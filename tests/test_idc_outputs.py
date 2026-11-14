"""Tests for the model-free parts of the IDC path: view routing, label adapters, DICOM SEG / SR writers.

No model weights or GPU needed; the DICOM writers run on a synthetic single-frame CR radiograph.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cxas.models.qseg_backend import apply_adapter, looks_frontal, view_from_position

ADAPTERS = Path(__file__).resolve().parent.parent / "cxas" / "data" / "adapters"


def _synthetic_cr(path, rows=64, cols=48):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"  # CR Image Storage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.PatientID, ds.PatientName, ds.PatientBirthDate, ds.PatientSex = "TEST", "Test^Patient", "", "O"
    ds.StudyDate, ds.StudyTime, ds.StudyID, ds.AccessionNumber, ds.ReferringPhysicianName = "20260101", "000000", "1", "", ""
    ds.Modality, ds.SeriesNumber, ds.InstanceNumber, ds.ViewPosition = "CR", 1, 1, "PA"
    ds.PatientOrientation = ["L", "F"]
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelSpacing = [0.2, 0.2]
    ds.PixelData = (np.random.RandomState(0).randint(0, 4096, (rows, cols))).astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return pydicom.dcmread(path)


class TestViewRouting(unittest.TestCase):
    def test_positions(self):
        for vp in ("PA", "AP", " pa "):
            self.assertEqual(view_from_position(vp), "frontal")
        for vp in ("LATERAL", "LL", "RL", "XTABLE LATERAL"):
            self.assertEqual(view_from_position(vp), "lateral")
        self.assertEqual(view_from_position("", default=""), "")
        self.assertEqual(view_from_position(None), "frontal")

    def test_looks_frontal(self):
        h, w = 100, 100
        left, right = np.zeros((h, w), bool), np.zeros((h, w), bool)
        right[20:80, 10:45], left[20:80, 55:90] = True, True        # side by side -> frontal
        self.assertTrue(looks_frontal({"left lung": (0.9, left), "right lung": (0.9, right)}, w))
        same = np.zeros((h, w), bool); same[20:80, 30:70] = True     # superimposed -> lateral
        self.assertFalse(looks_frontal({"left lung": (0.9, same), "right lung": (0.9, same)}, w))
        self.assertFalse(looks_frontal({"left lung": (0.9, left)}, w))


class TestAdapters(unittest.TestCase):
    def test_frontal_relabel_and_derive(self):
        ad = json.load(open(ADAPTERS / "anatomy187_frontal.json"))
        m = lambda: np.zeros((10, 10), bool)
        a, b, cl, cboth = m(), m(), m(), m()
        a[:, :5], b[:, 5:] = True, True
        cboth[0:2, 1:9], cl[0:2, 5:9] = True, True
        out = apply_adapter({"left hemidiaphragm": (0.9, a), "right hemidiaphragm": (0.8, b),
                             "clavicles": (0.7, cboth), "clavicle left": (0.6, cl), "clavicle right": (0.99, m())}, ad)
        self.assertTrue(out["right hemidiaphragm"][1][:, :5].all())          # sides swapped
        self.assertEqual(out["clavicle right"][0], 0.6)                       # derived: score = min of inputs
        self.assertTrue(out["clavicle right"][1][0:2, 1:5].all() and not out["clavicle right"][1][:, 5:].any())

    def test_lateral_zone_swap_and_lung_union(self):
        ad = json.load(open(ADAPTERS / "paxray177_lateral.json"))
        zones = {}
        for k, n in enumerate(["apical zone lung", "upper zone lung", "mid zone lung", "lung base"]):
            z = np.zeros((8, 4), bool); z[2 * k:2 * k + 2] = True
            zones[n] = (0.5 + 0.1 * k, z)
        out = apply_adapter(zones, ad)
        self.assertTrue(out["lung base"][1][2:4].all())                       # raw upper zone -> lung base
        self.assertTrue(out["lung"][1].all())
        self.assertIn("lung middle lobe right", out)


class TestDicomWriters(unittest.TestCase):
    def setUp(self):
        try:
            import highdicom  # noqa: F401
        except ImportError:
            self.skipTest("highdicom not installed")
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "cr.dcm")
        self.ds = _synthetic_cr(self.src)

    def test_seg_roundtrip(self):
        import pydicom
        from cxas.io_utils.dicomseg_2d import write_seg

        rng = np.random.RandomState(1)
        masks = {f"structure {i}": rng.rand(64, 48) > 0.6 for i in range(5)}
        masks["empty"] = np.zeros((64, 48), bool)
        out = write_seg(self.src, masks, os.path.join(self.tmp, "seg.dcm"), "test", "0")
        seg = pydicom.dcmread(out)
        self.assertEqual(seg.Modality, "SEG")
        labels = {s.SegmentNumber: s.SegmentLabel for s in seg.SegmentSequence}
        self.assertNotIn("empty", labels.values())
        px = seg.pixel_array
        for k, fg in enumerate(seg.PerFrameFunctionalGroupsSequence):
            name = labels[int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)]
            np.testing.assert_array_equal(px[k].astype(bool), masks[name])
        self.assertEqual(seg.ReferencedSeriesSequence[0].SeriesInstanceUID, self.ds.SeriesInstanceUID)

    def test_landmark_sr(self):
        import pydicom
        from cxas.io_utils.dicomseg_2d import write_landmark_sr

        lms = [{"class": "heart", "landmark": "left_cardiac_border", "x": 30.5, "y": 40.0, "confidence": 0.9},
               {"class": "trachea", "landmark": "carina", "x": 999.0, "y": 1.0, "confidence": 0.9}]  # outside
        out = write_landmark_sr(self.src, lms, os.path.join(self.tmp, "sr.dcm"), "test", "0")
        sr = pydicom.dcmread(out)
        self.assertEqual(sr.Modality, "SR")
        points = []

        def walk(items):
            for it in items:
                if getattr(it, "ValueType", "") == "SCOORD":
                    points.append(list(it.GraphicData))
                walk(getattr(it, "ContentSequence", []))
        walk(sr.ContentSequence)
        self.assertEqual(points, [[30.5, 40.0]])


if __name__ == "__main__":
    unittest.main()


class TestVendoredLandmarks(unittest.TestCase):
    def test_vocabulary(self):
        from cxas.landmarks.channels import VIEWS, Channels

        ch = Channels()
        self.assertEqual((ch.n, ch.n_classes), (311, 177))
        n_view = {v: sum(len(s) for s in ch.by_class[v].values()) for v in VIEWS}
        self.assertEqual(n_view, {"frontal": 203, "lateral": 108})

    def test_model_builds_offline(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        from cxas.landmarks.model import JointDenseModel

        m = JointDenseModel(311, 177, pretrained_encoder=False)        # architecture from the bundled config
        self.assertEqual(m.enc.config.patch_size, 14)


class TestMask2FormerPort(unittest.TestCase):
    """The vendored Mask2Former runs without detectron2 / qseg / MaskDINO."""

    def test_forward_without_detectron2(self):
        import sys
        import torch
        for blocked in ("detectron2", "qseg", "maskdino", "fvcore"):
            sys.modules.setdefault(blocked, None)
        from cxas.models.m2f.decoder import MultiScaleMaskedTransformerDecoder
        from cxas.models.m2f.model import MaskFormerHead
        from cxas.models.m2f.pixel_decoder import MSDeformAttnPixelDecoder, ShapeSpec

        shapes = {f"p{s}": ShapeSpec(32, s) for s in (4, 8, 16, 32)}
        pd = MSDeformAttnPixelDecoder(shapes, transformer_enc_layers=1, conv_dim=32, mask_dim=32,
                                      transformer_in_features=["p8", "p16", "p32"], common_stride=4)
        dec = MultiScaleMaskedTransformerDecoder(32, num_classes=5, hidden_dim=32, num_queries=7, nheads=4,
                                                 dim_feedforward=64, dec_layers=2, pre_norm=False, mask_dim=32,
                                                 enforce_input_project=False)
        feats = {f"p{s}": torch.randn(1, 32, 64 // s, 96 // s) for s in (4, 8, 16, 32)}
        out = MaskFormerHead(pd, dec).eval()(feats)
        self.assertEqual(tuple(out["pred_logits"].shape), (1, 7, 6))
        self.assertEqual(tuple(out["pred_masks"].shape), (1, 7, 16, 24))
        names = {k.split(".")[0] for k in pd.state_dict()}
        self.assertTrue({"adapter_1", "layer_1", "mask_features", "input_proj", "transformer"} <= names)


class TestWeightChecksums(unittest.TestCase):
    def test_verify_sha256(self):
        import hashlib
        from cxas.models import verify_sha256

        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "M2F", "frontal"))
        p = os.path.join(d, "M2F", "frontal", "model.safetensors")
        open(p, "wb").write(b"weights")
        open(os.path.join(d, "SHA256SUMS"), "w").write(
            hashlib.sha256(b"weights").hexdigest() + "  M2F/frontal/model.safetensors\n"
            + "0" * 64 + "  other/file.bin\n")
        verify_sha256(d, os.path.join(d, "SHA256SUMS"), "M2F/")          # other/ is outside the prefix
        open(p, "wb").write(b"tampered")
        with self.assertRaises(RuntimeError):
            verify_sha256(d, os.path.join(d, "SHA256SUMS"), "M2F/")


class TestUnits(unittest.TestCase):
    def test_spacing_precedence(self):
        from pydicom.dataset import Dataset
        from cxas.units import physical_spacing

        ds = Dataset(); ds.ImagerPixelSpacing = [0.2, 0.2]; ds.PixelSpacing = [0.18, 0.17]
        self.assertEqual(physical_spacing(ds)["source"], "PixelSpacing")
        self.assertEqual((physical_spacing(ds)["row_mm"], physical_spacing(ds)["col_mm"]), (0.18, 0.17))
        ds = Dataset(); ds.ImagerPixelSpacing = [0.2, 0.2]; ds.EstimatedRadiographicMagnificationFactor = 1.25
        self.assertAlmostEqual(physical_spacing(ds)["row_mm"], 0.16)
        ds = Dataset(); ds.ImagerPixelSpacing = [0.2, 0.2]; ds.DistanceSourceToDetector = 1800; ds.DistanceSourceToPatient = 1500
        self.assertAlmostEqual(physical_spacing(ds)["col_mm"], 0.2 / 1.2)
        ds = Dataset(); ds.ImagerPixelSpacing = [0.2, 0.2]
        self.assertIn("detector", physical_spacing(ds)["source"])
        self.assertIsNone(physical_spacing(Dataset()))

    def test_area_and_perimeter_on_a_stretched_grid(self):
        import cv2
        from cxas.units import features_in_mm
        # a disc of radius 100 mm on a 2000 (rows) x 1000 (cols) image with 0.2 mm pixels, resized to 512 x 512
        rows, cols, mm = 2000, 1000, 0.2
        yy, xx = np.mgrid[:rows, :cols]
        disc = ((yy - 1000) * mm) ** 2 + ((xx - 500) * mm) ** 2 <= 100 ** 2
        grid = cv2.resize(disc.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask = np.zeros((159, 512, 512), bool); mask[0] = grid
        spacing = {"row_mm": mm, "col_mm": mm, "source": "test"}
        area = features_in_mm("area", {"spine_area": float(grid.sum())}, mask, spacing, (rows, cols))
        self.assertAlmostEqual(area["spine_area_mm2"] / (np.pi * 100 ** 2), 1.0, delta=0.02)
        per = features_in_mm("perimeter", {}, mask, spacing, (rows, cols))
        self.assertAlmostEqual(per["spine_perimeter_mm"] / (2 * np.pi * 100), 1.0, delta=0.02)
        self.assertEqual(features_in_mm("area", {}, mask, None, (rows, cols)), {})


class TestLandmarkMeasurementsAndRegistration(unittest.TestCase):
    def test_ctr(self):
        from cxas.landmarks.measurements import measure
        lms = [{"class": "left lung", "landmark": "ctr_thorax", "x": 900, "y": 500, "confidence": 0.9},
               {"class": "right lung", "landmark": "ctr_thorax", "x": 100, "y": 500, "confidence": 0.9},
               {"class": "heart", "landmark": "left_cardiac_border", "x": 700, "y": 600, "confidence": 0.9},
               {"class": "heart", "landmark": "right_cardiac_border", "x": 300, "y": 600, "confidence": 0.9}]
        m = measure(lms, "frontal", spacing={"row_mm": 0.2, "col_mm": 0.2})
        self.assertAlmostEqual(m["ctr"], 0.5)
        self.assertAlmostEqual(m["thorax_width_mm"], 160.0)

    def test_dome_height_numpy2(self):
        # np.cross rejects 2-D vectors in NumPy 2; these paths crashed every cxas_idc run with numpy >= 2.x
        from cxas.landmarks.measurements import measure
        lm = lambda c, k, x, y: {"class": c, "landmark": k, "x": x, "y": y, "confidence": 0.9}
        frontal = [lm("right hemidiaphragm", "dome", 300, 700), lm("right lung", "costophrenic_pulmonary", 120, 760),
                   lm("heart", "cardiophrenic_cardiac_right", 480, 760)]
        self.assertAlmostEqual(measure(frontal, "frontal")["dome_height"], 60 / 360)
        lateral = [lm("diaphragm", "anterior_costophrenic", 100, 700), lm("diaphragm", "posterior_costophrenic", 500, 700),
                   lm("diaphragm", "dome", 300, 640)]
        self.assertAlmostEqual(measure(lateral, "lateral")["dome_height_lat"], 60 / 400)

    def _ref(self, n=40, seed=0):
        rng = np.random.RandomState(seed)
        return {(f"c{i}", "p"): rng.uniform(80, 430, 2) for i in range(n)}

    def test_recovers_a_similarity_and_ignores_an_outlier(self):
        from cxas.registration.landmark_fit import fit_landmarks
        ref = self._ref()
        a, s, t = np.deg2rad(7), 1.15, np.array([20.0, -35.0])
        R = s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        # image points that the true transform maps onto the reference: p = R^-1 (ref - t)
        lms = [{"class": c, "landmark": n, "x": q[0], "y": q[1], "confidence": 0.9}
               for (c, n), r in ref.items() for q in [np.linalg.solve(R, r - t)]]
        lms[0]["x"] += 300                                           # one grossly misplaced landmark
        M, info = fit_landmarks(lms, ref, (512, 512))
        self.assertTrue(np.allclose(M[:, :2], R, atol=1e-3) and np.allclose(M[:, 2], t, atol=0.5))
        self.assertEqual(info["n_inliers"], len(ref) - 1)

    def test_mirrored_lateral(self):
        from cxas.registration.landmark_fit import fit_landmarks
        ref = self._ref(seed=1)
        lms = [{"class": c, "landmark": n, "x": 511.0 - r[0], "y": r[1], "confidence": 0.9} for (c, n), r in ref.items()]
        M, info = fit_landmarks(lms, ref, (512, 512), allow_mirror=True)
        self.assertTrue(info["mirrored"])
        p = np.array([[l["x"], l["y"]] for l in lms])
        self.assertLess(np.abs(p @ M[:, :2].T + M[:, 2] - np.array(list(ref.values()))).max(), 1e-6)
        M2, info2 = fit_landmarks(lms, ref, (512, 512), allow_mirror=False)
        self.assertGreater(info2["rms_residual_px"], 10 * info["rms_residual_px"] + 1)
