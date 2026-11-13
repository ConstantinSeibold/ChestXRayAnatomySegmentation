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
