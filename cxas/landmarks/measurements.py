"""Vendored from cxr-landmarks (https://github.com/ConstantinSeibold/cxr-landmarks, MIT): src/cxrlandmarks/measurements.py.

Clinical measurements from named landmarks: the diagnostic criteria of "Semantic landmark detection in chest
radiographs" (Table 2), computed from the landmarks of one radiograph.

    from cxrlandmarks.measurements import measure, haller_index, CRITERIA
    m = measure(landmarks, view="frontal")                 # landmarks: [{class, landmark, x, y, confidence}, ...]
    m["ctr"], m["cp_angle_max"], m["cobb_axis"]
    m = measure(landmarks, view="frontal", spacing={"row_mm": 0.15, "col_mm": 0.15})   # adds *_mm lengths

Landmarks are in the image's own pixels. Sides are chosen by image position, not by class name (in PAXRay++ the
hemidiaphragm and clavicle names are mirrored relative to the lungs). Most measurements are angles or ratios and
need no pixel size; the two absolute lengths (thoracic width, sternovertebral distance) also come in mm when a
spacing is given. CRITERIA lists, per finding, which measurement the paper used, in which direction it indicates
the finding, the literature threshold where one exists, and the source.

Ported from the paper's evaluation code (kp_prep/diagnostic/measure.py, the --v1 path that the final model used):
the measurement functions are unchanged; `to_paper_names` maps the final landmark vocabulary onto the names they
read.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

FRONTAL_CORNERS = ("superior_left", "superior_right", "inferior_left", "inferior_right")
LATERAL_CORNERS = ("antero_superior", "antero_inferior", "postero_superior", "postero_inferior")
VERT_LAT = [f"vertebrae T{i}" for i in range(1, 13)] + [f"vertebrae L{i}" for i in range(1, 6)]

# finding -> view ("frontal", "lateral" or "both"), measurement, direction (+1: larger means the finding),
# literature threshold on the measurement (None: none used), source, description
CRITERIA = {
    "cardiomegaly":           dict(view="frontal", measurement="ctr", direction=+1, threshold=0.5,
                                   source="danzer1919", description="Cardiothoracic ratio > 0.5"),
    "cardiomegaly_lateral":   dict(view="lateral", measurement="hoffman_rigler", direction=+1, threshold=None,
                                   source="hoffman1965", description="Hoffman-Rigler sign: posterior heart border behind the IVC"),
    "pleural_effusion":       dict(view="frontal", measurement="cp_angle_max", direction=+1, threshold=None,
                                   source="blackmore1996", description="Blunted lateral costophrenic angle (degrees)"),
    "pleural_effusion_lateral": dict(view="lateral", measurement="post_sulcus_depth", direction=-1, threshold=None,
                                   source="collins1972", description="Posterior costophrenic sulcus filled"),
    "low_lung_volumes":       dict(view="frontal", measurement="rib_level", direction=-1, threshold=10.0,
                                   source="eur16260", description="Right dome above the 10th posterior rib"),
    "hyperinflation":         dict(view="frontal", measurement="rib_level", direction=+1, threshold=None,
                                   source="eur16260", description="Low diaphragm (posterior rib level of the right dome)"),
    "hyperinflation_flat_dome": dict(view="frontal", measurement="dome_height", direction=-1, threshold=None,
                                   source="pratt1987", description="Flattened hemidiaphragm dome"),
    "hyperinflation_lateral": dict(view="lateral", measurement="dome_height_lat", direction=-1, threshold=None,
                                   source="pratt1987", description="Flattened diaphragm on the lateral"),
    "hyperinflation_retrosternal": dict(view="lateral", measurement="retrosternal", direction=+1, threshold=None,
                                   source="pratt1987", description="Enlarged retrosternal clear space"),
    "hyperinflation_ap_depth": dict(view="lateral", measurement="ap_depth", direction=+1, threshold=None,
                                   source="", description="Antero-posterior chest depth (barrel chest)"),
    "elevated_hemidiaphragm": dict(view="frontal", measurement="dome_dy_abs", direction=+1, threshold=None,
                                   source="", description="Height difference of the hemidiaphragm domes"),
    "scoliosis":              dict(view="frontal", measurement="cobb_axis", direction=+1, threshold=10.0,
                                   source="cobb1948", description="Cobb angle > 10 degrees (spine-axis tangents)"),
    "kyphosis":               dict(view="lateral", measurement="kyphosis_axis", direction=+1, threshold=None,
                                   source="fon1980", description="Thoracic kyphosis T4-T12 (spine-axis angle, degrees)"),
    "vertebral_fracture":     dict(view="lateral", measurement="hloss_max", direction=+1, threshold=0.2,
                                   source="genant1993", description="Vertebral height loss >= 20% vs adjacent bodies (worst level T4-L2)"),
    "pectus_excavatum":       dict(view="both", measurement="haller", direction=+1, threshold=None,
                                   source="haller1987; khanna2010", description="Radiographic Haller index (see haller_index)"),
    "tortuous_aorta":         dict(view="frontal", measurement="knob_width", direction=+1, threshold=None,
                                   source="tomita2020", description="Aortic knob width"),
    "rotation":               dict(view="frontal", measurement="rotation_index", direction=+1, threshold=None,
                                   source="eur16260; marais2023", description="Clavicle-midline asymmetry (patient rotation)"),
}


# unit of every measurement: "deg", "mm", "ratio" (dimensionless), "px" (image pixels), "rib" (rib level), "count"
UNITS = {"ctr": "ratio", "thorax_width": "px", "thorax_width_mm": "mm", "cp_angle_max": "deg", "rib_level": "rib",
         "dome_height": "ratio", "dome_dy": "ratio", "dome_dy_abs": "ratio", "rotation_index": "ratio", "cobb": "deg",
         "cobb_axis": "deg", "knob_width": "ratio", "post_sulcus_depth": "ratio", "dome_height_lat": "ratio",
         "retrosternal": "ratio", "wedge_max_raw": "ratio", "hloss_max": "ratio", "n_vert_lat": "count",
         "kyphosis_cobb": "deg", "kyphosis_axis": "deg", "ap_depth": "ratio", "sv_dist": "px", "sv_dist_mm": "mm",
         "hoffman_rigler": "ratio", "haller": "ratio"}
UNITS.update({f"hloss_{v.split()[1]}": "ratio" for v in VERT_LAT})


def landmark_dict(landmarks, min_confidence: float = 0.1) -> Dict[str, Dict[str, tuple]]:
    """[{class, landmark, x, y, confidence}] -> {class: {landmark: (x, y)}}, keeping confident landmarks only."""
    out: Dict[str, Dict[str, tuple]] = {}
    for lm in landmarks:
        if float(lm.get("confidence", 1.0)) >= min_confidence:
            out.setdefault(lm["class"].strip(), {})[lm["landmark"]] = (float(lm["x"]), float(lm["y"]))
    return out


def to_paper_names(L: dict, lateral: bool) -> dict:
    """Nearest equivalent of each landmark the measurements read under their original names. Hemidiaphragm sides
    are matched by image position (names are mirrored vs lungs)."""
    out = {c: dict(d) for c, d in L.items()}
    heart = L.get("heart", {})
    for hemi in ("left hemidiaphragm", "right hemidiaphragm"):
        dome = L.get(hemi, {}).get("dome")
        if dome is None:
            continue
        lungs = [L[l] for l in ("left lung", "right lung") if L.get(l, {}).get("costophrenic_pulmonary")]
        if lungs:
            lung = min(lungs, key=lambda d: abs(d["costophrenic_pulmonary"][0] - dome[0]))
            out[hemi]["costophrenic_diaphragmatic"] = lung["costophrenic_pulmonary"]
            if lung.get("costophrenic_limb_diaphragm"):
                out[hemi]["ridge_lateral_third"] = lung["costophrenic_limb_diaphragm"]
        cph = [heart[k] for k in ("cardiophrenic_cardiac_left", "cardiophrenic_cardiac_right") if heart.get(k)]
        if cph:
            out[hemi]["cardiophrenic_diaphragmatic"] = min(cph, key=lambda p: abs(p[0] - dome[0]))
    for c, d in L.items():
        if "rib_head" in d:
            out[c]["costovertebral_costal"] = d["rib_head"]
        corners = [d.get(k) for k in (LATERAL_CORNERS if lateral else FRONTAL_CORNERS)]
        if c.startswith("vertebrae") and all(corners):
            out[c]["center"] = tuple(np.mean(corners, 0))
    if L.get("sternum", {}).get("manubriosternal_limb_body"):
        out["sternum"]["xiphoid"] = L["sternum"]["manubriosternal_limb_body"]
    if L.get("ascending aorta", {}).get("retrosternal_posterior"):
        out["ascending aorta"]["anterior_border"] = L["ascending aorta"]["retrosternal_posterior"]
    if L.get("inferior vena cava", {}).get("ivc_ra_caval"):
        out["inferior vena cava"]["center"] = L["inferior vena cava"]["ivc_ra_caval"]
    return out


def _get(L, cls, kp):
    return L.get(cls, {}).get(kp)


def _angle(v, a, b):
    u = np.subtract(a, v); w = np.subtract(b, v)
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu < 1e-6 or nw < 1e-6:
        return np.nan
    return math.degrees(math.acos(float(np.clip(np.dot(u, w) / (nu * nw), -1, 1))))


def measurements(L: dict, sx: float = 1.0, sy: float = 1.0) -> dict:
    """Frontal view. L: {class: {kp: (x, y)}} under the paper's names; sx, sy scale the coordinates first."""
    P = {c: {k: (p[0] * sx, p[1] * sy) for k, p in d.items()} for c, d in L.items()}
    g = lambda c, k: _get(P, c, k)
    m: dict[str, float] = {}

    # thoracic width + midline
    tl, tr = g("left lung", "ctr_thorax"), g("right lung", "ctr_thorax")
    width = abs(tl[0] - tr[0]) if tl and tr else np.nan
    # midline = vertebral-body centre (mean of the 4 endplate corners) of T1-T5
    mids = []
    for i in range(1, 8):
        cs = [g(f"vertebrae T{i}", k) for k in ("superior_left", "superior_right", "inferior_left", "inferior_right")]
        if all(cs):
            mids.append((np.mean([c[1] for c in cs]), np.mean([c[0] for c in cs])))
    mids.sort()

    def midline_at(y):
        if len(mids) < 2:
            return np.nan
        return float(np.interp(y, [m[0] for m in mids], [m[1] for m in mids]))
    mid_x = float(np.median([m[1] for m in mids[:5]])) if mids else np.nan

    # G01 CTR (danzer1919)
    hl, hr = g("heart", "left_cardiac_border"), g("heart", "right_cardiac_border")
    m["ctr"] = abs(hl[0] - hr[0]) / width if hl and hr and width > 0 else np.nan
    m["thorax_width"] = width                              # for the radiographic Haller index

    # sides by image position: hemidiaphragm whose dome is nearest each lung's CP point
    hemis = [h for h in ("left hemidiaphragm", "right hemidiaphragm") if g(h, "dome")]

    def hemi_near(pt):
        return min(hemis, key=lambda h: abs(g(h, "dome")[0] - pt[0])) if hemis and pt else None

    # G02 costophrenic angle (blackmore1996): vertex CP point, arms to lateral lung margin
    # and to the lateral third of the hemidiaphragm on the same side
    cps = []
    domes = {}
    for lung in ("left lung", "right lung"):
        v, a = g(lung, "costophrenic_pulmonary"), g(lung, "lateral_margin")
        h = hemi_near(v)
        b = g(h, "ridge_lateral_third") if h else None
        cps.append(_angle(v, a, b) if v and a and b else np.nan)
        if h:
            domes[lung] = g(h, "dome")
    m["cp_angle_max"] = np.nanmax(cps) if not all(np.isnan(cps)) else np.nan

    # G20 inspiration (eur16260 1.1.1): posterior-rib level of the right dome, fractional,
    # from the costovertebral ends of the right posterior ribs
    d = domes.get("right lung")
    ys = []
    for n in range(1, 13):
        sfx = {1: "st", 2: "nd", 3: "rd"}.get(n, "th")
        p = g(f"posterior {n}{sfx} rib right", "costovertebral_costal")
        ys.append(p[1] if p else np.nan)
    ys = np.array(ys)
    if d is not None and np.isfinite(ys).sum() >= 6:
        idx = np.where(np.isfinite(ys))[0]
        yy, rr = ys[idx], idx + 1.0
        order = np.argsort(yy)
        m["rib_level"] = float(np.interp(d[1], yy[order], rr[order], left=rr.min() - 0.5, right=rr.max() + 0.5))
    else:
        m["rib_level"] = np.nan

    # G04 dome flattening (pratt1987): dome height above the CP-cardiophrenic chord / chord
    flats = []
    for h in hemis:
        a, b, dm = g(h, "costophrenic_diaphragmatic"), g(h, "cardiophrenic_diaphragmatic"), g(h, "dome")
        if a and b and dm:
            ch = np.subtract(b, a); L_ = np.linalg.norm(ch)
            if L_ > 1e-6:
                flats.append(abs(np.cross(ch, np.subtract(dm, a))) / L_ / L_)
    m["dome_height"] = float(np.mean(flats)) if flats else np.nan

    # G04 hemidiaphragm asymmetry: (y of dome under left lung - under right lung) / thoracic width
    if "left lung" in domes and "right lung" in domes and width > 0:
        m["dome_dy"] = (domes["left lung"][1] - domes["right lung"][1]) / width
    else:
        m["dome_dy"] = np.nan
    m["dome_dy_abs"] = abs(m["dome_dy"])

    # G18 rotation (eur16260 1.1.2, marais2023): medial clavicle ends vs spinous processes
    sc = [g(c, "sternoclavicular_clavicular") for c in ("clavicle left", "clavicle right")]
    if not all(sc):
        sc = [g("clavicles", f"sternoclavicular_clavicular_{s}") for s in ("left", "right")]
    mx = midline_at((sc[0][1] + sc[1][1]) / 2) if all(sc) else np.nan
    if np.isfinite(mx):
        dA, dB = abs(sc[0][0] - mx), abs(sc[1][0] - mx)
        m["rotation_index"] = abs(dA - dB) / (dA + dB) if dA + dB > 0 else np.nan
    else:
        m["rotation_index"] = np.nan

    # G19 Cobb angle (cobb1948): max difference of superior-endplate tilts, T1-L5
    tilts = []
    for v in [f"vertebrae T{i}" for i in range(1, 13)] + [f"vertebrae L{i}" for i in range(1, 6)]:
        a, b = g(v, "superior_left"), g(v, "superior_right")
        if a and b:
            dx, dy = b[0] - a[0], b[1] - a[1]
            if dx < 0:
                dx, dy = -dx, -dy
            tilts.append(math.degrees(math.atan2(dy, dx)))
    m["cobb"] = (max(tilts) - min(tilts)) if len(tilts) >= 6 else np.nan

    # G19 Cobb angle from the spine axis: polynomial x(y) through the vertebral-body centres T1-L5,
    # angle = max - min tangent direction over the fitted levels (the model flattens the endplates)
    cen = [g(v, "center") for v in [f"vertebrae T{i}" for i in range(1, 13)] + [f"vertebrae L{i}" for i in range(1, 6)]]
    cen = [c for c in cen if c]
    if len(cen) >= 8:
        yy, xx = np.array([c[1] for c in cen]), np.array([c[0] for c in cen])
        poly = np.poly1d(np.polyfit(yy, xx, 4))
        ang = np.degrees(np.arctan(poly.deriv()(yy)))
        m["cobb_axis"] = float(ang.max() - ang.min())
    else:
        m["cobb_axis"] = np.nan

    # G09 aortic knob width (tomita2020-style): knob lateral extreme to midline / thoracic width
    k = g("aortic arch", "knob")
    m["knob_width"] = abs(k[0] - mid_x) / width if k and np.isfinite(mid_x) and width > 0 else np.nan
    return m


def measurements_lat(L: dict, sx: float = 1.0, sy: float = 1.0) -> dict:
    """Lateral view. Left/right superimpose, so only antero-posterior / vertical geometry;
    'anterior' is taken from the diaphragm's own anterior vs posterior sulcus points."""
    P = {c: {k: (p[0] * sx, p[1] * sy) for k, p in d.items()} for c, d in L.items()}
    g = lambda c, k: _get(P, c, k)
    m: dict[str, float] = {}
    ant, post, dome = g("diaphragm", "anterior_costophrenic"), g("diaphragm", "posterior_costophrenic"), g("diaphragm", "dome")
    depth = abs(ant[0] - post[0]) if ant and post else np.nan

    # G02 posterior costophrenic sulcus (collins1972, blackmore1996): how far the posterior
    # sulcus reaches below the dome; fluid fills the sulcus first
    if post and dome and depth > 0:
        m["post_sulcus_depth"] = (post[1] - dome[1]) / depth
    else:
        m["post_sulcus_depth"] = np.nan

    # G04 diaphragm flattening on the lateral (pratt1987): dome height above the
    # anterior-posterior sulcus chord, relative to the chord
    if ant and post and dome:
        ch = np.subtract(post, ant); L_ = np.linalg.norm(ch)
        m["dome_height_lat"] = abs(np.cross(ch, np.subtract(dome, ant))) / L_ / L_ if L_ > 1e-6 else np.nan
    else:
        m["dome_height_lat"] = np.nan

    # G08 retrosternal clear space (pratt1987 sign of hyperinflation): sternum (interpolated
    # between jugular notch and xiphoid) to the anterior border of the ascending aorta
    jn, xi, aa = g("sternum", "jugular_notch"), g("sternum", "xiphoid"), g("ascending aorta", "anterior_border")
    if jn and xi and aa and depth > 0 and abs(xi[1] - jn[1]) > 1e-6:
        t = (aa[1] - jn[1]) / (xi[1] - jn[1])
        sx_ = jn[0] + t * (xi[0] - jn[0])
        m["retrosternal"] = abs(aa[0] - sx_) / depth
    else:
        m["retrosternal"] = np.nan

    # G19 vertebral height loss (genant1993), T4-L2: 1 - anterior height vs a local reference
    # (own posterior height and the adjacent bodies' heights)
    H = {}
    for v in VERT_LAT:
        a_s, a_i, p_s, p_i = (g(v, k) for k in ("antero_superior", "antero_inferior", "postero_superior", "postero_inferior"))
        if a_s and a_i and p_s and p_i:
            H[v] = (math.dist(a_s, a_i), math.dist(p_s, p_i))
    raw, rel = [], []
    for j, v in enumerate(VERT_LAT):
        if v not in H or not (4 <= j + 1 <= 14):          # T4 (index 3) .. L2 (index 13)
            continue
        ha, hp = H[v]
        ref = [hp] + [h for u in VERT_LAT[j - 1: j + 2: 2] if u in H for h in H[u]]
        if hp > 1e-6:
            raw.append(1 - ha / hp)
        m[f"hloss_{v.split()[1]}"] = 1 - ha / np.median(ref) if len(ref) >= 3 else np.nan
        rel.append(m[f"hloss_{v.split()[1]}"])
    m["wedge_max_raw"] = max(raw) if len(raw) >= 5 else np.nan
    rel = [x for x in rel if np.isfinite(x)]
    m["hloss_max"] = max(rel) if len(rel) >= 5 else np.nan
    m["n_vert_lat"] = len(raw)

    # body-size reference for the lateral distances: median antero-posterior depth of the
    # vertebral bodies, which inflation and heart size leave alone
    vdepth = [math.dist(g(v, "antero_superior"), g(v, "postero_superior")) for v in VERT_LAT
              if g(v, "antero_superior") and g(v, "postero_superior")]
    vd = float(np.median(vdepth)) if len(vdepth) >= 3 else np.nan
    sternum = [p for p in (g("sternum", k) for k in ("jugular_notch", "sternal_angle", "xiphoid")) if p]
    spine_ant = [g(v, k) for v in VERT_LAT for k in ("antero_superior", "antero_inferior") if g(v, k)]
    post_dir = (np.sign(np.mean([p[0] for p in spine_ant]) - np.mean([p[0] for p in sternum]))
                if sternum and spine_ant else np.nan)

    # G19 thoracic kyphosis (fon1980): Cobb angle between the superior endplate of T4 and the
    # inferior endplate of T12 (nearest available level as fallback)
    up = next((v for v in ("vertebrae T4", "vertebrae T3", "vertebrae T5")
               if g(v, "antero_superior") and g(v, "postero_superior")), None)
    lo = next((v for v in ("vertebrae T12", "vertebrae T11") if g(v, "antero_inferior") and g(v, "postero_inferior")), None)
    if up and lo:
        a1 = math.degrees(math.atan2(*np.subtract(g(up, "postero_superior"), g(up, "antero_superior"))[::-1]))
        a2 = math.degrees(math.atan2(*np.subtract(g(lo, "postero_inferior"), g(lo, "antero_inferior"))[::-1]))
        d = abs(a1 - a2) % 180
        m["kyphosis_cobb"] = min(d, 180 - d)
    else:
        m["kyphosis_cobb"] = np.nan

    # G19 thoracic kyphosis from the spine axis: quadratic fit x(y) through the vertebral-body
    # centres T2-L1, angle between its tangents at T4 and T12
    cen = {v: g(v, "center") for v in VERT_LAT[1:13] if g(v, "center")}
    if len(cen) >= 6 and "vertebrae T4" in cen and "vertebrae T12" in cen:
        yy, xx = np.array([p[1] for p in cen.values()]), np.array([p[0] for p in cen.values()])
        q2, q1, _ = np.polyfit(yy, xx, 2)
        slope = lambda y: 2 * q2 * y + q1
        m["kyphosis_axis"] = abs(math.degrees(math.atan(slope(cen["vertebrae T4"][1])) - math.atan(slope(cen["vertebrae T12"][1]))))
    else:
        m["kyphosis_axis"] = np.nan

    # G04 antero-posterior chest depth ("barrel chest"): sternal angle to the line through the
    # anterior vertebral corners, in vertebral-body depths
    sa = g("sternum", "sternal_angle") or (tuple(np.mean([g("sternum", "jugular_notch"), g("sternum", "xiphoid")], 0))
                                           if g("sternum", "jugular_notch") and g("sternum", "xiphoid") else None)
    spine_line = None
    if len(spine_ant) >= 4:
        a, b = np.polyfit([p[1] for p in spine_ant], [p[0] for p in spine_ant], 1)  # x = a*y + b
        spine_line = lambda p: abs(p[0] - (a * p[1] + b)) / math.sqrt(1 + a * a)
    m["ap_depth"] = spine_line(sa) / vd if sa and spine_line and vd > 0 else np.nan

    # G19 pectus excavatum (haller1987, khanna2010): sternovertebral distance at the lower
    # sternum, where the depression is deepest; divided by the frontal thoracic width in haller_index
    xi = g("sternum", "xiphoid")
    lower = [p for p in (sa, xi, tuple(np.mean([sa, xi], 0)) if sa and xi else None) if p]
    m["sv_dist"] = min(spine_line(p) for p in lower) if lower and spine_line else np.nan

    # G01 cardiomegaly on the lateral (Hoffman-Rigler sign, hoffman1965): posterior heart border
    # behind the inferior vena cava, in vertebral-body depths
    hb, ivc = g("heart", "posterior_border"), g("inferior vena cava", "center")
    m["hoffman_rigler"] = (hb[0] - ivc[0]) * post_dir / vd if hb and ivc and vd > 0 and np.isfinite(post_dir) else np.nan
    return m


def measure(landmarks, view: str, spacing: Optional[dict] = None, min_confidence: float = 0.1) -> dict:
    """All measurements of one radiograph from its landmarks (image pixels).

    spacing: {"row_mm", "col_mm"} (e.g. from the DICOM PixelSpacing); adds thorax_width_mm (frontal) or
    sv_dist_mm (lateral). Angles and ratios are computed in pixels as in the paper; with non-square pixels they
    would be slightly distorted, which DICOM radiographs practically never have.
    """
    if view not in ("frontal", "lateral"):
        raise ValueError("view must be 'frontal' or 'lateral'")
    lateral = view == "lateral"
    L = to_paper_names(landmark_dict(landmarks, min_confidence), lateral)
    fn = measurements_lat if lateral else measurements
    m = {k: (float(v) if v is not None else np.nan) for k, v in fn(L).items()}
    if spacing:
        mm = fn(L, sx=float(spacing["col_mm"]), sy=float(spacing["row_mm"]))
        key = "sv_dist" if lateral else "thorax_width"
        m[key + "_mm"] = float(mm[key])
    return m


def haller_index(frontal: dict, lateral: dict) -> float:
    """Radiographic Haller index: frontal thoracic width / lateral sternovertebral distance (khanna2010).
    Uses mm when both views have a pixel spacing; otherwise pixels, which only works if both radiographs share
    the same pixel size (the paper's setting)."""
    if np.isfinite(frontal.get("thorax_width_mm", np.nan)) and np.isfinite(lateral.get("sv_dist_mm", np.nan)):
        w, d = frontal["thorax_width_mm"], lateral["sv_dist_mm"]
    else:
        w, d = frontal.get("thorax_width", np.nan), lateral.get("sv_dist", np.nan)
    return float(w / d) if np.isfinite(w) and np.isfinite(d) and d > 0 else np.nan
