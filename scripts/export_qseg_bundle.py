#!/usr/bin/env python
"""Export a qseg training run into a self-contained CXAS model bundle (developer tool: needs qseg to read the run;
the exported bundle is loaded by cxas.models.m2f without qseg).

Bundle layout (``<out>/``):
    config.yaml        qseg resolved config, dataset/output paths blanked, backbone.hf_id -> "backbone"
    categories.json    COCO categories of the run (keypoint declarations kept so the architecture matches)
    model.safetensors  model weights WITHOUT the backbone encoder (no optimizer / EMA / trainer state)
    backbone/          HF-format DINOv3 / RAD-DINO encoder (config.json + model.safetensors) taken from the
                       checkpoint, so inference never downloads the (gated) pretrained backbone
    adapter.json       label adapter applied to the raw instances (relabel / drop / derive), may be empty
    meta.json          provenance: source run, checkpoint sha256, class count, view, score floor

Usage:
    python scripts/export_qseg_bundle.py --run ~/workspace/qseg/outputs/anat_subdiaph187 \
        --cats ~/workspace/datasets/subdiaph_quadrants/annotations/val.json \
        --adapter cxas/data/adapters/anatomy187_frontal.json --view frontal \
        --out ~/.cxas/weights/M2F_anatomy_v2/frontal
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--ckpt", default=None, help="default: model_best.pth, else model_final.pth")
    ap.add_argument("--cats", required=True, type=Path, help="COCO json whose categories the run was trained on")
    ap.add_argument("--adapter", default=None, type=Path)
    ap.add_argument("--view", required=True, choices=["frontal", "lateral", "any"])
    ap.add_argument("--score-floor", type=float, default=0.05)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModel

    ckpt = Path(a.ckpt) if a.ckpt else next(p for p in (a.run / "model_best.pth", a.run / "model_final.pth") if p.exists())
    cfg = OmegaConf.load(a.run / "resolved_config.yaml")
    hf_id = str(cfg.model.backbone.hf_id)

    a.out.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model", state)

    enc_prefix = "backbone.encoder."
    enc = {k[len(enc_prefix):]: v.contiguous() for k, v in state.items() if k.startswith(enc_prefix)}
    rest = {k: v.contiguous() for k, v in state.items() if not k.startswith(enc_prefix)}
    assert enc, "no backbone.encoder.* weights in the checkpoint"

    bdir = a.out / "backbone"
    hf_cfg = AutoConfig.from_pretrained(hf_id)
    enc_model = AutoModel.from_config(hf_cfg)
    miss, unexp = enc_model.load_state_dict(enc, strict=False)
    assert not miss and not unexp, f"backbone mismatch: missing={miss[:5]} unexpected={unexp[:5]}"
    enc_model.save_pretrained(bdir, safe_serialization=True)
    save_file(rest, str(a.out / "model.safetensors"))

    for key in ("json_train", "json_val", "json_test", "json_pseudo", "json_ood", "image_root", "image_root_ood"):
        if key in cfg.data:
            cfg.data[key] = None
    cfg.train.output_dir = None
    for key in ("init_weights", "pretrained_head_path", "experiments_csv"):
        if key in cfg.train:
            cfg.train[key] = None
    cfg.model.backbone.hf_id = "backbone"
    text = OmegaConf.to_yaml(cfg)
    assert "/home/" not in text and "/tmp/" not in text, "absolute path left in the bundle config"
    (a.out / "config.yaml").write_text(text)

    cats = json.load(open(a.cats))["categories"]
    json.dump({"images": [], "annotations": [], "categories": cats}, open(a.out / "categories.json", "w"))
    if a.adapter:
        shutil.copy(a.adapter, a.out / "adapter.json")
    else:
        json.dump({"version": "none"}, open(a.out / "adapter.json", "w"))

    meta = {"source_run": a.run.name, "checkpoint": ckpt.name, "checkpoint_sha256": sha256(ckpt),
            "backbone": hf_id, "arch": str(cfg.model.arch), "num_classes": len(cats), "view": a.view,
            "score_floor": a.score_floor, "input_short_edge": int(cfg.aug.input_size[0])}
    json.dump(meta, open(a.out / "meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
