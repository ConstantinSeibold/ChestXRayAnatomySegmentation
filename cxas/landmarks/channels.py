"""Vendored from cxr-landmarks (https://github.com/ConstantinSeibold/cxr-landmarks, MIT) @ baa9bcc: inference only.

Landmark vocabulary -> model output channels.

Channels are view specific: one heatmap per (view, class, landmark) of the final vocabulary
(311 = 203 frontal + 108 lateral). Classes are the 177 PAXRay++ structures in category-id order;
a class has up to K_max = 13 landmark slots, named in its category's `keypoints` list.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

VIEWS = ("frontal", "lateral")


def _vocab_file(name: str) -> Path:
    return Path(__file__).parent / "data" / name


def view_of(file_name: str) -> str:
    """PAXRay++ convention: lateral images carry 'lateral' in their file name."""
    return "lateral" if "lateral" in Path(str(file_name)).name.lower() else "frontal"


class Channels:
    """(view, contiguous class, slot) <-> channel index."""

    def __init__(self, categories: list[dict] | None = None, vocab: dict | None = None):
        if categories is None:
            categories = json.load(open(_vocab_file("categories_kp_v1.json")))["categories"]
        if vocab is None:
            vocab = json.load(open(_vocab_file("final_vocab_v1.json")))
        cats = sorted(categories, key=lambda c: c["id"])
        self.categories = cats
        self.cat_ids = [c["id"] for c in cats]
        self.class_names = [c["name"] for c in cats]
        self.kp_names = [list(c.get("keypoints") or []) for c in cats]
        self.k_max = max(len(k) for k in self.kp_names)
        contig = {c["name"]: i for i, c in enumerate(cats)}
        self.items: list[tuple[str, int, int]] = []               # channel -> (view, class, slot)
        self.table: dict[str, dict[tuple[int, int], int]] = {v: {} for v in VIEWS}
        for p in vocab["provenance"]:
            c = contig[p["class"]]
            k = self.kp_names[c].index(p["kp"])
            self.table[p["view"]][(c, k)] = len(self.items)
            self.items.append((p["view"], c, k))
        self.n = len(self.items)
        self.view_mask = {v: torch.tensor([it[0] == v for it in self.items]) for v in VIEWS}
        self.by_class: dict[str, dict[int, list[tuple[int, int]]]] = {v: {} for v in VIEWS}
        self.lut = {v: torch.full((len(cats), self.k_max), -1, dtype=torch.long) for v in VIEWS}
        for v in VIEWS:
            for (c, k), ch in self.table[v].items():
                self.by_class[v].setdefault(c, []).append((k, ch))
                self.lut[v][c, k] = ch

    @property
    def n_classes(self) -> int:
        return len(self.cat_ids)
