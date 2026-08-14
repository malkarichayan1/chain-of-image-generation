#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence
from PIL import Image

import pandas as pd
from scipy import stats
import torch
from transformers import CLIPModel, CLIPProcessor

from anchor_common import build_agreement_rows, load_labels
from recompute_boxes import load_box_cache

def compute_clip_predictions(rows: List[dict], boxes_by_prompt: Dict[int, Dict[str, list]], artifacts_dir: Path) -> pd.DataFrame:
    print("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    out = []
    for r in rows:
        prompt_id = r["prompt_id"]
        boxes = boxes_by_prompt.get(prompt_id)
        row = dict(r)
        
        if not boxes:
            row.update(clip_predicted_owner=None, clip_matches_attention=None, clip_correct=None)
            out.append(row)
            continue
            
        img_path = artifacts_dir / "images" / f"p{prompt_id}.png"
        if not img_path.exists():
            img_path = artifacts_dir / "images" / f"{prompt_id:04d}.png"
        if not img_path.exists() and "image_path" in r and r["image_path"]:
            from anchor_common import resolve_image_path
            img_path = resolve_image_path(artifacts_dir, r["image_path"])
        if not img_path.exists():
            row.update(clip_predicted_owner=None, clip_matches_attention=None, clip_correct=None)
            out.append(row)
            continue
            
        image = Image.open(img_path).convert("RGB")
        
        subjects = list(boxes.keys())
        crops = [image.crop(tuple(boxes[s])) for s in subjects]
        
        attribute = r["attribute"]
        text_prompt = f"a photo of a {attribute}" if not attribute.startswith("a ") else attribute
        
        inp = clip_proc(text=[text_prompt], images=crops, return_tensors="pt", padding=True)
        with torch.no_grad():
            sim = clip_model(**inp).logits_per_image.squeeze(1).numpy()
            
        best_idx = int(sim.argmax())
        clip_pred = subjects[best_idx]
        
        row.update(
            clip_predicted_owner=clip_pred,
            clip_matches_attention=(clip_pred == r["predicted_owner"]),
            clip_correct=(clip_pred == r["human_label"]) if r["scored"] else None
        )
        out.append(row)
        
    return pd.DataFrame(out)

def agreement_report(df: pd.DataFrame) -> dict:
    sub = df.dropna(subset=["clip_predicted_owner", "predicted_owner"])
    n = len(sub)
    agreements = int(sub["clip_matches_attention"].sum())
    return dict(n=n, agreement_rate=(agreements / n) if n else None)

def accuracy_report(df: pd.DataFrame) -> dict:
    sub = df[(df["scored"] == True) & df["clip_predicted_owner"].notna()]
    n = len(sub)
    attention_correct = int(sub["correct"].sum())
    clip_correct = int((sub["clip_predicted_owner"] == sub["human_label"]).sum())
    return dict(n=n, attention_accuracy=(attention_correct / n) if n else None, 
                clip_accuracy=(clip_correct / n) if n else None)
                
def mcnemar_report(df: pd.DataFrame) -> dict:
    sub = df[(df["scored"] == True) & df["clip_predicted_owner"].notna()]
    attn_correct = sub["correct"].astype(bool)
    clip_correct = (sub["clip_predicted_owner"] == sub["human_label"])
    
    attn_only = int((attn_correct & ~clip_correct).sum())
    clip_only = int((~attn_correct & clip_correct).sum())
    n_discordant = attn_only + clip_only
    p = (stats.binomtest(attn_only, n_discordant, 0.5, alternative="two-sided").pvalue
         if n_discordant else None)
    return dict(n=len(sub), attention_only_correct=attn_only,
                clip_only_correct=clip_only, n_discordant=n_discordant, p_value=p)

def _print_report(name: str, report: dict) -> None:
    print(f"\n{name}")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--annotator", default="chayan")
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    boxes = load_box_cache(artifacts_dir / "boxes.json")

    rows = build_agreement_rows(manifest, labels, margin_threshold=0.02)
    df = compute_clip_predictions(rows, boxes, artifacts_dir)
    
    missing = df["clip_predicted_owner"].isna().sum()
    print(f"=== {artifacts_dir} ===  rows: {len(df)}  missing predictions: {missing}")

    _print_report("Agreement (Attention vs CLIP)", agreement_report(df))
    _print_report("Accuracy vs Human Labels", accuracy_report(df))
    _print_report("McNemar (paired, same rows)", mcnemar_report(df))
