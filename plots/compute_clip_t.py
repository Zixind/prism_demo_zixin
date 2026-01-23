#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from typing import Optional

#pip install -U pandas pillow tqdm transformers torch


def resolve_image_path(csv_path: Path, root: Optional[Path], rel: str) -> Path:
    base = root if root is not None else csv_path.parent
    return (base / rel).resolve()



@torch.no_grad()
def clip_t_batch(model, processor, images, texts, device):
    """
    Returns cosine similarity (normalized dot product) for each (image, text) pair.
    """
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
    txt_feat = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
    txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)

    # pairwise aligned: each row i with text i
    sims = (img_feat * txt_feat).sum(dim=-1)  # (B,)
    return sims.detach().cpu().numpy()

@torch.no_grad()
def clip_t_prompt_batch(model, processor, images, prompts, device):
    inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
    txt_feat = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )

    img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
    txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)

    # aligned pairs: i-th image with i-th prompt
    return (img_feat * txt_feat).sum(dim=-1).detach().cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to prompt_image_caption.csv",
        choices=["writing_symbols_outputs_flux1_duel_textprior/judge_gpt_4o_mini/n10_k10/flux1_duel/judge_gpt_4o_mini/n10_k10/prompt_image_caption.csv"])
    ap.add_argument("--root", type=str, default=None,
                    help="Root directory containing image_file paths (defaults to CSV folder).",
                    choices=["writing_symbols_outputs_flux1_duel_textprior/judge_gpt_4o_mini/n10_k10/flux1_duel/judge_gpt_4o_mini/n10_k10"])
    ap.add_argument("--model", type=str, default="openai/clip-vit-large-patch14",
                    help="CLIP model name (HF).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV path. Default: <csv>_with_clipt.csv")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    root = Path(args.root).resolve() if args.root else None

    df = pd.read_csv(csv_path)
    for col in ["image_file", "prompt"]:
        if col not in df.columns:
            raise RuntimeError(f"CSV missing column: {col}")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device != "cuda" else "cuda")
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    sims = []
    rows = df.to_dict("records")
    bs = args.batch_size

    for start in tqdm(range(0, len(rows), bs), desc="CLIP-T(prompt)"):
        batch = rows[start:start + bs]
        images, prompts = [], []

        for r in batch:
            img_path = resolve_image_path(csv_path, root, str(r["image_file"]))
            if not img_path.is_file():
                raise FileNotFoundError(f"Image not found: {img_path} (from image_file={r['image_file']})")

            images.append(Image.open(img_path).convert("RGB"))
            prompts.append(str(r["prompt"]))

        sims.extend(clip_t_prompt_batch(model, processor, images, prompts, device).tolist())

    out_df = pd.DataFrame({
        "image_file": df["image_file"].astype(str),
        "prompt": df["prompt"].astype(str),
        "clip_t_prompt": sims,
    })

    out_path = Path(args.out).resolve() if args.out else csv_path.with_name(csv_path.stem + "_clip_t_prompt.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")
    print(f"[info] device={device}, model={args.model}")


if __name__ == "__main__":
    main()
