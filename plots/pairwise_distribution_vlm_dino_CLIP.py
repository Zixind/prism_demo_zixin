#!/usr/bin/env python3
"""
pairwise_dino_vs_vlm.py

Compute DINOv3 similarity scores for each VLM duel pair and compare
them against the VLM judge winner.

Assumes directory layout like:

  ROOT/
    writing_symbols_outputs_all/
      qwen_image_fal_duel/ or flux1_fal_duel/ or gemini_duel/
          judge_gpt_5_nano/
            n6_k5/
              duel_pairs_images/
                round001_pair001_i3_A.png
                round001_pair001_j1_B.png
                ...
              vlm_pairs/
                01_Anime_..._vlm_pairs.csv
                ...
              01_Anime_..._ORIGINAL.png
              ...

Each *_vlm_pairs.csv has columns:
  round, pair_idx, i, j, imgA, imgB, scoreA, scoreB, winner, conf

This script adds DINO-based scores + winners and writes:

  *_vlm_pairs_with_dino.csv

for each object.
"""

from pathlib import Path
from PIL import Image
import torch
import numpy as np
import pandas as pd
import re

from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel
from scipy.stats import pearsonr, spearmanr   # pearson

# --------------------------------------------------------------------
# BASIC CONFIG – ADAPT THESE IF NEEDED
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # repo root

# JUDGE_TAG = "gpt_5_nano"        # which judge bucket you used
JUDGE_TAGS = [
    "gpt_5_nano",
    "gpt_4o_mini"
]
RUN_TAG   = "n6_k5"             # run subdir (nN_kK)

# T2I duel dirs you want to process
T2I_DIRS = {
    "flux1_fal_duel": "Flux1-duel",
    "gemini_duel": "Gemini-duel",
    "qwen_image_fal_duel": "Qwen-Image-duel",
}


# ====================== DINOv3 SETUP =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[DINO] Using device:", DEVICE)

DINO_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE)
model.eval()

BATCH_SIZE = 16  # tune based on memory


@torch.no_grad()
def dinov3_embed(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        emb = outputs.pooler_output
    else:
        emb = outputs.last_hidden_state.mean(dim=1)

    emb = emb.squeeze(0)
    emb = emb / (emb.norm() + 1e-8)
    return emb.cpu().numpy().astype(np.float32)


@torch.no_grad()
def dinov3_embed_batch(img_paths, batch_size: int = BATCH_SIZE) -> np.ndarray:
    all_embs = []
    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start:start + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            emb = outputs.last_hidden_state.mean(dim=1)

        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
        all_embs.append(emb.cpu().numpy().astype(np.float32))

    return np.concatenate(all_embs, axis=0)

# ====================== CLIP-I SETUP =========================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to(DEVICE)
clip_model.eval()

@torch.no_grad()
def clip_embed(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    feats = clip_model.get_image_features(**inputs)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    emb = feats.squeeze(0).cpu().numpy().astype(np.float32)
    return emb

@torch.no_grad()
def clip_embed_batch(img_paths, batch_size: int = BATCH_SIZE) -> np.ndarray:
    all_embs = []
    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start:start + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(DEVICE)
        feats = clip_model.get_image_features(**inputs)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        all_embs.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(all_embs, axis=0)

def run_dir_for_t2i(t2i_dir: str, judge_tag: str) -> Path:
    """Return run directory for given T2I duel model."""
    return (
        ROOT
        / "writing_symbols_outputs_all"
        / t2i_dir
        / f"judge_{judge_tag}"
        / RUN_TAG
    )


def compute_pairwise_backbones_for_object(run_dir: Path, vlm_csv: Path, judge_tag: str) -> pd.DataFrame:
    """
    For a single *_vlm_pairs.csv, compute:
      - DINO scores vs ORIGINAL: dino_scoreA/B, dino_margin, dino_winner, match
      - CLIP-I scores vs ORIGINAL: clip_scoreA/B, clip_margin, clip_winner, clip_match

    Returns the augmented DataFrame.
    """
    obj_name = vlm_csv.stem.replace("_vlm_pairs", "")
    print(f"\n[OBJ] {obj_name}")
    df = pd.read_csv(vlm_csv)

    # ORIGINAL image for this object
    df["obj_name"] = obj_name
    orig_candidates = sorted(run_dir.glob(f"{obj_name}_ORIGINAL.png"))
    if not orig_candidates:
        raise FileNotFoundError(f"No ORIGINAL image for {obj_name} in {run_dir}")
    orig_path = orig_candidates[0]
    print(f"[OBJ] Using ORIGINAL: {orig_path.name}")

    # orig_emb = dinov3_embed(orig_path)  # (D,)
    # Backbone embeddings for ORIGINAL
    orig_dino_emb = dinov3_embed(orig_path)  # (D_dino,)
    orig_clip_emb = clip_embed(orig_path)    # (D_clip,)
    # Already L2-normalized

    # Duel images
    duel_img_dir = run_dir / "duel_pairs_images"

    all_fnames = sorted(set(df["imgA"].tolist() + df["imgB"].tolist()))
    img_paths = [duel_img_dir / fn for fn in all_fnames]

    # Sanity check existence
    existing_paths = []
    existing_names = []
    for fn, p in zip(all_fnames, img_paths):
        if p.is_file():
            existing_paths.append(p)
            existing_names.append(fn)
        else:
            print(f"[OBJ][warn] missing duel image: {p}")

    if not existing_paths:
        raise RuntimeError(f"[OBJ] No duel images found for {obj_name}")

    # DINO embeddings
    dino_embs = dinov3_embed_batch(existing_paths, batch_size=BATCH_SIZE)  # (N, D_dino)
    # CLIP embeddings
    clip_embs = clip_embed_batch(existing_paths, batch_size=BATCH_SIZE)    # (N, D_clip)


    # Map filename -> embedding (for each backbone)
    name2emb_dino = {fn: emb for fn, emb in zip(existing_names, dino_embs)}
    name2emb_clip = {fn: emb for fn, emb in zip(existing_names, clip_embs)}


    # === per-pair scores ===
    dino_scoreA, dino_scoreB, dino_margin, dino_winner, dino_match = [], [], [], [], []
    clip_scoreA, clip_scoreB, clip_margin, clip_winner, clip_match = [], [], [], [], []

    for _, row in df.iterrows():
        fnA = row["imgA"]
        fnB = row["imgB"]

        if fnA not in name2emb_dino or fnB not in name2emb_dino:
            # if something missing, skip or use NaN
            dino_scoreA.append(np.nan)
            dino_scoreB.append(np.nan)
            dino_margin.append(np.nan)
            dino_winner.append(np.nan)
            dino_match.append(np.nan)

            clip_scoreA.append(np.nan)
            clip_scoreB.append(np.nan)
            clip_margin.append(np.nan)
            clip_winner.append(np.nan)
            clip_match.append(np.nan)
            continue

        # ----- DINO -----
        embA_dino = name2emb_dino[fnA]
        embB_dino = name2emb_dino[fnB]

        sA_d = float(np.dot(embA_dino, orig_dino_emb))
        sB_d = float(np.dot(embB_dino, orig_dino_emb))
        margin_d = sA_d - sB_d
        win_d = 0 if sA_d >= sB_d else 1

        dino_scoreA.append(sA_d)
        dino_scoreB.append(sB_d)
        dino_margin.append(margin_d)
        dino_winner.append(win_d)
        dino_match.append(1 if win_d == row["winner"] else 0)

        # ----- CLIP-I -----
        embA_c = name2emb_clip[fnA]
        embB_c = name2emb_clip[fnB]

        sA_c = float(np.dot(embA_c, orig_clip_emb))
        sB_c = float(np.dot(embB_c, orig_clip_emb))
        margin_c = sA_c - sB_c
        win_c = 0 if sA_c >= sB_c else 1

        clip_scoreA.append(sA_c)
        clip_scoreB.append(sB_c)
        clip_margin.append(margin_c)
        clip_winner.append(win_c)
        clip_match.append(1 if win_c == row["winner"] else 0)


    # Attach columns
    df["dino_scoreA"] = dino_scoreA
    df["dino_scoreB"] = dino_scoreB
    df["dino_margin"] = dino_margin
    df["dino_winner"] = dino_winner
    df["match"] = dino_match          # DINO vs VLM match

    df["clip_scoreA"] = clip_scoreA
    df["clip_scoreB"] = clip_scoreB
    df["clip_margin"] = clip_margin
    df["clip_winner"] = clip_winner
    df["clip_match"] = clip_match     # CLIP vs VLM match

    # Extra diagnostic columns
    df["vlm_margin"] = df["scoreA"] - df["scoreB"]
    df["vlm_sign"] = np.where(df["winner"] == 0, 1.0, -1.0)
    df["dino_sign"] = np.where(df["dino_winner"] == 0, 1.0, -1.0)
    df["clip_sign"] = np.where(df["clip_winner"] == 0, 1.0, -1.0)

    # ---------- Summary metrics (per object) ----------
    def _p_str(p):
        return "< 0.001" if p < 0.001 else f"= {p:.3g}"

    # DINO vs VLM
    valid_dino = df["match"].notna()
    if valid_dino.any():
        sub_d = df.loc[valid_dino].copy()
        acc_d = sub_d["match"].mean()

        if sub_d["vlm_sign"].nunique() > 1 and sub_d["dino_sign"].nunique() > 1:
            rP_sd, pP_sd = pearsonr(sub_d["vlm_sign"], sub_d["dino_sign"])
            rS_sd, pS_sd = spearmanr(sub_d["vlm_sign"], sub_d["dino_sign"])
        else:
            rP_sd = pP_sd = rS_sd = pS_sd = np.nan

        print(
            f"[OBJ] N_pairs={valid_dino.sum():3d} "
            f"acc(DINO vs VLM)={acc_d:.3f}\n"
            f"      DINO sign: Pearson r={rP_sd:.3f}, p{_p_str(pP_sd)}; "
            f"Spearman ρ={rS_sd:.3f}, p{_p_str(pS_sd)}"
        )
    else:
        print("[OBJ] No valid pairs for DINO metrics.")


    # CLIP vs VLM
    valid_clip = df["clip_match"].notna()
    if valid_clip.any():
        sub_c = df.loc[valid_clip].copy()
        acc_c = sub_c["clip_match"].mean()

        if sub_c["vlm_sign"].nunique() > 1 and sub_c["clip_sign"].nunique() > 1:
            rP_sc, pP_sc = pearsonr(sub_c["vlm_sign"], sub_c["clip_sign"])
            rS_sc, pS_sc = spearmanr(sub_c["vlm_sign"], sub_c["clip_sign"])
        else:
            rP_sc = pP_sc = rS_sc = pS_sc = np.nan

        print(
            f"      CLIP-I vs VLM acc={acc_c:.3f}\n"
            f"      CLIP sign: Pearson r={rP_sc:.3f}, p{_p_str(pP_sc)}; "
            f"Spearman ρ={rS_sc:.3f}, p{_p_str(pS_sc)}"
        )
    else:
        print("[OBJ] No valid pairs for CLIP metrics.")

    df["judge_tag"] = judge_tag
    return df


def main():
    overall_rows = []
    # directory for the triplet files
    triple_dir = ROOT / "figures" / "per_obj_judge_model"
    triple_dir.mkdir(parents=True, exist_ok=True)

    for judge_tag in JUDGE_TAGS:
        print("\n############################################")
        print(f"[JUDGE] {judge_tag}")
        print("############################################")

        for t2i_dir, label in T2I_DIRS.items():
            run_dir = run_dir_for_t2i(t2i_dir, judge_tag)
            if not run_dir.is_dir():
                print(f"[SKIP] No run_dir for {t2i_dir} with judge={judge_tag}: {run_dir}")
                continue

            print(f"\n==============================")
            print(f"[MODEL] {label} ({t2i_dir})")
            print(f"Run dir: {run_dir}")
            print(f"==============================")

            vlm_pairs_dir = run_dir / "vlm_pairs"
            if not vlm_pairs_dir.is_dir():
                print(f"[SKIP] No vlm_pairs dir in {run_dir}")
                continue

            vlm_files = sorted(vlm_pairs_dir.glob("*_vlm_pairs.csv"))
            if not vlm_files:
                print(f"[SKIP] No *_vlm_pairs.csv files in {vlm_pairs_dir}")
                continue

            for vlm_csv in vlm_files:
                try:
                    df_aug = compute_pairwise_backbones_for_object(run_dir, vlm_csv, judge_tag)
                    df_aug["t2i_dir"] = t2i_dir
                    df_aug["model_label"] = label
                    obj_name = df_aug["obj_name"].iloc[0]
                    safe_obj = re.sub(r"[^A-Za-z0-9._-]+", "_", obj_name)
                    triple_name = f"pairs_{safe_obj}_judge-{judge_tag}_t2i-{t2i_dir}.csv"
                    triple_path = triple_dir / triple_name
                    df_aug.to_csv(triple_path, index=False)
                    print(f"[SAVE] Triplet CSV: {triple_path}")
                    overall_rows.append(df_aug)
                except Exception as e:
                    print(f"[ERR] Failed on {vlm_csv} (judge={judge_tag}): {e}")

        if overall_rows:
            big_df = pd.concat(overall_rows, ignore_index=True)

            def _p_str(p):
                return "< 0.001" if p < 0.001 else f"= {p:.3g}"

            # DINO overall
            valid_dino = big_df["match"].notna()
            if valid_dino.any():
                acc_d = big_df.loc[valid_dino, "match"].mean()
                sub_d = big_df.loc[valid_dino].copy()

                if sub_d["vlm_sign"].nunique() > 1 and sub_d["dino_sign"].nunique() > 1:
                    rP_sd, pP_sd = pearsonr(sub_d["vlm_sign"], sub_d["dino_sign"])
                    rS_sd, pS_sd = spearmanr(sub_d["vlm_sign"], sub_d["dino_sign"])
                else:
                    rP_sd = pP_sd = rS_sd = pS_sd = np.nan
            else:
                acc_d = rP_sd = pP_sd = rS_sd = pS_sd = np.nan

            # CLIP overall
            valid_clip = big_df["clip_match"].notna()
            if valid_clip.any():
                acc_c = big_df.loc[valid_clip, "clip_match"].mean()
                sub_c = big_df.loc[valid_clip].copy()

                if sub_c["vlm_sign"].nunique() > 1 and sub_c["clip_sign"].nunique() > 1:
                    rP_sc, pP_sc = pearsonr(sub_c["vlm_sign"], sub_c["clip_sign"])
                    rS_sc, pS_sc = spearmanr(sub_c["vlm_sign"], sub_c["clip_sign"])
                else:
                    rP_sc = pP_sc = rS_sc = pS_sc = np.nan
            else:
                acc_c = rP_sc = pP_sc = rS_sc = pS_sc = np.nan


            print("\n===== OVERALL backbone vs VLM (all judges & models) =====")
            print(f"DINO:  N_pairs={valid_dino.sum():4d}, acc={acc_d:.3f}")
            print(
                f"       sign: Pearson r={rP_sd:.3f}, p{_p_str(pP_sd)}; "
                f"Spearman ρ={rS_sd:.3f}, p{_p_str(pS_sd)}"
            )
            print(f"CLIP-I: N_pairs={valid_clip.sum():4d}, acc={acc_c:.3f}")
            print(
                f"       sign: Pearson r={rP_sc:.3f}, p{_p_str(pP_sc)}; "
                f"Spearman ρ={rS_sc:.3f}, p{_p_str(pS_sc)}"
            )


        

if __name__ == "__main__":
    main()
