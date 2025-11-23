#!/usr/bin/env python3
# Computing DINOv3 similarity scores for T2I model outputs

from pathlib import Path
from PIL import Image
import torch
import numpy as np
import pandas as pd
import re

from transformers import AutoImageProcessor, AutoModel

# --------------------------------------------------------------------
# Basic config – ADAPT these three if needed
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]   # repo root
JUDGE_TAG = "gpt_5_nano"                     # whichever judge you used to run t2i
RUN_TAG   = "n6_k5"                          # run subdir
FIG_DIR   = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def find_score_csv_for_dino(t2i_dir: str) -> Path:
    """
    For DINO we just need any judge_scores_*.csv to locate the run dir.
    We ignore dinov3 similarity files here.
    """
    base = (
        ROOT
        / "writing_symbols_outputs_all"
        / t2i_dir
        / f"judge_{JUDGE_TAG}"
        / f"judge_{JUDGE_TAG}"
        / RUN_TAG
    )

    all_matches = sorted(base.glob("judge_scores_*.csv"))
    if not all_matches:
        raise FileNotFoundError(f"No judge_scores_*.csv in {base}")

    non_dino = [p for p in all_matches
                if "dino" not in p.name.lower() and "dinov3" not in p.name.lower()]
    matches = non_dino or all_matches

    if len(matches) > 1:
        print(f"[warn] multiple score files in {base}, using {matches[0].name}")
    return matches[0]

# ====================== DINOv3 SETUP =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device", DEVICE)

DINO_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE)
model.eval()
BATCH_SIZE = 4  # tune based on GPU/CPU memory

@torch.no_grad()
def dinov3_embed(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        emb = outputs.pooler_output
    else:
        emb = outputs.last_hidden_state.mean(dim=1)

    emb = emb.squeeze(0).cpu().numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb

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
        all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)

# ================== PER-MODEL DINO SCORE FUNCTION ===================

pattern = re.compile(r"iter(\d+)_stream(\d+)\.png")

def compute_dino_scores_for_t2i(t2i_dir: str) -> pd.DataFrame:
    """
    Compute DINOv3 similarity to ORIGINAL.png for one T2I model
    and save as judge_scores_dinov3_similarity.csv in that run dir.
    """
    score_csv = find_score_csv_for_dino(t2i_dir)
    run_dir = score_csv.parent
    images_dir = run_dir / "images"

    print(f"[{t2i_dir}] Images dir:", images_dir)

    orig_candidates = sorted(images_dir.glob("*ORIGINAL.png"))
    if not orig_candidates:
        raise FileNotFoundError(f"No *ORIGINAL.png found in {images_dir}")
    orig_path = orig_candidates[0]
    print(f"[{t2i_dir}] Using ORIGINAL: {orig_path.name}")

    orig_emb = dinov3_embed(orig_path).astype(np.float32)

    img_paths, iterations, stream_idxs, filenames = [], [], [], []

    for img_path in sorted(images_dir.glob("*.png")):
        if img_path == orig_path:
            continue
        m = pattern.match(img_path.name)
        if m is None:
            print(f"[{t2i_dir}] [warn] skipping unexpected file: {img_path.name}")
            continue
        iteration = int(m.group(1))
        stream_idx = int(m.group(2))

        img_paths.append(img_path)
        iterations.append(iteration)
        stream_idxs.append(stream_idx)
        filenames.append(img_path.name)

    if not img_paths:
        raise RuntimeError(f"[{t2i_dir}] No candidate images found to score.")

    print(f"[{t2i_dir}] Embedding {len(img_paths)} images in batches of {BATCH_SIZE}...")
    embs = dinov3_embed_batch(img_paths, batch_size=BATCH_SIZE)  # (N, D)

    sims = embs @ orig_emb  # cosine similarity (L2-normalized)

    rows = []
    for it, st, fname, sim in zip(iterations, stream_idxs, filenames, sims):
        sim = float(sim)
        print(f"[{t2i_dir}] DINO score {sim:.4f} for iteration {it} stream {st}")
        rows.append(
            dict(iteration=it, stream_idx=st, dino_score=sim, image_file=fname)
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(["iteration", "stream_idx"])
        [["iteration", "stream_idx", "dino_score", "image_file"]]
    )

    out_path = run_dir / "judge_scores_dinov3_similarity.csv"
    df.to_csv(out_path, index=False)
    print(f"[{t2i_dir}] Saved DINO scores to: {out_path}")
    return df

# =================== RUN FOR ALL 3 T2I MODELS =======================

if __name__ == "__main__":
    t2i_dirs = {
        "flux1_fal": "Flux1",
        "gemini": "Gemini",
        "qwen_image_fal": "Qwen-Image",
    }

    all_dino = []
    for t2i_dir, label in t2i_dirs.items():
        df_model = compute_dino_scores_for_t2i(t2i_dir)
        df_model["model"] = label
        all_dino.append(df_model)

    dino_scores = pd.concat(all_dino, ignore_index=True)
    combined_out = FIG_DIR / "dinov3_similarity_all_models.csv"
    combined_out.parent.mkdir(parents=True, exist_ok=True)
    dino_scores.to_csv(combined_out, index=False)
    print("Combined DINO scores saved to:", combined_out)
