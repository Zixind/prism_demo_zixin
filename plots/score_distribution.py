from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

ROOT = Path(__file__).resolve().parents[1]
# judge + run you want to analyse
# Run over BOTH judges
JUDGE_TAGS = ["gpt_5_nano", "gpt_4o_mini"]
RUN_TAG    = "n6_k5"   # the run subdir you used

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# T2I dirs and pretty labels
T2I_DIRS = {
    "flux1_fal": "Flux1",
    "gemini": "Gemini",
    "qwen_image_fal": "Qwen-Image",
}


def find_score_csv(t2i_dir: str, judge_tag: str) -> Path:
    """
    t2i_dir: 'flux1_fal', 'gemini', 'qwen_image_fal', ...
    judge_tag: 'gpt_5_nano' or 'gpt_4o_mini'
    Returns the path to *GPT-judge* judge_scores_*.csv for that model/judge/run,
    preferring non-DINO files over DINO similarity ones.
    """
    base = (
        ROOT
        / "writing_symbols_outputs_all"
        / t2i_dir
        / f"judge_{judge_tag}"
        / f"judge_{judge_tag}"
        / RUN_TAG
    )

    # Grab all judge_scores_*.csv files
    all_matches = sorted(base.glob("judge_scores_*.csv"))

    if not all_matches:
        raise FileNotFoundError(f"No judge_scores_*.csv in {base}")

    # Prefer NON-DINO files (GPT judges) over dinov3 similarity files
    non_dino = [
        p for p in all_matches
        if "dino" not in p.name.lower() and "dinov3" not in p.name.lower()
    ]

    if non_dino:
        matches = non_dino
    else:
        # Fallback: if we *only* have DINO scores, use them
        matches = all_matches

    if len(matches) > 1:
        print(f"[warn] multiple non-DINO score files in {base}, using {matches[0].name}")

    return matches[0]


def load_scores(path: Path, label: str) -> pd.DataFrame:
    """
    Load one judge_scores_*.csv and tag it with model label.
    Handles both old 3-col and new 4-col CSVs.
    """
    df = pd.read_csv(
        path,
        engine="python",  # tolerant of odd rows
        header=0,         # first line is header
        names=["iteration", "stream_idx", "score", "image_file"],
    )

    df["model"] = label
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["stream_idx"] = pd.to_numeric(df["stream_idx"], errors="coerce")

    df = df.dropna(subset=["iteration", "stream_idx", "score"])
    return df


# =============== PER-JUDGE ANALYSIS & PLOTS ===============

def analyse_judge(judge_tag: str) -> pd.DataFrame:
    """
    For a given judge_tag, load scores for all T2I models,
    print summary stats, and generate:

      - vlm_score_mean_per_iteration_{judge_tag}.png
      - vlm_score_boxplot_{judge_tag}.png
      - vlm_score_violin_{judge_tag}.png

    Returns combined score DataFrame with columns:
      iteration, stream_idx, score, image_file, model, judge
    """
    print("\n" + "=" * 60)
    print(f"Processing judge: {judge_tag}")
    print("=" * 60)

    # -------- load GPT-judge scores --------
    dfs = []
    for t2i_dir, label in T2I_DIRS.items():
        csv_path = find_score_csv(t2i_dir, judge_tag)
        print(f"[{judge_tag}] Loading scores from {csv_path}")
        df_model = load_scores(csv_path, label)
        dfs.append(df_model)

    scores = pd.concat(dfs, ignore_index=True)
    scores["iteration"]  = scores["iteration"].astype(int)
    scores["stream_idx"] = scores["stream_idx"].astype(int)
    scores["score"]      = scores["score"].astype(float)
    scores["judge"]      = judge_tag

    # -------- summary stats (optional) --------
    summary = (
        scores
        .groupby("model")["score"]
        .agg(["mean", "std", "min", "max", "nunique"])
    )

    max_score = scores["score"].max()
    summary["frac_at_max"] = (
        (scores["score"] == max_score)
        .groupby(scores["model"])
        .mean()
    )

    print(f"\n[Summary stats for judge={judge_tag}]")
    print(summary)

    # -------- per-iteration errorbar plot --------
    fig, ax = plt.subplots(figsize=(5.2, 3.5))

    models = list(T2I_DIRS.values())
    x_vals = np.array(sorted(scores["iteration"].unique()))
    offset = 0.1  # for slight horizontal shift per model

    for i, model in enumerate(models):
        df_m = scores[scores["model"] == model]
        means = df_m.groupby("iteration")["score"].mean().reindex(x_vals)
        stds  = df_m.groupby("iteration")["score"].std().reindex(x_vals)

        xs = x_vals + (i - (len(models)-1)/2) * offset
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=model)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title(f"Per-iteration judge scores (pool=6)\njudge={judge_tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"vlm_score_mean_per_iteration_{judge_tag}.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -------- boxplot --------
    data = [scores[scores["model"] == m]["score"].values for m in models]
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=models, showmeans=True)
    ax.set_ylabel("Score")
    ax.set_title(f"VLM score distribution (judge={judge_tag})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"vlm_score_boxplot_{judge_tag}.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -------- violin plot --------
    fig, ax = plt.subplots()
    ax.violinplot(data, showmeans=True, showextrema=True)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title(f"VLM score distributions (judge={judge_tag})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"vlm_score_violin_{judge_tag}.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)

    return scores





if __name__ == "__main__":
    all_scores = []

    for tag in JUDGE_TAGS:
        scores_tag = analyse_judge(tag)
        all_scores.append(scores_tag)

    # Optional: combined table over both judges
    combined_scores = pd.concat(all_scores, ignore_index=True)
    combined_scores.to_csv(FIG_DIR / "vlm_scores_all_judges.csv", index=False)
    print("\nCombined scores over all judges saved to:",
          FIG_DIR / "vlm_scores_all_judges.csv")


# flux   = load_scores(find_score_csv("flux1_fal"),   "Flux1")
# gemini = load_scores(find_score_csv("gemini"),       "Gemini")
# qwen   = load_scores(find_score_csv("qwen_image_fal"), "Qwen-Image")

# scores = pd.concat([flux, gemini, qwen], ignore_index=True)

# scores["iteration"]  = scores["iteration"].astype(int)
# scores["stream_idx"] = scores["stream_idx"].astype(int)
# scores["score"]      = scores["score"].astype(float)


# summary = (
#     scores
#     .groupby("model")["score"]
#     .agg(["mean", "std", "min", "max", "nunique"])
# )

# max_score = scores["score"].max()
# summary["frac_at_max"] = (
#     (scores["score"] == max_score)
#     .groupby(scores["model"])
#     .mean()
# )

# print(summary)

# round_stats = (
#     scores
#     .groupby(["model", "iteration"])["score"]
#     .agg(["mean", "std"])
#     .reset_index()
# )

# # os.makedirs("figures", exist_ok=True)
# fig, ax = plt.subplots(figsize=(5.2, 3.5))

# models = ["Flux1", "Gemini", "Qwen-Image"]
# colors = {m: None for m in models}  # let mpl pick defaults

# x_vals = np.array(sorted(scores["iteration"].unique()))
# offset = 0.1

# for i, model in enumerate(models):
#     df_m = scores[scores["model"] == model]
#     means = df_m.groupby("iteration")["score"].mean().reindex(x_vals)
#     stds  = df_m.groupby("iteration")["score"].std().reindex(x_vals)

#     xs = x_vals + (i - 1) * offset  # -0.1, 0, +0.1 for 3 models
#     ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=model)

# ax.set_xlabel("Iteration")
# ax.set_ylabel("Score")
# ax.set_title("Per-iteration judge scores (pool=6)")
# ax.legend()
# fig.tight_layout()
# fig.savefig(FIG_DIR/"vlm_score_mean_per_iteration.png", bbox_inches="tight", dpi=300)


# import matplotlib.pyplot as plt

# models = ["Flux1", "Gemini", "Qwen-Image"]
# data = [scores[scores["model"] == m]["score"].values for m in models]

# fig, ax = plt.subplots()
# ax.boxplot(data, labels=models, showmeans=True)
# ax.set_ylabel("Score")
# # ax.set_title("Distribution of VLM scores per T2I model")
# fig.tight_layout()
# fig.savefig(FIG_DIR/"vlm_score_boxplot.png", bbox_inches="tight", dpi=300)


# fig, ax = plt.subplots()
# parts = ax.violinplot(data, showmeans=True, showextrema=True)
# ax.set_xticks(range(1, len(models) + 1))
# ax.set_xticklabels(models)
# ax.set_ylabel("Score")
# # ax.set_title("VLM score distributions (judge on single image)")
# fig.tight_layout()
# fig.savefig(FIG_DIR/"vlm_score_violin.png", bbox_inches="tight", dpi=300)



# # pip install "transformers>=4.56.0" torch pillow
# from pathlib import Path
# from PIL import Image
# import torch
# import numpy as np
# import pandas as pd
# import re

# from transformers import AutoImageProcessor, AutoModel

# # --------------------------------------------------------------------
# # Existing: ROOT, JUDGE_TAG, RUN_TAG, find_score_csv, load_scores
# # --------------------------------------------------------------------


# # ====================== DINOv3 SETUP =========================

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print('Device {}'.format(DEVICE))
# DINO_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"  # adjustable facebook/dinov3-vit7b16-pretrain-lvd1689m

# processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
# model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE)
# model.eval()
# BATCH_SIZE = 4  # tune based on GPU/CPU memory

# @torch.no_grad()
# def dinov3_embed(img_path: Path) -> np.ndarray:
#     """Return a 1D DINOv3 embedding for a single image file."""
#     img = Image.open(img_path).convert("RGB")
#     inputs = processor(images=img, return_tensors="pt").to(DEVICE)

#     outputs = model(**inputs)

#     if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
#         emb = outputs.pooler_output
#     else:
#         emb = outputs.last_hidden_state.mean(dim=1)

#     emb = emb.squeeze(0).cpu().numpy()
#     emb = emb / (np.linalg.norm(emb) + 1e-8)  # L2-normalize
#     return emb

# @torch.no_grad()
# def dinov3_embed_batch(img_paths, batch_size: int = BATCH_SIZE) -> np.ndarray:
#     """
#     Embed a list of image paths with DINOv3 in batches.
#     Returns an array of shape (N, D), L2-normalized per row.
#     """
#     all_embs = []

#     for start in range(0, len(img_paths), batch_size):
#         batch_paths = img_paths[start:start + batch_size]
#         imgs = [Image.open(p).convert("RGB") for p in batch_paths]

#         inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
#         outputs = model(**inputs)

#         if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
#             emb = outputs.pooler_output       # (B, D)
#         else:
#             emb = outputs.last_hidden_state.mean(dim=1)  # (B, D)

#         emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
#         all_embs.append(emb.cpu().numpy())

#     return np.concatenate(all_embs, axis=0)  # (N, D)

# # ================== PER-MODEL DINO SCORE FUNCTION ===================

# pattern = re.compile(r"iter(\d+)_stream(\d+)\.png")

# def compute_dino_scores_for_t2i(t2i_dir: str) -> pd.DataFrame:
#     """
#     Compute DINOv3 similarity to ORIGINAL.png for one T2I model
#     and save as judge_scores_dinov3_similarity.csv in that run dir.
#     Returns the DataFrame with columns:
#       iteration, stream_idx, score, image_file
#     """
#     score_csv = find_score_csv(t2i_dir)
#     run_dir = score_csv.parent
#     images_dir = run_dir / "images"

#     print(f"[{t2i_dir}] Images dir:", images_dir)

#     # ORIGINAL image (reference)
#     orig_candidates = sorted(images_dir.glob("*ORIGINAL.png"))
#     if not orig_candidates:
#         raise FileNotFoundError(f"No *ORIGINAL.png found in {images_dir}")
#     orig_path = orig_candidates[0]
#     print(f"[{t2i_dir}] Using ORIGINAL: {orig_path.name}")

#     orig_emb = dinov3_embed(orig_path)
#     orig_emb = orig_emb.astype(np.float32)

#     rows = []
#     # 2) Collect all candidate images + parsed metadata
#     img_paths = []
#     iterations = []
#     stream_idxs = []
#     filenames = []


#     for img_path in sorted(images_dir.glob("*.png")):
#         if img_path == orig_path:
#             # keep ORIGINAL out of the score file, to mirror GPT judge format
#             continue

#         m = pattern.match(img_path.name)
#         if m is None:
#             print(f"[{t2i_dir}] [warn] skipping unexpected file: {img_path.name}")
#             continue

#         iteration = int(m.group(1))   # "01" -> 1
#         stream_idx = int(m.group(2))  # "00" -> 0

#         img_paths.append(img_path)
#         iterations.append(iteration)
#         stream_idxs.append(stream_idx)
#         filenames.append(img_path.name)

#     if not img_paths:
#         raise RuntimeError(f"[{t2i_dir}] No candidate images found to score.")

#     # 3) Batched DINO embeddings for all candidates
#     print(f"[{t2i_dir}] Embedding {len(img_paths)} images in batches of {BATCH_SIZE}...")
#     embs = dinov3_embed_batch(img_paths, batch_size=BATCH_SIZE)  # (N, D)

#     # 4) Cosine similarity vs ORIGINAL in one matmul
#     # orig_emb: (D,), embs: (N, D) -> sims: (N,)
#     sims = embs @ orig_emb  # already L2-normalized, so this is cosine

#     rows = []
#     for it, st, fname, sim in zip(iterations, stream_idxs, filenames, sims):
#         sim = float(sim)
#         print(f"[{t2i_dir}] DINO score {sim:.4f} for iteration {it} stream {st}")
#         rows.append(
#             {
#                 "iteration": it,
#                 "stream_idx": st,
#                 "dino_score": sim,
#                 "image_file": fname,
#             }
#         )

#     df = (
#         pd.DataFrame(rows)
#         .sort_values(["iteration", "stream_idx"])
#         [["iteration", "stream_idx", "dino_score", "image_file"]]
#     )

#     out_path = run_dir / "judge_scores_dinov3_similarity.csv"
#     df.to_csv(out_path, index=False)
#     print(f"[{t2i_dir}] Saved DINO scores to: {out_path}")

#     return df


# # =================== RUN FOR ALL 3 T2I MODELS =======================

# t2i_dirs = {
#     "flux1_fal": "Flux1",
#     "gemini": "Gemini",
#     "qwen_image_fal": "Qwen-Image",
# }

# all_dino = []

# for t2i_dir, label in t2i_dirs.items():
#     df_model = compute_dino_scores_for_t2i(t2i_dir)
#     df_model["model"] = label
#     all_dino.append(df_model)

# # Optional: combined DINO similarity table across all models
# dino_scores = pd.concat(all_dino, ignore_index=True)
# combined_out = FIG_DIR / "dinov3_similarity_all_models.csv"
# combined_out.parent.mkdir(parents=True, exist_ok=True)
# dino_scores.to_csv(combined_out, index=False)
# print("Combined DINO scores saved to:", combined_out)
