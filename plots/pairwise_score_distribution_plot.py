#!/usr/bin/env python3
"""
vlm_vs_dino_pairwise_confusion.py

Make 2x2 confusion-style heatmaps comparing pairwise VLM judge vs DINO:

    rows = VLM winner (A/B)
    cols = DINO winner (A/B)

Uses the combined table written by pairwise_dino_vs_vlm.py:
    figures/vlm_pairs_with_dino_all_models.csv
"""
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import re
from matplotlib.colors import LinearSegmentedColormap
JUDGES   = ["gpt_4o_mini", "gpt_5_nano"]
T2I_TAGS = ["flux1_fal_duel", "gemini_duel", "qwen_image_fal_duel"] 
# Make a darker variant of the "Blues" colormap
_blues = plt.cm.Blues
Blues_darker = LinearSegmentedColormap.from_list(
    "Blues_darker",
    _blues(np.linspace(0.3, 1.0, 256))  # 0.3→1.0 trims the pale colors
)

# # ---- global paper-style settings ----
# mpl.rcParams.update({
#     "figure.dpi": 300,
#     "axes.facecolor": "white",
#     "axes.edgecolor": "black",
#     "axes.spines.right": False,
#     "axes.spines.top": False,
#     "axes.grid": True,
#     "grid.color": "0.9",
#     "grid.linewidth": 0.8,
#     "axes.axisbelow": True,
#     "font.size": 11,
#     "axes.titlesize": 13,
#     "axes.labelsize": 12,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
# })


# def style_axes(ax):
#     """Apply paper-style tweaks to an axes."""
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.grid(True, color="0.9", linewidth=0.8)


# ROOT = Path(__file__).resolve().parents[1]
# FIG_DIR = ROOT / "figures"
# FIG_DIR.mkdir(parents=True, exist_ok=True)


# def load_pairs_with_dino(csv_path: Path) -> pd.DataFrame:
#     """
#     Load the combined VLM–DINO pairwise table.
#     Expected columns (from pairwise_dino_vs_vlm.py):

#       round, pair_idx, i, j, imgA, imgB,
#       scoreA, scoreB, winner, conf,
#       dino_scoreA, dino_scoreB, dino_margin, dino_winner, match,
#       vlm_margin, vlm_sign, dino_sign,
#       t2i_dir, model_label
#     """
#     df = pd.read_csv(csv_path)
#     # Ensure these columns exist; if not, raise a helpful error
#     required = ["winner", "dino_winner"]
#     for c in required:
#         if c not in df.columns:
#             raise ValueError(f"Column '{c}' missing in {csv_path}; "
#                              f"did you run pairwise_dino_vs_vlm.py?")
#     return df


# def compute_2x2(df: pd.DataFrame):
#     """
#     Build a 2x2 matrix:

#         rows = VLM winner  (0=A, 1=B)
#         cols = DINO winner (0=A, 1=B)

#     Returns:
#         mat_counts: 2x2 integer counts
#         mat_frac:   2x2 fractions over all valid pairs
#     """
#     # Keep only rows where both winners are defined (not NaN)
#     mask = df["winner"].notna() & df["dino_winner"].notna()
#     sub = df.loc[mask].copy()

#     mat = np.zeros((2, 2), dtype=int)
#     for _, row in sub.iterrows():
#         v = int(row["winner"])       # 0 or 1
#         d = int(row["dino_winner"])  # 0 or 1
#         if v in (0, 1) and d in (0, 1):
#             mat[v, d] += 1

#     total = mat.sum()
#     if total == 0:
#         frac = np.zeros_like(mat, dtype=float)
#     else:
#         frac = mat.astype(float) / total

#     return mat, frac, total


# def plot_confusion_2x2(mat_counts, mat_frac, total, title, out_path: Path):
#     """
#     Plot a 2x2 confusion-style heatmap with counts and percentages.
#     """
#     fig, ax = plt.subplots(figsize=(3.2, 3.0))
#     style_axes(ax)

#     im = ax.imshow(mat_frac, vmin=0.0, vmax=1.0, origin="upper")

#     # Tick labels
#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#     ax.set_xticklabels(["DINO: A", "DINO: B"])
#     ax.set_yticklabels(["VLM: A", "VLM: B"])

#     ax.set_xlabel("DINO winner")
#     ax.set_ylabel("VLM winner")
#     ax.set_title(title)

#     # Annotate cells with "count (xx%)"
#     for i in range(2):
#         for j in range(2):
#             count = mat_counts[i, j]
#             frac = mat_frac[i, j]
#             txt = f"{count}\n({frac*100:0.1f}%)" if total > 0 else "0\n(0.0%)"
#             ax.text(
#                 j, i, txt,
#                 ha="center", va="center",
#                 fontsize=9,
#             )

#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label("Fraction of pairs")

#     fig.tight_layout()
#     fig.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print(f"[plot] saved {out_path}")


# def main():
#     # ------------------------------------------------------------------
#     # 1) Load combined VLM–DINO pairwise table
#     # ------------------------------------------------------------------
#     combined_csv = FIG_DIR / "vlm_pairs_with_dino_all_models_pairs.csv"
#     if not combined_csv.is_file():
#         raise FileNotFoundError(
#             f"{combined_csv} not found.\n"
#             "Run pairwise_dino_vs_vlm.py first to create it."
#         )

#     df = load_pairs_with_dino(combined_csv)

#     # If you stored per-model label, use that:
#     if "model_label" in df.columns:
#         model_col = "model_label"
#     elif "model" in df.columns:
#         model_col = "model"
#     else:
#         model_col = None

#     # ------------------------------------------------------------------
#     # 2) Overall confusion matrix (all models, all objects)
#     # ------------------------------------------------------------------
#     mat_all, frac_all, total_all = compute_2x2(df)
#     plot_confusion_2x2(
#         mat_all,
#         frac_all,
#         total_all,
#         title=f"VLM vs DINO (all models, N={total_all})",
#         out_path=FIG_DIR / "vlm_vs_dino_confusion_overall.png",
#     )

#     # ------------------------------------------------------------------
#     # 3) Per-model confusion matrices (Flux1 / Gemini / Qwen-Image)
#     # ------------------------------------------------------------------
#     if model_col is not None:
#         for model_name in sorted(df[model_col].unique()):
#             sub = df[df[model_col] == model_name]
#             mat_m, frac_m, total_m = compute_2x2(sub)
#             title = f"{model_name} (N={total_m})"
#             out_path = FIG_DIR / f"vlm_vs_dino_confusion_{model_name.replace(' ', '_')}.png"
#             plot_confusion_2x2(mat_m, frac_m, total_m, title, out_path)

# --- paths ---
ROOT = Path(__file__).resolve().parents[1]  # prism_demo/
PAIR_DIR = ROOT / "figures" / "per_obj_judge_model"
OUT_DIR  = ROOT / "figures" / "agreement_mats_per_obj"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# use your actual tags
# JUDGES   = ["gpt_4o_mini", "gpt_5_nano"]
# T2I_TAGS = ["flux1_fal_duel", "gemini_duel", "qwen_image_fal_duel"]

# global style (optional, like your other plots)
mpl.rcParams.update({
    "figure.dpi": 300,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": False,
    "font.size": 11,
})

def agreement_rate(M: np.ndarray) -> float:
    total = M.sum()
    if total == 0:
        return np.nan
    return float(np.trace(M) / total)

def iter_pair_files(judge_tag: str, t2i_tag: str):
    """
    Yield (obj_name, df) for each per-object CSV for one (judge, model).
    obj_name is parsed from the filename:
        pairs_<OBJ>_judge-{judge}_t2i-{t2i}.csv
    """
    pattern = f"pairs_*_judge-{judge_tag}_t2i-{t2i_tag}.csv"
    files = sorted(PAIR_DIR.glob(pattern))
    if not files:
        raise RuntimeError(f"No CSVs for judge={judge_tag}, t2i={t2i_tag}")

    for f in files:
        # filename: pairs_<OBJ>_judge-..._t2i-...csv
        stem = f.stem  # "pairs_01_Anime_..._judge-gpt_4_1_mini_t2i-flux1_fal_duel"
        before_judge = stem.split("_judge-")[0]       # "pairs_01_Anime_..."
        obj_name = before_judge.replace("pairs_", "") # "01_Anime_..."

        df = pd.read_csv(f)
        yield obj_name, df

def add_preference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use existing sign columns to create {vlm_pref, clip_pref, dino_pref} ∈ {+1, -1}:
      +1 -> candidate A wins
      -1 -> candidate B wins
    """
    df = df.copy()

    # These columns already exist in your CSV
    # vlm_sign, dino_sign, clip_sign are floats in {+1.0, -1.0, 0.0}
    df["vlm_pref"]  = df["vlm_sign"].astype(int)
    df["clip_pref"] = df["clip_sign"].astype(int)
    df["dino_pref"] = df["dino_sign"].astype(int)

    # drop ties (sign == 0)
    for col in ["vlm_pref", "clip_pref", "dino_pref"]:
        df = df[df[col] != 0]

    return df


def compute_agreement_matrix(df: pd.DataFrame,
                             backbone_col: str) -> np.ndarray:
    """
    Build 2×2 matrix:
      rows = VLM pref (A, B)
      cols = backbone pref (A, B)
    using +1 (A) / -1 (B) encoding.
    """
    V = df["vlm_pref"].to_numpy()
    B = df[backbone_col].to_numpy()

    # map +1 -> 0 (A), -1 -> 1 (B)
    row_idx = (V == -1).astype(int)  # 0 = A, 1 = B
    col_idx = (B == -1).astype(int)

    M = np.zeros((2, 2), dtype=int)
    for r, c in zip(row_idx, col_idx):
        M[r, c] += 1
    return M
def pretty_judge_label(judge_tag: str) -> str:
    """Nicely formatted judge name for axis labels."""
    mapping = {
        "gpt_4_1_mini": "GPT-4.1 mini",
        "gpt_4o_mini": "GPT-4o mini",
        "gpt_5_nano": "GPT-5 nano",
    }
    return mapping.get(judge_tag, judge_tag)
def plot_agreement_matrix(M: np.ndarray,
                          judge_tag: str,
                          t2i_tag: str,
                          obj_name: str,
                          backbone_name: str,
                          out_path: Path,
                          add_colorbar: bool = False):
    total = M.sum()
    if total == 0:
        return

    frac = M.astype(float) / total
    judge_label = pretty_judge_label(judge_tag)
    agree = agreement_rate(M)  # scalar in [0,1]

    fig, ax = plt.subplots(figsize=(2.8, 2.6))

    # darker blue heatmap on fractions
    vmax = M.max() if M.max() > 0 else 1
    im = ax.imshow(M, cmap=Blues_darker, vmin=0, vmax=vmax)
    ax.set_aspect("equal")

    # ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["A", "B"])
    ax.set_yticklabels(["A", "B"])

    ax.set_xlabel(f"{judge_label}")      # columns
    ax.set_ylabel(f"{backbone_name}")    # rows

    # annotate cells with fractions
    for i in range(2):
        for j in range(2):
            # pct = frac[i, j] * 100.0
            # text_color = "white" if frac[i, j] > 0.5 else "black"
            # ax.text(
            #     j, i,
            #     f"{pct:0.1f}%",
            #     ha="center", va="center",
            #     fontsize=9,
            #     color=text_color,
            # )
            count = M[i, j]
            # use fraction just to decide white vs black text
            text_color = "white" if frac[i, j] > 0.5 else "black"
            ax.text(
                j, i,
                f"{count}",       # <-- only the raw number
                ha="center", va="center",
                fontsize=9,
                color=text_color,
            )

    # ---- ONLY SOME PANELS GET A COLORBAR ----
    # if add_colorbar:
    #     cbar = fig.colorbar(
    #         im,
    #         ax=ax,
    #         fraction=0.046,
    #         pad=0.10,
    #         location="left",
    #     )
    #     # cbar.set_label("Fraction of pairs")
    #     pos = cbar.ax.get_position()  # [x0, y0, width, height] in figure coords
    #     cbar.ax.set_position([
    #         pos.x0 - 0.12,  # subtract more to push it even lefter
    #         pos.y0,
    #         pos.width,
    #         pos.height,
    #     ])

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def wants_colorbar(judge_tag: str, t2i_tag: str, backbone_name: str) -> bool:
    # Only Flux1 + GPT-4o mini panels get bars (for CLIP-I and DINOv3)
    return (
        judge_tag == "gpt_4o_mini"
        and t2i_tag == "flux1_fal_duel"
        and backbone_name in {"CLIP-I", "DINOv3"}
    )


def main():
    for judge in JUDGES:
        for t2i in T2I_TAGS:
            print(f"\n=== judge={judge}, t2i={t2i} ===")

            for obj_name, df_raw in iter_pair_files(judge, t2i):
                df = add_preference_columns(df_raw)
                safe_obj = re.sub(r"[^A-Za-z0-9._-]+", "_", obj_name)

                # --- CLIP-I agreement ---
                M_clip = compute_agreement_matrix(df, "clip_pref")
                out_clip = OUT_DIR / (
                    f"agree_mat_clip_judge-{judge}_t2i-{t2i}_obj-{safe_obj}.png"
                )
                plot_agreement_matrix(
                    M_clip,
                    judge_tag=judge,
                    t2i_tag=t2i,
                    obj_name=obj_name,
                    backbone_name="CLIP-I",
                    out_path=out_clip,
                    add_colorbar=wants_colorbar(judge, t2i, "CLIP-I"),
                )

                # --- DINOv3 agreement ---
                M_dino = compute_agreement_matrix(df, "dino_pref")
                out_dino = OUT_DIR / (
                    f"agree_mat_dino_judge-{judge}_t2i-{t2i}_obj-{safe_obj}.png"
                )
                plot_agreement_matrix(
                    M_dino,
                    judge_tag=judge,
                    t2i_tag=t2i,
                    obj_name=obj_name,
                    backbone_name="DINOv3",
                    out_path=out_dino,
                    add_colorbar=wants_colorbar(judge, t2i, "DINOv3"),
                )


if __name__ == "__main__":
    main()
