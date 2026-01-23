#Assuming dino and CLIP-I scores are already computed, plotting dino and vlm judge score distributions
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, spearmanr   # pearson
ROOT = Path(__file__).resolve().parents[1]
from matplotlib.ticker import MaxNLocator  # add near your imports
from scipy.stats import pearsonr, spearmanr, gaussian_kde

# Run over BOTH judges in one go
JUDGE_TAGS = ["gpt_4o_mini", "gpt_5_nano"]
RUN_TAG    = "n6_k5"

PRETTY_JUDGE = {
    "gpt_5_nano": "GPT-5 nano",
    "gpt_4o_mini": "GPT-4o mini",
}

def pretty_judge_name(tag: str) -> str:
    return PRETTY_JUDGE.get(tag, tag)

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- global paper-style settings ----
mpl.rcParams.update({
    "figure.dpi": 300,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.color": "0.9",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})
def style_axes(ax):
    """Apply paper-style tweaks to an axes."""
    # reinforce in case rcParams change later
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.9", linewidth=0.8)
def find_vlm_score_csv(t2i_dir: str, judge_tag: str) -> Path:
    """
    Find the *VLM judge* CSV for a given T2I model + judge.
    Excludes the DINO csv (judge_scores_dinov3_similarity.csv).
    """
    base = (
        ROOT
        / "writing_symbols_outputs_all"
        / t2i_dir
        / f"judge_{judge_tag}"
        / f"judge_{judge_tag}"
        / RUN_TAG
    )
    # Ignore dinov3 similarity file
    matches = [p for p in base.glob("judge_scores_*.csv")
               if "dinov3" not in p.name
               and "clipi" not in p.name.lower()]

    if not matches:
        raise FileNotFoundError(f"No VLM judge_scores_*.csv in {base}")
    if len(matches) > 1:
        print(f"[warn] multiple VLM score files in {base}, using {matches[0].name}")
    return matches[0]


def load_vlm_scores(path: Path, label: str) -> pd.DataFrame:
    """
    Load VLM judge scores.
    CSV header: iteration,stream_idx,score,image_file
    We rename 'score' -> 'vlm_score'.
    """
    df = pd.read_csv(
        path,
        engine="python",
        header=0,
        names=["iteration", "stream_idx", "vlm_score", "image_file"],
    )
    df["model"] = label
    df["vlm_score"] = pd.to_numeric(df["vlm_score"], errors="coerce")
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["stream_idx"] = pd.to_numeric(df["stream_idx"], errors="coerce")
    df = df.dropna(subset=["iteration", "stream_idx", "vlm_score"])
    df["iteration"] = df["iteration"].astype(int)
    df["stream_idx"] = df["stream_idx"].astype(int)
    return df


def load_dino_scores_for_same_run(vlm_csv_path: Path) -> pd.DataFrame:
    """
    For the same run directory as vlm_csv_path, load DINO scores
    from judge_scores_dinov3_similarity.csv.
    Expected header: iteration,stream_idx,score,image_file
    We rename 'score' -> 'dino_score'.
    """
    run_dir = vlm_csv_path.parent
    dino_path = run_dir / "judge_scores_dinov3_similarity.csv"
    if not dino_path.exists():
        raise FileNotFoundError(f"No DINO csv at {dino_path}")

    df = pd.read_csv(dino_path)
    # standardize column names
    if "dino_score" in df.columns:
        pass
    elif "score" in df.columns:
        df = df.rename(columns={"score": "dino_score"})
    else:
        raise KeyError(f"No 'score' or 'dino_score' column in {dino_path}")

    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["stream_idx"] = pd.to_numeric(df["stream_idx"], errors="coerce")
    df["dino_score"] = pd.to_numeric(df["dino_score"], errors="coerce")
    df = df.dropna(subset=["iteration", "stream_idx", "dino_score"])
    df["iteration"] = df["iteration"].astype(int)
    df["stream_idx"] = df["stream_idx"].astype(int)
    return df

#loader for CLIP-I csv
def load_clipi_scores_for_same_run(vlm_csv_path: Path) -> pd.DataFrame:
    """
    For the same run directory as vlm_csv_path, load CLIP-I scores
    from judge_scores_clipi_similarity.csv.
    Expected header: iteration,stream_idx,clipI_score,image_file
    or iteration,stream_idx,score,image_file (then rename).
    """
    run_dir = vlm_csv_path.parent
    clip_path = run_dir / "judge_scores_clipi_similarity.csv"
    if not clip_path.exists():
        raise FileNotFoundError(f"No CLIP-I csv at {clip_path}")

    df = pd.read_csv(clip_path)

    if "clipI_score" in df.columns:
        pass
    elif "score" in df.columns:
        df = df.rename(columns={"score": "clipI_score"})
    else:
        raise KeyError(f"No 'score' or 'clipI_score' column in {clip_path}")

    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["stream_idx"] = pd.to_numeric(df["stream_idx"], errors="coerce")
    df["clipI_score"] = pd.to_numeric(df["clipI_score"], errors="coerce")
    df = df.dropna(subset=["iteration", "stream_idx", "clipI_score"])
    df["iteration"] = df["iteration"].astype(int)
    df["stream_idx"] = df["stream_idx"].astype(int)
    return df

def join_vlm_dino_clip(t2i_dir: str, label: str, judge_tag: str) -> pd.DataFrame:
    """
    Load VLM, DINO, and CLIP-I scores for one T2I model and one judge,
    and join them on (iteration, stream_idx, image_file).
    """
    vlm_csv = find_vlm_score_csv(t2i_dir, judge_tag)
    vlm  = load_vlm_scores(vlm_csv, label)
    dino = load_dino_scores_for_same_run(vlm_csv)
    clip = load_clipi_scores_for_same_run(vlm_csv)

    merged = pd.merge(
        vlm,
        dino,
        on=["iteration", "stream_idx", "image_file"],
        how="inner",
    )
    merged = pd.merge(
        merged,
        clip,
        on=["iteration", "stream_idx", "image_file"],
        how="inner",
    )
    merged["model"] = label
    merged["judge"] = judge_tag
    return merged

def _p_str(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"

# def make_scatter(df, x_col, y_col, xlabel, ylabel, title_suffix, out_name, add_label_on_y=False):
#     """linear version"""
#     fig, ax = plt.subplots(figsize=(5.2, 3.5))
#     style_axes(ax)

#     # integer x-axis limits / ticks (good for Likert-y VLM scores)
#     xmin = int(np.floor(df[x_col].min()))
#     xmax = int(np.ceil(df[x_col].max()))
#     ax.set_xlim(xmin - 0.1, xmax + 0.1)
#     ax.set_xticks(range(xmin, xmax + 1))
#     ax.tick_params(axis="both", which="major", labelsize=14)

#     ax.scatter(df[x_col], df[y_col], s=10, alpha=0.6)

#     # regression line
#     if df[x_col].nunique() > 1 and df[y_col].nunique() > 1:
#         z = np.polyfit(df[x_col], df[y_col], 1)
#         xgrid = np.linspace(df[x_col].min(), df[x_col].max(), 100)
#         yhat = np.polyval(z, xgrid)
#         ax.plot(xgrid, yhat, linewidth=1)

#         rS, pS = spearmanr(df[x_col], df[y_col])
#         text = f"Spearman ρ = {rS:.3f}, p {_p_str(pS)}"
#         ax.text(
#             0.5, 1.03,
#             text,
#             transform=ax.transAxes,
#             ha="center",
#             va="bottom",
#             fontsize=17
#         )

#     ax.set_xlabel(xlabel, fontsize=18)
#     if add_label_on_y:
#         ax.set_ylabel(ylabel, fontsize=18)

#     ax.grid(True, alpha=0.2)
#     fig.tight_layout()
#     out_path = FIG_DIR / out_name
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)
#     print(f"  Saved scatter to: {out_path}")

def main():
    # map run dirs -> pretty labels
    t2i_dirs = {
        "flux1_fal": "Flux1",
        "gemini": "Gemini",
        "qwen_image_fal": "Qwen-Image",
    }

    for judge_tag in JUDGE_TAGS:
        print(f"\n=== Judge: {judge_tag} ===")
        pretty_j = pretty_judge_name(judge_tag)

        for t2i_dir, label in t2i_dirs.items():
            df = join_vlm_dino_clip(t2i_dir, label, judge_tag)
            print(f"{label}: {len(df)} points")

            if df["vlm_score"].nunique() > 1 and df["dino_score"].nunique() > 1:
                rS_dino, pS_dino = spearmanr(df["vlm_score"], df["dino_score"])
                print(f"  Spearman ρ(VLM, DINO) = {rS_dino:.3f}, p {_p_str(pS_dino)}")
            else:
                rS_dino = pS_dino = np.nan
                print("  Not enough variation for VLM vs DINO correlation")

            if df["vlm_score"].nunique() > 1 and df["clipI_score"].nunique() > 1:
                rS_clip, pS_clip = spearmanr(df["vlm_score"], df["clipI_score"])
                print(f"  Spearman ρ(VLM, CLIP-I) = {rS_clip:.3f}, p {_p_str(pS_clip)}")
            else:
                rS_clip = pS_clip = np.nan
                print("  Not enough variation for VLM vs CLIP-I correlation")


            make_scatter(
                df,
                x_col="vlm_score",
                y_col="dino_score",
                xlabel=pretty_j,
                ylabel="DINO",
                title_suffix=f"{label} / {pretty_j}",
                out_name=f"vlm_vs_dino_scatter_{label}_{judge_tag}.png",
                add_label_on_y=(t2i_dir == "flux1_fal"),  # only leftmost column gets y-label in grids
            )
            # --- scatter: VLM vs CLIP-I ---
            make_scatter(
                df,
                x_col="vlm_score",
                y_col="clipI_score",
                xlabel=pretty_j,
                ylabel="CLIP-I",
                title_suffix=f"{label} / {pretty_j}",
                out_name=f"vlm_vs_clipi_scatter_{label}_{judge_tag}.png",
                add_label_on_y=(t2i_dir == "flux1_fal"),
            )

def make_scatter(df, x_col, y_col, xlabel, ylabel, title_suffix, out_name, add_label_on_y=False):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    style_axes(ax)

    # Compute KDE only if we have enough points
    if len(df) > 1:
        # 2D KDE over (x, y)
        x = df[x_col].values
        y = df[y_col].values
        values = np.vstack([x, y])
        kde = gaussian_kde(values)

        # Grid over data range
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        # Plot density as a heatmap
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
            alpha=0.85,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Density", fontsize=12)

    # Scatter on top (so points are visible over density)
    ax.scatter(df[x_col], df[y_col], s=8, alpha=0.4)

    # Integer x-axis limits / ticks (good for Likert-y VLM scores)
    xmin = int(np.floor(df[x_col].min()))
    xmax = int(np.ceil(df[x_col].max()))
    ax.set_xlim(xmin - 0.1, xmax + 0.1)
    ax.set_xticks(range(xmin, xmax + 1))
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Correlation text (no linear regression line anymore)
    if df[x_col].nunique() > 1 and df[y_col].nunique() > 1:
        rS, pS = spearmanr(df[x_col], df[y_col])
        text = f"Spearman ρ = {rS:.3f}, p {_p_str(pS)}"
        ax.text(
            0.5, 1.03,
            text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=17
        )

    ax.set_xlabel(xlabel, fontsize=18)
    if add_label_on_y:
        ax.set_ylabel(ylabel, fontsize=18)

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path = FIG_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved scatter+KDE to: {out_path}")


if __name__ == "__main__":
    main()
