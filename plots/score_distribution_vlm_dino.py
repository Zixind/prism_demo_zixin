from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]

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
               if "dinov3" not in p.name]

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


def join_vlm_and_dino(t2i_dir: str, label: str, judge_tag: str) -> pd.DataFrame:
    """
    Load VLM and DINO scores for one T2I model and one judge,
    and join them on (iteration, stream_idx, image_file).
    """
    vlm_csv = find_vlm_score_csv(t2i_dir, judge_tag)
    vlm = load_vlm_scores(vlm_csv, label)
    dino = load_dino_scores_for_same_run(vlm_csv)

    merged = pd.merge(
        vlm,
        dino,
        on=["iteration", "stream_idx", "image_file"],
        how="inner",
    )
    merged["model"] = label
    merged["judge"] = judge_tag
    return merged


def main():
    # map run dirs -> pretty labels
    t2i_dirs = {
        "flux1_fal": "Flux1",
        "gemini": "Gemini",
        "qwen_image_fal": "Qwen-Image",
    }

    for judge_tag in JUDGE_TAGS:
        print(f"\n=== Judge: {judge_tag} ===")
        for t2i_dir, label in t2i_dirs.items():
            df = join_vlm_and_dino(t2i_dir, label, judge_tag)
            print(f"{label}: {len(df)} points")

            if len(df) > 1:
                corr = np.corrcoef(df["vlm_score"], df["dino_score"])[0, 1]
                print(f"  Pearson r(VLM, DINO) = {corr:.3f}")

            # ---- scatter for THIS judge × THIS T2I model ----
            fig, ax = plt.subplots(figsize=(5.2, 3.5))
            style_axes(ax)

            ax.scatter(
                df["vlm_score"],
                df["dino_score"],
                s=10,
                alpha=0.6,
            )
            # ax.set_xlabel(f"VLM judge score (judge_{judge_tag})")
            if t2i_dir == "flux1_fal":
                ax.set_ylabel("DINOv3 similarity")
            ax.set_xlabel(pretty_judge_name(judge_tag))
            
            # ax.set_title(f"{label}: judge vs DINO")

            ax.grid(True, alpha=0.2)
            fig.tight_layout()

            out_path = FIG_DIR / f"vlm_vs_dino_scatter_{label}_{judge_tag}.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print("  Saved scatter to:", out_path)


if __name__ == "__main__":
    main()
