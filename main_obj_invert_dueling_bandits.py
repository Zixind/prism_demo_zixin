#!/usr/bin/env python3
# PRISM-DUEL for main_obj_invert.py with dueling bandits (pairwise A/B)
# DreamBooth-style multi-reference folders.

import warnings
warnings.filterwarnings("ignore")

import os, random, time, re, math, csv
from pathlib import Path
import argparse
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

from system_prompts import *
from loggers_duel import WandBLoggerDuel as WandBLogger
from judges import load_judge
from conversers import load_attack_and_target_models
from common import *

from utils import (
    SeedManager,
    save_vlm_pairs_csv,
    save_duel_trace_csv_compact,
    save_duel_trace_csv_detailed,
    save_caption_vs_prompt,
)

from prism_duel_text_prior import (
    sanitize_tag,
    auto_caption_goal_image,
)

from prism_duel_bandits import (
    pick_pairs_thompson,
    pick_pairs_ucb,
    pick_pairs_eps_greedy,
    pick_pairs_copeland_ucb,
)

from target_adapters import make_target_adapter, to_pil_list
from language_models import OpenAITextEmbedder

EMBEDDER = OpenAITextEmbedder()


def split_csv(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out or None
# ---------------- CSV logging for candidate images ----------------

def append_prompt_image_caption_csv(csv_path: Path, rows: list, header: Optional[list] = None):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if is_new and header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)

def save_candidate_images_and_log(
    *,
    out_dir: str,
    obj: str,
    model_tag: str,
    judge_tag: str,
    iteration: int,
    candidate_prompts: list,
    candidate_pils: list,          # list[PIL.Image.Image or None]
    goal_caption: str,
):
    img_dir = Path(out_dir) / "candidate_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(out_dir) / "prompt_image_caption.csv"
    header = ["obj", "iteration", "stream_idx", "image_file", "model", "judge_tag", "goal_caption", "prompt"]

    rows = []
    for k, (prompt, img) in enumerate(zip(candidate_prompts, candidate_pils)):
        if img is None:
            continue
        if getattr(img, "mode", None) and img.mode != "RGB":
            img = img.convert("RGB")

        fname = f"{obj}_iter{iteration:03d}_stream{k:02d}.png"
        fpath = img_dir / fname
        img.save(fpath)

        rows.append([
            obj,
            iteration,
            k,
            str(fpath.relative_to(Path(out_dir))),
            model_tag,
            judge_tag,
            goal_caption,
            prompt,
        ])

    append_prompt_image_caption_csv(csv_path, rows, header=header)

from PIL import Image, ImageDraw, ImageFont

def save_side_by_side(
    left_path: str,
    right_path: str,
    out_path: str,
    *,
    pad: int = 16,
    bg=(255, 255, 255),
    left_title: str = "ORIGINAL",
    right_title: str = "PRISM-DUEL",
    add_titles: bool = True,
):
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    # make same height (keep aspect)
    H = max(left.height, right.height)

    def resize_to_h(img, H):
        if img.height == H:
            return img
        w = int(round(img.width * (H / img.height)))
        return img.resize((w, H), Image.BICUBIC)

    left = resize_to_h(left, H)
    right = resize_to_h(right, H)

    title_h = 0
    if add_titles:
        title_h = 40  # simple fixed header

    W = left.width + right.width + pad * 3
    canvas = Image.new("RGB", (W, H + pad * 2 + title_h), bg)

    # paste
    x0 = pad
    y0 = pad + title_h
    canvas.paste(left, (x0, y0))
    canvas.paste(right, (x0 + left.width + pad, y0))

    # optional labels
    if add_titles:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((x0, pad), left_title, fill=(0, 0, 0), font=font)
        draw.text((x0 + left.width + pad, pad), right_title, fill=(0, 0, 0), font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path

# ---------------- Multi-ref loading ----------------

EXTS = (".jpg", ".jpeg", ".png", ".webp")

def _collect_goal_images(goal_dir: str, obj: str) -> List[str]:
    """
    Supports:
      - goal_dir/obj/   (folder with images)
      - goal_dir/obj.png (single file)
      - goal_dir/obj    (stem; try extensions)
    """
    base = os.path.join(goal_dir, obj)

    if os.path.isdir(base):
        img_paths = [
            os.path.join(base, f)
            for f in os.listdir(base)
            if f.lower().endswith(EXTS)
        ]
        img_paths.sort()
        return img_paths

    # file or stem
    candidates = [base] + [base + ext for ext in EXTS]
    img_paths = [p for p in candidates if os.path.isfile(p)]
    img_paths.sort()
    return img_paths

def _split_refs(img_paths: List[str], num_val: int) -> Tuple[List[str], List[str]]:
    """
    Deterministic split: last num_val are validation, rest are reference.
    If too few images, validation may be empty.
    """
    if num_val <= 0 or len(img_paths) <= 1:
        return img_paths, []
    num_val = min(num_val, max(0, len(img_paths) - 1))
    ref = img_paths[:-num_val]
    val = img_paths[-num_val:]
    return ref, val

# ---------------- Global prompt pool for Copeland across ALL rounds ----------------

P = []      # all unique prompts seen across all rounds
P2I = {}    # prompt -> idx
W_global = []  # W_global[i][j] = # times prompt i beat prompt j
S_global = []  # Beta stats per global prompt (optional)

def _ensure_prompt(p: str) -> int:
    global P, P2I, W_global, S_global
    if p not in P2I:
        P2I[p] = len(P)
        P.append(p)
        for row in W_global:
            row.append(0)
        W_global.append([0] * len(P))
        S_global.append(None)
    return P2I[p]

def _winrate_from_W(W, i):
    Wi = sum(W[i][k] for k in range(len(W)))
    Ni = sum(W[i][k] + W[k][i] for k in range(len(W)))
    return Wi / max(1, Ni), Wi, Ni

def _submatrix(W, idxs):
    m = len(idxs)
    Wsub = [[0]*m for _ in range(m)]
    for a, ia in enumerate(idxs):
        for b, ib in enumerate(idxs):
            Wsub[a][b] = W[ia][ib]
    return Wsub

def _copeland_scores(W):
    K = len(W)
    cs = [0]*K
    for i in range(K):
        s = 0
        for j in range(K):
            if i == j:
                continue
            if W[i][j] > W[j][i]:
                s += 1
        cs[i] = s
    return cs

def _select_copeland_winner(W, prompts):
    cs = _copeland_scores(W)
    max_cs = max(cs)
    cand = [i for i,v in enumerate(cs) if v == max_cs]
    if len(cand) == 1:
        return cand[0]

    # tie-break 1: head-to-head among candidates
    wins = {i: 0 for i in cand}
    for i in cand:
        for j in cand:
            if i == j:
                continue
            if W[i][j] > W[j][i]:
                wins[i] += 1
    best_h2h = max(wins.values())
    cand = [i for i,v in wins.items() if v == best_h2h]
    if len(cand) == 1:
        return cand[0]

    # tie-break 2: overall win-rate
    wr_list = []
    K = len(W)
    for i in cand:
        Wi = sum(W[i][k] for k in range(K))
        Ni = sum(W[i][k] + W[k][i] for k in range(K))
        wr_list.append((i, Wi / max(1, Ni)))
    wr_list.sort(key=lambda x: x[1], reverse=True)
    top_wr = [i for i,v in wr_list if abs(v - wr_list[0][1]) < 1e-12]
    if len(top_wr) == 1:
        return top_wr[0]

    # tie-break 3: shortest prompt
    top_wr.sort(key=lambda i: len(prompts[i]))
    return top_wr[0]

# ---------------- Multi-ref judge preference ----------------

def _agg(scores: List[float], mode: str) -> float:
    if not scores:
        return float("nan")
    if mode == "max":
        return float(max(scores))
    if mode == "mean":
        return float(sum(scores) / len(scores))
    # default
    return float(max(scores))

def judge_prefer(judgeLM, goal_img_b64, respA, respB):
    """
    Single-reference A/B judgment.

    Returns:
      winner (0=A, 1=B),
      scoreA, scoreB (floats, higher is better),
      conf (float or None),
      rationale (str)
    """
    score_max = float(getattr(judgeLM, "score_max", 100))

    # Handle render failures deterministically
    if respA is None or respB is None:
        if respA is None and respB is None:
            return 0, 0.0, 0.0, None, "Both candidates failed to render; treated as tie."
        if respA is None:
            return 1, 0.0, score_max, None, "Candidate A failed; Candidate B wins."
        return 0, score_max, 0.0, None, "Candidate B failed; Candidate A wins."

    # Preferred: explicit pairwise comparator if judge provides it
    cmp_fn = getattr(judgeLM, "compare", None)
    if callable(cmp_fn):
        try:
            out = cmp_fn(goal_img_b64, [respA, respB]) or {}
            w = out.get("winner", None)
            sA = out.get("scoreA", None)
            sB = out.get("scoreB", None)
            conf = out.get("conf", None)
            rationale = out.get("rationale", "") or "pairwise compare()"

            if w in (0, 1):
                # if compare() didn't return explicit numeric scores, synthesize a consistent margin
                if sA is None or sB is None:
                    conf_f = 0.55 if conf is None else float(conf)
                    base = score_max / 2.0
                    margin = max(score_max * 0.05, score_max * 0.4 * conf_f)
                    if w == 0:
                        sA = min(score_max, base + margin)
                        sB = max(0.0, base - margin)
                    else:
                        sA = max(0.0, base - margin)
                        sB = min(score_max, base + margin)

                return int(w), float(sA), float(sB), (None if conf is None else float(conf)), rationale
        except Exception as e:
            # fall back below
            pass

    # Fallback: independent scoring if judge exposes score()
    scores = judgeLM.score([goal_img_b64, goal_img_b64], [respA, respB])
    sA, sB = float(scores[0]), float(scores[1])
    winner = 0 if sA >= sB else 1
    return winner, sA, sB, None, "fallback score() per-candidate"


def judge_prefer_multi(
    judgeLM,
    goal_refs_b64: List[str],
    respA,
    respB,
    *,
    refs_per_duel: int,
    ref_agg: str,
) -> Tuple[int, float, float, Optional[float], str]:
    """
    Multi-reference A/B judgment by calling judge_prefer() on each selected ref and aggregating.

    Returns:
      winner (0=A, 1=B), agg_scoreA, agg_scoreB, agg_conf (or None), rationale
    """
    score_max = float(getattr(judgeLM, "score_max", 100))

    # Handle render failures deterministically
    if respA is None or respB is None:
        if respA is None and respB is None:
            return 0, 0.0, 0.0, None, "Both candidates failed to render; treated as tie."
        if respA is None:
            return 1, 0.0, score_max, None, "Candidate A failed; Candidate B wins."
        return 0, score_max, 0.0, None, "Candidate B failed; Candidate A wins."

    # Sample refs (cost control)
    if refs_per_duel is not None and refs_per_duel > 0 and len(goal_refs_b64) > refs_per_duel:
        refs = random.sample(goal_refs_b64, refs_per_duel)
    else:
        refs = list(goal_refs_b64)

    sA_list, sB_list, conf_list = [], [], []
    winA = 0

    for r in refs:
        w, sA, sB, conf, _why = judge_prefer(judgeLM, r, respA, respB)
        sA_list.append(float(sA))
        sB_list.append(float(sB))
        if conf is not None:
            conf_list.append(float(conf))
        if w == 0:
            winA += 1

    # Aggregate scores
    if ref_agg == "mean":
        sA_agg = float(sum(sA_list) / max(1, len(sA_list)))
        sB_agg = float(sum(sB_list) / max(1, len(sB_list)))
    else:  # "max" (default)
        sA_agg = float(max(sA_list)) if sA_list else 0.0
        sB_agg = float(max(sB_list)) if sB_list else 0.0

    # Decide winner:
    # - primary: aggregated score
    # - tie-break: majority wins across refs
    if abs(sA_agg - sB_agg) < 1e-12:
        winner = 0 if winA >= (len(refs) - winA) else 1
        tb = "tie-break=majority"
    else:
        winner = 0 if sA_agg >= sB_agg else 1
        tb = "tie-break=agg_score"

    conf_agg = None
    if conf_list:
        conf_agg = float(sum(conf_list) / len(conf_list))
    else:
        conf_agg = None

    rationale = f"multi-ref ({len(refs)} refs), agg={ref_agg}, {tb}"
    return winner, sA_agg, sB_agg, conf_agg, rationale


def _global_playoff(
    args,
    target_adapter,
    judgeLM,
    goal_refs_b64,
    seed_mgr,
    P,
    W_global,
    pool_idxs,
):
    """
    Cross-round duels among pool_idxs; updates W_global.
    Uses shared seeds per duel to reduce variance.
    Uses multi-ref judging.
    """
    pairs = []
    for a in range(len(pool_idxs)):
        for b in range(a + 1, len(pool_idxs)):
            pairs.append((pool_idxs[a], pool_idxs[b]))

    R = max(1, int(getattr(args, "num_reeval", 1)))
    for rep in range(R):
        for t, (gi, gj) in enumerate(pairs):
            seed = None
            if args.use_shared_seeds and target_adapter.supports_seeds():
                seed = seed_mgr.single(iteration=900000 + rep * 10000 + t)

            if seed is not None:
                resps = target_adapter.render_batch([P[gi], P[gj]], [seed, seed])
            else:
                resps = target_adapter.render_batch([P[gi], P[gj]], None)

            winner, *_ = judge_prefer_multi(
                judgeLM,
                goal_refs_b64,
                resps[0],
                resps[1],
                refs_per_duel=getattr(args, "refs_per_duel", 3),
                ref_agg=getattr(args, "ref_agg", "max"),
            )
            if winner == 0:
                W_global[gi][gj] += 1
            else:
                W_global[gj][gi] += 1

# ----------------------------- main -----------------------------

def main(args):
    global P, P2I, W_global, S_global
    P, P2I, W_global, S_global = [], {}, [], []

    # Target adapter (supports seeds for some models)
    target_adapter = make_target_adapter(
        args.target_model,
        save_dir_for_gemini=os.path.join(args.goal_dir, "images"),
    )
    seed_mgr = SeedManager(base=args.seed_base, antithetic=args.antithetic)

    # output dir naming (same as your img_invert)
    model_tag = sanitize_tag(args.target_model)
    judge_tag = sanitize_tag(args.judge_model).replace('-', '_').replace('.', '_')

    parts = Path(args.output_dir).parts
    has_judge = any(p.startswith("judge_") for p in parts)
    last = Path(args.output_dir).name
    has_duel = last.endswith("_duel") or last.endswith("_duel_textprior") or last.endswith("_obj_duel")

    if has_duel and has_judge:
        base_out = args.output_dir
    elif has_duel and not has_judge:
        base_out = os.path.join(args.output_dir, f"judge_{judge_tag}")
    else:
        base_out = os.path.join(args.output_dir, f"{model_tag}_obj_duel", f"judge_{judge_tag}")

    nk_leaf = f"n{args.duels_per_round}_k{args.n_iterations}"
    args.output_dir = base_out if str(base_out).endswith(nk_leaf) else os.path.join(base_out, nk_leaf)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "all_prompts"), exist_ok=True)

    # Load multi-reference images
    img_paths = _collect_goal_images(args.goal_dir, args.obj)
    if not img_paths:
        raise FileNotFoundError(
            f"No images found for obj='{args.obj}'. "
            f"Checked: goal_dir/obj/ folder and goal_dir/obj.(jpg|png|webp)."
        )

    ref_paths, val_paths = _split_refs(img_paths, args.num_val)
    if not ref_paths:
        raise RuntimeError("Reference set is empty after split.")

    goal_refs_b64 = [convert_image_to_base64(p) for p in ref_paths]
    goal_refs_pil = [load_img(p) for p in ref_paths]

    # For caption/text prior, pick a representative ref (first)
    rep_ref_b64 = goal_refs_b64[0]

    use_text_prior = not getattr(args, "no_text_prior", False)
    if use_text_prior:
        print("asking VLM to caption representative goal image...")
        goal_caption = (auto_caption_goal_image(args, rep_ref_b64) or "").strip()
        goal_emb = EMBEDDER.embed(goal_caption)[0]
        goal_emb = np.asarray(goal_emb, dtype=np.float32)
        print(f"[lf] goal_emb L2 norm = {np.linalg.norm(goal_emb):.4f}")
    else:
        print("[lf] text prior disabled (--no-text-prior); skipping caption + embeddings.")
        goal_caption = (auto_caption_goal_image(args, rep_ref_b64) or "").strip()
        goal_emb = None

    # LMs
    system_prompt = get_attacker_system_prompt_personalize_obj()
    attackLM, _targetLM_unused = load_attack_and_target_models(args)  # target handled via target_adapter
    judgeLM = load_judge(args)
    judgeLM.score_max = float(getattr(args, "judge_score_max", 100))
    logger = WandBLogger(args, system_prompt)

    # Init convs (same as your pattern)
    Kc = args.n_streams
    init_msg = get_init_msg()

    # Provide ONE representative goal image in context to the attacker prompt engineer
    processed_response_list = [process_init_msg(init_msg, rep_ref_b64) for _ in range(Kc)]
    convs_list = [conv_template(attackLM.template) for _ in range(Kc)]
    for conv in convs_list:
        conv.set_system_message(system_prompt)

    # Bandit pair selectors (same as yours)
    global_round_counter = 1
    last_target_responses = [None] * Kc
    last_scores_float = [0.0] * Kc

    PAIR_SELECTORS = {
        "thompson": lambda stats, args, t: pick_pairs_thompson(stats, args.duels_per_round),
        "ucb":      lambda stats, args, t: pick_pairs_ucb(stats, args.duels_per_round, t=t, c=2.0),
        "epsilon_greedy": lambda stats, args, t: pick_pairs_eps_greedy(len(stats), args.duels_per_round, eps=args.epsilon),
    }

    for round_idx in range(1, args.n_iterations + 1):
        random.seed(round_idx * 10)
        print(f"\n{'='*36}\nRound (iteration): {round_idx}\n{'='*36}\n")

        if round_idx > 1:
            processed_response_list = [
                process_target_response(resp, sc, rep_ref_b64)
                for resp, sc in zip(last_target_responses, last_scores_float)
            ]

        extracted_attack_list = attackLM.get_attack(convs_list, processed_response_list)
        print("Finished getting prompts.")
        candidate_prompts = [a["prompt"] for a in extracted_attack_list]
        if len(set(candidate_prompts)) != len(candidate_prompts):
            raise RuntimeError("Duplicate prompts in candidate_prompts; please enforce uniqueness.")
        assert len(candidate_prompts) == Kc, f"attackLM returned {len(candidate_prompts)} prompts but n_streams={Kc}"
        improv_list = [a["improvement"] for a in extracted_attack_list]

        # Register prompts globally
        local2global = [_ensure_prompt(p) for p in candidate_prompts]

        # Init Beta stats for new prompts (with optional text prior)
        new_local_idxs = [k for k, gi in enumerate(local2global) if S_global[gi] is None]
        if new_local_idxs:
            if use_text_prior and goal_emb is not None:
                new_prompts = [candidate_prompts[k] for k in new_local_idxs]
                prompt_embs = np.asarray(EMBEDDER.embed(new_prompts), dtype=np.float32)

                goal_vec  = np.asarray(goal_emb, dtype=np.float32)
                goal_norm = np.linalg.norm(goal_vec) + 1e-8
                p_norms   = np.linalg.norm(prompt_embs, axis=1, keepdims=True) + 1e-8
                cos_sims  = (prompt_embs @ goal_vec.reshape(-1, 1)) / (p_norms * goal_norm)
                cos_sims  = cos_sims.reshape(-1)
                sim_scaled = np.clip(0.5 * (cos_sims + 1.0), 0.0, 1.0)

                alpha = float(getattr(args, "text_prior_weight", 4.0))
                for k_local, psc in zip(new_local_idxs, sim_scaled):
                    gi = local2global[k_local]
                    p  = float(psc)
                    a0 = 1.0 + alpha * p
                    b0 = 1.0 + alpha * (1.0 - p)
                    S_global[gi] = {"a": a0, "b": b0, "wins": a0 - 1.0, "plays": a0 + b0 - 2.0}
            else:
                for k_local in new_local_idxs:
                    gi = local2global[k_local]
                    S_global[gi] = {"a": 1.0, "b": 1.0, "wins": 0.0, "plays": 0.0}

        stats = [S_global[gi] for gi in local2global]

        # Optional annealed similarity boost (same shape you used)
        if use_text_prior and goal_emb is not None and float(getattr(args, "beta_sim_boost", 0.0)) > 0:
            try:
                prompt_embs_all = np.asarray(EMBEDDER.embed(candidate_prompts), dtype=np.float32)
                goal_vec  = np.asarray(goal_emb, dtype=np.float32)
                goal_norm = np.linalg.norm(goal_vec) + 1e-8
                p_norms   = np.linalg.norm(prompt_embs_all, axis=1, keepdims=True) + 1e-8
                cos_all   = ((prompt_embs_all @ goal_vec.reshape(-1, 1)) / (p_norms * goal_norm)).reshape(-1)
                sim_all   = np.clip(0.5 * (cos_all + 1.0), 0.0, 1.0)

                beta_sim = float(getattr(args, "beta_sim_boost", 0.0))
                T = max(1, args.n_iterations - 1)
                lam = beta_sim * ((round_idx - 1) / T)
                for k, gi in enumerate(local2global):
                    S_global[gi]["a"] += lam * float(sim_all[k])
                    S_global[gi]["b"] += lam * float(1.0 - sim_all[k])
            except Exception as e:
                print(f"[warn] embedding similarity boost failed: {e}")

        # Choose duel pairs
        if args.bandit == "copeland_ucb":
            top_m = getattr(args, "copeland_top_m", 0)
            top_m = None if (top_m is None or top_m <= 0) else top_m
            pairs = pick_pairs_copeland_ucb(
                W_global=W_global,
                local2global=local2global,
                num_pairs=args.duels_per_round,
                t=global_round_counter,
                alpha=float(getattr(args, "copeland_alpha", 1.0)),
                top_m=top_m,
            )
        else:
            pairs = PAIR_SELECTORS[args.bandit](stats, args, global_round_counter)

        assert all((0 <= i < Kc) and (0 <= j < Kc) and (i != j) for (i, j) in pairs), \
            f"Invalid pairs from bandit='{args.bandit}': {pairs}"

        # Run duels
        duel_rows = []
        detail_rows = []
        vlm_pair_rows = []
        winner_scores_raw = []
        loser_scores_raw  = []

        duel_pair_img_dir = os.path.join(args.output_dir, "duel_pairs_images")
        os.makedirs(duel_pair_img_dir, exist_ok=True)

        t0 = time.time()
        for pair_idx, (i, j) in enumerate(pairs, start=1):
            # render with shared seed (CRN) if supported
            if args.use_shared_seeds and target_adapter.supports_seeds():
                seeds = seed_mgr.batch(2, iteration=round_idx * 1000 + pair_idx)
                if seeds and isinstance(seeds[0], (tuple, list)):
                    seeds = [s[0] for s in seeds]
                responses = target_adapter.render_batch([candidate_prompts[i], candidate_prompts[j]], seeds)
            else:
                responses = target_adapter.render_batch([candidate_prompts[i], candidate_prompts[j]], None)

            winner, sA, sB, conf, rationale = judge_prefer_multi(
                judgeLM,
                goal_refs_b64,
                responses[0],
                responses[1],
                refs_per_duel=getattr(args, "refs_per_duel", 3),
                ref_agg=getattr(args, "ref_agg", "max"),
            )

            if winner == 0:
                winner_scores_raw.append(sA); loser_scores_raw.append(sB)
                stats[i]["a"] += 1; stats[i]["wins"] += 1
                stats[j]["b"] += 1
            else:
                winner_scores_raw.append(sB); loser_scores_raw.append(sA)
                stats[j]["a"] += 1; stats[j]["wins"] += 1
                stats[i]["b"] += 1
            stats[i]["plays"] += 1
            stats[j]["plays"] += 1

            # Update global W
            gi, gj = local2global[i], local2global[j]
            if winner == 0:
                W_global[gi][gj] += 1
            else:
                W_global[gj][gi] += 1

            # Keep last responses
            last_target_responses[i] = responses[0]
            last_target_responses[j] = responses[1]

            duel_rows.append([round_idx, pair_idx, i, j, winner])
            detail_rows.append([round_idx, pair_idx, i, j, sA, sB, winner, conf])

            # Optional: save duel images
            imgA_name, imgB_name = None, None
            try:
                pair_pils = to_pil_list(responses)
                imgA_name = f"round{round_idx:03d}_pair{pair_idx:03d}_i{i}_A.png"
                imgB_name = f"round{round_idx:03d}_pair{pair_idx:03d}_j{j}_B.png"
                imgA_path = os.path.join(duel_pair_img_dir, imgA_name)
                imgB_path = os.path.join(duel_pair_img_dir, imgB_name)
                if pair_pils[0] is not None:
                    pair_pils[0].save(imgA_path)
                if pair_pils[1] is not None:
                    pair_pils[1].save(imgB_path)
            except Exception as e:
                print(f"[warn] failed to save duel images for round={round_idx}, pair={pair_idx}: {e}")

            vlm_pair_rows.append([
                round_idx, pair_idx, i, j, imgA_name, imgB_name, sA, sB, winner, conf
            ])

            print(
                f"[judge] round={round_idx} pair={pair_idx} i={i} vs j={j} "
                f"scoreA={sA:.4f} scoreB={sB:.4f} "
                f"winner={'A(i)' if winner==0 else 'B(j)'} "
                f"({rationale})"
            )

        print(f"[DEBUG] duels done in {time.time()-t0:.1f}s", flush=True)

        # pseudo-scores (win-rates) for logging & next-iteration feedback
        for k in range(Kc):
            n = max(1, stats[k]["plays"])
            last_scores_float[k] = stats[k]["wins"] / n

        # Ensure images for logging
        imgs_for_log = []
        for k in range(Kc):
            if last_target_responses[k] is None:
                try:
                    if args.use_shared_seeds and target_adapter.supports_seeds():
                        seed = seed_mgr.single(iteration=round_idx * 777 + k)
                        resp = target_adapter.render_batch([candidate_prompts[k]], [seed])[0]
                    else:
                        resp = target_adapter.render_batch([candidate_prompts[k]], None)[0]
                except Exception:
                    resp = None
                last_target_responses[k] = resp
            imgs_for_log.append(last_target_responses[k])
        imgs_for_log = to_pil_list(imgs_for_log)

        # Save candidate images + CSV log
        save_candidate_images_and_log(
            out_dir=args.output_dir,
            obj=args.obj,
            model_tag=model_tag,
            judge_tag=judge_tag,
            iteration=round_idx,
            candidate_prompts=candidate_prompts,
            candidate_pils=imgs_for_log,
            goal_caption=goal_caption,
        )

        # Log round
        logger.log(
            round_idx,
            [{"prompt": p, "improvement": imp} for p, imp in zip(candidate_prompts, improv_list)],
            imgs_for_log,
            last_scores_float,
            winner_scores_raw=winner_scores_raw,
            loser_scores_raw=loser_scores_raw,
        )

        # Save traces
        trace_path = save_duel_trace_csv_compact(args.output_dir, args.obj, round_idx, duel_rows)
        logger.log_duels(round_idx, duel_rows)
        print(f"[trace] saved duel trace: {trace_path}")

        detail_path = save_duel_trace_csv_detailed(args.output_dir, args.obj, round_idx, detail_rows)
        print(f"[trace] saved detailed duel trace: {detail_path}")

        vlm_pairs_path = save_vlm_pairs_csv(args.output_dir, args.obj, vlm_pair_rows)
        print(f"[trace] saved VLM pairwise scores: {vlm_pairs_path}")

        # Trim history
        keep_last_n = getattr(args, "keep_last_n", 5)
        for conv in convs_list:
            conv.messages = conv.messages[-2 * keep_last_n:]

        global_round_counter += 1

    # -------- Global playoff + selection --------
    K_global = len(P)
    pool_idxs = list(range(K_global))
    if K_global > 1:
        M = min(int(args.top_c), K_global)
        ranked = list(range(K_global))
        ranked.sort(key=lambda i: _winrate_from_W(W_global, i)[0], reverse=True)
        pool_idxs = ranked[:M]
        print(f"[global] playoff among top-{M} prompts by global win-rate (cross-round re-eval).")
        _global_playoff(
            args=args,
            target_adapter=target_adapter,
            judgeLM=judgeLM,
            goal_refs_b64=goal_refs_b64,
            seed_mgr=seed_mgr,
            P=P,
            W_global=W_global,
            pool_idxs=pool_idxs,
        )

    if not W_global or all(all(x == 0 for x in row) for row in W_global):
        # fallback: last round local winrates
        wr = [(i, (stats[i]["wins"] / max(1, stats[i]["plays"]))) for i in range(Kc)]
        wr.sort(key=lambda x: x[1], reverse=True)
        best_prompt = candidate_prompts[wr[0][0]]
        best_score_float = float(wr[0][1])
    else:
        W_pool = _submatrix(W_global, pool_idxs)
        P_pool = [P[i] for i in pool_idxs]
        winner_pool = _select_copeland_winner(W_pool, P_pool)
        winner_idx_global = pool_idxs[winner_pool]
        best_prompt = P[winner_idx_global]
        Wi = sum(W_global[winner_idx_global][k] for k in range(K_global))
        Ni = sum(W_global[winner_idx_global][k] + W_global[k][winner_idx_global] for k in range(K_global))
        best_score_float = float(Wi / max(1, Ni))

    # Persist chosen prompt
    with open(os.path.join(args.output_dir, "prompts", f"{args.obj}.txt"), "w") as f:
        f.write(best_prompt)

    # Final render
    final_name = f"{args.obj}_best_n{args.duels_per_round}_k{args.n_iterations}_PRISM-DUEL_OBJ.png"
    final_path = os.path.join(args.output_dir, final_name)

    print("[warmup] priming target model...", flush=True)
    final_resp = target_adapter.render_one(
        best_prompt,
        seed=(args.seed_base + 777) if target_adapter.supports_seeds() else None
    )
    pil = final_resp if isinstance(final_resp, Image.Image) else load_img(final_resp)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil.save(final_path)
    print(f"[FINAL] saved best image to: {final_path}")

    # Save a representative original ref
    rep_copy_path = os.path.join(args.output_dir, f"{args.obj}_REF0.png")
    try:
        rep = load_img(ref_paths[0])
        if rep.mode != "RGB":
            rep = rep.convert("RGB")
        rep.save(rep_copy_path)
    except Exception as e:
        print(f"[warn] failed to save REF0: {e}")
    
    # Side-by-side ORIGINAL vs PRISM-DUEL
    try:
        compare_name = f"{args.obj}_n{args.duels_per_round}_k{args.n_iterations}_ORIGINAL_vs_PRISM-DUEL.png"
        compare_path = os.path.join(args.output_dir, compare_name)
        save_side_by_side(rep_copy_path, final_path, compare_path)
        print(f"[FINAL] saved ORIGINAL vs PRISM-DUEL: {compare_path}")
    except Exception as e:
        print(f"[warn] failed to save side-by-side comparison: {e}")

    # Caption vs prompt artifact
    caption_for_save = goal_caption if goal_caption.strip() else (auto_caption_goal_image(args, rep_ref_b64) or "")
    txt_path, png_path = save_caption_vs_prompt(
        out_dir=args.output_dir,
        obj=args.obj,
        caption=caption_for_save,
        prompt=best_prompt,
    )
    print(f"[FINAL] saved caption vs prompt text: {txt_path}")
    print(f"[FINAL] saved caption vs prompt image: {png_path}")

    logger.log_final(
        final_image_path=final_path,
        best_prompt=best_prompt,
        best_score=best_score_float,
        extras={
            "final/goal_dir": args.goal_dir,
            "final/output_dir": args.output_dir,
            "final/n_streams": args.n_streams,
            "final/n_iterations": args.n_iterations,
            "mode": "PRISM-DUEL_OBJ",
            "ref0_path": rep_copy_path if os.path.isfile(rep_copy_path) else None,
            "num_refs": len(ref_paths),
            "num_val": len(val_paths),
            "refs_per_duel": getattr(args, "refs_per_duel", 3),
            "ref_agg": getattr(args, "ref_agg", "max"),
        },
    )
    try:
        logger.finish()
    except Exception as e:
        print("[warn] wandb finish failed:", e, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Assistant model
    parser.add_argument("--attack-model", default="gpt-4o-mini",
                        choices=["gpt-4.1-mini", "gpt-4o-mini"])
    parser.add_argument("--attack-max-n-tokens", type=int, default=500)
    parser.add_argument("--max-n-attack-attempts", type=int, default=5)

    # T2I model
    parser.add_argument("--target-model", default="sdxl-turbo",
                        choices=["dall-e-2", "dall-e-3", "sdxl-turbo", "flux1", "qwen-image", "gemini"])
    parser.add_argument("--target-max-n-tokens", type=int, default=150)

    # Judge
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        choices=["gpt-4.1-mini", "no-judge", "gpt-4o-mini","gpt-5-nano"])
    parser.add_argument("--judge-max-n-tokens", type=int, default=10)
    parser.add_argument("--judge-temperature", type=float, default=0)
    parser.add_argument("--judge-score-max", type=int, default=10, choices=[10, 100])

    # PRISM-DUEL parameters
    parser.add_argument("--n-streams", type=int, default=6)
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--duels-per-round", type=int, default=6)
    parser.add_argument("--bandit", type=str, default="thompson",
                        choices=["thompson", "ucb", "epsilon_greedy", "copeland_ucb"])
    parser.add_argument("--epsilon", type=float, default=0.1)

    # OBJ dataset IO
    parser.add_argument("--goal_dir", type=str, default="./dreambooth/dataset/",
                        help="Directory containing obj folder or obj file.")
    parser.add_argument("--obj", type=str, default="dog",
                        help="Either a folder name under goal_dir, or a filename/stem.")
    parser.add_argument("--output-dir", type=str, default="obj_duel_results/")
    parser.add_argument("--project-name", type=str, default="PRISM-DUEL_OBJ")
    parser.add_argument("--top-c", type=int, default=5)
    parser.add_argument("--num-reeval", type=int, default=2)
    parser.add_argument("--keep-last-n", type=int, default=5)
    parser.add_argument("--english", action="store_true")

    # Multi-ref judging controls
    parser.add_argument("--refs-per-duel", type=int, default=3,
                        help="How many reference images to sample per duel (cost control).")
    parser.add_argument("--ref-agg", type=str, default="max", choices=["max", "mean"],
                        help="How to aggregate multi-ref scores for A/B: max (best match) or mean (robust match).")
    parser.add_argument("--num-val", type=int, default=1,
                        help="Hold out last num_val images for validation (not used in duels; logged only).")

    # Seeds
    parser.add_argument("--use-shared-seeds", action="store_true")
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--antithetic", action="store_true")

    # Text prior knobs (optional)
    parser.add_argument("--no-text-prior", action="store_true")
    parser.add_argument("--text-prior-weight", type=float, default=4.0)
    parser.add_argument("--beta-sim-boost", type=float, default=0.5)
    parser.add_argument("--copeland-alpha", type=float, default=1.0)
    parser.add_argument("--copeland-top-m", type=int, default=4)

    args = parser.parse_args()
    main(args)
