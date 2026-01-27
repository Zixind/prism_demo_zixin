#!/usr/bin/env python3
# PRISM-DUEL runner with true dueling bandits (pairwise A/B)
import warnings
warnings.filterwarnings("ignore")
import os, csv, math, random, time, sys
from pathlib import Path
import argparse
from system_prompts import *
# from loggers import WandBLogger
from loggers_duel import WandBLoggerDuel as WandBLogger #change it to duel logger
from judges import load_judge, NoJudge, GPTVJudge
from conversers import load_attack_and_target_models
from common import *
from utils import (
    SeedManager, 
    save_vlm_pairs_csv,
    save_duel_trace_csv_compact,
    save_duel_trace_csv_detailed,
    save_caption_vs_prompt
)
from prism_duel_text_prior import (
    sanitize_tag,
    auto_caption_goal_image,
    build_text_prior_stats
)
from prism_duel_bandits import (
    pick_pairs_thompson,
    pick_pairs_ucb,
    pick_pairs_eps_greedy,
    pick_pairs_copeland_ucb,   
)
from target_adapters import make_target_adapter, to_pil_list
import re
from language_models import OpenAITextEmbedder
from openai import OpenAI  
import numpy as np
import textwrap
from PIL import Image, ImageDraw, ImageFont


### below is for <output_dir>/prompt_image_caption.csv
### save images under <output_dir>/candidate_images/*.png
import csv
from pathlib import Path
from typing import Optional

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

EMBEDDER = OpenAITextEmbedder()
def _winrate_from_W(W, i):
    Wi = sum(W[i][k] for k in range(len(W)))
    Ni = sum(W[i][k] + W[k][i] for k in range(len(W)))
    return Wi / max(1, Ni), Wi, Ni
def _submatrix(W, idxs):
    # W_sub[a][b] corresponds to W[idxs[a]][idxs[b]]
    m = len(idxs)
    Wsub = [[0]*m for _ in range(m)]
    for a, ia in enumerate(idxs):
        for b, ib in enumerate(idxs):
            Wsub[a][b] = W[ia][ib]
    return Wsub
def _global_playoff(
    args,
    target_adapter,
    judgeLM,
    goal_img_string,
    seed_mgr,
    P,
    W_global,
    pool_idxs,
    out_dir=None,
):
    """
    Run cross-round duels among pool_idxs and update W_global in-place.
    Uses shared seeds per duel to reduce variance.
    """
    # full round-robin among pool
    pairs = []
    for a in range(len(pool_idxs)):
        for b in range(a + 1, len(pool_idxs)):
            pairs.append((pool_idxs[a], pool_idxs[b]))

    # repeat each pair num_reeval times 
    R = max(1, int(getattr(args, "num_reeval", 1)))

    for rep in range(R):
        for t, (gi, gj) in enumerate(pairs):
            # shared seed for fairness
            seed = None
            if args.use_shared_seeds and target_adapter.supports_seeds():
                seed = seed_mgr.single(iteration=900000 + rep * 10000 + t)

            # render two prompts with SAME seed
            if seed is not None:
                resps = target_adapter.render_batch([P[gi], P[gj]], [seed, seed])
            else:
                resps = target_adapter.render_batch([P[gi], P[gj]], None)

            if resps[0] is None or resps[1] is None:
                # treat missing as automatic loss
                if resps[0] is None and resps[1] is None:
                    # winner = 0
                    continue
                elif resps[0] is None:
                    winner = 1
                else:
                    winner = 0
            else:
                winner, *_ = judge_prefer(judgeLM, goal_img_string, resps[0], resps[1])

            # update GLOBAL win matrix
            if winner == 0:
                W_global[gi][gj] += 1
            else:
                W_global[gj][gi] += 1

def _copeland_scores(W):
    K = len(W)
    cs = [0]*K
    for i in range(K):
        s = 0
        for j in range(K):
            if i == j: 
                continue
            if W[i][j] > W[j][i]:  # strict majority
                s += 1
        cs[i] = s
    return cs

def _select_copeland_winner(W, prompts):
    """Return index of Copeland winner with reasonable tie-breaks."""
    cs = _copeland_scores(W)
    max_cs = max(cs)
    cand = [i for i,v in enumerate(cs) if v == max_cs]
    if len(cand) == 1:
        return cand[0]

    # tie-break 1: head-to-head within candidate set
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

def judge_prefer(judgeLM, goal_img_b64, respA, respB):
    """
    Returns: winner (0=A,1=B), scoreA, scoreB, conf (or None)
    Tries pairwise compare() if available; falls back to score() otherwise.
    """
    score_max = getattr(judgeLM, "score_max", 100)
    if callable(getattr(judgeLM, "compare", None)):
        try:
            out = judgeLM.compare(goal_img_b64, [respA, respB]) or {}
            w   = out.get("winner", None)
            sA  = out.get("scoreA", None)
            sB  = out.get("scoreB", None)
            conf= out.get("conf", None)
            rationale = out.get("rationale", "")
            if w in (0, 1):
                if sA is None or sB is None:
                    conf_f = 0.55 if conf is None else float(conf)
                    base = score_max / 2.0
                    margin = max(score_max * 0.05, score_max * 0.4 * conf_f)
                    if w == 0:
                        sA = min(score_max, base + margin)
                        sB = max(0.0,      base - margin)
                    else:
                        sA = max(0.0,      base - margin)
                        sB = min(score_max, base + margin)

                return int(w), float(sA), float(sB), (None if conf is None else float(conf)), rationale
        except Exception:
            pass
    # Fallback: independent scores
    scores = judgeLM.score([goal_img_b64, goal_img_b64], [respA, respB])
    sA, sB = float(scores[0]), float(scores[1])
    winner = 0 if sA >= sB else 1
    rationale = "Fallback: independent per-candidate scores, no explicit pairwise rationale."
    return winner, sA, sB, None, rationale


# ---------- Global prompt pool for Copeland across ALL rounds ----------
P = []  # all unique prompts seen across all rounds
P2I = {}   # prompt -> idx
W_global = []   # W_global[i][j] = # times prompt i beat prompt j
S_global = []
def _ensure_prompt(p: str) -> int:
    """
    Register prompt p in the global pool, expand W_global if new, and return
    its global index in P.
    """
    global P, P2I, W_global, S_global
    if p not in P2I:
        P2I[p] = len(P)
        P.append(p)
        for row in W_global:
            row.append(0)
        W_global.append([0] * len(P))
        S_global.append(None)
    return P2I[p]
# ----------------------------- main -----------------------------

def main(args):
    global P, P2I, W_global, S_global
    P, P2I, W_global, S_global = [], {}, [], []
    def infer_goal_text(args):
        base = os.path.splitext(os.path.basename(args.obj))[0]
        base = re.sub(r'^\d+_', '', base)
        return base.replace("_", " ")
    
    target_adapter = make_target_adapter(
        args.target_model,
        save_dir_for_gemini=os.path.join(args.goal_dir, "images"),
    )
    # Seeds
    seed_mgr = SeedManager(base=args.seed_base, antithetic=args.antithetic)

    args.obj = os.path.splitext(os.path.basename(args.obj))[0]
    args.dirname = Path(args.goal_dir).name

    # FORCE a single canonical output root:
    model_tag = sanitize_tag(args.target_model)
    judge_tag = sanitize_tag(args.judge_model).replace('-', '_').replace('.', '_')

    parts = Path(args.output_dir).parts
    has_judge = any(p.startswith("judge_") for p in parts)
    last = Path(args.output_dir).name
    has_duel = last.endswith("_duel") or last.endswith("_duel_textprior")

    if has_duel and has_judge:
        # caller already provided .../<something_duel*>/judge_<tag>[/nN_kK]
        base_out = args.output_dir
    elif has_duel and not has_judge:
        # caller provided .../<something_duel*>  (Option A)
        base_out = os.path.join(args.output_dir, f"judge_{judge_tag}")
    else:
        # caller provided a high-level root
        base_out = os.path.join(args.output_dir, f"{model_tag}_duel", f"judge_{judge_tag}")

    nk_leaf = f"n{args.duels_per_round}_k{args.n_iterations}" #n args.duels_per_round
    args.output_dir = base_out if str(base_out).endswith(nk_leaf) else os.path.join(base_out, nk_leaf)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "all_prompts"), exist_ok=True)

    img_path = os.path.join(args.goal_dir, f"{args.obj}.png")
    goal_img_string = convert_image_to_base64(img_path)
    goal_img = load_img(img_path)

    use_text_prior = not getattr(args, "no_text_prior", False)
    if use_text_prior:
        #asking VLM to caption our reference image
        print("asking VLM to caption goal image...")
        goal_text = auto_caption_goal_image(args, goal_img_string)
        goal_text = (goal_text or "").strip()

        goal_caption = goal_text

        goal_emb = EMBEDDER.embed(goal_text)[0]  # shape (d,)
        goal_emb = np.asarray(goal_emb, dtype=np.float32)
        print(f"[lf] goal_emb L2 norm = {np.linalg.norm(goal_emb):.4f}")
    else:
        print("[lf] text prior disabled (--no-text-prior); skipping caption + embeddings.")
        goal_text = None
        goal_emb = None
        goal_caption = auto_caption_goal_image(args, goal_img_string)
    goal_caption = (goal_caption or "").strip()



    system_prompt = get_attacker_system_prompt_personalize_img()
    attackLM, targetLM = load_attack_and_target_models(args)
    judgeLM = load_judge(args)
    judgeLM.score_max = float(getattr(args, "judge_score_max", 100))
    logger = WandBLogger(args, system_prompt)


    # Initialize conversations
    Kc = args.n_streams  # number of candidates per round
    init_msg = get_init_msg()
    processed_response_list = [process_init_msg(init_msg, goal_img_string) for _ in range(Kc)]
    convs_list = [conv_template(attackLM.template) for _ in range(Kc)]
    for conv in convs_list:
        conv.set_system_message(system_prompt)

    # Keep running tallies across rounds (optional, used by UCB)
    global_round_counter = 1

    last_target_responses = [None] * Kc
    last_scores_float = [0.0] * Kc  

    # Select pairs to duel
    # Default is thompson
    PAIR_SELECTORS = {
        "thompson": lambda stats, args, t: pick_pairs_thompson(stats, args.duels_per_round),
        "ucb":      lambda stats, args, t: pick_pairs_ucb(stats, args.duels_per_round, t=t, c=2.0),
        "epsilon_greedy": lambda stats, args, t: pick_pairs_eps_greedy(len(stats), args.duels_per_round, eps=args.epsilon),
    }

    for round_idx in range(1, args.n_iterations + 1):
        random.seed(round_idx * 10)
        print(f"\n{'='*36}\nRound (iteration): {round_idx}\n{'='*36}\n")

        # Get candidate prompts this round
        if round_idx > 1:
            # Use last round's best image feedback as context
            processed_response_list = [
                process_target_response(resp, sc, goal_img_string)
                for resp, sc in zip(last_target_responses, last_scores_float)
            ]

        extracted_attack_list = attackLM.get_attack(convs_list, processed_response_list)
        print("Finished getting prompts.")
        candidate_prompts = [a["prompt"] for a in extracted_attack_list]
        if len(set(candidate_prompts)) != len(candidate_prompts):
            raise RuntimeError("Duplicate prompts in candidate_prompts; please dedupe or force attackLM to return unique prompts.")
        assert len(candidate_prompts) == Kc, f"attackLM returned {len(candidate_prompts)} prompts but n_streams={Kc}"
        improv_list = [a["improvement"] for a in extracted_attack_list]

        # Map local indices 0..Kc-1 to GLOBAL indices in P
        local2global = [_ensure_prompt(p) for p in candidate_prompts]
        # 1) initialize S_global for any new prompts (those still None)
        new_local_idxs = [k for k, gi in enumerate(local2global) if S_global[gi] is None]

        if new_local_idxs:
            if use_text_prior and goal_emb is not None:
                new_prompts = [candidate_prompts[k] for k in new_local_idxs]
                prompt_embs = np.asarray(EMBEDDER.embed(new_prompts), dtype=np.float32)  # (m,d)

                goal_vec  = np.asarray(goal_emb, dtype=np.float32)
                goal_norm = np.linalg.norm(goal_vec) + 1e-8
                p_norms   = np.linalg.norm(prompt_embs, axis=1, keepdims=True) + 1e-8

                cos_sims   = (prompt_embs @ goal_vec.reshape(-1, 1)) / (p_norms * goal_norm)
                cos_sims   = cos_sims.reshape(-1)  # (m,)
                sim_scaled = np.clip(0.5 * (cos_sims + 1.0), 0.0, 1.0)

                alpha = float(getattr(args, "text_prior_weight", 4.0))

                for k_local, psc in zip(new_local_idxs, sim_scaled):
                    gi = local2global[k_local]
                    p  = float(psc)  # in [0,1]
                    a0 = 1.0 + alpha * p
                    b0 = 1.0 + alpha * (1.0 - p)
                    S_global[gi] = {"a": a0, "b": b0, "wins": a0 - 1.0, "plays": a0 + b0 - 2.0}
            else:
                # runs when --no-text-prior (uniform Beta(1,1)) Assuming we do not update the prior distribution
                for k_local in new_local_idxs:
                    gi = local2global[k_local]
                    S_global[gi] = {"a": 1.0, "b": 1.0, "wins": 0.0, "plays": 0.0}
        stats = [S_global[gi] for gi in local2global]
        # low fidelity prior text embedding similarity
        if use_text_prior and goal_emb is not None:
            try:
                prompt_embs_all = np.asarray(EMBEDDER.embed(candidate_prompts), dtype=np.float32)
                goal_vec  = np.asarray(goal_emb, dtype=np.float32)
                goal_norm = np.linalg.norm(goal_vec) + 1e-8
                p_norms   = np.linalg.norm(prompt_embs_all, axis=1, keepdims=True) + 1e-8
                cos_all   = ((prompt_embs_all @ goal_vec.reshape(-1, 1)) / (p_norms * goal_norm)).reshape(-1)
                sim_all   = np.clip(0.5 * (cos_all + 1.0), 0.0, 1.0)
                print("[lf] text-embedding similarities (cosine) to goal caption:")
                for idx, (sim, ssc) in enumerate(zip(cos_all, sim_all)):
                    print(f"  cand {idx:02d}: cos_sim={sim:.4f}, scaled={ssc:.4f}")
            except Exception as e:
                print(f"[warn] text embedding debug print failed: {e}")

        beta_sim = float(getattr(args, "beta_sim_boost", 0.0))
        if beta_sim > 0 and sim_all is not None:
            T = max(1, args.n_iterations - 1)
            lam = beta_sim * ((round_idx - 1) / T)  # ramp up
            for k, gi in enumerate(local2global):
                S_global[gi]["a"] += lam * float(sim_all[k])
                S_global[gi]["b"] += lam * float(1.0 - sim_all[k])
        
        # pairs = PAIR_SELECTORS[args.bandit](stats, args, global_round_counter)
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
        Kc = len(candidate_prompts)  
        assert all(
            (0 <= i < Kc) and (0 <= j < Kc) and (i != j)
            for (i, j) in pairs
        ), f"Invalid pair(s) from bandit='{args.bandit}': {pairs}"

        # 4) Run duels
        duel_rows = []
        detail_rows = []
        # per-round pairwise ledgers over the Kc prompts of this round
        vlm_pair_rows = []

        winner_scores_raw = []
        loser_scores_raw  = []


        duel_pair_img_dir = os.path.join(args.output_dir, "duel_pairs_images")
        os.makedirs(duel_pair_img_dir, exist_ok=True)

        t0 = time.time()
        for pair_idx, (i, j) in enumerate(pairs, start=1):
            # Render A and B with shared seeds (CRN) for variance reduction
            if args.use_shared_seeds and target_adapter.supports_seeds():
                seeds = seed_mgr.batch(2, iteration=round_idx * 1000 + pair_idx)
                if seeds and isinstance(seeds[0], (tuple, list)):
                    seeds = [s[0] for s in seeds]
                responses = target_adapter.render_batch([candidate_prompts[i], candidate_prompts[j]], seeds)
            else:
                responses = target_adapter.render_batch([candidate_prompts[i], candidate_prompts[j]], None)

            # Decide winner
            # winner, sA, sB, conf = judge_prefer(judgeLM, goal_img_string, responses[0], responses[1])  # 0 => i, 1 => j
            if responses[0] is None or responses[1] is None:
                print(
                    f"[warn] render_batch returned None (round={round_idx}, pair={pair_idx}): "
                    f"respA is None? {responses[0] is None}, respB is None? {responses[1] is None}"
                )
                # use same max as judge_prefer
                score_max = getattr(judgeLM, "score_max", getattr(args, "judge_score_max", 100))
                # Caveat: Treats missing image as automatic loser to avoid calling judgeLM (avoid extra API call)
                if responses[0] is None and responses[1] is None:
                    # both failed and treat as tie with zero scores
                    # winner, sA, sB, conf = 0, 0.0, 0.0, None
                    # rationale = "Both candidates failed to render; treated as tie with zero scores."
                    continue
                elif responses[0] is None:
                    # A missing means B wins
                    winner, sA, sB, conf = 1, 0.0, float(score_max), None
                    rationale = "Candidate A failed to render; Candidate B automatically wins."
                else:
                    # B missing means A wins
                    winner, sA, sB, conf = 0, float(score_max), 0.0, None
                    rationale = "Candidate B failed to render; Candidate A automatically wins."
            else:
                winner, sA, sB, conf, rationale = judge_prefer(
                    judgeLM, goal_img_string, responses[0], responses[1]
                )  # 0 is i, 1 is j
            
            if winner == 0:
                winner_scores_raw.append(sA)
                loser_scores_raw.append(sB)
            else:
                winner_scores_raw.append(sB)
                loser_scores_raw.append(sA)
            # --- save the actual images for this duel ---
            try:
                pair_pils = to_pil_list(responses)  # [imgA, imgB] as PIL Images
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
                imgA_name, imgB_name = None, None

            if winner == 0:
                stats[i]["a"] += 1
                stats[i]["wins"] += 1
                stats[j]["b"] += 1
            else:
                stats[j]["a"] += 1
                stats[j]["wins"] += 1
                stats[i]["b"] += 1
            stats[i]["plays"] += 1
            stats[j]["plays"] += 1

            # Keep latest responses so we can show something in logger
            last_target_responses[i] = responses[0]
            last_target_responses[j] = responses[1]

            duel_rows.append([round_idx, pair_idx, i, j, winner])
            detail_rows.append([round_idx, pair_idx, i, j, sA, sB, winner, conf])
            # row for VLM–image pair correlation
            vlm_pair_rows.append([
                round_idx,
                pair_idx,
                i,
                j,
                imgA_name,
                imgB_name,
                sA,
                sB,
                winner,
                conf,
            ])
            gi, gj = local2global[i], local2global[j] #idx_map[i], idx_map[j]
            if winner == 0:
                W_global[gi][gj] += 1
            else:

                W_global[gj][gi] += 1
            print(
                f"[judge] round={round_idx} pair={pair_idx} i={i} vs j={j} "
                f"scoreA={sA:.4f} scoreB={sB:.4f} "
                f"{'conf='+str(round(conf,4)) if conf is not None else ''} "
                f"winner={'A(i)' if winner==0 else 'B(j)'}"
            )
            # Make it Optional: Short prompt previews (trim to 160 chars to make logs readable)
            pi = candidate_prompts[i].replace("\n", " ")
            pj = candidate_prompts[j].replace("\n", " ")
            print(f"  prompt[i]: {pi[:160]}{'...' if len(pi)>160 else ''}")
            print(f"  prompt[j]: {pj[:160]}{'...' if len(pj)>160 else ''}")
            if rationale:
                print(f"  rationale: {rationale}")

        print(f"[DEBUG] duels done in {time.time()-t0:.1f}s", flush=True)

        def summarize(vals):
            if not vals:
                return (float("nan"), float("nan"), float("nan"))
            mean_v = sum(vals) / len(vals)
            max_v  = max(vals)
            min_v  = min(vals)
            return (mean_v, max_v, min_v)

        w_mean, w_max, w_min = summarize(winner_scores_raw)
        l_mean, l_max, l_min = summarize(loser_scores_raw)

        print("=" * 14 + " RAW JUDGE SCORE STATS " + "=" * 14)
        print(f"Winner scores: mean={w_mean:.3f}, max={w_max:.3f}, min={w_min:.3f}")
        print(f"Loser  scores: mean={l_mean:.3f}, max={l_max:.3f}, min={l_min:.3f}")

        # Convert wins/plays to pseudo-scores for logging + next iteration context
        for k in range(Kc):
            n = max(1, stats[k]["plays"])
            last_scores_float[k] = stats[k]["wins"] / n

        # Prepare images for W&B logging (only where we have a response)
        imgs_for_log = []
        for k in range(Kc):
            if last_target_responses[k] is None:
                # ensure something to visualize: render once cheap
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
        # save candidate images and log
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

        # 7) Log iteration
        # We log improvements/prompts with pseudo-scores (win rates) for transparency.
        print("Finished dueling; logging results with pseudo-scores = win-rates.")
        logger.log(
            round_idx,
            [{"prompt": p, "improvement": imp} for p, imp in zip(candidate_prompts, improv_list)],
            imgs_for_log,
            last_scores_float,
            winner_scores_raw=winner_scores_raw,
            loser_scores_raw=loser_scores_raw,
        )

        # 8) Save duel trace CSV
        trace_path = save_duel_trace_csv_compact(args.output_dir, args.obj, round_idx, duel_rows)
        logger.log_duels(round_idx, duel_rows)
        print(f"[trace] saved duel trace: {trace_path}")
        detail_path = save_duel_trace_csv_detailed(args.output_dir, args.obj, round_idx, detail_rows)
        print(f"[trace] saved detailed duel trace: {detail_path}")
        vlm_pairs_path = save_vlm_pairs_csv(args.output_dir, args.obj, vlm_pair_rows)
        print(f"[trace] saved VLM pairwise scores: {vlm_pairs_path}")

        
        keep_last_n = getattr(args, "keep_last_n", 5)
        # 9) Trim conversation histories
        for conv in convs_list:
            conv.messages = conv.messages[-2 * keep_last_n:]
        # Prepare for next round feedback (already set: last_target_responses, last_scores_float)
        global_round_counter += 1

    K_global = len(P)
    pool_idxs = list(range(K_global))
    if K_global > 1:
        # pick a pool of finalists to make O(M^2) manageable
        # can also reuse args.top_c as the pool size, or add a new flag like --global-top-m
        M = min(int(args.top_c), K_global)

        # rank by global win-rate from W_global (more stable than sparse Copeland pre-playoff)
        ranked = list(range(K_global))
        ranked.sort(key=lambda i: _winrate_from_W(W_global, i)[0], reverse=True)
        pool_idxs = ranked[:M]

        print(f"[global] playoff among top-{M} prompts by global win-rate (cross-round re-eval).")
        _global_playoff(
            args=args,
            target_adapter=target_adapter,
            judgeLM=judgeLM,
            goal_img_string=goal_img_string,
            seed_mgr=seed_mgr,
            P=P,
            W_global=W_global,
            pool_idxs=pool_idxs,
            out_dir=args.output_dir,
        )
    else:
        pool_idxs = list(range(K_global))
    
    # ----- selection (no re-eval): choose Copeland winner if available -----
    if not W_global or all(all(x == 0 for x in row) for row in W_global):
        # Fallback: use per-arm win-rate (from LAST ROUND stats) to pick winner
        wr = [(i, (stats[i]["wins"] / max(1, stats[i]["plays"]))) for i in range(Kc)]
        wr.sort(key=lambda x: x[1], reverse=True)
        winner_idx_local = wr[0][0]
        best_prompt = candidate_prompts[winner_idx_local]

        # Files for auditing (fallback mode, last round only)
        with open(os.path.join(args.output_dir, "all_prompts", f"{args.dirname}_{args.obj}.txt"), "w") as f:
            for i, wv in wr:
                f.write(f"{candidate_prompts[i]},WR={wv:.4f},plays={stats[i]['plays']},idx={i}\n")

        with open(os.path.join(args.output_dir, f"candidates_{args.obj}.txt"), "w") as f:
            for i, wv in wr[:args.top_c]:
                f.write(f"{candidate_prompts[i]},WR={wv:.4f}\n")

        best_score_float = float(wr[0][1])
    else:
        # Normal path: GLOBAL Copeland over all prompts in P
        K_global = len(P)
        if K_global == 0:
            raise RuntimeError("No prompts collected for global selection.")

        # If you created pool_idxs above, use it; else fall back to all prompts
        try:
            pool_idxs
        except NameError:
            pool_idxs = list(range(K_global))

        W_pool = _submatrix(W_global, pool_idxs)
        P_pool = [P[i] for i in pool_idxs]

        winner_pool = _select_copeland_winner(W_pool, P_pool)
        winner_idx_global = pool_idxs[winner_pool]
        best_prompt = P[winner_idx_global]

        Wi = sum(W_global[winner_idx_global][k] for k in range(K_global))
        Ni = sum(W_global[winner_idx_global][k] + W_global[k][winner_idx_global] for k in range(K_global))
        best_score_float = float(Wi / max(1, Ni))

    # Persist the chosen prompt
    with open(os.path.join(args.output_dir, "prompts", f"{args.dirname}_{args.obj}.txt"), "w") as f:
        f.write(best_prompt)

    # --- Final image render & save ---
    import shutil
    from base64 import b64decode
    from PIL import Image

    os.makedirs(args.goal_dir, exist_ok=True)
    final_name = f"{args.obj}_best_n{args.duels_per_round}_k{args.n_iterations}_PRISM-DUEL.png"
    final_path = os.path.join(args.output_dir, final_name)

    def _save_pil(img, path):
        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path); return True
        return False

    print("[warmup] priming target model...", flush=True)
    try:
        final_resp = target_adapter.render_one(
            best_prompt,
            seed=(args.seed_base + 777) if target_adapter.supports_seeds() else None
        )
        print("[warmup] target model ready", flush=True)
    except Exception as e:
        print("[warmup] target model warmup failed:", e, flush=True)
        final_resp = None

    saved = False
    try:
        saved = _save_pil(final_resp, final_path)
    except Exception:
        saved = False
    if not saved and isinstance(final_resp, str):
        if os.path.isfile(final_resp):
            shutil.copyfile(final_resp, final_path); saved = True
        elif final_resp.startswith("data:image"):
            header, b64 = final_resp.split(",", 1)
            with open(final_path, "wb") as f:
                f.write(b64decode(b64)); saved = True
    if not saved:
        try:
            pil_img = load_img(final_resp)
            saved = _save_pil(pil_img, final_path)
        except Exception:
            pass
    if not saved:
        raise RuntimeError(f"Could not save final image from response type: {type(final_resp)}")

    print(f"[FINAL] saved best image to: {final_path}")
    try:
        from PIL import Image
        from shutil import copyfile

        # 1) copy/save the original goal image into output dir
        orig_src_path = os.path.join(args.goal_dir, f"{args.obj}.png")
        orig_copy_name = f"{args.obj}_ORIGINAL.png"
        orig_copy_path = os.path.join(args.output_dir, orig_copy_name)

        if os.path.isfile(orig_src_path):
            # direct copy from disk if available
            copyfile(orig_src_path, orig_copy_path)
        else:
            # fall back to in-memory PIL image we already loaded earlier (goal_img)
            if isinstance(goal_img, Image.Image):
                gi = goal_img.convert("RGB") if goal_img.mode != "RGB" else goal_img
                gi.save(orig_copy_path)

        # 2) optional: create side-by-side (Original | PRISM-DUEL) comparison
        try:
            orig_img = Image.open(orig_copy_path).convert("RGB")
            duel_img = Image.open(final_path).convert("RGB")

            # match heights while preserving aspect ratio
            H = max(orig_img.height, duel_img.height)

            def _resize_to_height(im, H):
                W = int(round(im.width * (H / im.height)))
                return im.resize((W, H), Image.Resampling.LANCZOS)

            o_resized = _resize_to_height(orig_img, H)
            d_resized = _resize_to_height(duel_img, H)

            comp = Image.new("RGB", (o_resized.width + d_resized.width, H), (255, 255, 255))
            comp.paste(o_resized, (0, 0))
            comp.paste(d_resized, (o_resized.width, 0))

            comp_name = f"{args.obj}_n{args.duels_per_round}_k{args.n_iterations}_ORIGINAL_vs_PRISM-DUEL.png"
            comp_path = os.path.join(args.output_dir, comp_name)
            comp.save(comp_path)
            print(f"[FINAL] saved side-by-side comparison to: {comp_path}")
        except Exception as e:
            print(f"[warn] failed to create comparison image: {e}", flush=True)

    except Exception as e:
        print(f"[warn] original image handling failed: {e}", flush=True)

    with open(os.path.join(args.output_dir, f"{args.obj}_best_prompt_n{args.n_streams}_k{args.n_iterations}.txt"), "w") as f:
        f.write(best_prompt)
    
    logger.log_final(
        final_image_path=final_path,
        best_prompt=best_prompt,
        best_score=best_score_float,
        extras={
            "final/goal_dir": args.goal_dir,
            "final/output_dir": args.output_dir,
            "final/n_streams": args.n_streams,
            "final/n_iterations": args.n_iterations,
            "mode": "PRISM-DUEL",
            "original_image_path": orig_copy_path if 'orig_copy_path' in locals() else None,
        },
    )
    # Ensure we have a caption string even if --no-text-prior
    caption_for_save = goal_text if (goal_text is not None and str(goal_text).strip()) else auto_caption_goal_image(args, goal_img_string)

    txt_path, png_path = save_caption_vs_prompt(
        out_dir=args.output_dir,
        obj=args.obj,
        caption=caption_for_save,
        prompt=best_prompt,
    )

    print(f"[FINAL] saved caption vs prompt text: {txt_path}")
    print(f"[FINAL] saved caption vs prompt image: {png_path}")

    try:
        logger.finish()
    except Exception as e:
        print("[warn] wandb finish failed:", e, flush=True)


if __name__ == '__main__':
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

    # PRISM-DUEL parameters (apple-to-apple with PRISM: n_streams=N, n_iterations=K)
    parser.add_argument("--n-streams", type=int, default=6,
                        help="Number of candidate prompts per round (N).")
    parser.add_argument("--n-iterations", type=int, default=5,
                        help="Number of rounds (K).")
    parser.add_argument("--duels-per-round", type=int, default=6,
                        help="Number of pairwise comparisons per round.")
    parser.add_argument("--bandit", type=str, default="thompson",
                        choices=["thompson", "ucb", "epsilon_greedy", "copeland_ucb"],
                        help="Bandit policy for scheduling duels.")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon for epsilon-greedy (if selected).")
    parser.add_argument("--window-size", type=int, default=5,
                        help="(Reserved) window of past prompts to optionally mix; not used in this minimal version.")

    # IO
    parser.add_argument("--goal_dir", type=str, default=".", help="Directory of the target images")
    parser.add_argument("--obj", type=str, default="0.png", help="Name of the target image")
    parser.add_argument("--project-name", type=str, default="PRISM-DUEL")
    parser.add_argument("--output-dir", type=str, default="prism_duel_results/")

    # Selection & re-eval
    parser.add_argument("--top-c", type=int, default=5,
                        help="Select the final prompt from top-c by win-rate, with re-eval.")
    parser.add_argument("--num-reeval", type=int, default=2,
                        help="Extra re-eval renders per candidate to reduce variance.")
    parser.add_argument('--english', action='store_true')

    # Seeds
    parser.add_argument("--use-shared-seeds", action="store_true",
                        help="Use shared (CRN) seeds per duel for variance reduction.")
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--antithetic", action="store_true")
    parser.add_argument(
        "--judge-score-max",
        type=int,
        default=10,          # or 100
        choices=[10, 100],   # restrict to 10 or 100 
        help="Max score for judge outputs (10 or 100).",
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=5,
        help="How many recent turns to keep per conversation."
    )
    parser.add_argument(
        "--text-prior-weight",
        type=float,
        default=4.0,
        help=(
            "Strength of the text-embedding prior (alpha). "
            "Larger values bias more strongly toward prompts that are "
            "textually similar to the auto-caption of the goal image."
        ),
    )
    parser.add_argument("--beta-sim-boost", type=float, default=0.5,
                    help="Annealed pseudo-count boost per round to Beta(a,b) using embedding similarity.")


    parser.add_argument("--copeland-alpha", type=float, default=1.0,
                    help="UCB exploration strength for Copeland-UCB.")
    parser.add_argument("--copeland-top-m", type=int, default=4,
                    help="Focus duels on top-m optimistic Copeland candidates (set <=0 to disable).")

    args = parser.parse_args()

    main(args)
