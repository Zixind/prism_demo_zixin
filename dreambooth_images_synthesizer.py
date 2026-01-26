#!/usr/bin/env python3
"""
DreamBooth-style synthesizing images (targets x*) using:
  - fal-ai via huggingface_hub.InferenceClient for FLUX and Qwen
  - Gemini via google-genai

Env:
  export HF_TOKEN="..."          # for fal-ai provider
  export GEMINI_API_KEY="..."    # for gemini backend

Run:
  python dreambooth_images_synthesizer.py \
    --out_dir outputs_dreambooth \
    --backends flux1_fal,qwen_image_fal,gemini \
    --num_prompts 25 --num_images_per_prompt 1 \
    --height 1024 --width 1024 \
    --seed_base 0

Notes:
- FLUX.1-dev may be gated on HF; you must accept access on the model page.
- Some backends may ignore seed/size depending on deployment.
"""

import argparse
import json
import os
import re
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from io import BytesIO


# ----------------------------
# Seeding (best-effort)
# ----------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed & 0xFFFFFFFF)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ----------------------------
# Parsing prompts_and_classes
# ----------------------------
def load_prompts_and_classes() -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Returns:
      subject2class: mapping subject_name -> class token
      object_prompts: list of 25 templates like "a {0} {1} in the jungle"
      live_prompts:   list of 25 templates (for cats/dogs)
    """
    path = hf_hub_download(
        repo_id="google/dreambooth",
        repo_type="dataset",
        filename="dataset/prompts_and_classes.txt",
    )
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")

    # Parse subject_name,class lines
    subject2class: Dict[str, str] = {}
    m = re.search(r"subject_name,class\s+(.*?)\s+Prompts", txt, flags=re.DOTALL)
    if not m:
        raise RuntimeError("Failed to parse subject->class mapping from prompts_and_classes.txt")
    mapping_block = m.group(1).strip().splitlines()
    for line in mapping_block:
        line = line.strip()
        if not line or "," not in line:
            continue
        subj, cls = [x.strip() for x in line.split(",", 1)]
        subject2class[subj] = cls

    def _extract_section(section_header: str) -> List[str]:
        sec = re.search(section_header + r".*?prompt_list\s*=\s*\[(.*?)\]\s*", txt, flags=re.DOTALL)
        if not sec:
            raise RuntimeError(f"Failed to parse prompt_list for section: {section_header}")
        block = sec.group(1)
        prompts = re.findall(r"'([^']+)'", block)
        return [p.strip() for p in prompts if p.strip()]

    object_prompts = _extract_section(r"Object Prompts")
    live_prompts   = _extract_section(r"Live Subject Prompts")

    if len(object_prompts) < 25 or len(live_prompts) < 25:
        raise RuntimeError(f"Unexpected prompt count: object={len(object_prompts)} live={len(live_prompts)}")

    return subject2class, object_prompts, live_prompts


def is_live_subject(class_token: str) -> bool:
    return class_token.strip().lower() in {"dog", "cat"}


# ----------------------------
# Backends
# ----------------------------
@dataclass
class GenConfig:
    width: int = 1024
    height: int = 1024
    timeout: int = 180
    sleep_s: float = 0.0          # rate-limit between calls
    max_retries: int = 3
    retry_backoff_s: float = 2.0  # exponential


class GeminiBackend:
    def __init__(self, model: str = "gemini-2.5-flash-image"):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment.")
        self.client = genai.Client()
        self.model = model

    def generate_one(self, prompt: str, seed: Optional[int] = None) -> List[Image.Image]:
        # Seed isn't reliably supported across Gemini image endpoints; we still set global seeds for consistency.
        set_global_seed(seed or 0)
        resp = self.client.models.generate_content(model=self.model, contents=[prompt])

        imgs: List[Image.Image] = []

        # Common layout:
        # resp.candidates[0].content.parts[*].inline_data.data (bytes)
        candidates = getattr(resp, "candidates", []) or []
        for cand in candidates[:1]:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", []) or []
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None) is not None:
                    imgs.append(Image.open(BytesIO(inline.data)))
        return imgs


class FalHFBackend:
    """
    HF InferenceClient with provider='fal-ai'.
    """
    def __init__(self, model: str, timeout: int = 180):
        from huggingface_hub import InferenceClient

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            raise RuntimeError("Missing HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in environment for fal-ai provider.")

        # provider='fal-ai' requires `pip install fal-client` on many setups
        self.client = InferenceClient(provider="fal-ai", api_key=hf_token, timeout=timeout)
        self.model = model

    def generate_one(
        self,
        prompt: str,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Image.Image:
        set_global_seed(seed or 0)

        # Some deployments accept width/height/seed; others don't. Try progressively.
        kw: Dict[str, Any] = {}
        if width is not None:
            kw["width"] = int(width)
        if height is not None:
            kw["height"] = int(height)
        if seed is not None:
            kw["seed"] = int(seed)

        # Try: with kwargs; then without.
        try:
            img = self.client.text_to_image(prompt, model=self.model, **kw)
        except TypeError:
            img = self.client.text_to_image(prompt, model=self.model)
        return img


# ----------------------------
# IO
# ----------------------------
def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------
# Generation helpers
# ----------------------------
def call_with_retries(fn, *, cfg: GenConfig):
    last_err = None
    for attempt in range(cfg.max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            sleep = cfg.retry_backoff_s * (2 ** attempt)
            print(f"[WARN] generation failed (attempt {attempt+1}/{cfg.max_retries}): {e}")
            print(f"       sleeping {sleep:.1f}s then retry...")
            time.sleep(sleep)
    raise RuntimeError(f"Generation failed after {cfg.max_retries} retries: {last_err}") from last_err


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("DreamBooth target synthesizer (fal-ai + gemini)")
    ap.add_argument("--out_dir", type=str, default="outputs_dreambooth")
    ap.add_argument("--backends", type=str, default="flux1_fal,qwen_image_fal,gemini",
                    help="Comma-separated: flux1_fal, qwen_image_fal, gemini")
    ap.add_argument("--subjects", type=str, nargs="*", default=None,
                    help="Subset names like backpack, dog, cat2. If omitted, runs all subjects.")
    ap.add_argument("--unique_token", type=str, default="V",
                    help="Unique token inserted as {0} in templates.")
    ap.add_argument("--unique_token_per_subject", action="store_true",
                    help="If set, uses unique_token + '_' + subject.")
    ap.add_argument("--num_prompts", type=int, default=25)
    ap.add_argument("--num_images_per_prompt", type=int, default=1,
                    help="Replicates per (subject, prompt_template, backend).")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--sleep_s", type=float, default=0.0,
                    help="Sleep between API calls (rate limit).")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--retry_backoff_s", type=float, default=2.0)
    ap.add_argument("--save_refs_only", action="store_true",
                    help="Only save DreamBooth reference images (no synthesis).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip generating target files that already exist.")
    args = ap.parse_args()

    cfg = GenConfig(
        width=args.width,
        height=args.height,
        timeout=args.timeout,
        sleep_s=args.sleep_s,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff_s,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject2class, obj_prompts, live_prompts = load_prompts_and_classes()
    subjects = args.subjects if args.subjects else sorted(subject2class.keys())

    backend_names = [b.strip() for b in args.backends.split(",") if b.strip()]

    # Instantiate backends
    backends: Dict[str, Any] = {}
    if not args.save_refs_only:
        for name in backend_names:
            if name == "gemini":
                backends[name] = GeminiBackend()
            elif name == "qwen_image_fal":
                backends[name] = FalHFBackend("Qwen/Qwen-Image", timeout=cfg.timeout)
            elif name == "flux1_fal":
                backends[name] = FalHFBackend("black-forest-labs/FLUX.1-dev", timeout=cfg.timeout)
            else:
                raise ValueError(f"Unknown backend: {name}")

    meta_path = out_dir / "metadata.jsonl"
    # Don’t delete by default; append to allow resumes.
    print(f"[INFO] Writing metadata to: {meta_path}")

    for sidx, subject in enumerate(subjects, start=1):
        if subject not in subject2class:
            print(f"[WARN] Unknown subject '{subject}', skipping.")
            continue

        class_token = subject2class[subject]
        templates = live_prompts if is_live_subject(class_token) else obj_prompts
        templates = templates[: args.num_prompts]

        unique_token = args.unique_token
        if args.unique_token_per_subject:
            unique_token = f"{unique_token}_{subject}"

        print(f"\n=== [{sidx}/{len(subjects)}] subject={subject} class={class_token} ===")

        # 1) Save reference images (4-6 per subject)
        ds = load_dataset("google/dreambooth", subject, split="train")
        ref_dir = out_dir / "refs" / subject
        for i, ex in enumerate(ds):
            img = ex["image"]
            ref_path = ref_dir / f"ref_{i:02d}.png"
            if args.resume and ref_path.exists():
                continue
            save_image(img, ref_path)

        if args.save_refs_only:
            continue

        # 2) Synthesize targets
        for backend_name, backend in backends.items():
            tgt_dir = out_dir / "targets" / backend_name / subject
            tgt_dir.mkdir(parents=True, exist_ok=True)

            for pid, template in enumerate(templates):
                prompt = template.format(unique_token, class_token)

                for rep in range(args.num_images_per_prompt):
                    seed = args.seed_base + (pid + 1) * 1000 + rep

                    out_path = tgt_dir / f"p{pid:02d}_r{rep:02d}_seed{seed}.png"
                    if args.resume and out_path.exists():
                        continue

                    def _do_gen():
                        if backend_name == "gemini":
                            imgs = backend.generate_one(prompt, seed=seed)
                            if not imgs:
                                raise RuntimeError("Gemini returned no images.")
                            return imgs[0]
                        else:
                            # fal-ai HF backend
                            return backend.generate_one(prompt, seed=seed, width=cfg.width, height=cfg.height)

                    img = call_with_retries(_do_gen, cfg=cfg)
                    save_image(img, out_path)

                    append_jsonl(meta_path, {
                        "subject": subject,
                        "class_token": class_token,
                        "unique_token": unique_token,
                        "backend": backend_name,
                        "prompt_id": pid,
                        "prompt_template": template,
                        "prompt": prompt,
                        "rep": rep,
                        "seed": seed,
                        "height": cfg.height,
                        "width": cfg.width,
                        "path": str(out_path.relative_to(out_dir)),
                        "timestamp": time.time(),
                    })

                    if cfg.sleep_s > 0:
                        time.sleep(cfg.sleep_s)

        print(f"[OK] subject={subject} done.")

    print(f"\n[DONE] Outputs under: {out_dir.resolve()}")
    print(f"[DONE] Metadata: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
