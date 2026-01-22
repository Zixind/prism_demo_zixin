
# pip install fal-client for huggingface provider
# pip install -U datasets pillow huggingface_hub google-genai pytesseract
# pip install -U "huggingface_hub>=0.28.0" pillow
# pip uninstall -y fschat gradio gradio-client
# pip install -U \
#   google-genai \
#   "pydantic>=2,<3" \
#   "websockets>=13,<15.1" \
#   "huggingface_hub>=0.28.0" \
#   pillow
# python -m pip check   # should be clean

# Generate images for 3 folders run: python3 parti_writing_symbols_gen.py --n 91 --start 0

import os, re, json, time, random
from dataclasses import dataclass
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import os
import fal_client
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
# Optional OCR (for quick legibility sanity check)
try:
    import pytesseract
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False

# ----------------------------
# Utils
# ----------------------------
def slugify(text: str, maxlen: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", (text or "")).strip().replace(" ", "_")
    text.lstrip("S_")
    return (text[:maxlen] or "prompt")

def set_global_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
    except Exception:
        pass

def norm_edit(s1: str, s2: str) -> float:
    """Normalized edit similarity in [0,1]; 1 means exact match."""
    try:
        import editdistance
        d = editdistance.eval(s1, s2)
    except Exception:
        # fallback levenshtein
        import difflib
        sm = difflib.SequenceMatcher(a=s1, b=s2)
        # convert ratio to distance-ish, then to similarity
        return sm.ratio()
    m = max(1, len(s1), len(s2))
    return 1.0 - d / m



# ----------------------------
# Dataset: Parti -> Writing & Symbols
# ----------------------------
def _lens(ex):
    txt = (ex["Prompt"] or "").strip()
    return {"char_len": len(txt), "word_len": len(txt.split())}


def load_parti_writing_symbols(n: int, start: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    ds = load_dataset("nateraw/parti-prompts")["train"]
    sub = ds.filter(lambda ex: (ex.get("Challenge") or "").strip().lower() == "writing & symbols")
    #91
    print('len(sub) is {}, how many challenge writing & symbols in partiprompts'.format(len(sub)))
    sub = sub.map(_lens)
    sub = sub.sort(["word_len", "char_len"], reverse=[True, False])

    sel = sub.select(range(start, start+n))         #sel = sub.select(range(min(n, len(sub))))
    prompts = [p.strip() for p in sel["Prompt"]]
    metas = []
    for i in range(len(sel)):
        metas.append({
            "Prompt": sel[i]["Prompt"],
            "Category": sel[i].get("Category"),
            "Challenge": sel[i].get("Challenge"),
            "Note": sel[i].get("Note"),
            "char_len": int(sel[i]["char_len"]),
            "word_len": int(sel[i]["word_len"]),
        })
    return prompts, metas


# # --------------------------
# # 2) Google GenAI client
# # --------------------------
# from google import genai

# API_KEY = os.getenv("GEMINI_API_KEY")
# if not API_KEY:
#     raise RuntimeError("Set GEMINI_API_KEY in your environment.")

# client = genai.Client()
# MODEL_NAME = "gemini-2.5-flash-image-preview"   # swap to your preferred image model if needed

# # --------------------------
# # 3) Generate & save images
# # --------------------------
# out_dir = "writing_symbols_outputs_nano_banana"
# os.makedirs(out_dir, exist_ok=True)

# ----------------------------
# Backends
# ----------------------------
@dataclass
class GenConfig:
    width: int = 1024
    height: int = 1024
    k_per_prompt: int = 1
    seed_base: int = 123456

# ---- Google GenAI (Gemini) ----
class GeminiBackend:
    def __init__(self, model: str = "gemini-2.5-flash-image"):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment.")
        self.client = genai.Client()
        self.model = model

    def generate_one(self, prompt: str, seed: Optional[int] = None) -> List[Image.Image]:
        # Note: seed control varies across providers; we try to keep global seed stable.
        set_global_seed(seed or 0)
        resp = self.client.models.generate_content(model=self.model, contents=[prompt])
        imgs = []
        for part in getattr(resp.candidates[0].content, "parts", []):
            if getattr(part, "inline_data", None) is not None:
                imgs.append(Image.open(BytesIO(part.inline_data.data)))
        return imgs

# ---- FAL AI via huggingface_hub InferenceClient ----
class FalHFBackend:
    def __init__(self, model: str, timeout: int = 180):
        from huggingface_hub import InferenceClient
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("Missing HF_TOKEN in environment for fal-ai provider.")
        self.client = InferenceClient(provider="fal-ai", api_key=hf_token, timeout=timeout)
        self.model = model

    def generate_one(self, prompt: str, seed: Optional[int] = None,
                     width: Optional[int] = None, height: Optional[int] = None) -> Image.Image:
        # Some deployed models respect width/height kwargs; if not, call minimal API.
        kw = {}
        if width:  kw["width"]  = width
        if height: kw["height"] = height
        set_global_seed(seed or 0)
        try:
            img = self.client.text_to_image(prompt, model=self.model, **kw)
        except TypeError:
            img = self.client.text_to_image(prompt, model=self.model)
        return img


# ----------------------------
# Runner
# ----------------------------
def save_manifest_line(path: Path, row: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def ocr_string(img: Image.Image) -> str:
    if not HAVE_OCR:
        return ""
    try:
        # Simple OCR; tweak language packs as needed (e.g., lang="eng")
        txt = pytesseract.image_to_string(img, config="--psm 6")
        return (txt or "").strip()
    except Exception:
        return ""

def run_backend(name: str,
                out_dir: Path,
                prompts: List[str],
                metas: List[Dict[str, Any]],
                cfg: GenConfig):

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    # if manifest_path.exists():
    #     manifest_path.unlink()  # start fresh

    # pick backend
    if name == "gemini":
        backend = GeminiBackend()
        gen = lambda p, s: backend.generate_one(p, s)  # returns List[Image]
    elif name == "qwen_image_fal":
        backend = FalHFBackend("Qwen/Qwen-Image")
        gen = lambda p, s: [backend.generate_one(p, s, cfg.width, cfg.height)]
    elif name == "flux1_fal":
        backend = FalHFBackend("black-forest-labs/FLUX.1-dev")
        gen = lambda p, s: [backend.generate_one(p, s, cfg.width, cfg.height)]
    else:
        raise ValueError(f"Unknown backend: {name}")

    for idx, (prompt, meta) in enumerate(zip(prompts, metas), start=1):
        print(f"\n[{idx}/{len(prompts)}] {prompt}")
        saved_files = []
        ocr_lines = []
        for j in range(cfg.k_per_prompt):
            seed = cfg.seed_base + idx * 10007 + j * 131
            images = gen(prompt, seed)
            if not images:
                print("  (no images returned)")
                continue
            for k, img in enumerate(images):
                if not isinstance(img, Image.Image):
                    # Some providers might return bytes/paths; normalize
                    try:
                        if isinstance(img, (bytes, bytearray)):
                            img = Image.open(BytesIO(img))
                        elif isinstance(img, str) and os.path.isfile(img):
                            img = Image.open(img)
                    except Exception:
                        continue
                if img.mode != "RGB":
                    img = img.convert("RGB")

                fname = f"{idx+args.start:02d}_{slugify(prompt[:60])}_{j}.png" #force idx + 15 because 15 has been generated
                slug = slugify(prompt[:60])
                print(f"  Saving image: {fname}")
                fpath = out_dir / fname
                img.save(fpath)

                # quick OCR pass (optional)
                txt = ocr_string(img) if HAVE_OCR else ""
                if txt:
                    ocr_lines.append(txt)
                saved_files.append(str(fpath))

        row = {
            "index": idx+args.start, #make it count for where starting args.start 
            "prompt": prompt,
            "files": saved_files,
            "backend": name,
            **meta,
        }
        if ocr_lines:
            row["ocr_preview"] = ocr_lines[:3]
            row["ocr_similarity_est"] = norm_edit(" ".join(ocr_lines)[:120], prompt[:120])
        save_manifest_line(manifest_path, row)

    print(f"\nSaved outputs under: {out_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Parti Writing & Symbols generator")
    p.add_argument("--out-root", type=str, default="writing_symbols_outputs",
                   help="Root folder for all backend outputs.")
    p.add_argument("--n", type=int, default=5, help="How many prompts to take from Parti.")
    p.add_argument("--k-per-prompt", type=int, default=1, help="Samples per prompt.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed-base", type=int, default=123456)
    p.add_argument("--backends", type=str, default="gemini,qwen_image_fal,flux1_fal",
                   help="Comma-separated: gemini, qwen_image_fal, flux1_fal")
    p.add_argument("--start", type=int, default=20, help="Starting index offset for prompt selection.")
    args = p.parse_args()

    cfg = GenConfig(width=args.width, height=args.height,
                    k_per_prompt=args.k_per_prompt, seed_base=args.seed_base)

    prompts, metas = load_parti_writing_symbols(n = args.n, start = args.start)

    root = Path(args.out_root)
    suffix_map = {
        "gemini": "nano_banana",
        "flux1_fal": "flux1",
        "qwen_image_fal": "qwen",
    }
    for b in [s.strip() for s in args.backends.split(",") if s.strip()]:
        out_dir = Path(f"{root}_{suffix_map.get(b, '_' + b)}")
        run_backend(b, out_dir, prompts, metas, cfg)