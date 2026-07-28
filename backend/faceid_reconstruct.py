
from __future__ import annotations

import os
import sys
import threading
from typing import Any

import numpy as np
from PIL import Image

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {}

def _disabled() -> bool:
    return os.environ.get("FACELAB_FACEID", "auto").lower() in ("0", "off", "no", "false")

def _load():
    """Load the IP-Adapter-FaceID pipeline once. Returns the model or None."""
    with _LOCK:
        if "model" in _STATE:
            return _STATE["model"]
        if _disabled():
            _STATE["model"] = None
            _STATE["error"] = "FACELAB_FACEID is off."
            return None
        try:
            repo = os.environ.get("FACELAB_FACEID_REPO", "").strip()
            if repo and repo not in sys.path:
                sys.path.append(repo)

            import torch
            from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
            from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

            base = os.environ.get("FACELAB_FACEID_BASE", "SG161222/Realistic_Vision_V4.0_noVAE")
            vae_path = os.environ.get("FACELAB_FACEID_VAE", "stabilityai/sd-vae-ft-mse")
            ip_ckpt = os.environ.get("FACELAB_FACEID_CKPT", "").strip()
            if not ip_ckpt or not os.path.exists(ip_ckpt):
                from huggingface_hub import hf_hub_download
                ip_ckpt = hf_hub_download("h94/IP-Adapter-FaceID", "ip-adapter-faceid_sd15.bin")

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            scheduler = DDIMScheduler(
                num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1,
            )
            vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=dtype)
            pipe = StableDiffusionPipeline.from_pretrained(
                base, torch_dtype=dtype, scheduler=scheduler, vae=vae,
                feature_extractor=None, safety_checker=None,
            )
            model = IPAdapterFaceID(pipe, ip_ckpt, device)
            _STATE["model"] = model
            _STATE["torch"] = torch
            _STATE["error"] = None
            return model
        except Exception as exc:
            _STATE["model"] = None
            _STATE["error"] = f"{type(exc).__name__}: {exc}"
            return None

def available() -> bool:
    return _load() is not None

def status() -> dict[str, Any]:
    ok = _load() is not None
    return {"available": ok, "error": None if ok else _STATE.get("error")}

_DEFAULT_PROMPT = (
    "front portrait of a person, passport-style photo, frontal, looking at the camera, "
    "neutral expression, even studio lighting, plain background, photorealistic, sharp focus"
)
_NEG = "side view, profile, blurry, deformed, cropped, mask, occluded, low quality, cartoon"

def reconstruct(embedding, prompt: str | None = None, seed: int = 7,
                steps: int = 30, width: int = 512, height: int = 512,
                guidance: float = 6.0) -> Image.Image:
    """Generate one realistic frontal face for the given 512-d ArcFace embedding."""
    model = _load()
    if model is None:
        raise RuntimeError(_STATE.get("error") or "IP-Adapter-FaceID is not available.")
    torch = _STATE["torch"]
    emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    if emb.shape[1] != 512:
        raise ValueError(f"Expected a 512-d ArcFace embedding, got {emb.shape[1]}.")
    faceid_embeds = torch.from_numpy(emb)
    with _LOCK:
        images = model.generate(
            prompt=(prompt or _DEFAULT_PROMPT), negative_prompt=_NEG,
            faceid_embeds=faceid_embeds, num_samples=1,
            width=width, height=height, num_inference_steps=int(steps),
            guidance_scale=float(guidance), seed=int(seed),
        )
    return images[0].convert("RGB")
