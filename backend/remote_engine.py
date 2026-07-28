"""Adapter engine: delegates generation/editing to an external /generate API."""
from __future__ import annotations

import io
import os
import threading
from typing import Any

import numpy as np
from PIL import Image

from identity_studio import IdentityStudio, ModelDefaults

def _remote_defaults(model: str) -> ModelDefaults:
    if model == "dev":
        return ModelDefaults(steps=28, guidance=4.0, min_size=512)
    return ModelDefaults(steps=4, guidance=1.0, min_size=512)

class RemoteEngine:
    """IdentityStudio-compatible engine backed by a remote /generate API."""

    is_mock = False
    remote = True

    def __init__(self) -> None:
        self.base = os.environ["FACELAB_REMOTE_API"].strip().rstrip("/")
        model = os.environ.get("FACELAB_REMOTE_MODEL", "klein").strip().lower()
        self.default_model = "dev" if model == "dev" else "klein"
        self.timeout = (15, float(os.environ.get("FACELAB_REMOTE_TIMEOUT", "1800")))
        self._face: Any | None = None
        self._model_lock = threading.RLock()
        self._multi_ref: bool | None = None

    @property
    def defaults(self) -> ModelDefaults:
        return _remote_defaults(self.default_model)

    def unload_pipe(self) -> None:
        pass

    def status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "mock": False,
            "engine": "remote",
            "remote_api": self.base,
            "remote_model": self.default_model,
            "image_model": f"remote {self.default_model} @ {self.base}",
            "model_kind": "remote",
            "device": "remote",
            "arcface_loaded": self._face is not None,
            "image_model_loaded": True,
        }
        try:
            import requests

            health = requests.get(self.base + "/health", timeout=8)
            status["remote_ok"] = health.ok
            try:
                payload = health.json()
            except Exception:
                payload = {"raw": health.text[:200]}
            status["remote_health"] = payload

            if isinstance(payload, dict) and "multi_ref" in payload:
                self._multi_ref = bool(payload.get("multi_ref"))
            status["multi_ref"] = self._multi_ref_supported()
        except Exception as exc:
            status["remote_ok"] = False
            status["remote_error"] = str(exc)
        return status

    def _multi_ref_supported(self) -> bool:
        """True when the remote API accepts a repeated `image` field.

        The final API documents repeated `image` fields (current first, anchor
        second) for BOTH klein and dev, so the default (auto) is ON.  A health
        payload may still turn it off by reporting `"multi_ref": false`.  Force
        either way with FACELAB_REMOTE_MULTIREF=1 / 0.  The edit path also falls
        back to a single image if a 2-image request ever errors, so enabling it
        is safe even against an older single-image server.
        """
        env = os.environ.get("FACELAB_REMOTE_MULTIREF", "auto").strip().lower()
        if env in {"1", "true", "on", "yes"}:
            return True
        if env in {"0", "false", "off", "no"}:
            return False
        return True if self._multi_ref is None else bool(self._multi_ref)

    def _get_face(self):
        with self._model_lock:
            if self._face is None:
                from insightface.app import FaceAnalysis

                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                try:
                    app = FaceAnalysis(name="buffalo_l", providers=providers)
                    app.prepare(ctx_id=0, det_size=(640, 640))
                except Exception:
                    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                    app.prepare(ctx_id=-1, det_size=(640, 640))
                self._face = app
            return self._face

    def signature(self, image: Image.Image | None) -> np.ndarray | None:
        if image is None:
            return None
        app = self._get_face()
        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]
        faces = app.get(bgr)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )
        vec = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else None

    def detect_faces(self, image: Image.Image | None) -> list[Any]:
        if image is None:
            return []
        try:
            app = self._get_face()
            bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]
            return list(app.get(bgr))
        except Exception:
            return []

    similarity = staticmethod(IdentityStudio.similarity)

    def _resolve(self, model: str | None) -> str:
        model = (model or self.default_model).strip().lower()
        return "dev" if model == "dev" else "klein"

    def _steps_for(self, model: str, steps: int | None) -> int:
        if steps:
            return max(1, min(50, int(steps)))
        return 28 if model == "dev" else 4

    def _post_generate(
        self,
        prompt: str,
        *,
        image: Image.Image | list[Image.Image] | None = None,
        model: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int = 42,
        width: int | None = None,
        height: int | None = None,
    ) -> Image.Image:
        import requests

        def _dim(v: int) -> int:

            return max(256, min(2048, int(round(int(v) / 16) * 16)))

        model = self._resolve(model)
        data: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "steps": self._steps_for(model, steps),
            "seed": max(0, int(seed)),
        }
        if width:
            data["width"] = _dim(width)
        if height:
            data["height"] = _dim(height)
        if model == "dev":
            data["guidance"] = float(self.defaults.guidance if guidance is None else guidance)

        files = None
        if image is not None:
            images = image if isinstance(image, list) else [image]
            images = images[:4]
            files = []
            for index, item in enumerate(images):
                buf = io.BytesIO()
                item.convert("RGB").save(buf, format="PNG")

                files.append(("image", (f"input{index}.png", buf.getvalue(), "image/png")))

        response = requests.post(self.base + "/generate", data=data, files=files, timeout=self.timeout)
        if not response.ok:
            detail = response.text[:300]
            raise RuntimeError(f"Remote API /generate returned HTTP {response.status_code}: {detail}")
        try:
            out = Image.open(io.BytesIO(response.content))
            out.load()
            return out.convert("RGB")
        except Exception as exc:
            raise RuntimeError("Remote API returned a non-image response.") from exc

    def generate(
        self,
        prompt: str,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int = 42,
        width: int = 768,
        height: int = 768,
        model: str | None = None,
    ) -> Image.Image:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("A non-empty face description is required.")
        return self._post_generate(
            prompt, model=model, steps=steps, guidance=guidance, seed=seed, width=width, height=height
        )

    def generate_many(
        self,
        prompt: str,
        count: int,
        steps: int | None = None,
        guidance: float | None = None,
        base_seed: int = 0,
        width: int = 640,
        height: int = 640,
        model: str | None = None,
    ) -> list[Image.Image]:
        """Fire `count` /generate calls CONCURRENTLY against the remote API.

        The remote server may still serialize on its own model lock, but issuing
        the requests in parallel overlaps network + any server-side queueing and
        is a large speed-up when the API can handle concurrency.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("A non-empty face description is required.")
        count = max(1, int(count))
        try:
            workers = max(1, int(os.environ.get("FACELAB_REMOTE_CONCURRENCY", "4")))
        except Exception:
            workers = 4

        from concurrent.futures import ThreadPoolExecutor

        def one(i: int) -> Image.Image:
            return self._post_generate(
                prompt, model=model, steps=steps, guidance=guidance,
                seed=base_seed + i, width=width, height=height,
            )

        with ThreadPoolExecutor(max_workers=min(workers, count)) as ex:
            return list(ex.map(one, range(count)))

    def edit(
        self,
        current: Image.Image,
        instruction: str,
        ref_image: Image.Image | None,
        mode: str = "preserve",
        steps: int | None = None,
        guidance: float | None = None,
        seed: int = 0,
        mask: dict[str, float] | None = None,
        model: str | None = None,
    ) -> Image.Image:
        if current is None:
            raise ValueError("A current image is required.")
        if mode.lower().strip() != "preserve":
            raise ValueError("mode must be 'preserve'.")
        current = current.convert("RGB")
        instruction = (instruction or "").strip() or "make a subtle natural edit"
        alpha_full = IdentityStudio._mask_alpha(mask, current.size)

        if alpha_full is not None:

            bx0, by0, bx1, by1 = alpha_full.getbbox()
            W, H = current.size
            bbox = {"x": bx0 / W, "y": by0 / H, "w": (bx1 - bx0) / W, "h": (by1 - by0) / H}
            context_box = IdentityStudio._region_context_box(
                current.size, bbox, padding=IdentityStudio._region_pad()
            )
            source = current.crop(context_box)
            prompt = (
                "This image is a small close-up patch cropped from a larger portrait. "
                f"Apply only this local edit to it: {instruction}. Keep the same skin, "
                "texture, colour and lighting and change nothing else. Do NOT add, insert, "
                "replace, shrink or duplicate a face, head, portrait or person — this is only "
                "a small region of a bigger photo, not a whole face."
            )
            w, h = IdentityStudio._aligned_size(source.width, source.height, self.defaults.min_size)
            out = self._post_generate(
                prompt, image=source, model=model, steps=steps, guidance=guidance,
                seed=seed, width=w, height=h,
            )
            return IdentityStudio._composite_region(current, out, context_box, alpha_full)

        w, h = IdentityStudio._aligned_size(current.width, current.height, self.defaults.min_size)

        def _single() -> Image.Image:

            prompt = (
                f"Apply this edit to the portrait: {instruction}. Keep the exact same person "
                "and identity: identical facial structure, proportions, skin tone, and "
                "distinguishing features. Apply only the requested change and do not turn "
                "this into a different person."
            )
            return self._post_generate(
                prompt, image=current, model=model, steps=steps, guidance=guidance,
                seed=seed, width=w, height=h,
            )

        if ref_image is not None and self._multi_ref_supported():

            prompt = (
                "Image 1 is the portrait to edit and image 2 is the identity anchor. "
                f"Apply this edit to image 1: {instruction}. Preserve the recognizable "
                "identity, facial structure and proportions from image 2. Keep the result "
                "photorealistic and do not blend in a different person."
            )
            try:
                out = self._post_generate(
                    prompt, image=[current, ref_image.convert("RGB")], model=model,
                    steps=steps, guidance=guidance, seed=seed, width=w, height=h,
                )
            except Exception:
                out = _single()
        else:
            out = _single()
        return out.resize(current.size, Image.Resampling.LANCZOS)
