"""Image generation, identity-preserving editing, and ArcFace embeddings."""
from __future__ import annotations

import base64
import gc
import hashlib
import io
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

def _cuda_available() -> bool:
    if os.environ.get("FACELAB_MOCK", "0") == "1":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False

MOCK = not _cuda_available()

@dataclass(frozen=True)
class ModelDefaults:
    steps: int
    guidance: float
    min_size: int

class IdentityStudio:
    """Reusable generation, editing, and ArcFace engine."""

    def __init__(self) -> None:
        requested = os.environ.get("FACELAB_IMAGE_MODEL", "klein").strip().lower()
        self.model_kind = "dev" if requested == "dev" else "klein"
        self.model_id = (
            os.environ.get("FACELAB_DEV_MODEL", "diffusers/FLUX.2-dev-bnb-4bit")
            if self.model_kind == "dev"
            else os.environ.get("FACELAB_KLEIN_MODEL", "black-forest-labs/FLUX.2-klein-4B")
        )
        self.defaults = (
            ModelDefaults(steps=28, guidance=4.0, min_size=512)
            if self.model_kind == "dev"
            else ModelDefaults(steps=4, guidance=1.0, min_size=512)
        )
        self._pipe: Any | None = None
        self._face: Any | None = None
        self._fill: Any | None = None
        self._model_lock = threading.RLock()

    @property
    def is_mock(self) -> bool:
        return MOCK

    def status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "mock": MOCK,
            "engine": "mock" if MOCK else "local",
            "image_model": "mock" if MOCK else self.model_id,
            "model_kind": "mock" if MOCK else self.model_kind,
            "image_model_loaded": self._pipe is not None,
            "arcface_loaded": self._face is not None,
        }
        if not MOCK:
            try:
                import torch

                status.update(
                    device=torch.cuda.get_device_name(0),
                    vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                )
            except Exception as exc:
                status.update(device="cuda", device_warning=str(exc))
        else:
            status["device"] = "mock"
        return status

    def unload_pipe(self) -> None:
        """Release the diffusion pipeline when another large model needs VRAM."""
        with self._model_lock:
            self._pipe = None
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    def _get_pipe(self):
        if MOCK:
            raise RuntimeError("The diffusion pipeline is unavailable in mock mode.")
        with self._model_lock:
            if self._pipe is not None:
                return self._pipe

            import torch

            if self.model_kind == "klein":
                from diffusers import Flux2KleinPipeline

                pipe = Flux2KleinPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                )
            else:
                from diffusers import Flux2Pipeline

                pipe = Flux2Pipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                )

            use_offload = os.environ.get("FACELAB_CPU_OFFLOAD", "auto").lower()
            if use_offload == "1":
                pipe.enable_model_cpu_offload()
            elif use_offload == "0":
                pipe.to("cuda")
            else:
                free_gb = torch.cuda.mem_get_info()[0] / 1e9
                needed = 18 if self.model_kind == "klein" else 34
                if free_gb >= needed:
                    pipe.to("cuda")
                else:
                    pipe.enable_model_cpu_offload()

            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            self._pipe = pipe
            return self._pipe

    def _get_face(self):
        if MOCK:
            return None
        with self._model_lock:
            if self._face is None:
                from insightface.app import FaceAnalysis

                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                app = FaceAnalysis(name="buffalo_l", providers=providers)
                try:
                    app.prepare(ctx_id=0, det_size=(640, 640))
                except Exception:

                    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                    app.prepare(ctx_id=-1, det_size=(640, 640))
                self._face = app
            return self._face

    def signature(self, image: Image.Image | None) -> np.ndarray | None:
        """Return the largest detected face's normalized 512-d ArcFace vector."""
        if image is None:
            return None
        image = image.convert("RGB")
        if MOCK:
            return _mock_signature(image)
        app = self._get_face()
        bgr = np.asarray(image)[:, :, ::-1]
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
        """Return the raw InsightFace detections (bbox, kps, embedding).

        Used by the public-figure scraper to crop a real reference face and
        rank candidates by size / frontal pose.  Returns [] in mock mode or
        when no detector is available.
        """
        if image is None or MOCK:
            return []
        app = self._get_face()
        if app is None:
            return []
        bgr = np.asarray(image.convert("RGB"))[:, :, ::-1]
        try:
            return list(app.get(bgr))
        except Exception:
            return []

    @staticmethod
    def similarity(a: np.ndarray | list[float] | None, b: np.ndarray | list[float] | None) -> float | None:
        if a is None or b is None:
            return None
        av = np.asarray(a, dtype=np.float32).reshape(-1)
        bv = np.asarray(b, dtype=np.float32).reshape(-1)
        if av.shape != bv.shape or av.size == 0:
            return None
        an = float(np.linalg.norm(av))
        bn = float(np.linalg.norm(bv))
        if an == 0 or bn == 0:
            return None
        return float(np.dot(av / an, bv / bn))

    @staticmethod
    def _aligned_size(width: int, height: int, min_size: int = 512, max_size: int = 1024) -> tuple[int, int]:
        """Preserve aspect ratio and use FLUX-friendly multiples of 16."""
        width, height = max(1, int(width)), max(1, int(height))
        scale = max(min_size / min(width, height), 1.0)
        if max(width, height) * scale > max_size:
            scale = max_size / max(width, height)
        w = max(256, int(round(width * scale / 16) * 16))
        h = max(256, int(round(height * scale / 16) * 16))
        return min(max_size, w), min(max_size, h)

    def generate(
        self,
        prompt: str,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int = 42,
        width: int = 768,
        height: int = 768,
    ) -> Image.Image:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("A non-empty face description is required.")
        if MOCK:
            return _mock_portrait(prompt, seed)

        import torch

        pipe = self._get_pipe()
        steps = int(steps or self.defaults.steps)
        guidance = float(self.defaults.guidance if guidance is None else guidance)
        width, height = self._aligned_size(width, height, self.defaults.min_size)
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        kwargs: dict[str, Any] = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        if self.model_kind == "dev":
            kwargs["caption_upsample_temperature"] = 0.15
        with self._model_lock, torch.inference_mode():
            return pipe(**kwargs).images[0].convert("RGB")

    def generate_many(
        self,
        prompt: str,
        count: int,
        steps: int | None = None,
        guidance: float | None = None,
        base_seed: int = 0,
        width: int = 640,
        height: int = 640,
    ) -> list[Image.Image]:


       
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("A non-empty face description is required.")
        count = max(1, int(count))
        if MOCK:
            return [_mock_portrait(prompt, base_seed + i) for i in range(count)]

        import torch

        pipe = self._get_pipe()
        steps = int(steps or self.defaults.steps)
        guidance = float(self.defaults.guidance if guidance is None else guidance)
        width, height = self._aligned_size(width, height, self.defaults.min_size)
        try:
            chunk = max(1, int(os.environ.get("FACELAB_BATCH", "6")))
        except Exception:
            chunk = 6

        out: list[Image.Image] = []
        i = 0
        while i < count:
            c = min(chunk, count - i)
            gens = [torch.Generator(device="cuda").manual_seed(int(base_seed + i + j)) for j in range(c)]
            kwargs: dict[str, Any] = dict(
                prompt=prompt,
                num_images_per_prompt=c,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gens,
            )
            if self.model_kind == "dev":
                kwargs["caption_upsample_temperature"] = 0.15
            try:
                with self._model_lock, torch.inference_mode():
                    images = pipe(**kwargs).images
                out.extend(im.convert("RGB") for im in images)
                i += c
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
        return out

    @staticmethod
    def _normalise_box(mask: dict[str, float] | None) -> dict[str, float] | None:
        if not mask:
            return None
        x = min(1.0, max(0.0, float(mask.get("x", 0.0))))
        y = min(1.0, max(0.0, float(mask.get("y", 0.0))))
        w = min(1.0 - x, max(0.0, float(mask.get("w", 0.0))))
        h = min(1.0 - y, max(0.0, float(mask.get("h", 0.0))))
        return None if w <= 0 or h <= 0 else {"x": x, "y": y, "w": w, "h": h}

    @staticmethod
    def _normalise_polygon(mask: dict | None) -> list[tuple[float, float]] | None:
        """Accept a rectangle {x,y,w,h} OR {polygon:[[x,y],…]} and return a
        clamped, normalised polygon (a rectangle becomes its 4 corners).

        A feature-shaped polygon is what lets a region edit REPLACE a whole
        feature (eyebrow, lips) instead of a rectangle slicing across it — the
        rectangle slice was the cause of the doubled / overlapping edge.
        """
        if not mask:
            return None
        pts = mask.get("polygon") if isinstance(mask, dict) else None
        if pts:
            poly = []
            for p in pts:
                try:
                    x = min(1.0, max(0.0, float(p[0])))
                    y = min(1.0, max(0.0, float(p[1])))
                except Exception:
                    continue
                poly.append((x, y))
            if len(poly) < 3:
                return None
            return poly
        box = IdentityStudio._normalise_box(mask)
        if not box:
            return None
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    @staticmethod
    def _poly_bbox(poly: list[tuple[float, float]]) -> dict[str, float]:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return {"x": x0, "y": y0, "w": max(0.0, x1 - x0), "h": max(0.0, y1 - y0)}

    @staticmethod
    def _mask_alpha(mask, size: tuple[int, int]) -> Image.Image | None:
        """Unify any mask spec into a full-resolution 'L' alpha (255=selected).

        Accepts (in priority order):
          * {"mask_png": dataurl}   a PAINTED raster mask (the brush tool) — this
            is the professional path: the selected pixels are exactly what the
            user painted, any shape, any size.
          * {"polygon": [[x,y],…]}  a normalised polygon (smart-click / lasso).
          * {"x","y","w","h"}       a normalised rectangle (legacy).
        Returns None when the mask is empty / unparseable (→ whole-face edit).
        """
        W, H = size
        if isinstance(mask, dict) and mask.get("mask_png"):
            try:
                data = str(mask["mask_png"])
                raw = base64.b64decode(data.split(",", 1)[1] if "," in data else data)
                m = Image.open(io.BytesIO(raw)).convert("L")
            except Exception:
                return None
            if m.size != (W, H):
                m = m.resize((W, H), Image.Resampling.LANCZOS)
            return m if m.getbbox() else None
        poly = IdentityStudio._normalise_polygon(mask)
        if not poly:
            return None
        alpha = Image.new("L", (W, H), 0)
        ImageDraw.Draw(alpha).polygon([(int(x * W), int(y * H)) for x, y in poly], fill=255)
        return alpha if alpha.getbbox() else None

    @staticmethod
    def _region_pad() -> float:
        """Context padding fraction around a selection. More context = cleaner blend;
        the generation canvas stays small either way, so cost barely changes."""
        try:
            return max(0.15, min(1.5, float(os.environ.get("FACELAB_REGION_PAD", "0.55"))))
        except Exception:
            return 0.55

    def _region_steps(self, steps: int) -> int:
        """A local inpaint converges in far fewer steps than a full generation.
        Klein is already 4-step distilled; only the heavy dev model is capped.
        This is the main time saver for region edits."""
        steps = max(1, int(steps))
        if self.model_kind == "dev":
            try:
                cap = int(os.environ.get("FACELAB_REGION_STEPS", "8"))
            except Exception:
                cap = 8
            return max(1, min(steps, cap))
        return steps

    def _get_fill_pipe(self):
        """Optional dedicated inpaint pipeline. Loaded ONLY if FACELAB_FILL_MODEL is
        set (e.g. a FLUX Fill checkpoint). Off by default because it adds a model
        download and VRAM; the default crop-edit path needs no second model."""
        if MOCK:
            return None
        model = os.environ.get("FACELAB_FILL_MODEL", "").strip()
        if not model:
            return None
        with self._model_lock:
            if self._fill is not None:
                return self._fill
            import torch
            import diffusers

            pipe = None
            for cls_name in ("Flux2FillPipeline", "FluxFillPipeline", "Flux2InpaintPipeline", "FluxInpaintPipeline"):
                cls = getattr(diffusers, cls_name, None)
                if cls is None:
                    continue
                try:
                    pipe = cls.from_pretrained(model, torch_dtype=torch.bfloat16)
                    break
                except Exception:
                    pipe = None
            if pipe is None:
                self._fill = None
                return None
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                try:
                    pipe.to("cuda")
                except Exception:
                    pass
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            self._fill = pipe
            return self._fill

    def _fill_region(self, fill, source_crop, alpha_full, context_box, current, instruction, seed):
        """True mask-conditioned inpaint, run ONLY on the small crop (cheap), then
        hard-composited so nothing outside the selection can change.

        `alpha_full` is the full-resolution painted mask; the crop mask is that
        mask cropped to the context box, so the inpaint follows the exact painted
        shape (any shape, and any number of disjoint areas).
        """
        import torch

        cx0, cy0, cx1, cy1 = context_box
        crop_w, crop_h = source_crop.size
        crop_mask = alpha_full.crop(context_box)

        w, h = self._aligned_size(crop_w, crop_h, self.defaults.min_size)
        src = source_crop.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        msk = crop_mask.resize((w, h), Image.Resampling.NEAREST)
        try:
            fill_guidance = float(os.environ.get("FACELAB_FILL_GUIDANCE", "30"))
        except Exception:
            fill_guidance = 30.0
        prompt = f"{instruction}. Blend naturally; keep the surrounding skin, texture and lighting unchanged."
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        with self._model_lock, torch.inference_mode():
            out = fill(
                prompt=prompt,
                image=src,
                mask_image=msk,
                width=w,
                height=h,
                num_inference_steps=self._region_steps(self.defaults.steps),
                guidance_scale=fill_guidance,
                generator=generator,
            ).images[0].convert("RGB")
        return self._composite_region(current, out, context_box, alpha_full)

    @staticmethod
    def _region_context_box(size: tuple[int, int], box: dict[str, float], padding: float = 0.45) -> tuple[int, int, int, int]:
        W, H = size
        x0 = box["x"] * W
        y0 = box["y"] * H
        x1 = (box["x"] + box["w"]) * W
        y1 = (box["y"] + box["h"]) * H
        pad_x = max(24.0, (x1 - x0) * padding)
        pad_y = max(24.0, (y1 - y0) * padding)
        return (
            max(0, int(x0 - pad_x)),
            max(0, int(y0 - pad_y)),
            min(W, int(x1 + pad_x)),
            min(H, int(y1 + pad_y)),
        )

    @staticmethod
    def _composite_region(
        original: Image.Image,
        edited_crop: Image.Image,
        context_box: tuple[int, int, int, int],
        alpha_or_mask,
    ) -> Image.Image:
        """Composite an edited crop back through a PAINTED / feature-shaped alpha.

        `alpha_or_mask` is a full-resolution 'L' mask (255 = user-selected pixels)
        — from the brush tool, a smart-click feature, or a legacy polygon/rect.
        The alpha is feathered INWARD only and hard-clipped to the selection, so
        every pixel the user did not select stays byte-for-byte identical, and
        whatever shape they painted is exactly what gets replaced.
        """
        original = original.convert("RGB")
        W, H = original.size
        cx0, cy0, cx1, cy1 = context_box
        edited_crop = edited_crop.convert("RGB").resize((cx1 - cx0, cy1 - cy0), Image.Resampling.LANCZOS)

        if isinstance(alpha_or_mask, Image.Image):
            hard = alpha_or_mask.convert("L")
            if hard.size != (W, H):
                hard = hard.resize((W, H), Image.Resampling.LANCZOS)
        else:
            hard = IdentityStudio._mask_alpha(alpha_or_mask, (W, H))
        if hard is None or not hard.getbbox():
            return original
        bx0, by0, bx1, by1 = hard.getbbox()

        if os.environ.get("FACELAB_REGION_TONE", "1") != "0":
            edited_crop = _match_crop_tone(
                edited_crop,
                original.crop(context_box),
                (bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0),
            )

        edited_full = original.copy()
        edited_full.paste(edited_crop, (cx0, cy0))

        feather = max(5, int(min(max(1, bx1 - bx0), max(1, by1 - by0)) * 0.10))
        alpha = hard.filter(ImageFilter.GaussianBlur(radius=feather))

        alpha = ImageChops.multiply(alpha, hard)
        return Image.composite(edited_full, original, alpha)

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
    ) -> Image.Image:
        
        if current is None:
            raise ValueError("A current image is required.")
        mode = mode.lower().strip()
        if mode != "preserve":
            raise ValueError("mode must be 'preserve'.")
        if ref_image is None:
            raise ValueError("A locked reference image is required.")

        current = current.convert("RGB")
        ref_image = ref_image.convert("RGB")
        alpha_full = self._mask_alpha(mask, current.size)

        if MOCK:
            edited = _mock_edit(current, instruction)
            if alpha_full is not None:
                box = (0, 0, current.width, current.height)
                return self._composite_region(current, edited, box, alpha_full)
            return edited

        import torch

        pipe = self._get_pipe()
        steps = int(steps or self.defaults.steps)
        guidance = float(self.defaults.guidance if guidance is None else guidance)
        instruction = (instruction or "").strip() or "make a subtle natural edit"

        context_box = None
        source = current
        if alpha_full is not None:
            bx0, by0, bx1, by1 = alpha_full.getbbox()
            W, H = current.size
            bbox = {"x": bx0 / W, "y": by0 / H, "w": (bx1 - bx0) / W, "h": (by1 - by0) / H}
            context_box = self._region_context_box(current.size, bbox, padding=self._region_pad())
            source = current.crop(context_box)
        region_edit = bool(alpha_full is not None and context_box is not None)

        if region_edit:
            fill = self._get_fill_pipe()
            if fill is not None:
                try:
                    return self._fill_region(fill, source, alpha_full, context_box, current, instruction, seed)
                except Exception:
                    pass

        if region_edit:

            prompt = (
                "This image is a small close-up patch cropped from a larger portrait. "
                f"Apply only this local edit to it: {instruction}. Keep the same skin, "
                "texture, colour and lighting and change nothing else. Do NOT add, insert, "
                "replace, shrink or duplicate a face, head, portrait or person — this is only "
                "a small region of a bigger photo, not a whole face."
            )
            edit_images = [source]
        else:
            prompt = (
                "Image 1 is the current portrait and image 2 is the identity anchor. "
                f"Apply this edit to image 1: {instruction}. Preserve the exact person, "
                "facial proportions, and recognizable identity from image 2. Keep the result "
                "photorealistic and do not blend in a different person."
            )
            edit_images = [source, ref_image]

        width, height = self._aligned_size(source.width, source.height, self.defaults.min_size)

        def run(pipe_images: list) -> Image.Image:
            kwargs: dict[str, Any] = dict(
                prompt=prompt,
                image=pipe_images,
                width=width,
                height=height,
                num_inference_steps=self._region_steps(steps) if region_edit else steps,
                guidance_scale=guidance,
                generator=torch.Generator(device="cuda").manual_seed(int(seed)),
            )
            if self.model_kind == "dev":
                kwargs["caption_upsample_temperature"] = 0.15
            with self._model_lock, torch.inference_mode():
                return pipe(**kwargs).images[0].convert("RGB")

        primary = edit_images if self.model_kind == "klein" else [edit_images]
        fallback = [edit_images] if self.model_kind == "klein" else edit_images
        try:
            out = run(primary)
        except Exception as exc:
            if "match number of prompts" in str(exc):
                out = run(fallback)
            else:
                raise

        if region_edit:
            return self._composite_region(current, out, context_box, alpha_full)
        return out.resize(current.size, Image.Resampling.LANCZOS)

def _match_crop_tone(
    edited_crop: Image.Image,
    original_crop: Image.Image,
    sel_box: tuple[int, int, int, int],
) -> Image.Image:
   
    edited = np.asarray(edited_crop.convert("RGB"), dtype=np.float32)
    original = np.asarray(
        original_crop.convert("RGB").resize(edited_crop.size), dtype=np.float32
    )
    h, w = edited.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in sel_box)
    ring = np.ones((h, w), dtype=bool)
    ring[max(0, y0):max(0, min(h, y1)), max(0, x0):max(0, min(w, x1))] = False
    if int(ring.sum()) < 256:
        return edited_crop
    corrected = np.empty_like(edited)
    for channel in range(3):
        e = edited[..., channel][ring]
        o = original[..., channel][ring]
        e_std = float(e.std())
        gain = (float(o.std()) / e_std) if e_std > 1e-3 else 1.0
        gain = min(1.5, max(0.66, gain))
        corrected[..., channel] = (edited[..., channel] - float(e.mean())) * gain + float(o.mean())
    return Image.fromarray(np.clip(corrected, 0.0, 255.0).astype(np.uint8))

def _code_from(value: object) -> float:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer % 1000003) / 1000003.0

def _embed_code(img: Image.Image, code: float) -> Image.Image:
    img = img.copy().convert("RGB")
    img.putpixel((0, 0), (int(max(0.0, min(1.0, code)) * 255), 7, 7))
    return img

def _read_code(img: Image.Image) -> float:
    return img.convert("RGB").getpixel((0, 0))[0] / 255.0

def _mock_signature(img: Image.Image) -> np.ndarray:
    angle = _read_code(img) * np.pi
    vec = np.zeros(512, np.float32)
    vec[0], vec[1] = np.cos(angle), np.sin(angle)
    return vec

def _mock_portrait(prompt: str, seed: int, code: float | None = None, tint: float = 0.0) -> Image.Image:
    code = _code_from(f"{prompt}|{seed}") if code is None else float(code)
    W = H = 512
    img = Image.new("RGB", (W, H), (18, 24, 38))
    d = ImageDraw.Draw(img)
    hue = int(code * 300)

    def skin(offset: int = 0) -> tuple[int, int, int]:
        r = 150 + int(70 * np.cos(code * 6.28)) + int(tint * 40)
        g = 120 + int(50 * np.sin(code * 6.28))
        b = 100 + int(40 * np.cos(code * 3.14))
        return tuple(max(0, min(255, value + offset)) for value in (r, g, b))

    d.ellipse([W * 0.29, H * 0.13, W * 0.71, H * 0.64], fill=(45 + hue % 65, 30, 20))
    d.ellipse([W * 0.33, H * 0.22, W * 0.67, H * 0.75], fill=skin())
    eye_spacing = 0.06 + 0.03 * code
    for side in (-1, 1):
        ex = W * (0.5 + side * eye_spacing)
        d.ellipse([ex - 14, H * 0.42 - 8, ex + 14, H * 0.42 + 8], fill=(245, 242, 235))
        d.ellipse([ex - 5, H * 0.42 - 5, ex + 5, H * 0.42 + 5], fill=(40, 28, 20))
    d.line([W * 0.5, H * 0.46, W * 0.49, H * 0.57], fill=(70, 50, 40), width=4)
    d.arc([W * 0.42, H * 0.56, W * 0.58, H * 0.67], 15, 165, fill=(120, 60, 55), width=5)
    d.ellipse([W * 0.35, H * 0.52, W * 0.65, H * 0.76], outline=(40, 28, 20), width=int(7 + 9 * code))
    return _embed_code(img, code)

def _mock_edit(current: Image.Image, instruction: str) -> Image.Image:
    """Identity-preserving mock edit: same identity code, instruction-driven tint."""
    current_code = _read_code(current)
    tint = _code_from(instruction) - 0.5
    return _mock_portrait("preserve", 0, code=current_code, tint=tint)

ENGINE = IdentityStudio()
