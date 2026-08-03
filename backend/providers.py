"""Remote image-generation back ends.

FaceLab can generate through four routes, chosen by ``FACELAB_PROVIDER``:

  ``fal``         FLUX.2 [dev] on fal.ai (``FAL_KEY``) — needs internet + a key
  ``openrouter``  hosted models over the OpenRouter API — needs internet + a key
  ``lab``         the FLUX.2 server on the lab machine (``FLUX_API_URL``) — offline
  ``local``       a diffusers pipeline inside this process (needs CUDA + weights)

``fal``, ``openrouter`` and ``lab`` are all "remote" to the rest of the app;
only the transport differs, so the switch is one environment variable. They are
not interchangeable per environment, though: the lab network has no internet, so
the hosted routes only work on an internet-connected host, and ``lab`` only
works where the FLUX server is reachable.

``fal`` is the route the pixel-exact half-face completion prefers, because it is
the only one that exposes a FLUX.2 *edit* endpoint honouring an exact pixel
``image_size`` — a resized canvas would move the seam. See ``fal_provider``.

OpenRouter's image endpoint is ``POST /api/v1/images``: model + prompt in,
base64 images out. It has no ``seed`` parameter, so the seeds FaceLab threads
through its call sites are accepted and ignored on this route — repeat calls
vary by sampling instead. That still satisfies the burst-then-average flow in
``face_pipeline.reduce_to_query``, but the variation is not reproducible.
"""
from __future__ import annotations

import base64
import io
import os
import threading
from typing import Any

from PIL import Image

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Continuity with the lab server: same model family, so switching routes does
# not change the look of a portrait much. Both are overridable.
DEFAULT_IMAGE_MODEL = "black-forest-labs/flux.2-pro"
# Editing wants strong instruction-following with reference images attached.
DEFAULT_EDIT_MODEL = "google/gemini-2.5-flash-image"
# OpenRouter caps n at 10 per request.
MAX_BATCH = 10
ASPECT_RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9")

_COST_LOCK = threading.Lock()
_SPEND = {"requests": 0, "images": 0, "cost": 0.0}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def api_key() -> str:
    return _env("OPENROUTER_API_KEY")


def base_url() -> str:
    return (_env("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def image_model() -> str:
    return _env("OPENROUTER_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL


def edit_model() -> str:
    return _env("OPENROUTER_EDIT_MODEL") or edit_fallback()


def edit_fallback() -> str:
    return DEFAULT_EDIT_MODEL


def ref_limit() -> int:
    try:
        return max(1, int(_env("OPENROUTER_MAX_REFS", "4")))
    except ValueError:
        return 4


def timeout() -> float:
    try:
        return float(_env("OPENROUTER_TIMEOUT", "300"))
    except ValueError:
        return 300.0


def resolve_provider() -> str:
    """Pick the generation back end. An explicit FACELAB_PROVIDER always wins.

    Auto-detection prefers the hosted routes when a key is present, because
    those are what work on a developer machine with no GPU. fal comes first: it
    serves FLUX.2 [dev] itself rather than a proxy, and it is the only route
    with an edit endpoint precise enough for half-face completion.
    """
    choice = _env("FACELAB_PROVIDER").lower()
    if choice in ("fal", "openrouter", "lab", "local"):
        return choice
    if choice in ("fal.ai", "flux2", "flux-2"):  # friendly aliases for the fal route
        return "fal"
    if choice in ("flux", "remote", "server"):  # friendly aliases for the lab route
        return "lab"
    try:
        import fal_provider

        if fal_provider.available():
            return "fal"
    except Exception:
        pass
    if api_key():
        return "openrouter"
    if _env("FLUX_API_URL"):
        return "lab"
    return "local"


def _ratio_value(ratio: str) -> float:
    left, right = ratio.split(":")
    return int(left) / int(right)


def aspect_ratio_for(width: int, height: int) -> str:
    """Nearest ratio OpenRouter accepts — it takes a ratio, not a pixel size."""
    target = max(1, int(width)) / max(1, int(height))
    return min(ASPECT_RATIOS, key=lambda r: abs(target - _ratio_value(r)))


def resolution_for(width: int, height: int) -> str:
    """Map the longest edge onto OpenRouter's resolution tiers."""
    longest = max(int(width), int(height))
    if longest <= 640:
        return "512"
    if longest <= 1280:
        return "1K"
    if longest <= 2560:
        return "2K"
    return "4K"


def to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def spend() -> dict[str, Any]:
    """Cumulative OpenRouter usage for this process — it is a paid API."""
    with _COST_LOCK:
        return {
            "requests": _SPEND["requests"],
            "images": _SPEND["images"],
            "cost_usd": round(_SPEND["cost"], 4),
        }


def _record(images: int, cost: float) -> None:
    with _COST_LOCK:
        _SPEND["requests"] += 1
        _SPEND["images"] += images
        _SPEND["cost"] += cost


def _explain(status: int, detail: str) -> str:
    hint = {
        401: "OPENROUTER_API_KEY is missing or invalid.",
        402: "OpenRouter credits exhausted — top up the account.",
        403: "The model refused this prompt, or the key lacks access to it.",
        404: f"Unknown model slug. Check OPENROUTER_IMAGE_MODEL against {base_url()}/images/models.",
        429: "OpenRouter rate limit — slow down or lower FACELAB_QUERY_N.",
    }.get(status, "")
    return f"OpenRouter error {status}: {detail}{(' ' + hint) if hint else ''}"


def _post(payload: dict[str, Any], request_timeout: float) -> dict[str, Any]:
    import requests

    key = api_key()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it in .env, or set "
            "FACELAB_PROVIDER=lab to generate on the lab FLUX server instead."
        )
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # Optional attribution headers OpenRouter uses for its rankings.
    referer = _env("OPENROUTER_SITE_URL")
    title = _env("OPENROUTER_APP_TITLE", "FaceLab")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    try:
        response = requests.post(
            f"{base_url()}/images", json=payload, headers=headers, timeout=request_timeout
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"OpenRouter unreachable at {base_url()} — this host needs internet access. "
            "On the offline lab network set FACELAB_PROVIDER=lab instead."
        ) from exc

    if response.status_code != 200:
        try:
            body = response.json()
            detail = str(body.get("error", {}).get("message") or body.get("detail") or body)
        except Exception:
            detail = response.text[:300]
        raise RuntimeError(_explain(response.status_code, detail[:400]))
    return response.json()


def _decode(payload: dict[str, Any]) -> list[Image.Image]:
    entries = payload.get("data") or []
    images: list[Image.Image] = []
    for entry in entries:
        raw = entry.get("b64_json")
        if not raw:
            continue
        # Some providers hand back a full data URL rather than bare base64.
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        try:
            images.append(Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB"))
        except Exception as exc:  # a corrupt entry should not sink the whole batch
            print(f"[openrouter] skipping undecodable image: {exc}", flush=True)
    cost = 0.0
    try:
        cost = float((payload.get("usage") or {}).get("cost") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    _record(len(images), cost)
    if not images:
        raise RuntimeError(
            "OpenRouter returned no image. The model may have declined the prompt; "
            "try rewording the description."
        )
    return images


def _payload(
    prompt: str,
    *,
    count: int,
    width: int,
    height: int,
    references: list[Image.Image] | None,
    model: str | None,
) -> dict[str, Any]:
    refs = list(references or [])
    chosen = model or (edit_model() if refs else image_model())
    body: dict[str, Any] = {
        "model": chosen,
        "prompt": prompt,
        "n": count,
        "aspect_ratio": aspect_ratio_for(width, height),
        "resolution": resolution_for(width, height),
        "output_format": "png",
    }
    quality = _env("OPENROUTER_QUALITY")
    if quality:
        body["quality"] = quality
    if refs:
        limit = ref_limit()
        if len(refs) > limit:
            print(
                f"[openrouter] sending {limit} of {len(refs)} reference images "
                "(raise OPENROUTER_MAX_REFS to send more)",
                flush=True,
            )
            refs = refs[:limit]
        body["input_references"] = [
            {"type": "image_url", "image_url": {"url": to_data_url(ref)}} for ref in refs
        ]
    return body


def generate_images(
    prompt: str,
    *,
    count: int = 1,
    width: int = 1024,
    height: int = 1024,
    references: list[Image.Image] | None = None,
    model: str | None = None,
    request_timeout: float | None = None,
) -> list[Image.Image]:
    """``count`` images from OpenRouter, batching up to MAX_BATCH per request.

    Falls back to one-at-a-time if the chosen model rejects n>1, which some
    image models do.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("A non-empty prompt is required.")
    count = max(1, int(count))
    limit = request_timeout or timeout()

    images: list[Image.Image] = []
    batched = True
    while len(images) < count:
        want = min(MAX_BATCH, count - len(images)) if batched else 1
        body = _payload(
            prompt,
            count=want,
            width=width,
            height=height,
            references=references,
            model=model,
        )
        try:
            got = _decode(_post(body, limit))
        except RuntimeError as exc:
            # Retry singly once: a model that only supports n=1 reports it as a
            # 400/422 on the n field, which is not worth failing the whole burst.
            if batched and want > 1 and _looks_like_batch_refusal(exc):
                print(f"[openrouter] batch refused ({exc}); falling back to n=1", flush=True)
                batched = False
                continue
            raise
        if not got:
            break
        images.extend(got[: count - len(images)])
    return images


def _looks_like_batch_refusal(exc: Exception) -> bool:
    text = str(exc).lower()
    return "\"n\"" in text or " n " in text or "batch" in text or "multiple images" in text


def list_models(request_timeout: float = 30.0) -> list[str]:
    """Slugs that can output images — handy when a configured slug 404s."""
    import requests

    key = api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = requests.get(
        f"{base_url()}/images/models", headers=headers, timeout=request_timeout
    )
    response.raise_for_status()
    payload = response.json()
    entries = payload.get("data") or payload.get("models") or []
    slugs = []
    for entry in entries:
        slug = entry.get("id") or entry.get("slug") if isinstance(entry, dict) else entry
        if slug:
            slugs.append(str(slug))
    return slugs


def health() -> dict[str, Any]:
    """Cheap reachability check that does not spend credits."""
    status: dict[str, Any] = {
        "provider": "openrouter",
        "base_url": base_url(),
        "key_present": bool(api_key()),
        "image_model": image_model(),
        "edit_model": edit_model(),
        "spend": spend(),
    }
    if not api_key():
        status["ready"] = False
        status["error"] = "OPENROUTER_API_KEY is not set"
        return status
    try:
        models = list_models(request_timeout=15.0)
        status["ready"] = True
        status["models_available"] = len(models)
        wanted = image_model()
        if models and wanted not in models:
            status["warning"] = (
                f"{wanted} is not in the image-model list; generation may 404."
            )
    except Exception as exc:
        status["ready"] = False
        status["error"] = str(exc)[:300]
    return status
