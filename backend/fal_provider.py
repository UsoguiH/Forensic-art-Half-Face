"""fal.ai FLUX.2 back end.

A fourth generation route alongside ``openrouter`` / ``lab`` / ``local`` (see
``providers.resolve_provider``).  fal.ai hosts the Black Forest Labs FLUX.2
family, and unlike OpenRouter it exposes a real *editing* endpoint that takes
reference images and honours an exact pixel ``image_size`` — both of which the
pixel-exact half-face completion in ``half_face.py`` depends on.

Endpoint slugs (verified against the fal catalogue):

  ``fal-ai/flux-2/edit``            FLUX.2 [dev] image-to-image editing   (default)
  ``fal-ai/flux-2/klein/9b/edit``   FLUX.2 [klein] 9B editing             (fallback)
  ``fal-ai/flux-2``                 FLUX.2 [dev] text-to-image
  ``fal-ai/flux-2/klein/9b``        FLUX.2 [klein] 9B text-to-image

The klein endpoints are 4-step distilled and take no ``guidance_scale``, so the
payload is trimmed per endpoint rather than sent blind.

Auth: ``FAL_KEY`` (fal's own convention) or ``FAL_API_KEY``.  As a convenience
for the offline lab image, a key may also live in a file pointed at by
``FACELAB_FAL_KEY_FILE``, defaulting to ``data/fal_key.txt``.
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

SYNC_URL = "https://fal.run"
QUEUE_URL = "https://queue.fal.run"

# FLUX.2 [dev] first — it is the model the forensic flow is tuned for. klein 9B
# is the documented fallback when dev is unavailable on the account.
EDIT_MODELS = ("fal-ai/flux-2/edit", "fal-ai/flux-2/klein/9b/edit")
IMAGE_MODELS = ("fal-ai/flux-2", "fal-ai/flux-2/klein/9b")

# fal rejects any side outside this range; FLUX likes multiples of 32.
MIN_SIDE, MAX_SIDE, SIDE_STEP = 512, 2048, 32
MAX_REFS = 4

_COST_LOCK = threading.Lock()
_SPEND = {"requests": 0, "images": 0}
_KEY_CACHE: dict[str, str] = {}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _key_file() -> Path:
    override = _env("FACELAB_FAL_KEY_FILE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "fal_key.txt"


def api_key() -> str:
    """The fal credential, from the environment or the on-disk key file."""
    for name in ("FAL_KEY", "FAL_API_KEY", "FALAI_API_KEY"):
        value = _env(name)
        if value:
            return value
    if "file" not in _KEY_CACHE:
        try:
            _KEY_CACHE["file"] = _key_file().read_text(encoding="utf-8").strip()
        except OSError:
            _KEY_CACHE["file"] = ""
    return _KEY_CACHE["file"]


def available() -> bool:
    return bool(api_key())


def timeout() -> float:
    try:
        return float(_env("FAL_TIMEOUT", "600"))
    except ValueError:
        return 600.0


def edit_models() -> tuple[str, ...]:
    """Edit endpoints to try in order; FAL_EDIT_MODEL prepends an override."""
    override = _env("FAL_EDIT_MODEL")
    return _with_override(override, EDIT_MODELS)


def image_models() -> tuple[str, ...]:
    override = _env("FAL_IMAGE_MODEL")
    return _with_override(override, IMAGE_MODELS)


def _with_override(override: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if not override:
        return defaults
    slug = resolve_slug(override, defaults)
    return (slug,) + tuple(m for m in defaults if m != slug)


def resolve_slug(model: str | None, defaults: tuple[str, ...]) -> str:
    """Accept a full fal slug, or the UI's short 'dev' / 'klein' names."""
    model = (model or "").strip()
    if not model:
        return defaults[0]
    short = model.lower()
    if short in ("dev", "flux2", "flux-2", "flux.2"):
        return defaults[0]
    if short in ("klein", "klein9b", "klein-9b", "flux2-klein", "flux.2-klein"):
        return defaults[-1]
    return model


def is_klein(slug: str) -> bool:
    return "klein" in slug.lower()


def spend() -> dict[str, Any]:
    with _COST_LOCK:
        return {"requests": _SPEND["requests"], "images": _SPEND["images"]}


def _record(images: int) -> None:
    with _COST_LOCK:
        _SPEND["requests"] += 1
        _SPEND["images"] += images


def align_side(value: int) -> int:
    """Clamp one dimension into fal's accepted range, on the FLUX grid."""
    value = int(round(max(1, int(value)) / SIDE_STEP) * SIDE_STEP)
    return max(MIN_SIDE, min(MAX_SIDE, value))


def fit_size(width: int, height: int) -> tuple[int, int]:
    """Nearest accepted (width, height) that keeps the aspect ratio.

    Both sides must land in [512, 2048].  A portrait half-face canvas is very
    tall and narrow, so the two clamps can fight; the long edge wins, because
    losing aspect on the short edge would misplace the seam.
    """
    width, height = max(1, int(width)), max(1, int(height))
    scale = 1.0
    longest, shortest = max(width, height), min(width, height)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
    elif shortest < MIN_SIDE:
        scale = MIN_SIDE / shortest
    return align_side(width * scale), align_side(height * scale)


def to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _headers() -> dict[str, str]:
    key = api_key()
    if not key:
        raise RuntimeError(
            "FAL_KEY is not set. Export it (FAL_KEY=<id>:<secret>) or write the key "
            f"to {_key_file()} to use the fal.ai FLUX.2 route."
        )
    return {"Authorization": f"Key {key}", "Content-Type": "application/json"}


def _explain(status: int, detail: str, slug: str) -> str:
    hint = {
        401: "FAL_KEY is missing or invalid.",
        403: "This fal account does not have access to that model.",
        404: f"Unknown fal endpoint '{slug}'.",
        422: "fal rejected the request body — check image_size and image_urls.",
        429: "fal rate limit — retry more slowly.",
        402: "fal balance exhausted — top up the account.",
    }.get(status, "")
    # fal reports an empty wallet as 403 "User is locked. Reason: Exhausted
    # balance." — the generic 403 hint would send someone hunting model
    # permissions when the fix is a top-up.
    if status == 403 and ("balance" in detail.lower() or "locked" in detail.lower()):
        hint = "The fal.ai account balance is exhausted — top up at fal.ai/dashboard/billing."
    return f"fal.ai error {status} on {slug}: {detail}{(' ' + hint) if hint else ''}"


def _detail(response: Any) -> str:
    try:
        body = response.json()
    except Exception:
        return response.text[:300]
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("error") or body)[:400]
    return str(body)[:400]


def _post(slug: str, payload: dict[str, Any], request_timeout: float) -> dict[str, Any]:
    """Synchronous call, falling back to the queue API on a 202/timeout."""
    import requests

    try:
        response = requests.post(
            f"{SYNC_URL}/{slug}", json=payload, headers=_headers(), timeout=request_timeout
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"fal.ai unreachable at {SYNC_URL} — this host needs internet access. "
            "On the offline lab network set FACELAB_PROVIDER=lab instead."
        ) from exc

    if response.status_code == 200:
        return response.json()
    if response.status_code in (202, 408, 504):
        return _queue(slug, payload, request_timeout)
    raise RuntimeError(_explain(response.status_code, _detail(response), slug))


def _queue(slug: str, payload: dict[str, Any], request_timeout: float) -> dict[str, Any]:
    """Submit + poll, for renders longer than one HTTP request can hold open."""
    import requests

    headers = _headers()
    submit = requests.post(f"{QUEUE_URL}/{slug}", json=payload, headers=headers, timeout=60)
    if submit.status_code not in (200, 201, 202):
        raise RuntimeError(_explain(submit.status_code, _detail(submit), slug))
    body = submit.json()
    status_url = body.get("status_url")
    response_url = body.get("response_url")
    request_id = body.get("request_id")
    if not status_url and request_id:
        status_url = f"{QUEUE_URL}/{slug}/requests/{request_id}/status"
        response_url = f"{QUEUE_URL}/{slug}/requests/{request_id}"
    if not status_url:
        raise RuntimeError(f"fal.ai queue accepted the job but returned no status URL: {body}")

    deadline = time.monotonic() + request_timeout
    while time.monotonic() < deadline:
        time.sleep(1.5)
        poll = requests.get(status_url, headers=headers, timeout=60)
        if poll.status_code != 200:
            raise RuntimeError(_explain(poll.status_code, _detail(poll), slug))
        state = str(poll.json().get("status", "")).upper()
        if state == "COMPLETED":
            final = requests.get(response_url, headers=headers, timeout=request_timeout)
            if final.status_code != 200:
                raise RuntimeError(_explain(final.status_code, _detail(final), slug))
            return final.json()
        if state in ("FAILED", "CANCELLED", "ERROR"):
            raise RuntimeError(f"fal.ai job {state.lower()} on {slug}: {_detail(poll)}")
    raise RuntimeError(f"fal.ai job on {slug} did not finish within {request_timeout:.0f}s.")


def _decode(payload: dict[str, Any], request_timeout: float) -> list[Image.Image]:
    import requests

    entries = payload.get("images") or payload.get("data") or []
    images: list[Image.Image] = []
    for entry in entries:
        url = entry.get("url") if isinstance(entry, dict) else entry
        if not url:
            continue
        try:
            if str(url).startswith("data:"):
                raw = base64.b64decode(str(url).split(",", 1)[1])
            else:
                raw = requests.get(url, timeout=request_timeout).content
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:  # one bad entry should not sink the batch
            print(f"[fal] skipping undecodable image: {exc}", flush=True)
    _record(len(images))
    if not images:
        raise RuntimeError(
            "fal.ai returned no image. The safety checker may have blocked it; "
            "try rewording the prompt."
        )
    return images


def _body(
    slug: str,
    prompt: str,
    *,
    count: int,
    width: int,
    height: int,
    references: list[Image.Image] | None,
    seed: int | None,
    steps: int | None,
    guidance: float | None,
) -> dict[str, Any]:
    klein = is_klein(slug)
    body: dict[str, Any] = {
        "prompt": prompt,
        "num_images": max(1, int(count)),
        "image_size": {"width": int(width), "height": int(height)},
        "output_format": "png",
        "sync_mode": True,
        "enable_safety_checker": _env("FAL_SAFETY_CHECKER", "0") == "1",
    }
    if seed is not None:
        body["seed"] = int(seed) % (2**31 - 1)
    if steps:
        # klein 9B is distilled and caps at 8 steps; sending dev's 28 breaks the
        # dev -> klein fallback with a 422.
        body["num_inference_steps"] = max(1, min(8 if klein else 50, int(steps)))
    elif klein:
        body["num_inference_steps"] = 4
    if guidance is not None and not klein:
        # klein 9B is distilled and exposes no guidance_scale.
        body["guidance_scale"] = float(guidance)
    refs = list(references or [])
    if refs:
        if len(refs) > MAX_REFS:
            print(f"[fal] sending {MAX_REFS} of {len(refs)} references (endpoint cap)", flush=True)
            refs = refs[:MAX_REFS]
        body["image_urls"] = [to_data_url(ref) for ref in refs]
    return body


def _try_models(
    slugs: tuple[str, ...],
    build: Any,
    request_timeout: float,
) -> tuple[dict[str, Any], str]:
    """Walk the fallback chain, moving on only for availability-style failures."""
    last: Exception | None = None
    for index, slug in enumerate(slugs):
        try:
            return _post(slug, build(slug), request_timeout), slug
        except RuntimeError as exc:
            last = exc
            text = str(exc)
            if "balance" in text.lower() or "user is locked" in text.lower():
                # An empty wallet locks the whole account; every fallback would
                # fail the same way, and the first message is the honest one.
                raise
            transient = any(code in text for code in (" 404", " 403", " 402", " 503", " 500"))
            if index + 1 < len(slugs) and transient:
                print(f"[fal] {slug} unavailable ({text[:120]}); trying {slugs[index + 1]}", flush=True)
                continue
            raise
    raise last or RuntimeError("No fal.ai endpoint was reachable.")


def edit_images(
    prompt: str,
    references: list[Image.Image],
    *,
    width: int,
    height: int,
    count: int = 1,
    seed: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    model: str | None = None,
    request_timeout: float | None = None,
) -> tuple[list[Image.Image], str]:
    """Reference-conditioned edit. Returns (images, endpoint slug actually used)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("A non-empty prompt is required.")
    if not references:
        raise ValueError("At least one reference image is required for an edit.")
    limit = request_timeout or timeout()
    slugs = _with_override(resolve_slug(model, edit_models()) if model else "", edit_models())

    def build(slug: str) -> dict[str, Any]:
        return _body(
            slug, prompt, count=count, width=width, height=height,
            references=references, seed=seed, steps=steps, guidance=guidance,
        )

    payload, slug = _try_models(slugs, build, limit)
    return _decode(payload, limit), slug


def generate_images(
    prompt: str,
    *,
    count: int = 1,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    model: str | None = None,
    request_timeout: float | None = None,
) -> tuple[list[Image.Image], str]:
    """Text-to-image. Returns (images, endpoint slug actually used)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("A non-empty prompt is required.")
    limit = request_timeout or timeout()
    slugs = _with_override(resolve_slug(model, image_models()) if model else "", image_models())

    def build(slug: str) -> dict[str, Any]:
        return _body(
            slug, prompt, count=count, width=width, height=height,
            references=None, seed=seed, steps=steps, guidance=guidance,
        )

    payload, slug = _try_models(slugs, build, limit)
    return _decode(payload, limit), slug


def health() -> dict[str, Any]:
    """Reachability check that does not spend credits."""
    status: dict[str, Any] = {
        "provider": "fal",
        "key_present": available(),
        "edit_model": edit_models()[0],
        "edit_fallback": edit_models()[-1],
        "image_model": image_models()[0],
        "spend": spend(),
    }
    if not available():
        status["ready"] = False
        status["error"] = f"FAL_KEY is not set (looked in the environment and {_key_file()})"
        return status
    import requests

    try:
        # The OpenAPI document is public and free; a 200 proves DNS + TLS + slug.
        response = requests.get(
            "https://fal.ai/api/openapi/queue/openapi.json",
            params={"endpoint_id": edit_models()[0]},
            timeout=20,
        )
        status["ready"] = response.status_code == 200
        if response.status_code != 200:
            status["error"] = f"fal returned {response.status_code} for the edit endpoint schema"
    except Exception as exc:
        status["ready"] = False
        status["error"] = str(exc)[:300]
    return status
