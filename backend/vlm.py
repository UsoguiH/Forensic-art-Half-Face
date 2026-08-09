"""Vision-language chat client — the auto-description engine's transport.

Speaks the plain OpenAI ``/chat/completions`` schema so ONE base-url env var
moves it between hosts with no code change:

  hosted   FACELAB_VLM_BASE_URL=https://openrouter.ai/api/v1   (default)
  lab      FACELAB_VLM_BASE_URL=http://<lab>:8001/v1           (vLLM/SGLang)

The VLM provider is deliberately independent of ``FACELAB_PROVIDER`` — fal can
draw the pixels while OpenRouter reads the face; the existing single switch
cannot express that combination.

Auth: FACELAB_VLM_KEY, then OPENROUTER_API_KEY, then data/openrouter_key.txt
(same file-fallback convention as fal_provider's data/fal_key.txt).
"""
from __future__ import annotations

import io
import base64
import os
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-vl-32b-instruct"
MAX_EDGE = 1024  # px; description needs detail, not megapixels

_SPEND_LOCK = threading.Lock()
_SPEND = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
_KEY_CACHE: dict[str, str] = {}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _key_file() -> Path:
    override = _env("FACELAB_VLM_KEY_FILE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "openrouter_key.txt"


def api_key() -> str:
    for name in ("FACELAB_VLM_KEY", "OPENROUTER_API_KEY"):
        value = _env(name)
        if value:
            return value
    if "file" not in _KEY_CACHE:
        try:
            _KEY_CACHE["file"] = _key_file().read_text(encoding="utf-8").strip()
        except OSError:
            _KEY_CACHE["file"] = ""
    return _KEY_CACHE["file"]


def base_url() -> str:
    return (_env("FACELAB_VLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def model() -> str:
    return _env("FACELAB_VLM_MODEL") or DEFAULT_MODEL


def timeout() -> float:
    try:
        return float(_env("FACELAB_VLM_TIMEOUT", "60"))
    except ValueError:
        return 60.0


def available() -> bool:
    return bool(api_key())


def spend() -> dict[str, Any]:
    with _SPEND_LOCK:
        return dict(_SPEND)


def _record(usage: dict[str, Any]) -> None:
    with _SPEND_LOCK:
        _SPEND["requests"] += 1
        _SPEND["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        _SPEND["completion_tokens"] += int(usage.get("completion_tokens") or 0)


def _data_url(image: Image.Image) -> str:
    img = image.convert("RGB")
    if max(img.size) > MAX_EDGE:
        s = MAX_EDGE / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)),
                         Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _explain(status: int, detail: str) -> str:
    hint = {
        401: "The VLM API key is missing or invalid.",
        402: "The VLM account has no credit.",
        404: f"The VLM endpoint or model '{model()}' was not found.",
        429: "VLM rate limit — retry more slowly.",
    }.get(status, "")
    return f"VLM error {status} at {base_url()}: {detail}{(' ' + hint) if hint else ''}"


def chat_vision(
    prompt: str,
    images: list[Image.Image],
    *,
    model_override: str | None = None,
    max_tokens: int = 400,
    temperature: float = 0.2,
    request_timeout: float | None = None,
) -> str:
    """One user turn of text + image(s) → the assistant's text reply."""
    import requests

    key = api_key()
    if not key:
        raise RuntimeError(
            "No VLM key. Set FACELAB_VLM_KEY / OPENROUTER_API_KEY or write the "
            f"key to {_key_file()}."
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _data_url(img)}})
    body = {
        "model": model_override or model(),
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    site = _env("OPENROUTER_SITE_URL")
    title = _env("OPENROUTER_APP_TITLE")
    if site:
        headers["HTTP-Referer"] = site
    if title:
        headers["X-Title"] = title

    limit = request_timeout or timeout()
    t0 = time.time()
    try:
        response = requests.post(
            f"{base_url()}/chat/completions", json=body, headers=headers, timeout=limit
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"VLM unreachable at {base_url()}: {exc}") from exc
    if response.status_code != 200:
        try:
            detail = str(response.json().get("error") or response.json())[:300]
        except Exception:
            detail = response.text[:300]
        raise RuntimeError(_explain(response.status_code, detail))

    payload = response.json()
    _record(payload.get("usage") or {})
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"VLM returned no message: {str(payload)[:300]}") from exc
    if isinstance(text, list):  # some hosts return content parts
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    print(f"[vlm] {body['model']} answered in {time.time() - t0:.1f}s", flush=True)
    return (text or "").strip()


def health() -> dict[str, Any]:
    """Reachability check that spends nothing."""
    status: dict[str, Any] = {
        "provider": "vlm",
        "base_url": base_url(),
        "model": model(),
        "key_present": available(),
        "spend": spend(),
    }
    if not available():
        status["ready"] = False
        status["error"] = (
            f"no key (looked at FACELAB_VLM_KEY, OPENROUTER_API_KEY, {_key_file()})"
        )
        return status
    import requests

    try:
        response = requests.get(
            f"{base_url()}/models",
            headers={"Authorization": f"Bearer {api_key()}"},
            timeout=20,
        )
        status["ready"] = response.status_code == 200
        if response.status_code != 200:
            status["error"] = f"model list returned {response.status_code}"
    except Exception as exc:
        status["ready"] = False
        status["error"] = str(exc)[:300]
    return status
