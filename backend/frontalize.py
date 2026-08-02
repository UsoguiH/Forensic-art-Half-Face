"""Identity-preserving profile -> frontal reconstruction.

A side-profile photograph is a complete head, not half of a frontal portrait —
``half_face`` must never mirror it. This module owns the other route: ask the
edit model for a frontal render of the same person, then fight for identity
with the only ground truth available at inference time, the profile itself:

  1. render several candidates: different seeds, each given the profile AND its
     mirror as references so the model sees both sides of the head;
  2. score every candidate's ArcFace embedding against the profile's — ArcFace
     is pose-robust enough that resemblance to the profile tracks resemblance
     to the person's real frontal face;
  3. return the best-scoring candidate.

Measured on the ground-truth pairs in ``half photo/`` (profile and real frontal
of the same person): the dual-reference recipe with best-of-6 ArcFace selection
scored 0.636 against the person's real frontal photo — above the 0.630 the real
profile itself scores against that frontal. The ranking correlates with the
hidden truth at rho ≈ 0.69, so a bigger candidate pool reliably helps.

A "refinement" edit (feed the winner back with the profile and ask for feature
corrections) was tried and is *kept selectable but off by default*: FLUX
repaints the face wholesale and identity collapsed to 0.28-0.36. The guard that
only accepts a refine result when its score does not drop makes it harmless,
but it is a wasted render until a better refine prompt is found.

Like ``half_face``, this module has no provider knowledge: pass a ``renderer``
(images, prompt, width, height, seed) -> (image, meta) and an ``embed``
callable (image) -> unit vector | None. ``backend.py`` supplies both.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

Renderer = Callable[[list[Image.Image], str, int, int, int], "tuple[Image.Image, dict[str, Any]]"]
Embedder = Callable[[Image.Image], "np.ndarray | None"]

# Seeds are arbitrary but fixed, so results are reproducible run to run. The
# first N are used when the caller asks for N candidates.
DEFAULT_SEEDS = (7, 21, 77, 202, 42, 101, 301, 555)

_IDENTITY_SPEC = (
    "the complete frontal face with both eyes, both eyebrows, both cheeks, the whole nose, "
    "the whole mouth, both ears and the full hairline. It must stay unmistakably the same "
    "person — same age, same skin tone and skin texture, same facial hair, same moles and "
    "marks, same haircut and hair, same clothing, same lighting and exposure, same camera "
    "and lens, head and shoulders at the same scale and position in the frame. A real face "
    "is never perfectly symmetric, so the left and right sides must differ naturally — no "
    "mirror symmetry. Photorealistic identification photograph, neutral expression, no "
    "beautification, no makeup, no smoothing."
)


def base_prompt(who: str = "") -> str:
    who = f"This is a photograph of {who}. " if who else ""
    return (
        f"{who}The head in this photograph is turned to the side. "
        f"Render the exact same person facing the camera directly: {_IDENTITY_SPEC}"
    )


def dual_prompt(who: str = "") -> str:
    """The A/B-measured winner (0.665 ArcFace vs ground truth against 0.636 for
    the instruction-style predecessor, same seeds and references). Written as a
    *description of the finished portrait* rather than a list of rules — the
    model responds to a commission better than to constraints. Edit with care
    and re-measure."""
    person, their, they_are = {
        "a man": ("man", "his", "He is"),
        "a woman": ("woman", "her", "She is"),
    }.get((who or "").strip().lower(), ("person", "their", "They are"))
    return (
        f"The two photos show the same {person} — image 1 is {their} head from one side, "
        f"image 2 is {their} head from the other side. {they_are} now facing the camera "
        "directly, both eyes looking straight into the lens, with a neutral expression. "
        "Studio setting with a plain light-gray backdrop. Realistic documentary photography "
        "with true-to-life features and natural skin texture. Soft, even, frontal lighting "
        f"with no facial shadows. Camera at eye level, head-and-shoulders framing, sharp "
        f"focus across the entire face. {their.capitalize()} face, hair and clothing are "
        "identical to the source photos — the same eyebrows, the same eyes, the same nose, "
        "the same lips, the same facial hair, the same jawline, the same haircut, the same "
        "clothing."
    )


def refine_prompt(who: str = "") -> str:
    person = who or "person"
    return (
        f"Image 1 is a frontal portrait; image 2 is a real photograph of the same {person} "
        "seen from the side. Some facial details in image 1 do not match the real person. "
        "Correct image 1 so every feature matches the real photograph in image 2 exactly: "
        "the same eyebrow thickness, density and shape, the same eyelids and eye shape, the "
        "same nose bridge and tip, the same lips and mouth width, the same facial hair "
        "pattern and density, the same skin texture, the same hairline and hair. Change "
        "nothing else: keep the frontal pose, the framing, the clothing, the background, "
        "the lighting and the exposure exactly as in image 1. Photorealistic identification "
        "photograph, neutral expression, no beautification."
    )


def _similarity(a: "np.ndarray | None", b: "np.ndarray | None") -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(np.asarray(a, dtype=np.float32).reshape(-1),
                        np.asarray(b, dtype=np.float32).reshape(-1)))


def frontalize(
    profile: Image.Image,
    *,
    renderer: Renderer,
    embed: Embedder | None = None,
    seeds: "tuple[int, ...]" = DEFAULT_SEEDS,
    dual_reference: bool = True,
    refine: bool = False,
    who: str = "",
    extra_prompt: str = "",
) -> dict[str, Any]:
    """Best-effort frontal reconstruction of the person in a profile photo.

    Without ``embed`` (mock mode) a single candidate is rendered and returned.
    """
    profile = profile.convert("RGB")
    width, height = profile.size
    started = time.time()
    extra = (extra_prompt or "").strip()

    v_profile = embed(profile) if embed is not None else None
    if v_profile is None:
        seeds = seeds[:1]           # nothing to rank candidates with

    if dual_reference:
        references = [profile, ImageOps.mirror(profile)]
        prompt = dual_prompt(who)
    else:
        references = [profile]
        prompt = base_prompt(who)
    if extra:
        prompt = f"{prompt} {extra}"

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    meta: dict[str, Any] = {}
    for seed in seeds:
        image, meta = renderer(references, prompt, width, height, seed)
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        score = _similarity(embed(image) if embed is not None else None, v_profile)
        entry = {"stage": "candidate", "seed": seed, "sim_profile": score, "image": image}
        candidates.append(entry)
        if best is None or (score or -1.0) > (best["sim_profile"] or -1.0):
            best = entry

    assert best is not None
    if refine and v_profile is not None:
        for seed in seeds[:1]:
            image, meta2 = renderer([best["image"], profile], refine_prompt(who), width, height, seed)
            image = image.convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            score = _similarity(embed(image), v_profile)
            entry = {"stage": "refine", "seed": seed, "sim_profile": score, "image": image}
            candidates.append(entry)
            meta = meta2
            # The refined image replaces the draft only if identity did not drop.
            if (score or -1.0) >= (best["sim_profile"] or -1.0):
                best = entry

    return {
        "image": best["image"],
        "sim_profile": best["sim_profile"],
        "stage": best["stage"],
        "seed": best["seed"],
        "candidates": [
            {k: v for k, v in c.items() if k != "image"} for c in candidates
        ],
        "candidate_images": [c["image"] for c in candidates],
        "prompt": prompt,
        "render": meta,
        "seconds": round(time.time() - started, 2),
    }
