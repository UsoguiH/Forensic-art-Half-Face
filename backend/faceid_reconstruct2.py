
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

Renderer = Callable[[list[Image.Image], str, int, int, int], "tuple[Image.Image, dict[str, Any]]"]
Embedder = Callable[[Image.Image], "np.ndarray | None"]

# Two real cross-pose photos of the same person land around 0.55-0.70; a
# candidate scoring far above that against the *exact source embedding* has
# copied the numbers rather than matched the person (see module docstring).
SUSPICIOUS_SIMILARITY = 0.74

DEFAULT_SEEDS = (7, 21, 77, 202)


def _unit(v: "np.ndarray | None") -> "np.ndarray | None":
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def extract_identity(
    profile: Image.Image, embed: Embedder | None
) -> "tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]":
    """(fused identity, profile embedding, mirror embedding). Fused = profile ⊕
    mirror, which cancels part of the pose bias and measured better on every
    subject; the separate vectors feed the both-sides ranking below."""
    if embed is None:
        return None, None, None
    v_p = _unit(embed(profile))
    if v_p is None:
        return None, None, None
    v_m = _unit(embed(ImageOps.mirror(profile)))
    fused = _unit((v_p + v_m) / 2.0) if v_m is not None else v_p
    return fused, v_p, v_m


def descriptive_prompt(who: str = "", traits: str = "") -> str:
    """Portrait commission. Won the prompt A/B at 0.665 (vs 0.636)."""
    person, their, they_are = {
        "a man": ("man", "his", "He is"),
        "a woman": ("woman", "her", "She is"),
    }.get((who or "").strip().lower(), ("person", "their", "They are"))
    traits = f" {traits.strip()}" if (traits or "").strip() else ""
    return (
        f"The two photos show the same {person} — image 1 is {their} head from one side, "
        f"image 2 is {their} head from the other side. {they_are} now facing the camera "
        "directly, both eyes looking straight into the lens, with a neutral expression. "
        "Studio setting with a plain light-gray backdrop. Realistic documentary photography "
        "with true-to-life features and natural skin texture. Soft, even, frontal lighting "
        "with no facial shadows. Camera at eye level, head-and-shoulders framing, sharp "
        f"focus across the entire face. {their.capitalize()} face, hair and clothing are "
        "identical to the source photos — the same eyebrows, the same eyes, the same nose, "
        "the same lips, the same facial hair, the same jawline, the same hairline and "
        f"haircut, the same clothing.{traits}"
    )


def idportrait_prompt(who: str = "", traits: str = "") -> str:
    """Scene-to-studio commission for far/surveillance shots.

    The scene-preserving prompt is actively wrong for these: it keeps the
    corridor.  This one commissions the deliverable itself — a studio ID
    portrait — while pinning everything identity-bearing to the source photos,
    clothing and headwear included (the collar and ghutra carry identity too).
    It also has to fight two poses at once: yaw (turn to camera) and the
    lowered gaze common in candid frames (head level, eyes into the lens).
    """
    person, their, they_are = {
        "a man": ("man", "his", "He is"),
        "a woman": ("woman", "her", "She is"),
    }.get((who or "").strip().lower(), ("person", "their", "They are"))
    traits = f" {traits.strip()}" if (traits or "").strip() else ""
    return (
        f"Both photos show the same {person}, photographed candidly from the side. "
        f"Now {they_are.lower()} posing for an official identification portrait: facing the "
        "camera directly, head level and upright, both eyes open and looking straight into "
        "the lens, neutral expression. Head-and-shoulders framing against a plain light-gray "
        "studio backdrop with soft, even, frontal lighting and no facial shadows. "
        f"{their.capitalize()} face is exactly the face in the source photos — the same "
        "eyebrows, the same eyes, the same nose, the same lips, the same facial hair, the "
        "same jawline, the same skin tone and texture, the same hairline and haircut — and "
        f"{they_are.lower()} wearing exactly the same clothing and any headwear from the "
        "source photos. Sharp focus across the entire face, true to life, no beautification, "
        f"no makeup, no smoothing.{traits}"
    )


def idportrait2_prompt(who: str = "", traits: str = "") -> str:
    """idportrait with the feature anchors spelled out one by one.

    Round-3 review found the failures are always *specific*: a face rendered
    slightly wider than the person's, a distinctive nose flattened to average,
    an age drifted younger. Generic "same face" phrasing lets the model fall
    back to its priors on exactly those axes, so this variant names each axis
    and pins it to the source. A/B'd against idportrait, never edited in place.
    """
    person, their, they_are = {
        "a man": ("man", "his", "He is"),
        "a woman": ("woman", "her", "She is"),
    }.get((who or "").strip().lower(), ("person", "their", "They are"))
    traits = f" {traits.strip()}" if (traits or "").strip() else ""
    return (
        f"Both photos show the same {person}, photographed candidly from the side. "
        f"Now {they_are.lower()} posing for an official identification portrait: facing the "
        "camera directly, head level and upright, both eyes open and looking straight into "
        "the lens, neutral expression. Head-and-shoulders framing against a plain light-gray "
        "studio backdrop with soft, even, frontal lighting. This must be the real person "
        "exactly as they are in the source photos, not an idealized version: the same exact "
        "head shape and face width, the same cheek fullness and jawline, the same distinctive "
        "nose with the same bridge and tip shape seen in the side view, the same eyebrow "
        "thickness, the same eye shape and eyelids, the same lips, the same skin tone and "
        "texture with any moles or marks kept, the same apparent age and weight, the same "
        "facial hair in the same pattern and density, the same hairline, hair length, hair "
        f"texture and haircut. {they_are} wearing exactly the same clothing and any headwear "
        "from the source photos. Photorealistic, sharp focus across the entire face, no "
        f"beautification, no makeup, no smoothing, no de-aging.{traits}"
    )


def preserving_prompt(who: str = "", traits: str = "") -> str:
    """Scene-preserving instruction. Keeps the original background, clothing and
    light — for photos whose identity lives in their context (distinctive
    beards, hairlines, lighting), where the studio re-stage drifts."""
    person = {
        "a man": "man", "a woman": "woman",
    }.get((who or "").strip().lower(), "person")
    traits = f" {traits.strip()}" if (traits or "").strip() else ""
    return (
        f"The two images show the same {person}'s head from its two sides. Turn the head to "
        f"face the camera directly, keeping everything else exactly as photographed: the same "
        "background, the same clothing, the same lighting and exposure, the same camera and "
        "framing, the head at the same scale and position. The frontal face must be the real "
        f"face of this same {person} — the same eyebrows, eyes, nose, lips, facial hair, "
        "jawline, hairline, haircut and skin texture, with both eyes looking into the lens, "
        f"neutral expression, photorealistic, no beautification, no smoothing.{traits}"
    )


def reconstruct(
    profile: Image.Image,
    *,
    renderer: Renderer,
    embed: Embedder | None = None,
    seeds: "tuple[int, ...]" = DEFAULT_SEEDS,
    prompt_styles: "tuple[str, ...]" = ("descriptive", "preserving"),
    who: str = "",
    traits: str = "",
    extra_prompt: str = "",
    out_size: "tuple[int, int] | None" = None,
    references: "list[Image.Image] | None" = None,
) -> dict[str, Any]:
    """Best-effort identical-identity frontal from one profile photo.

    Renders len(seeds) x len(prompt_styles) candidates and returns the one
    closest to the fused identity. Without ``embed`` (mock mode), renders one.

    ``out_size`` decouples the render from the input: a far shot arrives at
    scene size, but the deliverable is a portrait — rendering AT the scene size
    was spending the pixels on the background. Default keeps the old behaviour.
    """
    profile = profile.convert("RGB")
    width, height = out_size or profile.size
    started = time.time()

    fused, v_profile, v_mirror = extract_identity(profile, embed)
    if fused is None:
        seeds = seeds[:1]
        prompt_styles = prompt_styles[:1]

    prompts = {
        "descriptive": descriptive_prompt(who, traits),
        "preserving": preserving_prompt(who, traits),
        "idportrait": idportrait_prompt(who, traits),
        "idportrait2": idportrait2_prompt(who, traits),
    }
    extra = (extra_prompt or "").strip()
    # Default refs: the photo and its mirror ("both sides of the head").  A
    # caller can hand in richer evidence instead — e.g. a tight face crop plus
    # the head-and-shoulders crop, so more of the reference pixels are face.
    if references is None:
        references = [profile, ImageOps.mirror(profile)]

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    meta: dict[str, Any] = {}
    for style in prompt_styles:
        prompt = prompts.get(style)
        if not prompt:
            continue
        if extra:
            prompt = f"{prompt} {extra}"
        for seed in seeds:
            image, meta = renderer(references, prompt, width, height, seed)
            image = image.convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            v = _unit(embed(image)) if embed is not None else None
            score = None if (v is None or fused is None) else float(np.dot(v, fused))
            # Selection uses min(sim to profile, sim to mirror): a candidate must
            # resemble BOTH sides of the head. Measured over the 4 ground-truth
            # pairs this picks the pool's true best more often than the fused
            # score alone and lifts the worst case from 0.445 to 0.513.
            if v is not None and v_profile is not None and v_mirror is not None:
                rank_score = min(float(np.dot(v, v_profile)), float(np.dot(v, v_mirror)))
            else:
                rank_score = score
            entry = {
                "stage": style, "seed": seed, "sim_profile": score,
                "rank_score": rank_score, "image": image,
                "suspicious": bool(score is not None and score > SUSPICIOUS_SIMILARITY),
            }
            candidates.append(entry)
            # A suspicious score outranks honest ones only against other
            # suspicious ones (relevant when extra engines join the pool).
            rank = (not entry["suspicious"], rank_score if rank_score is not None else -1.0)
            if best is None or rank > (
                not best["suspicious"],
                best["rank_score"] if best["rank_score"] is not None else -1.0,
            ):
                best = entry

    assert best is not None
    return {
        "image": best["image"],
        "sim_profile": best["sim_profile"],
        "stage": best["stage"],
        "seed": best["seed"],
        "candidates": [
            {k: v for k, v in c.items() if k != "image"} for c in candidates
        ],
        "candidate_images": [c["image"] for c in candidates],
        "prompt": prompts.get(prompt_styles[0], ""),
        "render": meta,
        "seconds": round(time.time() - started, 2),
    }
