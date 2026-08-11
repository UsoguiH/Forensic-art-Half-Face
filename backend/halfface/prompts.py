"""v7 prompt system: pose block → identity notes → anti-idealization block.

Ordering is deliberate and measured:

1. POSE first — the v5 "forensic" finding: putting the deliverable's pose
   before everything else produced frontality 0.93–0.99 across the board and
   won the man2 pool on both metrics.
2. IDENTITY NOTES — the new v7 signal: concrete observed facts about THIS
   person (describe.compose output).  Facts, never axis names — the
   axis-enumeration variant (idportrait2) measured worse than saying nothing.
3. ANTI-IDEALIZATION — short, explicit, targeted at the four failure modes the
   visual audit found on every subject: facial-hair densification, hairline
   restoration, face narrowing, beautification/smoothing.

Styles:
- ``forensic``      studio ID-portrait framing (pool workhorse)
- ``forensicdoc``   documentary/neutral-wall variant — same three blocks, a
  second style because style pools split wins per subject in every battery
- ``idportrait``    the unchanged v5 control prompt (regression insurance in
  benchmark pools; carries no description)

All styles share the preamble logic: single-photo evidence is presented as
"photo and its mirrored copy — the two sides of the head" (the v1-measured
framing), several genuinely different photos as the multiref triangulation
framing, and the grid layout announces the identity sheet.
"""
from __future__ import annotations


def _pronouns(who: str) -> "tuple[str, str, str]":
    return {
        "a man": ("man", "his", "He is"),
        "a woman": ("woman", "her", "She is"),
    }.get((who or "").strip().lower(), ("person", "their", "They are"))


def _preamble(person: str, their: str, *, n_photos: int, grid: bool) -> str:
    if grid:
        return (
            f"The first image is a reference sheet: several photographs of one {person}'s "
            f"head seen from the side, including a mirrored copy. The next image shows the "
            f"same {person} in a wider view. Study every view together — each reveals part "
            f"of {their} true face, and every feature the views agree on is real."
        )
    if n_photos > 1:
        return (
            f"Every photo shows the same {person}, photographed on different occasions and "
            f"from different angles, always from the side. Study all of them together: each "
            f"view reveals part of {their} true face, and every feature they agree on is real."
        )
    return (
        f"The photos show the same {person}'s head from its two sides — one of them is a "
        "mirrored copy of the other."
    )


def _pose(they_are: str) -> str:
    return (
        f"{they_are} now posing for an official identification portrait: facing the camera "
        "directly, head level and upright, both eyes open and looking straight into the "
        "lens, neutral expression."
    )


def _identity(description: str) -> str:
    d = (description or "").strip()
    if not d:
        return ""
    if not d.endswith("."):
        d += "."
    return (
        "Identity notes recorded from the source photos — every one of them must appear "
        f"in the portrait exactly as written: {d}"
    )


def _anti_idealization(their: str) -> str:
    return (
        "This is an ordinary, unretouched person, not a model: do not beautify or "
        "idealize, do not thicken, darken or tidy the facial hair beyond what the photos "
        f"show, do not lower or fill in {their} hairline, do not narrow or lengthen the "
        "face, and keep the natural asymmetries, skin texture and apparent age exactly as "
        "photographed. Pay special attention to the eyes: reproduce exactly the eye "
        "visible in the side photos — the same eyelid shape and heaviness, the same eye "
        "size and depth, the same eyebrow-to-eye distance, with no added dark circles and "
        "no exaggerated tiredness."
    )


def _single_face(person: str) -> str:
    return (
        f"The picture must show this {person} exactly once: a single frontal head only — "
        "no side views, no profile views anywhere in the frame, no extra faces, no "
        "multi-view sheet, no duplicated heads at the edges of the image."
    )


def build(
    style: str,
    *,
    who: str = "",
    traits: str = "",
    description: str = "",
    n_photos: int = 1,
    grid: bool = False,
) -> str:
    person, their, they_are = _pronouns(who)
    traits_txt = f" {traits.strip()}" if (traits or "").strip() else ""

    if style == "idportrait":  # unchanged v5 control — no description block
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
            f"no makeup, no smoothing.{traits_txt}"
        )

    setting = {
        "forensic": (
            "Head-and-shoulders framing against a plain light-gray studio backdrop with "
            "soft, even, frontal lighting and no facial shadows."
        ),
        "forensicdoc": (
            "Head-and-shoulders documentary photograph against a plain neutral wall in "
            "natural, even light — realistic, unstyled, true to life."
        ),
    }.get(style)
    if setting is None:
        raise ValueError(f"Unknown prompt style: {style!r}")

    parts = [
        _preamble(person, their, n_photos=n_photos, grid=grid),
        _pose(they_are),
        setting,
        _identity(description),
        _anti_idealization(their),
        _single_face(person),
        (
            f"{they_are} wearing exactly the same clothing and any headwear as in the "
            "source photos. Photorealistic, sharp focus across the entire "
            f"face.{traits_txt}"
        ),
    ]
    return " ".join(p for p in parts if p)


ENRICH_SENTENCE = (
    "The last image is a draft frontal portrait of the same person; keep everything that "
    "matches the side photos and correct any feature that differs."
)
