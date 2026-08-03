"""Evidence intake: photos → crops, embeddings, anchors, tier.

Everything downstream reasons about an ``EvidenceSet``:

- ``working``     head-and-shoulders crop of the primary photo (renderer ref #1)
- ``tights``      tight face crops, primary first (identity-dense refs)
- ``refs_std``    the measured-best reference layout: [working] + extra tights
                  (≤3), mirror(working) appended when there are spare slots
- ``refs_grid``   optional composite "identity sheet" layout: one grid image of
                  the face views + the working crop
- anchors         unit embeddings: primary tight, its mirror, one fused
                  quality-weighted vector over every photo AND its mirror
                  (v4-measured: picked the pool oracle on every multi-photo run)
- ``tier``        "evidence-poor" (face < 200 px min-side → CCTV handling:
                  enrichment chain) or "evidence-rich"

Quality weight per photo = det_score × min(1, face_px / 200): a confident,
big face counts fully; a tiny CCTV sliver still contributes but cannot drag
the anchor.  Embeddings always come from the raw crops — plain LANCZOS is the
only resampling anywhere (generative SR/restoration measurably hallucinates
identity and is banned).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

import face_crop  # backend dir is on sys.path (flat-module convention of this repo)

Detector = Callable[[Image.Image], list]
Embedder = Callable[[Image.Image], "np.ndarray | None"]

EVIDENCE_POOR_PX = 200
MAX_REFS = 4          # fal FLUX.2 edit cap; keep layouts within it
MAX_EXTRA_TIGHTS = 3


def unit(v: "np.ndarray | None") -> "np.ndarray | None":
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def cos(a: "np.ndarray | None", b: "np.ndarray | None") -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


@dataclass
class EvidencePhoto:
    image: Image.Image                  # full source frame
    tight: Image.Image                  # tight face crop
    face: Any                           # detector Face object (bbox/kps/sex/age)
    face_px: int                        # min side of the face box
    quality: float                      # det_score × size factor
    embedding: "np.ndarray | None"      # unit embedding of the tight crop
    mirror_embedding: "np.ndarray | None"


@dataclass
class EvidenceSet:
    photos: list[EvidencePhoto]         # primary first
    working: Image.Image                # hs crop of primary
    crop_meta: dict[str, Any]
    mirror: Image.Image                 # mirror(working)
    refs_std: list[Image.Image]
    refs_grid: list[Image.Image] | None
    v_primary: "np.ndarray | None"      # primary tight embedding
    v_mirror: "np.ndarray | None"       # mirrored primary tight embedding
    fused: "np.ndarray | None"          # quality-weighted anchor (photos+mirrors)
    tier: str
    face_px: int
    who: str = ""
    traits: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def multi(self) -> bool:
        return len(self.photos) > 1

    def per_photo_sims(self, v: "np.ndarray | None") -> list[float | None]:
        return [cos(v, p.embedding) for p in self.photos]


def _photo(image: Image.Image, detector: Detector, embed: Embedder | None) -> EvidencePhoto | None:
    face = face_crop.biggest_face(detector(image)) if detector else None
    if face is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in face.bbox[:4])
    face_px = int(round(min(x1 - x0, y1 - y0)))
    tight = face_crop.tight_face_crop(image, face.bbox[:4])
    v = unit(embed(tight)) if embed else None
    vm = unit(embed(ImageOps.mirror(tight))) if embed else None
    det = float(getattr(face, "det_score", 1.0) or 1.0)
    quality = det * min(1.0, face_px / float(EVIDENCE_POOR_PX))
    return EvidencePhoto(image, tight, face, face_px, quality, v, vm)


class _MockFace:
    """Stand-in detection for mock mode (no models loaded): a centred
    portrait-ish box so crops, prompts and the ledger can be exercised."""

    def __init__(self, width: int, height: int):
        self.bbox = (width * 0.30, height * 0.18, width * 0.70, height * 0.62)
        self.kps = None
        self.det_score = 1.0
        self.sex = ""
        self.age = None


def _mock_photo(image: Image.Image) -> EvidencePhoto:
    face = _MockFace(*image.size)
    tight = face_crop.tight_face_crop(image, face.bbox)
    x0, y0, x1, y1 = face.bbox
    face_px = int(round(min(x1 - x0, y1 - y0)))
    return EvidencePhoto(image, tight, face, face_px, 1.0, None, None)


def identity_sheet(tiles: list[Image.Image], cell: int = 512, pad: int = 16) -> Image.Image:
    """Composite the face views into one 'identity sheet' reference.

    A single grid asserts co-identity structurally — the model reads it as a
    character sheet of one person rather than unrelated context images — and
    spends only one of the 4 reference slots.  2 tiles → 1×2, otherwise 2×2.
    """
    tiles = tiles[:4]
    cols = 2
    rows = 1 if len(tiles) <= 2 else 2
    sheet = Image.new("RGB", (cols * cell + pad * (cols + 1), rows * cell + pad * (rows + 1)), "white")
    for i, tile in enumerate(tiles):
        t = ImageOps.contain(tile.convert("RGB"), (cell, cell), Image.Resampling.LANCZOS)
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad) + (cell - t.width) // 2
        y = pad + r * (cell + pad) + (cell - t.height) // 2
        sheet.paste(t, (x, y))
    return sheet


def build_evidence(
    image: Image.Image,
    extra_images: "list[Image.Image] | None" = None,
    *,
    detector: Detector,
    embed: Embedder | None,
    who: str = "",
    traits: str = "",
    grid: bool = False,
) -> EvidenceSet:
    """Primary photo (+ optional extra photos of the same person) → EvidenceSet.

    Raises ValueError when no face is found in the primary photo.  Extra photos
    without a detectable face are skipped with a note, never fatal.
    """
    image = image.convert("RGB")
    notes: list[str] = []

    primary = _photo(image, detector, embed)
    if primary is None:
        if embed is None:  # mock mode: no models loaded, synthesize a face box
            primary = _mock_photo(image)
        else:
            raise ValueError("No face detected in the primary photo.")

    photos = [primary]
    for i, extra in enumerate(extra_images or []):
        p = _photo(extra.convert("RGB"), detector, embed)
        if p is None:
            notes.append(f"extra photo {i + 1}: no face detected, skipped")
            continue
        photos.append(p)

    working, crop_meta = face_crop.head_shoulders_crop(image, primary.face.bbox[:4])
    mirror = ImageOps.mirror(working)

    # Standard layout (v4-measured winner): density beats count — extras join
    # as TIGHT crops; scene-crop piles measurably diluted identity.
    extra_tights = [p.tight for p in photos[1:MAX_EXTRA_TIGHTS + 1]]
    refs_std = [working] + extra_tights
    if len(refs_std) < MAX_REFS - 1:
        refs_std.append(mirror)

    refs_grid = None
    if grid:
        tiles = [primary.tight, ImageOps.mirror(primary.tight)] + extra_tights[:2]
        refs_grid = [identity_sheet(tiles), working]

    # Fused quality-weighted anchor over photos AND mirrors.
    fused = None
    if embed is not None:
        acc, total = None, 0.0
        for p in photos:
            for v in (p.embedding, p.mirror_embedding):
                if v is None:
                    continue
                acc = v * p.quality if acc is None else acc + v * p.quality
                total += p.quality
        fused = unit(acc) if (acc is not None and total > 0) else None

    face_px = primary.face_px
    tier = "evidence-poor" if 0 < face_px < EVIDENCE_POOR_PX else "evidence-rich"

    if not who:
        sex = str(getattr(primary.face, "sex", "") or "")
        who = {"M": "a man", "F": "a woman"}.get(sex, "")
    if not traits and tier == "evidence-rich":
        age = getattr(primary.face, "age", None)
        if age:
            traits = f"About {int(age)} years old."

    return EvidenceSet(
        photos=photos,
        working=working,
        crop_meta=crop_meta,
        mirror=mirror,
        refs_std=refs_std,
        refs_grid=refs_grid,
        v_primary=primary.embedding,
        v_mirror=primary.mirror_embedding,
        fused=fused,
        tier=tier,
        face_px=face_px,
        who=who,
        traits=traits,
        notes=notes,
    )
