"""iterate_v7 — benchmark harness for the halfface v7 "Witness" pipeline.

One run = one subject through the v7 package: evidence → described pool →
gates → pick (→ CCTV enrichment chain).  Writes everything needed to judge a
run into Test_AI_Half/v7/<subject>__<config>/:

    crop.png                 head-shoulders working crop (renderer ref #1)
    cand_<style>_<layout>_<seed>.png
    winner.png
    board.png                input | winner | truth/target strip
    contact.png              every candidate, labelled
    report.json              per-candidate metrics + pick + ledger + prompts

Honest metrics: vs REAL truth for whatsapp AND young8 (same person — the real
frontal exists), fused-evidence/per-photo for multi-photo subjects, min_both
for single-photo ones.  AI-panel targets are reported as legacy columns only.

Budget discipline: --dry prints prompts and the projected render count and
exits; --max-renders aborts before the first render if the plan exceeds it;
the ledger (count, seconds, est USD) prints after every run.  ~$0.06/render.

Usage (from D:\\Claude\\Forensic_ART_Original\\Forensic_Art):
    python tools/iterate_v7.py --subject whatsapp --mock
    python tools/iterate_v7.py --subject whatsapp --dry
    python tools/iterate_v7.py --subject young8 --config v7-base
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]          # Forensic_Art
sys.path.insert(0, str(ROOT / "backend"))

import fal_provider  # noqa: E402
from halfface import build_evidence  # noqa: E402
from halfface.generate import Arm, EST_USD_PER_RENDER  # noqa: E402
from halfface.pipeline import Options, reconstruct_frontal  # noqa: E402
from halfface.rank import vs_image  # noqa: E402

HALF = ROOT / "half photo"
SCOPE = HALF / "Scope"
OUT_ROOT = ROOT.parent / "Test_AI_Half" / "v7"

# ---------------------------------------------------------------------------
# Examiner descriptions — written from the evidence photos (analyst pass,
# 2026-08-03).  On the lab box a local open VLM (Qwen2.5-VL) fills this role;
# the API exposes the same field as HalfFaceReq.description.
# ---------------------------------------------------------------------------
DESC_YOUNG = (
    "A short, broad face with full cheeks and a soft, rounded jawline. Thick, voluminous "
    "dark curly hair, high on top. A thin, sparse young-man's mustache and only faint "
    "sparse stubble on the chin — no beard on the cheeks. Medium-thick dark eyebrows with "
    "a soft natural arch. Medium-size dark brown eyes with slightly hooded lids. A "
    "prominent, straight nose with a rounded, slightly downturned tip. Full lips, the "
    "lower lip fuller. A rounded, slightly recessed chin. Warm light-brown skin with "
    "natural texture, around twenty years old"
)
DESC_MAN2 = (
    "A broad, full face with rounded cheeks and a wide jaw. Short, dark, tightly curled "
    "hair, neatly faded at the temples. A natural, uneven dark beard — moderately dense "
    "along the jawline and chin, patchier on the cheeks — joined to a modest mustache, "
    "not groomed or sculpted. A straight nose with a slightly drooping, rounded tip. "
    "Dark eyebrows of medium thickness. Slightly tired dark eyes with faint shadows "
    "beneath. Medium-brown skin, a solidly built man in his early thirties"
)
DESC_NAWAF = (
    "An oval face with full cheeks and a softly defined jaw. Short dark hair with a "
    "clearly receding hairline — deep set-back at both temples leaving a narrow front "
    "peak. A sparse, patchy dark beard: light stubble on the cheeks, denser along the "
    "jawline and chin, with a thin mustache. Dark, medium-thick eyebrows. Calm dark eyes "
    "with slightly heavy upper lids. A straight nose with a rounded tip. A fuller neck "
    "and solid build. Warm medium skin tone, a man around thirty"
)

# Contingency variant: the full description measurably hurt the CLEAN
# close-up tier (whatsapp pool best 0.56 vs champion 0.6339) — text competes
# with strong pixels. The lite variant keeps only the three facts FLUX got
# wrong there (mustache weight, eye shape, chin) inside the proven idportrait
# framing, plus a chain.
DESC_YOUNG_LITE = (
    "A thin, sparse young-man's mustache — much lighter than average. Medium-size, "
    "slightly hooded dark eyes. A rounded, slightly recessed chin and full cheeks"
)

SUBJECTS: dict[str, dict] = {
    "whatsapp-lite": {
        "primary": HALF / "WhatsApp Image 2026-08-02 at 6.38.18 AM (2).jpeg",
        "extras": [],
        "truth": HALF / "WhatsApp Image 2026-08-02 at 6.38.17 AM (1).jpeg",
        "target": None,
        "description": DESC_YOUNG_LITE,
        "arms": [
            Arm("forensic", (101, 202, 301)),   # lite facts in the studio skeleton
        ],
        "force_enrich": True,          # rich tier, but chains built the 0.6339
        "prev": {"vs_truth": 0.6339},
    },
    "whatsapp": {
        "primary": HALF / "WhatsApp Image 2026-08-02 at 6.38.18 AM (2).jpeg",
        "extras": [],
        "truth": HALF / "WhatsApp Image 2026-08-02 at 6.38.17 AM (1).jpeg",
        "target": None,
        "description": DESC_YOUNG,
        "arms": [
            Arm("forensic", (7, 21, 77)),
            Arm("forensicdoc", (7, 21, 101)),
            Arm("idportrait", (7, 101)),           # v5 control (regression insurance)
        ],
        "prev": {"vs_truth": 0.6339},
    },
    "young8": {
        "primary": SCOPE / "in_young8_cctv.png",
        "extras": [
            SCOPE / "in_clean_profile.png",
            SCOPE / "in_near_cctv.png",
            SCOPE / "in_young7_cctv.png",
        ],
        "truth": HALF / "WhatsApp Image 2026-08-02 at 6.38.17 AM (1).jpeg",  # same person
        "target": SCOPE / "tgt_young8.png",        # AI panel — legacy column only
        "description": DESC_YOUNG,
        "grid": True,
        "arms": [
            Arm("forensic", (7, 21, 77)),
            Arm("forensicdoc", (7, 21, 77)),
            Arm("forensic", (7, 21), layout="grid"),
        ],
        "prev": {"sim_fused": 0.5207, "vs_target": 0.5333},
    },
    "man2": {
        "primary": SCOPE / "in_man2_cctv.png",
        "extras": [SCOPE / "in_man1_cctv.png", SCOPE / "in_man3_cctv.png"],
        "truth": None,
        "target": SCOPE / "tgt_man2.png",          # AI panel — legacy column only
        "description": DESC_MAN2,
        "arms": [
            Arm("forensic", (7, 21, 77)),
            Arm("forensicdoc", (7, 21, 77)),
            Arm("idportrait", (7, 21)),
        ],
        "prev": {"sim_fused": 0.4711, "vs_target": 0.4849},
    },
    "nawaf": {
        "primary": HALF / "nawaf.jpeg",
        "extras": [],
        "truth": None,
        "target": None,
        "description": DESC_NAWAF,
        "arms": [                      # balanced A/B: description vs control
            Arm("forensic", (7, 21)),
            Arm("forensicdoc", (7, 21)),
            Arm("idportrait", (7, 21)),
        ],
        "prev": {"min_both": 0.5345, "min_both_v7anchor": 0.5185, "fused_v7anchor": 0.5533},
    },
}


def load_engine(skip: bool):
    """Local buffalo_l (CPU). Free — loaded even for --mock so the whole
    metric/gate path runs; --no-engine skips it for pure plumbing checks."""
    if skip:
        return None, None
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def detector(img: Image.Image):
        import numpy as np

        return app.get(np.asarray(img.convert("RGB"))[:, :, ::-1])

    def embed(img: Image.Image):
        faces = detector(img)
        if not faces:
            return None
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best.normed_embedding

    return detector, embed


def make_renderer(args):
    if args.mock:
        def renderer(refs, prompt, width, height, seed):
            img = refs[0].convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            return img, {"provider": "mock"}
        return renderer

    def renderer(refs, prompt, width, height, seed):
        w, h = fal_provider.fit_size(width, height)
        images, slug = fal_provider.edit_images(
            prompt, refs, width=w, height=h, seed=seed,
            steps=args.steps, guidance=args.guidance, model=args.model,
        )
        return images[0], {"provider": "fal", "model": slug}
    return renderer


def label_strip(images: "list[tuple[str, Image.Image]]", height: int = 720) -> Image.Image:
    font = ImageFont.load_default(size=22)
    tiles = []
    for label, img in images:
        img = img.convert("RGB")
        w = int(round(img.width * height / img.height))
        tiles.append((label, img.resize((w, height), Image.Resampling.LANCZOS)))
    total_w = sum(t.width for _, t in tiles) + 8 * (len(tiles) - 1)
    board = Image.new("RGB", (total_w, height + 34), "black")
    x = 0
    draw = ImageDraw.Draw(board)
    for label, t in tiles:
        board.paste(t, (x, 34))
        draw.text((x + 6, 6), label, fill="yellow", font=font)
        x += t.width + 8
    return board


def run_subject(name: str, args) -> dict:
    spec = SUBJECTS[name]
    out_dir = Path(args.out) if args.out else OUT_ROOT / f"{name}__{args.config}"
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = Image.open(spec["primary"]).convert("RGB")
    extras = [Image.open(p).convert("RGB") for p in spec["extras"]]
    truth = Image.open(spec["truth"]).convert("RGB") if spec.get("truth") else None
    target = Image.open(spec["target"]).convert("RGB") if spec.get("target") else None

    arms = spec["arms"]
    projected = sum(len(a.seeds) for a in arms)
    enrich_budget = 4 if not args.no_enrich else 0
    print(f"[{name}] projected base renders: {projected} (+<= {enrich_budget} enrichment)"
          f" ~ ${(projected + enrich_budget) * EST_USD_PER_RENDER:.2f} max")
    if args.dry:
        from halfface import prompts as P

        desc = spec["description"] if not args.no_desc else ""
        n_photos = 1 + len(spec["extras"])
        seen = set()
        for arm in arms:
            key = (arm.style, arm.layout)
            if key in seen:
                continue
            seen.add(key)
            text = P.build(arm.style, who="a man", description=desc,
                           n_photos=n_photos, grid=(arm.layout == "grid"))
            print(f"\n--- {name} :: {arm.style}/{arm.layout} ({len(text)} chars) ---\n{text}")
        return {"subject": name, "dry": True, "projected_renders": projected}
    if projected + enrich_budget > args.max_renders:
        raise SystemExit(f"[{name}] plan {projected}+{enrich_budget} exceeds --max-renders {args.max_renders}")

    detector, embed = load_engine(args.no_engine)
    renderer = make_renderer(args)

    ev = build_evidence(
        primary, extras, detector=detector or (lambda i: []), embed=embed,
        who="a man", grid=bool(spec.get("grid")),
    )
    print(f"[{name}] evidence: {len(ev.photos)} photo(s), face_px={ev.face_px}, tier={ev.tier}")
    ev.working.save(out_dir / "crop.png")

    description = spec["description"] if not args.no_desc else ""
    if description:
        ev.traits = ""  # the description states age; a conflicting detector age would fight it
    enrich: bool | None = None
    if args.mock or args.no_enrich:
        enrich = False
    elif spec.get("force_enrich"):
        enrich = True
    opts = Options(
        arms=arms,
        description=description,
        enrich=enrich,
    )
    t0 = time.time()
    result = reconstruct_frontal(ev, renderer=renderer, detector=detector, embed=embed, opts=opts)

    rows = []
    for cand in result.candidates:
        cand.image.save(out_dir / f"cand_{cand.style}_{cand.layout}_{cand.seed}.png")
        row = cand.meta()
        row["vs_truth"] = vs_image(cand.image, truth, embed)
        row["vs_target"] = vs_image(cand.image, target, embed)
        rows.append(row)
    result.winner.image.save(out_dir / "winner.png")

    picked = result.winner.meta()
    picked["vs_truth"] = vs_image(result.winner.image, truth, embed)
    picked["vs_target"] = vs_image(result.winner.image, target, embed)

    # Oracle (best candidate by the honest offline metric) — report only.
    oracle = None
    honest = "vs_truth" if truth is not None else ("sim_fused" if ev.multi else "min_both")
    scored = [r for r in rows if r.get(honest) is not None]
    if scored:
        oracle = max(scored, key=lambda r: r[honest])

    report = {
        "version": "v7.0",
        "subject": name,
        "config": args.config,
        "description_used": bool(spec["description"] and not args.no_desc),
        "evidence": [str(spec["primary"])] + [str(p) for p in spec["extras"]],
        "truth": str(spec["truth"]) if spec.get("truth") else None,
        "target_legacy": str(spec["target"]) if spec.get("target") else None,
        "tier": result.tier,
        "enriched": result.enriched,
        "honest_metric": honest,
        "prev_best": spec.get("prev", {}),
        "picked": picked,
        "oracle": oracle,
        "candidates": rows,
        "ledger": result.ledger.summary() if result.ledger else None,
        "prompts": result.prompts_used,
        "seconds_total": round(time.time() - t0, 1),
        "notes": ev.notes,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    strip = [("input", primary), ("winner", result.winner.image)]
    if truth is not None:
        strip.append(("REAL truth", truth))
    if target is not None:
        strip.append(("AI target (legacy)", target))
    label_strip(strip).save(out_dir / "board.png")
    label_strip(
        [(f"{c.style}/{c.layout}#{c.seed}" + (" GATED" if c.gated else ""), c.image)
         for c in result.candidates],
        height=480,
    ).save(out_dir / "contact.png")

    led = result.ledger.summary() if result.ledger else {}
    print(f"[{name}] picked {picked['style']}/{picked['layout']}#{picked['seed']} "
          f"min_both={picked['min_both']} fused={picked['sim_fused']} "
          f"vs_truth={picked['vs_truth']} vs_target={picked['vs_target']}")
    print(f"[{name}] prev best: {spec.get('prev')}")
    print(f"[{name}] ledger: {led}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True, choices=sorted(SUBJECTS) + ["all"])
    ap.add_argument("--config", default="v7-base")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mock", action="store_true", help="no fal spend; refs[0] echo renderer")
    ap.add_argument("--no-engine", action="store_true", help="skip insightface too (plumbing only)")
    ap.add_argument("--dry", action="store_true", help="print plan + prompts, render nothing")
    ap.add_argument("--no-desc", action="store_true", help="ablation: drop the description block")
    ap.add_argument("--no-enrich", action="store_true")
    ap.add_argument("--max-renders", type=int, default=14, help="hard per-subject cap")
    ap.add_argument("--model", default="dev")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    args = ap.parse_args()

    names = sorted(SUBJECTS) if args.subject == "all" else [args.subject]
    for name in names:
        run_subject(name, args)


if __name__ == "__main__":
    main()
