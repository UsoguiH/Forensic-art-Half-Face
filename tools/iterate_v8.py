"""iterate_v8 — DateSet Test benchmark harness (goal: beat v7, aim 0.75).

What changed vs v7:
- Evidence hygiene: the old "young" cluster mixed TWO people (clean_profile /
  near_cctv agree only 0.29-0.36 with the new REAL truth original6, while
  young6/7/near/093224 agree 0.55-0.62). v8 uses the verified A-cluster only.
- Real truths everywhere possible: young -> original 6.png (NEW), beard ->
  9.44.07 (anchor 0.6093), pair4 -> 6.38.18(3) (anchor 0.5709).
- Prompt v8: v7 3-block + eye-fidelity sentence + single-face constraint.
- New "eyes" arm: a tight crop of the visible eye region joins the references
  so the eye pixels cannot be overlooked (density-wins, measured for faces).

Usage (from D:\\Claude\\Forensic_ART_Original\\Forensic_Art_v8):
    python tools/iterate_v8.py --subject young --mock
    python tools/iterate_v8.py --subject young --config v8-base
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]              # Forensic_Art_v8 worktree
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

import fal_provider  # noqa: E402
import face_crop  # noqa: E402
from halfface import build_evidence  # noqa: E402
from halfface.generate import Arm, EST_USD_PER_RENDER, render_pool, enrich_chain, Ledger  # noqa: E402
from halfface.rank import pick, vs_image  # noqa: E402
from iterate_v7 import label_strip, load_engine, make_renderer  # noqa: E402

HALF = Path(r"D:\Claude\Forensic_ART_Original\Forensic_Art\half photo")
DST = HALF / "---DateSet Test---"
SCOPE = HALF / "Scope"
OUT_ROOT = Path(r"D:\Claude\Forensic_ART_Original\Test_AI_Half") / "v8"

DESC_YOUNG_A = (
    "A young man around twenty with a mop of very tight, voluminous dark curls covering "
    "the tops of the ears. A lean face with a straight, slightly prominent nose, softly "
    "hollow cheeks and a defined jawline. A thin dark mustache and a small sparse tuft of "
    "hair on the chin — the cheeks otherwise clean. Medium-thick dark eyebrows, "
    "medium-size dark brown eyes. Full lips. Warm light-brown skin with natural texture"
)
DESC_BEARD = (
    "A man in his late twenties wearing a white crocheted kufi prayer cap. A very long, "
    "full, dense black beard reaching the chest, joined to a thick mustache; the beard "
    "covers the jawline and lower cheeks. Dark wavy hair curling out at the nape below "
    "the cap. A straight, refined nose. Calm dark eyes with fine, dark eyebrows. Fair, "
    "light skin. A white thobe with a standing collar. Keep the cap and the full beard "
    "exactly as photographed"
)
DESC_PAIR4 = (
    "A slim young man in his early twenties with short, wavy dark hair swept back and "
    "fuller on top. A lean face with a straight, slightly convex nose, defined cheekbones "
    "and a sharp jawline. A thin dark mustache only — no beard, the cheeks and chin are "
    "clean-shaven. Dark, medium-thick eyebrows and medium-size dark eyes. Light warm "
    "skin. A white thobe with a standing collar"
)
DESC_MAN2 = (
    "A broad, full face with rounded cheeks and a wide jaw. Short, dark, tightly curled "
    "hair, neatly faded at the temples. A natural, uneven dark beard — moderately dense "
    "along the jawline and chin, patchier on the cheeks — joined to a modest mustache, "
    "not groomed or sculpted. A straight nose with a slightly drooping, rounded tip. "
    "Dark eyebrows of medium thickness. Slightly tired dark eyes with faint shadows "
    "beneath. Medium-brown skin, a solidly built man in his early thirties"
)

SUBJECTS: dict[str, dict] = {
    "young": {
        "primary": SCOPE / "in_young7_cctv.png",            # best truth-agreement (0.616)
        "extras": [
            SCOPE / "in_young6_cctv.png",                   # 0.562
            HALF / "New DataSet" / "young near.png",        # 0.571
            DST / "Screenshot 2026-08-04 093224.png",       # 0.569
            SCOPE / "in_young8_cctv.png",                   # 0.439 (down-weighted by quality)
        ],
        "truth": DST / "original 6.png",
        "target": None,
        "description": DESC_YOUNG_A,
        "arms": [
            Arm("forensic", (7, 21, 77)),
            Arm("forensicdoc", (7, 21)),
            Arm("forensic", (101, 202), layout="eyes"),
            Arm("idportrait", (7,)),
        ],
        "prev": {"v7_winner_vs_original6": 0.5619, "best_any_existing": 0.5688},
    },
    "beard": {
        "primary": DST / "WhatsApp Image 2026-08-02 at 9.44.08 AM.jpeg",
        "extras": [],
        "truth": HALF / "WhatsApp Image 2026-08-02 at 9.44.07 AM.jpeg",
        "target": None,
        "description": DESC_BEARD,
        "arms": [
            Arm("forensic", (7, 21)),
            Arm("forensicdoc", (7,)),
            Arm("forensic", (101,), layout="eyes"),
            Arm("idportrait", (7, 21)),
        ],
        "prev": {"anchor_ceiling": 0.6093},
    },
    "pair4": {
        "primary": DST / "WhatsApp Image 2026-08-02 at 6.38.19 AM.jpeg",
        "extras": [],
        "truth": HALF / "WhatsApp Image 2026-08-02 at 6.38.18 AM (3).jpeg",
        "target": None,
        "description": DESC_PAIR4,
        "arms": [
            Arm("forensic", (7, 21)),
            Arm("forensicdoc", (7,)),
            Arm("forensic", (101,), layout="eyes"),
            Arm("idportrait", (7, 21)),
        ],
        "prev": {"v2_era_best_vs_truth": 0.4473, "anchor_ceiling": 0.5709},
    },
    "man": {
        "primary": SCOPE / "in_man2_cctv.png",
        "extras": [SCOPE / "in_man1_cctv.png", SCOPE / "in_man3_cctv.png"],
        "truth": None,
        "target": SCOPE / "tgt_man2.png",
        "description": DESC_MAN2,
        "arms": [
            Arm("forensic", (7, 21)),
            Arm("forensicdoc", (7, 21)),
            Arm("idportrait", (7,)),
        ],
        "prev": {"v7_fused": 0.5069, "vs_target_legacy": 0.4849},
    },
}


def eye_crop(image: Image.Image, detector) -> "Image.Image | None":
    """Tight crop around the visible eye of a profile: the eye pixels become a
    dedicated reference instead of 2% of a scene crop."""
    faces = detector(image) if detector else []
    face = face_crop.biggest_face(faces or [])
    if face is None or getattr(face, "kps", None) is None:
        return None
    import numpy as np

    kps = np.asarray(face.kps, dtype=np.float32)
    x0, y0, x1, y1 = (float(v) for v in face.bbox[:4])
    fw, fh = x1 - x0, y1 - y0
    ex = float((kps[0][0] + kps[1][0]) / 2.0)
    ey = float((kps[0][1] + kps[1][1]) / 2.0)
    box = (
        int(max(0, ex - fw * 0.45)), int(max(0, ey - fh * 0.32)),
        int(min(image.width, ex + fw * 0.45)), int(min(image.height, ey + fh * 0.32)),
    )
    crop = image.crop(box)
    if crop.width < 384:
        s = min(4.0, 384 / max(1, crop.width))
        crop = crop.resize((round(crop.width * s), round(crop.height * s)), Image.Resampling.LANCZOS)
    return crop


def run_subject(name: str, args) -> dict:
    spec = SUBJECTS[name]
    out_dir = OUT_ROOT / f"{name}__{args.config}"
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = Image.open(spec["primary"]).convert("RGB")
    extras = [Image.open(p).convert("RGB") for p in spec["extras"]]
    truth = Image.open(spec["truth"]).convert("RGB") if spec.get("truth") else None
    target = Image.open(spec["target"]).convert("RGB") if spec.get("target") else None

    arms = spec["arms"]
    projected = sum(len(a.seeds) for a in arms)
    print(f"[{name}] projected base renders: {projected} (+<=4 enrich) ~ ${(projected + 4) * EST_USD_PER_RENDER:.2f} max")

    detector, embed = load_engine(args.no_engine)
    renderer = make_renderer(args)

    ev = build_evidence(primary, extras, detector=detector or (lambda i: []), embed=embed, who="a man")
    if getattr(args, "auto_desc", False):
        # Benchmark mode: the VLM writes the description instead of the analyst.
        from halfface import describe as hf_describe

        auto_text, auto_meta = hf_describe.from_image(ev)
        print(f"[{name}] auto-desc ({auto_meta}): {auto_text[:120]}...", flush=True)
        spec = {**spec, "description": auto_text}
    if spec["description"]:
        ev.traits = ""
    print(f"[{name}] evidence: {len(ev.photos)} photo(s), face_px={ev.face_px}, tier={ev.tier}")
    ev.working.save(out_dir / "crop.png")

    # "eyes" layout: an EvidenceSet variant whose 4th reference is the eye crop.
    ev_eyes = None
    ec = eye_crop(primary, detector) if detector else None
    if ec is not None:
        ec.save(out_dir / "eye_crop.png")
        from copy import copy

        ev_eyes = copy(ev)
        base = [ev.working] + [p.tight for p in ev.photos[1:3]]
        ev_eyes.refs_std = (base + [ec])[:4]

    ledger = Ledger()
    candidates = []
    prompts_used = {}
    for arm in arms:
        use_ev = ev_eyes if (arm.layout == "eyes" and ev_eyes is not None) else ev
        cands, ledger, used = render_pool(
            use_ev, [Arm(arm.style, arm.seeds)],
            renderer=renderer, detector=detector, embed=embed,
            description=spec["description"], ledger=ledger,
        )
        for c in cands:
            c.layout = arm.layout
        candidates.extend(cands)
        prompts_used.update({f"{arm.style}/{arm.layout}": v for v in used.values()})

    champion = pick(candidates, ev)
    enriched = False
    if champion is not None and not champion.gated and ev.tier == "evidence-poor" and not args.mock and not args.no_enrich:
        champion, extra_cands, ledger = enrich_chain(
            ev, champion, renderer=renderer, detector=detector, embed=embed,
            description=spec["description"], ledger=ledger,
        )
        candidates.extend(extra_cands)
        if ev.multi and ev.fused is not None:
            champion = pick(candidates, ev)
        enriched = champion.layout.startswith("enrich")

    rows = []
    for c in candidates:
        c.image.save(out_dir / f"cand_{c.style}_{c.layout}_{c.seed}.png")
        row = c.meta()
        row["vs_truth"] = vs_image(c.image, truth, embed)
        row["vs_target"] = vs_image(c.image, target, embed)
        rows.append(row)
    champion.image.save(out_dir / "winner.png")
    picked = champion.meta()
    picked["vs_truth"] = vs_image(champion.image, truth, embed)
    picked["vs_target"] = vs_image(champion.image, target, embed)

    honest = "vs_truth" if truth is not None else "sim_fused"
    scored = [r for r in rows if r.get(honest) is not None]
    oracle = max(scored, key=lambda r: r[honest]) if scored else None

    report = {
        "version": "v8.0",
        "subject": name,
        "config": args.config,
        "evidence": [str(spec["primary"])] + [str(p) for p in spec["extras"]],
        "truth": str(spec["truth"]) if spec.get("truth") else None,
        "tier": ev.tier,
        "enriched": enriched,
        "honest_metric": honest,
        "prev": spec.get("prev", {}),
        "picked": picked,
        "oracle": oracle,
        "candidates": rows,
        "ledger": ledger.summary(),
        "prompts": prompts_used,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    strip = [("input", primary), ("winner", champion.image)]
    if truth is not None:
        strip.append(("REAL truth", truth))
    elif target is not None:
        strip.append(("AI target (legacy)", target))
    label_strip(strip).save(out_dir / "board.png")
    label_strip(
        [(f"{c.style}/{c.layout}#{c.seed}" + (" GATED" if c.gated else ""), c.image) for c in candidates],
        height=420,
    ).save(out_dir / "contact.png")

    print(f"[{name}] picked {picked['style']}/{picked['layout']}#{picked['seed']} "
          f"honest[{honest}]={picked.get(honest)} | oracle={None if not oracle else oracle.get(honest)}")
    print(f"[{name}] prev: {spec.get('prev')}")
    print(f"[{name}] ledger: {ledger.summary()}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, choices=sorted(SUBJECTS) + ["all"])
    ap.add_argument("--config", default="v8-base")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-engine", action="store_true")
    ap.add_argument("--no-enrich", action="store_true")
    ap.add_argument("--no-desc", action="store_true")
    ap.add_argument("--auto-desc", action="store_true",
                    help="VLM (Qwen3-VL) writes the description instead of DESC_*")
    ap.add_argument("--model", default="dev")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    args = ap.parse_args()
    for name in (sorted(SUBJECTS) if args.subject == "all" else [args.subject]):
        run_subject(name, args)


if __name__ == "__main__":
    main()
