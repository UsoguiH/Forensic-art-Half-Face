"""Recipe iteration harness for the v3 profile→ID-portrait pipeline.

One run = one (input, config) pair: detect → head-and-shoulders crop →
candidate pool via faceid_reconstruct2 → rank → score. Everything a run
produces lands in one folder so configs can be eyeballed side by side:

    Test_AI_Half/v3/<subject>__<config>/
        crop.png            what the renderer actually saw
        cand_<style>_<seed>.png
        winner.png          best by min-both-sides (+ consensus, see report)
        sheet.png           input | crop | winner | target
        report.json         every score, both rankings, config, timings

The target (when the scope set has one) is used for EVALUATION ONLY — it never
influences generation or ranking, exactly as in production where it does not
exist. Both rankings (plain min-both-sides and consensus-blended) are reported
for every run, so a single paid pool doubles as the consensus-weight ablation.

    python tools/iterate_v3.py --input "half photo/Scope/in_man2_cctv.png" \
        --target "half photo/Scope/tgt_man2.png" --config crop-idp \
        --styles descriptive,idportrait --seeds 7,21 --crop --portrait
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import face_crop  # noqa: E402
import faceid_reconstruct2 as f2  # noqa: E402
import fal_provider  # noqa: E402

PORTRAIT = (832, 1216)


# --------------------------------------------------------------------------- #

def load_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def detect(app, image: Image.Image) -> list:
    try:
        return list(app.get(np.asarray(image.convert("RGB"))[:, :, ::-1]))
    except Exception:
        return []


def embed(app, image: Image.Image):
    face = face_crop.biggest_face(detect(app, image))
    if face is None:
        return None
    v = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def cos(a, b) -> float | None:
    if a is None or b is None:
        return None
    return round(float(np.dot(a, b)), 4)


def frontality(face) -> float:
    """0 = profile, 1 = frontal, from the 5-point kps (see tools/test_half_face.py)."""
    if face is None:
        return 0.0
    try:
        kps = np.asarray(face.kps, dtype=np.float32)
        x0, _, x1, _ = (float(v) for v in face.bbox[:4])
    except Exception:
        return 0.0
    le, re, nose = kps[0], kps[1], kps[2]
    gap = float(abs(re[0] - le[0]))
    box = max(1.0, x1 - x0)
    if gap < 1e-3:
        return 0.0
    centred = 1.0 - min(1.0, abs(float(nose[0]) - (float(le[0]) + float(re[0])) / 2.0) / (gap / 2.0))
    spread = min(1.0, (gap / box) / 0.34)
    return round(float(min(centred, spread)), 3)


MIN_FRONTALITY = 0.55
SUSPICIOUS = 0.74     # min_both above this = pose/embedding clone, not a match


# --------------------------------------------------------------------------- #

def contact_sheet(panels: list[Image.Image], labels: list[str]) -> Image.Image:
    from PIL import ImageDraw

    height = 720
    scaled = [
        p.resize((max(1, round(p.width * height / p.height)), height), Image.Resampling.LANCZOS)
        for p in panels
    ]
    gap, top = 10, 26
    sheet = Image.new(
        "RGB", (sum(p.width for p in scaled) + gap * (len(scaled) - 1), height + top), (14, 16, 20)
    )
    draw = ImageDraw.Draw(sheet)
    x = 0
    for panel, label in zip(scaled, labels):
        draw.text((x + 4, 5), label, fill=(255, 235, 120))
        sheet.paste(panel, (x, top))
        x += panel.width + gap
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", default=None, help="real/reference frontal for scoring only")
    parser.add_argument("--config", required=True, help="short label naming this recipe variant")
    parser.add_argument("--styles", default="descriptive,idportrait")
    parser.add_argument("--seeds", default="7,21")
    parser.add_argument("--crop", action="store_true", help="head-and-shoulders normalize first")
    parser.add_argument("--refs", default="mirror", choices=("mirror", "tight-hs", "tight-mirror"),
                        help="mirror: [working, mirror] | tight-hs: [tight face, working] | "
                             "tight-mirror: [tight face, mirrored tight face]")
    parser.add_argument("--enrich", default=None,
                        help="path to a prior winning candidate, appended as an extra "
                             "reference (Phase D enrichment experiment — evaluation decides "
                             "whether it helps; it never edits that image directly)")
    parser.add_argument("--portrait", action="store_true", help=f"render at fixed {PORTRAIT}")
    parser.add_argument("--model", default=None, help="dev | klein | full fal slug")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--who", default=None, help="override auto age/sex, e.g. 'a man'")
    parser.add_argument("--traits", default=None)
    parser.add_argument("--extra", default="", help="extra prompt sentence")
    parser.add_argument("--consensus-w", type=float, default=0.5)
    parser.add_argument("--out", default=str(ROOT.parent / "Test_AI_Half" / "v3"))
    args = parser.parse_args()

    if not fal_provider.available():
        print("FAL key missing.")
        return 2

    app = load_app()
    source = Image.open(args.input).convert("RGB")
    subject = Path(args.input).stem.replace("in_", "")
    out_dir = Path(args.out) / f"{subject}__{args.config}"
    out_dir.mkdir(parents=True, exist_ok=True)

    faces = detect(app, source)
    face = face_crop.biggest_face(faces)
    if face is None:
        print("No face found in input.")
        return 2

    # --- normalize -----------------------------------------------------------
    crop_meta = {}
    working = source
    if args.crop:
        working, crop_meta = face_crop.head_shoulders_crop(source, face.bbox[:4])
    working.save(out_dir / "crop.png")

    # --- who/traits ----------------------------------------------------------
    who = args.who
    if who is None:
        sex = getattr(face, "sex", None)
        who = {"M": "a man", "F": "a woman"}.get(sex, "")
    traits = args.traits
    if traits is None:
        age = getattr(face, "age", None)
        traits = f"About {int(age)} years old." if age else ""

    # --- references ----------------------------------------------------------
    references = None                      # f2 default: [working, mirror(working)]
    if args.refs != "mirror":
        tight = face_crop.tight_face_crop(source, face.bbox[:4])
        tight.save(out_dir / "tight.png")
        references = (
            [tight, working] if args.refs == "tight-hs"
            else [tight, ImageOps.mirror(tight)]
        )
    if args.enrich:
        if references is None:
            references = [working, ImageOps.mirror(working)]
        references.append(Image.open(args.enrich).convert("RGB"))

    # --- generate ------------------------------------------------------------
    styles = tuple(s.strip() for s in args.styles.split(",") if s.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    out_size = PORTRAIT if args.portrait else None
    render_log: list[dict] = []

    def renderer(images, prompt, width, height, seed):
        w, h = fal_provider.fit_size(width, height)
        started = time.time()
        out, slug = fal_provider.edit_images(
            prompt, images, width=w, height=h, seed=seed,
            steps=args.steps, guidance=args.guidance, model=args.model,
        )
        meta = {"model": slug, "render_size": [w, h], "seconds": round(time.time() - started, 2)}
        render_log.append(meta)
        print(f"    render {len(render_log)}: seed={seed} {w}x{h} {meta['seconds']}s")
        return out[0], meta

    print(f">>> {subject} [{args.config}] styles={styles} seeds={seeds} "
          f"crop={args.crop} portrait={args.portrait} who='{who}' traits='{traits}'")
    result = f2.reconstruct(
        working,
        renderer=renderer,
        embed=lambda im: embed(app, im),
        seeds=seeds,
        prompt_styles=styles,
        who=who,
        traits=traits,
        extra_prompt=args.extra,
        out_size=out_size,
        references=references,
    )

    # --- rank + score --------------------------------------------------------
    v_profile = embed(app, working)
    v_mirror = embed(app, ImageOps.mirror(working))
    v_target = embed(app, Image.open(args.target).convert("RGB")) if args.target else None

    cands = []
    vecs = []
    for meta, img in zip(result["candidates"], result["candidate_images"]):
        f = face_crop.biggest_face(detect(app, img))
        v = None
        if f is not None:
            v = np.asarray(f.normed_embedding, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(v))
            v = v / n if n > 0 else None
        vecs.append(v)
        cands.append({**meta, "image": img, "vec": v, "frontality": frontality(f)})

    for i, c in enumerate(cands):
        others = [v for j, v in enumerate(vecs) if j != i and v is not None]
        consensus = (
            round(float(np.mean([np.dot(c["vec"], o) for o in others])), 4)
            if c["vec"] is not None and others else None
        )
        min_both = (
            round(min(float(np.dot(c["vec"], v_profile)), float(np.dot(c["vec"], v_mirror))), 4)
            if c["vec"] is not None and v_profile is not None and v_mirror is not None else None
        )
        c["consensus"] = consensus
        c["min_both"] = min_both
        c["blend"] = (
            round(min_both + args.consensus_w * consensus, 4)
            if min_both is not None and consensus is not None else min_both
        )
        c["vs_target"] = cos(c["vec"], v_target)
        c["gated"] = bool(
            c["frontality"] < MIN_FRONTALITY
            or (c["min_both"] is not None and c["min_both"] > SUSPICIOUS)
        )

    pool = [c for c in cands if not c["gated"]] or cands
    ranked_min = max(pool, key=lambda c: c["min_both"] if c["min_both"] is not None else -1)
    ranked_blend = max(pool, key=lambda c: c["blend"] if c["blend"] is not None else -1)
    oracle = (
        max(cands, key=lambda c: c["vs_target"] if c["vs_target"] is not None else -1)
        if v_target is not None else None
    )

    for c in cands:
        c["image"].save(out_dir / f"cand_{c['stage']}_{c['seed']}.png")
    # Round-1 finding: the consensus blend picked worse than plain min-both in
    # a 4-candidate pool (consensus rewards the generic). min-both stays the
    # shipping pick; blend remains in the report for the ablation record.
    winner = ranked_min
    winner["image"].save(out_dir / "winner.png")

    panels = [source, working, winner["image"]]
    labels = ["input", "crop" if args.crop else "as-is", f"winner {winner['stage']}#{winner['seed']}"]
    if args.target:
        panels.append(Image.open(args.target).convert("RGB"))
        labels.append("target")
    contact_sheet(panels, labels).save(out_dir / "sheet.png")

    report = {
        "subject": subject, "config": args.config,
        "input": args.input, "target": args.target,
        "styles": list(styles), "seeds": list(seeds),
        "crop": bool(args.crop), "crop_meta": crop_meta, "refs": args.refs,
        "portrait": bool(args.portrait), "model": args.model,
        "who": who, "traits": traits, "extra": args.extra,
        "consensus_w": args.consensus_w,
        "candidates": [
            {k: v for k, v in c.items() if k not in ("image", "vec")} for c in cands
        ],
        "picked_min_both": {"stage": ranked_min["stage"], "seed": ranked_min["seed"],
                            "vs_target": ranked_min["vs_target"]},
        "picked_blend": {"stage": ranked_blend["stage"], "seed": ranked_blend["seed"],
                         "vs_target": ranked_blend["vs_target"]},
        "oracle": ({"stage": oracle["stage"], "seed": oracle["seed"],
                    "vs_target": oracle["vs_target"]} if oracle else None),
        "profile_vs_target": cos(v_profile, v_target),
        "render_log": render_log,
        "fal_spend": fal_provider.spend(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"    profile-vs-target : {report['profile_vs_target']}")
    for c in cands:
        print(f"    {c['stage']:>12}#{c['seed']:<4} front={c['frontality']} "
              f"min_both={c['min_both']} consensus={c['consensus']} "
              f"vs_target={c['vs_target']}{'  [GATED]' if c['gated'] else ''}")
    print(f"    pick(min_both)={report['picked_min_both']}  pick(blend)={report['picked_blend']}")
    if oracle:
        print(f"    oracle={report['oracle']}")
    print(f"    -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
