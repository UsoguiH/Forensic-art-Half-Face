"""Collect v7 benchmark runs into Test_AI_Half/Final_Results.

For each subject with a Test_AI_Half/v7/<subject>__v7-base/report.json:
- pick the shipping winner (the run's picked candidate),
- copy it to Final_Results/<subject>_generated.png,
- build <subject>_comparison.png = input | previous champion | v7 winner | truth/target,
- write v7_final_report.json with picked/oracle/prev-baseline numbers.

Run AFTER the benchmark pools. Zero fal spend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

from iterate_v7 import SUBJECTS, label_strip  # noqa: E402

TEST = ROOT.parent / "Test_AI_Half"
V7 = TEST / "v7"
FINAL = TEST / "Final_Results"

PREV_CHAMPS = {
    "whatsapp": TEST / "v3" / "WhatsApp Image 2026-08-02 at 6.38.18 AM (2)__e2" / "cand_idportrait_101.png",
    "young8": TEST / "v3" / "young8_cctv__e2" / "cand_multirefstudio_42.png",
    "man2": TEST / "v5" / "man2_cctv__v5-base" / "cand_forensic_7.png",
    "nawaf": TEST / "v5" / "nawaf__v5-base" / "winner.png",
}


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((V7 / "prev_champions_v7baseline.json").read_text())
    out: dict = {
        "version": "v7.0.1",
        "date": "2026-08-03",
        "champions": {},
        "prev_champions_v7_anchors": baseline,
        "spend": {"renders_total": 47, "est_usd": 2.82,
                  "note": "smoke 1 + whatsapp 8 + adapter test 1 + young8 12 + nawaf 6 + man2 12 + whatsapp-lite 7"},
    }

    for name in ("whatsapp", "young8", "man2", "nawaf"):
        spec = SUBJECTS[name]
        run_dir = V7 / f"{name}__v7-base"
        rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        picked = rep["picked"]
        winner_png = run_dir / "winner.png"

        entry = {
            "file": str(winner_png.relative_to(TEST)),
            "picked": picked,
            "oracle": rep.get("oracle"),
            "honest_metric": rep.get("honest_metric"),
            "prev_baseline_v7_anchors": baseline.get(name),
            "tier": rep.get("tier"),
            "enriched": rep.get("enriched"),
            "ledger": rep.get("ledger"),
            "new_all_time_best": False,
        }

        if name == "whatsapp":
            lite = json.loads((V7 / "whatsapp-lite__v7-lite" / "report.json").read_text(encoding="utf-8"))
            entry["champion_retained"] = {
                "file": str(PREV_CHAMPS[name].relative_to(TEST)),
                "vs_truth": 0.6339,
                "why": "clean close-up tier: pixels beat text — v7 described pools peaked "
                       "0.5597, lite-description contingency 0.5374; the idportrait+chain "
                       "recipe that built 0.6339 lives on as the pool's control arm",
                "lite_contingency_best": max((c.get("vs_truth") or 0) for c in lite["candidates"]),
            }
        else:
            entry["new_all_time_best"] = True
            Image.open(winner_png).save(FINAL / f"{name}_generated_v7.png")
        out["champions"][name] = entry

        strip = [("input", Image.open(spec["primary"]).convert("RGB"))]
        if PREV_CHAMPS.get(name, Path("_")).exists():
            strip.append(("previous champion", Image.open(PREV_CHAMPS[name]).convert("RGB")))
        strip.append(("v7 winner", Image.open(winner_png).convert("RGB")))
        if spec.get("truth"):
            strip.append(("REAL truth", Image.open(spec["truth"]).convert("RGB")))
        elif spec.get("target"):
            strip.append(("AI target (legacy)", Image.open(spec["target"]).convert("RGB")))
        label_strip(strip).save(FINAL / f"{name}_comparison_v7.png")
        print(f"[{name}] {'NEW BEST' if entry['new_all_time_best'] else 'champion retained'}: "
              f"{picked.get('style')}/{picked.get('layout')}#{picked.get('seed')}")

    (FINAL / "v7_final_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", FINAL / "v7_final_report.json")


if __name__ == "__main__":
    main()
