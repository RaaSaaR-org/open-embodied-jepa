"""TASK-100: Markdown tables from a score report (calibration, val, smoke or test; protocol §11).

Reads ``evaluate_task100.py score``'s JSON and prints the row (test), the gate table per
eligible arm, seed and h with every interval, and the reported quantities. Never recomputes a
gate: it prints what the report holds. ``--check-sha256`` refuses a report whose hash differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def fmt(stat, digits: int = 3) -> str:
    if stat is None:
        return "—"
    if isinstance(stat, dict) and "value" in stat:
        lo, hi = stat.get("ci", [None, None])
        if lo is None or hi is None:
            return f"{stat['value']:.{digits}f}"
        return f"{stat['value']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
    return f"{stat:.{digits}f}" if isinstance(stat, float) else str(stat)


def tick(flag) -> str:
    return "pass" if flag else "FAIL"


def pair_table(report: dict) -> list[str]:
    lines = [
        "| X | h | G1 coll. | rank ratio | std ratio | G1 (iii) diff. | copy ratio | X / N_X "
        "| sens | cube median (cm) | cube / copy (M) | cube / N_X (M) | jaw median (rad) "
        "| jaw / copy (M) | jaw / N_X (M) |",
        "|" + "---|" * 15,
    ]
    for name, entry in sorted(report.get("pairs", {}).items()):
        for h in sorted((k for k in entry if k.isdigit()), key=int):
            e = entry[h]
            g, g5 = e["g1_stats"], e["g5"]
            lines.append(
                f"| {name} | {h} | {g['collapsed_fraction']:.3f} | {g['effective_rank_ratio']:.3f} "
                f"| {g['std_ratio']:.3f} | {fmt(e['g1_iii'])} | {fmt(e['copy_ratio'])} "
                f"| {fmt(e['noaction_ratio'])} | {fmt(e['sens'])} | {fmt(g5['cube']['median'])} "
                f"| {fmt(g5['cube']['vs_copy'])} | {fmt(g5['cube']['vs_noaction'])} "
                f"| {fmt(g5['jaw']['median'], 4)} | {fmt(g5['jaw']['vs_copy'])} "
                f"| {fmt(g5['jaw']['vs_noaction'])} |"
            )
    return lines


def gate_table(report: dict) -> list[str]:
    lines = [
        "| arm | seed | h | G1 | G2 | G3 | G4 | G5 cube a/b/c | G5 jaw a/b/c | G6 |",
        "|" + "---|" * 10,
    ]
    for arm, seeds in sorted(report.get("gates", {}).items()):
        for seed, hs in sorted(seeds.items()):
            for h in sorted((k for k in hs if k.isdigit()), key=int):
                g = hs[h]
                c, j = g["G5"]["cube"], g["G5"]["jaw"]
                lines.append(
                    f"| {arm} | {seed} | {h} | {tick(g['G1']['all'])} | {tick(g['G2'])} "
                    f"| {tick(g['G3'])} | {tick(g['G4'])} "
                    f"| {'/'.join(tick(c[k]) for k in 'abc')} "
                    f"| {'/'.join(tick(j[k]) for k in 'abc')} "
                    f"| {tick(report['g6'][arm][seed])} |"
                )
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True)
    parser.add_argument("--check-sha256", default=None)
    args = parser.parse_args(argv)
    path = Path(args.report)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.check_sha256 and sha != args.check_sha256:
        raise SystemExit("the report differs from its recorded sha256")
    report = json.loads(path.read_text())
    out = [
        f"# TASK-100 {report.get('purpose')} report",
        "",
        f"- report: `{path}` (sha256 `{sha}`)",
        f"- revision: `{report.get('revision')}`; roots {report.get('roots')}, episodes "
        f"{report.get('episodes')}; outcome {report.get('outcome')}",
    ]
    if "row" in report:
        out += ["", f"**Row: {report['row'].get('row')}** — {json.dumps(report['row'])}"]
    if "calibration" in report:
        c = report["calibration"]
        out += [
            "",
            "## Stage 0 bars",
            "",
            f"- bars: {json.dumps(c['bars'])}",
            f"- G1 bar derivation: {json.dumps(c['g1_bars'])}",
            f"- moving subsets: {json.dumps(c['moving_thresholds'])}",
            f"- G4 calibration lb below 1.10: {c['g4_lb_below_1p10']}",
            f"- G5 calibration misses a bar: {json.dumps(c['g5_calibration_misses_bar'])}",
        ]
    out += ["", "## Pairs (gate-level intervals)", "", *pair_table(report)]
    if "gates" in report:
        out += ["", "## Gates", "", *gate_table(report)]
    ce = report.get("renc_ceiling", {})
    if ce:
        out += ["", "## R-enc on encoded frames (ceiling)", ""]
        for enc, v in ce.items():
            out.append(f"- {enc}: all frames {json.dumps(v.get('all_frames_median'))}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
