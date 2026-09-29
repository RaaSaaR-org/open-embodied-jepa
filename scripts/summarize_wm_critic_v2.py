"""TASK-073 summarizer: the offline decision (O1-O5) and results manifests from stage reports.

Protocol ``docs/experiments/apple_wm_critic_v2.md`` §6.6 and §10. Every mode reads reports by
path and sha256, refuses a V report, recomputes each row with the design module's decision
functions and refuses to write if the recomputation disagrees with the report.

- ``offline``: K0 (O5), train (O1, O2) and rank (O3, O4) -> ``decide_offline``; writes the
  offline decision the D3 stage requires (it pins the critic files and K0's condition).
- ``results``: one stage report -> a results manifest under ``benchmarks/manifests/`` (the
  numbers every results document restates, with the report's sha256).

    uv run --no-sync python scripts/summarize_wm_critic_v2.py offline \\
        --k0 <report> <sha> --train <report> <sha> --rank <report> <sha> \\
        --output outputs/task073-offline/run-1/decision.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import wm_critic_v2 as wc  # noqa: E402


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: str, sha: str, outcome: str | tuple) -> dict:
    if sha256_file(path) != sha:
        raise wc.GuardError(f"{path} differs from its sha256")
    report = json.loads(Path(path).read_text())
    wanted = (outcome,) if isinstance(outcome, str) else outcome
    if report.get("outcome") not in wanted:
        raise wc.GuardError(f"{path} is {report.get('outcome')!r}, not one of {wanted}")
    return report


def offline_decision(k0: dict, train: dict, rank: dict) -> dict:
    """The offline row from the three reports (every gate on all three W seeds)."""
    t, r = train["stages"], rank["stages"]["rank"]["statistics"]
    order = [t["primary_w_seed"], *[s for s in wc.MODEL_SEEDS if s != t["primary_w_seed"]]]
    seeds = {}
    for i, s in enumerate(order):
        name = f"W{i}"
        # the gate functions are re-applied to the stored statistics (reviewer N4)
        stats1 = {
            (k.split("@h")[0], int(k.split("@h")[1])): v
            for k, v in t["o1"][str(s)]["statistics"].items()
        }
        o1 = wc.o1_seed_passes(stats1)
        o2 = wc.o2_passes(t["o2"][str(s)]["statistics"])
        o3 = wc.o3_passes(r[name]["O3"])
        o4 = wc.o4_passes(r[name]["O4"])
        stored = (
            t["o1"][str(s)]["passes"],
            t["o2"][str(s)]["gate"],
            {"passes": r[name]["O3"]["passes"], "void": r[name]["O3"]["void"]},
            r[name]["O4"]["passes"],
        )
        if stored != (o1["passes"], o2, o3, o4["passes"]):
            raise wc.GuardError(f"W seed {s}: a stored gate flag differs from its recomputation")
        seeds[s] = {"O1": {"passes": o1["passes"]}, "O2": o2, "O3": o3, "O4": o4}
    o5 = k0["stages"]["k0"]["o5"]
    o0 = wc.o0_passes(r["regret_incumbent"])
    if o5 != o5 | wc.o5_passes(o5["values"]):
        raise wc.GuardError("O5's stored flag differs from its recomputation")
    decision = wc.decide_offline(seeds, o5, o0)
    selection = k0["stages"]["k0"]["selection"]
    return decision | {
        "per_seed": {str(s): v for s, v in seeds.items()},
        "o5": {"passes": o5["passes"], "median": o5["median"]},
        "o0": o0,
        "shift_step": selection["shift_step"],
        "shift_cm": selection["shift_cm"],
        "critic": t["critic"],
        "primary_w_seed": t["primary_w_seed"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    off = sub.add_parser("offline")
    for name in ("k0", "train", "rank"):
        off.add_argument(f"--{name}", nargs=2, required=True, metavar=("REPORT", "SHA256"))
    off.add_argument("--output", required=True)
    res = sub.add_parser("results")
    res.add_argument("--report", nargs=2, required=True, metavar=("REPORT", "SHA256"))
    res.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    if args.mode == "offline":
        k0 = read(*args.k0, "K0-PASS")
        train = read(*args.train, "TRAIN-COMPLETE")
        rank = read(*args.rank, "RANK-COMPLETE")
        kinds = {bool(r.get("smoke")) for r in (k0, train, rank)}
        if len(kinds) != 1:
            raise wc.GuardError("the offline decision mixes smoke and real reports")
        decision = offline_decision(k0, train, rank) | {
            "smoke": kinds.pop(),
            "reports": {
                n: {"path": getattr(args, n)[0], "sha256": getattr(args, n)[1]}
                for n in ("k0", "train", "rank")
            },
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(decision, indent=1, sort_keys=True) + "\n")
        print(decision["row"])
        return 0
    report = read(
        *args.report, tuple(wc.ALL_ROWS) + ("TRAIN-COMPLETE", "RANK-COMPLETE", "CORPUS-SEALED")
    )
    stages = report["stages"]
    if report["mode"] == "k0":
        cells = {
            tuple(int(x) for x in k.split("/")): v["counts"]
            for k, v in stages["k0"]["cells"].items()
        }
        if wc.k0_select(cells) != stages["k0"]["selection"]:
            raise wc.GuardError("the recomputed K0 selection differs from the report")
    if (
        report["mode"] == "d3"
        and wc.decide_d3(stages["d3"]["counts_of_16"]) != stages["d3"]["decision"]
    ):
        raise wc.GuardError("the recomputed D3 decision differs from the report")
    manifest = {
        "protocol": wc.PROTOCOL,
        "task": wc.TASK,
        "mode": report["mode"],
        "outcome": report["outcome"],
        "report_sha256": args.report[1],
        "revision": report.get("revision"),
        "stages": {k: v for k, v in stages.items() if k not in ("reproduction",)},
        "note": "generated by scripts/summarize_wm_critic_v2.py from report.json",
    }
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
