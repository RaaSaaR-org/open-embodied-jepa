"""TASK-090: Stage C, the data gate, and its Stage 0 checks (protocol
docs/experiments/grasp_play_corpus_v1.md §4, §13, §14). CPU only; reads sidecars, never frames.

* ``labels --corpus DIR --split S --output FILE``: the frozen label script -- one row per episode
  of the split (lift windows, grasp-contact windows, closed-hand-near windows, lift states, rises,
  per object; mode, skills, stop reason, the collector's lift checks). ``--split all`` labels every
  stored episode (the debug collections only). Every sidecar is checked against its manifest
  sha256.
* ``prototype --dev-output FILE --labels FILE --output FILE``: §13's check of the frozen label
  script against the re-run development script on ``play-v1`` train (the script's own convention
  reproduced exactly, then the protocol's).
* ``debug --v1 FILE --grasp FILE --v0 FILE --output FILE``: Stage 0's debug readings (§4).
* ``stage-c --v1 FILE --v2g FILE --v1p FILE --output FILE``: Stage C's row (§4).
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import grasp_corpus_wm as gc  # noqa: E402
from embodied_jepa import grounded_wm_data as gd  # noqa: E402


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def base() -> dict:
    return {
        "task": gc.TASK,
        "protocol": gc.PROTOCOL,
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "argv": sys.argv,
    }


def episodes_of(corpus: Path, split: str) -> list[tuple[Path, dict]]:
    """(shard dir, manifest row) of the split's episodes (``all``: every stored episode)."""
    out = []
    for shard in sorted(p for p in corpus.iterdir() if p.name.startswith("shard-")):
        if (shard / "EXCLUDED.json").exists():
            continue
        manifest = json.loads((shard / "meta" / "jepa_manifest.json").read_text())
        rows = {r["episode_id"]: r for r in manifest["episodes"]}
        if split == "all":
            ids = sorted(rows)
        else:
            ids = sorted(
                json.loads((shard / "meta" / "jepa_splits.json").read_text())["episodes"][split]
            )
        out += [(shard, rows[i]) for i in ids]
    return out


def _label(job):
    shard, row, dev = job
    meta = row["metadata"]
    path = Path(shard) / meta["sidecar"]
    if gd.sha256_file(path) != meta["sidecar_sha256"]:
        raise RuntimeError(f"sidecar hash mismatch: {path}")
    with np.load(path) as d:
        side = {k: d[k] for k in d.files}
    out = gc.episode_labels(side, meta["layout"], meta)
    out |= {"episode_id": row["episode_id"], "seed": int(meta["seed"]), "shard": Path(shard).name}
    if dev:
        out["dev"] = gc.dev_convention_counts(side)
    return out


def cmd_labels(args) -> int:
    report = base()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    corpus = Path(args.corpus)
    eps = episodes_of(corpus, args.split)
    if args.split == "test":
        raise SystemExit("test splits are never read (§2.2)")
    jobs = [(str(s), r, args.dev) for s, r in eps]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(_label, jobs, chunksize=16)
    report |= {
        "corpus": str(corpus),
        "corpus_json_sha256": gd.sha256_file(corpus / "corpus.json")
        if (corpus / "corpus.json").exists()
        else None,
        "split": args.split,
        "episodes": len(rows),
        "share": gc.share(rows),
        "rows": rows,
        "finished_utc": utc(),
    }
    gc.write_json(out, report)
    print(json.dumps({"episodes": len(rows), "share": report["share"]}))
    return 0


def cmd_prototype(args) -> int:
    """§13: the re-run development script's console output (its JSON line) against the frozen
    label script's counts on the same episodes."""
    report = base()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    dev_line = next(
        line for line in Path(args.dev_output).read_text().splitlines() if line.startswith("{")
    )
    dev = json.loads(dev_line)
    labels = json.loads(Path(args.labels).read_text())
    rows = labels["rows"]
    mine = {
        "eps": len(rows),
        "steps": sum(r["dev"]["steps"] for r in rows),
        "lift_any": sum(r["dev"]["lift_steps"] for r in rows),
        "win": sum(r["dev"]["windows"] for r in rows),
        "w_lift_any": sum(r["dev"]["lift_windows"] for r in rows),
    }
    match = {k: bool(mine[k] == dev[k]) for k in mine}
    report |= {
        "dev_output_sha256": gd.sha256_file(args.dev_output),
        "labels_sha256": gd.sha256_file(args.labels),
        "development_script": {k: dev[k] for k in mine},
        "label_script_dev_convention": mine,
        "match": match,
        "all_match": all(match.values()),
        "protocol_convention": {
            "windows": sum(r["windows"] for r in rows),
            "lift_windows": sum(r["lift_windows"] for r in rows),
            "share": gc.share(rows),
            "development_share": dev["w_lift_any"] / dev["win"],
        },
        "finished_utc": utc(),
    }
    gc.write_json(out, report)
    print(json.dumps({"all_match": report["all_match"], **report["protocol_convention"]}))
    return 0 if report["all_match"] else 1


def stop_joint(reason: str) -> str:
    """The joint that tripped the velocity stop ("... [joint speed]"), else the reason."""
    return reason.split("[")[-1].split(" ")[0] if "[" in reason else reason


def reported(rows) -> dict:
    """§4's reported quantities for one set of label rows."""
    n = len(rows)
    win = sum(r["windows"] for r in rows)
    modes = collections.Counter(r.get("mode") for r in rows)
    skills = collections.Counter(s for r in rows for s in r.get("skills", []))
    stops = collections.Counter(
        (
            r["stop_reason"].split("[")[-1].split(" ")[0]
            if "[" in r["stop_reason"]
            else r["stop_reason"]
        )
        for r in rows
        if r.get("stop_reason")
    )
    checks = [c for r in rows for c in r.get("pick_checks", [])]
    retries = [c for c in checks if c["attempt"] == 1]
    rises = [x for r in rows for x in r["max_rise"] if x is not None]
    names = ("apple", "cube", "banana", "can")
    states = sum(r["lift_states"] for r in rows)
    return {
        "episodes": n,
        "windows": win,
        "lift_window_share": gc.share(rows),
        "lift_episodes": sum(any(r["per_object_lifted"]) for r in rows),
        "per_object_lift_episodes": {
            k: sum(r["per_object_lifted"][i] for r in rows) for i, k in enumerate(names)
        },
        "per_object_lift_windows": {
            k: sum(r["per_object_lift_windows"][i] for r in rows) for i, k in enumerate(names)
        },
        "contact_window_share": sum(r["contact_windows"] for r in rows) / win,
        "closed_near_window_share": sum(r["closed_near_windows"] for r in rows) / win,
        "max_rise_quantiles": dict(
            zip(
                ("p10", "p50", "p90", "p99"),
                [float(x) for x in np.percentile(rises, [10, 50, 90, 99])],
                strict=True,
            )
        )
        if rises
        else None,
        "lift_states_ge5_share": (sum(r["lift_states_ge5"] for r in rows) / states)
        if states
        else None,
        "early_ends": sum(bool(r.get("stop_reason")) for r in rows),
        "early_end_joints": dict(stops.most_common()),
        "modes": dict(modes),
        "skills": dict(skills),
        "lift_checks": len(checks),
        "first_attempt_lifted": sum(c["lifted"] for c in checks if c["attempt"] == 0),
        "retries": len(retries),
        "retry_lifted": sum(c["lifted"] for c in retries),
        "retry_lift_rate": (sum(c["lifted"] for c in retries) / len(retries)) if retries else None,
    }


def _rows(path, *, split=None):
    rep = json.loads(Path(path).read_text())
    if split is not None and rep["split"] != split:
        raise SystemExit(f"{path} is the {rep['split']} split, not {split}")
    return rep, rep["rows"]


def check_v1(rep) -> None:
    """The play-v1 labels are TASK-085's sealed corpus (corpus.json pinned) and its 2 874 train
    episodes."""
    from embodied_jepa import grounded_wm as gw

    if rep["corpus_json_sha256"] != gw.CORPUS_JSON_SHA256 or len(rep["rows"]) != 2874:
        raise SystemExit("the play-v1 labels are not of TASK-085's sealed train split")


def cmd_debug(args) -> int:
    report = base()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    v1rep, v1 = _rows(args.v1, split="train")
    grep, grasp = _rows(args.grasp, split="all")
    v0rep, v0 = _rows(args.v0, split="all")
    from embodied_jepa import play_corpus_grasp as pg

    check_v1(v1rep)
    for r in grasp:
        pg.check_collection_seed("grasp", r["seed"], debug=True)
    for r in v0:
        pg.check_collection_seed("v0", r["seed"], debug=True)
    report |= {
        "inputs": {
            "v1": gd.sha256_file(args.v1),
            "grasp": gd.sha256_file(args.grasp),
            "v0": gd.sha256_file(args.v0),
        },
        "ratios": gc.debug_ratios(v1, grasp, v0),
        "reported": {
            "play-v1-train": reported(v1),
            "debug-grasp": reported(grasp),
            "debug-v0": reported(v0),
        },
        "finished_utc": utc(),
    }
    gc.write_json(out, report)
    print(json.dumps(report["ratios"], indent=1))
    return 0


def cmd_project(args) -> int:
    """Reported only: Stage C at full size by resampling (the declared and the revised bars)."""
    report = base()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    v1rep, v1 = _rows(args.v1, split="train")
    _, grasp = _rows(args.grasp, split="all")
    check_v1(v1rep)
    report["inputs"] = {"v1": gd.sha256_file(args.v1), "grasp": gd.sha256_file(args.grasp)}
    report["declared"] = gc.project_stage_c(
        v1, grasp, bars={"C1": 2.5, "C2": 1.75, "C3": gc.C3_BAND}
    )
    report["frozen"] = gc.project_stage_c(v1, grasp)
    report["finished_utc"] = utc()
    gc.write_json(out, report)
    print(json.dumps({k: report[k] for k in ("declared", "frozen")}, indent=1))
    return 0


def cmd_stage_c(args) -> int:
    report = base()
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    v1rep, v1 = _rows(args.v1, split="train")
    _, v2g = _rows(args.v2g, split="train")
    _, v1p = _rows(args.v1p, split="train")
    from embodied_jepa import play_corpus_grasp as pg

    check_v1(v1rep)
    if {r["seed"] for r in v2g} - set(pg.corpus_seeds("play-v2g")):
        raise SystemExit("play-v2g labels hold seeds outside 920000-923199")
    if {r["seed"] for r in v1p} - set(pg.corpus_seeds("play-v1p")):
        raise SystemExit("play-v1p labels hold seeds outside 910000-913199")
    report |= {
        "inputs": {
            "v1": gd.sha256_file(args.v1),
            "v2g": gd.sha256_file(args.v2g),
            "v1p": gd.sha256_file(args.v1p),
        },
        "stage_c": gc.stage_c(v1, v2g, v1p),
        "reported": {
            "play-v1-train": reported(v1),
            "play-v2g-added-train": reported(v2g),
            "play-v1p-added-train": reported(v1p),
        },
        "finished_utc": utc(),
    }
    gc.write_json(out, report)
    print(json.dumps(report["stage_c"], indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    la = sub.add_parser("labels")
    la.add_argument("--corpus", required=True)
    la.add_argument("--split", required=True, choices=("train", "val", "all"))
    la.add_argument("--output", required=True)
    la.add_argument("--workers", type=int, default=8)
    la.add_argument("--dev", action="store_true", help="also the development convention (§13)")
    pr = sub.add_parser("prototype")
    pr.add_argument("--dev-output", required=True)
    pr.add_argument("--labels", required=True)
    pr.add_argument("--output", required=True)
    de = sub.add_parser("debug")
    de.add_argument("--v1", required=True)
    de.add_argument("--grasp", required=True)
    de.add_argument("--v0", required=True)
    de.add_argument("--output", required=True)
    pj = sub.add_parser("project")
    pj.add_argument("--v1", required=True)
    pj.add_argument("--grasp", required=True)
    pj.add_argument("--output", required=True)
    sc = sub.add_parser("stage-c")
    sc.add_argument("--v1", required=True)
    sc.add_argument("--v2g", required=True)
    sc.add_argument("--v1p", required=True)
    sc.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return {
        "labels": cmd_labels,
        "prototype": cmd_prototype,
        "debug": cmd_debug,
        "project": cmd_project,
        "stage-c": cmd_stage_c,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
