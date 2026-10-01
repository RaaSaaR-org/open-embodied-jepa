"""Development harness: does the plate's colour limit how well its position is read?

**Development only** (``docs/experiments/apple_white_plate_dev.md``): not a gated run, not a
preregistered gate. It reuses TASK-075's runner (``scripts/run_obs_ceiling_v2.py``, imported
unmodified, with its guards, its preflight, its source checks that refuse the test roots, and its
readouts) and changes only:

- ``render``: the simulation workers are ``white_plate_dev_runtime``'s (TASK-075's worker plus
  ``plate_color.apply_plate_rgba``); the stage keeps the re-rendered 112 px onboard frames and
  fails only if a non-image array differs from the sealed root (the frames are expected to
  differ when the plate is recoloured; their difference is recorded).
- ``readouts --reference-from-views``: the reference view's frames are read from that store
  instead of the (blue) sealed corpus. Without the flag the readouts are TASK-075's own.
- ``compare``: the blue-against-white table from two readouts reports and their error files.
- ``sheet``: a contact sheet of the four views, blue against white, for one root and step.

Every report this harness writes carries ``development`` (the plate colour and this note).
**Learned Apple->Plate on the frozen benchmark is still 0 successes.**
"""

from __future__ import annotations

import os

os.environ.update(
    {
        "MKL_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "16",
    }
)

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_task075_runner", ROOT / "scripts" / "run_obs_ceiling_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load_runner()  # TASK-075's runner, unmodified
oc, lp, rg, R65 = R.oc, R.lp, R.rg, R.R65
from embodied_jepa import plate_color as pc  # noqa: E402
from embodied_jepa import white_plate_dev_runtime as wdr  # noqa: E402

PLATES = {"blue": None, "white": pc.WHITE_PLATE_RGBA}
DEV = {"plate": None, "reference_from_views": None}
NOTE = (
    "development diagnostic (owner request 2026-10-01): plate colour only; not a gated run, "
    "not a preregistered gate; docs/experiments/apple_white_plate_dev.md"
)


def _tag_reports():
    """Every report written through TASK-075's ``write_report`` carries ``development``."""
    original = R65.write_report

    def write(path, report):
        report["development"] = {
            "note": NOTE,
            "plate": DEV["plate"],
            "plate_rgba": None if PLATES.get(DEV["plate"]) is None else list(PLATES[DEV["plate"]]),
            "default_plate_rgba": list(pc.V2_PLATE_RGBA),
            "harness": "scripts/dev_white_plate.py",
            "reference_from_views": DEV["reference_from_views"],
        }
        return original(path, report)

    R65.write_report = write


class DevPool(R.Pool):
    """TASK-075's ``Pool`` (its ``map`` and G-look check) on the development worker, with the
    plate colour in the worker config."""

    def __init__(self, workers: int, config: dict):
        config = dict(config) | {"plate_rgba": PLATES[DEV["plate"]]}
        rg.BoundedPool.__init__(
            self, workers, wdr.run_task, initializer=wdr.worker_init, initargs=(config,)
        )
        self.look_reference = None


def dev_render(report, manifest, evidence, clock, args):
    """TASK-075's ``stage_render`` with the development worker; fails on any non-image array
    that differs from the sealed root, records the frame difference."""
    reader, source = R.open_source(args, False)
    report["_reader"] = reader
    out = Path(args.views)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    ids = R.read_ids(reader)
    per_root = manifest["render_smoke"]["bytes_per_root_max"] * 1.3  # + the onboard112 file
    out.parent.mkdir(parents=True, exist_ok=True)
    R.check_disk(report, out.parent, len(ids), per_root)
    report["_watch"].disk_path = str(out.parent)
    tasks = []
    for episode_id in ids:
        record = reader.manifest["episodes"][episode_id]
        meta_path = reader.corpus / "episodes" / f"{episode_id}.json"
        if R65.sha256_file(meta_path) != record["meta_sha256"]:
            raise lp.GuardError(f"G-data: {episode_id}'s metadata differs from its sealed hash")
        meta = json.loads(meta_path.read_text())
        tasks.append(
            {
                "kind": "views",
                "episode_id": episode_id,
                "meta": {
                    k: meta[k]
                    for k in ("seed", "reset", "shift", "misaim_m", "noise_level", "noise_seed")
                },
                "source_npz": str(reader.corpus / "episodes" / f"{episode_id}.npz"),
                "source_npz_sha256": record["npz_sha256"],
                "folder": str(out / "episodes"),
                "transitions": meta["transitions"],
            }
        )
        reader.decoded.add(episode_id)
    (out / "episodes").mkdir(parents=True)
    pool = R.Pool(
        R.sim_workers(), {"p3_checkpoint": R.L74.p3_checkpoint(evidence), "torch_threads": 1}
    )
    report["_pool"] = pool
    R.L74.mark_first_render(report)
    clock.check("render")
    records = pool.map(tasks, 4 * 3600.0, "views")
    by_id = {t["episode_id"]: t for t in tasks}
    bad = [
        r["episode_id"]
        for r in records
        if not all(r["arrays_equal"].values()) or r["rows"] != by_id[r["episode_id"]]["transitions"]
    ]
    plates = {json.dumps(r["plate"], sort_keys=True) for r in records}
    check = {
        "roots": len(records),
        "arrays_equal_roots": sum(all(r["arrays_equal"].values()) for r in records),
        "arrays_compared": list(wdr.ort.COMPARED_ARRAYS),
        "onboard112_vs_sealed_blue": {
            "frames_identical": sum(r["frame_check"]["identical"] for r in records),
            "frames_within_rule": sum(r["frame_check"]["within_rule"] for r in records),
            "frames_outside_rule": sum(r["frame_check"]["outside_rule"] for r in records),
            "max_level": max(r["frame_check"]["max_level"] for r in records),
            "max_pixels": max(r["frame_check"]["pixels"] for r in records),
        },
        "plate_records": [json.loads(p) for p in sorted(plates)],
        "failed_roots": bad,
        "bytes_total": sum(r["bytes"] + r["onboard112_bytes"] for r in records),
        "seconds_per_root_median": float(np.median([r["seconds"] for r in records])),
    }
    report["stages"]["render_check"] = check
    if bad:
        raise lp.GuardError(f"G-repro-render: {len(bad)} roots' non-image arrays differ")
    if len(plates) != 1:
        raise lp.GuardError(f"G-plate: the workers applied different plates {plates}")
    views_manifest = {
        "corpus": f"apple-far-shift-v2-views-dev-{DEV['plate']}-plate",
        "development": NOTE,
        "plate": json.loads(next(iter(plates))),
        "protocol": oc.PROTOCOL,
        "task": oc.TASK,
        "privileged_scripted_collector": True,
        "learned_control": False,
        "source": source,
        "views": list(oc.VIEWS),
        "view_specs": oc.VIEW_SPECS,
        "hidden_kinds": list(oc.HIDDEN_KINDS),
        "steps": {"render": list(oc.RENDER_STEPS), "hidden": list(oc.HIDDEN_STEPS)},
        "splits": {s: list(reader.manifest["splits"][s]) for s in oc.SOURCE_CORPUS["read_splits"]},
        "episodes": {
            r["episode_id"]: {
                "npz_sha256": r["npz_sha256"],
                "onboard112_npz_sha256": r["onboard112_npz_sha256"],
                "bytes": r["bytes"] + r["onboard112_bytes"],
            }
            for r in records
        },
        "provenance": {"revision": report["revision"], "frozen_sha256": oc.frozen_sha256()},
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(views_manifest, indent=1, sort_keys=True) + "\n")
    return {
        "outcome": "VIEWS-SEALED",
        "views": {
            "path": str(out),
            "manifest_sha256": R65.sha256_file(path),
            "source": source,
            "per_root": [{k: v for k, v in r.items() if k != "arrays_equal"} for r in records],
        },
    }


def view_frames_from_store(view, root, source, views):
    """TASK-075's ``view_frames``, with the reference view's visible frames read from the
    development store (its own re-render) instead of the sealed corpus."""
    if view != oc.REFERENCE_VIEW:
        return ORIGINAL_VIEW_FRAMES(view, root, source, views)
    want_vis, want_hid = R.expected_steps(root)
    keys = ["steps", "hidden_steps", *[f"{view}__{k}" for k in oc.HIDDEN_KINDS]]
    got = views.arrays(root["id"], keys)
    steps = [int(s) for s in got["steps"]]
    hidden_steps = [int(s) for s in got["hidden_steps"]]
    if [s for s in steps if s <= root["hi"]] != want_vis or [
        s for s in hidden_steps if s <= root["hi"]
    ] != want_hid:
        raise lp.GuardError(f"G-data: {root['id']}'s rendered steps are not the frozen ones")
    path = wdr.ref_path(views.path / "episodes", root["id"])
    if R65.sha256_file(path) != views.manifest["episodes"][root["id"]]["onboard112_npz_sha256"]:
        raise lp.GuardError(f"G-data: {root['id']}'s onboard112 frames differ from their hash")
    with np.load(path) as data:
        frames = data["onboard112"]
    if len(frames) != len(steps):
        raise lp.GuardError(f"G-data: {root['id']}'s onboard112 frames are not one per step")
    visible = frames[[steps.index(s) for s in want_vis]].copy()
    take_h = [hidden_steps.index(s) for s in want_hid]
    hidden = {k: got[f"{view}__{k}"][take_h] for k in oc.HIDDEN_KINDS}
    return visible, hidden


ORIGINAL_VIEW_FRAMES = R.view_frames


# ----- compare and sheet (light, offline) -------------------------------------------------------
def _stats(report_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    folder = report_path.parent
    out = {}
    for view in oc.VIEWS:
        st = report["stages"]["statistics"][view]
        sha = report["stages"]["errors_sha256"][view]
        path = folder / f"errors_{view}.npz"
        if R65.sha256_file(path) != sha:
            raise lp.GuardError(f"{path} differs from its recorded sha256")
        with np.load(path) as data:
            clusters = data["clusters"]
            errors = {k[3:]: data[k] for k in data.files if k.startswith("e__")}
        row = {}
        for key in ("r_off", "r_full", "r_pix"):
            e = errors[key]
            median = st["median_ci"][key]
            if not np.isclose(oc.cluster_median_ci(e, clusters)["median"], median["median"]):
                raise lp.GuardError(f"{view} {key}: the errors file disagrees with the report")
            row[key] = {
                "median": median["median"],
                "median_upper": median["ci95"][1],
                "p87_5": oc.cluster_quantile_ci(e, clusters, oc.REPORTED_PERCENTILE),
                "predicted_successes_of_32": oc.cluster_mean_ci(
                    oc.tau_curve_fraction(e) * oc.TAU["resets"], clusters
                ),
                "clock_ratio_upper": st["ratio_ci"][f"{key}/clock"]["ci95"][1],
                "clock_ratio": st["ratio_ci"][f"{key}/clock"]["ratio"],
                "plate_hidden_median": st["median_ci"][f"{key}_plate_hidden"]["median"],
            }
        reported = st["reported"]
        if not np.isclose(
            row["r_off"]["p87_5"]["value"], reported["r_off_percentile"]["value"]
        ) or not np.isclose(
            row["r_off"]["predicted_successes_of_32"]["mean"],
            reported["r_off_tau_curve_predicted_successes"]["mean"],
        ):
            raise lp.GuardError(f"{view}: recomputed r_off statistics disagree with the report")
        # TASK-075's reported-only pooled-token readouts of the plate's and the apple's own
        # world xy (the same features as R_off, another target): plate legibility is the part a
        # plate colour can change.
        for key in ("plate_target", "apple_target"):
            median = st["median_ci"][key]
            if not np.isclose(
                oc.cluster_median_ci(errors[key], clusters)["median"], median["median"]
            ):
                raise lp.GuardError(f"{view} {key}: the errors file disagrees with the report")
            row[key] = {
                "median": median["median"],
                "median_upper": median["ci95"][1],
                "p87_5": oc.cluster_quantile_ci(errors[key], clusters, oc.REPORTED_PERCENTILE),
            }
        row["clock_median"] = st["median_ci"]["clock"]["median"]
        row["windows"], row["roots"] = st["windows"], st["roots"]
        out[view] = row
    return {
        "row": report["stages"]["decision"]["row"],
        "revision": report.get("revision"),
        "views": out,
    }


def compare(args) -> int:
    blue, white = Path(args.blue), Path(args.white)
    result = {
        "development": NOTE,
        "blue": {"report": str(blue), "sha256": R65.sha256_file(blue)} | _stats(blue),
        "white": {"report": str(white), "sha256": R65.sha256_file(white)} | _stats(white),
    }
    if args.reproduced:
        rep = Path(args.reproduced)
        got = json.loads(rep.read_text())["stages"]["statistics"]
        want = json.loads(blue.read_text())["stages"]["statistics"]
        result["blue_reproduction"] = {
            "report": str(rep),
            "sha256": R65.sha256_file(rep),
            "statistics_identical": got == want,
        }
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(
        f"{'view':12} {'readout':7} | {'blue c_up':>9} {'white c_up':>10} | {'blue p87.5':>10} "
        f"{'white p87.5':>11} | {'blue pred':>9} {'white pred':>10} | {'blue clk':>8} "
        f"{'white clk':>9}"
    )
    for view in oc.VIEWS:
        for key in ("r_off", "r_full", "r_pix"):
            b, w = result["blue"]["views"][view][key], result["white"]["views"][view][key]
            print(
                f"{view:12} {key:7} | {b['median_upper']:9.2f} {w['median_upper']:10.2f} | "
                f"{b['p87_5']['value']:10.2f} {w['p87_5']['value']:11.2f} | "
                f"{b['predicted_successes_of_32']['mean']:9.1f} "
                f"{w['predicted_successes_of_32']['mean']:10.1f} | "
                f"{b['clock_ratio_upper']:8.3f} {w['clock_ratio_upper']:9.3f}"
            )
        for key in ("plate_target", "apple_target"):
            b, w = result["blue"]["views"][view][key], result["white"]["views"][view][key]
            print(
                f"{view:12} {key[:5]:7} | {b['median_upper']:9.2f} {w['median_upper']:10.2f} | "
                f"{b['p87_5']['value']:10.2f} {w['p87_5']['value']:11.2f} | (median "
                f"{b['median']:.3f} -> {w['median']:.3f})"
            )
    print("written", out, R65.sha256_file(out))
    return 0


PAIRED_KEYS = ("r_off", "r_full", "r_pix", "plate_target", "apple_target")


def paired(args) -> int:
    """``paired-white-over-blue.json``: per view and readout, ``oc.cluster_median_ratio(white,
    blue, clusters)`` over the same windows, from the two readouts runs' error files (each
    checked against its report's recorded sha256)."""
    folders = {"blue": Path(args.blue).parent, "white": Path(args.white).parent}
    shas = {
        k: json.loads(Path(v).read_text())["stages"]["errors_sha256"]
        for k, v in (("blue", args.blue), ("white", args.white))
    }
    out = {}
    for view in oc.VIEWS:
        data = {}
        for name, folder in folders.items():
            path = folder / f"errors_{view}.npz"
            if R65.sha256_file(path) != shas[name][view]:
                raise lp.GuardError(f"{path} differs from its recorded sha256")
            with np.load(path) as d:
                data[name] = {k: d[k] for k in d.files}
        b, w = data["blue"], data["white"]
        if not (
            np.array_equal(b["clusters"], w["clusters"])
            and np.array_equal(b["windows"], w["windows"])
        ):
            raise lp.GuardError(f"{view}: the two runs' windows differ")
        for key in PAIRED_KEYS:
            r = oc.cluster_median_ratio(w["e__" + key], b["e__" + key], b["clusters"])
            out[f"{view}/{key}"] = r
            print(
                f"{view:12} {key:13} white/blue {r['ratio']:.4f} "
                f"[{r['ci95'][0]:.5f}, {r['ci95'][1]:.5f}]"
            )
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(out, indent=1))
    print("written", path, R65.sha256_file(path))
    return 0


def sheet(args) -> int:
    from PIL import Image, ImageDraw

    blue_views, white_views = Path(args.blue_views), Path(args.white_views)
    root, step = args.root, args.step
    with np.load(blue_views / "episodes" / f"{root}.npz") as b:
        steps = [int(s) for s in b["steps"]]
        blue = {v: b[v][steps.index(step)] for v in oc.RENDERED_VIEWS}
    with np.load(Path(args.source) / "episodes" / f"{root}.npz") as src:
        blue["onboard112"] = src["frames"][step]
    with np.load(white_views / "episodes" / f"{root}.npz") as w:
        wsteps = [int(s) for s in w["steps"]]
        white = {v: w[v][wsteps.index(step)] for v in oc.RENDERED_VIEWS}
    with np.load(wdr.ref_path(white_views / "episodes", root)) as w:
        white["onboard112"] = w["onboard112"][wsteps.index(step)]
    cell, pad, head = 224, 6, 18
    canvas = Image.new(
        "RGB", (len(oc.VIEWS) * (cell + pad) + pad, 2 * (cell + pad + head) + pad), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for r, (name, frames) in enumerate(
        (("blue (v2 default)", blue), ("white 0.92 0.92 0.90", white))
    ):
        for c, view in enumerate(oc.VIEWS):
            x, y = pad + c * (cell + pad), pad + r * (cell + pad + head)
            image = Image.fromarray(np.asarray(frames[view], np.uint8)).resize((cell, cell))
            canvas.paste(image, (x, y + head))
            draw.text((x, y + 2), f"{name} | {view} ({frames[view].shape[0]} px)", fill="black")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    canvas.save(out)
    print("written", out, f"root {root} step {step}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("render", "readouts"):
        p = sub.add_parser(mode)
        p.add_argument("--plate", choices=sorted(PLATES), required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--evidence", required=True)
        p.add_argument("--source", required=True)
        p.add_argument("--views", required=True)
        if mode == "readouts":
            p.add_argument("--render-report", required=True)
            p.add_argument("--render-sha256", required=True)
            p.add_argument("--reference-from-views", action="store_true")
    p = sub.add_parser("paired")
    for name in ("blue", "white", "output"):
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("compare")
    for name in ("blue", "white", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--reproduced")
    p = sub.add_parser("sheet")
    for name in ("blue-views", "white-views", "source", "root", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--step", type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode == "paired":
        return paired(args)
    if args.mode == "compare":
        return compare(args)
    if args.mode == "sheet":
        return sheet(args)
    DEV["plate"] = args.plate
    DEV["reference_from_views"] = bool(getattr(args, "reference_from_views", False))
    _tag_reports()
    R.Pool = DevPool
    R.STAGES["render"] = dev_render
    if args.mode == "readouts" and args.reference_from_views:
        R.view_frames = view_frames_from_store
    run_args = argparse.Namespace(
        mode=args.mode,
        output=args.output,
        evidence=args.evidence,
        source=args.source,
        source_sha256=None,
        views=args.views,
        render_report=getattr(args, "render_report", None),
        render_sha256=getattr(args, "render_sha256", None),
        smoke=False,
        scale=False,
        workers=None,
    )
    started = time.monotonic()
    report = R.run(run_args)
    R.log(f"development {args.mode} ({args.plate}): {time.monotonic() - started:.0f} s")
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
