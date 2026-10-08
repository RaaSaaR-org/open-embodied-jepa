"""Kinematic replay of recorded MuJoCo episodes in the GR00T tutorial's Arena shelf scene.

Container-only, via ``scripts/isaac/run_isaac.sh`` with ``ISAAC_INPUT_DIR`` set to the state dump
of ``scripts/isaac_replay_lewm_cp_v2_demo.py dump`` (mounted at ``/oej/in``). Illustration only,
not evidence: **Isaac's physics never runs** (the timeline is never played) and nothing here is
an Isaac run of any controller. Every pose in a frame is MuJoCo's.

The stage holds, with no physics:

- the Arena background ``galileo_locomanip`` at Arena's pose, with the three boxes Arena's
  ``galileo_g1_static_pick_and_place`` deactivates also deactivated;
- our G1 + Dex3 USD (the MJCF conversion ``g1_29dof_with_hand-ref``, Physics variant ``none``),
  whose body prims nest like the MJCF's body tree. Each frame sets every body prim's local
  transform to MuJoCo's parent-relative pose, computed from the recorded world poses ``xpos`` /
  ``xquat``, so the posed robot is MuJoCo's to float precision whatever the joint conventions;
- Arena's apple (``apple_01``) and plate (``clay_plates``) meshes, rescaled to our sizes (apple
  diameter 5.4 cm, plate diameter 14.2 cm) and posed at MuJoCo's apple and plate body poses;
- markers as in the MuJoCo demo: a green sphere and stem at the committed aim, an orange ring of
  the scorer's 4 cm radius around the plate's final position (from step 405 on).

Our world maps to Arena's env-local frame by a translation only (``--offset``, default
(0.09, 0.08, -0.77)): our table top (z 0.74) goes to Arena's shelf top (z -0.030), our table's
front edge (x 0.13) to the shelf support's front edge (x 0.22), and y keeps Arena's robot y
(0.08). Our pelvis is fixed, so the feet hang 2.5 cm above Arena's floor (z -0.795); the table
height is matched instead of the floor.

``--probe`` renders one recorded step from every ``--camera`` candidate and exits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd", required=True)
parser.add_argument("--input", type=Path, default=Path("/oej/in"))
parser.add_argument("--clips", nargs="*", default=None, help="npz stems (default: all)")
parser.add_argument("--offset", type=float, nargs=3, default=(0.09, 0.08, -0.77))
parser.add_argument(
    "--camera",
    type=float,
    nargs=6,
    action="append",
    default=None,
    metavar=("EX", "EY", "EZ", "TX", "TY", "TZ"),
    help="eye and target in Arena env-local metres; the first one renders the replay",
)
parser.add_argument("--focal-mm", type=float, default=18.0)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--subframes", type=int, default=4, help="RTX subframes per frame")
parser.add_argument("--warmup", type=int, default=60, help="renders before the first frame")
parser.add_argument("--dome", type=float, default=0.0, help="extra dome light intensity")
parser.add_argument("--probe", type=int, default=None, metavar="STEP")
parser.add_argument("--max-frames", type=int, default=None, help="debug: frames per clip")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
args.output.mkdir(parents=True)
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402

BACKGROUND = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/galileo_locomanip/galileo_locomanip.usd"  # noqa: E501
BACKGROUND_POSE = (4.420, 1.408, -0.795)  # Arena's GalileoLocomanipBackground.initial_pose
BACKGROUND_OFF = (  # Arena's galileo_g1_static_pick_and_place deactivates these
    "BackgroundAssets/boxes/jetson_orin_06",
    "BackgroundAssets/boxes/jetson_orin_03",
    "BackgroundAssets/boxes/hesai_box_06",
)
OBJECTS = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects"
APPLE_USD = f"{OBJECTS}/objaverse/apple_01.usd"
PLATE_USD = f"{OBJECTS}/hot3d/clay_plates.usd"
APPLE_DIAMETER = 0.054  # our apple: sphere r 0.027
PLATE_DIAMETER = 0.142  # our plate: rim at r 0.067 + capsule r 0.004
PLATE_BOTTOM_BELOW_BODY = 0.006  # our plate base: cylinder half-height 0.006 below the body
SCORER_RADIUS = 0.04
DEFAULT_CAMERAS = [(0.05, -0.65, 0.65, 0.50, -0.03, -0.05)]  # over the right arm (isaac-probe-6)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    return quat_mul(quat_mul(q, np.array([0.0, *v])), quat_conj(q))[1:]


def local_pose(xpos, xquat, child: int, parent: int):
    """MuJoCo child body pose in its parent body's frame, from world poses (wxyz)."""
    qp_inv = quat_conj(xquat[parent])
    pos = quat_rotate(qp_inv, xpos[child] - xpos[parent])
    quat = quat_mul(qp_inv, xquat[child])
    return pos, quat / np.linalg.norm(quat)


class Posable:
    """A prim whose translate and orient ops are set each frame (keeps the authored op types)."""

    def __init__(self, prim):
        self.prim = prim
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        if "xformOp:translate" not in ops or "xformOp:orient" not in ops:
            xf.ClearXformOpOrder()
            ops = {
                "xformOp:translate": xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble),
                "xformOp:orient": xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble),
            }
        self.translate, self.orient = ops["xformOp:translate"], ops["xformOp:orient"]
        tname = str(self.translate.GetTypeName())
        oname = str(self.orient.GetTypeName())
        self.vec = Gf.Vec3f if tname.endswith("3f") else Gf.Vec3d
        self.quat = Gf.Quatf if oname == "quatf" else Gf.Quatd

    def set(self, pos, quat_wxyz) -> None:
        self.translate.Set(self.vec(*[float(v) for v in pos]))
        w, x, y, z = (float(v) for v in quat_wxyz)
        self.orient.Set(
            self.quat(w, Gf.Vec3d(x, y, z) if self.quat is Gf.Quatd else Gf.Vec3f(x, y, z))
        )


def xform(stage, path: str, pos=(0.0, 0.0, 0.0), scale=None):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    if scale is not None:
        x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(scale, scale, scale))
    return x


def relative_bbox(prim, ancestor):
    """The AABB of ``prim`` (its own transform included) in ``ancestor``'s frame."""
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    box = cache.ComputeRelativeBound(prim, ancestor).ComputeAlignedRange()
    return np.array(box.GetMin(), float), np.array(box.GetMax(), float)


def referenced_object(stage, path: str, url: str, *, size: float, size_axes, anchor: str):
    """``path`` is posed each frame; its child holds ``url`` rescaled so that its AABB extent over
    ``size_axes`` is ``size`` and recentred: ``anchor`` 'centre' puts the AABB centre at the
    pose, 'bottom' puts the AABB bottom centre there."""
    outer = UsdGeom.Xform.Define(stage, path)
    fit = UsdGeom.Xform.Define(stage, f"{path}/fit")  # our rescale and recentring
    asset = UsdGeom.Xform.Define(stage, f"{path}/fit/asset")  # the reference, untouched
    asset.GetPrim().GetReferences().AddReference(url)
    app.update()
    lo, hi = relative_bbox(asset.GetPrim(), fit.GetPrim())
    extent = float(np.max((hi - lo)[list(size_axes)]))
    if not np.isfinite(extent) or extent <= 0:
        raise RuntimeError(f"{url}: empty bounding box {lo} {hi}")
    s = size / extent
    centre = (lo + hi) / 2
    if anchor == "bottom":
        centre[2] = lo[2]
    fit.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(-s * centre)))
    fit.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(s, s, s))
    return Posable(outer.GetPrim()), {"url": url, "aabb_extent": (hi - lo).tolist(), "scale": s}


def sphere(stage, path, radius, rgb, pos=(0.0, 0.0, 0.0)):
    sp = UsdGeom.Sphere.Define(stage, path)
    sp.CreateRadiusAttr(radius)
    sp.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    sp.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    return sp


def build_markers(stage):
    aim = UsdGeom.Xform.Define(stage, "/World/Replay/aim")
    sphere(stage, "/World/Replay/aim/ball", 0.012, (0.1, 0.95, 0.2), (0, 0, 0.14))
    stem = UsdGeom.Cylinder.Define(stage, "/World/Replay/aim/stem")
    stem.CreateRadiusAttr(0.0025)
    stem.CreateHeightAttr(0.14)
    stem.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.95, 0.2)])
    stem.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0.07))
    ring = UsdGeom.Xform.Define(stage, "/World/Replay/ring")
    for i in range(36):
        a = 2 * np.pi * i / 36
        sphere(stage, f"/World/Replay/ring/b{i:02d}", 0.003, (1.0, 0.55, 0.0),
               (SCORER_RADIUS * np.cos(a), SCORER_RADIUS * np.sin(a), 0.0))  # fmt: skip
    return Posable(aim.GetPrim()), Posable(ring.GetPrim())


def set_visible(prim, visible: bool) -> None:
    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if visible else img.MakeInvisible()


def look_at_quat(eye, target):
    """USD camera looks down -Z with +Y up; world up is +Z."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = Gf.Matrix4d(1.0)
    m.SetRotate(Gf.Matrix3d(*right, *up, *(-fwd)))  # rows: the camera axes in world (row vectors)
    return m.ExtractRotationQuat()


def build_camera(stage, eye, target):
    cam = UsdGeom.Camera.Define(stage, "/World/ReplayCamera")
    cam.CreateFocalLengthAttr(float(args.focal_mm))
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * args.height / args.width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
    pose = Posable(cam.GetPrim())
    place_camera(pose, eye, target)
    return cam, pose


def place_camera(pose, eye, target):
    q = look_at_quat(eye, target)
    im = q.GetImaginary()
    pose.set(eye, (q.GetReal(), im[0], im[1], im[2]))


def say(msg: str) -> None:
    print(f"[replay] {msg}", flush=True)


def main() -> None:
    started = time.monotonic()
    say("building the stage")
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    bg = xform(stage, "/World/galileo_locomanip", BACKGROUND_POSE)
    bg.GetPrim().GetReferences().AddReference(BACKGROUND)
    app.update()
    deactivated = []
    for rel in BACKGROUND_OFF:
        prim = stage.GetPrimAtPath(f"/World/galileo_locomanip/{rel}")
        if prim.IsValid():
            prim.SetActive(False)
            deactivated.append(rel)
    say(f"background loaded; deactivated {deactivated}")
    if args.dome > 0:
        UsdLux.DomeLight.Define(stage, "/World/replay_dome").CreateIntensityAttr(float(args.dome))

    xform(stage, "/World/Replay", tuple(args.offset))
    robot = UsdGeom.Xform.Define(stage, "/World/Replay/Robot")
    robot.GetPrim().GetReferences().AddReference(args.usd)
    vset = robot.GetPrim().GetVariantSets().GetVariantSet("Physics")
    if vset and "none" in vset.GetVariantNames():
        vset.SetVariantSelection("none")
    app.update()
    apple, apple_info = referenced_object(stage, "/World/Replay/Apple", APPLE_USD,
                                          size=APPLE_DIAMETER, size_axes=(0, 1, 2),
                                          anchor="centre")  # fmt: skip
    plate, plate_info = referenced_object(stage, "/World/Replay/Plate", PLATE_USD,
                                          size=PLATE_DIAMETER, size_axes=(0, 1),
                                          anchor="bottom")  # fmt: skip
    aim, ring = build_markers(stage)
    say(f"robot and objects loaded: apple {apple_info}, plate {plate_info}")

    files = sorted(args.input.glob("*.npz"))
    if args.clips:
        files = [f for f in files if f.stem in set(args.clips)]
    if not files:
        raise SystemExit(f"no npz in {args.input}")
    first = np.load(files[0])
    names = [str(n) for n in first["body_names"]]
    parents = first["body_parent"].tolist()
    apple_id, plate_id = int(first["apple_body"]), int(first["plate_body"])

    # body prims: the shallowest prim of each MuJoCo body's name under the robot's Geometry
    root = stage.GetPrimAtPath("/World/Replay/Robot")
    found: dict[str, Usd.Prim] = {}
    for prim in Usd.PrimRange(root):
        n = prim.GetName()
        if n in names and n not in found and prim.IsA(UsdGeom.Xform):
            found[n] = prim
    bodies, mismatched = [], []
    for i, n in enumerate(names):
        if i in (0, apple_id, plate_id):
            continue
        prim = found.get(n)
        if prim is None:
            raise SystemExit(f"converted USD has no prim for MuJoCo body {n!r}")
        parent_name = names[parents[i]] if parents[i] != 0 else None
        usd_parent = prim.GetParent().GetName()
        if parent_name is not None and usd_parent != parent_name:
            mismatched.append((n, parent_name, usd_parent))
        bodies.append((i, parents[i], Posable(prim)))
    if mismatched:
        raise SystemExit(f"body tree differs from the MJCF's: {mismatched}")

    def pose_frame(z, row: int, t: int) -> None:
        xpos, xquat = z["xpos"][row], z["xquat"][row]
        for i, p, posable in bodies:
            if p == 0:
                posable.set(xpos[i], xquat[i])
            else:
                posable.set(*local_pose(xpos, xquat, i, p))
        apple.set(xpos[apple_id], xquat[apple_id])
        plate_pos = xpos[plate_id] - np.array([0.0, 0.0, PLATE_BOTTOM_BELOW_BODY])
        plate.set(plate_pos, xquat[plate_id])
        a = z["aim"][row]
        show_aim = bool(np.all(np.isfinite(a)))
        set_visible(aim.prim, show_aim)
        top = float(xpos[plate_id][2])
        if show_aim:
            aim.set((a[0], a[1], top), (1, 0, 0, 0))
        land = z["landing"]
        show_ring = bool(np.all(np.isfinite(land))) and t >= int(z["commit_step"])
        set_visible(ring.prim, show_ring)
        if show_ring:
            ring.set((land[0], land[1], top + 0.03), (1, 0, 0, 0))

    cameras = [tuple(c) for c in (args.camera or DEFAULT_CAMERAS)]
    cam, cam_pose = build_camera(stage, cameras[0][:3], cameras[0][3:])
    product = rep.create.render_product(str(cam.GetPath()), (args.width, args.height))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([product])

    kit = omni.kit.app.get_app()

    def render() -> np.ndarray:
        # Kit updates with the timeline stopped: Replicator's orchestrator.step waits for a
        # playing timeline (it hung in isaac-probe-4), and no physics must run anyway.
        for _ in range(int(args.subframes)):
            kit.update()
        data = np.asarray(annot.get_data())
        if data.ndim != 3:
            raise RuntimeError(f"no RGB data yet (shape {data.shape})")
        return data[..., :3].astype(np.uint8)

    z0 = first
    pose_frame(z0, 0, int(z0["t"][0]))
    say(f"{len(bodies)} bodies posed; render product ready; warming up")
    t_warm = time.monotonic()
    for k in range(int(args.warmup)):
        try:
            img = render()
            if k % 10 == 0:
                say(f"warm-up render {k}: mean {img.mean():.1f}")
        except RuntimeError as exc:
            say(f"warm-up render {k}: {exc}")
    record = {
        "what": "Isaac kinematic replay of MuJoCo episodes (illustration only; no physics)",
        "usd": args.usd,
        "background": BACKGROUND,
        "background_pose": BACKGROUND_POSE,
        "deactivated": deactivated,
        "offset": list(args.offset),
        "apple": apple_info,
        "plate": plate_info,
        "cameras": cameras,
        "focal_mm": args.focal_mm,
        "size": [args.width, args.height],
        "subframes": args.subframes,
        "warmup_s": time.monotonic() - t_warm,
        "bodies_posed": len(bodies),
        "clips": [],
    }
    frames_root = args.output / "frames"
    if args.probe is not None:
        steps = z0["t"].tolist()
        row = steps.index(int(args.probe))
        pose_frame(z0, row, int(args.probe))
        for k, c in enumerate(cameras):
            place_camera(cam_pose, c[:3], c[3:])
            for _ in range(3):
                img = render()
            frames_root.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(
                frames_root / f"probe_cam{k}_{files[0].stem}_{args.probe}.png"
            )
    else:
        for f in files:
            z = np.load(f)
            steps = z["t"].tolist()
            out = frames_root / f.stem
            out.mkdir(parents=True)
            t0 = time.monotonic()
            view_t = z["view_t"].tolist()
            if args.max_frames:
                view_t = view_t[: args.max_frames]
            for t in view_t:
                pose_frame(z, steps.index(t), t)
                img = render()
                Image.fromarray(img).save(out / f"{t:05d}.png")
            record["clips"].append({"name": f.stem, "frames": len(view_t),
                                    "seconds": time.monotonic() - t0})  # fmt: skip
            print(f"clip {f.stem}: {len(view_t)} frames", flush=True)
    record["seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(record, indent=1, default=str))
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()  # Kit's close exits the process before a traceback would print
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        sys.stdout.flush()
        app.close()
