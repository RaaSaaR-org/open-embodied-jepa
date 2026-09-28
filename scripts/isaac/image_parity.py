"""The declared onboard-image parity metric (``configs/isaac/onboard_camera_v1.json``).

Host-side, NumPy only (the MuJoCo segmentation helper imports ``mujoco`` lazily). Development
tooling for TASK-025: it compares an Isaac ``onboard_rgb`` frame with MuJoCo's frame of the same
joint state, photometrically (MAD, PSNR, luma SSIM) and geometrically (per-class IoU of
segmentation masks, apple/plate centroid distance). The classes are the robot, the table, the
apple, the plate, and the floor together with the background.
"""

from __future__ import annotations

import numpy as np

CLASSES = ("floor_or_background", "robot", "table", "apple", "plate")


def mujoco_classes(sim, camera: str = "onboard_rgb") -> np.ndarray:
    """Class index per pixel from MuJoCo's segmentation render of ``camera``."""
    import mujoco

    model = sim.model
    renderer = mujoco.Renderer(model, height=sim.height, width=sim.width)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera=camera)
        seg = renderer.render().copy()
    finally:
        renderer.close()
    out = np.zeros(seg.shape[:2], dtype=np.int64)
    geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    for g in np.unique(seg[..., 0][geom]):
        name = model.geom(int(g)).name
        body = model.body(int(model.geom_bodyid[int(g)])).name
        if name == "table":
            cls = 2
        elif body == "apple":
            cls = 3
        elif body == "plate":
            cls = 4
        elif body == "world":
            cls = 0
        else:
            cls = 1
        out[geom & (seg[..., 0] == g)] = cls
    return out


def isaac_classes(seg: np.ndarray, labels: dict) -> np.ndarray:
    """Class index per pixel from Isaac's instance-id segmentation and its id -> prim map."""
    out = np.zeros(seg.shape, dtype=np.int64)
    for key, path in labels.items():
        path = str(path)
        if path.startswith("/World/Robot"):
            cls = 1
        elif path.startswith("/World/table"):
            cls = 2
        elif path.startswith("/World/Apple"):
            cls = 3
        elif path.startswith("/World/Plate"):
            cls = 4
        else:
            cls = 0
        out[seg == int(key)] = cls
    return out


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])


def ssim_luma(a: np.ndarray, b: np.ndarray, window: int = 7) -> float:
    x, y = _luma(a), _luma(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    wx = np.lib.stride_tricks.sliding_window_view(x, (window, window))
    wy = np.lib.stride_tricks.sliding_window_view(y, (window, window))
    mx, my = wx.mean(axis=(-1, -2)), wy.mean(axis=(-1, -2))
    vx, vy = wx.var(axis=(-1, -2)), wy.var(axis=(-1, -2))
    cov = (wx * wy).mean(axis=(-1, -2)) - mx * my
    s = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
    return float(s.mean())


def compare(isaac_rgb, mujoco_rgb, isaac_cls=None, mujoco_cls=None) -> dict:
    a = np.asarray(isaac_rgb, dtype=np.uint8)
    b = np.asarray(mujoco_rgb, dtype=np.uint8)
    diff = a.astype(float) - b.astype(float)
    mse = float((diff**2).mean())
    out = {
        "mad": float(np.abs(diff).mean()),
        "psnr_db": float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf"),
        "ssim_luma": ssim_luma(a, b),
    }
    if isaac_cls is not None and mujoco_cls is not None:
        iou, frac = {}, {}
        for i, name in enumerate(CLASSES):
            p, q = isaac_cls == i, mujoco_cls == i
            union = int((p | q).sum())
            iou[name] = float((p & q).sum() / union) if union else None
            frac[name] = {"isaac": float(p.mean()), "mujoco": float(q.mean())}
        out["mask_iou"] = iou
        out["mask_fraction"] = frac
        cent = {}
        for i, name in ((3, "apple"), (4, "plate")):
            p, q = np.argwhere(isaac_cls == i), np.argwhere(mujoco_cls == i)
            cent[name] = float(np.linalg.norm(p.mean(0) - q.mean(0))) if len(p) and len(q) else None
        out["centroid_px"] = cent
    return out


def verdict(rows: dict, spec: dict) -> dict:
    """Apply the manifest's declared bars to the held-out reads."""
    g, ph = spec["thresholds"]["geometry"], spec["thresholds"]["photometric"]
    held = [rows[k] for k in spec["held_out_reads"] if k in rows]
    geometry_fail, photo_fail = [], []
    for k, r in zip(spec["held_out_reads"], held, strict=False):
        iou = r.get("mask_iou", {})
        for cls, bar in (
            ("robot", g["robot_iou_min"]),
            ("table", g["table_iou_min"]),
            ("apple", g["apple_iou_min"]),
            ("plate", g["plate_iou_min"]),
        ):
            if iou.get(cls) is None or iou[cls] < bar:
                geometry_fail.append(f"read {k}: {cls} IoU {iou.get(cls)} < {bar}")
        for obj, dist in r.get("centroid_px", {}).items():
            if dist is None or dist > g["centroid_px_max"]:
                geometry_fail.append(f"read {k}: {obj} centroid {dist} px > {g['centroid_px_max']}")
        if r["mad"] > ph["mad_max"]:
            photo_fail.append(f"read {k}: MAD {r['mad']:.1f} > {ph['mad_max']}")
        if r["ssim_luma"] < ph["ssim_luma_min"]:
            photo_fail.append(f"read {k}: SSIM {r['ssim_luma']:.3f} < {ph['ssim_luma_min']}")
    return {
        "held_out_reads": spec["held_out_reads"],
        "geometry": "PASS" if held and not geometry_fail else "FAIL",
        "geometry_failures": geometry_fail,
        "photometric": "PASS" if held and not photo_fail else "FAIL",
        "photometric_failures": photo_fail,
    }
