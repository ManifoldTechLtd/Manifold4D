"""Build a grid of validation trajectories around a dynamic-mask centre.

Given a preprocessed scene dir (the layout produced by
``scripts/preprocess/`` and consumed by ``manifold4d/generate.py`` —
``frames/``, ``predictions.npz``, ``trajectory/trajectory.npz``, optional
``mask*/``), this script generates **3 trajectories** per scene (f1/f2/f3):

* ``f1``: forward horizontal scan, yaw ``-60° → +60°``, source-radius
  preserving dolly ``1.0 → 1.0``.
* ``f2``: reverse moderate-angle scan, yaw ``+30° → -30°``, fixed pitch
  ``-10°`` and dolly ``1.3 → 1.3``.
* ``f3``: moderate reverse downward scan, yaw ``+30° → -30°``, fixed
  pitch ``+10°`` and dolly ``1.0 → 0.7``.

  For all trajectories, each frame is generated around the dynamic target
  using the source camera's per-frame target distance.  The camera rotation is
  rebuilt with ``look_at`` so the target remains on the optical axis.

Outputs (per source scene) under ``--out_dir <scene_dir>``::

    <scene_dir>/shared/                                  # one copy of immutable source assets
    <scene_dir>/f1/trajectory_new/trajectory.npz         # forward horizontal scan
    <scene_dir>/f2/trajectory_new/trajectory.npz         # reverse high-angle scan
    <scene_dir>/f3/trajectory_new/trajectory.npz         # moderate reverse scan

The ``trajectory_new/trajectory.npz`` carries the minimum 3 keys the
singleview generation reads (``R_world_from_cam``, ``centers``,
``intrinsic``) plus a ``meta`` dict with bucket parameters for downstream
manifest writing.

Reuses the geometry primitives of the training-time trajectory builder
where the geometry overlaps:
  * load_dynamic_center  — mask + depth + ref c2w → 3D centre
  * load_pcd_from_depth_npz — unproject predictions.npz → world points
  * world_up derivation  — -c2w[:, :3, 1].mean() (OpenCV cam Y points down)
  * TAE pad             — append 6 copies of the last pose (Wan TAE
                          temporal compression eats 6 frames off the tail)

Usage::

    # Single scene
    python scripts/preprocess/build_val_traj.py \
        --shared_dir data/dog_short_vggt \
        --out_dir    data/val_traj/dog \
        --scene_tag  dog

    # Override dynamic centre manually (skip mask auto-detect)
    python scripts/preprocess/build_val_traj.py --shared_dir ... --out_dir ... \
        --dynamic_center 0.32,-0.18,1.45
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation




TRAJECTORY_SPECS = {
    "f1": {
        "yaw_start":   -60.0, "yaw_end":   60.0,
        "pitch_start":   0.0, "pitch_end":  0.0,
        "dolly_start":  1.00, "dolly_end": 1.00,
    },
    "f2": {
        "yaw_start":    30.0, "yaw_end": -30.0,
        "pitch_start": -10.0, "pitch_end": -10.0,
        "dolly_start":   1.30, "dolly_end": 1.30,
    },
    "f3": {
        "yaw_start":    30.0, "yaw_end": -30.0,
        "pitch_start":  10.0, "pitch_end":  10.0,
        "dolly_start":   1.00, "dolly_end": 0.70,
    },
}



# TAE temporal compression: Wan VAE's t_downscale=8 with frames_to_trim=7
# eats the last 6 frames of the trajectory at decode time.  Mirroring the
# training-time trajectory builder we append 6 copies of the last pose.
TAE_PAD = 6


# ── Geometry primitives ────────────────────────────────────────────────────


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix from a unit ``axis`` and an angle in radians.

    Falls back to identity if the axis is degenerate (norm < 1e-9).
    """
    n = float(np.linalg.norm(axis))
    if n < 1e-9 or abs(angle_rad) < 1e-12:
        return np.eye(3, dtype=np.float32)
    return Rotation.from_rotvec(
        (axis / n) * angle_rad
    ).as_matrix().astype(np.float32)


def look_at(eye: np.ndarray, target: np.ndarray,
            world_up: np.ndarray) -> np.ndarray:
    """OpenCV-style c2w rotation: cam Z = target - eye (forward), cam Y = down.

    Convention matches ``predictions.npz``'s ``extrinsic`` (cam Y points
    down, cam Z points forward — i.e. into the scene).  Returns a 3×3 R
    such that ``R @ [0, 0, 1] = forward`` and ``R @ [0, 1, 0] ≈ down``.
    """
    forward = (target - eye).astype(np.float32)
    fn = float(np.linalg.norm(forward))
    if fn < 1e-9:
        # eye coincides with target — keep a sane fallback (identity-ish).
        return np.eye(3, dtype=np.float32)
    forward /= fn

    # OpenCV: cam-down ≈ world-down (= -world_up).  Pick "down" so that
    # cam Y axis ends up pointing down in world.
    down = -world_up.astype(np.float32)
    # Gram–Schmidt: re-orthogonalise down against forward.
    down = down - forward * float(np.dot(down, forward))
    dn = float(np.linalg.norm(down))
    if dn < 1e-9:
        # forward parallel to world_up — pick an arbitrary horizontal axis.
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(helper, forward))) > 0.95:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(helper, forward).astype(np.float32)
        right /= float(np.linalg.norm(right))
        down = np.cross(forward, right).astype(np.float32)
    else:
        down /= dn
        right = np.cross(down, forward).astype(np.float32)
        rn = float(np.linalg.norm(right))
        if rn < 1e-9:
            return np.eye(3, dtype=np.float32)
        right /= rn

    # Columns are cam axes expressed in world: [X_cam, Y_cam, Z_cam].
    return np.stack([right, down, forward], axis=1).astype(np.float32)


def derive_world_up(c2ws: np.ndarray) -> np.ndarray:
    """Average ``-c2w[:, :3, 1]`` across frames → world-up unit vector.

    OpenCV cam Y points down, so ``-cam_Y`` points up; averaging across the
    source clip is robust to small per-frame jitter.
    """
    up = -c2ws[:, :3, 1].mean(axis=0)
    up = up.astype(np.float32)
    return up / float(np.linalg.norm(up))


# ── Dynamic-mask centre (port of traj_4dgs.load_dynamic_center) ───────────


def _find_mask_dir(shared_dir: Path) -> Optional[Path]:
    """First ``shared_dir/mask*/`` subdir containing ``.png`` (lex order).

    Mirrors ``manifold4d.generate._find_mask_dir`` so the
    same auto-detection rule applies at trajectory-build and generation
    time.
    """
    cands = sorted(
        d for d in shared_dir.glob("mask*")
        if d.is_dir() and any(d.glob("*.png"))
    )
    return cands[0] if cands else None


def unproject_predictions(pred: dict, max_pts: int = 200_000,
                          conf_thresh: float = 2.0,
                          frame_step: int = 4,
                          rng_seed: int = 0) -> np.ndarray:
    """Port of traj_4dgs.load_pcd_from_depth_npz, kept self-contained.

    Returns a (N, 3) float32 world-space point cloud sampled across every
    ``frame_step``-th frame, gated by ``depth_conf >= conf_thresh``.
    """
    extrinsics = pred["extrinsic"]                  # (F, 3, 4) w2c
    intrinsics = pred["intrinsic"]                  # (F, 3, 3)
    depths = pred["depth"]                          # (F, H, W[, 1])
    confs = pred["depth_conf"]                      # (F, H, W)
    if depths.ndim == 4:
        depths = depths[..., 0]
    F, H, W = depths.shape

    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    out = []
    for fi in range(0, F, frame_step):
        K = intrinsics[fi]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = extrinsics[fi]
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        dep = depths[fi]
        conf = confs[fi]
        mask = (conf >= conf_thresh) & (dep > 0.05)
        if not mask.any():
            continue

        u = us[mask]
        v = vs[mask]
        d = dep[mask].astype(np.float32)
        x = (u - K[0, 2]) / K[0, 0] * d
        y = (v - K[1, 2]) / K[1, 1] * d
        cam = np.stack([x, y, d], axis=1)
        out.append((cam - t) @ R)

    if not out:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.concatenate(out, 0).astype(np.float32)
    if len(pts) > max_pts:
        rng = np.random.default_rng(rng_seed)
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    return pts




def load_dynamic_centers(mask_dir: Path, c2ws_ref: np.ndarray,
                         depths: np.ndarray, intrinsics: np.ndarray,
                         depth_min: float = 0.05) -> Optional[np.ndarray]:
    files = sorted(glob.glob(str(mask_dir / "*.png")))
    if not files:
        print(f"[dynamic] no mask PNGs under {mask_dir}")
        return None
    if depths.ndim == 4:
        depths = depths[..., 0]
    n = min(len(files), len(c2ws_ref), depths.shape[0], len(intrinsics))
    if n == 0:
        return None
    hd, wd = depths.shape[1:]
    centers = np.full((n, 3), np.nan, dtype=np.float32)
    prev_component = None
    prev_image_center = np.array([wd * 0.5, hd * 0.5], dtype=np.float32)
    image_center = np.array([wd * 0.5, hd * 0.5], dtype=np.float32)
    image_diag = float(np.hypot(wd, hd))
    tracked = 0

    for fi in range(n):
        mask = np.array(Image.open(files[fi])) > 128
        if mask.ndim == 3:
            mask = np.any(mask, axis=-1)
        if mask.shape != (hd, wd):
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255)
                .resize((wd, hd), Image.NEAREST)) > 128
        labels, num_labels = ndimage.label(mask, structure=np.ones((3, 3)))
        candidates = []
        for label_id in range(1, num_labels + 1):
            component = labels == label_id
            area = int(component.sum())
            if area < 16:
                continue
            cy, cx = np.mean(np.where(component), axis=1)
            center = np.array([cx, cy], dtype=np.float32)
            if prev_component is None:
                distance = float(np.linalg.norm(center - image_center))
                centrality = max(0.0, 1.0 - distance / (0.5 * image_diag))
                score = float(area) * (0.5 + 0.5 * centrality)
            else:
                intersection = np.count_nonzero(component & prev_component)
                union = np.count_nonzero(component | prev_component)
                cover = intersection / max(1, min(area, int(prev_component.sum())))
                iou = intersection / max(1, union)
                distance = float(np.linalg.norm(center - prev_image_center))
                proximity = max(0.0, 1.0 - distance / (0.25 * image_diag))
                score = 0.65 * cover + 0.25 * iou + 0.10 * proximity
            candidates.append((score, area, component, center))

        if not candidates:
            continue
        score, area, component, image_center_t = max(
            candidates, key=lambda item: item[0])
        prev_component = component
        prev_image_center = image_center_t
        tracked += 1

        ys, xs = np.where(component)
        d = depths[fi, ys, xs].astype(np.float32)
        valid = np.isfinite(d) & (d > depth_min)
        if not valid.any():
            continue
        ys, xs, d = ys[valid], xs[valid], d[valid]
        K = intrinsics[fi]
        cam = np.stack(((xs - K[0, 2]) * d / K[0, 0],
                        (ys - K[1, 2]) * d / K[1, 1], d), axis=1)
        centers[fi] = np.median(
            cam @ c2ws_ref[fi, :3, :3].T + c2ws_ref[fi, :3, 3], axis=0)

    valid = np.isfinite(centers).all(axis=1)
    if not valid.any():
        print(f"[dynamic] no tracked component with valid depth in {n} frames")
        return None
    valid_idx = np.flatnonzero(valid)
    for axis in range(3):
        centers[:, axis] = np.interp(
            np.arange(n), valid_idx, centers[valid_idx, axis])
    radius = 2
    padded = np.pad(centers, ((radius, radius), (0, 0)), mode="edge")
    centers = np.stack(
        [np.median(padded[i:i + 2 * radius + 1], axis=0)
         for i in range(n)], axis=0).astype(np.float32)
    if n >= 3:
        centers = ndimage.gaussian_filter1d(
            centers, sigma=2.0, axis=0, mode="nearest").astype(np.float32)
    print(f"[dynamic] tracked {tracked}/{n} frame components, "
          f"valid centres={int(valid.sum())}, "
          f"motion={float(np.linalg.norm(centers[-1] - centers[0])):.4f}")
    return centers


def load_dynamic_center(mask_dir: Path, c2ws_ref: np.ndarray,
                        c2ws_ref_unused: np.ndarray,
                        depths: np.ndarray, intrinsics: np.ndarray,
                        n_init_frames: int = 8,
                        depth_min: float = 0.05) -> Optional[np.ndarray]:
    centers = load_dynamic_centers(
        mask_dir, c2ws_ref, depths, intrinsics, depth_min=depth_min)
    return None if centers is None else centers.mean(axis=0).astype(np.float32)


# ── Per-bucket trajectory generator ────────────────────────────────────────


def orbit_dolly_traj(
    c2ws_src: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray,
    yaw_start: float,
    yaw_end: float,
    pitch_start: float,
    pitch_end: float,
    dolly_start: float,
    dolly_end: float,
    floor_thresh: Optional[float] = None,
) -> np.ndarray:
    """Generate a (T, 4, 4) c2w trajectory by linearly sweeping (yaw,
    pitch, dolly) across the clip and applying per-frame orbit + dolly
    + look-at to the source pose.

    At each frame ``t`` (with ``s = t / (T - 1)`` ∈ [0, 1])::

        yaw(t)   = yaw_start   + (yaw_end   - yaw_start)   * s   (degrees)
        pitch(t) = pitch_start + (pitch_end - pitch_start) * s   (degrees)
        dolly(t) = dolly_start + (dolly_end - dolly_start) * s

        offset(t)    = C_src(t) - target
        distance(t)  = ‖offset(t)‖
        direction(t) = offset(t) / distance(t)
        R_yaw    = rot(world_up,                  yaw(t))
        right    = normalise(cross(world_up, R_yaw·direction(t)))
        R_pitch  = rot(right,                     pitch(t))
        dir_new  = R_pitch @ R_yaw @ direction(t)
        dist_new = distance(t) · dolly(t)
        C_new(t) = target + dir_new · dist_new
        R_new(t) = look_at(C_new(t), target, world_up)

    ``roll`` is exactly 0 by construction.  Source camera intra-clip
    motion is preserved because each frame uses its own ``C_src(t)`` as
    the starting point.

    Args:
        c2ws_src:   (T, 4, 4) source camera c2w (pre-TAE-pad).
        target:     (3,) world-space dynamic centre.
        world_up:   (3,) unit world-up.
        yaw_start, yaw_end:    yaw at t=0 and t=T-1 (degrees, positive
                               = pan to the camera's right around world-up).
        pitch_start, pitch_end: pitch at t=0 and t=T-1 (degrees; for VGGT's
                               Y-down world, positive pitch projects the
                               camera DOWNWARD along world-up — so when
                               combined with a ``floor_thresh`` clamp
                               positive pitch is the dangerous direction).
        dolly_start, dolly_end: radial scale (× source distance to target)
                               at t=0 and t=T-1.  1.0 = same distance.
        floor_thresh: world-up projection lower bound.  Any generated
                      ``C_new`` whose ``C_new · world_up`` falls below
                      this is **lifted along world-up** to exactly the
                      threshold (look-at is re-derived).  ``None``
                      disables the clamp (legacy behaviour).  Matches the
                      semantics of ``traj_4dgs.py`` ``floor_thresh =
                      min_ref_height - max_down_offset`` but applies
                      per-frame clamping rather than rejection sampling
                      (we generate the trajectory deterministically).

    Returns:
        (T, 4, 4) float32 c2w trajectory at the source clip length
        (TAE pad is added later by ``write_traj_npz``).
    """
    T = c2ws_src.shape[0]
    if T <= 1:
        s = np.zeros((T,), dtype=np.float32)
    else:
        s = (np.arange(T, dtype=np.float32) / float(T - 1))
    yaw_t = np.deg2rad(yaw_start + (yaw_end - yaw_start) * s)
    pitch_t = np.deg2rad(pitch_start + (pitch_end - pitch_start) * s)
    dolly_t = dolly_start + (dolly_end - dolly_start) * s

    target = np.asarray(target, dtype=np.float32)
    if target.ndim == 1:
        target = np.tile(target[None], (T, 1))
    if target.shape != (T, 3):
        raise ValueError(f"target must have shape (3,) or ({T}, 3), got {target.shape}")

    out = np.zeros_like(c2ws_src)
    for t in range(T):
        target_t = target[t]
        C_src = c2ws_src[t, :3, 3].astype(np.float32)
        offset = C_src - target_t
        dist = float(np.linalg.norm(offset))
        if dist < 1e-6:
            # Degenerate: source camera sits on the target.  Skip the
            # spherical move and emit identity-ish look-at.
            C_new = C_src.copy()
        else:
            direction = offset / dist
            R_yaw = rotation_about_axis(world_up, float(yaw_t[t]))
            dir_y = R_yaw @ direction
            # Pitch axis = horizontal right *relative to* the current
            # direction vector.  Cross with world_up keeps it horizontal.
            right = np.cross(world_up, dir_y).astype(np.float32)
            rn = float(np.linalg.norm(right))
            if rn < 1e-9:
                # direction parallel to world_up — no horizontal right
                # axis; skip the pitch step (yaw still applies).
                dir_new = dir_y
            else:
                right /= rn
                R_pitch = rotation_about_axis(right, float(pitch_t[t]))
                dir_new = R_pitch @ dir_y
            dist_new = dist * float(dolly_t[t])
            C_new = target_t + dir_new * dist_new

        # Floor clamp.  Lift along world-up if the camera would fall
        # below ``floor_thresh`` — matches the semantics of
        # ``traj_4dgs.py``'s floor guard but applied deterministically
        # rather than as rejection sampling.  Keeps the (yaw, dolly) sweep
        # intact; effectively reduces the *effective* pitch only on
        # below-floor frames.
        if floor_thresh is not None:
            height = float(np.dot(C_new, world_up))
            if height < floor_thresh:
                C_new = C_new + (floor_thresh - height) * world_up

        R_new = look_at(C_new, target_t, world_up)
        out[t, :3, :3] = R_new
        out[t, :3, 3] = C_new
        out[t, 3, 3] = 1.0
    return out


# ── Bucket spec ────────────────────────────────────────────────────────────


def bucket_specs() -> list[dict]:
    """Materialise the active f1/f2 trajectories."""
    specs = []
    for trajectory_id, trajectory_kwargs in TRAJECTORY_SPECS.items():
        spec = {
            "id": trajectory_id,
            "orbit_bucket": "scan",
            "dolly_bucket": "source_radius",
            "anchor": "none",
            "direction_deg": float("nan"),
        }
        spec.update(trajectory_kwargs)
        specs.append(spec)
    return specs


# ── Disk helpers ──────────────────────────────────────────────────────────


def relink(link: Path, target: Path) -> None:
    """Make ``link`` a symlink to ``target``, replacing any existing path.

    Idempotent — safe to call repeatedly when iterating buckets.
    """
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def write_traj_npz(out_path: Path, c2ws: np.ndarray, K: np.ndarray,
                   meta: dict) -> None:
    """Write ``trajectory_new/trajectory.npz`` with the minimum 3 keys
    consumed by ``build_singleview_sample`` plus a TAE-padded tail.

    ``meta`` is JSON-encoded into a 0-D string array so downstream tools
    (e.g. the manifest writer) can introspect bucket parameters from disk
    without re-deriving them.
    """
    n = c2ws.shape[0]
    # TAE pad: append TAE_PAD copies of the final pose.  Mirrors
    # traj_4dgs.py:934 and prep_dog_traj_eval's per-scene length.
    if TAE_PAD > 0:
        c2ws = np.concatenate(
            [c2ws, np.tile(c2ws[-1:], (TAE_PAD, 1, 1))], axis=0)
    R = c2ws[:, :3, :3].astype(np.float32)
    C = c2ws[:, :3, 3].astype(np.float32)
    n_pad = c2ws.shape[0]
    if K.ndim == 2:
        K_tile = np.tile(K[None], (n_pad, 1, 1)).astype(np.float32)
    else:
        K_tile = K[:n_pad].astype(np.float32) if len(K) >= n_pad else \
            np.concatenate(
                [K, np.tile(K[-1:], (n_pad - len(K), 1, 1))]
            ).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        R_world_from_cam=R,
        centers=C,
        intrinsic=K_tile,
        meta=np.array(json.dumps(meta), dtype=object),
    )


def link_shared_assets(shared_dir: Path, source_dir: Path) -> None:
    """Link immutable source assets once into ``<scene>/shared``."""
    shared_dir.mkdir(parents=True, exist_ok=True)
    candidates = ["frames", "predictions.npz", "text_embedding.pt",
                  "trajectory"]
    for d in source_dir.glob("mask*"):
        if d.is_dir():
            candidates.append(d.name)
    for name in candidates:
        src = source_dir / name
        if src.exists():
            relink(shared_dir / name, src.resolve())


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_dynamic_center(s: Optional[str]) -> Optional[np.ndarray]:
    """Parse ``--dynamic_center 'x,y,z'`` into a (3,) float32 vector."""
    if s is None:
        return None
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--dynamic_center expects 3 floats, got {s!r}")
    return np.array(parts, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared_dir", required=True,
                   help="Path to the source scene's _shared dir (contains "
                        "frames/, predictions.npz, trajectory/, mask*/).")
    p.add_argument("--out_dir", required=True,
                   help="Per-scene output dir; f1/f2 land under "
                        "<out_dir>/f1 and <out_dir>/f2.")
    p.add_argument("--scene_tag", default=None,
                   help="Optional human-readable tag (defaults to "
                        "basename of --shared_dir); only used in printed "
                        "diagnostics and meta blob.")
    p.add_argument("--mask_dir", default=None,
                   help="Override mask dir for dynamic-centre detection. "
                        "Default = first <shared_dir>/mask* subdir.")
    p.add_argument("--dynamic_center", default=None,
                   help="Manually specify the dynamic centre as 'x,y,z' in "
                        "the source world frame; bypasses mask detection.")
    p.add_argument("--depth_conf_thresh", type=float, default=2.0,
                   help="Min depth_conf to include a pixel when "
                        "unprojecting predictions.npz for the point cloud "
                        "(used for dynamic-centre detection + scene_scale).")
    p.add_argument("--predictions_npz", default=None,
                   help="Override the predictions.npz path. Default = "
                        "<shared_dir>/predictions.npz.")
    p.add_argument("--max_down_offset", type=float, default=0.0,
                   help="World-units the generated cameras may drop below "
                        "the lowest source camera (measured along world_up). "
                        "0.0 ⇒ strict 'never below source floor' (matches "
                        "traj_4dgs.py default).  Negative ⇒ disable the "
                        "floor clamp entirely.  Recommended: a small "
                        "positive fraction of scene_scale (e.g. 0.05) "
                        "for visually softer landings.")
    args = p.parse_args()

    shared_dir = Path(args.shared_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    scene_tag = args.scene_tag or shared_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shared_out_dir = out_dir / "shared"
    link_shared_assets(shared_out_dir, shared_dir)

    # ── Load source trajectory ──────────────────────────────────────────────
    traj_in = np.load(shared_dir / "trajectory" / "trajectory.npz")
    R_src = traj_in["R_world_from_cam"].astype(np.float32)   # (T, 3, 3)
    C_src = traj_in["centers"].astype(np.float32)            # (T, 3)
    T_src = R_src.shape[0]
    c2ws_src = np.zeros((T_src, 4, 4), dtype=np.float32)
    c2ws_src[:, :3, :3] = R_src
    c2ws_src[:, :3, 3] = C_src
    c2ws_src[:, 3, 3] = 1.0
    print(f"[source] {scene_tag}: {T_src} poses loaded from "
          f"{shared_dir/'trajectory'/'trajectory.npz'}")

    # K is per-frame in trajectory.npz; use the first frame as the
    # constant ``K_ref`` for dynamic-centre projection.  Downstream
    # write_traj_npz tiles it across the TAE-padded length.
    K_src = traj_in["intrinsic"].astype(np.float32)          # (T, 3, 3)
    K_ref = K_src[0]
    print(f"[source] K[0] = fx={K_ref[0,0]:.1f} fy={K_ref[1,1]:.1f} "
          f"cx={K_ref[0,2]:.1f} cy={K_ref[1,2]:.1f}")

    # ── Point cloud + scene scale + dynamic centre ──────────────────────────
    pred_path = Path(args.predictions_npz) if args.predictions_npz else (
        shared_dir / "predictions.npz")
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.npz not found at {pred_path}")
    pred = np.load(pred_path)
    pts_world = unproject_predictions(
        {k: pred[k] for k in pred.files},
        conf_thresh=args.depth_conf_thresh,
    )
    if pts_world.shape[0] == 0:
        raise RuntimeError(
            f"unprojection produced 0 points "
            f"(conf_thresh={args.depth_conf_thresh}) — try lowering "
            f"--depth_conf_thresh.")
    scene_scale = float(np.ptp(pts_world, axis=0).max())
    print(f"[pcd] {len(pts_world)} world points, scene_scale={scene_scale:.4f}")

    # Depth resolution (for dynamic-centre projection — pcd is in world
    # frame so the projection K must match the depth grid).
    if pred["depth"].ndim == 4:
        Hd, Wd = pred["depth"].shape[1:3]
    else:
        Hd, Wd = pred["depth"].shape[1:]

    dyn_centre = parse_dynamic_center(args.dynamic_center)
    if dyn_centre is None:
        mask_dir = Path(args.mask_dir) if args.mask_dir else \
            _find_mask_dir(shared_dir)
        if mask_dir is None:
            raise RuntimeError(
                f"No dynamic-mask dir found under {shared_dir} and "
                f"--dynamic_center not given; cannot determine the orbit "
                f"target.  Either symlink a mask*/ subdir or pass "
                f"--dynamic_center 'x,y,z'.")
        dyn_centres = load_dynamic_centers(
            mask_dir, c2ws_src,
            depths=pred["depth"], intrinsics=K_src)
        if dyn_centres is None:
            raise RuntimeError(
                f"Dynamic centre detection failed (no projected points hit "
                f"the mask).  Pass --dynamic_center 'x,y,z' explicitly.")
    else:
        dyn_centres = np.tile(
            dyn_centre[None], (T_src, 1)).astype(np.float32)

    world_up = derive_world_up(c2ws_src)
    print(f"[world_up] {world_up}")

    dyn_centre = dyn_centres.mean(axis=0).astype(np.float32)
    src_dist = float(
        np.linalg.norm(c2ws_src[:, :3, 3] - dyn_centres, axis=1).mean())
    print(f"[source] mean camera-to-target distance = {src_dist:.4f}")

    # ── Floor threshold (matches traj_4dgs.py:794-801 semantics) ───────────
    # height(C) = C · world_up; floor_thresh is the lowest world-up value
    # any new camera may reach.  Negative max_down_offset disables.
    if args.max_down_offset < 0:
        floor_thresh = None
        print("[floor] clamp disabled (--max_down_offset < 0)")
    else:
        src_heights = c2ws_src[:, :3, 3] @ world_up
        min_src_h = float(src_heights.min())
        floor_thresh = min_src_h - args.max_down_offset
        print(f"[floor] source camera height range "
              f"[{min_src_h:.4f}, {float(src_heights.max()):.4f}]  "
              f"max_down_offset={args.max_down_offset:.4f}  "
              f"→ floor_thresh={floor_thresh:.4f}")

    # ── Materialise each bucket ────────────────────────────────────────────
    summary = []
    for spec in bucket_specs():
        bid = spec["id"]
        bucket_dir = out_dir / bid
        c2ws_new = orbit_dolly_traj(
            c2ws_src=c2ws_src,
            target=dyn_centres,
            world_up=world_up,
            yaw_start=spec["yaw_start"], yaw_end=spec["yaw_end"],
            pitch_start=spec["pitch_start"], pitch_end=spec["pitch_end"],
            dolly_start=spec["dolly_start"], dolly_end=spec["dolly_end"],
            floor_thresh=floor_thresh,
        )
        meta = {
            "scene_tag": scene_tag,
            "bucket_id": bid,
            "orbit_bucket": spec["orbit_bucket"],
            "dolly_bucket": spec["dolly_bucket"],
            "direction_deg": spec["direction_deg"],
            "yaw_start": spec["yaw_start"], "yaw_end": spec["yaw_end"],
            "pitch_start": spec["pitch_start"], "pitch_end": spec["pitch_end"],
            "dolly_start": spec["dolly_start"], "dolly_end": spec["dolly_end"],
            "anchor": spec["anchor"],
            "dynamic_center": dyn_centre.tolist(),
            "dynamic_centers": dyn_centres.tolist(),
            "world_up": world_up.tolist(),
            "scene_scale": scene_scale,
            "src_mean_dist_to_target": src_dist,
            "T_src": T_src,
            "tae_pad": TAE_PAD,
            "floor_thresh": (float(floor_thresh)
                             if floor_thresh is not None else None),
            "max_down_offset": float(args.max_down_offset),
        }
        write_traj_npz(
            bucket_dir / "trajectory_new" / "trajectory.npz",
            c2ws_new, K_src, meta,
        )
        summary.append((bid, spec))
        dd = spec["direction_deg"]
        dir_str = f"θ={dd:5.1f}°" if not np.isnan(dd) else "θ= (anchor)"
        print(f"  [{bid:>5}] {dir_str}  "
              f"yaw {spec['yaw_start']:+6.1f}°→{spec['yaw_end']:+6.1f}°  "
              f"pitch {spec['pitch_start']:+6.1f}°→{spec['pitch_end']:+6.1f}°  "
              f"dolly {spec['dolly_start']:.2f}→{spec['dolly_end']:.2f}  "
              f"→ {bucket_dir.relative_to(out_dir.parent)}")

    print(f"\nDone. {len(summary)} buckets written under {out_dir}/")


if __name__ == "__main__":
    main()
