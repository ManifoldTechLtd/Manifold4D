"""Foreground-occlusion ("leaked background") masking for the render.

When a moving subject is unprojected from the SOURCE view and the multi-frame
point cloud is rasterised into a novel TARGET view, the camera baseline reveals
the region *behind* the subject.  The subject is only a single-layer front
shell (one depth per source pixel), so that disoccluded region is not covered
by the subject's own points — instead it shows **leaked background** accumulated
from other frames that, in the source view, was hidden behind the subject.

This module masks that leak with a *source-occluder occlusion cone*:

  1. Fit the subject's **bounding ELLIPSE** as the 2D occluder region — the
     principal-axis minimum-enclosing ellipse of the dilated source silhouette
     (the dynamic / motion mask, morphologically dilated at load time); see
     :func:`fit_silhouette_ellipse`.
  2. Take the subject's **most-forward point** (min source depth) as the
     cone's front face and extend it to infinity behind the subject.
  3. Mask every covered TARGET pixel that reprojects **inside** the ellipse
     AND is **deeper** than the front face — it was occluded by the subject in
     source, so any content there is leaked background → hole.

Occluder shape: the subject's bounding ellipse is used rather than the raw
silhouette.  The ellipse closes the concave gaps of a spread / limbed subject,
so leaked background hiding in those gaps is masked too — at the cost of also
masking valid background that falls inside the ellipse beside the subject.  The
silhouette-only alternative (no over-masking, follows limbs exactly) remains
available: :func:`silhouette_cone_leak_mask` accepts ANY boolean occluder
region, so passing the raw silhouette instead of the fitted ellipse recovers
that behaviour.

The subject's own pixels are preserved by the caller via a dynamic-coverage
exemption ("don't filter the dynamic points"): the cone front is the subject's
nearest point, so without the exemption it would eat the subject's own body.

Both the offline PoC (``scripts/viz_conservative_mask_valtraj.py``) and the
training point-cloud renderer (``manifold4d/rendering/gpu_renderer.py``) call into here so
the geometry is defined in exactly one place.
"""

from typing import Optional

import numpy as np
import torch


@torch.no_grad()
def subject_front_depth(subj_pts_world: torch.Tensor, c2w_src: torch.Tensor,
                        *, depth_pct: float = 0.0,
                        min_points: int = 16) -> Optional[float]:
    """Most-forward source depth of the subject — the cone's front face.

    Projects the subject (dynamic) world points into the SOURCE camera and
    returns the ``depth_pct`` percentile of their camera-z.  ``depth_pct=0``
    (default) = the literal minimum = the subject's most-forward point.
    Returns ``None`` when too few points project in front of the camera.
    """
    if subj_pts_world.shape[0] < min_points:
        return None
    w2c = torch.linalg.inv(c2w_src.float())
    zs = subj_pts_world.float() @ w2c[:3, :3].t()[:, 2] + w2c[2, 3]   # cam-z only
    zs = zs[zs > 1e-4]
    if int(zs.numel()) < min_points:
        return None
    if depth_pct <= 0.0:
        return float(zs.min())
    return float(torch.quantile(zs, depth_pct / 100.0))


@torch.no_grad()
def silhouette_cone_leak_mask(
    depth_tgt: torch.Tensor,       # [H, W] rendered target depth (<=0 = empty)
    K: torch.Tensor,               # [3, 3] intrinsics (matching depth_tgt grid)
    c2w_tgt: torch.Tensor,         # [4, 4] target camera-to-world
    c2w_src: torch.Tensor,         # [4, 4] source camera-to-world
    src_occluder: torch.Tensor,    # [H, W] bool — dilated source subject silhouette
    z_front: float,                # subject's most-forward source depth
    *,
    eps: float = 0.02,
) -> torch.Tensor:
    """Per-frame leaked-background mask via the subject's source-silhouette cone.

    Args:
        depth_tgt: [H, W] rendered target depth; ``> 0`` where the render has
            coverage, ``<= 0`` (or non-finite) where it is empty.
        K: [3, 3] intrinsics for the ``depth_tgt`` / ``src_occluder`` grid
            (both must share this resolution).
        c2w_tgt / c2w_src: [4, 4] OpenCV camera-to-world for the target frame
            and the source view the subject was unprojected from.
        src_occluder: [H, W] bool — the (dilated) subject silhouette in the
            SOURCE image, i.e. the occluder region.
        z_front: the cone's front face = subject's most-forward source depth
            (see :func:`subject_front_depth`).
        eps: depth slack on the front face (Lyra uses 0.02).

    Returns:
        [H, W] bool — pixels behind the subject's source-silhouette cone
        (leaked background).  The caller must still exempt the subject's own
        rendered pixels (dynamic-coverage exemption).
    """
    H, W = depth_tgt.shape
    device = depth_tgt.device
    empty = torch.zeros(H, W, dtype=torch.bool, device=device)
    if not bool(src_occluder.any()):
        return empty

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    w2c_src = torch.linalg.inv(c2w_src.float())
    R_s, t_s = w2c_src[:3, :3], w2c_src[:3, 3]

    # Reproject every covered target pixel into the SOURCE camera.
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    z = depth_tgt.float()
    cam = torch.stack([(xx - cx) / fx * z, (yy - cy) / fy * z, z,
                       torch.ones_like(z)], dim=-1).reshape(-1, 4)
    world = (c2w_tgt.float() @ cam.t()).t()[:, :3]
    cam_s = world @ R_s.t() + t_s
    zt = cam_s[:, 2]
    ut = (fx * cam_s[:, 0] / zt + cx).round().long()
    vt = (fy * cam_s[:, 1] / zt + cy).round().long()

    # Inside the source silhouette AND deeper than the cone front → leaked.
    # ``depth_tgt`` empties are 0 (renderer) or +inf (point-depth viz) — reject
    # both so only genuinely covered pixels can be masked.
    zr = z.reshape(-1)
    covered = (torch.isfinite(zr) & (zr > 1e-6)
               & torch.isfinite(zt) & (zt > 1e-4))
    inb = covered & (ut >= 0) & (ut < W) & (vt >= 0) & (vt < H)
    leak = torch.zeros(H * W, dtype=torch.bool, device=device)
    idx = torch.nonzero(inb, as_tuple=True)[0]
    if idx.numel():
        in_sil = src_occluder.reshape(-1)[vt[idx] * W + ut[idx]]
        behind = in_sil & (zt[idx] > z_front + eps)
        leak[idx[behind]] = True
    return leak.view(H, W)


@torch.no_grad()
def fit_silhouette_ellipse(
    silhouette: torch.Tensor,       # [H, W] bool — subject silhouette pixels
    *,
    inflate: float = 1.0,
    min_points: int = 16,
) -> Optional[torch.Tensor]:
    """Filled principal-axis minimum-enclosing ellipse of a boolean silhouette.

    Fits the covariance ellipse oriented along the silhouette's principal axes
    and sized to the max squared Mahalanobis radius over the foreground pixels
    (so it tightly encloses every silhouette pixel), then rasterises it FILLED
    at the silhouette resolution.  A pixel ``p`` is inside iff
    ``(p - c)^T inv_cov (p - c) <= r2 * inflate``.

    Used as the occlusion-cone occluder region in place of the raw silhouette:
    the resulting mask is the subject's convex elliptical shadow, which closes
    the concave gaps of a spread / limbed subject (the behaviour the
    silhouette-only path deliberately avoided — see the module docstring).

    Args:
        silhouette: ``[H, W]`` bool occluder pixels (e.g. the dilated source
            motion mask).
        inflate: scale on the squared ellipse radius (1.0 = tight enclosing;
            ``> 1`` widens the ellipse).
        min_points: minimum foreground pixels required to fit; below this the
            ellipse is undefined.

    Returns:
        ``[H, W]`` bool — the filled ellipse region, or ``None`` when the
        silhouette has fewer than ``min_points`` pixels (caller should fall
        back to the raw silhouette).
    """
    H, W = silhouette.shape
    device = silhouette.device
    ys, xs = torch.nonzero(silhouette, as_tuple=True)
    if int(xs.numel()) < min_points:
        return None
    pts = torch.stack([xs.float(), ys.float()], dim=1)            # [N, 2] (x, y)
    c = pts.mean(dim=0)
    d = pts - c
    cov = (d.t() @ d) / max(1, d.shape[0] - 1) + torch.eye(2, device=device) * 1e-3
    inv = torch.linalg.inv(cov)
    r2 = (d @ inv * d).sum(dim=1).max().clamp(min=1e-6) * float(inflate)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    dd = torch.stack([xx - c[0], yy - c[1]], dim=-1)              # [H, W, 2]
    q = (dd @ inv * dd).sum(dim=-1)
    return q <= r2


def random_subject_crop_mask(
    pts_world: np.ndarray,                  # [N, 3] world points
    is_dyn: np.ndarray,                     # [N] bool — dynamic-subject points
    pts_frame_ids: Optional[np.ndarray] = None,   # [N] int source-frame id
    *,
    frac_min: float = 0.2,
    frac_max: float = 0.5,
    min_points: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Flag a slab of the dynamic subject's bbox for removal from the render RGB.

    Data-augmentation helper.  Picks **one** random world axis, **one** random
    side (the near or far end along that axis) and a fraction ``f`` in
    ``[frac_min, frac_max]``; the outer ``f`` **of the dynamic points** on that
    side (a point-count quantile, robust to depth flying-pixel outliers that
    would otherwise blow up a min/max bbox and make the cut land on a sparse
    tail) is flagged.  When ``pts_frame_ids`` is given the cut is recomputed
    **per source frame** so it stays a roughly-constant body portion as the
    subject moves — exact cross-frame consistency is *not* required (the slab
    plane may drift a little frame-to-frame).

    The returned points are meant to be dropped from the render RGB render only:
    the caller keeps them in the motion mask (full coverage) and in the
    occlusion cone (so the disoccluded slab shows a hole, not leaked
    background), forcing the model to inpaint the dynamic region from the
    motion cue instead of trusting the rendered colour.

    Returns:
        ``[N]`` bool — True = drop this (dynamic) point from the render RGB.
        All-False when there are too few dynamic points or a degenerate bbox.
    """
    is_dyn = np.asarray(is_dyn, dtype=bool)
    N = int(is_dyn.shape[0])
    crop = np.zeros(N, dtype=bool)
    dyn_idx = np.nonzero(is_dyn)[0]
    if dyn_idx.size < min_points:
        return crop
    if rng is None:
        rng = np.random.default_rng()

    axis = int(rng.integers(0, 3))
    far_side = bool(rng.integers(0, 2))      # True ⇒ crop the high-coord end
    frac = float(rng.uniform(frac_min, frac_max))
    coord = np.asarray(pts_world)[:, axis]

    def _flag(sel: np.ndarray) -> None:
        if sel.size < max(8, min_points // 2):
            return
        c = coord[sel]
        if float(np.ptp(c)) < 1e-6:
            return
        # Cut by POINT-COUNT quantile, NOT coordinate range.  The subject is a
        # 2.5D depth surface with flying-pixel outliers that blow up min/max,
        # so an "outer frac of the coord RANGE" cut lands on a sparse outlier
        # tail (a tiny rim hole) on one side and on ~the whole body on the
        # other (measured: 3% vs 92% for the two sides of one axis).  A
        # quantile threshold removes exactly ``frac`` of the points on the
        # chosen side regardless of outliers → a consistent, clearly visible
        # body slab.
        if far_side:
            crop[sel[c >= np.quantile(c, 1.0 - frac)]] = True
        else:
            crop[sel[c <= np.quantile(c, frac)]] = True

    if pts_frame_ids is not None:
        fids = np.asarray(pts_frame_ids)
        dyn_fids = fids[dyn_idx]
        for f in np.unique(dyn_fids):
            _flag(dyn_idx[dyn_fids == f])
    else:
        _flag(dyn_idx)
    return crop
