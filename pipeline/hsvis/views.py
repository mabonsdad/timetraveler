"""Running image models over an equirect via overlapping perspective views.

Models (local or remote) are trained on ordinary photos, so they never see the raw equirect:
the panorama is rendered into perspective views, each view is processed, and the results are
projected back. Label/depth results keep each pixel from the view where it is most central;
image results (stylisation) are feathered together.
"""
import cv2
import numpy as np

from .projection import (equirect_to_perspective, lonlat_grid, lonlat_to_dir, overlap_views,
                         perspective_to_equirect, rotation)

VIEW_FOV = np.radians(110)


def centrality(W, H, yaw, pitch):
    """Cosine between each equirect pixel's direction and the view's forward direction."""
    lon, lat = lonlat_grid(W, H)
    fwd = rotation(yaw, pitch) @ np.array([0, 0, 1.0])
    return lonlat_to_dir(lon, lat) @ fwd


def merge_views(W, H, run_view, size, dtype, interp, views=None, fov=VIEW_FOV, init=0):
    """run_view(yaw, pitch) -> perspective result (size x size); each equirect pixel takes its
    value from the view in which it is most central. Pixels a view marks with `init` don't count."""
    best = np.full((H, W), -2.0, np.float32)
    out = np.full((H, W), init, dtype)
    for yaw, pitch in views or overlap_views():
        res = run_view(yaw, pitch)
        if res is None:
            continue
        vals, valid = perspective_to_equirect(res, yaw, pitch, fov, W, H, interp=interp)
        c = centrality(W, H, yaw, pitch)
        take = valid & (c > best) & (vals != init)
        out[take] = vals[take]
        best[take] = c[take]
    return out


def tile_views(fov_deg=90):
    """Views covering the sphere with ~50% overlap for a given field of view: a ring on the
    horizon, rings above and below, and the poles."""
    step = fov_deg / 2
    views = []
    for pitch in (0.0, fov_deg * 0.55, -fov_deg * 0.55):
        n = int(np.ceil(360 / step * np.cos(np.radians(pitch))))
        offset = 0 if pitch == 0 else 180 / n
        views += [(np.radians(i * 360 / n + offset), np.radians(pitch)) for i in range(n)]
    views += [(0.0, np.pi / 2), (0.0, -np.pi / 2)]
    return views


def blend_views(rgb, process, fov_deg=90, size=1024, power=6, on_view=None):
    """Run an image->image `process(view_rgb, index) -> rgb` over overlapping views of `rgb`
    (an equirect) and feather the results back together. Centre-weighted so each region comes
    mostly from the view that saw it head-on."""
    H, W = rgb.shape[:2]
    fov = np.radians(fov_deg)
    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)
    views = tile_views(fov_deg)
    for i, (yaw, pitch) in enumerate(views):
        view = equirect_to_perspective(rgb, yaw, pitch, fov, size)
        out = process(view, i)
        if out.shape[:2] != (size, size):
            out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
        vals, valid = perspective_to_equirect(out, yaw, pitch, fov, W, H)
        c = np.clip(centrality(W, H, yaw, pitch), 0, 1)
        # weight falls to zero at the view's edge so seams fade rather than cut
        w = ((c - np.cos(fov / 2 * 1.2)).clip(0) ** power * valid)[..., None].astype(np.float32)
        acc += vals.astype(np.float32) * w
        wsum += w
        if on_view:
            on_view(i, len(views))
    out = acc / np.maximum(wsum, 1e-6)
    hole = wsum[..., 0] < 1e-6
    out[hole] = rgb[hole]
    return np.clip(out, 0, 255).astype(np.uint8)


def align_disparity(disp, prior_view, ground_view):
    """Fit 1/depth = a * disparity + b on visible ground, using the ground-plane prior as truth.
    Tries the map as given and inverted (providers differ on near = bright or dark) and keeps
    whichever fits better. Returns metric-ish depth for the view."""
    if ground_view.sum() < 500:  # looking at the sky or a wall: no ground to fit against
        return None
    target = 1 / prior_view[ground_view]
    best = None
    for d in (disp, disp.max() - disp):
        A = np.stack([d[ground_view], np.ones(ground_view.sum())], 1)
        (a, b), res, *_ = np.linalg.lstsq(A, target, rcond=None)
        err = float(res[0]) if len(res) else float(((A @ [a, b] - target) ** 2).sum())
        if a > 0 and (best is None or err < best[0]):
            best = (err, a, b, d)
    if best is None:
        return None
    _, a, b, d = best
    return (1 / np.maximum(a * d + b, 1e-4)).astype(np.float32)
