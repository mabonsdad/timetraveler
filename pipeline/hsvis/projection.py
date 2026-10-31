"""Equirectangular <-> perspective projection.

Conventions (shared with the player):
  column x = W/2 is the panorama's forward direction (yaw 0); yaw increases to the right
  row y = 0 is straight up (pitch +90), y = H straight down
  world axes: x right, y up, z forward
"""
import numpy as np
import cv2


def lonlat_grid(W, H):
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    return np.meshgrid(lon, lat)


def lonlat_to_dir(lon, lat):
    c = np.cos(lat)
    return np.stack([c * np.sin(lon), np.sin(lat), c * np.cos(lon)], axis=-1)


def dir_to_lonlat(d):
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    return lon, lat


def rotation(yaw, pitch):
    """Camera->world rotation. Radians; positive pitch looks up, positive yaw looks right."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    return ry @ rx


def _focal(fov, size):
    return 0.5 * size / np.tan(0.5 * fov)


def equirect_to_perspective(eq, yaw, pitch, fov, size, interp=cv2.INTER_LINEAR):
    """Render a square perspective view (size x size, horizontal fov in radians) from an equirect image."""
    H, W = eq.shape[:2]
    f = _focal(fov, size)
    i, j = np.meshgrid(np.arange(size) + 0.5 - size / 2, np.arange(size) + 0.5 - size / 2)
    cam = np.stack([i / f, -j / f, np.ones_like(i)], axis=-1)
    cam /= np.linalg.norm(cam, axis=-1, keepdims=True)
    world = cam @ rotation(yaw, pitch).T
    lon, lat = dir_to_lonlat(world)
    mx = ((lon / (2 * np.pi) + 0.5) * W - 0.5).astype(np.float32)
    my = ((0.5 - lat / np.pi) * H - 0.5).astype(np.float32)
    return cv2.remap(eq, mx, my, interp, borderMode=cv2.BORDER_WRAP)


def perspective_to_equirect(img, yaw, pitch, fov, W, H, interp=cv2.INTER_LINEAR, region=None):
    """Project a perspective image back into equirect space.

    Returns (values, valid) at full equirect size, or for the sub-rectangle
    region=(x0, y0, w, h) (x may wrap) if given.
    """
    size = img.shape[0]
    f = _focal(fov, size)
    if region is None:
        lon, lat = lonlat_grid(W, H)
    else:
        x0, y0, w, h = region
        xs = (np.arange(x0, x0 + w) % W + 0.5) / W * 2 * np.pi - np.pi
        ys = np.pi / 2 - (np.arange(y0, y0 + h) + 0.5) / H * np.pi
        lon, lat = np.meshgrid(xs, ys)
    world = lonlat_to_dir(lon, lat)
    cam = world @ rotation(yaw, pitch)  # world -> camera (R^T applied on the right)
    z = cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        px = cam[..., 0] / z * f + size / 2 - 0.5
        py = -cam[..., 1] / z * f + size / 2 - 0.5
    valid = (z > 1e-6) & (px >= 0) & (px <= size - 1) & (py >= 0) & (py <= size - 1)
    px = np.where(valid, px, -1).astype(np.float32)
    py = np.where(valid, py, -1).astype(np.float32)
    out = cv2.remap(img, px, py, interp, borderMode=cv2.BORDER_CONSTANT)
    return out, valid


def cube_views():
    """(name, yaw, pitch) for the six 90-degree cube faces."""
    h = np.pi / 2
    return [("front", 0, 0), ("right", h, 0), ("back", np.pi, 0),
            ("left", -h, 0), ("up", 0, h), ("down", 0, -h)]


def overlap_views():
    """Overlapping views (110 deg) for model inference: 8 around the horizon, 4 above/below, plus poles.
    Overlap lets us vote across views and hides face-edge errors."""
    views = [(np.deg2rad(a), 0.0) for a in range(0, 360, 45)]
    views += [(np.deg2rad(a), np.deg2rad(50)) for a in range(0, 360, 90)]
    views += [(np.deg2rad(a), np.deg2rad(-50)) for a in range(45, 405, 90)]
    views += [(0.0, np.pi / 2), (0.0, -np.pi / 2)]
    return views


def equirect_wrap_bbox(mask):
    """Bounding box of a mask in an equirect image, allowing it to wrap across the +-180 seam.

    Returns (x0, y0, w, h) where x0 may be such that x0 + w > W (wrapping), or None if empty.
    """
    H, W = mask.shape
    cols = mask.any(axis=0)
    rows = np.where(mask.any(axis=1))[0]
    if not cols.any():
        return None
    if cols.all():
        return (0, int(rows[0]), W, int(rows[-1] - rows[0] + 1))
    # Find the largest circular run of empty columns; the box starts right after it.
    empty = ~cols
    doubled = np.concatenate([empty, empty])
    best_len, best_end, run = 0, 0, 0
    for i, e in enumerate(doubled):
        run = run + 1 if e else 0
        if run > best_len and run <= W:
            best_len, best_end = run, i
    x0 = (best_end + 1) % W
    w = W - best_len
    return (int(x0), int(rows[0]), int(w), int(rows[-1] - rows[0] + 1))


def crop_wrap(img, bbox):
    x0, y0, w, h = bbox
    W = img.shape[1]
    xs = np.arange(x0, x0 + w) % W
    return img[y0:y0 + h][:, xs]
