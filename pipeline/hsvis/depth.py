"""Step 2: approximate metric depth per pixel.

Fixed point of view means depth isn't needed for parallax. It's used to:
  * order layers far -> near (draw order, removal order),
  * drive depth-based effects in the player (paint-over sweep, dawn mist),
  * estimate how far away a bird is (loudness, size) and which perches are plausible.

Backends:
  ground-plane  geometry only. A level camera at known height sees flat ground at
                distance h / sin(angle below horizon); upright things take the depth of
                the point where they meet the ground. Surprisingly good for lanes/fields.
  depth-anything  Depth Anything V2 per perspective view, each view aligned to the
                ground-plane prior so the views agree in scale (GPU machine; see ml.py).
  fal           the same model hosted on fal.ai (needs FAL_KEY).
"""
import cv2
import numpy as np

from .classes import CLASSES, GROUND, ID, NAMES
from .projection import lonlat_grid
from .segment import load_labels

SKY_DEPTH = 10000.0
FAR_DEPTH = 2000.0
MAX_GROUND = 1500.0


def depth(project, backend="ground-plane", dry_run=False, **kw):
    if backend == "fal" and not dry_run:
        from .ai import config
        config.fal_key()
    labels = load_labels(project)
    h = project.meta["settings"]["camera_height_m"]
    prior = ground_plane(labels, h)
    if backend == "ground-plane":
        d = prior
    elif backend == "depth-anything":
        from .ml import depth_anything
        W, H = project.work_size
        d = depth_anything(project.load_source((W, H)), labels, prior, **kw)
        d[labels == ID["sky"]] = SKY_DEPTH
    elif backend == "fal":
        from .ai import backends as ai
        W, H = project.work_size
        session = ai.session_for(project, dry_run, backend)
        d = ai.depth_fal(project.load_source((W, H)), labels, prior, session, **kw)
        d[labels == ID["sky"]] = SKY_DEPTH
        print(f"  {session.summary()}")
        if dry_run:
            return None
    else:
        raise SystemExit(f"unknown depth backend {backend}")
    np.save(project.p("work", "depth.npy"), d.astype(np.float32))
    vis = np.clip(np.log(d) / np.log(SKY_DEPTH), 0, 1)
    cv2.imwrite(project.p("work", "depth_vis.png"), (255 - vis * 255).astype(np.uint8))
    project.mark("depth", backend=backend)
    return d


def load_depth(project):
    return np.load(project.p("work", "depth.npy"))


def ground_distance(lat, cam_h):
    with np.errstate(divide="ignore"):
        return np.where(lat < 0, np.minimum(cam_h / np.sin(-lat), MAX_GROUND), np.inf)


# typical heights (m) for estimating distance from apparent size when the base is hidden
TYPICAL_HEIGHT = {"building": 7.0, "vegetation": 10.0, "hedge": 2.0, "wall": 1.5, "pole": 8.0,
                  "vehicle": 1.5, "person": 1.7, "sign": 2.5}


def size_prior(name, top_lat, cam_h):
    rise = TYPICAL_HEIGHT.get(name, 3.0) - cam_h
    if rise <= 0 or top_lat <= 0.002:
        return 0.0
    return min(rise / np.tan(top_lat), FAR_DEPTH)


def ground_plane(labels, cam_h):
    H, W = labels.shape
    _, lat = lonlat_grid(W, H)
    g = ground_distance(lat, cam_h)
    d = np.where(np.isfinite(g), g, FAR_DEPTH).astype(np.float32)
    d[labels == ID["sky"]] = SKY_DEPTH
    d[labels == ID["far"]] = FAR_DEPTH

    is_ground = np.isin(labels, [ID[n] for n in GROUND])
    upright = [ID[n] for n in NAMES if CLASSES[n]["upright"] and n != "far"]
    comps = []
    for cid in upright:
        n, comp, stats, _ = cv2.connectedComponentsWithStats((labels == cid).astype(np.uint8), connectivity=8)
        for k in range(1, n):
            x, y, w, hh, _ = stats[k]
            comps.append((y + hh, cid, k, comp, (x, y, w, hh)))
    # nearest first (lowest bottom edge), so things standing behind them can use their depth
    comps.sort(key=lambda c: -c[0])
    for bottom, cid, k, comp, (x, y, w, hh) in comps:
        m = comp[y:y + hh, x:x + w] == k
        rows = hh - 1 - np.argmax(m[::-1], axis=0)  # lowest pixel per column
        has = m.any(axis=0)
        cols = np.arange(x, x + w)[has]
        brow = y + rows[has]
        below = np.minimum(brow + 1, H - 1)
        touches = is_ground[below, cols] | (brow >= H - 1)
        if touches.any():
            # per column: where the foot is visible use the ground distance there, elsewhere
            # interpolate along the object (a long hedge recedes; one depth for it would be wrong)
            contact = np.minimum(g[brow[touches], cols[touches]], FAR_DEPTH)
            col_d = np.interp(cols, cols[touches], contact)
        else:  # foot hidden behind something nearer: at least that far, and sized like its class
            val = max(np.median(d[below, cols]) * 1.3, size_prior(NAMES[cid], lat[y, 0], cam_h))
            col_d = np.full(len(cols), val)
        per_col = np.zeros(w, np.float32)
        per_col[has] = np.minimum(col_d, FAR_DEPTH)
        sub = d[y:y + hh, x:x + w]
        sub[m] = np.broadcast_to(per_col, m.shape)[m]
    return d
