"""Step 1: per-pixel class labels at working resolution.

Backends:
  manual     a hand-painted label PNG using the colours in classes.py (any resolution, nearest
             colour wins; transparent pixels fall through to the base backend). This is also the
             human-moderation route: paint over or correct an ML result in Krita/Photoshop.
  heuristic  sky / ground / vegetation from colour and position. Crude, needs no models.
  mask2former  Mask2Former trained on Mapillary Vistas, run on overlapping perspective views (GPU
             machine; see ml.py).
"""
import cv2
import numpy as np
from PIL import Image

from .classes import CLASSES, ID, NAMES
from .projection import lonlat_grid


def segment(project, backend="heuristic", labels_path=None, **kw):
    W, H = project.work_size
    rgb = project.load_source((W, H))
    if backend == "manual" and labels_path is None:
        raise SystemExit("--labels is required for the manual backend")

    if backend == "manual":
        labels = np.full((H, W), ID["other"], np.uint8)
    elif backend == "heuristic":
        labels = heuristic(rgb)
    elif backend == "mask2former":
        from .ml import mask2former_segment
        labels = mask2former_segment(rgb, **kw)
    else:
        raise SystemExit(f"unknown segmentation backend {backend}")

    if labels_path:
        painted, valid = read_label_png(labels_path, (W, H))
        labels[valid] = painted[valid]

    save_labels(project, labels)
    project.mark("segment", backend=backend, labels=labels_path)
    return labels


def read_label_png(path, size):
    """Map a colour-coded label image onto class ids. Returns (labels, valid_mask)."""
    im = np.array(Image.open(path).convert("RGBA"))
    im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)
    rgb = im[..., :3].astype(np.int32)
    palette = np.array([CLASSES[n]["colour"] for n in NAMES], np.int32)
    # nearest palette colour, in chunks to keep memory sane at 8K
    out = np.zeros(rgb.shape[:2], np.uint8)
    dist_ok = np.zeros(rgb.shape[:2], bool)
    for y in range(0, rgb.shape[0], 256):
        blk = rgb[y:y + 256, :, None, :] - palette[None, None]
        d2 = (blk ** 2).sum(-1)
        out[y:y + 256] = d2.argmin(-1)
        dist_ok[y:y + 256] = d2.min(-1) < 45 ** 2  # tolerate JPEG/antialias drift
    return out, (im[..., 3] > 127) & dist_ok


def heuristic(rgb):
    H, W = rgb.shape[:2]
    _, lat = lonlat_grid(W, H)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v, s = hsv[..., 2] / 255, hsv[..., 1] / 255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), 3)
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]

    skyish = (v > 0.45) & (grad < 6) & (lat > np.deg2rad(-3)) & ((b >= g - 10) | (s < 0.25))
    # sky must connect to the zenith
    n, comp = cv2.connectedComponents(skyish.astype(np.uint8), connectivity=4)
    top_ids = np.unique(comp[: max(2, H // 50)][skyish[: max(2, H // 50)]])
    sky = np.isin(comp, top_ids[top_ids > 0])

    labels = np.full((H, W), ID["other"], np.uint8)
    green = (g > r * 1.05) & (g > b * 1.05)
    below = lat < 0
    labels[below] = ID["terrain"]
    labels[below & ~green & (s < 0.18)] = ID["road"]
    labels[~below & green] = ID["vegetation"]
    labels[sky] = ID["sky"]
    return labels


def save_labels(project, labels):
    im = Image.fromarray(labels, "P")
    pal = []
    for n in NAMES:
        pal += list(CLASSES[n]["colour"])
    im.putpalette(pal + [0] * (768 - len(pal)))
    im.save(project.p("work", "labels.png"))
    im.convert("RGB").save(project.p("work", "labels_vis.png"))


def load_labels(project):
    return np.array(Image.open(project.p("work", "labels.png")))
