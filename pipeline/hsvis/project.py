"""Project folder layout and image I/O.

<project>/
  project.json          capture metadata + settings + which steps have run
  source/pano.*         the stitched equirect from the Insta360 Pro 2 (untouched)
  work/                 intermediate data at working resolution
    labels.png            class index per pixel (palette PNG; see classes.py)
    labels_vis.png        colour view of the labels for checking
    depth.npy             float32 metres per pixel
    instances.npy         int32 object-layer id per pixel (0 = none)
  layers/               what the player loads (WebP with alpha)
  scene.json            the player manifest (layers, lifespans, birds)
  perches.json          candidate bird positions
"""
import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class Project:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.meta_path = os.path.join(self.root, "project.json")
        with open(self.meta_path) as f:
            self.meta = json.load(f)

    # ---- paths -------------------------------------------------------
    def p(self, *parts):
        return os.path.join(self.root, *parts)

    @property
    def source(self):
        return self.p("source", self.meta["source"])

    @property
    def work_size(self):
        """(W, H) used for segmentation/depth."""
        W = self.meta["settings"]["work_width"]
        return W, W // 2

    @property
    def out_size(self):
        """(W, H) of exported layers."""
        W = self.meta["settings"]["out_width"]
        return W, W // 2

    # ---- metadata ----------------------------------------------------
    def save(self):
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)

    def mark(self, step, **info):
        self.meta.setdefault("steps", {})[step] = info
        self.save()

    def load_scene(self):
        with open(self.p("scene.json")) as f:
            return json.load(f)

    def save_scene(self, scene):
        with open(self.p("scene.json"), "w") as f:
            json.dump(scene, f, indent=2)

    # ---- images ------------------------------------------------------
    def load_source(self, size=None):
        img = read_rgb(self.source)
        if size is not None and (img.shape[1], img.shape[0]) != size:
            interp = cv2.INTER_AREA if size[0] < img.shape[1] else cv2.INTER_CUBIC
            img = cv2.resize(img, size, interpolation=interp)
        return img


def init_project(root, pano, title=None, lat=None, lon=None, heading=None, capture_year=None,
                 camera_height=1.8, work_width=4096, out_width=None):
    os.makedirs(os.path.join(root, "source"), exist_ok=True)
    for d in ("work", "layers"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    name = "pano" + os.path.splitext(pano)[1].lower()
    shutil.copy2(pano, os.path.join(root, "source", name))
    img = read_rgb(os.path.join(root, "source", name))
    H, W = img.shape[:2]
    if abs(W - 2 * H) > 2:
        raise SystemExit(f"{pano} is {W}x{H}; expected a 2:1 equirectangular panorama")
    meta = dict(
        title=title or os.path.basename(os.path.abspath(root)),
        source=name,
        source_size=[W, H],
        location=dict(lat=lat, lon=lon, heading_deg=heading),
        capture_year=capture_year,
        settings=dict(
            camera_height_m=camera_height,
            work_width=min(work_width, W),
            # WebP (and many GPUs) top out at 16384 px; Pro 2 mono stills are 7680 wide.
            out_width=min(out_width or W, 16382),
        ),
        steps={},
    )
    with open(os.path.join(root, "project.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return Project(root)


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"could not read {path}")
    if img.dtype == np.uint16:  # 16-bit TIFF/PNG from PTGui or the Insta360 stitcher
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:  # float EXR/HDR: simple Reinhard tone-map
        img = np.clip(img / (1 + img) * 255 * 1.6, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb, quality=92):
    """Write RGB or RGBA (uint8). WebP keeps alpha and is far smaller than PNG for the player."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "RGBA" if rgb.shape[2] == 4 else "RGB"
    im = Image.fromarray(rgb, mode)
    if path.endswith(".webp"):
        im.save(path, quality=quality, method=4, exact=True)
    elif path.endswith(".jpg"):
        im.save(path, quality=quality)
    else:
        im.save(path)


def read_rgba(path):
    return np.array(Image.open(path).convert("RGBA"))
