"""Step 4: fill what's behind each removable layer.

Two jobs:
  plate fill      inpaint the plate wherever an object layer was cut out, so removing a car or a
                  house reveals verge/field/hedge rather than a hole. Done in perspective tiles
                  centred on each object (straight lines stay straight; diffusion models expect
                  perspective images), then projected back into the equirect.
  object completion  where a nearer object hides part of a farther one (a car in front of a house),
                  extend the farther layer behind it, so removing the car shows the whole house.

What sits behind a hole (sky or ground) is decided by propagating the nearest surrounding class;
holes that turn out to be sky become transparent in the plate (the sky layer shows through), which
also means poles/trees against the sky need no inpainting at all.

Backends: opencv (Telea, runs anywhere, smeary on big holes), sdxl (diffusion, GPU; see ml.py),
and hosted: fal-flux (FLUX.1 Fill on fal.ai), gemini / fal-edit (Nano Banana; see ai/backends.py).
"""
import math

import cv2
import numpy as np

from .classes import ID
from .depth import load_depth
from .layers import paste_wrap, write_depth_png
from .project import read_rgba, write_rgb
from .projection import crop_wrap, equirect_to_perspective, perspective_to_equirect
from .segment import load_labels


class OpenCVInpainter:
    def inpaint(self, rgb, mask, prompt=None, labels=None):
        """rgb uint8 HxWx3, mask uint8 (255 = fill). With `labels` (background class per pixel),
        each class is filled only from nearby pixels of the same class: road from road, verge
        from verge, instead of one smear across all of them."""
        if labels is None:
            return self._telea(rgb, mask)
        hole = mask > 0
        out = rgb.copy()
        for c in np.unique(labels[hole]):
            hc = hole & (labels == c)
            band = cv2.dilate(hc.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
            m = hole | (band & (labels != c))  # whole hole masked: removed pixels are never a source
            res = self._telea(rgb, m.astype(np.uint8) * 255)
            out[hc] = res[hc]
        return out

    def _telea(self, rgb, mask):
        h, w = mask.shape
        area = (mask > 0).sum()
        if area == 0:
            return rgb
        # big holes: inpaint coarse first (Telea is slow and smeary at full size), then add grain
        scale = 1 if area < 60_000 else max(2, int(math.sqrt(area / 60_000)) + 1)
        # mask-aware downscale: masked pixels (the thing being removed) must not tint their neighbours
        keep = (mask == 0).astype(np.float32)
        size = (w // scale, h // scale)
        num = cv2.resize(rgb.astype(np.float32) * keep[..., None], size, interpolation=cv2.INTER_AREA)
        den = cv2.resize(keep, size, interpolation=cv2.INTER_AREA)
        small = np.clip(num / np.maximum(den, 1e-6)[..., None], 0, 255).astype(np.uint8)
        sm = ((den < 0.999) * 255).astype(np.uint8)  # any masked pixel inside -> fill it
        res = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), sm, 6, cv2.INPAINT_TELEA)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        if scale > 1:
            res = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC)
            grain = np.random.default_rng(0).normal(0, 6, (h, w, 1)).astype(np.float32)
            res = np.clip(res.astype(np.float32) + grain, 0, 255).astype(np.uint8)
        m = (mask > 0)[..., None]
        return np.where(m, res, rgb)


AI_BACKENDS = ("fal-flux", "gemini", "fal-edit")


def get_inpainter(backend, project=None, dry_run=False, **kw):
    """Returns (inpainter, ai_session or None)."""
    if backend == "opencv":
        return OpenCVInpainter(), None
    if backend == "sdxl":
        from .ml import SDXLInpainter
        return SDXLInpainter(**kw), None
    if backend in AI_BACKENDS:
        from .ai.backends import get_inpainter as ai_inpainter
        return ai_inpainter(backend, project, dry_run)
    raise SystemExit(f"unknown fill backend {backend}")


# What the background is likely to be, as a prompt for diffusion backends.
BG_PROMPT = {
    "terrain": "grass verge and meadow", "road": "country lane surface", "path": "footpath",
    "hedge": "hawthorn hedgerow", "vegetation": "trees and hedgerow", "water": "pond water",
    "far": "distant rolling hills", "building": "stone wall", "wall": "garden wall",
}


def fill(project, backend="opencv", tile=1536, dry_run=False, **kw):
    scene = project.load_scene()
    Wo, Ho = project.out_size
    labels = load_labels(project)
    Hw, Ww = labels.shape
    inst = np.load(project.p("work", "instances.npy"))
    holes_full = cv2.imread(project.p("work", "holes.png"), cv2.IMREAD_GRAYSCALE) > 0
    painter, session = get_inpainter(backend, project, dry_run, **kw)
    if backend in AI_BACKENDS:
        tile = min(tile, 1024)  # hosted models work at ~1 MP; more tiles beats upscaled ones

    # ---- what's behind the holes: nearest surrounding class -----------------
    holes_w = cv2.resize(holes_full.astype(np.uint8), (Ww, Hw), interpolation=cv2.INTER_AREA) > 0
    bg = nearest_label(labels, holes_w | (inst > 0))
    np.save(project.p("work", "background_labels.npy"), bg)
    sky_bg = cv2.resize((bg == ID["sky"]).astype(np.float32), (Wo, Ho), interpolation=cv2.INTER_LINEAR)
    need = holes_full & (sky_bg < 0.5)

    plate = read_rgba(project.p(scene["plate"]["photo"]))[..., :3].copy()
    remaining = need.copy()                 # still to fill
    blocked = holes_full.copy()             # not usable as inpainting source (incl. sky-backed holes)

    # ---- plate fill, far objects first ------------------------------------
    for layer in scene["layers"]:
        x0, y0, w, h = layer["bbox"]
        region = crop_wrap(remaining, layer["bbox"])
        if not region.any():
            continue
        prompt = "rural English landscape, " + BG_PROMPT.get(dominant_bg(bg, layer, Wo), "countryside")
        for view in views_for_bbox(layer["bbox"], Wo, Ho, tile):
            yaw, pitch, fov, size = view
            rgb_t = equirect_to_perspective(plate, yaw, pitch, fov, size)
            hole_t = equirect_to_perspective(blocked.astype(np.uint8) * 255, yaw, pitch, fov, size,
                                             interp=cv2.INTER_NEAREST)
            # the tile is sampled slightly coarser than the panorama: grow the hole so its rim
            # pixels are inpainted too rather than keeping the removed object's colour
            hole_t = cv2.dilate(hole_t, np.ones((5, 5), np.uint8))
            lab_t = equirect_to_perspective(bg, yaw, pitch, fov, size, interp=cv2.INTER_NEAREST)
            out_t = painter.inpaint(rgb_t, hole_t, prompt, labels=lab_t)
            vals, valid = perspective_to_equirect(out_t, yaw, pitch, fov, Wo, Ho, region=layer["bbox"])
            reg = crop_wrap(remaining, layer["bbox"])
            write = reg & valid
            cur = crop_wrap(plate, layer["bbox"])
            cur[write] = vals[write]
            paste_wrap(plate, layer["bbox"], cur)
            paste_wrap(remaining, layer["bbox"], reg & ~write)
            paste_wrap(blocked, layer["bbox"], crop_wrap(blocked, layer["bbox"]) & ~write)

    if remaining.any():  # anything the tiles missed (e.g. at the poles): direct equirect inpaint
        print(f"  {remaining.sum()} px not covered by tiles; filling directly in the equirect")
        filled = painter.inpaint(plate, blocked.astype(np.uint8) * 255)
        plate[remaining] = filled[remaining]

    # removed things that stood against the sky: fully transparent, so no ghost edge survives
    plate_alpha = np.clip(1 - sky_bg, 0, 1)
    plate_alpha[holes_full & (sky_bg >= 0.5)] = 0
    plate_alpha = (plate_alpha * 255).astype(np.uint8)
    if not dry_run:
        write_rgb(project.p("layers", "plate_filled_photo.webp"), np.dstack([plate, plate_alpha]))
        scene["plate"]["photo"] = "layers/plate_filled_photo.webp"
        scene["plate"]["filled"] = True
        # background depth (objects removed) for the player's depth effects
        from .depth import ground_plane
        write_depth_png(project, ground_plane(bg, project.meta["settings"]["camera_height_m"]))

    complete_objects(project, scene, inst, painter, write=not dry_run)
    if session:
        print(f"  {session.summary()}")
    if dry_run:
        return  # nothing written: a dry run only counts calls
    project.save_scene(scene)
    project.mark("fill", backend=backend)


def complete_objects(project, scene, inst, painter, write=True):
    """Extend each object layer behind nearer objects that overlap it."""
    Wo, _ = project.out_size
    Ww = inst.shape[1]
    s = Wo / Ww
    ids = {}
    with open(project.p("work", "instance_ids.txt")) as f:
        for line in f:
            k, lid = line.split()
            ids[lid] = int(k)
    order = {l["id"]: i for i, l in enumerate(scene["layers"])}
    for layer in scene["layers"]:
        k = ids[layer["id"]]
        bo = layer["bbox"]
        bw = [int(round(v / s)) for v in bo]
        sub = crop_wrap(inst, bw)
        own = sub == k
        if not own.any():
            continue
        nearer = np.isin(sub, [ids[l["id"]] for l in scene["layers"] if order[l["id"]] > order[layer["id"]]])
        hull = span_hull(own)
        occ = hull & nearer & ~own
        if occ.sum() < 4:
            continue
        rgba = read_rgba(project.p(layer["photo"]))
        occ_o = cv2.resize(occ.astype(np.uint8), (bo[2], bo[3]), interpolation=cv2.INTER_NEAREST)
        occ_o = cv2.dilate(occ_o, np.ones((5, 5), np.uint8)) > 0
        a = rgba[..., 3]
        if getattr(painter, "generative", False):
            # generative models use the surrounding context; only the hidden part is filled
            mask = occ_o.astype(np.uint8) * 255
            prompt = f"the hidden part of the {layer['class'].replace('_', ' ')}, continuing its shape and materials"
        else:
            # source pixels = the object itself only: mask out everything else too
            mask = (occ_o | (a < 128)).astype(np.uint8) * 255
            prompt = layer["class"]
        filled = painter.inpaint(rgba[..., :3], mask, prompt)
        rgba[..., :3] = np.where(occ_o[..., None], filled, rgba[..., :3])
        soft = (cv2.GaussianBlur(occ_o.astype(np.float32), (0, 0), 1.0) * 255).astype(np.uint8)
        rgba[..., 3] = np.maximum(a, soft)
        path = layer["photo"].replace("_photo.webp", "_completed_photo.webp")
        if not write:
            continue
        write_rgb(project.p(path), rgba)
        layer["photo"] = path
        layer["completed_behind"] = True


def span_hull(m):
    """Pixels that lie between the object's extremes in their row or in their column - a cheap
    guess at its shape where something nearer hides it. Only pixels of nearer layers inside this
    hull are filled, so over-generous guesses stay contained."""
    rows = np.zeros_like(m)
    for y in np.where(m.any(1))[0]:
        xs = np.where(m[y])[0]
        rows[y, xs[0]:xs[-1] + 1] = True
    cols = np.zeros_like(m)
    for x in np.where(m.any(0))[0]:
        ys = np.where(m[:, x])[0]
        cols[ys[0]:ys[-1] + 1, x] = True
    # also extend downward to the object's lowest row (objects stand on the ground behind occluders)
    bottom = np.where(m.any(1))[0][-1]
    for x in np.where(m.any(0))[0]:
        ys = np.where(m[:, x])[0]
        cols[ys[0]:bottom + 1, x] |= rows[ys[0]:bottom + 1, x]
    return rows | cols


def nearest_label(labels, holes):
    """Replace labels inside holes with the nearest label outside them."""
    src = (~holes).astype(np.uint8)
    if src.all():
        return labels.copy()
    _, lab = cv2.distanceTransformWithLabels(1 - src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(src)  # label i (1-based) = i-th zero pixel of (1 - src) in row-major order
    lut = labels[ys, xs]
    out = labels.copy()
    out[holes] = lut[lab[holes] - 1]
    return out


def dominant_bg(bg, layer, Wo):
    s = Wo / bg.shape[1]
    sub = crop_wrap(bg, [int(round(v / s)) for v in layer["bbox"]])
    vals, counts = np.unique(sub, return_counts=True)
    from .classes import NAMES
    return NAMES[int(vals[counts.argmax()])]


def views_for_bbox(bbox, W, H, tile):
    """Perspective views that together cover an equirect bbox."""
    x0, y0, w, h = bbox
    lon0 = (x0 / W) * 2 * math.pi - math.pi
    wa = w / W * 2 * math.pi
    lat_top = math.pi / 2 - y0 / H * math.pi
    lat_bot = math.pi / 2 - (y0 + h) / H * math.pi
    ha = lat_top - lat_bot
    n = max(1, math.ceil(wa / math.radians(80)))
    views = []
    for i in range(n):
        c_lon = lon0 + wa * (i + 0.5) / n
        c_lat = (lat_top + lat_bot) / 2
        span = max(wa / n, ha)
        fov = min(math.radians(110), max(math.radians(20), span * 1.35 + math.radians(6)))
        # if the box is near a pole the tile must reach it; aim the view at the bbox centre
        size = int(min(tile, max(256, fov * W / (2 * math.pi))))
        views.append((c_lon, c_lat, fov, size))
    return views
