"""Step 3: cut the panorama into a layer stack and write the player manifest (scene.json).

Stack, back to front (fixed point of view, so layers are simply composited in order):
  sky     full-coverage sky; the player can replace/animate it for the dawn progression
  plate   everything that isn't sky or an object layer; holes left by objects are filled in fill.py
  objects one layer per removable/era-variable thing (building, car, pole, hedge...), cropped to
          its bounding box, sorted far -> near, each with a lifespan the research system can set
  additions (later) painted/generated layers for things that no longer exist (an old cottage,
          a narrower lane surface) - same format, just a lifespan in the past

Re-running keeps lifespans/feature ids/notes already edited in scene.json for layers whose id
and class still match.
"""
import os

import cv2
import numpy as np

from .classes import CLASSES, ID, NAMES
from .depth import SKY_DEPTH, load_depth
from .project import write_rgb
from .projection import crop_wrap, equirect_wrap_bbox
from .segment import load_labels

DEFAULT_ERAS = [2126, 2026, 1926, 1826, 1726]
BRIDGE_CLASSES = ("pole", "sign", "person", "vehicle")

# How each era is presented. Shared by the offline renderer and the player.
# sepia: 0..1 toward warm monochrome; sat: saturation multiplier; tint: RGB multiplier
DEFAULT_ERA_LOOKS = {
    "2126": dict(sepia=0.0, sat=0.9, tint=[0.95, 1.02, 1.04], label="2126 (scenario)"),
    "2026": dict(sepia=0.0, sat=1.0, tint=[1, 1, 1], label="Today"),
    "1926": dict(sepia=0.3, sat=0.8, tint=[1.03, 1.0, 0.95], label="1926"),
    "1826": dict(sepia=0.55, sat=0.65, tint=[1.05, 1.0, 0.9], label="1826"),
    "1726": dict(sepia=0.7, sat=0.55, tint=[1.06, 1.0, 0.88], label="1726"),
}


def build_layers(project, layer_classes=None, min_area=60, grow=3):
    labels = load_labels(project)
    depth = load_depth(project)
    Hw, Ww = labels.shape
    Wo, Ho = project.out_size
    s = Wo / Ww
    photo = project.load_source((Wo, Ho))
    for f in os.listdir(project.p("layers")):  # drop files from a previous run
        os.remove(project.p("layers", f))
    layer_classes = layer_classes or [n for n in NAMES if CLASSES[n]["layer"]]

    # ---- object instances -----------------------------------------------
    inst = np.zeros((Hw, Ww), np.int32)
    objs = []
    # Thin things in front (poles, signs, people, cars) often cut one house or hedge into
    # several pieces. Label connectivity through them so the pieces become one layer, which
    # fill.complete_objects then extends behind the occluder.
    bridge = np.isin(labels, [ID[n] for n in BRIDGE_CLASSES])
    for name in layer_classes:
        mask = labels == ID[name]
        through = mask if name in BRIDGE_CLASSES else (mask | bridge)
        n, comp = cv2.connectedComponents(through.astype(np.uint8), connectivity=8)
        comp = merge_seam(comp) * mask
        for k in np.unique(comp):
            if k == 0:
                continue
            m = comp == k
            if m.sum() < min_area:
                continue
            idx = len(objs) + 1
            inst[m] = idx
            objs.append(dict(idx=idx, cls=name, depth=float(np.median(depth[m]))))

    objs = occlusion_order(objs, inst, depth)
    holes = np.zeros((Ho, Wo), np.uint8)
    old = {}
    if os.path.exists(project.p("scene.json")):
        old = {l["id"]: l for l in project.load_scene().get("layers", [])}

    layers = []
    for n, o in enumerate(objs, 1):
        lid = f"{o['cls']}_{n:03d}"
        mask_w = inst == o["idx"]
        x0, y0, w, h = equirect_wrap_bbox(mask_w)
        pad = 2 + grow
        x0, w = x0 - pad, min(w + 2 * pad, Ww)
        y0, y1 = max(0, y0 - pad), min(Hw, y0 + h + pad)
        bw = (x0 % Ww, y0, w, y1 - y0)
        bo = (int(round(bw[0] * s)), int(round(bw[1] * s)), int(round(bw[2] * s)), int(round(bw[3] * s)))
        core_f = cv2.resize(crop_wrap(mask_w.astype(np.float32), bw), (bo[2], bo[3]), interpolation=cv2.INTER_LINEAR)
        alpha = soft_alpha(crop_wrap(mask_w.astype(np.float32), bw), (bo[2], bo[3]), grow)
        # labels come from a lower resolution than the photo, so trust only the object's inner core
        # for colour; erode by about one working pixel, but keep thin things (poles) intact
        core = core_f > 0.7
        k = int(np.ceil(s)) + 1
        eroded = cv2.erode(core.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        core = eroded if eroded.sum() > 0.25 * core.sum() else core
        rgba = np.dstack([decontaminate(crop_wrap(photo, bo), core), alpha])
        path = f"layers/{lid}_photo.webp"
        write_rgb(project.p(path), rgba)
        paste_wrap(holes, bo, alpha > 0, np.maximum)
        info = CLASSES[o["cls"]]
        prev = old.get(lid, {})
        keep = prev.get("class") == o["cls"]
        layers.append(dict(
            id=lid, kind="object", **{"class": o["cls"]},
            bbox=list(bo), depth_m=round(o["depth"], 1),
            photo=path, illus=None,
            lifespan=prev["lifespan"] if keep else list(info["lifespan"]),
            lifespan_source=prev.get("lifespan_source", "default") if keep else "default",
            feature_id=prev.get("feature_id") if keep else None,
            note=prev.get("note", "") if keep else "",
        ))
        o["lid"] = lid

    np.save(project.p("work", "instances.npy"), inst)
    cv2.imwrite(project.p("work", "holes.png"), holes * 255)
    with open(project.p("work", "instance_ids.txt"), "w") as f:
        f.writelines(f"{o['idx']} {o['lid']}\n" for o in objs)

    # ---- sky ------------------------------------------------------------
    sky_w = (labels == ID["sky"]).astype(np.float32)
    sky_alpha = cv2.resize(sky_w, (Wo, Ho), interpolation=cv2.INTER_LINEAR)
    write_rgb(project.p("layers", "sky_photo.webp"), extend_sky(photo, sky_alpha))

    # ---- plate (unfilled for now: objects still visible until fill.py runs) ------
    plate_a = (np.clip(1 - sky_alpha, 0, 1) * 255).astype(np.uint8)
    write_rgb(project.p("layers", "plate_photo.webp"), np.dstack([photo, plate_a]))
    write_depth_png(project, depth)

    scene = dict(
        version=1,
        title=project.meta["title"],
        location=project.meta["location"],
        capture_year=project.meta.get("capture_year") or 2026,
        size=[Wo, Ho],
        eras=DEFAULT_ERAS,
        era_looks=DEFAULT_ERA_LOOKS,
        sky=dict(photo="layers/sky_photo.webp", illus=None),
        plate=dict(photo="layers/plate_photo.webp", illus=None, depth="layers/plate_depth.png", filled=False),
        layers=layers,
        birds=[],
    )
    if os.path.exists(project.p("scene.json")):
        prev = project.load_scene()
        for k in ("eras", "era_looks", "birds"):
            if prev.get(k):
                scene[k] = prev[k]
    project.save_scene(scene)
    project.mark("layers", count=len(layers))
    return scene


def occlusion_order(objs, inst, depth, ring=3):
    """Back-to-front order. Start from median depth, then fix any pair that touches so the one
    that is nearer *where they meet* is drawn later."""
    objs = sorted(objs, key=lambda o: -o["depth"])
    H, W = inst.shape
    kernel = np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)
    front_of = []  # (front idx, back idx)
    for o in objs:
        m = inst == o["idx"]
        bx = equirect_wrap_bbox(m)
        x0, y0, w, h = bx
        box = (x0 - ring, max(0, y0 - ring), min(w + 2 * ring, W), min(h + 2 * ring, H - max(0, y0 - ring)))
        sub_i = crop_wrap(inst, box)
        sub_m = sub_i == o["idx"]
        grown = cv2.dilate(sub_m.astype(np.uint8), kernel) > 0
        sub_d = crop_wrap(depth, box)
        for other in np.unique(sub_i[grown & ~sub_m]):
            if other == 0 or other < o["idx"]:
                continue  # each pair once
            om = sub_i == other
            near_me = om & grown                                     # other's pixels next to me
            near_it = sub_m & (cv2.dilate(om.astype(np.uint8), kernel) > 0)
            if near_me.sum() < 3 or near_it.sum() < 3:
                continue
            d_me, d_it = np.median(sub_d[near_it]), np.median(sub_d[near_me])
            if abs(d_me - d_it) < 0.05 * min(d_me, d_it):
                continue
            front_of.append((o["idx"], other) if d_me < d_it else (other, o["idx"]))
    order = [o["idx"] for o in objs]
    for _ in range(len(order)):  # move each front layer after its back layer until stable
        changed = False
        for f, b in front_of:
            i, j = order.index(f), order.index(b)
            if i < j:
                order.insert(j, order.pop(i))
                changed = True
        if not changed:
            break
    by_idx = {o["idx"]: o for o in objs}
    return [by_idx[i] for i in order]


def merge_seam(comp):
    """Join components that touch across the left/right edge of the equirect."""
    left, right = comp[:, 0], comp[:, -1]
    pairs = {(a, b) for a, b in zip(left, right) if a and b and a != b}
    if not pairs:
        return comp
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    lut = np.arange(comp.max() + 1)
    for k in parent:
        lut[k] = find(k)
    return lut[comp]


def decontaminate(rgb, core):
    """Give every pixel outside the object's core the colour of the nearest core pixel, so the
    soft, slightly grown edge never carries the colour of whatever was next to it."""
    if core.all() or not core.any():
        return rgb
    _, lab = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(core)
    out = rgb.copy()
    out[~core] = rgb[ys[lab[~core] - 1], xs[lab[~core] - 1]]
    return out


def soft_alpha(mask_crop, size, grow):
    a = cv2.resize(mask_crop, size, interpolation=cv2.INTER_LINEAR)
    a = (a > 0.5).astype(np.uint8)
    if grow:
        a = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), max(0.8, grow / 2))
    return (np.clip(a * 1.6, 0, 1) * 255).astype(np.uint8)


def paste_wrap(dst, bbox, src, op=None):
    """Write src into dst at bbox (x may wrap). If op is given, dst = op(dst, src) there."""
    x0, y0, w, h = bbox
    xs = (x0 + np.arange(w)) % dst.shape[1]
    region = dst[y0:y0 + h][:, xs]
    val = op(region, src.astype(dst.dtype)) if op else src
    dst[y0:y0 + h, xs] = val


def extend_sky(photo, sky_alpha):
    """Full-coverage sky: real sky where visible, smoothly extended behind everything else."""
    Ho, Wo = sky_alpha.shape
    sw, sh = 512, 256
    small = cv2.resize(photo, (sw, sh), interpolation=cv2.INTER_AREA)
    known = cv2.resize(sky_alpha, (sw, sh), interpolation=cv2.INTER_AREA) > 0.95
    if known.sum() < 10:
        return photo.copy()
    filled = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), (~known).astype(np.uint8) * 255, 9,
                         cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), (0, 0), 6)
    big = cv2.resize(filled, (Wo, Ho), interpolation=cv2.INTER_CUBIC)
    # keep real sky only where it is certainly sky, so nothing that stood against it ghosts through
    a = cv2.erode((sky_alpha > 0.99).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 1.5)[..., None]
    return (photo * a + big * (1 - a)).astype(np.uint8)


def write_depth_png(project, depth, width=2048):
    d = cv2.resize(depth, (width, width // 2), interpolation=cv2.INTER_NEAREST)
    v = np.clip(np.log(np.maximum(d, 1)) / np.log(SKY_DEPTH), 0, 1)  # 0 near .. 1 sky
    cv2.imwrite(project.p("layers", "plate_depth.png"), (v * 255).astype(np.uint8))


def present(layer, year):
    a, b = layer.get("lifespan") or [None, None]
    return (a is None or year >= a) and (b is None or year <= b)
