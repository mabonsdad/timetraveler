"""Ray-cast a synthetic rural-lane equirect panorama plus an exact class-label PNG.

Lets the whole pipeline be exercised before real Insta360 captures exist.
    python tools/make_test_pano.py out_dir [--width 4096]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hsvis.classes import CLASSES, ID  # noqa: E402
from hsvis.projection import lonlat_grid, lonlat_to_dir  # noqa: E402

CAM_H = 1.8
CAM_X = 2.2  # camera stands at the lane edge, not on the centre line
SUN = np.array([-0.6, 0.25, 0.75])
SUN /= np.linalg.norm(SUN)


def noise(shape, scale, seed):
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, shape[0] // scale), max(2, shape[1] // scale))).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def hit_box(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d
        t2 = (hi - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
    t = np.where((tmax >= tmin) & (tmax > 0), np.where(tmin > 0, tmin, np.inf), np.inf)
    # normal: axis of tmin
    axis = np.argmax(np.minimum(t1, t2), axis=-1)
    n = np.zeros_like(d)
    np.put_along_axis(n, axis[..., None], -np.sign(np.take_along_axis(d, axis[..., None], -1)), -1)
    return t, n


def hit_sphere(o, d, c, r):
    oc = o - c
    b = (d * oc).sum(-1)
    disc = b * b - ((oc * oc).sum() - r * r)
    t = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
    t = np.where(t > 0, t, np.inf)
    p = o + d * t[..., None]
    n = (p - c) / r
    return t, n


def hit_cyl(o, d, cx, cz, r, y0, y1):
    ox, oz = o[0] - cx, o[2] - cz
    a = d[..., 0] ** 2 + d[..., 2] ** 2
    b = 2 * (ox * d[..., 0] + oz * d[..., 2])
    c = ox * ox + oz * oz - r * r
    disc = b * b - 4 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    y = o[1] + d[..., 1] * t
    t = np.where((disc > 0) & (t > 0) & (y >= y0) & (y <= y1), t, np.inf)
    p = o + d * t[..., None]
    n = np.stack([p[..., 0] - cx, np.zeros_like(t), p[..., 2] - cz], -1) / r
    return t, n


def scene():
    """List of (kind, params, class, base colour)."""
    g = -CAM_H
    S = []
    # houses (two sides of the lane) with roofs as separate thin boxes
    for (x, z, w, h, dpt, col) in [(11, 22, 8, 6, 9, (170, 120, 95)), (-14, 40, 10, 7, 8, (200, 190, 170)),
                                   (12, -30, 7, 5.5, 8, (150, 100, 80)), (-30, -12, 9, 6, 9, (185, 160, 130))]:
        S.append(("box", ((x - w / 2, g, z - dpt / 2), (x + w / 2, g + h, z + dpt / 2)), "building", col))
        S.append(("box", ((x - w / 2 - .4, g + h, z - dpt / 2 - .4), (x + w / 2 + .4, g + h + .5, z + dpt / 2 + .4)),
                  "building", (90, 60, 55)))
    # cars on the lane
    for (x, z, col) in [(1.4, 12, (40, 60, 160)), (-1.4, -18, (180, 30, 30)), (1.4, 45, (220, 220, 225))]:
        S.append(("box", ((x - .9, g + .2, z - 2.1), (x + .9, g + 1.45, z + 2.1)), "vehicle", col))
    # telegraph poles + streetlight + sign
    for z in (-60, -30, 0, 30, 60, 90):
        S.append(("cyl", (4.2, z + 7, .14, g, g + 8.5), "pole", (95, 75, 55)))
    S.append(("box", ((-4.6, g + 2.0, 8.0), (-3.4, g + 2.9, 8.1)), "sign", (230, 230, 230)))
    S.append(("cyl", (-4.0, 8.05, .05, g, g + 2.0), "sign", (120, 120, 120)))
    # a person on the verge
    S.append(("cyl", (3.3, 6.0, .25, g, g + 1.75), "person", (60, 70, 90)))
    # hedges along the lane, with gaps for gateways
    for side in (-1, 1):
        for (z0, z1) in [(-120, -40), (-34, 14), (20, 60), (66, 160)]:
            x = side * 4.8
            S.append(("box", ((x - .6, g, z0), (x + .6, g + 1.6, z1)), "hedge", (70, 110, 45)))
    # wall around a garden
    S.append(("box", ((6, g, 15), (16.5, g + 1.1, 15.4)), "wall", (160, 150, 130)))
    # trees
    rng = np.random.default_rng(3)
    for (x, z, r) in [(9, 2, 3.2), (-9, 18, 4), (20, 60, 5), (-22, 70, 5.5), (-8, -40, 3.5), (30, 10, 4.5),
                      (-35, 25, 6), (25, -55, 5), (-18, -70, 6), (45, 35, 6), (8, 90, 4)]:
        hgt = r * 1.6 + rng.random() * 2
        S.append(("cyl", (x, z, r * .12, g, g + hgt), "vegetation", (80, 60, 40)))
        S.append(("sph", ((x, g + hgt + r * .6, z), r), "vegetation", (55, 105, 40)))
    return S


def render(W):
    H = W // 2
    lon, lat = lonlat_grid(W, H)
    d = lonlat_to_dir(lon, lat).astype(np.float32)
    o = np.array([CAM_X, 0.0, 0.0])
    tbest = np.full((H, W), np.inf)
    nbest = np.zeros((H, W, 3))
    cls = np.full((H, W), ID["sky"], dtype=np.uint8)
    col = np.zeros((H, W, 3))

    # ground plane
    with np.errstate(divide="ignore"):
        tg = np.where(d[..., 1] < 0, -CAM_H / d[..., 1], np.inf)
    p = o + d * tg[..., None]
    upd = tg < np.inf
    tbest[upd] = tg[upd]
    nbest[upd] = (0, 1, 0)
    px, pz = p[..., 0], p[..., 2]
    road = (np.abs(px) < 3.0) & upd
    lane = road & (np.abs(px) < .08) & (np.mod(np.nan_to_num(pz, posinf=0), 9) < 4.5)
    pond = ((px + 18) ** 2 + (pz - 8) ** 2 < 36) & upd
    path = (np.abs(pz - 17) < .8) & (px > 3) & (px < 11) & upd
    cls[upd] = ID["terrain"]
    col[upd] = (95, 135, 55)
    cls[road] = ID["road"]; col[road] = (85, 85, 88)
    cls[lane] = ID["lane_marking"]; col[lane] = (235, 235, 225)
    cls[path] = ID["path"]; col[path] = (175, 155, 120)
    cls[pond] = ID["water"]; col[pond] = (60, 90, 120)

    for kind, prm, c, base in scene():
        if kind == "box":
            t, n = hit_box(o, d, np.array(prm[0]), np.array(prm[1]))
        elif kind == "sph":
            t, n = hit_sphere(o, d, np.array(prm[0]), prm[1])
        else:
            t, n = hit_cyl(o, d, *prm)
        m = t < tbest
        tbest[m] = t[m]
        nbest[m] = n[m]
        cls[m] = ID[c]
        col[m] = base

    # far hills: anything still unhit (or very distant ground) below a silhouette is "far"
    sil = np.deg2rad(2.0 + 2.5 * np.sin(lon * 3 + 1) + 1.5 * np.sin(lon * 7))
    far = ((tbest == np.inf) & (lat < sil)) | ((tbest > 900) & (tbest < np.inf))
    cls[far] = ID["far"]
    col[far] = (110, 115, 140)
    tbest[far] = 3000
    nbest[far] = (0, 0, 1)

    # shading
    lam = np.clip((nbest * SUN).sum(-1), 0, 1)
    shade = 0.55 + 0.6 * lam
    tex = 0.85 + 0.3 * noise((H, W), 3, 1)
    rgb = col * (shade * tex)[..., None]
    fog = 1 - np.exp(-np.minimum(tbest, 3000) / 900.0)
    sky = cls == ID["sky"]
    # dawn sky gradient
    t = np.clip(lat / (np.pi / 2), 0, 1)[..., None]
    glow = np.exp(-((lon + 0.6) ** 2) / 0.5)[..., None] * np.exp(-lat / 0.25)[..., None]
    skycol = (1 - t) * np.array([235, 200, 170]) + t * np.array([90, 130, 190]) + glow * np.array([20, 25, 0])
    skycol = np.clip(skycol * (0.97 + 0.06 * noise((H, W), 40, 2))[..., None], 0, 255)
    rgb = rgb * (1 - fog[..., None] * 0.6) + np.array([210, 195, 190]) * fog[..., None] * 0.6
    rgb[sky] = skycol[sky]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    labels = np.zeros((H, W, 3), np.uint8)
    for name, info in CLASSES.items():
        labels[cls == ID[name]] = info["colour"]
    return rgb, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=4096)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rgb, labels = render(a.width)
    cv2.imwrite(os.path.join(a.out, "test_pano.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(a.out, "test_labels.png"), cv2.cvtColor(labels, cv2.COLOR_RGB2BGR))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
