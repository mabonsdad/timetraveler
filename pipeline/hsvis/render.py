"""Offline compositing: what the player shows, as still images.

Used for stage plates (a fixed-POV LED volume can simply play the composited equirect, or a crop
of it for a curved wall), for checking results, and for bid material (contact sheets).
"""
import cv2
import numpy as np

from .layers import present
from .project import read_rgba, write_rgb
from .projection import equirect_to_perspective


def era_look(rgb, look):
    """Apply an era look (sepia/saturation/tint) - same maths as the player's shader."""
    f = rgb.astype(np.float32) / 255
    lum = (f @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    f = lum + (f - lum) * look.get("sat", 1)
    sep = lum * np.array([1.07, 0.95, 0.78], np.float32)
    s = look.get("sepia", 0)
    f = f * (1 - s) + sep * s
    f = f * np.array(look.get("tint", [1, 1, 1]), np.float32)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def composite(project, scene, year, style=0.0, width=None):
    """style 0 = photo, 1 = illustration, in between = blend."""
    W, H = scene["size"]
    cache = {}

    def load(entry, rgba=True):
        key = (entry.get("photo"), entry.get("illus"))
        if key in cache:
            return cache[key]
        ph = read_rgba(project.p(entry["photo"])).astype(np.float32)
        if style > 0 and entry.get("illus"):
            il = read_rgba(project.p(entry["illus"])).astype(np.float32)
            ph[..., :3] = ph[..., :3] * (1 - style) + il[..., :3] * style
        cache[key] = ph
        return ph

    out = load(scene["sky"])[..., :3].copy()
    over(out, load(scene["plate"]), (0, 0, W, H))
    for layer in scene["layers"]:
        if present(layer, year):
            over(out, load(layer), layer["bbox"])
    look = scene.get("era_looks", {}).get(str(year), {})
    out = era_look(out.astype(np.uint8), look)
    if width and width != W:
        out = cv2.resize(out, (width, width // 2), interpolation=cv2.INTER_AREA)
    return out


def over(dst, rgba, bbox):
    x0, y0, w, h = bbox
    xs = (x0 + np.arange(w)) % dst.shape[1]
    a = rgba[..., 3:4] / 255
    region = dst[y0:y0 + h][:, xs]
    dst[y0:y0 + h, xs] = region * (1 - a) + rgba[..., :3] * a


def draw_birds(img, scene, year):
    H, W = img.shape[:2]
    for b in scene.get("birds", []):
        if b["era"] != year:
            continue
        x = int((b["yaw"] + 180) / 360 * W)
        y = int((90 - b["pitch"]) / 180 * H)
        r = max(3, int(W / 600))
        cv2.circle(img, (x, y), r + 2, (255, 255, 255), -1)
        cv2.circle(img, (x, y), r, (200, 40, 40), -1)
    return img


def render(project, year, out, style=0.0, width=None, view=None, birds=False):
    scene = project.load_scene()
    img = composite(project, scene, year, style, None if view else width)
    if birds:
        img = draw_birds(img, scene, year)
    if view:  # (yaw_deg, pitch_deg, fov_deg, size)
        yaw, pitch, fov, size = view
        img = equirect_to_perspective(img, np.radians(yaw), np.radians(pitch), np.radians(fov), int(size))
    write_rgb(out, img)
    return img


def contact_sheet(project, out, width=1600, birds=True):
    """Rows = eras; columns = photo / illustration."""
    scene = project.load_scene()
    rows = []
    for year in scene["eras"]:
        cells = []
        for style in (0.0, 1.0):
            img = composite(project, scene, year, style, width)
            if birds:
                img = draw_birds(img, scene, year)
            label = f"{year}  {'photo' if style == 0 else 'illustration'}"
            cv2.putText(img, label, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(img, label, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(img)
        rows.append(np.hstack(cells))
    write_rgb(out, np.vstack(rows), quality=88)
