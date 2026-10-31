"""Step 5: photo -> house illustration style, applied per layer so every layer stays editable.

Consistency across layers (so a cut-out house matches the plate it sits on) comes from:
  * one colour palette learned from the whole panorama and reused for every layer,
  * one paper texture in panorama coordinates, sampled at each layer's own position.

Backends:
  opencv  edge-preserving wash + shared palette + ink lines + paper grain. A placeholder look,
          good enough to design transitions and test on the wall.
  sdxl    diffusion img2img with a style LoRA trained on the commissioned illustrator's key frames,
          on perspective tiles (GPU machine; see ml.py).
  fal-flux / gemini / fal-edit   the same idea on hosted models (see ai/backends.py and
          stylise_tiled below): FLUX + LoRA on fal.ai, or Nano Banana with reference key frames.
"""
import cv2
import numpy as np

from .project import read_rgba, write_rgb
from .projection import crop_wrap


class Painterly:
    def __init__(self, palette_src, size, colours=24, seed=7):
        self.W, self.H = size
        self.palette = learn_palette(palette_src, colours)
        self.paper = paper_texture(size, seed)

    def __call__(self, rgb, bbox=None, alpha=None, ink=True):
        h, w = rgb.shape[:2]
        smooth = cv2.edgePreservingFilter(rgb, flags=cv2.RECURS_FILTER, sigma_s=50, sigma_r=0.35)
        quant = quantise(smooth, self.palette)
        out = 0.55 * quant.astype(np.float32) + 0.45 * smooth.astype(np.float32)
        # lift shadows, soften contrast: reads as wash rather than photo
        out = 255 * (0.08 + 0.88 * (out / 255) ** 0.9)
        if ink:
            g = cv2.GaussianBlur(cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY), (0, 0), 1.2)
            edges = cv2.Canny(g, 35, 90)
            if alpha is not None:  # outline the layer's silhouette too
                edges |= cv2.Canny((alpha > 128).astype(np.uint8) * 255, 50, 150)
            k = max(1, round(self.W / 4096))
            edges = cv2.dilate(edges, np.ones((k, k), np.uint8))
            e = cv2.GaussianBlur(edges.astype(np.float32) / 255, (0, 0), 0.7 * k)[..., None]
            ink_col = np.array([55, 42, 34], np.float32)
            out = out * (1 - 0.7 * e) + ink_col * 0.7 * e
        paper = self.paper if bbox is None else crop_wrap(self.paper, bbox)
        out = out * paper[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


def learn_palette(rgb, k):
    small = cv2.resize(rgb, (512, 256), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centres = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return centres


def quantise(rgb, palette_lab):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = np.empty_like(lab)
    for y in range(0, lab.shape[0], 256):
        blk = lab[y:y + 256]
        d = ((blk[..., None, :] - palette_lab[None, None]) ** 2).sum(-1)
        out[y:y + 256] = palette_lab[d.argmin(-1)]
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2RGB)


def paper_texture(size, seed):
    W, H = size
    rng = np.random.default_rng(seed)
    acc = np.zeros((H, W), np.float32)
    for octave, amp in [(256, 0.5), (64, 0.3), (8, 0.2)]:
        sw, sh = max(4, W // octave), max(2, H // octave)
        n = rng.random((sh, sw)).astype(np.float32)
        n = np.hstack([n, n[:, :1]])  # wrap seamlessly at +-180
        big = cv2.resize(n, (W + W // sw, H), interpolation=cv2.INTER_CUBIC)[:, :W]
        acc += amp * big
    return 0.93 + 0.1 * acc


AI_BACKENDS = ("fal-flux", "gemini", "fal-edit")


def stylise(project, backend="opencv", dry_run=False, **kw):
    if backend in AI_BACKENDS:
        return stylise_tiled(project, backend, dry_run, **kw)
    scene = project.load_scene()
    size = project.out_size
    if backend == "opencv":
        style = Painterly(project.load_source((1024, 512)), size)
    elif backend == "sdxl":
        from .ml import SDXLStyliser
        style = SDXLStyliser(**kw)
    else:
        raise SystemExit(f"unknown stylise backend {backend}")

    sky = read_rgba(project.p(scene["sky"]["photo"]))[..., :3]
    write_rgb(project.p("layers", "sky_illus.webp"), style(sky, ink=False))
    scene["sky"]["illus"] = "layers/sky_illus.webp"

    plate = read_rgba(project.p(scene["plate"]["photo"]))
    plate[..., :3] = style(plate[..., :3])
    write_rgb(project.p("layers", "plate_illus.webp"), plate)
    scene["plate"]["illus"] = "layers/plate_illus.webp"

    for layer in scene["layers"]:
        rgba = read_rgba(project.p(layer["photo"]))
        rgba[..., :3] = style(rgba[..., :3], bbox=layer["bbox"], alpha=rgba[..., 3])
        path = f"layers/{layer['id']}_illus.webp"
        write_rgb(project.p(path), rgba)
        layer["illus"] = path

    project.save_scene(scene)
    project.mark("stylise", backend=backend)


def stylise_tiled(project, backend, dry_run=False, **kw):
    """Hosted generative restyle. The capture is restyled once and every layer is cut from that one
    illustration, so layers match exactly. Only areas the capture doesn't show (filled plate, the
    hidden parts of completed objects) are restyled separately."""
    from .ai.backends import get_styliser
    from .classes import ID
    from .layers import extend_sky
    from .segment import load_labels

    scene = project.load_scene()
    style, session = get_styliser(backend, project, dry_run, **kw)
    W, H = project.out_size
    src = project.load_source((W, H))
    illus = style(src)

    # exact masks from earlier steps (comparing pixels would trip on lossy WebP noise)
    holes = cv2.imread(project.p("work", "holes.png"), cv2.IMREAD_GRAYSCALE) > 0
    labels = cv2.resize(load_labels(project), (W, H), interpolation=cv2.INTER_NEAREST)
    plate = read_rgba(project.p(scene["plate"]["photo"]))
    filled = holes & (plate[..., 3] > 0)
    plate_il = illus
    if filled.any():
        extra = style(plate[..., :3], only=cv2.dilate(filled.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
        a = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 3)[..., None]
        plate_il = (extra * a + illus * (1 - a)).astype(np.uint8)

    # sky: real sky from the illustration, extended behind everything else
    sky_il = extend_sky(illus, (labels == ID["sky"]).astype(np.float32))

    layer_out = []
    for layer in scene["layers"]:
        rgba = read_rgba(project.p(layer["photo"]))
        il = crop_wrap(illus, layer["bbox"]).copy()
        if layer.get("completed_behind"):
            before = read_rgba(project.p(layer["photo"].replace("_completed_photo", "_photo")))[..., 3]
            hidden = rgba[..., 3].astype(np.int16) > before.astype(np.int16) + 64
            if hidden.any():
                a = cv2.GaussianBlur(hidden.astype(np.float32), (0, 0), 2)[..., None]
                il = (style.single(rgba[..., :3]) * a + il * (1 - a)).astype(np.uint8)
        layer_out.append((layer, np.dstack([il, rgba[..., 3]])))

    print(f"  {session.summary()}")
    if dry_run:
        return
    write_rgb(project.p("layers", "sky_illus.webp"), sky_il)
    scene["sky"]["illus"] = "layers/sky_illus.webp"
    write_rgb(project.p("layers", "plate_illus.webp"), np.dstack([plate_il, plate[..., 3]]))
    scene["plate"]["illus"] = "layers/plate_illus.webp"
    for layer, rgba in layer_out:
        path = f"layers/{layer['id']}_illus.webp"
        write_rgb(project.p(path), rgba)
        layer["illus"] = path
    project.save_scene(scene)
    project.mark("stylise", backend=backend)
