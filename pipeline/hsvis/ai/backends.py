"""Pipeline steps backed by hosted models (fal.ai, Google AI Studio).

  segment   sam3     SAM 3 on fal: one text prompt per class per view; best masks, ~$1/panorama
            gemini   Gemini 2.5 Flash masks: one call per view for all classes; cheap, coarser
  depth     fal      Depth Anything V2 on fal, aligned to the ground-plane prior per view
  fill      fal-flux FLUX.1 Fill [pro]: masked inpainting, prompt from the background class
            gemini / fal-edit   Nano Banana: remove what's painted magenta, composite inside the mask
  stylise   fal-flux FLUX img2img + style LoRA trained on the illustrator's key frames (train-style)
            gemini / fal-edit   Nano Banana restyle against reference key frames, no training
  describe  gemini   look at each building/wall/hedge layer and estimate when it was built

All of them run on perspective views or crops, cache every response under work/ai_cache, and stop
at the HSVIS_AI_BUDGET spending cap.
"""
import glob
import io
import os
import zipfile

import cv2
import numpy as np

from ..classes import GROUND, ID
from ..projection import equirect_to_perspective, overlap_views
from ..views import VIEW_FOV, align_disparity, blend_views, merge_views, tile_views
from .fal import Fal
from .gemini import Gemini
from .http import Session

# What to ask SAM 3 for, per class (several prompts per class are OR-ed together, at 1 call each:
# add synonyms only where a class is being missed). SAM 3 matches concepts, so "vehicle" catches
# cars and vans and "pole" catches telegraph poles and lamp posts.
SAM3_PROMPTS = {
    "sky": ["sky"], "far": ["distant hills"], "terrain": ["grass"], "water": ["water"],
    "road": ["road"], "path": ["footpath"], "vegetation": ["tree"], "hedge": ["hedge"],
    "wall": ["wall or fence"], "building": ["building"], "lane_marking": ["road marking"],
    "sign": ["road sign"], "pole": ["pole"], "vehicle": ["vehicle"], "person": ["person"],
}
# Later classes win where masks overlap: small things in front over big things behind.
PRIORITY = ["far", "sky", "terrain", "water", "road", "path", "vegetation", "hedge", "wall",
            "building", "lane_marking", "sign", "pole", "vehicle", "person"]
UNKNOWN = 255


PROVIDER = {"sam3": "fal", "fal": "fal", "fal-flux": "fal", "fal-edit": "fal", "gemini": "gemini"}


def session_for(project, dry_run=False, backend=None):
    """A caching, budgeted session. Checks the backend's key up front (unless dry-running), so a
    missing key fails before any work is done rather than halfway through."""
    if backend and not dry_run:
        from . import config
        config.fal_key() if PROVIDER[backend] == "fal" else config.gemini_key()
    return Session(project.p("work", "ai_cache"), dry_run=dry_run)


def _progress(i, n, what):
    print(f"  {what} {i + 1}/{n}", end="\r" if i + 1 < n else "\n", flush=True)


# ---- segmentation -------------------------------------------------------------------------
def segment_sam3(rgb, base, session, classes=None, size=1024):
    fal = Fal(session)
    classes = [c for c in PRIORITY if c in (classes or SAM3_PROMPTS)]
    views = overlap_views()
    H, W = rgb.shape[:2]
    state = {"i": 0}

    def run(yaw, pitch):
        view = equirect_to_perspective(rgb, yaw, pitch, VIEW_FOV, size)
        lab = np.full((size, size), UNKNOWN, np.uint8)
        for c in classes:
            m = np.zeros((size, size), bool)
            for prompt in SAM3_PROMPTS[c]:
                m |= fal.sam3(view, prompt)
            lab[m] = ID[c]
        _progress(state["i"], len(views), "SAM 3 view")
        state["i"] += 1
        return lab

    merged = merge_views(W, H, run, size, np.uint8, cv2.INTER_NEAREST, views=views, init=UNKNOWN)
    return np.where(merged == UNKNOWN, base, merged).astype(np.uint8)


GEMINI_LABELS = {  # what Gemini is asked for -> our class
    "sky": "sky", "distant hills": "far", "grass or field": "terrain", "water": "water",
    "road": "road", "footpath": "path", "tree": "vegetation", "hedge": "hedge", "wall or fence": "wall",
    "building": "building", "road marking": "lane_marking", "road sign": "sign",
    "pole or lamp post": "pole", "vehicle": "vehicle", "person": "person",
}


def segment_gemini(rgb, base, session, size=1024):
    gem = Gemini(session)
    views = overlap_views()
    H, W = rgb.shape[:2]
    rank = {c: i for i, c in enumerate(PRIORITY)}
    state = {"i": 0}

    def run(yaw, pitch):
        view = equirect_to_perspective(rgb, yaw, pitch, VIEW_FOV, size)
        lab = np.full((size, size), UNKNOWN, np.uint8)
        found = [(GEMINI_LABELS.get(l.lower().strip()), m) for l, m in gem.segment(view, list(GEMINI_LABELS))]
        for c, m in sorted((f for f in found if f[0]), key=lambda f: rank[f[0]]):
            lab[m] = ID[c]
        _progress(state["i"], len(views), "Gemini view")
        state["i"] += 1
        return lab

    merged = merge_views(W, H, run, size, np.uint8, cv2.INTER_NEAREST, views=views, init=UNKNOWN)
    return np.where(merged == UNKNOWN, base, merged).astype(np.uint8)


# ---- depth --------------------------------------------------------------------------------
def depth_fal(rgb, labels, prior, session, size=1024):
    fal = Fal(session)
    H, W = rgb.shape[:2]
    ground = np.isin(labels, [ID[n] for n in GROUND]) & (prior < 200)
    views = overlap_views()
    state = {"i": 0}

    def run(yaw, pitch):
        view = equirect_to_perspective(rgb, yaw, pitch, VIEW_FOV, size)
        disp = fal.depth(view)
        _progress(state["i"], len(views), "depth view")
        state["i"] += 1
        if disp is None:
            return None
        pv = equirect_to_perspective(prior.astype(np.float32), yaw, pitch, VIEW_FOV, size, cv2.INTER_NEAREST)
        gv = equirect_to_perspective(ground.astype(np.uint8), yaw, pitch, VIEW_FOV, size, cv2.INTER_NEAREST) > 0
        return align_disparity(disp, pv, gv)

    d = merge_views(W, H, run, size, np.float32, cv2.INTER_LINEAR, views=views, init=0)
    return np.minimum(np.where(d > 0, d, prior), 2000)


# ---- fill ---------------------------------------------------------------------------------
class _GenerativeInpainter:
    """Shared behaviour: composite the model's output only inside the (feathered) mask."""
    generative = True

    def inpaint(self, rgb, mask, prompt=None, labels=None):
        m = mask > 0
        if not m.any():
            return rgb
        out = self._fill(rgb, cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0,
                         prompt or "rural English countryside")
        a = cv2.GaussianBlur(cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.5)
        a = np.maximum(a, m)[..., None]
        return (out * a + rgb * (1 - a)).astype(np.uint8)


class FalFluxInpainter(_GenerativeInpainter):
    def __init__(self, session):
        self.fal = Fal(session)

    def _fill(self, rgb, m, prompt):
        return self.fal.flux_fill(rgb, m, f"{prompt}, natural photograph, seamless, soft early morning light")


MAGENTA = np.array([255, 0, 255], np.uint8)


class EditInpainter(_GenerativeInpainter):
    """Nano Banana has no mask input: paint the hole magenta and ask for it to be replaced."""

    def __init__(self, session, via="google"):
        self.client = Gemini(session) if via == "google" else Fal(session)
        self.via = via

    def _fill(self, rgb, m, prompt):
        marked = rgb.copy()
        marked[m] = MAGENTA
        instruction = ("Remove everything painted solid magenta and replace it with what would naturally be "
                       f"there: {prompt}. Match the surrounding light, perspective and texture. Change nothing "
                       "outside the magenta area, keep the exact framing, and return a photograph.")
        if self.via == "google":
            return self.client.edit_image([marked], instruction)
        return self.client.edit([marked], instruction)


def get_inpainter(backend, project, dry_run=False):
    s = session_for(project, dry_run, backend)
    if backend == "fal-flux":
        return FalFluxInpainter(s), s
    if backend in ("gemini", "fal-edit"):
        return EditInpainter(s, "google" if backend == "gemini" else "fal"), s
    raise SystemExit(f"unknown AI fill backend {backend}")


# ---- stylise ------------------------------------------------------------------------------
DEFAULT_STYLE = ("a delicate hand-drawn illustration: fine sepia ink linework with soft watercolour washes, "
                 "muted earthy palette, visible paper grain, in the manner of an early 20th-century natural "
                 "history book plate")


class TiledStyliser:
    """Restyles a whole equirect by overlapping views, feathered together. stylise.py calls this
    once for the capture and once for the filled plate, then cuts object layers from the result,
    so every layer shares one consistent illustration."""
    tiled = True

    def __init__(self, process, fov_deg=90, size=1024, name="style", session=None):
        self.process, self.fov_deg, self.size, self.name = process, fov_deg, size, name
        self.dry_run = bool(session and session.dry_run)

    def __call__(self, rgb, only=None, **_):
        """Restyle an equirect. With `only` (bool mask), views that don't see any of it are skipped."""
        views = tile_views(self.fov_deg)
        fov = np.radians(self.fov_deg)
        small_only = None if only is None else cv2.resize(only.astype(np.uint8), (1024, 512), interpolation=cv2.INTER_NEAREST)

        def needed(i):
            if small_only is None:
                return True
            yaw, pitch = views[i]
            return equirect_to_perspective(small_only, yaw, pitch, fov, 128, cv2.INTER_NEAREST).any()

        if self.dry_run:  # just count the calls, skip the full-resolution projection work
            blank = np.zeros((8, 8, 3), np.uint8)
            n = sum(1 for i in range(len(views)) if needed(i) and self.process(blank, i) is not None)
            print(f"  {self.name}: {n} of {len(views)} tiles")
            return rgb
        print(f"  {self.name}: up to {len(views)} tiles of {self.size}px at {self.fov_deg} deg")

        def proc(view, i):
            return self.process(view, i) if needed(i) else view

        return blend_views(rgb, proc, self.fov_deg, self.size,
                           on_view=lambda i, n: _progress(i, n, f"{self.name} tile"))

    def single(self, rgb):
        """Restyle one ordinary (perspective-ish) image, e.g. an object crop."""
        h, w = rgb.shape[:2]
        s = self.size / max(h, w)
        small = cv2.resize(rgb, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        side = max(small.shape[:2])
        square = np.zeros((side, side, 3), np.uint8)
        square[:small.shape[0], :small.shape[1]] = small
        out = self.process(square, -1)[:small.shape[0], :small.shape[1]]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


def load_refs(folder, max_refs=4, size=1024):
    if not folder:
        return []
    paths = sorted(p for ext in ("jpg", "jpeg", "png", "webp") for p in glob.glob(os.path.join(folder, f"*.{ext}")))
    refs = []
    for p in paths[:max_refs]:
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        s = size / max(im.shape[:2])
        refs.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im)
    return refs


def get_styliser(backend, project, dry_run=False, refs_dir=None, lora_url=None, strength=0.4,
                 style=None, tile_fov=90, tile_size=1024):
    s = session_for(project, dry_run, backend)
    style = style or DEFAULT_STYLE
    if backend == "fal-flux":
        fal = Fal(s)
        trigger = (project.meta.get("style", {}) or {}).get("trigger", "")
        lora_url = lora_url or (project.meta.get("style", {}) or {}).get("lora_url")
        if not lora_url:
            print("  no style LoRA given (--lora-url or `hsvis train-style`): using the text style prompt only")
        prompt = f"{trigger} {style}".strip()
        proc = lambda view, i: fal.flux_img2img(view, prompt, strength, lora_url)  # noqa: E731
        return TiledStyliser(proc, tile_fov, tile_size, "FLUX restyle", s), s
    if backend in ("gemini", "fal-edit"):
        refs = load_refs(refs_dir)
        client = Gemini(s) if backend == "gemini" else Fal(s)
        edit = client.edit_image if backend == "gemini" else client.edit
        anchor = {}

        def proc(view, i):
            # no reference frames yet: the first tile we style becomes the reference for the rest
            use = refs or ([anchor["img"]] if "img" in anchor else [])
            ref_txt = "the reference image(s) that follow" if use else "this description"
            prompt = (f"Repaint the first image as an illustration in exactly the style of {ref_txt}: {style}. "
                      "Keep the composition, perspective, horizon line and every object's position and outline "
                      "identical. Do not add, remove or move anything. Return only the repainted first image.")
            out = edit([view] + use, prompt)
            anchor.setdefault("img", out)
            return out

        return TiledStyliser(proc, tile_fov, tile_size, "Nano Banana restyle", s), s
    raise SystemExit(f"unknown AI stylise backend {backend}")


def train_style(project, frames_dir, trigger="hsvisstyle", steps=1000, dry_run=False):
    """Zip the illustrator's key frames, train a FLUX LoRA on fal, store its URL in project.json."""
    paths = sorted(p for ext in ("jpg", "jpeg", "png", "webp") for p in glob.glob(os.path.join(frames_dir, f"*.{ext}")))
    if len(paths) < 5:
        raise SystemExit(f"need at least 5 key frames in {frames_dir} (10-20 is better); found {len(paths)}")
    s = session_for(project, dry_run, "fal")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in paths:
            z.write(p, os.path.basename(p))
            caption = os.path.splitext(p)[0] + ".txt"
            if os.path.exists(caption):
                z.write(caption, os.path.basename(caption))
    url = Fal(s).train_lora(buf.getvalue(), trigger, steps)
    if url:
        project.meta["style"] = {"lora_url": url, "trigger": trigger, "frames": len(paths)}
        project.save()
    return url, s


# ---- describe / date layers ---------------------------------------------------------------
DATE_CLASSES = {"building", "wall", "hedge", "pole", "sign", "vegetation"}
DATE_PROMPT = """You are helping reconstruct how a place in the UK looked in past centuries.
Image 1 is a view of the whole scene; image 2 is a close crop of one feature, outlined in magenta.
Feature class: {cls}. Location: {loc}.
Describe the outlined feature and estimate when it first existed at this spot, using visible evidence
(building materials, architectural style, roof, windows, hedge structure, tree size, signage design).
Reply as JSON: {{"description": str, "evidence": [str], "earliest_year": int|null, "latest_year": int|null,
"confidence": "low"|"medium"|"high", "notes": str}}. earliest/latest bound the likely construction or
planting date. Use null when there is no basis for a date. Do not invent documentary sources."""


def describe_layers(project, apply=False, dry_run=False, classes=None):
    from ..project import read_rgba
    from ..projection import crop_wrap
    s = session_for(project, dry_run, "gemini")
    gem = Gemini(s)
    scene = project.load_scene()
    W, H = scene["size"]
    src = project.load_source((W, H))
    overview = cv2.resize(src, (2048, 1024), interpolation=cv2.INTER_AREA)
    loc = scene.get("location") or {}
    loc_txt = (f"lat {loc['lat']:.4f}, lon {loc['lon']:.4f}" if loc.get("lat") is not None else "unknown")
    todo = [l for l in scene["layers"] if l["class"] in (classes or DATE_CLASSES)]
    for i, layer in enumerate(todo):
        x0, y0, w, h = layer["bbox"]
        pad = int(max(w, h) * 0.4) + 16
        box = (x0 - pad, max(0, y0 - pad), w + 2 * pad, min(H - max(0, y0 - pad), h + 2 * pad))
        crop = crop_wrap(src, box).copy()
        a = read_rgba(project.p(layer["photo"]))[..., 3]
        m = np.zeros(crop.shape[:2], np.uint8)
        oy, ox = y0 - box[1], pad
        m[oy:oy + h, ox:ox + w] = (a > 127).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, (255, 0, 255), max(2, crop.shape[1] // 300))
        if max(crop.shape[:2]) > 1024:
            sc = 1024 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        est = gem.ask_json(DATE_PROMPT.format(cls=layer["class"], loc=loc_txt), [overview, crop])
        _progress(i, len(todo), "describing layer")
        if not est:
            continue
        layer["ai_estimate"] = est
        if apply and layer.get("lifespan_source") == "default" and est.get("earliest_year"):
            layer["lifespan"] = [int(est["earliest_year"]), layer["lifespan"][1]]
            layer["lifespan_source"] = "ai-estimate"
    if not dry_run:
        project.save_scene(scene)
    return todo, s
