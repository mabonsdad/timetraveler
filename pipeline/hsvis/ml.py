"""Model-based backends. Run on a machine with an NVIDIA GPU and `pip install -r requirements-ml.txt`.

NOT YET RUN: these were written without GPU or model-hub access, so expect a round of fixes on
first use. The classical backends exercise the same interfaces and are tested.

All models run on overlapping *perspective* views (projection.overlap_views), never on the raw
equirect: they were trained on normal photos and the equirect's stretched poles confuse them.
Each equirect pixel takes its value from the view in which it is most central.

Licences to check before production: Mask2Former / Depth Anything V2-Small are permissive
(Depth Anything V2 Base/Large are CC-BY-NC); SDXL is OpenRAIL++; FLUX.1-dev is non-commercial.
"""
import cv2
import numpy as np
from PIL import Image

from .classes import GROUND, ID, map_vistas
from .projection import equirect_to_perspective, overlap_views, perspective_to_equirect
from .views import VIEW_FOV, align_disparity, centrality as _centrality, merge_views as _merge_views


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ---- segmentation -----------------------------------------------------------
def mask2former_segment(rgb, model="facebook/mask2former-swin-large-mapillary-vistas-semantic", size=1024):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    dev = _device()
    proc = AutoImageProcessor.from_pretrained(model)
    net = Mask2FormerForUniversalSegmentation.from_pretrained(model).to(dev).eval()
    lut = np.array([ID[map_vistas(net.config.id2label[i])] for i in range(len(net.config.id2label))], np.uint8)
    H, W = rgb.shape[:2]

    def run(yaw, pitch):
        view = equirect_to_perspective(rgb, yaw, pitch, VIEW_FOV, size)
        inputs = proc(images=Image.fromarray(view), return_tensors="pt").to(dev)
        with torch.no_grad():
            out = net(**inputs)
        seg = proc.post_process_semantic_segmentation(out, target_sizes=[(size, size)])[0]
        return lut[seg.cpu().numpy()]

    return _merge_views(W, H, run, size, np.uint8, cv2.INTER_NEAREST)


# ---- depth ---------------------------------------------------------------------
def depth_anything(rgb, labels, prior, model="depth-anything/Depth-Anything-V2-Small-hf", size=768):
    """Relative depth per view, each view's scale/shift fitted to the ground-plane prior on visible
    ground, so views agree with each other and come out roughly in metres."""
    import torch
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model=model, device=_device())
    H, W = rgb.shape[:2]
    ground = np.isin(labels, [ID[n] for n in GROUND]) & (prior < 200)

    def run(yaw, pitch):
        view = equirect_to_perspective(rgb, yaw, pitch, VIEW_FOV, size)
        pred = pipe(Image.fromarray(view))["predicted_depth"]
        disp = torch.nn.functional.interpolate(pred[None] if pred.ndim == 3 else pred[None, None],
                                               size=(size, size), mode="bilinear")[0, 0].cpu().numpy()
        pv = equirect_to_perspective(prior.astype(np.float32), yaw, pitch, VIEW_FOV, size, cv2.INTER_NEAREST)
        gv = equirect_to_perspective(ground.astype(np.uint8), yaw, pitch, VIEW_FOV, size, cv2.INTER_NEAREST) > 0
        return align_disparity(disp, pv, gv)  # None (no ground in view) -> keep the prior there

    d = _merge_views(W, H, run, size, np.float32, cv2.INTER_LINEAR, init=0)
    return np.minimum(np.where(d > 0, d, prior), 2000)


# ---- diffusion ---------------------------------------------------------------
NEGATIVE = "car, vehicle, person, people, sign, text, lamp post, telegraph pole, road markings, blurry"


class SDXLInpainter:
    def __init__(self, model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", steps=30, seed=0):
        import torch
        from diffusers import AutoPipelineForInpainting
        self.torch = torch
        self.pipe = AutoPipelineForInpainting.from_pretrained(model, torch_dtype=torch.float16).to(_device())
        self.steps, self.seed = steps, seed

    def inpaint(self, rgb, mask, prompt=None, labels=None):
        if not (mask > 0).any():
            return rgb
        h, w = mask.shape
        img = Image.fromarray(rgb).resize((1024, 1024), Image.LANCZOS)
        m = Image.fromarray(cv2.dilate(mask, np.ones((9, 9), np.uint8))).resize((1024, 1024), Image.NEAREST)
        g = self.torch.Generator(device=self.pipe.device).manual_seed(self.seed)
        res = self.pipe(prompt=f"{prompt or 'rural English landscape'}, natural photo, dawn light",
                        negative_prompt=NEGATIVE, image=img, mask_image=m, strength=0.99,
                        guidance_scale=7.0, num_inference_steps=self.steps, generator=g).images[0]
        res = np.array(res.resize((w, h), Image.LANCZOS))
        return np.where((mask > 0)[..., None], res, rgb)


class SDXLStyliser:
    """img2img in the house style. Train the LoRA on the illustrator's key frames first
    (e.g. with kohya_ss or diffusers' train_dreambooth_lora_sdxl.py) and pass --lora."""

    def __init__(self, lora=None, model="stabilityai/stable-diffusion-xl-base-1.0", strength=0.45,
                 prompt="hsvis style illustration of an English rural landscape at dawn", seed=0):
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline
        self.torch = torch
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model, torch_dtype=torch.float16).to(_device())
        if lora:
            self.pipe.load_lora_weights(lora)
        self.strength, self.prompt, self.seed = strength, prompt, seed

    def _one(self, rgb):
        h, w = rgb.shape[:2]
        s = 1024 / max(h, w)
        tw, th = max(8, int(w * s) // 8 * 8), max(8, int(h * s) // 8 * 8)
        g = self.torch.Generator(device=self.pipe.device).manual_seed(self.seed)
        res = self.pipe(prompt=self.prompt, image=Image.fromarray(rgb).resize((tw, th), Image.LANCZOS),
                        strength=self.strength, guidance_scale=6.0, generator=g).images[0]
        return np.array(res.resize((w, h), Image.LANCZOS))

    def __call__(self, rgb, bbox=None, alpha=None, ink=True):
        if bbox is not None:  # an object crop: small enough to treat as one image
            return self._one(rgb)
        # full equirect: stylise overlapping views and feather them together
        H, W = rgb.shape[:2]
        acc = np.zeros((H, W, 3), np.float32)
        wsum = np.zeros((H, W, 1), np.float32)
        for yaw, pitch in overlap_views():
            view = self._one(equirect_to_perspective(rgb, yaw, pitch, VIEW_FOV, 1024))
            vals, valid = perspective_to_equirect(view, yaw, pitch, VIEW_FOV, W, H)
            wgt = (np.clip(_centrality(W, H, yaw, pitch), 0, 1) ** 8 * valid)[..., None]
            acc += vals * wgt
            wsum += wgt
        return (acc / np.maximum(wsum, 1e-6)).astype(np.uint8)
