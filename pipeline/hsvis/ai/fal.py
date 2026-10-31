"""fal.ai client (queue API) and the handful of models the pipeline uses.

Request flow: POST https://queue.fal.run/<model> -> poll status_url -> GET response_url, with
`Authorization: Key $FAL_KEY`. Images go up as data URIs (or via fal storage when large) and
result image URLs are downloaded straight away and stored inline, so cached results never point
at expired links.

Input/output field names follow fal's published schemas as of late 2026; when fal changes a
schema, the fix belongs in the small helper for that model below.
"""
import base64
import json
import time

import cv2
import numpy as np

from . import config
from .http import AIError, data_uri, fetch_image, request

DATA_URI_LIMIT = 8 * 1024 * 1024  # larger payloads go through fal storage


class Fal:
    def __init__(self, session):
        self.s = session
        self._key = None

    @property
    def headers(self):
        if self._key is None:
            self._key = config.fal_key()
        return {"Authorization": f"Key {self._key}"}

    # ---- core ----------------------------------------------------------------
    def run(self, price_name, app, args, inline=True, timeout=900):
        """Run a model through the queue. Returns the result JSON, or None in a dry run."""
        raw = self.s.cached_call(price_name, self.s.key(app, args),
                                 lambda: json.dumps(self._run(app, args, inline, timeout)).encode())
        return None if raw is None else json.loads(raw)

    def _run(self, app, args, inline, timeout):
        base = config.fal_queue_url()
        sub = request("POST", f"{base}/{app}", args, self.headers)
        rid = sub.get("request_id")
        status_url = sub.get("status_url") or f"{base}/{app}/requests/{rid}/status"
        response_url = sub.get("response_url") or f"{base}/{app}/requests/{rid}"
        t0, wait = time.time(), 0.5
        while True:
            st = request("GET", status_url, headers=self.headers)
            status = st.get("status")
            if status == "COMPLETED":
                break
            if status not in ("IN_QUEUE", "IN_PROGRESS"):
                raise AIError(f"fal {app}: unexpected status {st}")
            if time.time() - t0 > timeout:
                raise AIError(f"fal {app}: timed out after {timeout}s")
            time.sleep(wait)
            wait = min(wait * 1.5, 5)
        result = request("GET", response_url, headers=self.headers)
        if isinstance(result, dict) and result.get("detail"):
            raise AIError(f"fal {app}: {result['detail']}")
        return _inline_urls(result) if inline else result

    def upload(self, data, content_type, name="upload.bin"):
        """Put bytes on fal's CDN; returns a URL models can read."""
        init = request("POST", f"{config.fal_storage_url()}/storage/upload/initiate?storage_type=fal-cdn-v3",
                       {"content_type": content_type, "file_name": name}, self.headers)
        request("PUT", init["upload_url"], data, {"Content-Type": content_type}, raw=True)
        return init["file_url"]

    def image_arg(self, img, fmt="JPEG"):
        uri = data_uri(img, fmt)
        if len(uri) < DATA_URI_LIMIT:
            return uri
        mime = "image/png" if fmt == "PNG" else "image/jpeg"
        return self.upload(base64.b64decode(uri.split(",", 1)[1]), mime, "image." + fmt.lower())

    # ---- models ----------------------------------------------------------------
    def sam3(self, img, prompt):
        """Masks for every instance of `prompt` (text) in img. Returns a bool mask (img size)."""
        res = self.run("SAM3", config.model("SAM3"), {"image_url": self.image_arg(img), "prompt": prompt})
        h, w = img.shape[:2]
        out = np.zeros((h, w), bool)
        if not res:
            return out
        for m in res.get("masks") or []:
            out |= _mask_from(fetch_image(m["url"], "RGBA"), (w, h))
        return out

    def depth(self, img):
        """Relative depth map (float 0..1, orientation depends on the model) at img size."""
        res = self.run("FAL_DEPTH", config.model("FAL_DEPTH"), {"image_url": self.image_arg(img)})
        if not res:
            return None
        d = fetch_image(_first_image(res), "L").astype(np.float32) / 255
        return cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    def flux_fill(self, img, mask, prompt, seed=0):
        """Inpaint where mask > 0 (white = fill)."""
        h, w = img.shape[:2]
        W, H = _mult(w, 32), _mult(h, 32)
        args = {"image_url": self.image_arg(cv2.resize(img, (W, H))),
                "mask_url": self.image_arg(cv2.resize((mask > 0).astype(np.uint8) * 255, (W, H),
                                                      interpolation=cv2.INTER_NEAREST), "PNG"),
                "prompt": prompt, "seed": seed, "num_images": 1, "output_format": "png"}
        res = self.run("FLUX_FILL", config.model("FLUX_FILL"), args)
        return _resized(res, (w, h), img)

    def flux_img2img(self, img, prompt, strength=0.4, lora_url=None, lora_scale=1.0, seed=0):
        h, w = img.shape[:2]
        args = {"image_url": self.image_arg(img), "prompt": prompt, "strength": strength,
                "num_inference_steps": 28, "guidance_scale": 3.5, "seed": seed, "num_images": 1,
                "output_format": "png", "loras": [{"path": lora_url, "scale": lora_scale}] if lora_url else []}
        res = self.run("FLUX_IMG2IMG", config.model("FLUX_IMG2IMG"), args)
        return _resized(res, (w, h), img)

    def edit(self, images, prompt, resolution="2K", seed=0):
        """Gemini image editing ("Nano Banana") hosted on fal. images[0] is edited; the rest are references."""
        h, w = images[0].shape[:2]
        args = {"prompt": prompt, "image_urls": [self.image_arg(i) for i in images],
                "resolution": resolution, "aspect_ratio": "1:1" if h == w else "auto",
                "output_format": "png", "num_images": 1, "seed": seed}
        res = self.run("FAL_EDIT", config.model("FAL_EDIT"), args)
        return _resized(res, (w, h), images[0])

    def train_lora(self, zip_bytes, trigger, steps=1000):
        """Train a FLUX style LoRA from a zip of key frames. Returns the LoRA file URL."""
        url = self.upload(zip_bytes, "application/zip", "style_frames.zip")
        res = self.run("FLUX_LORA_TRAIN", config.model("FLUX_LORA_TRAIN"),
                       {"images_data_url": url, "trigger_word": trigger, "steps": steps, "is_style": True},
                       inline=False, timeout=3600)
        if not res:
            return None
        f = res.get("diffusers_lora_file") or {}
        return f.get("url")


def _mult(v, m):
    return max(m, int(round(v / m)) * m)


def _first_image(res):
    if res.get("images"):
        return res["images"][0]["url"]
    if res.get("image"):
        return res["image"]["url"]
    raise AIError(f"no image in result: {list(res)}")


def _resized(res, size, fallback):
    if not res:
        return fallback  # dry run
    out = fetch_image(_first_image(res), "RGB")
    return cv2.resize(out, size, interpolation=cv2.INTER_AREA if out.shape[1] > size[0] else cv2.INTER_CUBIC)


def _mask_from(rgba, size):
    a = rgba[..., 3]
    m = (a > 127) if a.min() < 250 else (rgba[..., :3].max(-1) > 127)
    return cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0


def _inline_urls(obj):
    """Replace result file URLs with data URIs so the cache is self-contained."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "url" and isinstance(v, str) and v.startswith("http"):
                ctype = obj.get("content_type") or "application/octet-stream"
                data = request("GET", v, raw=True)
                out[k] = f"data:{ctype};base64,{base64.b64encode(data).decode()}"
            else:
                out[k] = _inline_urls(v)
        return out
    if isinstance(obj, list):
        return [_inline_urls(v) for v in obj]
    return obj
