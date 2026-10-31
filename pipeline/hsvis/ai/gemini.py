"""Google AI Studio (Gemini API) client.

POST {base}/v1beta/models/<model>:generateContent with header x-goog-api-key.
Three uses:
  edit_image   Nano Banana image editing (restyle, remove objects); images in, image out
  segment      pixel masks from Gemini 2.5 Flash (Gemini 3.x returns boxes only)
  ask_json     vision + text -> structured JSON (dating buildings, describing layers)
"""
import base64
import json
import re

import cv2
import numpy as np

from . import config
from .http import AIError, decode, encode, request


class Gemini:
    def __init__(self, session):
        self.s = session
        self._key = None

    @property
    def headers(self):
        if self._key is None:
            self._key = config.gemini_key()
        return {"x-goog-api-key": self._key}

    def list_models(self):
        out, token = [], None
        while True:
            q = f"?pageSize=200{'&pageToken=' + token if token else ''}"
            r = request("GET", f"{config.gemini_url()}/v1beta/models{q}", headers=self.headers)
            out += [m["name"].removeprefix("models/") for m in r.get("models", [])]
            token = r.get("nextPageToken")
            if not token:
                return out

    def generate(self, price_name, model, parts, gen_config=None):
        body = {"contents": [{"role": "user", "parts": parts}]}
        if gen_config:
            body["generationConfig"] = gen_config
        url = f"{config.gemini_url()}/v1beta/models/{model}:generateContent"
        raw = self.s.cached_call(price_name, self.s.key(model, body),
                                 lambda: json.dumps(request("POST", url, body, self.headers)).encode())
        return None if raw is None else json.loads(raw)

    # ---- image editing ---------------------------------------------------------
    def edit_image(self, images, prompt, size="2K"):
        """images[0] is the image to change; any others are references. Returns RGB at images[0]'s size."""
        h, w = images[0].shape[:2]
        parts = [{"text": prompt}] + [_image_part(i) for i in images]
        cfg = {"responseModalities": ["IMAGE"],
               "imageConfig": {"aspectRatio": "1:1" if h == w else _aspect(w, h), "imageSize": size}}
        res = self.generate("GEMINI_IMAGE", config.model("GEMINI_IMAGE"), parts, cfg)
        if res is None:
            return images[0]
        for p in _parts(res):
            blob = p.get("inlineData") or p.get("inline_data")
            if blob:
                out = decode(base64.b64decode(blob["data"]))
                return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        raise AIError(f"Gemini returned no image: {_text(res)[:300] or res.get('promptFeedback')}")

    # ---- segmentation -------------------------------------------------------------
    def segment(self, img, labels):
        """Pixel masks for the listed labels. Returns [(label, bool mask at img size)]."""
        h, w = img.shape[:2]
        prompt = ("Give the segmentation masks for every instance of these things: " + ", ".join(labels) +
                  ". Output a JSON list where each entry has the 2D bounding box in \"box_2d\", the "
                  "segmentation mask in \"mask\", and the label in \"label\". The label must be exactly one of: "
                  + ", ".join(f'"{l}"' for l in labels) + ".")
        cfg = {"responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 0}}
        res = self.generate("GEMINI_SEGMENT", config.model("GEMINI_SEGMENT"), [_image_part(img), {"text": prompt}], cfg)
        if res is None:
            return []
        out = []
        for item in _json(_text(res)) or []:
            try:
                y0, x0, y1, x1 = [int(v) for v in item["box_2d"]]
                x0, x1 = int(x0 / 1000 * w), int(x1 / 1000 * w)
                y0, y1 = int(y0 / 1000 * h), int(y1 / 1000 * h)
                if x1 <= x0 or y1 <= y0:
                    continue
                png = item["mask"].split(",", 1)[-1]
                prob = decode(base64.b64decode(png), "L")
                prob = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                m = np.zeros((h, w), bool)
                m[y0:y1, x0:x1] = prob > 127
                out.append((item.get("label", ""), m))
            except (KeyError, ValueError, TypeError):
                continue  # one malformed entry shouldn't sink the view
        return out

    # ---- structured questions ---------------------------------------------------
    def ask_json(self, prompt, images=(), schema=None):
        parts = [_image_part(i) for i in images] + [{"text": prompt}]
        cfg = {"responseMimeType": "application/json"}
        if schema:
            cfg["responseSchema"] = schema
        res = self.generate("GEMINI_TEXT", config.model("GEMINI_TEXT"), parts, cfg)
        return None if res is None else _json(_text(res))


def _image_part(img):
    return {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(encode(img)).decode()}}


def _parts(res):
    try:
        return res["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        raise AIError(f"Gemini response had no content: {json.dumps(res)[:400]}") from None


def _text(res):
    return "".join(p.get("text", "") for p in _parts(res))


def _json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _aspect(w, h):
    supported = {"1:1": 1, "4:3": 4 / 3, "3:4": 3 / 4, "16:9": 16 / 9, "9:16": 9 / 16, "3:2": 1.5, "2:3": 2 / 3, "21:9": 21 / 9}
    r = w / h
    return min(supported, key=lambda k: abs(supported[k] - r))
