"""HTTP plumbing shared by the hosted backends: JSON requests with retries, image encoding,
an on-disk response cache (re-running a step never pays twice for the same request) and a
spending guard."""
import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from . import config


class AIError(RuntimeError):
    pass


def request(method, url, body=None, headers=None, timeout=300, retries=4, raw=False):
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = r.read()
                return payload if raw else (json.loads(payload) if payload else {})
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:800]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** attempt * 2)
                continue
            raise AIError(f"{method} {redact(url)} -> HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(2 ** attempt * 2)
                continue
            raise AIError(f"{method} {redact(url)} failed: {e}") from None


def redact(url):
    return url.split("?key=")[0]


# ---- images ----------------------------------------------------------------------
def encode(img, fmt="JPEG", quality=92):
    """numpy RGB/RGBA/gray -> bytes."""
    im = Image.fromarray(img)
    if fmt == "JPEG" and im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    return buf.getvalue()


def data_uri(img, fmt="JPEG"):
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(encode(img, fmt)).decode()}"


def decode(data, mode="RGB"):
    return np.array(Image.open(io.BytesIO(data)).convert(mode))


def fetch_image(url_or_data, mode="RGB"):
    if url_or_data.startswith("data:"):
        return decode(base64.b64decode(url_or_data.split(",", 1)[1]), mode)
    return decode(request("GET", url_or_data, raw=True), mode)


# ---- cache + budget -----------------------------------------------------------------
class Session:
    """One per pipeline command. Caches responses under <project>/work/ai_cache and stops when the
    estimated spend passes the budget (HSVIS_AI_BUDGET, default $5)."""

    def __init__(self, cache_dir=None, budget=None, dry_run=False):
        self.cache_dir = cache_dir
        self.budget = config.budget() if budget is None else budget
        self.dry_run = dry_run
        self.spent = 0.0
        self.calls = 0
        self.cached = 0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def key(self, *parts):
        h = hashlib.sha256()
        for p in parts:
            h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, default=str).encode())
        return h.hexdigest()[:32]

    def cached_call(self, price_name, key, fn):
        """fn() -> bytes. Returns cached bytes when this exact request was made before."""
        path = os.path.join(self.cache_dir, key) if self.cache_dir else None
        if path and os.path.exists(path):
            self.cached += 1
            with open(path, "rb") as f:
                return f.read()
        cost = config.price(price_name)
        if self.spent + cost > self.budget:
            raise AIError(f"stopping: estimated spend would pass the ${self.budget:.2f} budget "
                          f"(set HSVIS_AI_BUDGET to raise it). {self.calls} calls made so far.")
        if self.dry_run:
            self.spent += cost
            self.calls += 1
            return None
        out = fn()
        self.spent += cost
        self.calls += 1
        if path:
            with open(path + ".tmp", "wb") as f:
                f.write(out)
            os.replace(path + ".tmp", path)
        return out

    def summary(self):
        verb = "would make" if self.dry_run else "made"
        return f"{verb} {self.calls} API calls (~${self.spent:.2f}), {self.cached} served from cache"
