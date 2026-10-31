"""A local stand-in for fal.ai (queue + storage) and the Gemini API, for testing the plumbing
without keys or network access. Responses have the same shape as the real services; the image
"processing" is deliberately trivial (tint, posterise, flat fill) so results are recognisable.

    python tests/mock_services.py 8799     # serve until killed
"""
import base64
import io
import json
import re
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

FAL_KEY = "test-fal-key"
GEMINI_KEY = "test-gemini-key"
GEMINI_MODELS = ["gemini-nano-banana-2.1", "gemini-2.5-flash", "gemini-flash-latest", "gemini-3-pro-image"]


def _img(uri_or_b64):
    data = uri_or_b64.split(",", 1)[1] if uri_or_b64.startswith("data:") else uri_or_b64
    return np.array(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


def _png(arr, mode=None):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, "PNG")
    return buf.getvalue()


class State:
    def __init__(self):
        self.files = {}      # id -> (bytes, content type)
        self.requests = {}   # id -> {app, args, polls}
        self.calls = []      # (service, model)


class Handler(BaseHTTPRequestHandler):
    state: State = None

    def log_message(self, *a):
        pass

    # ---- helpers
    def _send(self, code, obj=None, raw=None, ctype="application/json"):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _base(self):
        return f"http://{self.headers['Host']}"

    def _file(self, data, ctype="image/png"):
        fid = uuid.uuid4().hex
        self.state.files[fid] = (data, ctype)
        return {"url": f"{self._base()}/files/{fid}", "content_type": ctype}

    def _fal_auth(self):
        if self.headers.get("Authorization") != f"Key {FAL_KEY}":
            self._send(401, {"detail": "Invalid API key"})
            return False
        return True

    def _gem_auth(self):
        if self.headers.get("x-goog-api-key") != GEMINI_KEY:
            self._send(403, {"error": {"code": 403, "message": "API key not valid"}})
            return False
        return True

    # ---- routes
    def do_GET(self):
        p = self.path
        if p.startswith("/files/"):
            data, ctype = self.state.files[p.split("/")[2]]
            return self._send(200, raw=data, ctype=ctype)
        if p.startswith("/gemini/v1beta/models"):
            if self._gem_auth():
                self._send(200, {"models": [{"name": f"models/{m}"} for m in GEMINI_MODELS]})
            return
        if p.startswith("/fal/"):
            if not self._fal_auth():
                return
            m = re.match(r"/fal/(.+)/requests/([0-9a-f-]+)(/status)?$", p)
            req = self.state.requests.get(m.group(2)) if m else None
            if not req:
                return self._send(404, {"detail": "Request not found"})
            if m.group(3):
                req["polls"] += 1
                return self._send(200, {"status": "COMPLETED" if req["polls"] > 1 else "IN_PROGRESS"})
            return self._send(200, self._fal_result(req["app"], req["args"]))
        self._send(404, {"detail": "no route"})

    def do_PUT(self):
        if self.path.startswith("/put/"):
            self.state.files[self.path.split("/")[2]] = (self._body(), self.headers.get("Content-Type"))
            return self._send(200, {})
        self._send(404, {})

    def do_POST(self):
        p = self.path
        body = self._body()
        if p.startswith("/storage/upload/initiate"):
            if not self._fal_auth():
                return
            fid = uuid.uuid4().hex
            return self._send(200, {"upload_url": f"{self._base()}/put/{fid}", "file_url": f"{self._base()}/files/{fid}"})
        if p.startswith("/fal/"):
            if not self._fal_auth():
                return
            app = p[len("/fal/"):]
            rid = str(uuid.uuid4())
            self.state.requests[rid] = {"app": app, "args": json.loads(body), "polls": 0}
            self.state.calls.append(("fal", app))
            # like the real queue, status lives under the app's base (first two path parts)
            base = "/".join(app.split("/")[:2])
            return self._send(200, {"request_id": rid,
                                    "status_url": f"{self._base()}/fal/{base}/requests/{rid}/status",
                                    "response_url": f"{self._base()}/fal/{base}/requests/{rid}"})
        m = re.match(r"/gemini/v1beta/models/(.+):generateContent$", p)
        if m:
            if not self._gem_auth():
                return
            if m.group(1) not in GEMINI_MODELS:
                return self._send(404, {"error": {"code": 404, "message": f"models/{m.group(1)} is not found"}})
            self.state.calls.append(("gemini", m.group(1)))
            return self._send(200, self._gemini(json.loads(body)))
        self._send(404, {"detail": "no route"})

    # ---- fake models
    def _fal_result(self, app, args):
        if "sam-3" in app:
            img = _img(args["image_url"])
            h, w = img.shape[:2]
            if args["prompt"] == "sky":  # bright, low-texture pixels in the upper half
                m = np.zeros((h, w), np.uint8)
                m[: h // 2] = (img[: h // 2].mean(-1) > 150) * 255
                return {"masks": [self._file(_png(m, "L"))], "image": None}
            return {"masks": []}
        if "depth-anything" in app:
            img = _img(args["image_url"])
            h, w = img.shape[:2]
            ramp = np.tile(np.linspace(0, 255, h, dtype=np.float32)[:, None], (1, w))  # nearer = lower = brighter
            return {"image": self._file(_png(ramp.astype(np.uint8), "L"))}
        if "flux-lora-fast-training" in app:
            assert args["images_data_url"].startswith(self._base()), "zip should arrive via storage upload"
            return {"diffusers_lora_file": {"url": "https://example.invalid/style.safetensors"}}
        if "fill" in app:
            img, mask = _img(args["image_url"]), _img(args["mask_url"])[..., 0] > 127
            out = img.copy()
            out[mask] = img[~mask].mean(0) if (~mask).any() else 128
            return {"images": [self._file(_png(out))]}
        if "image-to-image" in app:  # posterise = "style"
            return {"images": [self._file(_png((_img(args["image_url"]) // 64 * 64 + 32).astype(np.uint8)))]}
        if "nano-banana" in app:
            img = _img(args["image_urls"][0])
            return {"images": [self._file(_png(_tint(img)))], "description": ""}
        return {"detail": f"mock has no model {app}"}

    def _gemini(self, body):
        parts = body["contents"][0]["parts"]
        text = " ".join(p.get("text", "") for p in parts)
        images = [_img(p["inlineData"]["data"]) for p in parts if "inlineData" in p]
        cfg = body.get("generationConfig", {})
        if "IMAGE" in cfg.get("responseModalities", []):
            out = _tint(images[0])
            # remove magenta (inpainting by instruction)
            mag = (out[..., 0] > 200) & (out[..., 1] < 80) & (out[..., 2] > 200)
            out[mag] = 120
            data = base64.b64encode(_png(out)).decode()
            return {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": data}}]}}]}
        if "segmentation masks" in text:
            h, w = images[0].shape[:2]
            m = (images[0][: h // 2].mean(-1) > 150).astype(np.uint8) * 255
            mask = "data:image/png;base64," + base64.b64encode(_png(m, "L")).decode()
            ans = [{"box_2d": [0, 0, 500, 1000], "mask": mask, "label": "sky"}]
            return {"candidates": [{"content": {"parts": [{"text": "```json\n" + json.dumps(ans) + "\n```"}]}}]}
        ans = {"description": "a brick house with a slate roof", "evidence": ["machine-made brick"],
               "earliest_year": 1890, "latest_year": 1914, "confidence": "medium", "notes": "mock"}
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(ans)}]}}]}


def _tint(img):
    return np.clip(img.astype(np.float32) * [1.05, 0.95, 0.8] + [10, 5, 0], 0, 255).astype(np.uint8)


def start(port=0):
    state = State()
    handler = type("H", (Handler,), {"state": state})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


if __name__ == "__main__":
    srv, _ = start(int(sys.argv[1]) if len(sys.argv) > 1 else 8799)
    print(f"mock fal/gemini on http://127.0.0.1:{srv.server_port}")
    threading.Event().wait()
