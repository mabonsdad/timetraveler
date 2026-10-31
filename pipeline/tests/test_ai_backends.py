"""End-to-end test of the hosted-AI backends against the local mock services.

Runs the real CLI in subprocesses over real HTTP, so it covers request shapes, auth headers, queue
polling, storage upload, result download/inlining, caching, the budget guard and every compositing
path. It cannot tell you whether the real models give good results.

    cd pipeline && python -m unittest tests.test_ai_backends -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(HERE)
SAMPLES = os.path.join(PIPELINE, "..", "samples")
sys.path.insert(0, HERE)
import mock_services  # noqa: E402


class AIBackendsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, cls.state = mock_services.start()
        base = f"http://127.0.0.1:{cls.srv.server_port}"
        cls.tmp = tempfile.mkdtemp(prefix="hsvis_ai_")
        cls.proj = os.path.join(cls.tmp, "proj")
        cls.env = dict(os.environ,
                       FAL_KEY=mock_services.FAL_KEY, GEMINI_API_KEY=mock_services.GEMINI_KEY,
                       HSVIS_FAL_QUEUE_URL=f"{base}/fal", HSVIS_FAL_STORAGE_URL=base,
                       HSVIS_GEMINI_URL=f"{base}/gemini", HSVIS_AI_BUDGET="100",
                       NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
        cls.run_cli("init", cls.proj, "--pano", os.path.join(SAMPLES, "test_pano.jpg"),
                    "--work-width", "1024", "--out-width", "2048", "--lat", "51.45", "--lon", "-2.59")

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def run_cli(cls, *args, env=None, check=True):
        r = subprocess.run([sys.executable, "-m", "hsvis", *args], cwd=PIPELINE, env=env or cls.env,
                           capture_output=True, text=True, timeout=900)
        if check and r.returncode != 0:
            raise AssertionError(f"hsvis {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
        return r

    def scene(self):
        with open(os.path.join(self.proj, "scene.json")) as f:
            return json.load(f)

    # tests run in name order: the pipeline steps build on each other
    def test_01_check(self):
        r = self.run_cli("ai-check")
        self.assertIn("key works", r.stdout)
        self.assertIn("key accepted", r.stdout)
        bad = dict(self.env, FAL_KEY="wrong", GEMINI_API_KEY="wrong")
        r = self.run_cli("ai-check", env=bad, check=False)
        self.assertEqual(r.returncode, 1)
        self.assertIn("key rejected", r.stdout)

    def test_02_missing_key_message(self):
        env = {k: v for k, v in self.env.items() if k not in ("FAL_KEY",)}
        r = self.run_cli("depth", self.proj, "--backend", "fal", env=env, check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("FAL_KEY is not set", r.stdout + r.stderr)

    def test_03_segment(self):
        self.run_cli("segment", self.proj, "--backend", "gemini")
        r = self.run_cli("segment", self.proj, "--backend", "sam3")
        self.assertIn("API calls", r.stdout)
        labels = np.array(Image.open(os.path.join(self.proj, "work", "labels.png")))
        self.assertGreater((labels == 0).mean(), 0.2, "mock SAM 3 marks bright upper pixels as sky")
        again = self.run_cli("segment", self.proj, "--backend", "sam3")
        self.assertIn("made 0 API calls", again.stdout, "second run must be served from the cache")

    def test_04_depth(self):
        self.run_cli("depth", self.proj, "--backend", "fal")
        d = np.load(os.path.join(self.proj, "work", "depth.npy"))
        self.assertTrue(np.isfinite(d).all() and d.min() > 0)

    def test_05_layers_and_fill(self):
        # use the exact labels for layering so fills/completions have real objects to work on
        self.run_cli("segment", self.proj, "--backend", "manual", "--labels", os.path.join(SAMPLES, "test_labels.png"))
        self.run_cli("depth", self.proj)
        self.run_cli("layers", self.proj)
        dry = self.run_cli("fill", self.proj, "--backend", "fal-flux", "--dry-run")
        self.assertIn("would make", dry.stdout)
        self.assertFalse(self.scene()["plate"].get("filled"), "dry run must not write")
        self.run_cli("fill", self.proj, "--backend", "gemini")
        self.run_cli("fill", self.proj, "--backend", "fal-flux")
        sc = self.scene()
        self.assertTrue(sc["plate"]["filled"])
        self.assertTrue(any(l.get("completed_behind") for l in sc["layers"]))

    def test_06_stylise(self):
        refs = os.path.join(self.tmp, "refs")
        os.makedirs(refs, exist_ok=True)
        for i in range(2):
            Image.fromarray(np.full((64, 96, 3), 60 + 40 * i, np.uint8)).save(os.path.join(refs, f"ref{i}.png"))
        self.run_cli("stylise", self.proj, "--backend", "gemini", "--style-refs", refs, "--tile-fov", "100")
        self.run_cli("stylise", self.proj, "--backend", "fal-flux", "--tile-fov", "100")
        sc = self.scene()
        for entry in [sc["sky"], sc["plate"], *sc["layers"]]:
            self.assertTrue(entry["illus"] and os.path.exists(os.path.join(self.proj, entry["illus"])))

    def test_07_train_style(self):
        frames = os.path.join(self.tmp, "frames")
        os.makedirs(frames, exist_ok=True)
        for i in range(5):
            Image.fromarray(np.full((32, 32, 3), 40 * i, np.uint8)).save(os.path.join(frames, f"f{i}.png"))
        r = self.run_cli("train-style", self.proj, "--frames", frames)
        self.assertIn("LoRA:", r.stdout)
        with open(os.path.join(self.proj, "project.json")) as f:
            self.assertTrue(json.load(f)["style"]["lora_url"].endswith(".safetensors"))

    def test_08_describe(self):
        r = self.run_cli("describe", self.proj, "--apply")
        self.assertIn("1890-1914", r.stdout)
        layers = self.scene()["layers"]
        self.assertTrue(any(l.get("lifespan_source") == "ai-estimate" for l in layers))
        self.assertTrue(all(l["lifespan"][0] == 1950 for l in layers if l["class"] == "vehicle"),
                        "classes not sent for dating keep their defaults")

    def test_09_budget_guard(self):
        shutil.rmtree(os.path.join(self.proj, "work", "ai_cache"))
        r = self.run_cli("fill", self.proj, "--backend", "fal-flux", env=dict(self.env, HSVIS_AI_BUDGET="0.12"), check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("budget", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
