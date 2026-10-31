"""Secrets, model choices and prices for the hosted AI backends.

Secrets come from environment variables. For local work they can also live in a `.env` file
(never committed: it is in .gitignore) in the repository root or the pipeline folder:

    FAL_KEY=...             fal.ai  (Dashboard -> API keys)
    GEMINI_API_KEY=...      Google AI Studio (Get API key); GOOGLE_API_KEY is accepted too

Every model id can be overridden without code changes, because hosted model names change often:

    HSVIS_MODEL_GEMINI_IMAGE=gemini-3-pro-image
    HSVIS_MODEL_SAM3=fal-ai/sam-3/image

Run `python -m hsvis ai-check` to see which keys are set and which models your Gemini key can use.
"""
import os

_ROOTS = [os.path.join(os.path.dirname(__file__), "..", "..", ".."),   # repo root
          os.path.join(os.path.dirname(__file__), "..", "..")]         # pipeline/
_loaded = False


def load_dotenv():
    """Minimal .env loader: KEY=VALUE lines, # comments, optional quotes. Real environment wins."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for root in _ROOTS:
        path = os.path.abspath(os.path.join(root, ".env"))
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip().removeprefix("export ").strip(), v.strip().strip("'\"")
                os.environ.setdefault(k, v)


def secret(name, *aliases):
    load_dotenv()
    for n in (name, *aliases):
        if os.environ.get(n):
            return os.environ[n]
    return None


def require(name, *aliases, purpose=""):
    v = secret(name, *aliases)
    if not v:
        raise SystemExit(
            f"{name} is not set{' (needed for ' + purpose + ')' if purpose else ''}.\n"
            f"Add it to the environment or to a .env file - see SECRETS.md.")
    return v


def fal_key():
    return require("FAL_KEY", "FAL_API_KEY", purpose="fal.ai models")


def gemini_key():
    return require("GEMINI_API_KEY", "GOOGLE_API_KEY", purpose="Google AI Studio / Gemini models")


# ---- models -------------------------------------------------------------------
# Defaults as of late 2026. Override any of them with HSVIS_MODEL_<NAME>.
DEFAULT_MODELS = {
    # fal.ai
    "SAM3": "fal-ai/sam-3/image",                                # text-prompted segmentation
    "FAL_DEPTH": "fal-ai/image-preprocessors/depth-anything/v2",
    "FLUX_FILL": "fal-ai/flux-pro/v1/fill",                      # masked inpainting
    "FLUX_IMG2IMG": "fal-ai/flux-lora/image-to-image",           # restyle with a trained LoRA
    "FLUX_LORA_TRAIN": "fal-ai/flux-lora-fast-training",
    "FAL_EDIT": "fal-ai/nano-banana-pro/edit",                   # Gemini image editing via fal
    # Google AI Studio (Gemini API)
    "GEMINI_IMAGE": "gemini-nano-banana-2.1",                    # image editing / restyling
    "GEMINI_SEGMENT": "gemini-2.5-flash",                        # Gemini 3.x returns boxes, not masks
    "GEMINI_TEXT": "gemini-flash-latest",                        # scene description / dating
}


def model(name):
    return os.environ.get(f"HSVIS_MODEL_{name}") or DEFAULT_MODELS[name]


# Rough USD per call, for the budget guard and estimates. Check current pricing pages;
# override with HSVIS_PRICE_<NAME>.
DEFAULT_PRICES = {
    "SAM3": 0.005, "FAL_DEPTH": 0.002, "FLUX_FILL": 0.05, "FLUX_IMG2IMG": 0.035,
    "FLUX_LORA_TRAIN": 2.0, "FAL_EDIT": 0.15, "GEMINI_IMAGE": 0.07, "GEMINI_SEGMENT": 0.003,
    "GEMINI_TEXT": 0.002,
}


def price(name):
    return float(os.environ.get(f"HSVIS_PRICE_{name}") or DEFAULT_PRICES[name])


def budget():
    """Maximum estimated spend (USD) per pipeline command before it stops."""
    return float(os.environ.get("HSVIS_AI_BUDGET", "5"))


# Service endpoints (overridable for testing against a local mock)
def fal_queue_url():
    return os.environ.get("HSVIS_FAL_QUEUE_URL", "https://queue.fal.run")


def fal_storage_url():
    return os.environ.get("HSVIS_FAL_STORAGE_URL", "https://rest.alpha.fal.ai")


def gemini_url():
    return os.environ.get("HSVIS_GEMINI_URL", "https://generativelanguage.googleapis.com")
