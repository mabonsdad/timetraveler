"""`python -m hsvis ai-check`: are the keys set, do they work, and do the configured models exist?"""
from . import config
from .gemini import Gemini
from .http import AIError, Session, request


def _mask(v):
    return f"{v[:4]}…{v[-4:]} ({len(v)} chars)" if v and len(v) > 10 else "set"


def check():
    ok = True
    print("Keys")
    fal = config.secret("FAL_KEY", "FAL_API_KEY")
    gem = config.secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
    print(f"  FAL_KEY         {_mask(fal) if fal else 'NOT SET'}")
    print(f"  GEMINI_API_KEY  {_mask(gem) if gem else 'NOT SET'}")

    print("\nGoogle AI Studio")
    if gem:
        try:
            names = set(Gemini(Session()).list_models())
            print(f"  key works: {len(names)} models visible")
            for role in ("GEMINI_IMAGE", "GEMINI_SEGMENT", "GEMINI_TEXT"):
                m = config.model(role)
                found = m in names
                ok &= found
                print(f"  {role:<15} {m:<34} {'available' if found else 'NOT FOUND'}")
            image_models = sorted(n for n in names if "image" in n or "banana" in n)
            if image_models:
                print(f"  image-capable models on this key: {', '.join(image_models)}")
            print("  (override a model with HSVIS_MODEL_<ROLE>=<model id>)")
        except AIError as e:
            ok = False
            print(f"  key check failed: {e}")
    else:
        print("  skipped (no key)")

    print("\nfal.ai")
    if fal:
        # status of a request id that cannot exist: 401/403 = bad key, 404/422 etc. = key accepted
        url = f"{config.fal_queue_url()}/{config.model('SAM3')}/requests/00000000-0000-0000-0000-000000000000/status"
        try:
            request("GET", url, headers={"Authorization": f"Key {fal}"}, retries=0, timeout=30)
            print("  key accepted")
        except AIError as e:
            msg = str(e)
            if "HTTP 401" in msg or "HTTP 403" in msg:
                ok = False
                print(f"  key rejected: {msg[:200]}")
            elif "HTTP" in msg:
                print("  key accepted (test request id not found, as expected)")
            else:
                ok = False
                print(f"  could not reach fal.ai: {msg[:200]}")
        for role in ("SAM3", "FAL_DEPTH", "FLUX_FILL", "FLUX_IMG2IMG", "FLUX_LORA_TRAIN", "FAL_EDIT"):
            print(f"  {role:<15} {config.model(role)}")
        print("  (fal has no model list API; a wrong id fails on first use with HTTP 404)")
    else:
        print("  skipped (no key)")

    if not (fal or gem):
        ok = False
    print(f"\nBudget guard: ${config.budget():.2f} per command (HSVIS_AI_BUDGET)")
    print("\nall good" if ok else "\nsome checks failed - see SECRETS.md")
    return 0 if ok else 1
