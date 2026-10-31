# API keys and secrets

The pipeline can use two hosted AI services. Each needs one key:

| Variable | Service | Used for | Where to get it |
|---|---|---|---|
| `FAL_KEY` | [fal.ai](https://fal.ai) | SAM 3 segmentation, Depth Anything, FLUX Fill, FLUX restyle and LoRA training, Nano Banana via fal | fal dashboard → **Keys** → Add key (scope: API) |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com) | Nano Banana image editing, Gemini segmentation, dating layers (`describe`) | AI Studio → **Get API key** → Create API key |

You only need the key(s) for the backends you use. `GOOGLE_API_KEY` is accepted in place of
`GEMINI_API_KEY`, and `FAL_API_KEY` in place of `FAL_KEY`.

**Keys never go in the repository, the player or chat messages.** The player runs from files the
pipeline has already produced, so a stage or exhibition machine needs no keys at all.

## 1. On your own computer (the usual case)

```bash
cp .env.example .env
```

Edit `.env` and paste in your keys. `.env` is in `.gitignore`, so it won't be committed. The
pipeline reads it from the repository root (or `pipeline/`). Real environment variables take
priority over the file. Then check the keys work:

```bash
cd pipeline && python -m hsvis ai-check
```

`ai-check` shows which keys are set (masked) and whether each service accepts them. It also checks
that every configured Gemini model is available to your key, and lists the image models your key
can use.

If you'd rather not keep keys in a file, export them in your shell profile or load them from a
password manager (for example `export FAL_KEY=$(op read op://Private/fal/key)` with 1Password's CLI).

## 2. In a Claude Code cloud session

To let Claude run the AI steps for you in a cloud session:

1. Open the cloud environment menu in the session's title bar, then **Edit**.
2. Under **Network secrets** (called **API credentials** in older app versions), add `FAL_KEY` and
   `GEMINI_API_KEY`. If that section isn't offered, add them as environment variables.
3. Under **Network access**, choose **Limited** (or Custom) and add these to **Allowed domains**,
   keeping "Allow package managers" ticked:
   - `queue.fal.run`, `fal.run`, `rest.alpha.fal.ai`, `fal.media`, `v3.fal.media` (fal.ai)
   - `generativelanguage.googleapis.com` (Gemini)
   - `fal.ai`, `ai.google.dev` (optional: lets Claude read the current model docs)
4. Start a **new** session. Existing sessions don't pick up changed settings.

The steps are documented at https://code.claude.com/docs/en/cloud-environments.

## 3. GitHub Actions (if you automate runs later)

Repository **Settings → Secrets and variables → Actions → New repository secret**, one each for
`FAL_KEY` and `GEMINI_API_KEY`. Pass them to a job with:

```yaml
env:
  FAL_KEY: ${{ secrets.FAL_KEY }}
  GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

## Keeping costs and exposure down

- **Estimate first:** every AI step takes `--dry-run`, which counts the calls and estimated cost
  without spending anything.
- **Budget guard:** each command stops once its estimated spend passes `HSVIS_AI_BUDGET`
  (default `5`, in US dollars). Prices per call are in `pipeline/hsvis/ai/config.py`. Check them
  against fal's and Google's pricing pages, and override with `HSVIS_PRICE_<NAME>` if they've changed.
- **Cache:** every response is cached in `<project>/work/ai_cache/`, so re-running a step costs
  nothing unless its inputs changed. Delete that folder to force fresh results.
- **Provider-side limits:** set a spending limit on the fal account, and a budget alert on the
  Google Cloud project behind the AI Studio key. Image models on Gemini need billing enabled.
- **Restrict the Google key** to the Generative Language API (Google Cloud console → APIs &
  Services → Credentials → the key → API restrictions).
- **Rotate** a key immediately if it ever lands in a commit, a screenshot or a chat. Delete it in
  the provider's dashboard, then create a new one.

## Model IDs

Hosted model names change often. Gemini 2.5 models were being retired as this was written, and
Nano Banana 2.1 had just replaced 2.0. Override any of them without touching code:

```bash
HSVIS_MODEL_GEMINI_IMAGE=gemini-3-pro-image   # Nano Banana Pro instead of 2.1
HSVIS_MODEL_GEMINI_SEGMENT=gemini-2.5-flash    # Gemini 3.x returns boxes, not masks
HSVIS_MODEL_SAM3=fal-ai/sam-3/image
```

All roles and their defaults are listed in `pipeline/hsvis/ai/config.py`. Run `ai-check` after a change.
