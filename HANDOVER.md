# Handover: Historic Soundscapes visuals (timetraveler)

Read this first, then `README.md` (pipeline and player) and `SECRETS.md` (API keys).
Work is on branch **`ai-integrations`**; `main` has the first version without the AI backends.

## 1. Project aims

**Historic Soundscapes** is by a spatial sound designer. LLM research produces the probability of
each bird species being present at a UK location, stepping back 100 years at a time (and one
scenario 100 years forward), based on climate change, historic written references to birds, and
local habitat (habitat maps, distance to roads and rivers). The core experience is the **dawn
chorus** heard era by era. Conservationists have reviewed the bird modelling and are happy it is
well founded.

This repository adds the **visual layer**. It is for a funding bid to use a **Pinewood virtual
production (LED volume) stage** as a test bed, then a long-term exhibition space:

- A real 360° capture of the location is shown first, as the hook. It then "paints over" into an
  **illustrative style**, which gives leeway to depart from reality as we move between eras.
- Stepping between eras changes the environment: modern things disappear (cars, signs, poles,
  lane markings), roads narrow and become rougher, buildings appear or vanish, hedges and woodland change.
- **Birds from the audio data can be pulled out of the image interactively.** Each bird is a
  spatial audio source at a plausible perch. Click or point at it to solo it and see its card.
- **Fixed point of view** per environment (the stage has a fixed viewpoint), so parallax is not a priority.
- Later: the environment for each era is driven by **systematic research** (historic maps plus
  written sources, turned into feature lifespans), and a human can moderate what changes.
- The longer-term aim is any UK location (by postcode or GPS). The stage work focuses on hand-picked
  hero locations captured properly.

### Decisions already made
- **No Google Street View.** Its terms forbid scraping, caching and derived content, and its
  imagery is midday, low resolution and full of cars. Capture is with an **Insta360 Pro 2**
  (8K mono or stereo stills, RAW, HDR). Mapillary (CC BY-SA) is the fallback for the "any postcode"
  version later.
- The image is a **layer stack**: sky, background plate, and one object layer per removable or
  era-variable feature, each with a `lifespan` in years. An era just shows the layers present in that year.
- Illustration is the production look. Plan: commission an illustrator for 10–20 key frames of
  the pilot site and train a style model on them (we own the style, no imitation of living artists).
- Hosted AI comes from **fal.ai** and **Google AI Studio**, with keys in a git-ignored `.env`
  (already created on the user's machine, in the repository root).

## 2. State of play

### Done and tested (classical pipeline, no AI keys needed)
- `pipeline/hsvis/`: `init → segment → depth → layers → fill → stylise → perches → birds`, plus
  `render` and `sheet`.
  - Class labels per pixel. Ground-plane depth from camera height, with per-column contact depth
    for upright things.
  - Layer extraction joins pieces of one object split by a nearer thin occluder (a house cut in
    two by a pole). Layers are ordered by who is in front where they touch. Edges are
    decontaminated so neighbours' colours don't bleed in.
  - The plate is filled in perspective tiles, filling each background class from its own kind
    (road from road, verge from verge). Objects are completed behind nearer ones.
  - Placeholder painterly stylisation.
  - Bird perches tied to layers, so a bird on a roof goes when the house goes. Birds are placed
    from a CSV.
- `player/`: three.js viewer (vendored, works offline) with:
  - era transitions (nearest things vanish first) and an era colour look per era;
  - a photo-to-illustration "paint over" sweep from horizon to foreground;
  - a night-to-dawn grade slider;
  - HRTF spatial audio per bird (synth chirps for now) and click-to-pull-out cards;
  - a layer panel for editing lifespans, with JSON export;
  - Look, Flat 360° and Flat 180° modes, and URL presets.
- `pipeline/tools/make_test_pano.py` ray-casts a synthetic test lane with exact labels.
  `projects/test_lane/` is the processed result. Open it with
  `python3 -m http.server` from the repo root and
  `/player/?scene=../projects/test_lane/scene.json`.

### Done, but tested only against a mock (no real API calls made yet)
- `pipeline/hsvis/ai/`: fal.ai queue client and Gemini `generateContent` client. Responses are
  cached in `<project>/work/ai_cache/`, and a budget guard (`HSVIS_AI_BUDGET`, default $5 per
  command) stops runaway spend. Every AI step supports `--dry-run`, and keys are checked up front.
  - `segment --backend sam3` (SAM 3 on fal) or `gemini` (Gemini 2.5 Flash masks)
  - `depth --backend fal` (Depth Anything V2, aligned to the ground-plane prior)
  - `fill --backend fal-flux` (FLUX.1 Fill [pro]) or `gemini` / `fal-edit` (Nano Banana: the
    area to fill is painted magenta and the model is asked to replace it)
  - `train-style` (FLUX LoRA training on fal from key frames)
  - `stylise --backend fal-flux` (img2img plus LoRA) or `gemini` / `fal-edit` (with `--style-refs`)
  - `describe [--apply]`: Gemini estimates when each building, wall or hedge appeared
  - `ai-check`: are the keys set and accepted, and which configured model IDs your Gemini key can see
- `pipeline/tests/test_ai_backends.py` runs every AI backend through the CLI against
  `tests/mock_services.py`, a local fake of both APIs over HTTP. All 9 tests pass. This proves
  the plumbing, **not** the real schemas or output quality.
- Model IDs and prices are defaults in `pipeline/hsvis/ai/config.py`, all overridable with
  `HSVIS_MODEL_<ROLE>` and `HSVIS_PRICE_<ROLE>`. They were chosen from search results only:
  fal.ai and Google's docs were blocked in the cloud session, so **verify them on the first real run**.
  Points to watch:
  - Gemini 2.5 models were being retired around October 2026, and Gemini 3.x doesn't return masks.
    If `gemini-2.5-flash` is gone, use `segment --backend sam3`.
  - The default image model `gemini-nano-banana-2.1` comes from the Firebase model list.
  - `gemini-flash-latest` is used for `describe`.
  - fal field names assumed: SAM 3 `image_url`/`prompt` → `masks[].url`; depth `image_url` →
    `image.url`; FLUX Fill `image_url`/`mask_url`/`prompt` → `images[].url`; Nano Banana edit
    `image_urls`/`resolution` → `images[].url`; LoRA training `images_data_url`/`trigger_word` →
    `diffusers_lora_file.url`.

### Not run at all
- The local-GPU backends in `pipeline/hsvis/ml.py` (Mask2Former, Depth Anything, SDXL) need an
  NVIDIA GPU and `requirements-ml.txt`. These are optional now that hosted backends exist.

## 3. What still needs doing (in priority order)

1. **First real AI run (small and cheap).** In `pipeline/`:
   - Run `python -m hsvis ai-check`. Fix any NOT FOUND model IDs via `.env` overrides, or change
     the defaults in `config.py` if they're wrong.
   - On a scratch copy of the test project, run in this order, each with `--dry-run` first:
     `describe`, `segment --backend gemini`, `depth --backend fal`,
     `segment --backend sam3`, `fill --backend fal-flux`, `stylise --backend gemini`.
   - Fix whatever the real schemas break. The parsing lives in small helpers in `ai/fal.py` and
     `ai/gemini.py`. Update `tests/mock_services.py` to match, so the tests stay honest.
   - Check the outputs with `python -m hsvis sheet <project> -o sheet.jpg` and in the player.
2. **Process the first real Insta360 Pro 2 capture.** Follow the capture checklist in the README:
   level the horizon, measure camera height, record heading, take a burst to median-stack out
   traffic, use HDR, capture dawn and midday, add a nadir shot.
   - Run `init` with `--lat/--lon/--heading/--camera-height`, then the AI pipeline.
   - Expect to tune: the `SAM3_PROMPTS` per class in `ai/backends.py`, the tripod/nadir patch,
     and the `fill` prompts (`BG_PROMPT` in `fill.py`).
3. **House illustration style.** Gather illustrator key frames, then
   `train-style --frames <dir>` followed by `stylise --backend fal-flux` (tune `--strength`,
   `--tile-fov`). Or try `stylise --backend gemini --style-refs <dir>` with no training.
   - Check tile seams on the LED-resolution output.
   - Consider an upscaling step for 8K+ stage output.
4. **Road narrowing and surface change by era** (not implemented). Use the ground-plane
   coordinates of `road` pixels to derive the lane's centreline and width.
   - Generate per-era "addition" layers: the verge reclaiming the lane edges, and a
     gravel or rutted-earth surface.
   - Same layer format with past lifespans (see `scene.json` in the README). The player already
     supports additions.
5. **Research system** (planned, not started). Turn historic maps and written sources into
   feature lifespans that set the `lifespan` and `feature_id` on each layer (currently class
   defaults, flagged ● in the player; `describe --apply` gives AI estimates in the meantime).
   - **Sources:** NLS georeferenced OS maps (25-inch/6-inch, 1840s–1950s); tithe maps (c.1840);
     OS Old Series; Land Utilisation Survey (1930s); Britain from Above aerials; Historic Landscape
     Characterisation; Ancient Woodland Inventory; Environment Agency LiDAR; GB1900 gazetteer;
     Victoria County History; county avifaunas (Biodiversity Heritage Library); UKCP18 for 2126.
   - **Tools:** MapReader (Turing / Living with Machines) for map reading. LLM agent extracts
     cited claims → feature timeline (valid-time PostGIS/GeoPackage) → "Era Scene Spec" per year,
     which feeds both the visual layers and the bird model.
   - **Geo-registration:** project the map features into the panorama using camera pose, depth
     and the LiDAR skyline, so map changes become image masks automatically.
6. **Bird data hand-off.** Swap `samples/birds_example.csv` (an illustrative placeholder, not real
   data) for the sound designer's per-era species data, and wire real audio stems into
   `birds[].audio`.
7. **Stage output.** Confirm the Pinewood volume specs: pixel pitch, wall size and arc, playback
   system (Unreal nDisplay or disguise).
   - Simplest route: composited equirect (or a crop matching the wall) from `render`, or the
     player's flat modes.
   - Alternative: import the layers into Unreal as sphere sections in draw order.
8. **Pre-bid previs.** A 2–3 minute film at one site (dawn, step back to 1926 and 1826 and forward
   to 2126, birds appearing, one pulled out), plus a pipeline diagram, cost per location from
   the dry runs, and expert review (conservationists plus a landscape historian).
9. **Merge `ai-integrations` into `main`** (via pull request) once step 1 works with real keys.

## 4. Working notes
- Python 3.11+. In `pipeline/`:

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

- Run the tests:

```bash
cd pipeline && python -m unittest tests.test_ai_backends -v
```

  This takes about 6 minutes and needs no keys.
- Never commit `.env`, keys or `projects/<real sites>/`; `.gitignore` already covers them.
  Large captures stay out of git.
- Use `--dry-run` before any AI step on a new panorama; the cache makes re-runs free.
- Pronouns for collaborators haven't been stated, so use they/them.
