# Historic Soundscapes: visual pipeline and player

Turns a stitched Insta360 Pro 2 panorama into an **editable layer stack** (sky, background plate,
one layer per removable or era-variable feature), each with a **lifespan in years**, an
**illustrated version**, and **bird perches** tied to the layer they sit on. A fixed-POV
browser player composites the stack live, steps between eras, paints from photo to illustration,
and lets you pull birds out of the scene with spatial audio.

```
capture.jpg ─► segment ─► depth ─► layers ─► fill ─► stylise ─► perches ─► birds
                 │          │         │        │        │                    │
            labels.png  depth.npy  scene.json + layers/*.webp          bird CSV from the
                                   (the player's input)                soundscape model
```

Fixed point of view is assumed throughout: layers are pieces of one sphere, composited in order.
Depth is used for ordering, effects and bird distance, not for parallax.

## Quick start

```bash
cd pipeline
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m hsvis init ../projects/mysite --pano /path/to/stitched_8k.jpg \
  --title "Mill Lane, dawn" --lat 51.4545 --lon -2.5879 --heading 270 --year 2026 --camera-height 1.75
```

```bash
python -m hsvis all ../projects/mysite --csv ../samples/birds_example.csv
```

```bash
python -m hsvis sheet ../projects/mysite -o ../projects/mysite/sheet.jpg
```

Then serve the repository root and open the player:

```bash
cd .. && python3 -m http.server 8000
```

Open `http://localhost:8000/player/?scene=../projects/mysite/scene.json`.

A synthetic test scene, with an exact label map, lets you try everything before real captures exist:

```bash
python tools/make_test_pano.py ../samples
```

```bash
python -m hsvis init ../projects/test_lane --pano ../samples/test_pano.jpg --work-width 2048
```

```bash
python -m hsvis all ../projects/test_lane --seg manual --labels ../samples/test_labels.png --csv ../samples/birds_example.csv
```

## Pipeline steps

| Step | What it does | Backends |
|---|---|---|
| `segment` | Class per pixel (sky, vegetation, hedge, terrain, water, road, path, lane_marking, building, wall, vehicle, person, pole, sign, far, other). See `hsvis/classes.py`. | `heuristic` (no models, crude), `manual` (hand-painted label PNG), `mask2former` (GPU) |
| `depth` | Approximate metres per pixel. Ground distance from camera height and angle below the horizon; upright things take the distance where their base meets the ground, column by column. | `ground-plane`, `depth-anything` (GPU, aligned to the ground-plane prior) |
| `layers` | Cuts the panorama into sky, plate and object layers. Pieces of one house split by a pole are joined. Layers are ordered by which is in front where they touch. Edges are decontaminated. Writes `scene.json`. | — |
| `fill` | Inpaints the plate behind every object (in perspective tiles, filling each background class from its own kind) and completes objects behind nearer ones (a house behind a car). Holes backed by sky become transparent. | `opencv` (placeholder quality), `sdxl` (GPU) |
| `stylise` | Illustrated version of every layer, using one shared palette and paper texture so layers match. | `opencv` (placeholder look), `sdxl` + style LoRA (GPU) |
| `perches` | Candidate bird positions: tree crowns, hedge tops, roof ridges, walls, wires, near ground, water. | — |
| `birds` | Places birds from the soundscape CSV onto perches valid in that era. | — |
| `render` / `sheet` | Offline composites: a full equirect for an LED playback server, a perspective view, or a contact sheet of all eras. | — |

**Human moderation:** paint over `work/labels_vis.png` (or the panorama) in Krita or Photoshop
using the class colours, leave unpainted areas transparent, and run
`segment --backend mask2former --labels corrections.png`. Painted pixels override the model.
You can edit lifespans in the player's **Layers** panel and export `scene.json`. Re-running
`layers` keeps lifespans, feature IDs and notes for layers whose ID and class still match.

**GPU backends** (`hsvis/ml.py`, `pip install -r requirements-ml.txt`) were written without GPU
or model-hub access and **have not been run yet**, so expect a round of fixes on first use.
Every model runs on overlapping perspective views, not the raw equirect. Check model licences
before production: Depth Anything V2 Base/Large and FLUX.1-dev are non-commercial.

## scene.json (the hand-off format)

```jsonc
{
  "size": [7680, 3840],
  "eras": [2126, 2026, 1926, 1826, 1726],
  "era_looks": { "1826": { "sepia": 0.55, "sat": 0.65, "tint": [1.05, 1, 0.9], "label": "1826" } },
  "sky":   { "photo": "layers/sky_photo.webp", "illus": "layers/sky_illus.webp" },
  "plate": { "photo": "layers/plate_filled_photo.webp", "illus": "...", "depth": "layers/plate_depth.png" },
  "layers": [{
    "id": "building_009", "kind": "object", "class": "building",
    "bbox": [x0, y0, w, h],                 // in the equirect; x0 + w may pass W (wraps)
    "depth_m": 20.1, "photo": "...webp", "illus": "...webp",
    "lifespan": [1884, null],               // present from 1884, still standing
    "lifespan_source": "default|edited|research",
    "feature_id": "osm:way/123 | research db id",
    "completed_behind": true
  }],
  "birds": [{ "species": "emberiza_citrinella", "name": "Yellowhammer", "era": 1926,
              "yaw": -12.3, "pitch": 1.4, "depth_m": 38, "perch": "hedge", "layer": "hedge_013",
              "audio": "audio/yellowhammer_1926.ogg" }]
}
```

The research system's job is to set `lifespan` and `feature_id` per layer and to add **addition**
layers: things that no longer exist, such as a demolished cottage or a narrower lane surface.
Additions use the same format with a lifespan in the past, and can be painted by an illustrator
or generated.

## Player

`player/` is plain HTML and JavaScript with three.js vendored, so it runs offline on a stage machine.

- **Eras:** layers that stop existing disappear nearest first; new ones appear farthest first.
  Each era's look (sepia, saturation, tint) blends in over 2.5 s.
- **Paint over:** photo to illustration (or back) sweeps from the horizon towards the viewer, using the depth map.
- **Night to dawn** grade slider, as a starting point for the dawn chorus timeline.
- **Birds:** a glow pulses on each bird as it sings. Each one is an HRTF-panned audio source at its
  direction and distance (synth chirps until real stems are set in `audio`). Click a bird to
  "pull it out": the view turns to it, the others duck, and a species card appears.
- **Layers panel:** hover to highlight, toggle, edit lifespans, export `scene.json`.
- **Modes:** `Look` (drag and zoom), `Flat 360°` and `Flat 180°` (the composited equirect, or its front half,
  for capture or for feeding a playback server).
- **URL presets:** `era=1926&style=1&dawn=0.6&mode=flat360&yaw=20&fov=70&ui=0`.
- **Keys:** 1–5 eras, P paint over, F mode, L layers, H hide UI, Esc close card.

For the Pinewood volume, the simplest route is to play the composited equirect (or a crop of it
matching the wall's arc) through the stage's media server, or to load the same layers into
Unreal as sphere sections in this draw order. The layer format is engine-agnostic.

## Capturing with the Insta360 Pro 2

1. **Level the camera precisely.** The ground-plane depth assumes the horizon is the middle row.
   Use the stitcher's gyro or horizon correction, and check that the horizon is flat in the output.
2. **Measure the lens height** (tripod head to lens centre, plus the tripod) and pass it as `--camera-height`.
3. **Record the heading:** point the front lens at a known landmark or note its compass bearing
   (`--heading`), so the panorama can be registered to maps later.
4. **Shoot a burst for a clean plate:** 5–10 bracketed shots a few minutes apart. A median
   stack removes passing cars and people before the pipeline sees them. Keep one frame with
   traffic if you want "today" to include it.
5. **HDR bracketing** (3 or 5 stops) for dawn. Shoot the same position at dawn (the experience)
   and in flat midday light (better segmentation; transfer the masks).
6. **Mono 8K** (7680×3840) is the default input. 3D/stereo 8K is worth capturing too, as real depth later.
7. **Nadir:** take a handheld shot of the ground after removing the tripod, to patch the nadir.
   Otherwise paint the tripod as `other`.
8. **Stitch** (Insta360 Stitcher, or PTGui for control) to 8-bit JPG/PNG or 16-bit TIFF.
   Note that `init` copies the source into the project.
9. **Stand-off:** keep at least 1.5–2 m from the nearest objects to limit stitching parallax errors.

## Known limitations and next steps

- **OpenCV fill and stylise are placeholders.** They prove the data flow and the layer
  maths. The production look comes from the SDXL backends with a LoRA trained on commissioned
  illustration key frames.
- **Road narrowing and surface change by era** are not automated yet. Planned approach: use the
  ground-plane coordinates of road pixels to derive the lane's centreline and width, then generate
  a verge "addition" layer per era from the research data's width and surface.
- **Lifespans are class defaults** (flagged ● in the player) until the research database supplies them.
- **Heuristic segmentation** is only a fallback. Use `mask2former` or hand-painted labels on real captures.
- **Depth for occluded objects** uses typical heights (`depth.TYPICAL_HEIGHT`) when the base is hidden.
