"""hsvis - Historic Soundscapes visual pipeline.

    python -m hsvis init  projects/lane  --pano capture.jpg --lat 51.5 --lon -2.6 --heading 270
    python -m hsvis segment projects/lane --backend heuristic   [--labels painted.png]
    python -m hsvis depth   projects/lane
    python -m hsvis layers  projects/lane
    python -m hsvis fill    projects/lane
    python -m hsvis stylise projects/lane
    python -m hsvis perches projects/lane
    python -m hsvis birds   projects/lane --csv birds.csv
    python -m hsvis render  projects/lane --year 1926 --style 1 -o 1926.jpg
    python -m hsvis sheet   projects/lane -o sheet.jpg
    python -m hsvis all     projects/lane [--labels ...] [--csv ...]

Hosted AI (keys: see SECRETS.md):
    python -m hsvis ai-check
    python -m hsvis segment projects/lane --backend sam3 --dry-run     # estimate first
    python -m hsvis depth   projects/lane --backend fal
    python -m hsvis fill    projects/lane --backend fal-flux
    python -m hsvis train-style projects/lane --frames style/keyframes
    python -m hsvis stylise projects/lane --backend fal-flux           # or gemini --style-refs style/keyframes
    python -m hsvis describe projects/lane --apply
"""
import argparse
import time

from .project import Project, init_project

SEG = ["heuristic", "manual", "mask2former", "sam3", "gemini"]
DEPTH = ["ground-plane", "depth-anything", "fal"]
FILL = ["opencv", "sdxl", "fal-flux", "gemini", "fal-edit"]
STYLE = ["opencv", "sdxl", "fal-flux", "gemini", "fal-edit"]


def dry(q):
    q.add_argument("--dry-run", action="store_true",
                   help="count the API calls and estimated cost without calling anything")


def style_args(q):
    q.add_argument("--lora", help="style LoRA: local path (sdxl) - for fal-flux use --lora-url")
    q.add_argument("--lora-url", help="fal-flux: LoRA file URL (default: the one train-style saved)")
    q.add_argument("--style-refs", help="gemini/fal-edit: folder of illustrator key frames to match")
    q.add_argument("--style-prompt", help="text description of the house style")
    q.add_argument("--strength", type=float, default=0.4, help="fal-flux: how far to move from the photo")
    q.add_argument("--tile-fov", type=float, default=90, help="degrees per tile (smaller = sharper, more calls)")
    q.add_argument("--tile-size", type=int, default=1024)


def style_kw(a, backend):
    if backend == "sdxl":
        return {"lora": a.lora}
    if backend in ("fal-flux", "gemini", "fal-edit"):
        return {"refs_dir": a.style_refs, "lora_url": a.lora_url, "strength": a.strength,
                "style": a.style_prompt, "tile_fov": a.tile_fov, "tile_size": a.tile_size}
    return {}


def main():
    ap = argparse.ArgumentParser(prog="hsvis", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a project from a stitched equirect panorama")
    p.add_argument("project"); p.add_argument("--pano", required=True)
    p.add_argument("--title"); p.add_argument("--lat", type=float); p.add_argument("--lon", type=float)
    p.add_argument("--heading", type=float, help="compass bearing of the panorama centre, degrees")
    p.add_argument("--year", type=int, help="capture year (default 2026)")
    p.add_argument("--camera-height", type=float, default=1.8, help="lens height above ground, metres")
    p.add_argument("--work-width", type=int, default=4096)
    p.add_argument("--out-width", type=int, help="exported layer width (default: source width)")

    def proj(name, help_):
        q = sub.add_parser(name, help=help_)
        q.add_argument("project")
        return q

    q = proj("segment", "class label per pixel")
    q.add_argument("--backend", default="heuristic", choices=SEG)
    q.add_argument("--labels", help="hand-painted label PNG; overrides the backend where painted")
    dry(q)
    q = proj("depth", "approximate depth per pixel")
    q.add_argument("--backend", default="ground-plane", choices=DEPTH)
    dry(q)
    q = proj("layers", "cut into sky / plate / object layers and write scene.json")
    q.add_argument("--min-area", type=int, default=60)
    q = proj("fill", "fill behind removable layers")
    q.add_argument("--backend", default="opencv", choices=FILL)
    dry(q)
    q = proj("stylise", "make the illustrated version of every layer")
    q.add_argument("--backend", default="opencv", choices=STYLE)
    style_args(q)
    dry(q)
    q = proj("describe", "Gemini looks at each building/wall/hedge layer and estimates when it appeared")
    q.add_argument("--apply", action="store_true", help="set lifespans that are still class defaults")
    dry(q)
    q = proj("train-style", "train a FLUX style LoRA on fal.ai from the illustrator's key frames")
    q.add_argument("--frames", required=True, help="folder of 10-20 key frame images (+ optional .txt captions)")
    q.add_argument("--trigger", default="hsvisstyle"); q.add_argument("--steps", type=int, default=1000)
    dry(q)
    sub.add_parser("ai-check", help="check API keys and which configured models are available")
    proj("perches", "find candidate bird perches")
    q = proj("birds", "place birds from a CSV onto perches")
    q.add_argument("--csv", required=True)
    q = proj("render", "composite one era to an image")
    q.add_argument("--year", type=int, default=2026); q.add_argument("--style", type=float, default=0)
    q.add_argument("--width", type=int); q.add_argument("-o", "--out", required=True)
    q.add_argument("--view", nargs=4, type=float, metavar=("YAW", "PITCH", "FOV", "SIZE"))
    q.add_argument("--birds", action="store_true")
    q = proj("sheet", "contact sheet of all eras, photo and illustration")
    q.add_argument("-o", "--out", required=True); q.add_argument("--width", type=int, default=1600)
    q = proj("all", "segment -> depth -> layers -> fill -> stylise -> perches [-> birds]")
    q.add_argument("--seg", default="heuristic", choices=SEG)
    q.add_argument("--labels"); q.add_argument("--csv")
    q.add_argument("--depth", default="ground-plane", choices=DEPTH)
    q.add_argument("--fill", default="opencv", choices=FILL)
    q.add_argument("--style", default="opencv", choices=STYLE)
    style_args(q)

    a = ap.parse_args()
    if a.cmd == "ai-check":
        from .ai.check import check
        raise SystemExit(check())
    if a.cmd == "init":
        pr = init_project(a.project, a.pano, a.title, a.lat, a.lon, a.heading, a.year, a.camera_height,
                          a.work_width, a.out_width)
        print(f"created {pr.root}  ({pr.meta['source_size'][0]}x{pr.meta['source_size'][1]})")
        return
    pr = Project(a.project)

    from . import birds, depth, fill, layers, render, segment, stylise

    def step(name, fn, *args, **kw):
        t = time.time()
        print(f"[{name}] ...", flush=True)
        r = fn(*args, **kw)
        print(f"[{name}] done in {time.time() - t:.1f}s", flush=True)
        return r

    if a.cmd == "segment":
        step("segment", segment.segment, pr, a.backend, a.labels, dry_run=a.dry_run)
    elif a.cmd == "depth":
        step("depth", depth.depth, pr, a.backend, dry_run=a.dry_run)
    elif a.cmd == "layers":
        sc = step("layers", layers.build_layers, pr, min_area=a.min_area)
        print(f"  {len(sc['layers'])} object layers")
    elif a.cmd == "fill":
        step("fill", fill.fill, pr, a.backend, dry_run=a.dry_run)
    elif a.cmd == "stylise":
        step("stylise", stylise.stylise, pr, a.backend, dry_run=a.dry_run, **style_kw(a, a.backend))
    elif a.cmd == "describe":
        from .ai.backends import describe_layers
        todo, sess = step("describe", describe_layers, pr, a.apply, a.dry_run)
        print(f"  {len(todo)} layers; {sess.summary()}")
        for l in todo:
            est = l.get("ai_estimate") or {}
            if est:
                print(f"  {l['id']:<16} {est.get('earliest_year')}-{est.get('latest_year')} "
                      f"({est.get('confidence')}): {est.get('description', '')[:70]}")
    elif a.cmd == "train-style":
        from .ai.backends import train_style
        url, sess = step("train-style", train_style, pr, a.frames, a.trigger, a.steps, a.dry_run)
        print(f"  {sess.summary()}")
        if url:
            print(f"  LoRA: {url}\n  saved to project.json; `stylise --backend fal-flux` will use it")
    elif a.cmd == "perches":
        ps = step("perches", birds.find_perches, pr)
        print(f"  {len(ps)} perches")
    elif a.cmd == "birds":
        bs = step("birds", birds.place_birds, pr, a.csv)
        print(f"  {len(bs)} birds")
    elif a.cmd == "render":
        render.render(pr, a.year, a.out, a.style, a.width, a.view, a.birds)
    elif a.cmd == "sheet":
        step("sheet", render.contact_sheet, pr, a.out, a.width)
    elif a.cmd == "all":
        step("segment", segment.segment, pr, "manual" if (a.labels and a.seg == "manual") else a.seg, a.labels)
        step("depth", depth.depth, pr, a.depth)
        step("layers", layers.build_layers, pr)
        step("fill", fill.fill, pr, a.fill)
        step("stylise", stylise.stylise, pr, a.style, **style_kw(a, a.style))
        step("perches", birds.find_perches, pr)
        if a.csv:
            step("birds", birds.place_birds, pr, a.csv)
        print(f"done. open the player with ?scene=<path to {pr.p('scene.json')}>")


if __name__ == "__main__":
    main()
