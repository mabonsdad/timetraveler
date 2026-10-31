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
"""
import argparse
import time

from .project import Project, init_project


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
    q.add_argument("--backend", default="heuristic", choices=["heuristic", "manual", "mask2former"])
    q.add_argument("--labels", help="hand-painted label PNG; overrides the backend where painted")
    q = proj("depth", "approximate depth per pixel")
    q.add_argument("--backend", default="ground-plane", choices=["ground-plane", "depth-anything"])
    q = proj("layers", "cut into sky / plate / object layers and write scene.json")
    q.add_argument("--min-area", type=int, default=60)
    q = proj("fill", "fill behind removable layers")
    q.add_argument("--backend", default="opencv", choices=["opencv", "sdxl"])
    q = proj("stylise", "make the illustrated version of every layer")
    q.add_argument("--backend", default="opencv", choices=["opencv", "sdxl"])
    q.add_argument("--lora", help="style LoRA for the sdxl backend")
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
    q.add_argument("--seg", default="heuristic", choices=["heuristic", "manual", "mask2former"])
    q.add_argument("--labels"); q.add_argument("--csv")
    q.add_argument("--depth", default="ground-plane", choices=["ground-plane", "depth-anything"])
    q.add_argument("--fill", default="opencv", choices=["opencv", "sdxl"])
    q.add_argument("--style", default="opencv", choices=["opencv", "sdxl"])
    q.add_argument("--lora")

    a = ap.parse_args()
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
        step("segment", segment.segment, pr, a.backend, a.labels)
    elif a.cmd == "depth":
        step("depth", depth.depth, pr, a.backend)
    elif a.cmd == "layers":
        sc = step("layers", layers.build_layers, pr, min_area=a.min_area)
        print(f"  {len(sc['layers'])} object layers")
    elif a.cmd == "fill":
        step("fill", fill.fill, pr, a.backend)
    elif a.cmd == "stylise":
        kw = {"lora": a.lora} if a.backend == "sdxl" else {}
        step("stylise", stylise.stylise, pr, a.backend, **kw)
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
        step("stylise", stylise.stylise, pr, a.style, **({"lora": a.lora} if a.style == "sdxl" else {}))
        step("perches", birds.find_perches, pr)
        if a.csv:
            step("birds", birds.place_birds, pr, a.csv)
        print(f"done. open the player with ?scene=<path to {pr.p('scene.json')}>")


if __name__ == "__main__":
    main()
