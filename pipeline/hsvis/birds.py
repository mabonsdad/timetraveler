"""Steps 6-7: candidate perches from the scene, then birds from the soundscape data placed on them.

Perches are tied to their layer, so a bird on a roof disappears with the house in eras where the
house doesn't exist, and is never placed there in the first place.

Bird CSV (one row per species per era; this is the hand-off point from the soundscape model):
  species,common_name,era,count,perches,audio
  erithacus_rubecula,Robin,2026,3,hedge;canopy;wall,audio/robin.ogg
`perches` is a ;-separated list of perch types from classes.py (canopy, hedge, roof, wall, wire,
ground, water). `audio` is optional, relative to the project folder.
"""
import csv
import json
import math

import numpy as np

from .classes import CLASSES, ID, NAMES
from .depth import load_depth
from .layers import present
from .segment import load_labels


def find_perches(project, spacing_deg=1.2, ground_max_m=80, seed=1):
    labels = load_labels(project)
    depth = load_depth(project)
    inst = np.load(project.p("work", "instances.npy"))
    ids = {}
    with open(project.p("work", "instance_ids.txt")) as f:
        for line in f:
            k, lid = line.split()
            ids[int(k)] = lid
    H, W = labels.shape
    above = np.vstack([labels[:1], labels[:-1]])
    above_inst = np.vstack([inst[:1], inst[:-1]])
    cell = max(1, int(spacing_deg / 360 * W))
    rng = np.random.default_rng(seed)
    perches = []
    taken = set()

    def add(y, x, kind):
        key = (y // cell, x // cell)
        if key in taken:
            return
        taken.add(key)
        perches.append(dict(
            yaw=round((x + 0.5) / W * 360 - 180, 2), pitch=round(90 - (y + 0.5) / H * 180, 2),
            depth_m=round(float(min(depth[y, x], 2000)), 1), perch=kind,
            layer=ids.get(int(inst[y, x])), **{"class": NAMES[labels[y, x]]}))

    for name, info in CLASSES.items():
        kind = info["perch"]
        if kind in (None, "ground", "water"):
            continue
        cid = ID[name]
        # top edges: this class, with something different (or another instance) directly above
        edge = (labels == cid) & ((above != cid) | (above_inst != inst))
        ys, xs = np.nonzero(edge)
        for i in rng.permutation(len(ys)):
            add(ys[i], xs[i], kind)
    for name, kind in (("terrain", "ground"), ("water", "water")):
        ys, xs = np.nonzero((labels == ID[name]) & (depth < ground_max_m))
        for i in rng.permutation(len(ys))[:400]:
            add(ys[i], xs[i], kind)

    with open(project.p("perches.json"), "w") as f:
        json.dump(perches, f, indent=1)
    project.mark("perches", count=len(perches))
    return perches


def place_birds(project, csv_path, seed=2):
    scene = project.load_scene()
    with open(project.p("perches.json")) as f:
        perches = json.load(f)
    layers = {l["id"]: l for l in scene["layers"]}
    rng = np.random.default_rng(seed)
    birds = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            era = int(row["era"])
            kinds = {k.strip() for k in row["perches"].split(";")}
            ok = [p for p in perches if p["perch"] in kinds and
                  (p["layer"] is None or present(layers[p["layer"]], era))]
            if not ok:
                print(f"no perch of type {kinds} for {row['common_name']} in {era}")
                continue
            # prefer nearer perches a little, so birds can be seen and pulled out
            w = np.array([1 / math.sqrt(max(p["depth_m"], 3)) for p in ok])
            n = min(int(row["count"]), len(ok))
            for i in rng.choice(len(ok), n, replace=False, p=w / w.sum()):
                p = ok[i]
                birds.append(dict(
                    id=f"{row['species']}_{era}_{len(birds)}", species=row["species"],
                    name=row["common_name"], era=era, yaw=p["yaw"], pitch=p["pitch"],
                    depth_m=p["depth_m"], perch=p["perch"], layer=p["layer"],
                    audio=row.get("audio") or None))
    scene["birds"] = birds
    project.save_scene(scene)
    project.mark("birds", count=len(birds), csv=csv_path)
    return birds
