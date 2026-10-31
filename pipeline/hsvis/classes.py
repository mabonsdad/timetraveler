"""Scene classes shared by every segmentation backend.

Each class has:
  colour     RGB used in hand-painted label PNGs (the "manual" backend) and debug views
  upright    True if it stands up from the ground (its depth comes from where it meets the ground)
  layer      True if each connected region becomes its own removable/era-variable layer
  lifespan   default [from, to] years when the pipeline has no research data (None = open-ended)
  perch      what kind of bird perch its top edge / surface offers (None = not a perch)

Defaults are deliberately conservative placeholders. The research system overrides
lifespans per feature; anything still on a default is flagged `needs_research`.
"""

CLASSES = {
    # name           colour            upright layer  lifespan        perch
    "sky":          dict(colour=(135, 190, 235), upright=False, layer=False, lifespan=[None, None], perch=None),
    "far":          dict(colour=(120, 120, 170), upright=True,  layer=False, lifespan=[None, None], perch=None),
    "vegetation":   dict(colour=(40, 120, 40),   upright=True,  layer=False, lifespan=[None, None], perch="canopy"),
    "hedge":        dict(colour=(90, 150, 50),   upright=True,  layer=True,  lifespan=[None, None], perch="hedge"),
    "terrain":      dict(colour=(160, 200, 90),  upright=False, layer=False, lifespan=[None, None], perch="ground"),
    "water":        dict(colour=(40, 90, 200),   upright=False, layer=False, lifespan=[None, None], perch="water"),
    "road":         dict(colour=(110, 110, 110), upright=False, layer=False, lifespan=[None, None], perch=None),
    "path":         dict(colour=(190, 170, 140), upright=False, layer=False, lifespan=[None, None], perch=None),
    "lane_marking": dict(colour=(255, 255, 255), upright=False, layer=True,  lifespan=[1960, None], perch=None),
    "building":     dict(colour=(200, 80, 60),   upright=True,  layer=True,  lifespan=[None, None], perch="roof"),
    "wall":         dict(colour=(150, 100, 60),  upright=True,  layer=True,  lifespan=[None, None], perch="wall"),
    "vehicle":      dict(colour=(230, 30, 200),  upright=True,  layer=True,  lifespan=[1950, None], perch=None),
    "person":       dict(colour=(255, 200, 0),   upright=True,  layer=True,  lifespan=[9999, 9999], perch=None),
    "pole":         dict(colour=(0, 220, 220),   upright=True,  layer=True,  lifespan=[1930, None], perch="wire"),
    "sign":         dict(colour=(255, 120, 0),   upright=True,  layer=True,  lifespan=[1950, None], perch=None),
    "other":        dict(colour=(0, 0, 0),       upright=True,  layer=False, lifespan=[None, None], perch=None),
}

NAMES = list(CLASSES)
ID = {n: i for i, n in enumerate(NAMES)}
GROUND = {n for n, c in CLASSES.items() if not c["upright"] and n != "sky"}

# Mapillary Vistas label names (used by the Mask2Former backend) -> our classes.
# Matched case-insensitively on substrings, first match wins.
VISTAS_MAP = [
    ("lane marking", "lane_marking"), ("crosswalk", "lane_marking"),
    ("traffic sign frame", "pole"), ("street light", "pole"), ("utility pole", "pole"), ("pole", "pole"),
    ("traffic sign", "sign"), ("traffic light", "sign"), ("billboard", "sign"), ("banner", "sign"),
    ("signage", "sign"),
    ("bicyclist", "person"), ("motorcyclist", "person"), ("rider", "person"), ("person", "person"),
    ("car", "vehicle"), ("truck", "vehicle"), ("bus", "vehicle"), ("motorcycle", "vehicle"),
    ("bicycle", "vehicle"), ("caravan", "vehicle"), ("trailer", "vehicle"), ("on rails", "vehicle"),
    ("vehicle", "vehicle"), ("boat", "vehicle"),
    ("sidewalk", "path"), ("curb", "path"), ("pedestrian", "path"),
    ("road", "road"), ("parking", "road"), ("bike lane", "road"), ("service lane", "road"),
    ("rail track", "road"), ("manhole", "road"), ("pothole", "road"),
    ("building", "building"), ("bridge", "building"), ("tunnel", "building"),
    ("wall", "wall"), ("fence", "wall"), ("guard rail", "wall"), ("barrier", "wall"),
    ("vegetation", "vegetation"), ("terrain", "terrain"), ("sand", "terrain"),
    ("water", "water"), ("mountain", "far"), ("snow", "far"), ("sky", "sky"),
]


def map_vistas(label: str) -> str:
    low = label.lower()
    for key, ours in VISTAS_MAP:
        if key in low:
            return ours
    return "other"
