"""Frame borders read off a page, for border_pages.py.

The detector model draws boxes; this finds the frames themselves, as pixel
masks, on pages whose gutter is one flat colour (white or cream paper, or a
black page). Everything that is not gutter is art; art is opened so that
lettering, page numbers and figures bursting out of a frame fall away, and
each solid piece left is a frame. A frame that is a plain rectangle is then
cut along the thin dark or light rules that split touching panels.

Nothing here is trained and nothing is guessed: a page whose gutter is not
flat, whose art is not in solid pieces, or which is one full-bleed picture
gives no frames, and border_pages.py leaves it out. What comes back is
exact enough to paint over (the mask) and to classify (the kind):

  rect      axis-aligned rectangle, the usual frame
  slanted   four straight sides, at least one off the axes
  polygon   convex, three or five and more sides (wedges, cut corners)
  rounded   rounded corners
  round     circle or ellipse
  notched   straight-sided but not convex (an inset cut out of a corner)
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

TOL = 26             # max channel difference from the gutter colour
OPEN = 0.011         # opening radius, share of the page width
MIN_FRAME = 0.012    # smallest frame, share of the page area
MAX_LOOSE = 0.06     # art outside every frame, share of the page area
MIN_COVER = 0.45     # the frames together, share of the page area
MAX_FRAMES = 20
RULE_DARK, RULE_LIGHT = 80, 232   # grey levels of a rule between touching panels
RULE_SHARE = 0.92    # share of a row or column a rule must fill
RULE_THICK = 14      # px; thicker is art, not a rule
RULE_PART = 0.12     # each side of a rule, share of the frame's side
HAIRLINE = 0.012     # a shut-in gutter is at most twice this wide, share of the page width
MIN_SLANT = 1.5      # degrees off the axes before a four-sided frame counts as slanted
MERGED = 0.15        # a hairline this share of a frame's short side splits it
RING_MAX = 12        # px; a longer flat run from the edge is art, not a border line


@dataclass
class Frame:
    mask: np.ndarray   # uint8 0/1, the frame's pixels inside [box]
    box: tuple         # x, y, w, h in page pixels, border line included
    kind: str
    poly: list         # outline, [(x, y)] in page pixels
    slant: float       # degrees its most slanted long side is off the axes
    ring: int = 0      # px of border line inside the mask's edge
    ink: tuple = (0, 0, 0)   # the border line's colour, BGR
    cut: bool = False  # made by cutting a bigger rectangle along a rule


@dataclass
class Page:
    paper: tuple                 # the gutter colour, BGR
    frames: list = field(default_factory=list)
    loose: float = 0.0           # art outside the frames, share of the page
    cover: float = 0.0
    why: str = ""                # why the page gives no frames; "" when it does
    gutter: np.ndarray = None    # bool, the gutter's pixels (hairlines included)


# --------------------------------------------------------------- the gutter

def _mode_colour(px, share):
    """The commonest colour in [px] when at least [share] of it is that colour."""
    q = (px >> 4).astype(np.int32)
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    c = np.median(px[key == vals[counts.argmax()]], axis=0)
    if (np.abs(px.astype(np.int16) - c).max(1) <= TOL).mean() < share:
        return None
    return c


def paper_candidates(img):
    """Colours the gutter may have: the page's edge when it is flat, and
    nothing else then (with a white margin, black is art, and taking it for
    gutter makes one frame of the whole page). A full-bleed page has gutters
    but no margin: white and black, when enough of the page is one of them."""
    h, w = img.shape[:2]
    b = max(4, round(0.03 * min(h, w)))
    band = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                           img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    c = _mode_colour(band, 0.4)
    if c is not None:
        return [c]
    flat = img[::4, ::4].reshape(-1, 3)
    return [np.median(flat[sel], axis=0)
            for sel, share in ((flat.min(1) >= 232, 0.02), (flat.max(1) <= 30, 0.04)) if sel.mean() >= share]


def art_mask(img, paper):
    """1 where the page is not gutter. Gutter-coloured areas shut inside the
    art are filled (sky, balloons, the white of a black-and-white panel),
    except thin long ones: those are gutters closed off by a balloon lying
    across them, or the white line around an inset."""
    h, w = img.shape[:2]
    diff = np.abs(img.astype(np.int16) - np.asarray(paper, np.int16)).max(2)
    art = (diff > TOL).astype(np.uint8)   # not closed: a gutter may be 3 px thin
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - art, connectivity=4)
    widest = np.zeros(n, np.float32)      # per area: its widest point, as a radius
    np.maximum.at(widest, lab.ravel(), cv2.distanceTransform(1 - art, cv2.DIST_L2, 3).ravel())
    outside = np.zeros(n, bool)
    outside[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = True
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    # Thin all along, and either a straight strip or a sparse web in its box;
    # a balloon is as thin between its lines of lettering but fills its box.
    sparse = (np.minimum(bw, bh) <= 2.5 * HAIRLINE * w) | (stats[:, cv2.CC_STAT_AREA] < 0.2 * bw * bh)
    hairline = (widest <= HAIRLINE * w) & (np.maximum(bw, bh) >= 0.08 * w) & sparse
    fill = ~outside & ~hairline
    fill[0] = False                       # label 0 is the art itself
    return art | fill[lab].astype(np.uint8)


# --------------------------------------------------------------- the frames

def _regrow(seed, art, steps):
    """[seed] grown back [steps] px but only over [art]: restores the corners
    the opening rounded off without crossing a gutter to the next frame."""
    k = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        seed = cv2.dilate(seed, k) & art
    return seed


def _pieces(art, w, h):
    """(mask in its box, box) for every solid piece of [art] big enough to be a frame."""
    r = max(6, round(OPEN * w))
    opened = cv2.morphologyEx(art, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < MIN_FRAME * w * h:
            continue
        x0, y0, x1, y1 = max(0, x - r - 3), max(0, y - r - 3), min(w, x + bw + r + 3), min(h, y + bh + r + 3)
        m = _regrow((lab[y0:y1, x0:x1] == i).astype(np.uint8), art[y0:y1, x0:x1], r + 2)
        # Square opening: the stubs of balloon tails and thin break-outs go,
        # straight corners stay.
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        # Whole again: a hairline shut inside one frame is no gutter of its.
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = cv2.drawContours(np.zeros_like(m), contours, -1, 1, cv2.FILLED)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        out.append((m[by0:by1, bx0:bx1].copy(), (int(x0 + bx0), int(y0 + by0), int(bx1 - bx0), int(by1 - by0))))
    return out


def _slant(poly):
    """Degrees the most slanted long side of [poly] is off the axes."""
    pts = np.asarray(poly, np.float32)
    sides = [(pts[(i + 1) % len(pts)] - pts[i]) for i in range(len(pts))]
    longest = max(np.hypot(*s) for s in sides)
    worst = 0.0
    for s in sides:
        if np.hypot(*s) < 0.25 * longest:
            continue
        a = abs(math.degrees(math.atan2(s[1], s[0]))) % 90
        worst = max(worst, min(a, 90 - a))
    return worst


def _shape(mask, w):
    """(kind, outline) of one piece; kind is None when it is not a frame."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    area = max(cv2.contourArea(cnt), 1.0)
    bh, bw = mask.shape
    hull = cv2.convexHull(cnt)
    solid = area / max(cv2.contourArea(hull), 1.0)
    extent = area / (bw * bh)
    eps = max(3.0, 0.004 * w)
    if extent >= 0.965:
        return "rect", [(0, 0), (bw, 0), (bw, bh), (0, bh)]
    if solid >= 0.97:
        quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        pts = [tuple(map(float, p[0])) for p in quad]
        if len(pts) == 4 and area / max(cv2.contourArea(quad), 1.0) >= 0.97:
            # Under MIN_SLANT it is a rectangle with a ragged edge.
            return ("slanted" if _slant(pts) >= MIN_SLANT else "rect"), pts
        if abs(area / (math.pi / 4 * bw * bh) - 1) < 0.03:
            return "round", [tuple(map(float, p[0])) for p in cv2.approxPolyDP(hull, eps, True)]
        fine = [tuple(map(float, p[0])) for p in cv2.approxPolyDP(hull, eps, True)]
        return ("rounded" if extent >= 0.9 else "polygon"), fine
    poly = cv2.approxPolyDP(cnt, eps, True)
    if solid >= 0.75 and len(poly) <= 8 and area / max(cv2.contourArea(poly), 1.0) > 0.96:
        return "notched", [tuple(map(float, p[0])) for p in poly]
    return None, []


# ----------------------------------------------- rules between touching panels

def _flat_rows(gray, axis):
    """Per row (axis 0) or column: the share of it that is rule-dark, and rule-light."""
    return (gray < RULE_DARK).mean(axis=1 - axis), (gray > RULE_LIGHT).mean(axis=1 - axis)


def _rules(gray, axis):
    """(start, end) of the thin flat lines crossing [gray] along [axis]
    (0: rows), each at least RULE_PART of the side away from both ends.

    A rule is a line, not the edge of a dark or light area: the rows just
    beyond it on both sides must not be flat in the same way."""
    size = gray.shape[axis]
    edge = max(60, round(RULE_PART * size))
    runs = []
    for share in _flat_rows(gray, axis):
        hit = share >= RULE_SHARE
        i = 0
        while i < size:
            if not hit[i]:
                i += 1
                continue
            j = i
            while j < size and hit[j]:
                j += 1
            if j - i <= RULE_THICK and i >= edge and size - j >= edge \
                    and share[i - 24:i - 4].mean() < 0.5 and share[j + 4:j + 24].mean() < 0.5:
                runs.append((i, j))
            i = j
    return sorted(runs)


def _is_flat(part):
    """A piece between rules that is all rule colour is gutter, not a panel."""
    return (part < RULE_DARK).mean() > 0.9 or (part > RULE_LIGHT).mean() > 0.9


def _cut(gray, depth=0):
    """Sub-rectangles (x0, y0, x1, y1) of [gray] between its rules, recursively."""
    h, w = gray.shape
    if depth < 3:
        for axis in (0, 1):
            runs = _rules(gray, axis)
            if not runs:
                continue
            out, start = [], 0
            for a, b in runs + [(gray.shape[axis], None)]:
                part = gray[start:a] if axis == 0 else gray[:, start:a]
                for x0, y0, x1, y1 in _cut(part, depth + 1):
                    out.append((x0, y0 + start, x1, y1 + start) if axis == 0 else (x0 + start, y0, x1 + start, y1))
                start = b
            return out
    return [] if _is_flat(gray) else [(0, 0, w, h)]


def _split(frame, gray):
    """A rectangular frame cut along its rules into the panels it holds; the
    rule itself is in no panel, so border_pages.py keeps its pixels."""
    x, y, w, h = frame.box
    parts = _cut(gray[y:y + h, x:x + w])
    if len(parts) < 2 or any(pw * ph < MIN_FRAME * gray.size for pw, ph in ((x1 - x0, y1 - y0) for x0, y0, x1, y1 in parts)):
        return [frame]
    out = []
    for x0, y0, x1, y1 in parts:
        m = frame.mask[y0:y1, x0:x1].copy()
        out.append(Frame(m, (x + x0, y + y0, x1 - x0, y1 - y0), "rect",
                         [(x + x0, y + y0), (x + x1, y + y0), (x + x1, y + y1), (x + x0, y + y1)], 0.0, cut=True))
    return out


# ------------------------------------------------------------ the border line

def border_ring(img, frame, paper):
    """(How many pixels in from the frame's edge its border line reaches,
    the line's colour).

    The frame is peeled in layers one pixel deep. The line's colour is that
    of the darkest of layers 2 to 4 (layer 1 is the blend with the gutter);
    the line goes on while most of a layer is that colour. 0 when there is
    no line unlike the gutter, or when the colour runs too deep to be a line
    (flat art to the edge)."""
    x, y, w, h = frame.box
    crop = img[y:y + h, x:x + w].astype(np.int16)
    dist = cv2.distanceTransform(cv2.copyMakeBorder(frame.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT), cv2.DIST_L2, 3)[1:-1, 1:-1]
    layers = [crop[(dist > d - 1) & (dist <= d)] for d in range(1, RING_MAX + 2)]
    if any(len(px) < 20 for px in layers):
        return 0, (0, 0, 0)
    core = min(range(1, 4), key=lambda k: np.median(layers[k], axis=0).sum())
    ink = np.median(layers[core], axis=0)

    def is_line(px):
        return (np.abs(px - ink).max(1) <= 60).mean() >= 0.6

    colour = tuple(int(v) for v in ink)
    if np.abs(ink - np.asarray(paper)).max() <= TOL or not is_line(layers[core]):
        return 0, colour
    depth = core + 1    # the layers outside the darkest one are line or blend
    while depth < len(layers) and is_line(layers[depth]):
        depth += 1
    return (0 if depth == len(layers) else depth + 1), colour


# ------------------------------------------------------------------ the page

def _merged(frames, art):
    """True when a frame has gutter running through it that belongs to no
    frame inside it (an inset has its own white line): two panels joined by
    a balloon across their gutter, which cannot be told apart here."""
    h, w = art.shape
    for f in frames:
        x, y, bw, bh = f.box
        others = np.zeros((bh, bw), np.uint8)
        for g in frames:
            gx, gy, gw, gh = g.box
            if g is not f and gx >= x and gy >= y and gx + gw <= x + bw and gy + gh <= y + bh:
                others[gy - y:gy - y + gh, gx - x:gx - x + gw] |= g.mask
        near = cv2.dilate(others, np.ones((19, 19), np.uint8))
        inner = ((f.mask > 0) & (art[y:y + bh, x:x + bw] == 0) & (near == 0)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=8)
        if n > 1 and np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]).max() \
                >= max(MERGED * min(bw, bh), 0.12 * w):
            return True
    return False


def _frames(img, gray, paper):
    h, w = img.shape[:2]
    art = art_mask(img, paper)
    frames, used = [], np.zeros((h, w), np.uint8)
    for mask, (x, y, bw, bh) in _pieces(art, w, h):
        kind, poly = _shape(mask, w)
        if kind is None:
            continue
        f = Frame(mask, (x, y, bw, bh), kind, [(x + px, y + py) for px, py in poly], 0.0)
        f.slant = 0.0 if kind == "rect" else _slant(f.poly)
        used[y:y + bh, x:x + bw] |= mask
        frames.append(f)
    loose = float(((art > 0) & (used == 0)).mean())
    why = _why(frames, loose, float(used.mean())) or ("touching panels" if _merged(frames, art) else "")
    frames = [p for f in frames for p in (_split(f, gray) if f.kind == "rect" else [f])]
    return frames, loose, float(used.mean()), why, art == 0


def _why(frames, loose, cover):
    if not frames:
        return "no solid frames"
    if len(frames) > MAX_FRAMES:
        return "too many pieces"
    if loose > MAX_LOOSE:
        return "art outside the frames"
    if len(frames) == 1 and cover > 0.97:
        return "one full-bleed picture"
    if cover < MIN_COVER:
        return "frames cover too little"
    return ""


def find(img):
    """The frames of a page (BGR), or a Page saying why it has none.

    Each gutter colour the page may have is tried; of those that give a
    clean page the one with most frames wins.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best, fail = None, Page((0, 0, 0), why="no flat gutter")
    for paper in paper_candidates(img):
        frames, loose, cover, why, gutter = _frames(img, gray, paper)
        page = Page(tuple(int(v) for v in paper), frames, loose, cover, why, gutter)
        if page.why:
            fail = page
        elif best is None or len(frames) > len(best.frames):
            best = page
    if best is None:
        return fail
    for f in best.frames:
        f.ring, f.ink = border_ring(img, f, best.paper)
    return best
