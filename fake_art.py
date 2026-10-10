"""Made-up panel art for border_pages.py: noise that looks like a comic.

A fake page keeps a real page's gutters and border lines and nothing of its
art. What fills a frame instead has to give the detector the same problem
real art does, or it learns to tell noise from paper rather than to find
frames:

  - the frame's own colours, in the shares the real panel has them (taken
    with k-means, so a white-backed panel on white paper stays that hard,
    and so does a dark panel on a black page);
  - flat areas with ink lines between them, as cel-shaded art has;
  - straight strokes, hatching, speed lines and halftone dots inside the
    frame, because real art is full of lines that are not frame edges.

There are no balloons in it. A white ellipse with lettering in it and no
balloon label would teach the detector that balloons are background; real
balloons come from the labelled pages.
"""
import cv2
import numpy as np

COLOURS = 6          # k-means clusters per frame
SAMPLE = 3000        # pixels of the real frame the palette is taken from


def palette(crop, mask, rng):
    """(colours [K, 3] BGR, shares [K]) of the real frame under [mask]."""
    px = crop[mask > 0]
    if len(px) > SAMPLE:
        px = px[rng.choice(len(px), SAMPLE, replace=False)]
    k = min(COLOURS, max(1, len(px) // 50))
    cv2.setRNGSeed(int(rng.integers(1 << 30)))
    _, lab, centres = cv2.kmeans(px.astype(np.float32), k, None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0), 1, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(lab.ravel(), minlength=k) / len(px)
    return centres, shares


def _field(rng, h, w, size):
    """Smooth noise in 0..1 with features about [size] px across, in two octaves."""
    out = np.zeros((h, w), np.float32)
    for scale, weight in ((size, 1.0), (size / 3, 0.35)):
        gh, gw = max(2, round(h / scale) + 2), max(2, round(w / scale) + 2)
        out += weight * cv2.resize(rng.random((gh, gw), np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    out -= out.min()
    return out / max(float(out.max()), 1e-6)


def _areas(rng, h, w, shares):
    """A map of flat areas, one per colour, each about its share of the frame."""
    f = _field(rng, h, w, rng.uniform(0.12, 0.45) * min(h, w))
    order = rng.permutation(len(shares))
    cuts = np.quantile(f, np.clip(np.cumsum(shares[order])[:-1], 0, 1))
    return order[np.searchsorted(cuts, f)]


def _ink(colours):
    """The frame's darkest colour, darkened: what its lines are drawn in."""
    dark = colours[colours.sum(1).argmin()]
    return tuple(float(v) for v in dark * 0.35)


def _outlines(rng, art, areas, ink):
    """Ink lines where two flat areas meet."""
    edge = (areas != np.roll(areas, 1, 0)) | (areas != np.roll(areas, 1, 1))
    edge[0], edge[:, 0] = False, False
    t = int(rng.integers(1, 4))
    art[cv2.dilate(edge.astype(np.uint8), np.ones((t, t), np.uint8)) > 0] = ink


def _strokes(rng, art, ink):
    """Loose pen strokes: short random walks, some of them dead straight."""
    h, w = art.shape[:2]
    for _ in range(int(rng.integers(0, 14))):
        p = rng.random(2) * (w, h)
        step = rng.normal(0, 0.12 * min(h, w), 2)
        pts = [p]
        for _ in range(int(rng.integers(1, 6))):
            step = step * 0.8 + rng.normal(0, 0.05 * min(h, w), 2) * (rng.random() < 0.6)
            pts.append(pts[-1] + step)
        cv2.polylines(art, [np.int32(pts)], False, ink, int(rng.integers(1, 5)), cv2.LINE_AA)


def _hatching(rng, art, areas, ink):
    """Parallel lines over one flat area."""
    h, w = art.shape[:2]
    layer = art.copy()
    gap, a = int(rng.integers(5, 16)), rng.uniform(0, np.pi)
    dx, dy = np.cos(a), np.sin(a)
    for k in range(-(h + w), h + w, gap):
        p = np.array([w / 2 - dy * k, h / 2 + dx * k])
        cv2.line(layer, tuple(np.int32(p - (dx * (h + w), dy * (h + w)))), tuple(np.int32(p + (dx * (h + w), dy * (h + w)))),
                 ink, 1, cv2.LINE_AA)
    pick = areas == rng.integers(areas.max() + 1)
    art[pick] = layer[pick]


def _speed_lines(rng, art, ink):
    """Lines bursting from one point, as behind a punch or a shout."""
    h, w = art.shape[:2]
    c = rng.random(2) * (w, h)
    for a in rng.uniform(0, 2 * np.pi, int(rng.integers(12, 60))):
        r0, r1 = rng.uniform(0.15, 0.5) * max(h, w), 2.0 * max(h, w)
        d = np.array([np.cos(a), np.sin(a)])
        cv2.line(art, tuple(np.int32(c + d * r0)), tuple(np.int32(c + d * r1)), ink, int(rng.integers(1, 4)), cv2.LINE_AA)


def _halftone(rng, art, areas, ink):
    """Screentone dots over one flat area."""
    h, w = art.shape[:2]
    gap = int(rng.integers(6, 14))
    layer = art.copy()
    for y in range(0, h, gap):
        for x in range((y // gap % 2) * gap // 2, w, gap):
            cv2.circle(layer, (x, y), max(1, gap // 4), ink, -1, cv2.LINE_AA)
    pick = areas == rng.integers(areas.max() + 1)
    art[pick] = layer[pick]


def _shapes(rng, art, colours, ink):
    """A few solid figures. Never in a near-white colour: a white oval is a balloon."""
    h, w = art.shape[:2]
    dark = [c for c in colours if c.min() < 215] or [colours[0] * 0.6]
    for _ in range(int(rng.integers(0, 5))):
        c = tuple(float(v) for v in dark[rng.integers(len(dark))])
        centre = (int(rng.random() * w), int(rng.random() * h))
        axes = (int(rng.uniform(0.04, 0.3) * w) + 2, int(rng.uniform(0.04, 0.3) * h) + 2)
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(art, centre, axes, angle, 0, 360, c, -1, cv2.LINE_AA)
        if rng.random() < 0.7:
            cv2.ellipse(art, centre, axes, angle, 0, 360, ink, int(rng.integers(1, 4)), cv2.LINE_AA)


def panel(rng, h, w, colours, shares):
    """Fake art [h, w, 3] BGR in the given colours."""
    areas = _areas(rng, h, w, shares)
    jitter = np.clip(colours + rng.normal(0, 6, colours.shape), 0, 255)
    art = jitter[areas].astype(np.float32)
    if rng.random() < 0.6:   # soft shading across the flat colours
        art *= (0.85 + 0.3 * _field(rng, h, w, 0.6 * min(h, w)))[..., None]
    art = np.clip(art, 0, 255).astype(np.uint8)
    ink = _ink(colours)
    _shapes(rng, art, colours, ink)
    if rng.random() < 0.75:
        _outlines(rng, art, areas, ink)
    _strokes(rng, art, ink)
    if rng.random() < 0.3:
        _hatching(rng, art, areas, ink)
    if rng.random() < 0.2:
        _halftone(rng, art, areas, ink)
    if rng.random() < 0.15:
        _speed_lines(rng, art, ink)
    if rng.random() < 0.5:
        art = cv2.GaussianBlur(art, (3, 3), 0)
    grain = rng.normal(0, rng.uniform(1, 4), art.shape)
    return np.clip(art + grain, 0, 255).astype(np.uint8)
