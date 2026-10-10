#!/usr/bin/env python3
"""moredata's layouts with real comic art in the frames, for training.

The fake pages in moredata/ keep a real page's gutter and border lines and
fill each frame with made-up art in a palette of its own. A detector
trained on them learnt that a panel ends where the colours change, and
began to join real neighbouring panels of one palette across a thin
gutter (TRAINING.md, "Runs so far"). This keeps the layouts and puts real
art back in the frames: art cut from the labelled frames of ComicRedr's
clean training pages, as its spike/synth_modern.py does, all frames of a
page from one source page where they fit, so neighbouring panels share
colours as they do in a comic. Balloons and captions that come whole with
the art keep their labels, so the pages teach those too.

  python3 real_art.py moredata ../comicredr/spike/train_pages moredata-art [--seed 1]

A frame is its labelled box less the gutter-coloured pixels that reach
the gutter outside the boxes; its border line (borders.py's ring) and the
gutter stay as they are, and the art goes inside. A page with a frame
crossed by a thin gutter-coloured line is left out: those are two panels
border_pages.py read as one (54 of the 1,123 pages), and their labels
teach the detector to join panels across a thin gutter.
Output: <out>/<book>/pNNN.jpg and .json in labelkit's format and
<out>/index.tsv (moredata's rows of the pages made, plus their balloons
and captions), which
ComicRedr's spike/train.py takes beside the other folders. The same seed
gives the same pages. The art is public domain or CC BY, the layouts are
not, so the pages go with the model kept at home, as moredata does.
"""
import argparse
import json
import math
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SPIKE = Path(os.environ.get("COMICREDR", HERE.parent / "comicredr")) / "spike"
sys.path[:0] = [str(HERE), str(SPIKE)]
import borders  # noqa: E402
from synth_modern import load, sources  # noqa: E402

KEEP = 0.85        # a balloon stays labelled when this share of it shows
CUT = 0.15         # between CUT and KEEP it would show cut in half: try other art
SAME_PAGE = 0.8    # share of pages whose frames all try one source page first
HAIR_SHARE = 0.6   # a gutter-coloured line across this share of a frame splits it
HAIR_WIDE = 20     # px; a wider gutter-coloured band is art
EDGE = 0.08        # share of the frame's side next to its edges that is not looked at


def shape(free):
    """The frame inside its box: the convex hull of the largest piece of the
    box that is not outer gutter. Boxes of slanted frames overlap, and the
    neighbour's piece beyond the gutter falls away; paper-coloured patches
    of the fake art and pieces cut off by its dark lines are inside the hull
    and get art too."""
    free = free.astype(np.uint8)
    closed = cv2.morphologyEx(free, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=4)
    if n < 2:
        return free
    piece = (comp == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    mask = np.zeros_like(piece)
    cv2.fillPoly(mask, [cv2.convexHull(cv2.findNonZero(piece))], 1)
    return mask


def frames_of(img, lab):
    """A borders.Frame for every labelled box: its mask is the box less the
    gutter-coloured pixels that reach the gutter outside the boxes (the
    corners of a round frame, the wedge of a slanted neighbour), made whole
    by shape(); its ring the border line borders.py finds on it. None when the page has
    no flat gutter colour, or a frame that is two panels (hairline)."""
    papers = borders.paper_candidates(img)
    if not papers or hairline(img, lab, papers[0]) is not None:
        return None
    paper = papers[0]
    h, w = img.shape[:2]
    gutter = (np.abs(img.astype(np.int16) - np.asarray(paper, np.int16)).max(2) <= borders.TOL).astype(np.uint8)
    inside = np.zeros((h, w), bool)
    for x, y, bw, bh in lab["panels"]:
        inside[y:y + bh, x:x + bw] = True
    n, comp = cv2.connectedComponents(gutter, connectivity=4)
    reach = np.zeros(n, bool)
    reach[np.unique(comp[(gutter > 0) & ~inside])] = True
    reach[0] = False
    outer = reach[comp]
    frames = []
    for x, y, bw, bh in lab["panels"]:
        mask = shape(~outer[y:y + bh, x:x + bw])
        f = borders.Frame(mask, (x, y, bw, bh), "", [], 0.0)
        if mask.mean() > 0.5:
            f.ring, f.ink = borders.border_ring(img, f, paper)
        frames.append(f)
    return frames


def hairline(img, lab, paper):
    """The first frame crossed by a thin straight line of gutter colour, a
    row or column at least HAIR_SHARE of its side, clear of the frame's
    edges, with art on both sides: two panels border_pages.py took for one
    (a balloon across the gutter between them closed it off). None when
    there is none. Such a page is left out, since its label teaches the
    detector to join panels across a thin gutter."""
    g = np.abs(img.astype(np.int16) - np.asarray(paper, np.int16)).max(2) <= borders.TOL
    for i, (x, y, w, h) in enumerate(lab["panels"]):
        crop = g[y:y + h, x:x + w]
        for frac in (crop.mean(axis=1), crop.mean(axis=0)):
            n = len(frac)
            k, end = int(EDGE * n), int((1 - EDGE) * n)
            while k < end:
                if frac[k] < 0.5:
                    k += 1
                    continue
                e = k
                while e < end and frac[e] >= 0.5:
                    e += 1
                before, after = frac[max(0, k - 30):max(0, k - 8)], frac[e + 8:e + 30]
                if e - k <= HAIR_WIDE and frac[k:e].max() >= HAIR_SHARE and len(before) and len(after) \
                        and before.mean() <= 0.25 and after.mean() <= 0.25:
                    return i
                k = e
    return None


def fit_art(path, frame, extras, w, h, inner, rng):
    """Source art scaled to cover a w x h frame, and the balloons it brings,
    or None when one would show cut."""
    fx, fy, fw, fh = frame
    s = max(w / fw, h / fh) * rng.uniform(1.0, 1.3)
    ox, oy = rng.uniform(0, fw * s - w), rng.uniform(0, fh * s - h)
    kept = []
    for kind, (ex, ey, ew, eh) in extras:
        ax0, ay0 = (ex - fx) * s - ox, (ey - fy) * s - oy
        ax1, ay1 = ax0 + ew * s, ay0 + eh * s
        cx0, cy0, cx1, cy1 = max(0, int(ax0)), max(0, int(ay0)), min(w, math.ceil(ax1)), min(h, math.ceil(ay1))
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        seen = inner[cy0:cy1, cx0:cx1]
        share = seen.sum() / max(1.0, (ax1 - ax0) * (ay1 - ay0))
        if share >= KEEP:
            ys, xs = np.nonzero(seen)
            kept.append((kind, [cx0 + xs.min(), cy0 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]))
        elif share > CUT:
            return None
    M = np.float32([[s, 0, -(fx + ox / s) * s], [0, s, -(fy + oy / s) * s]])
    art = cv2.warpAffine(load(path), M, (w, h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    return art, kept


def candidates(pool, by_page, home, rng):
    """Source frames to try: the home page's first, then any."""
    first = list(by_page.get(home, [])) if home else []
    rng.shuffle(first)
    return first + [rng.choice(pool) for _ in range(40)]


def fill(img, frames, pool, by_page, rng):
    """The page with real art inside every frame; the labels it brings."""
    out = img.copy()
    owner = np.full(img.shape[:2], -1, np.int32)
    multi = [p for p, fs in by_page.items() if len(fs) >= min(len(frames), 3)]
    home = rng.choice(multi) if multi and rng.random() < SAME_PAGE else None
    found = []
    order = sorted(range(len(frames)), key=lambda i: -frames[i].box[2] * frames[i].box[3])   # insets on top
    for i in order:
        f = frames[i]
        x, y, w, h = f.box
        inner = f.mask
        if f.ring:
            inner = cv2.erode(f.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * f.ring + 1,) * 2))
        for path, frame, extras in candidates(pool, by_page, home, rng):
            # three framings of each source before the next: art with
            # balloons is kept unless every framing cuts one
            got = next((g for g in (fit_art(path, frame, extras, w, h, inner, rng) for _ in range(3)) if g), None)
            if got:
                break
        else:
            return None
        art, kept = got
        region = out[y:y + h, x:x + w]
        region[inner > 0] = art[inner > 0]
        owner[y:y + h, x:x + w][inner > 0] = i
        found += [(i, kind, [bx + x, by + y, bw, bh]) for kind, (bx, by, bw, bh) in kept]
    labels = {"balloons": [], "captions": []}
    for i, kind, (bx, by, bw, bh) in found:
        # an inset painted over a balloon later hides it
        if (owner[by:by + bh, bx:bx + bw] == i).mean() >= KEEP:
            labels[kind].append([int(bx), int(by), int(bw), int(bh)])
    return out, labels


_pool = _by_page = None


def setup(train_pages):
    global _pool, _by_page
    _pool = sources(train_pages)
    _by_page = {}
    for entry in _pool:
        _by_page.setdefault(entry[0], []).append(entry)


def make(job):
    src, dst, seed = job
    rng = random.Random(seed)
    img = cv2.imread(str(src))
    lab = json.loads(src.with_suffix(".json").read_text())
    frames = frames_of(img, lab)
    if frames is None:
        return "left out"
    done = fill(img, frames, _pool, _by_page, rng)
    if done is None:
        return "no art"
    out, extra = done
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    dst.with_suffix(".json").write_text(json.dumps({**lab, **extra}))
    return "made"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("moredata", help="the fake pages (moredata/)")
    ap.add_argument("train_pages", help="ComicRedr's labelled clean training pages (spike/train_pages)")
    ap.add_argument("out", help="output folder, e.g. moredata-art")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    root, out = Path(a.moredata), Path(a.out)
    pages = sorted(p for p in root.glob("*/*.jpg") if p.with_suffix(".json").exists())
    jobs = [(p, out / p.parent.name / p.name, f"{a.seed}:{p.parent.name}/{p.name}") for p in pages]
    with Pool(a.jobs, initializer=setup, initargs=(a.train_pages,)) as pool:
        got = pool.map(make, jobs, chunksize=8)
    counts = {k: got.count(k) for k in ("made", "left out", "no art")}
    # moredata's index, the pages made, with what the art brought
    rows = (root / "index.tsv").read_text().splitlines()
    index = [rows[0] + "\tballoons\tcaptions"]
    for row in rows[1:]:
        js = out / row.split("\t")[0].replace(".jpg", ".json")
        if js.exists():
            lab = json.loads(js.read_text())
            index.append(f"{row}\t{len(lab['balloons'])}\t{len(lab['captions'])}")
    (out / "index.tsv").write_text("\n".join(index) + "\n")
    labels = [json.loads(p.read_text()) for p in out.glob("*/*.json")]
    print(f"{counts['made']} of {len(pages)} pages made ({counts['left out']} left out: a frame of two panels, "
          f"or no flat gutter; {counts['no art']} without art that fits); "
          f"{sum(len(x['panels']) for x in labels)} frames, "
          f"{sum(len(x['balloons']) for x in labels)} balloons, {sum(len(x['captions']) for x in labels)} captions")


if __name__ == "__main__":
    main()
