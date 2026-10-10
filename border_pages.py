#!/usr/bin/env python3
"""Fake comic pages with real frame borders, as extra training data for
ComicRedr's panel detector.

Your own comics have the layouts the detector meets, but neither labels nor
a licence to train on their art. This takes only the layout: on every page
whose frames can be read off the page itself (borders.py: flat gutter,
solid frames), the gutter and the frames' border lines stay as they are and
the art inside each frame is replaced by made-up art in the frame's own
colours (fake_art.py). The labels are exact, because the label is what was
painted, not what was detected.

  python3 border_pages.py scan  COMICS scan [--weights MODEL.onnx] [--jobs 8]
  python3 border_pages.py stats scan
  python3 border_pages.py make  COMICS scan moredata [--weights MODEL.onnx]

scan   reads every book (PDF, CBZ/ZIP, folder of images) page by page and
       writes SCAN/<book>.json: the frames found on each page, or why none.
       With --weights it also asks the detector model, so stats can say
       which layouts it gets wrong today.
stats  prints the numbers: pages that could be read, frame shapes, and
       the model's score per shape.
make   picks pages from the scan and writes the fake books:
       OUT/<book>/pNNN.jpg and .json in labelkit's format (frames only, in
       reading order), which ComicRedr's spike/train.py takes as one more
       pages folder beside spike/train_pages and spike/synth_pages. OUT/index.tsv has
       one row per page (layout, shapes, how the model did on the fake
       page); OUT/sources.tsv says which book is which.

Pages are picked for what the training set lacks: every page with a frame
that is not a plain rectangle, and at most --plain-per-book pages of plain
grids per book. A page with one frame is worth little and at most
--single-per-book are kept. Books that are the same book twice (a PDF and a
CBZ) are taken once.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
# ComicRedr's spike/ has the model's interface and the reading order
# (evaluate.py); the repository sits beside this one unless COMICREDR says
# otherwise.
SPIKE = Path(os.environ.get("COMICREDR", HERE.parent / "comicredr")) / "spike"
sys.path[:0] = [str(HERE), str(SPIKE)]
import borders  # noqa: E402
import fake_art  # noqa: E402

IMG = (".jpg", ".jpeg", ".png", ".webp")
LONG = 1600           # px, as ComicRedr's spike/extract_pages.py writes pages
PLAIN = "rect"
_model = None         # one detector per worker process


# ------------------------------------------------------------------ books

def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_books(root, skip):
    """Every PDF, CBZ/ZIP and folder of page images under [root], sorted."""
    out = []
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or any(s in str(rel) for s in skip):
            continue
        if p.suffix.lower() in (".pdf", ".cbz", ".zip") and p.is_file():
            out.append(p)
        elif p.is_dir() and sum(c.suffix.lower() in IMG for c in p.iterdir()) >= 3:
            out.append(p)
    return out


def _fit(pil):
    pil = pil.convert("RGB")
    s = LONG / max(pil.size)
    if s < 1:
        pil = pil.resize((round(pil.width * s), round(pil.height * s)), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def open_book(path):
    """(page count, read(i) -> BGR page with its long side at most LONG)."""
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(path))

        def read(i):
            w, h = doc[i].get_size()
            return _fit(doc[i].render(scale=LONG / max(w, h)).to_pil())
        return len(doc), read
    if path.is_dir():
        names = sorted((p for p in path.iterdir() if p.suffix.lower() in IMG), key=lambda p: natural(p.name))
        return len(names), lambda i: _fit(Image.open(names[i]))
    z = zipfile.ZipFile(path)
    names = sorted((n for n in z.namelist() if n.lower().endswith(IMG) and "__MACOSX" not in n
                    and not Path(n).name.startswith(".")), key=natural)
    return len(names), lambda i: _fit(Image.open(io.BytesIO(z.read(names[i]))))


# ------------------------------------------------------------------ the model

def load_model(weights):
    """The detector as the app runs it, on one thread (the workers are the
    parallelism)."""
    import onnxruntime as ort
    import evaluate

    class OneThread(evaluate.Onnx):
        def __init__(self, path):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            inp = self.sess.get_inputs()[0]
            self.input, self.size = inp.name, inp.shape[2]
            self.names = eval(self.sess.get_modelmeta().custom_metadata_map.get("names", "{0: 'frame'}"))
            self.name = "model"
    return OneThread(weights)


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / (a[2] * a[3] + b[2] * b[3] - ix * iy + 1e-9)


def compare(img, boxes):
    """How the model does on [img] whose true frames are [boxes] (pixels):
    each frame's best IoU, the model's frames that match none, and whether
    two or more of those sit inside one true frame (the model sees panels
    where the page reader saw one)."""
    h, w = img.shape[:2]
    found = [[x * w, y * h, bw * w, bh * h] for x, y, bw, bh in _model(img)[0]]
    best = [max((iou(b, f) for f in found), default=0.0) for b in boxes]
    spare = [f for f in found if all(iou(b, f) < 0.5 for b in boxes)]
    finer = False
    for b in boxes:
        inside = [f for f in spare if iou_inside(f, b) >= 0.85 and f[2] * f[3] < 0.7 * b[2] * b[3]]
        finer |= len(inside) >= 2 and sum(f[2] * f[3] for f in inside) >= 0.5 * b[2] * b[3]
    return {"iou": [round(v, 3) for v in best], "spare": len(spare), "finer": bool(finer)}


def iou_inside(f, b):
    """Share of [f] that lies inside [b]."""
    ix = max(0.0, min(f[0] + f[2], b[0] + b[2]) - max(f[0], b[0]))
    iy = max(0.0, min(f[1] + f[3], b[1] + b[3]) - max(f[1], b[1]))
    return ix * iy / (f[2] * f[3] + 1e-9)


# ------------------------------------------------------------------ scan

def scan_page(img):
    page = borders.find(img)
    row = {"w": img.shape[1], "h": img.shape[0], "why": page.why, "paper": list(page.paper),
           "cover": round(page.cover, 3), "loose": round(page.loose, 3), "frames": []}
    if not page.why:
        row["frames"] = [{"box": list(map(int, f.box)), "kind": f.kind, "slant": round(f.slant, 1),
                          "ring": f.ring, "cut": f.cut} for f in page.frames]
        if _model is not None:
            row["model"] = compare(img, [f.box for f in page.frames])
    return row


def scan_book(job):
    book_id, path, out, weights = job
    global _model
    if weights and _model is None:
        _model = load_model(weights)
    dest = Path(out) / f"{book_id}.json"
    if dest.exists():
        return book_id, path, "kept"
    try:
        n, read = open_book(Path(path))
    except Exception as e:  # a broken archive must not stop the run
        return book_id, path, f"skipped: {e}"
    pages = []
    for i in range(n):
        try:
            pages.append(scan_page(read(i)))
        except Exception as e:
            pages.append({"why": f"unreadable: {e}", "frames": []})
    dest.write_text(json.dumps({"book": book_id, "path": path, "pages": pages}))
    return book_id, path, f"{sum(not p['why'] for p in pages)}/{n} pages read"


def scan(a):
    out = Path(a.scan)
    out.mkdir(parents=True, exist_ok=True)
    books = find_books(a.comics, a.skip)
    jobs = [(f"b{i + 1:03d}", str(p), str(out), a.weights) for i, p in enumerate(books)]
    with Pool(a.jobs) as pool:
        for book_id, path, note in pool.imap_unordered(scan_book, jobs):
            print(f"{book_id} {Path(path).name[:60]}: {note}", flush=True)


# ------------------------------------------------------------------ stats

def load_scan(folder):
    return [json.loads(p.read_text()) for p in sorted(Path(folder).glob("b*.json"))]


def layout(frames):
    """A page's layout family: the least plain shape on it."""
    kinds = {f["kind"] for f in frames}
    for k in ("round", "rounded", "notched", "polygon", "slanted"):
        if k in kinds:
            return k
    if any(f["cut"] for f in frames):
        return "touching"
    return "single" if len(frames) == 1 else "grid"


def _model_rows(pages):
    """Per layout family: pages, pages the model has all frames of at IoU
    0.5 with nothing spare, and its frame recall at IoU 0.5 and 0.75."""
    rows = {}
    for p in pages:
        m = p.get("model")
        if not m:
            continue
        r = rows.setdefault(layout(p["frames"]), [0, 0, 0, 0, 0])
        r[0] += 1
        r[1] += all(v >= 0.5 for v in m["iou"]) and m["spare"] == 0
        r[2] += len(m["iou"])
        r[3] += sum(v >= 0.5 for v in m["iou"])
        r[4] += sum(v >= 0.75 for v in m["iou"])
    return rows


def stats(a):
    books = load_scan(a.scan)
    pages = [p for b in books for p in b["pages"]]
    good = [p for p in pages if not p["why"]]
    print(f"{len(books)} books, {len(pages)} pages, {len(good)} with frames read off the page "
          f"({100 * len(good) / max(1, len(pages)):.0f}%)\n")
    print("Pages left out, and why:")
    for why, n in Counter(re.sub(r":.*", "", p["why"]) for p in pages if p["why"]).most_common():
        print(f"  {n:6d}  {why}")
    print("\nPages by layout:")
    for k, n in Counter(layout(p["frames"]) for p in good).most_common():
        print(f"  {n:6d}  {k}")
    frames = [f for p in good for f in p["frames"]]
    print(f"\n{len(frames)} frames by shape:")
    for k, n in Counter(f["kind"] for f in frames).most_common():
        print(f"  {n:6d}  {k}")
    print(f"  {sum(f['ring'] == 0 for f in frames):6d}  of them with no border line (art to the edge)")
    rows = _model_rows(good)
    if rows:
        print("\nThe model on these pages (layout: pages, pages all right, frame recall at IoU 0.5 / 0.75):")
        for k, (n, ok, fr, r50, r75) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
            print(f"  {k:9s} {n:5d} pages  {100 * ok / n:5.1f}% right   {100 * r50 / fr:5.1f}% / {100 * r75 / fr:5.1f}%")


# ------------------------------------------------------------------ make

LINE_TOL = 60         # max channel difference from a border line's colour


def border_line(crop, frame):
    """The frame's border line as it is on the page, [h, w, 3]. A pixel of
    the ring that is not the line's colour is art showing through (a figure
    over the frame's edge); it is painted in the line's colour, so none of
    the page's art is copied."""
    line = np.empty_like(crop)
    line[:] = frame.ink
    real = np.abs(crop.astype(np.int16) - np.asarray(frame.ink, np.int16)).max(2) <= LINE_TOL
    line[real] = crop[real]
    return line


def fake_page(img, page, rng):
    """The page with every frame's art replaced. Real pixels stay only where
    the page is gutter (paper colour) and in the frames' border lines."""
    out = np.empty_like(img)
    out[:] = page.paper
    for f in sorted(page.frames, key=lambda f: -f.box[2] * f.box[3]):   # insets go on top
        x, y, w, h = f.box
        crop = img[y:y + h, x:x + w]
        art = fake_art.panel(rng, h, w, *fake_art.palette(crop, f.mask, rng))
        region = out[y:y + h, x:x + w]
        if f.ring:
            inner = cv2.erode(f.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * f.ring + 1,) * 2))
            region[f.mask > 0] = border_line(crop, f)[f.mask > 0]
            region[inner > 0] = art[inner > 0]
        else:
            region[f.mask > 0] = art[f.mask > 0]
    out[page.gutter] = img[page.gutter]
    return out


def label(page, w, h):
    """labelkit's label: the frames' boxes in reading order, nothing else."""
    import evaluate
    boxes = [[x / w, y / h, bw / w, bh / h] for x, y, bw, bh in (f.box for f in page.frames)]
    ordered = evaluate.reading_order(boxes, aspect=w / h)
    return {"w": w, "h": h, "panels": [[round(x * w), round(y * h), round(bw * w), round(bh * h)] for x, y, bw, bh in ordered],
            "balloons": [], "captions": []}


def pick_pages(book, a):
    """Indices of the pages of one scanned book worth faking: all the pages
    with an unusual frame, and a few plain ones from all through the book."""
    by = {}
    for i, p in enumerate(book["pages"]):
        if not p["why"] and not p.get("model", {}).get("finer"):
            by.setdefault(layout(p["frames"]), []).append(i)
    by["grid"] = spread(by.get("grid", []), a.plain_per_book)
    by["single"] = spread(by.get("single", []), a.single_per_book)
    return sorted(i for pages in by.values() for i in pages)


def spread(indices, limit):
    """At most [limit] of [indices], evenly through the book."""
    if len(indices) <= limit:
        return indices
    if limit < 2:
        return indices[:max(0, limit)]
    return [indices[round(k * (len(indices) - 1) / (limit - 1))] for k in range(limit)]


def make_book(job):
    book, out, weights, seed, quality = job
    global _model
    if weights and _model is None:
        _model = load_model(weights)
    n, read = open_book(Path(book["path"]))
    dest = Path(out) / book["book"]
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in book["pick"]:
        img = read(i)
        page = borders.find(img)
        if page.why:
            continue
        rng = np.random.default_rng(int(hashlib.sha1(f"{seed}/{book['book']}/{i}".encode()).hexdigest()[:12], 16))
        fake = fake_page(img, page, rng)
        lab = label(page, img.shape[1], img.shape[0])
        cv2.imwrite(str(dest / f"p{i:03d}.jpg"), fake, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(quality[0], quality[1] + 1))])
        (dest / f"p{i:03d}.json").write_text(json.dumps(lab))
        rows.append(index_row(book["book"], i, page, lab, fake))
    return book["book"], rows


def index_row(book_id, i, page, lab, fake):
    frames = [{"kind": f.kind, "cut": f.cut} for f in page.frames]
    kinds = Counter(f.kind for f in page.frames)
    row = [f"{book_id}/p{i:03d}.jpg", layout(frames), len(page.frames),
           " ".join(f"{k}:{n}" for k, n in sorted(kinds.items())),
           f"{max(f.slant for f in page.frames):.1f}", sum(f.ring == 0 for f in page.frames),
           "dark" if sum(page.paper) < 200 else "light"]
    if _model is not None:
        m = compare(fake, lab["panels"])
        row += [sum(v >= 0.5 for v in m["iou"]), m["spare"]]
    return row


def same_books(books):
    """Books whose pages give the same frames as an earlier book's: the same
    comic in another format."""
    seen, dup = {}, set()
    for b in books:
        key = json.dumps([[f["box"] for f in p["frames"]] for p in b["pages"]])
        if any(p["frames"] for p in b["pages"]) and seen.setdefault(key, b["book"]) != b["book"]:
            dup.add(b["book"])
    return dup


def make(a):
    books = load_scan(a.scan)
    dup = same_books(books)
    for b in books:
        b["pick"] = [] if b["book"] in dup else pick_pages(b, a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(b, str(out), a.weights, a.seed, (75, 95)) for b in books if b["pick"]]
    rows = []
    with Pool(a.jobs) as pool:
        for book_id, book_rows in pool.imap_unordered(make_book, jobs):
            rows += book_rows
            print(f"{book_id}: {len(book_rows)} pages", flush=True)
    head = ["page", "layout", "frames", "shapes", "max_slant_deg", "frames_without_line", "gutter"]
    head += ["model_found", "model_spare"] if a.weights else []
    (out / "index.tsv").write_text("\n".join("\t".join(map(str, r)) for r in [head] + sorted(rows)) + "\n")
    (out / "sources.tsv").write_text("".join(f"{b['book']}\t{b['path']}\n" for b in books if b["pick"]))
    print(f"{len(rows)} pages from {len(jobs)} books in {out} ({len(dup)} duplicate books left out)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("comics")
    s.add_argument("scan")
    s.add_argument("--skip", nargs="*", default=[], help="leave out books whose path has one of these in it")
    s = sub.add_parser("stats")
    s.add_argument("scan")
    s = sub.add_parser("make")
    s.add_argument("comics")
    s.add_argument("scan")
    s.add_argument("out")
    s.add_argument("--plain-per-book", type=int, default=6)
    s.add_argument("--single-per-book", type=int, default=1)
    s.add_argument("--seed", type=int, default=1)
    for name in ("scan", "make"):
        sub.choices[name].add_argument("--weights", help="the detector model (.onnx) to compare with")
        sub.choices[name].add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    {"scan": scan, "stats": stats, "make": make}[a.cmd](a)


if __name__ == "__main__":
    main()
