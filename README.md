# ComicRedr training assets

Extra training data for the panel detector of
[ComicRedr](https://github.com/snonux/comicredr), and the scripts that
make it. ComicRedr's own training recipe (its labelled pages, its
synthetic pages, `spike/train.py`) stays in that repository; what is here
adds to it and replaces nothing.

## What is here

| | |
|---|---|
| `moredata/` | 1123 fake comic pages in 141 fake books, with exact frame labels |
| `moredata/index.tsv` | one row per page: layout, frame shapes, slant, gutter, and how the shipped model did on it |
| `border_pages.py` | scans comics, prints the numbers, makes the fake pages |
| `borders.py` | reads the frame borders off a page |
| `fake_art.py` | the made-up art that fills the frames |

A fake page is a real page's layout and nothing else. The gutter and the
frames' border lines are the real page's pixels; the art inside every
frame is replaced by made-up art. No drawing, lettering or balloon of the
source comics is in these files: a pixel is kept only when it has the
gutter's colour or the border line's colour.

The pages came from a private collection of 180 books (13,239 pages) of
mostly modern, digitally published comics. Which books is not recorded
here: `scan/` and `moredata/sources.tsv` name them and are git-ignored.

## How a page is made

1. **Gutter.** The page's edge gives the gutter colour when it is flat
   (white or cream paper, or a black page); a full-bleed page is tried
   with white and with black.
2. **Frames.** Everything that is not gutter is art. Gutter-coloured
   areas shut inside the art are filled (sky, balloons), except thin long
   ones, which are gutters closed off by a balloon lying across them.
   Opening the art drops lettering, page numbers and figures that burst
   out of a frame; each solid piece left is a frame. A plain rectangle is
   then cut along the thin rules that split touching panels.
3. **Shape.** Each frame is a `rect`, `slanted` (four straight sides, one
   at least 1.5 degrees off the axes), `polygon`, `rounded`, `round` or
   `notched` (straight-sided, not convex).
4. **Left out, not guessed.** A page is skipped when its gutter is not
   flat, its art is not in solid pieces, panels touch in a way that cannot
   be separated, or it is one full-bleed picture. With the model at hand, a
   page is also skipped when the model sees two or more panels inside one
   frame found here.
5. **Fake art.** Each frame is filled with flat areas in the frame's own
   colours (k-means of the real panel, in the same shares), ink lines
   between them, pen strokes, hatching, halftone dots and speed lines. A
   white-backed panel on white paper stays as hard as it was. There are no
   balloon-like shapes, because an unlabelled balloon would teach the
   detector that balloons are background.
6. **Label.** The axis-aligned box of what was painted, border line
   included, in reading order. It is exact for straight and slanted
   frames alike, because it is the box of the painted frame and not of a
   detection.

## The numbers

Of 13,239 pages, 3,387 (26%) could be read. The rest:

| Pages | Why not |
|---:|---|
| 4,792 | no solid frames (full-bleed art, covers, pin-ups, text) |
| 3,236 | art outside the frames (borderless panels, figures across the page) |
| 858 | touching panels that cannot be separated |
| 630 | one full-bleed picture |
| 294 | no flat gutter colour |
| 42 | frames cover too little of the page |

The 12,753 frames on the pages that could be read: 12,068 rectangles, 489
slanted, 85 notched, 54 rounded, 53 polygons, 4 round. 2,915 of them (23%)
have no border line; the art runs to the gutter.

The shipped model (`comicredr-panels.onnx`, D-FINE-S) against the frames
read off the **real** pages. "Right" is every frame found at IoU 0.5 and
no frame too many. These are not hand labels: where the two disagree,
either may be wrong.

| Layout | Pages | Pages right | Frames found at IoU 0.5 / 0.75 |
|---|---:|---:|---:|
| grid (rectangles with gutters) | 2,201 | 80.7% | 96.6% / 94.9% |
| single frame | 492 | 49.0% | 70.5% / 55.7% |
| slanted | 268 | 39.9% | 89.3% / 80.8% |
| touching (cut along rules) | 249 | 29.3% | 83.3% / 76.1% |
| notched | 77 | 39.0% | 89.7% / 79.7% |
| rounded | 51 | 52.9% | 88.6% / 83.3% |
| polygon | 46 | 8.7% | 46.2% / 38.8% |
| round | 3 | 33.3% | 91.7% / 83.3% |

So the model is good at plain grids and weak wherever a frame is not a
rectangle with a gutter round it, and its boxes get loose there (the drop
from IoU 0.5 to 0.75).

`moredata/` is picked for that: every readable page with a frame that is
not a plain rectangle, six plain grids and one single-frame page per book.
The same model on the **fake** pages:

| Layout | Pages | Frames | Frames found at IoU 0.5 | Pages right |
|---|---:|---:|---:|---:|
| grid | 526 | 2,415 | 99.6% | 91.8% |
| touching | 216 | 1,179 | 93.2% | 69.4% |
| slanted | 169 | 661 | 99.2% | 88.2% |
| single | 63 | 63 | 100.0% | 92.1% |
| notched | 60 | 253 | 95.7% | 78.3% |
| polygon | 44 | 66 | 97.0% | 84.1% |
| rounded | 43 | 115 | 97.4% | 93.0% |
| round | 2 | 9 | 88.9% | 50.0% |

122 of the 1,123 pages have a dark gutter; 1,191 of the 4,761 frames have
no border line.

The fake pages are easier for the model than the real pages they come
from. Much of what makes a real page hard is its art, and made-up art
does not carry that over. Expect these pages to help most with touching
and notched frames and with tight boxes on slanted ones; whether they
help at all is not measured yet. No model has been trained on them.

## Training with them

Check this repository out beside ComicRedr. Then, in ComicRedr:

```sh
make train-model LOCAL=1      # adds ../comicredr-training-assets/moredata when it is there
# or by hand
python3 spike/train.py spike/train_pages spike/synth_pages ../comicredr-training-assets/moredata --out spike/out/local/train
```

`spike/train.py` takes the folder as it is: `<book>/pNNN.jpg` with
`pNNN.json` beside it in labelkit's format (`w`, `h`, `panels` as
`[x, y, w, h]` in pixels, empty `balloons` and `captions`), pages 1600 px
on the long side like ComicRedr's own extracted pages. A tenth is held out
for validation by its usual rule.

What to keep in mind:

- **Score before and after.** ComicRedr's three test sets (original,
  modern, diagonal) decide; a model must beat the shipped one there.
- **Frames only.** The pages have no balloons or captions. With the 998
  labelled and 400 synthetic pages they are 45% of the set; if balloon
  scores slip, use fewer. `index.tsv` lets you pick: drop the `grid` and
  `single` rows first, they are what the labelled set already has.
- **Where the model is weakest is where there is least data**: 44 polygon
  pages, 2 round. ComicRedr's `synth_modern.py` can make more of those
  than any collection holds.
- **The pages left out are the harder half**: borderless panels and
  panels that touch at an angle. Nothing here covers them; they need hand
  labels.
- **Licence.** ComicRedr publishes a model only when it learned from
  public-domain, CC0 and CC BY books. These pages hold no art, but their
  layouts come from books outside that list, so ComicRedr adds them only
  to the model kept at home (`LOCAL=1`).

## Making them again, or from other comics

Needs Python 3.11 or later, the packages in `requirements.txt`, and
ComicRedr checked out beside this repository (or `COMICREDR=/path/to/it`).

```sh
M=../comicredr/assets/models/comicredr-panels.onnx
python3 border_pages.py scan  ~/Comics scan --weights $M    # every book, page by page
python3 border_pages.py stats scan                          # the tables above
python3 border_pages.py make  ~/Comics scan moredata --weights $M
```

`scan` takes PDFs, CBZ and ZIP files and folders of page images, about half
an hour for 13,000 pages on 8 cores. `make` is seeded: the same comics and
`--seed` give the same pages. `--plain-per-book` and `--single-per-book`
set how many plain pages a book gives.
