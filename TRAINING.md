# Training and retraining the detector

Everything needed to train ComicRedr's panel detector again, with or
without the fake pages in `moredata/`, from this repository's side. The
recipe itself lives in ComicRedr and is the same for every model:
[docs/training.md](https://github.com/snonux/comicredr/blob/main/docs/training.md)
there has the detail, and this page links to it rather than repeat it.

## Setup, once

Both repositories side by side, Python 3.11 or later, no GPU:

```sh
cd ~/git
git clone https://github.com/snonux/comicredr
git clone https://github.com/snonux/comicredr-training-assets
cd comicredr
# CPU-only torch first: without the index URL pip also fetches some 3 GB of CUDA libraries
python3 -m pip install --user torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user -r spike/requirements-train.txt
```

The first run needs the network: archive.org and peppercarrot.com for
the comics, huggingface.co for the COCO starting weights. archive.org
answers 500 now and then; the scripts retry, and a run started again
keeps what it fetched. About 7 GB of disk for comics and pages.

## The one command

Everything goes through `make train-model` in ComicRedr
(`tool/train_model.sh`). It fetches the training comics, extracts the
labelled pages, makes the 400 synthetic pages, trains, exports, keeps the
run and scores it against the shipped model.

| What | Command, in `~/git/comicredr` | Time on 4 cores |
|---|---|---|
| The shipped recipe, from COCO | `make train-model` | 30 epochs, 5 to 6 hours |
| On top of the shipped model | `make train-model FROM=shipped` | 8 epochs, about 2 hours |
| On top of a kept run | `make train-model FROM=../comicredr-training-assets/checkpoints/NAME/last` | 8 epochs, about 2 hours |
| With `moredata/` added, on top | `make train-model LOCAL=more FROM=shipped` | 8 epochs, about 4 hours |
| With the real-art pages instead | `make train-model LOCAL=more FROM=shipped MORE=../comicredr-training-assets/moredata-art` | 8 epochs, about 4 hours |
| With `moredata/` and the NC/ND/SA books | `make train-model LOCAL=1` | from COCO, 10 hours or more |
| A quick check that it all runs | `make train-model EPOCHS=1 SCORE=0` | half an hour or so |
| Score any model against the shipped one | `make score-model MODEL=file.onnx` | some 10 minutes a model |

`EPOCHS=`, `LR=` and `IMGSZ=` change the training; a fresh run defaults
to 30 epochs at 1e-4, a run on top to 8 at 3e-5. An epoch takes about 12
minutes per 1,000 training pages on 4 cores: 16 minutes for the 1,270
pages of the plain set, 27 with `moredata/` added (2,306). The old data
is always in the run: `moredata/` has no balloons or captions, and a
model trained on it alone would forget them.

A run without `LOCAL` also puts its model in ComicRedr's
`assets/models/`, in place of the shipped one, uncommitted: that is how a
clean model gets shipped. Until it is committed, `FROM=shipped` and the
scores go by that file (the script says so);
`git checkout -- assets/models/comicredr-panels.onnx` puts the shipped one
back. With `LOCAL` the model never goes there.

## The real-art pages

`moredata/`'s fake art taught the detector the wrong thing (see "Runs so
far"). `real_art.py` keeps the layouts, gutters and border lines and
fills each frame with real art cut from the labelled frames of
ComicRedr's clean training pages, all frames of a page from one source
page where they fit, with the balloons and captions that come whole with
the art. It leaves out the pages that label two panels as one frame (a
thin gutter-coloured line across a frame). It needs ComicRedr's
extracted training pages, which `make train-model` makes:

```sh
python3 real_art.py moredata ../comicredr/spike/train_pages moredata-art    # about 5 minutes on one core
```

1,068 of the 1,123 pages, with 1,392 balloons and 178 captions, the same
pages every time for the same `--seed`. `moredata-art/` is git-ignored:
it is made from the two repositories in minutes. The pages go with the
model kept at home, as `moredata/` does.

## Where checkpoints live

From 2026-10-10 every run is kept, so the next one can start where it
ended. With this repository beside ComicRedr, runs go in `checkpoints/`
here (git-ignored: tens of MB each, and models trained on `moredata/` are
local only), one folder a run, named `DATE-fresh`, `DATE-on-top`,
`DATE-on-top-local-more` and so on, or `RUN=name`:

```
checkpoints/2026-10-10-on-top-local-more/
  start/                  the checkpoint the run started from, when FROM=shipped
  last/  best/            checkpoints of the last epoch and of the lowest validation loss
  state.pt                to resume: the same RUN= again goes on after the last finished epoch
  run.json                page folders and counts, command, starting weights, losses and minutes per epoch
  comicredr-panels.onnx   the exported model (from last/)
  scores/summary.md       shipped against this run on the three test sets
```

All of it is written after every epoch, so a killed run loses one epoch
at most. `CHECKPOINTS=` puts runs elsewhere; in a cloud container use
`CHECKPOINTS=/mnt/project-files/comicredr-checkpoints`, which outlives
the container. Copy a run folder between machines as it is.

## When a checkpoint is lost

The checkpoint of the run that made the shipped model was not kept.
`FROM=shipped` gets round that: ComicRedr's `spike/onnx_to_checkpoint.py`
rebuilds a trainable checkpoint from the ONNX file. The file holds every
weight the model uses to find panels, with the batch norms folded into
the convolutions. The script exports the same architecture once with a
marked value in every weight to see where each one went, puts the model's
own weights back there (each folded norm as an identity after its
convolution), and checks that exporting the result gives the same file:
for the shipped model every tensor matches within 1.2e-7. Two small parts
the export drops are filled in and settle in the first training steps:
the class and quality heads of decoder layers 0 and 1, which only the
training losses use, and the denoising label embedding. By hand:

```sh
python3 spike/onnx_to_checkpoint.py assets/models/comicredr-panels.onnx /tmp/shipped/last \
    --pages spike/train_pages spike/synth_pages
```

## Scoring

`tool/score_model.sh` (or `make score-model`) scores models on ComicRedr's
three test sets, which nothing is trained on: original 100 pages, modern
66, diagonal 24. It fetches and extracts them the first time and writes
`summary.md`: guided right / whole / wrong, panel F1, balloon F1, balloon
stops on captions, milliseconds a page and file size, every model on the
same machine. A wrong camera move is worse than showing a page whole, so
more "right" bought with more "wrong" is no gain, and a balloon F1 that
falls is a regression even when frames get better. The sets are small
(24 diagonal pages): do not tune on them.

## The licence rule

ComicRedr publishes a model only when it learned from public-domain, CC0
and CC BY books. `moredata/` holds no art, but its layouts come from
books outside that list, so a model trained on it (`LOCAL=more` or
`LOCAL=1`) is kept at home: it stays in `checkpoints/` and
`spike/out/local/`, never in ComicRedr's `assets/models/`, never
committed. Whether one may ever ship is the owner's decision. To use one
in the app: `make install-model MODEL=../comicredr-training-assets/checkpoints/RUN/comicredr-panels.onnx`
from ComicRedr, then `make install KEEP_MODEL=1`. Never train on the test
books; start only from the shipped model, a kept run or
`ustc-community/dfine-small-coco` (no Ultralytics, Objects365 or
Manga109).

## Runs so far

Scored with `tool/score_model.sh` on the same 4-core machine. Guided
right / whole / wrong, then panel F1 and balloon F1. Every run on top
started from the shipped model rebuilt from its ONNX file and trained 8
epochs at 3e-5.

| Model | Original 100 | Modern 66 | Diagonal 24 | Balloon stops on captions (orig.) |
|---|---|---|---|---|
| Shipped (998 real + 400 synthetic, 30 epochs from COCO) | 73 / 17 / 10, 0.903, 0.787 | 47 / 15 / 4, 0.894, 0.863 | 4 / 15 / 5, 0.910, 0.944 | 14 |
| On top, + `moredata/` (1,123 fake-art pages) | 72 / 16 / 12, 0.910, 0.799 | 39 / 20 / 7, 0.826, 0.884 | 4 / 15 / 5, 0.914, 0.892 | 18 |
| On top, no new pages (the control) | 70 / 16 / 14, 0.889, 0.777 | 49 / 11 / 6, 0.919, 0.872 | 5 / 14 / 5, 0.921, 0.932 | 15 |
| On top, + `moredata-art/` (1,068 pages, real art) | 70 / 17 / 13, 0.890, 0.783 | 50 / 9 / 7, 0.917, 0.855 | 5 / 12 / 7, 0.918, 0.917 | 22 |

Every model takes 150 to 165 ms a page in ONNX Runtime on 4 cores (the
spread is noise between runs) and is 42.8 MB.

What it showed (2026-10-10):

- **No run beats the shipped model.** Each one guides a few more modern
  pages right and makes more wrong camera moves, which count most. The
  shipped model stays.
- **The fake art hurts.** With `moredata/` the model joins neighbouring
  panels across a thin gutter (IHOW, Stigkland, Zombie Pandemic): modern
  panel recall fell from 0.88 to 0.80 and 8 fewer modern pages were
  guided right than in the control. Two causes. Each fake frame has a
  palette of its own, so the model learns that a panel ends where the
  colours change, and real neighbours often share colours. And 54 pages
  label two panels as one frame with the thin gutter between them still
  showing: border_pages.py closes a gutter off when a balloon lies
  across it, and the check against that used the shipped model, which
  misses the same merges.
- **`real_art.py` repairs the damage but adds nothing.** With real art
  in the frames and those 54 pages left out, modern panel F1 is back to
  the control's (0.917 against 0.919), but no better: what the layouts
  teach, the 400 synthetic pages and the labelled ones already did.
  Balloon stops on captions rose on the original set (14 to 22).
- **Training on top costs something by itself.** The control, with no
  new pages at all, already makes more wrong moves on the original set
  (10 to 14).
- The test sets are small; a page or two either way is noise. Do not tune
  on them.

Worth trying next, if anything: `border_pages.py` splitting a frame at a
hairline gutter instead of keeping it whole (it needs the comics, so on
the machine that has them), then a fresh 30-epoch run from COCO with the
pages in (`make train-model LOCAL=more MORE=../comicredr-training-assets/moredata-art`),
since a fresh run is what the shipped model is and a run on top may not
compare.

The runs are kept in the project's shared files,
`comicredr-checkpoints/2026-10-10-on-top-moredata`, `-on-top-control` and
`-on-top-art` (each with `scores/`), and `2026-09-26-shipped-rebuilt`, the
shipped model as a checkpoint.
