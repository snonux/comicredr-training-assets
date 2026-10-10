# Handoff: train ComicRedr's panel detector on top of what it already learned

Written 2026-10-10 for whoever (person or LLM) does the next training run.
Read this whole page first; `README.md` here and `docs/training.md` in
ComicRedr have the detail.

> **Done, 2026-10-10.** This page is kept as the brief of that run. How to
> train now is [TRAINING.md](TRAINING.md); the results are in its "Runs
> so far". In short: no run beat the shipped model, which stays. What
> turned out wrong below:
>
> - The checkpoint is not needed. `make train-model FROM=shipped`
>   rebuilds it from the ONNX file (`spike/onnx_to_checkpoint.py`), so
>   path A works anywhere; path B was not taken.
> - The hook branch is merged into ComicRedr's `main`. `LOCAL=1` also
>   adds the NC/ND/SA books; `LOCAL=more` adds only `moredata/`.
> - `spike/requirements-train.txt` lacked scipy (D-FINE's matcher needs
>   it), and a plain `pip install torch` fetches some 3 GB of CUDA
>   libraries: install the CPU wheel first (TRAINING.md, "Setup").
>   Python 3.13 works as well as 3.11.
> - An epoch with `moredata/` took 26 to 28 minutes on 4 cores, not 20
>   (2,306 training and 215 validation pages, as expected).
> - Runs now go to `checkpoints/RUN/` here, kept every epoch and
>   resumable, not to `spike/out/local/`.
> - "The fake pages are easier, so a small gain or none": they made modern
>   pages worse (39 right instead of 47). The made-up art is the cause,
>   and 54 pages label two panels as one frame; `real_art.py` fixes both.

## The task

Continue training the shipped panel detector with the 1,123 fake pages in
`moredata/` **added to** the data it was trained on, score the result on
ComicRedr's three test sets, and report whether it beats the shipped
model. Nothing is replaced: the 998 labelled pages and the 400 synthetic
pages stay in the set.

Not done yet, by anyone: no model has been trained on `moredata/`. Whether
these pages help is the open question this run answers.

## Where things are

| | |
|---|---|
| `~/git/comicredr` | the app, `spike/train.py`, `spike/evaluate.py`, labels, test manifests, `tool/train_model.sh` |
| `~/git/comicredr-training-assets` | this repository: `moredata/` and the scripts that made it |
| `assets/models/comicredr-panels.onnx` (in ComicRedr) | the shipped model: D-FINE-S, 30 epochs at 640 px on 998 real + 400 synthetic pages |

Both repositories must sit side by side. In ComicRedr, the branch
`claude/training-assets-reference` holds the hook that adds `moredata/`
(`MORE=` in `tool/train_model.sh`) and the docs pointer. It is pushed but
**not merged into `main`**: check it out or merge it first.

```sh
cd ~/git/comicredr && git fetch && git switch claude/training-assets-reference
cd ~/git/comicredr-training-assets && git pull
python3 -m pip install --user -r ~/git/comicredr/spike/requirements-train.txt   # Python 3.11
```

## First: is the old checkpoint there?

"On top of the training already done" needs the PyTorch checkpoint of the
shipped run, `spike/out/train/last/` in ComicRedr. It is git-ignored. It
was **not** on the machine where `moredata/` was made (no `spike/out/` at
all), so look on the machine that trained the shipped model:

```sh
ls ~/git/comicredr/spike/out/train/last/    # config.json, model.safetensors, comicredr.json
```

The ONNX file in `assets/models/` cannot be trained further; it is an
export. So:

- **Checkpoint found** -> path A, fine-tune it. About 3 hours on 4 CPU
  cores.
- **No checkpoint** -> path B, train again from the COCO weights with all
  three sets together. The model then learns the old data and the new in
  one run, which gives the same "on top of" result; it takes 10 hours or
  more, since the set is about twice as big as the 5 to 6 hour shipped run.

Either way the old data must be in the run. Training on `moredata/` alone
would make the model forget balloons and captions: these pages have none.

## Prepare the pages (both paths)

Run from `~/git/comicredr`. This is what `tool/train_model.sh` does before
it trains; it needs the network (archive.org, peppercarrot.com).

```sh
python3 spike/fetch_corpus.py --manifest test/train.manifest.toml --out test/corpus-train
python3 spike/extract_pages.py test/corpus-train spike/train_pages --per-book 400 --manifest test/train.manifest.toml
find spike/train_pages -name '*.json' ! -name '*.cand.json' -delete
python3 spike/labelkit.py import spike/labels/train spike/train_pages
rm -rf spike/synth_pages
python3 spike/synth_modern.py spike/train_pages spike/synth_pages --count 400 --seed 1
```

archive.org answers 500 now and then; run the fetch again, it keeps what
it has. Expect 998 labelled pages and 400 synthetic ones.

## Path A: fine-tune the shipped checkpoint

`spike/train.py --weights` takes a checkpoint folder as well as a Hugging
Face name, so no code change is needed.

```sh
python3 spike/train.py spike/train_pages spike/synth_pages ../comicredr-training-assets/moredata \
    --weights spike/out/train/last --out spike/out/local/more \
    --epochs 8 --lr 3e-5 --imgsz 640
```

- `--out` must not be `spike/out/train`: that would overwrite the
  checkpoint being continued. Copy `spike/out/train/last` somewhere safe
  before starting anyway.
- `--epochs 8 --lr 3e-5` is a starting point, not a tested recipe. The
  shipped run used 30 epochs at 1e-4 from COCO; a third of that rate and a
  quarter of the epochs is a guess at "learn the new layouts without
  unlearning the old". If the scores below barely move, try
  `--epochs 12 --lr 5e-5`; if balloon scores drop, see "If it goes wrong".
- The script prints `dataset: N train, M val pages`. Expect about 2,270
  and 250 (a tenth is held out by a hash of the page's file name). Fewer
  than 2,400 in all means a folder is missing.
- An epoch is about 20 minutes on 4 cores (`--threads` sets the count).

## Path B: no checkpoint, train again with everything

```sh
make train-model LOCAL=1 EPOCHS=1     # optional: once through the pipeline, about an hour
make train-model LOCAL=1              # the real run
```

`LOCAL=1` adds `../comicredr-training-assets/moredata` when its
`index.tsv` is there, and says so (`== LOCAL=1: also the fake pages in
...`). If that line does not appear, the hook branch is not checked out or
the repositories are not side by side. `LOCAL=1` also adds the books of
`test/train-local.manifest.toml`; that is intended, see "Licence" below.
The model goes to `spike/out/local/comicredr-panels.onnx`, and the
checkpoint to `spike/out/local/train/last/`: keep that folder, it is what
the next "on top of" run starts from.

## Export (path A; path B does it itself)

```sh
python3 spike/train.py --export spike/out/local/more/last --out-onnx spike/out/local/comicredr-panels.onnx
python3 -c "
import onnxruntime as ort, numpy as np
s = ort.InferenceSession('spike/out/local/comicredr-panels.onnx')
print(s.run(None, {'images': np.zeros((1, 3, 800, 800), np.float32)})[0].shape)"   # must print (1, 300, 6)
```

Export `last/`, not `best/`: the shipped model was chosen that way because
the last epoch guided more test pages right than the lowest validation
loss did. Score both if there is time.

## Score it

The three test sets are never trained on. Fetch and extract them once:

```sh
python3 spike/fetch_corpus.py --out test/corpus
python3 spike/fetch_corpus.py --manifest test/modern.manifest.toml --out test/corpus-modern
python3 spike/fetch_corpus.py --manifest test/diagonal-eval.manifest.toml --out test/corpus-diagonal-eval
python3 spike/extract_pages.py test/corpus spike/eval_pages --per-book 400
python3 spike/extract_pages.py test/corpus-modern spike/modern_pages --per-book 400 --manifest test/modern.manifest.toml
python3 spike/extract_pages.py test/corpus-diagonal-eval spike/diagonal_pages --per-book 400 --manifest test/diagonal-eval.manifest.toml
python3 spike/labelkit.py import spike/labels/eval spike/eval_pages
python3 spike/labelkit.py import spike/labels/modern spike/modern_pages
python3 spike/labelkit.py import spike/labels/diagonal-eval spike/diagonal_pages
```

Then score the shipped model and the new one the same way, on the same
machine, so the comparison is fair:

```sh
cd spike
for s in eval modern diagonal; do
  python3 evaluate.py ${s}_pages --out out/base-$s --no-cv --trim --trained ../assets/models/comicredr-panels.onnx
  python3 evaluate.py ${s}_pages --out out/more-$s --no-cv --trim --trained out/local/comicredr-panels.onnx
done        # out/base-*/report.md and out/more-*/report.md
```

The numbers to beat (guided right / page shown whole / guided wrong), from
`docs/training.md`. Your `base-*` run should reproduce the last row; if it
does not, say so and compare against your own base numbers.

| Model | Original 100 | Modern 66 | Diagonal 24 |
|---|---|---|---|
| D-FINE-S, 998 real pages | 69 / 22 / 9 | 43 / 21 / 2 | 6 / 13 / 5 |
| D-FINE-S, 998 real + 400 synthetic (shipped) | 73 / 17 / 10 | 47 / 15 / 4 | 4 / 15 / 5 |

How to read the result:

- "Guided wrong" matters most. A wrong camera move is worse than showing
  the page whole. More "right" bought with more "wrong" is not a win.
- The reports also give panel F1, **balloon F1** and how often balloon mode
  stops on a caption. `moredata/` has no balloons or captions, so these are
  where harm would show. A drop there is a regression even if frames
  improve.
- Where gains are plausible: the diagonal and modern sets (slanted,
  touching, notched frames). The original 100 are mostly plain grids, which
  the model already does well; it should hold there, not rise.
- The fake pages are easier for the shipped model than the real pages they
  were made from (it already finds 93 to 100% of their frames). A small
  gain, or none, is a real possible outcome. Report it as it is.

## If it goes wrong

- **Balloon or caption scores drop.** Too large a share of balloon-free
  pages. Use fewer: `moredata/index.tsv` has a `layout` column; drop the
  `grid` and `single` rows first (589 pages, the layouts the labelled set
  already has most of), which leaves 534 pages of slanted, touching,
  notched, rounded, polygon and round frames. Copy the kept pages (each
  `.jpg` with its `.json`, keeping the `<book>/` folders) to another
  folder and pass that instead of `moredata`.
- **Path A forgets old pages** (the original 100 fall): lower `--lr` to
  2e-5 or fewer epochs.
- **Nothing changes**: the model already knows these layouts as far as
  made-up art can teach them. Stop there; do not keep tuning on the test
  sets, they are small (24 diagonal pages) and a few pages of noise look
  like progress.
- **`train.py` cannot load the checkpoint**: check that
  `spike/requirements-train.txt` versions are installed; the checkpoint
  was written by transformers 5.17.

## Rules that must hold

- **Licence.** ComicRedr publishes a model only if it learned from
  public-domain, CC0 and CC BY books. `moredata/` holds no art, but its
  layouts come from other books, so a model trained on it is a **local**
  model: it stays in `spike/out/local/`. Do not copy it to
  `assets/models/`, do not run `tool/fetch_model.sh` on it, do not commit
  it. Whether such a model may ever ship is the owner's decision, not
  yours; say what you measured and leave it there. To try it in the app:
  `make install-model MODEL=spike/out/local/comicredr-panels.onnx`, then
  `make install KEEP_MODEL=1`.
- **Never train on the test books**, and never label their pages.
- **No Ultralytics code or weights, no Objects365 checkpoints, nothing
  trained on Manga109.** Start only from the shipped checkpoint or from
  `ustc-community/dfine-small-coco`.
- **CPU only**, as the recipe is written; nothing here needs a GPU.
- Do not commit downloaded comics, extracted pages or checkpoints; they are
  git-ignored for that reason. `scan/` and `moredata/sources.tsv` in this
  repository list the owner's comic files and stay untracked.

## What to hand back

1. Which path was taken, and the exact `train.py` command.
2. The six `report.md` files, and one table: shipped against new, on the
   three sets, guided right / whole / wrong plus panel F1 and balloon F1.
3. A plain verdict: better, worse or no difference, and on which layouts.
4. Where the new checkpoint is (`spike/out/local/.../last/`), kept for the
   next run.
5. Anything in this page that turned out wrong.
