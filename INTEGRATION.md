# Integration guide — consuming `src/data`

For Mateo (models), Trina (training), Jamie (evaluation) and Ryan (product).
Melvin owns `src/data/`; this is what it guarantees and how to use it.

Authoritative references: [`DATA_CONTRACT.md`](DATA_CONTRACT.md) for exact
schemas and transform names, `src/data/README.md` for the API tour.

```bash
python -m pytest tests/ -q     # 595 passed
```

---

## The five-line version

```python
from src.data import (seed_everything, build_from_config, validate_splits,
                      ManifestDataset, PairedViewDataset, build_preprocess)

seed_everything(42)
splits = build_from_config("configs/data.yaml")      # train/val/test [+ demo]
validate_splits(splits["train"], splits["val"], splits["test"])
train = PairedViewDataset(splits["train"], preprocess=build_preprocess("ijepa", 224))
```

`validate_splits` raises rather than warns. If it raises, **do not train** —
the run would be scientifically invalid.

---

## Mateo — models

You receive tensors; you decide what preprocessing produces them.

```python
from src.data import build_preprocess, ImagePreprocessing, NORMALIZATION_PRESETS

build_preprocess("ijepa", image_size=224)    # ImageNet stats
build_preprocess("clip",  image_size=224)    # CLIP stats — genuinely different
ImagePreprocessing(image_size=224, normalization=((m...), (s...)))   # explicit
```

- The dataset yields **PIL RGB uint8 `[0,255]`** with no `preprocess`. Nothing
  model-specific is baked in — that is deliberate (rule 12) and tested.
- Pass any `PIL.Image -> Any` callable: `ImagePreprocessing`, a
  `torchvision.transforms.Compose`, or a HuggingFace `image_processor`.
- **Wrong normalization is a listed failure mode (§21).** CLIP and ImageNet
  statistics differ; using one backbone's stats with another silently degrades
  accuracy without erroring. Own this explicitly per backbone.
- `model.forward` returns **logits, no sigmoid** (rule 12). The data layer never
  applies one either.

For the forensic/FFT branch: take the *same* preprocessed tensor the visual
branch sees, or an explicitly documented parallel `preprocess`. Rule 21 lists
"FFT branch using inconsistent image scaling" — if the two branches disagree on
scaling, the fusion learns an artifact.

## Trina — training

```python
from src.data import (PairedViewDataset, RandomCompetitionTransform,
                      seed_everything, dataloader_kwargs)

seed_everything(42)
train = PairedViewDataset(splits["train"],
                          augment=RandomCompetitionTransform(seed=42),
                          preprocess=preprocess)
loader = DataLoader(train, batch_size=32, shuffle=True,
                    **dataloader_kwargs(seed=42, num_workers=4))

for batch in loader:
    logits_clean = model(batch["clean"])
    logits_aug   = model(batch["augmented"])
    labels       = batch["label"].float()       # BCEWithLogitsLoss wants float
```

- `clean` and `augmented` share `label` and `source_id` by construction, and
  have identical shapes.
- `batch["label"]` is `int64`; cast to float for `BCEWithLogitsLoss`, and pass
  **logits**, never probabilities (rule 21).
- `dataloader_kwargs` supplies `worker_init_fn`. **Without it, NumPy-based
  augmentation draws identical values in every worker** — a silent bug that
  quietly shrinks your effective augmentation diversity.
- `batch["transform_name"]` says which corruption was drawn, for logging or
  per-transform loss weighting.
- **macOS/Windows:** `num_workers > 0` spawns (not forks) processes, so any
  script that builds a DataLoader must guard its entry point with
  `if __name__ == "__main__":` or the workers re-import it and crash. Not a
  data-layer issue — it will bite `train.py` and `predict.py` equally.
- Threshold tuning belongs on **val only** (rule 11.D).

The consistency loss is yours; the data layer deliberately implements none.

## Jamie — evaluation

```python
from src.data import build_eval_datasets, list_eval_transforms, get_eval_transform

grid = build_eval_datasets(splits["test"], preprocess=preprocess)   # 20 datasets
for name, dataset in grid.items():
    ...   # every dataset holds the same records in the same order
```

- 20 named transforms: `clean` + 19 corruptions. Names are stable and
  machine-readable — write them into result files verbatim.
- Every grid dataset iterates the **same records in the same order**, so
  per-transform predictions align row-by-row and score drift is a subtraction.
- Evaluation transforms are **deterministic**, including noise (seeded). Rule 21
  lists "stochastic test transforms" as a failure mode; that cannot happen here,
  and `test_data_transform_audit.py` pins it.
- `describe_eval_transforms()` gives name/family/params/severity as JSON for
  your results table.
- AUROC needs **continuous scores**, not thresholded labels (rule 13).

Cross-generator and cross-source protocols:

```python
from src.data import split_by_field_holdout, list_field_values

split_by_field_holdout(records, field="generator", n_holdout=2, seed=0)
split_by_field_holdout(records, field="dataset", holdout=["sid_set"], holdout_label=None)
```

The holdout split contains **only** the held-out domains' AIGC images (plus real
images, so it stays two-class). Seen-generator AIGC never enters it, so the
cross-generator number is not diluted by in-distribution samples.

Before publishing headline numbers, run the shortcut audit and put its findings
in the write-up:

```python
from src.data import audit_shortcuts, format_audit_report
print(format_audit_report(audit_shortcuts(splits["train"])))
```

## Ryan — inference and product

**Use the shared loader.** Do not write a second image-reading path in
`predict.py` — if it diverges, the served model stops matching the evaluated one.

```python
from src.data import list_images, load_image, make_loader, verify_images

paths = list_images(args.input)              # sorted -> deterministic JSON order
loader = make_loader("placeholder")          # one bad file must not abort a run
```

- `list_images` returns **sorted** paths (rule 17 requires deterministic
  ordering) and accepts jpg/jpeg/png/webp/bmp/tif.
- Corrupt-file policy is explicit (rule 20.6). For the CLI, prefer
  `verify_images` up front and report unreadable files, or `make_loader("skip")`
  and count them. Never emit a fabricated score for a file you could not read —
  if a placeholder was used, say so; do not report it as a real prediction.
- The competition JSON takes `image_path` and `pred` **only** — no debug fields:

```json
[{"image_path": "images/a.jpg", "pred": 0.9342}]
```

- `pred` is P(AI-generated) = `sigmoid(logits)`. Applying sigmoid twice is a
  listed failure mode (§21) — the model returns logits, so sigmoid exactly once,
  in inference.
- For the demo's robustness panel, reuse `get_eval_transform(name)` rather than
  re-implementing corruptions, so the panel matches the benchmark.

---

## Run the audit CLI before you trust a number

```bash
# leakage + protected-data gate (exit 1 blocks the run)
python -m src.data.audit_cli splits   --config configs/baseline_clip.yaml
python -m src.data.audit_cli splits   --train t.csv --val v.csv --test s.csv

# rule 11.C: how easily could the model cheat on this corpus?
python -m src.data.audit_cli shortcut --config configs/baseline_clip.yaml [--strict]

# every image readable? (run before an inference batch)
python -m src.data.audit_cli verify   --input ./images

# rule 21: do these runs differ ONLY in `model`?
python -m src.data.audit_cli compare  configs/baseline_*.yaml
```

Exit codes: `0` clean, `1` blocking problem, `2` bad usage — so these drop into
CI unchanged. `--json` on any subcommand for machine-readable output.

## Comparable baselines (rule 21)

`configs/baseline_{clip,dino,ijepa}.yaml` share one split, seed, schedule and
evaluation protocol; only `model:` differs. `compare` verifies this
mechanically and prints a fingerprint per run — **identical fingerprints mean a
metric gap is attributable to the backbone.** If you change a training setting,
change it in all three (or the comparison is void, and `compare` will say so).

```python
from src.data import load_experiment, assert_comparable

configs = [load_experiment("configs/baseline_%s.yaml" % n) for n in ("clip","dino","ijepa")]
assert_comparable(configs)          # raises, naming the offending key
splits = configs[0].build_splits()  # the one split all three share
```

`config.model`, `config.training`, `config.evaluation` are passed through
untouched — Mateo and Trina own their contents and can add keys freely without
touching `src/data`.

---

## Guarantees you can rely on

1. No `source_id` spans two splits — no transformed derivative leaks (rule 11.A).
2. Demonstration-only data (COCO val2017, DALL·E Advanced) cannot enter
   train/val/test; `validate_splits` enforces this **by default** (rule 11.B).
3. Splits are deterministic given a seed and independent of input record order.
4. Labels are binary and validated at construction: `0` real, `1` AIGC.
5. Named evaluation transforms are deterministic and size-preserving.
6. Every batch matches the documented schema; `validate_batch` proves it.

## What the data layer does *not* do

- No losses (Trina), no models (Mateo), no metrics (Jamie), no CLI (Ryan).
- No dataset downloads. Point configs at data you already have.
- No guarantee the adapters match the real SID_Set/CIFAKE/WildFake layouts —
  they are validated against synthetic trees only. **First contact with real
  data may need `class_map` / `source_id_policy` tuning.** Tell Melvin rather
  than working around it locally.

## No-download fixture for your tests

```python
from src.data.synthetic import make_synthetic_dataset

bundle = make_synthetic_dataset(tmp_path)    # 24 images, 3 splits, manifests
```

Classes are deliberately learnable, so an integration test can assert accuracy
above chance without downloading anything. Rule 20.11: do not pull large
pretrained weights in unit tests — mock the backbone.
