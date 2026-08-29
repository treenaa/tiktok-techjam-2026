# `src/data` — data pipeline for binary AIGC detection

Real = `0`, AIGC = `1`. Everything here is dataset-agnostic: adapters turn a
folder layout into records, records go through a leakage-safe splitter, and
datasets hand the model whatever preprocessing *the model owner* supplies.

```
folders ──adapters──▶ [ManifestRecord] ──splitting──▶ train/val/test
                            │  ▲                          │
                       manifest.py (CSV/JSON)     validate_splits()  ← run before training
                                                          ▼
                                                  ManifestDataset          (standard mode)
                                                  PairedViewDataset        (paired mode)
                                                  TransformedEvalDataset   (robustness grid)
```

**The exact batch schemas and transform names live in
[`DATA_CONTRACT.md`](../../DATA_CONTRACT.md)** — read that first if you are
consuming this package. `src/data/contract.py` is its executable form.

## Quick start

```python
from src.data import (build_manifest, split_records, check_split_integrity,
                      PairedViewDataset, ManifestDataset, build_preprocess)
from src.data.transforms import RandomCompetitionTransform

records = build_manifest("cifake", "/data/cifake")        # or "sid_set", "wildfake", "folder"
splits  = split_records(records, ratios=(0.7, 0.15, 0.15), seed=42)
check_split_integrity(splits, original=records)           # raises on any leakage

preprocess = build_preprocess("ijepa", image_size=224)    # owned by the MODEL side
train = PairedViewDataset(splits["train"], augment=RandomCompetitionTransform(seed=0),
                          preprocess=preprocess)
val   = ManifestDataset(splits["val"], preprocess=preprocess)
```

## 1. The sample representation

`ManifestRecord` describes one image file. `ManifestDataset` turns it into:

```python
{"image": <PIL or tensor>, "label": 0|1, "source_id": str, "image_path": str,
 "dataset": str, "generator": str, "index": int, "transform_name": str}
```

**`source_id` is the load-bearing field.** It identifies the *underlying
original image*, so every transformed derivative of one image shares it and the
splitter cannot separate them. `make_source_id_fn` builds the policy:

| policy | id for `/d/train/REAL/cat_017_jpeg70.png` | use when |
|---|---|---|
| `stem` | `cat_017` | filenames are globally unique |
| `relpath` | `train/REAL/cat_017` (default in adapters) | filenames repeat across class folders |
| `parent` | `REAL` | all views of an image live in one folder |
| `regex` | first capture group | anything else |

Trailing transform markers are stripped by default, so `cat_017.png`,
`cat_017_jpeg70.png` and `cat_017_blur_1.0_jpeg30.png` all yield `cat_017`.
Every name in `EVAL_TRANSFORM_NAMES` is strippable. Pass `prefix=`/`dataset=` to
namespace ids so two datasets that both contain `img001.png` stay distinct
(`merge_manifests` does this for you).

## 2. Adapters and manifests

`build_manifest(name, root, **kw)` dispatches to `cifake`, `sid_set`,
`wildfake`, or the generic `folder`. All are thin wrappers over `from_folder`,
which labels images from directory names (`DEFAULT_CLASS_MAP`, deepest segment
wins) — so a new dataset is usually a `class_map=` argument, not new code.

- **CIFAKE** — `<root>/{train,test}/{REAL,FAKE}/*.jpg`. Its own train/test dirs
  land in `split`, but you can still re-split.
- **SID_Set** — `<root>/[split/]{real,synthetic,tampered}/`. Three classes are
  mapped to binary with **both `synthetic` and `tampered` → 1**; override with
  `class_map={"tampered": 0}`. `tampered_share_real_source_id=True` gives a
  tampered image the source id of the real image it was edited from.
- **WildFake** — `<root>/{real,fake}/<arch>/<model>/`. `generator_depth=0`
  records the architecture, `1` the concrete model.
- **Anything else** — `from_folder(root, class_map={"camera": 0, "midjourney": 1})`,
  `from_folder(root, label=1)` for a fixed-label directory, or
  `from_class_folders(real_dir, aigc_dir)`.

Manifests are the interchange format —
`image_path,label,source_id,dataset,generator[,split]` as CSV, TSV, JSON or
JSONL (picked by extension). Unknown columns survive a round-trip via
`record.extra`. Write with `relative_to=root` and read with `root=` to keep them
portable.

## 3. Leakage-safe splitting

```python
splits = split_records(records, ratios=(0.7, 0.15, 0.15), seed=42,
                       group_keys=("source_id",), stratify_keys=("label",))
```

- Splits **groups**, never images — no derivative can cross a split.
- `group_keys=("dataset", "source_id")` when pooling datasets with colliding ids.
- `stratify_keys` accepts any fields: `("label",)`, `("dataset", "label")`,
  `("generator", "label")`, or `None`.
- Deterministic in `seed` **and independent of input order** (groups are sorted
  before shuffling), so a reshuffled manifest reproduces the same split.
- `verify=True` (default) self-checks the result before returning it.

`assign_splits(...)` instead returns one flat list with `.split` set, for a
single manifest that carries its own splits (`read_manifest(path, split="train")`).

Checks, all raising `LeakageError`: `assert_no_source_id_leakage`,
`assert_no_path_overlap`, and `check_split_integrity(splits, original=records)`
which additionally proves the split is a *partition* of the input and no split
is empty. `split_report` / `format_split_report` summarise counts and balance.

Ratios are hit exactly at the group level: rounding remainders are carried
across strata, so a stratified split still totals 70/15/15 rather than drifting
to 70/16/14. Individual strata are therefore balanced only to within ±1 group.

## 4. Competition transforms

PIL-in/PIL-out callables with a stable `.name`, `.family` and `.params`. They
need no torchvision but compose with `torchvision.transforms.Compose`.

**A. Deterministic, for evaluation** — 20 named transforms in
`TRANSFORM_REGISTRY` / `EVAL_TRANSFORM_NAMES`:

| family | names |
|---|---|
| jpeg | `jpeg_q90` `jpeg_q70` `jpeg_q50` `jpeg_q30` |
| blur | `blur_sigma0.5` `blur_sigma1.0` `blur_sigma2.0` |
| resize | `resize_0.5x` `resize_0.25x` (down then back up) |
| noise | `noise_sigma0.02` `noise_sigma0.05` `noise_sigma0.10` |
| jitter | `jitter_{brightness,contrast,saturation}_{up,down}` (±20%) |
| crop | `crop_0.8` (centre 80% per side, resized back) |
| — | `clean` |

```python
get_transform("jpeg_q30")(img)      # same input -> byte-identical output
build_eval_datasets(test_records)   # {name: dataset}, rows aligned across the grid
```

Noise is seeded (`seed=0`) so evaluation is reproducible; pass `seed=None` for
fresh noise. All of them preserve input size and mode, including 1×1 images.

**B. Stochastic, for training** — `RandomCompetitionTransform` samples a family
and *continuous* parameters inside the competition ranges:

```python
aug = RandomCompetitionTransform(families=None, n_ops=(1, 2), p_identity=0.1, seed=0)
t   = aug.sample()     # a concrete transform; log t.name to know what was applied
img = aug(img)         # sample + apply
```

`sample()` returns a deterministic object, so a drawn augmentation can be
replayed. With `num_workers > 0`, re-seed per worker
(`worker_init_fn=lambda w: aug.set_seed(base_seed + w)`) or every worker draws
the same stream.

## 5. Paired views

```python
PairedViewDataset(records, augment=RandomCompetitionTransform(seed=0), preprocess=pre)[i]
# {"clean": x, "augmented": T(x), "label": ..., "source_id": ...,
#  "image_path": ..., "dataset": ..., "generator": ..., "index": ..., "transform": "jpeg_q37"}
```

Both views come from the same file and always carry the same `label` and
`source_id`. `augment` may be a sampler (fresh corruption per access — training)
or a fixed callable/registry name (reproducible — debugging). Use
`augmented_preprocess=` to preprocess the two branches differently.

**No loss is implemented here** — this class only produces views.

## 6. Model-aware normalization

Normalization is deliberately *not* baked into the dataset: with no
`preprocess`, datasets yield raw PIL images. The model owner passes any
`PIL.Image -> Any` callable — `ImagePreprocessing`, a torchvision `Compose`, or
a HuggingFace image processor:

```python
build_preprocess("ijepa", image_size=224)   # ImageNet stats (also "clip", "half", "none")
ImagePreprocessing(224, resize_mode="shortest", normalization=(mean, std))
```

Ordering is deliberate: **competition transforms run on the raw image at native
resolution, preprocessing runs after.** Corrupting after a resize to 224 would
destroy the very artifacts the detector keys on.

## Testing

`python3 -m pytest tests/ -q` — 259 tests, no dataset downloads (synthetic trees
are built in `tmp_path` by `tests/test_data_fixtures.py`).


## 9. Second-pass additions

### Batch schema validation (`contract.py`)

```python
from src.data import validate_batch, validate_sample, MODE_STANDARD, MODE_PAIRED

validate_batch(batch, mode=MODE_PAIRED, batch_size=32)   # raises SchemaError
ManifestDataset(records, preprocess=pre).validate_schema()
```

Standard mode requires `image, label, source_id, image_path`; paired mode
requires `clean, augmented, label, source_id, image_path`. Undocumented keys are
rejected by default, so a drifting key name fails immediately rather than
silently.

### The transform registry

```python
from src.data import get_eval_transform, list_eval_transforms, describe_eval_transforms

get_eval_transform("jpeg_30")        # jpeg_90/70/50/30
get_eval_transform("blur_2.0")       # blur_0.5/1.0/2.0
get_eval_transform("resize_0.25")    # resize_0.5/0.25
get_eval_transform("noise_0.10")     # noise_0.02/0.05/0.10
get_eval_transform("crop_0.80")      # crop_0.80
list_eval_transforms()               # all 20 (clean first), stable order
list_eval_transforms("jitter")       # 6 jitter variants, +-20%
describe_eval_transforms()           # JSON-serialisable: name/family/params/severity
```

Old spellings (`jpeg_q30`, `blur_sigma2.0`, `resize_0.25x`, `crop_0.8`) still
resolve through `canonical_transform_name`, but write canonical names into
results.

### Pre-training split gate

```python
from src.data import validate_splits

validate_splits("manifests/train.csv", "manifests/val.csv", "manifests/test.csv",
                forbidden={"test": {"generator": ["some_generator"]}})
```

Raises `LeakageError` listing *every* problem: `source_id` overlap, path
overlap, filename-inferred derivative leakage (catches a bad `source_id` policy),
forbidden dataset/generator combinations, empty and single-class splits. Pass
`raise_on_failure=False` to inspect `report.problems` instead.

### Generator-aware splits

No generator name is hard-coded anywhere:

```python
from src.data import (list_generators, filter_by_generator,
                      split_by_generator_holdout, assert_generators_disjoint)

list_generators(records)                                    # discover what exists
filter_by_generator(records, exclude=["sdxl"])              # keeps real images
splits = split_by_generator_holdout(records, holdout=["sdxl"])   # unseen-generator test
splits = split_by_generator_holdout(records, n_holdout=2, seed=0)
assert_generators_disjoint(splits)
```

Held-out generators appear only in the holdout split; real images are spread
across all splits so none is single-class; `source_id` grouping still holds.

### Manifest generation

```python
from src.data import generate_manifest, generate_split_manifests

generate_manifest("/data/cifake", adapter="cifake", out_path="manifests/cifake.csv")
generate_split_manifests("/data/cifake", "manifests/", adapter="cifake", seed=0)
```

```bash
python -m src.data.build --root /data/cifake --adapter cifake --out-dir manifests/
```

### Synthetic fixture (no downloads)

```python
from src.data.synthetic import make_synthetic_dataset

bundle = make_synthetic_dataset(tmp_path)      # 24 images, 3 splits, manifests written
bundle.train, bundle.val, bundle.test          # record lists
bundle.train_manifest                          # CSV path
```

The two classes are deliberately learnable (AIGC images carry a periodic
over-smooth structure), so an integration test can assert accuracy above chance
without the task being trivial. Used by `tests/test_data_synthetic.py` to cover
dataset → dataloader → train → evaluate end to end.
