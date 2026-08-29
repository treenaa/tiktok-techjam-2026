# Data Contract

The interface between `src/data/` and every consumer (training, evaluation,
inference). Owned by the data pipeline. If you need a change here, ask — do not
work around it locally.

The executable version of this document is `src/data/contract.py`; anything
below can be asserted at runtime with `validate_sample` / `validate_batch`.

---

## 1. Labels

| Value | Meaning |
| --- | --- |
| `0` | real |
| `1` | AIGC (fully synthetic **or** manipulated) |

Binary throughout. Multi-class source datasets are collapsed by their adapter
(SID_Set's `tampered` → `1`). The label domain is enforced when a record is
constructed, so an out-of-range label cannot reach a batch.

---

## 2. Batch schemas

Two modes. Keys are exact — no aliases, no renames.

### 2.1 Standard mode (`MODE_STANDARD`)

Single view. Emitted by `ManifestDataset` and `TransformedEvalDataset`.

```python
{
    "image":      Tensor,   # or PIL.Image if no `preprocess` was supplied
    "label":      Tensor,   # int64, shape (B,), values in {0, 1}
    "source_id":  list[str],
    "image_path": list[str],
    # optional:
    "dataset":        list[str],
    "generator":      list[str],   # "" for real images
    "index":          Tensor,
    "transform_name": list[str],   # "clean" when no transform was applied
}
```

### 2.2 Paired robustness mode (`MODE_PAIRED`)

Clean + corrupted view of the **same** image. Emitted by `PairedViewDataset`.

```python
{
    "clean":      Tensor,   # x
    "augmented":  Tensor,   # T(x), same shape as "clean"
    "label":      Tensor,   # applies to BOTH views
    "source_id":  list[str],
    "image_path": list[str],
    # optional:
    "transform_name": list[str],   # which T was drawn, e.g. "jpeg_47"
    "dataset":        list[str],
    "generator":      list[str],
    "index":          Tensor,
}
```

Guarantees: `clean` and `augmented` derive from one source image, share
`label`/`source_id`/`image_path`, and have identical tensor shapes so they can be
concatenated. **The data layer does not implement any loss.**

### 2.3 Checking a batch

```python
from src.data import validate_batch, MODE_PAIRED
validate_batch(batch, mode=MODE_PAIRED, batch_size=32)   # raises SchemaError
```

Before a batch dimension exists, per-sample: `dataset.validate_schema()`.

### 2.4 Collation

Default `torch.utils.data.default_collate` is sufficient — no custom
`collate_fn`. Tensors stack; `source_id`/`image_path`/`transform_name` become
`list[str]` of length `B`. Verified under `num_workers=2`.

---

## 3. Image range and preprocessing ownership

| Stage | Type | Range | Owner |
| --- | --- | --- | --- |
| Leaves the dataset (no `preprocess`) | `PIL.Image`, mode `RGB` | `uint8`, `[0, 255]`, native resolution | data |
| Competition transform `T(x)` | `PIL.Image`, mode `RGB` | `uint8`, `[0, 255]`, native resolution | data |
| After `preprocess` | `torch.Tensor`, `CHW` | `float32`, `[0, 1]` then normalized | **model** |

**The raw dataset bakes in no model-specific preprocessing.** No I-JEPA, CLIP or
DINO statistics are applied unless a caller passes a `preprocess` callable:

```python
from src.data import ManifestDataset, build_preprocess

pre = build_preprocess("ijepa", image_size=224)      # model owner's choice
ds  = ManifestDataset(records, preprocess=pre)
```

`preprocess` is any `PIL.Image -> Any` callable — `ImagePreprocessing`, a
`torchvision.transforms.Compose`, or a HuggingFace `image_processor`. Presets in
`NORMALIZATION_PRESETS`: `imagenet`, `ijepa` (= imagenet), `clip`, `half`,
`none`.

**Ordering is part of the contract**: corruptions are applied at native
resolution *before* `preprocess` resizes. Blurring at 224px is not the same
operation as blurring at native size and then downscaling.

---

## 4. Official transform names

Stable, machine-readable, safe as filenames and dict keys. Retrieve with
`get_eval_transform(name)`; enumerate with `list_eval_transforms()`.

| Name | Family | Parameters | Severity |
| --- | --- | --- | --- |
| `clean` | identity | -- | 0 |
| `jpeg_90` | jpeg | quality=90 | 0 |
| `jpeg_70` | jpeg | quality=70 | 1 |
| `jpeg_50` | jpeg | quality=50 | 2 |
| `jpeg_30` | jpeg | quality=30 | 3 |
| `blur_0.5` | blur | sigma=0.5 | 0 |
| `blur_1.0` | blur | sigma=1.0 | 1 |
| `blur_2.0` | blur | sigma=2.0 | 2 |
| `resize_0.5` | resize | scale=0.5 | 0 |
| `resize_0.25` | resize | scale=0.25 | 1 |
| `noise_0.02` | noise | sigma=0.02 | 0 |
| `noise_0.05` | noise | sigma=0.05 | 1 |
| `noise_0.10` | noise | sigma=0.1 | 2 |
| `jitter_brightness_up` | jitter | brightness=1.2, contrast=1.0, saturation=1.0 |  |
| `jitter_brightness_down` | jitter | brightness=0.8, contrast=1.0, saturation=1.0 |  |
| `jitter_contrast_up` | jitter | brightness=1.0, contrast=1.2, saturation=1.0 |  |
| `jitter_contrast_down` | jitter | brightness=1.0, contrast=0.8, saturation=1.0 |  |
| `jitter_saturation_up` | jitter | brightness=1.0, contrast=1.0, saturation=1.2 |  |
| `jitter_saturation_down` | jitter | brightness=1.0, contrast=1.0, saturation=0.8 |  |
| `crop_0.80` | crop | ratio=0.8, resize_back=True | 0 |
19 corruptions + `clean` = **20** entries. `severity` orders a family from
mildest (`0`) upward; jitter directions are symmetric and carry `None`.

```python
from src.data import get_eval_transform, list_eval_transforms, describe_eval_transforms

get_eval_transform("jpeg_30")      # exact competition setting
list_eval_transforms()             # all 20, "clean" first, stable order
list_eval_transforms("blur")       # one family
describe_eval_transforms()         # JSON-serialisable spec incl. parameters
```

Deprecated spellings (`jpeg_q30`, `blur_sigma2.0`, `resize_0.25x`,
`noise_sigma0.10`, `crop_0.8`, `identity`) still resolve via
`canonical_transform_name`, but **write the canonical name** into result files.

### Determinism

* Every named transform is deterministic: same name + same image ⇒ identical
  pixels, across instances and processes (noise is seeded at `seed=0`).
* Training-time sampling uses `RandomCompetitionTransform`, which is stochastic
  by default and reproducible when constructed with a `seed`. Each *drawn*
  transform is itself replayable, and reports what it did via `.name`.

---

## 5. Manifest format

CSV (default), TSV, JSON or JSONL. These five columns are **always** present:

```
image_path,label,source_id,dataset,generator
```

`split` and any extra columns appear only when populated. `image_path` may be
relative — pass `root=` to `read_manifest` or the dataset.

`source_id` identifies the **underlying original image**. Every transformed
derivative of one image carries the *same* `source_id`; this is what makes the
splits leakage-safe.

---

## 6. Split guarantees

Produced by `split_records` / `assign_splits` / `split_by_generator_holdout`:

1. Splitting is by `source_id` group, never by individual image.
2. No `source_id` appears in two splits ⇒ no transformed derivative crosses a
   split.
3. Deterministic given `seed`, and independent of input record order.
4. Optional stratification by `label`, `dataset`, `generator`, or any
   combination.

Before training, run the gate:

```python
from src.data import validate_splits
validate_splits("manifests/train.csv", "manifests/val.csv", "manifests/test.csv")
```

It raises `LeakageError` listing **every** problem found: `source_id` overlap,
path overlap, filename-inferred derivative leakage, configured forbidden
dataset/generator combinations, empty splits, single-class splits.

---

## 6b. Protected data (rule 11.B) — NON-NEGOTIABLE

The competition's demonstration subset must **never** be trained or
model-selected on:

| Subset | Label | Images |
| --- | --- | --- |
| COCO val2017 | `0` real | 4998 |
| DALL·E Advanced | `1` AIGC | 8843 |

`validate_splits` enforces this **by default**. Protected data is recognised
from its `dataset` column *or* its path, so a mislabelled column does not defeat
the guard. It is permitted only in a split named `demo`, `demonstration`,
`benchmark` or `reference` — anywhere else (train, **val**, **test**) is a hard
failure, because val/test drive model selection and the final number.

```python
from src.data import assert_not_trainable, partition_protected

assert_not_trainable(records)                    # before any fit()
trainable, demo_only = partition_protected(pooled_records)
```

`allow_protected=True` exists as an escape hatch but must be a deliberate,
reviewable change at the call site. Never set it to silence a failure.

---

## 6c. Shortcut auditing (rule 11.C)

`audit_shortcuts` reports how easily a model could cheat by learning dataset
identity instead of AI-ness — the "fake 98% accuracy" trap:

```python
from src.data import audit_shortcuts, format_audit_report
print(format_audit_report(audit_shortcuts(train_records)))
```

Checks provenance (is `dataset` a perfect label predictor?), resolution,
encoding, file size, and generator concentration. Advisory by default —
findings carry `info` / `warning` / `critical`. **Report the findings in the
write-up rather than hiding them**; some skew is unavoidable at hackathon scale,
but unreported skew invalidates the headline number.

---

## 6d. Reproducibility (rule 20.9)

```python
from src.data import seed_everything, dataloader_kwargs

seed_everything(42)                                     # python + numpy + torch
DataLoader(ds, batch_size=32, shuffle=True, **dataloader_kwargs(seed=42, num_workers=4))
```

`dataloader_kwargs` supplies a seeded `generator` and `worker_init_fn`. Without
the latter, NumPy-based augmentation can draw *identical* values in every worker.

---

## 7. Integration checklist

```python
from src.data import (read_manifest, ManifestDataset, PairedViewDataset,
                      build_preprocess, validate_splits, validate_batch,
                      get_eval_transform, list_eval_transforms)

seed_everything(42)                                        # 0. reproducibility
validate_splits(train_csv, val_csv, test_csv)              # 1. gate (incl. rule 11.B)
pre = build_preprocess("ijepa", image_size=224)            # 2. model's choice
ds  = PairedViewDataset(read_manifest(train_csv), preprocess=pre)
validate_batch(next(iter(DataLoader(ds, batch_size=32))), mode="paired")
```

No-download fixture for CI:

```python
from src.data.synthetic import make_synthetic_dataset
bundle = make_synthetic_dataset(tmp_path)   # 24 images, 3 splits, manifests
```
