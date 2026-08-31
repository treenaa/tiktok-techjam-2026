# Robust AIGC Image Detection

Prototype for TikTok TechJam 2026: classify an image as authentic (`0`) or
AI-generated (`1`) while remaining useful after JPEG compression, blur, resize,
noise, color changes, and cropping.

The required submission interface is implemented by `predict.py`. It accepts an
image directory and writes only the probability of AIGC:

```json
[
  {"image_path": "images/example.jpg", "pred": 0.9342}
]
```

`pred` is always `P(AI-generated)`. Models return raw logits; sigmoid is applied
exactly once in inference.

## Quickstart

Everything below runs on CPU. Verified on a clean macOS/M4 checkout.

```bash
git clone https://github.com/treenaa/tiktok-techjam-2026.git
cd tiktok-techjam-2026

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# the final model (robustness-aware DINOv2 + FFT fusion)
mkdir -p runs/robust_dino_fusion
curl -L --fail -o runs/robust_dino_fusion/best.pt \
  https://github.com/treenaa/tiktok-techjam-2026/releases/download/v0.1-baselines/robust_dino_fusion-best.pt

sha256sum runs/robust_dino_fusion/best.pt    # macOS: shasum -a 256
# expect 1fced519c8ecd6dc25b456c0ac872afa452b013a653b795cd24a8c483f39a20c

# required competition inference
python predict.py \
  --input ./images \
  --output predictions.json \
  --checkpoint runs/robust_dino_fusion/best.pt \
  --device auto

# judging demo
streamlit run app/studio.py -- \
  --checkpoint runs/robust_dino_fusion/best.pt \
  --device auto
```

**Downloads.** The checkpoint is ~348 MB. The first run also pulls the DINOv2
backbone (~330 MB) from Hugging Face, so budget roughly **680 MB** and a working
network on first execution. Both are cached afterwards; later runs are offline.

**Timings** on an M4 CPU, once cached: ~3 s to load the model, ~0.3 s for the
six-view analysis, ~0.9 s for all twenty transformations.

### Tested environment

`requirements.txt` carries lower bounds only. This exact combination was
verified end to end on a clean machine:

```text
Python        3.12.7
torch         2.13.0
transformers  5.16.1
streamlit     1.62.0
Pillow        12.3.0
numpy         2.5.2
pillow-heif   1.6.0
```

## Status

The data, model, training, evaluation, inference, and demo code paths are
implemented and covered by download-free tests.

Three phase-one baselines and one robustness-aware model have been trained and
evaluated on real data. Every number below comes from a real run recorded under
`results/`; none is inferred from synthetic fixtures.

The robustness-aware run (`robust_dino_fusion`, `configs/robust_dino_fusion.yaml`)
enables paired clean/transformed training, a prediction-consistency loss, and
the FFT forensic branch. It shares its dataset, split, seed, schedule and
threshold protocol with the CIFAKE baseline, so the two are directly
comparable; see [Robustness-aware training](#robustness-aware-training).

What has **not** been run yet, and is therefore not claimed anywhere:

- CLIP and I-JEPA baselines (`configs/baseline_clip.yaml`,
  `configs/baseline_ijepa.yaml` exist but have no results).
- An ablation separating the three robustness interventions. Paired
  augmentation, the consistency loss and the forensic branch were enabled
  together in a single run, so no individual contribution is claimed — in
  particular, the forensic branch is not shown to help on its own.
- Any evaluation outside the training augmentation ranges. Improved robustness
  is demonstrated on the corruption families the model was trained against; it
  is not shown to transfer to unseen corruptions.
- A combined multi-dataset model, and cross-dataset (train on A, test on B)
  generalisation.
- The organisers' demonstration subset (COCO val2017 + DALL-E Advanced). It is
  structurally blocked from training and has not been scored.

## Results

Phase-one baseline: DINOv2 ViT-B/14 (`facebook/dinov2-base`), frozen backbone
plus a linear head. 86,582,785 parameters, within the 2B competition limit.
Each model was trained and tested on a single dataset with a source-grouped
70/15/15 split, seed 42, threshold selected on validation.

| Test set | n | Clean AUROC | Mean transformed AUROC | Mean AUROC drop | Worst transform | Worst AUROC | Mean class-flip rate |
|---|---|---|---|---|---|---|---|
| CIFAKE | 18,000 | 0.9867 | 0.9565 | 0.0302 | `resize_0.25` | 0.8209 | 0.144 |
| SID_Set | 900 | 0.9353 | 0.9346 | 0.0007 | `jpeg_30` | 0.9322 | 0.019 |
| WildFake | 600 | 0.9904 | 0.9831 | 0.0073 | `noise_0.10` | 0.9527 | 0.031 |

These are the phase-one baselines. The robustness-aware model trained on CIFAKE
is reported separately below, against the CIFAKE row.

Read these with the following caveats:

- The SID_Set and WildFake test sets are small (900 and 600 images). Their
  robustness looks excellent, but the support is too thin to headline.
- CIFAKE is 32x32 imagery upscaled to 224. Aggressive blur and downscaling
  destroy its signal: at `resize_0.25` accuracy falls to 0.561 and recall to
  0.138 while specificity stays at 0.984, i.e. the model stops calling anything
  AI-generated rather than making balanced mistakes. This is the failure mode
  the robustness-aware run below addresses.
- Generator metadata is only populated for the WildFake run, so cross-generator
  slices are empty in the other reports.

### Robustness-aware training

`robust_dino_fusion` differs from the CIFAKE baseline in exactly four settings —
`augment: competition`, `consistency_weight: 0.5`, the clean/augmented loss
weights, and `forensic_branch: true`. Dataset, split, seed, schedule and
threshold protocol are identical, so this is a controlled before/after on the
same 18,000-image test set. The forensic branch adds 110,048 parameters
(86,582,785 → 86,692,833) and no measurable throughput cost (227 images/s in
both runs).

| CIFAKE, n=18,000 | Baseline | Robustness-aware | Δ |
|---|---|---|---|
| Clean AUROC | 0.9867 | 0.9920 | +0.0053 |
| Mean transformed AUROC | 0.9565 | 0.9713 | +0.0148 |
| Mean AUROC drop | 0.0302 | 0.0207 | −0.0095 |
| Worst transform (`resize_0.25`) | 0.8209 | 0.8796 | +0.0587 |
| Mean class-flip rate | 0.144 | 0.067 | −0.077 |
| Mean absolute score drift | 0.142 | 0.078 | −0.064 |

Clean accuracy did not regress, so there is no robustness/accuracy trade-off in
this run. The recall recovery is concentrated exactly where the baseline
collapsed:

| Recall (CIFAKE) | Baseline | Robustness-aware |
|---|---|---|
| `resize_0.25` | 0.138 | 0.637 |
| `noise_0.10` | 0.156 | 0.850 |
| `blur_2.0` | 0.232 | 0.724 |
| `resize_0.5` | 0.366 | 0.817 |
| `blur_1.0` | 0.504 | 0.861 |

Clean AUROC and 18 of the 19 transformed conditions improved. The exception is
`crop_0.80` at
−0.0023, which is within run-to-run noise but is not claimed as an improvement.
The recall gains are partly paid for in specificity under heavy degradation
(at `resize_0.25`, 0.984 → 0.906): the baseline was not "safer", it had
degenerated into predicting `real` for almost everything. See
[`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md).

Two limits on how far this result can be read. The training augmentation samples
continuous ranges that span every evaluated severity, so this demonstrates
robustness on the trained corruption families rather than transfer to unseen
ones. And all three interventions were enabled together, so the contribution of
the forensic branch specifically is not established.

Note that `python -m src.data.audit_cli compare` reports `robust_dino_fusion` as
NOT COMPARABLE against the baselines. That is correct and expected: rule 21
requires that only `model` varies, which is the right guard for the backbone
comparison and deliberately not satisfied by an intervention run.

Full per-transform metrics, stability, subgroup slices, representative errors,
and runtime are in `results/<run>/report.json` and `results/<run>/robustness.csv`.

## Pretrained checkpoints

Checkpoints are not stored in git; they are published as assets on the
[v0.1-baselines release](https://github.com/treenaa/tiktok-techjam-2026/releases/tag/v0.1-baselines).

**Use `robust_dino_fusion` unless you specifically want a baseline.** It is the
robustness-aware model reported above and the checkpoint every command in this
README assumes. Download it into `runs/`:

```bash
mkdir -p runs/robust_dino_fusion
curl -L -o runs/robust_dino_fusion/best.pt \
  https://github.com/treenaa/tiktok-techjam-2026/releases/download/v0.1-baselines/robust_dino_fusion-best.pt
```

Each file is roughly 350 MB and contains the model, optimizer, scheduler, and
scaler state plus the validation-selected threshold, so `predict.py` and
`evaluate.py` can reconstruct the model without extra flags.

| Run | Data | Validation threshold | Published |
|---|---|---|---|
| `baseline_dino_real` | CIFAKE | 0.4668375 | yes |
| `baseline_dino_sidset` | SID_Set | 0.4877687 | yes |
| `baseline_dino_wildfake_v3` | WildFake | 0.3831851 | yes |
| `robust_dino_fusion` | CIFAKE | 0.6042042 | yes |

Verify a download against the checksums below. Use `sha256sum` on Linux, or
`shasum -a 256` on macOS:

```text
bd2e27126805c474213cfca17a24479c6141426f1c156b80efa2ca9357f22a04  baseline_dino_real-best.pt
87c244f35b5b9ad2a79b1e08d98a512f48ead87e14125bc668b9e0713edc1e6b  baseline_dino_sidset-best.pt
215819264f3f83f0e666b4af5901ed63acb613eda96a07bd5d1060bd7d2c9f08  baseline_dino_wildfake_v3-best.pt
1fced519c8ecd6dc25b456c0ac872afa452b013a653b795cd24a8c483f39a20c  robust_dino_fusion-best.pt
```

## Approach

The project is designed to test three pretrained visual representations under
the same split, head, training schedule, seed, and robustness benchmark. Only
DINOv2 has been run so far; see Status. The robustness phase below has been run
on DINOv2 and is reported in Results.

Candidate backbones:

- I-JEPA
- DINOv2
- CLIP

The phase-one detector is a visual encoder plus a small binary head. The
robustness model adds, on top of that backbone:

- paired clean/transformed training;
- prediction-consistency loss;
- a lightweight forensic branch over normalized log FFT magnitude;
- feature-level visual/forensic fusion.

All four are enabled together in `robust_dino_fusion`, which is the run reported
in Results. Because they were switched on in a single run, the results measure
the combination and not any individual component.

The forensic branch explicitly reverses the visual backbone's input
normalization before the FFT. This avoids turning CLIP-versus-ImageNet scaling
into a fake forensic signal.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For a guided Google Colab GPU run, open
[`notebooks/techjam_aigc_colab.ipynb`](notebooks/techjam_aigc_colab.ipynb).
It downloads and audits CIFAKE, stores checkpoints on Google Drive, trains the
one-epoch DINOv2 integration run, and verifies the submission inference path.

Tests inject tiny local encoders and do not download pretrained weights.
Constructing a real I-JEPA, DINOv2, or CLIP detector may download its configured
Hugging Face checkpoint unless `local_files_only: true` is set and the model is
already cached.

## Data

Copy a configuration and replace the placeholder roots:

```bash
cp configs/data.example.yaml configs/data.yaml
```

Supported adapters include CIFAKE, WildFake, SID_Set, and generic class
folders. Splitting is by stable `source_id`; transformed derivatives of one
source cannot cross train/validation/test boundaries.

The following organizer demonstration/reference data is protected and must not
enter training, validation, or test model selection:

- COCO val2017: 4,998 real images
- DALL·E Advanced: 8,843 AIGC images

The data validator blocks these subsets outside a dedicated demonstration or
reference split. Dataset licenses and redistribution rights must be verified
before publishing any assets.

## Train

Phase-one backbone configurations are provided in `configs/`:

```bash
python train.py \
  --config configs/baseline_dino.yaml \
  --output-dir runs/baseline_dino \
  --device auto
```

Resume without changing the training method:

```bash
python train.py \
  --config configs/baseline_dino.yaml \
  --output-dir runs/baseline_dino \
  --resume runs/baseline_dino/last.pt
```

`train.py` builds train and validation loaders only. Threshold tuning, early
stopping, and checkpoint selection use validation. Test data is reserved for
the final `evaluate.py` run.

For robustness-aware training, set:

```yaml
training:
  augment: competition
  clean_loss_weight: 1.0
  augmented_loss_weight: 1.0
  consistency_weight: 0.5
```

## Evaluate

Run clean plus all 19 official corruptions:

```bash
python evaluate.py \
  --manifest manifests/test.csv \
  --model-factory src.models:create_model \
  --model-kwargs '{"backbone":"dinov2","architecture":"fusion"}' \
  --preprocess-factory src.models:create_preprocess \
  --preprocess-kwargs '{"backbone":"dinov2"}' \
  --checkpoint runs/robust_dino_fusion/best.pt \
  --threshold "$VALIDATION_THRESHOLD" \
  --output-dir results/final \
  --save-predictions
```

The threshold is stored in the checkpoint and is printed by `predict.py`; for
`robust_dino_fusion` it is `0.6042042`. Never select it on test data. The report
contains AUROC, accuracy, F1, precision, recall, specificity, false-positive
rate, AUROC degradation, score drift, class flips, dataset/generator slices,
representative errors, and runtime measurements.

## Required directory inference

Training checkpoints are self-describing, so the usual command is:

```bash
python predict.py \
  --input ./images \
  --output predictions.json \
  --checkpoint runs/robust_dino_fusion/best.pt
```

Files are sorted deterministically. Supported formats are JPG/JPEG, PNG, WebP,
BMP, and TIFF. By default one unreadable file aborts before output is written.
To explicitly skip corrupt files and receive a separate error report:

```bash
python predict.py \
  --input ./images \
  --output predictions.json \
  --checkpoint runs/robust_dino_fusion/best.pt \
  --on-error skip
```

Optional robustness diagnostics remain separate from competition JSON:

```bash
python predict.py \
  --input ./images \
  --output predictions.json \
  --checkpoint runs/robust_dino_fusion/best.pt \
  --diagnostics-output robustness.json
```

Plain or older checkpoints can use explicit `--model-factory`,
`--model-kwargs`, `--preprocess-factory`, and `--preprocess-kwargs` overrides.
State restoration remains strict.

## Interactive demo

`app/studio.py` ("Spectral Evidence") is the judging demo. It shows each image
beside the log-FFT magnitude the forensic branch actually consumes, and walks the
official degradation ladder so robustness is watched rather than read off a table:

```bash
streamlit run app/studio.py -- \
  --checkpoint runs/robust_dino_fusion/best.pt \
  --device auto
```

`app/streamlit_app.py` is a smaller panel over the same inference path:

```bash
streamlit run app/streamlit_app.py -- \
  --checkpoint runs/robust_dino_fusion/best.pt \
  --device auto
```

Both decode uploads through `src.data.load_image` and score through
`load_artifact` / `Predictor`, so neither can disagree with `predict.py`. See
[`app/README.md`](app/README.md) for the full description.

The demo shows the clean probability, thresholded label, transformed scores,
mean score drift, and class stability. Stability only means the answer changed
little after transformations; it does not prove that the answer is correct.

## Repository map

```text
configs/            experiment and data configurations
src/data/           manifests, leakage-safe splits, transforms, loaders
src/models/         backbone adapters, classifier, forensic branch, fusion
src/training/       losses, trainer, validation, checkpoints, resume
src/evaluation/     robustness metrics, error analysis, reporting
src/inference/      artifact loading, batch prediction, product reports
app/                Streamlit demos: studio.py (judging), streamlit_app.py
scripts/            dataset download/export and GPU readiness checks
notebooks/          guided Colab GPU run
results/            committed evaluation reports for each published run
train.py            training entry point
evaluate.py         held-out robustness evaluation
predict.py          required competition directory inference
ERROR_ANALYSIS.md   false positives, false negatives, and trade-offs
tests/              download-free unit and integration coverage
```

## Limitations

- Performance depends on the actual training datasets and generator coverage;
  no universal AI-image detector is implied.
- Unseen generators, heavy editing, composites, screenshots, adversarial
  processing, and domains unlike training data may fail.
- Dataset/source imbalance can create shortcuts despite leakage-safe splits;
  run and publish the shortcut audit.
- The robustness gains are measured on the same corruption families the model
  was trained against, at severities inside the training sampling ranges.
  Nothing here shows transfer to a corruption type the model has never seen.
- Paired augmentation, the consistency loss and the forensic branch were
  enabled in one run, so their individual contributions are unknown.
- Robustness was bought partly with specificity under heavy degradation, which
  raises the false-positive rate on real images in exactly the conditions where
  a moderation system would be least tolerant of it.
- A stable prediction can still be consistently wrong.
- Probability calibration may drift across domains and should be checked before
  interpreting a score as real-world confidence.
- The FFT branch depends on `torch.fft`; use CPU or CUDA if a PyTorch MPS build
  lacks the required complex operations.
- This is a hackathon proof of concept, not a moderation or authenticity verdict
  system.

## Team contributions

- Melvin: data pipeline, datasets, transforms, split integrity
- Mateo: encoders, classifier, forensic branch, fusion, parameter accounting
- Trina: losses, training loop, validation, checkpoints, early stopping
- Jamie: robustness evaluation, metrics, drift, errors, runtime
- Ryan: inference CLI, checkpoint reconstruction, demo, repository UX

## License

Source code is released under the MIT License; see [`LICENSE`](LICENSE). The
licence covers this repository's code only. Pretrained weights and datasets
(CIFAKE, WildFake, SID_Set, COCO val2017, DALL-E Advanced) remain under their
own terms and must be checked before redistribution.

See `SUBMISSION_CHECKLIST.md` before publishing or recording the demo.
