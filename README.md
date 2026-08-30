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

## Status

The data, model, training, evaluation, inference, and demo code paths are
implemented and covered by download-free tests.

Three phase-one baselines have been trained and evaluated on real data. Every
number below comes from a real run recorded under `results/`; none is inferred
from synthetic fixtures.

What has **not** been run yet, and is therefore not claimed anywhere:

- CLIP and I-JEPA baselines (`configs/baseline_clip.yaml`,
  `configs/baseline_ijepa.yaml` exist but have no results).
- Robustness-aware training. All three baselines use `augment: none`,
  `consistency_weight: 0.0`, and `forensic_branch: false`. The paired
  augmentation, consistency loss, and forensic branch are implemented but do
  not contribute to any published number.
- A combined multi-dataset model, and cross-dataset (train on A, test on B)
  generalisation.

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

Read these with the following caveats:

- The SID_Set and WildFake test sets are small (900 and 600 images). Their
  robustness looks excellent, but the support is too thin to headline.
- CIFAKE is 32x32 imagery upscaled to 224. Aggressive blur and downscaling
  destroy its signal: at `resize_0.25` accuracy falls to 0.561 and recall to
  0.138 while specificity stays at 0.984, i.e. the model stops calling anything
  AI-generated rather than making balanced mistakes. This is the failure mode
  robustness-aware training is meant to address, and it has not been run yet.
- Generator metadata is only populated for the WildFake run, so cross-generator
  slices are empty in the other two reports.

Full per-transform metrics, stability, subgroup slices, representative errors,
and runtime are in `results/<run>/report.json` and `results/<run>/robustness.csv`.

## Pretrained checkpoints

Checkpoints are not stored in git. Download `best.pt` for the baseline you want
from the repository's GitHub Releases page and place it under `runs/`:

```bash
mkdir -p runs/baseline_dino_real
curl -L -o runs/baseline_dino_real/best.pt   https://github.com/treenaa/tiktok-techjam-2026/releases/download/v0.1-baselines/baseline_dino_real-best.pt
```

Each file is roughly 346 MB and contains the model, optimizer, scheduler, and
scaler state plus the validation-selected threshold, so `predict.py` and
`evaluate.py` can reconstruct the model without extra flags. Available
checkpoints are `baseline_dino_real` (CIFAKE), `baseline_dino_sidset`, and
`baseline_dino_wildfake_v3`; their validation-selected thresholds are 0.4668,
0.4878, and 0.3832 respectively.

## Approach

The project is designed to test three pretrained visual representations under
the same split, head, training schedule, seed, and robustness benchmark. Only
DINOv2 has been run so far; see Status.

Candidate backbones:

- I-JEPA
- DINOv2
- CLIP

The phase-one detector is a visual encoder plus a small binary head. After the
winning backbone is selected empirically, the robustness model can add:

- paired clean/transformed training;
- prediction-consistency loss;
- a lightweight forensic branch over normalized log FFT magnitude;
- feature-level visual/forensic fusion.

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
  --checkpoint runs/final/best.pt \
  --threshold "$VALIDATION_THRESHOLD" \
  --output-dir results/final \
  --save-predictions
```

Use the validation-selected threshold stored in the best checkpoint. The report
contains AUROC, accuracy, F1, precision, recall, specificity, false-positive
rate, AUROC degradation, score drift, class flips, dataset/generator slices,
representative errors, and runtime measurements.

## Required directory inference

Training checkpoints are self-describing, so the usual command is:

```bash
python predict.py \
  --input ./images \
  --output predictions.json \
  --checkpoint runs/final/best.pt
```

Files are sorted deterministically. Supported formats are JPG/JPEG, PNG, WebP,
BMP, and TIFF. By default one unreadable file aborts before output is written.
To explicitly skip corrupt files and receive a separate error report:

```bash
python predict.py \
  --input ./images \
  --output predictions.json \
  --checkpoint runs/final/best.pt \
  --on-error skip
```

Optional robustness diagnostics remain separate from competition JSON:

```bash
python predict.py \
  --input ./images \
  --output predictions.json \
  --checkpoint runs/final/best.pt \
  --diagnostics-output robustness.json
```

Plain or older checkpoints can use explicit `--model-factory`,
`--model-kwargs`, `--preprocess-factory`, and `--preprocess-kwargs` overrides.
State restoration remains strict.

## Interactive demo

```bash
streamlit run app/streamlit_app.py -- \
  --checkpoint runs/final/best.pt \
  --device auto
```

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
app/                Streamlit demo
train.py            training entry point
evaluate.py         held-out robustness evaluation
predict.py          required competition directory inference
tests/              download-free unit and integration coverage
```

## Limitations

- Performance depends on the actual training datasets and generator coverage;
  no universal AI-image detector is implied.
- Unseen generators, heavy editing, composites, screenshots, adversarial
  processing, and domains unlike training data may fail.
- Dataset/source imbalance can create shortcuts despite leakage-safe splits;
  run and publish the shortcut audit.
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
