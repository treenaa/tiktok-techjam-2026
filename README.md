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
implemented and covered by download-free tests. This repository does not ship a
trained checkpoint or claim benchmark numbers yet. Results must come from real
experiments; they must not be inferred from synthetic fixtures or fabricated.

## Approach

The project tests three pretrained visual representations under the same split,
head, training schedule, seed, and robustness benchmark:

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

See `SUBMISSION_CHECKLIST.md` before publishing or recording the demo.
