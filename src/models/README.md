# Model subsystem

The subsystem implements the three comparable visual baselines and the
dual-domain fusion architecture from the project handoff. Its hard contract is:

```python
logits = model(images)  # shape (B,), raw logit for label 1 = AIGC
```

There is no sigmoid in `forward()`. Training should use
`BCEWithLogitsLoss`; evaluation and inference apply sigmoid exactly once.

## Real backbones

The default registered checkpoints are:

| Name | Checkpoint | Pooling | Input statistics |
| --- | --- | --- | --- |
| `ijepa` | `facebook/ijepa_vith14_1k` | mean token | ImageNet |
| `dinov2` | `facebook/dinov2-base` | CLS token | ImageNet |
| `clip` | `openai/clip-vit-base-patch32` | model pooler | CLIP |

They are loaded lazily through Hugging Face Transformers. Importing
`src.models` and running tests never downloads weights. Install `transformers`
before constructing a real backbone; use `local_files_only=true` in an offline
environment with a populated model cache.

The fusion branch uses `torch.fft`; choose CPU or CUDA if the installed PyTorch
build does not implement complex FFT operations on its MPS backend.

```python
from src.models import create_model, create_preprocess, parameter_report

model = create_model(
    backbone="dinov2",
    architecture="visual",  # fair phase-1 baseline
    freeze_backbone=True,
)
preprocess = create_preprocess("dinov2")
print(parameter_report(model))
```

After selecting a backbone empirically:

```python
model = create_model(
    backbone="dinov2",
    architecture="fusion",
    forensic_dim=128,
)
```

The forensic branch receives the same tensor as the visual branch. It explicitly
reverses that backbone's declared normalization to RGB `[0,1]`, converts to
luminance, removes the DC component, computes a centred log FFT magnitude, and
uses a small GroupNorm CNN. This prevents CLIP-versus-ImageNet normalization
from becoming an accidental forensic signal.

## Fair backbone comparison

`comparison_configs()` emits CLIP, DINOv2, and I-JEPA configs with identical
head size, dropout, architecture, and freezing policy. Data splits, optimizer,
epochs, augmentations, and seeds belong to the data/training configuration and
must also remain identical.

## Evaluation CLI integration

```bash
python evaluate.py \
  --manifest manifests/test.csv \
  --model-factory src.models:create_model \
  --model-kwargs '{"backbone":"dinov2","architecture":"fusion"}' \
  --preprocess-factory src.models:create_preprocess \
  --preprocess-kwargs '{"backbone":"dinov2"}' \
  --checkpoint checkpoints/best.pt \
  --threshold 0.5 \
  --output-dir results/final
```

The threshold in that command must be selected on validation. Checkpoints must
restore strictly unless a mismatch is being deliberately investigated.
