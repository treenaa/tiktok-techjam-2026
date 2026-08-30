# Evaluation subsystem

This package owns clean metrics, the 20-view robustness benchmark, score drift,
class flips, dataset/generator slices, representative errors, and runtime
measurements. It consumes the stable interfaces in `src.data` and does not tune
on the test set.

## Programmatic use

```python
from src.data import build_eval_datasets
from src.evaluation import evaluate_grid, build_report

grid = build_eval_datasets(test_records, preprocess=model_preprocess)
run = evaluate_grid(model, grid, batch_size=32, device="auto")
report = build_report(run, threshold=validation_selected_threshold)
```

The model must return one raw logit per image with shape `(B,)` or `(B, 1)`.
The evaluator applies sigmoid exactly once and records P(AIGC), where real is 0
and AIGC is 1. Two-logit heads are rejected rather than guessing which column
means AIGC.

## CLI

```bash
python evaluate.py \
  --manifest manifests/test.csv \
  --model-factory src.models.factory:create_model \
  --checkpoint checkpoints/best.pt \
  --threshold 0.5 \
  --output-dir results/final \
  --save-predictions
```

`--model-factory` is intentionally explicit because architecture construction
belongs to the model subsystem. It may receive JSON options through
`--model-kwargs`. The checkpoint must be a raw state dict or contain
`model_state_dict`/`state_dict`; use `--state-dict-key` for a different layout.

Outputs:

- `report.json`: full metrics, stability, dataset/generator protocols and errors
  for every transform, runtime semantics, transform specifications, and model
  metadata.
- `robustness.csv`: compact clean-versus-transformation table for the write-up.
- `predictions/*.jsonl`: optional per-image audit trail.

An undefined subgroup AUROC is JSON `null`, never a fabricated zero. Generator
slices use the stated `all_real_vs_generator_aigc` protocol. Stability measures
prediction consistency only; it is not evidence that a prediction is correct.
