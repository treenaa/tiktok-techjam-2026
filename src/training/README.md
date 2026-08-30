# Training subsystem

The trainer supports both project phases without changing the data contract:

- `augment: none`: a true single clean view for fair CLIP/DINOv2/I-JEPA
  baselines.
- `augment: competition`: paired clean and transformed views with
  `BCE(clean) + BCE(transformed) + lambda * MSE(P(clean), P(transformed))`.

Models must return raw logits shaped `(B,)`; sigmoid is used only for the
consistency term and reported probabilities. The test split is never passed to
`Trainer` or loaded by `train.py`.

## Run

```bash
python train.py \
  --config configs/baseline_dino.yaml \
  --output-dir runs/baseline_dino \
  --device auto
```

Resume the exact run in the same directory:

```bash
python train.py \
  --config configs/baseline_dino.yaml \
  --output-dir runs/baseline_dino \
  --resume runs/baseline_dino/last.pt
```

Outputs include `best.pt`, `last.pt`, `history.json`,
`training_summary.json`, and a resolved experiment config. Checkpoints contain
`model_state_dict` for strict compatibility with `evaluate.py`, optimizer and
scheduler state, validation-selected threshold, RNG state, loader state, and
the comparison fingerprint.

## Robustness phase

Set these under `training` in an experiment config:

```yaml
augment: competition
clean_loss_weight: 1.0
augmented_loss_weight: 1.0
consistency_weight: 0.5
augment_identity_probability: 0.0
augment_operations: 1
```

The validation threshold is selected each epoch using only validation
probabilities. `threshold_metric` can be `f1`, `balanced_accuracy`, or
`accuracy`. Early stopping defaults to `val_auroc`, which is computed from
continuous probabilities. Test metrics must be produced later with
`evaluate.py` and the threshold stored in `best.pt`.

With `num_workers > 0`, the custom worker initializer reseeds the private
`RandomCompetitionTransform` stream as well as Python/NumPy. Exact mid-run
resume is strongest with `num_workers: 0`; multi-worker scheduling can consume
prefetched augmentations in a platform-dependent order even when whole runs are
reproducible from the same seed.
