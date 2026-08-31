# Error analysis

Competition deliverable 5. Every number and every example here is read from the
committed evaluation artifacts — nothing is recomputed by hand or estimated.

Source: `results/baseline_dino_real/report.json` and
`results/robust_dino_fusion/report.json`, both on the CIFAKE test split
(n = 18,000; 9,000 real, 9,000 AIGC), decided at each run's validation-selected
threshold (0.4668375 baseline, 0.6042040 robustness-aware).

Regenerate the underlying data with:

```bash
python evaluate.py \
  --manifest manifests/test.csv \
  --model-factory src.models:create_model \
  --model-kwargs '{"backbone":"dinov2","architecture":"fusion"}' \
  --preprocess-factory src.models:create_preprocess \
  --preprocess-kwargs '{"backbone":"dinov2"}' \
  --checkpoint runs/robust_dino_fusion/best.pt \
  --threshold 0.6042042 \
  --output-dir results/robust_dino_fusion \
  --save-predictions
```

A note on the paths below: CIFAKE ships its own `train/` and `test/` folders,
but the project pools them and re-splits 70/15/15 grouped on `source_id`
(`src/data/adapters.py`, `cifake_adapter`). An image under `train/` in these
listings is therefore a legitimate member of *our* test split, not leakage.

---

## 1. The dominant failure mode is asymmetric

The baseline does not degrade gracefully under heavy corruption. It collapses
in one direction: it stops predicting `AIGC` at all.

| Transform | Model | TP | FN | FP | TN | Recall | Specificity |
|---|---|---|---|---|---|---|---|
| `clean` | baseline | 8550 | 450 | 577 | 8423 | 0.950 | 0.936 |
| `clean` | robust | 8637 | 363 | 398 | 8602 | 0.960 | 0.956 |
| `jpeg_30` | baseline | 6783 | 2217 | 290 | 8710 | 0.754 | 0.968 |
| `jpeg_30` | robust | 7531 | 1469 | 334 | 8666 | 0.837 | 0.963 |
| `blur_2.0` | baseline | 2089 | 6911 | 220 | 8780 | 0.232 | 0.976 |
| `blur_2.0` | robust | 6512 | 2488 | 947 | 8053 | 0.724 | 0.895 |
| `resize_0.25` | baseline | 1240 | 7760 | 143 | 8857 | 0.138 | 0.984 |
| `resize_0.25` | robust | 5733 | 3267 | 843 | 8157 | 0.637 | 0.906 |
| `noise_0.10` | baseline | 1401 | 7599 | 5 | 8995 | 0.156 | 0.999 |
| `noise_0.10` | robust | 7647 | 1353 | 911 | 8089 | 0.850 | 0.899 |

The `noise_0.10` baseline row is the clearest case. **Five** false positives out
of 9,000 real images looks like a near-perfect specificity of 0.999. It is not a
strength — the model has stopped flagging anything, and 7,599 of 9,000 AI images
pass through undetected. A specificity that rises while recall collapses is a
symptom of degeneracy, not of caution.

This is why the robustness benchmark reports recall and specificity separately
rather than accuracy alone. At `noise_0.10` the baseline's accuracy is 0.578 —
which reads as "somewhat weak" and hides a detector that has effectively
switched off.

## 2. The trade-off the robustness-aware run makes

Under heavy corruption the robustness-aware model buys recall with specificity.
Stated in absolute counts, at `noise_0.10`:

- false negatives: 7,599 → 1,353 (**6,246 fewer** missed AI images)
- false positives: 5 → 911 (**906 more** real images wrongly flagged)

At `resize_0.25`, 4,493 fewer false negatives for 700 more false positives.

Whether that trade is correct depends on the deployment. For a platform triage
queue where a flag means "route to review", it is clearly favourable — roughly
seven missed detections recovered per new false alarm. For an automated action
that penalises a creator with no human in the loop, an extra 906 wrongly flagged
authentic images per 9,000 is not acceptable, and the threshold should be moved
up from the validation-selected value.

Two things this trade-off is **not**:

- It is not a clean-accuracy sacrifice. At `clean` the robustness-aware model is
  strictly better on both error types (FN 450 → 363, FP 577 → 398). The
  trade-off appears only under degradation.
- It is not evidence that the baseline was "more conservative". The baseline's
  low false-positive count under corruption is a side effect of it predicting
  `real` almost everywhere.

## 3. Representative false positives — real images called AI-generated

Highest-confidence errors on clean inputs, robustness-aware model:

| P(AIGC) | Image |
|---|---|
| 0.9992 | `test/REAL/0077.jpg` |
| 0.9982 | `train/REAL/2098 (3).jpg` |
| 0.9955 | `train/REAL/0704 (2).jpg` |

Baseline, clean:

| P(AIGC) | Image |
|---|---|
| 0.9974 | `test/REAL/0478 (3).jpg` |
| 0.9941 | `train/REAL/3769 (2).jpg` |
| 0.9937 | `test/REAL/0077.jpg` |

`test/REAL/0077.jpg` is a confident false positive in **both** models — 0.9937
for the baseline and 0.9992 after robustness training. An error that survives a
substantially different training recipe is more likely to be a property of the
image (or its label) than of the model, and is the first thing we would inspect
manually given more time.

These errors are also the expensive ones. At a 0.999 score, no downstream
confidence threshold saves a wrongly accused authentic image.

## 4. Representative false negatives — AI images called real

Robustness-aware model, clean:

| P(AIGC) | Image |
|---|---|
| 0.0090 | `train/FAKE/4151 (9).jpg` |
| 0.0106 | `train/FAKE/4337 (5).jpg` |
| 0.0165 | `train/FAKE/1011.jpg` |

Baseline, clean:

| P(AIGC) | Image |
|---|---|
| 0.0013 | `train/FAKE/1373 (3).jpg` |
| 0.0018 | `train/FAKE/5019 (10).jpg` |
| 0.0026 | `train/FAKE/4337 (5).jpg` |

`train/FAKE/4337 (5).jpg` is missed by both models, and
`train/FAKE/1373 (3).jpg` is missed by the baseline both clean and at
`resize_0.25` (where its score reaches 0.0000). A small set of generated images
appears to carry no signal either model detects.

One encouraging detail: the robustness-aware model's false negatives are less
confident than the baseline's (0.0090 vs 0.0013 at the top of the list). It is
wrong less emphatically, which matters if a score is ever surfaced to a human
reviewer rather than thresholded.

## 5. Known limits of this analysis

- **Single dataset.** All of the above is CIFAKE, which is 32×32 imagery
  upscaled to 224. Its errors under blur and downscaling are partly a property
  of that resolution and should not be assumed to transfer.
- **No generator breakdown.** `generator_nonempty_aigc` is 0 for both CIFAKE
  runs, so we cannot say which generators are missed. Only the WildFake run
  carries generator metadata, and there every AIGC image is labelled
  `diffusion`, so its per-generator slice is uninformative too.
- **Errors are listed, not inspected.** These are the highest-confidence
  errors by score. Nobody has yet opened them to look for a common visual cause,
  which is the obvious next step.
- **No human-label audit.** A confident false positive shared across two models
  may indicate a mislabelled image; we have not verified any labels.

## 6. What we would do next

1. Open the shared errors — `test/REAL/0077.jpg` and `train/FAKE/4337 (5).jpg`
   first — and check for mislabelling before treating them as model failures.
2. Populate generator metadata for the CIFAKE runs so false negatives can be
   attributed to a generator family.
3. Sweep the decision threshold and publish the precision/recall curve under
   degradation, so the trade-off in §2 becomes a choice a deployer makes rather
   than one baked into the checkpoint.
4. Repeat on a higher-resolution dataset to separate CIFAKE's resolution
   artifacts from genuine detector weaknesses.
