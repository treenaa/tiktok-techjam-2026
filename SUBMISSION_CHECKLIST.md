# Submission checklist

Items requiring real experiments, credentials, licensing decisions, or video
recording are intentionally not marked complete by code generation.

## Repository and reproducibility

- [x] Structured data/model/training/evaluation/inference modules
- [x] Directory-to-competition-JSON inference script
- [x] Deterministic input ordering
- [x] Explicit corrupt-file behavior with separate error reporting
- [x] CPU fallback and download-free tests
- [x] Setup, training, evaluation, inference, limitations, and contribution docs
- [x] Choose and add the team's repository license (MIT, code only)
- [ ] Create the public GitHub repository and verify it from a clean clone
- [ ] Pin a tested lockfile or exact environment after the final training run
- [ ] Confirm every redistributed dataset/model asset permits publication

## Scientific validity

- [x] Canonical labels: real `0`, AIGC `1`
- [x] Model returns logits; sigmoid applied once outside `forward`
- [x] Source-level split and derivative-leakage validation
- [x] Protected demonstration data blocked from train/validation/test
- [x] Validation-only threshold tuning, early stopping, and checkpoint selection
- [x] Continuous probabilities used for AUROC
- [x] Deterministic official evaluation transformations
- [ ] Run and publish the dataset shortcut audit on the real training manifest
- [x] Verify real SID_Set/CIFAKE/WildFake layouts against their adapters
- [ ] Train CLIP, DINOv2, and I-JEPA under comparable configurations (only DINOv2 done)
- [x] Train at least one robustness-aware run (paired augmentation + consistency + forensic branch) — `robust_dino_fusion`, reported against the CIFAKE baseline it shares a split, seed and schedule with
- [ ] Ablate the three robustness interventions; they were enabled in one run, so no individual contribution is claimed
- [ ] Evaluate outside the training augmentation ranges, so robustness is shown to transfer rather than only to hold on the trained corruption families
- [ ] Train a combined multi-dataset model and measure cross-dataset generalisation
- [ ] Populate generator metadata for the CIFAKE and SID_Set runs so cross-generator slices are non-empty
- [ ] Select the backbone using validation evidence, not test performance
- [ ] Run the final test benchmark once with the selected checkpoint

## Results and error analysis

- [x] Populate clean plus all-transformation metrics from actual runs (three DINOv2 baselines plus `robust_dino_fusion`)
- [x] Report mean/worst transformed AUROC and degradation
- [ ] Report mean/median/p95 score drift and class-flip rate — mean and p95 drift and flip rate are in `robustness.csv`; median drift is not computed
- [ ] Report cross-dataset and cross-generator slices with support counts
- [x] Write up representative false positives and false negatives — see [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md)
- [ ] Open the representative errors and inspect them visually for a common cause; they are currently listed by score, not reviewed
- [ ] Measure parameter count, checkpoint size, latency, and throughput on the demo device — parameter count, throughput and p95 batch latency are recorded per run, but on the training GPU rather than the demo machine
- [x] Add one compact robustness/ablation table or figure to the submission — before/after table in the README Results section
- [x] Do not include synthetic-fixture metrics or fabricated numbers

## Product and judging demo

- [x] Competition JSON contains `image_path` and `pred` only
- [x] Interactive upload demo and robustness panel implemented
- [x] Demo warns that stability is not correctness
- [ ] Place the final `best.pt` where the demo/inference commands expect it
- [x] Publish the baseline `best.pt` files as GitHub Release assets and confirm the README download command works from a clean clone — verified for `baseline_dino_wildfake_v3`: downloaded from the v0.1-baselines release into a fresh clone, SHA-256 matched, and `load_artifact` reconstructed the model with no factory overrides
- [ ] Publish `robust_dino_fusion-best.pt` as a Release asset and add its SHA-256 to the README; the run's results are committed but its weights are not downloadable
- [x] Test JPG, JPEG, PNG, WebP, corrupt input, empty directory, and nested directory UX
- [ ] Rehearse the demo offline with pretrained weights already cached
- [ ] Record the end-to-end demo video
- [ ] Show at least one clean/transformed comparison in the video
- [ ] Add final screenshots/video link to README and Devpost

## Devpost narrative

- [ ] Explain the redistribution-versus-generation signal problem
- [ ] Describe the selected backbone using measured evidence
- [ ] Explain paired transformations, consistency learning, and forensic fusion
- [ ] List tools, frameworks, datasets, pretrained models, and licensed assets
- [ ] State limitations and likely failure cases clearly
- [ ] Verify every numerical claim against saved result artifacts
- [ ] Credit all team members and external assets
