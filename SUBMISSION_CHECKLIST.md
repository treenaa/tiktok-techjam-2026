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
- [ ] Choose and add the team's repository license
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
- [ ] Verify real SID_Set/CIFAKE/WildFake layouts against their adapters
- [ ] Train CLIP, DINOv2, and I-JEPA under comparable configurations
- [ ] Select the backbone using validation evidence, not test performance
- [ ] Run the final test benchmark once with the selected checkpoint

## Results and error analysis

- [ ] Populate clean plus all-transformation metrics from actual runs
- [ ] Report mean/worst transformed AUROC and degradation
- [ ] Report mean/median/p95 score drift and class-flip rate
- [ ] Report cross-dataset and cross-generator slices with support counts
- [ ] Review representative false positives and false negatives manually
- [ ] Measure parameter count, checkpoint size, latency, and throughput on the demo device
- [ ] Add one compact robustness/ablation table or figure to the submission
- [ ] Do not include synthetic-fixture metrics or fabricated numbers

## Product and judging demo

- [x] Competition JSON contains `image_path` and `pred` only
- [x] Interactive upload demo and robustness panel implemented
- [x] Demo warns that stability is not correctness
- [ ] Place the final `best.pt` where the demo/inference commands expect it
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
