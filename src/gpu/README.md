# GPU validation subsystem

Run this before anyone commits the GPU to a full training run. It answers five
questions and writes the answers down:

1. **Is the environment right?** CUDA availability, driver version, CUDA
   toolkit, GPU model and VRAM, and whether PyTorch is a CUDA build whose
   runtime the installed driver actually supports.
2. **Does the pipeline really run on the GPU?** A tiny forward/backward through
   every backbone x architecture, asserting that weights, buffers, logits and
   loss are all on the device — no silent CPU fallback — and that mixed
   precision does not quietly produce NaNs.
3. **What does it cost?** Peak and steady VRAM, images/sec for a training step
   and for inference, time to first batch, and total wall clock.
4. **Does it reproduce?** The same seed twice on the same GPU, compared within
   tolerance, with the cuDNN/cuBLAS/TF32 caveats printed rather than assumed
   away.
5. **What should we run instead?** Every configuration that will not fit the
   VRAM budget comes back with measured fallbacks.

The subsystem owns no data, model, or training code. It calls
`src.models.create_model`, `src.training.build_optimizer`,
`src.training.RobustBinaryObjective` and `src.data.seed_everything` so the
checks exercise the real code paths, and it adds nothing to them.

```bash
python scripts/gpu_check.py     --config configs/gpu_check_8gb.yaml
python scripts/benchmark_gpu.py --config configs/gpu_check_8gb.yaml
```

Exit codes match `src.data.audit_cli`: `0` clean, `1` a blocking problem, `2`
bad usage. `--strict` promotes warnings (an OOM-prone config, a loose
determinism knob) to failures, so this drops into CI unchanged.

## Configs

| Config | For | Notes |
| --- | --- | --- |
| `configs/gpu_check_8gb.yaml` | the team's own 8 GB card | pins `vram_budget_mb: 8172`, sweeps batches past the ceiling |
| `configs/gpu_check.yaml` | any instance | budget defaults to whatever the device reports |
| `configs/gpu_check_smoke.yaml` | a laptop, CI, no GPU | stand-in backbones, no downloads, runs in seconds |

Nothing is hard-coded: batch sizes, backbones, architectures, precisions,
tolerances and budgets all come from the config, and every environment fact is
detected rather than assumed.

```bash
# Fast plumbing check with no GPU and no downloads.
python scripts/gpu_check.py --config configs/gpu_check_smoke.yaml --allow-cpu

# Find the largest batch this card holds for one backbone.
python scripts/benchmark_gpu.py --config configs/gpu_check_8gb.yaml \
    --backbones dinov2 --batch-sizes 8 16 32 64 --modes train

# Deterministic run: sets CUBLAS_WORKSPACE_CONFIG before torch is imported.
python scripts/gpu_check.py --config configs/gpu_check_8gb.yaml --deterministic --strict
```

## The 8 GB budget

`budget.vram_budget_mb` is the ceiling every configuration is judged against.
An explicit budget wins over a larger measuring device, so a sweep run
somewhere else cannot bless a config the real card will not run; a *smaller*
device still wins, because you cannot use VRAM that is not there.

Configurations over the budget — measured OOM or merely over
`vram_headroom_fraction` — are reported with fallbacks derived from the same
sweep, not from guesswork:

- the largest batch size that was **measured** to fit;
- the gradient-accumulation factor that preserves
  `budget.target_train_batch_size` at that smaller physical batch;
- whether AMP fp16 was measured to fit where fp32 did not;
- a smaller `image_size`, when one is configured above 224;
- gradient checkpointing and keeping `freeze_backbone: true`.

Gradient checkpointing needs a hook in `src/models` (a Hugging Face backbone
exposes `gradient_checkpointing_enable()`). This subsystem deliberately does
not add it — it is proposed in the report for the model owner to decide.

I-JEPA ViT-H/14 is ~630M parameters and is the first backbone expected to
overflow 8 GB. Read the fallback block before deciding which backbones the
phase-1 comparison can afford.

## Backbone source

`model.backbone_source: stub` injects `StubVisionBackbone`, a small
patch-embedding transformer that satisfies the same interface
`src.models.VisionEncoder` requires (`last_hidden_state`, `pooler_output`,
`config.hidden_size`), so all three pooling modes — `mean` for I-JEPA, `cls`
for DINOv2, `pooler` for CLIP — are genuinely exercised with no downloads.

**Stub numbers are not backbone numbers.** The plumbing, device placement and
precision behaviour are real; throughput and VRAM are not. Every report using
the stub says so in its notes. Use `backbone_source: pretrained` on the GPU for
figures anyone should size a run against.

## Reproducibility, honestly

Each variant is run twice from the same seed in one process and compared
against `determinism.logit_tolerance` / `loss_tolerance`. Model initialisation
is compared too, so a seeding gap shows up separately from kernel noise.

Bitwise equality is not claimed. The report always carries `CUDNN_CAVEATS`, and
`determinism.controls` records which knobs were actually active — cuDNN
benchmark mode, cuDNN determinism, TF32, `CUBLAS_WORKSPACE_CONFIG`,
`torch.use_deterministic_algorithms`. `--deterministic` tightens them and sets
`CUBLAS_WORKSPACE_CONFIG` *before* torch is imported, which is the only point
where that variable has any effect.

## Failure handling

No CUDA error is silently caught. Everything is re-raised as a `GpuCheckError`
naming the backbone, architecture, batch size, precision and device, with the
original exception chained.

The single exception is allocator exhaustion during the benchmark sweep, which
is recorded as a `status: "oom"` row with the full message — finding the
ceiling is the sweep's job, and one OOM must not abort the other measurements.
Any non-OOM CUDA error still aborts the run.

## Report

`write_report` emits `<basename>.json` and `<basename>.txt` (default
`reports/gpu/`), written atomically. The JSON is versioned
(`GPU_REPORT_SCHEMA_VERSION`) and contains `environment`, `config`,
`parameters` (per-variant counts against the 2B limit), `benchmarks`,
`recommendations`, `determinism`, `checks` and an overall `status`.

Another agent can gate on it:

```python
from src.gpu import read_report

report = read_report("reports/gpu/gpu_report.json")
assert report["status"] != "fail", "do not start training on this box"
```

## Tests

```bash
python -m pytest tests/gpu -q
```

Runs in seconds with no GPU and no downloads. The CUDA-only tests in
`tests/gpu/test_gpu_cuda.py` are **skipped**, never quietly passed, when no GPU
is present — a green suite on a laptop is not a validated GPU instance.
