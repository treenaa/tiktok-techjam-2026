# Demos

Two entry points share the same inference path. Both decode uploads through
`src.data.load_image` and score through `load_artifact` / `Predictor`, so neither
can disagree with `predict.py`.

## `studio.py` — Spectral Evidence

The judging demo.

```bash
streamlit run app/studio.py -- \
  --checkpoint runs/baseline_dino_wildfake_v3/best.pt \
  --device auto
```

Every view is shown as a pair: the image as a person sees it, beside the log-FFT
magnitude the detector's forensic branch actually consumes. Walking the official
degradation ladder shows the picture and its frequency signature decay together
next to the probability, so robustness is something you watch rather than read
off a table.

- **Degradation ladder** — a card per transform, ordered by family then severity,
  each carrying the transformed image, its spectrum, `P(AI-generated)`, the delta
  against clean, and the change in the frequency signature. A red border marks a
  view whose predicted class flipped. Choose the core six or the full official
  twenty in the sidebar; all twenty score in well under a second on CPU.
- **Verdict band** — the clean probability against the decision threshold, with
  the threshold's provenance shown rather than assumed.
- **Score under degradation** — one line across the ladder with the threshold
  drawn, so every class flip reads as a crossing.
- **Published results** — the measured runs in `results/*/report.json`, read at
  page load. No number on that tab is typed by hand.

The spectrum is not a decorative visual: `app/spectral.py` runs
`src.models.forensic.LogMagnitudeFFT`, the same module the fusion architecture
feeds its forensic branch, so the panel shows the model's real input.

Supporting modules: `app/theme.py` (palette, ramp, injected CSS) and
`app/spectral.py` (spectrum rendering and the paired strips). The colour ramp
that renders the spectra is also the interface's accent, so the identity is
sampled from the signal being measured.

## `streamlit_app.py` — compact demo

The original, smaller panel.

```bash
streamlit run app/streamlit_app.py -- \
  --checkpoint runs/baseline_dino_wildfake_v3/best.pt \
  --device auto
```

The upload is decoded through the same shared image loader used by the CLI,
written only to a temporary file and removed immediately. The displayed
robustness scores reuse the official deterministic transforms. The panel states
explicitly that stable predictions are not necessarily correct predictions.

## Notes

Both scripts insert the repository root on `sys.path` before importing `src.*`.
`streamlit run app/<script>.py` puts `app/` on the path rather than the
repository root, so without that bootstrap neither app starts.

Without `--checkpoint`, both read `AIGC_CHECKPOINT` and `AIGC_DEVICE` from the
environment. `studio.py` renders an onboarding panel rather than an error when no
checkpoint is set.

`.streamlit/config.toml` themes Streamlit's own chrome to match `app/theme.py`;
keep the two in step if you change the palette.
