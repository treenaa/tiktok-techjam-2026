# Interactive demo

Launch with a self-describing checkpoint:

```bash
streamlit run app/streamlit_app.py -- \
  --checkpoint checkpoints/best.pt \
  --device auto
```

The upload is decoded through the same shared image loader used by the CLI,
written only to a temporary file and removed immediately. The displayed
robustness scores reuse the official deterministic transforms. The panel states
explicitly that stable predictions are not necessarily correct predictions.
