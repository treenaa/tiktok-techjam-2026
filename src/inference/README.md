# Inference subsystem

`load_artifact()` strictly reconstructs a model and preprocessing callable from
the metadata saved by `train.py`. Plain state-dict checkpoints require explicit
factory overrides. No missing or unexpected state keys are accepted.

`Predictor.predict_paths()` verifies every file through `src.data.load_image`
before inference, preserves input order, calls `model.eval()` with
`torch.inference_mode()`, and applies sigmoid exactly once. The supported error
policies are:

- `raise`: abort before writing output if any image is unreadable;
- `skip`: score readable files and return every unreadable path/reason.

Placeholder images are deliberately unsupported because they would create a
fabricated prediction for a file the system never decoded.

`Predictor.diagnose_image()` uses the official deterministic transform registry
and returns a separate rich object. Competition JSON is validated independently
and permits exactly `image_path` and numeric `pred` fields.
