"""Reproducibility helpers (project rule 20.9).

Seeds Python, NumPy and PyTorch from one call, and provides the DataLoader
plumbing needed for worker processes to be reproducible too.

``torch`` is optional: if it is not installed the torch parts are skipped and
the return value says so, so this module is safe to import anywhere.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

__all__ = [
    "seed_everything",
    "seed_worker",
    "make_generator",
    "dataloader_kwargs",
]


def seed_everything(
    seed: int = 0,
    deterministic_torch: bool = False,
    set_pythonhashseed: bool = True,
) -> Dict[str, Any]:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    Parameters
    ----------
    deterministic_torch:
        Also request deterministic cuDNN kernels.  Slower, and some ops have no
        deterministic implementation, so it is off by default; turn it on for
        runs whose exact numbers must be reproducible.
    set_pythonhashseed:
        Sets ``PYTHONHASHSEED`` for child processes.  It does **not** affect the
        current interpreter -- that is fixed at startup -- so rely on it only
        for subprocesses.

    Returns
    -------
    What was actually seeded, e.g. ``{"seed": 0, "python": True, "numpy": True,
    "torch": True, "cuda": False, "deterministic": False}``.
    """
    seed = int(seed)
    state: Dict[str, Any] = {"seed": seed, "python": True, "numpy": False, "torch": False}

    if set_pythonhashseed:
        os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        state["numpy"] = True
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        pass

    try:
        import torch

        torch.manual_seed(seed)
        state["torch"] = True
        state["cuda"] = bool(torch.cuda.is_available())
        if state["cuda"]:
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except AttributeError:  # pragma: no cover - CPU-only builds
                pass
            state["deterministic"] = True
        else:
            state["deterministic"] = False
    except ImportError:
        state["cuda"] = False
        state["deterministic"] = False

    return state


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for ``DataLoader``.

    Each worker derives its Python/NumPy seeds from torch's per-worker seed, so
    workers neither collide nor repeat across epochs.  Without this, NumPy-based
    augmentation can produce *identical* random draws in every worker -- a
    classic silent bug.
    """
    try:
        import torch

        worker_seed = torch.initial_seed() % (2 ** 32)
    except ImportError:  # pragma: no cover
        worker_seed = worker_id
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:  # pragma: no cover
        pass


def make_generator(seed: int = 0):
    """A seeded ``torch.Generator`` for DataLoader shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def dataloader_kwargs(seed: int = 0, num_workers: int = 0) -> Dict[str, Any]:
    """Reproducibility kwargs to splat into a ``DataLoader``::

        DataLoader(dataset, batch_size=32, shuffle=True,
                   **dataloader_kwargs(seed=0, num_workers=4))
    """
    kwargs: Dict[str, Any] = {"num_workers": num_workers}
    try:
        import torch  # noqa: F401

        kwargs["generator"] = make_generator(seed)
        if num_workers > 0:
            kwargs["worker_init_fn"] = seed_worker
    except ImportError:  # pragma: no cover
        pass
    return kwargs
