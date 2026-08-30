"""Failure types that refuse to lose the context a GPU error happened in.

A bare ``CUDA error: out of memory`` traceback is close to useless when a sweep
touches three backbones, two architectures, four batch sizes and two
precisions. Every failure raised from this package is re-raised as a
:class:`GpuCheckError` whose message names the exact configuration, with the
original exception chained so nothing is swallowed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional


class GpuCheckError(RuntimeError):
    """Raised when a GPU check fails, annotated with the failing config."""

    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None) -> None:
        self.context: Dict[str, Any] = dict(context or {})
        self.raw_message = str(message)
        super().__init__(format_context(self.raw_message, self.context))


def format_context(message: str, context: Mapping[str, Any]) -> str:
    """``"message [backbone=clip architecture=fusion batch_size=32]"``."""
    if not context:
        return str(message)
    rendered = " ".join("%s=%s" % (key, context[key]) for key in sorted(context))
    return "%s [%s]" % (message, rendered)


def is_out_of_memory(exc: BaseException) -> bool:
    """Recognise CUDA/CPU allocator exhaustion across PyTorch versions.

    ``torch.cuda.OutOfMemoryError`` only exists on newer builds, and the CPU
    allocator raises a plain ``RuntimeError``, so the message is checked too.
    """
    try:
        import torch

        cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
        if cuda_oom is not None and isinstance(exc, cuda_oom):
            return True
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass
    text = str(exc).lower()
    return "out of memory" in text or "can't allocate memory" in text


@contextmanager
def gpu_error_context(**context: Any) -> Iterator[None]:
    """Re-raise anything from the block as an annotated :class:`GpuCheckError`.

    This never swallows an error: the original exception stays chained as
    ``__cause__`` so the full traceback survives.
    """
    try:
        yield
    except GpuCheckError as exc:
        # Preserve the innermost context; add outer keys it does not already have.
        merged = dict(context)
        merged.update(exc.context)
        if merged != exc.context:
            raise GpuCheckError(exc.raw_message, merged) from exc.__cause__ or exc
        raise
    except BaseException as exc:  # noqa: BLE001 - re-raised immediately, never dropped
        message = str(exc) or type(exc).__name__
        raise GpuCheckError("%s: %s" % (type(exc).__name__, message), context) from exc


__all__ = ["GpuCheckError", "format_context", "is_out_of_memory", "gpu_error_context"]
