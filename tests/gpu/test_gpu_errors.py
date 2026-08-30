from __future__ import annotations

import pytest

from src.gpu import GpuCheckError, gpu_error_context, is_out_of_memory


def test_context_is_rendered_into_the_message():
    error = GpuCheckError("kernel exploded", {"backbone": "clip", "batch_size": 32})
    assert "kernel exploded" in str(error)
    assert "backbone=clip" in str(error)
    assert "batch_size=32" in str(error)
    assert error.context["backbone"] == "clip"


def test_errors_are_re_raised_with_context_and_the_cause_preserved():
    original = RuntimeError("CUDA error: device-side assert triggered")
    with pytest.raises(GpuCheckError) as info:
        with gpu_error_context(backbone="ijepa", batch_size=64):
            raise original
    assert info.value.__cause__ is original
    assert "backbone=ijepa" in str(info.value)
    assert "device-side assert" in str(info.value)


def test_nested_contexts_keep_the_innermost_detail_and_gain_the_outer():
    with pytest.raises(GpuCheckError) as info:
        with gpu_error_context(backbone="dinov2"):
            with gpu_error_context(batch_size=8):
                raise ValueError("boom")
    assert info.value.context == {"backbone": "dinov2", "batch_size": 8}


def test_nothing_is_swallowed_when_the_block_succeeds():
    with gpu_error_context(backbone="clip"):
        value = 1 + 1
    assert value == 2


@pytest.mark.parametrize(
    "message,expected",
    [
        ("CUDA out of memory. Tried to allocate 2.00 GiB", True),
        ("DefaultCPUAllocator: can't allocate memory: you tried", True),
        ("CUDA error: device-side assert triggered", False),
        ("shape mismatch", False),
    ],
)
def test_out_of_memory_is_recognised_by_message(message, expected):
    assert is_out_of_memory(RuntimeError(message)) is expected
