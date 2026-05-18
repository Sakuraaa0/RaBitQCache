"""
CUDA Events profiling utilities for RabitQ.

Usage:
    from vllm.v1.attention.backends.rabitq_profiling import CUDATimer

    with CUDATimer("my_operation"):
        # your GPU code here
        result = some_gpu_operation()
"""
import torch
from contextlib import contextmanager
from collections import defaultdict
import time

class CUDATimer:
    """Context manager for accurate CUDA timing."""

    # Class-level storage for aggregated timings
    timings = defaultdict(list)
    enabled = True  # Set to False to disable profiling

    def __init__(self, name: str, sync_before=True, sync_after=True):
        """
        Args:
            name: Name of the operation being timed
            sync_before: Whether to synchronize before starting timer
            sync_after: Whether to synchronize after stopping timer
        """
        self.name = name
        self.sync_before = sync_before
        self.sync_after = sync_after
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if not self.enabled:
            return self

        if self.sync_before:
            torch.cuda.synchronize()

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        self.end_event.record()

        if self.sync_after:
            torch.cuda.synchronize()

        elapsed_ms = self.start_event.elapsed_time(self.end_event)
        self.timings[self.name].append(elapsed_ms)

    @classmethod
    def print_stats(cls):
        """Print aggregated timing statistics."""
        if not cls.timings:
            print("No timing data collected")
            return

        print("\n" + "=" * 80)
        print("CUDA Timing Statistics (milliseconds)")
        print("=" * 80)
        print(f"{'Operation':<40} {'Count':>8} {'Total':>12} {'Mean':>12} {'Min':>12} {'Max':>12}")
        print("-" * 80)

        # Sort by total time descending
        sorted_timings = sorted(cls.timings.items(),
                               key=lambda x: sum(x[1]),
                               reverse=True)

        for name, times in sorted_timings:
            count = len(times)
            total = sum(times)
            mean = total / count
            min_time = min(times)
            max_time = max(times)

            print(f"{name:<40} {count:>8} {total:>12.3f} {mean:>12.3f} {min_time:>12.3f} {max_time:>12.3f}")

        print("=" * 80 + "\n")

    @classmethod
    def reset(cls):
        """Clear all timing data."""
        cls.timings.clear()

    @classmethod
    def enable(cls):
        """Enable profiling."""
        cls.enabled = True

    @classmethod
    def disable(cls):
        """Disable profiling (zero overhead)."""
        cls.enabled = False


# Convenience function for one-off timing
@contextmanager
def time_cuda(name: str, print_immediately=False):
    """
    Time a CUDA operation and optionally print immediately.

    Example:
        with time_cuda("mask_building", print_immediately=True):
            mask = build_mask(...)
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    yield

    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    if print_immediately:
        print(f"[CUDA Timer] {name}: {elapsed:.3f} ms")

    CUDATimer.timings[name].append(elapsed)
