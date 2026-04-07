"""
torch.profiler wrapper with NVTX annotations.

Wraps torch.profiler.profile for training loop profiling.

Annotates forward pass, backward pass, and data loading separately
using torch.cuda.nvtx ranges -- these appear as named regions in
Nsight Systems timeline view.

Usage:
    profiler = ProfilerWrapper(output_dir='profiler_output/')
    with profiler.profile():
        for batch in dataloader:
            with torch.cuda.nvtx.range('DataLoader'):
                x, y = batch
            with torch.cuda.nvtx.range('Forward'):
                out = model(x)
            with torch.cuda.nvtx.range('Backward'):
                loss.backward()

Why NVTX annotations:
    The profiler trace shows CPU and GPU timelines as colored bars.
    Without NVTX, all operations appear as raw CUDA kernel names.
    With NVTX, the trace groups kernels under your labels (DataLoader,
    Forward, Backward) so you can immediately see which phase is slow.
"""

import os
from contextlib import contextmanager
from typing import Optional

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)


class ProfilerWrapper:
    """
    Wraps torch.profiler.profile for training loop profiling.

    Provides:
    - Automatic NVTX range annotations for forward, backward, data loading
    - Chrome trace export for visualization
    - TensorBoard trace export for interactive analysis
    - Memory profiling (GPU memory allocation tracking)

    Args:
        output_dir: directory to save profiler traces
        wait_steps: number of warmup steps before profiling starts
        warmup_steps: number of steps to warm up the profiler
        active_steps: number of steps to actively profile
        repeat: number of profiling cycles (0 = profile continuously)
        record_shapes: whether to record tensor shapes in traces
        profile_memory: whether to profile GPU memory allocation
        with_stack: whether to record Python call stacks
    """

    def __init__(
        self,
        output_dir: str = "profiler_output/",
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 3,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ):
        self.output_dir = output_dir
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        os.makedirs(output_dir, exist_ok=True)

    @contextmanager
    def profile_context(self):
        """
        Context manager for profiling a training loop.

        Yields a torch.profiler.profile object that can be stepped
        after each training iteration.

        Usage:
            with profiler_wrapper.profile_context() as prof:
                for batch in dataloader:
                    # ... training step ...
                    prof.step()
        """
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        prof_schedule = schedule(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat,
        )

        with profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=tensorboard_trace_handler(self.output_dir),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        ) as prof:
            yield prof

    def export_chrome_trace(
        self, prof: torch.profiler.profile, filename: str = "trace.json"
    ):
        """
        Export profiler results as a Chrome trace JSON file.

        Open in chrome://tracing or Perfetto UI for interactive analysis.

        Args:
            prof: the profiler object after profiling is complete
            filename: output filename (saved in output_dir)
        """
        trace_path = os.path.join(self.output_dir, filename)
        prof.export_chrome_trace(trace_path)
        return trace_path

    def print_summary(
        self,
        prof: torch.profiler.profile,
        sort_by: str = "cpu_time_total",
        row_limit: int = 20,
    ):
        """
        Print a summary table of profiled operations.

        Args:
            prof: the profiler object after profiling is complete
            sort_by: column to sort by (cpu_time_total, cuda_time_total, etc.)
            row_limit: maximum number of rows to display
        """
        print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))


def get_memory_stats(device: Optional[str] = None) -> dict:
    """
    Get current GPU memory statistics.

    Returns a dictionary with memory usage broken down into:
    - allocated: memory currently occupied by tensors
    - reserved: memory reserved by the caching allocator (includes fragmentation)
    - max_allocated: peak memory allocated since last reset

    Args:
        device: CUDA device string (e.g., 'cuda:0'). If None, uses current device.

    Returns:
        dict with memory stats in MB, or empty dict if CUDA is unavailable
    """
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
        }

    if device is None:
        device = torch.cuda.current_device()

    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
    }
