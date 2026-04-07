"""
Roofline model plotter (FLOP/byte analysis).

The Roofline model is a performance analysis framework that bounds the
achievable throughput of a computation by two hardware limits:

1. **Peak compute throughput** (TFLOPS): The maximum floating-point
   operations per second the hardware can execute. This is the horizontal
   ceiling on the right side of the roofline plot.

2. **Peak memory bandwidth** (GB/s): The maximum rate at which data
   can be moved between HBM and compute units. This is the diagonal
   line on the left side of the roofline plot.

The **ridge point** is where these two limits intersect:
    ridge_point = peak_flops / peak_bandwidth  (in FLOP/byte)

Operations to the LEFT of the ridge point are **memory-bound**:
    They could compute faster but are waiting for data from HBM.
    Example: standard attention at small seq_len, elementwise ops.

Operations to the RIGHT of the ridge point are **compute-bound**:
    They have enough data but the compute units are the bottleneck.
    Example: large matrix multiplications, convolutions.

Why this matters for ViT vs ResNet:
    Standard attention is memory-bound (reading/writing the n x n matrix).
    FlashAttention moves it toward compute-bound by tiling in SRAM.
    ResNet convolutions are typically compute-bound (high arithmetic intensity).
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_roofline(
    peak_flops_tflops: float,
    peak_bandwidth_gbps: float,
    ops_list: List[Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Roofline Model",
) -> plt.Figure:
    """
    Draw the standard roofline chart with operations plotted.

    The roofline has two segments:
    1. Memory-bound region (left): throughput = bandwidth x arithmetic_intensity
    2. Compute-bound region (right): throughput = peak_flops (horizontal line)

    Each operation in ops_list is plotted as a dot based on its measured
    arithmetic intensity and throughput.

    Args:
        peak_flops_tflops: peak compute throughput in TFLOPS
        peak_bandwidth_gbps: peak memory bandwidth in GB/s
        ops_list: list of dicts, each with:
            - "name": str, operation name
            - "arithmetic_intensity": float, FLOP/byte ratio
            - "throughput": float, achieved throughput in TFLOPS
        save_path: if provided, save figure to this path
        title: plot title

    Returns:
        matplotlib Figure object

    Example:
        ops = [
            {"name": "Standard Attention", "arithmetic_intensity": 1.2,
             "throughput": 45.0},
            {"name": "FlashAttention", "arithmetic_intensity": 8.5,
             "throughput": 120.0},
            {"name": "Conv2d 3x3", "arithmetic_intensity": 20.0,
             "throughput": 140.0},
        ]
        fig = plot_roofline(312.0, 2039.0, ops, "roofline.png")
    """
    # Convert peak TFLOPS to GFLOPS for plotting (easier numbers)
    peak_flops_gflops = peak_flops_tflops * 1000
    peak_bandwidth = peak_bandwidth_gbps  # GB/s

    # Ridge point: where memory bandwidth and compute throughput meet
    ridge_point = peak_flops_gflops / peak_bandwidth  # FLOP/byte

    # Create arithmetic intensity range for plotting
    ai_min = 0.01
    ai_max = max(ridge_point * 10, 100)
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)

    # Roofline: min(peak_compute, bandwidth * arithmetic_intensity)
    roofline = np.minimum(peak_flops_gflops, peak_bandwidth * ai_range)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the roofline
    # Memory-bound region (left of ridge point) in blue
    memory_bound_mask = ai_range <= ridge_point
    compute_bound_mask = ai_range >= ridge_point

    ax.loglog(
        ai_range[memory_bound_mask],
        roofline[memory_bound_mask],
        "b-",
        linewidth=2.5,
        label="Memory Bandwidth Limit",
    )
    ax.loglog(
        ai_range[compute_bound_mask],
        roofline[compute_bound_mask],
        color="orange",
        linewidth=2.5,
        label="Compute Limit",
    )

    # Ridge point vertical line
    ax.axvline(
        x=ridge_point,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Ridge Point ({ridge_point:.1f} FLOP/byte)",
    )

    # Plot each operation
    markers = ["o", "s", "^", "D", "v", "p", "*", "h"]
    for i, op in enumerate(ops_list):
        ai = op["arithmetic_intensity"]
        tp = op["throughput"]
        name = op["name"]
        marker = markers[i % len(markers)]

        # Color based on memory-bound vs compute-bound
        color = "blue" if ai < ridge_point else "orange"

        ax.scatter(
            ai,
            tp,
            s=150,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
            label=f"{name} (AI={ai:.1f})",
        )

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Throughput (GFLOPS)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Add region labels
    ax.text(
        ai_min * 3,
        peak_flops_gflops * 0.5,
        "Memory\nBound",
        fontsize=11,
        color="blue",
        alpha=0.6,
        ha="left",
    )
    ax.text(
        ridge_point * 5,
        peak_flops_gflops * 0.5,
        "Compute\nBound",
        fontsize=11,
        color="orange",
        alpha=0.6,
        ha="left",
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
