#!/usr/bin/env python3
"""
Plot comparison of threading vs multiprocessing for different batch sizes.
Automatically selects the best num_workers configuration (highest throughput) for each data point.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter


def parse_summary_file(filepath):
    """
    Parse the summary.txt file and extract metrics.

    Returns:
        dict: Nested dict with structure:
              {worker_method: {batch_size: [(num_workers, samples_per_sec, final_memory_increase)]}}
    """
    data = {}

    with open(filepath, "r") as f:
        content = f.read()

    # Split by configuration blocks
    blocks = content.split("\n\n")

    for block in blocks:
        if (
            not block.strip()
            or block.startswith("Dataloader")
            or block.startswith("===")
        ):
            continue

        lines = block.strip().split("\n")

        # Parse header line: worker_method=X, num_workers=Y, batch_size=Z
        header = lines[0]
        match = re.search(
            r"worker_method=(\w+), num_workers=(\d+), batch_size=(\d+)", header
        )
        if not match:
            continue

        worker_method = match.group(1)
        num_workers = int(match.group(2))
        batch_size = int(match.group(3))

        # Extract metrics from subsequent lines
        samples_per_sec = None
        memory_increases = []

        for line in lines[1:]:
            # Extract samples per second
            if "Samples per second:" in line:
                samples_match = re.search(r"Samples per second:\s+([\d.]+)", line)
                if samples_match:
                    samples_per_sec = float(samples_match.group(1))

            # Extract all memory increase values
            if "Memory increase:" in line:
                mem_match = re.search(r"Memory increase:\s+([\d.]+)\s+MB", line)
                if mem_match:
                    memory_increases.append(float(mem_match.group(1)))

        # The last memory increase value is the final one
        final_memory_increase = memory_increases[-1] if memory_increases else None

        if samples_per_sec is not None and final_memory_increase is not None:
            # Store data
            if worker_method not in data:
                data[worker_method] = {}
            if batch_size not in data[worker_method]:
                data[worker_method][batch_size] = []

            data[worker_method][batch_size].append(
                (num_workers, samples_per_sec, final_memory_increase)
            )

    return data


def select_best_configs(data):
    """
    For each (worker_method, batch_size) pair, select the configuration
    with the highest samples per second.

    Returns:
        dict: {worker_method: {batch_size: (num_workers, samples_per_sec, final_memory_increase)}}
    """
    best_data = {}

    for worker_method, batch_data in data.items():
        best_data[worker_method] = {}

        for batch_size, configs in batch_data.items():
            # Find config with highest samples per second
            best_config = max(configs, key=lambda x: x[1])  # x[1] is samples_per_sec
            best_data[worker_method][batch_size] = best_config

    return best_data


def plot_comparison(best_data, output_dir):
    """
    Create two comparison plots: throughput and memory increase.
    """
    # Prepare data for plotting
    methods = sorted(best_data.keys())

    # Get all batch sizes (should be same for both methods)
    all_batch_sizes = set()
    for method_data in best_data.values():
        all_batch_sizes.update(method_data.keys())
    batch_sizes = sorted(all_batch_sizes)

    # Extract data for each method
    plot_data = {}
    for method in methods:
        plot_data[method] = {
            "batch_sizes": [],
            "samples_per_sec": [],
            "memory_increase": [],
            "num_workers": [],
        }

        for batch_size in batch_sizes:
            if batch_size in best_data[method]:
                num_workers, samples_per_sec, memory_increase = best_data[method][
                    batch_size
                ]
                plot_data[method]["batch_sizes"].append(batch_size)
                plot_data[method]["samples_per_sec"].append(samples_per_sec)
                plot_data[method]["memory_increase"].append(memory_increase)
                plot_data[method]["num_workers"].append(num_workers)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors and markers
    colors = {"multiprocessing": "#1f77b4", "thread": "#ff7f0e"}
    markers = {"multiprocessing": "o", "thread": "s"}

    # Plot 1: Samples per Second vs Batch Size
    for method in methods:
        data = plot_data[method]
        ax1.plot(
            data["batch_sizes"],
            data["samples_per_sec"],
            marker=markers[method],
            markersize=8,
            linewidth=2,
            label=method.capitalize(),
            color=colors[method],
        )

        # Add num_workers annotations
        for i, (bs, sps, nw) in enumerate(
            zip(data["batch_sizes"], data["samples_per_sec"], data["num_workers"])
        ):
            # if i % 2 == 0:  # Annotate every other point to avoid clutter
            ax1.annotate(
                f"w={nw}",
                xy=(bs, sps),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    ax1.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Samples per Second", fontsize=12, fontweight="bold")
    ax1.set_title(
        "DataLoader Throughput Comparison\n(Best num_workers selected for each point)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x)}"))

    # Plot 2: Memory Increase vs Batch Size
    for method in methods:
        data = plot_data[method]
        ax2.plot(
            data["batch_sizes"],
            data["memory_increase"],
            marker=markers[method],
            markersize=8,
            linewidth=2,
            label=method.capitalize(),
            color=colors[method],
        )

        # Add num_workers annotations
        for i, (bs, mem, nw) in enumerate(
            zip(data["batch_sizes"], data["memory_increase"], data["num_workers"])
        ):
            ax2.annotate(
                f"w={nw}",
                xy=(bs, mem),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    # Calculate and annotate memory ratios for each batch size
    # Find common batch sizes between both methods
    if "multiprocessing" in plot_data and "thread" in plot_data:
        mp_bs = plot_data["multiprocessing"]["batch_sizes"]
        mp_mem = plot_data["multiprocessing"]["memory_increase"]
        th_bs = plot_data["thread"]["batch_sizes"]
        th_mem = plot_data["thread"]["memory_increase"]

        # Create dictionaries for easy lookup
        mp_dict = dict(zip(mp_bs, mp_mem))
        th_dict = dict(zip(th_bs, th_mem))

        # Find common batch sizes and annotate ratios
        common_batch_sizes = set(mp_bs) & set(th_bs)

        for bs in sorted(common_batch_sizes):
            mp_memory = mp_dict[bs]
            th_memory = th_dict[bs]
            ratio = mp_memory / th_memory if th_memory > 0 else 0

            # Position annotation between the two lines
            mid_y = (mp_memory + th_memory) / 2

            ax2.annotate(
                f"{ratio:.1f}x",
                xy=(bs, mid_y),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="yellow",
                    edgecolor="orange",
                    alpha=0.7,
                ),
            )

    ax2.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Memory Increase (MB)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Memory Usage Comparison\n(Best num_workers selected for each point)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x)}"))

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / "comparison_plots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlots saved to: {output_path}")

    plt.show()

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Best Configuration for Each Batch Size")
    print("=" * 80)
    print(
        f"\n{'Method':<20} {'Batch Size':<12} {'Workers':<10} {'Throughput (sps)':<20} {'Memory (MB)':<15}"
    )
    print("-" * 80)

    for method in methods:
        data = plot_data[method]
        for bs, sps, mem, nw in zip(
            data["batch_sizes"],
            data["samples_per_sec"],
            data["memory_increase"],
            data["num_workers"],
        ):
            print(f"{method:<20} {bs:<12} {nw:<10} {sps:<20.2f} {mem:<15.2f}")
        print("-" * 80)

    # Print comparison insights
    print("\nKEY INSIGHTS:")
    print("-" * 80)

    mp_data = plot_data.get("multiprocessing", {})
    th_data = plot_data.get("thread", {})

    if mp_data and th_data:
        # Compare throughput
        avg_mp_throughput = np.mean(mp_data["samples_per_sec"])
        avg_th_throughput = np.mean(th_data["samples_per_sec"])
        throughput_diff = (
            (avg_mp_throughput - avg_th_throughput) / avg_th_throughput
        ) * 100

        print(
            f"• Average throughput (multiprocessing): {avg_mp_throughput:.2f} samples/sec"
        )
        print(f"• Average throughput (threading): {avg_th_throughput:.2f} samples/sec")
        print(
            f"• Throughput difference: {throughput_diff:+.1f}% "
            f"({'multiprocessing' if throughput_diff > 0 else 'threading'} is faster)"
        )

        # Compare memory
        avg_mp_memory = np.mean(mp_data["memory_increase"])
        avg_th_memory = np.mean(th_data["memory_increase"])
        memory_ratio = avg_mp_memory / avg_th_memory

        print(f"\n• Average memory (multiprocessing): {avg_mp_memory:.2f} MB")
        print(f"• Average memory (threading): {avg_th_memory:.2f} MB")
        print(
            f"• Memory ratio: {memory_ratio:.2f}x (multiprocessing uses {memory_ratio:.2f}x more memory)"
        )


def main():
    import sys

    if len(sys.argv) > 1:
        summary_file = sys.argv[1]
    else:
        # Default to most recent results directory
        results_dirs = sorted(Path(".").glob("results_*"))
        if not results_dirs:
            print("Error: No results directories found!")
            print("Usage: python plot_comparison.py [path/to/summary.txt]")
            sys.exit(1)
        summary_file = results_dirs[-1] / "summary.txt"

    summary_file = Path(summary_file)
    if not summary_file.exists():
        print(f"Error: File not found: {summary_file}")
        sys.exit(1)

    print(f"Parsing data from: {summary_file}")

    # Parse data
    data = parse_summary_file(summary_file)

    # Select best configurations (highest throughput for each batch size)
    best_data = select_best_configs(data)

    # Create plots
    output_dir = summary_file.parent
    plot_comparison(best_data, output_dir)


if __name__ == "__main__":
    main()
