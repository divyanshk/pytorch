#!/usr/bin/env python3
"""
Benchmark script for PyTorch DataLoader with different worker methods.

This script measures:
1. Dataloader initialization time
2. Dataloading speed (time per batch)
3. CPU memory utilization

Usage:
    # For local dataset
    python dataloader_benchmark.py --dataset_type local --data_path /path/to/dataset --batch_size 32 --num_workers 4

    # For HuggingFace streaming dataset
    python dataloader_benchmark.py --dataset_type huggingface --hf_dataset_name "imagenet-1k" --batch_size 32 --num_workers 4
"""

import argparse
import copy
import gc
import sys
import time

# Add local torchvision to path (use development version from pytorch/vision)
_VISION_DIR = "/pytorch/vision/"
sys.path.insert(0, _VISION_DIR)
import torchvision

import psutil

import torchvision.transforms as transforms
from torchvision.models import resnet18

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataset import ConcatDataset


def get_memory_usage(worker_method=None):
    """
    Get current memory usage in MB.

    For multiprocessing: Returns PSS of main process + sum of PSS for all child processes.
    For threading: Returns RSS of main process only (threads share the same address space).

    Args:
        worker_method: 'thread' or 'multiprocessing' (or None for multiprocessing)

    Returns:
        Total memory usage in MB
    """
    process = psutil.Process()

    if worker_method == "thread":
        # For threading, use RSS of main process only
        # Threads share the same address space, so child processes would be misleading
        main_memory = process.memory_info().rss
    else:
        # For multiprocessing, use PSS of main process + children
        # PSS accounts for shared memory proportionally
        main_memory = process.memory_full_info().pss

        # Add memory usage of all child processes
        for child in process.children(recursive=True):
            try:
                child_mem = child.memory_full_info().pss
                main_memory += child_mem
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                # Process might have terminated or doesn't support PSS, fall back to USS
                print(f"Failed to get PSS for {child}, falling back to USS")
                try:
                    child_mem = child.memory_info().uss
                    main_memory += child_mem
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Child might have terminated, skip it
                    pass

    return main_memory / (1024 * 1024)


def print_detailed_memory():
    """Print detailed memory information."""
    process = psutil.Process()
    print("\nDetailed memory information:")
    try:
        print(
            f"  USS (Unique Set Size): {process.memory_full_info().uss / (1024 * 1024):.2f} MB"
        )
        print(
            f"  PSS (Proportional Set Size): {process.memory_full_info().pss / (1024 * 1024):.2f} MB"
        )
        print(
            f"  RSS (Resident Set Size): {process.memory_info().rss / (1024 * 1024):.2f} MB"
        )
    except Exception:
        print("  Detailed memory info not available")


def create_model():
    """Create a simple model for benchmarking."""
    model = resnet18()
    return model


def benchmark_dataloader(
    dataset,
    dataset_type,
    batch_size,
    num_workers,
    num_epochs=1,
    max_batches=10,
    multiprocessing_context=None,
    worker_method=None,
    logging_freq=10,
):
    """Benchmark a dataloader with specific configuration."""
    print("\n--- Benchmarking DataLoader ---")
    print(f"Worker method: {worker_method if worker_method else 'process (default)'}")

    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    # Create model
    model = create_model()

    # Measure memory before dataloader creation
    memory_before = get_memory_usage(worker_method)
    print(f"Memory before DataLoader creation: {memory_before:.2f} MB")
    print_detailed_memory()

    # Measure dataloader initialization time
    start = time.perf_counter()

    # Build DataLoader kwargs
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": dataset_type != "huggingface",
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    if worker_method:
        dataloader_kwargs["worker_method"] = worker_method

    if worker_method != "thread" and multiprocessing_context:
        dataloader_kwargs["multiprocessing_context"] = multiprocessing_context

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    it = iter(dataloader)
    dataloader_init_time = time.perf_counter() - start

    # Measure memory after dataloader creation
    memory_after = get_memory_usage(worker_method)
    print(f"Memory after DataLoader creation: {memory_after:.2f} MB")
    print(f"Memory increase: {memory_after - memory_before:.2f} MB")

    # Create model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Benchmark dataloading speed
    model.train()
    total_batches = 0
    total_samples = 0
    total_time = 0
    total_data_load_time = 0

    # Measure peak memory during training
    peak_memory = memory_after

    print(
        f"\nStarting training loop with {num_epochs} epochs (max {max_batches} batches per epoch)"
    )

    for epoch in range(num_epochs):
        while total_batches < max_batches:
            batch_start = time.perf_counter()

            try:
                inputs, labels = next(it)
            except StopIteration:
                break

            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Capture data fetch time (including sending to device)
            data_load_time = time.perf_counter() - batch_start

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Capture batch time
            batch_time = time.perf_counter() - batch_start

            total_batches += 1
            total_samples += inputs.size(0)
            total_data_load_time += data_load_time
            total_time += batch_time

            # Update peak memory and log memory usage periodically
            if total_batches % 5 == 0:
                # Force garbage collection before measuring memory
                gc.collect()
                current_memory = get_memory_usage(worker_method)

                if current_memory > peak_memory:
                    peak_memory = current_memory

            if total_batches % logging_freq == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {total_batches}, "
                    f"Time: {batch_time:.4f}s, "
                    f"Memory: {current_memory:.2f} MB"
                )

    # Calculate statistics
    avg_data_load_time = (
        total_data_load_time / total_batches if total_batches > 0 else 0
    )
    avg_batch_time = total_time / total_batches if total_batches > 0 else 0
    samples_per_second = total_samples / total_time if total_time > 0 else 0

    results = {
        "dataloader_init_time": dataloader_init_time,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "total_batches": total_batches,
        "avg_batch_time": avg_batch_time,
        "avg_data_load_time": avg_data_load_time,
        "samples_per_second": samples_per_second,
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": peak_memory - memory_before,
    }

    print("\nResults:")
    print(f"  DataLoader init time: {dataloader_init_time:.4f} seconds")
    print(f"  Average data loading time: {avg_data_load_time:.4f} seconds")
    print(f"  Average batch time: {avg_batch_time:.4f} seconds")
    print(f"  Samples per second: {samples_per_second:.2f}")
    print(f"  Peak memory usage: {peak_memory:.2f} MB")
    print(f"  Memory increase: {peak_memory - memory_before:.2f} MB")

    # Clean up
    del model, optimizer
    del dataloader

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    return results


class HuggingFaceStreamingDataset(IterableDataset):
    """Wrapper for HuggingFace streaming datasets to work with PyTorch DataLoader."""

    def __init__(self, dataset_name, split="train", transform=None, max_samples=None):
        """
        Args:
            dataset_name: Name of the HuggingFace dataset (e.g., "imagenet-1k")
            split: Dataset split to use (e.g., "train", "validation")
            transform: Optional transform to apply to images
            max_samples: Maximum number of samples to fetch (for benchmarking)
        """
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise ImportError(
                "HuggingFace datasets library not installed. "
                "Install with: pip install datasets"
            ) from err

        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.max_samples = max_samples

        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True,  # This enables streaming without caching
        )

    def __iter__(self):
        """Iterate over the streaming dataset."""

        for count, item in enumerate(self.dataset):
            if self.max_samples is not None and count >= self.max_samples:
                break

            # This assumes standard image classification format: image and label,
            # where image is a PIL Image and label is an integer
            image = item["image"]
            label = item["label"]

            # Convert grayscale images to RGB to ensure consistent 3-channel format
            if hasattr(image, "mode") and image.mode != "RGB":
                image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            yield image, label
            count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch DataLoader with different worker methods"
    )
    parser.add_argument(
        "--dataset_type",
        choices=["local", "huggingface"],
        default="local",
        help="Type of dataset to use",
    )
    parser.add_argument("--data_path", help="Path to dataset (for local type)")
    parser.add_argument(
        "--hf_dataset_name",
        help="HuggingFace dataset name (e.g., 'imagenet-1k', 'cifar10')",
    )
    parser.add_argument(
        "--hf_split", default="train", help="HuggingFace dataset split (default: train)"
    )
    parser.add_argument(
        "--hf_max_samples",
        type=int,
        default=1000,
        help="Maximum samples to fetch from HuggingFace streaming dataset",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Maximum number of batches per epoch",
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--worker_method",
        choices=["multiprocessing", "thread"],
        default="multiprocessing",
        help="Worker method to use (multiprocessing or thread)",
    )
    parser.add_argument(
        "--multiprocessing_context",
        choices=["fork", "spawn", "forkserver"],
        default="forkserver",
        help="Multiprocessing context to use when worker_method is process (fork, spawn, forkserver)",
    )
    parser.add_argument(
        "--dataset_copies",
        type=int,
        default=1,
        help="Number of copies of the dataset to concatenate (for testing memory usage, local only)",
    )
    parser.add_argument(
        "--logging_freq",
        type=int,
        default=10,
        help="Frequency of logging memory usage during training",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.dataset_type == "local" and not args.data_path:
        parser.error("--data_path is required when --dataset_type is local")
    if args.dataset_type == "huggingface" and not args.hf_dataset_name:
        parser.error("--hf_dataset_name is required when --dataset_type is huggingface")

    # Print system info
    print("System Information:")
    # The following are handy for debugging if building from source worked correctly
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  PyTorch location: {torch.__file__}")
    print(f"  Torchvision version: {torchvision.__version__}")
    print(f"  Torchvision location: {torchvision.__file__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CPU count: {psutil.cpu_count(logical=True)}")
    print(f"  Physical CPU cores: {psutil.cpu_count(logical=False)}")
    print(f"  Total system memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset based on type
    if args.dataset_type == "local":
        print(
            f"\nLoading local dataset from {args.data_path} ({args.dataset_copies} copies)"
        )
        datasets = []
        for _ in range(args.dataset_copies):
            base_dataset = torchvision.datasets.ImageFolder(
                args.data_path, transform=transform
            )
            datasets.append(copy.deepcopy(base_dataset))
            del base_dataset
        dataset = ConcatDataset(datasets)
        print(f"Local dataset size: {len(dataset)}")

    elif args.dataset_type == "huggingface":
        print(f"\nLoading HuggingFace streaming dataset: {args.hf_dataset_name}")
        dataset = HuggingFaceStreamingDataset(
            dataset_name=args.hf_dataset_name,
            split=args.hf_split,
            transform=transform,
            max_samples=args.hf_max_samples,
        )
        print(f"Streaming dataset loaded (max samples: {args.hf_max_samples})")

    # Run benchmark with specified worker method
    benchmark_dataloader(
        dataset,
        args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        multiprocessing_context=args.multiprocessing_context,
        worker_method=args.worker_method,
        num_epochs=args.num_epochs,
        max_batches=args.max_batches,
        logging_freq=args.logging_freq,
    )


if __name__ == "__main__":
    main()
