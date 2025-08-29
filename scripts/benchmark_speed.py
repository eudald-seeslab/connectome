import argparse
import time
import os

import numpy as np
import torch
from torch import nn, optim
from torch_geometric.data import Batch

from configs import config as cfg
from connectome.core.data_processing import DataProcessor
from connectome.core.graph_models import FullGraphModel


def benchmark(num_iters: int, batch_size: int, synthetic: bool = True):
    """Benchmark data pipeline + forward/backward pass.

    Parameters
    ----------
    num_iters : int
        Number of iterations (batches) to benchmark (after 1 warm-up).
    batch_size : int
        Batch size to use.
    synthetic : bool, default True
        If True, generates random images instead of reading from disk. This is
        useful to isolate compute time from I/O.
    """

    # ---- Config tweaks ----
    cfg.batch_size = batch_size
    cfg.debugging = False

    device = cfg.DEVICE

    # ---- Build data processor & model ----
    input_dir = cfg.TRAINING_DATA_DIR if not synthetic else None
    dp = DataProcessor(cfg, input_images_dir=input_dir)
    model = FullGraphModel(dp, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    pixel_dim = 512  # fixed in current pipeline
    num_classes = len(cfg.CLASSES) if not synthetic else 2

    def create_batch():
        # Generate random uint8 images [B, H, W, 3]
        imgs = np.random.randint(0, 256, size=(batch_size, pixel_dim, pixel_dim, 3), dtype=np.uint8)
        labels = np.random.randint(0, num_classes, size=(batch_size,))
        return imgs, labels

    # Warm-up iteration (not timed)
    imgs, labels = create_batch()
    inputs, labels_tensor = dp.process_batch(imgs, labels)
    outputs = model(inputs)
    loss = criterion(outputs, labels_tensor)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device) if device.type == "cuda" else None

    # ---- Timed iterations ----
    data_times, fwd_times, bwd_times, total_times = [], [], [], []

    for _ in range(num_iters):
        start_total = time.perf_counter()

        # Data creation + pre-process
        t0 = time.perf_counter()
        imgs, labels = create_batch()
        inputs, labels_tensor = dp.process_batch(imgs, labels)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t1 = time.perf_counter()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels_tensor)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t2 = time.perf_counter()

        # Backward
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t3 = time.perf_counter()

        data_times.append(t1 - t0)
        fwd_times.append(t2 - t1)
        bwd_times.append(t3 - t2)
        total_times.append(t3 - start_total)

    print("\nBenchmark results (averaged over", num_iters, "iterations):")
    print(f"  Data   time: {np.mean(data_times):.4f} s (std {np.std(data_times):.4f})")
    print(f"  Fwd    time: {np.mean(fwd_times):.4f} s (std {np.std(fwd_times):.4f})")
    print(f"  Bwd    time: {np.mean(bwd_times):.4f} s (std {np.std(bwd_times):.4f})")
    print(f"  Total  time: {np.mean(total_times):.4f} s (std {np.std(total_times):.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark connectome training throughput")
    parser.add_argument("-n", "--num-iters", type=int, default=10, help="Number of iterations to benchmark")
    parser.add_argument("-b", "--batch-size", type=int, default=12, help="Batch size")
    parser.add_argument("--real-data", action="store_true", help="Use real images from training dir instead of synthetic")

    args = parser.parse_args()

    benchmark(num_iters=args.num_iters, batch_size=args.batch_size, synthetic=not args.real_data) 