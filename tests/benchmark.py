import time

import numpy as np

from paircloud.calcs import calculate_kde, calculate_kde_with_hdi, compute_stack_offsets


def py_compute_stack_offsets(data, position, width, side, bins=50):
    n = len(data)
    grid_bins = min(bins, n)
    hist, bin_edges = np.histogram(data, bins=grid_bins)

    bin_indices = np.digitize(data, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, grid_bins - 1)

    stack_counts = np.zeros(grid_bins)
    offsets = np.zeros(n)

    max_stack = hist.max()
    scale_factor = (width / 2) / max(1, max_stack)

    for i in range(n):
        b = bin_indices[i]
        if side in ('positive', 'top', 'right'):
            offsets[i] = position + stack_counts[b] * scale_factor
        elif side in ('negative', 'bottom', 'left'):
            offsets[i] = position - stack_counts[b] * scale_factor
        else:
            s_val = 1 if stack_counts[b] % 2 == 0 else -1
            step = (stack_counts[b] + 1) // 2
            offsets[i] = position + s_val * step * scale_factor
        stack_counts[b] += 1
    return offsets


def py_kde_with_hdi(data, intervals, grid_points=200, bandwidth=None):
    eval_points = np.linspace(data.min() - np.std(data), data.max() + np.std(data), grid_points)
    density = calculate_kde(data, eval_points, bandwidth=bandwidth)

    if not intervals:
        return eval_points, density, []

    sorted_idxs = np.argsort(density)[::-1]
    sorted_density = density[sorted_idxs]
    cumulative_mass = np.cumsum(sorted_density) * (eval_points[1] - eval_points[0])

    if cumulative_mass[-1] > 0:
        cumulative_mass /= cumulative_mass[-1]

    thresholds = []
    # Highest interval (e.g. 95) first -> lowest density threshold
    for interval in sorted(intervals, reverse=True):
        target = interval / 100.0
        idx = np.searchsorted(cumulative_mass, target)
        if idx >= len(sorted_density):
            idx = len(sorted_density) - 1
        thresholds.append(sorted_density[idx])

    return eval_points, density, thresholds


if __name__ == '__main__':
    print('--- Performance Benchmarks ---')
    sizes = [10_000, 100_000]

    for N in sizes:
        print(f'\nEvaluating N = {N:,} points')
        data = np.random.normal(0, 1, N)

        # Warmup
        py_compute_stack_offsets(data[:100], 0, 0.8, 'both')
        compute_stack_offsets(data[:100], 0, 0.8, 'both')
        py_kde_with_hdi(data[:100], [50, 95])
        calculate_kde_with_hdi(data[:100], [50, 95])

        # 1. Stack Offsets (Python loop vs Rust array offset)
        t0 = time.perf_counter_ns()
        py_compute_stack_offsets(data, 0, 0.8, 'both')
        py_time_ns = time.perf_counter_ns() - t0
        py_time_ms = py_time_ns / 1_000_000

        t0 = time.perf_counter_ns()
        compute_stack_offsets(data, 0, 0.8, 'both')
        rs_time_ns = time.perf_counter_ns() - t0
        rs_time_ms = rs_time_ns / 1_000_000

        print('Stack Offsets (dot placement jitter):')
        print(f'  Python: {py_time_ms:7.2f} ms')
        print(f'  Rust:   {rs_time_ms:7.2f} ms')
        if rs_time_ns > 0:
            print(f'  Speedup: {py_time_ns / rs_time_ns:5.1f}x')

        # 2. KDE + HDI processing
        t0 = time.perf_counter_ns()
        py_kde_with_hdi(data, [50, 95])
        py_time_ns = time.perf_counter_ns() - t0
        py_time_ms = py_time_ns / 1_000_000

        t0 = time.perf_counter_ns()
        calculate_kde_with_hdi(data, [50, 95])
        rs_time_ns = time.perf_counter_ns() - t0
        rs_time_ms = rs_time_ns / 1_000_000

        print('KDE Grid + HDI Interpolation:')
        print(f'  Python Numpy: {py_time_ms:7.2f} ms')
        print(f'  Rust Native:  {rs_time_ms:7.2f} ms')
        if rs_time_ns > 0:
            print(f'  Speedup:      {py_time_ns / rs_time_ns:5.1f}x')
