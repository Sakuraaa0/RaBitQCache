#!/usr/bin/env python3
"""
Benchmark test for INT4 x Binary GEMV kernels.

Compares:
1. Original kernel (with dp4a optimization)
2. Packed Binary kernel (8x memory bandwidth reduction)
3. Warp kernel with shared memory

Usage:
    conda run -n rabitq python test/test_kernel_benchmark.py
"""

import torch
import time
import sys
import os

# Add vllm to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vllm'))

from vllm import _custom_ops as ops


def pack_binary_to_uint32(x_b: torch.Tensor) -> torch.Tensor:
    """
    Pack binary tensor from [N, H, D] uint8 to [N, H, D/32] uint32.
    """
    N, H, D = x_b.shape
    assert D % 32 == 0, f"head_size must be divisible by 32, got {D}"
    x_b_reshaped = x_b.view(N, H, D // 32, 32)
    # Use int64 for bit positions since CUDA doesn't support uint32 arange
    bit_positions = torch.arange(32, device=x_b.device, dtype=torch.int64)
    # Convert x_b to int64, shift, sum, then convert to int32 (which will be reinterpreted as uint32)
    packed = (x_b_reshaped.to(torch.int64) << bit_positions).sum(dim=-1).to(torch.int32)
    return packed.contiguous()


def generate_test_data(
    num_tokens: int,
    num_quantized: int,
    num_heads: int,
    head_size: int,
    device: str = "cuda"
):
    """Generate random test data for benchmarking."""
    # Query data (INT4 quantized, stored as uint8)
    q_u = torch.randint(0, 16, (num_tokens, num_heads, head_size),
                        dtype=torch.uint8, device=device)

    # Binary keys (0 or 1, stored as uint8)
    x_b = torch.randint(0, 2, (num_quantized, num_heads, head_size),
                        dtype=torch.uint8, device=device)

    # Metadata tensors
    delta_vals = torch.rand(num_tokens, num_heads, dtype=torch.float32, device=device) * 0.1 + 0.01
    v_l_vals = torch.rand(num_tokens, num_heads, dtype=torch.float32, device=device) * 0.5 - 0.25
    sum_q_u = q_u.sum(dim=-1, dtype=torch.float32)
    sum_x_b = x_b.sum(dim=-1, dtype=torch.float32)

    key_norms = torch.rand(num_quantized, num_heads, dtype=torch.float32, device=device) + 0.5
    k_bar_dot_k = torch.rand(num_quantized, num_heads, dtype=torch.float32, device=device) * 2 - 1
    cq_dot_kr = torch.rand(num_quantized, num_heads, dtype=torch.float32, device=device)

    q_norms = torch.rand(num_tokens, num_heads, dtype=torch.float32, device=device) + 0.5
    qr_dot_ck = torch.rand(num_tokens, num_heads, dtype=torch.float32, device=device)
    cq_dot_ck = torch.rand(num_heads, dtype=torch.float32, device=device)

    sqrt_head_size = float(head_size ** 0.5)
    denom_eps = 1e-6

    return {
        'q_u': q_u,
        'delta_vals': delta_vals,
        'v_l_vals': v_l_vals,
        'sum_q_u': sum_q_u,
        'x_b': x_b,
        'sum_x_b': sum_x_b,
        'key_norms': key_norms,
        'k_bar_dot_k': k_bar_dot_k,
        'cq_dot_kr': cq_dot_kr,
        'q_norms': q_norms,
        'qr_dot_ck': qr_dot_ck,
        'cq_dot_ck': cq_dot_ck,
        'sqrt_head_size': sqrt_head_size,
        'denom_eps': denom_eps,
    }


def benchmark_original_kernel(data: dict, warmup: int = 10, iterations: int = 100):
    """Benchmark the original dp4a-optimized kernel."""
    # Warmup
    for _ in range(warmup):
        _ = ops.rabitq_int4_binary_scores(
            data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
            data['x_b'], data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
            data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
            data['sqrt_head_size'], data['denom_eps']
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        scores = ops.rabitq_int4_binary_scores(
            data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
            data['x_b'], data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
            data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
            data['sqrt_head_size'], data['denom_eps']
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return scores, elapsed / iterations * 1000  # ms per iteration


def benchmark_warp_kernel(data: dict, warmup: int = 10, iterations: int = 100):
    """Benchmark the warp-level parallel kernel."""
    # Pack x_b
    x_b_packed = pack_binary_to_uint32(data['x_b'])

    # Warmup
    for _ in range(warmup):
        _ = ops.rabitq_int4_packed_binary_scores_warp(
            data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
            x_b_packed, data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
            data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
            data['denom_eps']
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        scores = ops.rabitq_int4_packed_binary_scores_warp(
            data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
            x_b_packed, data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
            data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
            data['denom_eps']
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return scores, elapsed / iterations * 1000  # ms per iteration


def benchmark_packed_kernel(data: dict, warmup: int = 10, iterations: int = 100):
    """Benchmark the packed binary kernel."""
    # Pack x_b
    x_b_packed = pack_binary_to_uint32(data['x_b'])

    # Warmup
    for _ in range(warmup):
        _ = ops.rabitq_int4_packed_binary_scores(
            data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
            x_b_packed, data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
            data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
            data['denom_eps']
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        scores = ops.rabitq_int4_packed_binary_scores(
            data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
            x_b_packed, data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
            data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
            data['denom_eps']
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return scores, elapsed / iterations * 1000  # ms per iteration


def benchmark_packed_kernel_with_packing(data: dict, warmup: int = 10, iterations: int = 100):
    """Benchmark the packed binary kernel including packing overhead."""
    # Warmup
    for _ in range(warmup):
        x_b_packed = pack_binary_to_uint32(data['x_b'])
        _ = ops.rabitq_int4_packed_binary_scores(
            data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
            x_b_packed, data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
            data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
            data['denom_eps']
        )
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        x_b_packed = pack_binary_to_uint32(data['x_b'])
        scores = ops.rabitq_int4_packed_binary_scores(
            data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
            x_b_packed, data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
            data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
            data['denom_eps']
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return scores, elapsed / iterations * 1000  # ms per iteration


def verify_correctness(data: dict):
    """Verify that both kernels produce the same results."""
    print("\n" + "="*60)
    print("Correctness Verification")
    print("="*60)

    # Original kernel
    scores_orig = ops.rabitq_int4_binary_scores(
        data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
        data['x_b'], data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
        data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
        data['sqrt_head_size'], data['denom_eps']
    )

    # Packed kernel
    x_b_packed = pack_binary_to_uint32(data['x_b'])
    scores_packed = ops.rabitq_int4_packed_binary_scores(
        data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
        x_b_packed, data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
        data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
        data['denom_eps']
    )

    # Warp kernel
    scores_warp = ops.rabitq_int4_packed_binary_scores_warp(
        data['q_u'], data['delta_vals'], data['v_l_vals'], data['sum_q_u'],
        x_b_packed, data['sum_x_b'], data['key_norms'], data['k_bar_dot_k'],
        data['cq_dot_kr'], data['q_norms'], data['qr_dot_ck'], data['cq_dot_ck'],
        data['denom_eps']
    )

    # Compare packed kernel
    max_diff = (scores_orig - scores_packed).abs().max().item()
    mean_diff = (scores_orig - scores_packed).abs().mean().item()

    print(f"Scores shape: {scores_orig.shape}")
    print(f"Packed vs Original - Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

    # Compare warp kernel
    max_diff_warp = (scores_orig - scores_warp).abs().max().item()
    mean_diff_warp = (scores_orig - scores_warp).abs().mean().item()
    print(f"Warp vs Original - Max diff: {max_diff_warp:.6e}, Mean diff: {mean_diff_warp:.6e}")

    if max_diff < 1e-4 and max_diff_warp < 1e-4:
        print("PASS: All results are numerically equivalent")
        return True
    else:
        print("FAIL: Results differ significantly!")
        return False


def run_benchmark(
    num_tokens: int,
    num_quantized: int,
    num_heads: int,
    head_size: int,
    warmup: int = 10,
    iterations: int = 100
):
    """Run benchmark for a specific configuration."""
    print(f"\nConfig: tokens={num_tokens}, quantized={num_quantized}, "
          f"heads={num_heads}, head_size={head_size}")
    print("-" * 60)

    data = generate_test_data(num_tokens, num_quantized, num_heads, head_size)

    # Benchmark original kernel
    _, time_orig = benchmark_original_kernel(data, warmup, iterations)

    # Benchmark packed kernel (without packing overhead)
    _, time_packed = benchmark_packed_kernel(data, warmup, iterations)

    # Benchmark warp kernel
    _, time_warp = benchmark_warp_kernel(data, warmup, iterations)

    # Benchmark packed kernel (with packing overhead)
    _, time_packed_with_pack = benchmark_packed_kernel_with_packing(data, warmup, iterations)

    speedup_packed = time_orig / time_packed if time_packed > 0 else float('inf')
    speedup_warp = time_orig / time_warp if time_warp > 0 else float('inf')

    print(f"Original kernel (dp4a):     {time_orig:.4f} ms")
    print(f"Packed kernel:              {time_packed:.4f} ms  ({speedup_packed:.2f}x speedup)")
    print(f"Warp kernel:                {time_warp:.4f} ms  ({speedup_warp:.2f}x speedup)")

    return {
        'config': (num_tokens, num_quantized, num_heads, head_size),
        'time_orig': time_orig,
        'time_packed': time_packed,
        'time_warp': time_warp,
        'time_packed_with_pack': time_packed_with_pack,
        'speedup_packed': speedup_packed,
        'speedup_warp': speedup_warp,
    }


def main():
    print("="*60)
    print("RabitQ INT4 x Binary GEMV Kernel Benchmark")
    print("="*60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        return 1

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"CUDA version: {torch.version.cuda}")

    # First verify correctness with small data
    print("\n" + "="*60)
    print("Step 1: Verify Correctness")
    print("="*60)
    small_data = generate_test_data(
        num_tokens=2,
        num_quantized=64,
        num_heads=8,
        head_size=128
    )
    if not verify_correctness(small_data):
        print("Correctness check failed! Aborting benchmark.")
        return 1

    # Benchmark configurations
    print("\n" + "="*60)
    print("Step 2: Performance Benchmark")
    print("="*60)

    configs = [
        # (num_tokens, num_quantized, num_heads, head_size)
        (1, 1024, 8, 128),      # Single token, 1K KV cache
        (1, 4096, 8, 128),      # Single token, 4K KV cache
        (1, 8192, 8, 128),      # Single token, 8K KV cache
        (1, 16384, 8, 128),     # Single token, 16K KV cache
        (4, 4096, 8, 128),      # 4 tokens, 4K KV cache
        (8, 4096, 8, 128),      # 8 tokens, 4K KV cache
        (1, 4096, 32, 128),     # Single token, 32 heads (like Llama-70B)
        (1, 8192, 8, 64),       # Single token, smaller head size
    ]

    results = []
    for config in configs:
        try:
            result = run_benchmark(*config, warmup=20, iterations=100)
            results.append(result)
        except Exception as e:
            print(f"Error with config {config}: {e}")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Config':<30} {'Original':>10} {'Packed':>10} {'Warp':>10} {'Packed':>8} {'Warp':>8}")
    print(f"{'':30} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'speedup':>8} {'speedup':>8}")
    print("-"*78)
    for r in results:
        config_str = f"{r['config'][0]}x{r['config'][1]}x{r['config'][2]}x{r['config'][3]}"
        print(f"{config_str:<30} {r['time_orig']:>10.4f} {r['time_packed']:>10.4f} {r['time_warp']:>10.4f} {r['speedup_packed']:>7.2f}x {r['speedup_warp']:>7.2f}x")

    avg_speedup_packed = sum(r['speedup_packed'] for r in results) / len(results) if results else 0
    avg_speedup_warp = sum(r['speedup_warp'] for r in results) / len(results) if results else 0
    print("-"*78)
    print(f"Average speedup: Packed={avg_speedup_packed:.2f}x, Warp={avg_speedup_warp:.2f}x")

    return 0


if __name__ == "__main__":
    exit(main())
