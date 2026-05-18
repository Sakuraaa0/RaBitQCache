"""
RabitQ Triton Kernels for Bit-Quantized Operations.

This module provides optimized kernels for computing:
    <x̄_b, q̄_u> where x̄_b is binary {0,1} and q̄_u is B_q-bit quantized.

Formula:
    <x̄_b, q̄_u> = Σ_{j=0}^{B_q-1} 2^j · <x̄_b, q̄_u^(j)>
    where q̄_u^(j) is the j-th bit of q_u
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def rabitq_update_mask_kernel(
    mask_ptr,
    indices_ptr,
    num_indices,
    num_quantized,
    pending_start,
    total_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)

    zero_mask = offsets < total_tokens
    tl.store(mask_ptr + offsets, 0, mask=zero_mask)

    idx_mask = offsets < num_indices
    idx = tl.load(indices_ptr + offsets, mask=idx_mask, other=0)
    valid_idx = idx_mask & (idx >= 0) & (idx < num_quantized)
    tl.store(mask_ptr + idx, 1, mask=valid_idx)

    pending_len = total_tokens - pending_start
    pending_mask = offsets < pending_len
    tail_offsets = pending_start + offsets
    tl.store(mask_ptr + tail_offsets, 1, mask=pending_mask)


def update_topk_mask(
    mask_slice: torch.Tensor,
    indices: Optional[torch.Tensor],
    num_quantized: int,
    pending_start: int,
    total_tokens: int,
) -> None:
    if total_tokens == 0:
        return

    pending_start = max(0, min(pending_start, total_tokens))

    if mask_slice.device.type != "cuda":
        mask_slice[:total_tokens].zero_()
        if indices is not None and indices.numel() > 0:
            mask_slice[indices.to(torch.long)] = 1
        if pending_start < total_tokens:
            mask_slice[pending_start:total_tokens] = 1
        return

    if indices is None or indices.numel() == 0:
        indices_device = mask_slice.new_empty(0, dtype=torch.int32)
    else:
        indices_device = indices
        if indices_device.dtype != torch.int32:
            indices_device = indices_device.to(torch.int32)
        indices_device = indices_device.contiguous()

    block = 256
    zero_blocks = triton.cdiv(total_tokens, block)
    pending_len = total_tokens - pending_start
    pending_blocks = triton.cdiv(max(pending_len, 0), block)
    scatter_blocks = triton.cdiv(indices_device.numel(), block)
    grid = (max(zero_blocks, pending_blocks, scatter_blocks, 1), )

    rabitq_update_mask_kernel[grid](
        mask_slice,
        indices_device,
        indices_device.numel(),
        num_quantized,
        pending_start,
        total_tokens,
        BLOCK_SIZE=block,
    )


@triton.jit
def rabitq_binary_dot_quantized_kernel(
    # Input pointers
    x_b_ptr,  # [num_tokens, num_heads, head_size], uint8, values {0,1}
    q_u_ptr,  # [num_heads, head_size], uint8, values [0, 2^B_q - 1]
    # Output pointer
    output_ptr,  # [num_tokens, num_heads]
    # Dimensions
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    B_q: tl.constexpr,
    # Tile sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute <x̄_b, q̄_u> for all tokens and heads using bit operations.

    Each program handles one (token, head) pair.
    """
    # Program IDs
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Base offsets
    x_offset = (token_idx * num_heads + head_idx) * head_size
    q_offset = head_idx * head_size

    # Accumulator for final result
    result = 0.0

    # Process each bit position
    for bit_idx in range(B_q):
        bit_weight = 1 << bit_idx  # 2^bit_idx

        # Accumulator for this bit
        bit_sum = 0

        # Process head_size dimensions in blocks
        for block_start in range(0, head_size, BLOCK_SIZE):
            # Offsets for this block
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < head_size

            # Load x_b and q_u for this block
            x_vals = tl.load(x_b_ptr + x_offset + offsets, mask=mask, other=0).to(tl.uint8)
            q_vals = tl.load(q_u_ptr + q_offset + offsets, mask=mask, other=0).to(tl.uint8)

            # Extract bit_idx-th bit from q_vals
            q_bits = (q_vals >> bit_idx) & 1

            # Compute AND: x_vals & q_bits
            and_result = x_vals & q_bits

            # Accumulate
            bit_sum += tl.sum(and_result.to(tl.int32))

        # Add weighted bit sum to result
        result += bit_weight * bit_sum

    # Store result
    output_offset = token_idx * num_heads + head_idx
    tl.store(output_ptr + output_offset, result.to(tl.float32))


def binary_dot_quantized(
    x_b: torch.Tensor,  # [num_tokens, num_heads, head_size], uint8
    q_u: torch.Tensor,  # [num_heads, head_size], uint8
    B_q: int
) -> torch.Tensor:
    """
    Compute <x̄_b, q̄_u> using Triton kernel.

    Args:
        x_b: Binary vectors {0,1}, shape [num_tokens, num_heads, head_size]
        q_u: B_q-bit quantized vectors, shape [num_heads, head_size]
        B_q: Number of bits in quantization (e.g., 4)

    Returns:
        torch.Tensor: Inner products, shape [num_tokens, num_heads]
    """
    num_tokens, num_heads, head_size = x_b.shape

    # Output tensor
    output = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=x_b.device)

    # Choose block size (tune for performance)
    BLOCK_SIZE = triton.next_power_of_2(min(head_size, 1024))

    # Launch kernel
    grid = (num_tokens, num_heads)

    rabitq_binary_dot_quantized_kernel[grid](
        x_b, q_u, output,
        num_tokens, num_heads, head_size, B_q,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output


# Optional: Optimized version with autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['head_size'],
)
@triton.jit
def rabitq_binary_dot_quantized_kernel_autotuned(
    x_b_ptr, q_u_ptr, output_ptr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    B_q: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Autotuned version of the kernel."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    x_offset = (token_idx * num_heads + head_idx) * head_size
    q_offset = head_idx * head_size

    result = 0.0

    for bit_idx in range(B_q):
        bit_weight = 1 << bit_idx
        bit_sum = 0

        for block_start in range(0, head_size, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < head_size

            x_vals = tl.load(x_b_ptr + x_offset + offsets, mask=mask, other=0).to(tl.uint8)
            q_vals = tl.load(q_u_ptr + q_offset + offsets, mask=mask, other=0).to(tl.uint8)

            q_bits = (q_vals >> bit_idx) & 1
            and_result = x_vals & q_bits

            bit_sum += tl.sum(and_result.to(tl.int32))

        result += bit_weight * bit_sum

    output_offset = token_idx * num_heads + head_idx
    tl.store(output_ptr + output_offset, result.to(tl.float32))


def binary_dot_quantized_autotuned(
    x_b: torch.Tensor,
    q_u: torch.Tensor,
    B_q: int
) -> torch.Tensor:
    """Autotuned version with performance optimization."""
    num_tokens, num_heads, head_size = x_b.shape
    output = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=x_b.device)

    grid = (num_tokens, num_heads)

    rabitq_binary_dot_quantized_kernel_autotuned[grid](
        x_b, q_u, output,
        num_tokens, num_heads, head_size, B_q
    )

    return output


# ========== Fused Key Quantization Kernel ==========

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
    ],
    key=['head_size'],
)
@triton.jit
def quantize_keys_fused_kernel(
    # Input pointers
    keys_ptr,           # [num_tokens, num_heads, head_size]
    centroid_k_ptr,     # [num_heads, head_size]
    rotation_t_ptr,     # [num_heads, head_size, head_size]
    rotation_ptr,       # [num_heads, head_size, head_size]
    centroid_q_ptr,     # [num_heads, head_size]
    # Output pointers
    bits_ptr,           # [num_tokens, num_heads, head_size] uint8
    norms_ptr,          # [num_tokens, num_heads]
    inner_ptr,          # [num_tokens, num_heads]
    sum_bits_ptr,       # [num_tokens, num_heads]
    cq_dot_kr_ptr,      # [num_tokens, num_heads]
    # Dimensions
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    sqrt_head_size: tl.constexpr,
    # Block size
    BLOCK_D: tl.constexpr,
):
    """
    Fused kernel for key quantization:
    1. Center keys: centered = keys - centroid_k
    2. Normalize: normalized = centered / norm(centered)
    3. Rotate inverse: rotated_inv = normalized @ rotation_t
    4. Quantize: bits = (rotated_inv >= 0)
    5. Reconstruct: x_bar = (2*bits - 1)
    6. Rotate forward: rotated_x = x_bar @ rotation
    7. Inner product: inner = sum(rotated_x * normalized)
    8. Sum bits: sum_bits = sum(bits)
    9. Centroid dot: cq_dot_kr = keys · centroid_q
    """
    # Program IDs
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Base offsets
    key_offset = (token_idx * num_heads + head_idx) * head_size
    centroid_offset = head_idx * head_size
    rotation_offset = head_idx * head_size * head_size

    # Step 1: Load key and compute centered = key - centroid_k
    # Also compute cq_dot_kr = key · centroid_q in the same pass
    cq_dot_kr_acc = 0.0
    norm_sq = 0.0

    for d_block in range(0, head_size, BLOCK_D):
        d_offsets = d_block + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_size

        # Load key, centroid_k, centroid_q
        key_vals = tl.load(keys_ptr + key_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        ck_vals = tl.load(centroid_k_ptr + centroid_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        cq_vals = tl.load(centroid_q_ptr + centroid_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        # Center
        centered_block = key_vals - ck_vals

        # Accumulate norm squared
        norm_sq += tl.sum(centered_block * centered_block, axis=0)

        # Accumulate cq · kr
        cq_dot_kr_acc += tl.sum(key_vals * cq_vals, axis=0)

        # Store centered for later use (we'll reload keys and recompute if needed)

    # Compute norm
    norm = tl.sqrt(norm_sq + 1e-12)  # Add epsilon for numerical stability

    # Store norm
    norm_out_offset = token_idx * num_heads + head_idx
    tl.store(norms_ptr + norm_out_offset, norm)

    # Store cq_dot_kr
    tl.store(cq_dot_kr_ptr + norm_out_offset, cq_dot_kr_acc)

    # Step 2: Normalize, rotate, quantize
    # We need to process in blocks and accumulate for rotation (matrix multiply)
    rotated_inv = tl.zeros([head_size], dtype=tl.float32)

    # For each output dimension of rotated_inv
    for out_d in range(head_size):
        acc = 0.0

        # Load rotation_t row: rotation_t[head_idx, out_d, :]
        rotation_row_offset = rotation_offset + out_d * head_size

        for d_block in range(0, head_size, BLOCK_D):
            d_offsets = d_block + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < head_size

            # Reload key and centroid_k
            key_vals = tl.load(keys_ptr + key_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
            ck_vals = tl.load(centroid_k_ptr + centroid_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

            # Normalize
            normalized_vals = (key_vals - ck_vals) / norm

            # Load rotation_t weights
            rot_weights = tl.load(rotation_t_ptr + rotation_row_offset + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

            # Accumulate dot product
            acc += tl.sum(normalized_vals * rot_weights, axis=0)

        # Store in rotated_inv (will quantize in next pass)
        # For now, we compute bits directly
        bit_val = tl.where(acc >= 0.0, 1, 0)

        # Store bit
        bit_offset = key_offset + out_d
        tl.store(bits_ptr + bit_offset, bit_val.to(tl.uint8))

    # Step 3: Optimized inner product computation
    # Strategy: first compute rotated_normalized = P^T @ normalized,
    # then compute inner = <x_bar, rotated_normalized>, where x_bar = ±1
    # This avoids floating-point rotation on x_bar by exploiting the integer ±1 property

    inner_acc = 0.0
    sum_bits_acc = 0.0

    # For each dimension out_d, compute rotated_normalized[out_d] = (P^T @ normalized)[out_d]
    # then immediately compute inner += x_bar[out_d] * rotated_normalized[out_d]
    for out_d in range(head_size):
        rotated_norm_val = 0.0

        # Compute rotated_normalized[out_d] = sum_i (rotation_t[out_d, i] * normalized[i])
        # rotation_t[out_d, :] is the out_d-th row
        rotation_t_row_offset = rotation_offset + out_d * head_size

        for in_d in range(head_size):
            # Load normalized value
            key_val = tl.load(keys_ptr + key_offset + in_d).to(tl.float32)
            ck_val = tl.load(centroid_k_ptr + centroid_offset + in_d).to(tl.float32)
            normalized_val = (key_val - ck_val) / norm

            # Load rotation_t weight
            rot_t_weight = tl.load(rotation_t_ptr + rotation_t_row_offset + in_d).to(tl.float32)

            # Accumulate rotated_normalized[out_d]
            rotated_norm_val += normalized_val * rot_t_weight

        # Load bit and compute x_bar = ±1
        bit_val = tl.load(bits_ptr + key_offset + out_d).to(tl.float32)
        x_bar_val = 2.0 * bit_val - 1.0  # ±1, integer value

        # Accumulate inner product: <x_bar, rotated_normalized>
        inner_acc += x_bar_val * rotated_norm_val

        # Sum bits
        sum_bits_acc += bit_val

    # Store inner product and sum_bits
    tl.store(inner_ptr + norm_out_offset, inner_acc)
    tl.store(sum_bits_ptr + norm_out_offset, sum_bits_acc)


def quantize_keys_fused(
    keys: torch.Tensor,           # [num_tokens, num_heads, head_size]
    centroid_k: torch.Tensor,     # [num_heads, head_size]
    rotation_t: torch.Tensor,     # [num_heads, head_size, head_size]
    rotation: torch.Tensor,       # [num_heads, head_size, head_size]
    centroid_q: torch.Tensor,     # [num_heads, head_size]
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused quantization of keys using Triton kernel.

    Returns:
        bits: [num_tokens, num_heads, head_size] uint8
        norms: [num_tokens, num_heads]
        inner: [num_tokens, num_heads]
        sum_bits: [num_tokens, num_heads]
        cq_dot_kr: [num_tokens, num_heads]
    """
    num_tokens, num_heads, _ = keys.shape
    device = keys.device
    dtype = keys.dtype

    # Ensure contiguous inputs for Triton
    if not keys.is_contiguous():
        keys = keys.contiguous()
    if not centroid_k.is_contiguous():
        centroid_k = centroid_k.contiguous()
    if not rotation_t.is_contiguous():
        rotation_t = rotation_t.contiguous()
    if not rotation.is_contiguous():
        rotation = rotation.contiguous()
    if not centroid_q.is_contiguous():
        centroid_q = centroid_q.contiguous()

    # Allocate outputs
    bits = torch.empty((num_tokens, num_heads, head_size), dtype=torch.uint8, device=device)
    norms = torch.empty((num_tokens, num_heads), dtype=dtype, device=device)
    inner = torch.empty((num_tokens, num_heads), dtype=dtype, device=device)
    sum_bits = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=device)
    cq_dot_kr = torch.empty((num_tokens, num_heads), dtype=dtype, device=device)

    # Launch kernel
    grid = (num_tokens, num_heads)
    sqrt_head_size = float(head_size ** 0.5)

    quantize_keys_fused_kernel[grid](
        keys, centroid_k, rotation_t, rotation,
        centroid_q, bits, norms, inner, sum_bits, cq_dot_kr,
        num_tokens, num_heads, head_size, sqrt_head_size
    )

    return bits, norms, inner, sum_bits, cq_dot_kr


# ========== Batched Binary Dot Product Kernel ==========

@triton.jit
def rabitq_binary_dot_batched_kernel(
    # Input pointers
    x_b_ptr,      # [T_key, num_heads, head_size], uint8
    q_u_ptr,      # [T_query, num_heads, head_size], uint8
    # Output pointer
    output_ptr,   # [T_query, T_key, num_heads]
    # Dimensions - RUNTIME parameters (not constexpr) to avoid recompilation
    T_key,        # Runtime param
    T_query,      # Runtime param
    num_heads,    # Runtime param
    head_size: tl.constexpr,  # Keep as constexpr for loop unrolling
    B_q: tl.constexpr,
    # Strides
    x_stride_t,
    x_stride_h,
    q_stride_t,
    q_stride_h,
    o_stride_tq,
    o_stride_tk,
    # Tile sizes
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TQ: tl.constexpr,  # Process multiple queries per block
):
    """
    Optimized batched dot product - process multiple output elements per block.

    Key change: T_key, T_query, num_heads are runtime parameters to avoid recompilation
    when these values change during inference.
    """
    # Get base indices
    pid = tl.program_id(0)

    # Decode: (tq_base, tk, h) - use runtime division
    pairs_per_head = (T_query * T_key) // BLOCK_TQ
    head_idx = pid // pairs_per_head
    pair_idx = pid % pairs_per_head

    tk_idx = pair_idx // (T_query // BLOCK_TQ)
    tq_base_idx = (pair_idx % (T_query // BLOCK_TQ)) * BLOCK_TQ

    # Bounds check
    if head_idx >= num_heads or tk_idx >= T_key:
        return

    # Process BLOCK_TQ query tokens
    tq_offsets = tq_base_idx + tl.arange(0, BLOCK_TQ)
    tq_mask = tq_offsets < T_query

    # Results for this block [BLOCK_TQ]
    results = tl.zeros((BLOCK_TQ,), dtype=tl.float32)

    # Load keys once (shared across all queries)
    x_base = tk_idx * x_stride_t + head_idx * x_stride_h

    # Process dimension in blocks
    for d_start in range(0, head_size, BLOCK_SIZE):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE)
        d_mask = d_offsets < head_size

        # Load x_b [BLOCK_SIZE]
        x_vals = tl.load(x_b_ptr + x_base + d_offsets, mask=d_mask, other=0).to(tl.int32)

        # Load q_u for all queries [BLOCK_TQ, BLOCK_SIZE]
        q_offsets = (tq_offsets[:, None] * q_stride_t +
                    head_idx * q_stride_h +
                    d_offsets[None, :])
        q_vals = tl.load(q_u_ptr + q_offsets,
                        mask=tq_mask[:, None] & d_mask[None, :],
                        other=0).to(tl.int32)

        # Compute products [BLOCK_TQ, BLOCK_SIZE]
        products = q_vals * x_vals[None, :]

        # Sum over dimension [BLOCK_TQ]
        results += tl.sum(products, axis=1)

    # Store results
    output_offsets = (tq_offsets * o_stride_tq +
                     tk_idx * o_stride_tk +
                     head_idx)
    tl.store(output_ptr + output_offsets, results, mask=tq_mask)


def binary_dot_quantized_batched(
    x_b: torch.Tensor,  # [T_key, num_heads, head_size], uint8
    q_u: torch.Tensor,  # [T_query, num_heads, head_size], uint8
    B_q: int
) -> torch.Tensor:
    """
    Batched binary dot product for RabitQ quantized tensors.

    Args:
        x_b: Quantized keys [T_key, num_heads, head_size], dtype=uint8
        q_u: Quantized queries [T_query, num_heads, head_size], dtype=uint8
        B_q: Number of bits used for quantization

    Returns:
        output: [T_query, T_key, num_heads], dtype=float32
    """
    T_key, num_heads, head_size = x_b.shape
    T_query, num_heads_q, head_size_q = q_u.shape

    assert head_size == head_size_q, f"Head sizes must match: {head_size} vs {head_size_q}"
    assert num_heads == num_heads_q, f"Number of heads must match: {num_heads} vs {num_heads_q}"

    # Allocate output
    output = torch.empty((T_query, T_key, num_heads), dtype=torch.float32, device=x_b.device)

    # Choose block sizes
    BLOCK_SIZE = triton.next_power_of_2(min(head_size, 128))
    BLOCK_TQ = min(triton.next_power_of_2(T_query), 8)  # Process up to 8 queries per block

    # Grid: reduce by BLOCK_TQ factor
    num_blocks = triton.cdiv(T_query * T_key, BLOCK_TQ) * num_heads
    grid = (num_blocks,)

    rabitq_binary_dot_batched_kernel[grid](
        x_b, q_u, output,
        T_key, T_query, num_heads, head_size, B_q,
        x_b.stride(0), x_b.stride(1),
        q_u.stride(0), q_u.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_TQ=BLOCK_TQ
    )

    return output


# ========== Fused Query Preprocessing Kernel ==========
# Fuses: center + normalize + rotate + quantize into a single kernel

@triton.jit
def fused_query_preprocess_kernel(
    # Input pointers
    query_ptr,        # [num_tokens, num_heads * head_size], float16/bfloat16
    centroid_q_ptr,   # [num_kv_heads, head_size], float16/bfloat16
    rotation_ptr,     # [num_kv_heads, head_size, head_size], float16/bfloat16 (can be None)
    centroid_k_ptr,   # [num_kv_heads, head_size], float16/bfloat16
    # Output pointers
    q_u_ptr,          # [num_tokens, num_kv_heads, head_size], uint8
    delta_ptr,        # [num_tokens, num_kv_heads], float32
    v_l_ptr,          # [num_tokens, num_kv_heads], float32
    sum_q_u_ptr,      # [num_tokens, num_kv_heads], float32
    q_norm_ptr,       # [num_tokens, num_kv_heads], float32
    qr_dot_ck_ptr,    # [num_tokens, num_kv_heads], float32
    # Dimensions
    num_tokens,
    num_heads,
    num_kv_heads,
    head_size: tl.constexpr,
    num_queries_per_kv,
    B_q,              # quantization bits
    use_rotation,     # whether to apply rotation
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for query preprocessing:
    1. Center query: q_centered = query - centroid_q
    2. Normalize: q_normalized = q_centered / ||q_centered||
    3. Average over query groups (GQA)
    4. Rotate: q_rotated = q_normalized @ rotation_t
    5. Quantize: q_u = round((q_rotated - v_l) / delta)
    6. Compute auxiliary values: q_norm, qr_dot_ck, sum_q_u

    Grid: (num_tokens, num_kv_heads)
    Each thread block processes one (token, kv_head) pair.
    """
    token_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if token_idx >= num_tokens or kv_head_idx >= num_kv_heads:
        return

    # Offsets for head_size dimension
    offs_d = tl.arange(0, BLOCK_SIZE)
    mask_d = offs_d < head_size

    # Load centroid_q for this kv_head: [head_size]
    centroid_q_offset = kv_head_idx * head_size + offs_d
    centroid_q = tl.load(centroid_q_ptr + centroid_q_offset, mask=mask_d, other=0.0).to(tl.float32)

    # Load centroid_k for this kv_head: [head_size]
    centroid_k = tl.load(centroid_k_ptr + centroid_q_offset, mask=mask_d, other=0.0).to(tl.float32)

    # Accumulate normalized query across query group (for GQA)
    acc_normalized = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_q_norm = tl.zeros([1], dtype=tl.float32)
    acc_qr_dot_ck = tl.zeros([1], dtype=tl.float32)

    # Process each query head in this kv group
    for q_idx in range(num_queries_per_kv):
        head_idx = kv_head_idx * num_queries_per_kv + q_idx

        # Load query for this (token, head): [head_size]
        query_offset = token_idx * num_heads * head_size + head_idx * head_size + offs_d
        query = tl.load(query_ptr + query_offset, mask=mask_d, other=0.0).to(tl.float32)

        # Center: q_centered = query - centroid_q
        q_centered = query - centroid_q

        # Compute norm: ||q_centered||
        q_centered_sq = q_centered * q_centered
        norm_sq = tl.sum(q_centered_sq, axis=0)
        norm = tl.sqrt(norm_sq)
        norm = tl.maximum(norm, 1e-6)

        # Normalize
        q_normalized = q_centered / norm

        # Accumulate for averaging
        acc_normalized += q_normalized
        acc_q_norm += norm

        # Compute q_r dot centroid_k (using original query, not normalized)
        # q_r = mean of queries in group, we'll divide by num_queries_per_kv later
        q_view = query  # This is the original query
        qr_ck = tl.sum(q_view * centroid_k, axis=0)
        acc_qr_dot_ck += qr_ck

    # Average over query group
    normalized_kv = acc_normalized / num_queries_per_kv
    q_norm_avg = acc_q_norm / num_queries_per_kv
    qr_dot_ck_avg = acc_qr_dot_ck / num_queries_per_kv

    # Apply rotation if provided
    if use_rotation:
        # rotated_q = normalized_kv @ rotation_t
        # rotation_t is [num_kv_heads, head_size, head_size]
        # We compute: rotated[d] = sum_e(normalized_kv[e] * rotation[kv_head, e, d])
        rotated_q = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for e in range(head_size):
            # Load rotation[kv_head, e, :] - the e-th row
            rot_offset = kv_head_idx * head_size * head_size + e * head_size + offs_d
            rot_row = tl.load(rotation_ptr + rot_offset, mask=mask_d, other=0.0).to(tl.float32)
            # normalized_kv[e] is a scalar for this element
            norm_e_offset = kv_head_idx * head_size + e
            # We need to extract element e from normalized_kv
            # Since normalized_kv is in registers, we use a different approach:
            # We accumulate: rotated[d] += normalized_kv[e] * rotation[e, d]
            # But normalized_kv[e] needs extraction - use gather
            norm_e_mask = offs_d == e
            norm_e_val = tl.sum(tl.where(norm_e_mask, normalized_kv, 0.0), axis=0)
            rotated_q += norm_e_val * rot_row
    else:
        rotated_q = normalized_kv

    # Quantize: find min/max, compute delta, quantize
    v_l_val = tl.min(rotated_q, axis=0)
    v_r_val = tl.max(rotated_q, axis=0)

    # Compute delta = (v_r - v_l) / (2^B_q - 1)
    max_val = (1 << B_q) - 1  # 2^B_q - 1
    delta_val = (v_r_val - v_l_val) / max_val
    delta_val = tl.maximum(delta_val, 1e-8)

    # Quantize: q_u = round((rotated_q - v_l) / delta)
    # Use floor(x + 0.5) for rounding since tl.math.round may not exist
    scaled = (rotated_q - v_l_val) / delta_val
    q_quantized = tl.floor(scaled + 0.5)
    q_quantized = tl.maximum(tl.minimum(q_quantized, max_val), 0.0)

    # Compute sum_q_u
    sum_q_u_val = tl.sum(q_quantized, axis=0)

    # Store outputs
    # q_u: [num_tokens, num_kv_heads, head_size]
    q_u_offset = token_idx * num_kv_heads * head_size + kv_head_idx * head_size + offs_d
    tl.store(q_u_ptr + q_u_offset, q_quantized.to(tl.uint8), mask=mask_d)

    # Scalar outputs: [num_tokens, num_kv_heads]
    scalar_offset = token_idx * num_kv_heads + kv_head_idx
    tl.store(delta_ptr + scalar_offset, delta_val)
    tl.store(v_l_ptr + scalar_offset, v_l_val)
    tl.store(sum_q_u_ptr + scalar_offset, sum_q_u_val)
    tl.store(q_norm_ptr + scalar_offset, q_norm_avg)
    tl.store(qr_dot_ck_ptr + scalar_offset, qr_dot_ck_avg)


def fused_query_preprocess(
    query: torch.Tensor,           # [num_tokens, num_heads, head_size]
    centroid_q: torch.Tensor,      # [num_kv_heads, head_size]
    centroid_k: torch.Tensor,      # [num_kv_heads, head_size]
    rotation_t: torch.Tensor,      # [num_kv_heads, head_size, head_size] or None
    num_queries_per_kv: int,
    B_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused query preprocessing: center + normalize + rotate + quantize.

    Args:
        query: Query tensor [num_tokens, num_heads, head_size]
        centroid_q: Query centroid [num_kv_heads, head_size]
        centroid_k: Key centroid [num_kv_heads, head_size]
        rotation_t: Rotation matrix [num_kv_heads, head_size, head_size] or None
        num_queries_per_kv: Number of query heads per KV head (for GQA)
        B_q: Quantization bits

    Returns:
        q_u: Quantized query [num_tokens, num_kv_heads, head_size], uint8
        delta: Quantization step [num_tokens, num_kv_heads], float32
        v_l: Quantization min [num_tokens, num_kv_heads], float32
        sum_q_u: Sum of quantized values [num_tokens, num_kv_heads], float32
        q_norm: Query norms [num_tokens, num_kv_heads], float32
        qr_dot_ck: Query dot centroid_k [num_tokens, num_kv_heads], float32
    """
    num_tokens = query.size(0)
    num_heads = query.size(1)
    head_size = query.size(2)
    num_kv_heads = centroid_q.size(0)

    device = query.device

    # Flatten query for kernel: [num_tokens, num_heads * head_size]
    query_flat = query.view(num_tokens, num_heads * head_size).contiguous()

    # Allocate outputs
    q_u = torch.empty((num_tokens, num_kv_heads, head_size), dtype=torch.uint8, device=device)
    delta = torch.empty((num_tokens, num_kv_heads), dtype=torch.float32, device=device)
    v_l = torch.empty((num_tokens, num_kv_heads), dtype=torch.float32, device=device)
    sum_q_u = torch.empty((num_tokens, num_kv_heads), dtype=torch.float32, device=device)
    q_norm = torch.empty((num_tokens, num_kv_heads), dtype=torch.float32, device=device)
    qr_dot_ck = torch.empty((num_tokens, num_kv_heads), dtype=torch.float32, device=device)

    # Block size (must be >= head_size and power of 2)
    BLOCK_SIZE = triton.next_power_of_2(head_size)

    # Grid: (num_tokens, num_kv_heads)
    grid = (num_tokens, num_kv_heads)

    use_rotation = rotation_t is not None
    if rotation_t is None:
        # Create dummy tensor for rotation (won't be used)
        rotation_t = torch.empty((1,), device=device)

    fused_query_preprocess_kernel[grid](
        query_flat,
        centroid_q,
        rotation_t,
        centroid_k,
        q_u,
        delta,
        v_l,
        sum_q_u,
        q_norm,
        qr_dot_ck,
        num_tokens,
        num_heads,
        num_kv_heads,
        head_size,
        num_queries_per_kv,
        B_q,
        use_rotation,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return q_u, delta, v_l, sum_q_u, q_norm, qr_dot_ck
