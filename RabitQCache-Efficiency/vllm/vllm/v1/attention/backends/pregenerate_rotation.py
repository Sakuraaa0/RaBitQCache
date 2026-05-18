#!/usr/bin/env python3
"""
Pre-generate rotation matrices for RabitQ to avoid expensive runtime computation.

This script generates orthogonal rotation matrices using QR decomposition and saves them
to a file that can be loaded at runtime, eliminating the 150+ second initialization overhead.

Usage:
    python pregenerate_rotation.py --num-heads 8 --head-size 128 --output rotation_8h_128d.pt
"""

import argparse
import torch
import time


def generate_rotation_matrices(num_heads: int, head_size: int, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate orthogonal rotation matrices using QR decomposition.

    Args:
        num_heads: Number of KV heads
        head_size: Dimension of each head
        seed: Random seed for reproducibility

    Returns:
        (rotation, rotation_t): Forward and transposed rotation matrices
    """
    print(f"Generating rotation matrices: {num_heads} heads × {head_size}×{head_size} dimensions")
    start_time = time.time()

    # Use deterministic seed
    torch.manual_seed(seed)

    rotation = torch.empty(num_heads, head_size, head_size, dtype=torch.float32)

    for head in range(num_heads):
        # Generate random matrix
        random_matrix = torch.randn(head_size, head_size, dtype=torch.float32)

        # QR decomposition to get orthogonal matrix
        q, r = torch.linalg.qr(random_matrix)

        # Ensure uniform distribution over orthogonal group (correct the signs)
        d = torch.diag(r)
        ph = d.sign()
        q *= ph

        rotation[head] = q

        if (head + 1) % 10 == 0 or head == num_heads - 1:
            elapsed = time.time() - start_time
            print(f"  Progress: {head + 1}/{num_heads} heads ({elapsed:.2f}s)")

    rotation_t = rotation.transpose(-1, -2).contiguous()

    elapsed = time.time() - start_time
    print(f"✓ Generation completed in {elapsed:.2f}s")

    return rotation, rotation_t


def verify_orthogonality(rotation: torch.Tensor, tolerance: float = 1e-5) -> bool:
    """Verify that rotation matrices are orthogonal (R @ R^T = I)."""
    num_heads = rotation.shape[0]
    print(f"\nVerifying orthogonality (tolerance={tolerance})...")

    for head in range(num_heads):
        r = rotation[head]
        identity = torch.matmul(r, r.T)
        expected = torch.eye(r.shape[0], dtype=r.dtype)

        max_error = (identity - expected).abs().max().item()
        if max_error > tolerance:
            print(f"  ✗ Head {head}: Max error = {max_error:.2e} (exceeds tolerance)")
            return False

    print(f"  ✓ All {num_heads} matrices are orthogonal")
    return True


def save_rotation_matrices(
    rotation: torch.Tensor,
    rotation_t: torch.Tensor,
    output_path: str,
    metadata: dict = None
):
    """Save rotation matrices to a file with metadata."""
    print(f"\nSaving to {output_path}...")

    if metadata is None:
        metadata = {}

    metadata.update({
        "num_heads": rotation.shape[0],
        "head_size": rotation.shape[1],
        "dtype": str(rotation.dtype),
        "shape": list(rotation.shape),
    })

    torch.save({
        "rotation": rotation,
        "rotation_t": rotation_t,
        "metadata": metadata,
    }, output_path)

    file_size_mb = torch.cuda.FloatStorage.from_file(output_path, size=0).size() * 4 / 1024 / 1024 if False else 0
    import os
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024

    print(f"✓ Saved successfully ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Pre-generate rotation matrices for RabitQ")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-size", type=int, default=128, help="Dimension of each head")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--verify", action="store_true", help="Verify orthogonality after generation")

    args = parser.parse_args()

    # Generate default output path if not specified
    if args.output is None:
        args.output = f"rotation_{args.num_heads}h_{args.head_size}d.pt"

    print("=" * 80)
    print("RabitQ Rotation Matrix Pre-generation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Num heads:  {args.num_heads}")
    print(f"  Head size:  {args.head_size}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {args.output}")
    print("=" * 80)

    # Generate matrices
    rotation, rotation_t = generate_rotation_matrices(args.num_heads, args.head_size, args.seed)

    # Verify orthogonality
    if args.verify:
        is_valid = verify_orthogonality(rotation)
        if not is_valid:
            print("\n✗ Verification failed!")
            return 1

    # Save to file
    metadata = {
        "seed": args.seed,
        "generation_method": "qr_decomposition",
    }
    save_rotation_matrices(rotation, rotation_t, args.output, metadata)

    print("\n✓ All done! Load this file using:")
    print(f'  data = torch.load("{args.output}")')
    print(f'  rotation = data["rotation"]')
    print(f'  rotation_t = data["rotation_t"]')

    return 0


if __name__ == "__main__":
    exit(main())
