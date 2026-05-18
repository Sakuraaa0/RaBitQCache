# RabitQ-based Sparse Attention Selector (No Centroid Variant)
# Ablation experiment: centroids for both Q and K are fixed to zero.
# This tests the effect of removing the centroid from the RabitQ algorithm.

import torch
from typing import Optional
import math

from .rabitq import RabitQState, get_rabitq_state


class RabitQNoCenterState(RabitQState):
    """RabitQ state with centroids fixed to zero (ablation experiment)."""

    def train(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_id: int):
        """Train rotation matrix but set centroids to zero.

        Args:
            query_states: [bs, num_heads, prefill_len, head_dim]
            key_states: [bs, num_heads, prefill_len, head_dim]
            layer_id: current layer index
        """
        if layer_id not in self.is_trained or not self.is_trained[layer_id]:
            bs, num_heads, prefill_len, head_dim = query_states.shape
            device = query_states.device

            # Initialize random rotation matrix (same as original)
            if layer_id not in self.rotation_matrices:
                random_matrix = torch.randn(head_dim, head_dim, device=device, dtype=torch.float32)
                Q, R = torch.linalg.qr(random_matrix)
                self.rotation_matrices[layer_id] = Q

            # Fix centroids to zero
            self.q_centroids[layer_id] = torch.zeros(num_heads, head_dim, device=device, dtype=query_states.dtype)
            self.k_centroids[layer_id] = torch.zeros(num_heads, head_dim, device=device, dtype=key_states.dtype)
            self.cq_dot_ck[layer_id] = torch.zeros(num_heads, device=device, dtype=query_states.dtype)

            self.is_trained[layer_id] = True


# Global state for RabitQ No-Center (shared across all layers)
_rabitq_no_center_state = None


def get_rabitq_no_center_state(quantize_interval: int = 512) -> RabitQNoCenterState:
    """Get or create global RabitQ no-center state."""
    global _rabitq_no_center_state
    if _rabitq_no_center_state is None:
        _rabitq_no_center_state = RabitQNoCenterState(quantize_interval=quantize_interval)
    return _rabitq_no_center_state


def reset_rabitq_no_center_state():
    """Reset global RabitQ no-center state."""
    global _rabitq_no_center_state
    _rabitq_no_center_state = None


def rabitq_no_center_selector(
    model,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    head_dim: int = 64,
    quantize_interval: int = 128,
    top_p: float = 0.95,
) -> torch.Tensor:
    """RabitQ-based sparse attention selector with centroids fixed to zero.

    Same logic as rabitq_selector but uses RabitQNoCenterState.

    Args:
        model: attention module with layer_id
        query_states: [bs, num_heads, 1, head_dim]
        key_states: [bs, num_heads, seq_length, head_dim]
        quantize_interval: quantize every N tokens
        top_p: probability mass to keep for quantized tokens

    Returns:
        mask: [bs, num_heads, 1, seq_length] boolean mask
    """
    bs, num_heads, q_len, head_dim = query_states.shape
    seq_length = key_states.shape[2]
    device = query_states.device
    layer_id = model.layer_id

    mask = torch.zeros(bs, num_heads, q_len, seq_length, dtype=torch.bool, device=device)

    rabitq_state = get_rabitq_no_center_state(quantize_interval)

    if layer_id not in rabitq_state.is_trained or not rabitq_state.is_trained[layer_id]:
        mask[:, :, :, :] = True
        return mask

    # Always select first quantize_interval tokens
    first_chunk_end = min(quantize_interval, seq_length)
    mask[:, :, :, :first_chunk_end] = True

    if seq_length <= quantize_interval:
        return mask

    # Quantize new decode tokens if needed
    rabitq_state.quantize_new_decode_tokens(key_states, layer_id)

    num_quantized = rabitq_state.num_quantized_tokens.get(layer_id, quantize_interval)
    last_quantized_idx = num_quantized
    quantized_length = last_quantized_idx - quantize_interval

    # Always select unquantized tokens at the end
    if last_quantized_idx < seq_length:
        mask[:, :, :, last_quantized_idx:] = True

    # Handle quantized tokens using RabitQ estimation (no centroid)
    if quantized_length > 0:
        estimated_weights = rabitq_state.distances(query_states, layer_id, quantized_length) / math.sqrt(head_dim)

        if estimated_weights is not None:
            quantized_probs = torch.softmax(estimated_weights, dim=-1)
            sorted_probs, sorted_indices = torch.sort(quantized_probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumsum_probs <= top_p
            top_p_mask[:, :, :, 0] = True

            quantized_mask = torch.zeros(bs, num_heads, q_len, quantized_length, dtype=torch.bool, device=device)
            quantized_mask.scatter_(dim=-1, index=sorted_indices, src=top_p_mask)
            mask[:, :, :, quantize_interval:last_quantized_idx] = quantized_mask

    return mask
