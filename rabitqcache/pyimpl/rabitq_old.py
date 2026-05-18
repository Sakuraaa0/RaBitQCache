# RabitQ-based Sparse Attention Selector
# Adapted from faiss ReferenceRabitQ implementation

import torch
from typing import Optional
from torch import nn
import math


class RabitQState:
    """State manager for RabitQ across attention layers."""

    def __init__(self, quantize_interval: int = 512):
        self.quantize_interval = quantize_interval

        # Per-layer centroids (layer_id -> centroids dict)
        self.q_centroids = {}  # layer_id -> q_centroid tensor
        self.k_centroids = {}  # layer_id -> k_centroid tensor
        self.cq_dot_ck = {}    # layer_id -> dot product scalar

        # Per-layer rotation matrices
        self.rotation_matrices = {}  # layer_id -> rotation matrix P

        # Quantized data storage (per layer)
        # layer_id -> dict with quantization data
        self.quantized_data = {}

        # Track how many tokens have been quantized per layer
        self.num_quantized_tokens = {}

        # Track training status per layer
        self.is_trained = {}  # layer_id -> bool

    def rotation(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Apply rotation: x @ P"""
        P = self.rotation_matrices[layer_id]
        # x: [..., head_dim], P: [head_dim, head_dim]
        # Convert P to match x's dtype
        if P.dtype != x.dtype:
            P = P.to(x.dtype)
        return x @ P

    def inv_rotation(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Apply inverse rotation: x @ P.T"""
        P = self.rotation_matrices[layer_id]
        # x: [..., head_dim], P: [head_dim, head_dim]
        # Convert P to match x's dtype
        if P.dtype != x.dtype:
            P = P.to(x.dtype)
        return x @ P.T

    def train(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_id: int):
        """Train centroids using prefill stage data for a specific layer.

        Called during prefill stage (q_len > 1) with full Q and K.

        Args:
            query_states: [bs, num_heads, prefill_len, head_dim]
            key_states: [bs, num_heads, prefill_len, head_dim]
            layer_id: current layer index
        """
        if layer_id not in self.is_trained or not self.is_trained[layer_id]:
            # Compute centroids per head
            # For each head, compute the mean across all tokens (bs * prefill_len)
            # Output: [num_heads, head_dim]
            bs, num_heads, prefill_len, head_dim = query_states.shape
            device = query_states.device

            # Initialize random rotation matrix using QR decomposition (like Faiss)
            if layer_id not in self.rotation_matrices:
                # Create random rotation matrix P: [head_dim, head_dim]
                # Keep in float32 for better precision and avoid repeated conversions
                random_matrix = torch.randn(head_dim, head_dim, device=device, dtype=torch.float32)
                Q, R = torch.linalg.qr(random_matrix)
                self.rotation_matrices[layer_id] = Q  # Orthogonal rotation matrix in float32

            # Reshape to [bs * prefill_len, num_heads, head_dim] then compute mean over tokens
            q_reshaped = query_states.transpose(1, 2).reshape(bs * prefill_len, num_heads, head_dim)
            k_reshaped = key_states.transpose(1, 2).reshape(bs * prefill_len, num_heads, head_dim)

            # Compute per-head centroids: [num_heads, head_dim]
            self.q_centroids[layer_id] = q_reshaped.mean(dim=0)  # [num_heads, head_dim]
            self.k_centroids[layer_id] = k_reshaped.mean(dim=0)  # [num_heads, head_dim]

            # Compute per-head dot products: [num_heads]
            self.cq_dot_ck[layer_id] = (self.q_centroids[layer_id] * self.k_centroids[layer_id]).sum(dim=-1)

            self.is_trained[layer_id] = True

    def quantize_keys_batch(self, key_states: torch.Tensor, layer_id: int, start_idx: int, end_idx: int):
        """Quantize a batch of keys to 1-bit codes.

        Args:
            key_states: [bs, num_heads, seq_len, head_dim] - full key sequence
            layer_id: current layer index
            start_idx: starting token index for this batch
            end_idx: ending token index for this batch
        """
        if layer_id not in self.quantized_data:
            self.quantized_data[layer_id] = {
                'codes': [],  # List of quantized codes tensors
                'norms': [],  # List of norms tensors
                'o_Obar': [],  # List of <o, obar> dot products tensors
                'cq_dot_kr': [],  # List of c_q · k_r tensors
            }
            self.num_quantized_tokens[layer_id] = 0

        # Extract the batch to quantize
        keys_to_quantize = key_states[:, :, start_idx:end_idx, :]  # [bs, num_heads, batch_size, head_dim]

        bs, num_heads, batch_size, head_dim = keys_to_quantize.shape
        device = keys_to_quantize.device

        # Use layer-specific per-head centroids: [num_heads, head_dim]
        k_centroid = self.k_centroids[layer_id]  # [num_heads, head_dim]
        q_centroid = self.q_centroids[layer_id]  # [num_heads, head_dim]

        # Center keys: broadcast centroid across batch dimension
        # k_centroid: [num_heads, head_dim] -> [1, num_heads, 1, head_dim]
        Krc = keys_to_quantize - k_centroid.view(1, num_heads, 1, head_dim)

        # Compute norms before rotation
        K_norms = torch.sqrt((Krc ** 2).sum(dim=-1))  # [bs, num_heads, batch_size]

        # Normalize
        K = Krc / (K_norms.unsqueeze(-1) + 1e-8)

        # Apply inverse rotation before quantization (like Faiss: inv_rotation(Orc))
        Krc_rotated = self.inv_rotation(Krc, layer_id)  # [bs, num_heads, batch_size, head_dim]

        # Quantize to 1-bit (sign-based quantization) - on rotated data
        Xbarb = (Krc_rotated > 0).to(torch.int8)  # [bs, num_heads, batch_size, head_dim]

        # Reconstruct with forward rotation (like Faiss: rotation((2*Xbarb-1)/sqrt(d)))
        Kbar_unrotated = (2 * Xbarb.float() - 1) / math.sqrt(head_dim)
        Kbar = self.rotation(Kbar_unrotated, layer_id)  # Apply forward rotation

        # Compute <o, obar> with rotated reconstruction
        o_Kbar = (K * Kbar).sum(dim=-1)  # [bs, num_heads, batch_size]

        # Compute c_q · k_r (per-head dot product)
        # q_centroid: [num_heads, head_dim] -> [1, num_heads, 1, head_dim]
        cq_dot_kr = (keys_to_quantize * q_centroid.view(1, num_heads, 1, head_dim)).sum(dim=-1)  # [bs, num_heads, batch_size]

        # Store quantized data (store the rotated codes)
        self.quantized_data[layer_id]['codes'].append(Xbarb)
        self.quantized_data[layer_id]['norms'].append(K_norms)
        self.quantized_data[layer_id]['o_Obar'].append(o_Kbar)
        self.quantized_data[layer_id]['cq_dot_kr'].append(cq_dot_kr)

        self.num_quantized_tokens[layer_id] = end_idx

    def quantize_initial_keys(self, key_states: torch.Tensor, layer_id: int):
        """Quantize all keys from prefill stage, in batches of quantize_interval.

        Called during prefill stage with all prefill keys.

        Args:
            key_states: [bs, num_heads, total_seq_len, head_dim] - all keys from prefill stage
            layer_id: current layer index
        """
        total_seq_len = key_states.shape[2]

        # Quantize from [quantize_interval, total_seq_len) in batches
        # First quantize_interval tokens are always selected, no need to quantize
        start = self.quantize_interval

        while start < total_seq_len:
            end = min(start + self.quantize_interval, total_seq_len)
            self.quantize_keys_batch(key_states, layer_id, start, end)
            start = end

    def quantize_new_decode_tokens(self, key_states: torch.Tensor, layer_id: int):
        """Quantize new tokens accumulated during decode stage.

        Called during decode stage with all historical keys from cache.

        Args:
            key_states: [bs, num_heads, total_seq_len, head_dim] - all historical keys from cache
            layer_id: current layer index
        """
        if layer_id not in self.num_quantized_tokens:
            # Should not happen - quantize_initial_keys should be called first in prefill
            return

        total_seq_len = key_states.shape[2]
        num_already_quantized = self.num_quantized_tokens[layer_id]

        # Check if we have accumulated enough new tokens for a new batch
        if total_seq_len >= num_already_quantized + self.quantize_interval:
            # Quantize new complete batches
            start = num_already_quantized
            while start + self.quantize_interval <= total_seq_len:
                end = start + self.quantize_interval
                self.quantize_keys_batch(key_states, layer_id, start, end)
                start = end

    def distances(self, query_states: torch.Tensor, layer_id: int, quantized_seq_len: int):
        """Compute estimated inner products between query and quantized keys.

        Called during decode stage.

        Args:
            query_states: [bs, num_heads, 1, head_dim]
            layer_id: current layer index
            quantized_seq_len: number of quantized tokens to use

        Returns:
            estimated_inner_products: [bs, num_heads, 1, quantized_seq_len]
        """
        if layer_id not in self.quantized_data or len(self.quantized_data[layer_id]['codes']) == 0:
            return None

        bs, num_heads, q_len, head_dim = query_states.shape
        device = query_states.device

        # Use layer-specific per-head centroids: [num_heads, head_dim]
        q_centroid = self.q_centroids[layer_id]  # [num_heads, head_dim]
        k_centroid = self.k_centroids[layer_id]  # [num_heads, head_dim]
        cq_dot_ck = self.cq_dot_ck[layer_id]     # [num_heads]

        # Concatenate all quantized data
        codes = torch.cat(self.quantized_data[layer_id]['codes'], dim=2)  # [bs, num_heads, total_quantized, head_dim]
        norms = torch.cat(self.quantized_data[layer_id]['norms'], dim=2)  # [bs, num_heads, total_quantized]
        o_Kbar = torch.cat(self.quantized_data[layer_id]['o_Obar'], dim=2)  # [bs, num_heads, total_quantized]
        cq_dot_kr = torch.cat(self.quantized_data[layer_id]['cq_dot_kr'], dim=2)  # [bs, num_heads, total_quantized]

        # Only use the first quantized_seq_len tokens
        codes = codes[:, :, :quantized_seq_len, :]
        norms = norms[:, :, :quantized_seq_len]
        o_Kbar = o_Kbar[:, :, :quantized_seq_len]
        cq_dot_kr = cq_dot_kr[:, :, :quantized_seq_len]

        # Process query with per-head centroid
        # q_centroid: [num_heads, head_dim] -> [1, num_heads, 1, head_dim]
        Qrc = query_states - q_centroid.view(1, num_heads, 1, head_dim)

        # Apply inverse rotation to query (like Faiss: inv_rotation(Q))
        Qrc_rotated = self.inv_rotation(Qrc, layer_id)  # [bs, num_heads, 1, head_dim]

        # Use rotated query directly (no quantization)
        qbar = Qrc_rotated  # [bs, num_heads, 1, head_dim]

        # Reconstruct Kbar_unrotated from codes (before rotation)
        Kbar_unrotated = (2 * codes.float() - 1) / math.sqrt(head_dim)  # [bs, num_heads, quantized_seq_len, head_dim]

        # Compute dot product: qbar · Kbar_unrotated (both in rotated space)
        xbar_qbar = (qbar.unsqueeze(3) * Kbar_unrotated.unsqueeze(2)).sum(dim=-1)  # [bs, num_heads, 1, quantized_seq_len]

        # Debias the 1-bit reconstruction magnitude (Faiss / RabitQ: <obar, q> / <obar, o>)
        # This estimates <Qrc, Krc / ||Krc||> (query is NOT normalized here).
        q_dot_k_unit = xbar_qbar / (o_Kbar.unsqueeze(2) + 1e-8)

        # Compute q_r · c_k (per-head)
        # k_centroid: [num_heads, head_dim] -> [1, num_heads, 1, head_dim]
        qr_dot_ck = (query_states * k_centroid.view(1, num_heads, 1, head_dim)).sum(dim=-1, keepdim=True)  # [bs, num_heads, 1, 1]

        # Complete inner product formula (per-head cq_dot_ck)
        # <q_r, k_r> = <q_r-c_q, k_r-c_k> + q_r·c_k + c_q·k_r - c_q·c_k
        # and <q_r-c_q, k_r-c_k> = ||k_r-c_k|| * <Qrc, (k_r-c_k)/||k_r-c_k||>.
        # cq_dot_ck: [num_heads] -> [1, num_heads, 1, 1]
        inner_product = (norms.unsqueeze(2) * q_dot_k_unit
                        + qr_dot_ck
                        + cq_dot_kr.unsqueeze(2)
                        - cq_dot_ck.view(1, num_heads, 1, 1))

        return inner_product

    def reset_layer(self, layer_id: int):
        """Reset quantized data for a specific layer (called at start of new sequence)."""
        if layer_id in self.quantized_data:
            self.quantized_data[layer_id] = {
                'codes': [],
                'norms': [],
                'o_Obar': [],
                'cq_dot_kr': [],
            }
            self.num_quantized_tokens[layer_id] = 0


# Global state for RabitQ (shared across all layers)
_rabitq_state = None


def get_rabitq_state(quantize_interval: int = 512) -> RabitQState:
    """Get or create global RabitQ state."""
    global _rabitq_state
    if _rabitq_state is None:
        _rabitq_state = RabitQState(quantize_interval=quantize_interval)
    return _rabitq_state


def reset_rabitq_state():
    """Reset global RabitQ state (call at start of new sequence)."""
    global _rabitq_state
    _rabitq_state = None


def rabitq_selector(
    model,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    head_dim: int = 64,
    quantize_interval: int = 128,
    top_p: float = 0.95,
) -> torch.Tensor:
    """RabitQ-based sparse attention selector.

    Called ONLY in decode stage (q_len=1).
    Training and initial quantization happen in prefill stage via attention.py hooks.

    Selection strategy:
    1. First quantize_interval tokens: always select (initial context)
    2. Unquantized tokens: always select (most recent tokens)
    3. Quantized tokens in the middle: use RabitQ to estimate attention weights,
       then select top-p tokens (p=0.85)

    Args:
        model: attention module with layer_id
        query_states: [bs, num_heads, 1, head_dim] (decode stage, after repeat_kv for GQA)
        key_states: [bs, num_heads, seq_length, head_dim] (after repeat_kv for GQA, from cache)
        quantize_interval: quantize every N tokens
        top_p: probability mass to keep for quantized tokens

    Returns:
        mask: [bs, num_heads, 1, seq_length] boolean mask
    """
    bs, num_heads, q_len, head_dim = query_states.shape
    seq_length = key_states.shape[2]
    device = query_states.device
    layer_id = model.layer_id

    # Initialize mask (all False)
    mask = torch.zeros(bs, num_heads, q_len, seq_length, dtype=torch.bool, device=device)

    # Get RabitQ state (should already be trained in prefill)
    rabitq_state = get_rabitq_state(quantize_interval)

    if layer_id not in rabitq_state.is_trained or not rabitq_state.is_trained[layer_id]:
        # Should not happen - train() should be called in prefill stage
        # Fallback: select all tokens
        mask[:, :, :, :] = True
        return mask

    # Step 1: Always select first quantize_interval tokens
    first_chunk_end = min(quantize_interval, seq_length)
    mask[:, :, :, :first_chunk_end] = True

    if seq_length <= quantize_interval:
        # If sequence is too short, return mask with all tokens selected
        return mask

    # Check if we need to quantize new tokens accumulated during decode
    rabitq_state.quantize_new_decode_tokens(key_states, layer_id)

    # Calculate quantization boundaries
    num_quantized = rabitq_state.num_quantized_tokens.get(layer_id, quantize_interval)
    last_quantized_idx = num_quantized
    quantized_length = last_quantized_idx - quantize_interval

    # Step 2: Always select unquantized tokens at the end
    if last_quantized_idx < seq_length:
        mask[:, :, :, last_quantized_idx:] = True

    # Step 3: Handle quantized tokens in the middle using RabitQ estimation
    if quantized_length > 0:
        # Estimate attention weights for quantized region
        # estimated_weights = rabitq_state.distances(query_states, layer_id, quantized_length) # [bs, num_heads, 1, quantized_length]
        estimated_weights = rabitq_state.distances(query_states, layer_id, quantized_length) / math.sqrt(head_dim)  # [bs, num_heads, 1, quantized_length]

        if estimated_weights is not None:
            # Apply top-p selection on quantized region
            # Normalize to get probabilities
            quantized_probs = torch.softmax(estimated_weights, dim=-1)  # [bs, num_heads, 1, quantized_length]

            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(quantized_probs, dim=-1, descending=True)

            # Compute cumulative probabilities
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Find tokens that contribute to top-p mass
            top_p_mask = cumsum_probs <= top_p

            # Always include at least one token (the one with highest probability)
            top_p_mask[:, :, :, 0] = True

            # Map back to original indices in the quantized region
            quantized_mask = torch.zeros(bs, num_heads, q_len, quantized_length, dtype=torch.bool, device=device)
            quantized_mask.scatter_(dim=-1, index=sorted_indices, src=top_p_mask)

            # Place quantized mask into the full mask at correct position
            mask[:, :, :, quantize_interval:last_quantized_idx] = quantized_mask

    return mask
