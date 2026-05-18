"""
RabitQ-based attention backend scaffolding.

This backend extends FlashInfer attention while adding RabitQ quantized KV cache
workflow hooks. It uses FlashInfer's BatchDecodeWithPagedKVCacheWrapper for efficient
paged attention computation.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import AttentionType
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend, FlashInferImpl, FlashInferMetadata,
    FlashInferMetadataBuilder)
from vllm.v1.attention.backends.rabitq_ops import (
    quantize_keys_fused, update_topk_mask, fused_query_preprocess)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import RabitQAttentionSpec

# for profiling
from vllm.utils.profiling import cprofile
import torch.cuda.nvtx as nvtx

logger = init_logger(__name__)

from vllm import _custom_ops as ops
rabitq_int4_binary_scores = ops.rabitq_int4_binary_scores
rabitq_int4_packed_binary_scores = ops.rabitq_int4_packed_binary_scores
top_p_mask = ops.top_p_mask


def _query_preprocess_impl(
    query: torch.Tensor,
    centroid_q: torch.Tensor,
    centroid_k: torch.Tensor,
    rotation_t: Optional[torch.Tensor],
    num_kv_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    B_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused query preprocessing: center + normalize + rotate + quantize.
    This function is designed to be compiled with torch.compile.
    """
    num_tokens = query.size(0)

    # Center and normalize query
    centered = query - centroid_q.unsqueeze(0)
    centered_view = centered.view(num_tokens, num_kv_heads, num_queries_per_kv, head_size)
    q_norms = torch.linalg.norm(centered_view, dim=-1, keepdim=True).clamp_min_(1e-6)
    normalized = centered_view / q_norms
    normalized_kv = normalized.mean(dim=2)  # [num_tokens, num_kv_heads, head_size]

    # Rotate query
    if rotation_t is not None:
        rotated_q = torch.einsum('tnh,nhd->tnd', normalized_kv, rotation_t)
    else:
        rotated_q = normalized_kv

    # Quantize query
    v_l = rotated_q.min(dim=-1, keepdim=True).values
    v_r = rotated_q.max(dim=-1, keepdim=True).values
    max_val = (1 << B_q) - 1
    Delta = ((v_r - v_l) / max_val).clamp_min_(1e-8)

    q_quantized = torch.round((rotated_q - v_l) / Delta).clamp(0, max_val)
    q_u = q_quantized.to(torch.uint8).contiguous()
    sum_q_u = q_quantized.sum(dim=-1, dtype=torch.float32)

    v_l_sq = v_l.squeeze(-1)
    Delta_sq = Delta.squeeze(-1)

    # Compute auxiliary values
    q_r = query.view(num_tokens, num_kv_heads, num_queries_per_kv, head_size).mean(dim=2)
    q_norm = q_norms.squeeze(-1).mean(dim=-1)
    qr_dot_ck = (q_r * centroid_k).sum(dim=-1)

    return q_u, Delta_sq, v_l_sq, sum_q_u, q_norm, qr_dot_ck


# Compiled version of query preprocessing
_compiled_query_preprocess: Optional[callable] = None


def get_compiled_query_preprocess():
    """Get or create the compiled query preprocessing function."""
    global _compiled_query_preprocess
    if _compiled_query_preprocess is None:
        _compiled_query_preprocess = torch.compile(
            _query_preprocess_impl,
            mode="reduce-overhead",
            fullgraph=False,
        )
    return _compiled_query_preprocess


def pack_binary_to_int32(x_b: torch.Tensor) -> torch.Tensor:
    """
    Pack binary tensor from [N, H, D] uint8 to [N, H, D/32] int32.

    Each 32 consecutive binary values are packed into one int32.
    This provides 8x memory bandwidth reduction.

    Args:
        x_b: Binary tensor with values {0, 1}, shape [N, H, D], dtype uint8

    Returns:
        Packed tensor, shape [N, H, D//32], dtype int32
    """
    N, H, D = x_b.shape
    x_b_reshaped = x_b.view(N, H, D // 32, 32)
    bit_positions = torch.arange(32, device=x_b.device, dtype=torch.int64)
    packed = (x_b_reshaped.to(torch.int64) << bit_positions).sum(dim=-1).to(torch.int32)
    return packed


# Flag to enable batched optimization (can be controlled via environment variable)
USE_BATCHED_RABITQ = os.environ.get("RABITQ_USE_BATCHED", "1") == "2"
if USE_BATCHED_RABITQ:
    logger.info("RabitQ: Batched optimization enabled (set RABITQ_USE_BATCHED=0 to disable)")

# =============================================================================
# Attention Timing Infrastructure
# =============================================================================
import time
import json
from collections import defaultdict
from dataclasses import asdict

RABITQ_TIMING_ENABLED = os.environ.get("RABITQ_TIMING_ENABLED", "0") == "1"
RABITQ_TIMING_OUTPUT_FILE = os.environ.get("RABITQ_TIMING_OUTPUT_FILE", "/tmp/rabitq_timing.json")

class AttentionTimingCollector:
    """Collects attention timing statistics in the worker process."""

    def __init__(self):
        self.forward_times = []  # List of (layer_name, time_ms)
        self.topk_mask_times = []  # List of (layer_name, time_ms) for _compute_topk_mask
        self.enabled = RABITQ_TIMING_ENABLED
        self._save_counter = 0
        self._save_interval = 32  # Save every N records
        self._last_file_check = 0  # For periodic file existence check
        self._has_saved_once = False  # Track if we've saved to file at least once

    def _check_and_reset_if_needed(self):
        """Check if output file was deleted (signal to reset) and reset if so."""
        import time as time_module
        current_time = time_module.time()
        # Only check every 0.5 seconds to avoid excessive I/O
        if current_time - self._last_file_check < 0.5:
            return
        self._last_file_check = current_time

        # Only reset if:
        # 1. We have previously saved the file (so it existed at some point)
        # 2. The file no longer exists (indicating test script deleted it)
        # 3. We have data to reset
        if self._has_saved_once and self.forward_times and not os.path.exists(RABITQ_TIMING_OUTPUT_FILE):
            self.reset()
            self._has_saved_once = False  # Reset this flag too

    def record_forward(self, layer_name: str, time_ms: float):
        """Record a forward pass timing."""
        if self.enabled:
            self._check_and_reset_if_needed()
            self.forward_times.append((layer_name, time_ms))
            self._save_counter += 1
            if self._save_counter >= self._save_interval:
                self.save_to_file()
                self._save_counter = 0

    def record_topk_mask(self, layer_name: str, time_ms: float):
        """Record a _compute_topk_mask timing."""
        if self.enabled:
            self.topk_mask_times.append((layer_name, time_ms))

    def reset(self):
        self.forward_times.clear()
        self.topk_mask_times.clear()
        self._save_counter = 0

    def get_summary(self) -> dict:
        """Get timing summary."""
        by_layer = defaultdict(list)
        for layer_name, t in self.forward_times:
            by_layer[layer_name].append(t)

        topk_mask_by_layer = defaultdict(list)
        for layer_name, t in self.topk_mask_times:
            topk_mask_by_layer[layer_name].append(t)

        total_forward_ms = sum(t for _, t in self.forward_times)
        total_topk_mask_ms = sum(t for _, t in self.topk_mask_times)

        return {
            "forward": {
                "total_ms": total_forward_ms,
                "count": len(self.forward_times),
                "by_layer": {k: {"total": sum(v), "count": len(v), "avg": sum(v)/len(v) if v else 0}
                            for k, v in by_layer.items()},
            },
            "topk_mask": {
                "total_ms": total_topk_mask_ms,
                "count": len(self.topk_mask_times),
                "ratio": total_topk_mask_ms / total_forward_ms if total_forward_ms > 0 else 0,
                "by_layer": {k: {"total": sum(v), "count": len(v), "avg": sum(v)/len(v) if v else 0}
                            for k, v in topk_mask_by_layer.items()},
            },
        }

    def save_to_file(self, filepath: str = None):
        """Save timing stats to file."""
        if filepath is None:
            filepath = RABITQ_TIMING_OUTPUT_FILE
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        self._has_saved_once = True  # Mark that we've saved at least once
        logger.info(f"Timing stats saved to {filepath}")

# Global timing collector instance
_timing_collector = AttentionTimingCollector()


def get_timing_collector() -> AttentionTimingCollector:
    """Get the global timing collector instance."""
    return _timing_collector


def enable_attention_timing():
    """Enable attention timing collection."""
    global _timing_collector
    _timing_collector.enabled = True
    logger.info("RabitQ attention timing enabled")


def disable_attention_timing():
    """Disable attention timing collection."""
    global _timing_collector
    _timing_collector.enabled = False


def reset_attention_timing():
    """Reset timing statistics."""
    global _timing_collector
    _timing_collector.reset()


def save_attention_timing(filepath: str = None):
    """Save timing stats to file."""
    global _timing_collector
    _timing_collector.save_to_file(filepath)


@dataclass
class PendingQuantizationData:
    """Data saved for deferred async quantization."""
    keys: torch.Tensor  # [num_tokens, num_kv_heads, head_size]
    req_ids: list[str]
    start_locs: list[int]
    device: torch.device


@dataclass
class RabitQLayerState:
    """
    Per-layer RabitQ state (rotation matrices, request states, etc.)
    Each attention layer has its own independent state.
    """
    rotation: Optional[torch.Tensor] = None
    rotation_t: Optional[torch.Tensor] = None
    request_states: dict[str, "RabitQRequestState"] = field(
        default_factory=dict)

    # Per-layer quantization stream for async prefill quantization
    quantize_stream: Optional[torch.cuda.Stream] = None

    # Pending quantization data: saved after attention, quantized async
    pending_quantization: Optional[PendingQuantizationData] = None


@dataclass
class RabitQRuntimeState:
    """
    RabitQ-specific buffers that persist across prefill/decode iterations.

    This stores global configuration and manages per-layer states.
    Each layer has its own RabitQLayerState to avoid mixing data.
    """
    b_q: Optional[int]
    topk: Optional[int] = None
    topp: Optional[float] = None
    batch_size: int = 512  # Batch size for quantization (configurable)

    # Per-layer states keyed by layer_name (e.g., "model.layers.5.self_attn")
    layer_states: dict[str, RabitQLayerState] = field(default_factory=dict)

    # Global rotation matrix cache to avoid redundant orthogonal initialization
    # Key: (num_kv_heads, head_size, device, dtype) -> (rotation, rotation_t)
    _rotation_cache: dict = field(default_factory=dict)

    # ========== Phase 1 Optimization: Pre-allocated buffers for _compute_topk_mask ==========
    topk_mask_unpacked_buffer: Optional[torch.Tensor] = None
    mask_offsets_buffer: Optional[torch.Tensor] = None
    max_total_kv_tokens: int = 0
    max_batch_size: int = 0
    topk_index_buffer: Optional[torch.Tensor] = None
    topk_index_buffer_capacity: int = 0


@dataclass
class RabitQRequestState:
    sum_q: Optional[torch.Tensor] = None
    sum_k: Optional[torch.Tensor] = None
    count: int = 0
    prepared: bool = False
    centroid_q: Optional[torch.Tensor] = None
    centroid_k: Optional[torch.Tensor] = None
    pending_keys: list[torch.Tensor] = field(default_factory=list)

    total_tokens: int = 0
    """Total number of logical tokens processed for this request."""

    # ========== GPU quantization metadata (pre-allocated for efficiency) ==========
    x_b_gpu: Optional[torch.Tensor] = None
    x_b_packed_gpu: Optional[torch.Tensor] = None  # Packed binary: [N, H, D/32] uint32
    sum_x_b_gpu: Optional[torch.Tensor] = None
    key_norms_gpu: Optional[torch.Tensor] = None
    k_bar_dot_k_gpu: Optional[torch.Tensor] = None
    cq_dot_kr_gpu: Optional[torch.Tensor] = None
    cq_dot_ck: Optional[torch.Tensor] = None

    metadata_capacity: int = 0
    """Pre-allocated capacity for metadata tensors"""

    metadata_write_index: int = 0
    """Current write position in metadata tensors"""

    # ========== Pending decode tokens (not yet quantized) ==========
    num_pending_decode_tokens: int = 0
    """Number of pending decode tokens (not yet quantized)"""


@dataclass
class RabitQMetadata(FlashInferMetadata):
    """Extended metadata for RabitQ attention."""
    # RabitQ-specific fields
    rabitq_b_q: Optional[int] = None
    rabitq_topk: Optional[int] = None
    rabitq_topp: Optional[float] = None
    req_ids: Optional[list[str]] = None
    query_start_loc_cpu: Optional[torch.Tensor] = None

    # RabitQ batch decode wrapper (created in build, used in forward)
    rabitq_decode_wrapper: Optional[object] = None


class RabitQAttentionBackend(FlashInferBackend):

    @staticmethod
    def get_name() -> str:
        return "RABITQ_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["RabitQAttentionImpl"]:
        return RabitQAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[RabitQMetadata]:
        return RabitQMetadata

    @staticmethod
    def get_builder_cls() -> type["RabitQAttentionMetadataBuilder"]:
        return RabitQAttentionMetadataBuilder


class RabitQAttentionMetadataBuilder(FlashInferMetadataBuilder):
    """
    Wrapper around FlashInfer's metadata builder that records whether
    RabitQ is active for the current KV cache spec.
    """

    def __init__(self, kv_cache_spec: RabitQAttentionSpec,
                 layer_names: list[str], vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._rabitq_enabled = isinstance(kv_cache_spec, RabitQAttentionSpec)
        self._rabitq_b_q = kv_cache_spec.rabitq_b_q
        self._rabitq_topk = kv_cache_spec.rabitq_topk
        self._rabitq_topp = kv_cache_spec.rabitq_topp

        # Initialize RabitQ decode wrapper (similar to FlashInfer's _decode_wrapper)
        self._rabitq_decode_wrapper = None
        self._rabitq_workspace_buffer = None

    def _get_rabitq_workspace_buffer(self):
        """Get or create workspace buffer for RabitQ decode wrapper."""
        if self._rabitq_workspace_buffer is None:
            self._rabitq_workspace_buffer = torch.empty(
                128 * 1024 * 1024,  # 128MB workspace
                dtype=torch.uint8,
                device=self.device
            )
        return self._rabitq_workspace_buffer

    def _get_rabitq_decode_wrapper(self, num_qo_heads: int, num_kv_heads: int,
                                    head_dim: int, dtype: torch.dtype):
        """Get or create RabitQ batch decode wrapper (similar to FlashInfer's _get_decode_wrapper)."""
        if self._rabitq_decode_wrapper is None:
            try:
                from .rabitq_flashinfer_ops import create_rabitq_batch_decode_wrapper
                self._rabitq_decode_wrapper = create_rabitq_batch_decode_wrapper(
                    head_dim=head_dim,
                    dtype=dtype,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    workspace_buffer=self._get_rabitq_workspace_buffer(),
                    max_num_seqs=512,
                    max_num_pages=200000,
                )
            except Exception as e:
                logger.warning_once(f"RabitQ: Failed to create decode wrapper: {e}")
                return None
        return self._rabitq_decode_wrapper

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> RabitQMetadata:
        # Build base FlashInfer metadata
        base_metadata = super().build(common_prefix_len, common_attn_metadata,
                                      fast_build)

        # Create RabitQ metadata with extended fields
        metadata = RabitQMetadata(
            num_actual_tokens=base_metadata.num_actual_tokens,
            q_data_type=base_metadata.q_data_type,
            slot_mapping=base_metadata.slot_mapping,
            max_q_len=base_metadata.max_q_len,
            max_seq_len=base_metadata.max_seq_len,
            seq_lens=base_metadata.seq_lens,
            block_table_tensor=base_metadata.block_table_tensor,
            prefill_use_trtllm=base_metadata.prefill_use_trtllm,
            decode_use_trtllm=base_metadata.decode_use_trtllm,
            num_decodes=base_metadata.num_decodes,
            num_decode_tokens=base_metadata.num_decode_tokens,
            num_prefills=base_metadata.num_prefills,
            num_prefill_tokens=base_metadata.num_prefill_tokens,
            use_cascade=base_metadata.use_cascade,
            prefill_wrapper=base_metadata.prefill_wrapper,
            decode_wrapper=base_metadata.decode_wrapper,
            cascade_wrapper=base_metadata.cascade_wrapper,
            qo_indptr_gpu=base_metadata.qo_indptr_gpu,
            paged_kv_indptr_gpu=base_metadata.paged_kv_indptr_gpu,
        )

        # Add RabitQ-specific metadata
        if self._rabitq_enabled and self._rabitq_b_q is not None:
            metadata.rabitq_b_q = self._rabitq_b_q
        if self._rabitq_enabled and self._rabitq_topk is not None:
            metadata.rabitq_topk = self._rabitq_topk
        if self._rabitq_enabled and self._rabitq_topp is not None:
            metadata.rabitq_topp = self._rabitq_topp
        if common_attn_metadata.req_ids is not None:
            metadata.req_ids = list(common_attn_metadata.req_ids)
        if common_attn_metadata.query_start_loc_cpu is not None:
            metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        # Create RabitQ decode wrapper if we have decode requests
        if self._rabitq_enabled and metadata.num_decodes > 0:
            wrapper = self._get_rabitq_decode_wrapper(
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=self.model_config.dtype,
            )
            if wrapper is not None:
                # Prepare buffers and plan (similar to FlashInfer's fast_plan_decode)
                indptr_buf, indices_buf, last_page_len_buf = wrapper.get_buffers()

                # Fill buffers from metadata (only for decode requests)
                num_reqs = metadata.num_decodes
                indptr_buf[0] = 0
                num_indices = 0

                seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]
                block_table_tensor = metadata.block_table_tensor[:num_reqs]

                for i in range(num_reqs):
                    total_tokens = int(seq_lens_cpu[i])
                    num_blocks_needed = (total_tokens + self.page_size - 1) // self.page_size

                    # Copy block indices
                    indices_buf[num_indices:num_indices + num_blocks_needed].copy_(
                        block_table_tensor[i, :num_blocks_needed], non_blocking=True
                    )
                    num_indices += num_blocks_needed

                    indptr_buf[i + 1] = num_indices
                    last_page_len_buf[i] = total_tokens % self.page_size or self.page_size

                # Plan wrapper
                wrapper.plan_direct(
                    batch_size=num_reqs,
                    num_indices=num_indices,
                    page_size=self.page_size,
                )

                metadata.rabitq_decode_wrapper = wrapper

        return metadata


class RabitQAttentionImpl(FlashInferImpl):
    """
    FlashInfer implementation extended with RabitQ bookkeeping.

    This implementation extends FlashInferImpl while adding RabitQ's
    quantized KV cache workflow for efficient approximate attention.
    """

    # Dict to store per-worker shared state across all layers
    # Key: (process_id, thread_id) to isolate different LLM instances
    _worker_rabitq_states: dict[tuple[int, int], RabitQRuntimeState] = {}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        sinks: Optional[torch.Tensor] = None,
        *,
        rabitq_b_q: Optional[int] = None,
        rabitq_topk: Optional[int] = None,
        rabitq_topp: Optional[float] = None,
        **extra_kwargs,
    ) -> None:
        # Use worker-specific shared state across all layers
        import os
        import threading
        worker_key = (os.getpid(), threading.get_ident())
        if worker_key not in RabitQAttentionImpl._worker_rabitq_states:
            RabitQAttentionImpl._worker_rabitq_states[worker_key] = RabitQRuntimeState(
                b_q=rabitq_b_q, topk=rabitq_topk or 128, topp=rabitq_topp
            )
        self._rabitq_state = RabitQAttentionImpl._worker_rabitq_states[worker_key]

        # Store layer name to track per-layer state (will be set in forward())
        self._layer_name: Optional[str] = None

        if rabitq_b_q is not None:
            self._validate_b_q(rabitq_b_q, head_size)

        # Initialize FlashInfer base class
        super().__init__(
            num_heads, head_size, scale, num_kv_heads,
            alibi_slopes, sliding_window, kv_cache_dtype,
            logits_soft_cap, attn_type, kv_sharing_target_layer_name, sinks
        )

        if rabitq_b_q is not None:
            logger.info_once(
                "RabitQ backend initialized (B_q=%d, topk=%d, head_size=%d, "
                "num_kv_heads=%d). Using FlashInfer for paged attention.",
                rabitq_b_q, rabitq_topk or 128, head_size, num_kv_heads
            )
            # Warmup Triton kernels to avoid JIT compilation during inference
            self._warmup_triton_kernels(head_size, num_kv_heads)

        self._emitted_fallback_warning = rabitq_b_q is None
        self._prefill_triton_available = True

    # Class-level flag to track if warmup has been done
    _triton_warmup_done: bool = False

    def _warmup_triton_kernels(self, head_size: int, num_kv_heads: int) -> None:
        """Warmup Triton kernels to trigger JIT compilation before inference."""
        # Only warmup once across all instances
        if RabitQAttentionImpl._triton_warmup_done:
            return
        RabitQAttentionImpl._triton_warmup_done = True

        try:
            device = torch.device("cuda")
            dtype = torch.bfloat16

            # Pre-load rotation matrices into global cache during initialization
            # This avoids file I/O in the async quantization path
            self._preload_rotation_matrices(device, dtype, num_kv_heads, head_size)

            # Create small dummy tensors for warmup
            num_tokens = 4  # Small batch for warmup
            num_heads = num_kv_heads * self.num_queries_per_kv
            keys = torch.randn(num_tokens, num_kv_heads, head_size, device=device, dtype=dtype)
            centroid_k = torch.randn(num_kv_heads, head_size, device=device, dtype=dtype)
            rotation = torch.randn(num_kv_heads, head_size, head_size, device=device, dtype=dtype)
            rotation_t = rotation.transpose(-1, -2).contiguous()
            centroid_q = torch.randn(num_kv_heads, head_size, device=device, dtype=dtype)

            # Trigger JIT compilation for quantize_keys_fused
            bits, _, _, _, _ = quantize_keys_fused(keys, centroid_k, rotation_t, rotation, centroid_q, head_size)

            # Also warmup pack_binary_to_int32
            _ = pack_binary_to_int32(bits)

            # Warmup fused_query_preprocess kernel
            query = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
            _ = fused_query_preprocess(
                query, centroid_q, centroid_k, rotation_t,
                self.num_queries_per_kv, 4  # B_q=4
            )

            torch.cuda.synchronize()

            logger.info("RabitQ: Triton kernels warmed up successfully")
        except Exception as e:
            logger.warning(f"RabitQ: Triton warmup failed: {e}")

    def _preload_rotation_matrices(self, device: torch.device, dtype: torch.dtype,
                                    num_kv_heads: int, head_size: int) -> None:
        """Pre-load rotation matrices into global cache during initialization.

        Since all layers use the same rotation matrices (from the same .pt file),
        we load them once at init time to avoid file I/O in async paths.
        This allows _process_pending_quantization_async to start immediately
        without waiting for data_ready_event.
        """
        cache_key = (num_kv_heads, head_size, str(device), str(dtype))

        if cache_key in self._rabitq_state._rotation_cache:
            logger.info("RabitQ: Rotation matrices already in cache")
            return

        pregenerated_file = Path(__file__).parent / f"rabitq_rotation_{num_kv_heads}h_{head_size}d.pt"

        if pregenerated_file.exists():
            try:
                logger.info(f"Pre-loading rotation matrices from {pregenerated_file.name}")
                data = torch.load(pregenerated_file, map_location="cpu", weights_only=True)
                rotation_f32 = data["rotation"].to(device=device, dtype=torch.float32)
                rotation_t_f32 = data["rotation_t"].to(device=device, dtype=torch.float32)
                rotation = rotation_f32.to(dtype)
                rotation_t = rotation_t_f32.to(dtype)
                self._rabitq_state._rotation_cache[cache_key] = (rotation, rotation_t)
                logger.info("✓ Pre-loaded rotation matrices into global cache")
                return
            except Exception as e:
                logger.warning(f"Failed to pre-load rotation matrices: {e}. Will generate at runtime.")

        # Generate rotation matrices if pre-generated file doesn't exist
        logger.info("Pre-generating rotation matrices during initialization")
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)

        rotation_f32 = torch.empty(num_kv_heads, head_size, head_size,
                                   device=device, dtype=torch.float32)

        for head in range(num_kv_heads):
            random_matrix = torch.randn(head_size, head_size, device=device, dtype=torch.float32)
            q, r = torch.linalg.qr(random_matrix)
            d = torch.diag(r)
            ph = d.sign()
            q *= ph
            rotation_f32[head] = q

        torch.set_rng_state(rng_state)

        rotation = rotation_f32.to(dtype)
        rotation_t = rotation.transpose(-1, -2).contiguous()
        self._rabitq_state._rotation_cache[cache_key] = (rotation, rotation_t)
        logger.info("✓ Pre-generated rotation matrices into global cache")

    def _get_layer_state(self, layer_name: str) -> RabitQLayerState:
        """Get or create the RabitQLayerState for a specific layer."""
        if layer_name not in self._rabitq_state.layer_states:
            self._rabitq_state.layer_states[layer_name] = RabitQLayerState()
        return self._rabitq_state.layer_states[layer_name]

    def _get_current_layer_state(self) -> RabitQLayerState:
        """Get the current layer's state using self._layer_name."""
        if self._layer_name is None:
            raise RuntimeError("layer_name not set. Call forward() first.")
        return self._get_layer_state(self._layer_name)

    def _get_topk_index_buffer(self, device: torch.device,
                               size: int) -> torch.Tensor:
        """Return a cached [0, size) arange tensor on the requested device."""
        if size <= 0:
            return torch.empty(0, dtype=torch.int32, device=device)

        buffer = self._rabitq_state.topk_index_buffer
        needs_new = (
            buffer is None or buffer.device != device
            or self._rabitq_state.topk_index_buffer_capacity < size
        )

        if needs_new:
            buffer = torch.arange(size, dtype=torch.int32, device=device)
            self._rabitq_state.topk_index_buffer = buffer
            self._rabitq_state.topk_index_buffer_capacity = size

        return buffer[:size]

    def _get_request_state(self, req_id: str,
                           device: torch.device,
                           dtype: torch.dtype,
                           layer_name: Optional[str] = None) -> RabitQRequestState:
        """Get or create request state with tensors using the specified dtype."""
        if layer_name is None:
            layer_name = self._layer_name
        if layer_name is None:
            raise RuntimeError("layer_name not provided and self._layer_name not set")

        layer_state = self._get_layer_state(layer_name)
        state = layer_state.request_states.get(req_id)
        if state is None:
            sum_q = torch.zeros(self.num_heads,
                                self.head_size,
                                device=device,
                                dtype=dtype)
            sum_k = torch.zeros(self.num_kv_heads,
                                self.head_size,
                                device=device,
                                dtype=dtype)
            state = RabitQRequestState(sum_q=sum_q, sum_k=sum_k)
            layer_state.request_states[req_id] = state
        return state

    def _update_prefill_stats(self, query: torch.Tensor, key: torch.Tensor,
                              value: Optional[torch.Tensor],
                              attn_metadata: RabitQMetadata) -> None:
        """Accumulate Q/K statistics during prefill phase."""
        req_ids = attn_metadata.req_ids
        if not req_ids:
            return

        query_start_loc_cpu = attn_metadata.query_start_loc_cpu
        if query_start_loc_cpu is None:
            return

        start_locs = getattr(attn_metadata, "_cached_query_start_locs_list", None)
        if start_locs is None:
            start_locs = query_start_loc_cpu.tolist()
            setattr(attn_metadata, "_cached_query_start_locs_list", start_locs)

        num_requests = len(req_ids)
        assert len(start_locs) >= num_requests + 1

        for idx, req_id in enumerate(req_ids):
            start = int(start_locs[idx])
            end = int(start_locs[idx + 1])
            if end <= start:
                continue

            q_slice = query[start:end]
            k_slice = key[start:end]
            state = self._get_request_state(req_id, q_slice.device, q_slice.dtype)
            if state.prepared:
                continue

            # Accumulate statistics
            state.sum_q.add_(q_slice.sum(dim=0))
            state.sum_k.add_(k_slice.sum(dim=0))

            num_prefill_tokens = end - start
            state.count += num_prefill_tokens
            state.total_tokens += num_prefill_tokens


    def _ensure_rotation(self, device: torch.device, dtype: torch.dtype, layer_state: RabitQLayerState) -> None:
        """Initialize rotation matrices with the target dtype to avoid repeated conversions."""
        if layer_state.rotation is not None:
            return

        cache_key = (self.num_kv_heads, self.head_size, str(device), str(dtype))

        if cache_key in self._rabitq_state._rotation_cache:
            layer_state.rotation, layer_state.rotation_t = self._rabitq_state._rotation_cache[cache_key]
            return

        pregenerated_file = Path(__file__).parent / f"rabitq_rotation_{self.num_kv_heads}h_{self.head_size}d.pt"

        if pregenerated_file.exists():
            try:
                logger.info(f"Loading pre-generated rotation matrices from {pregenerated_file.name}")
                data = torch.load(pregenerated_file, map_location="cpu", weights_only=True)
                rotation_f32 = data["rotation"].to(device=device, dtype=torch.float32)
                rotation_t_f32 = data["rotation_t"].to(device=device, dtype=torch.float32)
                layer_state.rotation = rotation_f32.to(dtype)
                layer_state.rotation_t = rotation_t_f32.to(dtype)
                self._rabitq_state._rotation_cache[cache_key] = (layer_state.rotation, layer_state.rotation_t)
                logger.info(f"✓ Loaded pre-generated rotation matrices (instant)")
                return
            except Exception as e:
                logger.warning(f"Failed to load pre-generated matrices: {e}. Falling back to runtime generation.")

        rng_state = torch.get_rng_state()
        torch.manual_seed(42)

        rotation_f32 = torch.empty(self.num_kv_heads,
                                   self.head_size,
                                   self.head_size,
                                   device=device,
                                   dtype=torch.float32)

        for head in range(self.num_kv_heads):
            random_matrix = torch.randn(self.head_size, self.head_size,
                                       device=device, dtype=torch.float32)
            q, r = torch.linalg.qr(random_matrix)
            d = torch.diag(r)
            ph = d.sign()
            q *= ph
            rotation_f32[head] = q

        torch.set_rng_state(rng_state)

        layer_state.rotation = rotation_f32.to(dtype)
        layer_state.rotation_t = layer_state.rotation.transpose(-1, -2).contiguous()
        self._rabitq_state._rotation_cache[cache_key] = (layer_state.rotation, layer_state.rotation_t)
        logger.info(f"✓ Generated rotation matrices at runtime")

    def _quantize_prefill_keys_immediately(
        self,
        key: torch.Tensor,
        attn_metadata: RabitQMetadata,
        device: torch.device,
        quantize_stream: torch.cuda.Stream,
        layer_state: RabitQLayerState,
    ) -> None:
        """
        Quantize prefill keys immediately, running in parallel with attention computation.
        This avoids storing two copies of keys and enables parallel execution.

        Args:
            key: Full key tensor [num_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata containing request information
            device: Device for computation
            quantize_stream: CUDA stream for parallel execution
            layer_state: Per-layer state (passed to avoid dict lookup)
        """
        req_ids = attn_metadata.req_ids
        if not req_ids:
            return

        query_start_loc_cpu = attn_metadata.query_start_loc_cpu
        if query_start_loc_cpu is None:
            return

        start_locs = getattr(attn_metadata, "_cached_query_start_locs_list", None)
        if start_locs is None:
            start_locs = query_start_loc_cpu.tolist()

        # Execute quantization in parallel stream
        with torch.cuda.stream(quantize_stream):
            for idx, req_id in enumerate(req_ids):
                state = layer_state.request_states.get(req_id)
                if state is None or state.prepared or state.count == 0:
                    continue

                # Extract key slice for this request
                start = int(start_locs[idx])
                end = int(start_locs[idx + 1])
                if end <= start:
                    continue

                keys = key[start:end]
                num_prefill_tokens = keys.shape[0]

                # Compute centroids
                state.centroid_q = state.sum_q / state.count
                state.centroid_k = state.sum_k / state.count

                # Precompute cq_dot_ck for decode phase
                cq_reshaped = state.centroid_q.view(
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    self.head_size
                ).mean(dim=1)
                state.cq_dot_ck = (cq_reshaped * state.centroid_k).sum(dim=-1)

                # Ensure rotation matrices are initialized
                self._ensure_rotation(device, keys.dtype, layer_state)

                # Compute cq_dot_kr for approximate scoring and attempt fused Triton kernel
                cq_reshaped_gpu = state.centroid_q.view(
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    self.head_size
                ).mean(dim=1).to(keys.dtype)

                centroid_k_typed = state.centroid_k.to(keys.dtype)
                rotation = layer_state.rotation
                rotation_t = layer_state.rotation_t

                bits = key_norms = k_bar_dot_k = sum_x_b = cq_dot_kr_gpu = None
                if (getattr(self, "_prefill_triton_available", True)
                        and keys.is_cuda):
                    try:
                        bits, key_norms, k_bar_dot_k, sum_x_b, cq_dot_kr_gpu = quantize_keys_fused(
                            keys,
                            centroid_k_typed,
                            rotation_t,
                            rotation,
                            cq_reshaped_gpu,
                            self.head_size,
                        )
                    except Exception as e:
                        self._prefill_triton_available = False
                        logger.warning_once(
                            "RabitQ: Triton prefill quantization failed (%s); "
                            "falling back to torch implementation.", e)
                        bits = None

                if bits is None:
                    centroid_k_expanded = centroid_k_typed.unsqueeze(0)
                    centered = keys - centroid_k_expanded
                    norms = torch.linalg.norm(centered, dim=-1, keepdim=True).clamp_min_(1e-6)
                    normalized = centered / norms

                    rotated_inv = torch.einsum('tnh,nhd->tnd', normalized, rotation_t)
                    bits = (rotated_inv >= 0).to(torch.uint8)
                    # Optimization: new x_bar = ±1 (no division by sqrt(D)); the 1/sqrt(D) cancels between numerator and denominator
                    x_bar = (2.0 * bits.to(keys.dtype) - 1.0)

                    # Optimization: compute P^T @ normalized first, then inner product with x_bar (±1)
                    # Avoids floating-point rotation on x_bar by exploiting the integer ±1 property
                    rotated_normalized = torch.einsum('tnh,nhd->tnd', normalized, rotation_t)
                    k_bar_dot_k = (x_bar * rotated_normalized).sum(dim=-1)
                    key_norms = norms.squeeze(-1)
                    sum_x_b = bits.sum(dim=-1, dtype=torch.float32)
                    cq_dot_kr_gpu = torch.einsum('thd,hd->th', keys, cq_reshaped_gpu)

                # Allocate metadata storage with capacity for future decode tokens
                estimated_decode_tokens = max(num_prefill_tokens * 10, 512)
                state.metadata_capacity = num_prefill_tokens + estimated_decode_tokens

                state.x_b_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads, self.head_size,
                    dtype=torch.uint8, device=device
                )
                # Allocate packed binary storage: [capacity, num_heads, head_size/32]
                state.x_b_packed_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads, self.head_size // 32,
                    dtype=torch.int32, device=device
                )
                state.sum_x_b_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads,
                    dtype=torch.float32, device=device
                )
                state.key_norms_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads,
                    dtype=keys.dtype, device=device
                )
                state.k_bar_dot_k_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads,
                    dtype=keys.dtype, device=device
                )
                state.cq_dot_kr_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads,
                    dtype=keys.dtype, device=device
                )

                # Store quantized results
                state.x_b_gpu[:num_prefill_tokens] = bits
                # Pack binary data for optimized kernel
                state.x_b_packed_gpu[:num_prefill_tokens] = pack_binary_to_int32(bits)
                state.sum_x_b_gpu[:num_prefill_tokens] = sum_x_b
                state.key_norms_gpu[:num_prefill_tokens] = key_norms
                state.k_bar_dot_k_gpu[:num_prefill_tokens] = k_bar_dot_k
                state.cq_dot_kr_gpu[:num_prefill_tokens] = cq_dot_kr_gpu

                state.metadata_write_index = num_prefill_tokens

                # Clear sum tensors (no longer needed)
                state.sum_q.zero_()
                state.sum_k.zero_()

                # Mark as prepared
                state.prepared = True

    def _process_pending_quantization_async(
        self,
        layer_state: RabitQLayerState,
    ) -> None:
        """
        Process pending quantization data in the quantize_stream.
        This runs AFTER attention computation, using data saved from prefill.

        Key insight: By the time this is called, all data dependencies are satisfied
        because attention computation has completed in the default stream.
        """
        pending = layer_state.pending_quantization
        if pending is None:
            return

        # Clear pending immediately to avoid reprocessing
        layer_state.pending_quantization = None

        keys = pending.keys
        req_ids = pending.req_ids
        start_locs = pending.start_locs
        device = pending.device

        # Process in quantize_stream (already synchronized with default stream via wait_stream)
        with torch.cuda.stream(layer_state.quantize_stream):

            for idx, req_id in enumerate(req_ids):
                state = layer_state.request_states.get(req_id)
                if state is None or state.prepared or state.count == 0:
                    continue

                # Extract key slice for this request
                start = int(start_locs[idx])
                end = int(start_locs[idx + 1])
                if end <= start:
                    continue

                req_keys = keys[start:end]
                num_prefill_tokens = req_keys.shape[0]

                # Compute centroids
                state.centroid_q = state.sum_q / state.count
                state.centroid_k = state.sum_k / state.count

                # Precompute cq_dot_ck for decode phase
                cq_reshaped = state.centroid_q.view(
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    self.head_size
                ).mean(dim=1)
                state.cq_dot_ck = (cq_reshaped * state.centroid_k).sum(dim=-1)

                # Rotation should already be loaded (done before entering this function)
                rotation = layer_state.rotation
                rotation_t = layer_state.rotation_t

                # Compute cq_dot_kr for approximate scoring
                cq_reshaped_gpu = state.centroid_q.view(
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    self.head_size
                ).mean(dim=1).to(req_keys.dtype)

                centroid_k_typed = state.centroid_k.to(req_keys.dtype)

                bits = key_norms = k_bar_dot_k = sum_x_b = cq_dot_kr_gpu = None
                if (getattr(self, "_prefill_triton_available", True)
                        and req_keys.is_cuda):
                    try:
                        bits, key_norms, k_bar_dot_k, sum_x_b, cq_dot_kr_gpu = quantize_keys_fused(
                            req_keys,
                            centroid_k_typed,
                            rotation_t,
                            rotation,
                            cq_reshaped_gpu,
                            self.head_size,
                        )
                    except Exception as e:
                        self._prefill_triton_available = False
                        logger.warning_once(
                            "RabitQ: Triton prefill quantization failed (%s); "
                            "falling back to torch implementation.", e)
                        bits = None

                if bits is None:
                    centroid_k_expanded = centroid_k_typed.unsqueeze(0)
                    centered = req_keys - centroid_k_expanded
                    norms = torch.linalg.norm(centered, dim=-1, keepdim=True).clamp_min_(1e-6)
                    normalized = centered / norms

                    rotated_inv = torch.einsum('tnh,nhd->tnd', normalized, rotation_t)
                    bits = (rotated_inv >= 0).to(torch.uint8)
                    x_bar = (2.0 * bits.to(req_keys.dtype) - 1.0)

                    rotated_normalized = torch.einsum('tnh,nhd->tnd', normalized, rotation_t)
                    k_bar_dot_k = (x_bar * rotated_normalized).sum(dim=-1)
                    key_norms = norms.squeeze(-1)
                    sum_x_b = bits.sum(dim=-1, dtype=torch.float32)
                    cq_dot_kr_gpu = torch.einsum('thd,hd->th', req_keys, cq_reshaped_gpu)

                # Allocate metadata storage with capacity for future decode tokens
                estimated_decode_tokens = max(num_prefill_tokens * 10, 512)
                state.metadata_capacity = num_prefill_tokens + estimated_decode_tokens

                state.x_b_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads, self.head_size,
                    dtype=torch.uint8, device=device
                )
                state.x_b_packed_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads, self.head_size // 32,
                    dtype=torch.int32, device=device
                )
                state.sum_x_b_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads,
                    dtype=torch.float32, device=device
                )
                state.key_norms_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads,
                    dtype=req_keys.dtype, device=device
                )
                state.k_bar_dot_k_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads,
                    dtype=req_keys.dtype, device=device
                )
                state.cq_dot_kr_gpu = torch.empty(
                    state.metadata_capacity, self.num_kv_heads,
                    dtype=req_keys.dtype, device=device
                )

                # Store quantized results
                state.x_b_gpu[:num_prefill_tokens] = bits
                state.x_b_packed_gpu[:num_prefill_tokens] = pack_binary_to_int32(bits)
                state.sum_x_b_gpu[:num_prefill_tokens] = sum_x_b
                state.key_norms_gpu[:num_prefill_tokens] = key_norms
                state.k_bar_dot_k_gpu[:num_prefill_tokens] = k_bar_dot_k
                state.cq_dot_kr_gpu[:num_prefill_tokens] = cq_dot_kr_gpu

                state.metadata_write_index = num_prefill_tokens

                # Clear sum tensors
                state.sum_q.zero_()
                state.sum_k.zero_()

                # Mark as prepared
                state.prepared = True


    # def _prepare_requests(self, req_ids: list[str],
    #                       device: torch.device) -> None:
    #     """
    #     Prepare requests for decode phase (legacy method for compatibility).
    #     Now mostly a no-op since quantization happens in prefill with parallel stream.
    #     """
    #     if not req_ids:
    #         return

    #     layer_state = self._get_current_layer_state()
    #     for req_id in req_ids:
    #         state = layer_state.request_states.get(req_id)
    #         if state is None or state.prepared:
    #             continue

    #         # If request is not prepared by now, something went wrong
    #         if state.count > 0 and not state.prepared:
    #             logger.warning(
    #                 f"Request {req_id} has count={state.count} but not prepared. "
    #                 f"This indicates quantization didn't complete in prefill phase."
    #             )

    #             # Emergency fallback: compute centroids at least
    #             if state.centroid_q is None and state.count > 0:
    #                 state.centroid_q = state.sum_q / state.count
    #                 state.centroid_k = state.sum_k / state.count

    #                 cq_reshaped = state.centroid_q.view(
    #                     self.num_kv_heads,
    #                     self.num_queries_per_kv,
    #                     self.head_size
    #                 ).mean(dim=1)
    #                 state.cq_dot_ck = (cq_reshaped * state.centroid_k).sum(dim=-1)

    #             state.prepared = True

    def _quantize_pending_keys(
        self,
        state: RabitQRequestState,
        req_index: int,
        kv_cache: torch.Tensor,
        attn_metadata: RabitQMetadata,
        device: torch.device,
        dtype: torch.dtype,
        layer_state: RabitQLayerState,
    ) -> None:
        """Batch quantize pending decode keys and update metadata."""
        num_pending = state.num_pending_decode_tokens
        if num_pending == 0:
            return

        pending_start_pos = state.total_tokens - num_pending
        pending_indices = torch.arange(pending_start_pos, state.total_tokens, device=device)

        keys_pending = self._extract_keys_from_paged_cache(
            req_index, pending_indices, kv_cache, attn_metadata, device, dtype
        )

        if keys_pending.shape[0] == 0:
            logger.warning("Failed to extract pending keys from cache")
            return

        self._ensure_rotation(device, dtype, layer_state)

        centroid_k_typed = state.centroid_k.to(dtype)
        rotation = layer_state.rotation
        rotation_t = layer_state.rotation_t

        cq_reshaped_gpu = state.centroid_q.view(
            self.num_kv_heads,
            self.num_queries_per_kv,
            self.head_size
        ).mean(dim=1).to(dtype)
        bits = key_norms = k_bar_dot_k = sum_x_b = cq_dot_kr_gpu = None
        if (getattr(self, "_prefill_triton_available", True)
                and keys_pending.is_cuda):
            try:
                bits, key_norms, k_bar_dot_k, sum_x_b, cq_dot_kr_gpu = quantize_keys_fused(
                    keys_pending,
                    centroid_k_typed,
                    rotation_t,
                    rotation,
                    cq_reshaped_gpu,
                    self.head_size,
                )
            except Exception as e:
                self._prefill_triton_available = False
                logger.warning_once(
                    "RabitQ: Triton decode quantization failed (%s); "
                    "falling back to torch implementation.", e)
                bits = None

        if bits is None:
            centered = keys_pending - centroid_k_typed.unsqueeze(0)
            norms = torch.linalg.norm(centered, dim=-1, keepdim=True).clamp_min_(1e-6)
            normalized = centered / norms

            rotated_inv = torch.einsum('tnh,nhd->tnd', normalized, rotation_t)
            bits = (rotated_inv >= 0).to(torch.uint8)
            # Optimization: new x_bar = ±1 (no division by sqrt(D)); the 1/sqrt(D) cancels between numerator and denominator
            x_bar = (2.0 * bits.to(dtype) - 1.0)

            # Optimization: compute P^T @ normalized first, then inner product with x_bar (±1)
            # Avoids floating-point rotation on x_bar by exploiting the integer ±1 property
            rotated_normalized = torch.einsum('tnh,nhd->tnd', normalized, rotation_t)
            k_bar_dot_k = (x_bar * rotated_normalized).sum(dim=-1)
            key_norms = norms.squeeze(-1)
            sum_x_b = bits.sum(dim=-1, dtype=torch.float32)
            cq_dot_kr_gpu = torch.einsum('thd,hd->th', keys_pending, cq_reshaped_gpu)

        new_write_index = state.metadata_write_index + num_pending
        if new_write_index > state.metadata_capacity:
            new_capacity = max(new_write_index * 2, state.metadata_capacity + 512)
            logger.info(f"Expanding metadata capacity from {state.metadata_capacity} to {new_capacity}")

            x_b_new = torch.empty(new_capacity, self.num_kv_heads, self.head_size,
                                  dtype=torch.uint8, device=device)
            x_b_packed_new = torch.empty(new_capacity, self.num_kv_heads, self.head_size // 32,
                                         dtype=torch.int32, device=device)
            sum_x_b_new = torch.empty(new_capacity, self.num_kv_heads,
                                      dtype=torch.float32, device=device)
            key_norms_new = torch.empty(new_capacity, self.num_kv_heads,
                                        dtype=dtype, device=device)
            k_bar_dot_k_new = torch.empty(new_capacity, self.num_kv_heads,
                                          dtype=dtype, device=device)
            cq_dot_kr_new = torch.empty(new_capacity, self.num_kv_heads,
                                        dtype=dtype, device=device)

            x_b_new[:state.metadata_write_index] = state.x_b_gpu[:state.metadata_write_index]
            x_b_packed_new[:state.metadata_write_index] = state.x_b_packed_gpu[:state.metadata_write_index]
            sum_x_b_new[:state.metadata_write_index] = state.sum_x_b_gpu[:state.metadata_write_index]
            key_norms_new[:state.metadata_write_index] = state.key_norms_gpu[:state.metadata_write_index]
            k_bar_dot_k_new[:state.metadata_write_index] = state.k_bar_dot_k_gpu[:state.metadata_write_index]
            cq_dot_kr_new[:state.metadata_write_index] = state.cq_dot_kr_gpu[:state.metadata_write_index]

            state.x_b_gpu = x_b_new
            state.x_b_packed_gpu = x_b_packed_new
            state.sum_x_b_gpu = sum_x_b_new
            state.key_norms_gpu = key_norms_new
            state.k_bar_dot_k_gpu = k_bar_dot_k_new
            state.cq_dot_kr_gpu = cq_dot_kr_new
            state.metadata_capacity = new_capacity

        write_start = state.metadata_write_index
        write_end = write_start + num_pending

        state.x_b_gpu[write_start:write_end] = bits
        state.x_b_packed_gpu[write_start:write_end] = pack_binary_to_int32(bits)
        state.sum_x_b_gpu[write_start:write_end] = sum_x_b
        state.key_norms_gpu[write_start:write_end] = key_norms
        state.k_bar_dot_k_gpu[write_start:write_end] = k_bar_dot_k
        state.cq_dot_kr_gpu[write_start:write_end] = cq_dot_kr_gpu

        state.metadata_write_index = write_end
        state.num_pending_decode_tokens = 0

        logger.info(f"RabitQ: Batch quantized {num_pending} decode tokens, metadata_write_index now {state.metadata_write_index}")


    def _append_decode_kv(self, attn_metadata: RabitQMetadata, layer_state: RabitQLayerState) -> None:
        """Track decode bookkeeping when new KV blocks are appended."""
        req_ids = attn_metadata.req_ids
        if not req_ids:
            return

        start_locs_cpu = attn_metadata.query_start_loc_cpu
        if start_locs_cpu is None:
            return

        start_locs = getattr(attn_metadata, "_cached_query_start_locs_list", None)
        if start_locs is None:
            start_locs = start_locs_cpu.tolist()
            setattr(attn_metadata, "_cached_query_start_locs_list", start_locs)

        for idx, req_id in enumerate(req_ids):
            start = int(start_locs[idx])
            end = int(start_locs[idx + 1])
            num_new_tokens = end - start

            if num_new_tokens <= 0:
                continue

            state = layer_state.request_states.get(req_id)
            if state is None or not state.prepared:
                continue

            state.total_tokens += num_new_tokens
            state.num_pending_decode_tokens += num_new_tokens

    def _extract_keys_from_paged_cache(
        self,
        req_index: int,
        token_indices: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RabitQMetadata,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Extract key rows from paged cache using block_table-based lookup."""
        if token_indices.numel() == 0:
            return torch.empty(0, self.num_kv_heads, self.head_size, device=device, dtype=dtype)

        if attn_metadata is None or attn_metadata.block_table_tensor is None:
            logger.warning("RabitQ: block_table missing in attn_metadata")
            return torch.empty(0, self.num_kv_heads, self.head_size, device=device, dtype=dtype)

        # FlashInfer layout: [num_blocks, 2, block_size, num_kv_heads, head_size]
        key_cache = kv_cache[:, 0, :, :, :]
        block_size = key_cache.shape[1]

        block_table = attn_metadata.block_table_tensor[req_index]

        # Compute block indices and offsets
        block_indices = token_indices // block_size
        block_offsets = token_indices % block_size

        # Get physical block numbers
        physical_blocks = block_table[block_indices.long()]

        # Extract keys
        keys = key_cache[physical_blocks.long(), block_offsets.long(), :, :]
        if keys.dtype != dtype:
            keys = keys.to(dtype)

        return keys

    def _compute_topk_mask(
        self,
        batch_info: list[dict],
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RabitQMetadata,
        device: torch.device,
        layer_state: RabitQLayerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top-k mask for all decode requests in the batch.

        This function:
        1. Computes approximate attention scores using RabitQ quantization
        2. Selects top-k tokens per request
        3. Creates a packed bitmap mask for FlashInfer

        Returns:
            topk_mask: Packed bitmap [total_mask_bytes] (uint8)
            mask_offsets: Bit offset for each request [batch_size] (uint32)
        """
        topk = self._rabitq_state.topk
        dtype = query.dtype

        # Phase 1: Collect batch info and trigger quantization if needed
        for info in batch_info:
            state = info['state']
            num_pending = state.num_pending_decode_tokens
            if num_pending >= self._rabitq_state.batch_size:
                self._quantize_pending_keys(
                    state, info['req_idx'], kv_cache, attn_metadata, device, dtype, layer_state
                )
        
        batch_size = len(batch_info)
        if batch_size == 1:
            return self._compute_topk_mask_single(
                batch_info[0],
                query,
                device,
                dtype,
                topk,
                layer_state,
            )

        # Phase 2: Compute total KV tokens and mask offsets
        # ========== Phase 1 Optimization: Reuse pre-allocated buffers ==========
        total_kv_tokens = 0
        mask_offsets_list = []
        for info in batch_info:
            mask_offsets_list.append(total_kv_tokens)
            total_kv_tokens += info['state'].total_tokens

        # Allocate or reuse mask_offsets buffer
        if (self._rabitq_state.mask_offsets_buffer is None or
            batch_size > self._rabitq_state.max_batch_size):
            # Need to allocate a larger buffer
            self._rabitq_state.mask_offsets_buffer = torch.empty(
                batch_size, dtype=torch.uint32, device=device
            )
            self._rabitq_state.max_batch_size = batch_size

        # Fill mask_offsets from list (reusing buffer)
        mask_offsets = self._rabitq_state.mask_offsets_buffer[:batch_size]
        mask_offsets_src = mask_offsets.new_tensor(mask_offsets_list)
        mask_offsets.copy_(mask_offsets_src)

        # Allocate or reuse topk_mask_unpacked buffer
        if (self._rabitq_state.topk_mask_unpacked_buffer is None or
            total_kv_tokens > self._rabitq_state.max_total_kv_tokens):
            # Need to allocate a larger buffer
            self._rabitq_state.topk_mask_unpacked_buffer = torch.zeros(
                total_kv_tokens, dtype=torch.uint8, device=device
            )
            self._rabitq_state.max_total_kv_tokens = total_kv_tokens

        topk_mask_unpacked = self._rabitq_state.topk_mask_unpacked_buffer[:total_kv_tokens]

        # Phase 3: For each request, compute approx scores and select top-k
        # ========== Phase 2 Optimization: Vectorized TopK Selection ==========
        query_idx = 0
        for info in batch_info:
            state = info['state']
            offset = mask_offsets_list[info['req_idx']]
            num_tokens = info['num_tokens']
            num_quantized = state.metadata_write_index
            num_pending = state.num_pending_decode_tokens
            num_total_kv = state.total_tokens

            request_slice = topk_mask_unpacked[offset:offset + num_total_kv]

            # Get query slice for this request
            q_slice = query[query_idx:query_idx + num_tokens]
            query_idx += num_tokens

            # Compute approximate scores for quantized tokens
            idx_flat: Optional[torch.Tensor] = None
            topk_mask_direct: Optional[torch.Tensor] = None
            if num_quantized > 0 and state.x_b_gpu is not None:
                topk_limit = topk if topk is not None else num_quantized
                result = self._compute_quantized_topk_indices(
                    state,
                    q_slice,
                    dtype,
                    layer_state,
                    topk_limit,
                )
                if result is not None:
                    # Check if result is a mask (bool) or indices (int32)
                    if result.dtype == torch.bool:
                        # top-p returns mask directly: [num_tokens, num_kv_heads, num_quantized]
                        # Reduce across tokens and heads: select if ANY head/token selects it
                        topk_mask_direct = result.any(dim=0).any(dim=0)  # [num_quantized]
                    else:
                        # top-k returns indices
                        idx_flat = result.reshape(-1).to(torch.int32).contiguous()

            pending_start = num_total_kv - num_pending

            # Use direct mask if available (top-p), otherwise use indices (top-k)
            if topk_mask_direct is not None:
                # Direct mask assignment: much faster than scatter
                request_slice[:num_quantized].copy_(topk_mask_direct.to(torch.uint8))
                # Set pending tokens to 1
                if pending_start < num_total_kv:
                    request_slice[pending_start:num_total_kv].fill_(1)
            else:
                update_topk_mask(
                    request_slice,
                    idx_flat,
                    num_quantized,
                    pending_start,
                    num_total_kv,
                )

        return topk_mask_unpacked, mask_offsets
    
    def _compute_approx_scores_for_request(
        self,
        state: RabitQRequestState,
        query: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        layer_state: RabitQLayerState,
    ) -> Optional[torch.Tensor]:
        """
        Compute approximate attention scores for a single request.

        Args:
            state: Request state with quantization metadata
            query: Query tensor [num_tokens, num_heads, head_size]
            device: Device for computation
            dtype: Data type for computation
            layer_state: Per-layer state (passed to avoid dict lookup)

        Returns:
            approx_scores: [num_tokens, num_quantized_tokens, num_kv_heads]
        """
        num_quantized = state.metadata_write_index
        if num_quantized == 0 or state.x_b_gpu is None:
            return None

        B_q = self._rabitq_state.b_q
        if B_q is None or B_q <= 0:
            return None

        num_tokens = query.size(0)

        # Center and normalize query
        centroid_q = state.centroid_q.to(dtype)
        centered = query - centroid_q.unsqueeze(0)
        centered_view = centered.view(num_tokens, self.num_kv_heads,
                                      self.num_queries_per_kv, self.head_size)
        q_norms = torch.linalg.norm(centered_view, dim=-1, keepdim=True).clamp_min_(1e-6)
        normalized = centered_view / q_norms
        normalized_kv = normalized.mean(dim=2)  # [num_tokens, num_kv_heads, head_size]

        # Rotate query
        rotation_t = layer_state.rotation_t
        if rotation_t is not None:
            rotated_q = torch.einsum('tnh,nhd->tnd', normalized_kv, rotation_t)
        else:
            rotated_q = normalized_kv

        # Quantize query
        v_l = rotated_q.min(dim=-1, keepdim=True).values
        v_r = rotated_q.max(dim=-1, keepdim=True).values
        Delta = ((v_r - v_l) / (2**B_q - 1)).clamp_min_(1e-8)

        q_quantized = torch.round((rotated_q - v_l) / Delta).clamp(0, 2**B_q - 1)
        q_u = q_quantized.to(torch.uint8)
        sum_q_u = q_u.sum(dim=-1, dtype=torch.float32)

        # Compute inner product with stored quantized keys
        x_b = state.x_b_gpu[:num_quantized]
        inner_product = torch.einsum('thd,shd->tsh', q_u.float(), x_b.float())

        # Compute approximate <x_bar, q>
        sqrt_D = math.sqrt(self.head_size)
        v_l_sq = v_l.squeeze(-1)
        Delta_sq = Delta.squeeze(-1)
        sum_x_b = state.sum_x_b_gpu[:num_quantized]

        term1 = (2.0 * Delta_sq.unsqueeze(1) / sqrt_D) * inner_product
        term2 = (2.0 * v_l_sq.unsqueeze(1) / sqrt_D) * sum_x_b.unsqueeze(0)
        term3 = -(Delta_sq.unsqueeze(1) / sqrt_D) * sum_q_u.unsqueeze(1)
        term4 = -sqrt_D * v_l_sq.unsqueeze(1)

        x_bar_dot_q = term1 + term2 + term3 + term4

        x_bar = (2.0 * x_b.to(torch.bfloat16) - 1.0)

        # Compute exact dot product for debugging comparison
        x_bar_dot_q2 = torch.einsum('shd,thd->tsh', x_bar, rotated_q.to(x_bar.dtype))

        # Compute full approximate score
        k_bar_dot_k = state.k_bar_dot_k_gpu[:num_quantized].clamp_min_(1e-6)
        approx_kq = x_bar_dot_q / k_bar_dot_k.unsqueeze(0)

        key_norms = state.key_norms_gpu[:num_quantized]
        cq_dot_kr = state.cq_dot_kr_gpu[:num_quantized]
        cq_dot_ck = state.cq_dot_ck

        # q_r: original query averaged over GQA groups
        q_r = query.view(num_tokens, self.num_kv_heads,
                        self.num_queries_per_kv, self.head_size).mean(dim=2)
        q_norm = q_norms.squeeze(-1).mean(dim=-1)  # [num_tokens, num_kv_heads]

        centroid_k = state.centroid_k.to(dtype)
        qr_dot_ck = (q_r * centroid_k).sum(dim=-1)

        approx_scores = (
            q_norm.unsqueeze(1) * key_norms.unsqueeze(0) * approx_kq +
            qr_dot_ck.unsqueeze(1) +
            cq_dot_kr.unsqueeze(0) -
            cq_dot_ck.unsqueeze(0).unsqueeze(0)
        )

        return approx_scores
    
    def _compute_approx_scores_for_request_l2(
        self,
        state: RabitQRequestState,
        query: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        layer_state: RabitQLayerState,
    ) -> Optional[torch.Tensor]:
        """
        Compute approximate attention scores for a single request.

        Args:
            state: Request state with quantization metadata
            query: Query tensor [num_tokens, num_heads, head_size]
            device: Device for computation
            dtype: Data type for computation
            layer_state: Per-layer state (passed to avoid dict lookup)

        Returns:
            approx_scores: [num_tokens, num_quantized_tokens, num_kv_heads]
        """
        num_quantized = state.metadata_write_index
        if num_quantized == 0 or state.x_b_gpu is None:
            return None

        B_q = self._rabitq_state.b_q
        if B_q is None or B_q <= 0:
            return None

        num_tokens = query.size(0)

        # Center and normalize query
        centroid_q = state.centroid_q.to(dtype)
        centered = query - centroid_q.unsqueeze(0)
        centered_view = centered.view(num_tokens, self.num_kv_heads,
                                      self.num_queries_per_kv, self.head_size)
        q_norms = torch.linalg.norm(centered_view, dim=-1, keepdim=True).clamp_min_(1e-6)
        normalized = centered_view / q_norms
        normalized_kv = normalized.mean(dim=2)  # [num_tokens, num_kv_heads, head_size]

        # Rotate query
        rotation_t = layer_state.rotation_t
        if rotation_t is not None:
            rotated_q = torch.einsum('tnh,nhd->tnd', normalized_kv, rotation_t)
        else:
            rotated_q = normalized_kv

        # Quantize query
        v_l = rotated_q.min(dim=-1, keepdim=True).values
        v_r = rotated_q.max(dim=-1, keepdim=True).values
        Delta = ((v_r - v_l) / (2**B_q - 1)).clamp_min_(1e-8)

        q_quantized = torch.round((rotated_q - v_l) / Delta).clamp(0, 2**B_q - 1)
        q_u = q_quantized.to(torch.uint8)
        sum_q_u = q_u.sum(dim=-1, dtype=torch.float32)

        # Compute inner product with stored quantized keys
        x_b = state.x_b_gpu[:num_quantized]
        inner_product = torch.einsum('thd,shd->tsh', q_u.float(), x_b.float())

        # Compute approximate <x_bar, q>
        sqrt_D = math.sqrt(self.head_size)
        v_l_sq = v_l.squeeze(-1)
        Delta_sq = Delta.squeeze(-1)
        sum_x_b = state.sum_x_b_gpu[:num_quantized]

        term1 = (2.0 * Delta_sq.unsqueeze(1) ) * inner_product
        term2 = (2.0 * v_l_sq.unsqueeze(1) ) * sum_x_b.unsqueeze(0)
        term3 = -(Delta_sq.unsqueeze(1) ) * sum_q_u.unsqueeze(1)
        term4 = -self.head_size * v_l_sq.unsqueeze(1)

        x_bar_dot_q = term1 + term2 + term3 + term4

        # Optimization: new x_bar = ±1 (no division by sqrt(D)) - for debugging comparison
        x_bar = (2.0 * x_b.to(torch.bfloat16) - 1.0)

        # Compute exact dot product for debugging comparison
        x_bar_dot_q2 = torch.einsum('shd,thd->tsh', x_bar, rotated_q.to(x_bar.dtype))

        # Compute full approximate score
        k_bar_dot_k = state.k_bar_dot_k_gpu[:num_quantized].clamp_min_(1e-6)
        approx_kq = x_bar_dot_q / k_bar_dot_k.unsqueeze(0)

        key_norms = state.key_norms_gpu[:num_quantized]
        cq_dot_kr = state.cq_dot_kr_gpu[:num_quantized]
        cq_dot_ck = state.cq_dot_ck

        # q_r: original query averaged over GQA groups
        q_r = query.view(num_tokens, self.num_kv_heads,
                        self.num_queries_per_kv, self.head_size).mean(dim=2)
        q_norm = q_norms.squeeze(-1).mean(dim=-1)  # [num_tokens, num_kv_heads]

        centroid_k = state.centroid_k.to(dtype)
        centroid_q = state.centroid_q.to(dtype)

        # Compute dot products with centroids
        qr_dot_ck = (q_r * centroid_k).sum(dim=-1)  # [num_tokens, num_kv_heads]
        qr_dot_cq = (q_r * centroid_q).sum(dim=-1)  # [num_tokens, num_kv_heads]

        # Compute centroid norms squared
        cq_norm_sq = (centroid_q ** 2).sum(dim=-1)  # [num_kv_heads]
        ck_norm_sq = (centroid_k ** 2).sum(dim=-1)  # [num_kv_heads]

        # Compute L2 distance squared: ||q-k||²
        # ||q-k||² = ||q-c_q||² + ||k-c_k||² - ||c_q||² + ||c_k||²
        #           - 2·||q-c_q||·||k-c_k||·⟨q_c, k_c⟩
        #           + 2·⟨q, c_q⟩ - 2·⟨q, c_k⟩
        #           - 2·⟨c_q, k⟩ + 2·⟨c_q, c_k⟩
        l2_dist_sq = (
            q_norm.unsqueeze(1) ** 2  # [num_tokens, 1, num_kv_heads]
            + key_norms.unsqueeze(0) ** 2  # [1, num_quantized, num_kv_heads]
            - cq_norm_sq.unsqueeze(0).unsqueeze(0)  # [1, 1, num_kv_heads]
            + ck_norm_sq.unsqueeze(0).unsqueeze(0)  # [1, 1, num_kv_heads]
            - 2 * q_norm.unsqueeze(1) * key_norms.unsqueeze(0) * approx_kq
            + 2 * qr_dot_cq.unsqueeze(1)  # [num_tokens, 1, num_kv_heads]
            - 2 * qr_dot_ck.unsqueeze(1)  # [num_tokens, 1, num_kv_heads]
            - 2 * cq_dot_kr.unsqueeze(0)  # [1, num_quantized, num_kv_heads]
            + 2 * cq_dot_ck.unsqueeze(0).unsqueeze(0)  # [1, 1, num_kv_heads]
        )

        # Return negative L2 distance squared (so smaller distance = higher score)
        approx_scores = -l2_dist_sq

        return approx_scores


    def _compute_topk_mask_single(
        self,
        info: dict,
        query: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        topk: int,
        layer_state: RabitQLayerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fast path for batch_size == 1 to avoid per-request Python overhead."""
        state = info['state']
        num_tokens = info['num_tokens']
        num_quantized = state.metadata_write_index
        num_pending = state.num_pending_decode_tokens
        total_kv_tokens = state.total_tokens

        if (self._rabitq_state.mask_offsets_buffer is None or
            self._rabitq_state.max_batch_size < 1):
            self._rabitq_state.mask_offsets_buffer = torch.tensor(
                [0], dtype=torch.uint32, device=device)
            self._rabitq_state.max_batch_size = 1
        mask_offsets = self._rabitq_state.mask_offsets_buffer[:1]

        if (self._rabitq_state.topk_mask_unpacked_buffer is None or
            total_kv_tokens > self._rabitq_state.max_total_kv_tokens):
            self._rabitq_state.topk_mask_unpacked_buffer = torch.zeros(
                total_kv_tokens, dtype=torch.uint8, device=device)
            self._rabitq_state.max_total_kv_tokens = total_kv_tokens
        topk_mask_unpacked = self._rabitq_state.topk_mask_unpacked_buffer[:total_kv_tokens]

        # Avoid extra PyTorch slicing in the fast path. If approximate scores
        # are re-enabled, materialize query slices lazily inside the block.
        idx_flat: Optional[torch.Tensor] = None
        topk_mask_direct: Optional[torch.Tensor] = None
        if num_quantized > 0 and state.x_b_gpu is not None:
            topk_limit = topk if topk is not None else num_quantized
            result = self._compute_quantized_topk_indices(
                state,
                query,
                dtype,
                layer_state,
                topk_limit,
            )

            if result is not None:
                # Check if result is a mask (bool) or indices (int32)
                if result.dtype == torch.bool:
                    # top-p returns mask directly: [num_tokens, num_kv_heads, num_quantized]
                    # Reduce across tokens and heads: select if ANY head/token selects it
                    topk_mask_direct = result.any(dim=0).any(dim=0)  # [num_quantized]
                else:
                    # top-k returns indices
                    idx_flat = result.reshape(-1).to(torch.int32).contiguous()

        pending_start = total_kv_tokens - num_pending

        # Use direct mask if available (top-p), otherwise use indices (top-k)
        if topk_mask_direct is not None:
            # Direct mask assignment: much faster than scatter
            topk_mask_unpacked[:num_quantized].copy_(topk_mask_direct.to(torch.uint8))
            # Set pending tokens to 1
            if pending_start < total_kv_tokens:
                topk_mask_unpacked[pending_start:total_kv_tokens].fill_(1)
        else:
            update_topk_mask(
                topk_mask_unpacked,
                idx_flat,
                num_quantized,
                pending_start,
                total_kv_tokens,
            )

        return topk_mask_unpacked, mask_offsets

    def _compute_quantized_topk_indices(
        self,
        state: RabitQRequestState,
        query: torch.Tensor,
        dtype: torch.dtype,
        layer_state: RabitQLayerState,
        topk: Optional[int],
    ) -> Optional[torch.Tensor]:
        """Return per-token top-k/top-p indices over quantized tokens."""
        import time
        _PROFILE = False  # Set to True to enable profiling
        _USE_FUSED_KERNEL = False  # Disabled - Triton kernel needs more work

        num_quantized = state.metadata_write_index
        if num_quantized == 0 or state.x_b_gpu is None:
            return None

        B_q = self._rabitq_state.b_q
        if B_q is None or B_q <= 0:
            return None

        topp = self._rabitq_state.topp
        # If neither topk nor topp is set, return None
        if (topk is None or topk <= 0) and topp is None:
            return None

        num_tokens = query.size(0)

        if _PROFILE:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if _USE_FUSED_KERNEL:
            # Use fused kernel for center + normalize + rotate + quantize
            centroid_q = state.centroid_q.to(dtype)
            centroid_k = state.centroid_k.to(dtype)
            rotation_t = layer_state.rotation_t

            # query shape: [num_tokens, num_heads * head_size] -> reshape to [num_tokens, num_heads, head_size]
            query_3d = query.view(num_tokens, self.num_heads, self.head_size)

            q_u, Delta_sq, v_l_sq, sum_q_u, q_norm, qr_dot_ck = fused_query_preprocess(
                query_3d,
                centroid_q,
                centroid_k,
                rotation_t,
                self.num_queries_per_kv,
                B_q,
            )

            if _PROFILE:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
        else:
            # Use torch.compile optimized version
            centroid_q = state.centroid_q.to(dtype)
            centroid_k = state.centroid_k.to(dtype)
            rotation_t = layer_state.rotation_t

            compiled_fn = get_compiled_query_preprocess()
            q_u, Delta_sq, v_l_sq, sum_q_u, q_norm, qr_dot_ck = compiled_fn(
                query,
                centroid_q,
                centroid_k,
                rotation_t,
                self.num_kv_heads,
                self.num_queries_per_kv,
                self.head_size,
                B_q,
            )

            if _PROFILE:
                torch.cuda.synchronize()
                t1 = time.perf_counter()

        # Use pre-packed binary data for optimized kernel (8x memory bandwidth reduction)
        x_b_packed = state.x_b_packed_gpu[:num_quantized].contiguous()
        sum_x_b = state.sum_x_b_gpu[:num_quantized].to(torch.float32)
        key_norms = state.key_norms_gpu[:num_quantized].to(torch.float32)
        k_bar_dot_k = state.k_bar_dot_k_gpu[:num_quantized].to(torch.float32)
        cq_dot_kr = state.cq_dot_kr_gpu[:num_quantized].to(torch.float32)

        if _PROFILE:
            torch.cuda.synchronize()
            t2 = time.perf_counter()

        scores = rabitq_int4_packed_binary_scores(
            q_u,
            Delta_sq.to(torch.float32),
            v_l_sq.to(torch.float32),
            sum_q_u.to(torch.float32),
            x_b_packed,
            sum_x_b,
            key_norms,
            k_bar_dot_k,
            cq_dot_kr,
            q_norm.to(torch.float32),
            qr_dot_ck.to(torch.float32),
            state.cq_dot_ck.to(torch.float32),
            1e-6,
        )

        if _PROFILE:
            torch.cuda.synchronize()
            t3 = time.perf_counter()

        # Use top-p if set, otherwise use top-k
        if topp is not None:
            # Top-p selection using CUDA ternary search kernel (no sorting)
            # scores shape: [num_tokens, num_kv_heads, num_quantized]
            # Scale by 1/sqrt(head_size) before softmax (standard attention scaling)
            scaled_scores = scores / math.sqrt(self.head_size)
            probs = torch.softmax(scores, dim=-1)

            # CUDA kernel: returns mask where probs > threshold
            # Directly return mask (bool tensor) - no need to convert to indices
            mask = top_p_mask(probs, topp)  # [num_tokens, num_kv_heads, num_quantized]

            if _PROFILE:
                torch.cuda.synchronize()
                t4 = time.perf_counter()
                if _USE_FUSED_KERNEL:
                    print(f"[RabitQ Topk Profile] num_q={num_quantized} | "
                          f"fused_preprocess: {(t1-t0)*1000:.3f}ms | "
                          f"prepare_kv: {(t2-t1)*1000:.3f}ms | "
                          f"cuda_scores: {(t3-t2)*1000:.3f}ms | "
                          f"top_p_mask: {(t4-t3)*1000:.3f}ms | "
                          f"TOTAL: {(t4-t0)*1000:.3f}ms")
                else:
                    print(f"[RabitQ Topk Profile] num_q={num_quantized} | "
                          f"preprocess: {(t1-t0)*1000:.3f}ms | "
                          f"prepare_kv: {(t2-t1)*1000:.3f}ms | "
                          f"cuda_scores: {(t3-t2)*1000:.3f}ms | "
                          f"top_p_mask: {(t4-t3)*1000:.3f}ms | "
                          f"TOTAL: {(t4-t0)*1000:.3f}ms")

            # Return mask directly (is_mask=True flag via dtype check: bool means mask)
            return mask
        else:
            topk_quantized = min(topk, num_quantized)
            if topk_quantized <= 0:
                return None
            _, idx_topk = torch.topk(scores, topk_quantized, dim=-1, sorted=False)

            if _PROFILE:
                torch.cuda.synchronize()
                t4 = time.perf_counter()
                if _USE_FUSED_KERNEL:
                    print(f"[RabitQ Topk Profile] num_q={num_quantized} | "
                          f"fused_preprocess: {(t1-t0)*1000:.3f}ms | "
                          f"prepare_kv: {(t2-t1)*1000:.3f}ms | "
                          f"cuda_scores: {(t3-t2)*1000:.3f}ms | "
                          f"topk: {(t4-t3)*1000:.3f}ms | "
                          f"TOTAL: {(t4-t0)*1000:.3f}ms")
                else:
                    print(f"[RabitQ Topk Profile] num_q={num_quantized} | "
                          f"preprocess: {(t1-t0)*1000:.3f}ms | "
                          f"prepare_kv: {(t2-t1)*1000:.3f}ms | "
                          f"cuda_scores: {(t3-t2)*1000:.3f}ms | "
                          f"topk: {(t4-t3)*1000:.3f}ms | "
                          f"TOTAL: {(t4-t0)*1000:.3f}ms")

            return idx_topk.to(torch.int32)

    def _run_rabitq_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RabitQMetadata,
        output: torch.Tensor,
        layer: torch.nn.Module,
        layer_state: RabitQLayerState,
    ) -> None:
        """
        Run RabitQ decode attention, writing results directly to output tensor.

        This follows FlashInfer's pattern: decode_wrapper.run(..., out=output)
        """
        req_ids = attn_metadata.req_ids
        if not req_ids:
            return

        start_locs_cpu = attn_metadata.query_start_loc_cpu
        if start_locs_cpu is None:
            return

        start_locs = getattr(attn_metadata, "_cached_query_start_locs_list", None)
        if start_locs is None:
            start_locs = start_locs_cpu.tolist()
            setattr(attn_metadata, "_cached_query_start_locs_list", start_locs)

        device = query.device
        dtype = query.dtype

        # Build batch info for prepared requests
        batch_info = []
        for idx, req_id in enumerate(req_ids):
            start = int(start_locs[idx])
            end = int(start_locs[idx + 1])
            if end <= start:
                continue

            state = layer_state.request_states.get(req_id)
            if state is None or not state.prepared:
                continue

            batch_info.append({
                'req_id': req_id,
                'req_idx': idx,
                'start': start,
                'end': end,
                'num_tokens': end - start,
                'state': state
            })

        if not batch_info:
            return

        # Try batch decode with RabitQ wrapper from metadata
        if (attn_metadata.rabitq_decode_wrapper is not None and
            all(info['num_tokens'] == 1 for info in batch_info)):

            success = self._run_batch_decode(
                batch_info, query, kv_cache, attn_metadata, output, layer_state
            )
            if success:
                return

        # Fallback: process each request individually (should rarely happen)
        logger.warning_once("RabitQ: Falling back to per-request decode")
        self._run_individual_decode(
            batch_info, query, kv_cache, attn_metadata, output, layer_state
        )

    def _run_batch_decode(
        self,
        batch_info: list[dict],
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RabitQMetadata,
        output: torch.Tensor,
        layer_state: RabitQLayerState,
    ) -> bool:
        """
        Run batched decode using RabitQ wrapper from metadata.
        Wrapper is already planned in build(), just compute mask and run.
        Returns True on success, False to trigger fallback.
        """
        try:
            device = query.device

            # Timing for _compute_topk_mask only
            _do_timing = _timing_collector.enabled
            if _do_timing:
                torch.cuda.synchronize()
                _topk_mask_start = time.perf_counter()

            topk_mask, mask_offsets = self._compute_topk_mask(
                batch_info, query, kv_cache, attn_metadata, device, layer_state
            )

            if _do_timing:
                torch.cuda.synchronize()
                _topk_mask_time_ms = (time.perf_counter() - _topk_mask_start) * 1000
                _timing_collector.record_topk_mask(self._layer_name, _topk_mask_time_ms)

            # === CONSTRUCT ALL-SELECT MASK FOR TESTING ===
            # total_kv_tokens = sum(info['state'].total_tokens for info in batch_info)
            # # Create unpacked mask with all 1s (select all tokens)
            # topk_mask = torch.ones(total_kv_tokens, dtype=torch.uint8, device=device)
            # # Create mask_offsets: cumulative token offsets for each request
            # mask_offsets = torch.zeros(len(batch_info), dtype=torch.uint32, device=device)
            # cumulative_tokens = 0
            # for i, info in enumerate(batch_info):
            #     mask_offsets[i] = cumulative_tokens
            #     cumulative_tokens += info['state'].total_tokens
            # === END ALL-SELECT MASK CONSTRUCTION ===

            # Get wrapper from metadata (already planned in build())
            wrapper = attn_metadata.rabitq_decode_wrapper

            # Run attention - writes directly to output (for all decode tokens)
            # Note: wrapper was planned for all decode requests in build()
            num_decode_tokens = attn_metadata.num_decode_tokens
            wrapper.run(
                query=query[:num_decode_tokens],
                kv_cache=kv_cache,
                topk_mask=topk_mask,
                mask_offsets=mask_offsets,
                sm_scale=self.scale,
                out=output[:num_decode_tokens],
            )

            return True

        except Exception as e:
            logger.warning_once(f"RabitQ: Batch decode failed: {e}")
            return False

    def _run_individual_decode(
        self,
        batch_info: list[dict],
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RabitQMetadata,
        output: torch.Tensor,
        layer_state: RabitQLayerState,
    ) -> None:
        """Fallback: process each request individually using SDPA."""
        device = query.device
        dtype = query.dtype
        topk = self._rabitq_state.topk

        for info in batch_info:
            state = info['state']
            req_idx = info['req_idx']
            start = info['start']
            end = info['end']

            # Trigger quantization if needed
            if state.num_pending_decode_tokens >= self._rabitq_state.batch_size:
                self._quantize_pending_keys(
                    state, req_idx, kv_cache, attn_metadata, device, dtype, layer_state
                )

            for token_idx in range(start, end):
                q = query[token_idx:token_idx + 1]

                # Compute approximate scores
                topk_limit = topk if topk is not None else state.metadata_write_index
                approx_indices = self._compute_quantized_topk_indices(
                    state, q, dtype, layer_state, topk_limit
                )

                # Select top-k indices
                num_quantized = state.metadata_write_index
                num_pending = state.num_pending_decode_tokens

                if num_quantized > 0 and approx_indices is not None:
                    # Check if result is a mask (bool) or indices (int32)
                    if approx_indices.dtype == torch.bool:
                        # top-p returns mask: [num_tokens, num_kv_heads, num_quantized]
                        # Reduce across tokens and heads, then get indices
                        mask_1d = approx_indices.any(dim=0).any(dim=0)  # [num_quantized]
                        idx_topk_sorted = mask_1d.nonzero(as_tuple=False).squeeze(-1).to(torch.long)
                    else:
                        # top-k returns indices
                        idx_topk_sorted, _ = torch.sort(approx_indices[0].to(torch.long))
                else:
                    idx_topk_sorted = torch.tensor([], device=device, dtype=torch.long)

                # Add pending tokens
                if num_pending > 0:
                    pending_start = state.total_tokens - num_pending
                    pending_indices = torch.arange(pending_start, state.total_tokens, device=device)
                    idx_topk_sorted = torch.cat([idx_topk_sorted, pending_indices])

                # Extract KV for selected tokens
                keys, values = self._extract_kv_from_paged_cache(
                    req_idx, idx_topk_sorted, kv_cache, attn_metadata, device, dtype
                )

                # Run SDPA
                query_sdpa = q.transpose(0, 1).unsqueeze(0)  # [1, num_heads, 1, head_size]
                key_sdpa = keys.transpose(0, 1).unsqueeze(0)  # [1, num_kv_heads, k, head_size]
                value_sdpa = values.transpose(0, 1).unsqueeze(0)

                if self.num_queries_per_kv > 1:
                    key_sdpa = key_sdpa.repeat_interleave(self.num_queries_per_kv, dim=1)
                    value_sdpa = value_sdpa.repeat_interleave(self.num_queries_per_kv, dim=1)

                with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
                    context = F.scaled_dot_product_attention(
                        query_sdpa, key_sdpa, value_sdpa,
                        attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
                    )

                # Write to output
                output[token_idx].copy_(context.squeeze(0).squeeze(1).view(-1))

    def _extract_kv_from_paged_cache(
        self,
        req_index: int,
        token_indices: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RabitQMetadata,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract KV rows from paged cache."""
        if token_indices.numel() == 0:
            empty = torch.empty(0, self.num_kv_heads, self.head_size, device=device, dtype=dtype)
            return empty, empty.clone()

        # FlashInfer layout: [num_blocks, 2, block_size, num_kv_heads, head_size]
        key_cache = kv_cache[:, 0, :, :, :]
        value_cache = kv_cache[:, 1, :, :, :]
        block_size = key_cache.shape[1]

        block_table = attn_metadata.block_table_tensor[req_index]

        block_indices = token_indices // block_size
        block_offsets = token_indices % block_size

        physical_blocks = block_table[block_indices.long()]

        keys = key_cache[physical_blocks.long(), block_offsets.long(), :, :]
        values = value_cache[physical_blocks.long(), block_offsets.long(), :, :]

        if keys.dtype != dtype:
            keys = keys.to(dtype)
        if values.dtype != dtype:
            values = values.to(dtype)

        return keys, values

    @staticmethod
    def _validate_b_q(b_q: int, head_size: int) -> None:
        if b_q <= 0:
            raise ValueError("rabitq_b_q must be positive.")
        min_bits = max(1, math.ceil(1 / math.sqrt(head_size)))
        if b_q < min_bits:
            raise ValueError(
                f"rabitq_b_q={b_q} violates the minimum requirement of "
                f"{min_bits} derived from head_size={head_size}.")
        
    # @cprofile("/root/code/RabitQCache/profile/rabitq_forward.prof")
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Optional[RabitQMetadata],
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if attn_metadata is None:
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)

        # Extract layer name for per-layer state management
        layer_name = getattr(layer, 'layer_name', 'unknown_layer')
        self._layer_name = layer_name
        layer_state = self._get_current_layer_state()

        # Timing instrumentation - start timing at the beginning of forward
        _do_timing = _timing_collector.enabled
        if _do_timing:
            # torch.cuda.synchronize()
            _timing_start = time.perf_counter()

        # Wait for THIS layer's previous quantization to complete
        # Only needed in decode phase - prefill quantization can run fully async
        # and will be synchronized when the first decode token arrives at this layer
        # if self._rabitq_state.b_q is not None and layer_state.quantize_stream is not None:
        #     torch.cuda.current_stream().wait_stream(layer_state.quantize_stream)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Initialize RabitQ parameters from metadata if not set
        if attn_metadata.rabitq_b_q is not None and self._rabitq_state.b_q is None:
            self._rabitq_state.b_q = attn_metadata.rabitq_b_q
        if attn_metadata.rabitq_topk is not None and self._rabitq_state.topk is None:
            self._rabitq_state.topk = attn_metadata.rabitq_topk
        if attn_metadata.rabitq_topp is not None and self._rabitq_state.topp is None:
            self._rabitq_state.topp = attn_metadata.rabitq_topp

        if self._rabitq_state.b_q is not None and not self._emitted_fallback_warning:
            logger.info_once(
                "RabitQ attention active; using FlashInfer for paged attention, "
                "quantized compute for top-k selection.")
            self._emitted_fallback_warning = True

        # Determine prefill vs decode
        max_query_len = attn_metadata.max_q_len
        is_prefill = max_query_len > 1
        num_decode_tokens = attn_metadata.num_decode_tokens

        if is_prefill and attn_metadata.req_ids:
            self._update_prefill_stats(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                attn_metadata,
            )

            # Use FlashInfer for actual prefill attention FIRST (runs in default stream)
            result = super().forward(layer, query, key, value, kv_cache,
                                attn_metadata, output, output_scale,
                                output_block_scale)

            # AFTER attention completes, start quantization in parallel stream
            if self._rabitq_state.b_q is not None:
                # Create per-layer quantization stream if not exists
                if layer_state.quantize_stream is None:
                    layer_state.quantize_stream = torch.cuda.Stream()

                # Rotation matrices are pre-loaded during initialization into _rotation_cache
                # Just copy from cache to layer_state if needed (fast, no I/O)
                self._ensure_rotation(query.device, key.dtype, layer_state)

                # Synchronize quantize_stream with default stream
                # This ensures quantize_stream waits for attention to complete
                layer_state.quantize_stream.wait_stream(torch.cuda.current_stream())

                # Get start_locs for request slicing (should already be cached from _update_prefill_stats)
                query_start_loc_cpu = attn_metadata.query_start_loc_cpu
                if query_start_loc_cpu is not None:
                    # Use cached start_locs to avoid redundant .tolist() call
                    start_locs = getattr(attn_metadata, "_cached_query_start_locs_list", None)
                    if start_locs is None:
                        start_locs = query_start_loc_cpu.tolist()

                    # Save pending quantization data (just references, no copy needed)
                    layer_state.pending_quantization = PendingQuantizationData(
                        keys=key[:num_actual_tokens],
                        req_ids=list(attn_metadata.req_ids),
                        start_locs=start_locs,
                        device=query.device,
                    )

                    # Launch async quantization (non-blocking)
                    self._process_pending_quantization_async(layer_state)

            # Quantization continues in background - will be synchronized at next forward() call
            # This allows quantization to overlap with subsequent layer computations

            # Record forward timing
            if _do_timing:
                # torch.cuda.synchronize()
                _timing_collector.record_forward(layer_name, (time.perf_counter() - _timing_start) * 1000)

            return result

        elif num_decode_tokens > 0 and self._rabitq_state.b_q is not None and attn_metadata.req_ids:

            # Decode: use RabitQ approximate attention

            # Wait for this layer's prefill quantization to complete (if any)
            # This is the sync point - quantization from prefill runs fully async
            # until the first decode token arrives at this layer
            if layer_state.quantize_stream is not None:
                torch.cuda.current_stream().wait_stream(layer_state.quantize_stream)

            # Prepare per-request state (centroids, quantization)
            # self._prepare_requests(attn_metadata.req_ids, query.device)

            # Write new K/V to paged cache

            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )


            # Track new decode tokens
            self._append_decode_kv(attn_metadata, layer_state)



            # Run RabitQ decode attention (writes directly to output)
            self._run_rabitq_decode(
                query[:num_actual_tokens],
                kv_cache,
                attn_metadata,
                output[:num_actual_tokens],
                layer,
                layer_state,
            )

            # Record forward timing
            if _do_timing:
                # torch.cuda.synchronize()
                _timing_collector.record_forward(layer_name, (time.perf_counter() - _timing_start) * 1000)

            return output

        # Fallback to FlashInfer
        result = super().forward(layer, query, key, value, kv_cache,
                            attn_metadata, output, output_scale,
                            output_block_scale)

        # Record forward timing for fallback path
        if _do_timing:
            # torch.cuda.synchronize()
            _timing_collector.record_forward(layer_name, (time.perf_counter() - _timing_start) * 1000)

        return result
