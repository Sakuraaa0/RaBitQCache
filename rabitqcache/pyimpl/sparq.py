# This is a Python implementation of the paper:
#   "SparQ Attention: Bandwidth-Efficient LLM Inference"
#   (https://arxiv.org/pdf/2312.04985.pdf)

# Note that this Python version is just for testing accuracy, not for efficiency.
# Hence we use a "naive" implementation, which will compute full attention weights.


import torch

import math

from typing import Optional

from .top_k import top_k


def sparq(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    r: int,
) -> torch.Tensor:
    """SparQ weight estimator.

    r = head_dim * compression_rate

    e.g. compression_rate = 1/8, head_dim = 128, r = 16

    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    query_norm = torch.abs(query_states)

    _, channel_indices = torch.topk(query_norm, dim=-1, k=r)

    partial_query = torch.gather(query_states, dim=-1, index=channel_indices)
    channel_indices = channel_indices.repeat(1, 1, key_states.shape[-2], 1)
    partial_key = torch.gather(key_states, dim=-1, index=channel_indices)

    estimated_weights = torch.matmul(
        partial_query, partial_key.transpose(2, 3)
    ) / math.sqrt(
        query_states.shape[-1]
    )  # Divided by original d, not r
    return estimated_weights


def sparq_selector(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    r: int,
    token_budget: Optional[int] = None,
    token_ratio: Optional[float] = None,
) -> torch.Tensor:
    """SparQ index selector.
    Returns:
        mask: A mask with "True" or "False indicating whether the token is selected.
    """

    estimated_weights = sparq(query_states, key_states, r)

    if token_ratio is not None:
        if token_ratio <= 0:
            raise ValueError(f"token_ratio must be > 0, got {token_ratio}")
        seq_len = int(estimated_weights.shape[-1])
        token_budget = max(1, math.ceil(seq_len * float(token_ratio)))
    elif token_budget is None:
        raise ValueError("Either token_budget or token_ratio must be provided for SparQ.")

    return top_k(estimated_weights, int(token_budget))
