# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compatibility module for LMCache integration.

LMCache expects certain utilities to be in vllm.utils.torch_utils,
but they are currently in vllm.utils. This module provides the necessary
imports for compatibility.
"""

from vllm.utils import get_kv_cache_torch_dtype

__all__ = ["get_kv_cache_torch_dtype"]
