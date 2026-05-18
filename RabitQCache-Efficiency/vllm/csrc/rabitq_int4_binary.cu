#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cfloat>
#include <cub/cub.cuh>
#include <cuda/functional>
#include <numeric>  // for std::gcd

#include "cuda_utils.h"

namespace vllm {

namespace {

// ============================================================================
// dp4a intrinsic: 4-element uint8 dot product with int32 accumulation
// Computes: d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + c
// Available on SM61+ (Pascal and later)
// ============================================================================
__device__ __forceinline__ int32_t dp4a_u32(uint32_t a, uint32_t b, int32_t c) {
  int32_t result;
  asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
  return result;
}

// ============================================================================
// Optimized kernel using dp4a instruction for INT4 x Binary inner product
// Requires head_size to be divisible by 4
// ============================================================================
__global__ void rabitq_int4_binary_scores_kernel(
    const uint8_t* __restrict__ q_u,
    const float* __restrict__ delta_vals,
    const float* __restrict__ v_l_vals,
    const float* __restrict__ sum_q_u,
    const uint8_t* __restrict__ x_b,
    const float* __restrict__ sum_x_b,
    const float* __restrict__ key_norms,
    const float* __restrict__ k_bar_dot_k,
    const float* __restrict__ cq_dot_kr,
    const float* __restrict__ q_norms,
    const float* __restrict__ qr_dot_ck,
    const float* __restrict__ cq_dot_ck,
    const float sqrt_head_size,
    const float denom_eps,
    const int64_t num_tokens,
    const int64_t num_quantized,
    const int64_t num_heads,
    const int64_t head_size,
    float* __restrict__ scores_out) {
  const int token_idx = blockIdx.x;
  const int kv_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (token_idx >= num_tokens || kv_idx >= num_quantized) {
    return;
  }

  const int64_t q_token_stride = num_heads * head_size;
  const int64_t meta_stride = num_heads;
  const int packed_size = head_size / 4;  // Number of uint32 elements

  const uint8_t* q_token_ptr = q_u + token_idx * q_token_stride;
  const float* delta_ptr = delta_vals + token_idx * meta_stride;
  const float* v_l_ptr = v_l_vals + token_idx * meta_stride;
  const float* sum_q_ptr = sum_q_u + token_idx * meta_stride;
  const float* q_norm_ptr = q_norms + token_idx * meta_stride;
  const float* qr_dot_ck_ptr = qr_dot_ck + token_idx * meta_stride;

  const uint8_t* x_token_ptr = x_b + kv_idx * q_token_stride;
  const float* sum_x_ptr = sum_x_b + kv_idx * meta_stride;
  const float* key_norm_ptr = key_norms + kv_idx * meta_stride;
  const float* k_bar_ptr = k_bar_dot_k + kv_idx * meta_stride;
  const float* cq_dot_kr_ptr = cq_dot_kr + kv_idx * meta_stride;

  float max_score = -FLT_MAX;

  for (int head = 0; head < num_heads; ++head) {
    // Reinterpret as uint32 pointers for dp4a
    const uint32_t* q_vec = reinterpret_cast<const uint32_t*>(
        q_token_ptr + static_cast<int64_t>(head) * head_size);
    const uint32_t* x_vec = reinterpret_cast<const uint32_t*>(
        x_token_ptr + static_cast<int64_t>(head) * head_size);

    // Compute inner product using dp4a: processes 4 uint8 elements per iteration
    int32_t inner_product = 0;
#pragma unroll 8
    for (int i = 0; i < packed_size; ++i) {
      inner_product = dp4a_u32(q_vec[i], x_vec[i], inner_product);
    }

    const float delta_val = delta_ptr[head];
    const float v_l_val = v_l_ptr[head];
    const float sum_q_val = sum_q_ptr[head];
    const float sum_x_val = sum_x_ptr[head];

    const float term1 = 2.f * delta_val * static_cast<float>(inner_product);
    const float term2 = 2.f * v_l_val * sum_x_val;
    const float term3 = -delta_val * sum_q_val;
    const float term4 = -static_cast<float>(head_size) * v_l_val;

    const float x_bar_dot_q = term1 + term2 + term3 + term4;

    float denom = k_bar_ptr[head];
    if (fabsf(denom) < denom_eps) {
      denom = denom >= 0.f ? denom_eps : -denom_eps;
    }

    const float approx_kq = x_bar_dot_q / denom;

    float score = q_norm_ptr[head] * key_norm_ptr[head] * approx_kq;
    score += qr_dot_ck_ptr[head] + cq_dot_kr_ptr[head] - cq_dot_ck[head];

    if (score > max_score) {
      max_score = score;
    }
  }

  scores_out[token_idx * num_quantized + kv_idx] = max_score;
}

// ============================================================================
// Optimized Warp-level kernel with shared memory tiling
// Key insight: Cache q_u in shared memory, then each thread processes one KV
// This combines the memory bandwidth benefit of shared memory with high parallelism
// ============================================================================
template <int BLOCK_SIZE = 256, int TILE_KV = 32>
__global__ void rabitq_int4_packed_binary_scores_warp_kernel(
    const uint8_t* __restrict__ q_u,           // [num_tokens, num_heads, head_size]
    const float* __restrict__ delta_vals,
    const float* __restrict__ v_l_vals,
    const float* __restrict__ sum_q_u,
    const uint32_t* __restrict__ x_b_packed,   // [num_quantized, num_heads, head_size/32]
    const float* __restrict__ sum_x_b,
    const float* __restrict__ key_norms,
    const float* __restrict__ k_bar_dot_k,
    const float* __restrict__ cq_dot_kr,
    const float* __restrict__ q_norms,
    const float* __restrict__ qr_dot_ck,
    const float* __restrict__ cq_dot_ck,
    const float denom_eps,
    const int64_t num_tokens,
    const int64_t num_quantized,
    const int64_t num_heads,
    const int64_t head_size,
    float* __restrict__ scores_out) {

  // Shared memory for caching q_u data for current token
  // Layout: [num_heads][head_size] - we cache one token's q data
  extern __shared__ uint8_t smem[];
  uint8_t* q_shared = smem;  // [num_heads * head_size]

  const int token_idx = blockIdx.x;
  const int kv_idx = blockIdx.y * BLOCK_SIZE + threadIdx.x;

  if (token_idx >= num_tokens) return;

  const int64_t packed_dim = head_size / 32;
  const int64_t q_token_stride = num_heads * head_size;
  const int64_t x_packed_stride = num_heads * packed_dim;
  const int64_t meta_stride = num_heads;

  // Cooperatively load q_u for this token into shared memory
  const uint8_t* q_token_ptr = q_u + token_idx * q_token_stride;
  const int total_q_elements = num_heads * head_size;

  for (int i = threadIdx.x; i < total_q_elements; i += BLOCK_SIZE) {
    q_shared[i] = q_token_ptr[i];
  }
  __syncthreads();

  // Now each thread processes one KV index
  if (kv_idx >= num_quantized) return;

  // Load token-level metadata (same for all KV in this block)
  const float* delta_ptr = delta_vals + token_idx * meta_stride;
  const float* v_l_ptr = v_l_vals + token_idx * meta_stride;
  const float* sum_q_ptr = sum_q_u + token_idx * meta_stride;
  const float* q_norm_ptr = q_norms + token_idx * meta_stride;
  const float* qr_dot_ck_ptr = qr_dot_ck + token_idx * meta_stride;

  // KV-specific pointers
  const uint32_t* x_packed_ptr = x_b_packed + kv_idx * x_packed_stride;
  const float* sum_x_ptr = sum_x_b + kv_idx * meta_stride;
  const float* key_norm_ptr = key_norms + kv_idx * meta_stride;
  const float* k_bar_ptr = k_bar_dot_k + kv_idx * meta_stride;
  const float* cq_dot_kr_ptr = cq_dot_kr + kv_idx * meta_stride;

  float max_score = -FLT_MAX;

  for (int head = 0; head < num_heads; ++head) {
    const uint8_t* q_vec = q_shared + head * head_size;  // Read from shared memory
    const uint32_t* x_packed = x_packed_ptr + head * packed_dim;

    // Compute inner product using efficient bit extraction
    int32_t inner_product = 0;

#pragma unroll 4
    for (int p = 0; p < packed_dim; ++p) {
      uint32_t bits = x_packed[p];
      const uint8_t* q_segment = q_vec + p * 32;

      // Vectorized load: 32 bytes as 8 x uint32
      const uint32_t* q_u32 = reinterpret_cast<const uint32_t*>(q_segment);

      // Process 8 groups of 4 bytes each
#pragma unroll 8
      for (int g = 0; g < 8; ++g) {
        uint32_t q_packed = q_u32[g];
        uint32_t bits4 = (bits >> (g * 4)) & 0xF;

        // Extract 4 bytes from q_packed
        uint8_t q0 = q_packed & 0xFF;
        uint8_t q1 = (q_packed >> 8) & 0xFF;
        uint8_t q2 = (q_packed >> 16) & 0xFF;
        uint8_t q3 = (q_packed >> 24) & 0xFF;

        // Multiply by corresponding bits (0 or 1)
        inner_product += q0 * (bits4 & 1);
        inner_product += q1 * ((bits4 >> 1) & 1);
        inner_product += q2 * ((bits4 >> 2) & 1);
        inner_product += q3 * ((bits4 >> 3) & 1);
      }
    }

    const float delta_val = delta_ptr[head];
    const float v_l_val = v_l_ptr[head];
    const float sum_q_val = sum_q_ptr[head];
    const float sum_x_val = sum_x_ptr[head];

    const float term1 = 2.f * delta_val * static_cast<float>(inner_product);
    const float term2 = 2.f * v_l_val * sum_x_val;
    const float term3 = -delta_val * sum_q_val;
    const float term4 = -static_cast<float>(head_size) * v_l_val;

    const float x_bar_dot_q = term1 + term2 + term3 + term4;

    float denom = k_bar_ptr[head];
    if (fabsf(denom) < denom_eps) {
      denom = denom >= 0.f ? denom_eps : -denom_eps;
    }

    const float approx_kq = x_bar_dot_q / denom;

    float score = q_norm_ptr[head] * key_norm_ptr[head] * approx_kq;
    score += qr_dot_ck_ptr[head] + cq_dot_kr_ptr[head] - cq_dot_ck[head];

    if (score > max_score) {
      max_score = score;
    }
  }

  scores_out[token_idx * num_quantized + kv_idx] = max_score;
}

// ============================================================================
// Packed Binary kernel: x_b is packed as uint32 (32 binary bits per element)
// This reduces memory bandwidth by 8x for x_b tensor
//
// Optimization: Use vectorized loads + efficient bit extraction
// For head_size=128: packed_dim=4, inner loop processes 32 bits each
// ============================================================================
__global__ void rabitq_int4_packed_binary_scores_kernel(
    const uint8_t* __restrict__ q_u,           // [num_tokens, num_heads, head_size]
    const float* __restrict__ delta_vals,
    const float* __restrict__ v_l_vals,
    const float* __restrict__ sum_q_u,
    const uint32_t* __restrict__ x_b_packed,   // [num_quantized, num_heads, head_size/32]
    const float* __restrict__ sum_x_b,
    const float* __restrict__ key_norms,
    const float* __restrict__ k_bar_dot_k,
    const float* __restrict__ cq_dot_kr,
    const float* __restrict__ q_norms,
    const float* __restrict__ qr_dot_ck,
    const float* __restrict__ cq_dot_ck,
    const float denom_eps,
    const int64_t num_tokens,
    const int64_t num_quantized,
    const int64_t num_heads,
    const int64_t head_size,
    float* __restrict__ scores_out) {
  const int token_idx = blockIdx.x;
  const int kv_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (token_idx >= num_tokens || kv_idx >= num_quantized) {
    return;
  }

  const int64_t packed_dim = head_size / 32;  // Number of uint32 per head
  const int64_t q_token_stride = num_heads * head_size;
  const int64_t x_packed_stride = num_heads * packed_dim;
  const int64_t meta_stride = num_heads;

  const uint8_t* q_token_ptr = q_u + token_idx * q_token_stride;
  const float* delta_ptr = delta_vals + token_idx * meta_stride;
  const float* v_l_ptr = v_l_vals + token_idx * meta_stride;
  const float* sum_q_ptr = sum_q_u + token_idx * meta_stride;
  const float* q_norm_ptr = q_norms + token_idx * meta_stride;
  const float* qr_dot_ck_ptr = qr_dot_ck + token_idx * meta_stride;

  const uint32_t* x_packed_ptr = x_b_packed + kv_idx * x_packed_stride;
  const float* sum_x_ptr = sum_x_b + kv_idx * meta_stride;
  const float* key_norm_ptr = key_norms + kv_idx * meta_stride;
  const float* k_bar_ptr = k_bar_dot_k + kv_idx * meta_stride;
  const float* cq_dot_kr_ptr = cq_dot_kr + kv_idx * meta_stride;

  float max_score = -FLT_MAX;

  for (int head = 0; head < num_heads; ++head) {
    const uint8_t* q_vec = q_token_ptr + static_cast<int64_t>(head) * head_size;
    const uint32_t* x_packed = x_packed_ptr + static_cast<int64_t>(head) * packed_dim;

    // Compute inner product: sum(q_u[i]) where x_b[i] == 1
    // Each uint32 contains 32 binary bits
    int32_t inner_product = 0;

#pragma unroll 4
    for (int p = 0; p < packed_dim; ++p) {
      uint32_t bits = x_packed[p];
      const uint8_t* q_segment = q_vec + p * 32;

      // Process 32 bits using vectorized 128-bit loads
      // Load 16 bytes at a time (4 uint32)
      const uint4* q_vec4 = reinterpret_cast<const uint4*>(q_segment);

      // First 16 bytes (bits 0-15)
      uint4 q_data0 = q_vec4[0];
      const uint8_t* q0 = reinterpret_cast<const uint8_t*>(&q_data0);

      // Second 16 bytes (bits 16-31)
      uint4 q_data1 = q_vec4[1];
      const uint8_t* q1 = reinterpret_cast<const uint8_t*>(&q_data1);

      // Unrolled accumulation for first 16 bits
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        inner_product += q0[i] * ((bits >> i) & 1);
      }

      // Unrolled accumulation for second 16 bits
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        inner_product += q1[i] * ((bits >> (16 + i)) & 1);
      }
    }

    const float delta_val = delta_ptr[head];
    const float v_l_val = v_l_ptr[head];
    const float sum_q_val = sum_q_ptr[head];
    const float sum_x_val = sum_x_ptr[head];

    const float term1 = 2.f * delta_val * static_cast<float>(inner_product);
    const float term2 = 2.f * v_l_val * sum_x_val;
    const float term3 = -delta_val * sum_q_val;
    const float term4 = -static_cast<float>(head_size) * v_l_val;

    const float x_bar_dot_q = term1 + term2 + term3 + term4;

    float denom = k_bar_ptr[head];
    if (fabsf(denom) < denom_eps) {
      denom = denom >= 0.f ? denom_eps : -denom_eps;
    }

    const float approx_kq = x_bar_dot_q / denom;

    float score = q_norm_ptr[head] * key_norm_ptr[head] * approx_kq;
    score += qr_dot_ck_ptr[head] + cq_dot_kr_ptr[head] - cq_dot_ck[head];

    if (score > max_score) {
      max_score = score;
    }
  }

  scores_out[token_idx * num_quantized + kv_idx] = max_score;
}

}  // namespace

}  // namespace vllm

torch::Tensor rabitq_int4_binary_scores(
    torch::Tensor q_u, torch::Tensor delta_vals, torch::Tensor v_l_vals,
    torch::Tensor sum_q_u, torch::Tensor x_b, torch::Tensor sum_x_b,
    torch::Tensor key_norms, torch::Tensor k_bar_dot_k,
    torch::Tensor cq_dot_kr, torch::Tensor q_norms,
    torch::Tensor qr_dot_ck, torch::Tensor cq_dot_ck,
    double sqrt_head_size, double denom_eps) {
  TORCH_CHECK(q_u.is_cuda(), "q_u must be on CUDA device");
  TORCH_CHECK(x_b.is_cuda(), "x_b must be on CUDA device");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_u));

  const int64_t num_tokens = q_u.size(0);
  const int64_t num_heads = q_u.size(1);
  const int64_t head_size = q_u.size(2);
  const int64_t num_quantized = x_b.size(0);

  TORCH_CHECK(delta_vals.size(0) == num_tokens &&
                  delta_vals.size(1) == num_heads,
              "delta shape mismatch");
  TORCH_CHECK(v_l_vals.sizes() == delta_vals.sizes(),
              "v_l shape mismatch");
  TORCH_CHECK(sum_q_u.sizes() == delta_vals.sizes(),
              "sum_q_u shape mismatch");
  TORCH_CHECK(q_norms.sizes() == delta_vals.sizes(),
              "q_norm shape mismatch");
  TORCH_CHECK(qr_dot_ck.sizes() == delta_vals.sizes(),
              "qr_dot_ck shape mismatch");
  TORCH_CHECK(cq_dot_ck.numel() == num_heads,
              "cq_dot_ck shape mismatch");
  TORCH_CHECK(x_b.sizes() ==
                  torch::IntArrayRef({num_quantized, num_heads, head_size}),
              "x_b shape mismatch");
  TORCH_CHECK(sum_x_b.size(0) == num_quantized &&
                  sum_x_b.size(1) == num_heads,
              "sum_x_b shape mismatch");
  TORCH_CHECK(key_norms.sizes() == sum_x_b.sizes(),
              "key_norms shape mismatch");
  TORCH_CHECK(k_bar_dot_k.sizes() == sum_x_b.sizes(),
              "k_bar_dot_k shape mismatch");
  TORCH_CHECK(cq_dot_kr.sizes() == sum_x_b.sizes(),
              "cq_dot_kr shape mismatch");

  auto options = q_norms.options().dtype(torch::kFloat32);
  auto scores = torch::empty({num_tokens, num_quantized}, options);

  const dim3 block_dim(
      std::min<int64_t>(256, (num_quantized + 31) / 32 * 32), 1, 1);
  const dim3 grid_dim(num_tokens,
                      (num_quantized + block_dim.x - 1) / block_dim.x, 1);

  vllm::rabitq_int4_binary_scores_kernel<<<grid_dim, block_dim, 0,
                                     at::cuda::getCurrentCUDAStream()>>>(
      q_u.data_ptr<uint8_t>(), delta_vals.data_ptr<float>(),
      v_l_vals.data_ptr<float>(), sum_q_u.data_ptr<float>(),
      x_b.data_ptr<uint8_t>(), sum_x_b.data_ptr<float>(),
      key_norms.data_ptr<float>(), k_bar_dot_k.data_ptr<float>(),
      cq_dot_kr.data_ptr<float>(), q_norms.data_ptr<float>(),
      qr_dot_ck.data_ptr<float>(), cq_dot_ck.data_ptr<float>(),
      static_cast<float>(sqrt_head_size), static_cast<float>(denom_eps),
      num_tokens, num_quantized, num_heads, head_size,
      scores.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return scores;
}

// ============================================================================
// Packed Binary version: x_b is packed as uint32 (32 bits per element)
// This provides 8x memory bandwidth reduction for x_b tensor
// ============================================================================
torch::Tensor rabitq_int4_packed_binary_scores(
    torch::Tensor q_u, torch::Tensor delta_vals, torch::Tensor v_l_vals,
    torch::Tensor sum_q_u, torch::Tensor x_b_packed, torch::Tensor sum_x_b,
    torch::Tensor key_norms, torch::Tensor k_bar_dot_k,
    torch::Tensor cq_dot_kr, torch::Tensor q_norms,
    torch::Tensor qr_dot_ck, torch::Tensor cq_dot_ck,
    double denom_eps) {
  TORCH_CHECK(q_u.is_cuda(), "q_u must be on CUDA device");
  TORCH_CHECK(x_b_packed.is_cuda(), "x_b_packed must be on CUDA device");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_u));

  const int64_t num_tokens = q_u.size(0);
  const int64_t num_heads = q_u.size(1);
  const int64_t head_size = q_u.size(2);
  const int64_t num_quantized = x_b_packed.size(0);
  const int64_t packed_dim = head_size / 32;

  TORCH_CHECK(head_size % 32 == 0,
              "head_size must be divisible by 32 for packed binary, got ", head_size);
  TORCH_CHECK(delta_vals.size(0) == num_tokens &&
                  delta_vals.size(1) == num_heads,
              "delta shape mismatch");
  TORCH_CHECK(v_l_vals.sizes() == delta_vals.sizes(),
              "v_l shape mismatch");
  TORCH_CHECK(sum_q_u.sizes() == delta_vals.sizes(),
              "sum_q_u shape mismatch");
  TORCH_CHECK(q_norms.sizes() == delta_vals.sizes(),
              "q_norm shape mismatch");
  TORCH_CHECK(qr_dot_ck.sizes() == delta_vals.sizes(),
              "qr_dot_ck shape mismatch");
  TORCH_CHECK(cq_dot_ck.numel() == num_heads,
              "cq_dot_ck shape mismatch");
  TORCH_CHECK(x_b_packed.sizes() ==
                  torch::IntArrayRef({num_quantized, num_heads, packed_dim}),
              "x_b_packed shape mismatch, expected [", num_quantized, ", ",
              num_heads, ", ", packed_dim, "]");
  TORCH_CHECK(sum_x_b.size(0) == num_quantized &&
                  sum_x_b.size(1) == num_heads,
              "sum_x_b shape mismatch");
  TORCH_CHECK(key_norms.sizes() == sum_x_b.sizes(),
              "key_norms shape mismatch");
  TORCH_CHECK(k_bar_dot_k.sizes() == sum_x_b.sizes(),
              "k_bar_dot_k shape mismatch");
  TORCH_CHECK(cq_dot_kr.sizes() == sum_x_b.sizes(),
              "cq_dot_kr shape mismatch");

  auto options = q_norms.options().dtype(torch::kFloat32);
  auto scores = torch::empty({num_tokens, num_quantized}, options);

  const dim3 block_dim(
      std::min<int64_t>(256, (num_quantized + 31) / 32 * 32), 1, 1);
  const dim3 grid_dim(num_tokens,
                      (num_quantized + block_dim.x - 1) / block_dim.x, 1);

  vllm::rabitq_int4_packed_binary_scores_kernel<<<grid_dim, block_dim, 0,
                                     at::cuda::getCurrentCUDAStream()>>>(
      q_u.data_ptr<uint8_t>(), delta_vals.data_ptr<float>(),
      v_l_vals.data_ptr<float>(), sum_q_u.data_ptr<float>(),
      reinterpret_cast<const uint32_t*>(x_b_packed.data_ptr<int32_t>()), sum_x_b.data_ptr<float>(),
      key_norms.data_ptr<float>(), k_bar_dot_k.data_ptr<float>(),
      cq_dot_kr.data_ptr<float>(), q_norms.data_ptr<float>(),
      qr_dot_ck.data_ptr<float>(), cq_dot_ck.data_ptr<float>(),
      static_cast<float>(denom_eps),
      num_tokens, num_quantized, num_heads, head_size,
      scores.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return scores;
}

// ============================================================================
// Warp-level parallel version: uses warp shuffle for reduction
// This provides better parallelism for the inner product computation
// ============================================================================
torch::Tensor rabitq_int4_packed_binary_scores_warp(
    torch::Tensor q_u, torch::Tensor delta_vals, torch::Tensor v_l_vals,
    torch::Tensor sum_q_u, torch::Tensor x_b_packed, torch::Tensor sum_x_b,
    torch::Tensor key_norms, torch::Tensor k_bar_dot_k,
    torch::Tensor cq_dot_kr, torch::Tensor q_norms,
    torch::Tensor qr_dot_ck, torch::Tensor cq_dot_ck,
    double denom_eps) {
  TORCH_CHECK(q_u.is_cuda(), "q_u must be on CUDA device");
  TORCH_CHECK(x_b_packed.is_cuda(), "x_b_packed must be on CUDA device");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_u));

  const int64_t num_tokens = q_u.size(0);
  const int64_t num_heads = q_u.size(1);
  const int64_t head_size = q_u.size(2);
  const int64_t num_quantized = x_b_packed.size(0);
  const int64_t packed_dim = head_size / 32;

  TORCH_CHECK(head_size % 32 == 0,
              "head_size must be divisible by 32 for packed binary, got ", head_size);
  TORCH_CHECK(delta_vals.size(0) == num_tokens &&
                  delta_vals.size(1) == num_heads,
              "delta shape mismatch");
  TORCH_CHECK(v_l_vals.sizes() == delta_vals.sizes(),
              "v_l shape mismatch");
  TORCH_CHECK(sum_q_u.sizes() == delta_vals.sizes(),
              "sum_q_u shape mismatch");
  TORCH_CHECK(q_norms.sizes() == delta_vals.sizes(),
              "q_norm shape mismatch");
  TORCH_CHECK(qr_dot_ck.sizes() == delta_vals.sizes(),
              "qr_dot_ck shape mismatch");
  TORCH_CHECK(cq_dot_ck.numel() == num_heads,
              "cq_dot_ck shape mismatch");
  TORCH_CHECK(x_b_packed.sizes() ==
                  torch::IntArrayRef({num_quantized, num_heads, packed_dim}),
              "x_b_packed shape mismatch, expected [", num_quantized, ", ",
              num_heads, ", ", packed_dim, "]");
  TORCH_CHECK(sum_x_b.size(0) == num_quantized &&
                  sum_x_b.size(1) == num_heads,
              "sum_x_b shape mismatch");
  TORCH_CHECK(key_norms.sizes() == sum_x_b.sizes(),
              "key_norms shape mismatch");
  TORCH_CHECK(k_bar_dot_k.sizes() == sum_x_b.sizes(),
              "k_bar_dot_k shape mismatch");
  TORCH_CHECK(cq_dot_kr.sizes() == sum_x_b.sizes(),
              "cq_dot_kr shape mismatch");

  auto options = q_norms.options().dtype(torch::kFloat32);
  auto scores = torch::empty({num_tokens, num_quantized}, options);

  // New kernel: grid is (num_tokens, ceil(num_quantized/BLOCK_SIZE))
  // Each block processes one token and BLOCK_SIZE KV pairs
  constexpr int64_t BLOCK_SIZE = 256;
  const dim3 block_dim(BLOCK_SIZE, 1, 1);
  const dim3 grid_dim(num_tokens, (num_quantized + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

  // Shared memory size: num_heads * head_size bytes for q_u cache
  const size_t smem_size = num_heads * head_size * sizeof(uint8_t);

  vllm::rabitq_int4_packed_binary_scores_warp_kernel<BLOCK_SIZE><<<grid_dim, block_dim, smem_size,
                                     at::cuda::getCurrentCUDAStream()>>>(
      q_u.data_ptr<uint8_t>(), delta_vals.data_ptr<float>(),
      v_l_vals.data_ptr<float>(), sum_q_u.data_ptr<float>(),
      reinterpret_cast<const uint32_t*>(x_b_packed.data_ptr<int32_t>()), sum_x_b.data_ptr<float>(),
      key_norms.data_ptr<float>(), k_bar_dot_k.data_ptr<float>(),
      cq_dot_kr.data_ptr<float>(), q_norms.data_ptr<float>(),
      qr_dot_ck.data_ptr<float>(), cq_dot_ck.data_ptr<float>(),
      static_cast<float>(denom_eps),
      num_tokens, num_quantized, num_heads, head_size,
      scores.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return scores;
}

// ==================== Top-P Mask Kernel ====================
// Based on FlashInfer's TopPRenormProbKernel (sampling.cuh:1545-1715)
// Outputs boolean mask instead of renormalized probs

namespace vllm {

// GetCudaComputeCapability matching FlashInfer's utils.cuh:335-342
inline std::pair<int, int> GetCudaComputeCapability() {
  int device_id = 0;
  cudaGetDevice(&device_id);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  return std::make_pair(major, minor);
}

using namespace cub;

constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 ceil_div(const T1 x, const T2 y) noexcept {
  return (x + y - 1) / y;
}

// Simplified vec_t for float, matching FlashInfer's vec_dtypes.cuh
template <size_t vec_size>
struct vec_t {
  static_assert(vec_size == 1 || vec_size == 2 || vec_size == 4 || vec_size % 4 == 0,
                "Invalid vector size");
};

template <>
struct vec_t<1> {
  float data;
  __device__ __forceinline__ float& operator[](size_t i) { return (&data)[i]; }
  __device__ __forceinline__ const float& operator[](size_t i) const { return (&data)[i]; }
  __device__ __forceinline__ void fill(float val) { data = val; }
  __device__ __forceinline__ void load(const float* ptr) { data = *ptr; }
  __device__ __forceinline__ void store(float* ptr) const { *ptr = data; }
};

template <>
struct vec_t<2> {
  float2 data;
  __device__ __forceinline__ float& operator[](size_t i) { return ((float*)&data)[i]; }
  __device__ __forceinline__ const float& operator[](size_t i) const { return ((const float*)&data)[i]; }
  __device__ __forceinline__ void fill(float val) { data = make_float2(val, val); }
  __device__ __forceinline__ void load(const float* ptr) { data = *((float2*)ptr); }
  __device__ __forceinline__ void store(float* ptr) const { *((float2*)ptr) = data; }
};

template <>
struct vec_t<4> {
  float4 data;
  __device__ __forceinline__ float& operator[](size_t i) { return ((float*)&data)[i]; }
  __device__ __forceinline__ const float& operator[](size_t i) const { return ((const float*)&data)[i]; }
  __device__ __forceinline__ void fill(float val) { data = make_float4(val, val, val, val); }
  __device__ __forceinline__ void load(const float* ptr) { data = *((float4*)ptr); }
  __device__ __forceinline__ void store(float* ptr) const { *((float4*)ptr) = data; }
};

// MinReduceOp and MaxReduceOp matching FlashInfer's sampling.cuh
// CUDA 13 (12.9+) deprecated cub::Max/Min in favor of cuda::maximum/minimum
#if CUDA_VERSION >= 12090
using MaxReduceOp = cuda::maximum<>;
using MinReduceOp = cuda::minimum<>;
#else
using MaxReduceOp = cub::Max;
using MinReduceOp = cub::Min;
#endif

// DISPATCH_ALIGNED_VEC_SIZE matching FlashInfer's utils.cuh:275-304
// For float (4 bytes): vec_size = gcd(4, d), max is 4
#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 4: {                                                              \
      constexpr uint32_t ALIGNED_VEC_SIZE = 4;                             \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr uint32_t ALIGNED_VEC_SIZE = 2;                             \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr uint32_t ALIGNED_VEC_SIZE = 1;                             \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default:                                                               \
      break;                                                               \
  }

// DISPATCH_COMPUTE_CAP_NUM_THREADS matching FlashInfer's sampling.cuh:65-72
// SM >= 8.0 (Ampere+): BLOCK_THREADS = 1024
// SM < 8.0: BLOCK_THREADS = 512
#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...) \
  if (compute_capacity.first >= 8) {                                           \
    constexpr uint32_t BLOCK_THREADS = 1024;                                   \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    constexpr uint32_t BLOCK_THREADS = 512;                                    \
    __VA_ARGS__                                                                \
  }

// RenormTempStorage matching FlashInfer's sampling.cuh:1519-1543
template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM>
struct RenormTempStorage {
  union {
    typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
  } block_prim;
  struct {
    float max_val;
    float min_val;
    union {
      struct {
        float values[2];
      };
    } block_aggregate;
  };
};

// GetMaxValue matching FlashInfer's sampling.cuh:248-277
template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM>
__device__ __forceinline__ float GetMaxValue(const float* in_data, uint32_t row_idx, uint32_t d,
                                             RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>& temp_storage) {
  const uint32_t tx = threadIdx.x;
  vec_t<VEC_SIZE> in_data_vec;

  // Thread-local max accumulation (deferred reduction)
  float thread_max = 0.0f;
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_data_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      in_data_vec.load(in_data + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_max = max(thread_max, in_data_vec[j]);
    }
  }

  // Single block reduction after loop completes
  float max_val =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
          .Reduce(thread_max, MaxReduceOp{});
  if (tx == 0) {
    temp_storage.max_val = max_val;
  }
  __syncthreads();
  return temp_storage.max_val;
}

// TopPMaskKernel based on FlashInfer's TopPRenormProbKernel (sampling.cuh:1545-1715)
// Modified to output boolean mask instead of renormalized probs
template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE>
__global__ void TopPMaskKernel(const float* probs, bool* mask, float top_p_val, uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = bx;
  float p = top_p_val;

  extern __shared__ __align__(alignof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>))
      uint8_t smem_renorm[];
  auto& temp_storage =
      reinterpret_cast<RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>&>(smem_renorm);
  vec_t<VEC_SIZE> probs_vec;

  // Fast-path: when p >= 1.0, all tokens are selected (sampling.cuh:1559-1608 simplified)
  if (p >= 1.0f) {
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          uint32_t idx = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
          if (idx < d) {
            mask[row_idx * d + idx] = true;
          }
        }
      }
    }
    return;
  }

  // Original Top-P renormalization logic (sampling.cuh:1610-1714)
  temp_storage.max_val = 0;
  float max_val = GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM>(probs, row_idx, d, temp_storage);

  double low = 0, high = max_val;
  float min_gt_low, max_le_high;
  float sum_low = 1;
  // f(x) = sum(probs[probs > x]), f(x) is non-increasing
  // min_gt_low = min{p \in probs | p > low}, max_le_high = max{p \in probs | p <= high}
  // loop invariant:
  // - f(low) >= p, f(high) < p
  // - f(low) > f(min_gt_low) >= f(max_le_high) == f(high)
  // stopping condition
  // - f(low) >= p, f(min_gt_low) == f(max_le_high) == f(high) < p
  do {
    double pivot_0 = (high + 2 * low) / 3;
    double pivot_1 = (2 * high + low) / 3;

    float aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
    min_gt_low = high;
    max_le_high = low;
    float threadlocal_aggregate_gt_pivot_0 = 0;
    float threadlocal_aggregate_gt_pivot_1 = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      float probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
        probs_gt_pivot_1[j] = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;

        if (probs_vec[j] > low && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
          min_gt_low = min(min_gt_low, probs_vec[j]);
        }
        if (probs_vec[j] <= high && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
          max_le_high = max(max_le_high, probs_vec[j]);
        }
        threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Sum(threadlocal_aggregate_gt_pivot_0);
    __syncthreads();
    aggregate_gt_pivot_1 =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Sum(threadlocal_aggregate_gt_pivot_1);
    __syncthreads();

    min_gt_low = BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                     .Reduce(min_gt_low, MinReduceOp{});
    __syncthreads();
    max_le_high =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .Reduce(max_le_high, MaxReduceOp{});
    if (tx == 0) {
      temp_storage.block_aggregate.values[0] = aggregate_gt_pivot_0;
      temp_storage.block_aggregate.values[1] = aggregate_gt_pivot_1;
      temp_storage.min_val = min_gt_low;
      temp_storage.max_val = max_le_high;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.values[0];
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.values[1];
    min_gt_low = temp_storage.min_val;
    max_le_high = temp_storage.max_val;

    if (aggregate_gt_pivot_1 >= p) {
      low = pivot_1;
      sum_low = aggregate_gt_pivot_1;
    } else if (aggregate_gt_pivot_0 >= p) {
      low = pivot_0;
      high = min(pivot_1, max_le_high);
      sum_low = aggregate_gt_pivot_0;
    } else {
      high = min(pivot_0, max_le_high);
    }
  } while (min_gt_low != max_le_high);

  // Output mask: probs > low (instead of normalize in sampling.cuh:1697-1714)
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      uint32_t idx = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
      if (idx < d) {
        mask[row_idx * d + idx] = (probs_vec[j] > low);
      }
    }
  }
}

}  // namespace vllm

// top_p_mask based on FlashInfer's TopPRenormProb (sampling.cuh:1717-1737)
torch::Tensor top_p_mask(torch::Tensor probs, double top_p) {
  TORCH_CHECK(probs.is_cuda(), "probs must be on CUDA device");
  TORCH_CHECK(probs.dim() >= 1, "probs must have at least 1 dimension");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(probs));

  // Flatten to 2D
  auto original_shape = probs.sizes().vec();
  int64_t d = probs.size(-1);
  int64_t batch_size = probs.numel() / d;
  auto probs_2d = probs.view({batch_size, d}).contiguous();

  // Allocate output
  auto mask = torch::empty({batch_size, d}, probs.options().dtype(torch::kBool));

  // Match FlashInfer's vec_size calculation: std::gcd(16 / sizeof(DType), d)
  // For float (4 bytes): 16/4 = 4, so vec_size = gcd(4, d), max is 4
  const uint32_t vec_size = std::gcd(16u / static_cast<uint32_t>(sizeof(float)),
                                     static_cast<uint32_t>(d));

  // Match FlashInfer's DISPATCH_COMPUTE_CAP_NUM_THREADS pattern
  auto compute_capacity = vllm::GetCudaComputeCapability();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(vllm::RenormTempStorage<BLOCK_THREADS, vllm::REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);

    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
      auto kernel = vllm::TopPMaskKernel<BLOCK_THREADS, vllm::REDUCE_ALGO, VEC_SIZE>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      kernel<<<nblks, nthrs, smem_size, stream>>>(
          probs_2d.data_ptr<float>(),
          mask.data_ptr<bool>(),
          static_cast<float>(top_p),
          static_cast<uint32_t>(d));
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Reshape to original shape
  return mask.view(original_shape);
}
