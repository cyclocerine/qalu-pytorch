#include <cuda_runtime.h>
#include <math.h>

/**
 * Q-ALU CUDA Kernels
 * Optimized for NVIDIA Tensor Core architectures (Ampere / Ada Lovelace / Blackwell / RTX 5090).
 */

// =========================================================================
// FORWARD KERNEL: Q-ALU Canonical (FP32)
// =========================================================================
__global__ void qalu_forward_kernel_fp32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size,
    float alpha,
    float beta,
    float gamma,
    float delta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        if (x >= 0.0f) {
            float u = x / beta;
            float t = tanhf(u);
            output[idx] = alpha * x * (1.0f + t);
        } else {
            float u = x / delta;
            output[idx] = -gamma * x * expm1f(u);
        }
    }
}

// =========================================================================
// BACKWARD KERNEL: Q-ALU Canonical (FP32)
// =========================================================================
__global__ void qalu_backward_kernel_fp32(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int size,
    float alpha,
    float beta,
    float gamma,
    float delta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float dy = grad_output[idx];
        float dx = 0.0f;

        if (x >= 0.0f) {
            float u = x / beta;
            float t = tanhf(u);
            float sech2 = 1.0f - t * t;
            dx = alpha * (1.0f + t + u * sech2);
        } else {
            float u = x / delta;
            float exp_u = expf(u);
            dx = -gamma * (expm1f(u) + u * exp_u);
        }
        grad_input[idx] = dy * dx;
    }
}

// =========================================================================
// FORWARD KERNEL: Q-ALU Variant (FP32)
// =========================================================================
__global__ void qalu2_forward_kernel_fp32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size,
    float alpha,
    float beta,
    float gamma,
    float delta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        if (x >= 0.0f) {
            float u = x / beta;
            float t = tanhf(u);
            output[idx] = alpha * x * (t + t * t);
        } else {
            float u = x / delta;
            output[idx] = -gamma * x * expm1f(u);
        }
    }
}

// =========================================================================
// BACKWARD KERNEL: Q-ALU 2.0 (FP32)
// =========================================================================
__global__ void qalu2_backward_kernel_fp32(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int size,
    float alpha,
    float beta,
    float gamma,
    float delta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float dy = grad_output[idx];
        float dx = 0.0f;

        if (x >= 0.0f) {
            float u = x / beta;
            float t = tanhf(u);
            float sech2 = 1.0f - t * t;
            dx = alpha * (t + t * t + u * sech2 * (1.0f + 2.0f * t));
        } else {
            float u = x / delta;
            float exp_u = expf(u);
            dx = -gamma * (expm1f(u) + u * exp_u);
        }
        grad_input[idx] = dy * dx;
    }
}

// =========================================================================
// FORWARD KERNEL: Q-ALU Ultra (Dynamic Volatility Conditioning)
// =========================================================================
__global__ void qalu_ultra_forward_kernel_fp32(
    const float* __restrict__ input,
    const float* __restrict__ vol_ratios, // Rasio volatilitas per sampel/batch
    float* __restrict__ output,
    int size,
    float alpha,
    float beta,
    float delta,
    float fee_threshold,
    float loss_aversion)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float v_ratio = (vol_ratios != nullptr) ? vol_ratios[idx] : 1.0f;
        float eff_beta = (beta + fee_threshold) * v_ratio;
        float eff_delta = (delta + fee_threshold) * v_ratio;
        float gamma = 2.0f * alpha * loss_aversion;

        if (x >= 0.0f) {
            float u = x / eff_beta;
            float t = tanhf(u);
            output[idx] = alpha * x * (t + t * t);
        } else {
            float u = x / eff_delta;
            output[idx] = -gamma * x * expm1f(u);
        }
    }
}

// =========================================================================
// FORWARD KERNEL: FastQALU Rational Approximation (FP32)
// =========================================================================
__global__ void fast_qalu_forward_kernel_fp32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size,
    float alpha,
    float beta,
    float gamma,
    float delta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        if (x >= 0.0f) {
            float u = x / beta;
            float t = u * rsqrtf(1.0f + u * u);
            output[idx] = alpha * x * (1.0f + t);
        } else {
            float u = -x / delta;
            float u2 = u * u;
            float num = u2 + 2.0f * u;
            float den = u2 + 3.0f * u + 2.5f;
            output[idx] = gamma * x * (num / den);
        }
    }
}

// =========================================================================
// BACKWARD KERNEL: FastQALU Rational Approximation (FP32)
// =========================================================================
__global__ void fast_qalu_backward_kernel_fp32(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int size,
    float alpha,
    float beta,
    float gamma,
    float delta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float dy = grad_output[idx];
        float dx = 0.0f;

        if (x >= 0.0f) {
            float u = x / beta;
            float inv_s = rsqrtf(1.0f + u * u);
            float t = u * inv_s;
            float dt_du = inv_s * inv_s * inv_s;
            dx = alpha * (1.0f + t + u * dt_du);
        } else {
            float u = -x / delta;
            float num = u * u * u + 2.0f * u * u;
            float den = u * u + 3.0f * u + 2.5f;
            float dnum = 3.0f * u * u + 4.0f * u;
            float dden = 2.0f * u + 3.0f;
            float d_ratio = (dnum * den - num * dden) / (den * den);
            dx = gamma * d_ratio;
        }
        grad_input[idx] = dy * dx;
    }
}
