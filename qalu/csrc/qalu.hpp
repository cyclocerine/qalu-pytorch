#ifndef QALU_HPP
#define QALU_HPP

/**
 * Q-ALU: Quant Asymmetric Leaky Unit - C++ High Performance Header
 * Header-only library untuk sistem High-Frequency Trading (HFT) sub-mikrodetik.
 */

#include <cmath>
#include <algorithm>
#include <cstddef>

namespace qalu {

// Q-ALU Canonical Forward (Stable FP32 via expm1)
inline float forward(float x, float alpha = 1.0f, float beta = 1.0f, float gamma = 1.5f, float delta = 0.5f) {
    if (x >= 0.0f) {
        return alpha * x * (1.0f + std::tanh(x / beta));
    } else {
        return -gamma * x * std::expm1(x / delta);
    }
}

// Q-ALU Canonical Backward (dL/dx)
inline float backward(float x, float alpha = 1.0f, float beta = 1.0f, float gamma = 1.5f, float delta = 0.5f) {
    if (x >= 0.0f) {
        float u = x / beta;
        float t = std::tanh(u);
        float sech2 = 1.0f - t * t;
        return alpha * (1.0f + t + u * sech2);
    } else {
        float u = x / delta;
        float exp_u = std::exp(u);
        return -gamma * (std::expm1(u) + u * exp_u);
    }
}

inline float forward_v1(float x, float alpha = 1.0f, float beta = 1.0f, float gamma = 1.5f, float delta = 0.5f) {
    return forward(x, alpha, beta, gamma, delta);
}

inline float backward_v1(float x, float alpha = 1.0f, float beta = 1.0f, float gamma = 1.5f, float delta = 0.5f) {
    return backward(x, alpha, beta, gamma, delta);
}

// Q-ALU v2.0 Forward (C1 Smooth & Symmetrical Deadband)
inline float forward_v2(float x, float alpha = 1.0f, float beta = 1.0f, float gamma = 2.5f, float delta = 0.5f) {
    if (x >= 0.0f) {
        float u = x / beta;
        float t = std::tanh(u);
        return alpha * x * (t + t * t);
    } else {
        float u = x / delta;
        return -gamma * x * std::expm1(u);
    }
}

// Q-ALU v2.0 Backward (dL/dx)
inline float backward_v2(float x, float alpha = 1.0f, float beta = 1.0f, float gamma = 2.5f, float delta = 0.5f) {
    if (x >= 0.0f) {
        float u = x / beta;
        float t = std::tanh(u);
        float sech2 = 1.0f - t * t;
        return alpha * (t + t * t + u * sech2 * (1.0f + 2.0f * t));
    } else {
        float u = x / delta;
        float exp_u = std::exp(u);
        return -gamma * (std::expm1(u) + u * exp_u);
    }
}

// Q-ALU Ultra (Adaptive Volatility & Fee-Aware)
inline float forward_ultra(float x, float vol_ratio, float fee_threshold = 0.01f, 
                           float alpha = 1.0f, float beta = 1.0f, float gamma = 2.5f, float delta = 0.5f) {
    float eff_beta = (beta + fee_threshold) * vol_ratio;
    float eff_delta = (delta + fee_threshold) * vol_ratio;
    if (x >= 0.0f) {
        float u = x / eff_beta;
        float t = std::tanh(u);
        return alpha * x * (t + t * t);
    } else {
        float u = x / eff_delta;
        return -gamma * x * std::expm1(u);
    }
}

// FastQALU (Non-transcendental Algebraic Rational Approximation)
inline float forward_fast(float x, float alpha = 1.0f, float beta = 1.0f, float gamma = 1.5f, float delta = 0.5f) {
    if (x >= 0.0f) {
        float u = x / beta;
        float t = u / std::sqrt(1.0f + u * u);
        return alpha * x * (1.0f + t);
    } else {
        float u = -x / delta;
        float u2 = u * u;
        float num = u2 + 2.0f * u;
        float den = u2 + 3.0f * u + 2.5f;
        return gamma * x * (num / den);
    }
}

// FastQALU Canonical Backward (dL/dx)
inline float backward_fast(float x, float alpha = 1.0f, float beta = 1.0f, float gamma = 1.5f, float delta = 0.5f) {
    if (x >= 0.0f) {
        float u = x / beta;
        float inv_s = 1.0f / std::sqrt(1.0f + u * u);
        float t = u * inv_s;
        float dt_du = inv_s * inv_s * inv_s; // 1 / (1 + u^2)^(3/2)
        return alpha * (1.0f + t + u * dt_du);
    } else {
        // u = -x / delta => x = -delta * u, dx/du = -delta
        // f(x) = -gamma * delta * (u^3 + 2u^2) / (u^2 + 3u + 2.5)
        // df/dx = (-1/delta) * df/du = gamma * d/du [(u^3 + 2u^2) / (u^2 + 3u + 2.5)]
        float u = -x / delta;
        float num = u * u * u + 2.0f * u * u;
        float den = u * u + 3.0f * u + 2.5f;
        float dnum = 3.0f * u * u + 4.0f * u;
        float dden = 2.0f * u + 3.0f;
        float d_ratio = (dnum * den - num * dden) / (den * den);
        return gamma * d_ratio;
    }
}

// Batch vectorised processor for CPU
inline void batch_forward_v2(const float* __restrict__ input, float* __restrict__ output, size_t n,
                             float alpha = 1.0f, float beta = 1.0f, float gamma = 2.5f, float delta = 0.5f) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        output[i] = forward_v2(input[i], alpha, beta, gamma, delta);
    }
}

} // namespace qalu

#endif // QALU_HPP
