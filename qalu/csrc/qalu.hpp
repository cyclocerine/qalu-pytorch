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
// gamma follows the Python reference: gamma = 2 * alpha * (1 + loss_aversion).
inline float forward_ultra(float x, float vol_ratio, float fee_threshold = 0.01f,
                           float alpha = 1.0f, float beta = 1.0f, float delta = 0.5f,
                           float loss_aversion = 2.25f) {
    float eff_beta = (beta + fee_threshold) * vol_ratio;
    float eff_delta = (delta + fee_threshold) * vol_ratio;
    float gamma = 2.0f * alpha * (1.0f + loss_aversion);
    if (x >= 0.0f) {
        float u = x / eff_beta;
        float t = std::tanh(u);
        return alpha * x * (t + t * t);
    } else {
        float u = x / eff_delta;
        return -gamma * x * std::expm1(u);
    }
}

// Q-ALU Ultra Backward (dL/dx), same conditioning as forward_ultra
inline float backward_ultra(float x, float vol_ratio, float fee_threshold = 0.01f,
                            float alpha = 1.0f, float beta = 1.0f, float delta = 0.5f,
                            float loss_aversion = 2.25f) {
    float eff_beta = (beta + fee_threshold) * vol_ratio;
    float eff_delta = (delta + fee_threshold) * vol_ratio;
    float gamma = 2.0f * alpha * (1.0f + loss_aversion);
    if (x >= 0.0f) {
        float u = x / eff_beta;
        float t = std::tanh(u);
        float sech2 = 1.0f - t * t;
        return alpha * (t + t * t + u * sech2 * (1.0f + 2.0f * t));
    } else {
        float u = x / eff_delta;
        return -gamma * (std::expm1(u) + u * std::exp(u));
    }
}

// Rational surrogate of 1 - exp(-u) for u >= 0 (see paper, Eq. 12).
// [4/4] Pade-type approximant, r(0)=0, r'(0)=1, r(inf)=1, denominator >= 1.
inline float fast_neg_shape(float u) {
    float n = u * (1.0f + u * (0.55f + u * (0.05f + 0.10f * u)));
    return n / (1.0f + n);
}

// Saturation point: for u >= 1e4 the surrogate is 1.0 to within FP32 resolution.
// Clamping there stops u^4 from overflowing to +inf (inf/inf = NaN) for |x| > 1e7.
#define QALU_FAST_U_MAX 1.0e4f

// FastQALU (Non-transcendental Algebraic Rational Approximation)
inline float forward_fast(float x, float alpha = 1.0f, float beta = 1.0f, float gamma = 1.5f, float delta = 0.5f) {
    if (x >= 0.0f) {
        float u = x / beta;
        float t = u / std::sqrt(1.0f + u * u);
        return alpha * x * (1.0f + t);
    } else {
        float u = -x / delta;
        if (u > QALU_FAST_U_MAX) u = QALU_FAST_U_MAX;
        return gamma * x * fast_neg_shape(u);
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
        // u = -x/delta, r = n/(1+n) => df/dx = gamma * (r + u * r')
        // with r' = (n'/n) * r * (1-r) and p = n/u, q = n':
        //     df/dx = gamma * (r + (q/p) * r * (1-r))
        // Overflow-free: n*(1+n) would exceed FP32 range for u > ~1e5.
        float u = -x / delta;
        if (u > QALU_FAST_U_MAX) u = QALU_FAST_U_MAX;
        float n = u * (1.0f + u * (0.55f + u * (0.05f + 0.10f * u)));
        float p = 1.0f + u * (0.55f + u * (0.05f + 0.10f * u));  // n / u
        float q = 1.0f + u * (1.1f + u * (0.15f + 0.4f * u));    // n'
        float r = n / (1.0f + n);
        return gamma * (r + (q / p) * r * (1.0f - r));
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
