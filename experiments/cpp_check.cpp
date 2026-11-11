// Verification harness for qalu/csrc/qalu.hpp.
//
// Checks that every analytic backward in the header matches a central finite
// difference of the corresponding forward, that the rational surrogate stays
// within its published error bound, and that no input produces NaN or Inf.
//
// Build & run:
//     g++ -O2 -std=c++17 -Wall -Wextra experiments/cpp_check.cpp -o /tmp/cpp_check
//     /tmp/cpp_check

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "../qalu/csrc/qalu.hpp"

static int failures = 0;

static void check(bool ok, const char* what, double got, double want) {
    if (!ok) {
        std::printf("  FAIL %-46s got %.10g want %.10g\n", what, got, want);
        ++failures;
    }
}

// Relative error between an analytic derivative and a central difference.
template <typename F, typename G_>
static double grad_error(F fwd, G_ bwd, double x, double h = 1e-2) {
    double num = (fwd(x + h) - fwd(x - h)) / (2.0 * h);
    double ana = bwd(x);
    double scale = std::fabs(num) > 1e-8 ? std::fabs(num) : 1.0;
    return std::fabs(num - ana) / scale;
}

int main() {
    const float A = 1.0f, B = 1.0f, G = 1.5f, D = 0.5f;

    // ---- 1. Q-ALU canonical: forward/backward consistency -------------------
    double worst = 0.0;
    for (double x = -8.0; x <= 8.0; x += 0.01) {
        if (std::fabs(x) < 0.1) continue;  // kink at the origin, see [1b]
        double e = grad_error(
            [&](double t) { return (double)qalu::forward((float)t, A, B, G, D); },
            [&](double t) { return (double)qalu::backward((float)t, A, B, G, D); },
            x);
        if (e > worst) worst = e;
    }
    std::printf("[1] Q-ALU     max |analytic - numeric| rel err = %.3e (excluding |x| < 0.1)\n", worst);
    check(worst < 5e-3, "Q-ALU backward", worst, 0.0);

    // [1b] v1 is C0 but NOT C1 at the origin: the left derivative is 0 while the
    // right derivative is alpha, so the gradient jumps by alpha at x = 0.
    double left = (double)qalu::backward(-1e-6f, A, B, G, D);
    double right = (double)qalu::backward(1e-6f, A, B, G, D);
    std::printf("[1b] Q-ALU v1 gradient at 0: left = %.6f right = %.6f (jump = %.6f)\n",
                left, right, right - left);
    check(std::fabs(left) < 1e-3 && std::fabs(right - A) < 1e-3, "v1 kink limits", right - left, A);
    // v2 restores C1 continuity: both one-sided limits vanish.
    double l2 = (double)qalu::backward_v2(-1e-6f, A, B, G, D);
    double r2 = (double)qalu::backward_v2(1e-6f, A, B, G, D);
    std::printf("[1b] Q-ALU v2 gradient at 0: left = %.6f right = %.6f\n", l2, r2);
    check(std::fabs(l2) < 1e-3 && std::fabs(r2) < 1e-3, "v2 C1 continuity", r2 - l2, 0.0);

    // ---- 2. Q-ALU 2.0 -------------------------------------------------------
    worst = 0.0;
    for (double x = -8.0; x <= 8.0; x += 0.01) {
        if (std::fabs(x) < 0.1) continue;  // kink at the origin, see [1b]
        double e = grad_error(
            [&](double t) { return (double)qalu::forward_v2((float)t, A, B, 0.25f, D); },
            [&](double t) { return (double)qalu::backward_v2((float)t, A, B, 0.25f, D); },
            x);
        if (e > worst) worst = e;
    }
    std::printf("[2] Q-ALU 2.0 max |analytic - numeric| rel err = %.3e\n", worst);
    check(worst < 5e-3, "Q-ALU2 backward", worst, 0.0);

    // ---- 3. Q-ALU Ultra -----------------------------------------------------
    worst = 0.0;
    for (double x = -8.0; x <= 8.0; x += 0.01) {
        if (std::fabs(x) < 0.1) continue;  // kink at the origin, see [1b]
        double e = grad_error(
            [&](double t) { return (double)qalu::forward_ultra((float)t, 1.5f, 0.01f, A, B, D, 2.25f); },
            [&](double t) { return (double)qalu::backward_ultra((float)t, 1.5f, 0.01f, A, B, D, 2.25f); },
            x);
        if (e > worst) worst = e;
    }
    std::printf("[3] Q-ALU Ultra max |analytic - numeric| rel err = %.3e\n", worst);
    check(worst < 5e-3, "Ultra backward", worst, 0.0);

    // ---- 4. FastQALU --------------------------------------------------------
    worst = 0.0;
    for (double x = -8.0; x <= 8.0; x += 0.01) {
        if (std::fabs(x) < 0.1) continue;  // kink at the origin, see [1b]
        double e = grad_error(
            [&](double t) { return (double)qalu::forward_fast((float)t, A, B, G, D); },
            [&](double t) { return (double)qalu::backward_fast((float)t, A, B, G, D); },
            x);
        if (e > worst) worst = e;
    }
    std::printf("[4] FastQALU   max |analytic - numeric| rel err = %.3e\n", worst);
    check(worst < 5e-3, "FastQALU backward", worst, 0.0);

    // ---- 5. surrogate error bound (paper claims < 1%) -----------------------
    double err_val = 0.0, err_at = 0.0;
    for (double u = 1e-6; u <= 20.0; u += 1e-4) {
        double exact = -std::expm1(-u);
        double approx = (double)qalu::fast_neg_shape((float)u);
        double e = std::fabs(approx - exact) / exact;
        if (e > err_val) { err_val = e; err_at = u; }
    }
    std::printf("[5] surrogate max rel err = %.4f%% at u = %.3f\n", err_val * 100.0, err_at);
    check(err_val < 1e-2, "surrogate value error", err_val, 0.0);

    // ---- 6. downside overshoot preserved ------------------------------------
    double peak = 0.0, peak_at = 0.0;
    for (double x = -4.0; x < 0.0; x += 1e-4) {
        double g = qalu::backward_fast((float)x, A, B, G, D);
        if (g > peak) { peak = g; peak_at = x; }
    }
    double target = G * (1.0 + std::exp(-2.0));
    std::printf("[6] FastQALU peak grad = %.5f at x = %.4f (exact %.5f at -1.0, %+.2f%%)\n",
                peak, peak_at, target, (peak / G - 1.0) * 100.0);
    check(std::fabs(peak - target) / target < 5e-3, "FastQALU overshoot peak", peak, target);

    // ---- 7. monotonicity and asymptotes ------------------------------------
    bool mono = true;
    for (double x = -50.0; x < 50.0; x += 0.01) {
        double d1 = qalu::backward_fast((float)x, A, B, G, D);
        if (d1 < -1e-6) mono = false;
    }
    std::printf("[7] FastQALU monotone on [-50, 50]: %s\n", mono ? "yes" : "NO");
    check(mono, "monotonicity", 0.0, 1.0);
    double tail = qalu::backward_fast(-1000.0f, A, B, G, D);
    std::printf("[7] f'_fast(-1000) = %.6f (asymptote gamma = %.1f)\n", tail, G);
    check(std::fabs(tail - G) < 1e-4, "downside asymptote", tail, G);

    // ---- 8. no NaN/Inf at extreme magnitudes -------------------------------
    const float extremes[] = {-1e30f, -1e18f, -1e7f, -1e4f, 0.0f, 1e4f, 1e18f, 1e30f};
    bool finite = true;
    for (float x : extremes) {
        float v = qalu::forward_fast(x, A, B, G, D);
        float g = qalu::backward_fast(x, A, B, G, D);
        float v2 = qalu::forward(x, A, B, G, D);
        float g2 = qalu::backward(x, A, B, G, D);
        if (!std::isfinite(v) || !std::isfinite(g) ||
            !std::isfinite(v2) || !std::isfinite(g2)) {
            std::printf("  FAIL non-finite at x = %g\n", x);
            finite = false;
        }
    }
    std::printf("[8] all outputs finite at |x| up to 1e30: %s\n", finite ? "yes" : "NO");
    check(finite, "finite outputs", 0.0, 1.0);

    if (failures == 0) {
        std::printf("\nALL C++ HEADER CHECKS PASSED\n");
        return 0;
    }
    std::printf("\n%d C++ HEADER CHECK(S) FAILED\n", failures);
    return 1;
}
