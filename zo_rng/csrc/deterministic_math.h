/*
 * deterministic_math.h — Cross-device deterministic math functions
 *
 * Provides det_logf, det_sqrtf, and det_sincos_2pi using ONLY IEEE 754
 * float32 operations (+, -, *, /) to guarantee bit-exact results across
 * CPU and GPU.
 *
 * Included by BOTH generate_cpu.c (compiled as C) and generate_cuda.cu
 * (compiled as CUDA C++).
 */
#ifndef DETERMINISTIC_MATH_H
#define DETERMINISTIC_MATH_H

#include <stdint.h>

#ifdef __CUDACC__
#define DET_FUNC __device__ __forceinline__
#else
#define DET_FUNC static inline
#endif

/* ---- IEEE 754 float32 bit manipulation ---- */

typedef union {
    float f;
    uint32_t u;
} det_float_bits_t;

/* ---- Deterministic sqrtf ----
 * Uses fast inverse sqrt (Quake III style) + Newton-Raphson iterations.
 * Only uses float32 *, -, operations (no libm/CUDA intrinsics).
 */
DET_FUNC float det_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;

    det_float_bits_t conv;
    conv.f = x;
    /* Fast inverse sqrt initial estimate (~8 correct bits) */
    conv.u = 0x5F3759DFu - (conv.u >> 1);
    float y = conv.f; /* y ≈ 1/sqrt(x) */

    /* Newton-Raphson for 1/sqrt(x): y = y * (1.5 - 0.5*x*y*y)
     * Each iteration doubles the number of correct bits.
     * 3 iterations: ~8 -> ~16 -> ~32 -> ~64 bits (well beyond float32's 24) */
    float xhalf = 0.5f * x;
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);

    return x * y; /* sqrt(x) = x * rsqrt(x) */
}

/* ---- Deterministic logf (natural logarithm) ----
 * For x > 0. Uses:
 *   1. Bit manipulation to decompose x = 2^e * m, m in [sqrt(2)/2, sqrt(2)]
 *   2. Series: log(m) = 2*t*(1 + t^2/3 + t^4/5 + t^6/7 + t^8/9)
 *      where t = (m - 1) / (m + 1), |t| < 0.172
 *   3. log(x) = log(m) + e * ln(2)
 */
DET_FUNC float det_logf(float x) {
    det_float_bits_t bits;
    bits.f = x;

    /* Extract exponent and set mantissa to [1, 2) */
    int e = (int)((bits.u >> 23) & 0xFFu) - 127;
    bits.u = (bits.u & 0x007FFFFFu) | 0x3F800000u;
    float m = bits.f; /* m in [1, 2) */

    /* Adjust range to [sqrt(2)/2, sqrt(2)] for faster convergence */
    if (m > 1.41421356f) {
        m = m * 0.5f;
        e = e + 1;
    }

    /* t = (m - 1) / (m + 1), |t| < 0.172 */
    float t = (m - 1.0f) / (m + 1.0f);
    float t2 = t * t;

    /* Horner evaluation of 1 + t^2/3 + t^4/5 + t^6/7 + t^8/9 */
    float poly = 0.111111111f; /* 1/9 */
    poly = poly * t2 + 0.142857143f; /* 1/7 */
    poly = poly * t2 + 0.200000000f; /* 1/5 */
    poly = poly * t2 + 0.333333333f; /* 1/3 */
    poly = poly * t2 + 1.000000000f;

    float log_m = 2.0f * t * poly;

    /* Cody-Waite: use two-part ln(2) for better accuracy
     * LN2_HI is exactly representable; LN2_LO = ln(2) - LN2_HI */
    float e_f = (float)e;
    float result = log_m + e_f * 1.4286068e-6f;  /* e * LN2_LO */
    result = result + e_f * 0.6931457519531250f;  /* e * LN2_HI */
    return result;
}

/* ---- Polynomial approximations for sin/cos on [0, pi/2] ----
 * Taylor series with enough terms for < 2^-22 error on [0, pi/2].
 */

/* sin(x) ≈ x * (1 - x^2/6 + x^4/120 - x^6/5040 + x^8/362880 - x^10/39916800) */
DET_FUNC float det_sin_poly(float x) {
    float x2 = x * x;
    float p = -2.50521084e-8f;  /* -1/39916800 */
    p = p * x2 + 2.75573192e-6f;  /* 1/362880 */
    p = p * x2 + (-1.98412698e-4f);  /* -1/5040 */
    p = p * x2 + 8.33333333e-3f;  /* 1/120 */
    p = p * x2 + (-1.66666667e-1f);  /* -1/6 */
    p = p * x2 + 1.0f;
    return x * p;
}

/* cos(x) ≈ 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320 - x^10/3628800 + x^12/479001600 */
DET_FUNC float det_cos_poly(float x) {
    float x2 = x * x;
    float p = 2.08767570e-9f;  /* 1/479001600 */
    p = p * x2 + (-2.75573192e-7f);  /* -1/3628800 */
    p = p * x2 + 2.48015873e-5f;  /* 1/40320 */
    p = p * x2 + (-1.38888889e-3f);  /* -1/720 */
    p = p * x2 + 4.16666667e-2f;  /* 1/24 */
    p = p * x2 + (-5.00000000e-1f);  /* -1/2 */
    p = p * x2 + 1.0f;
    return p;
}

/* ---- sin(2*pi*u) and cos(2*pi*u) for u in [0, 1) ----
 * Uses quadrant-based range reduction via u (not theta) to
 * avoid precision loss from dividing by pi/2.
 *
 * Quadrant j = floor(4*u):
 *   j=0: sin = sin_poly(theta), cos = cos_poly(theta)
 *   j=1: sin = cos_poly(theta), cos = -sin_poly(theta)
 *   j=2: sin = -sin_poly(theta), cos = -cos_poly(theta)
 *   j=3: sin = -cos_poly(theta), cos = sin_poly(theta)
 * where theta = (u - j*0.25) * 2*pi, in [0, pi/2).
 */
DET_FUNC void det_sincos_2pi(float u, float *sin_val, float *cos_val) {
    float u4 = u * 4.0f;
    int j = (int)u4;
    if (j < 0) j = 0;
    if (j > 3) j = 3;

    float u_red = u - (float)j * 0.25f;
    /* theta_red = u_red * 2*pi, in [0, pi/2)
     * Use two-part 2*pi for accuracy:
     *   TWO_PI_HI = 6.28125 (exactly representable: 6 + 9/32)
     *   TWO_PI_LO = 2*pi - 6.28125 ≈ -0.00006469... */
    float theta_red = u_red * 6.28125f + u_red * (-6.46918e-5f);

    float s = det_sin_poly(theta_red);
    float c = det_cos_poly(theta_red);

    switch (j) {
    case 0: *sin_val = s;  *cos_val = c;  break;
    case 1: *sin_val = c;  *cos_val = -s; break;
    case 2: *sin_val = -s; *cos_val = -c; break;
    default: *sin_val = -c; *cos_val = s;  break;
    }
}

/* ---- Box-Muller: 4 × uint32 -> 4 × normal float ---- */
DET_FUNC void uint32x4_to_normal4(uint32_t raw[4], float out[4]) {
    /* Convert to uniform (0, 1) by mapping 24-bit integer to float.
     * raw >> 8 gives [0, 2^24 - 1]; multiply by 2^-24. */
    float u0 = (float)(raw[0] >> 8) * 5.96046448e-8f; /* 0x1.0p-24 */
    float u1 = (float)(raw[1] >> 8) * 5.96046448e-8f;
    float u2 = (float)(raw[2] >> 8) * 5.96046448e-8f;
    float u3 = (float)(raw[3] >> 8) * 5.96046448e-8f;

    /* Clamp radial uniforms away from 0 (avoids log(0) = -inf) */
    if (u0 < 5.96046448e-8f) u0 = 5.96046448e-8f; /* min = 2^-24 */
    if (u2 < 5.96046448e-8f) u2 = 5.96046448e-8f;

    /* Box-Muller pair 1 */
    float r0 = det_sqrtf(-2.0f * det_logf(u0));
    float sin0, cos0;
    det_sincos_2pi(u1, &sin0, &cos0);
    out[0] = r0 * cos0;
    out[1] = r0 * sin0;

    /* Box-Muller pair 2 */
    float r1 = det_sqrtf(-2.0f * det_logf(u2));
    float sin1, cos1;
    det_sincos_2pi(u3, &sin1, &cos1);
    out[2] = r1 * cos1;
    out[3] = r1 * sin1;
}

#endif /* DETERMINISTIC_MATH_H */
