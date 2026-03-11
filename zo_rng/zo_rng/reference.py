"""Pure-Python reference implementation of zo_rng.

Implements Philox4x32-10 + deterministic Box-Muller using the same
algorithm as the C/CUDA kernels. Useful for cross-checking correctness.

All arithmetic is done in Python integers (for Philox) and numpy float32
(for Box-Muller) to match the C implementation bit-for-bit.
"""

import numpy as np
import torch

# Philox4x32 constants
PHILOX_M4x32_0 = 0xD2511F53
PHILOX_M4x32_1 = 0xCD9E8D57
PHILOX_W32_0 = 0x9E3779B9
PHILOX_W32_1 = 0xBB67AE85

MASK32 = 0xFFFFFFFF


def _mulhilo32(a, b):
    """Multiply two uint32 values, return (lo, hi) as Python ints."""
    product = a * b  # Python int: no overflow
    lo = product & MASK32
    hi = (product >> 32) & MASK32
    return lo, hi


def _philox4x32_round(ctr, key):
    """One round of Philox4x32."""
    lo0, hi0 = _mulhilo32(PHILOX_M4x32_0, ctr[0])
    lo1, hi1 = _mulhilo32(PHILOX_M4x32_1, ctr[2])
    return (
        (hi1 ^ ctr[1] ^ key[0]) & MASK32,
        lo1,
        (hi0 ^ ctr[3] ^ key[1]) & MASK32,
        lo0,
    )


def _philox4x32_bumpkey(key):
    """Bump the Philox key."""
    return (
        (key[0] + PHILOX_W32_0) & MASK32,
        (key[1] + PHILOX_W32_1) & MASK32,
    )


def philox4x32_10(counter, key):
    """Philox4x32-10: 10-round counter-based PRNG.

    Args:
        counter: tuple of 4 uint32 values.
        key: tuple of 2 uint32 values.

    Returns:
        tuple of 4 uint32 values (the random output).
    """
    ctr = tuple(c & MASK32 for c in counter)
    k = (key[0] & MASK32, key[1] & MASK32)

    # Round 1 (no key bump)
    ctr = _philox4x32_round(ctr, k)

    # Rounds 2-10 (bump key before each)
    for _ in range(9):
        k = _philox4x32_bumpkey(k)
        ctr = _philox4x32_round(ctr, k)

    return ctr


# ---- Deterministic math in numpy float32 ----

def _det_sqrtf(x):
    """Deterministic sqrtf matching the C implementation."""
    x = np.float32(x)
    if x <= np.float32(0.0):
        return np.float32(0.0)

    # Fast inverse sqrt via bit trick
    bits = np.frombuffer(x.tobytes(), dtype=np.uint32)[0]
    bits = np.uint32(0x5F3759DF) - np.uint32(bits >> 1)
    y = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]

    xhalf = np.float32(0.5) * x
    y = y * (np.float32(1.5) - xhalf * y * y)
    y = y * (np.float32(1.5) - xhalf * y * y)
    y = y * (np.float32(1.5) - xhalf * y * y)

    return x * y


def _det_logf(x):
    """Deterministic logf matching the C implementation."""
    x = np.float32(x)
    bits = np.frombuffer(x.tobytes(), dtype=np.uint32)[0]

    # Extract exponent
    e = int((bits >> 23) & 0xFF) - 127

    # Set mantissa to [1, 2)
    m_bits = (bits & np.uint32(0x007FFFFF)) | np.uint32(0x3F800000)
    m = np.frombuffer(m_bits.tobytes(), dtype=np.float32)[0]

    # Adjust to [sqrt(2)/2, sqrt(2)]
    if m > np.float32(1.41421356):
        m = m * np.float32(0.5)
        e = e + 1

    t = (m - np.float32(1.0)) / (m + np.float32(1.0))
    t2 = t * t

    # Horner evaluation
    poly = np.float32(0.111111111)
    poly = poly * t2 + np.float32(0.142857143)
    poly = poly * t2 + np.float32(0.200000000)
    poly = poly * t2 + np.float32(0.333333333)
    poly = poly * t2 + np.float32(1.000000000)

    log_m = np.float32(2.0) * t * poly

    e_f = np.float32(float(e))
    result = log_m + e_f * np.float32(1.4286068e-6)    # LN2_LO
    result = result + e_f * np.float32(0.6931457519531250)  # LN2_HI
    return result


def _det_sin_poly(x):
    """sin(x) polynomial for x in [0, pi/2]."""
    x = np.float32(x)
    x2 = x * x
    p = np.float32(-2.50521084e-8)
    p = p * x2 + np.float32(2.75573192e-6)
    p = p * x2 + np.float32(-1.98412698e-4)
    p = p * x2 + np.float32(8.33333333e-3)
    p = p * x2 + np.float32(-1.66666667e-1)
    p = p * x2 + np.float32(1.0)
    return x * p


def _det_cos_poly(x):
    """cos(x) polynomial for x in [0, pi/2]."""
    x = np.float32(x)
    x2 = x * x
    p = np.float32(2.08767570e-9)
    p = p * x2 + np.float32(-2.75573192e-7)
    p = p * x2 + np.float32(2.48015873e-5)
    p = p * x2 + np.float32(-1.38888889e-3)
    p = p * x2 + np.float32(4.16666667e-2)
    p = p * x2 + np.float32(-5.00000000e-1)
    p = p * x2 + np.float32(1.0)
    return p


def _det_sincos_2pi(u):
    """sin(2*pi*u) and cos(2*pi*u) for u in [0, 1)."""
    u = np.float32(u)
    u4 = u * np.float32(4.0)
    j = int(u4)
    if j < 0:
        j = 0
    if j > 3:
        j = 3

    u_red = u - np.float32(float(j)) * np.float32(0.25)
    theta_red = u_red * np.float32(6.28125) + u_red * np.float32(-6.46918e-5)

    s = _det_sin_poly(theta_red)
    c = _det_cos_poly(theta_red)

    if j == 0:
        return s, c
    elif j == 1:
        return c, -s
    elif j == 2:
        return -s, -c
    else:
        return -c, s


def _uint32x4_to_normal4(raw):
    """Box-Muller: 4 uint32 -> 4 normal floats (as numpy float32 array)."""
    u = [np.float32(float(r >> 8)) * np.float32(5.96046448e-8) for r in raw]

    # Clamp radial uniforms
    MIN_U = np.float32(5.96046448e-8)
    if u[0] < MIN_U:
        u[0] = MIN_U
    if u[2] < MIN_U:
        u[2] = MIN_U

    out = np.zeros(4, dtype=np.float32)

    # Pair 1
    r0 = _det_sqrtf(np.float32(-2.0) * _det_logf(u[0]))
    sin0, cos0 = _det_sincos_2pi(u[1])
    out[0] = r0 * cos0
    out[1] = r0 * sin0

    # Pair 2
    r1 = _det_sqrtf(np.float32(-2.0) * _det_logf(u[2]))
    sin1, cos1 = _det_sincos_2pi(u[3])
    out[2] = r1 * cos1
    out[3] = r1 * sin1

    return out


def generate_normal_reference(seed, counter, n):
    """Generate n normal float32 values using pure-Python Philox + Box-Muller.

    Returns a 1-D torch float32 tensor on CPU.
    """
    seed = seed & 0xFFFFFFFFFFFFFFFF
    key = (seed & MASK32, (seed >> 32) & MASK32)
    num_blocks = (n + 3) // 4

    result = np.empty(num_blocks * 4, dtype=np.float32)

    for i in range(num_blocks):
        ctr_val = counter + i
        ctr = (ctr_val & MASK32, (ctr_val >> 32) & MASK32, 0, 0)
        raw = philox4x32_10(ctr, key)
        normals = _uint32x4_to_normal4(raw)
        result[i * 4: i * 4 + 4] = normals

    return torch.from_numpy(result[:n].copy())
