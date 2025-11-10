# jdsp-core
```python
"""
jdsp_core.py

JAX-based DSP core:
- Pure functional, no classes
- Uses jax.numpy, jax.random, jax.lax for vectorized / scan-friendly ops
- Compatible with JIT, vmap, grad where meaningful
- Plotting / I/O intentionally omitted (handled externally if needed)
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
from jax import lax, random

from jax import config
config.update("jax_enable_x64", True)
# ================================================================
# Helpers
# ================================================================

def _as_array(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _to_control_sig(val, N, name="param"):
    """
    Scalar -> (N,) array
    (N,)    -> returned as float64
    Anything else: runtime error (outside jit) or undefined (inside jit).
    """
    val = jnp.asarray(val)

    # 0-d behaves like scalar
    if val.ndim == 0:
        return jnp.full((N,), float(val), dtype=jnp.float64)

    if val.ndim == 1:
        if val.shape[0] != N:
            # For JIT safety, avoid raise in jitted context; this is best-effort.
            raise ValueError(f"{name} must be scalar or length {N}, got {val.shape}")
        return val.astype(jnp.float64)

    raise ValueError(f"{name} must be scalar or 1D array")


# ================================================================
# dB / linear
# ================================================================


def db_to_lin(db):
    db = _as_array(db)
    return 10.0 ** (db / 20.0)



def lin_to_db(lin):
    lin = _as_array(lin)
    lin = jnp.maximum(lin, 1e-12)
    return 20.0 * jnp.log10(lin)


# ================================================================
# Noise (functional RNG)
# ================================================================


def noise_white(key, N, amp=1.0):
    """
    White noise in [-amp, +amp].
    key : PRNGKey
    """
    return random.uniform(key, (int(N),), minval=-amp, maxval=amp)



def noise_gauss(key, N, std=1.0):
    """
    Gaussian noise, mean 0, std.
    key : PRNGKey
    """
    return random.normal(key, (int(N),)) * std


# ================================================================
# Basic signal sources
# ================================================================


def dc(N, value=0.0):
    return jnp.full((int(N),), float(value), dtype=jnp.float64)



def ramp(N, start=0.0, end=1.0):
    N = int(N)
    return jnp.linspace(float(start), float(end), N, dtype=jnp.float64)


# ================================================================
# Nonlinear / limiting
# ================================================================


def hard_clip(x, limit=1.0):
    x = _as_array(x)
    limit = abs(limit)
    return jnp.clip(x, -limit, limit)



def soft_clip(x, drive=1.0):
    x = _as_array(x)
    return jnp.tanh(x * drive)


# ================================================================
# DC / Nyquist blockers & filter coeff helpers
# ================================================================

def dc_blocker_coef(fc_hz, fs):
    return float(jnp.exp(-2.0 * jnp.pi * fc_hz / fs))



def dc_blocker(x, a=0.995):
    """
    First-order DC blocker:
    y[n] = x[n] - x[n-1] + a * y[n-1]
    Starts from x[0], y[0] = 0.
    """
    x = _as_array(x)
    N = x.shape[0]

    def step(carry, n):
        x_prev, y_prev = carry
        xn = x[n]
        y = xn - x_prev + a * y_prev
        return (xn, y), y

    (_, _), y = lax.scan(step, (x[0], 0.0), jnp.arange(1, N))
    y0 = jnp.array([0.0])
    return jnp.concatenate([y0, y])


def one_pole_coef(fc_hz, fs):
    if fc_hz <= 0:
        return 1.0
    return float(jnp.exp(-2.0 * jnp.pi * fc_hz / fs))


def hp_coef(fc_hz, fs):
    if fc_hz <= 0:
        return 0.0
    return float(jnp.exp(-2.0 * jnp.pi * fc_hz / fs))



def one_pole_lp(x, a):
    """
    y[n] = (1-a)*x[n] + a*y[n-1], y[-1] = 0
    """
    x = _as_array(x)
    one_minus_a = 1.0 - a

    def step(y_prev, xn):
        y = one_minus_a * xn + a * y_prev
        return y, y

    _, y = lax.scan(step, 0.0, x)
    return y



def one_pole_lp_var(x, a_arr):
    """
    a_arr: (N,) per-sample coefficient
    """
    x = _as_array(x)
    a_arr = _as_array(a_arr)

    def step(y_prev, xa):
        xn, a = xa
        y = (1.0 - a) * xn + a * y_prev
        return y, y

    _, y = lax.scan(step, 0.0, (x, a_arr))
    return y



def one_pole_hp(x, a):
    """
    High-pass:
    y[n] = a * (y[n-1] + x[n] - x[n-1])
    """
    x = _as_array(x)
    N = x.shape[0]

    def step(carry, n):
        y_prev, x_prev = carry
        xn = x[n]
        y = a * (y_prev + xn - x_prev)
        return (y, xn), y

    (y_last, x_last), y = lax.scan(
        step,
        (0.0, x[0]),
        jnp.arange(1, N),
    )
    y0 = jnp.array([0.0])
    return jnp.concatenate([y0, y])



def nyquist_blocker(x, fs, cutoff_ratio=0.45):
    """
    Simple anti-imaging LP using one-pole low-pass.
    cutoff = cutoff_ratio * Nyquist
    """
    fc = cutoff_ratio * 0.5 * fs
    a = one_pole_coef(fc, fs)
    return one_pole_lp(x, a)


# ================================================================
# Time constants / slew limiting
# ================================================================

def time_const_to_coef(tau_s, fs):
    if tau_s <= 0:
        return 0.0
    return float(jnp.exp(-1.0 / (tau_s * fs)))



def slew_limit(x, rate_up, rate_down):
    """
    Per-sample delta clamp.
    rate_up/down: max change per sample.
    """
    x = _as_array(x)

    def step(y_prev, xn):
        delta = xn - y_prev
        delta = jnp.clip(delta, -rate_down, rate_up)
        y = y_prev + delta
        return y, y

    _, y = lax.scan(step, x[0], x)
    return y



def slew_limit_time(x, fs, rise_time=0.01, fall_time=0.01):
    """
    Time-based slew. 1.0 FS over rise_time/fall_time.
    """
    x = _as_array(x)
    rate_up = 1.0 / jnp.maximum(rise_time * fs, 1e-9)
    rate_down = 1.0 / jnp.maximum(fall_time * fs, 1e-9)

    def step(y_prev, xn):
        delta = xn - y_prev
        delta = jnp.clip(delta, -rate_down, rate_up)
        y = y_prev + delta
        return y, y

    _, y = lax.scan(step, x[0], x)
    return y



def slew_limit_db(x_db, fs, rise_time=0.02, fall_time=0.05, range_db=60.0):
    """
    Slew limiter in dB domain.
    rise/fall time: time to traverse `range_db`.
    """
    x_db = _as_array(x_db)
    rate_up = range_db / jnp.maximum(rise_time * fs, 1e-9)
    rate_down = range_db / jnp.maximum(fall_time * fs, 1e-9)

    def step(y_prev, xn):
        delta = xn - y_prev
        delta = jnp.clip(delta, -rate_down, rate_up)
        y = y_prev + delta
        return y, y

    _, y = lax.scan(step, x_db[0], x_db)
    return y



def slew_limit_time_var(x, fs, rise_time, fall_time):
    """
    Variable-time slew; rise_time/fall_time arrays.
    """
    x = _as_array(x)
    rise_time = _as_array(rise_time)
    fall_time = _as_array(fall_time)

    def step(y_prev, args):
        xn, rt, ft = args
        rt = jnp.maximum(rt, 1e-9)
        ft = jnp.maximum(ft, 1e-9)
        rate_up = 1.0 / (rt * fs)
        rate_down = 1.0 / (ft * fs)
        delta = xn - y_prev
        delta = jnp.clip(delta, -rate_down, rate_up)
        y = y_prev + delta
        return y, y

    _, y = lax.scan(step, x[0], (x, rise_time, fall_time))
    return y



def slew_limit_db_var(x_db, fs, rise_time, fall_time, range_db=60.0):
    """
    Variable-time dB slew.
    """
    x_db = _as_array(x_db)
    rise_time = _as_array(rise_time)
    fall_time = _as_array(fall_time)

    def step(y_prev, args):
        xn, rt, ft = args
        rt = jnp.maximum(rt, 1e-9)
        ft = jnp.maximum(ft, 1e-9)
        rate_up = range_db / (rt * fs)
        rate_down = range_db / (ft * fs)
        delta = xn - y_prev
        delta = jnp.clip(delta, -rate_down, rate_up)
        y = y_prev + delta
        return y, y

    _, y = lax.scan(step, x_db[0], (x_db, rise_time, fall_time))
    return y


# ================================================================
# Curves & splines (JAX-compatible, vectorized where possible)
# ================================================================


def bspline3(x):
    """
    Cubic B-spline basis B3(x), defined over [-2, 2].
    """
    ax = jnp.abs(x)
    y = jnp.where(
        ax < 1.0,
        (4 - 6 * ax**2 + 3 * ax**3) / 6.0,
        jnp.where(
            (ax >= 1.0) & (ax < 2.0),
            ((2 - ax) ** 3) / 6.0,
            0.0,
        ),
    )
    return y



def bspline_envelope(x, xp, yp):
    """
    Smooth envelope via cubic B-spline blending.
    Assumes uniform xp spacing.
    """
    x = _as_array(x)
    xp = _as_array(xp)
    yp = _as_array(yp)
    step = xp[1] - xp[0]

    def contrib(i, acc):
        xi = xp[i]
        w = bspline3((x - xi) / step)
        return acc + yp[i] * w

    acc0 = jnp.zeros_like(x)
    y = lax.fori_loop(0, xp.shape[0], contrib, acc0)
    return y



def catmull_rom(x, xp, yp):
    """
    Catmull-Rom spline interpolation (stateless).
    xp sorted ascending.
    """
    x = _as_array(x)
    xp = _as_array(xp)
    yp = _as_array(yp)
    N = xp.shape[0]

    def interp(xv):
        # clamp index
        i = jnp.clip(jnp.searchsorted(xp, xv) - 1, 1, N - 3)
        i0 = i - 1
        i1 = i
        i2 = i + 1
        i3 = i + 2

        x0 = xp[i0]
        x1 = xp[i1]
        x2 = xp[i2]
        x3 = xp[i3]
        y0 = yp[i0]
        y1 = yp[i1]
        y2 = yp[i2]
        y3 = yp[i3]

        t = (xv - x1) / (x2 - x1)

        a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
        a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        a2 = -0.5 * y0 + 0.5 * y2
        a3 = y1

        return ((a0 * t + a1) * t + a2) * t + a3

    return jax.vmap(interp)(x)


# (Full cubic_spline could be added similarly using lax.fori_loop; omitted for brevity)


# ================================================================
# Analytic signal / Hilbert
# ================================================================


def hilbert_analytic(x):
    """
    Hilbert transform via FFT (analytic signal).
    """
    x = _as_array(x)
    N = x.size
    X = jnp.fft.fft(x)
    H = jnp.zeros(N, dtype=jnp.float64)

    def even_case(H):
        H = H.at[0].set(1.0)
        H = H.at[N // 2].set(1.0)
        H = H.at[1:N // 2].set(2.0)
        return H

    def odd_case(H):
        H = H.at[0].set(1.0)
        H = H.at[1:(N + 1) // 2].set(2.0)
        return H

    H = lax.cond(N % 2 == 0, even_case, odd_case, H)
    return jnp.fft.ifft(X * H).astype(jnp.complex128)


# ================================================================
# Modulation
# ================================================================


def ring_mod(modulator, carrier):
    m = _as_array(modulator)
    c = _as_array(carrier)
    return m * c



def amplitude_mod(modulator, carrier, depth=1.0):
    m = _as_array(modulator)
    c = _as_array(carrier)
    return (1.0 + depth * m) * c



def ssb_modulate(modulator, carrier, sideband_is_upper=True, remove_dc=True):
    """
    Single sideband frequency shifter via analytic signals.
    sideband_is_upper: True (upper), False (lower)
    """
    m = _as_array(modulator)
    c = _as_array(carrier)

    if remove_dc:
        m = m - jnp.mean(m)
        c = c - jnp.mean(c)

    ma = hilbert_analytic(m)
    ca = hilbert_analytic(c)

    def upper(_):
        return jnp.real(ma * ca)

    def lower(_):
        return jnp.real(ma * jnp.conj(ca))

    return lax.cond(sideband_is_upper, upper, lower, operand=None)


# ================================================================
# Envelope follower / AM demod
# ================================================================

def env_ar(x, ga, gr):
    """
    Attack/release envelope follower.
    ga, gr: "timescale" constants in [0,1), smaller = faster.
    """
    x = _as_array(x)
    # convert to effective coefficients (faster decay)
    ga = 1.0 - (1.0 - ga) * 0.1
    gr = 1.0 - (1.0 - gr) * 0.5

    def step(y_prev, xn):
        xn = jnp.abs(xn)
        y = jnp.where(
            xn > y_prev,
            ga * y_prev + (1 - ga) * xn,
            gr * y_prev + (1 - gr) * xn,
        )
        return y, y

    _, env = lax.scan(step, 0.0, x)
    return jnp.clip(env, 0.0, 1.0)


def am_demod_envelope(x, fs, tau_a=0.001, tau_r=0.03):
    x = _as_array(x)
    env_inst = jnp.abs(hilbert_analytic(x))
    ga = time_const_to_coef(tau_a, fs)
    gr = time_const_to_coef(tau_r, fs)
    return env_ar(env_inst, ga, gr)


# ================================================================
# LFOs
# ================================================================


def lfo_wave(shape, phase):
    shape = shape.lower()
    p = phase % 1.0
    if shape == "sine":
        return jnp.sin(2 * jnp.pi * p)
    elif shape == "tri":
        return 4.0 * jnp.abs(p - 0.5) - 1.0
    elif shape == "saw":
        return 2.0 * p - 1.0
    elif shape == "square":
        s = jnp.sign(jnp.sin(2 * jnp.pi * p))
        s = jnp.where(s == 0, 0.0, s)
        return jnp.array(s, dtype=jnp.float64)
    else:
        raise ValueError(f"Unknown LFO shape '{shape}'")



def lfo(fs, freq, N, phase=0.0, shape="sine", phase_offset=0.0):
    t = jnp.arange(int(N), dtype=jnp.float64) / fs
    ph = (phase + freq * t + phase_offset) % 1.0
    return lfo_wave(shape, ph)


# ================================================================
# ADSR & asymptotic envelopes
# ================================================================


def adsr(
    fs,
    attack=0.01,
    decay=0.1,
    sustain=0.7,
    release=0.2,
    gate_length=1.0,
    curve="exp",
):
    """
    Stateless ADSR (block).
    """
    fs = float(fs)
    Na = int(jnp.round(attack * fs))
    Nd = int(jnp.round(decay * fs))
    Nr = int(jnp.round(release * fs))
    Ng = int(jnp.round(gate_length * fs))

    if curve == "exp":
        def seg_exp(N, start, end, tau):
            N = int(N)
            if N <= 0:
                return jnp.zeros((0,), dtype=jnp.float64)
            k = jnp.exp(-1.0 / jnp.maximum(tau * fs, 1.0))
            n = jnp.arange(N, dtype=jnp.float64)
            return end + (start - end) * (k ** n)

        a = seg_exp(Na, 0.0, 1.0, attack)
        d = seg_exp(Nd, 1.0, sustain, decay)
        s_len = jnp.maximum(0, Ng - Na - Nd)
        s = jnp.full((s_len,), sustain)
        r = seg_exp(Nr, sustain, 0.0, release)
    else:
        def seg_lin(N, start, end):
            N = int(N)
            if N <= 0:
                return jnp.zeros((0,), dtype=jnp.float64)
            return jnp.linspace(start, end, N, endpoint=False)

        a = seg_lin(Na, 0.0, 1.0)
        d = seg_lin(Nd, 1.0, sustain)
        s_len = jnp.maximum(0, Ng - Na - Nd)
        s = jnp.full((s_len,), sustain)
        r = seg_lin(Nr, sustain, 0.0)

    return jnp.concatenate([a, d, s, r])



def asymp_env(fs, duration, start=0.0, target=1.0, tau=0.3, threshold=1e-4):
    """
    Asymptotic exponential envelope to target.
    """
    fs = float(fs)
    N = int(jnp.round(duration * fs))
    if N <= 0:
        return jnp.zeros((0,), dtype=jnp.float64)

    a = jnp.exp(-1.0 / (tau * fs))
    c = (1.0 - a) * target

    def step(val, _):
        val = a * val + c
        return val, val

    _, env = lax.scan(step, start, jnp.arange(N))
    # no early termination in JAX scan; approximate full curve
    return env



def asymp_adsr(
    fs,
    attack=0.01,
    decay=0.1,
    sustain=0.7,
    release=0.2,
    gate=0.5,
    tau_attack=0.01,
    tau_decay=0.05,
    tau_release=0.1,
    threshold=1e-4,
):
    """
    Multi-segment asymptotic ADSR (stateless).
    """
    a = asymp_env(fs, attack, 0.0, 1.0, tau_attack, threshold)
    d = asymp_env(fs, decay, 1.0, sustain, tau_decay, threshold)
    s_len = int(jnp.round(gate * fs))
    s = jnp.full((s_len,), sustain)
    r = asymp_env(fs, release, sustain, 0.0, tau_release, threshold)
    return jnp.concatenate([a, d, s, r])


# ================================================================
# Amplifiers / Panning
# ================================================================


def amp_db(x, gain_db):
    x = _as_array(x)
    g = _as_array(gain_db)
    # broadcast
    g = jnp.broadcast_to(g, x.shape)
    gain_lin = 10.0 ** (g / 20.0)
    return x * gain_lin



def stereo_panner(x, pan):
    """
    Equal-power panner.
    x: mono (N,) or stereo (N,2)
    pan: [-1..1], scalar or (N,)
    """
    x = _as_array(x)

    if x.ndim == 1:
        xL = xR = x
    elif x.ndim == 2 and x.shape[1] == 2:
        xL = x[:, 0]
        xR = x[:, 1]
    else:
        raise ValueError("x must be mono (N,) or stereo (N,2)")

    N = xL.shape[0]
    pan_arr = _to_control_sig(pan, N, "pan")
    pan_arr = jnp.clip(pan_arr, -1.0, 1.0)

    angle = (pan_arr + 1.0) * 0.25 * jnp.pi
    gL = jnp.cos(angle)
    gR = jnp.sin(angle)

    yL = xL * gL
    yR = xR * gR
    return jnp.stack([yL, yR], axis=-1)



def amp_mod(x, gain_db=0.0, mod=None, mod_depth_db=0.0):
    """
    Gain with optional modulation in dB.
    """
    x = _as_array(x)
    N = x.shape[0]

    base = _to_control_sig(gain_db, N, "gain_db")

    if mod is not None:
        m = _as_array(mod)
        if m.shape[0] != N:
            raise ValueError("mod must be length N")
        base = base + m * mod_depth_db

    gain_lin = 10.0 ** (base / 20.0)
    return x * gain_lin



def amp_pan_mod(
    x,
    gain_db=0.0,
    mod_gain=None,
    mod_gain_depth_db=0.0,
    pan=0.0,
    mod_pan=None,
    mod_pan_depth=1.0,
):
    """
    Amp + stereo pan with modulation.
    """
    x = _as_array(x)

    if x.ndim == 1:
        xL = xR = x
    elif x.ndim == 2 and x.shape[1] == 2:
        xL = x[:, 0]
        xR = x[:, 1]
    else:
        raise ValueError("x must be mono (N,) or stereo (N,2)")

    N = xL.shape[0]

    # gain path
    g_db = _to_control_sig(gain_db, N, "gain_db")

    if mod_gain is not None:
        mg = _as_array(mod_gain)
        if mg.shape[0] != N:
            raise ValueError("mod_gain must be length N")
        g_db = g_db + mg * mod_gain_depth_db

    gain_lin = 10.0 ** (g_db / 20.0)

    # pan path
    pan_arr = _to_control_sig(pan, N, "pan")

    if mod_pan is not None:
        mp = _as_array(mod_pan)
        if mp.shape[0] != N:
            raise ValueError("mod_pan must be length N")
        pan_arr = pan_arr + mp * mod_pan_depth

    pan_arr = jnp.clip(pan_arr, -1.0, 1.0)

    angle = (pan_arr + 1.0) * 0.25 * jnp.pi
    gL = jnp.cos(angle) * gain_lin
    gR = jnp.sin(angle) * gain_lin

    yL = xL * gL
    yR = xR * gR
    return jnp.stack([yL, yR], axis=-1)


#!/usr/bin/env python3
"""
examples_jdsp_core_plots.py
---------------------------
Demonstration suite for jdsp_core.py

Covers:
- Noise and basic signals
- Nonlinear processing
- Filters (DC, HP, LP)
- Slew limiting and envelopes
- Splines and modulation
- Hilbert / AM / SSB
- Amplifiers and panning

Author: James Theory / GPT-5
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jdsp_core import *


# ================================================================
# Utility for plotting
# ================================================================
def plot_signal(x, fs, title="", dur_s=None):
    N = len(x)
    t = np.arange(N) / fs
    if dur_s is not None:
        N = int(dur_s * fs)
        x = np.asarray(x[:N])
        t = t[:N]
    plt.figure(figsize=(8, 3))
    plt.plot(t, np.asarray(x))
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectrum(x, fs, title=""):
    N = len(x)
    X = np.fft.rfft(np.asarray(x) * np.hanning(N))
    f = np.linspace(0, fs/2, len(X))
    mag = 20 * np.log10(np.maximum(np.abs(X), 1e-12))
    plt.figure(figsize=(8, 3))
    plt.plot(f, mag)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ================================================================
# 1. Noise and basic signals
# ================================================================
def demo_noise_and_signals():
    fs = 48000
    key = random.PRNGKey(0)
    N = 4096
    n_white = noise_white(key, N)
    n_gauss = noise_gauss(key, N)
    dc_sig = dc(N, 0.5)
    ramp_sig = ramp(N, 0.0, 1.0)

    plot_signal(n_white, fs, "White noise")
    plot_signal(n_gauss, fs, "Gaussian noise")
    plot_signal(dc_sig, fs, "DC signal")
    plot_signal(ramp_sig, fs, "Ramp 0–1")
    plot_spectrum(n_white, fs, "White noise spectrum")


# ================================================================
# 2. Nonlinearities
# ================================================================
def demo_nonlinearities():
    x = jnp.linspace(-2, 2, 1000)
    hard = hard_clip(x)
    soft = soft_clip(x, drive=2.0)
    plt.figure(figsize=(6, 4))
    plt.plot(x, hard, label="Hard clip")
    plt.plot(x, soft, label="Soft clip (tanh drive=2)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Hard vs. Soft Clipping")
    plt.tight_layout()
    plt.show()


# ================================================================
# 3. Filters
# ================================================================
def demo_filters():
    fs = 48000
    t = jnp.arange(0, 0.05, 1/fs)
    x = jnp.sin(2*jnp.pi*440*t) + 0.5*jnp.sin(2*jnp.pi*2000*t)

    a_lp = one_pole_coef(1000, fs)
    y_lp = one_pole_lp(x, a_lp)

    a_hp = one_pole_coef(1000, fs)
    y_hp = one_pole_hp(x, a_hp)

    plot_signal(x, fs, "Input (440 + 2000 Hz)")
    plot_signal(y_lp, fs, "Low-pass output (1 kHz)")
    plot_signal(y_hp, fs, "High-pass output (1 kHz)")
    plot_spectrum(y_lp, fs, "LP spectrum")
    plot_spectrum(y_hp, fs, "HP spectrum")


# ================================================================
# 4. Slew limiting and envelopes
# ================================================================
def demo_slew_envelope():
    fs = 48000
    N = int(fs * 0.1)
    step = jnp.concatenate([jnp.zeros(N//2), jnp.ones(N//2)])
    slew = slew_limit_time(step, fs, rise_time=0.01, fall_time=0.05)
    adsr_env = adsr(fs, attack=0.01, decay=0.05, sustain=0.6, release=0.2)
    asymp = asymp_env(fs, 0.2, 0.0, 1.0, tau=0.05)

    plot_signal(step, fs, "Input step for slew limiter")
    plot_signal(slew, fs, "Slew-limited output")
    plot_signal(adsr_env, fs, "ADSR envelope")
    plot_signal(asymp, fs, "Asymptotic envelope")


# ================================================================
# 5. Splines
# ================================================================
def demo_splines():
    xp = jnp.linspace(0, 1, 6)
    yp = jnp.array([0.0, 0.3, 1.0, 0.5, 0.8, 0.2])
    x = jnp.linspace(0, 1, 500)
    y_bs = bspline_envelope(x, xp, yp)
    y_cat = catmull_rom(x, xp, yp)

    plt.figure(figsize=(8, 3))
    plt.plot(xp, yp, "o", label="control points")
    plt.plot(x, y_bs, label="B-spline")
    plt.plot(x, y_cat, "--", label="Catmull-Rom")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Spline Interpolation")
    plt.tight_layout()
    plt.show()


# ================================================================
# 6. Hilbert / modulation
# ================================================================
def demo_modulation():
    fs = 48000
    t = jnp.arange(0, 0.01, 1/fs)
    mod = jnp.sin(2*jnp.pi*200*t)
    car = jnp.sin(2*jnp.pi*2000*t)
    ring = ring_mod(mod, car)
    am = amplitude_mod(mod, car, depth=0.8)
    ssb = ssb_modulate(mod, car, sideband_is_upper=True)

    plot_signal(ring, fs, "Ring modulation (200 Hz × 2 kHz)")
    plot_signal(am, fs, "AM modulation (depth=0.8)")
    plot_signal(ssb, fs, "SSB modulation (upper)")


# ================================================================
# 7. Envelope follower / demod
# ================================================================
def demo_envelope_following():
    fs = 48000
    t = jnp.arange(0, 0.05, 1/fs)
    sig = jnp.sin(2*jnp.pi*200*t) * (1.0 + 0.5*jnp.sin(2*jnp.pi*2*t))
    env = am_demod_envelope(sig, fs)
    plot_signal(sig, fs, "AM signal")
    plot_signal(env, fs, "Detected envelope (Hilbert + AR smoothing)")


# ================================================================
# 8. LFOs
# ================================================================
def demo_lfos():
    fs = 48000
    N = int(fs * 1.0)
    lfo_sine = lfo(fs, 2.0, N, shape="sine")
    lfo_tri = lfo(fs, 2.0, N, shape="tri")
    lfo_saw = lfo(fs, 2.0, N, shape="saw")
    lfo_square = lfo(fs, 2.0, N, shape="square")

    plt.figure(figsize=(8, 3))
    plt.plot(np.asarray(lfo_sine), label="sine")
    plt.plot(np.asarray(lfo_tri), label="tri")
    plt.plot(np.asarray(lfo_saw), label="saw")
    plt.plot(np.asarray(lfo_square), label="square")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("LFO Waveforms")
    plt.tight_layout()
    plt.show()


# ================================================================
# 9. Amplifiers and panning
# ================================================================
def demo_amp_panning():
    fs = 48000
    t = jnp.arange(0, 0.02, 1/fs)
    x = jnp.sin(2*jnp.pi*440*t)
    mod = jnp.sin(2*jnp.pi*1*t)

    gain_mod = amp_mod(x, gain_db=-6.0, mod=mod, mod_depth_db=12.0)
    stereo_pan = stereo_panner(x, pan=jnp.linspace(-1, 1, len(x)))
    amp_pan = amp_pan_mod(x, gain_db=-3.0, mod_gain=mod, mod_gain_depth_db=6.0,
                           pan=0.0, mod_pan=mod, mod_pan_depth=1.0)

    plot_signal(gain_mod, fs, "Amplitude modulation in dB domain")
    plot_signal(stereo_pan[:, 0], fs, "Stereo panner — Left channel")
    plot_signal(stereo_pan[:, 1], fs, "Stereo panner — Right channel")
    plot_signal(amp_pan[:, 0], fs, "Amp+Pan Modulated — Left channel")
    plot_signal(amp_pan[:, 1], fs, "Amp+Pan Modulated — Right channel")


# ================================================================
# Main
# ================================================================
def main():
    demo_noise_and_signals()
    demo_nonlinearities()
    demo_filters()
    demo_slew_envelope()
    demo_splines()
    demo_modulation()
    demo_envelope_following()
    demo_lfos()
    demo_amp_panning()


if __name__ == "__main__":
    main()
````
