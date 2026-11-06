"""Synthesize the AgentDunker outro score (6.0 s, resolving, ends in silence). All sounds are generated here."""
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt
from scipy.io import wavfile

SR = 48000
DUR = 6.0
N = int(SR * DUR)
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(7)


def place(sig, start, pan=0.0, gain=1.0):
    i = int(start * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(sig)] += sig * gl * 1.414
    R[i:i + len(sig)] += sig * gr * 1.414


def lp(x, f, order=2):
    return sosfilt(butter(order, f, btype='low', fs=SR, output='sos'), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, btype='high', fs=SR, output='sos'), x)


def saw(freq, n, detune=0.0):
    ph = np.cumsum(np.full(n, freq * (1 + detune)) / SR)
    return 2 * (ph % 1) - 1


def pad(freqs, dur, fade_in, fade_out, cutoff=1600, gain=0.05):
    n = int(dur * SR); x = np.arange(n) / SR
    p = sum(lp(saw(f, n, d), cutoff) for f in freqs for d in (-0.005, 0.0, 0.005))
    return p * np.minimum(1, x / fade_in) * np.minimum(1, np.maximum(0, (dur - x) / fade_out)) * gain


def bell(f, gain=0.2, dur=2.5):
    m = int(dur * SR); x = np.arange(m) / SR
    return sum(a * np.sin(2 * np.pi * f * k * x) * np.exp(-x * d) for k, a, d in [(1, .5, 2.2), (2.0, .25, 3.5), (3.01, .12, 5), (4.2, .06, 7)]) * gain


# reverse swell into the title hit (0 -> 0.35)
n = int(0.4 * SR); x = np.arange(n) / SR
place(hp(rng.standard_normal(n), 2500) * (x / 0.4) ** 3 * 0.35, 0.0)

# soft title hit at 0.35
m = int(2.5 * SR); x = np.arange(m) / SR
hit = np.sin(2 * np.pi * np.cumsum(44 + 90 * np.exp(-x * 16)) / SR) * np.exp(-x * 2.6) * 0.7
hit += lp(rng.standard_normal(m), 900) * np.exp(-x * 8) * 0.25
place(hit, 0.35)

# chord progression: Am(add9) -> Fmaj7 -> Am(add9), resolving
place(pad([110, 220, 261.63, 329.63, 493.88], 2.3, 0.4, 0.5, gain=0.045), 0.35)
place(pad([87.31, 174.61, 220, 261.63, 329.63], 2.2, 0.5, 0.6, gain=0.045), 2.2)
place(pad([110, 220, 261.63, 329.63, 493.88, 659.25], 2.3, 0.4, 1.6, cutoff=1300, gain=0.045), 3.7)

# sparkle arpeggio (slow, thinning out)
arp = [659.25, 987.77, 880.0, 1318.5, 987.77, 783.99, 659.25, 1174.7]
for i, f in enumerate(arp):
    place(bell(f, 0.07, 1.6), 1.0 + i * 0.36, pan=(-0.5 if i % 2 else 0.5))

# shimmer bells on subtitle (1.0) and logos (2.2)
place(bell(1318.5, 0.18), 1.0, -0.2)
place(bell(1975.5, 0.14), 2.22, 0.3); place(bell(2637.0, 0.08), 2.3, -0.3)

# soft whoosh with the light sweep (3.0)
n = int(0.8 * SR); x = np.arange(n) / SR
place(lp(hp(rng.standard_normal(n), 1500), 6000) * np.sin(np.pi * x / 0.8) ** 2 * 0.18, 3.0, 0.2)

# final low boom at 4.3 as the picture fades
m = int(1.7 * SR); x = np.arange(m) / SR
boom = np.sin(2 * np.pi * np.cumsum(40 + 60 * np.exp(-x * 10)) / SR) * np.exp(-x * 2.0) * 0.6
place(boom, 4.3)

# reverb
ir_n = int(2.2 * SR); ix = np.arange(ir_n) / SR
irL = rng.standard_normal(ir_n) * np.exp(-ix * 2.4); irR = rng.standard_normal(ir_n) * np.exp(-ix * 2.4)
irL[0] = irR[0] = 0
outL = L + fftconvolve(L, irL)[:N] * 0.025
outR = R + fftconvolve(R, irR)[:N] * 0.025

# master: gentle fade to silence by 6.0
env = np.ones(N)
i0 = int(4.6 * SR)
env[i0:] = np.linspace(1, 0, N - i0) ** 1.6
mix = np.stack([outL * env, outR * env], 1)
mix /= np.max(np.abs(mix))
mix = np.tanh(mix * 1.3) / np.tanh(1.3) * 0.7
wavfile.write('outro_score.wav', SR, (mix * 32767).astype(np.int16))
print('wrote outro_score.wav')
