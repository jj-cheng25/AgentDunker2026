"""SFX bed for the 4.4 s "3D 战术解读" transition (+0.8 s tail). Mixed quietly under the original voiceover."""
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt
from scipy.io import wavfile

SR = 48000
DUR = 5.2
N = int(SR * DUR)
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(3)


def place(sig, start, pan=0.0, gain=1.0):
    i = int(start * SR)
    sig = sig[: N - i] * gain
    gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(sig)] += sig * gl * 1.414
    R[i:i + len(sig)] += sig * gr * 1.414


def filt(x, f, kind):
    return sosfilt(butter(2, f, btype=kind, fs=SR, output='sos'), x)


def sweep_noise(n, f0, f1, q=0.3):
    x = rng.standard_normal(n); out = np.zeros(n); blk = 1024
    for s in range(0, n, blk):
        fc = f0 * (f1 / f0) ** (s / n)
        seg = x[max(0, s - 2048): s + blk]
        y = filt(seg, [max(20, fc * (1 - q)), min(SR / 2 - 100, fc * (1 + q))], 'band')
        out[s: s + blk] = y[-len(x[s:s + blk]):]
    return out


# 0.0 freeze: digital zap + short low thump
m = int(0.6 * SR); x = np.arange(m) / SR
zap = np.sin(2 * np.pi * np.cumsum(2400 * np.exp(-x * 9) + 200) / SR) * np.exp(-x * 10) * 0.35
thump = np.sin(2 * np.pi * np.cumsum(50 + 120 * np.exp(-x * 25)) / SR) * np.exp(-x * 7) * 0.7
glitch = filt(rng.standard_normal(m), 3000, 'high') * (np.sign(np.sin(2 * np.pi * 37 * x)) > 0) * np.exp(-x * 14) * 0.25
place(zap + thump + glitch, 0.0)

# 0.08-0.6 scan sweep (top -> bottom)
n = int(0.55 * SR); x = np.arange(n) / SR
place(sweep_noise(n, 6000, 900, 0.25) * np.sin(np.pi * x / 0.55) ** 1.5 * 0.5, 0.08, 0.0)

# 0.5-1.7 screen tilts back: low whoosh down
n = int(1.3 * SR); x = np.arange(n) / SR
place(sweep_noise(n, 1200, 120, 0.5) * np.sin(np.pi * x / 1.3) ** 2 * 0.7, 0.5)
place(np.sin(2 * np.pi * 36 * x) * np.sin(np.pi * x / 1.3) ** 2 * 0.35, 0.5)

# 1.0 title hit + shimmer
m = int(2.0 * SR); x = np.arange(m) / SR
hit = np.sin(2 * np.pi * np.cumsum(44 + 100 * np.exp(-x * 18)) / SR) * np.exp(-x * 3.0) * 0.75
hit += filt(rng.standard_normal(m), 1800, 'high') * np.exp(-x * 20) * 0.3
place(hit, 1.33)
shim = sum(a * np.sin(2 * np.pi * f * x) * np.exp(-x * d) for f, a, d in [(1760, .3, 3), (2637, .2, 4.5), (3520, .1, 6)])
place(shim * 0.35, 1.36, 0.2)

# 1.35-1.7 per-character ticks, 1.75-2.25 typing for the English line
for i in range(4):
    mm = int(0.06 * SR); xx = np.arange(mm) / SR
    place(np.sin(2 * np.pi * (1400 + 180 * i) * xx) * np.exp(-xx * 60) * 0.25, 1.35 + i * 0.08, -0.3 + 0.2 * i)
for i in range(21):
    mm = int(0.025 * SR); xx = np.arange(mm) / SR
    place(filt(rng.standard_normal(mm), 3500, 'high') * np.exp(-xx * 180) * 0.16, 1.75 + i * 0.024, 0.2)

# 2.5 light sweep shimmer, 2.55-3.6 court lines drawing: soft data texture
n = int(0.8 * SR); x = np.arange(n) / SR
place(filt(filt(rng.standard_normal(n), 2500, 'high'), 9000, 'low') * np.sin(np.pi * x / 0.8) ** 2 * 0.22, 2.5, 0.3)
for i in range(6):
    mm = int(0.08 * SR); xx = np.arange(mm) / SR
    place(np.sin(2 * np.pi * 2200 * xx) * np.exp(-xx * 50) * 0.12, 2.6 + i * 0.16, 0.5 if i % 2 else -0.5)

# 3.4-4.4 riser into the whiteout, hit at 4.4 (the 3D scene)
n = int(1.0 * SR); x = np.arange(n) / SR
place(sweep_noise(n, 600, 9000, 0.35) * (x / 1.0) ** 2.5 * 0.7, 3.4)
place(np.sin(2 * np.pi * np.cumsum(250 * 4 ** x) / SR) * x ** 3 * 0.12, 3.4)
m = int(0.8 * SR); x = np.arange(m) / SR
place(np.sin(2 * np.pi * np.cumsum(46 + 110 * np.exp(-x * 20)) / SR) * np.exp(-x * 5) * 0.6 + filt(rng.standard_normal(m), 2500, 'high') * np.exp(-x * 16) * 0.25, 4.4)

ir_n = int(1.2 * SR); ix = np.arange(ir_n) / SR
irL = rng.standard_normal(ir_n) * np.exp(-ix * 4); irR = rng.standard_normal(ir_n) * np.exp(-ix * 4); irL[0] = irR[0] = 0
oL = L + fftconvolve(L, irL)[:N] * 0.02; oR = R + fftconvolve(R, irR)[:N] * 0.02
env = np.ones(N); i1 = int(4.5 * SR); env[i1:] = np.linspace(1, 0, N - i1) ** 2
mix = np.stack([oL * env, oR * env], 1); mix /= np.max(np.abs(mix)); mix = np.tanh(mix * 1.3) / np.tanh(1.3) * 0.7
wavfile.write('tr_sfx.wav', SR, (mix * 32767).astype(np.int16))
print('wrote tr_sfx.wav')
