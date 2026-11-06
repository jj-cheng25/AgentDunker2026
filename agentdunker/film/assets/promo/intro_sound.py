"""Synthesize the AgentDunker intro score (6.0 s intro + 2.0 s tail under the video). All sounds are generated here."""
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt
from scipy.io import wavfile

SR = 48000
DUR = 8.0
N = int(SR * DUR)
t_all = np.arange(N) / SR
L = np.zeros(N); Rr = np.zeros(N)
rng = np.random.default_rng(42)


def env_adsr(n, a, d, s, r, sustain_len):
    a, d, r, sl = int(a * SR), int(d * SR), int(r * SR), int(sustain_len * SR)
    e = np.concatenate([np.linspace(0, 1, max(a, 1)), np.linspace(1, s, max(d, 1)), np.full(max(sl, 0), s), np.linspace(s, 0, max(r, 1))])
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def place(sig, start, pan=0.0, gain=1.0):
    i = int(start * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(sig)] += sig * gl * 1.414
    Rr[i:i + len(sig)] += sig * gr * 1.414


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype='band', fs=SR, output='sos'), x)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, btype='low', fs=SR, output='sos'), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, btype='high', fs=SR, output='sos'), x)


def saw(freq, n, detune=0.0):
    ph = np.cumsum(np.full(n, freq * (1 + detune)) / SR)
    return 2 * (ph % 1) - 1


def sweep_noise(n, f0, f1, q=0.25):
    """Noise whose band-pass centre glides from f0 to f1 (processed in short blocks)."""
    x = rng.standard_normal(n); out = np.zeros(n); blk = 1024
    for s in range(0, n, blk):
        fr = s / n; fc = f0 * (f1 / f0) ** fr
        lo, hi = max(20, fc * (1 - q)), min(SR / 2 - 100, fc * (1 + q))
        seg = x[max(0, s - 2048): s + blk]
        y = bp(seg, lo, hi)
        out[s: s + blk] = y[-len(x[s:s + blk]):]
    return out


# ---- 1. low rumble + riser (0 -> 2.4 s) ----
n = int(2.45 * SR); tt = np.arange(n) / SR
rumble = np.sin(2 * np.pi * 38 * tt) * 0.5 + lp(rng.standard_normal(n), 120) * 0.6
place(rumble * (tt / 2.45) ** 1.5, 0.0, 0, 0.55)
riser_tone = np.sin(2 * np.pi * np.cumsum(220 * (6.0 ** (tt / 2.45))) / SR)
riser_tone += 0.5 * np.sin(2 * np.pi * np.cumsum(330 * (6.0 ** (tt / 2.45))) / SR)
place(riser_tone * (tt / 2.45) ** 3 * 0.18, 0.0, 0, 1)
place(sweep_noise(n, 300, 7000) * (tt / 2.45) ** 2.2 * 0.9, 0.0, 0, 1)

# ---- 2. HUD / line-draw ticks ----
def blip(f, dur=0.06, gain=0.35):
    m = int(dur * SR); x = np.arange(m) / SR
    return np.sin(2 * np.pi * f * x) * np.exp(-x * 60) * gain
for i, tk in enumerate([0.25, 0.38, 0.5, 0.7, 0.85, 1.0, 1.25]):
    place(blip(2400 + 300 * (i % 3)), tk, pan=(-0.6 if i % 2 else 0.6))

# ---- 3. player pops (cyan 1.0.., orange 1.08..) ----
for i in range(5):
    for side, t0, f in [(0, 1.0, 880), (1, 1.08, 660)]:
        m = int(0.18 * SR); x = np.arange(m) / SR
        fr = f * (1 + 0.12 * i) * (1 + 0.6 * np.exp(-x * 40))
        s = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-x * 22) * 0.28
        place(s, t0 + i * 0.16, pan=-0.5 + 0.25 * i)

# ---- 4. ball whoosh (1.55 -> 2.4) ----
n = int(0.9 * SR); tt = np.arange(n) / SR
wh = sweep_noise(n, 500, 4500, 0.4) * np.sin(np.pi * np.clip(tt / 0.9, 0, 1)) ** 1.5
for k, (s0) in enumerate([0.0]):
    seg = wh * 1.1
    # pan right -> centre
    i0 = int(1.55 * SR)
    pans = np.linspace(0.7, 0.0, n)
    L[i0:i0 + n] += seg * np.cos((pans + 1) * np.pi / 4) * 1.414
    Rr[i0:i0 + n] += seg * np.sin((pans + 1) * np.pi / 4) * 1.414

# ---- 5. IMPACT at 2.4 ----
T_HIT = 2.4
m = int(2.6 * SR); x = np.arange(m) / SR
kick = np.sin(2 * np.pi * np.cumsum(45 + 140 * np.exp(-x * 18)) / SR) * np.exp(-x * 3.2)
crack = hp(rng.standard_normal(m), 1500) * np.exp(-x * 26) * 0.7
sub = np.sin(2 * np.pi * 41 * x) * np.exp(-x * 1.1) * 0.7
# "braam": detuned low saws through an opening low-pass
braam_n = int(2.2 * SR); bx = np.arange(braam_n) / SR
braam = sum(saw(f, braam_n, d) for f in (55, 82.5, 110) for d in (-0.006, 0.0, 0.007))
braam = lp(braam, 900) * np.exp(-bx * 1.3) * (1 - np.exp(-bx * 40)) * 0.16
hit = kick * 1.0 + crack + sub
hit[:braam_n] += braam
place(hit, T_HIT, 0, 1.0)

# ---- 6. groove 2.4 -> 5.3 (kick / clap / hats / bass), 128 BPM ----
beat = 60 / 128
def kick1(g=0.55):
    m = int(0.35 * SR); x = np.arange(m) / SR
    return np.sin(2 * np.pi * np.cumsum(48 + 110 * np.exp(-x * 30)) / SR) * np.exp(-x * 9) * g
def clap(g=0.9):
    m = int(0.25 * SR); x = np.arange(m) / SR
    return bp(rng.standard_normal(m), 900, 4000) * np.exp(-x * 18) * g
def hat(g=0.34):
    m = int(0.05 * SR); x = np.arange(m) / SR
    return hp(rng.standard_normal(m), 7000) * np.exp(-x * 90) * g
start = T_HIT + 2 * beat
k = 0
tb = start
while tb < 5.25:
    place(kick1(), tb)
    if k % 2 == 1:
        place(clap(), tb, pan=0.1)
    place(hat(), tb + beat / 2, pan=0.4)
    place(hat(0.08), tb + beat / 4 * 3, pan=-0.4)
    # bass pulse: A1 / F1 / C2 / G1 pattern
    notes = [55.0, 43.65, 65.41, 49.0]
    f = notes[(k // 2) % 4]
    m = int(beat * 0.9 * SR); x = np.arange(m) / SR
    bs = lp(saw(f, m) + 0.6 * saw(f * 2, m, 0.004), 900) * np.exp(-x * 4) * 0.16
    place(bs, tb + beat / 2)
    k += 1; tb += beat

# 16th-note pluck arpeggio (A minor 7) for mid-range drive on phone speakers
arp = [440.0, 523.25, 659.25, 783.99, 659.25, 523.25, 880.0, 659.25]
ta = T_HIT + 0.5; j = 0
while ta < 5.2:
    f = arp[j % len(arp)]
    m = int(0.16 * SR); x = np.arange(m) / SR
    pl = (saw(f, m, 0.003) + saw(f, m, -0.003)) * 0.5
    pl = lp(pl, 3800) * np.exp(-x * 22) * 0.30 * min(1, (ta - T_HIT) / 0.8)
    place(pl, ta, pan=(-0.45 if j % 2 else 0.45))
    j += 1; ta += beat / 4

# pad swell under the title (A minor add9)
n = int(3.4 * SR); px = np.arange(n) / SR
pad = sum(lp(saw(f, n, d), 1800) for f in (220, 261.63, 329.63, 493.88) for d in (-0.004, 0.004))
pad *= np.minimum(1, px / 0.8) * np.minimum(1, (3.4 - px) / 0.6) * 0.05
place(pad, 2.5, 0, 1)

# ---- 7. typing ticks for the subtitle (2.92 s, 45 ms per char, 17 chars) ----
for i in range(17):
    m = int(0.03 * SR); x = np.arange(m) / SR
    s = hp(rng.standard_normal(m), 3000) * np.exp(-x * 160) * 0.22 + np.sin(2 * np.pi * 3200 * x) * np.exp(-x * 120) * 0.08
    place(s, 2.92 + i * 0.045, pan=-0.3 + 0.04 * i)

# HUD readout blips
pass  # readout blips removed with the on-screen readouts

# ---- 8. logo shimmer (3.85) : bell partials ----
m = int(2.0 * SR); x = np.arange(m) / SR
bell = sum(a * np.sin(2 * np.pi * f * x) * np.exp(-x * dcy) for f, a, dcy in [(1318.5, .5, 2.5), (1975.5, .3, 3.5), (2637, .2, 5), (3951, .1, 7)])
place(bell * 0.22, 3.88, -0.2); place(bell * 0.16, 3.95, 0.3)

# ---- 9. transition: riser + reverse cymbal 5.0 -> 6.0, hit at 6.0 ----
n = int(1.0 * SR); tt = np.arange(n) / SR
rev = hp(rng.standard_normal(n), 4000) * (tt / 1.0) ** 3 * 0.55
place(rev, 5.0, 0, 1)
place(sweep_noise(n, 800, 9000, 0.35) * (tt / 1.0) ** 2.5 * 0.8, 5.0, 0, 1)
up = np.sin(2 * np.pi * np.cumsum(300 * 4 ** tt) / SR) * (tt ** 3) * 0.12
place(up, 5.0)
m = int(2.0 * SR); x = np.arange(m) / SR
hit2 = np.sin(2 * np.pi * np.cumsum(42 + 120 * np.exp(-x * 20)) / SR) * np.exp(-x * 4.5) * 0.8 + hp(rng.standard_normal(m), 2000) * np.exp(-x * 14) * 0.35
place(hit2, 6.0, 0, 0.8)

# ---- reverb (synthetic stereo IR) on everything, then master ----
ir_n = int(1.6 * SR); ix = np.arange(ir_n) / SR
irL = rng.standard_normal(ir_n) * np.exp(-ix * 3.2); irR = rng.standard_normal(ir_n) * np.exp(-ix * 3.2)
irL[0] = irR[0] = 0
wetL = fftconvolve(L, irL)[:N] * 0.018; wetR = fftconvolve(Rr, irR)[:N] * 0.018
outL, outR = L + wetL, Rr + wetR

# master envelope: after 6.0 s duck the tail so the commentary stays clear, fade out by 8.0
env = np.ones(N)
i6 = int(6.0 * SR)
env[i6:] = np.linspace(1.0, 0.0, N - i6) ** 2 * 0.55
env[:int(0.05 * SR)] = np.linspace(0, 1, int(0.05 * SR))
outL *= env; outR *= env
mix = np.stack([outL, outR], 1)
# soft clip + normalise peak; loudness is matched later with ffmpeg loudnorm
print('pre-clip peak', np.max(np.abs(mix)))
mix /= np.max(np.abs(mix))
mix = np.tanh(mix * 1.5) / np.tanh(1.5)
mix *= 0.7
wavfile.write('intro_score.wav', SR, (mix * 32767).astype(np.int16))
print('wrote intro_score.wav', mix.shape)
