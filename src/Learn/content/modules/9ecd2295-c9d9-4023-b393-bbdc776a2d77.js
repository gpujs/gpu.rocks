// Module: Spectrograms — uuid 9ecd2295-c9d9-4023-b393-bbdc776a2d77
// (short id 9ecd2295). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module postdates
// the uuid switch, and it declares no track (track membership lives in
// content/tracks.js).
//
// Spectrograms — Signal Processing. A single spectrum says what frequencies a
// signal contains and nothing at all about when. The short-time Fourier
// transform slides a window along and transforms each slice, which turns a
// one-dimensional signal into a two-dimensional picture — and into a genuinely
// good GPU workload, because every (frame, bin) cell is independent.
//
// Five tasks: one spectrum and its blindness to order → the STFT, one thread
// per (frame, bin) → the time/frequency trade, measured rather than asserted →
// decibels and colour, which is what makes the picture legible → the payoff,
// reading a chirp's sweep rate off its own ridge.
//
// COMPLEX NUMBERS. gpu.js has no complex type, so this track carries a complex
// signal as TWO PLANES of floats: `output: [n, 2]` is indexed `result[p][i]`,
// plane 0 real, plane 1 imaginary (a 2D output [w, h] is read [y][x], so
// [n, 2] gives exactly [plane][index]). Task 1 uses it; tasks 2–5 only ever
// need the magnitude, which is a real number, so they stay 2D.
//
// NO AUDIO. Learner code runs in a Web Worker, which has no AudioContext, no
// OfflineAudioContext and no navigator.mediaDevices — measured in a real
// browser, not assumed. Nothing here records, synthesises through WebAudio or
// plays back. Every signal is built deterministically below, which is also what
// makes every assertion exact; task 5 shows the real getUserMedia →
// AudioContext → AudioWorklet wiring as code and says plainly that the sandbox
// cannot run it.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, loop
// bounds come from this.constants (compile-time known, which is also what lets
// one kernel source be compiled twice at two window lengths), and every task
// passes in CPU mode. Sizes: n = 512 for the single DFT, 64 frames × 128 bins
// for the spectrograms, signals of 2,272–4,288 samples.
//
// NUMERIC CONTRACT (this has bitten three previous modules — do not rediscover
// it). Tests compute in float64; the GL backend computes in float32. A naive
// DFT sums N terms of magnitude ≤ 1 and divides by N, so per-term trig error
// averages out rather than accumulating: over 512 samples the worst case is
// ~1.2e-4 and over a 256-sample window ~6e-5. Tolerances are therefore
// DFT_EPS = 1e-3 (values ≤ 0.125, so 8× headroom) and SPEC_EPS = 2e-3 (values
// ≤ 0.25, so 30× headroom). Every frequency in this module is an exact
// multiple of SR / window, so no peak sits between two bins and no assertion
// sits near a decision boundary. The two tones in task 3 are 128 Hz apart —
// exactly two bins of the short window, so their relative phase repeats every
// hop and the peak count is identical in every frame.

import { seededRandom } from '../../engine/utils.js';

// ---- the grid ---------------------------------------------------------------

const SR = 4096; // nominal sample rate, Hz. A power of two, so every frequency
                 // below lands on an exact bin of every window length used.

// Task 1 — one spectrum of a whole 512-sample signal.
const DFT_N = 512;
const DFT_BINS = 256; // bins 0 … just below Nyquist

// Tasks 2, 4, 5 — the module's standard spectrogram.
const WIN = 256; // window length in samples: 62.5 ms at SR
const HOP = 64; // slide between frames: 15.625 ms — every sample seen 4 times
const FRAMES = 64; // (SIGNAL_N - WIN) / HOP + 1
const BINS = 128; // bins kept per frame: 0 … just below Nyquist
const SIGNAL_N = 4288; // (FRAMES - 1) * HOP + WIN — the last frame ends exactly
                       // on the last sample, so nothing reads past the end.

// Task 4's module card. On the card the picture is 512×512 with one pixel per
// (frame, bin) rather than a 4×2 block per cell, so it needs eight times the
// frames — which is a shorter hop, not a longer signal: the same second of
// audio analysed more often. The signal does have to grow by the 56 samples
// that keep the SIGNAL_N invariant above true at the finer hop, or the last
// frames would read past the end. (The frequency axis is interpolated in the
// kernel instead — see the spectrograms entry in capture-module-renders.mjs —
// because a longer window would buy bins by spending time resolution, which is
// the trade task 3 is about, and would change the picture rather than sharpen
// it.)
//
// The ANIMATED card sweeps the window anyway — but as task 4's TAPER, not as
// WIN, which is the whole reason the two were split. WIN stays 256 whatever the
// dial says, so the transform length, the frame count and this signal length are
// all untouched by the sweep; only the Hann bell inside each frame shortens.
// Frequency resolution moves and the frequency axis does not, which is what
// makes the trade legible instead of a picture rescaling itself.
const CARD_HOP = HOP / 8; // 8 samples between frames instead of 64
const CARD_FRAMES = 512;
const CARD_N = (CARD_FRAMES - 1) * CARD_HOP + WIN; // 4344

// Task 3 — the same signal through two window lengths, same hop, same frames.
const T_HOP = 32;
const T_FRAMES = 64;
const SHORT_WIN = 64;
const SHORT_BINS = 32;
const LONG_WIN = 256;
const LONG_BINS = 128;
const TRADE_N = 2272; // (T_FRAMES - 1) * T_HOP + LONG_WIN
const CLICK_AT = 1040; // the one-sample click, mid-window in the short frames

// The chirp: 256 Hz sweeping up at 1024 Hz/s. Frame f is centred at sample
// f·HOP + WIN/2, so its instantaneous frequency is 288 + 16f Hz — exactly bin
// 18 + f. The ridge is a perfect diagonal, one bin per frame.
const CHIRP_F0 = 256;
const CHIRP_RATE = 1024;

const DFT_EPS = 1e-3;
const SPEC_EPS = 2e-3;

// ---- signal construction ----------------------------------------------------

const round6 = v => Math.round(v * 1e6) / 1e6;

function hann(i, len) {
  return 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (len - 1));
}

// Task 1. A 256 Hz burst for the first half, a quieter 640 Hz burst for the
// second, each shaped by a Hann bell so the two tones read as clean lines
// rather than as the splatter a hard edit would produce. Both frequencies fit
// a whole number of cycles into their own 256-sample half AND into the full
// 512-sample buffer, so the peaks land on bins 32 and 80 exactly.
const memo = new Map();
function once(key, build) {
  if (!memo.has(key)) memo.set(key, build());
  return memo.get(key);
}

function twoToneAB() {
  return once('ab', () => {
    const half = DFT_N / 2;
    const s = new Array(DFT_N).fill(0);
    for (let i = 0; i < half; i++) {
      s[i] = round6(hann(i, half) * Math.cos((2 * Math.PI * 256 * i) / SR));
      s[half + i] = round6(0.6 * hann(i, half) * Math.cos((2 * Math.PI * 640 * i) / SR));
    }
    return s;
  });
}

// The SAME signal read backwards, which is the honest way to say "the other
// order": reversing puts the 640 Hz burst first and the 256 Hz burst second,
// and reversal conjugates a spectrum, so the two magnitude spectra agree to
// the last bit rather than merely closely.
function twoToneBA() {
  return once('ba', () => twoToneAB().slice().reverse());
}

// A different two-tone pair, for the private test: bins 24 and 96.
function twoToneAlt() {
  return once('alt', () => {
    const half = DFT_N / 2;
    const s = new Array(DFT_N).fill(0);
    for (let i = 0; i < half; i++) {
      s[i] = round6(0.8 * hann(i, half) * Math.cos((2 * Math.PI * 192 * i) / SR));
      s[half + i] = round6(0.5 * hann(i, half) * Math.sin((2 * Math.PI * 768 * i) / SR));
    }
    return s;
  });
}

function chirpSignal(f0 = CHIRP_F0, rate = CHIRP_RATE) {
  return once(`chirp${f0}:${rate}`, () => {
    const s = new Array(SIGNAL_N);
    for (let n = 0; n < SIGNAL_N; n++) {
      const t = n / SR;
      s[n] = round6(Math.sin(2 * Math.PI * (f0 * t + 0.5 * rate * t * t)));
    }
    return s;
  });
}

// Task 3. Two steady tones 128 Hz apart plus one sample driven hard — the
// sharpest event a sampled signal can hold.
function tradeSignal() {
  return once('trade', () => {
    const s = new Array(TRADE_N);
    for (let n = 0; n < TRADE_N; n++) {
      const t = n / SR;
      s[n] = Math.sin(2 * Math.PI * 512 * t) + 0.65 * Math.sin(2 * Math.PI * 640 * t);
    }
    s[CLICK_AT] += 6;
    return s.map(round6);
  });
}

// Task 4. Three plucked notes, staggered, each four harmonics deep with an
// exponential decay, over a faint seeded noise floor — which is what every
// real recording has, and what stops the quiet parts of the picture being a
// flat black rectangle.
//
// Takes a length so the card can have the same notes analysed at a finer hop
// (see CARD_N above): the voices keep their absolute sample positions, so a
// longer signal is this exact signal with more of the last note's tail — the
// first SIGNAL_N samples come out identical, sample for sample.
function noteSignal(length = SIGNAL_N) {
  return once(`notes:${length}`, () => {
    const s = new Array(length).fill(0);
    const voices = [
      { f: 192, start: 0, tau: 0.34 },
      { f: 256, start: 1408, tau: 0.3 },
      { f: 320, start: 2816, tau: 0.3 },
    ];
    const partials = [1, 0.5, 0.28, 0.15];
    for (const voice of voices) {
      for (let n = voice.start; n < length; n++) {
        const t = (n - voice.start) / SR;
        const env = Math.exp(-t / voice.tau) * (1 - Math.exp(-t / 0.004));
        let acc = 0;
        for (let h = 0; h < partials.length; h++) {
          acc += partials[h] * Math.sin(2 * Math.PI * voice.f * (h + 1) * t);
        }
        s[n] += env * acc;
      }
    }
    let peak = 0;
    for (let n = 0; n < length; n++) peak = Math.max(peak, Math.abs(s[n]));
    const rand = seededRandom(9137);
    for (let n = 0; n < length; n++) {
      s[n] = round6(s[n] / peak + 0.006 * (rand() * 2 - 1));
    }
    return s;
  });
}

// ---- reference transforms (float64, host side) ------------------------------

// The forward DFT, normalised by n: bin k of a real signal, as two planes.
function refSpectrum(signal, bins) {
  const n = signal.length;
  const re = new Array(bins);
  const im = new Array(bins);
  for (let k = 0; k < bins; k++) {
    let r = 0;
    let i = 0;
    for (let t = 0; t < n; t++) {
      const angle = (-2 * Math.PI * k * t) / n;
      r += signal[t] * Math.cos(angle);
      i += signal[t] * Math.sin(angle);
    }
    re[k] = r / n;
    im[k] = i / n;
  }
  return { re, im };
}

// The reference spectrogram: [bin][frame] magnitudes, Hann-windowed and
// normalised by the window length. `windowed: false` is the "forgot the
// window" mistake, built only when a test is already failing.
function refStft(signal, { win, hop, frames, bins, windowed = true }) {
  const out = [];
  for (let b = 0; b < bins; b++) out.push(new Float64Array(frames));
  for (let f = 0; f < frames; f++) {
    const start = f * hop;
    for (let b = 0; b < bins; b++) {
      let re = 0;
      let im = 0;
      for (let t = 0; t < win; t++) {
        const sample = signal[start + t];
        const s = windowed ? sample * hann(t, win) : sample;
        const angle = (-2 * Math.PI * b * t) / win;
        re += s * Math.cos(angle);
        im += s * Math.sin(angle);
      }
      out[b][f] = Math.sqrt(re * re + im * im) / win;
    }
  }
  return out;
}

// The standard spectrogram of a named signal, built at most once per session:
// a 64 × 128 reference costs about 2 million multiply-adds.
function standardStft(name, signal) {
  return once(`stft:${name}`, () =>
    refStft(signal, { win: WIN, hop: HOP, frames: FRAMES, bins: BINS })
  );
}

function ridgeOf(spec) {
  const line = [];
  for (let f = 0; f < spec[0].length; f++) {
    let best = 0;
    for (let b = 1; b < spec.length; b++) if (spec[b][f] > spec[best][f]) best = b;
    line.push(best);
  }
  return line;
}

// ---- near-miss diagnosis ----------------------------------------------------
//
// A failing assert that reports two numbers tells a learner nothing about WHICH
// slip produced them. A probe pairs the value one specific known mistake would
// produce with a sentence naming that mistake; diagnose() speaks only when the
// observed value matches a probe within the test's own tolerance AND the
// correct value does not — so a bin where two candidates coincide (bin 32,
// where the imaginary part is zero and conjugating changes nothing) stays
// silent, as do observations matching probes that disagree with each other.
// A wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The grid form. One matching cell is weak evidence when a candidate is a
// whole rearrangement of the data — a transposed spectrogram agrees with the
// real one wherever the picture happens to be symmetric — so these probes must
// predict EVERY cell in the list, and disagree with the right answer somewhere,
// before they may speak. A missing cell makes the comparison NaN, which fails.
function diagnoseCells(cells, got, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (const [r, c] of cells) {
        const candidate = value(r, c);
        if (!(Math.abs(got(r, c) - candidate) <= eps)) return false;
        if (Math.abs(expected(r, c) - candidate) > eps) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// First cell of a [bin][frame] grid that disagrees with the reference, or null.
function firstMismatch(out, ref, eps) {
  for (let b = 0; b < ref.length; b++) {
    const row = out[b];
    for (let f = 0; f < ref[b].length; f++) {
      const got = row ? row[f] : undefined;
      if (!(Math.abs(got - ref[b][f]) <= eps)) return { b, f, got };
    }
  }
  return null;
}

// ---- task 1 probes ----------------------------------------------------------

const PLANES_SWAPPED =
  'the two planes are swapped — plane 0 (this.thread.y === 0) carries the real ' +
  'part and plane 1 the imaginary part';
const NOT_AVERAGED =
  'the sums were never divided by this.constants.n — every bin is 512 times too big';
const CONJUGATED =
  'that is the conjugate: the angle came out as +2πki/n. The forward transform ' +
  'winds the other way — angle = -2 * Math.PI * k * i / this.constants.n. The ' +
  'magnitudes still look right, which is exactly why this one hides';

function spectrumProbes(ref, plane, k) {
  if (plane === 0) {
    return [
      [ref.im[k], PLANES_SWAPPED],
      [ref.re[k] * DFT_N, NOT_AVERAGED],
    ];
  }
  return [
    [-ref.im[k], CONJUGATED],
    [ref.re[k], PLANES_SWAPPED],
    [ref.im[k] * DFT_N, NOT_AVERAGED],
  ];
}

// ---- spectrogram probes -----------------------------------------------------

const NO_WINDOW =
  'every frame was transformed unwindowed. A slice cut straight out of a signal ' +
  'ends on a step, and the transform reports that step as energy in every bin — ' +
  'multiply each sample by its Hann value before you accumulate it';
const HOP_IS_WIN =
  'the frames were started this.constants.win apart instead of this.constants.hop ' +
  'apart. Hop and window are different numbers: frame f starts at ' +
  'f * this.constants.hop, and the window it then reads is this.constants.win ' +
  'samples long';
const NOT_NORMALISED =
  'the magnitude was never divided by this.constants.win — every cell is a ' +
  'window-length times too big';
const POWER_NOT_MAGNITUDE =
  'that is re * re + im * im — the power, not the magnitude. Take the square root';
const SPEC_TRANSPOSED =
  'time and frequency are swapped: cell [bin][frame] is holding the value that ' +
  'belongs to [frame][bin]. With output [FRAMES, BINS] the coordinates are ' +
  'frame = this.thread.x and bin = this.thread.y — x is time';
const READ_PAST_END =
  'some cells came back NaN, so a frame read past the end of the signal. Frame f ' +
  'covers samples f * hop … f * hop + win - 1, and the last of them ends on the ' +
  'very last sample — so check both ends: start is frame * this.constants.hop, ' +
  'and the loop runs t < this.constants.win, not t <= it';

// Builds the diagnosis for a wrong spectrogram. Every reference it needs
// beyond the correct one is built HERE, on the failure path only: a rectangular
// reference costs another two million multiply-adds and no passing run should
// pay for it.
//
// ORDER MATTERS. Both structural mistakes below — the transpose and the
// window-sized hop — ALSO run some frames off the end of the signal, so a
// NaN check placed first would answer every one of them with the same wrong
// sentence. The structural probes read only cells that stay in range under
// their own mistake, so they get asked first; NaN is the fallback for the
// overruns nothing else explains.
function spectrogramHint(out, ref, signal, opts) {
  const { win, hop, frames, bins, eps } = opts;
  const got = (r, c) => (out[r] ? out[r][c] : NaN);
  const expected = (r, c) => ref[r][c];

  const wide = [];
  for (const b of [8, 20, 40, 63, 90, 120]) {
    for (const f of [1, 5, 17, 33, 50, 62]) {
      if (b < bins && f < frames) wide.push([b, f]);
    }
  }

  // Cheap probes first: both are pure rescalings of the right answer.
  const scaled = diagnoseCells(wide, got, expected, eps, [
    [(r, c) => ref[r][c] * win, NOT_NORMALISED],
    [(r, c) => ref[r][c] * ref[r][c] * win, POWER_NOT_MAGNITUDE],
    [(r, c) => ref[r][c] * ref[r][c], POWER_NOT_MAGNITUDE],
  ]);
  if (scaled) return scaled;

  // Time and frequency swapped. Only cells inside the square corner have a
  // transpose at all, and a cell on the diagonal is its own transpose, so those
  // can never show the mistake.
  const square = Math.min(bins, frames);
  const corner = [];
  for (const r of [5, 17, 33, 50]) {
    for (const c of [8, 20, 40, 60]) {
      if (r < square && c < square && r !== c) corner.push([r, c]);
    }
  }
  if (corner.length) {
    const flipped = diagnoseCells(corner, got, expected, eps, [
      [(r, c) => ref[c][r], SPEC_TRANSPOSED],
    ]);
    if (flipped) return flipped;
  }

  // Frames started a whole window apart. Only the frames that still fit inside
  // the signal can be predicted — past that the mistake runs off the end, which
  // is a different symptom.
  const fits = [];
  for (const [b, f] of wide) {
    if (f * win + win <= signal.length) fits.push([b, f]);
  }
  if (fits.length) {
    const stride = refStft(signal, { win, hop: win, frames, bins });
    const strideHint = diagnoseCells(fits, got, expected, eps, [
      [(r, c) => stride[r][c], HOP_IS_WIN],
    ]);
    if (strideHint) return strideHint;
  }

  // Nothing structural explained it. An overrun that gets this far is the
  // ordinary off-by-one, and NaN is the only trace the CPU backend leaves of
  // one (the GL backend clamps the read instead).
  for (let b = 0; b < bins; b++) {
    for (let f = 0; f < frames; f++) {
      if (Number.isNaN(got(b, f))) return READ_PAST_END;
    }
  }

  const rect = refStft(signal, { win, hop, frames, bins, windowed: false });
  return diagnoseCells(wide, got, expected, eps, [[(r, c) => rect[r][c], NO_WINDOW]]);
}

// The one assertion every spectrogram task makes, with the whole grid checked
// and nothing expensive computed unless it fails.
function assertSpectrogram(ctx, out, ref, signal, opts) {
  const miss = firstMismatch(out, ref, opts.eps);
  if (!miss) return;
  const hint = spectrogramHint(out, ref, signal, opts);
  ctx.assertClose(
    miss.got,
    ref[miss.b][miss.f],
    opts.eps,
    hint || `bin ${miss.b}, frame ${miss.f}`
  );
}

// ---- task 4: the colour ramp, host side -------------------------------------

// Byte-for-byte the ramp the task hands the learner. Loudness is carried by
// value, which is the part that has to be monotone; the hue sweep is for
// legibility (Colour Spaces argues the case).
function rampColour(v) {
  return [
    Math.min(1, 1.43 * v),
    0.95 * Math.max(0, Math.min(1, (v - 0.45) / 0.55)),
    0.55 * Math.min(1, v / 0.32) -
      0.37 * Math.max(0, Math.min(1, (v - 0.32) / 0.4)) +
      0.54 * Math.max(0, Math.min(1, (v - 0.72) / 0.28)),
  ];
}

const DB_RANGE = 60;

function dbValue(mag, peak) {
  const db = 20 * Math.log10(mag / peak + 1e-9);
  return Math.max(0, Math.min(1, 1 + db / DB_RANGE));
}

const LINEAR_SCALING =
  'your kernel is fine — your scaling is wrong. Those pixels are the magnitude ' +
  'mapped straight to the ramp, and the magnitudes span three orders of ' +
  'magnitude, so almost every cell lands in the bottom sliver of it. Convert to ' +
  'decibels first: 20 * Math.log10(mag / peak + 1e-9)';
const NATURAL_LOG =
  'that is Math.log, the natural logarithm, where the decibel definition wants ' +
  'the base-10 one. Everything comes out 2.3 times further down the scale — use ' +
  'Math.log10, which gpu.js supports directly';
const NO_PEAK_DIVISION =
  'the magnitude was not divided by peak before the logarithm, so the whole ' +
  'picture sits about 18 dB lower than it should. The loudest cell has to map ' +
  'to 0 dB for the range to mean anything';

// ---- shared kernel sources (given to the learner in later tasks) -------------

const STFT_BODY = `const spectrogram = gpu.createKernel(function (signal) {
  const frame = this.thread.x;
  const bin = this.thread.y;
  const start = frame * this.constants.hop;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.win; t++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / (this.constants.win - 1));
    const s = signal[start + t] * w;
    const angle = (-2 * Math.PI * bin * t) / this.constants.win;
    re += s * Math.cos(angle);
    im += s * Math.sin(angle);
  }
  return Math.sqrt(re * re + im * im) / this.constants.win;
}, {
  output: [FRAMES, BINS],
  constants: { win: WIN, hop: HOP },
});`;

// Task 4's copy of the same kernel, with the window length pulled out onto a
// dial. Two lengths, not one: `win` is the TRANSFORM length and stays 256, so
// bin b is always frequency b·SR/win and the frequency axis never moves; `taper`
// is how many of those 256 samples the Hann bell actually covers, and the rest
// are multiplied by nothing and contribute zero. That is zero-padding — librosa
// spells it `n_fft` against `win_length` — and it is the only way to put task
// 3's trade on a slider without the picture changing shape underneath it.
// taper === win is byte-for-byte the kernel above, which is why the default
// leaves this task exactly where it was.
const STFT_TAPERED = `const spectrogram = gpu.createKernel(function (signal) {
  const frame = this.thread.x;
  const bin = this.thread.y;
  const start = frame * this.constants.hop;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.taper; t++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / (this.constants.taper - 1));
    const s = signal[start + t] * w;
    const angle = (-2 * Math.PI * bin * t) / this.constants.win;
    re += s * Math.cos(angle);
    im += s * Math.sin(angle);
  }
  return Math.sqrt(re * re + im * im) / this.constants.win;
}, {
  output: [FRAMES, BINS],
  constants: { win: WIN, hop: HOP, taper: TAPER },
});`;

const RAMP_SOURCE = `  // Already written: the ramp. Black → indigo → magenta → orange → cream.
  const r = Math.min(1, 1.43 * v);
  const g = 0.95 * Math.max(0, Math.min(1, (v - 0.45) / 0.55));
  const b = 0.55 * Math.min(1, v / 0.32)
          - 0.37 * Math.max(0, Math.min(1, (v - 0.32) / 0.4))
          + 0.54 * Math.max(0, Math.min(1, (v - 0.72) / 0.28));
  this.color(r, g, b, 1);`;

// ---- kernel lookup ----------------------------------------------------------

// Kernels are identified by the shape they produce, never by creation order:
// a learner is free to declare them the other way round.
function kernelWithOutput(ctx, dims) {
  return (
    ctx.kernels.find(k => {
      if (!k.kernel || !k.kernel.output) return false;
      const out = Array.from(k.kernel.output);
      return out.length === dims.length && dims.every((d, i) => out[i] === d);
    }) || null
  );
}

function graphicalKernel(ctx) {
  return ctx.kernels.find(k => k.kernel && k.kernel.graphical) || null;
}

function loggedNumbers(logs) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?/g);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

export default {
  uuid: '9ecd2295-c9d9-4023-b393-bbdc776a2d77',
  version: 1,
  slug: 'spectrograms',
  title: 'Spectrograms',
  blurb:
    'Slide a window along a signal and transform every slice — a picture of frequency over time, one thread per <code>(frame, bin)</code>.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'one-spectrum',
      title: 'One Spectrum, No Clock',
      intro: `<p>The discrete Fourier transform asks one question per <strong>bin</strong>: how much
        of this signal looks like a wave that fits exactly <em>k</em> whole cycles into the buffer?
        Answering it is a sum over every sample, and every bin's sum is independent of every other
        one — so it is one thread per bin, each pulling the whole signal. The answer is a
        <em>complex</em> number: how much, and at what phase.</p>
        <p>gpu.js has no complex type, so this track carries complex data as <strong>two planes of
        floats</strong>. <code>output: [n, 2]</code> is read <code>result[p][i]</code> — plane
        <code>p = 0</code> is the real part, plane 1 the imaginary part, <code>i</code> the bin. (A
        2D output <code>[w, h]</code> is indexed <code>[y][x]</code>, so <code>[n, 2]</code> gives
        exactly <code>[plane][bin]</code>.) What you usually plot is the <strong>magnitude</strong>,
        <code>√(re² + im²)</code>: how much of that frequency is present, phase discarded.</p>
        <p>Two signals here. <code>orderAB</code> plays a 256 Hz tone and then a quieter 640 Hz
        one. <code>orderBA</code> plays the same two tones the other way round — it <em>is</em>
        <code>orderAB</code> read backwards. Transform both. The two spectra will not merely look
        alike; they agree to the last bit, and that is a theorem rather than a coincidence:
        reversing a signal conjugates its spectrum, and conjugating does not change a magnitude.
        A spectrum knows what is in a signal. It has no idea when.</p>`,
      goal: `<strong>Goal:</strong> fill in both planes of the spectrum — plane 0 the real part,
        plane 1 the imaginary part, each divided by <code>this.constants.n</code>.`,
      requirements: [
        'Keep <code>output: [256, 2]</code> — 256 bins across, two planes; the result is read <code>spectrum[plane][bin]</code>',
        'Sum over all <code>this.constants.n</code> samples with <code>angle = -2 * Math.PI * k * i / this.constants.n</code>',
        'Accumulate <code>re += signal[i] * Math.cos(angle)</code> and <code>im += signal[i] * Math.sin(angle)</code>',
        'Divide both sums by <code>this.constants.n</code>, and return <code>re</code> on plane 0, <code>im</code> on plane 1',
      ],
      hints: [
        {
          title: 'Hint 1 — which cell is mine?',
          body: `<p><code>this.thread.x</code> is the bin (0…255) and <code>this.thread.y</code> is
            the plane (0 or 1). Both threads of a bin do the <em>same</em> sum and each keeps half
            of it — wasteful, and at 512 samples not worth caring about. An FFT is the version that
            stops paying for it.</p>`,
        },
        {
          title: 'Hint 2 — the loop',
          body: `<pre><code>for (let i = 0; i &lt; this.constants.n; i++) {
  const angle = (-2 * Math.PI * k * i) / this.constants.n;
  re += signal[i] * Math.cos(angle);
  im += signal[i] * Math.sin(angle);
}
re = re / this.constants.n;
im = im / this.constants.n;</code></pre>`,
        },
        {
          title: 'Hint 3 — the sign',
          body: `<p>The forward transform winds <em>clockwise</em>: the angle is negative. Drop the
            minus and every bin comes back conjugated — the imaginary plane flips sign while the
            magnitudes stay exactly right, which is what makes this one so good at hiding.</p>`,
        },
      ],
      transfer: `Split-complex — one array of real parts, one of imaginary — is how cuFFT, rocFFT,
        Apple's vDSP and NumPy's <code>.real</code>/<code>.imag</code> views all lay complex data
        out, and it is what you just wrote. Nobody ships a naive DFT, though: it is O(n²), and the
        FFT rebuilds it as a log₂ n ladder of butterflies — the same halving-ladder shape the
        Reductions module climbs, driven by a JS loop over a stride-taking kernel. At n = 512 the
        naive version is 262,144 multiply-adds, which a GPU eats without noticing.`,
      starterCode: `// One thread per (bin, plane). 512 samples in, 256 bins out — and each
// bin's answer is a complex number, carried as two planes of floats.
const gpu = new GPU({ mode });

const spectrum = gpu.createKernel(function (signal) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;

  // TODO: sum over all this.constants.n samples.
  //   angle = -2 * Math.PI * k * i / this.constants.n
  //   re += signal[i] * Math.cos(angle)
  //   im += signal[i] * Math.sin(angle)
  // Then divide BOTH by this.constants.n.

  if (this.thread.y === 0) {
    return re;
  }
  return im;
}, {
  output: [256, 2],
  constants: { n: 512 },
});

// Magnitude is the length of the complex number.
function magnitudes(spec) {
  const mag = [];
  for (let k = 0; k < 256; k++) {
    mag.push(Math.sqrt(spec[0][k] * spec[0][k] + spec[1][k] * spec[1][k]));
  }
  return mag;
}

function peakBin(mag) {
  let best = 0;
  for (let k = 1; k < mag.length; k++) if (mag[k] > mag[best]) best = k;
  return best;
}

const magAB = magnitudes(await spectrum(orderAB));
const magBA = magnitudes(await spectrum(orderBA));

let worst = 0;
for (let k = 0; k < 256; k++) worst = Math.max(worst, Math.abs(magAB[k] - magBA[k]));

console.log('A-then-B peaks at bin', peakBin(magAB));
console.log('B-then-A peaks at bin', peakBin(magBA));
console.log('largest disagreement between the two spectra:', worst);
`,
      solutionCode: `// One thread per (bin, plane). 512 samples in, 256 bins out — and each
// bin's answer is a complex number, carried as two planes of floats.
const gpu = new GPU({ mode });

const spectrum = gpu.createKernel(function (signal) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;

  for (let i = 0; i < this.constants.n; i++) {
    const angle = (-2 * Math.PI * k * i) / this.constants.n;
    re += signal[i] * Math.cos(angle);
    im += signal[i] * Math.sin(angle);
  }
  re = re / this.constants.n;
  im = im / this.constants.n;

  if (this.thread.y === 0) {
    return re;
  }
  return im;
}, {
  output: [256, 2],
  constants: { n: 512 },
});

// Magnitude is the length of the complex number.
function magnitudes(spec) {
  const mag = [];
  for (let k = 0; k < 256; k++) {
    mag.push(Math.sqrt(spec[0][k] * spec[0][k] + spec[1][k] * spec[1][k]));
  }
  return mag;
}

function peakBin(mag) {
  let best = 0;
  for (let k = 1; k < mag.length; k++) if (mag[k] > mag[best]) best = k;
  return best;
}

const magAB = magnitudes(await spectrum(orderAB));
const magBA = magnitudes(await spectrum(orderBA));

let worst = 0;
for (let k = 0; k < 256; k++) worst = Math.max(worst, Math.abs(magAB[k] - magBA[k]));

console.log('A-then-B peaks at bin', peakBin(magAB));
console.log('B-then-A peaks at bin', peakBin(magBA));
console.log('largest disagreement between the two spectra:', worst);
`,
      inputs: () => ({ orderAB: twoToneAB(), orderBA: twoToneBA() }),
      publicTests: [
        {
          name: 'the spectrum comes back as two planes of 256 bins',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(twoToneAB());
            ctx.assert(out && out.length, 'the kernel returned nothing');
            ctx.assert(
              out.length === 2,
              out.length === 256
                ? 'the output is [2, 256]; it should be [256, 2] — a 2D output [w, h] is read [y][x], so [256, 2] gives spectrum[plane][bin]'
                : `expected 2 planes, got ${out.length}`
            );
            ctx.assert(
              out[0] && out[0].length === DFT_BINS,
              `each plane should hold ${DFT_BINS} bins, got ${out[0] && out[0].length}`
            );
          },
        },
        {
          name: 'plane 0 is the real part, plane 1 the imaginary part',
          run: async ctx => {
            const signal = twoToneAB();
            const out = await ctx.kernel(signal);
            const ref = once('ref:ab', () => refSpectrum(signal, DFT_BINS));
            for (const k of [0, 16, 31, 32, 33, 79, 80, 81, 120, 200]) {
              for (const plane of [0, 1]) {
                const expected = plane === 0 ? ref.re[k] : ref.im[k];
                const got = out[plane][k];
                const hint = diagnose(got, expected, DFT_EPS, spectrumProbes(ref, plane, k));
                ctx.assertClose(
                  got,
                  expected,
                  DFT_EPS,
                  hint || `${plane === 0 ? 'real' : 'imaginary'} part of bin ${k}`
                );
              }
            }
          },
        },
        {
          name: 'both orders give the same magnitudes, bin for bin',
          run: async ctx => {
            const ab = await ctx.kernel(twoToneAB());
            const ba = await ctx.kernel(twoToneBA());
            const mag = spec => {
              const m = [];
              for (let k = 0; k < DFT_BINS; k++) {
                m.push(Math.sqrt(spec[0][k] * spec[0][k] + spec[1][k] * spec[1][k]));
              }
              return m;
            };
            const magAB = mag(ab);
            const magBA = mag(ba);
            let worst = 0;
            for (let k = 0; k < DFT_BINS; k++) {
              worst = Math.max(worst, Math.abs(magAB[k] - magBA[k]));
            }
            ctx.assert(
              worst <= 2 * DFT_EPS,
              `the two spectra should be identical, but they differ by up to ${worst.toExponential(2)} — ` +
                'reversing a signal conjugates its spectrum, which leaves every magnitude alone, so ' +
                'either the transform is wrong or the two runs did not use the same kernel'
            );
            const peak = m => {
              let best = 0;
              for (let k = 1; k < m.length; k++) if (m[k] > m[best]) best = k;
              return best;
            };
            ctx.assert(
              peak(magAB) === 32 && peak(magBA) === 32,
              `both spectra should peak at bin 32 (the 256 Hz tone), got ${peak(magAB)} and ${peak(magBA)}`
            );
            ctx.assertClose(magAB[80], magBA[80], 2 * DFT_EPS, 'the 640 Hz tone, bin 80');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const signal = twoToneAlt();
            const out = await ctx.kernel(signal);
            const ref = once('ref:alt', () => refSpectrum(signal, DFT_BINS));
            for (let k = 0; k < DFT_BINS; k++) {
              for (const plane of [0, 1]) {
                const expected = plane === 0 ? ref.re[k] : ref.im[k];
                const hint = diagnose(out[plane][k], expected, DFT_EPS, spectrumProbes(ref, plane, k));
                ctx.assertClose(out[plane][k], expected, DFT_EPS, hint || `plane ${plane}, bin ${k}`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The reversal identity on a signal this task never showed.
            const forward = twoToneAlt();
            const backward = forward.slice().reverse();
            const a = await ctx.kernel(forward);
            const b = await ctx.kernel(backward);
            for (let k = 0; k < DFT_BINS; k++) {
              const ma = Math.sqrt(a[0][k] * a[0][k] + a[1][k] * a[1][k]);
              const mb = Math.sqrt(b[0][k] * b[0][k] + b[1][k] * b[1][k]);
              ctx.assertClose(mb, ma, 2 * DFT_EPS, `magnitude of bin ${k} under reversal`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'stft',
      title: 'Slice, Window, Transform',
      intro: `<p>The fix falls out of saying the problem aloud: stop transforming the whole signal.
        Chop it into short <strong>frames</strong>, transform each one on its own, and stand the
        answers side by side. That is the <strong>short-time Fourier transform</strong>, and the
        picture it makes — time across, frequency up, brightness for how much — is a spectrogram.</p>
        <p>Two numbers describe the chopping and they are <em>not</em> the same number.
        <strong>Window</strong> is how much signal one frame sees (256 samples here, 62.5 ms).
        <strong>Hop</strong> is how far the window slides between frames (64 samples). Setting hop
        equal to the window puts the frames edge to edge, so anything straddling a boundary is
        split in half; setting hop larger leaves gaps the transform never looks at. Hop smaller —
        <strong>overlap</strong> — costs work and buys a smooth time axis: here every sample is
        seen by four different frames. One frame more: cut a slice out with scissors and it ends on
        a step, which the transform faithfully reports as energy at every frequency. Windowing &amp;
        Spectral Leakage makes a module of that; here, just take the fix and multiply each frame by
        a Hann bell first.</p>
        <p>The GPU shape is the good part: <strong>one thread per (frame, bin)</strong>. No cell
        needs anything another cell computed, so the entire picture is one launch of 8,192 threads,
        each gathering its own 256 samples. <code>output: [FRAMES, BINS]</code> puts time on x and
        frequency on y — which is how the picture is drawn, and means the result reads
        <code>spec[bin][frame]</code>.</p>`,
      goal: `<strong>Goal:</strong> build the spectrogram of <code>signal</code> — get the frame
        count right, then fill in the kernel body so each cell holds the windowed magnitude.`,
      requirements: [
        'Set <code>FRAMES</code> to the number of <em>whole</em> windows that fit: <code>Math.floor((signal.length - WIN) / HOP) + 1</code>',
        'Frame <code>this.thread.x</code> starts at <code>this.thread.x * this.constants.hop</code> and reads <code>this.constants.win</code> samples',
        'Multiply each sample by its Hann value <code>0.5 - 0.5 * Math.cos(2 * Math.PI * t / (this.constants.win - 1))</code> before accumulating',
        'Return <code>Math.sqrt(re * re + im * im) / this.constants.win</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — which sample is mine?',
          body: `<p>Frame <code>f</code>, tap <code>t</code> of that frame, is
            <code>signal[f * hop + t]</code>. The tap index <code>t</code> also drives the window
            and the angle — both are measured from the <em>start of the frame</em>, never from the
            start of the signal.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<pre><code>for (let t = 0; t &lt; this.constants.win; t++) {
  const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / (this.constants.win - 1));
  const s = signal[start + t] * w;
  const angle = (-2 * Math.PI * bin * t) / this.constants.win;
  re += s * Math.cos(angle);
  im += s * Math.sin(angle);
}
return Math.sqrt(re * re + im * im) / this.constants.win;</code></pre>`,
        },
        {
          title: 'Hint 3 — counting frames',
          body: `<p><code>signal.length / HOP</code> counts frames that start inside the signal,
            including three at the end whose windows run off it. A frame needs a <em>whole</em>
            window, so the last legal start is <code>signal.length - WIN</code>:</p>
<pre><code>const FRAMES = Math.floor((signal.length - WIN) / HOP) + 1;</code></pre>
<p>With 4,288 samples that is 64, and the last frame ends exactly on the last sample.</p>`,
        },
      ],
      transfer: `You have written <code>librosa.stft</code> / <code>torchaudio.Spectrogram</code> /
        MATLAB's <code>spectrogram</code>, and the production version differs in exactly one place:
        the inner loop becomes an FFT and the whole thing becomes a <em>batched</em> transform —
        one cuFFT or VkFFT plan, all 64 frames at once. The 2D launch you just wrote is already the
        right decomposition; only the O(n) inner sum gets replaced by an O(log n) ladder. Note what
        you did <em>not</em> need: no atomics, no shared memory, no thread writing to a cell it does
        not own. Every thread pulled what it wanted. That is the gather formulation, and it is why
        this maps onto every platform unchanged.`,
      starterCode: `// The short-time Fourier transform: one thread per (frame, bin).
const gpu = new GPU({ mode });

const WIN = 256;   // samples each frame sees — 62.5 ms at 4096 Hz
const HOP = 64;    // how far the window slides between frames
const BINS = 128;  // bins kept per frame: 0 … just below Nyquist

// TODO: how many WHOLE windows fit? A frame whose window runs off the end
// of the signal is not a frame.
const FRAMES = Math.floor(signal.length / HOP);

const spectrogram = gpu.createKernel(function (signal) {
  const frame = this.thread.x;
  const bin = this.thread.y;
  const start = frame * this.constants.hop;
  let re = 0;
  let im = 0;

  // TODO: walk this frame's this.constants.win samples.
  //   w     = 0.5 - 0.5 * Math.cos(2 * Math.PI * t / (this.constants.win - 1))
  //   s     = signal[start + t] * w
  //   angle = -2 * Math.PI * bin * t / this.constants.win
  //   re += s * Math.cos(angle);   im += s * Math.sin(angle);
  // Then return the magnitude, divided by this.constants.win.

  return 0;
}, {
  output: [FRAMES, BINS],
  constants: { win: WIN, hop: HOP },
});

const spec = await spectrogram(signal);
console.log('spectrogram:', spec.length, 'bins ×', spec[0].length, 'frames');

// signal is a chirp — a tone sliding steadily upward — so the loudest bin
// should climb, one bin per frame.
for (const f of [0, 16, 32, 48]) {
  let best = 0;
  for (let b = 1; b < BINS; b++) if (spec[b][f] > spec[best][f]) best = b;
  console.log('frame', f, 'peaks at bin', best);
}
`,
      solutionCode: `// The short-time Fourier transform: one thread per (frame, bin).
const gpu = new GPU({ mode });

const WIN = 256;   // samples each frame sees — 62.5 ms at 4096 Hz
const HOP = 64;    // how far the window slides between frames
const BINS = 128;  // bins kept per frame: 0 … just below Nyquist

// A frame needs a WHOLE window, so the last legal start is signal.length - WIN.
const FRAMES = Math.floor((signal.length - WIN) / HOP) + 1;

const spectrogram = gpu.createKernel(function (signal) {
  const frame = this.thread.x;
  const bin = this.thread.y;
  const start = frame * this.constants.hop;
  let re = 0;
  let im = 0;

  for (let t = 0; t < this.constants.win; t++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / (this.constants.win - 1));
    const s = signal[start + t] * w;
    const angle = (-2 * Math.PI * bin * t) / this.constants.win;
    re += s * Math.cos(angle);
    im += s * Math.sin(angle);
  }

  return Math.sqrt(re * re + im * im) / this.constants.win;
}, {
  output: [FRAMES, BINS],
  constants: { win: WIN, hop: HOP },
});

const spec = await spectrogram(signal);
console.log('spectrogram:', spec.length, 'bins ×', spec[0].length, 'frames');

// signal is a chirp — a tone sliding steadily upward — so the loudest bin
// should climb, one bin per frame.
for (const f of [0, 16, 32, 48]) {
  let best = 0;
  for (let b = 1; b < BINS; b++) if (spec[b][f] > spec[best][f]) best = b;
  console.log('frame', f, 'peaks at bin', best);
}
`,
      inputs: () => ({ signal: chirpSignal() }),
      publicTests: [
        {
          name: 'the picture is <code>128</code> bins tall and <code>64</code> frames wide',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(chirpSignal());
            ctx.assert(out && out.length, 'the kernel returned nothing');
            ctx.assert(
              out.length === BINS,
              out.length === FRAMES
                ? 'the output is [BINS, FRAMES]; it should be [FRAMES, BINS] — a 2D output [w, h] is read [y][x], and time belongs on x'
                : `expected ${BINS} rows (one per bin), got ${out.length}`
            );
            const wide = out[0] && out[0].length;
            ctx.assert(
              wide === FRAMES,
              wide === 67
                ? 'got 67 frames — that is signal.length / HOP, which counts three frames whose windows run off the end of the signal. A frame needs a whole window: Math.floor((signal.length - WIN) / HOP) + 1 = 64'
                : `expected ${FRAMES} frames, got ${wide}`
            );
          },
        },
        {
          name: 'every cell is the windowed magnitude of its own frame',
          run: async ctx => {
            const signal = chirpSignal();
            const out = await ctx.kernel(signal);
            ctx.assert(out && out.length === BINS && out[0].length === FRAMES, 'wrong output shape');
            assertSpectrogram(ctx, out, standardStft('chirp', signal), signal, {
              win: WIN,
              hop: HOP,
              frames: FRAMES,
              bins: BINS,
              eps: SPEC_EPS,
            });
          },
        },
        {
          name: 'the chirp draws a diagonal — the loudest bin climbs by one per frame',
          run: async ctx => {
            const out = await ctx.kernel(chirpSignal());
            for (const f of [0, 16, 32, 48, 63]) {
              let best = 0;
              for (let b = 1; b < BINS; b++) if (out[b][f] > out[best][f]) best = b;
              ctx.assert(
                best === 18 + f,
                `frame ${f} should peak at bin ${18 + f}, got ${best} — the chirp rises 16 Hz per ` +
                  'frame and the bins are 16 Hz apart, so the ridge is exactly one bin per frame'
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A chirp sweeping the other way: same kernel, a signal it never saw.
            const signal = chirpSignal(1536, -1024);
            const out = await ctx.kernel(signal);
            assertSpectrogram(ctx, out, standardStft('down', signal), signal, {
              win: WIN,
              hop: HOP,
              frames: FRAMES,
              bins: BINS,
              eps: SPEC_EPS,
            });
            const line = ridgeOf(out);
            for (let f = 0; f < FRAMES; f++) {
              ctx.assert(line[f] === 94 - f, `descending chirp: frame ${f} should peak at bin ${94 - f}, got ${line[f]}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Silence in, silence out — and no NaN from a frame that overran.
            const quiet = new Array(SIGNAL_N).fill(0);
            const out = await ctx.kernel(quiet);
            for (let b = 0; b < BINS; b++) {
              for (let f = 0; f < FRAMES; f++) {
                ctx.assert(
                  Number.isFinite(out[b][f]),
                  `bin ${b}, frame ${f} is ${out[b][f]} — ${READ_PAST_END}`
                );
                ctx.assertClose(out[b][f], 0, 1e-6, `a silent signal should give a silent picture (bin ${b}, frame ${f})`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'resolution-trade',
      title: 'Fine in Time, or Fine in Frequency',
      intro: `<p>Shorten the window and the time axis sharpens. Lengthen it and the frequency axis
        does. You cannot have both, and this is not a gpu.js limitation or a numerical one — it is
        a theorem. A window <em>w</em> samples long cannot tell two frequencies apart unless they
        differ by roughly <code>2 · SR / w</code>, and it cannot place an event in time to better
        than <em>w</em> samples. Multiply those two together and the window length cancels: the
        product is a constant, and picking a window is choosing which end of it to spend.</p>
        <p>This signal has both kinds of detail. Two steady tones, 512 Hz and 640 Hz, run the whole
        way through; and at 0.254 s there is a single sample driven hard — a <strong>click</strong>,
        the sharpest event a sampled signal can hold. Build the same spectrogram twice: same hop,
        same 64 frames, one with a 64-sample window and one with a 256-sample window. Then measure
        both, on both axes. The two measurements are already written; do not take anyone's word for
        which window wins, because neither of them does.</p>
        <p>One kernel <em>source</em>, two compilations. <code>this.constants.win</code> is fixed
        when a kernel is created — which is precisely why it is legal as a loop bound — so handing
        <code>createKernel</code> a different constants object turns the same function body into a
        different kernel. Same trick a CUDA template parameter plays.</p>`,
      goal: `<strong>Goal:</strong> compile the spectrogram twice — once with a 64-sample window
        and once with 256 — and let the two measurements say what changed.`,
      requirements: [
        '<code>shortSpec</code> uses <code>win: 64</code> and keeps <code>32</code> bins',
        '<code>longSpec</code> uses <code>win: 256</code> and keeps <code>128</code> bins',
        'Both keep the same hop (<code>32</code>) and the same <code>64</code> frames, so their time axes line up',
      ],
      hints: [
        {
          title: 'Hint 1 — how many bins?',
          body: `<p>A window of <em>w</em> real samples has <em>w</em>/2 bins below Nyquist, each
            <code>SR / w</code> hertz wide. A 64-sample window gives 32 bins 64 Hz apart; a
            256-sample window gives 128 bins 16 Hz apart.</p>`,
        },
        {
          title: 'Hint 2 — the two calls',
          body: `<pre><code>const shortSpec = makeSpectrogram(64, 32);
const longSpec = makeSpectrogram(256, 128);</code></pre>
<p><code>makeSpectrogram</code> is plain JavaScript — <code>win</code> and <code>bins</code> reach
the kernel through its settings object, never through a closure, which is why this compiles at
all.</p>`,
        },
        {
          title: 'Hint 3 — what you should see',
          body: `<p>The short window finds <strong>one</strong> peak where there are two tones, and
            pins the click to <strong>2</strong> frames. The long window resolves
            <strong>two</strong> peaks and smears the same click across <strong>6</strong>. Neither
            is the right answer; they are the same information spent differently.</p>`,
        },
      ],
      transfer: `Window length is the first knob in every audio pipeline, and the choice is always
        this trade: a speech recogniser takes 25 ms windows with a 10 ms hop because phonemes move
        fast, a music transcriber takes 100 ms because it needs to tell a semitone from its
        neighbour, and a vibration monitor looking for a bearing fault takes seconds. Recompiling
        one kernel source against different compile-time constants is equally universal —
        <code>template&lt;int WIN&gt;</code> in CUDA, <code>override</code> constants in WGSL,
        function constants in Metal. The specialisation is what lets the compiler unroll the loop.`,
      starterCode: `// One kernel source, two window lengths. Everything else held equal.
const gpu = new GPU({ mode });

const SR = 4096;    // samples per second
const HOP = 32;     // both spectrograms slide by this much
const FRAMES = 64;  // and both are this wide

// Already written: the STFT body from the last task, with the window length
// left as a constant so the same source can be compiled more than once.
function makeSpectrogram(win, bins) {
  return gpu.createKernel(function (signal) {
    const frame = this.thread.x;
    const bin = this.thread.y;
    const start = frame * this.constants.hop;
    let re = 0;
    let im = 0;
    for (let t = 0; t < this.constants.win; t++) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / (this.constants.win - 1));
      const s = signal[start + t] * w;
      const angle = (-2 * Math.PI * bin * t) / this.constants.win;
      re += s * Math.cos(angle);
      im += s * Math.sin(angle);
    }
    return Math.sqrt(re * re + im * im) / this.constants.win;
  }, {
    output: [FRAMES, bins],
    constants: { win: win, hop: HOP },
  });
}

// TODO: one SHORT window and one LONG one. Right now this is the same
// kernel twice, which measures the same thing twice.
const shortSpec = makeSpectrogram(256, 128);
const longSpec = makeSpectrogram(256, 128);

// Already written. FREQUENCY resolution: how many separate peaks stand up
// between 400 Hz and 800 Hz in a quiet frame? Two tones live in there.
function tonePeaks(spec, win) {
  const lo = Math.round((400 * win) / SR);
  const hi = Math.round((800 * win) / SR);
  const f = 20;
  let peak = 0;
  for (let b = 0; b < spec.length; b++) peak = Math.max(peak, spec[b][f]);
  let n = 0;
  for (let b = lo; b <= hi; b++) {
    const v = spec[b][f];
    if (v > 0.1 * peak && v >= spec[b - 1][f] && v > spec[b + 1][f]) n++;
  }
  return n;
}

// Already written. TIME resolution: how many frames does the one-sample
// click light up? It is broadband, so look above the tones.
function clickFrames(spec) {
  const top = Math.floor(spec.length / 2);
  const energy = [];
  let peak = 0;
  for (let f = 0; f < FRAMES; f++) {
    let e = 0;
    for (let b = top; b < spec.length; b++) e += spec[b][f];
    energy.push(e);
    peak = Math.max(peak, e);
  }
  let n = 0;
  for (const e of energy) if (e > 0.1 * peak) n++;
  return n;
}

const shortPic = await shortSpec(signal);
const longPic = await longSpec(signal);

console.log('short window:', tonePeaks(shortPic, 64), 'tone peaks,', clickFrames(shortPic), 'frames of click');
console.log('long  window:', tonePeaks(longPic, 256), 'tone peaks,', clickFrames(longPic), 'frames of click');
`,
      solutionCode: `// One kernel source, two window lengths. Everything else held equal.
const gpu = new GPU({ mode });

const SR = 4096;    // samples per second
const HOP = 32;     // both spectrograms slide by this much
const FRAMES = 64;  // and both are this wide

// Already written: the STFT body from the last task, with the window length
// left as a constant so the same source can be compiled more than once.
function makeSpectrogram(win, bins) {
  return gpu.createKernel(function (signal) {
    const frame = this.thread.x;
    const bin = this.thread.y;
    const start = frame * this.constants.hop;
    let re = 0;
    let im = 0;
    for (let t = 0; t < this.constants.win; t++) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / (this.constants.win - 1));
      const s = signal[start + t] * w;
      const angle = (-2 * Math.PI * bin * t) / this.constants.win;
      re += s * Math.cos(angle);
      im += s * Math.sin(angle);
    }
    return Math.sqrt(re * re + im * im) / this.constants.win;
  }, {
    output: [FRAMES, bins],
    constants: { win: win, hop: HOP },
  });
}

// 64 samples: 32 bins, 64 Hz apart. 256 samples: 128 bins, 16 Hz apart.
const shortSpec = makeSpectrogram(64, 32);
const longSpec = makeSpectrogram(256, 128);

// Already written. FREQUENCY resolution: how many separate peaks stand up
// between 400 Hz and 800 Hz in a quiet frame? Two tones live in there.
function tonePeaks(spec, win) {
  const lo = Math.round((400 * win) / SR);
  const hi = Math.round((800 * win) / SR);
  const f = 20;
  let peak = 0;
  for (let b = 0; b < spec.length; b++) peak = Math.max(peak, spec[b][f]);
  let n = 0;
  for (let b = lo; b <= hi; b++) {
    const v = spec[b][f];
    if (v > 0.1 * peak && v >= spec[b - 1][f] && v > spec[b + 1][f]) n++;
  }
  return n;
}

// Already written. TIME resolution: how many frames does the one-sample
// click light up? It is broadband, so look above the tones.
function clickFrames(spec) {
  const top = Math.floor(spec.length / 2);
  const energy = [];
  let peak = 0;
  for (let f = 0; f < FRAMES; f++) {
    let e = 0;
    for (let b = top; b < spec.length; b++) e += spec[b][f];
    energy.push(e);
    peak = Math.max(peak, e);
  }
  let n = 0;
  for (const e of energy) if (e > 0.1 * peak) n++;
  return n;
}

const shortPic = await shortSpec(signal);
const longPic = await longSpec(signal);

console.log('short window:', tonePeaks(shortPic, 64), 'tone peaks,', clickFrames(shortPic), 'frames of click');
console.log('long  window:', tonePeaks(longPic, 256), 'tone peaks,', clickFrames(longPic), 'frames of click');
`,
      inputs: () => ({ signal: tradeSignal() }),
      publicTests: [
        {
          name: 'two kernels — <code>64×32</code> and <code>64×128</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const short = kernelWithOutput(ctx, [T_FRAMES, SHORT_BINS]);
            const long = kernelWithOutput(ctx, [T_FRAMES, LONG_BINS]);
            const both = ctx.kernels.filter(k => {
              const out = k.kernel && k.kernel.output && Array.from(k.kernel.output);
              return out && out[0] === T_FRAMES && out[1] === LONG_BINS;
            });
            ctx.assert(
              short,
              both.length >= 2
                ? 'both kernels were compiled with the same 256-sample window — the whole exercise is to change win (and the bin count that implies) between them'
                : `no kernel with output [${T_FRAMES}, ${SHORT_BINS}] found — the short window keeps ${SHORT_BINS} bins`
            );
            ctx.assert(long, `no kernel with output [${T_FRAMES}, ${LONG_BINS}] found — the long window keeps ${LONG_BINS} bins`);
          },
        },
        {
          name: 'the 64-sample window: 32 bins, 64 Hz apart',
          run: async ctx => {
            const short = kernelWithOutput(ctx, [T_FRAMES, SHORT_BINS]);
            ctx.assert(short, 'no short-window kernel found');
            const signal = tradeSignal();
            const out = await short(signal);
            const ref = once('stft:short', () =>
              refStft(signal, { win: SHORT_WIN, hop: T_HOP, frames: T_FRAMES, bins: SHORT_BINS })
            );
            assertSpectrogram(ctx, out, ref, signal, {
              win: SHORT_WIN,
              hop: T_HOP,
              frames: T_FRAMES,
              bins: SHORT_BINS,
              eps: SPEC_EPS,
            });
          },
        },
        {
          name: 'the 256-sample window: 128 bins, 16 Hz apart',
          run: async ctx => {
            const long = kernelWithOutput(ctx, [T_FRAMES, LONG_BINS]);
            ctx.assert(long, 'no long-window kernel found');
            const signal = tradeSignal();
            const out = await long(signal);
            const ref = once('stft:long', () =>
              refStft(signal, { win: LONG_WIN, hop: T_HOP, frames: T_FRAMES, bins: LONG_BINS })
            );
            assertSpectrogram(ctx, out, ref, signal, {
              win: LONG_WIN,
              hop: T_HOP,
              frames: T_FRAMES,
              bins: LONG_BINS,
              eps: SPEC_EPS,
            });
          },
        },
        {
          name: 'the trade, measured: <code>1</code> peak / <code>2</code> frames against <code>2</code> peaks / <code>6</code> frames',
          run: async ctx => {
            const short = kernelWithOutput(ctx, [T_FRAMES, SHORT_BINS]);
            const long = kernelWithOutput(ctx, [T_FRAMES, LONG_BINS]);
            ctx.assert(short && long, 'expected a short-window kernel and a long-window kernel');
            const signal = tradeSignal();
            const s = await short(signal);
            const l = await long(signal);
            const peaks = (spec, win) => {
              const lo = Math.round((400 * win) / SR);
              const hi = Math.round((800 * win) / SR);
              let peak = 0;
              for (let b = 0; b < spec.length; b++) peak = Math.max(peak, spec[b][20]);
              let n = 0;
              for (let b = lo; b <= hi; b++) {
                const v = spec[b][20];
                if (v > 0.1 * peak && v >= spec[b - 1][20] && v > spec[b + 1][20]) n++;
              }
              return n;
            };
            const lit = spec => {
              const top = Math.floor(spec.length / 2);
              const energy = [];
              let peak = 0;
              for (let f = 0; f < T_FRAMES; f++) {
                let e = 0;
                for (let b = top; b < spec.length; b++) e += spec[b][f];
                energy.push(e);
                peak = Math.max(peak, e);
              }
              return energy.filter(e => e > 0.1 * peak).length;
            };
            ctx.assert(
              peaks(s, SHORT_WIN) === 1,
              `the 64-sample window should merge the two tones into one peak, found ${peaks(s, SHORT_WIN)} — ` +
                'its bins are 64 Hz apart and the tones are only 128 Hz apart, which a Hann main lobe covers'
            );
            ctx.assert(
              peaks(l, LONG_WIN) === 2,
              `the 256-sample window should resolve two peaks, found ${peaks(l, LONG_WIN)} — ` +
                'its bins are 16 Hz apart, so the two tones sit 8 bins away from each other'
            );
            ctx.assert(
              lit(s) === 2,
              `the click should light 2 frames of the short spectrogram, got ${lit(s)}`
            );
            ctx.assert(
              lit(l) === 6,
              `the click should smear across 6 frames of the long spectrogram, got ${lit(l)} — ` +
                'a 256-sample window is still looking at the click eight hops after it happened'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The same two kernels on a signal with only the click in it: the
            // time axis has to behave with no tones to hide behind.
            const short = kernelWithOutput(ctx, [T_FRAMES, SHORT_BINS]);
            const long = kernelWithOutput(ctx, [T_FRAMES, LONG_BINS]);
            ctx.assert(short && long, 'expected a short-window kernel and a long-window kernel');
            const signal = new Array(TRADE_N).fill(0);
            signal[CLICK_AT] = 6;
            const refShort = refStft(signal, {
              win: SHORT_WIN, hop: T_HOP, frames: T_FRAMES, bins: SHORT_BINS,
            });
            const refLong = refStft(signal, {
              win: LONG_WIN, hop: T_HOP, frames: T_FRAMES, bins: LONG_BINS,
            });
            assertSpectrogram(ctx, await short(signal), refShort, signal, {
              win: SHORT_WIN, hop: T_HOP, frames: T_FRAMES, bins: SHORT_BINS, eps: SPEC_EPS,
            });
            assertSpectrogram(ctx, await long(signal), refLong, signal, {
              win: LONG_WIN, hop: T_HOP, frames: T_FRAMES, bins: LONG_BINS, eps: SPEC_EPS,
            });
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      // Pinned to WebGL, like the other tasks whose paint sits downstream of a
      // pipelined chain. This one awaits its STFT before creating the paint
      // kernel, so under 'auto' the adapter decision has already settled and
      // gpu.js builds the graphical kernel as a WebGPU kernel directly —
      // dropping the GPU-level canvas, which in a worker is the only one there
      // is, and failing with "graphical mode requires a canvas".
      //
      // Injecting a per-kernel canvas in the engine was tried and reverted: it
      // fixed this task and broke four others, whose readbacks then returned
      // zeros. Pinning is the mechanism the course already has for exactly
      // this shape, and it costs nothing here — explicit webgpu still works.
      backend: 'webgl',
      slug: 'paint-it',
      title: 'Decibels, Then Colour',
      intro: `<p>The numbers in a spectrogram span orders of magnitude. In this one the loudest cell
        is about <code>0.126</code> and the median cell about <code>0.00014</code> — a factor of
        nearly 900. Hand that straight to a colour ramp and the median cell gets a tenth of one
        percent of the range: you get a black rectangle with a few bright streaks, and the obvious
        conclusion — "my kernel is broken" — is wrong. Your kernel is fine. Your scaling is
        wrong.</p>
        <p>The fix is what ears already do: work in <strong>decibels</strong>.
        <code>db = 20 · log₁₀(mag / peak)</code> puts the loudest cell at 0 and every halving 6 dB
        below it, so a 60 dB display range covers a factor of 1,000 in one legible ramp and
        everything quieter clamps to black. That single transform is the difference between a
        picture and a black rectangle.</p>
        <p>Then colour. Loudness rides on <strong>value</strong> — the part that has to be monotone,
        or the picture lies about which cell is louder — and the hue sweep is there for legibility,
        which is the case Colour Spaces argues at length. The ramp below is written for you: black →
        indigo → magenta → orange → cream. What is yours is the two lines above it, and the canvas
        is 256×256, so each spectrogram cell paints a 4×2 block.</p>
        <p>One thing has moved since task 2: the window length is on a <strong>slider</strong>, and
        the kernel now takes two lengths instead of one. <code>WIN</code> is the length of the
        <em>transform</em> and stays 256, which is what pins bin <em>b</em> to
        <em>b·SR/256</em> and stops the frequency axis sliding around under you. <code>TAPER</code>
        is the length of the <em>window</em> — how much of that 256 the Hann bell covers, the rest
        being zeros, which is exactly what <code>librosa</code> means by <code>win_length</code>
        against <code>n_fft</code>. Solve the task at the default, where <code>TAPER === WIN</code>
        and the kernel is task 2's to the last bit. Then drag it left and watch task 3 happen: the
        harmonics swell into bands as the bins coarsen, and the plucks tighten into hard vertical
        edges as time sharpens up.</p>`,
      goal: `<strong>Goal:</strong> finish the paint kernel — magnitude to decibels, decibels to a
        0…1 value — and render the result.`,
      requirements: [
        'Convert against the peak: <code>db = 20 * Math.log10(mag / peak + 1e-9)</code>',
        'Map the top <code>this.constants.range</code> decibels onto 0…1 and clamp: <code>Math.max(0, Math.min(1, 1 + db / this.constants.range))</code>',
        'Leave the 4×2 zoom and the ramp alone — the canvas stays <code>256×256</code>',
        'Keep the <code>render(paint.canvas)</code> call',
      ],
      hints: [
        {
          title: 'Hint 1 — log base 10',
          body: `<p><code>Math.log10</code> is on gpu.js's whitelist, so write it directly — no need
            for <code>Math.log(x) / Math.LN10</code>. <code>Math.log</code> is the <em>natural</em>
            logarithm and gives answers 2.3 times too far down the scale.</p>`,
        },
        {
          title: 'Hint 2 — the two lines',
          body: `<pre><code>const db = 20 * Math.log10(mag / peak + 1e-9);
const v = Math.max(0, Math.min(1, 1 + db / this.constants.range));</code></pre>`,
        },
        {
          // Plain text: the app renders hint titles as JSX text, so any markup
          // here is escaped and the learner sees the tags themselves.
          title: 'Hint 3 — why the 1e-9',
          body: `<p><code>log₁₀(0)</code> is <code>-Infinity</code>, and <code>-Infinity / 60</code>
            stays infinite: one silent cell would paint <code>NaN</code>, which lands on screen as
            whatever the driver felt like. The tiny floor costs nothing (it is 180 dB down) and
            makes the silent case land on plain black instead.</p>`,
        },
      ],
      transfer: `Amplitude-to-decibels is the universal display transform for audio —
        <code>librosa.amplitude_to_db</code>, every DAW meter, every analyser you have ever looked
        at — and the wider lesson generalises well past sound: a GPU can compute a perfectly correct
        answer and a bad transfer function will still show you nothing. HDR tone mapping, log-scale
        depth buffers and gamma correction are the same move made for the same reason. On any
        platform this is a fragment shader reading a storage texture; the arithmetic does not
        change.`,
      starterCode: `// The spectrogram from task 2, already built. What is missing is the part
// that makes it legible. 64 frames × 128 bins onto a 256×256 canvas.
const gpu = new GPU({ mode });

const WIN = 256;   // the TRANSFORM length. Bin b is b * SR / WIN, always.
const HOP = 64;
const BINS = 128;
const FRAMES = Math.floor((signal.length - WIN) / HOP) + 1;

// A dial, not a constant: slider() re-runs the whole program when you drag it.
// TAPER is the WINDOW length — how many of the frame's 256 samples the Hann
// bell actually covers; the rest are zeros. Left: coarse bins, sharp onsets.
// Right: fine bins, smeared onsets. Task 3 measured that trade. This is the
// handle on it, and it starts where the task was written: the full 256.
const TAPER = slider('window', { min: 32, max: WIN, value: WIN, step: 16 });

${STFT_TAPERED}

const spec = await spectrogram(signal);

// How wide is the range you are about to map?
let peak = 0;
const all = [];
for (let b = 0; b < BINS; b++) {
  for (let f = 0; f < FRAMES; f++) {
    all.push(spec[b][f]);
    if (spec[b][f] > peak) peak = spec[b][f];
  }
}
all.sort((a, b) => a - b);
console.log('loudest cell:', peak);
console.log('median cell: ', all[all.length >> 1]);
console.log('ratio:       ', peak / all[all.length >> 1]);

const paint = gpu.createKernel(function (spec, peak) {
  const frame = Math.floor(this.thread.x / 4);
  const bin = Math.floor(this.thread.y / 2);
  const mag = spec[bin][frame];

  // TODO: magnitude → decibels → 0…1.
  //   db = 20 * Math.log10(mag / peak + 1e-9)
  //   v  = 1 + db / this.constants.range, clamped to 0…1
  const v = mag; // ← linear. Run it and look at what that gets you.

${RAMP_SOURCE}
}, {
  output: [256, 256],
  graphical: true,
  constants: { range: 60 },
});

await paint(spec, peak);
render(paint.canvas);
`,
      solutionCode: `// The spectrogram from task 2, already built. What is missing is the part
// that makes it legible. 64 frames × 128 bins onto a 256×256 canvas.
const gpu = new GPU({ mode });

const WIN = 256;   // the TRANSFORM length. Bin b is b * SR / WIN, always.
const HOP = 64;
const BINS = 128;
const FRAMES = Math.floor((signal.length - WIN) / HOP) + 1;

// A dial, not a constant: slider() re-runs the whole program when you drag it.
// TAPER is the WINDOW length — how many of the frame's 256 samples the Hann
// bell actually covers; the rest are zeros. Left: coarse bins, sharp onsets.
// Right: fine bins, smeared onsets. Task 3 measured that trade. This is the
// handle on it, and it starts where the task was written: the full 256.
const TAPER = slider('window', { min: 32, max: WIN, value: WIN, step: 16 });

${STFT_TAPERED}

const spec = await spectrogram(signal);

// How wide is the range you are about to map?
let peak = 0;
const all = [];
for (let b = 0; b < BINS; b++) {
  for (let f = 0; f < FRAMES; f++) {
    all.push(spec[b][f]);
    if (spec[b][f] > peak) peak = spec[b][f];
  }
}
all.sort((a, b) => a - b);
console.log('loudest cell:', peak);
console.log('median cell: ', all[all.length >> 1]);
console.log('ratio:       ', peak / all[all.length >> 1]);

const paint = gpu.createKernel(function (spec, peak) {
  const frame = Math.floor(this.thread.x / 4);
  const bin = Math.floor(this.thread.y / 2);
  const mag = spec[bin][frame];

  const db = 20 * Math.log10(mag / peak + 1e-9);
  const v = Math.max(0, Math.min(1, 1 + db / this.constants.range));

${RAMP_SOURCE}
}, {
  output: [256, 256],
  graphical: true,
  constants: { range: 60 },
});

await paint(spec, peak);
render(paint.canvas);
`,
      inputs: () => ({ signal: noteSignal() }),
      // The lesson paints 64 frames × 128 bins as 4×2 blocks on a 256×256
      // canvas — which the catalogue card then shows at ~300 CSS px, so every
      // cell is a lump. The card renders the same second of audio at 512×512
      // with one pixel per cell: the same notes, hopped eight times as often
      // (CARD_HOP) and read at four times the bin spacing, both set up by the
      // capture script's CARD_SCALE entry. All this has to supply is the 56
      // extra samples the finer hop needs to reach the right edge.
      cardInputs: () => ({ signal: noteSignal(CARD_N) }),
      publicTests: [
        {
          name: 'a spectrogram kernel and a <code>256×256</code> graphical kernel',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            ctx.assert(
              kernelWithOutput(ctx, [FRAMES, BINS]),
              `no spectrogram kernel with output [${FRAMES}, ${BINS}] found`
            );
            const paint = graphicalKernel(ctx);
            ctx.assert(paint, 'no graphical kernel found — the paint kernel needs graphical: true');
            ctx.assert(ctx.canvas, 'no canvas — did you keep the render(paint.canvas) call?');
            ctx.assert(
              ctx.canvas.width === 256 && ctx.canvas.height === 256,
              `expected a 256×256 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
          },
        },
        {
          name: 'pixels follow <code>magnitude → decibels → colour</code>',
          run: async ctx => {
            const stft = kernelWithOutput(ctx, [FRAMES, BINS]);
            const paint = graphicalKernel(ctx);
            ctx.assert(stft && paint, 'expected a spectrogram kernel and a graphical paint kernel');
            const signal = noteSignal();
            const spec = await stft(signal);
            let peak = 0;
            for (let b = 0; b < BINS; b++) {
              for (let f = 0; f < FRAMES; f++) if (spec[b][f] > peak) peak = spec[b][f];
            }
            await paint(spec, peak);
            const pixels = await ctx.getPixels();

            // Sample the cells nearest a spread of loudnesses, so the check
            // covers the whole ramp rather than the bright end of it. Brightest
            // first: near the floor every wrong scaling paints the same black,
            // so a failure reported there would say nothing.
            const samples = [];
            for (const target of [0.95, 0.8, 0.65, 0.5, 0.35, 0.2, 0.05]) {
              let best = null;
              let bestDistance = Infinity;
              for (let b = 0; b < BINS; b++) {
                for (let f = 0; f < FRAMES; f++) {
                  const d = Math.abs(dbValue(spec[b][f], peak) - target);
                  if (d < bestDistance) {
                    bestDistance = d;
                    best = [b, f];
                  }
                }
              }
              samples.push(best);
            }

            // getPixels() hands rows back top-down and the top row is the
            // highest this.thread.y on both backends here — but resolve it
            // from the data anyway, so a backend that ever disagreed reports a
            // colour mismatch rather than a phantom one.
            const read = (tx, ty, flipped) => {
              const row = flipped ? ty : 255 - ty;
              const i = (row * 256 + tx) * 4;
              return [pixels[i], pixels[i + 1], pixels[i + 2]];
            };
            const wanted = ([b, f]) => rampColour(dbValue(spec[b][f], peak)).map(c => c * 255);
            const near = (got, want, eps) => got.every((c, i) => Math.abs(c - want[i]) <= eps);
            const score = flipped =>
              samples.filter(([b, f]) => near(read(4 * f + 1, 2 * b, flipped), wanted([b, f]), 4)).length;
            const flipped = score(true) > score(false);

            // A wrong scaling has to explain EVERY sampled pixel before it may
            // speak. One pixel is not evidence: down near the floor all four
            // mistakes below paint the same black, and a confident wrong
            // diagnosis is worse than a plain numeric one.
            const clamp01 = v => Math.max(0, Math.min(1, v));
            const mistakes = [
              [mag => clamp01(mag), LINEAR_SCALING],
              [mag => clamp01(mag / peak), LINEAR_SCALING],
              [mag => clamp01(1 + (20 * Math.log(mag / peak + 1e-9)) / DB_RANGE), NATURAL_LOG],
              [mag => clamp01(1 + (20 * Math.log10(mag + 1e-9)) / DB_RANGE), NO_PEAK_DIVISION],
            ];
            const hits = mistakes
              .filter(([value]) => {
                let differs = false;
                for (const [b, f] of samples) {
                  const colour = rampColour(value(spec[b][f])).map(c => c * 255);
                  if (!near(read(4 * f + 1, 2 * b, flipped), colour, 4)) return false;
                  if (!near(wanted([b, f]), colour, 4)) differs = true;
                }
                return differs;
              })
              .map(p => p[1]);
            const hint = hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;

            for (const [b, f] of samples) {
              const got = read(4 * f + 1, 2 * b, flipped);
              const want = wanted([b, f]);
              const mag = spec[b][f];
              for (let c = 0; c < 3; c++) {
                ctx.assertClose(
                  got[c],
                  want[c],
                  4,
                  hint || `channel ${'rgb'[c]} of the cell at bin ${b}, frame ${f} (magnitude ${mag.toExponential(2)})`
                );
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Every cell, not seven of them: the ramp has to be right all the
            // way down, including the cells that clamp to black.
            const stft = kernelWithOutput(ctx, [FRAMES, BINS]);
            const paint = graphicalKernel(ctx);
            ctx.assert(stft && paint, 'expected a spectrogram kernel and a graphical paint kernel');
            const spec = await stft(noteSignal());
            let peak = 0;
            for (let b = 0; b < BINS; b++) {
              for (let f = 0; f < FRAMES; f++) if (spec[b][f] > peak) peak = spec[b][f];
            }
            await paint(spec, peak);
            const pixels = await ctx.getPixels();
            const read = (tx, ty, flipped) => {
              const row = flipped ? ty : 255 - ty;
              const i = (row * 256 + tx) * 4;
              return [pixels[i], pixels[i + 1], pixels[i + 2]];
            };
            const near = (got, want, eps) => got.every((c, i) => Math.abs(c - want[i]) <= eps);
            const wanted = (b, f) => rampColour(dbValue(spec[b][f], peak)).map(c => c * 255);
            const score = flipped => {
              let n = 0;
              for (let b = 0; b < BINS; b += 16) {
                for (let f = 0; f < FRAMES; f += 8) {
                  if (near(read(4 * f + 1, 2 * b, flipped), wanted(b, f), 4)) n++;
                }
              }
              return n;
            };
            const flipped = score(true) > score(false);
            for (let b = 0; b < BINS; b++) {
              for (let f = 0; f < FRAMES; f++) {
                const got = read(4 * f + 1, 2 * b, flipped);
                const want = wanted(b, f);
                for (let c = 0; c < 3; c++) {
                  ctx.assertClose(got[c], want[c], 4, `channel ${'rgb'[c]} at bin ${b}, frame ${f}`);
                }
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A signal this task never showed, and one whose peak is somewhere
            // else entirely: a single steady tone. Nothing may be hardcoded,
            // and the whole picture below the tone has to clamp to the floor
            // rather than to whatever the previous peak happened to be.
            const stft = kernelWithOutput(ctx, [FRAMES, BINS]);
            const paint = graphicalKernel(ctx);
            ctx.assert(stft && paint, 'expected a spectrogram kernel and a graphical paint kernel');
            const tone = new Array(SIGNAL_N);
            for (let n = 0; n < SIGNAL_N; n++) tone[n] = 0.4 * Math.sin((2 * Math.PI * 640 * n) / SR);
            const spec = await stft(tone);
            let peak = 0;
            for (let b = 0; b < BINS; b++) {
              for (let f = 0; f < FRAMES; f++) if (spec[b][f] > peak) peak = spec[b][f];
            }
            await paint(spec, peak);
            const pixels = await ctx.getPixels();
            const read = (tx, ty, flipped) => {
              const row = flipped ? ty : 255 - ty;
              const i = (row * 256 + tx) * 4;
              return [pixels[i], pixels[i + 1], pixels[i + 2]];
            };
            const wanted = (b, f) => rampColour(dbValue(spec[b][f], peak)).map(c => c * 255);
            const near = (got, want, eps) => got.every((c, i) => Math.abs(c - want[i]) <= eps);
            const score = flipped => {
              let n = 0;
              for (let b = 0; b < BINS; b += 8) {
                for (let f = 0; f < FRAMES; f += 8) {
                  if (near(read(4 * f + 1, 2 * b, flipped), wanted(b, f), 4)) n++;
                }
              }
              return n;
            };
            const flipped = score(true) > score(false);
            for (let b = 0; b < BINS; b++) {
              for (let f = 0; f < FRAMES; f++) {
                const got = read(4 * f + 1, 2 * b, flipped);
                const want = wanted(b, f);
                for (let c = 0; c < 3; c++) {
                  ctx.assertClose(got[c], want[c], 4, `steady tone: channel ${'rgb'[c]} at bin ${b}, frame ${f}`);
                }
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'read-the-ridge',
      title: 'Payoff: Read the Sweep Off the Picture',
      intro: `<p>A spectrogram is not only for looking at. This signal is a <strong>chirp</strong> —
        a tone sliding steadily upward — and everything needed to characterise it is already in the
        picture. For each frame, find the bin carrying the most energy. The sequence of winners is
        the chirp's <strong>ridge</strong>, and the slope of that ridge is the sweep rate, in hertz
        per second, recovered from nothing but the image.</p>
        <p>The GPU shape: one thread per frame, each scanning its own column of 128 bins for the
        largest value and returning <strong>the index</strong>, not the value. An argmax along an
        axis is a reduction — Reductions climbs the general log₂ ladder for the case where the axis
        is huge — but with 128 values per column, one thread walking them is the right call. What
        matters more is what it does not need: no atomics, no shared memory, no thread writing to a
        cell it does not own. Every thread pulls its own column and writes its own answer.</p>
        <p>Where would this come from on a real page? A microphone, and the wiring is short:</p>
<pre><code>const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const audio = new AudioContext({ sampleRate: 48000 });
await audio.audioWorklet.addModule('tap.js');   // posts Float32Array frames
const tap = new AudioWorkletNode(audio, 'tap');
audio.createMediaStreamSource(stream).connect(tap);

const gpu = new GPU();
const spectrogram = gpu.createKernel(/* the kernel you already wrote */);

tap.port.onmessage = async event =&gt; {
  const spec = await spectrogram(event.data);   // event.data is the ring buffer
  await paint(spec, peakOf(spec));
};</code></pre>
        <p><strong>That code cannot run here, and this course is not going to pretend otherwise.</strong>
        Your code executes inside a Web Worker — which is what lets a runaway kernel be killed
        instead of freezing the page — and a Worker has no <code>AudioContext</code>, no
        <code>OfflineAudioContext</code> and no <code>navigator.mediaDevices</code>. There is no
        microphone to reach and nothing to play back through. The buffers this module builds are the
        stand-in, and they buy something a microphone never could: every assertion below is exact.
        Every kernel you have written works unchanged the day you paste it onto a page with the
        wiring above.</p>`,
      goal: `<strong>Goal:</strong> find the ridge with a kernel, then turn its slope into a sweep
        rate in hertz per second and log it.`,
      requirements: [
        '<code>output: [FRAMES]</code> — one thread per frame, scanning all <code>this.constants.bins</code> bins',
        'Return the <em>bin index</em> of the largest magnitude in that column, not the magnitude itself',
        'Convert the slope to hertz per second and <code>console.log</code> it — bins are <code>SR / WIN</code> Hz apart, frames <code>HOP / SR</code> seconds apart',
      ],
      hints: [
        {
          title: 'Hint 1 — tracking an argmax',
          body: `<p>Two locals: the best value seen so far, and the index it came from. Update both
            together, and return the <em>index</em>. Returning <code>best</code> instead of
            <code>bestBin</code> is the classic slip, and it looks plausible right up until the
            numbers turn out to be 0.2 rather than 40.</p>`,
        },
        {
          title: 'Hint 2 — the kernel',
          body: `<pre><code>let best = 0;
let bestBin = 0;
for (let b = 0; b &lt; this.constants.bins; b++) {
  const v = spec[b][this.thread.x];
  if (v &gt; best) {
    best = v;
    bestBin = b;
  }
}
return bestBin;</code></pre>
<p>Note <code>spec[b][this.thread.x]</code>: the column belongs to this thread, and
<code>b</code> walks down it.</p>`,
        },
        {
          title: 'Hint 3 — the arithmetic',
          body: `<pre><code>const binsPerFrame = (line[FRAMES - 1] - line[0]) / (FRAMES - 1);
const hzPerSecond = (binsPerFrame * (SR / WIN)) / (HOP / SR);</code></pre>
<p>Bins are <code>4096 / 256</code> = 16 Hz apart and frames are <code>64 / 4096</code> = 15.625 ms
apart, so one bin per frame is 1,024 Hz per second — which is exactly what went in.</p>`,
        },
      ],
      transfer: `Ridge tracking is how a radar measures range rate, how a guitar tuner follows a bent
        note, how a birdsong classifier segments syllables and how a bearing monitor spots a
        harmonic drifting with load. The primitive is <code>argmax</code> along an axis:
        <code>torch.argmax(dim=…)</code>, <code>cub::DeviceSegmentedReduce</code> with
        <code>ArgMax</code>, <code>simd_max</code> plus a ballot in Metal. The awkward part on every
        platform is the same one you just stepped around — a maximum is easy, but carrying the
        <em>index</em> of the maximum through a parallel reduction means reducing pairs, not
        numbers.`,
      starterCode: `// The spectrogram is already built. What is new is reading it.
const gpu = new GPU({ mode });

const SR = 4096;   // samples per second
const WIN = 256;
const HOP = 64;
const BINS = 128;
const FRAMES = Math.floor((signal.length - WIN) / HOP) + 1;

${STFT_BODY}

const ridge = gpu.createKernel(function (spec) {
  // TODO: scan this frame's column of this.constants.bins bins and return
  // the INDEX of the loudest one.
  return 0;
}, {
  output: [FRAMES],
  constants: { bins: BINS },
});

const spec = await spectrogram(signal);
const line = await ridge(spec);
console.log('ridge bins:', Array.from(line).join(' '));

// TODO: turn the slope into hertz per second and log it.
// const binsPerFrame = (line[FRAMES - 1] - line[0]) / (FRAMES - 1);
// const hzPerSecond = (binsPerFrame * (SR / WIN)) / (HOP / SR);
// console.log('sweep rate:', hzPerSecond, 'Hz per second');
`,
      solutionCode: `// The spectrogram is already built. What is new is reading it.
const gpu = new GPU({ mode });

const SR = 4096;   // samples per second
const WIN = 256;
const HOP = 64;
const BINS = 128;
const FRAMES = Math.floor((signal.length - WIN) / HOP) + 1;

${STFT_BODY}

const ridge = gpu.createKernel(function (spec) {
  let best = 0;
  let bestBin = 0;
  for (let b = 0; b < this.constants.bins; b++) {
    const v = spec[b][this.thread.x];
    if (v > best) {
      best = v;
      bestBin = b;
    }
  }
  return bestBin;
}, {
  output: [FRAMES],
  constants: { bins: BINS },
});

const spec = await spectrogram(signal);
const line = await ridge(spec);
console.log('ridge bins:', Array.from(line).join(' '));

const binsPerFrame = (line[FRAMES - 1] - line[0]) / (FRAMES - 1);
const hzPerSecond = (binsPerFrame * (SR / WIN)) / (HOP / SR);
console.log('sweep rate:', hzPerSecond, 'Hz per second');
`,
      inputs: () => ({ signal: chirpSignal() }),
      publicTests: [
        {
          name: 'the ridge is 64 bin <em>indices</em>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const ridge = kernelWithOutput(ctx, [FRAMES]);
            ctx.assert(ridge, `no kernel with output [${FRAMES}] found — one thread per frame`);
            const stft = kernelWithOutput(ctx, [FRAMES, BINS]);
            ctx.assert(stft, 'the spectrogram kernel went missing');
            const line = Array.from(await ridge(await stft(chirpSignal())));
            ctx.assert(line.length === FRAMES, `expected ${FRAMES} values, got ${line.length}`);
            const fractional = line.some(v => v > 0 && v !== Math.round(v));
            const allSmall = line.every(v => v < 1);
            ctx.assert(
              !(allSmall && fractional),
              'those are magnitudes, not bin indices — the kernel returned the largest VALUE it ' +
                'found instead of the bin it sits in. Keep two locals and return the index'
            );
            ctx.assert(
              new Set(line).size > 1,
              'every frame reports the same bin — the scan is not using this.thread.x, so all 64 ' +
                'threads are looking at the same column'
            );
            for (const v of line) {
              ctx.assert(
                Number.isInteger(v) && v >= 0 && v < BINS,
                `${v} is not a bin index — every answer must be a whole number in 0…${BINS - 1}`
              );
            }
          },
        },
        {
          name: 'the ridge climbs one bin per frame: <code>18 + f</code>',
          run: async ctx => {
            const ridge = kernelWithOutput(ctx, [FRAMES]);
            const stft = kernelWithOutput(ctx, [FRAMES, BINS]);
            ctx.assert(ridge && stft, 'expected a spectrogram kernel and a ridge kernel');
            const spec = await stft(chirpSignal());
            const line = Array.from(await ridge(spec));
            // Scanning the wrong axis reads along a row of the picture instead
            // of down a column, and on this signal that produces a distinctive
            // sequence of its own — it has to predict every frame before it
            // may speak.
            const byRow = [];
            for (let f = 0; f < FRAMES; f++) {
              let best = 0;
              for (let k = 1; k < FRAMES; k++) if (spec[f][k] > spec[f][best]) best = k;
              byRow.push(best);
            }
            const wrongAxis = line.every((v, f) => v === byRow[f]) && byRow.some((v, f) => v !== 18 + f);
            for (let f = 0; f < FRAMES; f++) {
              ctx.assert(
                line[f] === 18 + f,
                wrongAxis
                  ? 'the scan ran along the wrong axis — it walked a row (one bin across all ' +
                    'frames) instead of a column. The column belongs to this thread: ' +
                    'spec[b][this.thread.x], with b running over the bins'
                  : `frame ${f} should peak at bin ${18 + f}, got ${line[f]}`
              );
            }
          },
        },
        {
          name: 'the sweep rate <code>1024</code> Hz per second is logged',
          run: async ctx => {
            const numbers = loggedNumbers(ctx.logs);
            ctx.assert(
              numbers.some(v => Math.abs(v - 1024) <= 1),
              'log the sweep rate — the ridge climbs one bin per frame, bins are 16 Hz apart and ' +
                'frames are 15.625 ms apart, so the answer is 1024 Hz per second'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A chirp sweeping the other way, through the same two kernels.
            const ridge = kernelWithOutput(ctx, [FRAMES]);
            const stft = kernelWithOutput(ctx, [FRAMES, BINS]);
            ctx.assert(ridge && stft, 'expected a spectrogram kernel and a ridge kernel');
            const line = Array.from(await ridge(await stft(chirpSignal(1536, -1024))));
            for (let f = 0; f < FRAMES; f++) {
              ctx.assert(
                line[f] === 94 - f,
                `a chirp sweeping down: frame ${f} should peak at bin ${94 - f}, got ${line[f]}`
              );
            }
            const binsPerFrame = (line[FRAMES - 1] - line[0]) / (FRAMES - 1);
            const hz = (binsPerFrame * (SR / WIN)) / (HOP / SR);
            ctx.assertClose(hz, -1024, 1, 'the recovered sweep rate of a descending chirp');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A steady tone: the ridge must be flat, and it must sit on the
            // tone rather than on bin 0.
            const ridge = kernelWithOutput(ctx, [FRAMES]);
            const stft = kernelWithOutput(ctx, [FRAMES, BINS]);
            ctx.assert(ridge && stft, 'expected a spectrogram kernel and a ridge kernel');
            const tone = new Array(SIGNAL_N);
            for (let n = 0; n < SIGNAL_N; n++) tone[n] = Math.sin((2 * Math.PI * 640 * n) / SR);
            const line = Array.from(await ridge(await stft(tone)));
            for (let f = 0; f < FRAMES; f++) {
              ctx.assert(line[f] === 40, `a steady 640 Hz tone is bin 40 in every frame; frame ${f} gave ${line[f]}`);
            }
          },
        },
      ],
    },
  ],
};
