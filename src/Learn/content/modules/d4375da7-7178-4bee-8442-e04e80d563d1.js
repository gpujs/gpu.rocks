// Module: The FFT Butterfly — uuid d4375da7-7178-4bee-8442-e04e80d563d1
// (short id d4375da7). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module
// postdates the uuid scheme.
//
// Signal Processing — The FFT Butterfly.
//
// Six tasks: the even/odd split verified numerically → the butterfly as a
// gather → bit reversal → the log₂n stage loop driven from JS → the naive
// DFT for proof and for the showdown → the inverse transform by conjugation.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// every value arrives as an argument / this.thread.* / this.constants.*,
// statically bounded loops, Math.* per gpu.js's whitelist. Every task passes
// in cpu mode as well as gpu mode. Tasks 1–4 and 6 stay at n ≤ 256; task 5 is
// the payoff task and is sized differently — see below.
//
// ---------------------------------------------------------------------------
// WHY TASK 5 IS BIG — n = 8,192, and why not 512, 4,096 or 16,384
//
// Task 5 races the naive DFT against the FFT, so it is only worth running at a
// size where the FFT actually wins. It did not, at 512. Measured in a real
// browser on an Apple M1 Max (ANGLE/Metal), gpu mode, one call each after two
// warm-ups, median of 40:
//
//   n       naive DFT     FFT (this task)   FFT as task 4 wrote it
//   512       0.7 ms        —                 5.1 ms   ← the FFT LOSES 7×
//   4,096     3.0 ms      1.4 ms              7.6 ms   ← 2.0×, and 1.25× at worst
//   8,192     3.5 ms      1.5 ms              8.2 ms   ← 2.3–2.7×, never below 1.9×
//   16,384    9.8 ms      1.8 ms              9.8 ms   ← 5.4×, and unshippable
//
// Two separate things had to change to get there.
//
//  1. THE FFT STOPS COPYING ITSELF BACK. Task 4 chains its passes through
//     JavaScript arrays, which is fourteen readbacks at n = 8,192 — measured at
//     ~0.5 ms each, against ~0.07 ms for the launch they wrap. That is the
//     whole reason a small FFT loses to a brute-force kernel, so task 5 keeps
//     the intermediate spectra on the card (pipeline: true) and reads back
//     once: 8.2 ms → 1.5 ms. Two butterfly kernels take turns, because a
//     pipelined kernel writes into its own texture and cannot also be the one
//     reading it. gpu.js's CPU backend accepts `pipeline` and simply hands back
//     the array, which is why fft() ends
//     `buffer.toArray ? await buffer.toArray() : buffer`.
//
//     Every stage of that ladder is AWAITED IN ORDER. Under gpu.js's async mode
//     a kernel call is a promise on every backend, and each pass reads the
//     texture the previous one wrote — gathering the thirteen launches into a
//     Promise.all would feed a pass a half-written input and silently corrupt
//     the transform. `await` on a non-promise is a no-op, so the one awaited
//     shape is also what the synchronous backends run.
//
//  2. n GOES UP UNTIL THE ARITHMETIC OUTWEIGHS THE LAUNCHES. At 4,096 the
//     margin is 2.0× typical but 1.25× worst-case — inside the noise. 8,192 is
//     the smallest power of two that wins by a stable multiple.
//
// n = 16,384 would win by 5.4× and is NOT used: a [w, 2] kernel needs a
// w-wide texture, and MAX_TEXTURE_SIZE is 8,192 on Chrome's software WebGL
// (SwiftShader) — measured, the kernel refuses to build — so 16,384 makes the
// task fail outright on every machine without a GPU, including CI runners.
// 8,192 runs everywhere. Learner-visible numbers off the real console, through
// the worker sandbox: 1.6 ms vs 3.5 ms on the M1 (2.0–2.5× typical, 5.5× on a
// cold device), and 10.4 ms vs 1,408.7 ms — 135× — on SwiftShader, where all
// five tests still pass.
//
// THE CPU BACKEND CANNOT PAY FOR THE FULL SPECTRUM. One thread does ~3.1e7
// of the naive transform's multiply-accumulates per second, so all 8,192 bins
// is ~2.2 s per call and scripts/verify-learn.mjs calls it six times. So in cpu
// mode the definition is asked for a 1,024-bin SLICE (`bins`) — an 8× head
// start, and it still loses by 36× (7.3 ms against 271 ms). Everything about
// the FFT side is identical in both modes; only how much of the spectrum the
// naive side is asked for changes, and the task says so in the code, in the
// prose and in the console.
//
// ---------------------------------------------------------------------------
// COMPLEX NUMBERS — TWO PLANES (the Signal Processing convention)
//
// gpu.js has no complex type, so a complex signal is two planes of floats:
//   kernel output  output: [n, 2]  → result[p][i], p = 0 real, p = 1 imaginary
//   kernel input   a [2][n] nested array → signal[0][i] real, signal[1][i] imag
// (output: [w, h] is indexed [y][x], so [n, 2] gives exactly [plane][i].)
// A purely real input signal is a plain [n] array; tasks 1, 3, 4 and 5 take
// one and the kernel fills plane 1 with zeros.
//
// ---------------------------------------------------------------------------
// BIT REVERSAL: ARITHMETIC, NOT BITWISE — measured, not assumed
//
// `>>`, `&` and `|` all produce CORRECT answers in gpu.js 2.20: verified in a
// real browser over all 512 indices at 9 bits, in cpu and gpu mode, against a
// float64 reference — zero mismatches. They are still the wrong spelling here.
// Both WebGL backends compile every one of them to a helper that walks up to
// BIT_COUNT = 32 bits in a loop (src/backend/web-gl{,2}/fragment-shader.js:
// bitwiseAnd / bitwiseOr / bitwiseZeroFillLeftShift / bitwiseSignedRightShift);
// the native GLSL integer operators are never emitted. Bitonic Sort found the
// same thing about `^`. Three of those per bit, thirteen bits at task 5's size,
// is ~1,250 shader loop iterations to compute one index — against thirteen
// float operations for
// `reversed = reversed * 2 + v % 2; v = Math.floor(v / 2);`. The arithmetic
// form is cheaper, portable, and it shows the learner what bit reversal IS —
// and it stays exact where it now has to: re-checked over all 8,192 indices at
// 13 bits, cpu and gpu, against the float64 reference — zero mismatches.
//
// ---------------------------------------------------------------------------
// FLOAT MARGINS — chosen from measurements, not from hope
//
// Tests compute in float64; the GL backend computes in float32, and a
// trigonometric sum accumulates error fast. Measured worst absolute error
// against the float64 reference, in gpu mode, on the signals these tasks ship:
//
//   task 1  4-point DFT, n = 8 .................. 4e-15   tolerance 1e-3
//   task 2  one butterfly pass, n = 8 ........... 4e-7    tolerance 1e-3
//   task 3  a permutation (exact integers) ...... 0       tolerance 1e-3
//   task 4  8-stage FFT, n = 256, peak 128 ...... 1.5e-5  tolerance 5e-3
//   task 5  naive DFT, n = 8,192 ................ see below
//   task 6  FFT round trip, n = 256 ............. 5.3e-7  tolerance 5e-3
//
// TASK 5 IS THE ONE PLACE A TOLERANCE IS A FRACTION OF THE SPECTRUM RATHER
// THAN A FIXED NUMBER, and the reason is worth stating: at n = 8,192 the
// naive transform is the INACCURATE one. Its phase is -2π·k·t/n with k·t up to
// 6.7e7 — past float32's 2^24 of integer resolution — so the angle it feeds
// cos/sin carries ~4e-3 rad of error, and 8,192 terms of that accumulate.
// Measured against a float64 reference in gpu mode, worst absolute error over
// the whole spectrum:
//
//   naive DFT, two tones .... 2.41  (peak 4,096 → 5.9e-4 relative)
//   naive DFT, noise ........ 0.43  (peak   307 → 1.4e-3 relative)
//   THE FFT, same signals ... 0.0015 and 0.0001  (3.6e-7 relative)
//
// The FFT is ~1,600× more accurate here, which is task 5's third punchline and
// not a coincidence: thirteen passes accumulate thirteen roundings, n² terms
// accumulate n². The assertions therefore use eps = 3% of the reference's own
// peak: ≥21× the measured error, and ≥33× smaller than the gap to the nearest
// wrong answer a probe names (every one of them is O(peak) away). Every other
// tolerance in this module is at least 20× the measured error and at least 30×
// smaller than that gap, so no assertion here sits near a decision boundary.
//
// TWO RULES THAT FALL OUT OF THAT, BOTH LEARNED THE HARD WAY ELSEWHERE:
//
//  1. A probe for a mistake that SCALES the answer scales its float error too,
//     and needs its own tolerance. See probeEps() and inverseProbes(): the
//     "missing 1/n" prediction is n times larger than the signal and therefore
//     n times noisier, and a probe held to the test's own epsilon would go
//     quiet on a backend noisier than the one it was measured on.
//
//  2. NOTHING HERE ASSERTS PHASE. atan2 on a bin whose imaginary part is float
//     noise flips by a full turn between backends, so a phase assertion near
//     the ±π seam passes on one machine and fails on the next. Every test in
//     this module compares real and imaginary parts, or magnitudes — never an
//     angle. If a future task wants one, assert it only on bins where BOTH
//     parts carry real energy, and say so in the test.

const TAU = 2 * Math.PI;

// ---- deterministic signals -------------------------------------------------
//
// No microphone and no WebAudio in a Web Worker, so every signal is built here
// and is exactly reproducible. Tones land on whole bin numbers, which makes the
// expected spectrum an exact number rather than a smear: a sine of amplitude A
// at bin b in n samples puts magnitude A·n/2 in bins b and n − b, and nothing
// anywhere else.

// The first eight digits of π. Small, distinct integers: every intermediate in
// task 1 comes out a whole number a learner can check by hand.
const PI_DIGITS = [3, 1, 4, 1, 5, 9, 2, 6];

// Task 2's complex input, as this module's two planes. Held here rather than
// built inside inputs() so the tests assert against the same eight numbers the
// learner is looking at, without going through the task object to find them.
const EIGHT_COMPLEX = [
  [1, -2, 3, 0.5, -1.5, 2, 4, -3],
  [0.5, 1, -1, 2, 0, -2.5, 1.5, 3],
];

// Sum of pure tones, sampled at n points. `tones` is [[bin, amplitude, 'sin'|'cos'], …].
function makeTones(n, tones) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (const [bin, amp, kind] of tones) {
      const phase = (TAU * bin * i) / n;
      v += amp * (kind === 'cos' ? Math.cos(phase) : Math.sin(phase));
    }
    out[i] = v;
  }
  return out;
}

// A plucked string: four decaying harmonics. Nothing lands on a clean bin, so
// task 6's round trip has to reproduce a genuinely messy waveform.
function makePluck(n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const decay = Math.exp((-3 * i) / n);
    let v = 0;
    for (let h = 1; h <= 4; h++) v += (1 / h) * Math.sin((TAU * 6 * h * i) / n);
    out[i] = Math.round(v * decay * 1000) / 1000;
  }
  return out;
}

// n deterministic samples, −2 … 2 with 3 decimals, all distinct enough that a
// permutation test can tell any two positions apart.
function makeSamples(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.round((rand() * 4 - 2) * 1000) / 1000;
  return out;
}

// ---- float64 references ----------------------------------------------------

// The definition, straight off the page: X[k] = Σ x[t]·e^(sign·2πi·kt/n).
// sign −1 is the forward transform, +1 the (unscaled) inverse.
function dftRef(re, im, sign = -1) {
  const n = re.length;
  const outRe = new Array(n).fill(0);
  const outIm = new Array(n).fill(0);
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < n; t++) {
      const angle = (sign * TAU * k * t) / n;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      outRe[k] += re[t] * c - im[t] * s;
      outIm[k] += re[t] * s + im[t] * c;
    }
  }
  return [outRe, outIm];
}

// The same definition, evaluated at a LIST of bins instead of all of them.
// Task 5 runs at n = 8,192, where a whole-spectrum dftRef is 67 million float64
// terms — about a wall-clock second per call, and its tests would want six of
// them. The assertions only ever read the bins in `list`, so those are the only
// bins worth predicting; results come back indexed by POSITION IN THE LIST, not
// by bin number.
function dftRefAt(x, list, sign = -1) {
  const n = x.length;
  const outRe = new Array(list.length).fill(0);
  const outIm = new Array(list.length).fill(0);
  for (let q = 0; q < list.length; q++) {
    const k = list[q];
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t++) {
      const angle = (sign * TAU * k * t) / n;
      re += x[t] * Math.cos(angle);
      im += x[t] * Math.sin(angle);
    }
    outRe[q] = re;
    outIm[q] = im;
  }
  return [outRe, outIm];
}

// Which bins task 5's assertions read out of the `bins` the naive kernel
// produced. The low end in full — DC, the tones, the first harmonics, where a
// spectrum is easiest to read by eye — then a stride to the very top, so no
// mistake can hide in a stretch nobody looked at. Around 96 bins either way,
// which is ~0.8 million float64 terms to predict rather than 67 million.
function checkBins(bins) {
  const list = [];
  for (let k = 0; k < 24 && k < bins; k++) list.push(k);
  const stride = Math.max(1, Math.floor(bins / 72));
  for (let k = 24; k < bins; k += stride) list.push(k);
  if (list[list.length - 1] !== bins - 1) list.push(bins - 1);
  return list;
}

// The biggest magnitude in a reference spectrum. Task 5's tolerance is a
// fraction of this rather than a fixed number — see the FLOAT MARGINS note.
function peakOf(re, im) {
  let peak = 0;
  for (let i = 0; i < re.length; i++) peak = Math.max(peak, Math.hypot(re[i], im[i]));
  return peak;
}

// The m-point DFT of every second sample of `x`, starting at `offset` — task 1's
// kernel, in plain JS.
function halfDftRef(x, offset, m) {
  const slice = new Array(m);
  for (let t = 0; t < m; t++) slice[t] = x[2 * t + offset];
  return dftRef(slice, new Array(m).fill(0), -1);
}

// ONE butterfly pass over a two-plane spectrum — the JS twin of the kernel
// tasks 2 and 4 build. `flaws` switches on exactly one deliberate mistake at a
// time, so every near-miss signature in this module comes out of the same
// function as the answer it is being compared against and the two can never
// drift apart.
function passRef(re, im, half, flaws = {}) {
  const n = re.length;
  const outRe = new Array(n);
  const outIm = new Array(n);
  for (let i = 0; i < n; i++) {
    const j = Math.floor(i / half) % 2; // 0 = lower member of my pair
    const base = i - j * half;
    const r = i % half;
    const angle = ((flaws.plusTwiddle ? 1 : -1) * Math.PI * r) / half;
    const wr = Math.cos(angle);
    const wi = Math.sin(angle);
    const ar = re[base];
    const ai = im[base];
    const br = re[base + half];
    const bi = im[base + half];
    if (flaws.twiddleOnA) {
      // w applied to the wrong half of the pair
      const tr = wr * ar - wi * ai;
      const ti = wr * ai + wi * ar;
      outRe[i] = j === 0 ? tr + br : tr - br;
      outIm[i] = j === 0 ? ti + bi : ti - bi;
      continue;
    }
    const tr = flaws.conjugateMultiply ? wr * br + wi * bi : wr * br - wi * bi;
    const ti = flaws.conjugateMultiply ? wr * bi - wi * br : wr * bi + wi * br;
    outRe[i] = j === 0 || flaws.alwaysAdd ? ar + tr : ar - tr;
    outIm[i] = j === 0 || flaws.alwaysAdd ? ai + ti : ai - ti;
  }
  return [outRe, outIm];
}

// reverse the low `bits` bits of i — 1 (0001) at 4 bits is 8 (1000).
function bitReverse(i, bits) {
  let v = i;
  let reversed = 0;
  for (let b = 0; b < bits; b++) {
    reversed = reversed * 2 + (v % 2);
    v = Math.floor(v / 2);
  }
  return reversed;
}

function scrambleRef(arr, bits) {
  const out = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[bitReverse(i, bits)];
  return out;
}

// The whole transform in float64: permute, then log₂n butterfly passes.
// `flaws` carries the pass-level switches plus three schedule-level ones —
// noScramble, oneStageShort and halvesDescending.
function fftRef(re, im, flaws = {}) {
  const n = re.length;
  const bits = Math.round(Math.log2(n));
  let R = flaws.noScramble ? Array.from(re) : scrambleRef(re, bits);
  let I = flaws.noScramble ? Array.from(im) : scrambleRef(im, bits);
  const halves = [];
  for (let half = 1; half < n; half *= 2) halves.push(half);
  if (flaws.oneStageShort) halves.pop();
  if (flaws.halvesDescending) halves.reverse();
  for (const half of halves) {
    const [nr, ni] = passRef(R, I, half, flaws);
    R = nr;
    I = ni;
  }
  return [R, I];
}

function magnitudeAt(re, im, i) {
  return Math.hypot(re[i], im[i]);
}

// ---- console helpers -------------------------------------------------------

// Every number that appeared in a console.log line.
function loggedNumbers(logs) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?/g);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

function logged(logs, value, eps) {
  return loggedNumbers(logs).some(v => Math.abs(v - value) <= eps);
}

function loggedText(logs, text) {
  return logs.some(line => line.type === 'log' && line.text && line.text.includes(text));
}

// ---- near-miss diagnosis ---------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake, and it may speak only when
// the observation matches it AND the correct answer does not — so candidates
// that coincide stay silent, as do observations matching probes that disagree
// with each other. A wrong diagnosis is worse than none.
//
// Every probe in this module is whole-array, because a spectrum is index-shaped
// and one matching bin is worth nothing: bin 0 is the plain sum of the samples
// under every mistake here, and a twiddle of W⁰ = 1 hides the twiddle mistakes
// entirely. So a probe must predict EVERY element of BOTH planes (and disagree
// with the right answer somewhere) before it is allowed to speak. Probe values
// are functions of (plane, index); a missing element makes the comparison NaN,
// which fails, which keeps the probe quiet.
//
// A PROBE MAY CARRY ITS OWN TOLERANCE, as an optional third slot. This is not a
// convenience — it is a correctness requirement for any mistake that SCALES the
// answer. A learner whose inverse transform forgot its 1/n produces values n
// times too large, and float32 error scales with them: at n = 256 an observation
// carrying 5e-4 of round-trip error arrives 0.13 away from the prediction, and a
// probe held to the test's own 5e-3 would silently never fire on a backend one
// order of magnitude noisier than the one it was written on. The probe's
// tolerance governs "does the observation match me"; the TEST's tolerance still
// governs "and does the right answer not", so a loose probe can never claim an
// observation that the correct answer explains just as well.
function probeEps(probe, eps) {
  return probe.length > 2 && probe[2] != null ? probe[2] : eps;
}

function diagnoseSpectrum(n, got, expected, eps, probes) {
  const hits = probes
    .filter(probe => {
      const [value] = probe;
      const mine = probeEps(probe, eps);
      let differs = false;
      for (let p = 0; p < 2; p++) {
        for (let i = 0; i < n; i++) {
          const c = value(p, i);
          if (!(Math.abs(got(p, i) - c) <= mine)) return false;
          if (Math.abs(expected(p, i) - c) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Same rule for a single plane of numbers (task 1's half-length DFTs, task 3's
// permutation, task 6's recovered samples): predict all of it or say nothing.
function diagnoseAll(count, got, expected, eps, probes) {
  const hits = probes
    .filter(probe => {
      const [value] = probe;
      const mine = probeEps(probe, eps);
      let differs = false;
      for (let i = 0; i < count; i++) {
        if (!(Math.abs(got(i) - value(i)) <= mine)) return false;
        if (Math.abs(expected(i) - value(i)) > eps) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// ---- per-task probe sets ---------------------------------------------------

// Task 1: the half-length DFT of a strided slice. Three ways to miss.
function halfDftProbes(x, offset, m) {
  // "four samples in a row" has two spellings — ignoring `offset` altogether,
  // and using it to pick a contiguous block. At offset 0 they are the same
  // numbers, so both carry the SAME sentence: two probes that agree can speak,
  // two that disagree cancel each other into silence.
  const inARow =
    'that is the DFT of four samples in a ROW, not of every second sample. The split is by ' +
    'PARITY: element t of the half is x[2 * t + offset], so offset 0 takes x[0], x[2], x[4], ' +
    'x[6] and offset 1 takes x[1], x[3], x[5], x[7]';
  const blockAt = start => {
    const slice = new Array(m);
    for (let t = 0; t < m; t++) slice[t] = x[start + t];
    return dftRef(slice, new Array(m).fill(0), -1);
  };
  const ignoredOffset = blockAt(0);
  const contiguousBlock = blockAt(offset * m);
  const wholeLengthDenominator = (() => {
    const outRe = new Array(m).fill(0);
    const outIm = new Array(m).fill(0);
    for (let k = 0; k < m; k++) {
      for (let t = 0; t < m; t++) {
        const angle = (-TAU * k * t) / (2 * m);
        outRe[k] += x[2 * t + offset] * Math.cos(angle);
        outIm[k] += x[2 * t + offset] * Math.sin(angle);
      }
    }
    return [outRe, outIm];
  })();
  const flipped = (() => {
    const slice = new Array(m);
    for (let t = 0; t < m; t++) slice[t] = x[2 * t + offset];
    return dftRef(slice, new Array(m).fill(0), 1);
  })();
  return [
    [(p, i) => ignoredOffset[p][i], inARow],
    [(p, i) => contiguousBlock[p][i], inARow],
    [(p, i) => wholeLengthDenominator[p][i],
      'the exponent still divides by 8 — but this is a 4-point transform, so its angles are ' +
      '-2π·k·t / 4. The whole point is that the half is a smaller DFT in its own right'],
    [(p, i) => flipped[p][i],
      'every imaginary part has the wrong sign: that is the exponent + rather than -. ' +
      'The forward transform is e^(-2πi·kt/n)'],
  ];
}

// Task 2 and 4: the four ways one butterfly pass goes wrong. Two notes on why
// the messages are shaped the way they are.
//
// At half = 1 the twiddle is W⁰ = 1, so plusTwiddle, conjugateMultiply and
// twiddleOnA all produce byte-identical output there — every caller therefore
// probes at half ≥ 2 as well, where they separate by 6 or more.
//
// And plusTwiddle and conjugateMultiply produce identical output at EVERY half,
// because multiplying by the conjugate of w is the same arithmetic as
// conjugating the multiply. Two probes with different sentences would cancel
// each other into silence, so they share one that is true of both — which is
// also the more useful sentence, since it names the observable (a conjugated
// spectrum) and then both ways of causing it.
function passProbes(re, im, half) {
  const variant = flaws => passRef(re, im, half, flaws);
  const plus = variant({ plusTwiddle: true });
  const conjMul = variant({ conjugateMultiply: true });
  const onA = variant({ twiddleOnA: true });
  const add = variant({ alwaysAdd: true });
  const conjugated =
    'your result is the complex conjugate of the right one — every imaginary part negated. ' +
    'Two slips give exactly this, and they are indistinguishable from the numbers: an angle of ' +
    '+π·r / half instead of -π·r / half, and a complex multiply with its signs crossed. ' +
    '(wr + i·wi)(br + i·bi) is (wr·br - wi·bi) + i(wr·bi + wi·br) — the minus belongs in the ' +
    'REAL part';
  return [
    [(p, i) => plus[p][i], conjugated],
    [(p, i) => conjMul[p][i], conjugated],
    [(p, i) => onA[p][i],
      'the twiddle is on the wrong half of the pair. It multiplies b, the upper element — ' +
      'a is added and subtracted untouched: a + w·b and a - w·b'],
    [(p, i) => add[p][i],
      'both members of every pair returned a + w·b, so the minus branch never happened — ' +
      'and a butterfly that only adds throws half its information away'],
    [(p, i) => (p === 0 ? re[i] : im[i]),
      'the spectrum came back unchanged — no thread combined anything'],
  ];
}

// `base = i` — the pairing mistake that looks right for exactly half the
// threads, because the lower member of every pair has base = i anyway. It is
// not a whole-array probe: where it reads past the end of the array the value
// is unknowable (the CPU backend gives NaN, the GL backend clamps), so those
// cells are skipped. Every OTHER cell has to match, at least half the array has
// to be predictable, and the prediction has to disagree with the right answer
// somewhere, before it may speak.
function baseIsIHint(n, got, expected, eps, re, im, half) {
  let matched = 0;
  let differs = false;
  for (let i = 0; i < n; i++) {
    if (i + half >= n) continue;
    const j = Math.floor(i / half) % 2;
    const r = i % half;
    const angle = (-Math.PI * r) / half;
    const wr = Math.cos(angle);
    const wi = Math.sin(angle);
    const tr = wr * re[i + half] - wi * im[i + half];
    const ti = wr * im[i + half] + wi * re[i + half];
    const vr = j === 0 ? re[i] + tr : re[i] - tr;
    const vi = j === 0 ? im[i] + ti : im[i] - ti;
    if (Math.abs(got(0, i) - vr) > eps || Math.abs(got(1, i) - vi) > eps) return null;
    matched++;
    if (Math.abs(expected(0, i) - vr) > eps || Math.abs(expected(1, i) - vi) > eps) differs = true;
  }
  return matched >= n / 2 && differs
    ? 'the pair distance is right but the pair is not. Every thread read data[i] and ' +
      'data[i + half], so the UPPER member of each pair took the wrong partner — and near the ' +
      'top of the array it read past the end. Both members of a pair have to start from the ' +
      'same element: base = i - j * half'
    : null;
}

// Everything task 2 knows about a wrong butterfly pass, in the order the
// messages are worth hearing.
function butterflyHint(n, got, expected, eps, re, im, half) {
  return diagnoseSpectrum(n, got, expected, eps, passProbes(re, im, half)) ||
    baseIsIHint(n, got, expected, eps, re, im, half);
}

// Task 3: the permutation. Reversing over the wrong number of bits and not
// reversing at all are the two that actually happen.
function scrambleProbes(x, bits) {
  const probes = [
    [i => x[i], 'the values came back in their original order — no index was reversed'],
  ];
  for (const wrong of [bits - 1, bits + 1]) {
    if (wrong < 1) continue;
    const permuted = new Array(x.length);
    let inRange = true;
    for (let i = 0; i < x.length; i++) {
      const src = bitReverse(i, wrong);
      if (src >= x.length) inRange = false;
      permuted[i] = x[src];
    }
    if (!inRange) continue;
    probes.push([i => permuted[i],
      'the reversal is over ' + wrong + ' bits, not ' + bits + '. The width is log2(n) — ' +
      'every index of an n-point transform has to be turned around in full']);
  }
  return probes;
}

// Task 4: the three ways the SCHEDULE goes wrong, each one exact. Every
// signature comes from fftRef, so it is the same arithmetic as the answer.
function scheduleProbes(signal, n) {
  const zeros = new Array(n).fill(0);
  const noScramble = fftRef(signal, zeros, { noScramble: true });
  const short = fftRef(signal, zeros, { oneStageShort: true });
  const descending = fftRef(signal, zeros, { halvesDescending: true });
  const plus = fftRef(signal, zeros, { plusTwiddle: true });
  return [
    [(p, i) => noScramble[p][i],
      'the butterflies ran, and they ran on the samples in their original order — so what came ' +
      'back is the exact spectrum of a DIFFERENT signal: yours with its sample indices ' +
      'bit-reversed. The permutation belongs on the input, before the first pass'],
    [(p, i) => short[p][i],
      'one pass short. What you have is two independent 128-point spectra sitting side by side ' +
      'in one 256-slot array — the even samples in the first half, the odd samples in the ' +
      'second — which is exactly the state the very first task left them in. The last pass is ' +
      'the one that combines them: run halves while half < n, not while half < n / 2'],
    [(p, i) => descending[p][i],
      'the halves run the wrong way. A decimation-in-time transform starts with adjacent pairs ' +
      'and reaches further every pass: 1, 2, 4, … n / 2, not n / 2 down to 1'],
    [(p, i) => plus[p][i],
      'the twiddle turns the wrong way, so this is the inverse transform: the complex conjugate ' +
      'of the right spectrum, mirrored about bin 0. Either the angle is +π·r / half instead of ' +
      '-π·r / half, or the complex multiply has its signs crossed — the two are indistinguishable ' +
      'from the numbers. Every MAGNITUDE is identical to the right answer, which is why a ' +
      'magnitude plot would have told you nothing'],
  ];
}

// Task 5: the naive DFT itself. Everything here is indexed by POSITION IN
// `list` (see dftRefAt): the kernel produces `bins` of them, the assertions
// read ~96, and a probe has to predict exactly the cells the assertions read.
// `correct` is passed in because the caller has already paid for it, and `eps`
// because two of these predictions are a factor of n SMALLER than the spectrum
// and would otherwise be swallowed by a tolerance sized for the spectrum: at
// n = 8,192 the test's eps is 3% of a peak of 4,096, i.e. 123, while a spectrum
// divided by n peaks at 0.5. Held to 123 that prediction matches a kernel that
// returned nothing at all, and the learner whose loop body is still a TODO gets
// told their scale factor is wrong. Both small predictions therefore carry
// eps/n, which is ~50× their own float error and still nowhere near zero.
function naiveProbes(signal, list, correctRe, correctIm, eps) {
  const n = signal.length;
  const smallEps = eps / n;
  // The forward transform of a REAL signal and its inverse differ only in the
  // sign of the imaginary part, so the "wrong exponent" prediction is free.
  const inverse = [correctRe, correctIm.map(v => -v)];
  const scaled = [correctRe.map(v => v / n), correctIm.map(v => v / n)];
  const swapped = (() => {
    // k and t exchanged: the sum runs over bins instead of samples.
    const outRe = new Array(list.length).fill(0);
    const outIm = new Array(list.length).fill(0);
    for (let q = 0; q < list.length; q++) {
      const k = list[q];
      for (let t = 0; t < n; t++) {
        const angle = (-TAU * k * t) / n;
        outRe[q] += signal[k] * Math.cos(angle);
        outIm[q] += signal[k] * Math.sin(angle);
      }
    }
    return [outRe, outIm];
  })();
  return [
    [(p, i) => inverse[p][i],
      'every imaginary part has the wrong sign — that is e^(+2πi·kt/n), the INVERSE transform. ' +
      'Its spectrum is the correct one mirrored about bin 0, and for a real signal the ' +
      'magnitudes are identical, so nothing but the sign gives it away'],
    // Dividing by n divides the accumulated float error by n too, so this
    // observation is QUIETER than the test's own tolerance, not noisier — the
    // override above is what keeps it from swallowing an empty spectrum.
    [(p, i) => scaled[p][i],
      'the whole spectrum is divided by n. Some texts put the 1/n on the forward transform; ' +
      'this course puts it on the inverse (task 6), so the forward transform must not scale',
      smallEps],
    [(p, i) => swapped[p][i],
      'the sample index and the bin index are the wrong way round — the loop variable indexes ' +
      'the signal, x[t], and this.thread.x is the bin k'],
    [() => 0,
      'every bin came back zero, in both planes — the accumulation loop is running but nothing ' +
      'inside it is adding to re and im',
      smallEps],
  ];
}

// Task 6: the four ways to get the inverse nearly right.
//
// The first two SCALE the answer, so they scale its float error with it and
// carry their own tolerances (see probeEps): a recovered signal n times too big
// is n times as noisy, and holding it to the test's 5e-3 would make the probe
// go quiet on any backend noisier than the one these numbers were measured on.
// The margins stay enormous either way — at n = 256 the "no 1/n" prediction is
// 255·|x| from the right answer and its tolerance is n·eps, a factor of 289.
function inverseProbes(signal, n, eps) {
  const half = signal.map(v => v / 2);
  const scaledByRootN = signal.map(v => v * Math.sqrt(n));
  const timesN = signal.map(v => v * n);
  return [
    [i => timesN[i],
      'every value is exactly n times too big — the 1/n is missing. A forward transform ' +
      'followed by an unscaled inverse multiplies the signal by n',
      eps * n],
    [i => scaledByRootN[i],
      'the scale is 1/sqrt(n), the symmetric convention where both directions carry half the ' +
      'factor. This module puts the whole 1/n on the inverse, so the forward transform is ' +
      'left alone',
      eps * Math.sqrt(n)],
    [i => half[i], 'the signal came back at half amplitude — check the scale factor'],
    [i => -signal[i],
      'the sign is inverted across the board: the real part is being negated somewhere. ' +
      'Conjugation negates the IMAGINARY part and leaves the real part alone'],
  ];
}

// Find the kernel whose output is [w, 2] — every kernel in this module returns a
// two-plane spectrum, so this is how a test picks one out of a run that built
// several without depending on the order they were created in.
function kernelWithWidth(ctx, width) {
  for (let i = ctx.kernels.length - 1; i >= 0; i--) {
    const k = ctx.kernels[i];
    const output = k && k.kernel && k.kernel.output;
    if (output && output[0] === width && output[1] === 2) return k;
  }
  return null;
}

// How many arguments a kernel's function declares. gpu.js parses the names at
// createKernel() time, before any build, so this costs nothing and — unlike
// invoking a kernel to see what happens — cannot disturb the argument types
// gpu.js locked on the learner's own first call.
function argCount(k) {
  const names = k && k.kernel && k.kernel.argumentNames;
  return Array.isArray(names) ? names.length : -1;
}

function twoPlaneKernels(ctx, n) {
  return ctx.kernels.filter(k => {
    const output = k && k.kernel && k.kernel.output;
    return output && output[0] === n && output[1] === 2;
  });
}

// The two kernels an FFT is made of, told apart by their ARITY rather than by
// the order they happen to have been created in: the permutation takes the
// signal alone, the butterfly takes the spectrum and the half-block size.
function findFftKernels(ctx, n) {
  const candidates = twoPlaneKernels(ctx, n);
  return {
    scramble: candidates.find(k => argCount(k) === 1) || null,
    butterfly: candidates.find(k => argCount(k) === 2) || null,
  };
}

// ---- task 5: finding the learner's naive DFT --------------------------------
//
// Task 5's run builds four kernels — the permutation, two butterfly passes and
// the naive transform — and in gpu mode all four declare the same [n, 2]
// output, so width alone cannot pick one out. Arity narrows it to two (the
// permutation and the naive transform each take just the signal), and BEHAVIOUR
// settles it: handed a unit impulse at t = 0, a forward transform returns 1 in
// every bin, while a permutation returns the impulse somewhere else. The
// butterfly kernels are never invoked, so nothing is ever called at the wrong
// arity — and the permutation, being pipelined, hands back a texture rather
// than an array, which rules it out before any arithmetic is looked at.
//
// In cpu mode the naive side is asked for a 1,024-bin slice, so it is the only
// kernel of its width and the search ends on the first candidate.

const NAIVE_N = 8192; // the transform length task 5 runs at, both modes
const NAIVE_CPU_BINS = 1024; // how much of the spectrum the CPU backend is asked for

// The search runs ONCE PER RUN rather than once per test: on the CPU backend a
// speculative call to the naive kernel costs ~0.27 s, and task 5's three tests
// would otherwise pay for the same answer three times. Keyed by the run's
// kernel array, which is a fresh object for every run.
const naiveCache = new WeakMap();

function naiveCandidates(ctx, width) {
  return twoPlaneKernels(ctx, width).filter(k => argCount(k) === 1);
}

// Every bin of the transform of a unit impulse at t = 0 is exactly 1.
// Async because it invokes a kernel: under gpu.js's async mode every call comes
// back as a Promise, so the probe has to be awaited before it can be read.
async function transformsAnImpulseFlat(k, width) {
  const impulse = new Array(NAIVE_N).fill(0);
  impulse[0] = 1;
  let out;
  try {
    out = await k(impulse);
  } catch (e) {
    return false; // an argument type it was not built for
  }
  // a pipelined kernel hands back a texture, not indexable planes
  if (!out || !out[0] || out[0].length !== width) return false;
  for (let i = 0; i < width; i++) {
    if (Math.abs(out[0][i] - 1) > 0.05 || Math.abs(out[1][i]) > 0.05) return false;
  }
  return true;
}

// When nothing passes the impulse test the learner has still built something,
// and reporting only "nothing looked like a transform" throws away every probe
// below. So fall back to the one-argument kernel that is plainly NOT the
// permutation (its output on a ramp is not a rearrangement of that ramp) and
// let the spectrum probes say what actually went wrong with it.
async function rearrangesARamp(k, width) {
  const ramp = new Array(NAIVE_N);
  for (let i = 0; i < NAIVE_N; i++) ramp[i] = i;
  let out;
  try {
    out = await k(ramp);
  } catch (e) {
    return true; // cannot be read, so it is not the candidate we want
  }
  if (!out || !out[0] || out[0].length !== width) return true;
  const seen = new Set();
  for (let i = 0; i < width; i++) {
    const v = Math.round(out[0][i]);
    if (Math.abs(out[0][i] - v) > 1e-3 || v < 0 || v >= NAIVE_N || seen.has(v)) return false;
    seen.add(v);
  }
  return true;
}

// Async, and the .find() calls are now sequential for loops: an async predicate
// handed to Array.prototype.find returns a Promise, which is truthy, so find()
// would settle on the first candidate whatever it did. One candidate is probed
// at a time — these probes are ~0.27 s each on the CPU backend and firing them
// concurrently would only queue them anyway.
async function findNaiveDft(ctx) {
  if (naiveCache.has(ctx.kernels)) return naiveCache.get(ctx.kernels);
  const widths = [NAIVE_N, NAIVE_CPU_BINS];
  let found = null;
  for (const width of widths) {
    for (const k of naiveCandidates(ctx, width)) {
      if (await transformsAnImpulseFlat(k, width)) {
        found = k;
        break;
      }
    }
    if (found) break;
  }
  if (!found) {
    for (const width of widths) {
      for (const k of naiveCandidates(ctx, width)) {
        if (!(await rearrangesARamp(k, width))) {
          found = k;
          break;
        }
      }
      if (found) break;
    }
  }
  naiveCache.set(ctx.kernels, found);
  return found;
}

// How many bins the learner actually asked their naive kernel for — the tests
// read it off the kernel rather than re-deriving it from the mode, so a learner
// who changes `bins` is still measured against what they built.
function binsOf(k) {
  const output = k && k.kernel && k.kernel.output;
  return output ? output[0] : 0;
}

// Task 6 builds three kernels, two of which take two arguments — so arity is
// not enough. The conjugate is found by what it DOES (with scale = 1 the real
// plane comes back untouched and the imaginary one negated, which the butterfly
// fails at almost every cell), and the butterfly is then simply the other one.
// `planes` must be the same container shape the learner's own run used, or
// gpu.js rejects the call against the argument types it locked.
async function findInverseKernels(ctx, n, planes) {
  const twoArg = twoPlaneKernels(ctx, n).filter(k => argCount(k) === 2);
  let conjugate = null;
  const behaviour = new Map();
  for (const k of twoArg) {
    let out;
    try {
      out = await k(planes, 1);
    } catch (e) {
      continue;
    }
    if (!out || !out[0] || out[0].length !== n) continue;
    let conjugates = true;
    let unchanged = true;
    let bothNegated = true;
    for (let i = 0; i < n; i++) {
      if (Math.abs(out[0][i] - planes[0][i]) > 5e-3 || Math.abs(out[1][i] + planes[1][i]) > 5e-3) {
        conjugates = false;
      }
      if (Math.abs(out[0][i] - planes[0][i]) > 5e-3 || Math.abs(out[1][i] - planes[1][i]) > 5e-3) {
        unchanged = false;
      }
      if (Math.abs(out[0][i] + planes[0][i]) > 5e-3 || Math.abs(out[1][i] + planes[1][i]) > 5e-3) {
        bothNegated = false;
      }
    }
    behaviour.set(k, { unchanged, bothNegated });
    if (conjugates && !conjugate) conjugate = k;
  }
  return {
    scramble: twoPlaneKernels(ctx, n).find(k => argCount(k) === 1) || null,
    butterfly: twoArg.find(k => k !== conjugate) || null,
    conjugate,
    behaviour,
  };
}

// When no kernel conjugated, say which of the two near misses happened.
function conjugateHint(behaviour) {
  for (const { unchanged, bothNegated } of behaviour.values()) {
    if (bothNegated) {
      return 'both planes came back negated. Conjugation flips the sign of the IMAGINARY part ' +
        'only — negating the real part as well is multiplying by -1, which is a different ' +
        'operation entirely and does not invert anything';
    }
    if (unchanged) {
      return 'the kernel handed its input straight back — conjugation has to negate the ' +
        'imaginary plane, the one this.thread.y === 1 owns';
    }
  }
  return null;
}

export default {
  uuid: 'd4375da7-7178-4bee-8442-e04e80d563d1',
  version: 1,
  slug: 'fft-butterfly',
  title: 'The FFT Butterfly',
  blurb:
    'Split the sum by parity and the transform collapses from n² terms to log₂n passes of a ' +
    'two-line <code>butterfly</code> — the same multi-pass gather every ladder in this course uses.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'split-the-sum',
      title: 'Where the Saving Comes From',
      intro: `<p>The discrete Fourier transform asks one question n times: <em>how much of the
        signal is a wave that fits exactly k times into the window?</em> The answer is a sum over
        every sample, so n answers cost n² terms:</p>
<pre><code>X[k] = Σ  x[t] · e^(-2πi·k·t / n)      t = 0 … n-1</code></pre>
        <p>At n = 512 that is 262,144 sine-cosine terms per transform, and the count grows with the
        square. Now split the sum by the <strong>parity of t</strong> — the even-indexed samples in
        one pile, the odd-indexed in the other. Pull the shared factor out of the odd pile and
        something remarkable is left standing:</p>
<pre><code>X[k]       = E[k] + W·O[k]        W = e^(-2πi·k / n)
X[k + n/2] = E[k] - W·O[k]</code></pre>
        <p>where <code>E</code> and <code>O</code> are the (n/2)-point DFTs of the even and the odd
        samples. Two half-length transforms and one multiplication give you the whole thing — and
        <em>they</em> can be split the same way, and so on down. That recursion is the entire
        algorithm: n² becomes n·log₂n. Do not take that on faith. Check it.</p>
        <p>Complex numbers arrive here as <strong>two planes of floats</strong>, the convention every
        module in this track shares: a kernel with <code>output: [n, 2]</code> is indexed
        <code>result[p][i]</code>, plane <code>0</code> real and plane <code>1</code> imaginary.
        <code>this.thread.y</code> tells a thread which plane it owns.</p>`,
      goal: `<strong>Goal:</strong> compute the two 4-point half-transforms of <code>signal</code>
        with one kernel, combine them by hand, and confirm the result is the 8-point DFT the
        <code>dft8</code> kernel already computes.`,
      requirements: [
        'One <code>halfDft</code> kernel, called twice — <code>offset 0</code> for the even samples, <code>offset 1</code> for the odd',
        'It reads every SECOND sample: element <code>t</code> is <code>x[2 * t + offset]</code>',
        'Its angles divide by 4, not 8 — it is a 4-point transform',
        'Combine with <code>W = e^(-2πi·k/8)</code> and <code>console.log</code> the largest disagreement',
      ],
      hints: [
        {
          title: 'Hint 1 — which samples are mine?',
          body: `<p>The even pile is <code>x[0], x[2], x[4], x[6]</code>; the odd pile is
            <code>x[1], x[3], x[5], x[7]</code>. Both are "every second sample", differing only in
            where they start — so element <code>t</code> of the pile is
            <code>x[2 * t + offset]</code>, and <code>offset</code> is an ordinary kernel
            argument.</p>`,
        },
        {
          title: 'Hint 2 — multiplying two complex numbers',
          body: `<p>There is no complex type, so you write it out. With
            <code>w = wr + i·wi</code> and <code>o = or + i·oi</code>:</p>
<pre><code>// the minus lives in the REAL part
const tr = wr * or - wi * oi;
const ti = wr * oi + wi * or;</code></pre>`,
        },
        {
          title: 'Hint 3 — the check',
          body: `<p>For each <code>k</code> in 0…3, all four of these have to hold:</p>
<pre><code>E[0][k] + tr  →  X[0][k]
E[1][k] + ti  →  X[1][k]
E[0][k] - tr  →  X[0][k + 4]
E[1][k] - ti  →  X[1][k + 4]</code></pre>
<p>Four bins of <code>E</code> and <code>O</code> reconstruct all eight bins of
            <code>X</code>. That is the saving, in miniature.</p>`,
        },
      ],
      transfer: `Every FFT in the world is this recursion made iterative — FFTW, cuFFT, rocFFT,
        vDSP, Metal Performance Shaders. The radix varies (4, 8 and split-radix are common, and
        cuFFT picks one per size), and Bluestein's algorithm rescues the sizes that will not
        factor, but the move is always the same: turn one transform into several smaller ones plus
        a twiddle.`,
      starterCode: `// Two 4-point transforms and one multiplication should rebuild all 8 bins.
const gpu = new GPU({ mode });

// The definition, written out: 8 bins × 8 samples = 64 terms.
const dft8 = gpu.createKernel(function (x) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.n; t++) {
    const angle = (-2 * Math.PI * k * t) / this.constants.n;
    re += x[t] * Math.cos(angle);
    im += x[t] * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [8, 2], constants: { n: 8 } });

const halfDft = gpu.createKernel(function (x, offset) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.m; t++) {
    const angle = (-2 * Math.PI * k * t) / this.constants.m;
    // TODO: read every SECOND sample, starting at \`offset\`.
    const v = x[t];
    re += v * Math.cos(angle);
    im += v * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [4, 2], constants: { m: 4 } });

const X = await dft8(signal);
const E = await halfDft(signal, 0);
const O = await halfDft(signal, 1);
console.log('E real:', E[0]);
console.log('O real:', O[0]);

let worst = 0;
for (let k = 0; k < 4; k++) {
  const angle = (-2 * Math.PI * k) / 8;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);

  // TODO: t = W · O[k] — a complex multiply, two lines.
  const tr = 0;
  const ti = 0;

  worst = Math.max(worst,
    Math.abs(E[0][k] + tr - X[0][k]), Math.abs(E[1][k] + ti - X[1][k]),
    Math.abs(E[0][k] - tr - X[0][k + 4]), Math.abs(E[1][k] - ti - X[1][k + 4]));
}

console.log('largest disagreement:', worst);
console.log('identity holds:', worst < 1e-3);
`,
      solutionCode: `// Two 4-point transforms and one multiplication should rebuild all 8 bins.
const gpu = new GPU({ mode });

// The definition, written out: 8 bins × 8 samples = 64 terms.
const dft8 = gpu.createKernel(function (x) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.n; t++) {
    const angle = (-2 * Math.PI * k * t) / this.constants.n;
    re += x[t] * Math.cos(angle);
    im += x[t] * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [8, 2], constants: { n: 8 } });

const halfDft = gpu.createKernel(function (x, offset) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.m; t++) {
    const angle = (-2 * Math.PI * k * t) / this.constants.m;
    const v = x[2 * t + offset];
    re += v * Math.cos(angle);
    im += v * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [4, 2], constants: { m: 4 } });

const X = await dft8(signal);
const E = await halfDft(signal, 0);
const O = await halfDft(signal, 1);
console.log('E real:', E[0]);
console.log('O real:', O[0]);

let worst = 0;
for (let k = 0; k < 4; k++) {
  const angle = (-2 * Math.PI * k) / 8;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);

  const tr = wr * O[0][k] - wi * O[1][k];
  const ti = wr * O[1][k] + wi * O[0][k];

  worst = Math.max(worst,
    Math.abs(E[0][k] + tr - X[0][k]), Math.abs(E[1][k] + ti - X[1][k]),
    Math.abs(E[0][k] - tr - X[0][k + 4]), Math.abs(E[1][k] - ti - X[1][k + 4]));
}

console.log('largest disagreement:', worst);
console.log('identity holds:', worst < 1e-3);
`,
      inputs: () => ({ signal: PI_DIGITS.slice() }),
      publicTests: [
        {
          name: 'the even half: <code>halfDft(signal, 0)</code> transforms <code>x[0], x[2], x[4], x[6]</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, 'expected two kernels — dft8 and halfDft');
            const half = kernelWithWidth(ctx, 4);
            ctx.assert(half, 'no kernel with output [4, 2] found — the half transform has 4 bins and 2 planes');
            const out = await half(PI_DIGITS.slice(), 0);
            ctx.assert(
              out && out.length === 2 && out[0].length === 4,
              'expected a [4, 2] result — 2 planes of 4 bins'
            );
            const [wantRe, wantIm] = halfDftRef(PI_DIGITS, 0, 4);
            const hint = diagnoseSpectrum(
              4, (p, i) => out[p][i], (p, i) => (p === 0 ? wantRe[i] : wantIm[i]),
              1e-3, halfDftProbes(PI_DIGITS, 0, 4)
            );
            for (let k = 0; k < 4; k++) {
              ctx.assertClose(out[0][k], wantRe[k], 1e-3, hint || `even half, real part of bin ${k}`);
              ctx.assertClose(out[1][k], wantIm[k], 1e-3, hint || `even half, imaginary part of bin ${k}`);
            }
          },
        },
        {
          name: 'the odd half: <code>halfDft(signal, 1)</code> transforms <code>x[1], x[3], x[5], x[7]</code>',
          run: async ctx => {
            const half = kernelWithWidth(ctx, 4);
            ctx.assert(half, 'no kernel with output [4, 2] found');
            const out = await half(PI_DIGITS.slice(), 1);
            const [wantRe, wantIm] = halfDftRef(PI_DIGITS, 1, 4);
            const hint = diagnoseSpectrum(
              4, (p, i) => out[p][i], (p, i) => (p === 0 ? wantRe[i] : wantIm[i]),
              1e-3, halfDftProbes(PI_DIGITS, 1, 4)
            );
            for (let k = 0; k < 4; k++) {
              ctx.assertClose(out[0][k], wantRe[k], 1e-3, hint || `odd half, real part of bin ${k}`);
              ctx.assertClose(out[1][k], wantIm[k], 1e-3, hint || `odd half, imaginary part of bin ${k}`);
            }
          },
        },
        {
          name: 'the identity is checked and reported as holding',
          run: async ctx => {
            // The verdict, not the residual: `worst` is a number like 2.6e-14
            // in cpu mode and 1e-5 in gpu mode, and JavaScript prints the first
            // of those in exponential notation — so a numeric probe on the
            // logged text would pass in one mode and fail in the other.
            ctx.assert(
              loggedText(ctx.logs, 'identity holds: true'),
              'the identity does not hold yet — E[k] + W·O[k] has to reproduce X[k], and ' +
              'E[k] - W·O[k] has to reproduce X[k + 4]. If the halves are right, the twiddle is ' +
              'what is missing: W = e^(-2πi·k/8), applied to O and to O alone'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different 8-sample signal, with negatives: nothing about the
            // split cares, and a kernel that quietly hard-coded a stride would.
            const half = kernelWithWidth(ctx, 4);
            ctx.assert(half, 'no kernel with output [4, 2] found');
            const x = makeSamples(ctx.utils, 8, 4375);
            for (const offset of [0, 1]) {
              const out = await half(x, offset);
              const [wantRe, wantIm] = halfDftRef(x, offset, 4);
              const hint = diagnoseSpectrum(
                4, (p, i) => out[p][i], (p, i) => (p === 0 ? wantRe[i] : wantIm[i]),
                1e-3, halfDftProbes(x, offset, 4)
              );
              for (let k = 0; k < 4; k++) {
                ctx.assertClose(out[0][k], wantRe[k], 1e-3, hint || `offset ${offset}, real bin ${k}`);
                ctx.assertClose(out[1][k], wantIm[k], 1e-3, hint || `offset ${offset}, imaginary bin ${k}`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The identity itself, recomputed here from the learner's own
            // halves: bin 0 always works, so this checks all eight.
            const half = kernelWithWidth(ctx, 4);
            const full = kernelWithWidth(ctx, 8);
            ctx.assert(half && full, 'expected an 8-bin kernel and a 4-bin kernel');
            const x = makeSamples(ctx.utils, 8, 90210);
            const E = await half(x, 0);
            const O = await half(x, 1);
            const [XR, XI] = dftRef(x, new Array(8).fill(0), -1);
            for (let k = 0; k < 4; k++) {
              const angle = (-TAU * k) / 8;
              const wr = Math.cos(angle);
              const wi = Math.sin(angle);
              const tr = wr * O[0][k] - wi * O[1][k];
              const ti = wr * O[1][k] + wi * O[0][k];
              ctx.assertClose(E[0][k] + tr, XR[k], 1e-3, `E[${k}] + W·O[${k}] should be bin ${k}`);
              ctx.assertClose(E[1][k] + ti, XI[k], 1e-3, `E[${k}] + W·O[${k}], imaginary part`);
              ctx.assertClose(E[0][k] - tr, XR[k + 4], 1e-3, `E[${k}] - W·O[${k}] should be bin ${k + 4}`);
              ctx.assertClose(E[1][k] - ti, XI[k + 4], 1e-3, `E[${k}] - W·O[${k}], imaginary part`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'the-butterfly',
      title: 'The Butterfly: One Thread, One Output',
      intro: `<p>Written as a recursion the split is elegant and useless: a GPU cannot recurse, and
        allocating a tree of half-length arrays would cost more than the arithmetic saves. Turn it
        inside out instead. Every level of that recursion is one <strong>pass</strong> over the
        whole array, and every pass is the same tiny operation repeated: take two elements
        <code>a</code> and <code>b</code>, and produce</p>
<pre><code>a + w·b        and        a - w·b</code></pre>
        <p>Two in, two out, one twiddle factor <code>w</code>. Drawn on paper the crossing lines
        look like a butterfly, and the name stuck.</p>
        <p>Here is where a half-remembered recursive formulation has to be let go. A thread does not
        <em>own a pair</em> and write two cells — it owns <strong>one output cell</strong>, exactly
        as everywhere else in this course, and it works out for itself which two inputs and which
        twiddle that cell needs. Nothing is ever swapped. Given the pass's half-block size
        <code>half</code>, thread <code>i</code> asks three questions:</p>
<pre><code>// am I the lower member of my pair, or the upper?
const j = Math.floor(i / half) % 2;
// where the LOWER member of my pair sits
const base = i - j * half;
// my position inside the half-block
const r = i % half;</code></pre>
        <p>Then <code>a</code> is element <code>base</code>, <code>b</code> is element
        <code>base + half</code>, and <code>w = e^(-iπ·r / half)</code>. Both members of a pair
        compute the same <code>w</code> and read the same two elements; they differ only in the sign
        between them. No communication, no barrier inside a pass — the same reason the halving
        ladder in Reductions needs none.</p>`,
      goal: `<strong>Goal:</strong> write one butterfly pass over the 8-point complex
        <code>spectrum</code> — the kernel takes <code>(spectrum, half)</code> and returns one
        number per output cell.`,
      requirements: [
        'Work out <code>base</code> and <code>r</code> from <code>this.thread.x</code> and <code>half</code> alone',
        'Read <em>both</em> planes of both elements — a complex multiply needs all four numbers',
        'Return <code>a + w·b</code> at the lower index and <code>a - w·b</code> at the upper one',
        'It must be right for <code>half = 1</code>, <code>2</code> and <code>4</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — which pair am I in?',
          body: `<p>At <code>half = 2</code> the eight threads split 0,1 | 2,3 | 4,5 | 6,7 into
            blocks of four: threads 0, 1, 4, 5 have <code>j = 0</code> and pair upward; threads
            2, 3, 6, 7 have <code>j = 1</code> and pair downward. Subtracting
            <code>j * half</code> lands both members of a pair on the same
            <code>base</code>.</p>`,
        },
        {
          title: 'Hint 2 — which twiddle?',
          body: `<p><code>r = i % half</code> is your position inside the half-block, and it is
            all <code>w</code> depends on. At <code>half = 1</code> every <code>r</code> is 0, so
            <code>w = 1</code> and the first pass is pure addition and subtraction — which is why
            a sign mistake in the twiddle will not show up until <code>half = 2</code>.</p>`,
        },
        {
          title: 'Hint 3 — the ending',
          body: `<p>Compute the whole complex result, then return only the half this thread's
            plane asked for:</p>
<pre><code>const tr = wr * br - wi * bi;
const ti = wr * bi + wi * br;
if (this.thread.y === 0) {
  if (j === 0) return ar + tr;
  return ar - tr;
}
if (j === 0) return ai + ti;
return ai - ti;</code></pre>`,
        },
      ],
      transfer: `The butterfly is gather-shaped on every platform, for the reason Thinking in
        Parallel gives: a thread writes its own cell and nobody else's. CUDA's cuFFT keeps a whole
        radix-N butterfly in registers and exchanges through shared memory or
        <code>__shfl_xor_sync</code>; WGSL and Metal compute shaders do the same through workgroup
        memory plus a barrier. What none of them do is have one thread write two results.`,
      starterCode: `// One pass. Eight complex numbers in, eight complex numbers out.
const gpu = new GPU({ mode });

const butterfly = gpu.createKernel(function (spectrum, half) {
  const i = this.thread.x;
  const j = Math.floor(i / half) % 2; // 0 = lower member of my pair, 1 = upper

  // TODO: which pair am I in, and where do I sit inside the half-block?
  const base = i;
  const r = 0;

  const angle = (-Math.PI * r) / half;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);

  const ar = spectrum[0][base];
  const ai = spectrum[1][base];
  const br = spectrum[0][base + half];
  const bi = spectrum[1][base + half];

  // TODO: multiply w by b, then return a + w·b at the lower index of the
  // pair and a - w·b at the upper one — this thread's plane only.
  if (this.thread.y === 0) return ar;
  return ai;
}, { output: [8, 2] });

const once = await butterfly(spectrum, 1);
console.log('after half = 1, real:', once[0]);
console.log('after half = 1, imag:', once[1]);
console.log('after half = 2, real:', (await butterfly(spectrum, 2))[0]);
`,
      solutionCode: `// One pass. Eight complex numbers in, eight complex numbers out.
const gpu = new GPU({ mode });

const butterfly = gpu.createKernel(function (spectrum, half) {
  const i = this.thread.x;
  const j = Math.floor(i / half) % 2; // 0 = lower member of my pair, 1 = upper

  const base = i - j * half;
  const r = i % half;

  const angle = (-Math.PI * r) / half;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);

  const ar = spectrum[0][base];
  const ai = spectrum[1][base];
  const br = spectrum[0][base + half];
  const bi = spectrum[1][base + half];

  const tr = wr * br - wi * bi;
  const ti = wr * bi + wi * br;

  if (this.thread.y === 0) {
    if (j === 0) return ar + tr;
    return ar - tr;
  }
  if (j === 0) return ai + ti;
  return ai - ti;
}, { output: [8, 2] });

const once = await butterfly(spectrum, 1);
console.log('after half = 1, real:', once[0]);
console.log('after half = 1, imag:', once[1]);
console.log('after half = 2, real:', (await butterfly(spectrum, 2))[0]);
`,
      inputs: () => ({ spectrum: EIGHT_COMPLEX.map(plane => plane.slice()) }),
      inputNotes: {
        spectrum:
          'Two planes, not two rows: spectrum[0][i] is the real part of element i and ' +
          'spectrum[1][i] its imaginary part. Inside the kernel this.thread.y is the plane and ' +
          'this.thread.x is the element.',
      },
      publicTests: [
        {
          name: 'the first pass, <code>half = 1</code> — adjacent pairs, <code>w = 1</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const [re, im] = EIGHT_COMPLEX;
            const out = await ctx.kernel([re, im], 1);
            ctx.assert(
              out && out.length === 2 && out[0].length === 8,
              'expected a [8, 2] result — 2 planes of 8 elements'
            );
            const [wantRe, wantIm] = passRef(re, im, 1);
            const hint = butterflyHint(
              8, (p, i) => out[p][i], (p, i) => (p === 0 ? wantRe[i] : wantIm[i]),
              1e-3, re, im, 1
            );
            for (let i = 0; i < 8; i++) {
              ctx.assertClose(out[0][i], wantRe[i], 1e-3, hint || `half = 1, real part of cell ${i}`);
              ctx.assertClose(out[1][i], wantIm[i], 1e-3, hint || `half = 1, imaginary part of cell ${i}`);
            }
          },
        },
        {
          name: 'the wider passes, <code>half = 2</code> and <code>half = 4</code> — where the twiddle bites',
          run: async ctx => {
            const [re, im] = EIGHT_COMPLEX;
            for (const half of [2, 4]) {
              const out = await ctx.kernel([re, im], half);
              const [wantRe, wantIm] = passRef(re, im, half);
              const hint = butterflyHint(
                8, (p, i) => out[p][i], (p, i) => (p === 0 ? wantRe[i] : wantIm[i]),
                1e-3, re, im, half
              );
              for (let i = 0; i < 8; i++) {
                ctx.assertClose(out[0][i], wantRe[i], 1e-3, hint || `half = ${half}, real part of cell ${i}`);
                ctx.assertClose(out[1][i], wantIm[i], 1e-3, hint || `half = ${half}, imaginary part of cell ${i}`);
              }
            }
          },
        },
        {
          name: 'a pass doubles the energy — exactly, and at every <code>half</code>',
          run: async ctx => {
            // |a + w·b|² + |a - w·b|² = 2(|a|² + |b|²), for any unit w. It is
            // one level of Parseval's theorem, it holds independently of every
            // index probe above, and a pass that lost the minus branch or
            // dropped a plane misses it by a mile.
            const [re, im] = EIGHT_COMPLEX;
            let before = 0;
            for (let i = 0; i < 8; i++) before += re[i] * re[i] + im[i] * im[i];
            for (const half of [1, 2, 4]) {
              const out = await ctx.kernel([re, im], half);
              let after = 0;
              for (let i = 0; i < 8; i++) after += out[0][i] * out[0][i] + out[1][i] * out[1][i];
              ctx.assertClose(
                after, 2 * before, 1e-2,
                `at half = ${half} the pass changed the total energy. One butterfly doubles it ` +
                'exactly — |a + w·b|² + |a - w·b|² = 2(|a|² + |b|²) — so a pass that loses or ' +
                'invents energy is not a butterfly'
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A fresh complex vector through all three passes. A kernel that is
            // right only where w = 1 does not survive this.
            const re = makeSamples(ctx.utils, 8, 1717);
            const im = makeSamples(ctx.utils, 8, 2828);
            for (const half of [1, 2, 4]) {
              const out = await ctx.kernel([re, im], half);
              const [wantRe, wantIm] = passRef(re, im, half);
              const hint = butterflyHint(
                8, (p, i) => out[p][i], (p, i) => (p === 0 ? wantRe[i] : wantIm[i]),
                1e-3, re, im, half
              );
              for (let i = 0; i < 8; i++) {
                ctx.assertClose(out[0][i], wantRe[i], 1e-3, hint || `half = ${half}, real cell ${i}`);
                ctx.assertClose(out[1][i], wantIm[i], 1e-3, hint || `half = ${half}, imaginary cell ${i}`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Three passes over a bit-reversed input ARE the 8-point DFT. This
            // is the whole module in miniature, three tasks early, and a pass
            // that is right only at one value of `half` cannot fake it.
            const signal = makeSamples(ctx.utils, 8, 5309);
            const scrambled = scrambleRef(signal, 3);
            let re = scrambled;
            let im = new Array(8).fill(0);
            for (const half of [1, 2, 4]) {
              const out = await ctx.kernel([re, im], half);
              re = Array.from(out[0]);
              im = Array.from(out[1]);
            }
            const [wantRe, wantIm] = dftRef(signal, new Array(8).fill(0), -1);
            for (let k = 0; k < 8; k++) {
              ctx.assertClose(
                re[k], wantRe[k], 1e-3,
                `three passes over the bit-reversed samples are the 8-point DFT — real part of ` +
                `bin ${k} does not match the definition`
              );
              ctx.assertClose(
                im[k], wantIm[k], 1e-3,
                `three passes over the bit-reversed samples are the 8-point DFT — imaginary part ` +
                `of bin ${k} does not match the definition`
              );
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'bit-reversal',
      title: 'Put the Input in the Right Order',
      intro: `<p>One thing about the last task was quietly wrong. The first pass paired
        <em>adjacent</em> elements — but the split that started all this was by parity, so the first
        pass should be combining <code>x[0]</code> with <code>x[4]</code>, <code>x[1]</code> with
        <code>x[5]</code>… at n = 8, and with strides that change every pass. Writing a kernel whose
        pair distance depends on the stage in a different way each time is possible and horrible.</p>
        <p>The trick every iterative FFT uses instead: <strong>permute the input once</strong>, so
        that the pairs are adjacent at every stage afterwards. The permutation turns out to be
        beautifully simple — send element <code>i</code> to the position whose index is <code>i</code>
        with its <strong>bits reversed</strong>. At n = 8, sample 1 (binary 001) goes to position 4
        (binary 100); sample 3 (011) goes to position 6 (110).</p>
        <p>gpu.js will happily give you <code>&gt;&gt;</code>, <code>&amp;</code> and
        <code>|</code>, and they produce correct answers — but each one compiles to a helper that
        loops over up to 32 bits in the shader, so the one-line spelling costs around eight hundred
        loop iterations per index. So do it in arithmetic instead, which is nine operations and, more
        to the point, says out loud what a bit reversal <em>is</em>: peel the low digit off one
        number and push it onto the front of another.</p>
<pre><code>let v = this.thread.x;
let reversed = 0;
for (let b = 0; b &lt; this.constants.bits; b++) {
  // push v's lowest bit onto the front of reversed
  reversed = reversed * 2 + (v % 2);
  // and drop it from v
  v = Math.floor(v / 2);
}</code></pre>
        <p>One happy accident is worth naming: reversing an index twice gives it back, so this
        permutation is its own inverse. It is the one rearrangement in the whole course where
        "push my value there" and "pull the value that belongs here" land on the same answer. Write
        it as the gather anyway — that habit is what makes the other ninety-nine cases work.</p>`,
      goal: `<strong>Goal:</strong> write the kernel that reads the 16 real samples of
        <code>signal</code> into bit-reversed order and presents them as a complex spectrum — real
        part permuted, imaginary part zero.`,
      requirements: [
        'Compute the reversal arithmetically, over <code>this.constants.bits</code> = 4 bits',
        'It is a <em>gather</em>: cell <code>i</code> returns <code>x[reversed]</code>',
        'Plane 1 (imaginary) is all zeros — the input signal is real',
      ],
      hints: [
        {
          title: 'Hint 1 — one digit at a time',
          body: `<p><code>v % 2</code> is the lowest bit of <code>v</code>;
            <code>Math.floor(v / 2)</code> throws that bit away. Multiplying
            <code>reversed</code> by 2 before adding shifts everything already collected one place
            up, so the first bit taken ends up highest.</p>`,
        },
        {
          title: 'Hint 2 — how many bits?',
          body: `<p>Exactly <code>log₂(n)</code> — 4 for 16 elements. Fewer and the top bits never
            move; more and every index lands outside the array. It is already in
            <code>this.constants.bits</code>.</p>`,
        },
        {
          title: 'Hint 3 — the whole thing',
          body: `<p>Check yourself against the first three: 0 → 0, 1 → 8, 2 → 4. Then:</p>
<pre><code>if (this.thread.y === 0) return x[reversed];
return 0;</code></pre>`,
        },
      ],
      transfer: `The bit-reversal permutation is a named, tuned primitive everywhere: cuFFT and
        rocFFT fold it into the first pass's addressing so the array is never physically permuted,
        FFTW spends real effort on cache-friendly reversal orders, and hardware FFT blocks in DSPs
        ship a bit-reversed addressing mode in the address generator itself. It is also why so many
        library APIs offer a "bit-reversed output" variant — if you are about to multiply two
        spectra together and transform back, neither of them ever needs to be in order.`,
      starterCode: `// Reverse the bits of your own index, then go and fetch that element.
const gpu = new GPU({ mode });

const scramble = gpu.createKernel(function (x) {
  let v = this.thread.x;
  let reversed = 0;
  for (let b = 0; b < this.constants.bits; b++) {
    // TODO: push v's low bit onto the front of \`reversed\` before dropping it.
    v = Math.floor(v / 2);
  }

  if (this.thread.y === 0) return x[reversed];
  return 0;
}, { output: [16, 2], constants: { bits: 4 } });

const ordered = await scramble(signal);
console.log('original: ', signal);
console.log('scrambled:', ordered[0]);
console.log('imaginary:', ordered[1]);
`,
      solutionCode: `// Reverse the bits of your own index, then go and fetch that element.
const gpu = new GPU({ mode });

const scramble = gpu.createKernel(function (x) {
  let v = this.thread.x;
  let reversed = 0;
  for (let b = 0; b < this.constants.bits; b++) {
    reversed = reversed * 2 + (v % 2);
    v = Math.floor(v / 2);
  }

  if (this.thread.y === 0) return x[reversed];
  return 0;
}, { output: [16, 2], constants: { bits: 4 } });

const ordered = await scramble(signal);
console.log('original: ', signal);
console.log('scrambled:', ordered[0]);
console.log('imaginary:', ordered[1]);
`,
      inputs: utils => ({ signal: makeSamples(utils, 16, 43751) }),
      publicTests: [
        {
          name: 'cell <code>i</code> holds the sample whose index is <code>i</code> reversed',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const x = makeSamples(ctx.utils, 16, 43751);
            const out = await ctx.kernel(x);
            ctx.assert(
              out && out.length === 2 && out[0].length === 16,
              'expected a [16, 2] result — 2 planes of 16 elements'
            );
            const want = scrambleRef(x, 4);
            const hint = diagnoseAll(16, i => out[0][i], i => want[i], 1e-3, scrambleProbes(x, 4));
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(out[0][i], want[i], 1e-3, hint || `cell ${i} should hold sample ${bitReverse(i, 4)}`);
            }
          },
        },
        {
          name: 'the imaginary plane is all zeros — a real signal has no imaginary part',
          run: async ctx => {
            const out = await ctx.kernel(makeSamples(ctx.utils, 16, 43751));
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(
                out[1][i], 0, 1e-6,
                `plane 1 of cell ${i} is not zero — the input is a real signal, so the imaginary ` +
                'plane starts empty and the transform fills it in'
              );
            }
          },
        },
        {
          name: 'nothing is lost — the 16 samples are the same 16, rearranged',
          run: async ctx => {
            const x = makeSamples(ctx.utils, 16, 1234);
            const out = await ctx.kernel(x);
            const before = Array.from(x).sort((a, b) => a - b);
            const after = Array.from(out[0]).sort((a, b) => a - b);
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(
                after[i], before[i], 1e-3,
                'a permutation rearranges values, it never drops or duplicates one — this result ' +
                'is not a permutation of the input, so some index was computed twice and another ' +
                'never at all'
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The permutation is its own inverse: applying it twice must give
            // back the original order. A reversal over the wrong width does not
            // have that property.
            const x = makeSamples(ctx.utils, 16, 60613);
            const once = await ctx.kernel(x);
            const twice = await ctx.kernel(Array.from(once[0]));
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(
                twice[0][i], x[i], 1e-3,
                'scrambling twice should give back the original order — reversing an index and ' +
                'then reversing it again is the identity. It is not here, so the reversal is ' +
                `over the wrong number of bits (cell ${i})`
              );
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Positions read straight off the index, with a signal whose value
            // IS its index: cell i must literally contain bitReverse(i).
            const x = new Array(16);
            for (let i = 0; i < 16; i++) x[i] = i;
            const out = await ctx.kernel(x);
            const hint = diagnoseAll(
              16, i => out[0][i], i => bitReverse(i, 4), 1e-3, scrambleProbes(x, 4)
            );
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(out[0][i], bitReverse(i, 4), 1e-3, hint || `cell ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'the-stage-loop',
      title: 'Drive log₂n Passes',
      intro: `<p>Everything is in place: a permutation to run once, and a butterfly pass to run
        log₂(n) times with the half-block size doubling — 1, 2, 4, … n/2. That is the same skeleton
        Reductions and Prefix Sums drive their ladders with, a plain JavaScript loop over a kernel
        that takes the stage as an argument, ping-ponging between buffers. The difference is only
        what comes out the far end.</p>
        <p>And as with a sorting network, the entire access pattern is fixed <em>before any data is
        seen</em>. Build the schedule first and print it:</p>
<pre><code>const schedule = [];
for (let half = 1; half &lt; n; half *= 2) schedule.push(half);</code></pre>
        <p>Eight numbers for n = 256, and not one of them could have been different for a different
        signal. Which pairs combine, in which pass, with which twiddle: all of it is a function of
        the index and the stage, so no thread ever waits on a value, no branch depends on data, and
        the whole transform is eight kernel launches with a barrier between them.</p>
        <p>The signal below is a sine at bin 5 plus a half-amplitude cosine at bin 12. A tone of
        amplitude A sitting exactly on bin b puts magnitude A·n/2 into bin b and its mirror bin
        n − b, and near enough nothing anywhere else — so a correct transform of this input has
        exactly four non-zero bins, and you know all four numbers in advance.</p>`,
      goal: `<strong>Goal:</strong> build and print the 8-pass schedule before touching the data,
        run scramble + 8 butterfly passes over the 256 samples, and log the magnitude of bins 5 and
        12.`,
      requirements: [
        '<code>schedule</code> holds the half-block sizes, doubling from <code>1</code> up to but not including <code>n</code>',
        'Build it without reading <code>signal</code> — the schedule is complete before the first kernel call',
        'Scramble once, then run one <code>butterfly</code> pass per entry, feeding each result into the next',
        '<code>console.log</code> the pass count, the schedule, and the magnitudes of bins 5 and 12',
      ],
      hints: [
        {
          title: 'Hint 1 — the loop bound',
          body: `<p><code>half &lt; n</code>, not <code>half &lt; n / 2</code>. The last pass, the
            one with <code>half = 128</code>, is the one that finally combines the even-sample
            spectrum with the odd-sample spectrum — stop before it and you are left holding
            precisely the two halves task 1 started with.</p>`,
        },
        {
          title: 'Hint 2 — ping-pong',
          body: `<p>Each pass consumes the previous result and produces a new one, so a single
            variable is all the bookkeeping needed:</p>
<pre><code>let buffer = await scramble(signal);
for (let s = 0; s &lt; schedule.length; s++) {
  buffer = await butterfly(buffer, schedule[s]);
}</code></pre>
<p>gpu.js locks an argument's type on a kernel's first call, and every pass hands
            back the same shape it took, so the chain is type-stable from the start.</p>`,
        },
        {
          title: 'Hint 3 — reading a magnitude',
          body: `<p>A bin is a complex number spread across the two planes, so its magnitude is
            <code>Math.hypot(buffer[0][k], buffer[1][k])</code>. For this signal bin 5 should come
            to 128 and bin 12 to 64.</p>`,
        },
      ],
      transfer: `One host-side loop issuing one launch per stage is exactly how an FFT ships:
        cuFFT's plan is a precomputed list of stages, WebGPU records one dispatch per pass into a
        command encoder, Metal encodes one compute pass each. The launches are the
        synchronisation — everything inside a pass is independent, and the boundary between passes
        is the only barrier anyone needs. It is also why FFT libraries make you build a "plan"
        before you hand over any data: the plan IS the schedule you just printed.`,
      starterCode: `// The schedule first, the signal second. One launch per stage.
const gpu = new GPU({ mode });
const n = 256;

const scramble = gpu.createKernel(function (x) {
  let v = this.thread.x;
  let reversed = 0;
  for (let b = 0; b < this.constants.bits; b++) {
    reversed = reversed * 2 + (v % 2);
    v = Math.floor(v / 2);
  }
  if (this.thread.y === 0) return x[reversed];
  return 0;
}, { output: [n, 2], constants: { bits: 8 } });

const butterfly = gpu.createKernel(function (spectrum, half) {
  const i = this.thread.x;
  const j = Math.floor(i / half) % 2;
  const base = i - j * half;
  const r = i % half;

  const angle = (-Math.PI * r) / half;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);

  const ar = spectrum[0][base];
  const ai = spectrum[1][base];
  const br = spectrum[0][base + half];
  const bi = spectrum[1][base + half];
  const tr = wr * br - wi * bi;
  const ti = wr * bi + wi * br;

  if (this.thread.y === 0) {
    if (j === 0) return ar + tr;
    return ar - tr;
  }
  if (j === 0) return ai + ti;
  return ai - ti;
}, { output: [n, 2] });

const schedule = [];
// TODO: fill \`schedule\` with the half-block size of every pass — doubling
// from 1 while it stays below n. Notice that nothing in here can look at
// \`signal\`. That is the point.

console.log('passes:', schedule.length);
console.log('schedule:', JSON.stringify(schedule));

// TODO: scramble once, then run one butterfly pass per schedule entry.
let buffer = await scramble(signal);

console.log('bin 5 magnitude: ', Math.hypot(buffer[0][5], buffer[1][5]));
console.log('bin 12 magnitude:', Math.hypot(buffer[0][12], buffer[1][12]));
`,
      solutionCode: `// The schedule first, the signal second. One launch per stage.
const gpu = new GPU({ mode });
const n = 256;

const scramble = gpu.createKernel(function (x) {
  let v = this.thread.x;
  let reversed = 0;
  for (let b = 0; b < this.constants.bits; b++) {
    reversed = reversed * 2 + (v % 2);
    v = Math.floor(v / 2);
  }
  if (this.thread.y === 0) return x[reversed];
  return 0;
}, { output: [n, 2], constants: { bits: 8 } });

const butterfly = gpu.createKernel(function (spectrum, half) {
  const i = this.thread.x;
  const j = Math.floor(i / half) % 2;
  const base = i - j * half;
  const r = i % half;

  const angle = (-Math.PI * r) / half;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);

  const ar = spectrum[0][base];
  const ai = spectrum[1][base];
  const br = spectrum[0][base + half];
  const bi = spectrum[1][base + half];
  const tr = wr * br - wi * bi;
  const ti = wr * bi + wi * br;

  if (this.thread.y === 0) {
    if (j === 0) return ar + tr;
    return ar - tr;
  }
  if (j === 0) return ai + ti;
  return ai - ti;
}, { output: [n, 2] });

const schedule = [];
for (let half = 1; half < n; half *= 2) {
  schedule.push(half);
}

console.log('passes:', schedule.length);
console.log('schedule:', JSON.stringify(schedule));

let buffer = await scramble(signal);
for (let s = 0; s < schedule.length; s++) {
  buffer = await butterfly(buffer, schedule[s]);
}

console.log('bin 5 magnitude: ', Math.hypot(buffer[0][5], buffer[1][5]));
console.log('bin 12 magnitude:', Math.hypot(buffer[0][12], buffer[1][12]));
`,
      inputs: () => ({ signal: makeTones(256, [[5, 1, 'sin'], [12, 0.5, 'cos']]) }),
      publicTests: [
        {
          name: 'the schedule is built and printed — <code>8</code> passes, doubling from 1',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            const right = [1, 2, 4, 8, 16, 32, 64, 128];
            const hasRun = pattern => {
              for (let i = 0; i + pattern.length <= nums.length; i++) {
                let hit = true;
                for (let j = 0; j < pattern.length; j++) {
                  if (nums[i + j] !== pattern[j]) { hit = false; break; }
                }
                if (hit) return true;
              }
              return false;
            };
            let hint = null;
            if (!hasRun(right)) {
              if (hasRun([1, 2, 4, 8, 16, 32, 64]) ) {
                hint =
                  'the schedule stops at 64 — the loop has to run while half < n, not while ' +
                  'half < n / 2. The pass at half = 128 is the one that combines the even-sample ' +
                  'spectrum with the odd-sample spectrum, and without it you have two half-length ' +
                  'transforms instead of one whole one';
              } else if (hasRun([128, 64, 32, 16, 8, 4, 2, 1])) {
                hint =
                  'the halves run the wrong way. This transform takes its input bit-reversed and ' +
                  'starts with ADJACENT pairs, reaching further every pass: 1, 2, 4, … 128';
              }
            }
            ctx.assert(
              hasRun(right),
              hint || 'log the whole schedule — expected its 8 half-block sizes in order, ' +
              '[1,2,4,8,16,32,64,128], built before the first kernel call'
            );
          },
        },
        {
          name: 'the transform of a fresh two-tone signal is correct in every bin',
          run: async ctx => {
            const { scramble, butterfly } = findFftKernels(ctx, 256);
            ctx.assert(scramble, 'no one-argument kernel with output [256, 2] found — the permutation');
            ctx.assert(butterfly, 'no two-argument kernel with output [256, 2] found — the butterfly pass');
            // Drive the learner's own two kernels through the whole schedule.
            const x = makeTones(256, [[9, 1, 'cos'], [40, 0.75, 'sin']]);
            let buffer = await scramble(x);
            for (let half = 1; half < 256; half *= 2) buffer = await butterfly(buffer, half);
            const zeros = new Array(256).fill(0);
            const [wantRe, wantIm] = dftRef(x, zeros, -1);
            const hint = diagnoseSpectrum(
              256, (p, i) => buffer[p][i], (p, i) => (p === 0 ? wantRe[i] : wantIm[i]),
              5e-3, scheduleProbes(x, 256)
            );
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(buffer[0][i], wantRe[i], 5e-3, hint || `real part of bin ${i}`);
              ctx.assertClose(buffer[1][i], wantIm[i], 5e-3, hint || `imaginary part of bin ${i}`);
            }
          },
        },
        {
          name: 'the magnitudes of bins 5 and 12 are logged — <code>128</code> and <code>64</code>',
          run: async ctx => {
            const zeros = new Array(256).fill(0);
            const [R, I] = fftRef(makeTones(256, [[5, 1, 'sin'], [12, 0.5, 'cos']]), zeros);
            const five = magnitudeAt(R, I, 5);
            const twelve = magnitudeAt(R, I, 12);
            const short = fftRef(makeTones(256, [[5, 1, 'sin'], [12, 0.5, 'cos']]), zeros, { oneStageShort: true });
            const halfHint =
              logged(ctx.logs, magnitudeAt(short[0], short[1], 5), 1e-2) && !logged(ctx.logs, five, 1e-2)
                ? 'that is exactly half the magnitude it should be, at the right bin — the signature ' +
                  'of a transform that ran one pass short. Seven passes leave two independent ' +
                  '128-point spectra side by side, each carrying half the amplitude; the eighth ' +
                  'pass is what merges them'
                : null;
            ctx.assert(
              logged(ctx.logs, five, 1e-2),
              halfHint || `log the magnitude of bin 5 — expected ≈${five.toFixed(2)} in the console output`
            );
            ctx.assert(
              logged(ctx.logs, twelve, 1e-2),
              `log the magnitude of bin 12 — expected ≈${twelve.toFixed(2)} in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // An impulse: every bin of its spectrum is exactly 1, and there is
            // nowhere for an ordering mistake to hide in a flat answer that is
            // wrong everywhere at once.
            const { scramble, butterfly } = findFftKernels(ctx, 256);
            ctx.assert(scramble && butterfly, 'expected a scramble kernel and a butterfly kernel');
            const impulse = new Array(256).fill(0);
            impulse[0] = 1;
            let buffer = await scramble(impulse);
            for (let half = 1; half < 256; half *= 2) buffer = await butterfly(buffer, half);
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(
                buffer[0][i], 1, 5e-3,
                `the transform of a unit impulse at t = 0 is 1 in every bin — bin ${i} is not`
              );
              ctx.assertClose(buffer[1][i], 0, 5e-3, `impulse spectrum, imaginary part of bin ${i}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A noisy signal with nothing structured about it: every bin is
            // some arbitrary complex number, so an accidental symmetry cannot
            // rescue a wrong schedule.
            const { scramble, butterfly } = findFftKernels(ctx, 256);
            ctx.assert(scramble && butterfly, 'expected a scramble kernel and a butterfly kernel');
            const x = makeSamples(ctx.utils, 256, 31415);
            let buffer = await scramble(x);
            for (let half = 1; half < 256; half *= 2) buffer = await butterfly(buffer, half);
            const [wantRe, wantIm] = dftRef(x, new Array(256).fill(0), -1);
            const hint = diagnoseSpectrum(
              256, (p, i) => buffer[p][i], (p, i) => (p === 0 ? wantRe[i] : wantIm[i]),
              5e-3, scheduleProbes(x, 256)
            );
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(buffer[0][i], wantRe[i], 5e-3, hint || `real part of bin ${i}`);
              ctx.assertClose(buffer[1][i], wantIm[i], 5e-3, hint || `imaginary part of bin ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'dft-versus-fft',
      title: 'Same Answer, Two Centuries Apart',
      // A payoff task is only worth running at a size where the payoff shows.
      // At n = 512 it did not: the FFT's ten kernel launches cost more than the
      // arithmetic they saved, and the better algorithm lost its own race. This
      // one runs at n = 8,192 — see WHY TASK 5 IS BIG at the top of this file —
      // and 20 s is what that size costs on the SLOWEST backend it has to work
      // on, not on the one it was written on. Measured, whole task (the run
      // plus its five tests):
      //
      //   M1 Max, gpu mode ........  0.9 s   (run 0.8 s, tests 0.1 s)
      //   cpu backend .............  2.2 s   (run 1.0 s, tests 1.2 s)
      //   SwiftShader, gpu mode ... 10.8 s   (run 5.1 s, tests 5.7 s)
      //
      // Software WebGL is where the number comes from: on any machine with no
      // GPU one naive transform of 8,192 samples is ~1.4 s, and this task does
      // several. That is legitimately slow, not a runaway, and it leaves no
      // useful headroom under the 10 s run / 15 s test / 30 s benchmark
      // defaults in engine/runner.js — the ⏱ Benchmark chip alone is two full
      // runs plus two adaptive timing loops. 20 s doubles (runner.js's
      // WATCHDOG_HEADROOM) to a 40 s watchdog on all three, which is ~4× the
      // slowest measurement above.
      //
      // NOT the reason, and worth writing down so nobody assumes it is: the
      // pre-flight guard in engine/sandbox.js never fires here. It only judges
      // runs above PROBE_MIN_THREADS = 65,536 threads, and an [8192, 2] kernel
      // asks for 16,384 — verified by running this task at budgetMs = 1000 in
      // both modes and watching it go through untouched.
      budgetMs: 20000,
      intro: `<p>A permutation and a stack of index arithmetic produced <em>something</em>. The
        only way to know it is the Fourier transform is to compute the Fourier transform the way
        the definition says and compare, bin by bin. That is this task: write the naive DFT, run
        both at n = 8,192 — thirteen passes this time, since log₂(8,192) = 13 — and check they
        agree.</p>
        <p>Then count. The DFT visits every sample for every bin: n² = <strong>67,108,864</strong>
        complex multiplies. The FFT does log₂(n) passes of n/2 butterflies:
        <strong>53,248</strong>. A factor of 1,260 — and it is not a constant, it is a ratio that
        grows without bound. At n = 65,536, a window of about 1.5 seconds of CD audio, it is
        <strong>8,192×</strong>, which is the difference between a spectrogram that renders live
        and one that does not render at all. Gauss found the trick in 1805 and left it
        unpublished; Cooley and Tukey published it in 1965 and it went on to underwrite digital
        audio, JPEG's cousin the DCT, radio astronomy, MRI, and the modem you are reading this
        through.</p>
        <p>Now the stopwatch, and the reason this task is the size it is. <strong>Kernel launches
        are not free.</strong> A launch plus the readback around it costs about half a millisecond
        here, and the FFT needs fourteen of them where the definition needs one. At n = 512 that
        overhead <em>is</em> the transform, and the algorithm doing 114× less arithmetic loses the
        race outright. That is a real and permanent GPU lesson — a small transform is not worth
        sending to a GPU at all — but it is no longer this task's punchline, because the fix for
        it is the one change the code below makes to the FFT you built in task 4: every pass hands
        the next one a <em>texture</em> (<code>pipeline: true</code>) instead of a JavaScript
        array, so the spectrum crosses back to the CPU once instead of fourteen times. Measured,
        that single change takes the FFT from 8.2 ms to 1.5 ms.</p>
        <p>With both things settled — enough arithmetic that it outweighs the launches, and no
        copies that buy nothing — the race is not close. On the machine these notes were written
        on the FFT comes in somewhere between <strong>1 and 2 ms</strong>, and the definition takes
        <strong>three to five times as long</strong> — nearer the top of that range on WebGPU,
        which <strong>Auto</strong> picks, than on WebGL, because a thirteen-launch ladder gains
        less from the newer backend than one big arithmetic sweep does.
        Run it and read your own two numbers off the console. Then switch <strong>Mode</strong>
        from Auto to CPU, where there is no launch overhead left and nothing but the operation
        count decides it, and watch the gap open past <strong>35×</strong> — against a naive
        transform that has been handed only a 1,024-bin slice of the spectrum instead of all
        8,192, because the whole thing is two seconds of a single thread. A machine with no GPU
        at all, running WebGL in software, reports about 135×.</p>
        <p>One last thing in the console, which is not about speed. The two answers agree to about
        six parts in ten thousand, and nearly all of that gap is the <em>naive</em> one being
        wrong: 8,192 float32 terms summed into one running total, against thirteen roundings for
        the FFT. The fast algorithm is also the accurate one, and that is not a coincidence.
        Measuring Speed Honestly is the module that owns the benchmarking argument; the ⏱
        <strong>Benchmark</strong> chip answers a different question again, timing the whole file
        at once rather than either transform alone.</p>`,
      goal: `<strong>Goal:</strong> write the naive DFT kernel, confirm it agrees with the FFT
        bin by bin, and log the two operation counts and both timings.`,
      requirements: [
        'The DFT kernel is one thread per output cell, looping over all <code>this.constants.n</code> samples',
        'The output is <code>bins</code> wide but every bin sums over all <code>n</code> samples — the output width and the loop bound are different numbers',
        'Angles are <code>-2π·k·t / n</code> with <code>k = this.thread.x</code> — no scaling by <code>n</code>',
        'Compare the two spectra bin by bin and <code>console.log</code> the verdict',
        '<code>console.log</code> both operation counts and both elapsed times (already wired up)',
      ],
      hints: [
        {
          title: 'Hint 1 — which index is the loop, which is the thread?',
          body: `<p><code>this.thread.x</code> is the bin <code>k</code>, fixed for the whole
            thread. The loop variable <code>t</code> walks the samples. So the array lookup is
            <code>x[t]</code> — if <code>this.thread.x</code> appears inside the brackets, the
            two have been swapped.</p>`,
        },
        {
          title: 'Hint 2 — the accumulation',
          body: `<pre><code>const angle = (-2 * Math.PI * k * t) / this.constants.n;
re += x[t] * Math.cos(angle);
im += x[t] * Math.sin(angle);</code></pre>
<p>— the input is real, so there is no fourth term. Task 6 needs the general
            version.</p>`,
        },
        {
          title: 'Hint 3 — what "agree" should mean',
          body: `<p>Not equality. The two computations do wildly different numbers of float32
            operations and will not land on the same bits — the DFT accumulates 8,192 terms into
            one running total, the FFT thirteen. Compare the largest difference against the size
            of the spectrum, not against zero: <code>worst / peak &lt; 5e-3</code> is a generous
            margin and a truthful one. Expect about <code>6e-4</code>, and expect nearly all of it
            to belong to the DFT: its phase <code>-2π·k·t / n</code> runs <code>k·t</code> up past
            67 million, which float32 cannot even hold to the nearest whole number.</p>`,
        },
      ],
      transfer: `Checking a fast implementation against the slow definition on a small input is
        how every FFT library is tested — cuFFT, FFTW and rocFFT all ship exactly this comparison
        in their accuracy suites, usually reported as relative error against a reference computed
        at higher precision. It is also the honest answer to "is my GPU port correct": not "the
        pictures look the same", but the definition, at a size where you can afford it. The
        second habit here travels just as far: when a chain of kernels loses to a brute-force
        one, count the round trips before you touch the arithmetic. Keeping intermediates on the
        device is what <code>pipeline: true</code> is for in gpu.js, what CUDA graphs and Metal
        command buffers are for elsewhere, and it is very often the whole difference.`,
      starterCode: `// The definition versus the algorithm. Same answer, different bill.
const gpu = new GPU({ mode });
const n = 8192;                 // log2(8192) = 13 butterfly passes

// The naive transform reads the WHOLE signal once for every bin it produces, so
// a full spectrum is n * n work. Every GPU backend eats that; one CPU thread
// needs about two seconds of it. So the CPU backend — and only it — is asked for
// a 1,024-bin slice instead: an 8x head start, and it loses anyway. The question
// is "am I the slow single-threaded backend?", so the test is mode === 'cpu'.
const bins = mode === 'cpu' ? 1024 : n;

// ---- the fast one, complete (tasks 3 and 4), with ONE change
// Every pass hands the next one a TEXTURE instead of a JavaScript array. Copying
// the spectrum off the card after each of thirteen passes costs far more than the
// passes do, and that — not the arithmetic — is why a small FFT loses this race.
// Two butterfly kernels take turns because a pipelined kernel writes into its own
// texture and so cannot also be the one reading it.
const stage = { output: [n, 2], pipeline: true };

const scramble = gpu.createKernel(function (x) {
  let v = this.thread.x;
  let reversed = 0;
  for (let b = 0; b < this.constants.bits; b++) {
    reversed = reversed * 2 + (v % 2);
    v = Math.floor(v / 2);
  }
  if (this.thread.y === 0) return x[reversed];
  return 0;
}, { ...stage, constants: { bits: 13 } });

const onePass = function (spectrum, half) {
  const i = this.thread.x;
  const j = Math.floor(i / half) % 2;
  const base = i - j * half;
  const r = i % half;
  const angle = (-Math.PI * r) / half;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);
  const ar = spectrum[0][base];
  const ai = spectrum[1][base];
  const br = spectrum[0][base + half];
  const bi = spectrum[1][base + half];
  const tr = wr * br - wi * bi;
  const ti = wr * bi + wi * br;
  if (this.thread.y === 0) {
    if (j === 0) return ar + tr;
    return ar - tr;
  }
  if (j === 0) return ai + ti;
  return ai - ti;
};

const evenPass = gpu.createKernel(onePass, stage);
const oddPass = gpu.createKernel(onePass, stage);

// Every stage is awaited IN ORDER. Each pass reads the texture the previous one
// wrote, so the ladder is strictly sequential — firing all thirteen and awaiting
// them together would hand a pass a half-finished input.
async function fft(samples) {
  let buffer = await scramble(samples);
  let pass = 0;
  for (let half = 1; half < n; half *= 2) {
    buffer = await (pass++ % 2 === 0 ? evenPass : oddPass)(buffer, half);
  }
  // one read back, at the very end. The CPU backend has no textures and has
  // handed back a plain array already, so there is nothing to convert there.
  return buffer.toArray ? await buffer.toArray() : buffer;
}

// ---- the slow one
const dft = gpu.createKernel(function (x) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.n; t++) {
    // TODO: accumulate x[t] · e^(-2πi·k·t / n) into re and im.
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [bins, 2], constants: { n } });

// Warm both up before timing: the first call compiles a shader, and a GPU handed
// its first work of the day is still spinning its clocks up.
await fft(signal);
await dft(signal);
await fft(signal);
await dft(signal);

// Each timed call is AWAITED inside its own brackets. A kernel call returns a
// promise, so timing an unawaited one would measure how long it took to queue
// the work rather than how long the work took.
const t0 = performance.now();
const fast = await fft(signal);
const t1 = performance.now();
const slow = await dft(signal);
const t2 = performance.now();

let worst = 0;
let peak = 0;
for (let k = 0; k < bins; k++) {
  worst = Math.max(worst, Math.abs(fast[0][k] - slow[0][k]), Math.abs(fast[1][k] - slow[1][k]));
  peak = Math.max(peak, Math.hypot(slow[0][k], slow[1][k]));
}

console.log('bins the definition was asked for:', bins, 'of', n);
console.log('DFT complex multiplies:', bins * n);
console.log('FFT complex multiplies:', (n / 2) * Math.log2(n));
console.log('worst bin difference:', worst, 'against a peak of', peak);
console.log('agree:', worst / peak < 5e-3);
console.log('fft ms:', (t1 - t0).toFixed(3));
console.log('dft ms:', (t2 - t1).toFixed(3));
console.log('the definition took', ((t2 - t1) / (t1 - t0)).toFixed(1), 'times as long');
`,
      solutionCode: `// The definition versus the algorithm. Same answer, different bill.
const gpu = new GPU({ mode });
const n = 8192;                 // log2(8192) = 13 butterfly passes

// The naive transform reads the WHOLE signal once for every bin it produces, so
// a full spectrum is n * n work. Every GPU backend eats that; one CPU thread
// needs about two seconds of it. So the CPU backend — and only it — is asked for
// a 1,024-bin slice instead: an 8x head start, and it loses anyway. The question
// is "am I the slow single-threaded backend?", so the test is mode === 'cpu'.
const bins = mode === 'cpu' ? 1024 : n;

// ---- the fast one, complete (tasks 3 and 4), with ONE change
// Every pass hands the next one a TEXTURE instead of a JavaScript array. Copying
// the spectrum off the card after each of thirteen passes costs far more than the
// passes do, and that — not the arithmetic — is why a small FFT loses this race.
// Two butterfly kernels take turns because a pipelined kernel writes into its own
// texture and so cannot also be the one reading it.
const stage = { output: [n, 2], pipeline: true };

const scramble = gpu.createKernel(function (x) {
  let v = this.thread.x;
  let reversed = 0;
  for (let b = 0; b < this.constants.bits; b++) {
    reversed = reversed * 2 + (v % 2);
    v = Math.floor(v / 2);
  }
  if (this.thread.y === 0) return x[reversed];
  return 0;
}, { ...stage, constants: { bits: 13 } });

const onePass = function (spectrum, half) {
  const i = this.thread.x;
  const j = Math.floor(i / half) % 2;
  const base = i - j * half;
  const r = i % half;
  const angle = (-Math.PI * r) / half;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);
  const ar = spectrum[0][base];
  const ai = spectrum[1][base];
  const br = spectrum[0][base + half];
  const bi = spectrum[1][base + half];
  const tr = wr * br - wi * bi;
  const ti = wr * bi + wi * br;
  if (this.thread.y === 0) {
    if (j === 0) return ar + tr;
    return ar - tr;
  }
  if (j === 0) return ai + ti;
  return ai - ti;
};

const evenPass = gpu.createKernel(onePass, stage);
const oddPass = gpu.createKernel(onePass, stage);

// Every stage is awaited IN ORDER. Each pass reads the texture the previous one
// wrote, so the ladder is strictly sequential — firing all thirteen and awaiting
// them together would hand a pass a half-finished input.
async function fft(samples) {
  let buffer = await scramble(samples);
  let pass = 0;
  for (let half = 1; half < n; half *= 2) {
    buffer = await (pass++ % 2 === 0 ? evenPass : oddPass)(buffer, half);
  }
  // one read back, at the very end. The CPU backend has no textures and has
  // handed back a plain array already, so there is nothing to convert there.
  return buffer.toArray ? await buffer.toArray() : buffer;
}

// ---- the slow one
const dft = gpu.createKernel(function (x) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.n; t++) {
    const angle = (-2 * Math.PI * k * t) / this.constants.n;
    re += x[t] * Math.cos(angle);
    im += x[t] * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [bins, 2], constants: { n } });

// Warm both up before timing: the first call compiles a shader, and a GPU handed
// its first work of the day is still spinning its clocks up.
await fft(signal);
await dft(signal);
await fft(signal);
await dft(signal);

// Each timed call is AWAITED inside its own brackets. A kernel call returns a
// promise, so timing an unawaited one would measure how long it took to queue
// the work rather than how long the work took.
const t0 = performance.now();
const fast = await fft(signal);
const t1 = performance.now();
const slow = await dft(signal);
const t2 = performance.now();

let worst = 0;
let peak = 0;
for (let k = 0; k < bins; k++) {
  worst = Math.max(worst, Math.abs(fast[0][k] - slow[0][k]), Math.abs(fast[1][k] - slow[1][k]));
  peak = Math.max(peak, Math.hypot(slow[0][k], slow[1][k]));
}

console.log('bins the definition was asked for:', bins, 'of', n);
console.log('DFT complex multiplies:', bins * n);
console.log('FFT complex multiplies:', (n / 2) * Math.log2(n));
console.log('worst bin difference:', worst, 'against a peak of', peak);
console.log('agree:', worst / peak < 5e-3);
console.log('fft ms:', (t1 - t0).toFixed(3));
console.log('dft ms:', (t2 - t1).toFixed(3));
console.log('the definition took', ((t2 - t1) / (t1 - t0)).toFixed(1), 'times as long');
`,
      inputs: () => ({ signal: makeTones(NAIVE_N, [[7, 1, 'sin'], [29, 0.5, 'cos']]) }),
      publicTests: [
        {
          name: 'the DFT kernel computes the transform from the definition',
          run: async ctx => {
            ctx.assert(
              ctx.kernels.length >= 4,
              'expected four kernels — the permutation, two butterfly passes and the naive DFT'
            );
            const dft = await findNaiveDft(ctx);
            ctx.assert(
              dft,
              'no kernel behaved like a transform of a real signal — handed a unit impulse at ' +
              't = 0, a forward transform returns 1 in every bin'
            );
            const bins = binsOf(dft);
            const x = makeTones(NAIVE_N, [[7, 1, 'sin'], [29, 0.5, 'cos']]);
            const out = await dft(x);
            const list = checkBins(bins);
            const [wantRe, wantIm] = dftRefAt(x, list);
            const eps = 0.03 * peakOf(wantRe, wantIm);
            const hint = diagnoseSpectrum(
              list.length,
              (p, q) => out[p][list[q]],
              (p, q) => (p === 0 ? wantRe[q] : wantIm[q]),
              eps, naiveProbes(x, list, wantRe, wantIm, eps)
            );
            for (let q = 0; q < list.length; q++) {
              ctx.assertClose(out[0][list[q]], wantRe[q], eps, hint || `real part of bin ${list[q]}`);
              ctx.assertClose(out[1][list[q]], wantIm[q], eps, hint || `imaginary part of bin ${list[q]}`);
            }
          },
        },
        {
          name: 'the two transforms are compared and reported as agreeing',
          run: async ctx => {
            ctx.assert(
              loggedText(ctx.logs, 'agree: true'),
              'the FFT and the DFT do not agree yet — log the verdict, and if it says false the ' +
              'DFT kernel is not yet computing the same thing the thirteen passes are'
            );
          },
        },
        {
          name: 'both operation counts and both timings are logged',
          run: async ctx => {
            const dft = await findNaiveDft(ctx);
            const bins = dft ? binsOf(dft) : NAIVE_N;
            ctx.assert(
              logged(ctx.logs, bins * NAIVE_N, 0.5),
              `log the naive transform's cost — ${bins.toLocaleString('en-US')} bins × ` +
              `${NAIVE_N.toLocaleString('en-US')} samples = ` +
              `${(bins * NAIVE_N).toLocaleString('en-US')} complex multiplies`
            );
            ctx.assert(
              logged(ctx.logs, (NAIVE_N / 2) * 13, 0.5),
              'log the FFT\'s cost — 13 passes × 4,096 butterflies = 53,248 complex multiplies'
            );
            const timings = ctx.logs.filter(
              line => line.type === 'log' && line.text && /\bms\b/.test(line.text)
            );
            ctx.assert(
              timings.length >= 2,
              'log both elapsed times — one line for the FFT and one for the DFT'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A signal with no clean structure at all, so a spectrum that is
            // right only where symmetry helps does not survive.
            const dft = await findNaiveDft(ctx);
            ctx.assert(dft, 'no naive DFT kernel found');
            const bins = binsOf(dft);
            const x = makeSamples(ctx.utils, NAIVE_N, 26535);
            const out = await dft(x);
            const list = checkBins(bins);
            const [wantRe, wantIm] = dftRefAt(x, list);
            const eps = 0.03 * peakOf(wantRe, wantIm);
            const hint = diagnoseSpectrum(
              list.length,
              (p, q) => out[p][list[q]],
              (p, q) => (p === 0 ? wantRe[q] : wantIm[q]),
              eps, naiveProbes(x, list, wantRe, wantIm, eps)
            );
            for (let q = 0; q < list.length; q++) {
              ctx.assertClose(out[0][list[q]], wantRe[q], eps, hint || `real part of bin ${list[q]}`);
              ctx.assertClose(out[1][list[q]], wantIm[q], eps, hint || `imaginary part of bin ${list[q]}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Properties of a forward transform of a REAL signal that a
            // magnitude plot cannot see AND that survive being asked for only
            // the bottom of the spectrum — which is what the cpu backend gets,
            // so conjugate symmetry about bin n − k is not available here.
            const dft = await findNaiveDft(ctx);
            ctx.assert(dft, 'no naive DFT kernel found');
            const n = NAIVE_N;
            // 0.7 of DC on top of the tones, so bin 0 is 5,734.4 rather than 0 —
            // a spectrum quietly divided by n would pass a bin-0 check against
            // zero and fails this one by a factor of 8,192.
            const x = makeTones(n, [[3, 1, 'sin'], [11, 0.4, 'cos'], [60, 0.8, 'sin']])
              .map(v => v + 0.7);
            const out = await dft(x);
            // 1% of one tone's magnitude: 220× the worst float32 error this
            // kernel was measured at, and 140× smaller than the smallest gap any
            // assertion below has to see across.
            const eps = n / 200;
            let sum = 0;
            for (let t = 0; t < n; t++) sum += x[t];
            ctx.assertClose(
              out[0][0], sum, eps,
              'bin 0 of a forward transform is the plain sum of the samples, with no scaling — ' +
              `expected ≈${sum.toFixed(2)}`
            );
            // A sine of amplitude A sitting on whole bin b puts −A·n/2 in the
            // IMAGINARY part of bin b and nothing in the real part; a cosine
            // does the opposite, with a PLUS. Only the sign of the exponent
            // decides which way the imaginary parts go, and no magnitude plot
            // would ever show it.
            const sine = 'a sine of amplitude A on whole bin b puts −A·n/2 in the IMAGINARY part ' +
              'of bin b — a plus there is e^(+2πi·kt/n), the inverse transform';
            ctx.assertClose(out[1][3], -n / 2, eps, sine);
            ctx.assertClose(out[1][60], -0.8 * (n / 2), eps, sine);
            ctx.assertClose(
              out[0][11], 0.4 * (n / 2), eps,
              'a cosine of amplitude A on whole bin b puts +A·n/2 in the REAL part of bin b — ' +
              `bin 11 should be ≈${(0.4 * (n / 2)).toFixed(1)}`
            );
            ctx.assertClose(out[0][3], 0, eps, 'the real part of bin 3 — a pure sine has none');
            ctx.assertClose(out[1][11], 0, eps, 'the imaginary part of bin 11 — a pure cosine has none');
            ctx.assertClose(
              out[0][25], 0, eps,
              'bin 25 carries no tone in this signal, so both of its parts are zero — energy ' +
              'showing up there means the bins are not lining up with the samples'
            );
            ctx.assertClose(out[1][25], 0, eps, 'the imaginary part of empty bin 25');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'inverse-by-conjugation',
      title: 'Run It Backwards for Free',
      intro: `<p>The inverse transform differs from the forward one in two small ways: the exponent
        is <code>+2πi·kt/n</code> instead of <code>-</code>, and the result is divided by n. You
        could write a second set of kernels. You do not have to — there is an identity that gets the
        inverse out of the forward transform you already have:</p>
<pre><code>ifft(X) = conj( fft( conj(X) ) ) / n</code></pre>
        <p>Conjugating flips the sign of every imaginary part, which is exactly the sign flip the
        exponent needs; doing it on the way in and again on the way out leaves everything else where
        it was. Nine lines of kernel and a division, and the transform runs both ways.</p>
        <p>And then the warning, which is the real content of this task. A round trip is a
        <strong>consistency</strong> check, not a correctness check — it proves your inverse undoes
        your forward transform, and nothing else. Flip the twiddle sign in the butterfly and the
        forward transform computes the conjugate of the right spectrum; run it through this same
        inverse and the signal comes back <em>perfectly</em>, to the last bit. The magnitudes were
        already identical, the round trip is clean, and every plot looks right. The only thing that
        catches it is the comparison you did in the last task: against the definition.</p>`,
      goal: `<strong>Goal:</strong> write the conjugate-and-scale kernel, use it at both ends of a
        forward transform to invert one, and confirm the 256 samples come back.`,
      requirements: [
        'One kernel <code>conjugate(spectrum, scale)</code>, used at both ends — it negates the imaginary plane and multiplies both planes by <code>scale</code>',
        'Going in the scale is <code>1</code>; coming out it is <code>1 / n</code>',
        'The recovered imaginary plane is ~0 — a real signal in, a real signal out',
        '<code>console.log</code> the largest round-trip error and the verdict',
      ],
      hints: [
        {
          title: 'Hint 1 — conjugation, both planes at once',
          body: `<p>The kernel owns one output cell, and which plane it is decides the sign:</p>
<pre><code>if (this.thread.y === 0) return spectrum[0][this.thread.x] * scale;
return -spectrum[1][this.thread.x] * scale;</code></pre>
<p>The real part keeps its sign. Only the imaginary part turns over.</p>`,
        },
        {
          title: 'Hint 2 — the whole inverse, one line',
          body: `<pre><code>const recovered = await conjugate(await fft(await conjugate(spectrum, 1)), 1 / n);</code></pre>
<p>Inside out: conjugate, transform, conjugate again and scale. The
            <code>1</code> on the way in is not decoration — the same kernel has to take a scale
            both times or gpu.js sees two different call shapes.</p>`,
        },
        {
          title: 'Hint 3 — where the n goes',
          body: `<p>All of it, on the inverse. A forward transform followed by an unscaled inverse
            gives you the signal multiplied by n — so if every recovered sample is 256 times too
            big, the division is missing rather than misplaced.</p>`,
        },
      ],
      transfer: `The conjugation trick is standard practice, not a curiosity: it is why cuFFT,
        rocFFT and FFTW all describe their inverse as "unnormalized" and leave the 1/n to you —
        the same kernels run both directions, and a library that scaled automatically would make
        the common case (transform, filter, transform back, and the scales cancel) pay for
        something nobody asked for. Where the 1/n lives, or whether it is split as 1/√n between the
        two directions, is a convention you have to read off each library's documentation rather
        than assume.`,
      starterCode: `// The forward transform, run backwards, using nothing new.
const gpu = new GPU({ mode });
const n = 256;

// ---- the forward transform, complete — now taking a COMPLEX input
const scramble = gpu.createKernel(function (spectrum) {
  let v = this.thread.x;
  let reversed = 0;
  for (let b = 0; b < this.constants.bits; b++) {
    reversed = reversed * 2 + (v % 2);
    v = Math.floor(v / 2);
  }
  return spectrum[this.thread.y][reversed];
}, { output: [n, 2], constants: { bits: 8 } });

const butterfly = gpu.createKernel(function (spectrum, half) {
  const i = this.thread.x;
  const j = Math.floor(i / half) % 2;
  const base = i - j * half;
  const r = i % half;
  const angle = (-Math.PI * r) / half;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);
  const ar = spectrum[0][base];
  const ai = spectrum[1][base];
  const br = spectrum[0][base + half];
  const bi = spectrum[1][base + half];
  const tr = wr * br - wi * bi;
  const ti = wr * bi + wi * br;
  if (this.thread.y === 0) {
    if (j === 0) return ar + tr;
    return ar - tr;
  }
  if (j === 0) return ai + ti;
  return ai - ti;
}, { output: [n, 2] });

async function fft(planes) {
  let buffer = await scramble(planes);
  // one pass at a time: each reads what the last one wrote
  for (let half = 1; half < n; half *= 2) buffer = await butterfly(buffer, half);
  return buffer;
}

const conjugate = gpu.createKernel(function (spectrum, scale) {
  // TODO: negate the IMAGINARY plane, leave the real plane alone, and
  // multiply whichever plane this thread owns by \`scale\`.
  return spectrum[this.thread.y][this.thread.x];
}, { output: [n, 2] });

// A real signal becomes two planes: the samples, and an empty imaginary plane.
const planes = [Float32Array.from(signal), new Float32Array(n)];
const spectrum = await fft(planes);
console.log('bin 24 magnitude:', Math.hypot(spectrum[0][24], spectrum[1][24]));

// TODO: conjugate, transform, conjugate and scale by 1 / n.
const recovered = spectrum;

let worst = 0;
let leftover = 0;
for (let i = 0; i < n; i++) {
  worst = Math.max(worst, Math.abs(recovered[0][i] - signal[i]));
  leftover = Math.max(leftover, Math.abs(recovered[1][i]));
}

console.log('worst round-trip error:', worst);
console.log('largest imaginary leftover:', leftover);
console.log('round trip clean:', worst < 5e-3 && leftover < 5e-3);
`,
      solutionCode: `// The forward transform, run backwards, using nothing new.
const gpu = new GPU({ mode });
const n = 256;

// ---- the forward transform, complete — now taking a COMPLEX input
const scramble = gpu.createKernel(function (spectrum) {
  let v = this.thread.x;
  let reversed = 0;
  for (let b = 0; b < this.constants.bits; b++) {
    reversed = reversed * 2 + (v % 2);
    v = Math.floor(v / 2);
  }
  return spectrum[this.thread.y][reversed];
}, { output: [n, 2], constants: { bits: 8 } });

const butterfly = gpu.createKernel(function (spectrum, half) {
  const i = this.thread.x;
  const j = Math.floor(i / half) % 2;
  const base = i - j * half;
  const r = i % half;
  const angle = (-Math.PI * r) / half;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);
  const ar = spectrum[0][base];
  const ai = spectrum[1][base];
  const br = spectrum[0][base + half];
  const bi = spectrum[1][base + half];
  const tr = wr * br - wi * bi;
  const ti = wr * bi + wi * br;
  if (this.thread.y === 0) {
    if (j === 0) return ar + tr;
    return ar - tr;
  }
  if (j === 0) return ai + ti;
  return ai - ti;
}, { output: [n, 2] });

async function fft(planes) {
  let buffer = await scramble(planes);
  // one pass at a time: each reads what the last one wrote
  for (let half = 1; half < n; half *= 2) buffer = await butterfly(buffer, half);
  return buffer;
}

const conjugate = gpu.createKernel(function (spectrum, scale) {
  if (this.thread.y === 0) return spectrum[0][this.thread.x] * scale;
  return -spectrum[1][this.thread.x] * scale;
}, { output: [n, 2] });

// A real signal becomes two planes: the samples, and an empty imaginary plane.
const planes = [Float32Array.from(signal), new Float32Array(n)];
const spectrum = await fft(planes);
console.log('bin 24 magnitude:', Math.hypot(spectrum[0][24], spectrum[1][24]));

const recovered = await conjugate(await fft(await conjugate(spectrum, 1)), 1 / n);

let worst = 0;
let leftover = 0;
for (let i = 0; i < n; i++) {
  worst = Math.max(worst, Math.abs(recovered[0][i] - signal[i]));
  leftover = Math.max(leftover, Math.abs(recovered[1][i]));
}

console.log('worst round-trip error:', worst);
console.log('largest imaginary leftover:', leftover);
console.log('round trip clean:', worst < 5e-3 && leftover < 5e-3);
`,
      inputs: () => ({ signal: makePluck(256) }),
      publicTests: [
        {
          name: '<code>conjugate</code> negates the imaginary plane and scales both',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 3, 'expected three kernels — scramble, butterfly and conjugate');
            const re = makeSamples(ctx.utils, 256, 8080);
            const im = makeSamples(ctx.utils, 256, 9090);
            const planes = [Float32Array.from(re), Float32Array.from(im)];
            const { conjugate: conj, behaviour } = await findInverseKernels(ctx, 256, planes);
            ctx.assert(
              conj,
              conjugateHint(behaviour) ||
              'no kernel conjugated its input — with scale = 1 the real plane must come back ' +
              'unchanged and the imaginary plane negated'
            );
            const scaled = await conj(planes, 0.25);
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(
                scaled[0][i], re[i] * 0.25, 5e-3,
                `the scale is not reaching the real plane (cell ${i})`
              );
              ctx.assertClose(
                scaled[1][i], -im[i] * 0.25, 5e-3,
                `the scale is not reaching the imaginary plane (cell ${i})`
              );
            }
          },
        },
        {
          name: 'the round trip recovers all 256 samples',
          run: async ctx => {
            // The scale mistakes announce themselves in the logged error: with
            // no 1/n at all every sample is n times too big, so the largest
            // error is (n - 1)·max|signal|, and with 1/sqrt(n) it is
            // (sqrt(n) - 1)·max|signal|. Those two differ by a factor of 17, so
            // matching one within 5% cannot be matching the other.
            const amplitude = Math.max(...makePluck(256).map(Math.abs));
            const nums = loggedNumbers(ctx.logs);
            const scaleHint = (() => {
              const probes = [
                [amplitude * 255,
                  'every recovered sample is n times too big — a forward transform followed by ' +
                  'an unscaled inverse multiplies the signal by n, so the whole 1/n is missing'],
                [amplitude * (Math.sqrt(256) - 1),
                  'the scale looks like 1/sqrt(n) — the symmetric convention, where each ' +
                  'direction carries half the factor. This module puts the whole 1/n on the inverse'],
              ];
              for (const [value, message] of probes) {
                if (nums.some(v => Math.abs(v - value) <= 0.05 * value)) return message;
              }
              return null;
            })();
            // The verdict, not the residual: a round-trip error of 1.7e-7 is
            // printed in exponential notation, which no numeric probe over the
            // console text can read.
            ctx.assert(
              loggedText(ctx.logs, 'round trip clean: true'),
              scaleHint ||
              'the round trip does not come back clean yet — conjugate, transform, conjugate ' +
              'again, and divide by n'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Drive the learner's own kernels through the whole round trip on a
            // different signal: conjugate, fft, conjugate-and-scale.
            const signal = makeTones(256, [[3, 1, 'sin'], [17, 0.6, 'cos'], [44, 0.3, 'sin']]);
            const planes = [Float32Array.from(signal), new Float32Array(256)];
            const { scramble, butterfly, conjugate } = await findInverseKernels(ctx, 256, planes);
            ctx.assert(
              scramble && butterfly && conjugate,
              'expected scramble, butterfly and conjugate kernels'
            );
            const fft = async p => {
              let buffer = await scramble(p);
              for (let half = 1; half < 256; half *= 2) buffer = await butterfly(buffer, half);
              return buffer;
            };
            const spectrum = await fft(planes);
            const recovered = await conjugate(await fft(await conjugate(spectrum, 1)), 1 / 256);
            const hint = diagnoseAll(
              256, i => recovered[0][i], i => signal[i], 5e-3, inverseProbes(signal, 256, 5e-3)
            );
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(recovered[0][i], signal[i], 5e-3, hint || `recovered sample ${i}`);
              ctx.assertClose(
                recovered[1][i], 0, 5e-3,
                `recovered sample ${i} has an imaginary part — a real signal transformed and ` +
                'inverted comes back real'
              );
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The forward transform is still the FORWARD transform. The inverse
            // recipe would round-trip just as cleanly with the twiddle sign
            // flipped, so check the spectrum itself against the definition.
            const signal = makeTones(256, [[5, 1, 'sin']]);
            const planes = [Float32Array.from(signal), new Float32Array(256)];
            const { scramble, butterfly } = await findInverseKernels(ctx, 256, planes);
            ctx.assert(scramble && butterfly, 'expected a scramble kernel and a butterfly kernel');
            let buffer = await scramble(planes);
            for (let half = 1; half < 256; half *= 2) buffer = await butterfly(buffer, half);
            const [wantRe, wantIm] = dftRef(signal, new Array(256).fill(0), -1);
            for (const k of [5, 251]) {
              ctx.assertClose(buffer[0][k], wantRe[k], 5e-3, `forward transform, real part of bin ${k}`);
              ctx.assertClose(
                buffer[1][k], wantIm[k], 5e-3,
                `forward transform, imaginary part of bin ${k} — a clean round trip would not ` +
                'have noticed this, which is the whole point of checking against the definition'
              );
            }
          },
        },
      ],
    },
  ],
};
