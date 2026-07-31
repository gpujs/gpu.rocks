// Module: Filtering in the Frequency Domain — uuid 8c225e10-d7d6-4473-8099-1c45f40a7668
// (short id 8c225e10). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module post-dates
// the pre-uuid urls.
//
// Filtering in the Frequency Domain — where the transform stops being a party
// trick and starts paying rent.
//
// Five tasks: the convolution theorem, verified against a direct convolution
// rather than asserted → the brick-wall low-pass and the ringing it cannot
// avoid → a Gaussian roll-off that trades that ringing for a slower edge →
// gating a spectrum to pull three tones out of noise, measured in dB → and the
// size question, where multiplying spectra convolves CIRCULARLY and eats the
// start of your own answer.
//
// THE COMPLEX CONVENTION, stated once and obeyed everywhere in this file (it is
// the Signal Processing track's, shared by every module in it):
//   * kernel OUTPUT is `output: [n, 2]`, indexed result[p][i] — plane p = 0 is
//     the real part, p = 1 the imaginary part, i the sample or bin index.
//     (`output: [w, h]` is indexed [y][x], so [n, 2] gives exactly [plane][i].)
//   * kernel INPUT is the same thing one level up: a [2][n] nested array, so
//     spec[0][i] is real and spec[1][i] imaginary. A purely real input signal
//     is a plain [n] array, and every task below says which it is taking.
// Verified in a real browser on gpu.js 2.20.0: a [n, 2] kernel hands back an
// Array of two Float32Arrays on BOTH backends, and feeding that straight into
// another kernel that a test also calls with a plain nested array is fine on
// this version (utils.typeFitsValue treats any array as fitting an Array type).
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested number arrays as inputs, this.thread.* for indexing,
// this.constants.* for compile-time values (legal as loop bounds), 2D output
// [w, h] indexed [y][x]. Every task passes in CPU mode, comfortably: the naive
// O(n^2) transforms below cost ~7 ms per round trip at n = 256 on the CPU
// backend and ~90 ms on WebGL, and no task builds more than four kernels.
//
// WHY A NAIVE DFT. This module is about what you do BETWEEN the transforms.
// Every task therefore hands you a complete O(n^2) `dft`/`idft` pair as
// scaffolding and asks you to write only the filtering step. Which transform
// produced the spectrum changes none of the arithmetic below.
//
// FLOAT MARGINS (the track's standing hazard: tests compute in float64, the GL
// backend in float32, and trigonometric sums accumulate). Measured, not
// guessed:
//   * MEASURED ON BOTH RENDERERS, because the gap between them is the whole
//     hazard: an Apple M1 Max (ANGLE Metal) and a forced SwiftShader build,
//     which is what a CI runner gets. The learner's own end-to-end console
//     figures — three float32 kernels deep, forward, multiply, inverse — come
//     to 1.6e-6 / 8.7e-5 for task 1 and 1.5e-6 / 7.1e-5 for task 5. Those two
//     assertions therefore sit at 5e-3, which is ~60x clear of the slower
//     renderer and still ~80x below what the tasks' own starters produce.
//     Nothing in this module asserts a logged figure more tightly than that.
//   * spectrum assertions feed a float64 reference INTO the learner's kernel,
//     so the only error is float32 storage plus that kernel's own arithmetic.
//     Measured on WebGL against the module's real data: 2.1e-7 for task 1's
//     complex multiply (bins up to 3.2), exactly 0 for the pass-through masks
//     of tasks 2 and 4, and 1.5e-5 (Metal) / 3.8e-6 (SwiftShader) for task 3's
//     Math.exp on bins up to 163 — GLSL's exp is only required to be good to
//     3 ULP, which is where that last one comes from, and the software
//     renderer happens to be the more accurate of the two. All of them use
//     eps = 5e-3: a 300x margin at worst.
//   * every near-miss probe in this file is separated from the right answer by
//     at least 2.1 in absolute value, so no probe sits anywhere near a
//     tolerance and none can fire on a merely-rounded answer.
//   * a probe whose predicted value is at a DIFFERENT SCALE from the right
//     answer needs its own epsilon, scaled the same way — see probeEps() below.
//     The two "divided by n a second time" probes predict values 64x smaller
//     and are judged at eps / 64 accordingly. Reusing the assertion's epsilon
//     would make them over-tolerant here, and the mirror mistake (a probe
//     predicting an n times LARGER value) would stop firing on the GL backend
//     altogether, silently.
//   * nothing in this module asserts a PHASE. atan2 on a bin whose imaginary
//     part is near zero swings by a whole turn on float32 noise, so every
//     assertion here is on a plane or on a magnitude, both of which stay
//     continuous through the seam.
//   * task 4's magnitude gate is 30, with the loudest noise bin at 19.5 and the
//     quietest tone at 41.9 — a 1.5x margin on each side against a float error
//     of ~1e-2. No assertion in this module sits near a decision boundary.

// ---- task 1: the convolution theorem ---------------------------------------

const N1 = 64; // transform length
const SUPPORT1 = 20; // how much of that buffer the signal actually occupies
const TAPS1 = [0.4, 0.3, 0.2, 0.08, 0.02]; // deliberately asymmetric — see below

// ---- tasks 2 and 3: one square wave, two filters ---------------------------

const N = 256;
const PERIOD = 128; // two full cycles fill the buffer, so four edges to ring at
const CUT = 20; // task 2's brick-wall cutoff, as a folded bin index
const SIGMA = 12; // task 3's Gaussian width, in bins

// ---- task 4: three tones under noise ---------------------------------------

// [bin, amplitude, phase]. Integer bins, so each tone lands in exactly one bin
// (plus its mirror) and the spectrum is genuinely sparse rather than smeared.
const TONES = [[6, 1, 0], [19, 0.55, 1.1], [44, 0.3, 2.3]];
const NOISE = 0.8; // uniform noise amplitude
const NOISE_SEED = 7331;
const GATE = 30; // magnitude below which a bin is called noise

// ---- task 5: the size question ---------------------------------------------

const N5 = 32; // both source buffers are this long
const TAPS5 = [0.25, 0.2, 0.16, 0.12, 0.1, 0.07, 0.05, 0.03, 0.02]; // sums to 1
const LINEAR5 = N5 + TAPS5.length - 1; // 40 — the length of the honest answer

// ---- deterministic signal building -----------------------------------------

// Three decimals, and never negative zero: an input's own documentation prints
// these values, and "-0" in that list reads as a mistake.
function round3(v) {
  const r = Math.round(v * 1000) / 1000;
  return r === 0 ? 0 : r;
}

// A plucked-string decay occupying the first `support` samples of an n-buffer.
// Asymmetric in time, which is what makes the time-reversal near-miss in task 1
// tellable apart from the right answer.
function pluck(n, support) {
  const out = new Array(n).fill(0);
  for (let i = 0; i < support; i++) {
    out[i] = round3(Math.sin((2 * Math.PI * i) / 9) * Math.exp(-i / 11));
  }
  return out;
}

function decay(n, support) {
  const out = new Array(n).fill(0);
  for (let i = 0; i < support; i++) {
    out[i] = round3(Math.cos((2 * Math.PI * i) / 13) * (1 - i / support));
  }
  return out;
}

// A short filter dropped into the front of an n-sample buffer. A filter is
// almost always shorter than the buffer it travels in; the zeros are not
// padding-for-its-own-sake, they are simply where the filter isn't.
function inBuffer(taps, n) {
  const out = new Array(n).fill(0);
  for (let j = 0; j < taps.length; j++) out[j] = taps[j];
  return out;
}

// Square wave, +1 for half a period then -1, mean exactly zero (which is what
// makes the mirror-bin probe in task 2 exact rather than approximate).
function squareWave() {
  const out = new Array(N);
  for (let i = 0; i < N; i++) out[i] = i % PERIOD < PERIOD / 2 ? 1 : -1;
  return out;
}

function toneSignal() {
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    let v = 0;
    for (let t = 0; t < TONES.length; t++) {
      v += TONES[t][1] * Math.sin((2 * Math.PI * TONES[t][0] * i) / N + TONES[t][2]);
    }
    out[i] = round3(v);
  }
  return out;
}

function noisySignal(utils, seed) {
  const rand = utils.seededRandom(seed);
  const clean = toneSignal();
  const out = new Array(N);
  for (let i = 0; i < N; i++) out[i] = round3(clean[i] + (rand() * 2 - 1) * NOISE);
  return out;
}

// Task 5's source: 32 samples with no quiet stretch anywhere, so the wrapped
// tail lands on top of real signal instead of on top of zeros.
function shortSignal() {
  const out = new Array(N5);
  for (let i = 0; i < N5; i++) {
    out[i] = round3(0.8 * Math.sin((2 * Math.PI * i) / 7) + 0.5 * Math.sin((2 * Math.PI * i) / 13 + 0.7));
  }
  return out;
}

// ---- float64 references -----------------------------------------------------
//
// Every expectation in this file is computed here, in double precision, from
// the same numbers the learner's kernels are handed.

function dftRef(x) {
  const n = x.length;
  const re = new Array(n);
  const im = new Array(n);
  for (let k = 0; k < n; k++) {
    let r = 0;
    let i = 0;
    for (let t = 0; t < n; t++) {
      const angle = (-2 * Math.PI * k * t) / n;
      r += x[t] * Math.cos(angle);
      i += x[t] * Math.sin(angle);
    }
    re[k] = r;
    im[k] = i;
  }
  return [re, im];
}

function idftRef(spec) {
  const n = spec[0].length;
  const re = new Array(n);
  const im = new Array(n);
  for (let t = 0; t < n; t++) {
    let r = 0;
    let i = 0;
    for (let k = 0; k < n; k++) {
      const angle = (2 * Math.PI * k * t) / n;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      r += spec[0][k] * c - spec[1][k] * s;
      i += spec[0][k] * s + spec[1][k] * c;
    }
    re[t] = r / n;
    im[t] = i / n;
  }
  return [re, im];
}

// Direct convolution, the sliding window Convolution & Filters taught: output
// sample i gathers taps.length products. No wrap — samples before 0 and after
// the end of the source simply do not contribute.
function linearConv(x, taps, n) {
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < taps.length; j++) {
      const t = i - j;
      if (t >= 0 && t < x.length) sum += x[t] * taps[j];
    }
    out[i] = sum;
  }
  return out;
}

// The circular convolution multiplying two length-n spectra actually produces:
// the same sum, with the source index taken modulo n.
function circularConv(x, taps, n) {
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < taps.length; j++) sum += taps[j] * x[(((i - j) % n) + n) % n];
    out[i] = sum;
  }
  return out;
}

// Complex product of two spectra, bin by bin. `variant` selects a deliberate
// mistake so a probe can predict exactly what it would have produced:
//   'nocross'  the two cross terms never happen — (ar*br, ai*bi)
//   'conj'     a * conj(b): the correct product of the wrong number, which
//              correlates instead of convolving and reverses the filter
//   'imflip'   the sign inside the imaginary part reversed
//   'reflip'   the sign inside the real part reversed
//   'divn'     the right answer, divided by n a second time
function productRef(a, b, variant) {
  const n = a[0].length;
  const re = new Array(n);
  const im = new Array(n);
  for (let k = 0; k < n; k++) {
    const ar = a[0][k];
    const ai = a[1][k];
    const br = b[0][k];
    const bi = b[1][k];
    if (variant === 'nocross') {
      re[k] = ar * br;
      im[k] = ai * bi;
    } else if (variant === 'conj') {
      re[k] = ar * br + ai * bi;
      im[k] = ai * br - ar * bi;
    } else if (variant === 'imflip') {
      re[k] = ar * br - ai * bi;
      im[k] = ar * bi - ai * br;
    } else if (variant === 'reflip') {
      re[k] = ar * br + ai * bi;
      im[k] = ar * bi + ai * br;
    } else {
      re[k] = ar * br - ai * bi;
      im[k] = ar * bi + ai * br;
    }
    if (variant === 'divn') {
      re[k] /= n;
      im[k] /= n;
    }
  }
  return [re, im];
}

// How far bin k sits from DC once the spectrum's mirror symmetry is taken into
// account: bin n - 6 is six bins from DC, not n - 6.
function fold(k, n) {
  return Math.min(k, n - k);
}

// The brick-wall low-pass of task 2. `variant`:
//   'unfolded' compares the raw index instead of the folded one, so every
//              mirror bin dies too and the inverse transform comes back complex
//   'inverted' keeps exactly what it should have thrown away
function brickRef(spec, cut, variant) {
  const n = spec[0].length;
  const re = new Array(n).fill(0);
  const im = new Array(n).fill(0);
  for (let k = 0; k < n; k++) {
    const d = variant === 'unfolded' ? k : fold(k, n);
    const keep = variant === 'inverted' ? d > cut : d <= cut;
    if (keep) {
      re[k] = spec[0][k];
      im[k] = spec[1][k];
    }
  }
  return [re, im];
}

// The Gaussian roll-off of task 3. `variant`:
//   'unfolded' the response is built from the raw index, so the upper half of
//              the spectrum is crushed and the mirrors go with it
//   'nohalf'   exp(-kk*kk / sigma^2) — the factor of two under the exponent
//              dropped, which is a narrower filter than asked for
//   'oneplane' the response applied to the real plane only
function gaussRef(spec, sigma, variant) {
  const n = spec[0].length;
  const re = new Array(n);
  const im = new Array(n);
  for (let k = 0; k < n; k++) {
    const d = variant === 'unfolded' ? k : fold(k, n);
    const h =
      variant === 'nohalf'
        ? Math.exp(-(d * d) / (sigma * sigma))
        : Math.exp(-(d * d) / (2 * sigma * sigma));
    re[k] = spec[0][k] * h;
    im[k] = spec[1][k] * (variant === 'oneplane' ? 1 : h);
  }
  return [re, im];
}

// The magnitude gate of task 4. `variant`:
//   'reonly'   the gate reads the real plane only, so a tone whose energy sits
//              in the imaginary part is thrown away
//   'nosqrt'   re*re + im*im compared against the gate without the square root
//   'inverted' keeps the noise and discards the tones
//   'perplane' each plane gated on its own magnitude
function gateRef(spec, gate, variant) {
  const n = spec[0].length;
  const re = new Array(n).fill(0);
  const im = new Array(n).fill(0);
  for (let k = 0; k < n; k++) {
    const r = spec[0][k];
    const i = spec[1][k];
    if (variant === 'perplane') {
      if (Math.abs(r) >= gate) re[k] = r;
      if (Math.abs(i) >= gate) im[k] = i;
      continue;
    }
    const measure =
      variant === 'reonly' ? Math.abs(r) : variant === 'nosqrt' ? r * r + i * i : Math.sqrt(r * r + i * i);
    const keep = variant === 'inverted' ? measure < gate : measure >= gate;
    if (keep) {
      re[k] = r;
      im[k] = i;
    }
  }
  return [re, im];
}

// Signal-to-noise ratio in dB: how much bigger the clean signal is than
// whatever `test` gets wrong about it.
function snrDb(clean, test) {
  let signal = 0;
  let error = 0;
  for (let i = 0; i < clean.length; i++) {
    signal += clean[i] * clean[i];
    error += (test[i] - clean[i]) * (test[i] - clean[i]);
  }
  return 10 * Math.log10(signal / error);
}

// ---- near-miss diagnosis ----------------------------------------------------
//
// The course's shared discipline: when a failing value is exactly what some
// specific mistake would produce, name that mistake instead of reporting two
// numbers. A probe pairs such a value with its sentence, and a probe may only
// speak when the observation matches it AND the right answer does not — so a
// bin where two candidates coincide stays silent, as do observations matching
// probes that contradict each other. A confident wrong diagnosis is worse than
// a plain numeric mismatch.

// A probe may carry its OWN epsilon as a third element, and it HAS TO whenever
// the value it predicts sits at a different scale from the right answer. A probe
// for "divided by n a second time" predicts a value n times smaller, and its
// float error shrinks with it, so judging it at the assertion's epsilon makes it
// wildly over-tolerant; the mirror case — a probe predicting an n times LARGER
// value, whose error is n times larger too — is worse, because it simply never
// fires and does so silently. Scale a probe's epsilon the way its value scales.
function probeEps(probe, eps) {
  return probe.length > 2 && probe[2] != null ? probe[2] : eps;
}

function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => {
      const e = probeEps(p, eps);
      return Number.isFinite(p[0]) && Math.abs(got - p[0]) <= e && Math.abs(expected - p[0]) > e;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The array form: one matching cell is weak evidence, so a probe here has to
// predict EVERY element (and disagree with the right answer somewhere) before
// it may speak. `got` and `value` are functions of the index; a missing element
// makes the comparison NaN, which fails.
function diagnoseAll(count, got, expected, eps, probes) {
  const hits = probes
    .filter(probe => {
      const value = probe[0];
      const e = probeEps(probe, eps);
      let differs = false;
      for (let i = 0; i < count; i++) {
        if (!(Math.abs(got(i) - value(i)) <= e)) return false;
        if (Math.abs(expected(i) - value(i)) > e) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The complex form. Half of the near-misses in this module are indistinguishable
// on one plane and obvious across both — a conjugated product and a sign flip in
// the real part agree exactly on plane 0 — so a spectrum probe has to predict
// BOTH planes of EVERY bin. `got(p, k)` and `value(k) -> [re, im]`.
function diagnoseSpectrum(count, got, expected, eps, probes) {
  const hits = probes
    .filter(probe => {
      const value = probe[0];
      const e = probeEps(probe, eps);
      let differs = false;
      for (let k = 0; k < count; k++) {
        const c = value(k);
        if (!(Math.abs(got(0, k) - c[0]) <= e)) return false;
        if (!(Math.abs(got(1, k) - c[1]) <= e)) return false;
        if (Math.abs(expected(0, k) - c[0]) > e || Math.abs(expected(1, k) - c[1]) > e) {
          differs = true;
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// A spectrum handed to diagnoseSpectrum as a [re[], im[]] pair.
function planeReader(pair) {
  return (p, k) => pair[p][k];
}

// ---- console readers --------------------------------------------------------

function lineWith(logs, needle) {
  for (const line of logs) {
    if (line.type === 'log' && line.text && line.text.indexOf(needle) !== -1) return line.text;
  }
  return null;
}

// The number that follows `label` on that line — 'overshoot: 9.20 %' -> 9.2.
// Reading a LABELLED number rather than every number on the line keeps a bin
// index or an array length from being mistaken for the answer. Exponent
// notation is accepted because a learner who prints a raw float gets it.
function numberAfter(text, label) {
  if (!text) return null;
  const match = new RegExp(`${label}\\s*(-?\\d+(?:\\.\\d+)?(?:e[-+]?\\d+)?)`, 'i').exec(text);
  return match ? parseFloat(match[1]) : null;
}

// ---- shared kernel source ---------------------------------------------------
//
// The transforms are scaffolding in every task, so their text is written once
// and pasted into each starter/solution pair. `size` is spliced in as a plain
// number, which is what both the output shape and the loop bound need.

function transformsSource(size) {
  return `// GIVEN — the forward transform. A real signal in, a complex spectrum
// out: with output [n, 2] the result is indexed spec[p][k], where plane 0
// holds the real part of bin k and plane 1 the imaginary part.
const dft = gpu.createKernel(function (sig) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.n; t++) {
    const angle = (-2 * Math.PI * k * t) / this.constants.n;
    re += sig[t] * Math.cos(angle);
    im += sig[t] * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [${size}, 2], constants: { n: ${size} } });

// GIVEN — the inverse. A complex spectrum in as spec[0][k] / spec[1][k], a
// complex signal out in the same two planes. The 1/n lives HERE and nowhere
// else: divide a second time "to be safe" and everything comes back n times
// too small.
const idft = gpu.createKernel(function (spec) {
  const t = this.thread.x;
  let re = 0;
  let im = 0;
  for (let k = 0; k < this.constants.n; k++) {
    const angle = (2 * Math.PI * k * t) / this.constants.n;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    re += spec[0][k] * c - spec[1][k] * s;
    im += spec[0][k] * s + spec[1][k] * c;
  }
  if (this.thread.y === 0) return re / this.constants.n;
  return im / this.constants.n;
}, { output: [${size}, 2], constants: { n: ${size} } });`;
}

const TRANSFORMS_1 = transformsSource(N1);
const TRANSFORMS_N = transformsSource(N);

export default {
  uuid: '8c225e10-d7d6-4473-8099-1c45f40a7668',
  version: 1,
  slug: 'frequency-filtering',
  title: 'Filtering in the Frequency Domain',
  blurb:
    'Convolution becomes multiplication — the trade that makes the FFT worth its complexity, plus the ringing, the wrap-around and the cross terms it hides.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'convolution-theorem',
      title: 'Convolution Becomes Multiplication',
      intro: `<p>Convolution &amp; Filters slid a window along a signal: every output sample gathered
        <code>m</code> products, so the whole pass cost <code>n·m</code>. This task makes that
        sliding window <em>disappear</em>.</p>
        <p>The convolution theorem says: transform both signals, multiply the two spectra
        <strong>bin by bin</strong>, transform back — and what comes out is the convolution.
        No window, no <code>m</code>. The cost collapses to the cost of the transforms, which is
        why every audio plug-in, every software radio and every large-kernel blur is written this
        way. We are going to check that claim rather than believe it.</p>
        <p>Spectra are complex, and gpu.js has no complex type, so this track carries a complex
        signal as <strong>two planes of floats</strong>. A kernel with <code>output: [n, 2]</code>
        is indexed <code>result[p][i]</code> — plane 0 the real part, plane 1 the imaginary part,
        <code>i</code> the bin. Handed back <em>into</em> a kernel the same thing is a
        <code>[2][n]</code> nested array: <code>spec[0][i]</code> real, <code>spec[1][i]</code>
        imaginary. Every module in this track uses that shape.</p>
        <p>Multiplying two complex numbers is where people slip. It is not "reals times reals,
        imaginaries times imaginaries" — that is four products, and two of them cross:</p>
<pre><code>(ar + ai·i)(br + bi·i) = (ar·br − ai·bi) + (ar·bi + ai·br)·i</code></pre>
        <p>Drop the cross terms and you get a specific, recognisable wrong answer — which is
        exactly what the starter below does, so run it first and watch the two roads disagree.</p>`,
      goal: `<strong>Goal:</strong> finish <code>mulSpectra</code> so the frequency-domain road
        reproduces the sliding window's answer — the two should agree to five or six decimal
        places, and whatever is left is float32 rounding rather than physics.`,
      requirements: [
        'Keep <code>output: [64, 2]</code> — plane 0 is the real part of each bin, plane 1 the imaginary part',
        'Read both planes of both arguments: <code>a[0][i]</code>, <code>a[1][i]</code>, <code>b[0][i]</code>, <code>b[1][i]</code>',
        'Plane 0 returns <code>ar * br - ai * bi</code>; plane 1 returns <code>ar * bi + ai * br</code> — all four products, both cross terms',
        'Leave the <code>1/n</code> to <code>idft</code>: the inverse transform already has it',
      ],
      hints: [
        {
          title: 'Hint 1 — which plane am I?',
          body: `<p>With <code>output: [64, 2]</code> there are 128 threads.
            <code>this.thread.x</code> is the bin, 0…63; <code>this.thread.y</code> is the plane,
            0 for real and 1 for imaginary. Every thread reads all four numbers — the four
            products mix both planes of both inputs — and returns just the one belonging to its
            own plane.</p>`,
        },
        {
          title: 'Hint 2 — the four products',
          body: `<p>Name them first, then pick:</p>
<pre><code>const i = this.thread.x;
const ar = a[0][i];
const ai = a[1][i];
const br = b[0][i];
const bi = b[1][i];

if (this.thread.y === 0) return ar * br - ai * bi;
return ar * bi + ai * br;</code></pre>
          <p>Both signs matter. Flip the one in the real part and you have computed
            <code>a</code> times the <em>conjugate</em> of <code>b</code>, which correlates
            instead of convolving — the filter comes out reversed in time.</p>`,
        },
      ],
      transfer: `This is the one trade every FFT library exists to sell. In CUDA it is three calls:
        <code>cufftExecC2C</code> forward, a handful of lines of <code>cuComplex</code> multiply,
        <code>cufftExecC2C</code> inverse — and <code>cuComplex</code> is there so the four products
        get written once instead of in every kernel. ROCm ships rocFFT, Apple ships vDSP and MPS, and
        WebGPU ships nothing at all, so everyone writes the same radix-2 ladder over a storage
        buffer. The two planes you are indexing here are what a <code>float2</code> buffer or an
        <code>rg32float</code> texture holds on every one of them.`,
      starterCode: `// The convolution theorem, both roads at once.
const gpu = new GPU({ mode });

// GIVEN — road 1: direct convolution, the sliding window from
// Convolution & Filters. Output sample x gathers 5 products.
const directConv = gpu.createKernel(function (sig, filt) {
  let sum = 0;
  for (let j = 0; j < this.constants.taps; j++) {
    const t = this.thread.x - j;
    if (t >= 0) {
      sum += sig[t] * filt[j];
    }
  }
  return sum;
}, { output: [64], constants: { taps: 5 } });

${TRANSFORMS_1}

// YOUR JOB — road 2, the middle step. Multiply the two spectra bin by bin.
const mulSpectra = gpu.createKernel(function (a, b) {
  const i = this.thread.x;
  const ar = a[0][i];
  const ai = a[1][i];
  const br = b[0][i];
  const bi = b[1][i];

  // TODO: the two CROSS terms are missing. A complex product is
  // (ar·br − ai·bi) + (ar·bi + ai·br)·i — four products, not two.
  if (this.thread.y === 0) return ar * br;
  return ai * bi;
}, { output: [64, 2] });

const slow = directConv(signal, filt);
const fast = idft(mulSpectra(dft(signal), dft(filt)));

let worst = 0;
let leftover = 0;
for (let i = 0; i < 64; i++) {
  worst = Math.max(worst, Math.abs(fast[0][i] - slow[i]));
  leftover = Math.max(leftover, Math.abs(fast[1][i]));
}
console.log('largest disagreement:', worst.toFixed(9));
console.log('largest imaginary part:', leftover.toFixed(9));
`,
      solutionCode: `// The convolution theorem, both roads at once.
const gpu = new GPU({ mode });

// GIVEN — road 1: direct convolution, the sliding window from
// Convolution & Filters. Output sample x gathers 5 products.
const directConv = gpu.createKernel(function (sig, filt) {
  let sum = 0;
  for (let j = 0; j < this.constants.taps; j++) {
    const t = this.thread.x - j;
    if (t >= 0) {
      sum += sig[t] * filt[j];
    }
  }
  return sum;
}, { output: [64], constants: { taps: 5 } });

${TRANSFORMS_1}

// YOUR JOB — road 2, the middle step. Multiply the two spectra bin by bin.
const mulSpectra = gpu.createKernel(function (a, b) {
  const i = this.thread.x;
  const ar = a[0][i];
  const ai = a[1][i];
  const br = b[0][i];
  const bi = b[1][i];

  if (this.thread.y === 0) return ar * br - ai * bi;
  return ar * bi + ai * br;
}, { output: [64, 2] });

const slow = directConv(signal, filt);
const fast = idft(mulSpectra(dft(signal), dft(filt)));

let worst = 0;
let leftover = 0;
for (let i = 0; i < 64; i++) {
  worst = Math.max(worst, Math.abs(fast[0][i] - slow[i]));
  leftover = Math.max(leftover, Math.abs(fast[1][i]));
}
console.log('largest disagreement:', worst.toFixed(9));
console.log('largest imaginary part:', leftover.toFixed(9));
`,
      inputs: () => ({
        signal: pluck(N1, SUPPORT1),
        filt: inBuffer(TAPS1, N1),
      }),
      publicTests: [
        {
          name: 'the multiply returns two planes of 64 bins',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const a = dftRef(pluck(N1, SUPPORT1));
            const b = dftRef(inBuffer(TAPS1, N1));
            const out = ctx.kernel(a, b);
            ctx.assert(out && out.length === 2, `expected 2 planes, got ${out && out.length}`);
            ctx.assert(
              out[0] && out[0].length === N1,
              `each plane should hold ${N1} bins — is the output still [64, 2]?`
            );
          },
        },
        {
          name: 'every bin holds the complex product <code>a·b</code>',
          run: async ctx => {
            const a = dftRef(pluck(N1, SUPPORT1));
            const b = dftRef(inBuffer(TAPS1, N1));
            const out = ctx.kernel(a, b);
            const want = productRef(a, b);
            const eps = 5e-3;
            const probe = variant => planeReader(productRef(a, b, variant));
            const nocross = probe('nocross');
            const conj = probe('conj');
            const imflip = probe('imflip');
            const reflip = probe('reflip');
            const divn = probe('divn');
            const hint = diagnoseSpectrum(
              N1,
              (p, k) => out[p][k],
              (p, k) => want[p][k],
              eps,
              [
                [k => [nocross(0, k), nocross(1, k)],
                  'that is the real parts multiplied and the imaginary parts multiplied, with no cross terms — a complex product has four: (ar·br − ai·bi) + (ar·bi + ai·br)·i'],
                [k => [conj(0, k), conj(1, k)],
                  'both signs are reversed, which is a times the CONJUGATE of b — that correlates instead of convolving, and the filter comes back reversed in time'],
                [k => [imflip(0, k), imflip(1, k)],
                  'the imaginary part is ar·bi − ai·br; the two cross terms are ADDED there — ar·bi + ai·br'],
                [k => [reflip(0, k), reflip(1, k)],
                  'the real part is ar·br + ai·bi; the product of the two imaginary parts is SUBTRACTED there — ar·br − ai·bi'],
                // n times smaller than the right answer, and so is its float
                // error: eps / N1, not eps.
                [k => [divn(0, k), divn(1, k)],
                  'every bin is the right product divided by 64 — the 1/n belongs to the inverse transform, which already applies it',
                  eps / N1],
              ]
            );
            for (const k of [0, 1, 3, 7, 13, 20, 31, 32, 33, 44, 57, 63]) {
              ctx.assertClose(out[0][k], want[0][k], eps, hint || `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], eps, hint || `bin ${k}, imaginary part`);
            }
          },
        },
        {
          name: 'the theorem holds: the round trip reproduces the direct convolution',
          run: async ctx => {
            const signal = pluck(N1, SUPPORT1);
            const filt = inBuffer(TAPS1, N1);
            const out = ctx.kernel(dftRef(signal), dftRef(filt));
            // Inverse-transform what the learner's kernel produced, in float64,
            // and hold it against the sliding window's answer.
            const back = idftRef([out[0], out[1]]);
            const direct = linearConv(signal, TAPS1, N1);
            const eps = 1e-4;
            // What a conjugated product actually produces, computed rather than
            // guessed at: correlation, which reverses the filter in time and
            // wraps it around the buffer.
            const correlated = idftRef(productRef(dftRef(signal), dftRef(filt), 'conj'))[0];
            for (let i = 0; i < N1; i++) {
              const hint = diagnose(back[0][i], direct[i], eps, [
                [direct[i] / N1, 'the whole result is 64 times too small — something divided by n twice', eps / N1],
                [correlated[i], 'the filter has been applied backwards — that is what multiplying by the conjugate does: it correlates instead of convolving'],
              ]);
              ctx.assertClose(back[0][i], direct[i], eps, hint || `sample ${i} of the round trip`);
            }
            let leftover = 0;
            for (let i = 0; i < N1; i++) leftover = Math.max(leftover, Math.abs(back[1][i]));
            ctx.assert(
              leftover <= eps,
              `the round trip comes back complex (largest imaginary part ${leftover.toFixed(6)}) — ` +
                `two real signals convolve to a real one, so an imaginary part left over means the ` +
                `bin arithmetic is not a complex product`
            );
          },
        },
        {
          name: 'the disagreement between the two roads is logged, and it is tiny',
          run: async ctx => {
            const text = lineWith(ctx.logs, 'largest disagreement');
            ctx.assert(text, 'no disagreement was logged — keep the two console.log lines at the end');
            const worst = numberAfter(text, 'disagreement:');
            ctx.assert(worst !== null, `could not read a number out of "${text}"`);
            // Measured end to end through three float32 kernels: 1.6e-6 on an
            // M1 Max, 8.7e-5 under SwiftShader. 5e-3 keeps a 58x margin on the
            // slower renderer while still being 80x below what the starter's
            // missing cross terms produce (0.41).
            ctx.assert(
              worst <= 5e-3,
              `the two roads still disagree by ${worst} — with the multiply right what is left is ` +
                `float32 rounding and nothing else: about 0.000002 on a GPU, 0.0001 on a software ` +
                `renderer`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Every bin of a different pair, not just the twelve sampled above.
            const signal = decay(N1, SUPPORT1);
            const filt = inBuffer(TAPS1, N1);
            const a = dftRef(signal);
            const b = dftRef(filt);
            const out = ctx.kernel(a, b);
            const want = productRef(a, b);
            const eps = 5e-3;
            const nocross = planeReader(productRef(a, b, 'nocross'));
            const conj = planeReader(productRef(a, b, 'conj'));
            const hint = diagnoseSpectrum(
              N1,
              (p, k) => out[p][k],
              (p, k) => want[p][k],
              eps,
              [
                [k => [nocross(0, k), nocross(1, k)],
                  'the two cross terms are missing — a complex product is (ar·br − ai·bi) + (ar·bi + ai·br)·i'],
                [k => [conj(0, k), conj(1, k)],
                  'both signs are reversed, which is a times the conjugate of b — correlation, not convolution'],
              ]
            );
            for (let k = 0; k < N1; k++) {
              ctx.assertClose(out[0][k], want[0][k], eps, hint || `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], eps, hint || `bin ${k}, imaginary part`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The theorem again, on the second pair, end to end.
            const signal = decay(N1, SUPPORT1);
            const filt = inBuffer(TAPS1, N1);
            const back = idftRef(ctx.kernel(dftRef(signal), dftRef(filt)));
            const direct = linearConv(signal, TAPS1, N1);
            for (let i = 0; i < N1; i++) {
              ctx.assertClose(back[0][i], direct[i], 1e-4, `sample ${i} of the round trip`);
              ctx.assertClose(back[1][i], 0, 1e-4, `sample ${i} should be real`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'brick-wall',
      title: 'The Brick Wall Rings',
      intro: `<p>If multiplying by a spectrum is a filter, then just <em>choose</em> the spectrum.
        Want a low-pass? Set every bin above a cutoff to zero and transform back. Nothing in signal
        processing is simpler, and nothing punishes simplicity faster.</p>
        <p>What comes back is not a smoothed square wave. It is a square wave with
        <strong>ringing</strong>: an overshoot at every edge, plus ripples running the whole length
        of the buffer. Widening the cutoff does not help: it squeezes the ripple closer to the edge,
        but the height of that first overshoot sits at about <strong>9% of the jump</strong> however
        many bins you keep — right up until you keep them all and there is no filter left. This is
        the <strong>Gibbs phenomenon</strong>, and it is not a bug in your arithmetic: a rectangle in
        frequency is a <code>sinc</code> in time, and a sinc rings forever.</p>
        <p>One structural detail first, because it is the most common way this task goes wrong.
        The signal is real, which makes its spectrum <strong>conjugate-symmetric</strong>: bin
        <code>n − k</code> always holds the conjugate of bin <code>k</code>. Those upper bins are
        not extra high frequencies — they are the mirror halves of the low ones. Bin 250 of 256 is
        <em>six</em> bins from DC. Zero a bin without zeroing its mirror and the inverse transform
        comes back complex, so fold the index before you compare it to the cutoff:</p>
<pre><code>const k = this.thread.x;
const dist = Math.min(k, this.constants.n - k);</code></pre>
        <p>The starter below forgets that fold, on purpose. Run it and look at the second log line:
        a real signal has come back with an imaginary part bigger than 1.</p>`,
      goal: `<strong>Goal:</strong> finish <code>lowPass</code> so that every bin more than
        <code>20</code> bins from DC is zeroed on <em>both</em> planes and every other bin passes
        through untouched — then read the overshoot off the console.`,
      requirements: [
        'Keep <code>output: [256, 2]</code>; the incoming spectrum is read as <code>spec[0][k]</code> and <code>spec[1][k]</code>',
        'Measure a bin\'s distance from DC as <code>Math.min(k, this.constants.n - k)</code> — never the raw index',
        'Return <code>0</code> on both planes when that distance exceeds <code>this.constants.cut</code>',
        'Otherwise return the bin unchanged, real part on plane 0 and imaginary part on plane 1',
      ],
      hints: [
        {
          title: 'Hint 1 — the fold, and why',
          body: `<p>For a real signal the spectrum is a palindrome of conjugates: bin
            <code>255</code> is the mirror of bin <code>1</code>, bin <code>236</code> the mirror of
            bin <code>20</code>. Keeping a bin and dropping its mirror keeps half of a cosine and
            throws the other half away, and what is left is a complex exponential. So:</p>
<pre><code>const k = this.thread.x;
const dist = Math.min(k, this.constants.n - k);
if (dist &gt; this.constants.cut) return 0;</code></pre>`,
        },
        {
          title: 'Hint 2 — one return per plane',
          body: `<p>The zero is the same on both planes, so it can be returned before you ever
            look at <code>this.thread.y</code>. What survives is a straight pass-through:</p>
<pre><code>if (this.thread.y === 0) return spec[0][k];
return spec[1][k];</code></pre>`,
        },
        {
          title: 'Hint 3 — reading the answer',
          body: `<p>The console prints the peak as a fraction of the wave's 2-unit jump. You are
            looking for about <code>9.2</code>%. Gibbs' constant is 8.95%; the difference is only
            that our samples do not land exactly on the peak of the ripple.</p>`,
        },
      ],
      transfer: `Every real filter design is a negotiation with this fact. An ideal low-pass has an
        infinitely long, non-causal impulse response, so hardware and DSP libraries ship windowed
        approximations instead — <code>scipy.signal.firwin</code>, MATLAB's <code>fir1</code>, the
        biquad cascades in Web Audio. It is also why a JPEG rings around sharp edges: quantising
        away a block's high-frequency DCT coefficients is exactly the brick wall you just built.`,
      starterCode: `// A brick-wall low-pass, and the ringing it cannot avoid.
const gpu = new GPU({ mode });

${TRANSFORMS_N}

// YOUR JOB — zero everything above the cutoff.
const lowPass = gpu.createKernel(function (spec) {
  const k = this.thread.x;

  // TODO: this compares the RAW bin index. Bin 250 of 256 is the mirror of
  // bin 6, not a high frequency — fold the index first:
  //   const dist = Math.min(k, this.constants.n - k);
  if (k > this.constants.cut) return 0;

  if (this.thread.y === 0) return spec[0][k];
  return spec[1][k];
}, { output: [256, 2], constants: { n: 256, cut: 20 } });

const filtered = idft(lowPass(dft(wave)));

let peak = filtered[0][0];
let leftover = 0;
for (let i = 0; i < 256; i++) {
  peak = Math.max(peak, filtered[0][i]);
  leftover = Math.max(leftover, Math.abs(filtered[1][i]));
}
// samples 16..47 sit at least 16 away from any edge: whatever wobbles there
// is ringing that has travelled, not the edge itself
let ripple = 0;
for (let i = 16; i < 48; i++) ripple = Math.max(ripple, Math.abs(Math.abs(filtered[0][i]) - 1));

// the wave steps from -1 to +1, so the jump is 2 units wide
console.log('overshoot:', (((peak - 1) / 2) * 100).toFixed(2), '% of the jump');
console.log('ripple in the flat stretch:', ripple.toFixed(6));
console.log('largest imaginary part:', leftover.toFixed(6));
`,
      solutionCode: `// A brick-wall low-pass, and the ringing it cannot avoid.
const gpu = new GPU({ mode });

${TRANSFORMS_N}

// YOUR JOB — zero everything above the cutoff.
const lowPass = gpu.createKernel(function (spec) {
  const k = this.thread.x;
  const dist = Math.min(k, this.constants.n - k);

  if (dist > this.constants.cut) return 0;

  if (this.thread.y === 0) return spec[0][k];
  return spec[1][k];
}, { output: [256, 2], constants: { n: 256, cut: 20 } });

const filtered = idft(lowPass(dft(wave)));

let peak = filtered[0][0];
let leftover = 0;
for (let i = 0; i < 256; i++) {
  peak = Math.max(peak, filtered[0][i]);
  leftover = Math.max(leftover, Math.abs(filtered[1][i]));
}
// samples 16..47 sit at least 16 away from any edge: whatever wobbles there
// is ringing that has travelled, not the edge itself
let ripple = 0;
for (let i = 16; i < 48; i++) ripple = Math.max(ripple, Math.abs(Math.abs(filtered[0][i]) - 1));

// the wave steps from -1 to +1, so the jump is 2 units wide
console.log('overshoot:', (((peak - 1) / 2) * 100).toFixed(2), '% of the jump');
console.log('ripple in the flat stretch:', ripple.toFixed(6));
console.log('largest imaginary part:', leftover.toFixed(6));
`,
      inputs: () => ({ wave: squareWave() }),
      publicTests: [
        {
          name: 'the mask returns two planes of 256 bins',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(dftRef(squareWave()));
            ctx.assert(out && out.length === 2, `expected 2 planes, got ${out && out.length}`);
            ctx.assert(
              out[0] && out[0].length === N,
              `each plane should hold ${N} bins — is the output still [256, 2]?`
            );
          },
        },
        {
          name: 'bins within <code>20</code> of DC survive — <strong>including their mirrors</strong>',
          run: async ctx => {
            const spec = dftRef(squareWave());
            const out = ctx.kernel(spec);
            const want = brickRef(spec, CUT);
            const eps = 5e-3;
            const unfolded = planeReader(brickRef(spec, CUT, 'unfolded'));
            const inverted = planeReader(brickRef(spec, CUT, 'inverted'));
            const hint = diagnoseSpectrum(
              N,
              (p, k) => out[p][k],
              (p, k) => want[p][k],
              eps,
              [
                [k => [unfolded(0, k), unfolded(1, k)],
                  'the cutoff was compared against the raw bin index, so every mirror bin died with it — bin 250 is six bins from DC, not 250. Fold first: Math.min(k, this.constants.n - k)'],
                [k => [inverted(0, k), inverted(1, k)],
                  'the comparison is the wrong way round — this kept everything ABOVE the cutoff and threw away everything below it'],
                [k => [spec[0][k], spec[1][k]],
                  'nothing was removed at all — every bin came through untouched'],
              ]
            );
            // The mirrors of the kept harmonics are the whole point, so they are
            // named explicitly alongside a sweep of the rest.
            for (const k of [0, 2, 6, 10, 14, 18, 20, 21, 30, 128, 226, 236, 242, 246, 250, 254]) {
              ctx.assertClose(out[0][k], want[0][k], eps, hint || `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], eps, hint || `bin ${k}, imaginary part`);
            }
          },
        },
        {
          name: 'the filtered wave is real, and it rings',
          run: async ctx => {
            const spec = dftRef(squareWave());
            const back = idftRef(ctx.kernel(spec));
            const want = idftRef(brickRef(spec, CUT));
            let leftover = 0;
            for (let i = 0; i < N; i++) leftover = Math.max(leftover, Math.abs(back[1][i]));
            // The unfolded mask keeps half of every cosine, and half a cosine is
            // a complex exponential: the real part comes out at exactly half the
            // right answer (this wave has no DC), with a large imaginary part
            // beside it. Both halves of that signature are checked before it is
            // named, so a merely wrong answer still gets the plain message.
            let halfEverywhere = true;
            for (let i = 0; i < N; i++) {
              if (Math.abs(back[0][i] - want[0][i] / 2) > 1e-3) halfEverywhere = false;
            }
            ctx.assert(
              leftover <= 1e-3,
              halfEverywhere
                ? `the result came back complex (largest imaginary part ${leftover.toFixed(4)}) and ` +
                    `its real part is exactly half of the right answer — the classic signature of ` +
                    `zeroing the conjugate mirrors along with the high bins. Fold the index: ` +
                    `Math.min(k, this.constants.n - k)`
                : `the result came back complex (largest imaginary part ${leftover.toFixed(4)}) — ` +
                    `a real signal filtered by a symmetric mask must stay real`
            );
            let peak = -Infinity;
            for (let i = 0; i < N; i++) {
              ctx.assertClose(back[0][i], want[0][i], 1e-3, `sample ${i} of the filtered wave`);
              peak = Math.max(peak, back[0][i]);
            }
            ctx.assert(
              peak > 1.05,
              `the filtered wave peaks at ${peak.toFixed(4)} — a brick wall always overshoots, so ` +
                `a peak at or below the wave's own amplitude means the mask never did anything`
            );
          },
        },
        {
          name: 'the overshoot is measured and logged',
          run: async ctx => {
            const text = lineWith(ctx.logs, 'overshoot');
            ctx.assert(text, 'no overshoot was logged — keep the console.log at the end');
            const got = numberAfter(text, 'overshoot:');
            ctx.assert(got !== null, `could not read a number out of "${text}"`);
            // A tighter eps for the probes than for the assertion: 0.092 and 0
            // are only 0.092 apart, so a 0.6-wide probe would cheerfully
            // misdiagnose any small wrong answer as a missing factor of 100.
            const hint = diagnose(got, 9.2, 0.05, [
              [18.41, 'that is the overshoot as a fraction of the wave\'s amplitude — the wave steps from -1 to +1, so the jump it overshoots is 2 units wide, not 1'],
              [0.092, 'that is the overshoot as a fraction rather than a percentage — multiply by 100'],
            ]);
            ctx.assertClose(got, 9.2, 0.6, hint || 'the logged overshoot, in % of the jump');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Every bin, not just the sixteen sampled above.
            const spec = dftRef(squareWave());
            const out = ctx.kernel(spec);
            const want = brickRef(spec, CUT);
            const eps = 5e-3;
            const unfolded = planeReader(brickRef(spec, CUT, 'unfolded'));
            const hint = diagnoseSpectrum(
              N,
              (p, k) => out[p][k],
              (p, k) => want[p][k],
              eps,
              [
                [k => [unfolded(0, k), unfolded(1, k)],
                  'the cutoff was compared against the raw bin index, so the conjugate mirrors died with the high bins — fold first: Math.min(k, this.constants.n - k)'],
              ]
            );
            for (let k = 0; k < N; k++) {
              ctx.assertClose(out[0][k], want[0][k], eps, hint || `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], eps, hint || `bin ${k}, imaginary part`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A different signal through the same mask: a single rectangular
            // pulse, whose spectrum has no zero bins at all, so a mask that
            // merely happens to agree on the square wave's harmonics fails here.
            const pulse = new Array(N).fill(-1);
            for (let i = 40; i < 110; i++) pulse[i] = 1;
            const spec = dftRef(pulse);
            const out = ctx.kernel(spec);
            const want = brickRef(spec, CUT);
            for (let k = 0; k < N; k++) {
              ctx.assertClose(out[0][k], want[0][k], 5e-3, `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], 5e-3, `bin ${k}, imaginary part`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'smooth-rolloff',
      title: 'Trade the Ringing for a Softer Edge',
      intro: `<p>The ringing came from the cliff. A rectangle in frequency has a discontinuity, and
        a discontinuity in one domain is a long, slowly-decaying tail in the other. Round the cliff
        off and the tail goes with it.</p>
        <p>The most satisfying way to round it off is a <strong>Gaussian</strong>, because the
        Fourier transform of a Gaussian is a Gaussian. The impulse response of this filter is
        therefore a Gaussian too — <em>strictly positive</em>, with no negative lobes anywhere. A
        filter that never subtracts cannot overshoot. The ringing does not shrink here; it
        vanishes, from about 9.2% to about 0.00%.</p>
        <p>You pay for it twice. A Gaussian never reaches zero, so it never removes anything
        completely: at <code>σ = 12</code> the bin at <code>k = 26</code> is not deleted, it is
        scaled by <code>exp(−26²/(2·12²)) ≈ 0.096</code>. And it starts attenuating immediately —
        the harmonic at <code>k = 18</code>, which the brick wall passed at full strength, comes
        through at <code>0.325</code>. The edge that swung in 7 samples now takes 13. That is the
        whole trade, and it is the same one the windowing module makes from the other end: a
        smooth taper buys clean tails at the cost of sharpness.</p>
        <p>Convolution &amp; Filters convolved a Gaussian blur directly, tap by tap. This is that
        same filter, arriving from the other side.</p>`,
      goal: `<strong>Goal:</strong> finish <code>rollOff</code> so every bin is scaled by
        <code>exp(−dist²/(2σ²))</code>, where <code>dist</code> is the folded distance from DC —
        and watch the overshoot collapse.`,
      requirements: [
        'Keep <code>output: [256, 2]</code>; the spectrum arrives as <code>spec[0][k]</code> / <code>spec[1][k]</code>',
        'Fold the bin index first, exactly as the brick wall did: <code>Math.min(k, this.constants.n - k)</code>',
        'Build the response <code>h = Math.exp(-(dist * dist) / (2 * sigma * sigma))</code> with <code>sigma</code> from <code>this.constants</code>',
        'Scale <strong>both</strong> planes by the same <code>h</code> — a response that touches only the real part is not a filter',
      ],
      hints: [
        {
          title: 'Hint 1 — the response, then the scaling',
          body: `<p>Compute <code>h</code> once, before you look at which plane you are on. Every
            thread of a bin must agree on it:</p>
<pre><code>const k = this.thread.x;
const dist = Math.min(k, this.constants.n - k);
const s = this.constants.sigma;
const h = Math.exp(-(dist * dist) / (2 * s * s));</code></pre>`,
        },
        {
          title: 'Hint 2 — the finish',
          body: `<p>No branch on the cutoff at all — there is no cutoff. Every bin is scaled,
            most of them by almost nothing:</p>
<pre><code>if (this.thread.y === 0) return spec[0][k] * h;
return spec[1][k] * h;</code></pre>
          <p>The <code>2</code> under the exponent is not decoration. Leave it out and you have a
            filter <code>√2</code> narrower than the one you asked for.</p>`,
        },
      ],
      transfer: `Choosing a window is the oldest trade in the field: rectangular, Hann, Hamming,
        Blackman, Kaiser — each one moves ringing into resolution or back at a different exchange
        rate. cuFFT and rocFFT will happily transform anything; the window is always yours to pick.
        And the Gaussian's other half is why <code>cv::GaussianBlur</code> and every game engine's
        bloom pass use a Gaussian rather than a box: the box is a brick wall in time, and it rings
        in frequency.`,
      starterCode: `// The same wave, the same cutoff, a filter with no cliff in it.
const gpu = new GPU({ mode });

${TRANSFORMS_N}

// YOUR JOB — replace the cliff with a Gaussian roll-off.
const rollOff = gpu.createKernel(function (spec) {
  const k = this.thread.x;
  const dist = Math.min(k, this.constants.n - k);

  // TODO: this is still a brick wall — h is 1 inside the cutoff and 0
  // outside it. Make h fall off smoothly instead:
  //   h = exp(-(dist * dist) / (2 * sigma * sigma))
  let h = 0;
  if (dist <= this.constants.sigma) h = 1;

  if (this.thread.y === 0) return spec[0][k] * h;
  return spec[1][k] * h;
}, { output: [256, 2], constants: { n: 256, sigma: 12 } });

const filtered = idft(rollOff(dft(wave)));

let peak = filtered[0][0];
for (let i = 0; i < 256; i++) peak = Math.max(peak, filtered[0][i]);
let ripple = 0;
for (let i = 16; i < 48; i++) ripple = Math.max(ripple, Math.abs(Math.abs(filtered[0][i]) - 1));

console.log('overshoot:', (((peak - 1) / 2) * 100).toFixed(4), '% of the jump');
console.log('ripple in the flat stretch:', ripple.toFixed(6));
// the eight samples after the rising edge: does the climb ever turn back?
console.log('just after the edge:', Array.from({ length: 8 },
  (unused, j) => Number(filtered[0][j].toFixed(3))));
`,
      solutionCode: `// The same wave, the same cutoff, a filter with no cliff in it.
const gpu = new GPU({ mode });

${TRANSFORMS_N}

// YOUR JOB — replace the cliff with a Gaussian roll-off.
const rollOff = gpu.createKernel(function (spec) {
  const k = this.thread.x;
  const dist = Math.min(k, this.constants.n - k);
  const s = this.constants.sigma;
  const h = Math.exp(-(dist * dist) / (2 * s * s));

  if (this.thread.y === 0) return spec[0][k] * h;
  return spec[1][k] * h;
}, { output: [256, 2], constants: { n: 256, sigma: 12 } });

const filtered = idft(rollOff(dft(wave)));

let peak = filtered[0][0];
for (let i = 0; i < 256; i++) peak = Math.max(peak, filtered[0][i]);
let ripple = 0;
for (let i = 16; i < 48; i++) ripple = Math.max(ripple, Math.abs(Math.abs(filtered[0][i]) - 1));

console.log('overshoot:', (((peak - 1) / 2) * 100).toFixed(4), '% of the jump');
console.log('ripple in the flat stretch:', ripple.toFixed(6));
// the eight samples after the rising edge: does the climb ever turn back?
console.log('just after the edge:', Array.from({ length: 8 },
  (unused, j) => Number(filtered[0][j].toFixed(3))));
`,
      inputs: () => ({ wave: squareWave() }),
      publicTests: [
        {
          name: 'the response scales every bin, on both planes',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const spec = dftRef(squareWave());
            const out = ctx.kernel(spec);
            ctx.assert(out && out.length === 2, `expected 2 planes, got ${out && out.length}`);
            ctx.assert(
              out[0] && out[0].length === N,
              `each plane should hold ${N} bins — is the output still [256, 2]?`
            );
            const want = gaussRef(spec, SIGMA);
            const eps = 5e-3;
            const unfolded = planeReader(gaussRef(spec, SIGMA, 'unfolded'));
            const nohalf = planeReader(gaussRef(spec, SIGMA, 'nohalf'));
            const oneplane = planeReader(gaussRef(spec, SIGMA, 'oneplane'));
            const brick = planeReader(brickRef(spec, SIGMA));
            const hint = diagnoseSpectrum(
              N,
              (p, k) => out[p][k],
              (p, k) => want[p][k],
              eps,
              [
                [k => [unfolded(0, k), unfolded(1, k)],
                  'the response was built from the raw bin index, so the mirror half of the spectrum is crushed and the result will come back complex — fold first: Math.min(k, this.constants.n - k)'],
                [k => [nohalf(0, k), nohalf(1, k)],
                  'the 2 under the exponent is missing: exp(-d²/σ²) is a filter √2 narrower than exp(-d²/(2σ²))'],
                [k => [oneplane(0, k), oneplane(1, k)],
                  'only the real plane was scaled — the imaginary plane came through untouched, which is not a filter, it is a rotation'],
                [k => [brick(0, k), brick(1, k)],
                  'that is still the brick wall from the previous task: h is 1 inside σ and 0 outside it, with no roll-off in between'],
              ]
            );
            for (const k of [0, 2, 6, 10, 14, 18, 22, 26, 34, 50, 90, 128, 166, 206, 234, 246, 250, 254]) {
              ctx.assertClose(out[0][k], want[0][k], eps, hint || `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], eps, hint || `bin ${k}, imaginary part`);
            }
          },
        },
        {
          name: 'the response is symmetric, so the filtered wave is still real',
          run: async ctx => {
            const spec = dftRef(squareWave());
            const back = idftRef(ctx.kernel(spec));
            let leftover = 0;
            for (let i = 0; i < N; i++) leftover = Math.max(leftover, Math.abs(back[1][i]));
            ctx.assert(
              leftover <= 1e-3,
              `the filtered wave came back complex (largest imaginary part ${leftover.toFixed(4)}) — ` +
                `the response has to be the same at bin k and bin n − k, which is what folding the ` +
                `index buys you`
            );
          },
        },
        {
          name: 'the ringing is gone — overshoot below 0.5% of the jump',
          run: async ctx => {
            const spec = dftRef(squareWave());
            const back = idftRef(ctx.kernel(spec));
            const want = idftRef(gaussRef(spec, SIGMA));
            let peak = -Infinity;
            for (let i = 0; i < N; i++) {
              ctx.assertClose(back[0][i], want[0][i], 1e-3, `sample ${i} of the filtered wave`);
              peak = Math.max(peak, back[0][i]);
            }
            const overshoot = ((peak - 1) / 2) * 100;
            ctx.assert(
              overshoot < 0.5,
              `the wave still overshoots by ${overshoot.toFixed(2)}% of the jump — a Gaussian ` +
                `impulse response is strictly positive, so a correct roll-off cannot overshoot at all`
            );
            // Overshoot is the edge; ripple is the tail that used to run the
            // whole buffer. The brick wall leaves 0.077 of it here, the
            // Gaussian about 0.000001.
            let ripple = 0;
            for (let i = 16; i < 48; i++) {
              ripple = Math.max(ripple, Math.abs(Math.abs(back[0][i]) - 1));
            }
            ctx.assert(
              ripple < 1e-3,
              `the flat stretch between the edges still wobbles by ${ripple.toFixed(4)} — that is ` +
                `ringing arriving from an edge 16 samples away, and a Gaussian roll-off does not ` +
                `produce any`
            );
          },
        },
        {
          name: 'the collapsed overshoot is logged',
          run: async ctx => {
            const text = lineWith(ctx.logs, 'overshoot');
            ctx.assert(text, 'no overshoot was logged — keep the console.log at the end');
            const got = numberAfter(text, 'overshoot:');
            ctx.assert(got !== null, `could not read a number out of "${text}"`);
            // Any brick wall on this wave overshoots by 9-10% of the jump,
            // whatever its cutoff, and the right answer is ~0 — so one wide
            // probe covers the whole family without ever coming near it.
            const hint = diagnose(got, 0, 0.7, [
              [9.3, 'that is a brick wall\'s overshoot — the response is still a cliff, not a roll-off'],
            ]);
            ctx.assert(
              got < 0.5,
              hint || `the logged overshoot is ${got}% of the jump; a Gaussian roll-off drives it to about 0.00%`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const spec = dftRef(squareWave());
            const out = ctx.kernel(spec);
            const want = gaussRef(spec, SIGMA);
            const eps = 5e-3;
            const unfolded = planeReader(gaussRef(spec, SIGMA, 'unfolded'));
            const nohalf = planeReader(gaussRef(spec, SIGMA, 'nohalf'));
            const hint = diagnoseSpectrum(
              N,
              (p, k) => out[p][k],
              (p, k) => want[p][k],
              eps,
              [
                [k => [unfolded(0, k), unfolded(1, k)],
                  'the response was built from the raw bin index — fold it: Math.min(k, this.constants.n - k)'],
                [k => [nohalf(0, k), nohalf(1, k)],
                  'the 2 under the exponent is missing: exp(-d²/σ²) is √2 narrower than asked for'],
              ]
            );
            for (let k = 0; k < N; k++) {
              ctx.assertClose(out[0][k], want[0][k], eps, hint || `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], eps, hint || `bin ${k}, imaginary part`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A rectangular pulse: no zero bins anywhere, so every bin's scaling
            // is observable rather than multiplied into a zero.
            const pulse = new Array(N).fill(-1);
            for (let i = 40; i < 110; i++) pulse[i] = 1;
            const spec = dftRef(pulse);
            const out = ctx.kernel(spec);
            const want = gaussRef(spec, SIGMA);
            for (let k = 0; k < N; k++) {
              ctx.assertClose(out[0][k], want[0][k], 5e-3, `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], 5e-3, `bin ${k}, imaginary part`);
            }
            const back = idftRef(out);
            let peak = -Infinity;
            for (let i = 0; i < N; i++) peak = Math.max(peak, back[0][i]);
            ctx.assert(peak <= 1.005, `the pulse overshoots to ${peak.toFixed(4)} — a Gaussian must not`);
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'denoise',
      title: 'A Tone Buried in Noise',
      intro: `<p>Noise is broadband; a tone is not. In the time domain they are hopelessly mixed —
        <code>noisy</code> looks like grass. In the frequency domain they come apart: three tones
        put nearly all their energy into three bins, while the noise spreads its energy thinly over
        all 256. Concentration versus dilution is the entire trick, and it is the best argument
        anyone has ever made for paying the cost of a transform.</p>
        <p>The numbers here: the three tones land at magnitudes <strong>124</strong>,
        <strong>71</strong> and <strong>42</strong>, and the loudest noise bin reaches
        <strong>19.5</strong>. Anything between about 22 and 40 separates them; the gate is set to
        <strong>30</strong>, comfortably in the middle of that gap.</p>
        <p>A convenience falls out of the conjugate symmetry that made task 2 painful. Bin
        <code>k</code> and bin <code>n − k</code> are conjugates, and conjugates have
        <em>equal magnitude</em> — so a gate that reads magnitude keeps or drops both mirrors
        together, automatically, and the result comes back real without you doing anything about
        it. Magnitude needs both planes though:</p>
<pre><code>const mag = Math.sqrt(re * re + im * im);</code></pre>
        <p>The starter gates on <code>Math.abs(re)</code> alone, which throws away any tone whose
        energy happens to sit in the imaginary part. Run it and watch the SNR go the wrong way.</p>
        <p>You are given <code>clean</code> as well as <code>noisy</code> — you would not have it
        in real life, and that is the point: it is here so the improvement can be a number instead
        of an impression.</p>`,
      goal: `<strong>Goal:</strong> finish <code>gate</code> so every bin quieter than
        <code>this.constants.gate</code> is zeroed on both planes, then read the two SNR figures
        off the console. Before is about <strong>4.8 dB</strong>; after should be over
        <strong>20 dB</strong>.`,
      requirements: [
        'Keep <code>output: [256, 2]</code>; the spectrum arrives as <code>spec[0][k]</code> / <code>spec[1][k]</code>',
        'Compute the bin\'s magnitude from <strong>both</strong> planes: <code>Math.sqrt(re * re + im * im)</code>',
        'Return <code>0</code> on both planes when that magnitude is below <code>this.constants.gate</code>; otherwise pass the bin through unchanged',
        'Do not fold the index and do not special-case the mirrors — equal magnitudes take care of them',
      ],
      hints: [
        {
          title: 'Hint 1 — read both planes, decide once',
          body: `<p>Every thread of a bin has to reach the same verdict, so read both planes
            whichever one you are on:</p>
<pre><code>const k = this.thread.x;
const re = spec[0][k];
const im = spec[1][k];
const mag = Math.sqrt(re * re + im * im);
if (mag &lt; this.constants.gate) return 0;</code></pre>
          <p>Dropping the <code>Math.sqrt</code> and comparing <code>re*re + im*im</code> against
            30 is not the same test — it is a gate at magnitude 5.5, and 171 of the 256 bins
            clear it.</p>`,
        },
        {
          title: 'Hint 2 — the SNR arithmetic',
          body: `<p>Signal-to-noise in dB is ten times the log of a power ratio: the clean
            signal's energy over the energy of whatever you got wrong.</p>
<pre><code>let signal = 0;
let error = 0;
for (let i = 0; i &lt; 256; i++) {
  signal += clean[i] * clean[i];
  error += (test[i] - clean[i]) * (test[i] - clean[i]);
}
const db = 10 * Math.log10(signal / error);</code></pre>`,
        },
      ],
      transfer: `Spectral gating is the backbone of noise reduction: Audacity's Noise Reduction,
        RNNoise, and the "spectral subtraction" every phone applies to your voice are all this idea
        with a smarter threshold — estimated per band, from a stretch of silence, instead of one
        number. The same shape reappears as thresholding in wavelet denoising, and as top-k pruning
        of a weight matrix. Keeping the few coefficients that carry the energy is one idea wearing
        several hats.`,
      starterCode: `// Three tones, a lot of noise, and a gate in the frequency domain.
const gpu = new GPU({ mode });

${TRANSFORMS_N}

// YOUR JOB — zero the bins where noise dominates.
const gate = gpu.createKernel(function (spec) {
  const k = this.thread.x;
  const re = spec[0][k];
  const im = spec[1][k];

  // TODO: a bin's magnitude needs BOTH planes — Math.sqrt(re*re + im*im).
  // This one reads the real part only, so a tone whose energy sits in the
  // imaginary part is thrown away with the noise.
  const mag = Math.abs(re);

  if (mag < this.constants.gate) return 0;

  if (this.thread.y === 0) return re;
  return im;
}, { output: [256, 2], constants: { gate: 30 } });

const cleaned = idft(gate(dft(noisy)));

function snr(test) {
  let signal = 0;
  let error = 0;
  for (let i = 0; i < 256; i++) {
    signal += clean[i] * clean[i];
    error += (test[i] - clean[i]) * (test[i] - clean[i]);
  }
  return 10 * Math.log10(signal / error);
}

console.log('SNR before:', snr(noisy).toFixed(2), 'dB');
console.log('SNR after:', snr(cleaned[0]).toFixed(2), 'dB');
`,
      solutionCode: `// Three tones, a lot of noise, and a gate in the frequency domain.
const gpu = new GPU({ mode });

${TRANSFORMS_N}

// YOUR JOB — zero the bins where noise dominates.
const gate = gpu.createKernel(function (spec) {
  const k = this.thread.x;
  const re = spec[0][k];
  const im = spec[1][k];

  const mag = Math.sqrt(re * re + im * im);

  if (mag < this.constants.gate) return 0;

  if (this.thread.y === 0) return re;
  return im;
}, { output: [256, 2], constants: { gate: 30 } });

const cleaned = idft(gate(dft(noisy)));

function snr(test) {
  let signal = 0;
  let error = 0;
  for (let i = 0; i < 256; i++) {
    signal += clean[i] * clean[i];
    error += (test[i] - clean[i]) * (test[i] - clean[i]);
  }
  return 10 * Math.log10(signal / error);
}

console.log('SNR before:', snr(noisy).toFixed(2), 'dB');
console.log('SNR after:', snr(cleaned[0]).toFixed(2), 'dB');
`,
      inputs: utils => ({
        noisy: noisySignal(utils, NOISE_SEED),
        clean: toneSignal(),
      }),
      publicTests: [
        {
          name: 'the gate returns two planes of 256 bins',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(dftRef(noisySignal(ctx.utils, NOISE_SEED)));
            ctx.assert(out && out.length === 2, `expected 2 planes, got ${out && out.length}`);
            ctx.assert(
              out[0] && out[0].length === N,
              `each plane should hold ${N} bins — is the output still [256, 2]?`
            );
          },
        },
        {
          name: 'exactly the six tone bins survive the gate',
          run: async ctx => {
            const spec = dftRef(noisySignal(ctx.utils, NOISE_SEED));
            const out = ctx.kernel(spec);
            const want = gateRef(spec, GATE);
            const eps = 5e-3;
            const reonly = planeReader(gateRef(spec, GATE, 'reonly'));
            const nosqrt = planeReader(gateRef(spec, GATE, 'nosqrt'));
            const inverted = planeReader(gateRef(spec, GATE, 'inverted'));
            const perplane = planeReader(gateRef(spec, GATE, 'perplane'));
            const hint = diagnoseSpectrum(
              N,
              (p, k) => out[p][k],
              (p, k) => want[p][k],
              eps,
              [
                [k => [reonly(0, k), reonly(1, k)],
                  'the gate read the real plane only — a bin\'s magnitude is Math.sqrt(re*re + im*im), and a tone whose energy sits in the imaginary part is invisible to Math.abs(re)'],
                [k => [nosqrt(0, k), nosqrt(1, k)],
                  're*re + im*im was compared against the gate without the square root — that is a gate at magnitude 5.5, and it lets 171 of the 256 bins through'],
                [k => [inverted(0, k), inverted(1, k)],
                  'the comparison is the wrong way round — this kept the noise and threw away the tones'],
                [k => [perplane(0, k), perplane(1, k)],
                  'each plane was gated on its own value; a bin is kept or dropped as a whole, on the magnitude the two planes make together'],
                [k => [spec[0][k], spec[1][k]],
                  'nothing was removed at all — every bin came through untouched'],
              ]
            );
            const bins = [0, 1, 5, 6, 7, 19, 20, 44, 45, 100, 128, 212, 237, 250, 255];
            for (const k of bins) {
              ctx.assertClose(out[0][k], want[0][k], eps, hint || `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], eps, hint || `bin ${k}, imaginary part`);
            }
            let kept = 0;
            for (let k = 0; k < N; k++) {
              if (Math.abs(out[0][k]) > eps || Math.abs(out[1][k]) > eps) kept++;
            }
            ctx.assert(
              kept === 6,
              hint ||
                `${kept} bins survived the gate — three tones and their three conjugate mirrors is ` +
                  `six, so any other count means the gate is measuring the wrong thing`
            );
          },
        },
        {
          name: 'the cleaned signal is real and much closer to <code>clean</code>',
          run: async ctx => {
            const noisy = noisySignal(ctx.utils, NOISE_SEED);
            const clean = toneSignal();
            const back = idftRef(ctx.kernel(dftRef(noisy)));
            let leftover = 0;
            for (let i = 0; i < N; i++) leftover = Math.max(leftover, Math.abs(back[1][i]));
            ctx.assert(
              leftover <= 1e-3,
              `the cleaned signal came back complex (largest imaginary part ${leftover.toFixed(4)}) — ` +
                `a magnitude gate keeps conjugate pairs together, so this means the two planes were ` +
                `gated on different values`
            );
            const before = snrDb(clean, noisy);
            const after = snrDb(clean, back[0]);
            ctx.assert(
              after > 20,
              `the cleaned signal scores ${after.toFixed(2)} dB against ${before.toFixed(2)} dB ` +
                `before — keeping just the six tone bins should reach about 24 dB`
            );
          },
        },
        {
          name: 'both SNR figures are logged',
          run: async ctx => {
            const beforeLine = lineWith(ctx.logs, 'SNR before');
            const afterLine = lineWith(ctx.logs, 'SNR after');
            ctx.assert(beforeLine, 'no "SNR before" line — keep both console.log lines at the end');
            ctx.assert(afterLine, 'no "SNR after" line — keep both console.log lines at the end');
            const before = numberAfter(beforeLine, 'SNR before:');
            const after = numberAfter(afterLine, 'SNR after:');
            ctx.assert(before !== null && after !== null, 'could not read the two dB figures off those lines');
            ctx.assertClose(before, 4.77, 0.2, 'the logged SNR before filtering');
            ctx.assert(
              after > 20,
              `the logged SNR after filtering is ${after} dB — the gate should lift it past 20 dB, ` +
                `an improvement of about 19.5 dB (roughly 90x in power)`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const spec = dftRef(noisySignal(ctx.utils, NOISE_SEED));
            const out = ctx.kernel(spec);
            const want = gateRef(spec, GATE);
            const eps = 5e-3;
            const reonly = planeReader(gateRef(spec, GATE, 'reonly'));
            const nosqrt = planeReader(gateRef(spec, GATE, 'nosqrt'));
            const hint = diagnoseSpectrum(
              N,
              (p, k) => out[p][k],
              (p, k) => want[p][k],
              eps,
              [
                [k => [reonly(0, k), reonly(1, k)],
                  'the gate read the real plane only — magnitude is Math.sqrt(re*re + im*im)'],
                [k => [nosqrt(0, k), nosqrt(1, k)],
                  'the square root is missing — re*re + im*im against 30 gates at magnitude 5.5'],
              ]
            );
            for (let k = 0; k < N; k++) {
              ctx.assertClose(out[0][k], want[0][k], eps, hint || `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], eps, hint || `bin ${k}, imaginary part`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A different noise draw: same tones, different grass. The gate has
            // to be a property of the code, not of one lucky seed.
            const noisy = noisySignal(ctx.utils, 4242);
            const clean = toneSignal();
            const spec = dftRef(noisy);
            const out = ctx.kernel(spec);
            const want = gateRef(spec, GATE);
            for (let k = 0; k < N; k++) {
              ctx.assertClose(out[0][k], want[0][k], 5e-3, `bin ${k}, real part`);
              ctx.assertClose(out[1][k], want[1][k], 5e-3, `bin ${k}, imaginary part`);
            }
            const back = idftRef(out);
            ctx.assert(
              snrDb(clean, back[0]) > 20,
              `on the second noise draw the cleaned signal only reaches ${snrDb(clean, back[0]).toFixed(2)} dB`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'zero-padding',
      title: 'Multiplying Spectra Convolves in a Circle',
      intro: `<p>Task 1 had an accident built into it: the signal occupied 20 of 64 samples and the
        filter 5, so the 24-sample answer had room to spare. Take the room away and the theorem
        bites.</p>
        <p>Multiplying two length-<code>n</code> spectra does not produce the convolution you want.
        It produces the <strong>circular</strong> convolution — the one that treats both buffers as
        loops, so whatever runs off the end comes back on at the beginning:</p>
<pre><code>circular[i] = linear[i] + linear[i + n]</code></pre>
        <p>Here <code>sig</code> is 32 samples and <code>filt</code> carries 9 taps, so the honest
        answer is <code>32 + 9 − 1 = 40</code> samples long. Multiply their 32-bin spectra and the
        last 8 samples fold back onto the first 8, corrupting them by as much as
        <strong>0.37</strong> against a signal whose largest value is 0.66. The signature is
        unmistakable once you have seen it: <em>the tail of the answer appears at its head</em>.</p>
        <p>The fix is to give the answer somewhere to live. Zero-pad both buffers to at least
        <code>n + m − 1</code> before transforming — 40 here, and since every real FFT wants a
        power of two, <strong>64</strong>. The overhang then lands on padding instead of on your
        data. The starter transforms at 32 and prints the damage; fix <code>PAD</code> and the
        guard inside <code>pad</code>, and the error should collapse to float32 rounding — a few
        millionths on a GPU, a few hundred-thousandths on a software renderer.</p>
        <p>This is also where the cost story gets honest. Padding to a power of two means
        transforming 64 points to convolve 32 with 9 — and 9 taps is short enough that the sliding
        window wins outright, on any hardware. Frequency-domain convolution starts paying somewhere
        around a few dozen taps, and exactly where depends on the machine. The Benchmark button will
        price this pipeline for you on both backends; Measuring Speed Honestly owns the rest of the
        methodology, including why a single timing is a rumour.</p>`,
      goal: `<strong>Goal:</strong> pick a <code>PAD</code> long enough to hold the whole answer,
        and finish <code>pad</code> so it copies the first 32 samples and zeroes the rest.`,
      requirements: [
        'Set <code>PAD</code> to at least <code>32 + 9 − 1 = 40</code>; round up to <code>64</code>, the next power of two',
        'Inside <code>pad</code>, return <code>src[this.thread.x]</code> only while <code>this.thread.x</code> is below <code>this.constants.src</code>, and <code>0</code> beyond it',
        'Guard that read: past the end of the source there is nothing to read, and gpu.js answers <code>NaN</code> on the CPU backend and garbage on WebGL',
        'Leave the rest of the pipeline alone — the transforms and the complex multiply are task 1\'s, unchanged',
      ],
      hints: [
        {
          title: 'Hint 1 — how long is the answer?',
          body: `<p>Convolving <code>n</code> samples with <code>m</code> taps produces
            <code>n + m − 1</code> samples: the filter's last tap is still hanging over the
            signal's last sample <code>m − 1</code> steps after the signal has ended. With 32 and
            9 that is 40. Any transform length of 40 or more works; 64 is what you would use in
            practice, because that is what an FFT wants.</p>`,
        },
        {
          title: 'Hint 2 — the guard',
          body: `<p>Two lines, and the <code>if</code> matters — a ternary would still describe an
            out-of-range read to the GLSL compiler:</p>
<pre><code>if (this.thread.x &lt; this.constants.src) {
  return src[this.thread.x];
}
return 0;</code></pre>
          <p><code>filt</code> already arrives in a 32-sample buffer with 9 taps and 23 zeros, so
            the same kernel pads it correctly with no special case.</p>`,
        },
        {
          title: 'Hint 3 — what wrapping looks like',
          body: `<p>Before you fix it, compare the first eight samples the starter prints against
            the last eight of <code>direct</code>. They are the same numbers. That is not a
            coincidence and it is not rounding — it is the tail of the convolution, delivered to
            the front of the buffer.</p>`,
        },
      ],
      transfer: `Every FFT convolution in production carries this bookkeeping. cuFFT, FFTW and
        <code>scipy.signal.fftconvolve</code> all pad internally and hand back the
        <code>n + m − 1</code> answer; <code>numpy.fft</code> does not, which is why the "why is my
        filtered audio clicking at the block boundaries" question outlives every generation of
        programmers. Streaming filters solve the same problem with overlap-add or overlap-save:
        pad each block, convolve, and add the overhang into the next block instead of throwing it
        at the start of this one.`,
      starterCode: `// Zero-padding, or: where does the answer live?
const SRC = 32;   // both source buffers are 32 samples long; filt has 9 taps

// TODO: SRC is exactly the source length, which leaves the answer nowhere to
// go. How long is the convolution of 32 samples with 9 taps?
const PAD = SRC;

const gpu = new GPU({ mode });

// GIVEN — the forward transform (see task 1).
const dft = gpu.createKernel(function (sig) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.n; t++) {
    const angle = (-2 * Math.PI * k * t) / this.constants.n;
    re += sig[t] * Math.cos(angle);
    im += sig[t] * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [PAD, 2], constants: { n: PAD } });

// GIVEN — the inverse.
const idft = gpu.createKernel(function (spec) {
  const t = this.thread.x;
  let re = 0;
  let im = 0;
  for (let k = 0; k < this.constants.n; k++) {
    const angle = (2 * Math.PI * k * t) / this.constants.n;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    re += spec[0][k] * c - spec[1][k] * s;
    im += spec[0][k] * s + spec[1][k] * c;
  }
  if (this.thread.y === 0) return re / this.constants.n;
  return im / this.constants.n;
}, { output: [PAD, 2], constants: { n: PAD } });

// GIVEN — task 1's complex multiply, unchanged.
const mulSpectra = gpu.createKernel(function (a, b) {
  const i = this.thread.x;
  const ar = a[0][i];
  const ai = a[1][i];
  const br = b[0][i];
  const bi = b[1][i];
  if (this.thread.y === 0) return ar * br - ai * bi;
  return ar * bi + ai * br;
}, { output: [PAD, 2] });

// YOUR JOB — copy the source into a longer buffer and zero the rest.
const pad = gpu.createKernel(function (src) {
  // TODO: only the first this.constants.src samples exist. Past that there
  // is nothing to read — return 0 instead of reading off the end.
  return src[this.thread.x];
}, { output: [PAD], constants: { src: SRC } });

const result = idft(mulSpectra(dft(pad(sig)), dft(pad(filt))));

let worst = 0;
const compared = Math.min(40, result[0].length);
for (let i = 0; i < compared; i++) worst = Math.max(worst, Math.abs(result[0][i] - direct[i]));
console.log('worst error vs the direct convolution:', worst.toFixed(9));
console.log('first 8 of result:', Array.from({ length: 8 },
  (unused, i) => Number(result[0][i].toFixed(3))));
console.log('last 8 of direct: ', Array.from({ length: 8 },
  (unused, i) => Number(direct[32 + i].toFixed(3))));
`,
      solutionCode: `// Zero-padding, or: where does the answer live?
const SRC = 32;   // both source buffers are 32 samples long; filt has 9 taps

// 32 + 9 - 1 = 40 samples of answer; 64 is the next power of two.
const PAD = 64;

const gpu = new GPU({ mode });

// GIVEN — the forward transform (see task 1).
const dft = gpu.createKernel(function (sig) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let t = 0; t < this.constants.n; t++) {
    const angle = (-2 * Math.PI * k * t) / this.constants.n;
    re += sig[t] * Math.cos(angle);
    im += sig[t] * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [PAD, 2], constants: { n: PAD } });

// GIVEN — the inverse.
const idft = gpu.createKernel(function (spec) {
  const t = this.thread.x;
  let re = 0;
  let im = 0;
  for (let k = 0; k < this.constants.n; k++) {
    const angle = (2 * Math.PI * k * t) / this.constants.n;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    re += spec[0][k] * c - spec[1][k] * s;
    im += spec[0][k] * s + spec[1][k] * c;
  }
  if (this.thread.y === 0) return re / this.constants.n;
  return im / this.constants.n;
}, { output: [PAD, 2], constants: { n: PAD } });

// GIVEN — task 1's complex multiply, unchanged.
const mulSpectra = gpu.createKernel(function (a, b) {
  const i = this.thread.x;
  const ar = a[0][i];
  const ai = a[1][i];
  const br = b[0][i];
  const bi = b[1][i];
  if (this.thread.y === 0) return ar * br - ai * bi;
  return ar * bi + ai * br;
}, { output: [PAD, 2] });

// YOUR JOB — copy the source into a longer buffer and zero the rest.
const pad = gpu.createKernel(function (src) {
  if (this.thread.x < this.constants.src) {
    return src[this.thread.x];
  }
  return 0;
}, { output: [PAD], constants: { src: SRC } });

const result = idft(mulSpectra(dft(pad(sig)), dft(pad(filt))));

let worst = 0;
const compared = Math.min(40, result[0].length);
for (let i = 0; i < compared; i++) worst = Math.max(worst, Math.abs(result[0][i] - direct[i]));
console.log('worst error vs the direct convolution:', worst.toFixed(9));
console.log('first 8 of result:', Array.from({ length: 8 },
  (unused, i) => Number(result[0][i].toFixed(3))));
console.log('last 8 of direct: ', Array.from({ length: 8 },
  (unused, i) => Number(direct[32 + i].toFixed(3))));
`,
      inputs: () => ({
        sig: shortSignal(),
        filt: inBuffer(TAPS5, N5),
        direct: linearConv(shortSignal(), TAPS5, LINEAR5),
      }),
      publicTests: [
        {
          name: 'the padded buffer is long enough to hold the whole convolution',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(shortSignal());
            ctx.assert(out && out.length, 'the padding kernel returned nothing');
            ctx.assert(
              out.length >= LINEAR5,
              out.length === N5
                ? `the buffer is still ${N5} samples — exactly the source length, which leaves the ` +
                    `answer nowhere to go. 32 samples convolved with 9 taps is ${LINEAR5} samples long, ` +
                    `and anything shorter wraps the overhang onto the start`
                : `the buffer is ${out.length} samples; the convolution needs at least ${LINEAR5} ` +
                    `(32 + 9 - 1) or its tail folds back onto its head`
            );
            ctx.assert(
              out.length <= 1024,
              `${out.length} samples is far more padding than this needs — 64 is the next power of ` +
                `two above 40, and every extra bin costs a full column of the transform`
            );
          },
        },
        {
          name: 'padding copies the source and zeroes the rest',
          run: async ctx => {
            const sig = shortSignal();
            const out = ctx.kernel(sig);
            for (let i = 0; i < N5; i++) {
              ctx.assert(
                Number.isFinite(out[i]),
                `sample ${i} of the padded buffer is ${out[i]} — the source read went off the end`
              );
              ctx.assertClose(out[i], sig[i], 1e-4, `sample ${i} should be the source value`);
            }
            for (let i = N5; i < out.length; i++) {
              ctx.assert(
                Number.isFinite(out[i]),
                `sample ${i} of the padded buffer is ${out[i]} — reading src past its end gives NaN ` +
                  `on the CPU backend. Guard it: if (this.thread.x < this.constants.src) …`
              );
              ctx.assertClose(out[i], 0, 1e-6, `sample ${i} is past the source and should be 0`);
            }
            // The filter travels in a buffer of the same shape, so the same
            // kernel has to pad it too.
            const filt = inBuffer(TAPS5, N5);
            const padded = ctx.kernel(filt);
            for (let i = 0; i < TAPS5.length; i++) {
              ctx.assertClose(padded[i], TAPS5[i], 1e-4, `tap ${i} of the padded filter`);
            }
            for (let i = TAPS5.length; i < padded.length; i++) {
              ctx.assertClose(padded[i], 0, 1e-6, `sample ${i} of the padded filter should be 0`);
            }
          },
        },
        {
          name: 'the padded pipeline reproduces the direct convolution, tail and all',
          run: async ctx => {
            const sig = shortSignal();
            const filt = inBuffer(TAPS5, N5);
            const direct = linearConv(sig, TAPS5, LINEAR5);
            // Run the theorem over the learner's OWN padded buffers, in float64:
            // whatever length they chose, this is what their pipeline computes.
            const paddedSig = Array.from(ctx.kernel(sig));
            const paddedFilt = Array.from(ctx.kernel(filt));
            const product = productRef(dftRef(paddedSig), dftRef(paddedFilt));
            const back = idftRef(product);
            const wrapped = circularConv(sig, TAPS5, N5);
            const eps = 1e-4;
            const hint = diagnoseAll(
              Math.min(N5, back[0].length),
              i => back[0][i],
              i => direct[i],
              eps,
              [
                [i => wrapped[i],
                  'this is the circular convolution: the last 8 samples of the answer have folded ' +
                    'back onto the first 8, so result[i] came out as direct[i] + direct[i + 32]. ' +
                    'Pad both buffers to 40 samples or more before transforming'],
              ]
            );
            for (let i = 0; i < LINEAR5; i++) {
              ctx.assertClose(back[0][i], direct[i], eps, hint || `sample ${i} of the convolution`);
            }
            for (let i = 0; i < LINEAR5; i++) {
              ctx.assertClose(back[1][i], 0, eps, `sample ${i} should be real`);
            }
          },
        },
        {
          name: 'the remaining error is logged, and it is float rounding',
          run: async ctx => {
            const text = lineWith(ctx.logs, 'worst error');
            ctx.assert(text, 'no error figure was logged — keep the console.log at the end');
            const worst = numberAfter(text, 'convolution:');
            ctx.assert(worst !== null, `could not read a number out of "${text}"`);
            // 1.5e-6 on an M1 Max, 7.1e-5 under SwiftShader — so 5e-3 leaves a
            // 70x margin there and still sits 75x below the wrapped answer.
            const hint = diagnose(worst, 0, 5e-3, [
              [0.3736, 'that is exactly the size of the wrapped tail — the transform is still ' +
                'running at the source length, so the answer is being convolved in a circle'],
            ]);
            ctx.assert(
              worst <= 5e-3,
              hint ||
                `the pipeline is still ${worst} away from the direct convolution; with enough ` +
                  `padding what is left is float32 rounding — about 0.000002 on a GPU`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different source through the same padding kernel: the guard has
            // to be about the buffer, not about these particular numbers.
            const other = new Array(N5);
            for (let i = 0; i < N5; i++) other[i] = round3(Math.cos((2 * Math.PI * i) / 5) - i / 40);
            const out = ctx.kernel(other);
            ctx.assert(out.length >= LINEAR5, `the padded buffer is only ${out.length} samples`);
            for (let i = 0; i < N5; i++) {
              ctx.assertClose(out[i], other[i], 1e-4, `sample ${i} should be the source value`);
            }
            for (let i = N5; i < out.length; i++) {
              ctx.assertClose(out[i], 0, 1e-6, `sample ${i} is past the source and should be 0`);
            }
            const paddedFilt = Array.from(ctx.kernel(inBuffer(TAPS5, N5)));
            const back = idftRef(productRef(dftRef(Array.from(out)), dftRef(paddedFilt)));
            const direct = linearConv(other, TAPS5, LINEAR5);
            for (let i = 0; i < LINEAR5; i++) {
              ctx.assertClose(back[0][i], direct[i], 1e-4, `sample ${i} of the convolution`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Nothing beyond the honest answer may be non-zero: a convolution of
            // 32 samples with 9 taps is 40 samples and then silence. A buffer
            // that still wraps fails this even where the head happens to agree.
            const sig = shortSignal();
            const paddedSig = Array.from(ctx.kernel(sig));
            const paddedFilt = Array.from(ctx.kernel(inBuffer(TAPS5, N5)));
            const back = idftRef(productRef(dftRef(paddedSig), dftRef(paddedFilt)));
            for (let i = LINEAR5; i < paddedSig.length; i++) {
              ctx.assertClose(
                back[0][i],
                0,
                1e-4,
                `sample ${i} is past the end of the convolution and should be silent`
              );
            }
          },
        },
      ],
    },
  ],
};
