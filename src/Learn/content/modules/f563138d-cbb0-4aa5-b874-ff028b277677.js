// Module: Windowing & Spectral Leakage — uuid f563138d-cbb0-4aa5-b874-ff028b277677
// (short id f563138d). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId — this module postdates
// the uuid migration — and no track field: track membership lives in tracks.js.
//
// Signal Processing — Windowing & Spectral Leakage.
//
// Five tasks: measure leakage on a tone that fits the window and one that does
// not → find the discontinuity in the implicit periodic extension that causes
// it → kill it with a Hann taper → put numbers on what the taper costs (main
// lobe versus side lobes, rectangular / Hann / Blackman) → give the amplitude
// back with the window's coherent gain.
//
// COMPLEX CONVENTION (this track's, stated to the learner in task 1): gpu.js has
// no complex type, so a spectrum is two planes of floats — output: [n, 2],
// indexed result[p][k], p = 0 real, p = 1 imaginary. output: [w, h] is indexed
// [y][x], so [n, 2] is exactly [plane][bin]. The DFT kernels are GIVEN in every
// starter: this module is about windows, not about re-deriving a transform the
// track teaches elsewhere.
//
// CONTAINER DISCIPLINE. gpu.js compiles a kernel value for the constructor of
// the argument it first saw (backend/web-gl/kernel-value/single-array.js:
// `value.constructor !== this.initialValueConstructor` → switchKernels), so
// handing one kernel a plain Array on one call and a Float32Array on the next
// forces a recompile mid-run. Kernel results come back as Float32Arrays and the
// injected signals are plain Arrays, so tasks 3-5 window EVERY signal before
// transforming it — including the rectangular case, which is a window kernel
// that multiplies by 1. That is not a workaround dressed up as pedagogy: "no
// window" really is the rectangular window, and task 4 measures it as one.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, loop
// bounds come from this.constants, Math.PI and Math.cos/sin/sqrt are on gpu.js's
// whitelist, `%` transpiles to GLSL mod(). One more, learned here and passed on
// to the learner in task 4's first hint: a kernel LOCAL must not share its name
// with a constant. `const n = this.constants.n` transpiles on the CPU backend to
// `const constants_n = constants_n`, which throws "cannot access before
// initialization" — and only on the CPU backend, so the GL run looks fine.
// Every task passes in CPU mode as well as GPU mode, and
// the largest transform is 1,024 bins over a 256-sample signal (~1.6 M inner
// iterations across the three windows of task 4), comfortably inside the
// sandbox's 5 s pre-flight budget.
//
// FLOAT MARGINS (the signals spec asks for these to be recorded, because three
// earlier modules learned them the hard way):
//   • Every given DFT kernel reduces its angle with (k * i) % nfft before the
//     trig call. Without it the argument reaches ~1,600 rad, where a float32
//     cos() carries ~1e-4 of absolute error and a driver with sloppy range
//     reduction carries more; with it every angle sits inside one turn and the
//     GL backend agrees with float64 to ~1e-6 relative. k * i peaks at
//     1023 * 255 = 260,865, well under 2^24, so the mod is exact.
//   • A float32 simulation of every asserted quantity here was run against the
//     float64 reference: peak amplitudes agreed to 6 decimals, the dB figures
//     of tasks 3 and 4 to 4 decimals, and the main-lobe widths were identical.
//     Tolerances below are therefore set by how far the nearest WRONG answer
//     sits, not by float drift:
//       task 1  amplitude 1.000 / 0.655             — tol 5e-3
//       task 2  seam 0.0192 / 1.9783                — tol 2e-3, 100x apart
//       task 3  leak -17.68 dB / -47.20 dB          — tol 0.3 dB, 29 dB apart
//       task 4  widths 2 / 4 / 6 bins (exact integers), peak side lobes
//               -13.31 / -32.17 / -58.23 dB — tol 0.4 dB, nearest wrong
//               candidate 41 dB away
//       task 5  amplitude 0.600 against 0.300 / 0.252 / 0.800 / 0.827
//   • The first null of a windowed on-bin tone is a genuine zero, so task 4's
//     "walk out while the magnitude falls" stops on a sample ~1e-6 of the peak
//     whose neighbours are 2.6e-3 and 5.8e-4 of it. Three orders of margin:
//     float32 cannot move that stopping point.

const N = 256; // window length, every task
const ON_BIN = 8; // whole cycles inside the window
const OFF_BIN = 8.5; // half a bin off — the worst case
const NFFT = 1024; // zero-padded transform length, task 4
const OVER = NFFT / N; // 4 spectrum samples per bin
const TRADE_BIN = 32; // the on-bin tone task 4 analyses
const GAIN_BIN = 20; // the on-bin tone task 5 calibrates
const GAIN_AMP = 0.6; // …and its true amplitude
const TAU = 2 * Math.PI;

// ---- the signals, built deterministically (shared by inputs() and tests) ----

// Rounded to 12 decimals purely for looks: cos(π/2) comes out of float64 as
// 6.12e-17, and the task-inputs panel showing a learner "[1, 0.707, 6.12e-17,
// …]" reads like a bug in the signal. 1e-12 is eight orders below the tightest
// tolerance in this file, so nothing measured moves.
function tone(cycles, amp = 1, len = N) {
  const x = new Array(len);
  for (let i = 0; i < len; i++) {
    x[i] = Math.round(amp * Math.cos((TAU * cycles * i) / len) * 1e12) / 1e12;
  }
  return x;
}

function ones(len = N) {
  return new Array(len).fill(1);
}

// ---- host-side reference DFT ------------------------------------------------
//
// Same angle reduction as the kernels, so the reference and the GPU differ only
// by float32 rounding, never by a different formula.
function dftMag(x, nfft = x.length) {
  const mag = new Array(nfft);
  for (let k = 0; k < nfft; k++) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < x.length; i++) {
      const angle = (-TAU * ((k * i) % nfft)) / nfft;
      re += x[i] * Math.cos(angle);
      im += x[i] * Math.sin(angle);
    }
    mag[k] = Math.sqrt(re * re + im * im);
  }
  return mag;
}

// ---- windows ----------------------------------------------------------------
//
// PERIODIC (DFT-even) forms — divisor n, not n - 1. The task 3 intro says why,
// and the symmetric variants are carried here because they are exactly what the
// near-miss probes look for.
const rectWindow = () => 1;
const hannWindow = i => 0.5 - 0.5 * Math.cos((TAU * i) / N);
const hannSymmetric = i => 0.5 - 0.5 * Math.cos((TAU * i) / (N - 1));
const blackmanWindow = i =>
  0.42 - 0.5 * Math.cos((TAU * i) / N) + 0.08 * Math.cos((2 * TAU * i) / N);
const blackmanSymmetric = i =>
  0.42 - 0.5 * Math.cos((TAU * i) / (N - 1)) + 0.08 * Math.cos((2 * TAU * i) / (N - 1));

function applyWindow(x, w) {
  return x.map((v, i) => v * w(i));
}

function samplesOf(w, len = N) {
  const out = new Array(len);
  for (let i = 0; i < len; i++) out[i] = w(i);
  return out;
}

function sumOf(values) {
  let total = 0;
  for (let i = 0; i < values.length; i++) total += values[i];
  return total;
}

// ---- the measurements the tasks ask for, as host-side references -------------

// Peak magnitude over the half-spectrum (bins 0…n/2 — the rest is the mirror).
function peakOf(mag, half = N / 2) {
  let peak = 0;
  for (let k = 0; k <= half; k++) if (mag[k] > peak) peak = mag[k];
  return peak;
}

// Task 1: how many of bins 0…128 hold more than 1% of the peak.
function busyBins(mag, half = N / 2) {
  const limit = 0.01 * peakOf(mag, half);
  let count = 0;
  for (let k = 0; k <= half; k++) if (mag[k] > limit) count++;
  return count;
}

// Task 2: the step from each sample to the next IN THE REPEATED signal — index
// n - 1's neighbour is index 0, which is the seam the DFT actually sees.
function wrappedSteps(x) {
  return x.map((v, i) => x[(i + 1) % x.length] - v);
}

// Task 3: the worst leak, in dB, outside the tone's own neighbourhood. Bins
// 5…12 are the 8.5-cycle tone and its immediate skirt; everything else in the
// half-spectrum is leakage by construction.
const LEAK_LOW = 5;
const LEAK_HIGH = 12;

function leakDb(mag) {
  const peak = peakOf(mag);
  let worst = 0;
  for (let k = 0; k <= N / 2; k++) {
    if (k >= LEAK_LOW && k <= LEAK_HIGH) continue;
    if (mag[k] > worst) worst = mag[k];
  }
  return 20 * Math.log10(worst / peak);
}

// Task 4: main-lobe width and peak side-lobe level, read off a 4x zero-padded
// spectrum. Walking outward while the magnitude keeps falling lands exactly on
// the first null; everything past it is side lobe. Only the right-hand side is
// searched — the spectrum is symmetric about grid index NFFT / 2, and the tone's
// own negative-frequency image sits on the far side of that.
function analyseLobes(mag, peakAt = TRADE_BIN * OVER) {
  let i = peakAt;
  while (i + 1 < mag.length && mag[i + 1] < mag[i]) i++;
  const half = i - peakAt;
  let worst = 0;
  for (let k = peakAt + half; k <= NFFT / 2; k++) if (mag[k] > worst) worst = mag[k];
  return { bins: (2 * half) / OVER, db: 20 * Math.log10(worst / mag[peakAt]) };
}

// The rows of task 4's table, memoized: a 1,024-point transform over 256
// samples is 262 k inner iterations and several tests want the same answer.
let tradeTableCache = null;

function tradeTable() {
  if (!tradeTableCache) {
    const signal = tone(TRADE_BIN);
    tradeTableCache = {};
    for (const [name, w] of [
      ['rect', rectWindow],
      ['hann', hannWindow],
      ['blackman', blackmanWindow],
      // never offered to the learner — this row exists only as a near-miss probe
      ['blackmanFlipped', i =>
        0.42 - 0.5 * Math.cos((TAU * i) / N) - 0.08 * Math.cos((2 * TAU * i) / N)],
    ]) {
      tradeTableCache[name] = analyseLobes(dftMag(applyWindow(signal, w), NFFT));
    }
  }
  return tradeTableCache;
}

// ---- log reading -------------------------------------------------------------
//
// Every task here asks for ONE labelled line per signal or per window ("rect …",
// "hann …"), which is what lets a probe say WHICH row is wrong instead of "some
// number is missing somewhere". The first matching line wins.
function lineNumbers(logs, label) {
  const needle = label.toLowerCase();
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    if (!line.text.toLowerCase().includes(needle)) continue;
    const found = line.text.match(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi);
    return found ? found.map(Number) : [];
  }
  return null;
}

function hasNumber(nums, target, tol) {
  return Boolean(nums) && nums.some(v => Math.abs(v - target) <= tol);
}

// The one number on a line that could plausibly be the quantity under test —
// used to give diagnose() something to look at when the expected value is
// missing. NaN when there is no candidate, which makes every probe stay silent.
function pick(nums, test) {
  if (!nums) return NaN;
  const found = nums.filter(test);
  return found.length ? found[0] : NaN;
}

function pickLast(nums, test) {
  if (!nums) return NaN;
  const found = nums.filter(test);
  return found.length ? found[found.length - 1] : NaN;
}

// ---- kernel lookup -----------------------------------------------------------
//
// ctx.kernel is only the LAST kernel created, and every starter here builds one
// or more GIVEN kernels before the one the learner writes, so tests find their
// kernel by shape. `lastArgs` (recorded by the sandbox on every invocation) is
// what separates a window kernel from the given magnitude pass: both have
// output [256], but only one of them was handed a flat 256-number signal. No
// candidate is ever probed with an argument of a shape it has not already seen
// — that would force gpu.js to recompile mid-test.
function kernelWithOutput(ctx, dims) {
  return (
    ctx.kernels.find(k => {
      const out = k.kernel && k.kernel.output;
      return out && out.length === dims.length && dims.every((d, i) => out[i] === d);
    }) || null
  );
}

function takesFlatSignal(k) {
  const out = k.kernel && k.kernel.output;
  if (!out || out.length !== 1 || out[0] !== N) return false;
  const args = k.lastArgs;
  return (
    Array.isArray(args) &&
    args.length === 1 &&
    args[0] &&
    args[0].length === N &&
    typeof args[0][0] === 'number'
  );
}

// Task 5's calibrated pass: output [256], invoked with (spectrum, sumW).
function amplitudeKernel(ctx) {
  return (
    ctx.kernels
      .filter(k => {
        const out = k.kernel && k.kernel.output;
        const args = k.lastArgs;
        return (
          out && out.length === 1 && out[0] === N && Array.isArray(args) && args.length === 2
        );
      })
      .pop() || null
  );
}

// Window kernels, in creation order. Every starter creates the one the learner
// must write LAST, so the last entry is the one under test.
function windowKernels(ctx) {
  return ctx.kernels.filter(k => takesFlatSignal(k));
}

// The window kernel under test, plus its shape on a signal of 256 ones — which
// is the window itself, since every window kernel here is "sample x taper".
function windowUnderTest(ctx, what) {
  const candidates = windowKernels(ctx);
  if (!candidates.length) {
    return { error: `no kernel with output: [256] taking a 256-sample signal was found — ${what}` };
  }
  const kernel = candidates[candidates.length - 1];
  const shape = Array.from(kernel(ones()));
  return { kernel, shape };
}

// ---- near-miss diagnosis -----------------------------------------------------
//
// A failing assert that reports two numbers says nothing about WHICH slip
// produced them. A probe pairs the value one specific known mistake would
// produce with a sentence naming it; diagnose() speaks only when the observed
// value matches a probe within the test's own tolerance AND the correct value
// does not — so a case where two candidates coincide stays silent, as do
// observations matching probes that disagree with each other. A confident wrong
// diagnosis is worse than a plain numeric mismatch.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Where a candidate differs from the right answer at only a few indices — the
// periodic-versus-symmetric window is the whole reason this exists, since the
// two agree to 1e-4 over most of the window and part company by 9e-3 around
// i = 190 — one matching cell is worthless evidence. This form demands that a
// probe predict EVERY element and differ from the correct answer somewhere
// before it may speak.
function diagnoseAll(count, got, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (let i = 0; i < count; i++) {
        if (!(Math.abs(got(i) - value(i)) <= eps)) return false;
        if (Math.abs(expected(i) - value(i)) > eps) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 1: the four ways |X| = sqrt(re² + im²) goes wrong.
function magnitudeProbes(re, im) {
  return [
    [re * re + im * im, 'that is re² + im², the power — the magnitude is its square root'],
    [Math.abs(re), 'only the real plane was read — a bin is a complex number, and plane 1 holds the other half of it'],
    [Math.abs(im), 'only the imaginary plane was read — plane 0 holds the real part, plane 1 the imaginary part'],
    [Math.abs(re) + Math.abs(im), 'the two parts were added instead of combined as a hypotenuse: Math.sqrt(re * re + im * im)'],
  ];
}

// Task 2: the wrapped difference, mis-wrapped.
function stepProbes(x) {
  const n = x.length;
  return [
    [i => x[i] - x[(i + 1) % n],
      'the difference runs the wrong way — the step from sample i is signal[(i + 1) % 256] - signal[i]'],
    [i => x[(i + 1) % n],
      'the subtraction is missing — this is the next sample itself, not the step to it'],
    [i => (i === n - 1 ? 0 : x[i + 1] - x[i]),
      'the last cell came back 0 — you clamped where you should have wrapped. The transform has no "last sample": index 255 is followed by index 0 of the next copy, which is what (i + 1) % 256 says'],
  ];
}

// Tasks 3 and 4: what a window kernel returns when the formula slips. `signal`
// is whatever the test fed in, so these work on a constant AND on a real tone.
function hannProbes(signal) {
  return [
    [i => signal[i] * hannSymmetric(i),
      'that is the SYMMETRIC Hann window, cos(2πi / (n - 1)). It is the right window for designing a filter and the wrong one for analysing a spectrum — this course wants the periodic form, divisor n'],
    [i => signal[i] * (0.5 + 0.5 * Math.cos((TAU * i) / N)),
      'the sign is flipped: 0.5 + 0.5·cos peaks at both ENDS and pinches to zero in the middle, which is the exact opposite of a taper'],
    [i => signal[i] * (1 - Math.cos((TAU * i) / N)),
      'the window runs 0…2 instead of 0…1 — both halves of the raised cosine need their 0.5'],
    [i => hannWindow(i),
      'that is the window on its own — it never got multiplied by the signal'],
  ];
}

function blackmanProbes(signal) {
  return [
    [i => signal[i] * blackmanSymmetric(i),
      'that is the SYMMETRIC Blackman window, divisor n - 1. For spectral analysis use the periodic form, divisor n'],
    [i => signal[i] * (0.42 - 0.5 * Math.cos((TAU * i) / N) - 0.08 * Math.cos((2 * TAU * i) / N)),
      'the third term has the wrong sign — it is + 0.08·cos(4πi / n), and with a minus the main lobe stops being a whole number of bins wide and the side lobes climb 41 dB'],
    [i => signal[i] * (0.42 - 0.5 * Math.cos((TAU * i) / N) + 0.08 * Math.cos((TAU * i) / N)),
      'the third term runs at 2πi / n like the second — it is the DOUBLE frequency, 4πi / n'],
    [i => signal[i] * (0.54 - 0.46 * Math.cos((TAU * i) / N)),
      'those are the Hamming coefficients (0.54, 0.46) — Blackman is the three-term 0.42, 0.5, 0.08'],
    [i => blackmanWindow(i),
      'that is the window on its own — it never got multiplied by the signal'],
  ];
}

// The given transform, verbatim in four starters. Kept in one place so the
// prose, the hints and every starter agree character for character.
const SPECTRUM_KERNEL = `const spectrum = gpu.createKernel(function (x) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = (-2 * Math.PI * ((k * i) % this.constants.n)) / this.constants.n;
    re += x[i] * Math.cos(angle);
    im += x[i] * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [256, 2], constants: { n: 256 } });`;

const MAGNITUDE_KERNEL = `const magnitude = gpu.createKernel(function (spec) {
  const re = spec[0][this.thread.x];
  const im = spec[1][this.thread.x];
  return Math.sqrt(re * re + im * im);
}, { output: [256] });`;

const RECT_KERNEL = `// The rectangular window: multiply by 1 and stop dead at the edges. It is
// still a window — just the one that does nothing but chop.
const rect = gpu.createKernel(function (signal) {
  return signal[this.thread.x];
}, { output: [256] });`;

const HANN_KERNEL = `const hann = gpu.createKernel(function (signal) {
  const i = this.thread.x;
  const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / this.constants.n);
  return signal[i] * w;
}, { output: [256], constants: { n: 256 } });`;

export default {
  uuid: 'f563138d-cbb0-4aa5-b874-ff028b277677',
  version: 1,
  slug: 'windowing',
  title: 'Windowing & Spectral Leakage',
  blurb: 'Why the same tone looks clean or filthy depending only on how many samples you took — and what a window costs to fix it.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'leakage',
      title: 'A Tone the Window Does Not Fit',
      intro: `<p>Two tones, same amplitude, same length, the same 256-sample window. One
        difference: <code>onBin</code> completes exactly <strong>8</strong> cycles inside the
        window and <code>offBin</code> completes <strong>8.5</strong>. Transform both and one
        comes back as a single clean spike, while the other smears across the whole spectrum
        with a peak <em>lower</em> than the amplitude actually in the signal. That smear is
        <strong>spectral leakage</strong>, and this module exists so that it never fools
        you.</p>
        <p>The transform is written for you below — it is the naive DFT, and it follows this
        track's convention for complex numbers. gpu.js has no complex type, so a spectrum is
        <strong>two planes of floats</strong>: <code>output: [n, 2]</code>, indexed
        <code>result[p][k]</code>, where plane <code>p = 0</code> holds the real part of bin
        <code>k</code> and plane <code>p = 1</code> the imaginary part. (A 2D output
        <code>[w, h]</code> is indexed <code>[y][x]</code>, so <code>[n, 2]</code> is exactly
        <code>[plane][bin]</code>.) Your job is the pass after it: one thread per bin, turning
        two planes into one magnitude.</p>
        <p>Then measure both tones rather than taking my word for it. Two numbers each: the
        peak <strong>amplitude</strong> — a real cosine of amplitude <em>A</em> splits its
        energy between bin <em>k</em> and its mirror above Nyquist, so
        <code>A = 2·|X| / n</code> — and how many of the 129 bins up to Nyquist hold more than
        1% of that peak.</p>`,
      goal: `<strong>Goal:</strong> write the magnitude kernel, then log for each signal its
        peak amplitude and how many of bins 0…128 carry more than 1% of the peak.`,
      requirements: [
        'Create <code>magnitude</code> with <code>output: [256]</code> — one thread per bin',
        'Read both planes of your own bin: <code>spec[0][this.thread.x]</code> and <code>spec[1][this.thread.x]</code>',
        'Return <code>Math.sqrt(re * re + im * im)</code>',
        'Log one line per signal, labelled <code>onBin</code> / <code>offBin</code>, carrying the amplitude <code>2 * peak / 256</code> and the busy-bin count',
      ],
      hints: [
        {
          title: 'Hint 1 — which bin is mine?',
          body: `<p>With <code>output: [256]</code> there are 256 threads and
            <code>this.thread.x</code> is the bin number. The spectrum handed in is two rows:
            row 0 is every bin's real part, row 1 is every bin's imaginary part. So your own
            bin's two halves are <code>spec[0][this.thread.x]</code> and
            <code>spec[1][this.thread.x]</code>.</p>`,
        },
        {
          title: 'Hint 2 — the kernel body',
          body: `<pre><code>const re = spec[0][this.thread.x];
const im = spec[1][this.thread.x];
return Math.sqrt(re * re + im * im);</code></pre>`,
        },
        {
          title: 'Hint 3 — the measuring, in plain JavaScript',
          body: `<p>Only bins 0…128 matter: above Nyquist the spectrum of a real signal is
            just the mirror image.</p>
<pre><code>let peak = 0;
for (let k = 0; k &lt;= 128; k++) if (mag[k] &gt; peak) peak = mag[k];
let busy = 0;
for (let k = 0; k &lt;= 128; k++) if (mag[k] &gt; 0.01 * peak) busy++;</code></pre>`,
        },
      ],
      transfer: `Split real/imaginary planes are how every serious FFT ships: cuFFT and rocFFT
        expose both interleaved (<code>cufftComplex</code>) and planar layouts, WebGPU compute
        pipelines almost always pick planar because a storage buffer of <code>f32</code> is
        what the hardware wants, and the magnitude pass you just wrote is one
        <code>length()</code> call in WGSL or HLSL.`,
      starterCode: `// Two tones, one window. Only one of them fits.
const gpu = new GPU({ mode });

// GIVEN — the naive DFT, in this track's two-plane form:
//   output: [256, 2]  →  result[0][k] = real part of bin k
//                        result[1][k] = imaginary part of bin k
// (k * i) % n keeps every angle inside one turn: a float32 cos() has lost
// four digits by the time its argument reaches a thousand radians.
${SPECTRUM_KERNEL}

const magnitude = gpu.createKernel(function (spec) {
  // TODO: |X[k]| = sqrt(re² + im²), with re = spec[0][k] and im = spec[1][k].
  return 0;
}, { output: [256] });

function report(label, signal) {
  const mag = magnitude(spectrum(signal));
  // TODO: peak      = the largest magnitude in bins 0…128
  //       amplitude = 2 * peak / 256
  //       busy      = how many of bins 0…128 exceed 0.01 * peak
  console.log(label, 'amplitude:', 0, 'busy bins:', 0);
}

report('onBin ', onBin);
report('offBin', offBin);
`,
      solutionCode: `// Two tones, one window. Only one of them fits.
const gpu = new GPU({ mode });

// GIVEN — the naive DFT, in this track's two-plane form:
//   output: [256, 2]  →  result[0][k] = real part of bin k
//                        result[1][k] = imaginary part of bin k
// (k * i) % n keeps every angle inside one turn: a float32 cos() has lost
// four digits by the time its argument reaches a thousand radians.
${SPECTRUM_KERNEL}

${MAGNITUDE_KERNEL}

function report(label, signal) {
  const mag = magnitude(spectrum(signal));
  let peak = 0;
  for (let k = 0; k <= 128; k++) if (mag[k] > peak) peak = mag[k];
  let busy = 0;
  for (let k = 0; k <= 128; k++) if (mag[k] > 0.01 * peak) busy++;
  console.log(label, 'amplitude:', 2 * peak / 256, 'busy bins:', busy);
}

report('onBin ', onBin);
report('offBin', offBin);
`,
      inputs: () => ({ onBin: tone(ON_BIN), offBin: tone(OFF_BIN) }),
      publicTests: [
        {
          name: 'a magnitude kernel with one thread per bin',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(
              kernelWithOutput(ctx, [N]),
              'no kernel with output: [256] found — the magnitude pass wants one thread per bin'
            );
          },
        },
        {
          name: 'every bin is <code>sqrt(re² + im²)</code> of its two planes',
          run: async ctx => {
            const magnitude = kernelWithOutput(ctx, [N]);
            ctx.assert(magnitude, 'no kernel with output: [256] found');
            // Both backends hand a 2D kernel's result back as Array<Float32Array>,
            // so the synthetic spectrum is built the same way: one container type
            // per kernel, or gpu.js recompiles mid-run.
            const re = new Float32Array(N);
            const im = new Float32Array(N);
            for (let k = 0; k < N; k++) {
              re[k] = ((k * 7) % 11) - 4.75;
              im[k] = ((k * 5) % 13) - 5.5;
            }
            const out = magnitude([re, im]);
            ctx.assert(out && out.length === N, `expected 256 magnitudes, got ${out && out.length}`);
            for (const k of [0, 1, 3, 17, 64, 128, 255]) {
              const expected = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
              const hint = diagnose(out[k], expected, 1e-3, magnitudeProbes(re[k], im[k]));
              ctx.assertClose(out[k], expected, 1e-3, hint || `bin ${k}`);
            }
          },
        },
        {
          name: 'the tone that fits reads amplitude <code>1.00</code>, in one bin',
          run: async ctx => {
            const mag = dftMag(tone(ON_BIN));
            const nums = lineNumbers(ctx.logs, 'onBin');
            ctx.assert(nums, 'no console.log line mentioning "onBin" — log one labelled line per signal');
            const amp = (2 * peakOf(mag)) / N;
            ctx.assert(
              hasNumber(nums, amp, 5e-3),
              `the onBin line should carry its peak amplitude ≈${amp.toFixed(3)} (2 * peak / 256), got [${nums.join(', ')}]`
            );
            ctx.assert(
              hasNumber(nums, busyBins(mag), 0.5),
              `the onBin line should carry its busy-bin count, ${busyBins(mag)} — a tone with a whole number of cycles occupies exactly one bin`
            );
          },
        },
        {
          name: 'half a bin off costs a third of the amplitude and fills every bin',
          run: async ctx => {
            const mag = dftMag(tone(OFF_BIN));
            const nums = lineNumbers(ctx.logs, 'offBin');
            ctx.assert(nums, 'no console.log line mentioning "offBin" — log one labelled line per signal');
            const amp = (2 * peakOf(mag)) / N;
            const hint = diagnose(pick(nums, v => v > 0.05 && v < 2), amp, 5e-3, [
              [(2 * peakOf(dftMag(tone(ON_BIN)))) / N,
                'that is the onBin answer — check that report() transforms the signal it was handed rather than a captured one'],
            ]);
            ctx.assert(
              hasNumber(nums, amp, 5e-3),
              hint || `the offBin line should carry its peak amplitude ≈${amp.toFixed(3)} — lower than the 1.00 that is actually in the signal, which is the whole point`
            );
            ctx.assert(
              hasNumber(nums, busyBins(mag), 0.5),
              `the offBin line should carry its busy-bin count, ${busyBins(mag)} — every bin up to Nyquist holds more than 1% of the peak`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const magnitude = kernelWithOutput(ctx, [N]);
            ctx.assert(magnitude, 'no kernel with output: [256] found');
            const re = new Float32Array(N);
            const im = new Float32Array(N);
            for (let k = 0; k < N; k++) {
              re[k] = Math.cos(k * 0.37) * 3 + 0.6;
              im[k] = Math.sin(k * 0.11) * 5 - 0.9;
            }
            const out = magnitude([re, im]);
            for (let k = 0; k < N; k++) {
              const expected = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
              const hint = diagnose(out[k], expected, 1e-3, magnitudeProbes(re[k], im[k]));
              ctx.assertClose(out[k], expected, 1e-3, hint || `bin ${k}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // End to end against the reference: the learner's magnitude pass over
            // the given transform must reproduce |X| for a fresh tone.
            const magnitude = kernelWithOutput(ctx, [N]);
            const spectrum = kernelWithOutput(ctx, [N, 2]);
            ctx.assert(magnitude && spectrum, 'expected the given spectrum kernel and a magnitude kernel');
            const signal = tone(11.25);
            const got = magnitude(spectrum(signal));
            const expected = dftMag(signal);
            for (let k = 0; k <= N / 2; k++) {
              ctx.assertClose(got[k], expected[k], 2e-2, `bin ${k} of a fresh 11.25-cycle tone`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'periodic-extension',
      title: 'What the Transform Actually Sees',
      intro: `<p>Why should half a cycle wreck a spectrum? Because every basis function the DFT
        measures against completes a whole number of cycles in the window. What it can
        represent exactly, then, is a signal that <strong>repeats with period n</strong> — so
        that is what it assumes you gave it. Hand it 256 samples and it does not see a
        256-sample excerpt of something longer. It sees those 256 samples tiled end to end,
        forever.</p>
        <p>Tile a tone with 8 whole cycles and the copies join invisibly: sample 255 runs into
        sample 0 exactly as if the cosine had never stopped. Tile one with 8.5 and every join
        is a cliff — the wave ends near the bottom and restarts at the top, half a cycle
        skipped. A cliff is broadband: no small set of smooth sinusoids can build a step, so
        the transform pays for it with a little energy in every bin it has. That is where the
        smear in the last task came from. It was never in your signal; the transform put it
        there, faithfully describing a discontinuity you never intended.</p>
        <p>You can measure that cliff without transforming anything, and one kernel does it:
        the step from each sample to the next <em>in the repeated signal</em>. The wrap is the
        whole point — sample 255's neighbour is sample 0 of the next copy.</p>`,
      goal: `<strong>Goal:</strong> write the wrapped-difference kernel and log, for both
        tones, the size of the step at the seam and the largest step anywhere in the window.`,
      requirements: [
        '<code>output: [256]</code> — cell <code>i</code> holds <code>signal[(i + 1) % 256] - signal[i]</code>',
        'The last cell wraps rather than clamping: sample 255 is followed by sample 0',
        'Log one line per signal, labelled <code>onBin</code> / <code>offBin</code>, carrying <code>Math.abs(step[255])</code> and the largest <code>Math.abs(step[i])</code> anywhere',
      ],
      hints: [
        {
          title: 'Hint 1 — the wrap',
          body: `<p><code>%</code> works inside a kernel (it transpiles to GLSL's
            <code>mod()</code>), so the neighbour of sample <code>i</code> in the repeated
            signal is <code>signal[(i + 1) % 256]</code>. At <code>i = 255</code> that is
            <code>signal[0]</code> — the seam.</p>`,
        },
        {
          title: 'Hint 2 — the kernel body',
          body: `<pre><code>const i = this.thread.x;
return signal[(i + 1) % this.constants.n] - signal[i];</code></pre>`,
        },
        {
          title: 'Hint 3 — what you should see',
          body: `<p>For the tone that fits, the seam step is the <em>smallest</em> kind of
            step in the window — about a tenth of the biggest. For the one that does not, the
            seam step <em>is</em> the biggest step, ten times anything the wave does inside
            the window. That factor of a hundred between the two seams is the leakage, before
            you have transformed anything at all.</p>`,
        },
      ],
      transfer: `Circular boundaries are the default in frequency-domain work everywhere:
        <code>cufftExecR2C</code>, WGSL FFT compute passes and every convolution-theorem
        implementation treat the buffer as a ring. It is the same wraparound the Cellular
        Automata module uses for a toroidal grid — and exactly why frequency-domain
        convolution has to be zero-padded before use, or the tail of the filter wraps round
        and lands on the beginning of the signal.`,
      starterCode: `// The DFT tiles your window end to end. Measure the seam.
const gpu = new GPU({ mode });

const wrappedStep = gpu.createKernel(function (signal) {
  const i = this.thread.x;
  // TODO: the step from sample i to the next sample IN THE REPEATED signal.
  // Sample 255's neighbour is sample 0 — use % this.constants.n.
  return 0;
}, { output: [256], constants: { n: 256 } });

function report(label, signal) {
  const step = wrappedStep(signal);
  // TODO: seam    = Math.abs(step[255])
  //       biggest = the largest Math.abs(step[i]) over all 256 cells
  console.log(label, 'seam:', 0, 'biggest step:', 0);
}

report('onBin ', onBin);
report('offBin', offBin);
`,
      solutionCode: `// The DFT tiles your window end to end. Measure the seam.
const gpu = new GPU({ mode });

const wrappedStep = gpu.createKernel(function (signal) {
  const i = this.thread.x;
  return signal[(i + 1) % this.constants.n] - signal[i];
}, { output: [256], constants: { n: 256 } });

function report(label, signal) {
  const step = wrappedStep(signal);
  const seam = Math.abs(step[255]);
  let biggest = 0;
  for (let i = 0; i < step.length; i++) {
    if (Math.abs(step[i]) > biggest) biggest = Math.abs(step[i]);
  }
  console.log(label, 'seam:', seam, 'biggest step:', biggest);
}

report('onBin ', onBin);
report('offBin', offBin);
`,
      inputs: () => ({ onBin: tone(ON_BIN), offBin: tone(OFF_BIN) }),
      publicTests: [
        {
          name: 'cell <code>i</code> holds <code>signal[(i + 1) % 256] - signal[i]</code>',
          run: async ctx => {
            const step = kernelWithOutput(ctx, [N]);
            ctx.assert(step, 'no kernel with output: [256] found — one thread per sample');
            // A ramp puts wrap, clamp and sign-flip as far apart as they go:
            // every interior step is +1/256 and the seam is -255/256.
            const ramp = new Array(N);
            for (let i = 0; i < N; i++) ramp[i] = i / N;
            const out = step(ramp);
            ctx.assert(out && out.length === N, `expected 256 steps, got ${out && out.length}`);
            const expected = wrappedSteps(ramp);
            const hint = diagnoseAll(N, i => out[i], i => expected[i], 1e-4, stepProbes(ramp));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], expected[i], 1e-4, hint ||
                (i === N - 1
                  ? 'cell 255 is the seam — sample 255 is followed by sample 0 of the next copy, not by nothing'
                  : `cell ${i}`));
            }
          },
        },
        {
          name: 'the tone that fits has no seam',
          run: async ctx => {
            const steps = wrappedSteps(tone(ON_BIN));
            const seam = Math.abs(steps[N - 1]);
            const biggest = Math.max(...steps.map(Math.abs));
            const nums = lineNumbers(ctx.logs, 'onBin');
            ctx.assert(nums, 'no console.log line mentioning "onBin" — log one labelled line per signal');
            ctx.assert(
              hasNumber(nums, seam, 2e-3),
              `the onBin line should carry its seam step ≈${seam.toFixed(4)}, got [${nums.join(', ')}]`
            );
            ctx.assert(
              hasNumber(nums, biggest, 2e-3),
              `the onBin line should carry its biggest step ≈${biggest.toFixed(4)} — ten times the seam, which is what "no seam" looks like`
            );
          },
        },
        {
          name: 'half a bin off, the seam <em>is</em> the biggest step',
          run: async ctx => {
            const steps = wrappedSteps(tone(OFF_BIN));
            const seam = Math.abs(steps[N - 1]);
            let inside = 0;
            for (let i = 0; i < N - 1; i++) inside = Math.max(inside, Math.abs(steps[i]));
            const nums = lineNumbers(ctx.logs, 'offBin');
            ctx.assert(nums, 'no console.log line mentioning "offBin" — log one labelled line per signal');
            const hint = diagnose(pick(nums, v => v > 0), seam, 2e-3, [
              [inside, 'that is the biggest step INSIDE the window — the seam is cell 255, the step from the last sample round to the first'],
            ]);
            ctx.assert(
              hasNumber(nums, seam, 2e-3),
              hint || `the offBin line should carry its seam step ≈${seam.toFixed(4)} — about a hundred times the onBin seam`
            );
            ctx.assert(
              nums.filter(v => Math.abs(v - seam) <= 2e-3).length >= 2,
              `the offBin line should carry its biggest step too, and it is the same ≈${seam.toFixed(4)} — here the seam and the biggest step are the same cell`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const step = kernelWithOutput(ctx, [N]);
            ctx.assert(step, 'no kernel with output: [256] found');
            const signal = tone(6.25, 0.8);
            const out = step(signal);
            const expected = wrappedSteps(signal);
            const hint = diagnoseAll(N, i => out[i], i => expected[i], 1e-4, stepProbes(signal));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], expected[i], 1e-4, hint || `cell ${i}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Whatever the signal, the 256 wrapped steps must sum to zero: they
            // walk once round the circle and back to where they started. Only a
            // real wrap can manage that.
            const step = kernelWithOutput(ctx, [N]);
            ctx.assert(step, 'no kernel with output: [256] found');
            const signal = tone(3.5, 1.4);
            const out = step(signal);
            ctx.assertClose(sumOf(Array.from(out)), 0, 1e-3,
              'the 256 wrapped steps should sum to 0 — they walk once round the circle and back to the start, which only works if the last cell wraps');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'hann-window',
      title: 'Taper the Edges',
      intro: `<p>You cannot make an arbitrary signal fit the window: you rarely know its
        frequency in advance, and a real signal has many at once. What you <em>can</em> do is
        make it end where it starts. Multiply every sample by a taper that falls to zero at
        both edges, and the tiled copies join at zero with no cliff left for the transform to
        explain.</p>
        <p>The <strong>Hann window</strong> is the workhorse:
        <code>w[i] = 0.5 - 0.5·cos(2πi / n)</code>, a single raised cosine. One multiply per
        sample, no neighbours, no accumulation — this is the most trivially parallel kernel in
        the course, and on a GPU it is free next to the transform that follows it.</p>
        <p>One honest footnote, because you will meet both spellings. The <em>symmetric</em>
        window divides by <code>n - 1</code> and is genuinely zero at both ends; it is the
        right choice when the window is a filter's impulse response. The <em>periodic</em>
        (DFT-even) window divides by <code>n</code>, so its <code>n</code> samples are exactly
        one period of the cosine — which is what makes a windowed on-bin tone land on three
        bins and nothing else. The two differ by under one part in a hundred and the argument
        is decades old, but for spectral analysis the answer is not in doubt: divide by
        <code>n</code>.</p>
        <p>And nothing is free. The taper throws most of the signal away near the edges, so
        the measured peak comes back <em>smaller</em> — worse than before, until task 5 gives
        it back — and it <strong>widens the main lobe</strong>: one sharp bin becomes three. A
        window buys dynamic range and pays for it in resolution. The next task puts numbers on
        that trade.</p>`,
      goal: `<strong>Goal:</strong> write the Hann window kernel, then measure the worst leak
        with no window and with Hann on the same 8.5-cycle tone.`,
      requirements: [
        'Create <code>hann</code> with <code>output: [256]</code>, returning <code>signal[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / 256))</code>',
        'Window the signal <em>before</em> the transform — a taper applied to a finished spectrum is a different, and useless, operation',
        'Leak = <code>20 * Math.log10(worst / peak)</code>, where <code>peak</code> is the largest magnitude in bins 0…128 and <code>worst</code> the largest outside bins 5…12',
        'Log one line labelled <code>rect</code> and one labelled <code>hann</code>, each with its leak in dB, and put the Hann peak amplitude on the Hann line',
      ],
      hints: [
        {
          title: 'Hint 1 — the kernel body',
          body: `<p><code>Math.PI</code> and <code>Math.cos</code> both work inside kernels,
            and <code>this.constants.n</code> is already wired up:</p>
<pre><code>const i = this.thread.x;
const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / this.constants.n);
return signal[i] * w;</code></pre>`,
        },
        {
          title: 'Hint 2 — and then use it',
          body: `<p>The taper goes on the samples, so it has to run <em>before</em> the
            transform sees them:</p>
<pre><code>const tapered = measure(magnitude(spectrum(hann(offBin))));</code></pre>
<p>If the number you get back is the rectangular one again, the windowed signal
            never reached <code>spectrum</code>.</p>`,
        },
        {
          title: 'Hint 3 — what you should see',
          body: `<p>The leak drops from about <code>-18 dB</code> to about
            <code>-47 dB</code>: the worst stray bin goes from an eighth of the peak to a
            four-hundredth of it. And the Hann peak amplitude comes back around
            <code>0.42</code> — <em>further</em> from the true 1.00 than the unwindowed 0.65
            was. That is not a mistake; it is the window's coherent gain, and task 5 divides
            it out.</p>`,
        },
      ],
      transfer: `Every real analyser windows: an <code>AnalyserNode</code> in the Web Audio API
        applies a Blackman window before its FFT and does not offer you the choice, numpy ships
        <code>hanning</code>/<code>hamming</code>/<code>blackman</code> as one-liners, and
        MATLAB's <code>pwelch</code> defaults to Hamming. On a GPU the taper is a per-element
        multiply you fuse into whatever pass already touches the samples — a
        <code>thrust::transform</code> in CUDA, two lines at the top of the load in WGSL.`,
      starterCode: `// A taper the signal can end on. Then transform it.
const gpu = new GPU({ mode });

// GIVEN — the same two-plane DFT and magnitude pass as task 1.
${SPECTRUM_KERNEL}

${MAGNITUDE_KERNEL}

${RECT_KERNEL}

const hann = gpu.createKernel(function (signal) {
  const i = this.thread.x;
  // TODO: multiply this sample by the Hann taper
  //   w = 0.5 - 0.5 * cos(2π i / n)      ← divisor n, not n - 1
  return signal[i];
}, { output: [256], constants: { n: 256 } });

function measure(mag) {
  let peak = 0;
  for (let k = 0; k <= 128; k++) if (mag[k] > peak) peak = mag[k];
  let worst = 0;
  for (let k = 0; k <= 128; k++) {
    if (k >= 5 && k <= 12) continue;
    if (mag[k] > worst) worst = mag[k];
  }
  return { db: 20 * Math.log10(worst / peak), amplitude: 2 * peak / 256 };
}

const plain = measure(magnitude(spectrum(rect(offBin))));
console.log('rect  leak:', plain.db, 'dB');

// TODO: taper offBin with hann FIRST, then transform the windowed samples.
const tapered = measure(magnitude(spectrum(rect(offBin))));
console.log('hann  leak:', tapered.db, 'dB   amplitude:', tapered.amplitude);
`,
      solutionCode: `// A taper the signal can end on. Then transform it.
const gpu = new GPU({ mode });

// GIVEN — the same two-plane DFT and magnitude pass as task 1.
${SPECTRUM_KERNEL}

${MAGNITUDE_KERNEL}

${RECT_KERNEL}

${HANN_KERNEL}

function measure(mag) {
  let peak = 0;
  for (let k = 0; k <= 128; k++) if (mag[k] > peak) peak = mag[k];
  let worst = 0;
  for (let k = 0; k <= 128; k++) {
    if (k >= 5 && k <= 12) continue;
    if (mag[k] > worst) worst = mag[k];
  }
  return { db: 20 * Math.log10(worst / peak), amplitude: 2 * peak / 256 };
}

const plain = measure(magnitude(spectrum(rect(offBin))));
console.log('rect  leak:', plain.db, 'dB');

// The taper goes on the SAMPLES, before the transform.
const tapered = measure(magnitude(spectrum(hann(offBin))));
console.log('hann  leak:', tapered.db, 'dB   amplitude:', tapered.amplitude);
`,
      inputs: () => ({ offBin: tone(OFF_BIN) }),
      publicTests: [
        {
          name: 'the window kernel is the Hann taper',
          run: async ctx => {
            const found = windowUnderTest(ctx, 'the Hann taper needs one thread per sample');
            ctx.assert(!found.error, found.error);
            ctx.assert(
              Math.abs(found.shape[0] - 1) > 1e-6 || Math.abs(found.shape[N / 2] - 1) > 1e-6,
              'no tapered window has run: the last window kernel to be invoked returned its signal unchanged, which is the rectangular window. Write the Hann taper, and make sure it actually runs on offBin'
            );
            const flat = ones();
            const hint = diagnoseAll(N, i => found.shape[i], i => hannWindow(i), 1e-4,
              hannProbes(flat));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(found.shape[i], hannWindow(i), 1e-4, hint ||
                `window sample ${i} — on a signal of 256 ones a window kernel returns its own shape`);
            }
          },
        },
        {
          name: 'the taper multiplies the signal, not just itself',
          run: async ctx => {
            const found = windowUnderTest(ctx, 'the Hann taper needs one thread per sample');
            ctx.assert(!found.error, found.error);
            const signal = tone(5.5, 0.9);
            const out = found.kernel(signal);
            const hint = diagnoseAll(N, i => out[i], i => signal[i] * hannWindow(i), 1e-4,
              hannProbes(signal));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], signal[i] * hannWindow(i), 1e-4, hint || `sample ${i}`);
            }
          },
        },
        {
          name: 'the unwindowed leak is about <code>-18 dB</code>',
          run: async ctx => {
            const expected = leakDb(dftMag(tone(OFF_BIN)));
            const nums = lineNumbers(ctx.logs, 'rect');
            ctx.assert(nums, 'no console.log line mentioning "rect" — log one labelled line per window');
            ctx.assert(
              hasNumber(nums, expected, 0.3),
              `the rect line should carry the unwindowed leak ≈${expected.toFixed(2)} dB, got [${nums.join(', ')}]`
            );
          },
        },
        {
          name: 'Hann drops the leak to about <code>-47 dB</code>',
          run: async ctx => {
            const rectMag = dftMag(tone(OFF_BIN));
            const hannMag = dftMag(applyWindow(tone(OFF_BIN), hannWindow));
            const expected = leakDb(hannMag);
            const nums = lineNumbers(ctx.logs, 'hann');
            ctx.assert(nums, 'no console.log line mentioning "hann" — log one labelled line per window');
            // The two ways a taper misses the transform, both exactly computable.
            const hint = diagnose(pick(nums, v => v < -0.001), expected, 0.3, [
              [leakDb(rectMag),
                'that is the unwindowed number again — the windowed samples never reached the transform. hann(offBin) returns a new signal, and it is that return value the DFT has to see'],
              [leakDb(rectMag.map((v, k) => v * hannWindow(k))),
                'the taper landed on the finished spectrum instead of on the samples. A window multiplies the signal BEFORE the DFT; scaling bins afterwards only reshapes the leak you already have'],
            ]);
            ctx.assert(
              hasNumber(nums, expected, 0.3),
              hint || `the hann line should carry a leak of ≈${expected.toFixed(2)} dB, got [${nums.join(', ')}]`
            );
            const amp = (2 * peakOf(hannMag)) / N;
            ctx.assert(
              hasNumber(nums, amp, 5e-3),
              `the hann line should also carry the windowed peak amplitude ≈${amp.toFixed(3)} — lower than the unwindowed 0.655, which task 5 explains and repairs`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const found = windowUnderTest(ctx, 'the Hann taper needs one thread per sample');
            ctx.assert(!found.error, found.error);
            const signal = tone(9.25, 1.3);
            const out = found.kernel(signal);
            const hint = diagnoseAll(N, i => out[i], i => signal[i] * hannWindow(i), 1e-4,
              hannProbes(signal));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], signal[i] * hannWindow(i), 1e-4, hint || `sample ${i}`);
            }
            // The window's own sum is what makes task 5 work, so pin it here: a
            // correctly shaped but rescaled taper cannot slip through.
            ctx.assertClose(sumOf(found.shape), N / 2, 0.05,
              'a periodic Hann window sums to exactly n / 2 = 128');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'window-trade',
      title: 'Measure the Trade',
      intro: `<p>"Use a Hann window" is folklore until you can say what it costs. Two numbers
        settle it, and both are in the picture above.</p>
        <p><strong>Main-lobe width</strong> is how many bins a single pure tone occupies — the
        frequency resolution you have left. <strong>Peak side-lobe level</strong> is how far
        down, in dB, the worst ripple outside that lobe sits — the dynamic range you have
        bought. The rectangular window (which is all "no window" ever meant: multiply by 1 and
        stop dead at the edges) wins the first and loses the second, badly. Every other window
        trades one for the other, and choosing between them is the whole skill.</p>
        <p>To see either you have to look <em>between</em> the bins, because a windowed on-bin
        tone lands on integer bins where these curves are exactly zero. The standard trick is
        <strong>zero padding</strong>: run a 1,024-point transform over a 256-sample signal, so
        768 of the terms are zero. Nothing is added and nothing is lost, but the spectrum is now
        sampled four times per bin. Be clear about what that buys — interpolation, never
        resolution. Zero padding draws the same curve with more dots; only a longer window makes
        the curve narrower.</p>`,
      goal: `<strong>Goal:</strong> write the Blackman window, then tabulate main-lobe width
        and peak side-lobe level for rectangular, Hann and Blackman on the same on-bin tone.`,
      requirements: [
        'Create <code>blackman</code> with <code>output: [256]</code>: the sample times <code>0.42 - 0.5·cos(2πi / n) + 0.08·cos(4πi / n)</code>',
        'Main lobe: from the peak at grid index 128, step outward while the magnitude keeps falling — that stops on the first null. Width in bins is <code>2 * steps / 4</code>',
        'Peak side lobe: <code>20 * Math.log10(worst / peak)</code>, with <code>worst</code> the largest magnitude from that null out to grid index 512',
        'Log one line per window, labelled <code>rect</code> / <code>hann</code> / <code>blackman</code>, carrying its width in bins and its side-lobe level in dB',
      ],
      hints: [
        {
          title: 'Hint 1 — the Blackman kernel',
          body: `<p>Three cosine terms, and the third runs at double the frequency — so name
            the angle once and the whole window is one line:</p>
<pre><code>const i = this.thread.x;
const a = 2 * Math.PI * i / this.constants.n;
const w = 0.42 - 0.5 * Math.cos(a) + 0.08 * Math.cos(2 * a);
return signal[i] * w;</code></pre>
<p>One gpu.js trap while you are in here: do not name a kernel local after one of
            your own constants. <code>const n = this.constants.n;</code> transpiles to
            <code>const constants_n = constants_n;</code> and the kernel dies with "cannot
            access before initialization".</p>`,
        },
        {
          title: 'Hint 2 — walking out to the first null',
          body: `<p>A main lobe falls monotonically to its first zero, and only then do the
            side lobes start climbing again. So keep stepping while the next sample is smaller
            than this one:</p>
<pre><code>let i = peakAt;
while (mag[i + 1] &lt; mag[i]) i++;
const bins = 2 * (i - peakAt) / 4;   // 4 grid points per bin</code></pre>`,
        },
        {
          title: 'Hint 3 — what the table should say',
          body: `<p>Widths of <code>2</code>, <code>4</code> and <code>6</code> bins, and side
            lobes at roughly <code>-13</code>, <code>-32</code> and <code>-58 dB</code>. Read
            it as a menu: three extra bins of width buy 45 dB of dynamic range. Hamming, not
            asked for here, sits between Hann and Blackman at 4 bins and about
            <code>-41 dB</code>.</p>`,
        },
      ],
      transfer: `Those two numbers are the datasheet entry for every window in every DSP
        library — Harris's 1978 survey tabulates a few dozen of them and is still the paper
        people cite. Choosing one is a real engineering decision: a spectrum analyser hunting a
        -60 dB spur beside a strong carrier needs Blackman's side lobes and can afford the
        width, while a pitch tracker separating two close partials wants Hann, or no window at
        all.`,
      starterCode: `// Two numbers decide which window you want. Measure both.
const gpu = new GPU({ mode });

// GIVEN — a 1024-point transform over the same 256 samples: 768 implied zeros,
// so the spectrum comes out sampled 4x per bin. Interpolation, not resolution.
const paddedSpectrum = gpu.createKernel(function (x) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = (-2 * Math.PI * ((k * i) % this.constants.nfft)) / this.constants.nfft;
    re += x[i] * Math.cos(angle);
    im += x[i] * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [1024, 2], constants: { n: 256, nfft: 1024 } });

const paddedMagnitude = gpu.createKernel(function (spec) {
  const re = spec[0][this.thread.x];
  const im = spec[1][this.thread.x];
  return Math.sqrt(re * re + im * im);
}, { output: [1024] });

${RECT_KERNEL}

// GIVEN — the Hann window from the last task.
${HANN_KERNEL}

const blackman = gpu.createKernel(function (signal) {
  const i = this.thread.x;
  // TODO: w = 0.42 - 0.5·cos(2πi/n) + 0.08·cos(4πi/n), times the sample
  return signal[i];
}, { output: [256], constants: { n: 256 } });

function analyse(windowed) {
  const mag = paddedMagnitude(paddedSpectrum(windowed));
  const peakAt = 32 * 4;        // the tone sits on bin 32, 4 grid points per bin
  const peak = mag[peakAt];

  // TODO 1: step out from peakAt while the next magnitude is smaller than this
  //         one. Where you stop is the first null.
  let i = peakAt;

  // TODO 2: main-lobe width in bins = 2 * (i - peakAt) / 4
  const bins = 0;

  // TODO 3: worst = the largest magnitude from i out to grid index 512
  let worst = 0;

  return { bins, db: 20 * Math.log10(worst / peak) };
}

for (const [label, windowed] of [
  ['rect    ', rect(signal)],
  ['hann    ', hann(signal)],
  ['blackman', blackman(signal)],
]) {
  const r = analyse(windowed);
  console.log(label, 'main lobe:', r.bins, 'bins   peak side lobe:', r.db, 'dB');
}
`,
      solutionCode: `// Two numbers decide which window you want. Measure both.
const gpu = new GPU({ mode });

// GIVEN — a 1024-point transform over the same 256 samples: 768 implied zeros,
// so the spectrum comes out sampled 4x per bin. Interpolation, not resolution.
const paddedSpectrum = gpu.createKernel(function (x) {
  const k = this.thread.x;
  let re = 0;
  let im = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = (-2 * Math.PI * ((k * i) % this.constants.nfft)) / this.constants.nfft;
    re += x[i] * Math.cos(angle);
    im += x[i] * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, { output: [1024, 2], constants: { n: 256, nfft: 1024 } });

const paddedMagnitude = gpu.createKernel(function (spec) {
  const re = spec[0][this.thread.x];
  const im = spec[1][this.thread.x];
  return Math.sqrt(re * re + im * im);
}, { output: [1024] });

${RECT_KERNEL}

// GIVEN — the Hann window from the last task.
${HANN_KERNEL}

const blackman = gpu.createKernel(function (signal) {
  const i = this.thread.x;
  const a = 2 * Math.PI * i / this.constants.n;
  const w = 0.42 - 0.5 * Math.cos(a) + 0.08 * Math.cos(2 * a);
  return signal[i] * w;
}, { output: [256], constants: { n: 256 } });

function analyse(windowed) {
  const mag = paddedMagnitude(paddedSpectrum(windowed));
  const peakAt = 32 * 4;        // the tone sits on bin 32, 4 grid points per bin
  const peak = mag[peakAt];

  let i = peakAt;
  while (i + 1 < mag.length && mag[i + 1] < mag[i]) i++;

  const bins = 2 * (i - peakAt) / 4;

  let worst = 0;
  for (let k = i; k <= 512; k++) if (mag[k] > worst) worst = mag[k];

  return { bins, db: 20 * Math.log10(worst / peak) };
}

for (const [label, windowed] of [
  ['rect    ', rect(signal)],
  ['hann    ', hann(signal)],
  ['blackman', blackman(signal)],
]) {
  const r = analyse(windowed);
  console.log(label, 'main lobe:', r.bins, 'bins   peak side lobe:', r.db, 'dB');
}
`,
      inputs: () => ({ signal: tone(TRADE_BIN) }),
      publicTests: [
        {
          name: 'the Blackman kernel is the three-term window',
          run: async ctx => {
            const found = windowUnderTest(ctx, 'the Blackman taper needs one thread per sample');
            ctx.assert(!found.error, found.error);
            ctx.assert(
              Math.abs(found.shape[N / 4] - 1) > 1e-6,
              'the last window kernel still returns the sample unchanged — the Blackman taper has not been written yet'
            );
            const flat = ones();
            const hint = diagnoseAll(N, i => found.shape[i], i => blackmanWindow(i), 1e-4,
              blackmanProbes(flat));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(found.shape[i], blackmanWindow(i), 1e-4, hint ||
                `window sample ${i} — on a signal of 256 ones a window kernel returns its own shape, and Blackman is 0.34 a quarter of the way in`);
            }
          },
        },
        {
          name: 'Blackman multiplies the signal it is given',
          run: async ctx => {
            const found = windowUnderTest(ctx, 'the Blackman taper needs one thread per sample');
            ctx.assert(!found.error, found.error);
            const signal = tone(7.5, 1.2);
            const out = found.kernel(signal);
            const hint = diagnoseAll(N, i => out[i], i => signal[i] * blackmanWindow(i), 1e-4,
              blackmanProbes(signal));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], signal[i] * blackmanWindow(i), 1e-4, hint || `sample ${i}`);
            }
          },
        },
        {
          name: 'rectangular: <code>2</code> bins wide, side lobes only <code>-13 dB</code> down',
          run: async ctx => {
            const table = tradeTable();
            const nums = lineNumbers(ctx.logs, 'rect');
            ctx.assert(nums, 'no console.log line mentioning "rect" — log one labelled line per window');
            const widthHint = diagnose(pick(nums, v => v > 0), table.rect.bins, 0.05, [
              [table.rect.bins / 2, 'that is the HALF-width — the main lobe runs to the first null on both sides, so the width is 2 * steps / 4'],
              [table.rect.bins * OVER, 'that is the width in grid points, not bins — divide by the 4x zero-padding factor'],
            ]);
            ctx.assert(hasNumber(nums, table.rect.bins, 0.05),
              widthHint || `the rect line should say ${table.rect.bins} bins of main lobe, got [${nums.join(', ')}]`);
            ctx.assert(hasNumber(nums, table.rect.db, 0.4),
              `the rect line should carry a peak side lobe of ≈${table.rect.db.toFixed(2)} dB`);
          },
        },
        {
          name: 'Hann and Blackman buy dynamic range with width',
          run: async ctx => {
            const table = tradeTable();
            for (const label of ['hann', 'blackman']) {
              const row = table[label];
              const nums = lineNumbers(ctx.logs, label);
              ctx.assert(nums, `no console.log line mentioning "${label}" — log one labelled line per window`);
              const widthHint = diagnose(pick(nums, v => v > 0), row.bins, 0.05, [
                [row.bins / 2, 'that is the HALF-width — the main lobe runs to the first null on both sides, so the width is 2 * steps / 4'],
                [row.bins * OVER, 'that is the width in grid points, not bins — divide by the 4x zero-padding factor'],
              ]);
              ctx.assert(hasNumber(nums, row.bins, 0.05),
                widthHint || `the ${label} line should say ${row.bins} bins of main lobe, got [${nums.join(', ')}]`);
              const dbHint = label === 'blackman'
                ? diagnose(pick(nums, v => v < -1), row.db, 0.4, [
                    [table.blackmanFlipped.db,
                      'that side-lobe level is the signature of a minus on the third term — Blackman is + 0.08·cos(4πi / n), and with the sign flipped the main lobe stops being a whole number of bins wide too'],
                  ])
                : null;
              ctx.assert(hasNumber(nums, row.db, 0.4),
                dbHint || `the ${label} line should carry a peak side lobe of ≈${row.db.toFixed(2)} dB, got [${nums.join(', ')}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const found = windowUnderTest(ctx, 'the Blackman taper needs one thread per sample');
            ctx.assert(!found.error, found.error);
            const signal = tone(13.5, 0.7);
            const out = found.kernel(signal);
            const hint = diagnoseAll(N, i => out[i], i => signal[i] * blackmanWindow(i), 1e-4,
              blackmanProbes(signal));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], signal[i] * blackmanWindow(i), 1e-4, hint || `sample ${i}`);
            }
            ctx.assertClose(sumOf(found.shape), 0.42 * N, 0.05,
              'a periodic Blackman window sums to exactly 0.42 n = 107.52');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The trade itself, as one assertion: every row of the table.
            const table = tradeTable();
            for (const label of ['rect', 'hann', 'blackman']) {
              const nums = lineNumbers(ctx.logs, label);
              ctx.assert(nums, `no console.log line mentioning "${label}"`);
              const row = table[label];
              ctx.assert(hasNumber(nums, row.bins, 0.05) && hasNumber(nums, row.db, 0.4),
                `the ${label} row should read ${row.bins} bins / ${row.db.toFixed(2)} dB, got [${nums.join(', ')}]`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'coherent-gain',
      title: 'Give the Amplitude Back',
      intro: `<p>Task 3 left a loose end. The Hann window collapsed the leak by 29 dB and also
        dropped the measured amplitude from 0.65 to 0.42 — <em>further</em> from the truth than
        before. That is neither a bug nor a mystery. The window multiplies the signal by
        something whose average is exactly 0.5, so the peak comes back exactly half size.
        Divide it back out and you are finished.</p>
        <p>The number to divide by is the window's <strong>coherent gain</strong>,
        CG = Σw / n — the taper's mean value, 1 for rectangular, 0.5 for Hann, 0.42 for
        Blackman. Folding it into the amplitude formula from task 1 leaves something pleasantly
        compact:</p>
<pre><code>amplitude = 2 · |X[k]| / Σw</code></pre>
        <p>Miss it and every amplitude your analyser reports is low by a constant factor — 50%
        for Hann, 58% for Blackman. That is exactly the kind of bug that survives for years,
        because a clean scale error never looks broken.</p>
        <p>Its sister number is the <strong>noise gain</strong>, Σw² / n, which is 0.375 for
        Hann. That one is for <em>power</em>: the mean square of a broadband noise floor scales
        with the sum of the squares, not with the square of the sum. Amplitude of a tone → Σw.
        Power of a noise floor → Σw². Reach for the wrong one and you are wrong by a factor
        nothing will ever flag, so it is worth carrying both.</p>
        <p>And a tidy way to get either: hand your window kernel a signal of 256 ones, and it
        hands you back the window.</p>`,
      goal: `<strong>Goal:</strong> build a calibrated amplitude spectrum and recover the true
        amplitude of one tone through all three windows.`,
      requirements: [
        'Create <code>amplitudeSpectrum(spec, sumW)</code> with <code>output: [256]</code> — a plain number passes into a kernel like any other argument',
        'Return <code>2 * Math.sqrt(re * re + im * im) / sumW</code>',
        'Get each window\'s samples by running its kernel on a signal of 256 ones, then sum them and their squares',
        'Log one line per window, labelled <code>rect</code> / <code>hann</code> / <code>blackman</code>, with its raw peak and its recovered amplitude, plus a line carrying Hann\'s coherent and noise gains',
      ],
      hints: [
        {
          title: 'Hint 1 — a scalar argument',
          body: `<p>Kernel arguments do not have to be arrays. <code>sumW</code> arrives as a
            plain number and is used like one:</p>
<pre><code>const re = spec[0][this.thread.x];
const im = spec[1][this.thread.x];
return 2 * Math.sqrt(re * re + im * im) / sumW;</code></pre>`,
        },
        {
          title: 'Hint 2 — the window, from the window kernel',
          body: `<p>Every window kernel here is "sample × taper", so a signal of ones makes it
            return the taper:</p>
<pre><code>const w = hann(flat);            // flat = new Array(256).fill(1)
let sumW = 0;
let sumW2 = 0;
for (let i = 0; i &lt; w.length; i++) {
  sumW += w[i];
  sumW2 += w[i] * w[i];
}</code></pre>
<p>Which also gives the rectangular case for free: <code>rect(flat)</code> is 256
            ones, so its <code>sumW</code> is 256.</p>`,
        },
        {
          title: 'Hint 3 — what you should see',
          body: `<p>Three raw peaks — <code>76.8</code>, <code>38.4</code>,
            <code>32.256</code> — and one amplitude, <code>0.600</code>, three times over.
            Hann's coherent gain is <code>0.5</code> exactly and its noise gain
            <code>0.375</code> exactly: a periodic Hann window sums to n/2 and its squares to
            3n/8, which doubles as a check that your window is the periodic one.</p>`,
        },
      ],
      transfer: `Coherent gain is why a home-made analyser and a real one disagree by a
        constant: scipy's <code>welch</code> takes <code>scaling='spectrum'</code> versus
        <code>'density'</code> precisely to pick Σw versus Σw² normalisation, and every vendor's
        spectrum-analyser manual carries a window table with both columns. On the GPU it is one
        scalar uniform folded into a pass you already run — the cheapest correctness fix in this
        module, and the most commonly skipped.`,
      starterCode: `// Three windows, three different peaks, one true amplitude.
const gpu = new GPU({ mode });

// GIVEN — the transform and the windows you have already built.
${SPECTRUM_KERNEL}

${RECT_KERNEL}

${HANN_KERNEL}

const blackman = gpu.createKernel(function (signal) {
  const i = this.thread.x;
  const a = 2 * Math.PI * i / this.constants.n;
  const w = 0.42 - 0.5 * Math.cos(a) + 0.08 * Math.cos(2 * a);
  return signal[i] * w;
}, { output: [256], constants: { n: 256 } });

const amplitudeSpectrum = gpu.createKernel(function (spec, sumW) {
  // TODO: 2 * |X[k]| / sumW — task 1's magnitude pass, calibrated.
  return 0;
}, { output: [256] });

const flat = new Array(256).fill(1);

function report(label, window, windowed) {
  // TODO: sumW = the sum of the window, sumW2 = the sum of its squares
  const sumW = 256;
  const sumW2 = 256;

  const amp = amplitudeSpectrum(spectrum(windowed), sumW);
  let peak = 0;
  for (let k = 0; k <= 128; k++) if (amp[k] > peak) peak = amp[k];

  console.log(label, 'raw peak:', peak * sumW / 2, 'amplitude:', peak);
  return { sumW, sumW2 };
}

report('rect    ', rect(flat), rect(signal));
const h = report('hann    ', hann(flat), hann(signal));
report('blackman', blackman(flat), blackman(signal));
console.log('hann gains — coherent:', h.sumW / 256, 'noise:', h.sumW2 / 256);
`,
      solutionCode: `// Three windows, three different peaks, one true amplitude.
const gpu = new GPU({ mode });

// GIVEN — the transform and the windows you have already built.
${SPECTRUM_KERNEL}

${RECT_KERNEL}

${HANN_KERNEL}

const blackman = gpu.createKernel(function (signal) {
  const i = this.thread.x;
  const a = 2 * Math.PI * i / this.constants.n;
  const w = 0.42 - 0.5 * Math.cos(a) + 0.08 * Math.cos(2 * a);
  return signal[i] * w;
}, { output: [256], constants: { n: 256 } });

const amplitudeSpectrum = gpu.createKernel(function (spec, sumW) {
  const re = spec[0][this.thread.x];
  const im = spec[1][this.thread.x];
  return 2 * Math.sqrt(re * re + im * im) / sumW;
}, { output: [256] });

const flat = new Array(256).fill(1);

function report(label, window, windowed) {
  let sumW = 0;
  let sumW2 = 0;
  for (let i = 0; i < window.length; i++) {
    sumW += window[i];
    sumW2 += window[i] * window[i];
  }

  const amp = amplitudeSpectrum(spectrum(windowed), sumW);
  let peak = 0;
  for (let k = 0; k <= 128; k++) if (amp[k] > peak) peak = amp[k];

  console.log(label, 'raw peak:', peak * sumW / 2, 'amplitude:', peak);
  return { sumW, sumW2 };
}

report('rect    ', rect(flat), rect(signal));
const h = report('hann    ', hann(flat), hann(signal));
report('blackman', blackman(flat), blackman(signal));
console.log('hann gains — coherent:', h.sumW / 256, 'noise:', h.sumW2 / 256);
`,
      inputs: () => ({ signal: tone(GAIN_BIN, GAIN_AMP) }),
      publicTests: [
        {
          name: 'the amplitude kernel divides by the <code>sumW</code> it is handed',
          run: async ctx => {
            const amp = amplitudeKernel(ctx);
            ctx.assert(amp,
              'no kernel taking (spectrum, sumW) found — amplitudeSpectrum needs both arguments, and output: [256]');
            const re = new Float32Array(N);
            const im = new Float32Array(N);
            for (let k = 0; k < N; k++) {
              re[k] = ((k * 3) % 7) + 1.5;
              im[k] = ((k * 5) % 9) - 3.25;
            }
            const got = amp([re, im], 128);
            ctx.assert(got && got.length === N, `expected 256 values, got ${got && got.length}`);
            for (const k of [0, 5, 64, 200]) {
              const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
              const expected = (2 * mag) / 128;
              const hint = diagnose(got[k], expected, 1e-3, [
                [mag / 128, 'the factor of 2 is missing — a real cosine of amplitude A puts A/2 in bin k and A/2 in its mirror, so the amplitude is 2·|X| / Σw'],
                [(2 * mag) / N, 'that divides by n rather than by the sumW argument — with no window they are the same 256, but a Hann window sums to 128 and the two answers differ by exactly the factor of 2 that makes every amplitude read 50% low'],
                [(2 * (re[k] * re[k] + im[k] * im[k])) / 128, 'the square root is missing — |X| is the hypotenuse of the two planes'],
              ]);
              ctx.assertClose(got[k], expected, 1e-3, hint || `bin ${k}`);
            }
            // Structural: halve sumW and every value must exactly double.
            const halved = amp([re, im], 64);
            ctx.assertClose(halved[7] / got[7], 2, 1e-3,
              'halving sumW must double every value — this kernel is ignoring its sumW argument and dividing by a constant instead');
          },
        },
        {
          name: 'three windows, three different raw peaks',
          run: async ctx => {
            const signal = tone(GAIN_BIN, GAIN_AMP);
            for (const [label, w] of [
              ['rect', rectWindow],
              ['hann', hannWindow],
              ['blackman', blackmanWindow],
            ]) {
              const peak = peakOf(dftMag(applyWindow(signal, w)));
              const nums = lineNumbers(ctx.logs, label);
              ctx.assert(nums, `no console.log line mentioning "${label}" — log one labelled line per window`);
              ctx.assert(hasNumber(nums, peak, 0.05),
                `the ${label} line should carry its raw peak magnitude ≈${peak.toFixed(3)}, got [${nums.join(', ')}]`);
            }
          },
        },
        {
          name: 'all three recover the same amplitude, <code>0.600</code>',
          run: async ctx => {
            const signal = tone(GAIN_BIN, GAIN_AMP);
            // rect is checked FIRST, and it is the one row where a missing factor
            // of 2 is distinguishable (its uncorrected amplitude is already the
            // right answer). Because ctx.assert throws, reaching the hann row at
            // all proves the factor of 2 — which is what lets a hann reading of
            // 0.300 be named as the missing coherent gain instead of staying
            // ambiguous between the two mistakes.
            for (const [label, w] of [
              ['rect', rectWindow],
              ['hann', hannWindow],
              ['blackman', blackmanWindow],
            ]) {
              const window = samplesOf(w);
              const sumW = sumOf(window);
              const sumW2 = sumOf(window.map(v => v * v));
              const peak = peakOf(dftMag(applyWindow(signal, w)));
              const nums = lineNumbers(ctx.logs, label);
              ctx.assert(nums, `no console.log line mentioning "${label}"`);
              const probes = [
                [(2 * peak) / N,
                  'that is 2·peak / n — the coherent-gain correction is missing. Divide by the window\'s own sum, not by the sample count'],
                [(2 * peak) / sumW2,
                  'that divides by Σw², the NOISE gain. Σw² normalises the power of a broadband noise floor; a tone\'s amplitude wants Σw'],
              ];
              if (label === 'rect') {
                probes.push([peak / sumW,
                  'the factor of 2 is missing — half of a real cosine\'s energy sits in the mirror bin above Nyquist, so the amplitude is 2·|X| / Σw']);
              }
              const hint = diagnose(pickLast(nums, v => v > 0 && v < 3), GAIN_AMP, 3e-3, probes);
              ctx.assert(hasNumber(nums, GAIN_AMP, 3e-3),
                hint || `the ${label} line should recover the true amplitude ${GAIN_AMP.toFixed(3)}, got [${nums.join(', ')}]`);
            }
          },
        },
        {
          name: 'the two gains: coherent <code>0.5</code>, noise <code>0.375</code>',
          run: async ctx => {
            const window = samplesOf(hannWindow);
            const cg = sumOf(window) / N;
            const ng = sumOf(window.map(v => v * v)) / N;
            const nums = lineNumbers(ctx.logs, 'coherent');
            ctx.assert(nums, 'no console.log line mentioning "coherent" — log Hann\'s two gains');
            ctx.assert(hasNumber(nums, cg, 2e-3),
              `expected Hann's coherent gain Σw/256 = ${cg} on that line, got [${nums.join(', ')}]`);
            const hint = diagnose(pickLast(nums, v => Math.abs(v - cg) > 2e-3), ng, 2e-3, [
              [cg * cg, 'that is CG², not the noise gain — the noise gain is the mean of the SQUARED window samples, Σw²/n, which for a periodic Hann window is exactly 0.375'],
            ]);
            ctx.assert(hasNumber(nums, ng, 2e-3),
              hint || `expected Hann's noise gain Σw²/256 = ${ng} on that line, got [${nums.join(', ')}]`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const amp = amplitudeKernel(ctx);
            ctx.assert(amp, 'no (spectrum, sumW) kernel found');
            const re = new Float32Array(N);
            const im = new Float32Array(N);
            for (let k = 0; k < N; k++) {
              re[k] = Math.cos(k * 0.21) * 4 + 1.1;
              im[k] = Math.sin(k * 0.43) * 2 - 0.7;
            }
            const got = amp([re, im], 107.52);
            for (let k = 0; k < N; k++) {
              const expected = (2 * Math.sqrt(re[k] * re[k] + im[k] * im[k])) / 107.52;
              ctx.assertClose(got[k], expected, 1e-3, `bin ${k}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The lesson, restated as an assertion: three windows, three peaks,
            // one amplitude.
            const signal = tone(GAIN_BIN, GAIN_AMP);
            const windows = [
              ['rect', rectWindow],
              ['hann', hannWindow],
              ['blackman', blackmanWindow],
            ];
            for (const [label, w] of windows) {
              const nums = lineNumbers(ctx.logs, label);
              ctx.assert(nums, `no console.log line mentioning "${label}"`);
              const peak = peakOf(dftMag(applyWindow(signal, w)));
              ctx.assert(hasNumber(nums, peak, 0.05),
                `the ${label} row should carry its raw peak ≈${peak.toFixed(3)} — three different peaks is the point`);
              ctx.assert(hasNumber(nums, GAIN_AMP, 3e-3),
                `the ${label} row should recover ${GAIN_AMP}`);
            }
          },
        },
      ],
    },
  ],
};
