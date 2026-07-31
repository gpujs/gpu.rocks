// Module: The DFT, Honestly — uuid 7b1e3f9b-baf5-4b75-9ad1-3c05f445a3db
// (short id 7b1e3f9b). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module is new.
//
// Signal Processing — The DFT, Honestly.
//
// Five tasks: one bin correlated by one thread → why a single real number
// throws the phase away, and the two-plane complex convention → the whole
// spectrum, one thread per (bin, plane) → magnitude, phase, and the
// conjugate mirror → the inverse transform and the round trip.
//
// The thesis: before the clever algorithm, the transform. The naive DFT is a
// pile of dot products, one per frequency, every one of them independent —
// O(n²) work a GPU eats happily, and the honest baseline an FFT has to beat.
//
// ---------------------------------------------------------------------------
// THE COMPLEX CONVENTION THIS TRACK USES — introduced here, inherited by every
// other Signal Processing module. gpu.js has no complex type, so a complex
// sequence is carried as TWO PLANES of floats:
//
//   • kernel OUTPUT — `output: [n, 2]`, indexed `result[p][i]`: plane p = 0 is
//     the real part, p = 1 the imaginary part, i the sample or bin index.
//     (`output: [w, h]` is indexed [y][x], so [n, 2] gives exactly [plane][i];
//     Optical Flow already ships the 3-plane form [w, h, 3] → [z][y][x], so
//     this is the established shape rather than a new invention.)
//   • kernel INPUT — a nested [2][n] array: spectrum[0][i] real,
//     spectrum[1][i] imaginary. Which is EXACTLY what an `output: [n, 2]`
//     kernel hands back, so a forward pass drops straight into an inverse one
//     with no reshaping (task 5 leans on that).
//   • a purely real input signal stays a plain [n] array; each task says which
//     of the two it takes.
//
// One container type per kernel, always: gpu.js binds the container on the
// first call and WebGL fails hard afterwards (gpujs/gpu.js#857). Every signal
// here is a plain Array of numbers and every spectrum a 2-row nested array.
//
// ---------------------------------------------------------------------------
// Kernel-authoring rules (contract): no closures inside kernel functions, loop
// bounds come from this.constants (compile-time known), Math.* is limited to
// gpu.js's whitelist (Math.atan2 is on it, Math.hypot is NOT), and every task
// passes in CPU mode as well as GPU mode.
//
// SIZE AND THE FLOAT MARGIN. n = 256 throughout. The naive DFT is O(n²), so a
// full spectrum is 512 threads × 256 terms — a few milliseconds on the GPU and
// well under a second on the CPU backend, which is what verification runs.
// Tests compute in float64 while the GL backend computes in float32, and 256
// trigonometric terms accumulate: MEASURED worst case over the whole spectrum
// is 4.5e-3 on a real GPU (Apple M1 Max, WebGL2, single precision) and 7.9e-3
// under SwiftShader; a round-tripped sample lands within 2.4e-5 and 5.3e-4
// respectively. So spectrum comparisons use eps = 0.05 (≈6× the worst
// measurement, and 2,500× smaller than the 128-tall peaks these signals are
// built to produce) and round-trip comparisons eps = 5e-3. Nothing asserted in
// this file sits near a decision boundary: every expected value is either a
// designed peak (128, 102.4, 96, 76.8, 64, 57.6, 38.4) or an analytic zero.

const N = 256; // samples per signal, and bins per spectrum, everywhere here
const SAMPLE_RATE = 8192; // Hz — so one bin is 8192 / 256 = 32 Hz wide

// ---- the signals ----------------------------------------------------------
//
// Learner code runs in a Web Worker, which has no AudioContext and no
// microphone, so every signal is BUILT here, deterministically. That is a
// feature: a sum of cosines whose frequencies land exactly on bins makes every
// expected value exact rather than approximate.
//
// spec entries are [bin, amplitude, phase]; bin 0 is the DC (constant) level.
// A cosine of amplitude A at bin k contributes exactly A·n/2 to |X[k]| — which
// is where 128, 102.4, 76.8, 64, 57.6 and 38.4 all come from.
function tones(spec, n = N) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (const [k, amp, phase] of spec) {
      v += amp * Math.cos((2 * Math.PI * k * i) / n + (phase || 0));
    }
    out[i] = v;
  }
  return out;
}

// The same tone, arriving `delay` samples late. A delay of n/(4k) samples is a
// quarter turn at bin k — which is the whole point of task 2.
function delayedTone(k, delay, n = N) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.cos((2 * Math.PI * k * (i - delay)) / n);
  return out;
}

// A plucked string: two harmonics under an exponential decay. Deliberately NOT
// a sum of whole bins — its energy is smeared across the whole spectrum, so
// the round trip in task 5 tests every bin rather than three of them.
function pluck(n = N) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const decay = Math.exp((-3 * i) / n);
    out[i] = decay * (Math.cos((2 * Math.PI * 9 * i) / n) + 0.5 * Math.cos((2 * Math.PI * 18 * i) / n));
  }
  return out;
}

// ---- float64 references ---------------------------------------------------

// The forward DFT: X[k] = Σ x[i]·e^(−2πi·ki/n), as two plain arrays [re, im].
// `radiansPerTurn` exists only so the tests can build the "the 2π went
// missing" probe from the same code path as the right answer.
function referenceDft(signal, radiansPerTurn = 2 * Math.PI) {
  const n = signal.length;
  const re = new Array(n).fill(0);
  const im = new Array(n).fill(0);
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      const angle = (radiansPerTurn * k * i) / n;
      re[k] += signal[i] * Math.cos(angle);
      im[k] -= signal[i] * Math.sin(angle);
    }
  }
  return [re, im];
}

// One real (cosine-only) correlation — task 1's whole job.
function cosineBin(signal, k, radiansPerTurn = 2 * Math.PI) {
  let acc = 0;
  for (let i = 0; i < signal.length; i++) {
    acc += signal[i] * Math.cos((radiansPerTurn * k * i) / signal.length);
  }
  return acc;
}

function magnitudeOf(spectrum, k) {
  return Math.sqrt(spectrum[0][k] * spectrum[0][k] + spectrum[1][k] * spectrum[1][k]);
}

function sumOf(arr) {
  let total = 0;
  for (let i = 0; i < arr.length; i++) total += arr[i];
  return total;
}

function sumAbsOf(arr) {
  let total = 0;
  for (let i = 0; i < arr.length; i++) total += Math.abs(arr[i]);
  return total;
}

// A spectrum in the shape a kernel HANDS BACK: an Array of two Float32Arrays.
// Tests that feed a spectrum into a learner's inverse kernel use this, so the
// container matches the one their own forward kernel already bound.
function asPlanes(re, im) {
  return [Float32Array.from(re), Float32Array.from(im)];
}

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

function someLogged(logs, value, eps) {
  return loggedNumbers(logs).some(v => Math.abs(v - value) <= eps);
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so a bin where two candidates coincide (bin 0,
// where the 2π cancels; any bin whose imaginary part is zero, where a flipped
// exponent is invisible) stays silent, as do observations matching probes that
// disagree with each other. A confident wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The strong form, for mistakes that reshape a WHOLE array: a probe must
// predict every element (and disagree with the right answer somewhere) before
// it may speak. One matching bin is weak evidence — a conjugated spectrum
// agrees with the real one wherever the imaginary part happens to vanish — so
// nothing here is diagnosed from a single coincidence. Probe values are
// functions of the index; a missing element makes the comparison NaN, which
// fails the test and rejects the probe.
//
// A probe may carry its own tolerance as a third element, and one kind of
// probe NEEDS to: a mistake that multiplies the answer by n multiplies the
// float error by n too, so an un-normalised inverse misses a 5e-3 window by
// exactly the factor it is being accused of. Widening the window for that
// probe alone costs nothing — it is still 250× tighter than the difference it
// has to detect.
function diagnoseAll(count, got, expected, eps, probes) {
  const hits = probes
    .filter(([value, , probeEps]) => {
      const tol = probeEps || eps;
      let differs = false;
      for (let i = 0; i < count; i++) {
        if (!(Math.abs(got(i) - value(i)) <= tol)) return false;
        if (Math.abs(expected(i) - value(i)) > tol) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// A [2][n] result read as one flat run of 2n values: index < n is plane 0.
function flatPlanes(planes, n = N) {
  return idx => {
    const row = planes[idx < n ? 0 : 1];
    return row ? row[idx % n] : NaN;
  };
}

// ---- the shared probe sentences -------------------------------------------

const SIGN_FLIPPED =
  'the sign of the exponent is flipped — the imaginary plane came back negated. The forward ' +
  'transform is e^(-2πi·ki/n), so the imaginary part SUBTRACTS: im -= signal[i] * Math.sin(angle). ' +
  'For a real signal that negation mirrors the whole spectrum, so bin k comes back holding what ' +
  'belongs to bin n - k';

const PLANES_SWAPPED =
  'the two planes are the other way round — plane 0 (this.thread.y === 0) is the REAL part and ' +
  'plane 1 the imaginary part';

const NO_TWO_PI =
  'the 2π is missing — the angle for term i is 2 * Math.PI * k * i / n, which is k whole turns ' +
  'across the signal. Without it the test wave barely moves and the sum measures nothing';

const BOTH_PLANES_REAL =
  'both planes computed the same thing — the kernel never branched on this.thread.y, so the ' +
  'imaginary plane is a copy of the real one';

// Task 1: the ways one real correlation goes wrong.
function binProbes(signal, k) {
  return [
    [cosineBin(signal, k, 1), NO_TWO_PI],
    [sumOf(signal),
      'every term used the same angle — with output: [1] this.thread.x is always 0, so the ' +
      'angle has to advance with the LOOP variable i, not with the thread index'],
    [cosineBin(signal, k) / signal.length,
      'that is the correlation divided by n — the forward transform does not scale. The 1/n ' +
      'belongs to the inverse transform, and only there'],
  ];
}

// Task 3: the four ways a whole two-plane spectrum goes wrong. Task 2 builds
// the same probes over its single bin, where they are two values rather than
// 2n; task 5's mistakes are different ones and live with its tests.
function spectrumProbes(signal, reference) {
  const flipped = referenceDft(signal, -2 * Math.PI);
  const noTwoPi = referenceDft(signal, 1);
  const n = signal.length;
  return [
    [idx => (idx < n ? flipped[0][idx] : flipped[1][idx % n]), SIGN_FLIPPED],
    [idx => (idx < n ? reference[1][idx] : reference[0][idx % n]), PLANES_SWAPPED],
    [idx => (idx < n ? noTwoPi[0][idx] : noTwoPi[1][idx % n]), NO_TWO_PI],
    [idx => reference[0][idx % n], BOTH_PLANES_REAL],
  ];
}

// ---- kernel identification ------------------------------------------------
//
// ctx.kernel is the LAST kernel a run created, which is all a single-kernel
// task needs. Tasks 4 and 5 create two, so they are told apart by BEHAVIOUR
// (task 4) or by the shape of the argument the learner actually passed them
// (task 5) — never by creation order, which a learner is free to change.

// Every kernel that answers a 256-value numeric array — task 4's two.
function unitKernels(ctx, probeSignal) {
  const out = [];
  for (const k of ctx.kernels) {
    let v;
    try {
      v = k(probeSignal);
    } catch (e) {
      continue; // a kernel of a different shape — not a candidate
    }
    if (v && v.length === N && typeof v[0] === 'number') out.push({ kernel: k, values: v });
  }
  return out;
}

// Task 4's probe tone: one cosine at bin 8, amplitude 1, phase 1.0 radians. A
// magnitude spectrum of it is non-negative with its peak at bin 8; a phase
// spectrum is an angle, so the whole thing fits inside ±π. Those two SHAPES
// tell the kernels apart — deliberately not the exact values 128 and 1.0,
// because a magnitude kernel that is merely wrong (no square root, a stray
// factor of 2) should still be recognised as the magnitude kernel and get a
// numeric diagnosis rather than "not found".
function magnitudeAndPhase(ctx) {
  const probe = tones([[8, 1, 1]]);
  const candidates = unitKernels(ctx, probe);
  let magnitude = null;
  let phase = null;
  for (const c of candidates) {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < c.values.length; i++) {
      if (c.values[i] < min) min = c.values[i];
      if (c.values[i] > max) max = c.values[i];
    }
    if (!magnitude && min >= -0.05 && max > 5 && c.values[8] >= max * 0.99) {
      magnitude = c.kernel;
    } else if (!phase && min >= -3.2 && max <= 3.2) {
      phase = c.kernel;
    }
  }
  return { magnitude, phase, candidates, probe };
}

// One wrong answer is common enough to be worth naming before any bin is even
// compared: taking each TERM's magnitude and summing those. |x·e^(−iθ)| is just
// |x|, so every bin comes back holding Σ|x[i]| — the same flat number, which is
// a spectrum of nothing. Checked at four spread-out bins, so a real magnitude
// spectrum (peaked at bin 8, near zero at bin 40) can never match.
function flatSpectrumHint(candidates, probe) {
  const flat = sumAbsOf(probe);
  for (const c of candidates) {
    let all = true;
    for (const k of [0, 8, 40, 199]) {
      if (!(Math.abs(c.values[k] - flat) <= 0.5)) all = false;
    }
    if (all) {
      return 'one kernel returns the same number for every bin — you took the magnitude of each ' +
        'TERM and summed those, and |x[i] · e^(-iθ)| is just |x[i]|, so every bin collapses to ' +
        'Σ|x[i]|. Accumulate the real and imaginary parts separately and take the magnitude once, ' +
        'after the loop';
    }
  }
  return null;
}

// Task 5's pair, told apart by what the learner passed them: the forward
// transform takes a flat signal of 256 numbers, the inverse takes a [2][256]
// spectrum. The sandbox records .lastArgs on every invocation.
function transformKernels(ctx) {
  let forward = null;
  let inverse = null;
  for (const k of ctx.kernels) {
    const args = Array.isArray(k.lastArgs) ? k.lastArgs : null;
    const first = args && args.length ? args[0] : null;
    if (!first || typeof first.length !== 'number') continue;
    if (first.length === N && typeof first[0] === 'number') forward = k;
    else if (first.length === 2 && first[0] && typeof first[0].length === 'number') inverse = k;
  }
  return { forward, inverse };
}

export default {
  uuid: '7b1e3f9b-baf5-4b75-9ad1-3c05f445a3db',
  version: 1,
  slug: 'the-dft',
  title: 'The DFT, Honestly',
  blurb:
    'One thread per frequency bin, each summing over every sample — the honest O(n²) transform, ' +
    'complex arithmetic and all.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'one-bin',
      title: 'One Bin, One Thread',
      intro: `<p>A <em>transform</em> sounds grand. One <strong>bin</strong> of it is not: pick a
        frequency, build a test wave at that frequency, multiply it against the signal sample by
        sample, and add the products up. One number falls out, and it answers exactly one
        question — <em>how much of this frequency is in here?</em></p>
        <p>That is a dot product, and it is the same instinct Template Matching used on images:
        slide a pattern over the data and let the sum of the products score the match. Here the
        pattern is a cosine and the data is a row of samples.</p>
        <p>Start with one bin and one thread. <code>output: [1]</code> gives you a single thread
        that walks the whole signal — the conceptual atom. Task 3 hands every bin a thread of its
        own, which is where this stops being a loop and starts being a GPU workload.</p>`,
      goal: `<strong>Goal:</strong> correlate <code>signal</code> against a cosine at bin
        <code>k</code> — sum <code>signal[i] × cos(2π·k·i/n)</code> over all 256 samples — and
        report bins 8, 20 and 13.`,
      requirements: [
        'Keep <code>output: [1]</code>: one thread owns the one answer',
        'Loop <code>for (let i = 0; i &lt; this.constants.n; i++)</code> — WebGL needs a compile-time bound',
        'Accumulate <code>signal[i] * Math.cos(2 * Math.PI * k * i / this.constants.n)</code>',
        '<code>console.log</code> bins 8, 20 and 13 — expect <code>128</code>, <code>64</code> and <code>0</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — what the angle has to do',
          body: `<p>Over the whole signal, the test wave for bin <code>k</code> must complete
            exactly <code>k</code> turns. Term <code>i</code> is therefore a fraction
            <code>i / n</code> of the way through <code>k</code> turns, and one turn is
            <code>2π</code> radians:</p>
<pre><code>const angle = 2 * Math.PI * k * i / this.constants.n;</code></pre>`,
        },
        {
          title: 'Hint 2 — the accumulator',
          body: `<p>Same shape as any other loop-and-total kernel: declare
            <code>let acc = 0;</code> before the loop, add one product per iteration, and
            <code>return acc;</code> after it.</p>
<pre><code>acc += signal[i] * Math.cos(angle);</code></pre>`,
        },
        {
          title: 'Hint 3 — asking for a bin',
          body: `<p><code>k</code> is an ordinary kernel argument, so one kernel answers every
            bin — call it again with a different number:</p>
<pre><code>console.log('bin 8:', bin(signal, 8)[0]);
console.log('bin 20:', bin(signal, 20)[0]);</code></pre>`,
        },
      ],
      transfer: `Correlating against a basis function is the move behind far more than audio:
        it is what a matched filter does in radar, what a lock-in amplifier does in a lab, and
        what a single row of a matrix–vector product does in linear algebra. On any platform the
        kernel is the same — CUDA, WGSL and Metal all give you a thread, a loop and a fused
        multiply-add, and that is the entire ingredient list.`,
      starterCode: `// One bin of a transform is one dot product: signal · test wave.
const gpu = new GPU({ mode });

const bin = gpu.createKernel(function (signal, k) {
  // TODO: loop i over all this.constants.n samples and accumulate
  // signal[i] * cos(2π · k · i / n) into a local total, then return it.
  return 0;
}, {
  output: [1],
  constants: { n: 256 },
});

console.log('bin 8  (the loud tone):', bin(signal, 8)[0]);
console.log('bin 20 (the quiet one):', bin(signal, 20)[0]);
console.log('bin 13 (nothing there):', bin(signal, 13)[0]);
`,
      solutionCode: `// One bin of a transform is one dot product: signal · test wave.
const gpu = new GPU({ mode });

const bin = gpu.createKernel(function (signal, k) {
  let acc = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = 2 * Math.PI * k * i / this.constants.n;
    acc += signal[i] * Math.cos(angle);
  }
  return acc;
}, {
  output: [1],
  constants: { n: 256 },
});

console.log('bin 8  (the loud tone):', bin(signal, 8)[0]);
console.log('bin 20 (the quiet one):', bin(signal, 20)[0]);
console.log('bin 13 (nothing there):', bin(signal, 13)[0]);
console.log('bin 0  (the DC level):', bin(signal, 0)[0]);
`,
      inputs: () => ({ signal: tones([[0, 0.375], [8, 1], [20, 0.5]]) }),
      publicTests: [
        {
          name: 'one thread, one number',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const signal = tones([[0, 0.375], [8, 1], [20, 0.5]]);
            const out = ctx.kernel(signal, 8);
            ctx.assert(out && out.length === 1, `expected 1 output value, got ${out && out.length}`);
          },
        },
        {
          name: 'bin 8 reads <code>128</code>, bin 20 reads <code>64</code>, bin 13 reads <code>0</code>',
          run: async ctx => {
            const signal = tones([[0, 0.375], [8, 1], [20, 0.5]]);
            for (const [k, expected] of [[8, 128], [20, 64], [13, 0], [0, 96]]) {
              const got = ctx.kernel(signal, k)[0];
              const hint = diagnose(got, expected, 0.05, binProbes(signal, k));
              ctx.assertClose(got, expected, 0.05, hint || `bin ${k}`);
            }
          },
        },
        {
          name: 'the three bins are logged',
          run: async ctx => {
            ctx.assert(
              someLogged(ctx.logs, 128, 0.5),
              'log the value of bin 8 — expected ≈128 in the console output'
            );
            ctx.assert(
              someLogged(ctx.logs, 64, 0.5),
              'log the value of bin 20 — expected ≈64 in the console output'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Different tones, different amplitudes, and one of them carrying a
            // phase — a cosine-only correlation reads A·n/2·cos(φ) there, which
            // is the crack task 2 prises open.
            const signal = tones([[0, 0.5], [5, 0.25], [17, 1], [31, 0.75, 0.6]]);
            for (const k of [0, 5, 17, 31, 9, 64]) {
              const expected = cosineBin(signal, k);
              const got = ctx.kernel(signal, k)[0];
              const hint = diagnose(got, expected, 0.05, binProbes(signal, k));
              ctx.assertClose(got, expected, 0.05, hint || `bin ${k}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'why-complex',
      title: 'One Number Is Not Enough',
      intro: `<p>Task 1's answer has a hole in it. Take a pure tone at bin 8 and correlate it
        against a cosine at bin 8: you get 128, the whole amplitude. Now delay that identical tone
        by 8 samples out of 256 — nothing about the sound has changed — and the same correlation
        reads <strong>0</strong>. The tone is right there and the number says it is absent, because
        a delay of 8 samples is a quarter turn at bin 8, and a cosine and a sine of the same
        frequency are orthogonal.</p>
        <p>The fix is to measure against <em>both</em>: a cosine and a sine. Two numbers per bin,
        and the pair rotates as the signal shifts while its length — <code>√(re² + im²)</code> —
        stays put. That pair is what "complex" means here. It is arithmetic on pairs of floats,
        not a mysterious new number system: <code>re</code> is what the cosine measured,
        <code>im</code> is what the sine measured (with a minus sign, from the standard
        <code>e^(−2πi·ki/n)</code>), and nothing else about it needs believing.</p>
        <p><strong>The convention this whole track uses.</strong> gpu.js has no complex type, so a
        complex result is <strong>two planes</strong> of floats: <code>output: [n, 2]</code>, read
        as <code>result[p][i]</code> — plane <code>p = 0</code> is the real part, plane
        <code>p = 1</code> the imaginary part, <code>i</code> the bin. Since
        <code>output: [w, h]</code> is indexed <code>[y][x]</code>, <code>this.thread.y</code>
        <em>is</em> the plane and <code>this.thread.x</code> is the bin. Here <code>n</code> is 1 —
        one bin, two planes.</p>`,
      goal: `<strong>Goal:</strong> compute both parts of bin 8 with one kernel —
        <code>output: [1, 2]</code>, plane 0 the cosine correlation, plane 1 the sine correlation
        with a minus sign — and log the real part, the imaginary part and the magnitude for both
        <code>tone</code> and <code>shiftedTone</code>.`,
      requirements: [
        '<code>output: [1, 2]</code> — read back as <code>result[plane][0]</code>',
        'Branch on <code>this.thread.y</code>: plane 0 <em>adds</em> <code>signal[i] * Math.cos(angle)</code>, plane 1 <em>subtracts</em> <code>signal[i] * Math.sin(angle)</code>',
        'In JavaScript, take <code>Math.sqrt(re * re + im * im)</code> for each signal',
        'Log both magnitudes — both read <code>128</code>, while the real parts read <code>128</code> and <code>0</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — one kernel, two jobs',
          body: `<p>Every thread runs the same body, so the plane has to be a branch inside it.
            <code>this.thread.y</code> is 0 for the real plane and 1 for the imaginary one:</p>
<pre><code>if (this.thread.y === 0) acc += signal[i] * Math.cos(angle);
else acc -= signal[i] * Math.sin(angle);</code></pre>`,
        },
        {
          title: 'Hint 2 — why the minus sign',
          body: `<p>The forward transform correlates against <code>e^(−2πi·ki/n)</code>, and
            <code>e^(−iθ) = cos(θ) − i·sin(θ)</code>. The minus rides on the sine, which is why
            plane 1 subtracts. Getting that sign wrong mirrors the entire spectrum, so it is worth
            fixing here where there is only one bin to look at.</p>`,
        },
        {
          title: 'Hint 3 — reading the pair back',
          body: `<p>With <code>output: [1, 2]</code> the result is two rows of one value:</p>
<pre><code>const out = bin8(tone);
const re = out[0][0];
const im = out[1][0];
console.log('magnitude:', Math.sqrt(re * re + im * im));</code></pre>`,
        },
      ],
      transfer: `Interleaved or planar complex data is a decision every signal library makes:
        cuFFT and rocFFT let you choose between <code>cufftComplex</code> (interleaved re/im pairs)
        and planar layouts, VkFFT does the same, and WebGPU compute shaders usually reach for a
        <code>vec2&lt;f32&gt;</code>. Planes suit gpu.js because an output axis is free; the
        arithmetic underneath is identical whichever way the bytes are arranged.`,
      starterCode: `// Two planes, one bin: plane 0 real, plane 1 imaginary.
const gpu = new GPU({ mode });

const bin8 = gpu.createKernel(function (signal) {
  let acc = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = 2 * Math.PI * this.constants.k * i / this.constants.n;
    // TODO: branch on this.thread.y — plane 0 ADDS signal[i] * Math.cos(angle),
    // plane 1 SUBTRACTS signal[i] * Math.sin(angle).
    acc += signal[i] * Math.cos(angle);
  }
  return acc;
}, {
  output: [1, 2],
  constants: { n: 256, k: 8 },
});

function report(name, out) {
  const re = out[0][0];
  const im = out[1][0];
  // TODO: log the magnitude too — Math.sqrt(re * re + im * im)
  console.log(name, 'real:', re, 'imaginary:', im);
}

report('tone       ', bin8(tone));
report('shiftedTone', bin8(shiftedTone));
`,
      solutionCode: `// Two planes, one bin: plane 0 real, plane 1 imaginary.
const gpu = new GPU({ mode });

const bin8 = gpu.createKernel(function (signal) {
  let acc = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = 2 * Math.PI * this.constants.k * i / this.constants.n;
    if (this.thread.y === 0) acc += signal[i] * Math.cos(angle);
    else acc -= signal[i] * Math.sin(angle);
  }
  return acc;
}, {
  output: [1, 2],
  constants: { n: 256, k: 8 },
});

function report(name, out) {
  const re = out[0][0];
  const im = out[1][0];
  console.log(name, 'real:', re, 'imaginary:', im,
    'magnitude:', Math.sqrt(re * re + im * im));
}

report('tone       ', bin8(tone));
report('shiftedTone', bin8(shiftedTone));
`,
      inputs: () => ({ tone: tones([[8, 1]]), shiftedTone: delayedTone(8, 8) }),
      publicTests: [
        {
          name: 'the result is two planes of one bin',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(tones([[8, 1]]));
            ctx.assert(out && out.length === 2, `expected 2 planes, got ${out && out.length}`);
            ctx.assert(
              out[0] && typeof out[0] !== 'number' && out[0].length === 1,
              'each plane should hold one bin — is your output still 1D?'
            );
          },
        },
        {
          name: 'the plain tone is all real: <code>(128, 0)</code>',
          run: async ctx => {
            const signal = tones([[8, 1]]);
            const reference = referenceDft(signal);
            const expected = [reference[0][8], reference[1][8]];
            const out = ctx.kernel(signal);
            const got = p => (out[p] ? out[p][0] : NaN);
            const hint = diagnoseAll(2, got, p => expected[p], 0.05, [
              [p => (p === 0 ? reference[0][8] : -reference[1][8]), SIGN_FLIPPED],
              [p => (p === 0 ? reference[1][8] : reference[0][8]), PLANES_SWAPPED],
              [() => reference[0][8], BOTH_PLANES_REAL],
            ]);
            ctx.assertClose(got(0), expected[0], 0.05, hint || 'the real part of bin 8');
            ctx.assertClose(got(1), expected[1], 0.05, hint || 'the imaginary part of bin 8');
          },
        },
        {
          name: 'delayed by 8 samples it is all imaginary: <code>(0, −128)</code>',
          run: async ctx => {
            const signal = delayedTone(8, 8);
            const reference = referenceDft(signal);
            const expected = [reference[0][8], reference[1][8]];
            const out = ctx.kernel(signal);
            const got = p => (out[p] ? out[p][0] : NaN);
            const noTwoPi = referenceDft(signal, 1);
            const hint = diagnoseAll(2, got, p => expected[p], 0.05, [
              [p => (p === 0 ? reference[0][8] : -reference[1][8]), SIGN_FLIPPED],
              [p => (p === 0 ? reference[1][8] : reference[0][8]), PLANES_SWAPPED],
              [p => (p === 0 ? noTwoPi[0][8] : noTwoPi[1][8]), NO_TWO_PI],
              [() => reference[0][8], BOTH_PLANES_REAL],
            ]);
            ctx.assertClose(got(0), expected[0], 0.05, hint || 'the real part of the delayed tone');
            ctx.assertClose(got(1), expected[1], 0.05, hint || 'the imaginary part of the delayed tone');
          },
        },
        {
          name: 'both magnitudes are computed and logged as <code>128</code>',
          run: async ctx => {
            ctx.assert(
              someLogged(ctx.logs, 128, 0.5),
              'log the magnitude — Math.sqrt(re * re + im * im) is ≈128 for both signals'
            );
            ctx.assert(
              someLogged(ctx.logs, -128, 0.5),
              "log the imaginary part too — the delayed tone's is ≈−128, and that minus sign is " +
                'the whole point of the exponent'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // An arbitrary phase, not a quarter turn: the pair swings to some
            // unremarkable angle and the magnitude still reads 128.
            const signal = tones([[8, 1, 0.7]]);
            const reference = referenceDft(signal);
            const out = ctx.kernel(signal);
            const got = p => (out[p] ? out[p][0] : NaN);
            const hint = diagnoseAll(2, got, p => reference[p][8], 0.05, [
              [p => (p === 0 ? reference[0][8] : -reference[1][8]), SIGN_FLIPPED],
              [p => (p === 0 ? reference[1][8] : reference[0][8]), PLANES_SWAPPED],
              [() => reference[0][8], BOTH_PLANES_REAL],
            ]);
            ctx.assertClose(got(0), reference[0][8], 0.05, hint || 'the real part');
            ctx.assertClose(got(1), reference[1][8], 0.05, hint || 'the imaginary part');
            ctx.assertClose(
              Math.sqrt(got(0) * got(0) + got(1) * got(1)),
              128,
              0.05,
              'the magnitude must not depend on the phase'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'the-whole-spectrum',
      title: 'Every Bin, One Thread Each',
      intro: `<p>Nothing about bin 8 was special, and nothing about it depended on bin 9. Every bin
        is its own dot product over the same samples, and no bin reads another bin's answer — which
        makes the whole transform 512 threads that never speak to each other. Change
        <code>output: [1, 2]</code> to <code>output: [256, 2]</code>, take the bin from
        <code>this.thread.x</code> instead of a constant, and you are done.</p>
        <p>Be honest about the cost: every one of the 256 bins sums over all 256 samples, so this
        is O(n²) — 65,536 multiply-adds where the FFT needs about 2,000. The GPU does not care.
        Perfectly independent quadratic work is the shape hardware likes best, and this is the
        baseline any clever algorithm has to actually beat, not merely out-argue.</p>
        <p>The signal is three cosines buried in one buffer. Bin <code>k</code> covers
        <code>k × sampleRate / n</code> hertz — with a sample rate of 8,192 Hz and 256 samples,
        each bin is 32 Hz wide. Scan the first half of the spectrum only; task 4 explains why the
        second half has nothing new to say.</p>`,
      goal: `<strong>Goal:</strong> transform the whole signal with one kernel —
        <code>output: [256, 2]</code>, one thread per (bin, plane) — then find every bin in
        1…128 whose magnitude clears 20 and log its frequency in hertz.`,
      requirements: [
        '<code>output: [256, 2]</code>: <code>this.thread.x</code> is the bin, <code>this.thread.y</code> the plane',
        'Plane 0 adds <code>signal[i] * Math.cos(angle)</code>, plane 1 subtracts <code>signal[i] * Math.sin(angle)</code>',
        'In JavaScript, scan bins 1…128 and keep those with <code>Math.sqrt(re * re + im * im) &gt; 20</code>',
        'Log each survivor in <strong>hertz</strong>: <code>k * sampleRate / 256</code> — expect 320, 800 and 1280',
      ],
      hints: [
        {
          title: 'Hint 1 — the only two changes',
          body: `<p>The kernel body from the last task already works. Swap the constant bin for the
            thread's own:</p>
<pre><code>const angle = 2 * Math.PI * this.thread.x * i / this.constants.n;</code></pre>
<p>and widen the output to <code>[256, 2]</code>. That is the whole diff.</p>`,
        },
        {
          title: 'Hint 2 — reading the spectrum back',
          body: `<p><code>output: [256, 2]</code> comes back as two rows of 256:</p>
<pre><code>const spectrum = dft(signal);
const re = spectrum[0][k];
const im = spectrum[1][k];</code></pre>`,
        },
        {
          title: 'Hint 3 — bins are not hertz',
          body: `<p>A bin index is a count of whole cycles across the buffer. To turn it into a
            frequency you need to know how long the buffer <em>is</em>, and that is what the sample
            rate tells you — <code>n</code> alone cannot:</p>
<pre><code>const hz = k * sampleRate / 256;</code></pre>`,
        },
      ],
      transfer: `This kernel is a matrix–vector product in disguise: the DFT matrix times the
        signal, which is exactly how cuBLAS-style GEMV would express it and why a naive DFT is
        embarrassingly parallel on every platform. It is also the honest control in any FFT
        benchmark — cuFFT, VkFFT and FFTW all publish against it, and Measuring Speed Honestly
        has the methodology for running that comparison yourself.`,
      starterCode: `// 256 bins × 2 planes = 512 threads, none of them talking to each other.
const gpu = new GPU({ mode });

const dft = gpu.createKernel(function (signal) {
  let acc = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = 2 * Math.PI * this.thread.x * i / this.constants.n;
    // TODO: plane 0 (this.thread.y === 0) adds signal[i] * Math.cos(angle),
    // plane 1 subtracts signal[i] * Math.sin(angle).
    acc += signal[i] * Math.cos(angle);
  }
  return acc;
}, {
  output: [256, 2],
  constants: { n: 256 },
});

const spectrum = dft(signal);
console.log('bin 10 — real:', spectrum[0][10], 'imaginary:', spectrum[1][10]);

// TODO: scan bins 1…128, and for every bin whose magnitude clears 20,
// log its frequency in hertz: k * sampleRate / 256.
`,
      solutionCode: `// 256 bins × 2 planes = 512 threads, none of them talking to each other.
const gpu = new GPU({ mode });

const dft = gpu.createKernel(function (signal) {
  let acc = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = 2 * Math.PI * this.thread.x * i / this.constants.n;
    if (this.thread.y === 0) acc += signal[i] * Math.cos(angle);
    else acc -= signal[i] * Math.sin(angle);
  }
  return acc;
}, {
  output: [256, 2],
  constants: { n: 256 },
});

const spectrum = dft(signal);
console.log('bin 10 — real:', spectrum[0][10], 'imaginary:', spectrum[1][10]);

for (let k = 1; k <= 128; k++) {
  const re = spectrum[0][k];
  const im = spectrum[1][k];
  const magnitude = Math.sqrt(re * re + im * im);
  if (magnitude > 20) {
    console.log('bin', k, 'is', k * sampleRate / 256, 'Hz — magnitude', magnitude);
  }
}
`,
      inputs: () => ({
        signal: tones([[10, 1, 0.4], [25, 0.6, 0.9], [40, 0.3, -1.7]]),
        sampleRate: SAMPLE_RATE,
      }),
      publicTests: [
        {
          name: 'the spectrum is two planes of 256 bins',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(tones([[10, 1, 0.4], [25, 0.6, 0.9], [40, 0.3, -1.7]]));
            ctx.assert(out && out.length === 2, `expected 2 planes, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 256, 'each plane should hold 256 bins');
          },
        },
        {
          name: 'every bin of both planes matches the transform',
          run: async ctx => {
            const signal = tones([[10, 1, 0.4], [25, 0.6, 0.9], [40, 0.3, -1.7]]);
            const reference = referenceDft(signal);
            const out = ctx.kernel(signal);
            const got = flatPlanes(out);
            const expected = idx => (idx < N ? reference[0][idx] : reference[1][idx - N]);
            const hint = diagnoseAll(2 * N, got, expected, 0.05, spectrumProbes(signal, reference));
            for (const k of [0, 1, 10, 25, 40, 77, 128, 216, 231, 246, 255]) {
              ctx.assertClose(out[0][k], reference[0][k], 0.05, hint || `the real part of bin ${k}`);
              ctx.assertClose(out[1][k], reference[1][k], 0.05, hint || `the imaginary part of bin ${k}`);
            }
          },
        },
        {
          name: 'the three tones are reported in hertz: <code>320</code>, <code>800</code>, <code>1280</code>',
          run: async ctx => {
            const numbers = loggedNumbers(ctx.logs);
            const has = v => numbers.some(x => Math.abs(x - v) <= 0.5);
            const binsInstead = [10, 25, 40].every(has) && !has(320);
            const hint = binsInstead
              ? 'those are bin indices, not hertz — a bin index counts whole cycles across the ' +
                'buffer, and only the sample rate says how long the buffer is: ' +
                'hz = k * sampleRate / 256, which makes one bin 32 Hz here'
              : null;
            for (const hz of [320, 800, 1280]) {
              ctx.assert(has(hz), hint || `log ${hz} Hz — one of the three tones is at bin ${hz / 32}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Different tones, one of them on the DC bin, and a full sweep of
            // all 512 values rather than a sample of them.
            const signal = tones([[0, 0.5], [6, 0.8, -0.3], [51, 1, 2.2], [90, 0.4, 1.4]]);
            const reference = referenceDft(signal);
            const out = ctx.kernel(signal);
            ctx.assert(out && out.length === 2 && out[0].length === 256, 'expected a [2][256] spectrum');
            const got = flatPlanes(out);
            const expected = idx => (idx < N ? reference[0][idx] : reference[1][idx - N]);
            const hint = diagnoseAll(2 * N, got, expected, 0.05, spectrumProbes(signal, reference));
            for (let k = 0; k < N; k++) {
              ctx.assertClose(out[0][k], reference[0][k], 0.05, hint || `the real part of bin ${k}`);
              ctx.assertClose(out[1][k], reference[1][k], 0.05, hint || `the imaginary part of bin ${k}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'magnitude-and-mirror',
      title: 'Magnitude, Phase and the Mirror',
      intro: `<p>Two planes of numbers are rarely what you want to look at. The pair
        <code>(re, im)</code> is a point, and its two natural readings are the <strong>magnitude</strong>
        <code>√(re² + im²)</code> — how much of that frequency is present, immune to any shift —
        and the <strong>phase</strong> <code>atan2(im, re)</code> — where in its cycle it started.
        Task 2's delay moved the phase by a quarter turn and left the magnitude untouched; that is
        the same fact, seen from the other end.</p>
        <p>Then the part worth knowing: for a <strong>real</strong> input the spectrum is
        conjugate-symmetric. Bin <code>n − k</code> always holds the mirror of bin <code>k</code> —
        same magnitude, opposite phase — so the top half of every spectrum in this module has been
        redundant all along. Bins 0…n/2, and only those, decide everything. That redundancy is
        exactly what a real-input FFT sells when it claims to be twice as fast.</p>
        <p>A magnitude spectrum is also what a live analyser hands you. In a page (not here — a
        Web Worker has no <code>AudioContext</code> and no microphone, so every signal in this
        module is built in the content file instead) the wiring is:</p>
<pre><code>const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const audio = new AudioContext();
const analyser = audio.createAnalyser();
audio.createMediaStreamSource(stream).connect(analyser);

const samples = new Float32Array(analyser.fftSize);
analyser.getFloatTimeDomainData(samples);
const magnitudes = magnitude(Array.from(samples)); // ← this task's kernel</code></pre>`,
      goal: `<strong>Goal:</strong> write two kernels over <code>signal</code>, each
        <code>output: [256]</code> — one returning the magnitude of every bin, one returning the
        phase — then confirm the mirror in JavaScript and log how many bins are genuinely
        independent.`,
      requirements: [
        'Both kernels keep <strong>two</strong> accumulators, <code>re</code> and <code>im</code>, in one loop',
        '<code>magnitude</code> returns <code>Math.sqrt(re * re + im * im)</code>; <code>phase</code> returns <code>Math.atan2(im, re)</code>',
        'Check the mirror in JavaScript: <code>mag[k]</code> against <code>mag[256 - k]</code>',
        'Log the count of independent bins — 0…128 inclusive, so <code>129</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — two accumulators, one pass',
          body: `<p>Nothing needs a second loop: the cosine and the sine share an angle, so read
            the sample once and feed both totals.</p>
<pre><code>let re = 0;
let im = 0;
for (let i = 0; i &lt; this.constants.n; i++) {
  const angle = 2 * Math.PI * this.thread.x * i / this.constants.n;
  re += signal[i] * Math.cos(angle);
  im -= signal[i] * Math.sin(angle);
}</code></pre>`,
        },
        {
          title: 'Hint 2 — the magnitude goes last',
          body: `<p>Take the square root <em>after</em> the loop, on the finished pair. Doing it
            inside — accumulating <code>√(…)</code> per term — throws the phase away before the sum
            can use it, and every bin ends up holding the same number.
            <code>Math.hypot</code> is not on gpu.js's whitelist, so write
            <code>Math.sqrt(re * re + im * im)</code>.</p>`,
        },
        {
          title: 'Hint 3 — checking the mirror',
          body: `<p>A plain loop over the bottom half, comparing each bin with its partner:</p>
<pre><code>let worst = 0;
for (let k = 1; k &lt; 128; k++) {
  worst = Math.max(worst, Math.abs(mag[k] - mag[256 - k]));
}
console.log('largest mirror difference:', worst);</code></pre>
<p>Bins 0 and 128 are their own partners, which is why the independent count is
            <code>128 + 1</code> and not 128.</p>`,
        },
      ],
      transfer: `Every serious FFT library sells this symmetry as an API: FFTW's
        <code>r2c</code> plans, cuFFT's <code>R2C</code> transform and rocFFT's real-to-complex
        mode all return n/2 + 1 bins and refuse to compute the rest, halving both the work and the
        memory. Reading a spectrum as magnitude and phase instead of real and imaginary is the same
        change of coordinates Colour Spaces makes when it trades RGB for hue and value.`,
      starterCode: `// One thread per bin, two accumulators inside it.
const gpu = new GPU({ mode });

const magnitude = gpu.createKernel(function (signal) {
  let re = 0;
  let im = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = 2 * Math.PI * this.thread.x * i / this.constants.n;
    re += signal[i] * Math.cos(angle);
    // TODO: accumulate the imaginary part as well — it SUBTRACTS
    // signal[i] * Math.sin(angle).
  }
  // TODO: return the magnitude of the finished pair, not just re
  return re;
}, {
  output: [256],
  constants: { n: 256 },
});

const phase = gpu.createKernel(function (signal) {
  // TODO: the same two accumulators, but return Math.atan2(im, re)
  return 0;
}, {
  output: [256],
  constants: { n: 256 },
});

const mag = magnitude(signal);
const ph = phase(signal);
console.log('peak bin 12:', mag[12], 'phase', ph[12]);

// TODO: compare mag[k] with mag[256 - k] across the bottom half, and log how
// many bins are genuinely independent (bins 0 … 128 inclusive).
`,
      solutionCode: `// One thread per bin, two accumulators inside it.
const gpu = new GPU({ mode });

const magnitude = gpu.createKernel(function (signal) {
  let re = 0;
  let im = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = 2 * Math.PI * this.thread.x * i / this.constants.n;
    re += signal[i] * Math.cos(angle);
    im -= signal[i] * Math.sin(angle);
  }
  return Math.sqrt(re * re + im * im);
}, {
  output: [256],
  constants: { n: 256 },
});

const phase = gpu.createKernel(function (signal) {
  let re = 0;
  let im = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = 2 * Math.PI * this.thread.x * i / this.constants.n;
    re += signal[i] * Math.cos(angle);
    im -= signal[i] * Math.sin(angle);
  }
  return Math.atan2(im, re);
}, {
  output: [256],
  constants: { n: 256 },
});

const mag = magnitude(signal);
const ph = phase(signal);
console.log('peak bin 12:', mag[12], 'phase', ph[12]);
console.log('its mirror 244:', mag[244], 'phase', ph[244]);

let worst = 0;
for (let k = 1; k < 128; k++) {
  worst = Math.max(worst, Math.abs(mag[k] - mag[256 - k]));
}
console.log('largest mirror difference:', worst);

const independent = 256 / 2 + 1;
console.log('independent bins:', independent);
`,
      inputs: () => ({ signal: tones([[0, 0.25], [12, 0.8, 1.1], [33, 0.45, -0.6]]) }),
      publicTests: [
        {
          name: 'a magnitude kernel and a phase kernel, 256 bins each',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const { magnitude, phase, candidates, probe } = magnitudeAndPhase(ctx);
            const flat = flatSpectrumHint(candidates, probe);
            ctx.assert(
              magnitude,
              flat || 'no magnitude kernel found — on a unit cosine at bin 8, one kernel should ' +
                'answer 256 non-negative values that peak at bin 8'
            );
            ctx.assert(!flat, flat); // a flat "spectrum" is a spectrum of nothing
            ctx.assert(
              phase,
              'no phase kernel found — one kernel should answer 256 angles, all of them inside ±π'
            );
          },
        },
        {
          name: 'magnitude: <code>64</code> at bin 0, <code>102.4</code> at bin 12, <code>57.6</code> at bin 33',
          run: async ctx => {
            const { magnitude, candidates, probe } = magnitudeAndPhase(ctx);
            ctx.assert(magnitude, flatSpectrumHint(candidates, probe) || 'no magnitude kernel found');
            const signal = tones([[0, 0.25], [12, 0.8, 1.1], [33, 0.45, -0.6]]);
            const reference = referenceDft(signal);
            const flat = sumAbsOf(signal);
            const out = magnitude(signal);
            for (const k of [0, 12, 33, 50, 128, 223, 244]) {
              const expected = magnitudeOf(reference, k);
              const hint = diagnose(out[k], expected, 0.05, [
                [flat,
                  'that is Σ|signal[i]| — the magnitude was taken of each TERM and those summed, ' +
                  'which destroys the phase before it can cancel anything. Accumulate re and im ' +
                  'separately and take the magnitude once, after the loop'],
                [expected * expected,
                  'that is the magnitude SQUARED — re * re + im * im still needs its Math.sqrt'],
                [reference[0][k], 'that is the real part alone — the imaginary part never joined in'],
              ]);
              ctx.assertClose(out[k], expected, 0.05, hint || `the magnitude of bin ${k}`);
            }
          },
        },
        {
          name: 'phase: <code>1.1</code> at bin 12, <code>−0.6</code> at bin 33',
          run: async ctx => {
            const { phase } = magnitudeAndPhase(ctx);
            ctx.assert(phase, 'no phase kernel found');
            const signal = tones([[0, 0.25], [12, 0.8, 1.1], [33, 0.45, -0.6]]);
            const reference = referenceDft(signal);
            const out = phase(signal);
            // Only bins with real energy in BOTH parts: where the magnitude is
            // float noise the phase is float noise too, and a bin whose
            // imaginary part is zero sits on atan2's ±π seam, where a hair of
            // float error flips the answer by a full turn. Neither is a fair
            // thing to assert on. Bins 12, 33 and their mirrors are neither.
            for (const k of [12, 33, 223, 244]) {
              const expected = Math.atan2(reference[1][k], reference[0][k]);
              const hint = diagnose(out[k], expected, 5e-3, [
                [Math.atan2(reference[0][k], reference[1][k]),
                  'Math.atan2 takes (y, x) — the imaginary part comes first: Math.atan2(im, re)'],
                [-expected,
                  'the phase came back negated — check the sign on the imaginary accumulator, ' +
                  'which subtracts signal[i] * Math.sin(angle)'],
              ]);
              ctx.assertClose(out[k], expected, 5e-3, hint || `the phase of bin ${k}`);
            }
          },
        },
        {
          name: 'the mirror is checked and <code>129</code> independent bins are logged',
          run: async ctx => {
            ctx.assert(
              someLogged(ctx.logs, 129, 0.01),
              'log how many bins are genuinely independent — bins 0 through 128 inclusive, so 129'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { magnitude, phase, candidates, probe } = magnitudeAndPhase(ctx);
            ctx.assert(magnitude, flatSpectrumHint(candidates, probe) || 'no magnitude kernel found');
            ctx.assert(phase, 'no phase kernel found');
            const signal = tones([[0, -0.4], [7, 1, 0.25], [44, 0.5, -2.4], [61, 0.9, 1.9]]);
            const reference = referenceDft(signal);
            const flat = sumAbsOf(signal);
            const mag = magnitude(signal);
            const ph = phase(signal);
            for (let k = 0; k < N; k++) {
              const expected = magnitudeOf(reference, k);
              const hint = diagnose(mag[k], expected, 0.05, [
                [flat, 'every bin holds Σ|signal[i]| — the magnitude was summed per term instead of ' +
                  'being taken once from the finished (re, im) pair'],
                [expected * expected, 'that is the magnitude squared — the Math.sqrt is missing'],
              ]);
              ctx.assertClose(mag[k], expected, 0.05, hint || `the magnitude of bin ${k}`);
              // the mirror, asserted on the learner's own output rather than
              // merely described in the brief
              ctx.assertClose(mag[k], mag[(N - k) % N], 0.05,
                `bin ${k} and bin ${(N - k) % N} must come back with the same magnitude`);
              // Both parts have to carry real energy before the phase is worth
              // asserting on — see the public phase test for why the imaginary
              // one matters as much as the magnitude.
              if (expected > 1 && Math.abs(reference[1][k]) > 1) {
                const expectedPhase = Math.atan2(reference[1][k], reference[0][k]);
                const phaseHint = diagnose(ph[k], expectedPhase, 5e-3, [
                  [Math.atan2(reference[0][k], reference[1][k]),
                    'Math.atan2 takes (y, x) — the imaginary part comes first: Math.atan2(im, re)'],
                  [-expectedPhase, 'the phase came back negated — check the sign on the imaginary accumulator'],
                ]);
                ctx.assertClose(ph[k], expectedPhase, 5e-3, phaseHint || `the phase of bin ${k}`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'inverse-and-round-trip',
      title: 'There and Back Again',
      intro: `<p>A transform earns the name by being reversible. The inverse DFT is the forward one
        with two edits: the exponent's sign flips to <code>+</code>, and the whole thing is divided
        by <code>n</code>. Both edits are load-bearing, and both are where people go wrong —
        forget the <code>1/n</code> and your signal comes back 256 times too loud; leave the sign
        negative and it comes back <em>time-reversed</em>, which is a spectacular way to fail a
        test that only checks sample 0.</p>
        <p>Summing complex terms means real arithmetic on pairs. With <code>c = cos(angle)</code>
        and <code>s = sin(angle)</code>, one term of <code>(re + i·im) · (c + i·s)</code> is:</p>
<pre><code>real      += re * c - im * s;
imaginary += re * s + im * c;</code></pre>
        <p>The shapes line up for free. <code>output: [n, 2]</code> hands back exactly the
        <code>[2][n]</code> nested array a kernel wants as input — <code>spectrum[0][k]</code> real,
        <code>spectrum[1][k]</code> imaginary — so the forward transform's result drops into the
        inverse with nothing in between. The original signal is real, so plane 1 of the round trip
        should come back as float noise: that residue is a free correctness check.</p>`,
      goal: `<strong>Goal:</strong> write the inverse transform — <code>output: [256, 2]</code>,
        taking the <code>[2][256]</code> spectrum — and confirm that transforming
        <code>signal</code> and transforming it back recovers all 256 samples.`,
      requirements: [
        'Loop over <strong>bins</strong> <code>k</code>, and use <code>this.thread.x</code> as the SAMPLE index',
        'The exponent turns positive: <code>real += re * c - im * s</code>, <code>imaginary += re * s + im * c</code>',
        'Divide the finished sum by <code>this.constants.n</code> — once, at the end',
        'Log how many of the 256 samples come back within <code>1e-3</code> — all of them should',
      ],
      hints: [
        {
          title: 'Hint 1 — the loop runs the other way round',
          body: `<p>The forward kernel's thread owned a bin and looped over samples. Here the
            thread owns a <em>sample</em> and loops over bins, so <code>this.thread.x</code> is
            <code>i</code> and the loop variable is <code>k</code>:</p>
<pre><code>const angle = 2 * Math.PI * k * this.thread.x / this.constants.n;</code></pre>`,
        },
        {
          title: 'Hint 2 — the two planes of the answer',
          body: `<p>Same branch as the forward transform, over the complex product:</p>
<pre><code>const c = Math.cos(angle);
const s = Math.sin(angle);
if (this.thread.y === 0) acc += spectrum[0][k] * c - spectrum[1][k] * s;
else acc += spectrum[0][k] * s + spectrum[1][k] * c;</code></pre>`,
        },
        {
          title: 'Hint 3 — the scaling',
          body: `<p><code>return acc / this.constants.n;</code> — once, on the finished sum, not
            inside the loop. Forward does not scale, inverse divides by n; put the 1/n on the wrong
            side and the round trip is off by a factor of 65,536.</p>`,
        },
      ],
      transfer: `The 1/n is a convention, not a law, and libraries disagree loudly about it: FFTW
        computes an unnormalised inverse and leaves the scaling to you, numpy puts 1/n on
        <code>ifft</code>, and cuFFT documents its own choice per plan — which is why "my round
        trip is n times too big" is one of the most-asked FFT questions on every platform. The
        sign of the exponent is equally conventional; what is not optional is that the forward and
        inverse pair disagree about it.`,
      starterCode: `// Forward, then back again. The spectrum's [2][256] shape is exactly what
// an output: [256, 2] kernel hands you — no reshaping in between.
const gpu = new GPU({ mode });

// The forward transform, unchanged from the last task.
const dft = gpu.createKernel(function (signal) {
  let acc = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = 2 * Math.PI * this.thread.x * i / this.constants.n;
    if (this.thread.y === 0) acc += signal[i] * Math.cos(angle);
    else acc -= signal[i] * Math.sin(angle);
  }
  return acc;
}, {
  output: [256, 2],
  constants: { n: 256 },
});

const idft = gpu.createKernel(function (spectrum) {
  // TODO: loop over the 256 BINS k, with this.thread.x as the sample index.
  // The exponent turns positive here:
  //   plane 0 accumulates spectrum[0][k] * c - spectrum[1][k] * s
  //   plane 1 accumulates spectrum[0][k] * s + spectrum[1][k] * c
  // and the finished sum is divided by n.
  return spectrum[this.thread.y][this.thread.x];
}, {
  output: [256, 2],
  constants: { n: 256 },
});

const spectrum = dft(signal);
const back = idft(spectrum);

let worst = 0;
let matched = 0;
for (let i = 0; i < 256; i++) {
  worst = Math.max(worst, Math.abs(back[0][i] - signal[i]), Math.abs(back[1][i]));
  if (Math.abs(back[0][i] - signal[i]) < 1e-3 && Math.abs(back[1][i]) < 1e-3) matched++;
}
console.log('largest round-trip error:', worst.toExponential(2));
console.log('samples recovered within 1e-3:', matched);
`,
      solutionCode: `// Forward, then back again. The spectrum's [2][256] shape is exactly what
// an output: [256, 2] kernel hands you — no reshaping in between.
const gpu = new GPU({ mode });

// The forward transform, unchanged from the last task.
const dft = gpu.createKernel(function (signal) {
  let acc = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = 2 * Math.PI * this.thread.x * i / this.constants.n;
    if (this.thread.y === 0) acc += signal[i] * Math.cos(angle);
    else acc -= signal[i] * Math.sin(angle);
  }
  return acc;
}, {
  output: [256, 2],
  constants: { n: 256 },
});

const idft = gpu.createKernel(function (spectrum) {
  let acc = 0;
  for (let k = 0; k < this.constants.n; k++) {
    const angle = 2 * Math.PI * k * this.thread.x / this.constants.n;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    if (this.thread.y === 0) acc += spectrum[0][k] * c - spectrum[1][k] * s;
    else acc += spectrum[0][k] * s + spectrum[1][k] * c;
  }
  return acc / this.constants.n;
}, {
  output: [256, 2],
  constants: { n: 256 },
});

const spectrum = dft(signal);
const back = idft(spectrum);

let worst = 0;
let matched = 0;
for (let i = 0; i < 256; i++) {
  worst = Math.max(worst, Math.abs(back[0][i] - signal[i]), Math.abs(back[1][i]));
  if (Math.abs(back[0][i] - signal[i]) < 1e-3 && Math.abs(back[1][i]) < 1e-3) matched++;
}
console.log('largest round-trip error:', worst.toExponential(2));
console.log('samples recovered within 1e-3:', matched);
`,
      inputs: () => ({ signal: pluck() }),
      publicTests: [
        {
          name: 'a forward kernel and an inverse kernel',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const { forward, inverse } = transformKernels(ctx);
            ctx.assert(forward, 'no kernel was invoked with a 256-sample signal — is the forward transform still being called?');
            ctx.assert(
              inverse,
              'no kernel was invoked with a [2][256] spectrum — feed the forward transform\'s ' +
                'result straight into the inverse one'
            );
          },
        },
        {
          name: 'the inverse of a known spectrum is the signal it came from',
          run: async ctx => {
            const { inverse } = transformKernels(ctx);
            ctx.assert(inverse, 'no inverse kernel found');
            const signal = pluck();
            const reference = referenceDft(signal);
            const out = inverse(asPlanes(reference[0], reference[1]));
            ctx.assert(out && out.length === 2 && out[0].length === 256,
              'the inverse should return [2][256] — plane 0 real, plane 1 imaginary');
            const reversed = i => signal[(N - i) % N];
            const hint = diagnoseAll(N, i => out[0][i], i => signal[i], 5e-3, [
              [i => signal[i] * N,
                'every sample came back n times too large — the inverse divides the finished sum ' +
                'by this.constants.n, and that division is missing', 1],
              [reversed,
                'the signal came back TIME-REVERSED: sample i holds sample n - i. The inverse ' +
                'transform flips the sign of the exponent, so its imaginary term ADDS ' +
                '(spectrum[0][k] * s + spectrum[1][k] * c) where the forward one subtracts'],
              [i => reversed(i) * N,
                'time-reversed AND n times too large — the exponent kept the forward sign and the ' +
                '1/n never happened', 1],
            ]);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[0][i], signal[i], 5e-3, hint || `sample ${i}`);
            }
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[1][i], 0, 5e-3,
                `sample ${i} of the imaginary plane — the original signal is real, so this plane ` +
                'must cancel to nothing');
            }
          },
        },
        {
          name: 'all <code>256</code> samples are recovered, and the count is logged',
          run: async ctx => {
            ctx.assert(
              someLogged(ctx.logs, 256, 0.01),
              'log how many samples came back within 1e-3 — all 256 of them should'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A fresh signal through the learner's OWN pair of kernels: forward
            // out, inverse back, nothing of the reference in between.
            const { forward, inverse } = transformKernels(ctx);
            ctx.assert(forward && inverse, 'expected a forward and an inverse kernel');
            const signal = tones([[0, 0.3], [4, 1, 1.3], [29, 0.65, -0.8], [77, 0.2, 2.9]]);
            const back = inverse(forward(signal));
            const reversed = i => signal[(N - i) % N];
            const hint = diagnoseAll(N, i => back[0][i], i => signal[i], 5e-3, [
              [i => signal[i] * N,
                'the round trip came back scaled by n — the inverse is missing its 1/n', 1],
              [reversed,
                "the round trip came back time-reversed — the inverse kept the forward transform's exponent sign"],
            ]);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(back[0][i], signal[i], 5e-3, hint || `sample ${i} of the round trip`);
              ctx.assertClose(back[1][i], 0, 5e-3, `sample ${i} of the imaginary residue`);
            }
          },
        },
      ],
    },
  ],
};
