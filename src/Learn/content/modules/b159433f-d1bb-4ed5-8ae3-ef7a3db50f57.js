// Module: Autocorrelation & Pitch — uuid b159433f-d1bb-4ed5-8ae3-ef7a3db50f57
// (short id b159433f). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module is new.
//
// Signal Processing — Autocorrelation & Pitch.
//
// Five tasks: correlate a signal with a shifted copy of itself, one thread per
// lag → turn the peak lag into hertz, with parabolic interpolation for the
// digits between the samples → the three ways a naive peak-pick lies (lag 0,
// the shrinking overlap, the octave error) → the missing fundamental, where
// the spectrum's loudest peak is an octave above the note you hear → the same
// curve via the FFT (Wiener–Khinchin), verified against brute force and timed.
//
// The thesis: pitch is not the loudest frequency. It is the shift at which a
// signal first agrees with itself, and that is a correlation question — this
// module is Template Matching's normalised cross-correlation with the template
// replaced by the signal itself.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested numeric arrays as inputs, this.thread.* for indexing,
// this.constants.* for compile-time loop bounds. The complex convention this
// track shares: a kernel writes output: [n, 2] read back as result[plane][i]
// (plane 0 real, plane 1 imaginary) and reads a nested [2][n] argument as
// data[0][i] / data[1][i]. Every task passes in CPU mode.
//
// NO AUDIO, MEASURED NOT ASSUMED. Learner code runs in a Web Worker, which has
// no AudioContext, no OfflineAudioContext and no navigator.mediaDevices. Every
// signal here is therefore synthesised deterministically below; task 2 shows
// the real getUserMedia wiring as a code block and says plainly why it cannot
// run here.
//
// FLOAT MARGINS. The GL backend computes in float32 and the tests compute in
// float64; gpu.js hands back Float32Arrays on the CPU backend too, so both
// paths carry float32 storage error. Measured in a real browser through the
// worker sandbox, against a float64 reference computed in the page:
//   • raw correlation, sums over 512 samples peaking at 140.57 — worst drift
//     8.8e-5, asserted at ±0.02. Margin 228×.
//   • normalised score, O(1) — worst drift 1.3e-6, asserted at ±2e-3. 1600×.
//   • the parabolic refinement divides differences of numbers ~150 whose gaps
//     are ~30 — worst drift 1.9e-6 lags / 4.4e-5 Hz, asserted at ±0.02 lags and
//     ±0.5 Hz, while the two mistakes it must be told apart from sit 0.37 lags
//     and 8.7 Hz away.
//   • the two-plane DFT, bins peaking at 256 — worst drift 7.8e-3 under
//     SwiftShader (which is what headless verification runs on, and the harsher
//     of the two), asserted at ±0.1. Margin 13×.
//   • the FFT route's intermediates reach 1.4e5 before the 1/1024 — worst
//     disagreement with brute force 1.07e-4, asserted at ±0.05. Margin 470×.
//
// ONE PROBE NEEDS ITS OWN EPSILON, AND MUST KEEP IT. A mistake that scales the
// answer by n scales its float error by n too: the "missing 1/1024" probe in
// task 5 predicts 143,806, and the value the GPU actually produces is 1024×
// a float32 result. It is given eps 500 rather than the test's own tolerance
// for exactly that reason — measured at 143,806.54 on the backend against
// 143,806.56 in float64. Tighten it and the probe goes silent on the GPU while
// still firing in cpu mode, which is the worst possible way for it to fail.
// Nothing asserted here sits near a decision boundary. The one genuine tie in
// the module (task 3, where every EVEN multiple of the period scores exactly
// 1.000, so "the tallest peak" is decided by float noise) is deliberately never
// asserted on — it is diagnosed instead, with one message covering 2×, 4× and
// 6× so whichever one the hardware picks gets named.

// ---- the sound ------------------------------------------------------------

const SR = 8192; // sample rate, Hz. A power of two, so periods come out exact.
const N = 512; // samples per analysis window — 62.5 ms
const MAX_LAG = 256; // lags 0…255: periods down to 32 Hz
const MIN_LAG = 8; // 8 samples = 1024 Hz — above any instrument's fundamental
const THRESHOLD = 0.8; // "this peak is real" cut on the normalised score
const PAD = 1024; // zero-padded length for the FFT route

// Four decimal places: clean in the Task inputs panel, and identical in the
// test and in the kernel, so an expectation computed here is what the kernel
// actually saw.
function q4(value) {
  return Math.round(value * 10000) / 10000;
}

// A sum of harmonics under an envelope — the shape a plucked string, a bowed
// note or a vowel actually has. `partials` are [harmonic, amplitude, phase];
// `tau` decays the whole thing; `alternate` makes every other period quieter,
// which is a real thing voices do and the cause of task 3's octave trap.
function synth({ f0, partials, tau = Infinity, alternate = 0, period = 0, n = N }) {
  const x = new Array(n);
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let p = 0; p < partials.length; p++) {
      const harmonic = partials[p];
      v += harmonic[1] * Math.sin((2 * Math.PI * harmonic[0] * f0 * i) / SR + (harmonic[2] || 0));
    }
    let amp = Math.exp(-i / tau);
    if (alternate) amp *= Math.floor(i / period) % 2 === 0 ? 1 + alternate : 1 - alternate;
    x[i] = q4(v * amp);
  }
  return x;
}

// Task 1 and 5: a plucked string at 256 Hz — period exactly 32 samples.
const TONE = { f0: 256, partials: [[1, 1], [2, 0.6], [3, 0.35], [4, 0.2]], tau: 400 };
// Task 1's private test: the same pluck an octave down, period 64.
const TONE_LOW = { f0: 128, partials: [[1, 1], [2, 0.6], [3, 0.35], [4, 0.2]], tau: 400 };

// Task 2: concert A. 8192 / 440 = 18.618 samples — deliberately nowhere near
// an integer, which is the whole point of the task.
const NOTE_A = { f0: 440, partials: [[1, 1], [2, 0.5], [3, 0.3], [4, 0.15]], tau: 600 };
const NOTE_E = { f0: 330, partials: [[1, 1], [2, 0.5], [3, 0.3], [4, 0.15]], tau: 600 };

// Task 3: period 40 samples (204.8 Hz), with every other period 15% quieter.
// The waveform still repeats every 40 samples and a listener still hears
// 204.8 Hz — but a copy shifted by 80 samples matches fractionally better, so
// the normalised score peaks at 1.000 at every EVEN multiple of the period and
// at 0.956 at the odd ones. Global-maximum peak picking therefore reports an
// octave too low. That is the classic octave error, built to order.
const VOICED = {
  f0: SR / 40, partials: [[1, 1], [2, 0.5], [3, 0.25], [4, 0.12]], alternate: 0.15, period: 40,
};
const VOICED_LOW = {
  f0: SR / 50, partials: [[1, 1], [2, 0.5], [3, 0.25], [4, 0.12]], alternate: 0.15, period: 50,
};

// Task 4: harmonics 2, 3 and 4 of 128 Hz — 256, 384 and 512 Hz — and nothing
// at 128 Hz at all. With SR / N = 16 Hz per bin those land on bins 16, 24 and
// 32 exactly, so the spectrum is clean and bin 8 (128 Hz) is empty to 1e-3.
// The waveform still repeats every 64 samples, because 128 Hz is what the
// three partials have in common.
const MISSING = { f0: 128, partials: [[2, 1, 0.6], [3, 0.7, 2.1], [4, 0.5, 4.0]] };
const MISSING_LOW = { f0: 64, partials: [[2, 1, 1.3], [3, 0.7, 0.2], [4, 0.5, 2.7]] };

const makeTone = () => synth(TONE);
const makeToneLow = () => synth(TONE_LOW);
const makeNoteA = () => synth(NOTE_A);
const makeNoteE = () => synth(NOTE_E);
const makeVoiced = () => synth(VOICED);
const makeVoicedLow = () => synth(VOICED_LOW);
const makeMissing = () => synth(MISSING);
const makeMissingLow = () => synth(MISSING_LOW);

// ---- reference implementations (host side, for the tests) -----------------

// Raw autocorrelation: the overlap shrinks as the lag grows, which is exactly
// the bias task 3 has to talk about.
function rawAuto(x, maxLag) {
  const r = new Array(maxLag);
  for (let lag = 0; lag < maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i + lag < x.length; i++) sum += x[i] * x[i + lag];
    r[lag] = sum;
  }
  return r;
}

// Normalised (Template Matching's NCC, against a shifted copy of the same
// signal): each lag divided by the geometric mean of the two overlapping
// slices' energies, so rho[0] is exactly 1 and lags are comparable.
function nccAuto(x, maxLag) {
  const rho = new Array(maxLag);
  for (let lag = 0; lag < maxLag; lag++) {
    let dot = 0;
    let head = 0;
    let tail = 0;
    for (let i = 0; i + lag < x.length; i++) {
      dot += x[i] * x[i + lag];
      head += x[i] * x[i];
      tail += x[i + lag] * x[i + lag];
    }
    rho[lag] = dot / Math.sqrt(head * tail);
  }
  return rho;
}

// The energy of the SHIFTED slice alone — what a kernel returns when both
// factors got the + lag by mistake.
function tailEnergy(x, maxLag) {
  const e = new Array(maxLag);
  for (let lag = 0; lag < maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i + lag < x.length; i++) sum += x[i + lag] * x[i + lag];
    e[lag] = sum;
  }
  return e;
}

// The energy of the UNSHIFTED slice alone — what a kernel returns when the
// second factor never got its + lag.
function headEnergy(x, maxLag) {
  const e = new Array(maxLag);
  for (let lag = 0; lag < maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i + lag < x.length; i++) sum += x[i] * x[i];
    e[lag] = sum;
  }
  return e;
}

// dot / head — the same energy used twice instead of once from each slice.
function halfNormalised(x, maxLag) {
  const r = new Array(maxLag);
  for (let lag = 0; lag < maxLag; lag++) {
    let dot = 0;
    let head = 0;
    for (let i = 0; i + lag < x.length; i++) {
      dot += x[i] * x[i + lag];
      head += x[i] * x[i];
    }
    r[lag] = dot / head;
  }
  return r;
}

function argmaxFrom(values, lo, hi) {
  let best = lo;
  for (let i = lo; i <= hi; i++) if (values[i] > values[best]) best = i;
  return best;
}

// The first LOCAL peak at or above `threshold` — the fix task 3 arrives at.
function firstPeakOver(values, lo, hi, threshold) {
  for (let i = lo; i <= hi; i++) {
    if (values[i] >= threshold && values[i] > values[i - 1] && values[i] >= values[i + 1]) return i;
  }
  return -1;
}

// Sub-sample peak offset from three samples around the peak. Negative when the
// true peak lies to the LEFT of the sampled one, which is the case that catches
// people: the refinement has to move the answer down, not up.
function parabolicDelta(values, p) {
  const a = values[p - 1];
  const b = values[p];
  const c = values[p + 1];
  return (0.5 * (a - c)) / (a - 2 * b + c);
}

// Reference DFT in the track's two-plane layout: planes[0][k] real,
// planes[1][k] imaginary, k over all N bins.
function dftPlanes(x) {
  const n = x.length;
  const re = new Array(n).fill(0);
  const im = new Array(n).fill(0);
  for (let k = 0; k < n; k++) {
    let sr = 0;
    let si = 0;
    for (let i = 0; i < n; i++) {
      const angle = (-2 * Math.PI * k * i) / n;
      sr += x[i] * Math.cos(angle);
      si += x[i] * Math.sin(angle);
    }
    re[k] = sr;
    im[k] = si;
  }
  return [re, im];
}

function magnitudesOf(planes) {
  const mag = new Array(planes[0].length);
  for (let k = 0; k < mag.length; k++) mag[k] = Math.hypot(planes[0][k], planes[1][k]);
  return mag;
}

// The zero-padded complex input the FFT route starts from. Float32Arrays from
// the first call: every pass returns Float32Arrays and gpu.js locks an
// argument's container type on the first invocation (gpujs/gpu.js#857).
function paddedSignal(x) {
  const re = new Float32Array(PAD);
  const im = new Float32Array(PAD);
  for (let i = 0; i < x.length; i++) re[i] = x[i];
  return [re, im];
}

// Drive a butterfly-pass kernel log2(PAD) = 10 times. sign −1 forward, +1 back.
function runTransform(pass, data, sign) {
  let cur = data;
  for (let ns = 1; ns < PAD; ns *= 2) cur = pass(cur, ns, sign);
  return cur;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports two numbers tells a learner nothing about WHICH
// slip produced them. A probe pairs the value one specific known mistake would
// produce with a sentence naming it; diagnose() speaks only when the observed
// value matches a probe within the test's own tolerance AND the correct value
// does not — so a lag where two candidates coincide (lag 0, where the raw sum
// and the tail energy are the same number) stays silent, as do observations
// matching probes that disagree with each other. A wrong diagnosis is worse
// than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The same idea for a whole CURVE. One matching lag is weak evidence — the raw
// sum and the tail energy agree at lag 0, and every candidate agrees wherever
// the correlation happens to cross — so a probe must predict EVERY lag the test
// looks at, and disagree with the right answer somewhere, before it may speak.
function diagnoseCurve(lags, got, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (const lag of lags) {
        if (!(Math.abs(got(lag) - value(lag)) <= eps)) return false;
        if (Math.abs(expected(lag) - value(lag)) > eps) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Every number that appeared in a console.log line. When any line mentions
// `label`, only those lines count — so "period: 40" and "pitch: 204.8 Hz" can
// be told apart even when one task's right answer is another's wrong one.
function loggedNumbers(logs, label) {
  const scan = lines => {
    const out = [];
    for (const line of lines) {
      const matches = line.match(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi);
      if (matches) for (const m of matches) out.push(parseFloat(m));
    }
    return out;
  };
  const all = logs.filter(l => l.type === 'log' && l.text).map(l => l.text);
  if (label) {
    const tagged = all.filter(text => text.toLowerCase().includes(label));
    if (tagged.length) return scan(tagged);
  }
  return scan(all);
}

// A reported answer is right when SOME logged number matches it. Only when
// none does may a probe speak about the numbers that are there.
function reportedHint(nums, expected, eps, probes) {
  if (nums.some(v => Math.abs(v - expected) <= eps)) return null;
  const hits = probes
    .filter(p => nums.some(v => Math.abs(v - p[0]) <= (p[2] === undefined ? eps : p[2])))
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// ---- kernel discovery (tasks 4 and 5 build more than one) -----------------

function outputOf(k) {
  try {
    return k.kernel && k.kernel.output ? Array.from(k.kernel.output) : null;
  } catch (e) {
    return null;
  }
}

function argCountOf(k) {
  try {
    return k.kernel && k.kernel.argumentNames ? k.kernel.argumentNames.length : -1;
  } catch (e) {
    return -1;
  }
}

function kernelsShaped(ctx, width, height, args) {
  return ctx.kernels.filter(k => {
    const out = outputOf(k);
    if (!out || out[0] !== width) return false;
    if (height == null ? out.length > 1 && out[1] > 1 : out[1] !== height) return false;
    return args == null || argCountOf(k) === args;
  });
}

// The kernel that answers like a plain correlation: 256 lags from one argument.
function findCorrelator(ctx) {
  return kernelsShaped(ctx, MAX_LAG, null, 1)[0] || null;
}

// The two-plane spectrum kernel: [N, 2] out, one argument in.
function findSpectrum(ctx) {
  return kernelsShaped(ctx, N, 2, 1)[0] || null;
}

// ---- shared prose ---------------------------------------------------------

const COMPLEX_NOTE =
  `<p class="layout-note"><strong>Two planes, one complex array.</strong> gpu.js has no complex
    number, so this track carries one as two planes of floats: a kernel declares
    <code>output: [n, 2]</code> and the result reads <code>result[0][i]</code> for the real part
    and <code>result[1][i]</code> for the imaginary part. Going the other way, a complex
    <em>argument</em> is a nested <code>[2][n]</code> array read as <code>data[0][i]</code> and
    <code>data[1][i]</code>. It is the same trick Optical Flow uses to return two flow components
    from one kernel — <code>output: [w, h]</code> is indexed <code>[y][x]</code>, so
    <code>[n, 2]</code> is indexed <code>[plane][i]</code>.</p>`;

export default {
  uuid: 'b159433f-d1bb-4ed5-8ae3-ef7a3db50f57',
  version: 1,
  slug: 'autocorrelation',
  title: 'Autocorrelation & Pitch',
  blurb: 'Finding the note in a sound by asking how well it resembles itself, shifted — and the octave error that catches every naive detector once.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'shift-and-compare',
      title: 'Compare a Signal With Itself',
      intro: `<p>"What note is this?" sounds like a question about frequency. Mostly it isn't. Ask
        a different one: <strong>how much does this signal look like itself, shifted?</strong>
        Slide a copy along by <em>lag</em> samples, multiply it against the original point by
        point, and total the products. Land the shift on a whole period and peaks meet peaks —
        a big positive score. Land it half a period out and peaks meet troughs — a big negative
        one. The periodicity falls straight out as a row of peaks.</p>
        <p>That is <strong>autocorrelation</strong>, and it is not a new idea if you have done
        Template Matching: it is that module's cross-correlation with the template replaced by
        the signal itself. It also parallelises without an argument — <strong>one thread per
        lag</strong>, every thread reading the same array and writing only its own cell. No
        atomics, no scatter, nothing to coordinate.</p>
        <p>Below is <code>signal</code>: ${N} samples of a plucked string recorded at
        ${SR}&nbsp;Hz — 62.5 milliseconds, a fundamental plus three harmonics, decaying. Its
        period is in there. Go and find it.</p>`,
      goal: `<strong>Goal:</strong> score <code>signal</code> against a shifted copy of itself at
        every lag from 0 to ${MAX_LAG - 1}, then find the lag of the biggest peak and log the
        period.`,
      requirements: [
        `<code>output: [${MAX_LAG}]</code> — one thread per lag, and <code>this.thread.x</code> <em>is</em> the lag`,
        'Loop over all <code>this.constants.n</code> samples, adding <code>signal[i] * signal[i + lag]</code> only while <code>i + lag</code> is still inside the signal',
        `Scan the returned array from lag <code>${MIN_LAG}</code> upward for the largest value — not from 0`,
        "Log it: <code>console.log('period:', bestLag, 'samples')</code>",
      ],
      hints: [
        {
          title: 'Hint 1 — which lag is mine?',
          body: `<p>With <code>output: [${MAX_LAG}]</code> there are ${MAX_LAG} threads, and thread
            <code>this.thread.x</code> owns exactly one lag. Read it into a
            <code>const lag</code> on the first line and the rest of the kernel reads like the
            formula.</p>`,
        },
        {
          title: 'Hint 2 — the loop, and its guard',
          body: `<p>A kernel loop needs a bound known at compile time, so loop over the
            <em>whole</em> signal and skip the samples whose partner has fallen off the end:</p>
<pre><code>for (let i = 0; i &lt; this.constants.n; i++) {
  if (i + lag &lt; this.constants.n) {
    sum += signal[i] * signal[i + lag];
  }
}</code></pre>
<p>Long lags therefore add up fewer terms than short ones. Remember that — task 3 sends
            the bill.</p>`,
        },
        {
          title: 'Hint 3 — finding the peak in JavaScript',
          body: `<p>${MAX_LAG} numbers is nothing; finish in plain JS.</p>
<pre><code>let bestLag = ${MIN_LAG};
for (let lag = ${MIN_LAG}; lag &lt; ${MAX_LAG}; lag++) {
  if (r[lag] &gt; r[bestLag]) bestLag = lag;
}</code></pre>
<p>Start at ${MIN_LAG}, not 0. Lag 0 is the signal against an unshifted copy of itself:
            it wins every time and means nothing.</p>`,
        },
      ],
      transfer: `Correlation-by-lag is a stock primitive everywhere — NPP and cuFFT on CUDA,
        vDSP's <code>conv</code>/<code>corr</code> on Apple silicon, a WGSL compute shader indexed
        by <code>global_invocation_id</code> in WebGPU. The guarded inner loop is the same edge
        handling every stencil kernel needs, and the read pattern (all threads sweeping the same
        array in step) is the one caches and texture units are built to serve.`,
      starterCode: `// One thread per lag. Each one slides a copy of the signal over itself.
const gpu = new GPU({ mode });

const correlate = gpu.createKernel(function (signal) {
  const lag = this.thread.x;
  let sum = 0;
  // TODO: walk the whole signal, and whenever i + lag is still inside it,
  // add signal[i] * signal[i + lag] to sum.
  return sum;
}, {
  output: [${MAX_LAG}],
  constants: { n: ${N} },
});

const r = correlate(signal);
console.log('r[0], the signal against itself:', r[0]);

// TODO: scan lags ${MIN_LAG}…${MAX_LAG - 1} for the biggest value, then
// console.log('period:', bestLag, 'samples');
`,
      solutionCode: `// One thread per lag. Each one slides a copy of the signal over itself.
const gpu = new GPU({ mode });

const correlate = gpu.createKernel(function (signal) {
  const lag = this.thread.x;
  let sum = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i + lag < this.constants.n) {
      sum += signal[i] * signal[i + lag];
    }
  }
  return sum;
}, {
  output: [${MAX_LAG}],
  constants: { n: ${N} },
});

const r = correlate(signal);
console.log('r[0], the signal against itself:', r[0]);

// Lag 0 always wins and always lies, so the search starts past it.
let bestLag = ${MIN_LAG};
for (let lag = ${MIN_LAG}; lag < ${MAX_LAG}; lag++) {
  if (r[lag] > r[bestLag]) bestLag = lag;
}
console.log('period:', bestLag, 'samples');
`,
      inputs: () => ({ signal: makeTone() }),
      publicTests: [
        {
          name: `${MAX_LAG} lags out, and lag 0 is the signal's own energy`,
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const x = makeTone();
            const out = ctx.kernel(x);
            ctx.assert(
              out && out.length === MAX_LAG,
              `expected ${MAX_LAG} correlation values, got ${out && out.length}`
            );
            const energy = rawAuto(x, 1)[0];
            ctx.assertClose(out[0], energy, 0.02,
              'r[0] should be Σ signal[i]² — the signal multiplied by an unshifted copy of itself');
          },
        },
        {
          name: 'cell <code>lag</code> holds <code>Σ signal[i] · signal[i + lag]</code>',
          run: async ctx => {
            const x = makeTone();
            const out = ctx.kernel(x);
            const ref = rawAuto(x, MAX_LAG);
            const head = headEnergy(x, MAX_LAG);
            const tail = tailEnergy(x, MAX_LAG);
            const lags = [1, 8, 16, 32, 64, 128, 255];
            const hint = diagnoseCurve(lags, l => out[l], l => ref[l], 0.02, [
              [l => head[l],
                'that is the signal multiplied by itself, unshifted — the second factor never got its + lag. It is signal[i] * signal[i + lag]'],
              [l => tail[l],
                'that is the energy of the shifted copy on its own — both factors got the + lag. Only the second one should: signal[i] * signal[i + lag]'],
            ]);
            for (const lag of lags) {
              ctx.assertClose(out[lag], ref[lag], 0.02, hint || `lag ${lag}`);
            }
          },
        },
        {
          name: 'the period is found and logged',
          run: async ctx => {
            const x = makeTone();
            const ref = rawAuto(x, MAX_LAG);
            const expected = argmaxFrom(ref, MIN_LAG, MAX_LAG - 1);
            const out = ctx.kernel(x);
            // Only diagnose the SEARCH once the kernel itself is right —
            // otherwise an unfinished kernel logs 0 and gets told about lag 0.
            const kernelOk = Math.abs(out[expected] - ref[expected]) <= 0.02;
            const nums = loggedNumbers(ctx.logs, 'period');
            const hint = kernelOk
              ? reportedHint(nums, expected, 0.5, [
                [0, 'lag 0 is the signal against an unshifted copy of itself. It scores highest every time, for every signal, and tells you nothing — start the search past it'],
                [1, 'the correlation falls away from lag 0 as a broad shoulder, so a search that starts at lag 1 just walks down that shoulder and stops. Start at lag ' + MIN_LAG],
              ])
              : null;
            ctx.assert(
              nums.some(v => Math.abs(v - expected) <= 0.5),
              hint || `log the period — expected "period: ${expected} samples" in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // An octave lower: same kernel, period 64 instead of 32.
            const x = makeToneLow();
            const out = ctx.kernel(x);
            const ref = rawAuto(x, MAX_LAG);
            const head = headEnergy(x, MAX_LAG);
            const tail = tailEnergy(x, MAX_LAG);
            const lags = [];
            for (let lag = 0; lag < MAX_LAG; lag++) lags.push(lag);
            const hint = diagnoseCurve([1, 16, 32, 64, 128], l => out[l], l => ref[l], 0.02, [
              [l => head[l], 'that is the signal multiplied by itself, unshifted — the second factor never got its + lag'],
              [l => tail[l], 'that is the energy of the shifted copy alone — both factors got the + lag'],
            ]);
            for (const lag of lags) {
              ctx.assertClose(out[lag], ref[lag], 0.02, hint || `lag ${lag}`);
            }
            ctx.assert(
              argmaxFrom(Array.from(out), MIN_LAG, MAX_LAG - 1) === 64,
              'the biggest peak past lag 8 should sit at lag 64 for this signal'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'lag-to-pitch',
      title: 'From Lag to Pitch',
      intro: `<p>A lag is a count of samples; a pitch is a count of hertz. One division bridges
        them, and it is easy to write upside down. The peak lag <em>is</em> the period, so the
        period in seconds is <code>lag / sampleRate</code> and the frequency is its reciprocal:
        <strong><code>sampleRate / lag</code></strong>. Put those the wrong way round and the
        answer is out by four or five orders of magnitude, which at least fails loudly.</p>
        <p>Now the real problem. Lags are integers. At ${SR}&nbsp;Hz, lag 18 means 455.1&nbsp;Hz,
        lag 19 means 431.2&nbsp;Hz and lag 20 means 409.6&nbsp;Hz — steps of more than
        20&nbsp;Hz, which up here is nearly a semitone. The signal below is concert A, 440&nbsp;Hz,
        whose true period is 18.618 samples. The best integer lag is 19, and a tuner built on it
        would report 431&nbsp;Hz: <strong>35 cents flat</strong>, audibly wrong, and no amount of
        GPU makes the grid any finer.</p>
        <p>Three numbers fix it. Near its top the correlation curve is smooth, so fit a parabola
        through the peak sample and its two neighbours and take the apex — the peak <em>between</em>
        the samples. This is the one part of the module that belongs on the CPU: it is three
        numbers and a divide, done once.</p>`,
      goal: `<strong>Goal:</strong> find the peak lag, refine it to a fraction of a sample with
        parabolic interpolation, convert it to hertz, and log both.`,
      requirements: [
        `Scan from lag <code>${MIN_LAG}</code> for the peak lag <code>p</code> (the kernel is already written for you)`,
        'Interpolate: <code>δ = ½(r[p−1] − r[p+1]) / (r[p−1] − 2·r[p] + r[p+1])</code>, and the refined lag is <code>p + δ</code>',
        "Log the refined lag: <code>console.log('refined lag:', p + delta)</code>",
        "Log the pitch: <code>console.log('pitch:', sampleRate / (p + delta), 'Hz')</code>",
      ],
      hints: [
        {
          title: 'Hint 1 — the parabola, in one line',
          body: `<pre><code>const a = r[p - 1];
const b = r[p];
const c = r[p + 1];
const delta = (0.5 * (a - c)) / (a - 2 * b + c);</code></pre>
<p><code>delta</code> always lands between −0.5 and +0.5 — it is a nudge, not a jump.</p>`,
        },
        {
          title: 'Hint 2 — which way does it move?',
          body: `<p>Here the true peak is to the <em>left</em> of the sampled one, so
            <code>delta</code> comes out <strong>negative</strong> and the refined lag is smaller
            than <code>p</code>. If your refinement pushes the answer further from 440&nbsp;Hz than
            the raw integer lag did, the two terms in the numerator are the wrong way round: it is
            <code>r[p - 1] - r[p + 1]</code>, left minus right.</p>`,
        },
        {
          title: 'Hint 3 — "but I want to hum into my laptop"',
          body: `<p>So would we. In a browser page the wiring is short:</p>
<pre><code>const mic = await navigator.mediaDevices
  .getUserMedia({ audio: true });
const audio = new AudioContext({ sampleRate: ${SR} });
const analyser = audio.createAnalyser();
analyser.fftSize = ${N};
audio.createMediaStreamSource(mic).connect(analyser);

const frame = new Float32Array(analyser.fftSize);
analyser.getFloatTimeDomainData(frame);
const r = correlate(frame);   // the same kernel, real audio</code></pre>
<p>None of that runs here, and not for a policy reason: your code executes inside a Web
            Worker, and a Worker has no <code>AudioContext</code>, no
            <code>OfflineAudioContext</code> and no <code>navigator.mediaDevices</code>. So the
            signals in this module are synthesised in the course's own source, which has the
            consolation that every expected answer is exact.</p>`,
        },
      ],
      transfer: `Sub-sample peak refinement is everywhere in GPU work under other names: it is
        the same quadratic fit stereo matchers use to get sub-pixel disparity, that keypoint
        detectors use to place a corner between pixels, and that time-of-flight sensors use
        between range bins. The pattern is identical — the grid is coarse, the underlying function
        is smooth, so fit and solve rather than sample harder.`,
      starterCode: `// The correlation kernel is done. The arithmetic after it is the lesson.
const gpu = new GPU({ mode });

const correlate = gpu.createKernel(function (signal) {
  const lag = this.thread.x;
  let sum = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i + lag < this.constants.n) {
      sum += signal[i] * signal[i + lag];
    }
  }
  return sum;
}, {
  output: [${MAX_LAG}],
  constants: { n: ${N} },
});

const r = correlate(signal);

let p = ${MIN_LAG};
for (let lag = ${MIN_LAG}; lag < ${MAX_LAG - 1}; lag++) {
  if (r[lag] > r[p]) p = lag;
}
console.log('peak lag:', p, '->', sampleRate / p, 'Hz (integer lags only)');

// TODO: fit a parabola through r[p - 1], r[p], r[p + 1] and take its apex.
//   const delta = ...
// Then log the refined lag and the pitch:
//   console.log('refined lag:', p + delta);
//   console.log('pitch:', sampleRate / (p + delta), 'Hz');
`,
      solutionCode: `// The correlation kernel is done. The arithmetic after it is the lesson.
const gpu = new GPU({ mode });

const correlate = gpu.createKernel(function (signal) {
  const lag = this.thread.x;
  let sum = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i + lag < this.constants.n) {
      sum += signal[i] * signal[i + lag];
    }
  }
  return sum;
}, {
  output: [${MAX_LAG}],
  constants: { n: ${N} },
});

const r = correlate(signal);

let p = ${MIN_LAG};
for (let lag = ${MIN_LAG}; lag < ${MAX_LAG - 1}; lag++) {
  if (r[lag] > r[p]) p = lag;
}
console.log('peak lag:', p, '->', sampleRate / p, 'Hz (integer lags only)');

// The apex of the parabola through the peak and its two neighbours.
const a = r[p - 1];
const b = r[p];
const c = r[p + 1];
const delta = (0.5 * (a - c)) / (a - 2 * b + c);

console.log('refined lag:', p + delta);
console.log('pitch:', sampleRate / (p + delta), 'Hz');
`,
      inputs: () => ({ signal: makeNoteA(), sampleRate: SR }),
      publicTests: [
        {
          name: 'the correlation kernel still answers correctly',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const x = makeNoteA();
            const out = ctx.kernel(x);
            ctx.assert(
              out && out.length === MAX_LAG,
              `expected ${MAX_LAG} correlation values, got ${out && out.length}`
            );
            const ref = rawAuto(x, MAX_LAG);
            for (const lag of [0, 18, 19, 20, 38]) {
              ctx.assertClose(out[lag], ref[lag], 0.02, `lag ${lag}`);
            }
          },
        },
        {
          name: 'the refined lag lands between the samples',
          run: async ctx => {
            const ref = rawAuto(makeNoteA(), MAX_LAG);
            const p = argmaxFrom(ref, MIN_LAG, MAX_LAG - 2);
            const delta = parabolicDelta(ref, p);
            const nums = loggedNumbers(ctx.logs, 'refined lag');
            const hint = reportedHint(nums, p + delta, 0.02, [
              [p, 'that is the integer peak lag itself — the parabolic step never moved it', 0.02],
              [p - delta,
                'the refinement moved the peak the wrong way. The numerator is r[p - 1] - r[p + 1], left minus right; flipped, it pushes the answer further from the truth than doing nothing at all',
                0.02],
            ]);
            ctx.assert(
              nums.some(v => Math.abs(v - (p + delta)) <= 0.02),
              hint || `log the refined lag — expected ≈${(p + delta).toFixed(3)} in the console output`
            );
          },
        },
        {
          name: 'the pitch is <code>sampleRate / refinedLag</code>, and it is 440 Hz',
          run: async ctx => {
            const ref = rawAuto(makeNoteA(), MAX_LAG);
            const p = argmaxFrom(ref, MIN_LAG, MAX_LAG - 2);
            const expected = SR / (p + parabolicDelta(ref, p));
            const nums = loggedNumbers(ctx.logs, 'pitch');
            const hint = reportedHint(nums, expected, 0.5, [
              [SR / p, `that is ${SR} / ${p} — the pitch of the raw integer lag, ${(SR / p - expected).toFixed(1)} Hz out. The refinement is what closes that gap`, 0.5],
              [SR / (p - parabolicDelta(ref, p)),
                'the refinement moved the peak the wrong way — it is r[p - 1] - r[p + 1] on top, left minus right', 0.5],
              [p / SR, 'that is lag / sampleRate: the period in seconds, not the pitch. Turn it over — sampleRate / lag', 1e-5],
              [(p + parabolicDelta(ref, p)) / SR, 'that is lag / sampleRate: the period in seconds, not the pitch. Turn it over — sampleRate / lag', 1e-5],
            ]);
            ctx.assert(
              nums.some(v => Math.abs(v - expected) <= 0.5),
              hint || `log the pitch — expected ≈${expected.toFixed(1)} Hz in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different note through the same kernel, and a tighter cut on the
            // reported pitch than the raw integer lag could ever satisfy.
            const other = makeNoteE();
            const out = ctx.kernel(other);
            const ref = rawAuto(other, MAX_LAG);
            for (let lag = 0; lag < MAX_LAG; lag++) {
              ctx.assertClose(out[lag], ref[lag], 0.02, `lag ${lag} of a 330 Hz note`);
            }
            const a = rawAuto(makeNoteA(), MAX_LAG);
            const p = argmaxFrom(a, MIN_LAG, MAX_LAG - 2);
            const expected = SR / (p + parabolicDelta(a, p));
            const nums = loggedNumbers(ctx.logs, 'pitch');
            ctx.assert(
              nums.some(v => Math.abs(v - expected) <= 0.2),
              `the logged pitch should be ≈${expected.toFixed(2)} Hz — within a fifth of a hertz of concert A`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'zero-lag-and-octaves',
      title: 'Three Ways to Get It Wrong',
      intro: `<p>Task 1 worked. It worked on a clean synthetic tone, with a search range chosen
        to dodge two traps you were never told about, and it would fall over on the signal below.
        Here is the full charge sheet.</p>
        <p><strong>One: lag 0 always wins.</strong> Every signal correlates perfectly with an
        unshifted copy of itself, so <code>r[0]</code> is the maximum of every autocorrelation
        ever computed. Worse, it is not a spike but a <em>shoulder</em> — nearby lags are nearly
        as good, because a smooth signal barely changes in one sample. That is why the search
        starts at ${MIN_LAG}.</p>
        <p><strong>Two: long lags are handicapped.</strong> The overlap shrinks as the lag grows —
        lag 40 sums 472 products, lag 200 only 312 — so raw sums fall away with lag whether or not
        the signal does. You cannot compare two lags until you divide that out. The honest fix is
        the one Template Matching already made for a different reason:
        <strong>normalise</strong>. Divide by the geometric mean of the two overlapping slices'
        energies and every lag comes back on the same −1…+1 scale, with lag 0 exactly 1.</p>
        <p><strong>Three: and now the octave error.</strong> Normalise honestly and a new door
        opens. A signal that repeats every ${40} samples also repeats every 80, and 120, and 160 —
        every multiple is a real peak, and they all score close to 1. The voice below has one
        common quirk: every other period is slightly quieter (creaky voice does this, and so do
        plenty of instruments). The waveform still repeats every 40 samples and you would still
        hear 204.8&nbsp;Hz — but a copy shifted 80 samples matches <em>fractionally better</em>,
        so the tallest peak is at 80 and the naive answer is an octave too low. The fix is not a
        better maximum. It is to stop taking the maximum: walk up from the shortest lag and take
        the <strong>first</strong> peak that clears a threshold.</p>`,
      goal: `<strong>Goal:</strong> return the <em>normalised</em> score at every lag, then report
        the first peak that reaches ${THRESHOLD} — not the biggest one.`,
      requirements: [
        'One guarded loop, three accumulators: the dot product, the energy of <code>signal[i]</code>, and the energy of <code>signal[i + lag]</code>',
        'Return <code>dot / Math.sqrt(headEnergy * tailEnergy)</code>, so lag 0 comes back as exactly 1',
        `Walk lags from ${MIN_LAG} upward and take the FIRST local peak whose score is at least <code>${THRESHOLD}</code>`,
        "Log both: <code>console.log('period:', lag, 'samples')</code> and <code>console.log('pitch:', sampleRate / lag, 'Hz')</code>",
      ],
      hints: [
        {
          title: 'Hint 1 — three accumulators, one pass',
          body: `<p>Read each sample once and use it three times — the same fusion Reductions
            makes for mean and RMS:</p>
<pre><code>const a = signal[i];
const b = signal[i + lag];
dot += a * b;
head += a * a;
tail += b * b;</code></pre>`,
        },
        {
          title: 'Hint 2 — what counts as a peak',
          body: `<p>A local peak is a value at least as large as both its neighbours, and here it
            also has to clear the threshold:</p>
<pre><code>for (let lag = ${MIN_LAG}; lag &lt; ${MAX_LAG - 1}; lag++) {
  const peak = rho[lag] &gt; rho[lag - 1]
    &amp;&amp; rho[lag] &gt;= rho[lag + 1];
  if (peak &amp;&amp; rho[lag] &gt;= ${THRESHOLD}) {
    period = lag;
    break;
  }
}</code></pre>
<p>The <code>break</code> is the whole idea. Drop it and you are back to taking the
            largest.</p>`,
        },
        {
          title: 'Hint 3 — how to tell you got it wrong',
          body: `<p>If your answer is 80 samples / 102.4&nbsp;Hz, the code is finding the tallest
            peak rather than the first one. If it is 40 samples / 204.8&nbsp;Hz, you have it — and
            you have also written a pitch detector that beats the obvious one.</p>`,
        },
      ],
      transfer: `Normalising a similarity score before comparing it is a rule that outlives GPUs:
        it is why <code>cv::matchTemplate</code> ships <code>TM_CCOEFF_NORMED</code> alongside the
        raw version, why cosine similarity beats a dot product for embeddings, and why every
        production pitch tracker (YIN, and the "cumulative mean normalised difference" at its
        heart) is a normalisation trick bolted onto exactly the kernel you just wrote.`,
      starterCode: `// Same shape as task 1, three accumulators instead of one.
const gpu = new GPU({ mode });

const correlate = gpu.createKernel(function (signal) {
  const lag = this.thread.x;
  let dot = 0;
  let head = 0;
  let tail = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i + lag < this.constants.n) {
      dot += signal[i] * signal[i + lag];
      // TODO: accumulate the two energies as well —
      // head gets signal[i] squared, tail gets signal[i + lag] squared.
    }
  }
  // TODO: return the normalised score, dot / sqrt(head * tail)
  return dot;
}, {
  output: [${MAX_LAG}],
  constants: { n: ${N} },
});

const rho = correlate(signal);
console.log('rho[0] (should be exactly 1):', rho[0]);

// TODO: walk from lag ${MIN_LAG} and take the FIRST local peak that reaches ${THRESHOLD}.
// Then log:
//   console.log('period:', period, 'samples');
//   console.log('pitch:', sampleRate / period, 'Hz');
`,
      solutionCode: `// Same shape as task 1, three accumulators instead of one.
const gpu = new GPU({ mode });

const correlate = gpu.createKernel(function (signal) {
  const lag = this.thread.x;
  let dot = 0;
  let head = 0;
  let tail = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i + lag < this.constants.n) {
      const a = signal[i];
      const b = signal[i + lag];
      dot += a * b;
      head += a * a;
      tail += b * b;
    }
  }
  return dot / Math.sqrt(head * tail);
}, {
  output: [${MAX_LAG}],
  constants: { n: ${N} },
});

const rho = correlate(signal);
console.log('rho[0] (should be exactly 1):', rho[0]);

// The FIRST peak over the threshold — not the tallest. Every whole number of
// periods is a real peak, so the tallest one is an octave lottery.
let period = -1;
for (let lag = ${MIN_LAG}; lag < ${MAX_LAG - 1}; lag++) {
  if (rho[lag] >= ${THRESHOLD} && rho[lag] > rho[lag - 1] && rho[lag] >= rho[lag + 1]) {
    period = lag;
    break;
  }
}

console.log('period:', period, 'samples');
console.log('pitch:', sampleRate / period, 'Hz');
`,
      inputs: () => ({ signal: makeVoiced(), sampleRate: SR }),
      publicTests: [
        {
          name: 'the score is normalised — <code>rho[0]</code> is exactly 1',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const x = makeVoiced();
            const out = ctx.kernel(x);
            ctx.assert(
              out && out.length === MAX_LAG,
              `expected ${MAX_LAG} values, got ${out && out.length}`
            );
            const raw = rawAuto(x, 1)[0];
            const hint = diagnose(out[0], 1, 2e-3, [
              [raw, 'that is the raw sum of products — the division by √(head · tail) is missing, and without it no two lags can be compared'],
            ]);
            ctx.assertClose(out[0], 1, 2e-3, hint || 'rho[0]');
          },
        },
        {
          name: 'each lag is divided by <em>its own</em> two overlapping slices',
          run: async ctx => {
            const x = makeVoiced();
            const out = ctx.kernel(x);
            const ref = nccAuto(x, MAX_LAG);
            const raw = rawAuto(x, MAX_LAG);
            const half = halfNormalised(x, MAX_LAG);
            const lags = [20, 40, 80, 120, 160, 200];
            const hint = diagnoseCurve(lags, l => out[l], l => ref[l], 2e-3, [
              [l => raw[l] / raw[0],
                'every lag was divided by the WHOLE window\'s energy instead of its own overlapping slices. That is the shrinking overlap again: it hands short lags an advantage they have not earned'],
              [l => half[l],
                'the same energy is doing both jobs — √(head · tail) needs one energy from signal[i] and a second from signal[i + lag], not one of them twice'],
            ]);
            for (const lag of lags) {
              ctx.assertClose(out[lag], ref[lag], 2e-3, hint || `lag ${lag}`);
            }
          },
        },
        {
          name: 'the FIRST peak over the threshold is reported, not the tallest',
          run: async ctx => {
            const x = makeVoiced();
            const ref = nccAuto(x, MAX_LAG);
            const expected = firstPeakOver(ref, MIN_LAG, MAX_LAG - 2, THRESHOLD);
            const out = ctx.kernel(x);
            const kernelOk = Math.abs(out[expected] - ref[expected]) <= 2e-3;
            const octave =
              'that is twice the period — the octave error. Every whole number of periods is a ' +
              'real peak, and on this voice the even multiples all score 1.000, so "the tallest ' +
              'peak" is a coin toss between 80, 160 and 240. Take the FIRST peak over the ' +
              'threshold instead';
            const nums = loggedNumbers(ctx.logs, 'period');
            const hint = kernelOk
              ? reportedHint(nums, expected, 0.5, [
                [2 * expected, octave],
                [4 * expected, octave],
                [6 * expected, octave],
                [0, 'lag 0 scores exactly 1 by construction now — it is the one peak that can never mean anything'],
              ])
              : null;
            ctx.assert(
              nums.some(v => Math.abs(v - expected) <= 0.5),
              hint || `log the period — expected "period: ${expected} samples" in the console output`
            );
            const pitch = loggedNumbers(ctx.logs, 'pitch');
            ctx.assert(
              pitch.some(v => Math.abs(v - SR / expected) <= 0.5),
              `log the pitch too — expected ≈${(SR / expected).toFixed(1)} Hz in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The whole curve on a different voice — period 50, trap at 100.
            const x = makeVoicedLow();
            const out = ctx.kernel(x);
            const ref = nccAuto(x, MAX_LAG);
            const raw = rawAuto(x, MAX_LAG);
            const half = halfNormalised(x, MAX_LAG);
            const hint = diagnoseCurve([25, 50, 100, 150, 200], l => out[l], l => ref[l], 2e-3, [
              [l => raw[l] / raw[0], 'every lag was divided by the whole window\'s energy, not its own overlapping slices'],
              [l => half[l], 'one energy is doing both jobs — √(head · tail) needs one from each slice'],
            ]);
            for (let lag = 0; lag < MAX_LAG; lag++) {
              ctx.assertClose(out[lag], ref[lag], 2e-3, hint || `lag ${lag}`);
            }
            ctx.assert(
              firstPeakOver(Array.from(out), MIN_LAG, MAX_LAG - 2, THRESHOLD) === 50,
              'the first peak over the threshold should be lag 50 for this voice'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'missing-fundamental',
      title: 'The Note That Isn\'t There',
      intro: `<p>Play a note through a telephone, or a small speaker, or a church organ's mixture
        stop, and something strange happens: the fundamental is gone — the hardware simply cannot
        move that slowly — and you hear it anyway. Your auditory system is not reading the
        loudest peak off a spectrum. It is finding the period the partials <em>share</em>.</p>
        <p><code>signal</code> below is built from harmonics 2, 3 and 4 of 128&nbsp;Hz — that is
        256, 384 and 512&nbsp;Hz — and there is no energy at 128&nbsp;Hz at all. Not
        attenuated: absent, by construction. Take its spectrum and the loudest bin is
        256&nbsp;Hz, an octave above the note. Take its autocorrelation and the first peak is at
        lag 64, which is 128&nbsp;Hz, because 256, 384 and 512 all come back into step every 64
        samples whether or not anything is oscillating at that rate.</p>
        <p>So you need a spectrum to accuse. Write one — a direct transform, one thread per bin,
        which is the O(n²) way and perfectly fine at ${N} samples. <em>The DFT, Honestly</em> has
        a great deal more to say about what a transform is <em>for</em> and what it costs; here it
        is only the witness for the prosecution.</p>
        ${COMPLEX_NOTE}`,
      goal: `<strong>Goal:</strong> write the two-plane DFT kernel, find the loudest bin and its
        frequency, and log it beside the answer the (already written) autocorrelation gives.`,
      requirements: [
        `<code>output: [${N}, 2]</code> — one thread per (plane, bin); <code>this.thread.x</code> is the bin, <code>this.thread.y</code> is the plane`,
        'Each thread sums over all <code>this.constants.n</code> samples with angle <code>−2π · bin · i / n</code>: plane 0 accumulates <code>signal[i] · cos(angle)</code>, plane 1 <code>signal[i] · sin(angle)</code>',
        `In JavaScript, take <code>Math.hypot(re, im)</code> per bin and find the loudest one below bin ${N / 2}; its frequency is <code>bin * sampleRate / ${N}</code>`,
        "Log both verdicts: <code>console.log('loudest partial:', hz, 'Hz')</code> and <code>console.log('autocorrelation:', hz, 'Hz')</code>",
      ],
      hints: [
        {
          title: 'Hint 1 — one thread, one number',
          body: `<p>Each thread owns a single output cell, so it computes a single sum — its own
            plane's, for its own bin:</p>
<pre><code>const bin = this.thread.x;
let re = 0;
let im = 0;
for (let i = 0; i &lt; this.constants.n; i++) {
  const angle = (-2 * Math.PI * bin * i) / this.constants.n;
  re += signal[i] * Math.cos(angle);
  im += signal[i] * Math.sin(angle);
}
if (this.thread.y === 0) return re;
return im;</code></pre>
<p>Both sums get computed by both threads and one of them is thrown away. That is
            wasteful and it is also the shape gpu.js gives you; the FFT in the next task earns the
            waste back a hundred times over.</p>`,
        },
        {
          title: 'Hint 2 — reading two planes back',
          body: `<p>The result is two rows, not two columns:</p>
<pre><code>const spec = spectrum(signal);
const mag = Math.hypot(spec[0][bin], spec[1][bin]);</code></pre>
<p>Plane first, bin second — <code>output: [n, 2]</code> is indexed <code>[y][x]</code>
            like any other 2D output.</p>`,
        },
        {
          title: 'Hint 3 — bins go the other way',
          body: `<p>Watch the two conversions in this task, because they are reciprocals of each
            other and mixing them up is the classic slip. A <em>lag</em> is a period, so
            <code>hz = sampleRate / lag</code>. A <em>bin</em> is already a frequency count, so
            <code>hz = bin * sampleRate / n</code>. With ${SR}&nbsp;Hz over ${N} samples each bin
            is ${SR / N}&nbsp;Hz wide, and the answer you are looking for is bin 16.</p>`,
        },
      ],
      transfer: `Two outputs per thread packed into planes is how every platform returns
        multi-component results without a struct: an RG32F texture in WebGPU, a
        <code>float2</code> buffer in CUDA (which is exactly what cuFFT's
        <code>cufftComplex</code> is), <code>half2</code> pairs in Metal. The layout question —
        interleaved <code>[re, im, re, im…]</code> or planar <code>[re…][im…]</code> — is the same
        one everywhere, and planar is what coalesces.`,
      starterCode: `// One kernel to write. The autocorrelation is yours from task 3.
const gpu = new GPU({ mode });

// Two planes: result[0][bin] is the real part, result[1][bin] the imaginary.
const spectrum = gpu.createKernel(function (signal) {
  const bin = this.thread.x;
  // TODO: sum signal[i] * cos(angle) and signal[i] * sin(angle) over the
  // whole signal, with angle = -2 * PI * bin * i / n, then return the sum
  // belonging to THIS thread's plane (this.thread.y).
  return 0;
}, {
  output: [${N}, 2],
  constants: { n: ${N} },
});

const correlate = gpu.createKernel(function (signal) {
  const lag = this.thread.x;
  let dot = 0;
  let head = 0;
  let tail = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i + lag < this.constants.n) {
      const a = signal[i];
      const b = signal[i + lag];
      dot += a * b;
      head += a * a;
      tail += b * b;
    }
  }
  return dot / Math.sqrt(head * tail);
}, {
  output: [${MAX_LAG}],
  constants: { n: ${N} },
});

const spec = spectrum(signal);
const rho = correlate(signal);

// The autocorrelation's verdict — the first peak over the threshold, as before.
let period = -1;
for (let lag = ${MIN_LAG}; lag < ${MAX_LAG - 1}; lag++) {
  if (rho[lag] >= ${THRESHOLD} && rho[lag] > rho[lag - 1] && rho[lag] >= rho[lag + 1]) {
    period = lag;
    break;
  }
}

// TODO: find the loudest bin below ${N / 2} from spec[0][bin] and spec[1][bin],
// convert it to hertz, and log:
//   console.log('loudest partial:', hz, 'Hz');
//   console.log('autocorrelation:', sampleRate / period, 'Hz');
`,
      solutionCode: `// One kernel to write. The autocorrelation is yours from task 3.
const gpu = new GPU({ mode });

// Two planes: result[0][bin] is the real part, result[1][bin] the imaginary.
const spectrum = gpu.createKernel(function (signal) {
  const bin = this.thread.x;
  let re = 0;
  let im = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const angle = (-2 * Math.PI * bin * i) / this.constants.n;
    re += signal[i] * Math.cos(angle);
    im += signal[i] * Math.sin(angle);
  }
  if (this.thread.y === 0) return re;
  return im;
}, {
  output: [${N}, 2],
  constants: { n: ${N} },
});

const correlate = gpu.createKernel(function (signal) {
  const lag = this.thread.x;
  let dot = 0;
  let head = 0;
  let tail = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i + lag < this.constants.n) {
      const a = signal[i];
      const b = signal[i + lag];
      dot += a * b;
      head += a * a;
      tail += b * b;
    }
  }
  return dot / Math.sqrt(head * tail);
}, {
  output: [${MAX_LAG}],
  constants: { n: ${N} },
});

const spec = spectrum(signal);
const rho = correlate(signal);

// The autocorrelation's verdict — the first peak over the threshold, as before.
let period = -1;
for (let lag = ${MIN_LAG}; lag < ${MAX_LAG - 1}; lag++) {
  if (rho[lag] >= ${THRESHOLD} && rho[lag] > rho[lag - 1] && rho[lag] >= rho[lag + 1]) {
    period = lag;
    break;
  }
}

// The spectrum's verdict — the loudest bin in the lower half.
let loudest = 1;
for (let bin = 1; bin < ${N / 2}; bin++) {
  const mag = Math.hypot(spec[0][bin], spec[1][bin]);
  if (mag > Math.hypot(spec[0][loudest], spec[1][loudest])) loudest = bin;
}

const f0Bin = Math.round(${N} / period);
console.log('loudest partial:', (loudest * sampleRate) / ${N}, 'Hz');
console.log('energy at the fundamental:', Math.hypot(spec[0][f0Bin], spec[1][f0Bin]));
console.log('autocorrelation:', sampleRate / period, 'Hz');
`,
      inputs: () => ({ signal: makeMissing(), sampleRate: SR }),
      publicTests: [
        {
          name: `the spectrum kernel returns two planes of ${N} bins`,
          run: async ctx => {
            const spectrum = findSpectrum(ctx);
            ctx.assert(
              spectrum,
              `no kernel with output [${N}, 2] taking one argument found — the spectrum kernel takes the signal and returns two planes`
            );
            const out = spectrum(makeMissing());
            ctx.assert(out && out.length === 2, `expected 2 planes, got ${out && out.length}`);
            ctx.assert(
              out[0] && out[0].length === N,
              `expected ${N} bins per plane, got ${out[0] && out[0].length}`
            );
          },
        },
        {
          name: 'plane 0 is the real part, plane 1 the imaginary',
          run: async ctx => {
            const spectrum = findSpectrum(ctx);
            ctx.assert(spectrum, `no kernel with output [${N}, 2] found`);
            const x = makeMissing();
            const out = spectrum(x);
            const [re, im] = dftPlanes(x);
            for (const bin of [0, 8, 16, 24, 32, 100]) {
              // The forward transform's sign convention flips the sign of the
              // imaginary part and nothing else, so only its magnitude is
              // asserted — a learner who wrote +2π is not wrong about anything
              // this module goes on to claim.
              const swapped = diagnose(out[0][bin], re[bin], 0.1, [
                [im[bin], 'plane 0 is holding the imaginary part — output [n, 2] is indexed result[plane][bin], and plane 0 (this.thread.y === 0) is the real one'],
                [-im[bin], 'plane 0 is holding the imaginary part — output [n, 2] is indexed result[plane][bin], and plane 0 (this.thread.y === 0) is the real one'],
              ]);
              ctx.assertClose(out[0][bin], re[bin], 0.1, swapped || `the real part of bin ${bin}`);
              ctx.assertClose(
                Math.abs(out[1][bin]), Math.abs(im[bin]), 0.1,
                `the imaginary part of bin ${bin}`
              );
            }
          },
        },
        {
          name: 'the spectrum peaks an octave above the note, and the fundamental is empty',
          run: async ctx => {
            const spectrum = findSpectrum(ctx);
            ctx.assert(spectrum, `no kernel with output [${N}, 2] found`);
            const x = makeMissing();
            const out = spectrum(x);
            const mag = bin => Math.hypot(out[0][bin], out[1][bin]);
            let loudest = 1;
            for (let bin = 1; bin < N / 2; bin++) if (mag(bin) > mag(loudest)) loudest = bin;
            ctx.assert(
              loudest === 16,
              `the loudest bin should be 16 (${(16 * SR) / N} Hz — harmonic 2), got ${loudest}`
            );
            ctx.assert(
              mag(8) < 0.01 * mag(16),
              `bin 8 (${SR / 64} Hz, the fundamental) should be empty — got ${mag(8).toFixed(3)} against ${mag(16).toFixed(1)} at bin 16`
            );
          },
        },
        {
          name: 'both verdicts are logged — 256 Hz from the spectrum, 128 Hz from the lags',
          run: async ctx => {
            const spectrumHz = (16 * SR) / N;
            const pitchHz = SR / 64;
            const loud = loggedNumbers(ctx.logs, 'loudest');
            const loudHint = reportedHint(loud, spectrumHz, 0.5, [
              [16, `16 is the bin NUMBER, not a frequency — multiply by sampleRate / ${N} = ${SR / N} Hz to get ${spectrumHz} Hz`, 0.5],
              [pitchHz, 'the spectrum has no peak at the fundamental — that is the whole point of this signal. Its loudest bin is harmonic 2', 0.5],
            ]);
            ctx.assert(
              loud.some(v => Math.abs(v - spectrumHz) <= 0.5),
              loudHint || `log the loudest partial — expected ≈${spectrumHz} Hz in the console output`
            );
            const auto = loggedNumbers(ctx.logs, 'autocorrelation');
            const autoHint = reportedHint(auto, pitchHz, 0.5, [
              [64, `64 is the peak LAG in samples — the pitch is sampleRate / lag = ${pitchHz} Hz`, 0.5],
              [spectrumHz, 'that is the spectrum\'s answer, an octave high. The autocorrelation\'s first peak sits at lag 64', 0.5],
            ]);
            ctx.assert(
              auto.some(v => Math.abs(v - pitchHz) <= 0.5),
              autoHint || `log the autocorrelation's answer — expected ≈${pitchHz} Hz in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Another missing fundamental, an octave down: partials at 128, 192
            // and 256 Hz, nothing at 64 Hz, period 128 samples.
            const spectrum = findSpectrum(ctx);
            const correlate = findCorrelator(ctx);
            ctx.assert(spectrum && correlate, 'expected a spectrum kernel and a correlation kernel');
            const x = makeMissingLow();
            const out = spectrum(x);
            const [re, im] = dftPlanes(x);
            for (let bin = 0; bin < N; bin += 7) {
              ctx.assertClose(out[0][bin], re[bin], 0.1, `the real part of bin ${bin}`);
              ctx.assertClose(Math.abs(out[1][bin]), Math.abs(im[bin]), 0.1, `the imaginary part of bin ${bin}`);
            }
            const mag = bin => Math.hypot(out[0][bin], out[1][bin]);
            let loudest = 1;
            for (let bin = 1; bin < N / 2; bin++) if (mag(bin) > mag(loudest)) loudest = bin;
            ctx.assert(loudest === 8, `expected bin 8 to be loudest for this signal, got ${loudest}`);
            ctx.assert(mag(4) < 0.01 * mag(8), 'bin 4 (the fundamental) should be empty');
            const rho = Array.from(correlate(x));
            ctx.assert(
              firstPeakOver(rho, MIN_LAG, MAX_LAG - 2, THRESHOLD) === 128,
              'the autocorrelation should still find the period at lag 128, with no energy at that frequency anywhere in the spectrum'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'fft-autocorrelation',
      title: 'Payoff: The Same Curve, via the FFT',
      intro: `<p>The brute-force correlator does <code>${MAX_LAG} × ${N}</code> multiply-adds. Ask
        for every lag of a one-second buffer and it is O(n²) with a large constant. There is a way
        out, and it is one of the prettiest results in the subject: the
        <strong>Wiener–Khinchin theorem</strong>. The autocorrelation of a signal is the inverse
        transform of its <strong>power spectrum</strong>. Transform, square the magnitudes, transform
        back — O(n log n), and the whole correlation comes out at once.</p>
        <p>Two details make it true rather than nearly true. First, the transform believes your
        signal repeats forever, so a shifted copy wraps around from the end onto the beginning; the
        cure is to <strong>zero-pad</strong> to twice the length, which is why ${N} samples go into
        a ${PAD}-point transform. Second, the inverse transform of this library — like most —
        leaves a factor of ${PAD} behind, so divide by it at the end.</p>
        <p>One thing this route is <em>not</em> is fragile. The power spectrum throws phase away,
        so nothing here ever calls <code>Math.atan2</code> — and a bin whose imaginary part is
        near zero has a phase that flips a whole turn on float32 noise. Squaring magnitudes has no
        such seam, which is why a ${PAD}-point round trip through ${Math.log2(PAD) * 2} kernel
        launches still agrees with brute force to five decimal places.</p>
        <p>The FFT itself is written for you below — <em>The FFT Butterfly</em> derives that pass
        properly; here it is a given — and its shape should look familiar even so: a JavaScript
        loop calling one kernel ${Math.log2(PAD)} times with a doubling stride, which is the
        halving ladder from Reductions read backwards. It carries a complex array as two planes,
        exactly as the last task's spectrum did. Your job is the theorem: the power kernel, and
        four lines of wiring.</p>
        <p>Then press the point home with a stopwatch — warm up first, and read the numbers the way
        Measuring Speed Honestly insists on. Be ready for an unflattering answer. At ${N} samples
        the brute force is only ${(MAX_LAG * N).toLocaleString('en-US')} multiply-adds in a single
        launch, while the FFT route costs ${2 * Math.log2(PAD)} launches plus a full download and
        upload between every one of them. The asymptotics are not in doubt; at this size they have
        not started paying yet, and saying so is the honest version of the lesson.</p>
        ${COMPLEX_NOTE}`,
      goal: `<strong>Goal:</strong> write the power-spectrum kernel, wire up the
        transform → power → inverse transform route, show its answer matches brute force, and time
        both.`,
      requirements: [
        `<code>power</code> takes the two-plane spectrum and returns <code>output: [${PAD}, 2]</code>: plane 0 is <code>re² + im²</code>, plane 1 is <code>0</code>`,
        'Wire up <code>transform(padded, −1)</code> → <code>power</code> → <code>transform(…, +1)</code>, and divide the real plane by <code>' + PAD + '</code>',
        "Log the largest disagreement with brute force: <code>console.log('max difference:', d)</code> — it should be under 0.01",
        "Warm up, then time five runs of each and log <code>'brute force:'</code> and <code>'fft route:'</code> in milliseconds",
      ],
      hints: [
        {
          title: 'Hint 1 — the power kernel',
          body: `<p>The power spectrum is real, so plane 1 goes home empty:</p>
<pre><code>const i = this.thread.x;
if (this.thread.y === 0) {
  const re = spec[0][i];
  const im = spec[1][i];
  return re * re + im * im;
}
return 0;</code></pre>
<p>Note <code>re * re + im * im</code> and not <code>Math.sqrt(...)</code>. The theorem wants
            the magnitude <em>squared</em>; a square root here produces a curve that is smooth,
            plausible and wrong.</p>`,
        },
        {
          title: 'Hint 2 — four lines of wiring',
          body: `<pre><code>const spec = transform(padded(), -1);
const p = power(spec);
const back = transform(p, +1);
for (let lag = 0; lag &lt; ${MAX_LAG}; lag++) {
  out[lag] = back[0][lag] / ${PAD};
}</code></pre>
<p>If every value comes out exactly ${PAD} times too big, the last division is the piece
            that is missing.</p>`,
        },
        {
          title: 'Hint 3 — timing it honestly',
          body: `<p>The first call to anything here compiles a shader, so throw it away:</p>
<pre><code>correlate(signal); viaFFT();          // warm up, discard

let t0 = Date.now();
for (let i = 0; i &lt; 5; i++) correlate(signal);
console.log('brute force:', (Date.now() - t0) / 5, 'ms');</code></pre>
<p>The <strong>Benchmark</strong> button will disagree with your own numbers, and it is not
            wrong: it times every kernel separately with a forced readback after each, which for a
            ten-pass ladder measures the ladder taken apart.</p>`,
        },
      ],
      transfer: `Convolution and correlation via the transform is the reason cuFFT, vkFFT and
        Apple's vDSP exist, and the reason every deep-learning framework once shipped an
        FFT-based convolution path. The crossover point is real and worth knowing: below a few
        hundred taps the direct form wins on hardware that likes dense arithmetic, which is
        exactly what you are about to measure.`,
      starterCode: `// Two routes to the same curve. One is O(n²), the other O(n log n).
const gpu = new GPU({ mode });

// ---- route 1: brute force, exactly as in task 1 --------------------------
const correlate = gpu.createKernel(function (signal) {
  const lag = this.thread.x;
  let sum = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i + lag < this.constants.n) {
      sum += signal[i] * signal[i + lag];
    }
  }
  return sum;
}, {
  output: [${MAX_LAG}],
  constants: { n: ${N} },
});

// ---- route 2: one radix-2 butterfly pass, run ${Math.log2(PAD)} times -----
// GIVEN, and the same two-plane layout as before: data[0][i] real,
// data[1][i] imaginary, output [${PAD}, 2] read back as result[plane][i].
const fftPass = gpu.createKernel(function (data, ns, sign) {
  const i = this.thread.x;
  const block = Math.floor(i / (2 * ns));
  const r = i - block * 2 * ns;
  const k = r % ns;
  const t = block * ns + k;
  const partner = t + this.constants.half;
  const angle = (sign * Math.PI * k) / ns;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);
  const ar = data[0][t];
  const ai = data[1][t];
  const br = data[0][partner];
  const bi = data[1][partner];
  const tr = br * wr - bi * wi;
  const ti = br * wi + bi * wr;
  if (r < ns) {
    if (this.thread.y === 0) return ar + tr;
    return ai + ti;
  }
  if (this.thread.y === 0) return ar - tr;
  return ai - ti;
}, {
  output: [${PAD}, 2],
  constants: { half: ${PAD / 2} },
});

// TODO: the one kernel you have to write.
const power = gpu.createKernel(function (spec) {
  // plane 0 → re * re + im * im, plane 1 → 0
  return 0;
}, { output: [${PAD}, 2] });

// The ladder: ${Math.log2(PAD)} passes, the stride doubling each time.
function transform(data, sign) {
  let cur = data;
  for (let ns = 1; ns < ${PAD}; ns *= 2) cur = fftPass(cur, ns, sign);
  return cur;
}

// ${N} samples zero-padded to ${PAD}, plus an all-zero imaginary plane.
// Float32Array from the start: every pass hands the next one Float32Arrays,
// and gpu.js locks an argument's container type on the first call.
function padded() {
  const re = new Float32Array(${PAD});
  const im = new Float32Array(${PAD});
  for (let i = 0; i < signal.length; i++) re[i] = signal[i];
  return [re, im];
}

function viaFFT() {
  const out = new Float32Array(${MAX_LAG});
  // TODO: transform → power → inverse transform → divide by ${PAD}.
  return out;
}

const brute = correlate(signal);
const fast = viaFFT();

let maxDiff = 0;
for (let lag = 0; lag < ${MAX_LAG}; lag++) {
  maxDiff = Math.max(maxDiff, Math.abs(fast[lag] - brute[lag]));
}
console.log('max difference:', maxDiff);

// TODO: warm both routes up, then time five runs of each and log
//   console.log('brute force:', ms, 'ms');
//   console.log('fft route:', ms, 'ms');
`,
      solutionCode: `// Two routes to the same curve. One is O(n²), the other O(n log n).
const gpu = new GPU({ mode });

// ---- route 1: brute force, exactly as in task 1 --------------------------
const correlate = gpu.createKernel(function (signal) {
  const lag = this.thread.x;
  let sum = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i + lag < this.constants.n) {
      sum += signal[i] * signal[i + lag];
    }
  }
  return sum;
}, {
  output: [${MAX_LAG}],
  constants: { n: ${N} },
});

// ---- route 2: one radix-2 butterfly pass, run ${Math.log2(PAD)} times -----
// GIVEN, and the same two-plane layout as before: data[0][i] real,
// data[1][i] imaginary, output [${PAD}, 2] read back as result[plane][i].
const fftPass = gpu.createKernel(function (data, ns, sign) {
  const i = this.thread.x;
  const block = Math.floor(i / (2 * ns));
  const r = i - block * 2 * ns;
  const k = r % ns;
  const t = block * ns + k;
  const partner = t + this.constants.half;
  const angle = (sign * Math.PI * k) / ns;
  const wr = Math.cos(angle);
  const wi = Math.sin(angle);
  const ar = data[0][t];
  const ai = data[1][t];
  const br = data[0][partner];
  const bi = data[1][partner];
  const tr = br * wr - bi * wi;
  const ti = br * wi + bi * wr;
  if (r < ns) {
    if (this.thread.y === 0) return ar + tr;
    return ai + ti;
  }
  if (this.thread.y === 0) return ar - tr;
  return ai - ti;
}, {
  output: [${PAD}, 2],
  constants: { half: ${PAD / 2} },
});

// |X|², not |X|: the theorem is about power, and the imaginary plane is zero.
const power = gpu.createKernel(function (spec) {
  const i = this.thread.x;
  if (this.thread.y === 0) {
    const re = spec[0][i];
    const im = spec[1][i];
    return re * re + im * im;
  }
  return 0;
}, { output: [${PAD}, 2] });

// The ladder: ${Math.log2(PAD)} passes, the stride doubling each time.
function transform(data, sign) {
  let cur = data;
  for (let ns = 1; ns < ${PAD}; ns *= 2) cur = fftPass(cur, ns, sign);
  return cur;
}

// ${N} samples zero-padded to ${PAD}, plus an all-zero imaginary plane.
// Float32Array from the start: every pass hands the next one Float32Arrays,
// and gpu.js locks an argument's container type on the first call.
function padded() {
  const re = new Float32Array(${PAD});
  const im = new Float32Array(${PAD});
  for (let i = 0; i < signal.length; i++) re[i] = signal[i];
  return [re, im];
}

function viaFFT() {
  const spec = transform(padded(), -1);
  const p = power(spec);
  const back = transform(p, +1);
  const out = new Float32Array(${MAX_LAG});
  for (let lag = 0; lag < ${MAX_LAG}; lag++) out[lag] = back[0][lag] / ${PAD};
  return out;
}

const brute = correlate(signal);
const fast = viaFFT();

let maxDiff = 0;
for (let lag = 0; lag < ${MAX_LAG}; lag++) {
  maxDiff = Math.max(maxDiff, Math.abs(fast[lag] - brute[lag]));
}
console.log('max difference:', maxDiff);

// Warm up first — the first call to each route compiles shaders.
correlate(signal);
viaFFT();

let t0 = Date.now();
for (let i = 0; i < 5; i++) correlate(signal);
console.log('brute force:', (Date.now() - t0) / 5, 'ms');

t0 = Date.now();
for (let i = 0; i < 5; i++) viaFFT();
console.log('fft route:', (Date.now() - t0) / 5, 'ms');
`,
      inputs: () => ({ signal: makeTone() }),
      publicTests: [
        {
          name: 'the power kernel squares the magnitudes and empties plane 1',
          run: async ctx => {
            const candidates = kernelsShaped(ctx, PAD, 2, 1);
            ctx.assert(
              candidates.length,
              `no kernel with output [${PAD}, 2] taking one argument found — power(spec) is the kernel to write`
            );
            const re = new Float32Array(PAD);
            const im = new Float32Array(PAD);
            re[0] = 3; im[0] = 4; // |z|² = 25, |z| = 5
            re[1] = -2; im[1] = 1; // |z|² = 5
            let power = null;
            let magOnly = false;
            for (const k of candidates) {
              let out;
              try {
                out = k([re, im]);
              } catch (e) {
                continue;
              }
              if (!out || out.length !== 2) continue;
              if (Math.abs(out[0][0] - 25) <= 1e-3 && Math.abs(out[0][1] - 5) <= 1e-3) {
                power = k;
                ctx.assertClose(out[1][0], 0, 1e-3,
                  'plane 1 of the power spectrum should be 0 — the power spectrum is real, and leaving the old imaginary part there feeds noise into the inverse transform');
                break;
              }
              if (Math.abs(out[0][0] - 5) <= 1e-3) magOnly = true;
            }
            ctx.assert(
              power,
              magOnly
                ? 'that kernel returns the magnitude, not the magnitude squared — Wiener–Khinchin is the inverse transform of |X|², so it is re * re + im * im with no square root'
                : `no kernel turned (3 + 4i) into 25 — plane 0 of power(spec) should be re * re + im * im`
            );
          },
        },
        {
          name: 'the FFT route reproduces the brute-force curve',
          run: async ctx => {
            const x = makeTone();
            const ref = rawAuto(x, MAX_LAG);
            const nums = loggedNumbers(ctx.logs, 'max difference');
            const hint = reportedHint(nums, 0, 0.05, [
              [143806, `every value is ${PAD}× too big — the inverse transform leaves a factor of ${PAD} behind, so the last step is back[0][lag] / ${PAD}`, 500],
              [137, 'that curve came from |X| rather than |X|² — the theorem squares the magnitudes, so no Math.sqrt and no Math.hypot', 5],
            ]);
            ctx.assert(
              nums.length,
              "log the largest disagreement — console.log('max difference:', maxDiff)"
            );
            ctx.assert(
              nums.some(v => Math.abs(v) <= 0.05),
              hint || `the two routes should agree to well under 0.01, on a curve whose peak is ${ref[0].toFixed(1)}`
            );
          },
        },
        {
          name: 'both routes are warmed up and timed',
          run: async ctx => {
            const texts = ctx.logs.filter(l => l.type === 'log' && l.text).map(l => l.text.toLowerCase());
            ctx.assert(
              texts.some(t => t.includes('brute force')),
              "time the brute-force route and log it — console.log('brute force:', ms, 'ms')"
            );
            ctx.assert(
              texts.some(t => t.includes('fft route')),
              "time the FFT route and log it — console.log('fft route:', ms, 'ms')"
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Rebuild the whole route from the learner's own kernels, on a
            // signal their code never saw.
            const passes = kernelsShaped(ctx, PAD, 2, 3);
            const powers = kernelsShaped(ctx, PAD, 2, 1);
            ctx.assert(passes.length, `no butterfly-pass kernel with output [${PAD}, 2] found`);
            ctx.assert(powers.length, `no power kernel with output [${PAD}, 2] found`);
            const pass = passes[0];
            const power = powers[0];
            const x = makeToneLow();
            const spec = runTransform(pass, paddedSignal(x), -1);
            const back = runTransform(pass, power(spec), +1);
            const ref = rawAuto(x, MAX_LAG);
            for (let lag = 0; lag < MAX_LAG; lag++) {
              ctx.assertClose(back[0][lag] / PAD, ref[lag], 0.05,
                `lag ${lag} of the transform route`);
            }
          },
        },
      ],
    },
  ],
};
