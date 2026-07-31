// Module: Sampling & Aliasing — uuid ad14836c-62d4-4243-afd8-401694d13c75
// (short id ad14836c). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId — this module postdates
// the uuid switch.
//
// The opener of the Signal Processing track. Five tasks: synthesise a signal
// one thread per sample → fold a tone above Nyquist and watch it come back as
// a different one → rebuild what fell between the samples with a windowed-sinc
// gather → quantise the amplitude axis and measure the ~6 dB a bit buys →
// throw away three samples in four and predict exactly what breaks.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, loop
// bounds come from this.constants (compile-time known), a real signal is a
// plain [n] array of floats, paired planes use output: [n, 2] indexed
// result[plane][i], and every task passes in CPU mode. Sizes stay ≤ 256
// samples so verification is fast.
//
// ---------------------------------------------------------------------------
// FLOAT MARGINS — read before changing any number in this file.
//
// Tests compute in float64; the GL backend computes in float32, and GLSL's
// sin() loses accuracy as its argument grows. Three deliberate choices keep
// every assertion far from a decision boundary:
//
//   1. sampleRate = 256 (a power of two) so every i / sampleRate, f / sampleRate
//      and 2 / 2^bits is a dyadic rational — exact in both precisions. The
//      alias fold f − round(f / fs) · fs therefore returns exact integers, and
//      Math.round's tie at 128 / 256 = 0.5 resolves the same way in JS and in
//      GLSL (both are floor(x + 0.5)).
//   2. Task 2 samples a 200 Hz tone, whose largest phase argument is
//      2π · 200 · 255 / 256 ≈ 1252 rad. float32 spacing there is ~6e-5, so the
//      two planes are asserted equal to 5e-3, ~250× the expected drift — while
//      the mistake the test is hunting (folding to +56 instead of −56) moves
//      values by up to 1.24, i.e. 250× the tolerance the other way.
//   3. Task 4 quantises a PROVIDED sample array rather than one the kernel
//      synthesises, so the only float32 error is the array's own round trip.
//      The signal's closest approach to a rounding boundary, over every bit
//      depth the tests use, is 3.5e-4 of a code (at 9 bits); the float32 noise
//      there is 1.3e-5 of a code — a 27× margin. Re-check with the phase search
//      in the task's comment before retuning TONES or SCALE.
//
// Task 3's sinc weights only ever see |πx| ≤ 25, where float32 sin is accurate
// to ~1e-7, so its tolerances are set by the physics (Lanczos truncation,
// ~1.1e-3 in the interior) rather than by arithmetic.

// ---- the signals -----------------------------------------------------------

const SAMPLE_RATE = 256; // Hz — a power of two, so i / SAMPLE_RATE is exact
const N = 256; // samples → exactly one second of signal

// [amplitude, frequency in Hz, phase in radians]. 5 and 12 Hz sit far below
// Nyquist; 40 Hz survives this rate but not task 5's decimation by four.
const TONES = [[1.0, 5, 0.6], [0.5, 12, 1.9], [0.25, 40, 2.7]];

// The one formula this whole module turns on: sample i happens at time
// i / sampleRate seconds, and each component contributes A·sin(2πft + φ).
function sampleAt(tones, i, sampleRate) {
  let s = 0;
  for (let c = 0; c < tones.length; c++) {
    s += tones[c][0] * Math.sin(2 * Math.PI * tones[c][1] * (i / sampleRate) + tones[c][2]);
  }
  return s;
}

function synthesise(tones, n, sampleRate) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = sampleAt(tones, i, sampleRate);
  return out;
}

// Fold a frequency into the base band [−fs/2, +fs/2] by subtracting the NEAREST
// whole multiple of the sample rate. Signed on purpose: a negative result is a
// tone that comes back mirrored in time, which is why wagon wheels spin
// backwards, and is the single most-missed detail in the whole idea.
function fold(f, sampleRate) {
  return f - Math.round(f / sampleRate) * sampleRate;
}

// ---- task 2: the frequencies the fold kernel is fed -------------------------

const FOLD_FREQS = [5, 40, 128, 200, 300, 500];
const ALIAS_PHASE = 0.9; // ≠ 0, so the 128 Hz entry is not degenerately zero
const ALIAS_TEST = 200; // the tone the two planes compare; folds to −56 Hz

// ---- task 3: reconstruction ------------------------------------------------

const RECON_TONES = [[0.8, 5, 0.4], [0.45, 9, 2.2]];
const RECON_FS = 64;
const RECON_N = 64;
const UP = 4; // output points per input sample
const RECON_M = RECON_N * UP;
const TAPS = 8; // Lanczos a — kernel half-width, in input samples
const WIDTH = 2 * TAPS + 1;

function reconSamples() {
  return synthesise(RECON_TONES, RECON_N, RECON_FS);
}

// The continuous signal the samples were taken from — the thing reconstruction
// is trying to get back.
function reconContinuous(t) {
  let s = 0;
  for (let c = 0; c < RECON_TONES.length; c++) {
    s += RECON_TONES[c][0] * Math.sin(2 * Math.PI * RECON_TONES[c][1] * t + RECON_TONES[c][2]);
  }
  return s;
}

// Lanczos: sinc(x) tapered by sinc(x / a) so the sum can stop after a few taps
// without ringing. sinc(0) = 1 by continuity — and x is exactly 0 whenever the
// output point lands on an input sample, since j / UP and k are both dyadic.
function lanczos(x) {
  if (x <= -TAPS || x >= TAPS) return 0;
  if (x === 0) return 1;
  const px = Math.PI * x;
  return (Math.sin(px) / px) * (Math.sin(px / TAPS) / (px / TAPS));
}

// A gather: output point j pulls WIDTH neighbouring samples and weights them.
function gatherWith(weight, samples, j) {
  const p = j / UP;
  const centre = Math.floor(p);
  let sum = 0;
  for (let d = 0; d < WIDTH; d++) {
    const k = centre - TAPS + d;
    if (k >= 0 && k < RECON_N) sum += samples[k] * weight(p - k);
  }
  return sum;
}

function reconstruct(samples, j) {
  return gatherWith(lanczos, samples, j);
}

// The three near-misses task 3's probes look for.
function truncatedSinc(x) {
  if (x <= -TAPS || x >= TAPS) return 0;
  if (x === 0) return 1;
  const px = Math.PI * x;
  return Math.sin(px) / px;
}

function piFreeSinc(x) {
  if (x <= -TAPS || x >= TAPS) return 0;
  if (x === 0) return 1;
  return (Math.sin(x) / x) * (Math.sin(x / TAPS) / (x / TAPS));
}

function linearInterp(samples, j) {
  const p = j / UP;
  const c = Math.floor(p);
  const next = c + 1 < RECON_N ? samples[c + 1] : samples[c];
  return samples[c] + (next - samples[c]) * (p - c);
}

function nearestSample(samples, j) {
  const k = Math.min(RECON_N - 1, Math.round(j / UP));
  return samples[k];
}

// ---- task 4: quantisation --------------------------------------------------

const QUANT_SCALE = 0.5; // peaks at 0.849 — clear of the ±1 clipping edge
const QUANT_DEPTHS = [4, 6, 8, 10, 12];

// Six decimal places, so what the learner sees in the Task inputs panel is
// exactly what the kernel quantises. See the FLOAT MARGINS note: the phase
// search that justified TONES/QUANT_SCALE measures min |0.5 − |v/step − round(v/step)||
// over every depth in QUANT_DEPTHS and wants it ≫ the float32 noise 6e-8 · peak / step.
function quantSignal() {
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    out[i] = Math.round(QUANT_SCALE * sampleAt(TONES, i, SAMPLE_RATE) * 1e6) / 1e6;
  }
  return out;
}

function quantStep(bits) {
  return 2 / Math.pow(2, bits);
}

function quantise(value, bits) {
  const step = quantStep(bits);
  return Math.round(value / step) * step;
}

// 10·log10(signal power / error power). The textbook rule is ≈6.02 dB per bit;
// this signal delivers a mean slope of 5.99 dB over 4…12 bits.
function snrDb(signal, bits, round) {
  const step = quantStep(bits);
  let sigPower = 0;
  let errPower = 0;
  for (let i = 0; i < signal.length; i++) {
    const q = round(signal[i] / step) * step;
    sigPower += signal[i] * signal[i];
    errPower += (q - signal[i]) * (q - signal[i]);
  }
  return 10 * Math.log10(sigPower / errPower);
}

// ---- task 5: decimation ----------------------------------------------------

const FACTOR = 4;
const DECIM_FS = SAMPLE_RATE / FACTOR; // 64 Hz — Nyquist drops to 32 Hz
const DECIM_N = N / FACTOR; // 64 samples

// What the decimated signal IS, exactly: the same three components, each folded
// into the new base band. 40 Hz becomes −24 Hz; 5 and 12 Hz survive untouched.
function decimatedPrediction(tones, i) {
  let s = 0;
  for (let c = 0; c < tones.length; c++) {
    s += tones[c][0] *
      Math.sin(2 * Math.PI * fold(tones[c][1], DECIM_FS) * (i / DECIM_FS) + tones[c][2]);
  }
  return s;
}

// ---- near-miss diagnosis ---------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so an index where two candidates coincide (any
// signal at i = 0, where dividing by the sample rate changes nothing) stays
// silent, as do observations matching probes that disagree with each other. A
// wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Where a candidate is a whole SIGNAL rather than one number, one matching
// sample is weak evidence — every sinusoid agrees with every other one
// somewhere. This form demands that a probe predict EVERY element (and differ
// from the right answer somewhere) before it may speak. Probe values are
// functions of the index; a missing element makes the comparison NaN, which
// fails.
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

// ---- task 1 probes ---------------------------------------------------------
//
// Dropping the "/ sampleRate" makes the argument 2π·f·i — a whole number of
// turns at every integer i when f is an integer — so every sample collapses onto
// the same value, Σ A·sin(φ). Constancy AND that value have to hold before the
// probe speaks, because "return 0" is constant too and means something else
// entirely. The tolerance is loose: on the GL backend sin() of an argument as
// large as 2π·40·255 has lost most of its accuracy, so the flat line is only
// approximately flat there. When it is not flat enough the probe simply stays
// quiet and the plain numeric message does the work.
function noDivideValue(tones) {
  let s = 0;
  for (let c = 0; c < tones.length; c++) s += tones[c][0] * Math.sin(tones[c][2]);
  return s;
}

function signalProbes(out, tones, n, sampleRate) {
  const flat = noDivideValue(tones);
  const flatHit = diagnoseAll(n, i => out[i], i => sampleAt(tones, i, sampleRate), 0.05, [
    [() => flat,
      'every sample came out the same number — that is what happens when the thread index is used as the time. ' +
      'sin(2π · f · i) is a whole number of turns for every integer i, so the signal flatlines on sin(φ). ' +
      'Sample i happens at t = i / this.constants.sampleRate seconds.'],
  ]);
  if (flatHit) return flatHit;
  return diagnoseAll(n, i => out[i], i => sampleAt(tones, i, sampleRate), 1e-3, [
    [i => {
      let s = 0;
      for (let c = 0; c < tones.length; c++) {
        s += tones[c][0] * Math.sin(tones[c][1] * (i / sampleRate) + tones[c][2]);
      }
      return s;
    },
      'the 2π is missing — sin() counts radians, not cycles, so a frequency in Hz has to be multiplied ' +
      'by 2π before it goes in: Math.sin(2 * Math.PI * f * t + phase)'],
    [i => {
      let s = 0;
      for (let c = 0; c < tones.length; c++) {
        s += tones[c][0] * Math.sin(2 * Math.PI * tones[c][1] * (i / sampleRate));
      }
      return s;
    },
      'the phase never made it in — each component is A · sin(2πft + φ), and φ is tones[c][2]'],
    [i => tones[0][0] * Math.sin(2 * Math.PI * tones[0][1] * (i / sampleRate) + tones[0][2]),
      'only the first component is there — the loop has to run over all this.constants.parts of them and add them up'],
  ]);
}

// ---- task 2 probes ---------------------------------------------------------

function foldProbes(out, freqs, sampleRate) {
  return diagnoseAll(freqs.length, i => out[i], i => fold(freqs[i], sampleRate), 1e-3, [
    [i => Math.abs(fold(freqs[i], sampleRate)),
      'the folds came back as magnitudes — the base band runs from −fs/2 to +fs/2 and the sign matters: ' +
      'a negative alias is a tone mirrored in time, which is exactly why a filmed wagon wheel can turn backwards'],
    [i => freqs[i] - Math.floor(freqs[i] / sampleRate) * sampleRate,
      'that is f mod fs, which lands in 0…fs and can still sit above Nyquist — fold to the NEAREST whole ' +
      'multiple of the sample rate: f − Math.round(f / fs) * fs'],
    [i => sampleRate - freqs[i],
      'the subtraction is the wrong way round — it is f − fs, not fs − f (which goes negative as soon as f exceeds the sample rate)'],
    [i => freqs[i],
      'the frequencies came back untouched — nothing was folded'],
  ]);
}

// The alias plane. Note what is NOT probed here: folding by the wrong whole
// multiple of fs (Math.floor instead of Math.round, or no fold at all) produces
// the SAME samples, because that is precisely the claim the task is making.
// Only the sign of the fold changes the numbers, so only the sign is probed.
function aliasRowProbes(out, alias, phase, sampleRate, n) {
  const ref = (freq, i) => Math.sin(2 * Math.PI * freq * (i / sampleRate) + phase);
  return diagnoseAll(n, i => out[i], i => ref(alias, i), 5e-3, [
    [i => ref(Math.abs(alias), i),
      `that plane is +${Math.abs(alias)} Hz, not ${alias} Hz — the fold is signed, and a sine at a negative ` +
      'frequency is its own mirror image, so the two are NOT the same set of samples'],
    [() => Math.sin(phase),
      'every sample in that plane came out the same number — the thread index was used as the time. ' +
      't = this.thread.x / this.constants.sampleRate'],
  ]);
}

// ---- task 3 probes ---------------------------------------------------------

function reconProbes(out, samples, count) {
  return diagnoseAll(count, j => out[j], j => reconstruct(samples, j), 2e-3, [
    [j => linearInterp(samples, j),
      'that is straight-line interpolation between neighbouring samples — it is the obvious guess and it is ' +
      'wrong by about 0.026 RMS here, fifty times the windowed-sinc sum\'s error. The samples do not lie on ' +
      'the chords between themselves'],
    [j => nearestSample(samples, j),
      'that is a zero-order hold — each output point copied its nearest sample, which is a staircase, not a signal'],
    [j => gatherWith(truncatedSinc, samples, j),
      'the sinc was chopped off rather than tapered — multiply by the window sinc(x / a) as well, or the ' +
      'truncation rings (that is the whole point of the word "windowed")'],
    [j => gatherWith(piFreeSinc, samples, j),
      'the π is missing from the sinc — it is sin(πx) / (πx), so that the weight is exactly 0 at every ' +
      'non-zero whole x and the curve passes through every sample'],
  ]);
}

// ---- task 4 probes ---------------------------------------------------------

function quantProbes(out, signal, bits, count) {
  const step = quantStep(bits);
  return diagnoseAll(count, i => out[i], i => quantise(signal[i], bits), step / 100, [
    [i => Math.floor(signal[i] / step) * step,
      'Math.floor where Math.round belongs — rounding down always errs the same way, so the error stops ' +
      'being centred on zero and picks up a −step/2 bias. That costs you a whole bit of SNR'],
    [i => Math.round(signal[i] / step),
      'the code index came back instead of a value — multiply by the step to get back onto the signal\'s scale'],
    [i => Math.round(signal[i] * Math.pow(2, bits)) / Math.pow(2, bits),
      'that step is half the size it should be — 2^bits codes have to cover the whole −1…+1 range, so the ' +
      'step is 2 / 2^bits'],
    [i => signal[i],
      'the samples came back unchanged — nothing was quantised'],
  ]);
}

// ---- task 5 probes ---------------------------------------------------------

function decimateProbes(out, signal, count) {
  return diagnoseAll(count, i => out[i], i => signal[i * FACTOR], 1e-3, [
    [i => signal[i],
      'that is the first 64 samples, not every fourth one — decimating by 4 means thread i owns ' +
      'signal[i * this.constants.factor]'],
    [i => signal[Math.min(N - 1, i * FACTOR + 1)],
      'the stride is right but it starts one sample late — thread 0 keeps signal[0]'],
  ]);
}

function predictionProbes(out, tones, count) {
  return diagnoseAll(count, i => out[i], i => decimatedPrediction(tones, i), 1e-3, [
    [i => {
      let s = 0;
      for (let c = 0; c < tones.length; c++) {
        s += tones[c][0] *
          Math.sin(2 * Math.PI * Math.abs(fold(tones[c][1], DECIM_FS)) * (i / DECIM_FS) + tones[c][2]);
      }
      return s;
    },
      'the fold lost its sign — 40 Hz lands on −24 Hz here, not +24 Hz, and the two are different signals'],
    [i => {
      let s = 0;
      for (let c = 0; c < tones.length; c++) {
        s += tones[c][0] *
          Math.sin(2 * Math.PI * fold(tones[c][1], DECIM_FS) * (i / SAMPLE_RATE) + tones[c][2]);
      }
      return s;
    },
      'the folded frequencies are right but the clock is not — after decimating, sample i happens at ' +
      'i / 64 seconds, not i / 256'],
    [i => {
      let s = 0;
      for (let c = 0; c < tones.length; c++) {
        s += tones[c][0] * Math.sin(2 * Math.PI * fold(tones[c][1], DECIM_FS) * (i / DECIM_FS));
      }
      return s;
    },
      'the phases were dropped — each component is still A · sin(2πft + φ), only f has changed'],
    // Deliberately NOT probed: synthesising at the UNFOLDED frequencies. That
    // produces the identical samples (40 Hz and −24 Hz differ by exactly one
    // sample rate), which is the whole claim of the task, so diagnoseAll would
    // filter such a probe out anyway — it never differs from the right answer.
  ]);
}

// ---- kernel identification -------------------------------------------------
//
// Tasks with more than one kernel find each by its declared output SHAPE rather
// than by creation order, so a learner who writes them in the other order still
// gets graded on the right thing. `k.kernel` is the built Kernel instance;
// `output` is set from the createKernel settings, before the first invocation.
function kernelWithOutput(ctx, dims) {
  return ctx.kernels.find(k => {
    const out = k.kernel && k.kernel.output;
    return out && out.length === dims.length && dims.every((d, i) => out[i] === d);
  }) || null;
}

export default {
  uuid: 'ad14836c-62d4-4243-afd8-401694d13c75',
  version: 1,
  slug: 'sampling-and-aliasing',
  title: 'Sampling & Aliasing',
  blurb: 'One thread per sample: build a signal, watch a tone come back as the wrong one, and rebuild what fell between.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'one-thread-one-sample',
      title: 'One Thread, One Sample',
      intro: `<p>A signal, to a computer, is a list of numbers: <code>signal[i]</code> is how big the
        thing was at moment <em>i</em>. Nothing else survives. Which moment is that? Sample
        <code>i</code> happens at <strong>t = i / sampleRate</strong> seconds — and forgetting that
        division is the single most common bug in this whole field, because the code still runs and
        the answer is only wrong by a factor of the sample rate.</p>
        <p>Building samples is the friendliest shape a GPU has: every sample is computed from its own
        index and nothing else, so 256 threads do 256 independent sums. No neighbours, no ordering, no
        coordination — the easiest possible parallel problem, and it produces the input every other
        task in this module runs on.</p>
        <p>Your <code>tones</code> input is three components, each an
        <code>[amplitude, frequency in Hz, phase in radians]</code> triple. Add them up at this
        thread's time:</p>
<pre><code>signal[i] = Σ  A · sin(2π · f · t + φ)        with t = i / sampleRate</code></pre>
        <p>Two things that bite. <code>Math.sin</code> counts <em>radians</em>, so a frequency in Hz
        has to be multiplied by 2π before it goes in. And a real signal like this one is just a plain
        <code>[n]</code> array of floats — the paired <code>output: [n, 2]</code> shape this track
        uses for complex signals arrives later, with the DFT.</p>
        <p>One boundary, stated plainly: your code runs in a Web Worker, which has no
        <code>AudioContext</code>, no <code>OfflineAudioContext</code> and no
        <code>navigator.mediaDevices</code>. Nothing here can record or play anything. Outside the
        sandbox the samples would arrive like this, and the kernel would not know the difference:</p>
<pre><code>const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const audio = new AudioContext({ sampleRate: 48000 });
const analyser = audio.createAnalyser();
audio.createMediaStreamSource(stream).connect(analyser);

const buffer = new Float32Array(analyser.fftSize);
analyser.getFloatTimeDomainData(buffer);   // ← 2048 real samples
await kernel(buffer);</code></pre>`,
      goal: `<strong>Goal:</strong> fill 256 samples of a one-second signal — each thread turns its own
        index into a time and sums the three components of <code>tones</code> at that time.`,
      requirements: [
        'Turn the thread index into seconds: <code>this.thread.x / this.constants.sampleRate</code>',
        'Loop over <code>this.constants.parts</code> components and accumulate <code>A * Math.sin(2 * Math.PI * f * t + phase)</code>',
        'Return the sum — <code>output: [256]</code>, one thread per sample',
      ],
      hints: [
        {
          title: 'Hint 1 — reading a component',
          body: `<p><code>tones</code> is a 3×3 nested array. Component <code>c</code> is
            <code>tones[c][0]</code> (amplitude), <code>tones[c][1]</code> (frequency, Hz) and
            <code>tones[c][2]</code> (phase, radians).</p>`,
        },
        {
          title: 'Hint 2 — the time of this sample',
          body: `<p>With <code>sampleRate</code> = 256, thread 0 is at t = 0 s, thread 64 is at
            t = 0.25 s, and thread 255 is at t = 0.996 s:</p>
<pre><code>const t = this.thread.x / this.constants.sampleRate;</code></pre>`,
        },
        {
          title: 'Hint 3 — the loop body',
          body: `<pre><code>s += tones[c][0] * Math.sin(2 * Math.PI * tones[c][1] * t + tones[c][2]);</code></pre>`,
        },
      ],
      transfer: `Per-sample synthesis is the "embarrassingly parallel" case every platform opens with:
        a CUDA kernel with one thread per sample, a WebGPU compute shader whose
        <code>global_invocation_id.x</code> is the sample index, a Metal kernel over a 1D grid. It is
        also what an <code>AudioWorkletProcessor</code> does per render quantum on the CPU — the same
        arithmetic, 128 samples at a time instead of all of them at once.`,
      starterCode: `// 256 samples of a one-second signal. One thread per sample.
const gpu = new GPU({ mode });

const synth = gpu.createKernel(function (tones) {
  // TODO: turn this thread's INDEX into a time in seconds,
  //       then sum the three components of \`tones\` at that time.
  //   const t = this.thread.x / this.constants.sampleRate;
  return 0;
}, {
  output: [256],
  constants: { sampleRate: 256, parts: 3 },
});

const signal = await synth(tones);
console.log('samples:', signal.length);
console.log('first four:', signal[0], signal[1], signal[2], signal[3]);
`,
      solutionCode: `// 256 samples of a one-second signal. One thread per sample.
const gpu = new GPU({ mode });

const synth = gpu.createKernel(function (tones) {
  const t = this.thread.x / this.constants.sampleRate;
  let s = 0;
  for (let c = 0; c < this.constants.parts; c++) {
    s += tones[c][0] * Math.sin(2 * Math.PI * tones[c][1] * t + tones[c][2]);
  }
  return s;
}, {
  output: [256],
  constants: { sampleRate: 256, parts: 3 },
});

const signal = await synth(tones);
console.log('samples:', signal.length);
console.log('first four:', signal[0], signal[1], signal[2], signal[3]);
`,
      inputs: () => ({ tones: TONES.map(t => t.slice()) }),
      publicTests: [
        {
          name: 'the kernel returns 256 samples',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(TONES.map(t => t.slice()));
            ctx.assert(out && out.length === N, `expected ${N} samples, got ${out && out.length}`);
          },
        },
        {
          name: 'sample <code>i</code> is the sum of the components at <code>t = i / 256</code>',
          run: async ctx => {
            const tones = TONES.map(t => t.slice());
            const out = await ctx.kernel(tones);
            const hint = signalProbes(out, tones, N, SAMPLE_RATE);
            for (const i of [0, 1, 7, 64, 128, 200, 255]) {
              ctx.assertClose(out[i], sampleAt(tones, i, SAMPLE_RATE), 2e-3, hint || `sample ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Different amplitudes, frequencies and phases: nothing may be baked in.
            const tones = [[0.7, 3, 0.25], [0.9, 17, 2.4], [0.4, 31, 1.1]];
            const out = await ctx.kernel(tones);
            ctx.assert(out && out.length === N, `expected ${N} samples`);
            const hint = signalProbes(out, tones, N, SAMPLE_RATE);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], sampleAt(tones, i, SAMPLE_RATE), 2e-3, hint || `sample ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'above-nyquist',
      title: 'Above Nyquist',
      intro: `<p>Here is the part that surprises people. Sample a 200 Hz tone at 256 Hz and you do not
        get a bad 200 Hz tone, or a warning, or noise. You get a <strong>perfectly good tone at a
        different frequency</strong> — and there is no way to tell from the samples that anything went
        wrong, because the samples of the two are <em>the same numbers</em>.</p>
        <p>The arithmetic is one line. Sampling at <code>fs</code> adds a whole turn to the phase every
        time <code>f</code> moves by <code>fs</code>, and a whole turn is invisible. So every frequency
        in the family <code>f, f ± fs, f ± 2fs, …</code> produces an identical sample sequence, and the
        one you actually perceive is the member nearest zero — fold <code>f</code> into the
        <strong>base band</strong> −fs/2 … +fs/2:</p>
<pre><code>alias = f - Math.round(f / fs) * fs
//  300 Hz at fs = 256 →  +44 Hz     (the picture above)
//  200 Hz at fs = 256 →  −56 Hz     (what you are about to sample)</code></pre>
        <p>Round to the <em>nearest</em> multiple, and keep the sign. A negative alias is not a mistake:
        a sine at −56 Hz is a sine at 56 Hz mirrored in time, and it is genuinely what you hear and see.
        (It is also why a filmed wagon wheel turns backwards, which task 5 comes back to.) Writing
        <code>fs - f</code> instead of <code>f - fs</code> flips that mirror — and goes negative for the
        wrong reason the moment <code>f</code> exceeds <code>fs</code>.</p>
        <p>fs/2 is <strong>Nyquist</strong>, the highest frequency a rate can carry. Exactly <em>at</em>
        it the rule is degenerate, which is why the safe test is <code>f &gt;= fs / 2</code> and not
        <code>f &gt; fs / 2</code>: a 128 Hz sine sampled at 256 Hz gives you sin(φ), −sin(φ), sin(φ), …
        — two samples per period, forever, so you can never recover both amplitude and phase. And if φ
        happens to be 0 you get 256 zeros and the tone disappears completely. Watch what the fold does
        to the 128 Hz entry in <code>freqs</code>.</p>
        <p>The second kernel writes <strong>two planes</strong>: <code>output: [256, 2]</code> is indexed
        <code>result[plane][i]</code>, the shape this track carries a complex signal in (plane 0 real,
        plane 1 imaginary) once the DFT arrives. Here the planes are simply two ordinary real signals,
        side by side, so you can compare them sample for sample.</p>`,
      goal: `<strong>Goal:</strong> fold six frequencies into the base band with one kernel, then use a
        second kernel to sample <code>200 Hz</code> and its alias into two planes and show they are the
        same numbers.`,
      requirements: [
        'The fold kernel returns <code>f - Math.round(f / fs) * fs</code> — one thread per frequency, sign kept',
        'The sample kernel uses <code>output: [256, 2]</code> and picks its frequency with <code>this.thread.y</code>',
        'Feed the folded 200 Hz alias into the second kernel and <code>console.log</code> it',
      ],
      hints: [
        {
          title: 'Hint 1 — which multiple to subtract',
          body: `<p><code>f / fs</code> says how many whole sample rates fit inside <code>f</code>.
            Rounding it to the nearest integer (not flooring it) is what lands the answer inside
            −fs/2 … +fs/2 rather than 0 … fs.</p>`,
        },
        {
          title: 'Hint 2 — two planes, one kernel',
          body: `<p>With <code>output: [256, 2]</code>, <code>this.thread.y</code> is 0 or 1 and
            <code>this.thread.x</code> is the sample index. Pick the frequency with the plane index and
            everything else is task 1 again:</p>
<pre><code>const f = pair[this.thread.y];
const t = this.thread.x / this.constants.sampleRate;
return Math.sin(2 * Math.PI * f * t + this.constants.phase);</code></pre>`,
        },
      ],
      transfer: `Every sampled system on every platform has this hazard and the same fix: filter above
        Nyquist <em>before</em> you sample, never after. Graphics calls it the same thing —
        a texture minified without mipmaps aliases, and MSAA/anisotropic filtering are anti-aliasing in
        the literal signal-processing sense. GPU texture units implement that filter in hardware
        precisely because doing it afterwards is impossible.`,
      starterCode: `// Two kernels: fold the frequencies, then sample two of them side by side.
const gpu = new GPU({ mode });

// One thread per frequency — pure arithmetic, no data at all.
const fold = gpu.createKernel(function (freqs) {
  const f = freqs[this.thread.x];
  // TODO: subtract the NEAREST whole multiple of the sample rate, keeping the sign.
  return f;
}, {
  output: [6],
  constants: { sampleRate: 256 },
});

// output: [256, 2] → result[plane][i]. Plane 0 samples pair[0], plane 1 pair[1].
const sample = gpu.createKernel(function (pair) {
  // TODO: choose this plane's frequency with this.thread.y
  const f = pair[0];
  const t = this.thread.x / this.constants.sampleRate;
  return Math.sin(2 * Math.PI * f * t + this.constants.phase);
}, {
  output: [256, 2],
  constants: { sampleRate: 256, phase: 0.9 },
});

const folded = await fold(freqs);
console.log('folded:', Array.from(folded));

const alias = folded[3];   // freqs[3] is the 200 Hz tone
console.log('200 Hz sampled at 256 Hz comes back as', alias, 'Hz');

const rows = await sample([200, alias]);
console.log('plane 0:', rows[0][0], rows[0][1], rows[0][2], rows[0][3]);
console.log('plane 1:', rows[1][0], rows[1][1], rows[1][2], rows[1][3]);
`,
      solutionCode: `// Two kernels: fold the frequencies, then sample two of them side by side.
const gpu = new GPU({ mode });

// One thread per frequency — pure arithmetic, no data at all.
const fold = gpu.createKernel(function (freqs) {
  const f = freqs[this.thread.x];
  return f - Math.round(f / this.constants.sampleRate) * this.constants.sampleRate;
}, {
  output: [6],
  constants: { sampleRate: 256 },
});

// output: [256, 2] → result[plane][i]. Plane 0 samples pair[0], plane 1 pair[1].
const sample = gpu.createKernel(function (pair) {
  const f = pair[this.thread.y];
  const t = this.thread.x / this.constants.sampleRate;
  return Math.sin(2 * Math.PI * f * t + this.constants.phase);
}, {
  output: [256, 2],
  constants: { sampleRate: 256, phase: 0.9 },
});

const folded = await fold(freqs);
console.log('folded:', Array.from(folded));

const alias = folded[3];   // freqs[3] is the 200 Hz tone
console.log('200 Hz sampled at 256 Hz comes back as', alias, 'Hz');

const rows = await sample([200, alias]);
console.log('plane 0:', rows[0][0], rows[0][1], rows[0][2], rows[0][3]);
console.log('plane 1:', rows[1][0], rows[1][1], rows[1][2], rows[1][3]);
`,
      inputs: () => ({ freqs: FOLD_FREQS.slice() }),
      publicTests: [
        {
          name: 'the fold lands every frequency in <code>−128 … +128</code>, sign and all',
          run: async ctx => {
            const folder = kernelWithOutput(ctx, [FOLD_FREQS.length]);
            ctx.assert(folder, `no kernel with output: [${FOLD_FREQS.length}] found — that is the fold kernel`);
            const out = await folder(FOLD_FREQS.slice());
            ctx.assert(out && out.length === FOLD_FREQS.length,
              `expected ${FOLD_FREQS.length} folded frequencies, got ${out && out.length}`);
            const hint = foldProbes(out, FOLD_FREQS, SAMPLE_RATE);
            for (let i = 0; i < FOLD_FREQS.length; i++) {
              const expected = fold(FOLD_FREQS[i], SAMPLE_RATE);
              const note = FOLD_FREQS[i] === 128
                ? `${FOLD_FREQS[i]} Hz is exactly Nyquist — f / fs is exactly 0.5, and Math.round takes ` +
                  'a tie upward, so it folds to −128'
                : `${FOLD_FREQS[i]} Hz`;
              ctx.assertClose(out[i], expected, 1e-3, hint || note);
            }
          },
        },
        {
          name: 'both planes hold 256 samples, and they are the same numbers',
          run: async ctx => {
            const sampler = kernelWithOutput(ctx, [N, 2]);
            ctx.assert(sampler, 'no kernel with output: [256, 2] found — the sample kernel needs two planes');
            const alias = fold(ALIAS_TEST, SAMPLE_RATE);
            const rows = await sampler([ALIAS_TEST, alias]);
            ctx.assert(rows && rows.length === 2, `expected 2 planes, got ${rows && rows.length}`);
            ctx.assert(rows[0] && rows[0].length === N && rows[1] && rows[1].length === N,
              'each plane should hold 256 samples');
            const hint = aliasRowProbes(rows[1], alias, ALIAS_PHASE, SAMPLE_RATE, N);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(
                rows[1][i],
                Math.sin(2 * Math.PI * alias * (i / SAMPLE_RATE) + ALIAS_PHASE),
                5e-3,
                hint || `plane 1, sample ${i}`
              );
              ctx.assertClose(rows[1][i], rows[0][i], 5e-3,
                `sample ${i} differs between the planes — a ${ALIAS_TEST} Hz tone and a ${alias} Hz tone ` +
                'sampled at 256 Hz have to produce identical numbers');
            }
          },
        },
        {
          name: 'the alias of <code>200 Hz</code> is logged',
          run: async ctx => {
            const alias = fold(ALIAS_TEST, SAMPLE_RATE);
            const nums = loggedNumbers(ctx.logs);
            const hint = !nums.some(v => Math.abs(v - alias) <= 1e-6) &&
              nums.some(v => Math.abs(v - Math.abs(alias)) <= 1e-6)
              ? `the console shows +${Math.abs(alias)} — the fold is signed, and 200 Hz at fs = 256 lands ` +
                `on ${alias}, a ${Math.abs(alias)} Hz tone mirrored in time`
              : `log the folded frequency — expected ${alias} in the console output`;
            ctx.assert(nums.some(v => Math.abs(v - alias) <= 1e-6), hint);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // 255 → −1 and 257 → +1: mirror images either side of the sample
            // rate. 384 sits on a tie (384 / 256 = 1.5) and folds to −128.
            const folder = kernelWithOutput(ctx, [FOLD_FREQS.length]);
            ctx.assert(folder, 'expected a fold kernel with 6 outputs');
            const fresh = [3, 77, 129, 255, 257, 384];
            const out = await folder(fresh);
            const hint = foldProbes(out, fresh, SAMPLE_RATE);
            for (let i = 0; i < fresh.length; i++) {
              ctx.assertClose(out[i], fold(fresh[i], SAMPLE_RATE), 1e-3, hint || `${fresh[i]} Hz`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A different aliasing pair: 300 Hz folds to +44, so this time the
            // alias is positive and both planes must still agree exactly.
            const sampler = kernelWithOutput(ctx, [N, 2]);
            ctx.assert(sampler, 'expected a sample kernel with output: [256, 2]');
            const f = 300;
            const alias = fold(f, SAMPLE_RATE);
            const rows = await sampler([f, alias]);
            const hint = aliasRowProbes(rows[1], alias, ALIAS_PHASE, SAMPLE_RATE, N);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(
                rows[0][i],
                Math.sin(2 * Math.PI * f * (i / SAMPLE_RATE) + ALIAS_PHASE),
                5e-3,
                `plane 0, sample ${i}`
              );
              ctx.assertClose(
                rows[1][i],
                Math.sin(2 * Math.PI * alias * (i / SAMPLE_RATE) + ALIAS_PHASE),
                5e-3,
                hint || `plane 1, sample ${i}`
              );
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'sinc-reconstruction',
      title: 'What Happened Between the Samples?',
      intro: `<p>You have 64 numbers. The signal they came from existed at every instant in between. Can
        you get it back?</p>
        <p>The obvious answer — join the dots with straight lines — is wrong, and not by a little. A
        sinusoid does not lie on the chords between its own samples, so linear interpolation shaves every
        peak; on the signal below it misses by about <strong>0.026 RMS</strong>. The right answer is
        stranger and much better: if nothing in the signal was above Nyquist, the samples determine it
        <em>completely</em>, and the formula that rebuilds it sums a <strong>sinc</strong> centred on
        every sample:</p>
<pre><code>x(t) = Σ  x[k] · sinc(t · fs − k)          sinc(u) = sin(πu) / (πu),  sinc(0) = 1</code></pre>
        <p>sinc is 1 at its own sample and exactly 0 at every other whole offset, so the curve threads
        every sample point and interpolates smoothly between them. In principle the sum runs over all
        <em>k</em>; in practice you stop after a handful of taps and <strong>window</strong> what is
        left, or the abrupt cut rings. The Lanczos window is the classic: multiply by
        <code>sinc(u / a)</code>, with <code>a</code> the half-width in samples.</p>
        <p>On a GPU this is a <strong>gather</strong>, the shape "Thinking in Parallel" makes the case
        for: each output point pulls the 17 samples around it and weights them. Nothing is written
        anywhere but this thread's own cell, so 256 output points cost one pass, no coordination and no
        ordering.</p>
        <p>The ends are honestly ragged — near the edges the window runs off the array and the sum is
        missing terms, so the tests only judge the interior. Real resamplers pad, mirror or taper the
        edges; the middle is where the idea lives.</p>`,
      goal: `<strong>Goal:</strong> reconstruct the signal at 4× the sample rate — 256 output points from
        64 samples — with a 17-tap Lanczos-windowed sinc gather.`,
      requirements: [
        'Output point <code>j</code> sits at sample position <code>p = j / this.constants.up</code>',
        'Gather <code>this.constants.width</code> taps centred on <code>Math.floor(p)</code>, skipping indices outside the array',
        'Weight tap <code>k</code> by <code>sinc(p − k) · sinc((p − k) / a)</code>, and by <code>1</code> when <code>p − k</code> is 0',
        'Return the weighted sum — <code>output: [256]</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — which samples are mine?',
          body: `<p><code>p</code> is generally between two samples. Take
            <code>centre = Math.floor(p)</code> and walk <code>k</code> from
            <code>centre - a</code> to <code>centre + a</code> — that is
            <code>this.constants.width</code> = 17 taps. Guard each one:
            <code>if (k &gt;= 0 &amp;&amp; k &lt; this.constants.n)</code>.</p>`,
        },
        {
          title: 'Hint 2 — the weight, carefully',
          body: `<p><code>x = p - k</code> is exactly 0 when the output point lands on a sample, and
            <code>sin(πx) / (πx)</code> is 0/0 there — special-case it to 1. Outside
            <code>±a</code> the window is 0, so those taps contribute nothing.</p>`,
        },
        {
          title: 'Hint 3 — the whole weight',
          body: `<pre><code>const px = Math.PI * x;
w = (Math.sin(px) / px) * (Math.sin(px / this.constants.a) / (px / this.constants.a));</code></pre>
<p>— the first factor is the sinc, the second is the Lanczos taper that lets you stop after
            17 taps.</p>`,
        },
      ],
      transfer: `Windowed-sinc resampling is what every audio SRC (libsamplerate, SoX, WebAudio's own
        <code>playbackRate</code>) and every image resizer's Lanczos option actually does. On a GPU it
        is the same gather a separable convolution uses — CUDA reads the taps through the texture cache,
        WebGPU binds the sample buffer as read-only storage, and a fragment shader gets a cheap 2-tap
        version for free in <code>textureSample</code>'s bilinear filter, which is precisely the
        straight-line guess this task rejects.`,
      starterCode: `// 64 samples in, 256 reconstructed points out. A gather: pull, never push.
const gpu = new GPU({ mode });

const rebuild = gpu.createKernel(function (samples) {
  const p = this.thread.x / this.constants.up;
  const centre = Math.floor(p);

  // TODO: replace this straight-line guess with a 17-tap windowed-sinc sum.
  const frac = p - centre;
  let next = samples[centre];
  if (centre + 1 < this.constants.n) {
    next = samples[centre + 1];
  }
  return samples[centre] + (next - samples[centre]) * frac;
}, {
  output: [256],
  constants: { up: 4, n: 64, a: 8, width: 17 },
});

const curve = await rebuild(samples);
console.log('reconstructed points:', curve.length);
console.log('on top of sample 10:', curve[40], 'vs', samples[10]);
`,
      solutionCode: `// 64 samples in, 256 reconstructed points out. A gather: pull, never push.
const gpu = new GPU({ mode });

const rebuild = gpu.createKernel(function (samples) {
  const p = this.thread.x / this.constants.up;
  const centre = Math.floor(p);
  let sum = 0;
  for (let d = 0; d < this.constants.width; d++) {
    const k = centre - this.constants.a + d;
    const x = p - k;
    let w = 0;
    if (x === 0) {
      w = 1;
    } else if (Math.abs(x) < this.constants.a) {
      const px = Math.PI * x;
      w = (Math.sin(px) / px) * (Math.sin(px / this.constants.a) / (px / this.constants.a));
    }
    if (k >= 0 && k < this.constants.n) {
      sum += samples[k] * w;
    }
  }
  return sum;
}, {
  output: [256],
  constants: { up: 4, n: 64, a: 8, width: 17 },
});

const curve = await rebuild(samples);
console.log('reconstructed points:', curve.length);
console.log('on top of sample 10:', curve[40], 'vs', samples[10]);
`,
      inputs: () => ({ samples: reconSamples() }),
      publicTests: [
        {
          name: 'the curve passes exactly through every original sample',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const samples = reconSamples();
            const out = await ctx.kernel(samples);
            ctx.assert(out && out.length === RECON_M,
              `expected ${RECON_M} reconstructed points, got ${out && out.length}`);
            for (let k = 0; k < RECON_N; k++) {
              ctx.assertClose(out[k * UP], samples[k], 2e-3,
                `output point ${k * UP} sits exactly on sample ${k}, so it must equal it — sinc is 1 at ` +
                'its own sample and 0 at every other whole offset');
            }
          },
        },
        {
          name: 'each point is the windowed-sinc sum of its 17 neighbours',
          run: async ctx => {
            const samples = reconSamples();
            const out = await ctx.kernel(samples);
            const hint = reconProbes(out, samples, RECON_M);
            for (let j = 0; j < RECON_M; j++) {
              ctx.assertClose(out[j], reconstruct(samples, j), 2e-3, hint || `point ${j}`);
            }
          },
        },
        {
          name: 'the interior matches the signal the samples came from',
          run: async ctx => {
            // The payoff: 64 numbers, and the curve between them is right.
            const samples = reconSamples();
            const out = await ctx.kernel(samples);
            const hint = reconProbes(out, samples, RECON_M);
            for (let j = TAPS * UP; j < RECON_M - TAPS * UP; j++) {
              ctx.assertClose(out[j], reconContinuous(j / (UP * RECON_FS)), 4e-3,
                hint || `point ${j} against the original continuous signal`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Fresh samples — a different pair of tones, still under Nyquist.
            const fresh = synthesise([[0.6, 7, 1.3], [0.5, 11, 0.2]], RECON_N, RECON_FS);
            const out = await ctx.kernel(fresh);
            const hint = reconProbes(out, fresh, RECON_M);
            for (let j = 0; j < RECON_M; j++) {
              ctx.assertClose(out[j], reconstruct(fresh, j), 2e-3, hint || `point ${j}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'quantise',
      title: 'The Other Axis Is Discrete Too',
      intro: `<p>Sampling chops up <em>time</em>. Storing the samples chops up <strong>amplitude</strong>:
        with <code>bits</code> bits there are 2<sup>bits</sup> code values to cover the whole −1…+1
        range, so the step between neighbouring codes is <code>2 / 2^bits</code> and every sample gets
        nudged to the nearest one.</p>
<pre><code>step = 2 / Math.pow(2, bits);
q    = Math.round(value / step) * step;      // NOT Math.floor
error = q - value;                           //  in −step/2 … +step/2</code></pre>
        <p><code>Math.round</code> is load-bearing. Rounding <em>down</em> quantises just as
        successfully, but every error then has the same sign, so instead of jitter centred on zero you
        get a DC offset of −step/2 riding on the signal — and you measurably throw away a whole bit.
        (Run this task with <code>Math.floor</code> if you want to watch it happen: every SNR drops by
        about 6 dB.)</p>
        <p>Which is the rule worth having. Signal-to-noise ratio is
        <code>10·log10(signal power / error power)</code>, and halving the step — one more bit —
        quarters the error power, which is 10·log10(4) ≈ <strong>6 dB</strong>. Do not take that on
        faith: the loop below runs 4, 6, 8, 10 and 12 bits and you can read the slope straight off the
        console.</p>
        <p>The kernel writes two planes again — <code>output: [256, 2]</code>, plane 0 the quantised
        signal and plane 1 the error — so one pass gives you both the result and the thing you want to
        measure. That is fusion, the same trick "Reductions" ends on: compute it where you already have
        the value in hand rather than paying for a second pass.</p>`,
      goal: `<strong>Goal:</strong> quantise <code>signal</code> to <code>bits</code> bits into plane 0
        and the rounding error into plane 1, then log the SNR at each of the five depths.`,
      requirements: [
        'Take <code>bits</code> as a kernel argument and derive <code>step = 2 / Math.pow(2, bits)</code>',
        'Plane 0 (<code>this.thread.y === 0</code>) holds <code>Math.round(value / step) * step</code>',
        'Plane 1 holds the error, <code>quantised - value</code>',
        '<code>console.log</code> the SNR in dB for each of 4, 6, 8, 10 and 12 bits',
      ],
      hints: [
        {
          title: 'Hint 1 — one value, two planes',
          body: `<p>Both planes are computed from the same sample, so read it once and branch at the
            end:</p>
<pre><code>const v = signal[this.thread.x];
const step = 2 / Math.pow(2, bits);
const q = Math.round(v / step) * step;
let out = q - v;
if (this.thread.y === 0) {
  out = q;
}
return out;</code></pre>`,
        },
        {
          title: 'Hint 2 — the SNR',
          body: `<p>Sum the squares of the samples and the squares of plane 1, then</p>
<pre><code>const snr = 10 * Math.log10(sigPower / errPower);</code></pre>
<p>— and print it with the bit depth beside it so the ~6 dB step is readable.</p>`,
        },
      ],
      transfer: `The bits-versus-noise trade is the same everywhere it shows up: 16-bit audio buys about
        98 dB, 8-bit textures about 50, and a GPU's <code>f16</code> half-precision path is this exact
        bargain — half the bandwidth and half the memory for a coarser step. Modern inference hardware
        pushes it much further with int8 and int4 kernels, and the reason quantisation-aware training
        works at all is that the error stays small, zero-mean noise instead of a bias.`,
      starterCode: `// Amplitude is discrete too. Two planes: the quantised signal, and the error.
const gpu = new GPU({ mode });

const quantise = gpu.createKernel(function (signal, bits) {
  const v = signal[this.thread.x];
  // TODO: step = 2 / 2^bits, round v to the nearest step (Math.round, not floor),
  //       return the quantised value in plane 0 and (quantised - v) in plane 1.
  return v;
}, {
  output: [256, 2],
});

for (const bits of [4, 6, 8, 10, 12]) {
  const planes = await quantise(signal, bits);
  let sigPower = 0;
  let errPower = 0;
  for (let i = 0; i < signal.length; i++) {
    sigPower += signal[i] * signal[i];
    errPower += planes[1][i] * planes[1][i];
  }
  const snr = 10 * Math.log10(sigPower / errPower);
  console.log(bits + ' bits: SNR ' + snr.toFixed(2) + ' dB');
}
`,
      solutionCode: `// Amplitude is discrete too. Two planes: the quantised signal, and the error.
const gpu = new GPU({ mode });

const quantise = gpu.createKernel(function (signal, bits) {
  const v = signal[this.thread.x];
  const step = 2 / Math.pow(2, bits);
  const q = Math.round(v / step) * step;
  let out = q - v;
  if (this.thread.y === 0) {
    out = q;
  }
  return out;
}, {
  output: [256, 2],
});

for (const bits of [4, 6, 8, 10, 12]) {
  const planes = await quantise(signal, bits);
  let sigPower = 0;
  let errPower = 0;
  for (let i = 0; i < signal.length; i++) {
    sigPower += signal[i] * signal[i];
    errPower += planes[1][i] * planes[1][i];
  }
  const snr = 10 * Math.log10(sigPower / errPower);
  console.log(bits + ' bits: SNR ' + snr.toFixed(2) + ' dB');
}
`,
      inputs: () => ({ signal: quantSignal() }),
      publicTests: [
        {
          name: 'plane 0 rounds each sample to the nearest of <code>2^bits</code> steps',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const signal = quantSignal();
            for (const bits of [4, 6, 8]) {
              const planes = await ctx.kernel(signal, bits);
              ctx.assert(planes && planes.length === 2, `expected 2 planes, got ${planes && planes.length}`);
              ctx.assert(planes[0] && planes[0].length === N, 'each plane should hold 256 values');
              const hint = quantProbes(planes[0], signal, bits, N);
              for (let i = 0; i < N; i++) {
                ctx.assertClose(planes[0][i], quantise(signal[i], bits), quantStep(bits) / 100,
                  hint || `sample ${i} at ${bits} bits`);
              }
            }
          },
        },
        {
          name: 'plane 1 is the error, <code>quantised − original</code>',
          run: async ctx => {
            const signal = quantSignal();
            const bits = 6;
            const planes = await ctx.kernel(signal, bits);
            const step = quantStep(bits);
            const hint = diagnoseAll(N, i => planes[1][i], i => quantise(signal[i], bits) - signal[i],
              step / 100, [
                [i => Math.floor(signal[i] / step) * step - signal[i],
                  'every error came out negative — that is Math.floor, not Math.round. Rounding down ' +
                  'biases the whole signal down by about half a step instead of leaving the error centred'],
                [i => signal[i] - quantise(signal[i], bits),
                  'the error has the wrong sign — plane 1 is quantised − original, so that a positive error ' +
                  'means the code sits above the sample'],
                [i => Math.abs(quantise(signal[i], bits) - signal[i]),
                  'the error was made positive — its sign is the interesting part, because it is what shows ' +
                  'the rounding is unbiased'],
              ]);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(planes[1][i], quantise(signal[i], bits) - signal[i], step / 100,
                hint || `error at sample ${i}`);
            }
            // Round, not floor: the mean error must sit on zero, not at −step/2.
            let mean = 0;
            for (let i = 0; i < N; i++) mean += planes[1][i];
            mean /= N;
            ctx.assert(Math.abs(mean) < step / 20,
              `the mean error is ${mean.toFixed(5)}, not ≈0 — rounding down (or up) biases every sample the ` +
              `same way by about ${(step / 2).toFixed(5)}; Math.round centres it`);
          },
        },
        {
          name: 'the SNR is logged for all five depths, and climbs ~6 dB per bit',
          run: async ctx => {
            const signal = quantSignal();
            const nums = loggedNumbers(ctx.logs);
            for (const bits of QUANT_DEPTHS) {
              const expected = snrDb(signal, bits, Math.round);
              const floorSnr = snrDb(signal, bits, Math.floor);
              const hint = nums.some(v => Math.abs(v - floorSnr) <= 0.3)
                ? `the ${bits}-bit SNR came out ≈${floorSnr.toFixed(2)} dB instead of ` +
                  `≈${expected.toFixed(2)} dB — that is the ~6 dB Math.floor costs you over Math.round`
                : `log the SNR at ${bits} bits — expected ≈${expected.toFixed(2)} dB in the console output`;
              ctx.assert(nums.some(v => Math.abs(v - expected) <= 0.5), hint);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A depth the driver loop never runs, so nothing can be memoised.
            const signal = quantSignal();
            const bits = 5;
            const planes = await ctx.kernel(signal, bits);
            const step = quantStep(bits);
            const hint = quantProbes(planes[0], signal, bits, N);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(planes[0][i], quantise(signal[i], bits), step / 100,
                hint || `sample ${i} at ${bits} bits`);
              ctx.assertClose(planes[1][i], quantise(signal[i], bits) - signal[i], step / 100,
                `error at sample ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'downsample',
      title: 'Payoff: Keep One Sample in Four',
      intro: `<p>Everything at once. Take the 256-sample second you built in task 1 — components at 5, 12
        and 40 Hz — and throw away three samples in four. What you have left is 64 samples of the same
        second, which is a signal at <strong>64 Hz</strong>. Nyquist has just dropped from 128 Hz to
        32 Hz, and the 40 Hz component is now on the wrong side of it.</p>
        <p>You can say exactly what happens, in advance, with the fold from task 2 — applied to the
        <em>new</em> rate:</p>
<pre><code>5 Hz  → fold(5, 64)  =   5 Hz      survives
12 Hz  → fold(12, 64) =  12 Hz      survives
40 Hz  → fold(40, 64) = −24 Hz      comes back as 24 Hz, mirrored</code></pre>
        <p>Not "roughly", not "some artefacts": the decimated samples are <em>identical</em>, number for
        number, to a signal synthesised from 5, 12 and −24 Hz at 64 Hz. Prove it — plane 0 keeps every
        fourth sample, plane 1 synthesises the prediction, and the two planes agree.</p>
        <p>This is the wagon wheel. A 24 frames-per-second camera pointed at a wheel turning 30 times a
        second folds it to 30 − 24 = 6 rev/s forwards; at 20 rev/s it folds to 20 − 24 = −4 rev/s, and
        the wheel appears to roll backwards. Same arithmetic, and the negative sign is doing real work.</p>
        <p>And this is why every resampler <strong>filters before it decimates</strong>: once the 40 Hz
        component is sitting on top of the 24 Hz band there is no undoing it — the two are the same
        numbers, and no amount of cleverness downstream can separate them. Remove it while it is still
        distinguishable, which is what "Convolution &amp; Filters" is for.</p>`,
      goal: `<strong>Goal:</strong> decimate <code>signal</code> by 4 into plane 0, synthesise the folded
        prediction into plane 1, and show the two are the same signal.`,
      requirements: [
        'Plane 0 keeps every fourth sample: <code>signal[this.thread.x * this.constants.factor]</code>',
        'Plane 1 sums the components of <code>tones</code>, each folded into the new base band at <code>64 Hz</code>',
        'Use the new rate for time as well: <code>t = this.thread.x / 64</code>, not <code>/ 256</code>',
        '<code>console.log</code> the three folded frequencies',
      ],
      hints: [
        {
          title: 'Hint 1 — the new sample rate',
          body: `<p>Keeping one sample in <code>factor</code> divides the rate by the same number:</p>
<pre><code>const newRate = this.constants.sampleRate / this.constants.factor;   // 64</code></pre>
<p>Everything in plane 1 — the fold and the time — uses <code>newRate</code>, never 256.</p>`,
        },
        {
          title: 'Hint 2 — the fold, again',
          body: `<p>Straight from task 2, with the new rate:</p>
<pre><code>const f = tones[c][1];
const a = f - Math.round(f / newRate) * newRate;</code></pre>
<p>Keep the sign. <code>Math.abs</code> here turns −24 Hz into a different signal.</p>`,
        },
        {
          title: 'Hint 3 — branching between the planes',
          body: `<p>Compute the decimated sample, then overwrite it when this thread is in plane 1:</p>
<pre><code>let out = signal[this.thread.x * this.constants.factor];
if (this.thread.y === 1) {
  // … the loop over this.constants.parts …
  out = s;
}
return out;</code></pre>`,
        },
      ],
      transfer: `Decimation with a pre-filter is the bottom half of every mipmap, every audio
        sample-rate converter and every image pyramid: CUDA's NPP and Metal Performance Shaders ship it,
        and a GPU that generates mipmaps without filtering produces exactly the shimmering this task
        predicts. It is also why "Measuring Speed Honestly" matters here — the filter is the expensive
        part, and skipping it is the tempting, wrong optimisation.`,
      starterCode: `// Keep one sample in four. Predict exactly what that does to each component.
const gpu = new GPU({ mode });

const decimate = gpu.createKernel(function (signal, tones) {
  // TODO: plane 0 — keep every this.constants.factor-th sample.
  // TODO: plane 1 — synthesise the same components at the NEW rate,
  //       each frequency folded into the new base band.
  return 0;
}, {
  output: [64, 2],
  constants: { sampleRate: 256, factor: 4, parts: 3 },
});

const planes = await decimate(signal, tones);
console.log('kept:', planes[0].length, 'samples at', 256 / 4, 'Hz');
console.log('first four kept:     ', planes[0][0], planes[0][1], planes[0][2], planes[0][3]);
console.log('first four predicted:', planes[1][0], planes[1][1], planes[1][2], planes[1][3]);

// TODO: log where each component of tones lands at the new rate — fold each
//       frequency into the 64 Hz base band, exactly as in task 2.
`,
      solutionCode: `// Keep one sample in four. Predict exactly what that does to each component.
const gpu = new GPU({ mode });

const decimate = gpu.createKernel(function (signal, tones) {
  const newRate = this.constants.sampleRate / this.constants.factor;
  const t = this.thread.x / newRate;
  let s = 0;
  for (let c = 0; c < this.constants.parts; c++) {
    const f = tones[c][1];
    const a = f - Math.round(f / newRate) * newRate;
    s += tones[c][0] * Math.sin(2 * Math.PI * a * t + tones[c][2]);
  }
  let out = signal[this.thread.x * this.constants.factor];
  if (this.thread.y === 1) {
    out = s;
  }
  return out;
}, {
  output: [64, 2],
  constants: { sampleRate: 256, factor: 4, parts: 3 },
});

const planes = await decimate(signal, tones);
console.log('kept:', planes[0].length, 'samples at', 256 / 4, 'Hz');
console.log('first four kept:     ', planes[0][0], planes[0][1], planes[0][2], planes[0][3]);
console.log('first four predicted:', planes[1][0], planes[1][1], planes[1][2], planes[1][3]);

for (const t of tones) {
  const newRate = 256 / 4;
  console.log(t[1] + ' Hz folds to ' + (t[1] - Math.round(t[1] / newRate) * newRate) + ' Hz');
}
`,
      inputs: () => ({
        signal: synthesise(TONES, N, SAMPLE_RATE),
        tones: TONES.map(t => t.slice()),
      }),
      publicTests: [
        {
          name: 'plane 0 keeps every fourth sample',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const signal = synthesise(TONES, N, SAMPLE_RATE);
            const tones = TONES.map(t => t.slice());
            const planes = await ctx.kernel(signal, tones);
            ctx.assert(planes && planes.length === 2, `expected 2 planes, got ${planes && planes.length}`);
            ctx.assert(planes[0] && planes[0].length === DECIM_N,
              `expected ${DECIM_N} kept samples, got ${planes[0] && planes[0].length}`);
            const hint = decimateProbes(planes[0], signal, DECIM_N);
            for (let i = 0; i < DECIM_N; i++) {
              ctx.assertClose(planes[0][i], signal[i * FACTOR], 1e-3, hint || `kept sample ${i}`);
            }
          },
        },
        {
          name: 'plane 1 is the same components, folded into the <code>64 Hz</code> base band',
          run: async ctx => {
            const signal = synthesise(TONES, N, SAMPLE_RATE);
            const tones = TONES.map(t => t.slice());
            const planes = await ctx.kernel(signal, tones);
            const hint = predictionProbes(planes[1], tones, DECIM_N);
            for (let i = 0; i < DECIM_N; i++) {
              ctx.assertClose(planes[1][i], decimatedPrediction(tones, i), 1e-3,
                hint || `predicted sample ${i}`);
            }
          },
        },
        {
          name: 'the two planes are the same signal, and the folds are logged',
          run: async ctx => {
            const signal = synthesise(TONES, N, SAMPLE_RATE);
            const tones = TONES.map(t => t.slice());
            const planes = await ctx.kernel(signal, tones);
            for (let i = 0; i < DECIM_N; i++) {
              ctx.assertClose(planes[0][i], planes[1][i], 2e-3,
                `sample ${i} differs between the planes — decimating by 4 must give exactly the folded signal`);
            }
            for (const tone of tones) {
              const folded = fold(tone[1], DECIM_FS);
              const wrong = Math.abs(folded);
              const hint = folded !== wrong && logged(ctx.logs, wrong, 1e-6) && !logged(ctx.logs, folded, 1e-6)
                ? `the console shows +${wrong} — after decimating, ${tone[1]} Hz lands on ${folded} Hz, and ` +
                  'the minus sign is the mirroring that makes wagon wheels turn backwards'
                : `log where ${tone[1]} Hz lands at 64 Hz — expected ${folded} in the console output`;
              ctx.assert(logged(ctx.logs, folded, 1e-6), hint);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Fresh components, including two that alias: 45 Hz → −19, 60 Hz → −4.
            const tones = [[0.8, 7, 0.35], [0.6, 45, 2.1], [0.35, 60, 1.4]];
            const signal = synthesise(tones, N, SAMPLE_RATE);
            const planes = await ctx.kernel(signal, tones);
            const keptHint = decimateProbes(planes[0], signal, DECIM_N);
            const predHint = predictionProbes(planes[1], tones, DECIM_N);
            for (let i = 0; i < DECIM_N; i++) {
              ctx.assertClose(planes[0][i], signal[i * FACTOR], 1e-3, keptHint || `kept sample ${i}`);
              ctx.assertClose(planes[1][i], decimatedPrediction(tones, i), 1e-3,
                predHint || `predicted sample ${i}`);
            }
          },
        },
      ],
    },
  ],
};
