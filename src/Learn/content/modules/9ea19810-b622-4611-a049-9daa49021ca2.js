// Module: Monte Carlo Methods — uuid 9ea19810-b622-4611-a049-9daa49021ca2 (short id 9ea19810).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. Legacy id (pre-uuid URLs, localStorage migration): 2-4.
//
// Module 2.4 — Monte Carlo Methods.
//
// Four tasks: the inside-the-circle dart test (one sample per thread) → a
// parallel reduction that turns hits into π → mean-value integration of
// e^(−x²), a function with no elementary antiderivative → pricing a European
// call option from simulated market paths.
//
// Determinism rule: kernels never call Math.random. Every random sample is
// precomputed in JavaScript with utils.seededRandom and handed to the kernel
// as an array argument, so every run — and every test — sees identical data.
// Kernel-authoring rules per the contract: no closures, arguments + literals
// + this.thread.* only, statically bounded loops. Every task passes on CPU.

// gpu.js quirk: on the GL backend a built kernel silently reuses its stale
// argument texture when re-invoked with an argument whose CONSTRUCTOR differs
// from the build-time one (plain Array vs Float32Array). Tests therefore
// always re-invoke kernels with the same constructor the reference solution
// used: plain Arrays for injected inputs, Float32Array for kernel-to-kernel
// hand-offs (kernel outputs are Float32Arrays on both backends).

// n uniforms in [0, 1) from a fixed seed.
function makeUniforms(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = rand();
  return out;
}

// n (x, y) dart positions in the unit square, as two parallel arrays.
function makePairs(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const xs = new Array(n);
  const ys = new Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = rand();
    ys[i] = rand();
  }
  return { xs, ys };
}

// Float64 reference: how many darts land inside the quarter circle.
function countInside(xs, ys) {
  let hits = 0;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] * xs[i] + ys[i] * ys[i] <= 1) hits++;
  }
  return hits;
}

// Float64 reference: sum of e^(−x²) over the samples.
function gaussSum(xs) {
  let sum = 0;
  for (let i = 0; i < xs.length; i++) sum += Math.exp(-xs[i] * xs[i]);
  return sum;
}

// n standard-normal draws via Box–Muller over seeded uniforms.
function makeNormals(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const out = new Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = 1 - rand(); // (0, 1] — keeps Math.log finite
    const u2 = rand();
    const radius = Math.sqrt(-2 * Math.log(u1));
    out[i] = radius * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) out[i + 1] = radius * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

// Option contract shared by task 4's inputs, starter/solution and tests.
const OPT = { s0: 100, strike: 105, rate: 0.03, sigma: 0.2, t: 1 };
const OPT_DRIFT = (OPT.rate - (OPT.sigma * OPT.sigma) / 2) * OPT.t;
const OPT_VOLT = OPT.sigma * Math.sqrt(OPT.t);

// Abramowitz–Stegun normal CDF (|error| < 7.5e-8).
function normalCdf(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp((-x * x) / 2);
  const p =
    d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}

// Black–Scholes closed form for the European call — the "right answer".
function blackScholesCall() {
  const { s0, strike, rate, sigma, t } = OPT;
  const d1 = (Math.log(s0 / strike) + (rate + (sigma * sigma) / 2) * t) / (sigma * Math.sqrt(t));
  const d2 = d1 - sigma * Math.sqrt(t);
  return s0 * normalCdf(d1) - strike * Math.exp(-rate * t) * normalCdf(d2);
}

// Float64 reference: discounted mean payoff over the given normals.
function referencePrice(normals) {
  let sum = 0;
  for (let i = 0; i < normals.length; i++) {
    const st = OPT.s0 * Math.exp(OPT_DRIFT + OPT_VOLT * normals[i]);
    sum += Math.max(st - OPT.strike, 0);
  }
  return Math.exp(-OPT.rate * OPT.t) * (sum / normals.length);
}

// Float64 reference for the un-floored payoff — the price a learner gets when
// Math.max is missing, which the task's own hint puts near −1.9.
function referencePriceUnfloored(normals) {
  let sum = 0;
  for (let i = 0; i < normals.length; i++) {
    const st = OPT.s0 * Math.exp(OPT_DRIFT + OPT_VOLT * normals[i]);
    sum += st - OPT.strike;
  }
  return Math.exp(-OPT.rate * OPT.t) * (sum / normals.length);
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so samples where two candidates coincide stay
// silent, as do observations that match probes disagreeing with each other.
// A wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 1: verdicts are 0 or 1, so any single dart is a coin flip's worth of
// evidence — these patterns are matched across ALL the planted darts at once,
// which is what tells "x + y ≤ 1" apart from "inverted".
function verdictHint(out, xs, ys, expected) {
  const matchesAll = predict => expected.every((_, i) => Math.abs(out[i] - predict(i)) <= 1e-6);
  if (matchesAll(i => (xs[i] + ys[i] <= 1 ? 1 : 0))) {
    return 'those verdicts are x + y ≤ 1 — the inside test compares squared distance: x * x + y * y <= 1';
  }
  if (matchesAll(i => 1 - expected[i])) {
    return 'the verdicts are inverted — return 1 when the dart lands inside, 0 when it misses';
  }
  return null;
}

// Tasks 2 and 3: a thread's slice, summed as-is or through a function.
function sliceSum(arr, start, len, f) {
  let s = 0;
  for (let i = 0; i < len; i++) {
    const v = arr[start + i];
    s += f ? f(v) : v;
  }
  return s;
}

// Task 2: the slice offset, and not summing a slice at all.
function sliceProbes(arr, t, len) {
  const probes = [
    [arr[t], 'that is a single verdict — this thread has to total all 256 in its own slice'],
  ];
  if (t + len <= arr.length) {
    probes.push([sliceSum(arr, t, len),
      'the slice starts at this.thread.x * 256 — with this.thread.x alone every thread walks an overlapping window']);
  }
  return probes;
}

// Task 3: accumulating the wrong thing inside the fused loop.
function integrandProbes(arr, start, len) {
  return [
    [sliceSum(arr, start, len),
      'that is the sum of the samples themselves — the accumulator wants Math.exp(-x * x)'],
    [sliceSum(arr, start, len, v => Math.exp(-v)),
      'that is e^(−x), not e^(−x²) — square x inside the exponent'],
    [sliceSum(arr, start, len, v => Math.exp(v * v)),
      'the exponent is missing its minus sign — e^(−x²) falls off as x grows'],
  ];
}

// Task 4: the payoff floor, dropped or pointed the wrong way.
function payoffProbes(st, strike) {
  return [
    [st - strike,
      'that is st − strike with no floor — a losing path pays exactly 0, never a negative amount'],
    [Math.max(strike - st, 0),
      'that is the put payoff — a call pays max(st − strike, 0)'],
  ];
}

export default {
  uuid: '9ea19810-b622-4611-a049-9daa49021ca2',
  version: 1,
  slug: 'monte-carlo-methods',
  legacyId: '2-4',
  title: 'Monte Carlo Methods',
  blurb: 'Estimate π, price an option, integrate the un-integrable — with a million random samples.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'darts-at-a-circle',
      title: 'Darts at a Quarter Circle',
      intro: `<p>Monte Carlo is statistics as a weapon: throw random darts at a square, and the
        <em>fraction</em> that lands inside the quarter circle inscribed in it approaches its area —
        π/4. No geometry beyond the Pythagorean check <code>x² + y² ≤ 1</code>.</p>
        <p>The method is embarrassingly parallel: every dart is judged independently, so every dart
        gets its own thread. One rule, though — the randomness is made <strong>outside</strong> the
        kernel. <code>xs</code> and <code>ys</code> hold 4,096 seeded dart positions; the kernel's
        job is only the verdict. Deterministic data in, deterministic verdicts out — that's what
        makes GPU Monte Carlo debuggable.</p>`,
      goal: `<strong>Goal:</strong> make each thread return <code>1</code> if its dart
        <code>(xs[x], ys[x])</code> lands inside the unit quarter circle, else <code>0</code>.`,
      requirements: [
        'Read this thread\'s dart: <code>xs[this.thread.x]</code> and <code>ys[this.thread.x]</code>',
        'Inside means <code>x² + y² ≤ 1</code> — no <code>Math.sqrt</code> needed',
        'Return exactly <code>1</code> or <code>0</code>, nothing in between',
      ],
      hints: [
        {
          title: 'Hint 1 — skip the square root',
          body: `<p>The dart is inside when its distance to the origin is ≤ 1 — and distances
            compare the same way squared: <code>x * x + y * y &lt;= 1</code> is the whole test.</p>`,
        },
        {
          title: 'Hint 2 — the verdict',
          body: `<pre><code>if (x * x + y * y &lt;= 1) {
  return 1;
}
return 0;</code></pre>
<p>— a branch is
            fine in a kernel as long as every path returns.</p>`,
        },
      ],
      transfer: `Real GPU Monte Carlo keeps the random numbers on-device — CUDA ships cuRAND, and
        WebGPU/Metal compute shaders run counter-based generators like Philox per thread — but the
        shape is exactly this: one thread, one sample, one verdict.`,
      starterCode: `// 4,096 seeded darts. One thread judges one dart.
const gpu = new GPU({ mode });

const inside = gpu.createKernel(function (xs, ys) {
  const x = xs[this.thread.x];
  const y = ys[this.thread.x];
  // TODO: return 1 if this dart lands inside the unit quarter
  // circle (x² + y² ≤ 1), otherwise 0.
  return 0;
}, { output: [4096] });

const hits = inside(xs, ys);

let count = 0;
for (let i = 0; i < hits.length; i++) count += hits[i];
console.log(count, 'of 4096 darts hit — π ≈', (4 * count) / 4096);
`,
      solutionCode: `// 4,096 seeded darts. One thread judges one dart.
const gpu = new GPU({ mode });

const inside = gpu.createKernel(function (xs, ys) {
  const x = xs[this.thread.x];
  const y = ys[this.thread.x];
  if (x * x + y * y <= 1) {
    return 1;
  }
  return 0;
}, { output: [4096] });

const hits = inside(xs, ys);

let count = 0;
for (let i = 0; i < hits.length; i++) count += hits[i];
console.log(count, 'of 4096 darts hit — π ≈', (4 * count) / 4096);
`,
      inputs: utils => makePairs(utils, 4096, 9001),
      publicTests: [
        {
          name: 'clearly-inside darts return 1, clearly-outside darts return 0',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const xs = new Array(4096).fill(0.5);
            const ys = new Array(4096).fill(0.5);
            // planted verdicts, all far from the boundary
            xs[0] = 0.1; ys[0] = 0.1;   // inside
            xs[1] = 0.9; ys[1] = 0.9;   // outside
            xs[2] = 0.0; ys[2] = 0.0;   // bullseye
            xs[3] = 0.99; ys[3] = 0.3;  // outside
            xs[4] = 0.6; ys[4] = 0.6;   // inside (0.72 ≤ 1)
            const out = ctx.kernel(xs, ys);
            ctx.assert(out && out.length === 4096, `expected 4096 verdicts, got ${out && out.length}`);
            const expected = [1, 0, 1, 0, 1];
            const hint = verdictHint(out, xs, ys, expected);
            for (let i = 0; i < expected.length; i++) {
              ctx.assertClose(out[i], expected[i], 1e-6,
                hint || `dart ${i} at (${xs[i]}, ${ys[i]})`);
            }
          },
        },
        {
          name: 'hit fraction over the seeded darts approaches <code>π/4</code>',
          run: async ctx => {
            const { xs, ys } = makePairs(ctx.utils, 4096, 9001);
            const out = ctx.kernel(xs, ys);
            let count = 0;
            for (let i = 0; i < out.length; i++) {
              ctx.assert(out[i] === 0 || out[i] === 1, `verdict ${i} is ${out[i]} — return exactly 1 or 0`);
              count += out[i];
            }
            ctx.assertClose(count, countInside(xs, ys), 2, 'hit count over the seeded darts');
            ctx.assertClose((4 * count) / 4096, Math.PI, 0.06, 'π estimate from 4096 darts');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { xs, ys } = makePairs(ctx.utils, 4096, 4242);
            const out = ctx.kernel(xs, ys);
            let count = 0;
            for (let i = 0; i < out.length; i++) count += out[i];
            ctx.assertClose(count, countInside(xs, ys), 2, 'hit count on unseen darts');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'reduce-to-pi',
      title: 'Reduce 65,536 Hits to π',
      intro: `<p>Last task summed the verdicts with a JavaScript loop — fine for 4,096 darts,
        wasteful for 65,536 and absurd for a billion. The GPU answer is a
        <strong>parallel reduction</strong>: don't ship every verdict home, ship
        <em>partial sums</em>. A second kernel with 256 threads gives each thread its own
        256-verdict slice to total, collapsing 65,536 numbers to 256 in one launch.</p>
        <p>Thread <code>t</code> owns the slice starting at <code>t * 256</code> — a statically
        bounded <code>for</code> loop walks it. JavaScript then folds the 256 partials into the
        final count, and <code>4 × hits / 65536</code> is your π.</p>`,
      goal: `<strong>Goal:</strong> complete the <code>partialSums</code> kernel so each of its
        256 threads returns the sum of its own 256-element slice of <code>hits</code>, then log
        the π estimate.`,
      requirements: [
        'Kernel 1 (<code>inside</code>) is last task\'s dart test — leave it as is',
        'In <code>partialSums</code>, thread <code>x</code> starts at <code>this.thread.x * 256</code>',
        'Loop <code>i = 0…255</code> and accumulate <code>hits[base + i]</code>',
        'Total the 256 partials in JavaScript and log <code>4 * total / 65536</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — who sums what',
          body: `<p>Thread 0 sums <code>hits[0…255]</code>, thread 1 sums <code>hits[256…511]</code>,
            and so on. The starting offset is <code>this.thread.x * 256</code>.</p>`,
        },
        {
          title: 'Hint 2 — the loop',
          body: `<pre><code>const base = this.thread.x * 256;
let sum = 0;
for (let i = 0; i &lt; 256; i++) {
  sum += hits[base + i];
}
return sum;</code></pre>
<p>The bound is a literal, so gpu.js can unroll it safely.</p>`,
        },
      ],
      transfer: `Reduction is <em>the</em> fundamental pattern of GPU computing — CUDA has warp
        shuffles and the CUB library for it, Metal has SIMD-group reductions, WebGPU builds them
        from workgroup shared memory. Chunked partial sums like yours are always the first rung.`,
      starterCode: `// 65,536 darts. Kernel 1 judges them; kernel 2 sums them — in parallel.
const gpu = new GPU({ mode });

const inside = gpu.createKernel(function (xs, ys) {
  const x = xs[this.thread.x];
  const y = ys[this.thread.x];
  if (x * x + y * y <= 1) {
    return 1;
  }
  return 0;
}, { output: [65536] });

const partialSums = gpu.createKernel(function (hits) {
  // TODO: sum THIS thread's 256-element slice of hits.
  // Slice start: this.thread.x * 256.
  return hits[this.thread.x];
}, { output: [256] });

const hits = inside(xs, ys);
const partials = partialSums(hits);

let total = 0;
for (let i = 0; i < partials.length; i++) total += partials[i];
console.log('π ≈', (4 * total) / 65536);
`,
      solutionCode: `// 65,536 darts. Kernel 1 judges them; kernel 2 sums them — in parallel.
const gpu = new GPU({ mode });

const inside = gpu.createKernel(function (xs, ys) {
  const x = xs[this.thread.x];
  const y = ys[this.thread.x];
  if (x * x + y * y <= 1) {
    return 1;
  }
  return 0;
}, { output: [65536] });

const partialSums = gpu.createKernel(function (hits) {
  const base = this.thread.x * 256;
  let sum = 0;
  for (let i = 0; i < 256; i++) {
    sum += hits[base + i];
  }
  return sum;
}, { output: [256] });

const hits = inside(xs, ys);
const partials = partialSums(hits);

let total = 0;
for (let i = 0; i < partials.length; i++) total += partials[i];
console.log('π ≈', (4 * total) / 65536);
`,
      inputs: utils => makePairs(utils, 65536, 1337),
      publicTests: [
        {
          name: 'reduction kernel collapses a known array to correct partial sums',
          run: async ctx => {
            const reduce = ctx.kernels.find(
              k => k.kernel && Array.isArray(k.kernel.output) && k.kernel.output[0] === 256
            );
            ctx.assert(reduce, 'no kernel with output [256] found — keep the partialSums kernel');
            // hits[i] = i % 3 → slice sums are computable exactly
            const fake = new Array(65536);
            for (let i = 0; i < 65536; i++) fake[i] = i % 3;
            const partials = reduce(new Float32Array(fake));
            ctx.assert(partials && partials.length === 256, `expected 256 partials, got ${partials && partials.length}`);
            for (const t of [0, 1, 17, 128, 255]) {
              let expected = 0;
              for (let i = 0; i < 256; i++) expected += (t * 256 + i) % 3;
              const hint = diagnose(partials[t], expected, 0.5, sliceProbes(fake, t, 256));
              ctx.assertClose(partials[t], expected, 0.5, hint || `partial sum for thread ${t}`);
            }
          },
        },
        {
          name: 'π comes out within <code>±0.05</code> over the 65,536 seeded darts',
          run: async ctx => {
            const inside = ctx.kernels.find(
              k => k.kernel && Array.isArray(k.kernel.output) && k.kernel.output[0] === 65536
            );
            const reduce = ctx.kernels.find(
              k => k.kernel && Array.isArray(k.kernel.output) && k.kernel.output[0] === 256
            );
            ctx.assert(inside && reduce, 'expected the inside kernel [65536] and the partialSums kernel [256]');
            const { xs, ys } = makePairs(ctx.utils, 65536, 1337);
            const hits = inside(xs, ys);
            const partials = reduce(new Float32Array(hits));
            let total = 0;
            for (let i = 0; i < partials.length; i++) total += partials[i];
            ctx.assertClose(total, countInside(xs, ys), 4, 'total hit count after reduction');
            ctx.assertClose((4 * total) / 65536, Math.PI, 0.05, 'π estimate');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const inside = ctx.kernels.find(
              k => k.kernel && Array.isArray(k.kernel.output) && k.kernel.output[0] === 65536
            );
            const reduce = ctx.kernels.find(
              k => k.kernel && Array.isArray(k.kernel.output) && k.kernel.output[0] === 256
            );
            ctx.assert(inside && reduce, 'expected the inside kernel [65536] and the partialSums kernel [256]');
            const { xs, ys } = makePairs(ctx.utils, 65536, 2718);
            const hits = inside(xs, ys);
            const partials = reduce(new Float32Array(hits));
            let total = 0;
            for (let i = 0; i < partials.length; i++) total += partials[i];
            ctx.assertClose(total, countInside(xs, ys), 4, 'hit count on unseen darts');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'integrate-the-unintegrable',
      title: 'Integrate the Un-integrable',
      intro: `<p><code>e^(−x²)</code> — the bell curve — famously has <strong>no elementary
        antiderivative</strong>. No substitution, no parts, no closed form. Monte Carlo doesn't
        care: for uniform samples on [0, 1], the <em>average</em> of <code>f(x)</code> converges
        to <code>∫₀¹ f(x) dx</code>. Sampling beats symbolic calculus.</p>
        <p>And here's the efficiency move over last task: instead of one kernel to evaluate and
        another to reduce, <strong>fuse them</strong>. Each of 256 threads walks its own 64-sample
        slice, evaluating <code>e^(−x²)</code> and accumulating in one pass — 16,384 evaluations,
        one launch, 256 numbers back.</p>`,
      goal: `<strong>Goal:</strong> make each thread return the sum of <code>e^(−x²)</code> over
        its 64-sample slice of <code>samples</code>, so the logged mean lands on
        <code>≈ 0.7468</code>.`,
      requirements: [
        'Thread <code>x</code> owns the slice starting at <code>this.thread.x * 64</code>',
        'Evaluate <code>Math.exp(-x * x)</code> for each sample — inside the loop, inside the kernel',
        'Return the slice sum; JavaScript divides the grand total by 16384',
      ],
      hints: [
        {
          title: 'Hint 1 — mean value, not area sampling',
          body: `<p>No darts this time: the estimator is just the average height of the curve,
            <code>(1/N) Σ f(xᵢ)</code>, times the interval width (here 1). You only need
            <code>f</code>, not a hit test.</p>`,
        },
        {
          title: 'Hint 2 — one line changes',
          body: `<p>The loop skeleton is last task's reduction. Swap what you accumulate:</p>
<pre><code>const x = xs[base + i];
sum += Math.exp(-x * x);</code></pre>`,
        },
      ],
      transfer: `Fusing the map into the reduction halves the memory traffic — the same reasoning
        behind kernel fusion in CUDA and ROCm, and behind doing per-workgroup sums in a single
        WebGPU compute pass instead of two. Bandwidth, not arithmetic, is usually the bill.`,
      starterCode: `// ∫₀¹ e^(−x²) dx has no closed form. Estimate it: average f over
// 16,384 seeded samples — 256 threads × 64 samples each, fused map+reduce.
const gpu = new GPU({ mode });

const partials = gpu.createKernel(function (xs) {
  const base = this.thread.x * 64;
  let sum = 0;
  for (let i = 0; i < 64; i++) {
    const x = xs[base + i];
    // TODO: accumulate f(x) = e^(−x²) — not x itself.
    sum += x;
  }
  return sum;
}, { output: [256] });

const sums = partials(samples);

let total = 0;
for (let i = 0; i < sums.length; i++) total += sums[i];
console.log('∫₀¹ e^(−x²) dx ≈', total / 16384, '(truth ≈ 0.746824)');
`,
      solutionCode: `// ∫₀¹ e^(−x²) dx has no closed form. Estimate it: average f over
// 16,384 seeded samples — 256 threads × 64 samples each, fused map+reduce.
const gpu = new GPU({ mode });

const partials = gpu.createKernel(function (xs) {
  const base = this.thread.x * 64;
  let sum = 0;
  for (let i = 0; i < 64; i++) {
    const x = xs[base + i];
    sum += Math.exp(-x * x);
  }
  return sum;
}, { output: [256] });

const sums = partials(samples);

let total = 0;
for (let i = 0; i < sums.length; i++) total += sums[i];
console.log('∫₀¹ e^(−x²) dx ≈', total / 16384, '(truth ≈ 0.746824)');
`,
      inputs: utils => ({ samples: makeUniforms(utils, 16384, 6077) }),
      publicTests: [
        {
          name: 'each thread sums <code>e^(−x²)</code> over its own 64-sample slice',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            // evenly spaced samples → every partial is independently checkable
            const grid = new Array(16384);
            for (let i = 0; i < 16384; i++) grid[i] = i / 16384;
            const out = ctx.kernel(grid);
            ctx.assert(out && out.length === 256, `expected 256 partial sums, got ${out && out.length}`);
            for (const t of [0, 3, 100, 255]) {
              const slice = grid.slice(t * 64, t * 64 + 64);
              const hint = diagnose(out[t], gaussSum(slice), 0.05, integrandProbes(grid, t * 64, 64));
              ctx.assertClose(out[t], gaussSum(slice), 0.05, hint || `partial sum for thread ${t}`);
            }
          },
        },
        {
          name: 'estimate lands within <code>±0.01</code> of the true value <code>0.746824</code>',
          run: async ctx => {
            const samples = makeUniforms(ctx.utils, 16384, 6077);
            const out = ctx.kernel(samples);
            let total = 0;
            for (let i = 0; i < out.length; i++) total += out[i];
            const hint = diagnose(total / 16384, 0.7468241328124271, 0.01,
              integrandProbes(samples, 0, 16384).map(p => [p[0] / 16384, p[1]]));
            ctx.assertClose(total / 16384, 0.7468241328124271, 0.01,
              hint || 'Monte Carlo integral estimate');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const samples = makeUniforms(ctx.utils, 16384, 1912);
            const out = ctx.kernel(samples);
            let total = 0;
            for (let i = 0; i < out.length; i++) total += out[i];
            const hint = diagnose(total / 16384, gaussSum(samples) / 16384, 2e-3,
              integrandProbes(samples, 0, 16384).map(p => [p[0] / 16384, p[1]]));
            ctx.assertClose(total / 16384, gaussSum(samples) / 16384, 2e-3,
              hint || 'estimate vs float64 reference');
            ctx.assertClose(total / 16384, 0.7468241328124271, 0.01, 'estimate vs the true integral');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'price-an-option',
      title: 'Price an Option',
      intro: `<p>The payoff. A <strong>European call option</strong> is the right to buy a stock at
        a fixed strike price K on a future date — worth <code>max(S_T − K, 0)</code> when the stock
        finishes at <code>S_T</code>, and its fair price today is the <em>discounted expected
        payoff</em>. Expectations are integrals, and you just learned to integrate by sampling.</p>
        <p>Each thread simulates one possible market: under the standard log-normal model, a
        pre-drawn normal shock <code>z</code> gives
        <code>S_T = S0 · e^(drift + volT · z)</code>. Your kernel turns 16,384 shocks into
        16,384 payoffs; JavaScript averages and discounts. Stock at 100, strike 105, one year out —
        the Black–Scholes formula says the answer is ≈ 7.13. Your simulation should agree.</p>`,
      goal: `<strong>Goal:</strong> complete the payoff kernel — simulate this thread's final stock
        price and return the option payoff <code>max(S_T − strike, 0)</code>.`,
      requirements: [
        'Simulate the final price: <code>s0 * Math.exp(drift + volT * z)</code> (already wired)',
        'Return the call payoff: <code>Math.max(st - strike, 0)</code> — an option never goes negative',
        'Average the payoffs and discount by <code>Math.exp(-RATE * T)</code> in JavaScript',
      ],
      hints: [
        {
          title: 'Hint 1 — why the max?',
          body: `<p>If the stock ends below the strike you simply don't exercise — the option
            expires worthless, payoff 0, never negative. Forgetting the <code>max</code> drags the
            average down by every losing path (the price comes out near −1.9 instead of ≈ 7.1).</p>`,
        },
        {
          title: 'Hint 2 — the kernel body',
          body: `<p><code>return Math.max(st - strike, 0);</code> — <code>Math.max</code> works
            inside kernels, and beats an <code>if</code> here.</p>`,
        },
      ],
      transfer: `This is production reality: quant desks run exactly this workload on CUDA and ROCm
        — millions of simulated paths per pricing call, one thread per path, then a reduction —
        because exotic options have no closed form at all. You now hold the whole recipe.`,
      starterCode: `// Fair price = discounted average payoff over simulated futures.
// Stock at 100, strike 105, 3% rate, 20% volatility, 1 year to expiry.
const S0 = 100, STRIKE = 105, RATE = 0.03, SIGMA = 0.2, T = 1;

const gpu = new GPU({ mode });

const payoff = gpu.createKernel(function (normals, s0, strike, drift, volT) {
  const z = normals[this.thread.x];
  const st = s0 * Math.exp(drift + volT * z); // this thread's final stock price
  // TODO: return the call payoff — st minus strike, but never below zero.
  return st - strike;
}, { output: [16384] });

const payoffs = payoff(normals, S0, STRIKE, (RATE - SIGMA * SIGMA / 2) * T, SIGMA * Math.sqrt(T));

let sum = 0;
for (let i = 0; i < payoffs.length; i++) sum += payoffs[i];
const price = Math.exp(-RATE * T) * (sum / payoffs.length);
console.log('Monte Carlo price:', price, '— Black–Scholes says ≈ 7.13');
`,
      solutionCode: `// Fair price = discounted average payoff over simulated futures.
// Stock at 100, strike 105, 3% rate, 20% volatility, 1 year to expiry.
const S0 = 100, STRIKE = 105, RATE = 0.03, SIGMA = 0.2, T = 1;

const gpu = new GPU({ mode });

const payoff = gpu.createKernel(function (normals, s0, strike, drift, volT) {
  const z = normals[this.thread.x];
  const st = s0 * Math.exp(drift + volT * z); // this thread's final stock price
  return Math.max(st - strike, 0);
}, { output: [16384] });

const payoffs = payoff(normals, S0, STRIKE, (RATE - SIGMA * SIGMA / 2) * T, SIGMA * Math.sqrt(T));

let sum = 0;
for (let i = 0; i < payoffs.length; i++) sum += payoffs[i];
const price = Math.exp(-RATE * T) * (sum / payoffs.length);
console.log('Monte Carlo price:', price, '— Black–Scholes says ≈ 7.13');
`,
      inputs: utils => ({ normals: makeNormals(utils, 16384, 8128) }),
      publicTests: [
        {
          name: 'payoffs are <code>max(S_T − K, 0)</code> — losing paths pay exactly zero',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const shocks = new Array(16384).fill(0);
            shocks[0] = -3;   // crash  → deep out of the money → payoff 0
            shocks[1] = 2;    // rally  → deep in the money
            shocks[2] = 0.5;  // mildly up
            shocks[3] = -0.5; // mildly down → out of the money
            const out = ctx.kernel(shocks, OPT.s0, OPT.strike, OPT_DRIFT, OPT_VOLT);
            ctx.assert(out && out.length === 16384, `expected 16384 payoffs, got ${out && out.length}`);
            for (const i of [0, 1, 2, 3]) {
              const st = OPT.s0 * Math.exp(OPT_DRIFT + OPT_VOLT * shocks[i]);
              const expected = Math.max(st - OPT.strike, 0);
              const hint = diagnose(out[i], expected, 0.05, payoffProbes(st, OPT.strike));
              ctx.assertClose(out[i], expected, 0.05, hint || `payoff for shock z = ${shocks[i]}`);
              ctx.assert(out[i] >= 0, `payoff for z = ${shocks[i]} is negative (${out[i]}) — options never go below zero`);
            }
          },
        },
        {
          name: 'simulated price agrees with Black–Scholes (<code>≈ 7.13</code>) within <code>±0.4</code>',
          run: async ctx => {
            const normals = makeNormals(ctx.utils, 16384, 8128);
            const out = ctx.kernel(normals, OPT.s0, OPT.strike, OPT_DRIFT, OPT_VOLT);
            let sum = 0;
            for (let i = 0; i < out.length; i++) sum += out[i];
            const price = Math.exp(-OPT.rate * OPT.t) * (sum / out.length);
            const hint = diagnose(price, referencePrice(normals), 0.05, [
              [referencePriceUnfloored(normals),
                'that is the average of st − strike with no floor — every losing path dragged the mean below zero'],
            ]);
            ctx.assertClose(price, referencePrice(normals), 0.05,
              hint || 'price vs float64 reference simulation');
            ctx.assertClose(price, blackScholesCall(), 0.4, 'price vs the Black–Scholes closed form');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const normals = makeNormals(ctx.utils, 16384, 6174);
            const out = ctx.kernel(normals, OPT.s0, OPT.strike, OPT_DRIFT, OPT_VOLT);
            let sum = 0;
            for (let i = 0; i < out.length; i++) sum += out[i];
            const price = Math.exp(-OPT.rate * OPT.t) * (sum / out.length);
            const hint = diagnose(price, referencePrice(normals), 0.05, [
              [referencePriceUnfloored(normals),
                'that is the average of st − strike with no floor — every losing path dragged the mean below zero'],
            ]);
            ctx.assertClose(price, referencePrice(normals), 0.05,
              hint || 'price vs float64 reference on unseen shocks');
            ctx.assertClose(price, blackScholesCall(), 0.5, 'price vs Black–Scholes on unseen shocks');
          },
        },
      ],
    },
  ],
};
