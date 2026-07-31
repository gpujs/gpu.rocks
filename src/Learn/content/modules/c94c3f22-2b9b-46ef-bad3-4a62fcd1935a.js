// Module: Gradient Descent — uuid c94c3f22-2b9b-46ef-bad3-4a62fcd1935a
// (short id c94c3f22). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module postdates
// the uuid migration.
//
// Five tasks: the least-squares loss as a fused partial-sum reduction → both
// gradient components from one walk over the data → the descent loop, driven
// from JavaScript → 256 learning rates swept one per thread, which puts the
// stability limit on screen → 1,024 starts on a non-convex surface landing in
// three different minima.
//
// This is numerical optimisation, not machine learning. The gradients are two
// lines of calculus and are written out by hand, because for a model this small
// they can be; nothing here is a neural network and the course does not teach
// one.
//
// ---------------------------------------------------------------------------
// THE DATA, AND WHY EVERY NUMBER IN THIS MODULE IS EXACT
//
// makeFit(n) tiles a 64-point block. Inside the block the points come in
// mirrored pairs — point 2p sits at x = +MAGS[p % 8], point 2p+1 at
// x = −MAGS[p % 8] — and BOTH members of a pair carry the same scatter
// NOISE[p]. Three exact identities follow, for every n that is a multiple
// of 64:
//
//   Σ x = 0                (the pairs cancel)
//   Σ x² / n = 1           (Σ MAGS² = 8, twice per pair, 8 pairs per 16 points)
//   Σ e = 0, Σ e·x = 0     (Σ NOISE = 0; the scatter is constant on a ± pair)
//   Σ e² / n = 0.5         (Σ NOISE² = 16 over 32 pairs)
//
// so the least-squares optimum is EXACTLY the generating line y = 3x + 4, and
// the loss surface is exactly
//
//     L(m, c) = (m − 3)² + (c − 4)² + 0.5
//
// — a circular paraboloid with its floor at 0.5. Hence L(0, 0) = 25.5,
// ∇L(0, 0) = (−6, −8), the Hessian is 2·I, λ_max = 2, and the learning-rate
// stability limit is exactly η < 2/λ_max = 1. Tasks 1–3 use n = 4,096 and
// task 4 uses n = 64; the surface is the same to the last bit either way,
// which is the point of the construction.
//
// FLOAT MARGINS (measured, not guessed — scripted in float64 and again with
// Math.fround on every kernel-side operation, which is what the GL backend
// does):
//   • every x is a multiple of 0.25 in [−1.5, 1.5] and every y a multiple of
//     0.25 in [−1.5, 9.25], so the inputs are bit-identical on both backends;
//   • task 3, 60 steps at η = 0.1: |m − 3| = 4.5e−6 in float32 and 4.6e−6 in
//     float64 (the geometric tail dominates, not rounding) — asserted at 1e−3;
//   • task 4, worst |L − 0.5| over 0.125 ≤ η ≤ 0.875: 1.2e−7 float32,
//     2.2e−16 float64 — asserted at 1e−4. The converged/diverged split is
//     identical in both precisions (125 rates below loss 1, 124 above 1e6);
//   • task 5, worst |w_final − minimum| over all 1,024 threads: 2.0e−6 in
//     both precisions — asserted at 0.01, and the basin counts (308/409/307)
//     are identical in both.
//
// Kernel-authoring rules (contract): no closures, arguments + this.thread.* +
// this.constants.* only, statically bounded loops, no kernel local sharing a
// name with a constant, no boolean stored in a variable. Every fraction a
// kernel divides by arrives as a float constant (gradScale, invPoints) rather
// than as `2 / this.constants.points`, because two integer constants divide as
// integers in GLSL. Every task passes in cpu mode as well as gpu mode; the
// largest launch is 1,024 threads, comfortably under the pre-flight guard's
// 65,536-thread floor, so no task needs budgetMs.

// Eight |x| magnitudes whose squares sum to exactly 8. The order is cosmetic —
// every identity below is a sum, so it survives any permutation — and this one
// is chosen so the first samples the Task inputs panel prints look like a
// spread of measurements rather than a repeated pair.
const MAGS = [1.5, 0.25, 1, 0.5, 1.25, 0.25, 0.75, 1.5];

// Thirty-two scatter values, one per mirrored pair: Σ = 0, Σ² = 16.
const NOISE = [
  0.75, -1, 0.5, -0.25, 1, -0.75, -0.5, 0.25,
  -1, 0.75, -0.75, 0.5, 0.25, 1, -0.5, -0.25,
  0.5, -0.75, 1, -1, -0.25, 0.75, 0.25, -0.5,
  -0.75, 1, -1, 0.75, -0.5, 0.5, -0.75, 0.75,
];

const M_STAR = 3;
const C_STAR = 4;
const L_STAR = 0.5;

// n points on the line y = 3x + 4 plus exactly-cancelling scatter.
// n must be a multiple of 64.
function makeFit(n) {
  const xs = new Array(n);
  const ys = new Array(n);
  for (let i = 0; i < n; i++) {
    const pair = (i >> 1) % 32;
    const x = (i % 2 === 0 ? 1 : -1) * MAGS[pair % 8];
    xs[i] = x;
    ys[i] = M_STAR * x + C_STAR + NOISE[pair];
  }
  return { xs, ys };
}

// The analytic truth, used to build every expectation in the tests.
function trueLoss(m, c) {
  return (m - M_STAR) ** 2 + (c - C_STAR) ** 2 + L_STAR;
}

function trueGradient(m, c) {
  return [2 * (m - M_STAR), 2 * (c - C_STAR)];
}

// Float64 references for the strided 64×64 walk tasks 1–3 use.
function stridedResidualSum(xs, ys, t, weight, square) {
  let sum = 0;
  for (let k = 0; k < 64; k++) {
    const at = k * 64 + t;
    const r = weight.m * xs[at] + weight.c - ys[at];
    sum += square ? r * r : r * (weight.byX ? xs[at] : 1);
  }
  return sum;
}

// The contiguous walk a learner writes instead: thread t taking the block
// [t·64, t·64 + 63] rather than every 64th element from t.
function contiguousResidualSum(xs, ys, t, weight, square) {
  let sum = 0;
  for (let k = 0; k < 64; k++) {
    const at = t * 64 + k;
    const r = weight.m * xs[at] + weight.c - ys[at];
    sum += square ? r * r : r * (weight.byX ? xs[at] : 1);
  }
  return sum;
}

// Whole-array float64 references.
function meanOver(xs, ys, m, c, f) {
  let sum = 0;
  for (let i = 0; i < xs.length; i++) sum += f(m * xs[i] + c - ys[i], xs[i]);
  return sum / xs.length;
}

// ---- task 4 and 5 references ----------------------------------------------

// 256 learning rates: (k + 1)/128, so 0.0078125 … 2, hitting 1/8, 1/2, 1 and
// 3/2 exactly. The stability wall sits at exactly 1.
function makeRates() {
  const rates = new Array(256);
  for (let k = 0; k < 256; k++) rates[k] = (k + 1) / 128;
  return rates;
}

// 1,024 starting points across [−2.5, 2.5). The grid deliberately misses the
// two ridges at ±1 (they would need k = 307.2 and 716.8), so no thread starts
// balanced on one.
function makeStarts() {
  const starts = new Array(1024);
  for (let k = 0; k < 1024; k++) starts[k] = -2.5 + k * (5 / 1024);
  return starts;
}

// Where a start ends up: the well it is already inside. Ridges at ±1.
function basinOf(start) {
  return start < -1 ? -2 : start < 1 ? 0 : 2;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so a point where two candidates coincide (the
// optimum, where the mean residual and the mean signed residual are both zero)
// stays silent, as do observations matching probes that disagree with each
// other. A wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 1: the three ways a mean squared residual comes out wrong.
function lossProbes(xs, ys, m, c) {
  return [
    [meanOver(xs, ys, m, c, r => r),
      'that is the mean residual, not the mean SQUARED residual — the squaring never happened, so positive and negative misses cancelled'],
    [meanOver(xs, ys, m, c, r => Math.abs(r)),
      'that is the mean ABSOLUTE residual — least squares squares each residual, it does not take its size'],
    [meanOver(xs, ys, m, c, r => r * r) * xs.length,
      'that is the total squared residual — the loss is the MEAN, so divide the total by the number of points'],
  ];
}

// Tasks 1 and 2: the walk itself. Only checked per-thread, where a contiguous
// block and a strided stride actually disagree (their totals never do).
function walkProbes(xs, ys, t, weight, square) {
  return [
    [contiguousResidualSum(xs, ys, t, weight, square),
      'that is the sum over a contiguous block — the walk is strided, i * this.constants.threads + this.thread.x, so neighbouring threads touch neighbouring points (the reason Reductions builds its partials that way)'],
    [stridedResidualSum(xs, ys, 0, weight, square),
      "every thread summed thread 0's slice — this.thread.x has to appear in the index"],
  ];
}

// Task 2: the five ways a two-row gradient comes out wrong. A single component
// cannot tell them apart — swapping the rows and dropping the x weight both
// put ∂L/∂c's value in ∂L/∂m's slot — so the whole PAIR has to match a
// candidate, and the candidate has to disagree with the truth in at least one
// component, before anything is said. At m = 1, c = 2 the two components are
// equal and every one of these candidates is invisible; that is the point of
// probing several (m, c).
function gradientPairHint(got, m, c, eps) {
  const [g0, g1] = trueGradient(m, c);
  const candidates = [
    [[g1, g0],
      'the two rows are the wrong way round — row 0 of the output is the Σ r·x partials (∂L/∂m) and row 1 the Σ r partials (∂L/∂c)'],
    [[g0 / 2, g1 / 2],
      'that is half the gradient — differentiating a square leaves a factor of 2 out front: ∂L/∂m = (2/n) Σ r·x'],
    [[g0 * 2048, g1 * 2048],
      'those are the raw sums, never averaged — both components carry a 2/n, and n is 4096 here'],
    [[g1, g1],
      'the slope row summed r with no weight — ∂L/∂m weights each residual by its own x: sm += r * xs[at]'],
    [[g0, g0],
      'both rows came back with the slope sum — the kernel needs the branch on this.thread.y, so that row 1 returns sc'],
  ];
  const hits = candidates
    .filter(([value]) =>
      Math.abs(got[0] - value[0]) <= eps && Math.abs(got[1] - value[1]) <= eps &&
      (Math.abs(g0 - value[0]) > eps || Math.abs(g1 - value[1]) > eps))
    .map(c2 => c2[1]);
  return hits.length && hits.every(h => h === hits[0]) ? hits[0] : null;
}

// Every number that appeared in a console.log line.
function loggedNumbers(logs) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

// Task 3: a run that walked the wrong way, or stepped with the raw sum. The
// two are told apart by magnitude, measured: ascent at η = 0.1 grows the error
// by 1.2 per step and ends finite near −1.7e5, while a step taken with the
// unaveraged sum grows it by 409 per step and overflows float32 at step 15.
// Infinity and NaN reach the console as words, not digits, so the text is
// searched before the numbers are.
function runawayHint(logs, nums) {
  if (logs.some(line => line.text && /\b(?:Infinity|NaN)\b/.test(line.text))) {
    return 'the parameters overflowed to Infinity — the step used the raw sums instead of the gradient. Both components carry a 2/n out front; without it every step is 2,048 times too long';
  }
  if (nums.some(v => Math.abs(v) > 1e3)) {
    return 'the parameters ran away to ±10⁵ instead of settling — the update ADDED the gradient. The gradient points uphill, so descent subtracts it';
  }
  return null;
}

// Task 3: find the two kernels by their output shape.
function findByOutput(ctx, want) {
  return ctx.kernels.find(k => {
    const out = k.kernel && k.kernel.output;
    return Array.isArray(out) && out.length === want.length &&
      want.every((v, i) => out[i] === v);
  }) || null;
}

// Task 3: drive a learner's gradient kernel through the descent ourselves.
function driveDescent(gradKernel, xs, ys, rate, steps, m0, c0) {
  let m = m0;
  let c = c0;
  for (let s = 0; s < steps; s++) {
    const rows = gradKernel(xs, ys, m, c);
    let sumMx = 0;
    let sumC = 0;
    for (let i = 0; i < 64; i++) {
      sumMx += rows[0][i];
      sumC += rows[1][i];
    }
    m -= (rate * 2 * sumMx) / xs.length;
    c -= (rate * 2 * sumC) / xs.length;
  }
  return [m, c];
}

// Task 4: what a sweep looks like when every thread is wrong the same way. The
// three signatures are measured and disjoint, and each is checked at several
// rates so that no single reading can be mistaken for another:
//   • returning m or c instead of the loss puts ≈ 3 or ≈ 4 at rate 1/2, where
//     a working sweep reads exactly 0.5;
//   • dropping gradScale multiplies every step by 32, which moves the wall
//     from η = 1 down to η = 1/32: rate 1/64 still converges, rates 1/8 and
//     1/2 detonate — something a working sweep never does;
//   • stepping the wrong way leaves no converging rate anywhere, and even the
//     smallest rate comes back near 300.
// The order matters: the parameter-not-loss case is ruled out first, because
// its readings can look like anything.
function sweepShapeHint(out) {
  const converged = k => out[k] < 1;
  const blown = k => !(out[k] < 1e6);
  if (Math.abs(out[63] - M_STAR) < 0.05 || Math.abs(out[63] - C_STAR) < 0.05) {
    return 'rate 1/2 came back as ≈ 3 or ≈ 4 — that is the fitted slope or intercept, not the loss. Return the mean squared residual measured after the last step';
  }
  if (converged(1) && blown(15) && blown(63)) {
    return 'the wall has moved to η ≈ 1/32 — rate 1/64 converges and rate 1/8 already explodes. Every step is 32× too long, which is exactly what dropping this.constants.gradScale (the 2/n) does';
  }
  if (!out.some((v, k) => converged(k)) && out[0] > 100) {
    return 'not one of the 256 rates converged, and even the smallest overshot — that is what stepping the wrong way looks like. Descent SUBTRACTS the gradient';
  }
  return null;
}

// Task 5: every thread reporting the same number, or every thread climbing.
function wellsHint(out, starts) {
  const first = out[0];
  if (out.every(v => Math.abs(v - first) < 1e-6)) {
    return `all 1,024 threads returned the same value (${first.toFixed(4)}) — each thread has to read its OWN start: starts[this.thread.x]`;
  }
  const finite = out.filter(Number.isFinite);
  const onRidges = finite.filter(v => Math.abs(Math.abs(v) - 1) < 1e-3).length;
  if (finite.length && onRidges > finite.length * 0.9) {
    return 'nearly every thread finished at w = ±1 — those are the two RIDGES, the peaks between the valleys. Climbing to them means the step added the slope instead of subtracting it';
  }
  if (starts.length && out.some(v => !Number.isFinite(v))) {
    return 'some threads escaped to Infinity — at rate 0.02 the slope w⁵ − 5w³ + 4w keeps every one of these starts bounded, so the expression in the loop is producing something far steeper than that';
  }
  return null;
}

export default {
  uuid: 'c94c3f22-2b9b-46ef-bad3-4a62fcd1935a',
  version: 1,
  slug: 'gradient-descent',
  title: 'Gradient Descent',
  blurb: 'Fit a line by walking downhill — the gradient as a reduction, the learning rate as a stability limit, and 1,024 searches in one launch.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'mean-squared-error',
      title: 'The Loss You Can Check',
      intro: `<p>Fitting is optimisation. Pick a model — here the straight line
        <code>y = m·x + c</code> — pick a single number that says how badly it fits, then go
        looking for the parameters that make that number small. The number is the
        <strong>loss</strong>; for least squares it is the mean squared residual. Gradient
        descent has a reputation as the engine inside model training, but underneath it is
        nothing more than a numerical method for walking downhill, and that is all this
        module asks of it.</p>
        <p>Look at what the loss actually <em>is</em>:</p>
<pre><code>L(m, c) = (1/n) · Σ (m·xᵢ + c − yᵢ)²</code></pre>
        <p>A sum over every data point, divided by n — a <strong>reduction</strong>, the same
        many-in-one-out shape the Reductions module builds its halving ladder for. So the first
        kernel here is a partial-sum kernel with the squaring fused into the read: 64 threads,
        64 points each, one pass over memory.</p>
        <p>These 4,096 points were built so that every claim in this module is checkable to the
        last digit. The best-fitting line is exactly <code>y = 3x + 4</code>, the lowest
        reachable loss is exactly <code>0.5</code>, and the whole surface is
        <code>L(m, c) = (m − 3)² + (c − 4)² + 0.5</code>. From <code>m = c = 0</code>, then,
        the loss must read <code>25.5</code>.</p>`,
      goal: `<strong>Goal:</strong> finish the partial-sum kernel so it accumulates
        <strong>squared</strong> residuals, then divide the grand total by 4,096 and log the
        loss at <code>m = 0, c = 0</code> — it should be <code>25.5</code>.`,
      requirements: [
        'Each of the 64 threads walks its strided slice: <code>i * this.constants.threads + this.thread.x</code>',
        'The residual of point <code>at</code> is <code>m * xs[at] + c - ys[at]</code>',
        'Accumulate its <em>square</em> — squaring happens as the value is read, not in a second pass',
        'Divide the total of the 64 partials by 4096 before logging',
      ],
      hints: [
        {
          title: 'Hint 1 — read once, square immediately',
          body: `<p>Name the residual, then square the name — no second pass over the data:</p>
<pre><code>const r = m * xs[at] + c - ys[at];
sum += r * r;</code></pre>`,
        },
        {
          title: 'Hint 2 — mean, not total',
          body: `<p>The 64 partials add up to <code>Σ r²</code> over all 4,096 points. The loss
            is the <em>mean</em> squared residual, so the last step is
            <code>total / 4096</code>. Forget it and you get 104,448 instead of 25.5.</p>`,
        },
      ],
      transfer: `Every training loop on every platform begins with exactly this: a per-example
        loss, summed and averaged. CUB and Thrust reduce it with a tree, a WGSL compute shader
        does it with workgroup shared memory and subgroup adds. Fusing the square into the read
        rather than running a separate squaring pass is <code>thrust::transform_reduce</code>
        in one line, and it halves the memory traffic.`,
      starterCode: `// 4,096 points, 64 threads, 64 points each — strided, so neighbouring
// threads read neighbouring points at every step of the loop.
const gpu = new GPU({ mode });

const lossPartials = gpu.createKernel(function (xs, ys, m, c) {
  let sum = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    const at = i * this.constants.threads + this.thread.x;
    // TODO: the residual of point \`at\` is m * xs[at] + c - ys[at].
    // Accumulate its SQUARE into sum.
    sum += 0;
  }
  return sum;
}, {
  output: [64],
  constants: { threads: 64, chunk: 64 },
});

const partials = lossPartials(xs, ys, 0, 0);

let total = 0;
for (let i = 0; i < partials.length; i++) total += partials[i];

// TODO: the loss is the MEAN squared residual — divide by 4096.
console.log('loss at m = 0, c = 0:', total);
`,
      solutionCode: `// 4,096 points, 64 threads, 64 points each — strided, so neighbouring
// threads read neighbouring points at every step of the loop.
const gpu = new GPU({ mode });

const lossPartials = gpu.createKernel(function (xs, ys, m, c) {
  let sum = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    const at = i * this.constants.threads + this.thread.x;
    const r = m * xs[at] + c - ys[at];
    sum += r * r;
  }
  return sum;
}, {
  output: [64],
  constants: { threads: 64, chunk: 64 },
});

const partials = lossPartials(xs, ys, 0, 0);

let total = 0;
for (let i = 0; i < partials.length; i++) total += partials[i];

console.log('loss at m = 0, c = 0:', total / 4096);
`,
      inputs: () => makeFit(4096),
      publicTests: [
        {
          name: '64 partial sums, and the loss at <code>(0, 0)</code> is <code>25.5</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const { xs, ys } = makeFit(4096);
            const out = ctx.kernel(xs, ys, 0, 0);
            ctx.assert(out && out.length === 64, `expected 64 partial sums, got ${out && out.length}`);
            let total = 0;
            for (let i = 0; i < out.length; i++) total += out[i];
            const got = total / 4096;
            const hint = diagnose(got, 25.5, 2e-3, lossProbes(xs, ys, 0, 0));
            ctx.assertClose(got, 25.5, 2e-3, hint || 'the loss at m = 0, c = 0');
          },
        },
        {
          name: 'each partial is its own strided slice of squared residuals',
          run: async ctx => {
            const { xs, ys } = makeFit(4096);
            for (const [m, c] of [[0, 0], [1, 2]]) {
              const out = ctx.kernel(xs, ys, m, c);
              const weight = { m, c };
              for (const t of [0, 1, 7, 63]) {
                const expected = stridedResidualSum(xs, ys, t, weight, true);
                const hint = diagnose(out[t], expected, 0.02, walkProbes(xs, ys, t, weight, true));
                ctx.assertClose(out[t], expected, 0.02, hint ||
                  `partial ${t} at m = ${m}, c = ${c}`);
              }
            }
          },
        },
        {
          name: 'the loss <code>25.5</code> is logged',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - 25.5) <= 2e-3),
              'log the loss — expected to see 25.5 in the console output (104448 means the total was never divided by 4096)'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { xs, ys } = makeFit(4096);
            // The surface is L(m, c) = (m − 3)² + (c − 4)² + 0.5 exactly.
            for (const [m, c] of [[3, 4], [2, 5], [-1, 6], [4.5, 4]]) {
              const out = ctx.kernel(xs, ys, m, c);
              ctx.assert(out && out.length === 64, 'expected 64 partial sums');
              let total = 0;
              for (let i = 0; i < out.length; i++) total += out[i];
              const got = total / 4096;
              const expected = trueLoss(m, c);
              const hint = diagnose(got, expected, 2e-3, lossProbes(xs, ys, m, c));
              ctx.assertClose(got, expected, 2e-3, hint || `the loss at m = ${m}, c = ${c}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'the-gradient',
      title: 'The Gradient Is a Reduction',
      intro: `<p>To walk downhill you need the direction of steepest <em>ascent</em> — the
        <strong>gradient</strong> — and then you go the other way. Differentiating the loss is
        two lines of calculus, and for a model this small there is no reason to reach for
        anything cleverer than writing them out:</p>
<pre><code>rᵢ      = m·xᵢ + c − yᵢ
∂L/∂m  = (2/n) · Σ rᵢ · xᵢ
∂L/∂c  = (2/n) · Σ rᵢ</code></pre>
        <p>Two sums, over the same residuals, in the same order. There is no reason to walk the
        data twice: one kernel computes both, holding each residual in a register and pushing it
        into two accumulators. What comes back is a 2D output — <code>output: [64, 2]</code>,
        64 threads wide and 2 rows tall — where <strong>row 0</strong> holds the
        <code>Σ r·x</code> partials and <strong>row 1</strong> the <code>Σ r</code> partials.</p>
        <p>Notice where the branch that picks the row goes: <em>after</em> the loop, never
        inside it. Every thread then runs the identical straight-line body and only the last
        statement differs. At <code>m = c = 0</code> the gradient must come out
        <code>(−6, −8)</code>, because on this data <code>∇L = (2(m − 3), 2(c − 4))</code>.</p>`,
      goal: `<strong>Goal:</strong> fill in the two accumulators and the row selection, then
        assemble the gradient in JavaScript and log it — <code>(−6, −8)</code> at
        <code>m = 0, c = 0</code>.`,
      requirements: [
        '<code>sm</code> accumulates <code>r * xs[at]</code>, <code>sc</code> accumulates <code>r</code>',
        'Row 0 of the output returns <code>sm</code>, row 1 returns <code>sc</code> — branch on <code>this.thread.y</code> <em>after</em> the loop',
        'Total each row in JavaScript and multiply by <code>2 / 4096</code>',
        'Log both components',
      ],
      hints: [
        {
          title: 'Hint 1 — one residual, two homes',
          body: `<p>Read the residual once and use it twice, exactly as the fused loss kernel
            reused it:</p>
<pre><code>const r = m * xs[at] + c - ys[at];
sm += r * xs[at];
sc += r;</code></pre>`,
        },
        {
          title: 'Hint 2 — which row am I?',
          body: `<p><code>this.thread.y</code> is 0 for the first row and 1 for the second. Put
            the choice after the loop so the loop body stays identical for every thread:</p>
<pre><code>if (this.thread.y === 0) {
  return sm;
}
return sc;</code></pre>`,
        },
        {
          title: 'Hint 3 — the 2/n out front',
          body: `<p>A 2D result is indexed <code>rows[y][x]</code>, so <code>rows[0]</code> is
            the 64 slope partials and <code>rows[1]</code> the 64 intercept partials. Both
            totals then need the same scaling: <code>(2 * total) / 4096</code>.</p>`,
        },
      ],
      transfer: `Accumulating several statistics in one pass over the data is standard practice
        everywhere: a CUDA kernel keeps both partials in registers across a single grid-stride
        loop, CUB instantiates one <code>BlockReduce</code> per accumulator but reads the tile
        once, and WGSL does the same with two workgroup arrays. The 2D output is just gpu.js's
        spelling of <em>one launch, two results</em>.`,
      starterCode: `// One walk over the data, two sums out of it.
// Row 0 → the Σ r·x partials, row 1 → the Σ r partials.
const gpu = new GPU({ mode });

const gradPartials = gpu.createKernel(function (xs, ys, m, c) {
  let sm = 0;
  let sc = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    const at = i * this.constants.threads + this.thread.x;
    const r = m * xs[at] + c - ys[at];
    // TODO: sm accumulates the residual weighted by x, sc the residual itself.
    sm += 0;
    sc += 0;
  }
  // TODO: row 0 of the output is sm, row 1 is sc. Branch on this.thread.y.
  return sm;
}, {
  output: [64, 2],
  constants: { threads: 64, chunk: 64 },
});

const rows = gradPartials(xs, ys, 0, 0);

let sumMx = 0;
let sumC = 0;
for (let i = 0; i < 64; i++) {
  sumMx += rows[0][i];
  sumC += rows[1][i];
}

// TODO: both components carry a factor of 2/n. n is 4096.
const gm = sumMx;
const gc = sumC;
console.log('gradient at m = 0, c = 0:', gm, gc);
`,
      solutionCode: `// One walk over the data, two sums out of it.
// Row 0 → the Σ r·x partials, row 1 → the Σ r partials.
const gpu = new GPU({ mode });

const gradPartials = gpu.createKernel(function (xs, ys, m, c) {
  let sm = 0;
  let sc = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    const at = i * this.constants.threads + this.thread.x;
    const r = m * xs[at] + c - ys[at];
    sm += r * xs[at];
    sc += r;
  }
  // The branch is outside the loop: every thread ran the same body to get here.
  if (this.thread.y === 0) {
    return sm;
  }
  return sc;
}, {
  output: [64, 2],
  constants: { threads: 64, chunk: 64 },
});

const rows = gradPartials(xs, ys, 0, 0);

let sumMx = 0;
let sumC = 0;
for (let i = 0; i < 64; i++) {
  sumMx += rows[0][i];
  sumC += rows[1][i];
}

const gm = (2 * sumMx) / 4096;
const gc = (2 * sumC) / 4096;
console.log('gradient at m = 0, c = 0:', gm, gc);
console.log('so downhill from here is:', -gm, -gc);
`,
      inputs: () => makeFit(4096),
      publicTests: [
        {
          name: 'the output is 2 rows of 64 partials',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const { xs, ys } = makeFit(4096);
            const out = ctx.kernel(xs, ys, 0, 0);
            ctx.assert(out && out.length === 2, `expected 2 rows, got ${out && out.length}`);
            ctx.assert(
              out[0] && typeof out[0] !== 'number' && out[0].length === 64,
              'each row should hold 64 partials — is the output still 1D?'
            );
          },
        },
        {
          name: 'the assembled gradient at <code>(0, 0)</code> is <code>(−6, −8)</code>',
          run: async ctx => {
            const { xs, ys } = makeFit(4096);
            for (const [m, c] of [[0, 0], [1, 2], [4, 4]]) {
              const rows = ctx.kernel(xs, ys, m, c);
              let sumMx = 0;
              let sumC = 0;
              for (let i = 0; i < 64; i++) {
                sumMx += rows[0][i];
                sumC += rows[1][i];
              }
              const got = [(2 * sumMx) / 4096, (2 * sumC) / 4096];
              const expected = trueGradient(m, c);
              const hint = gradientPairHint(got, m, c, 2e-3);
              for (const component of [0, 1]) {
                ctx.assertClose(got[component], expected[component], 2e-3, hint ||
                  `${component === 0 ? '∂L/∂m' : '∂L/∂c'} at m = ${m}, c = ${c}`);
              }
            }
          },
        },
        {
          name: 'each row is a strided slice, not a contiguous block',
          run: async ctx => {
            const { xs, ys } = makeFit(4096);
            const rows = ctx.kernel(xs, ys, 1, 2);
            for (const t of [0, 5, 63]) {
              const wantM = stridedResidualSum(xs, ys, t, { m: 1, c: 2, byX: true }, false);
              const hintM = diagnose(rows[0][t], wantM, 0.02,
                walkProbes(xs, ys, t, { m: 1, c: 2, byX: true }, false));
              ctx.assertClose(rows[0][t], wantM, 0.02, hintM || `row 0, partial ${t}`);
              const wantC = stridedResidualSum(xs, ys, t, { m: 1, c: 2, byX: false }, false);
              const hintC = diagnose(rows[1][t], wantC, 0.02,
                walkProbes(xs, ys, t, { m: 1, c: 2, byX: false }, false));
              ctx.assertClose(rows[1][t], wantC, 0.02, hintC || `row 1, partial ${t}`);
            }
          },
        },
        {
          name: 'the gradient is logged',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v + 6) <= 2e-3) && nums.some(v => Math.abs(v + 8) <= 2e-3),
              'log both components — expected to see -6 and -8 in the console output'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { xs, ys } = makeFit(4096);
            // At the optimum the gradient is exactly zero — the one place every
            // wrong scaling agrees with the right one, so it is checked for its
            // own sake and the diagnosis is left to the points around it.
            for (const [m, c] of [[3, 4], [-2, 9], [6.5, 1.25]]) {
              const rows = ctx.kernel(xs, ys, m, c);
              let sumMx = 0;
              let sumC = 0;
              for (let i = 0; i < 64; i++) {
                sumMx += rows[0][i];
                sumC += rows[1][i];
              }
              const got = [(2 * sumMx) / 4096, (2 * sumC) / 4096];
              const expected = trueGradient(m, c);
              const hint = gradientPairHint(got, m, c, 3e-3);
              for (const component of [0, 1]) {
                ctx.assertClose(got[component], expected[component], 3e-3, hint ||
                  `${component === 0 ? '∂L/∂m' : '∂L/∂c'} at m = ${m}, c = ${c}`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'walk-downhill',
      title: 'Take the Step',
      intro: `<p>The whole algorithm, now: stand somewhere, ask which way is up, move the other
        way, repeat.</p>
<pre><code>m ← m − η · ∂L/∂m
c ← c − η · ∂L/∂c</code></pre>
        <p>η is the <strong>learning rate</strong> — how far you move per step. Both kernels
        below are already written (task 1's loss, task 2's gradient), and the driving loop lives
        in JavaScript, exactly as the halving ladder's driver does: the arithmetic that touches
        all 4,096 points stays on the GPU, and the two numbers that say where you are stay on
        the host.</p>
        <p>Sixty steps at <code>η = 0.1</code> from <code>(0, 0)</code> land on
        <code>(3, 4)</code> to five decimal places, and the loss falls
        <code>25.5 → 0.79 → 0.503 → 0.5</code> along the way. Watch where it stops:
        <strong>0.5</strong>, not zero. The scatter in this data is real, and no straight line
        can explain it away — the optimum is where the loss stops falling, not where it
        vanishes.</p>`,
      goal: `<strong>Goal:</strong> write the two lines that move <code>m</code> and
        <code>c</code> one step downhill, and land on <code>y = 3x + 4</code> with a final loss
        of <code>0.5</code>.`,
      requirements: [
        'Step <em>against</em> the gradient — subtract, do not add',
        'Scale each step by <code>rate</code>: <code>m = m - rate * g[0]</code>',
        'Use the averaged gradient <code>gradientAt()</code> returns, not a raw sum',
        'The logged final slope and intercept should read <code>3</code> and <code>4</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — which way is downhill?',
          body: `<p>The gradient points in the direction the loss <em>increases</em> fastest.
            At <code>(0, 0)</code> it is <code>(−6, −8)</code>, so downhill is
            <code>(+6, +8)</code>, and <code>m − 0.1·(−6) = 0.6</code> is the first step. That
            minus sign is the whole difference between fitting and exploding.</p>`,
        },
        {
          title: 'Hint 2 — the two lines',
          body: `<pre><code>m = m - rate * g[0];
c = c - rate * g[1];</code></pre>
<p>Both parameters move on the <em>same</em> gradient — the one measured before either of
            them changed.</p>`,
        },
      ],
      transfer: `Host-driven, device-computed is how a real optimiser runs: a Python training
        loop issues CUDA kernels one step at a time, a WebGPU trainer records one dispatch per
        step. Parameters are small and data is not, so the parameters live where the control
        flow is — and the 60 round trips you just paid are exactly the cost real frameworks
        fight by fusing whole steps into one launch.`,
      starterCode: `// Both kernels are already written. The algorithm is yours.
const gpu = new GPU({ mode });

const gradPartials = gpu.createKernel(function (xs, ys, m, c) {
  let sm = 0;
  let sc = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    const at = i * this.constants.threads + this.thread.x;
    const r = m * xs[at] + c - ys[at];
    sm += r * xs[at];
    sc += r;
  }
  if (this.thread.y === 0) {
    return sm;
  }
  return sc;
}, { output: [64, 2], constants: { threads: 64, chunk: 64 } });

const lossPartials = gpu.createKernel(function (xs, ys, m, c) {
  let sum = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    const at = i * this.constants.threads + this.thread.x;
    const r = m * xs[at] + c - ys[at];
    sum += r * r;
  }
  return sum;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

function gradientAt(m, c) {
  const rows = gradPartials(xs, ys, m, c);
  let sumMx = 0;
  let sumC = 0;
  for (let i = 0; i < 64; i++) {
    sumMx += rows[0][i];
    sumC += rows[1][i];
  }
  return [(2 * sumMx) / 4096, (2 * sumC) / 4096];
}

function lossAt(m, c) {
  const partials = lossPartials(xs, ys, m, c);
  let total = 0;
  for (let i = 0; i < 64; i++) total += partials[i];
  return total / 4096;
}

const rate = 0.1;
let m = 0;
let c = 0;

for (let step = 1; step <= 60; step++) {
  const g = gradientAt(m, c);
  // TODO: move m and c one step DOWNHILL.
  // g[0] is dL/dm and g[1] is dL/dc — both point UPHILL.

  if (step % 10 === 0) console.log('step', step, '· loss', lossAt(m, c));
}

console.log('fitted slope:', m);
console.log('fitted intercept:', c);
console.log('final loss:', lossAt(m, c));
`,
      solutionCode: `// Both kernels are already written. The algorithm is yours.
const gpu = new GPU({ mode });

const gradPartials = gpu.createKernel(function (xs, ys, m, c) {
  let sm = 0;
  let sc = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    const at = i * this.constants.threads + this.thread.x;
    const r = m * xs[at] + c - ys[at];
    sm += r * xs[at];
    sc += r;
  }
  if (this.thread.y === 0) {
    return sm;
  }
  return sc;
}, { output: [64, 2], constants: { threads: 64, chunk: 64 } });

const lossPartials = gpu.createKernel(function (xs, ys, m, c) {
  let sum = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    const at = i * this.constants.threads + this.thread.x;
    const r = m * xs[at] + c - ys[at];
    sum += r * r;
  }
  return sum;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

function gradientAt(m, c) {
  const rows = gradPartials(xs, ys, m, c);
  let sumMx = 0;
  let sumC = 0;
  for (let i = 0; i < 64; i++) {
    sumMx += rows[0][i];
    sumC += rows[1][i];
  }
  return [(2 * sumMx) / 4096, (2 * sumC) / 4096];
}

function lossAt(m, c) {
  const partials = lossPartials(xs, ys, m, c);
  let total = 0;
  for (let i = 0; i < 64; i++) total += partials[i];
  return total / 4096;
}

const rate = 0.1;
let m = 0;
let c = 0;

for (let step = 1; step <= 60; step++) {
  const g = gradientAt(m, c);
  m = m - rate * g[0];
  c = c - rate * g[1];

  if (step % 10 === 0) console.log('step', step, '· loss', lossAt(m, c));
}

console.log('fitted slope:', m);
console.log('fitted intercept:', c);
console.log('final loss:', lossAt(m, c));
`,
      inputs: () => makeFit(4096),
      publicTests: [
        {
          name: 'both kernels are intact — gradient <code>(−6, −8)</code>, loss <code>25.5</code>',
          run: async ctx => {
            const grad = findByOutput(ctx, [64, 2]);
            const loss = findByOutput(ctx, [64]);
            ctx.assert(grad, 'no kernel with output [64, 2] found — keep the gradient kernel');
            ctx.assert(loss, 'no kernel with output [64] found — keep the loss kernel');
            const { xs, ys } = makeFit(4096);
            const rows = grad(xs, ys, 0, 0);
            let sumMx = 0;
            let sumC = 0;
            for (let i = 0; i < 64; i++) {
              sumMx += rows[0][i];
              sumC += rows[1][i];
            }
            ctx.assertClose((2 * sumMx) / 4096, -6, 2e-3, '∂L/∂m at m = 0, c = 0');
            ctx.assertClose((2 * sumC) / 4096, -8, 2e-3, '∂L/∂c at m = 0, c = 0');
            const partials = loss(xs, ys, 0, 0);
            let total = 0;
            for (let i = 0; i < 64; i++) total += partials[i];
            ctx.assertClose(total / 4096, 25.5, 2e-3, 'the loss at m = 0, c = 0');
          },
        },
        {
          name: 'the fitted line <code>y = 3x + 4</code> is logged',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            const hint = runawayHint(ctx.logs, nums);
            ctx.assert(
              nums.some(v => Math.abs(v - 3) <= 1e-3),
              hint || 'log the fitted slope — expected to see 3 in the console output'
            );
            ctx.assert(
              nums.some(v => Math.abs(v - 4) <= 1e-3),
              hint || 'log the fitted intercept — expected to see 4 in the console output'
            );
          },
        },
        {
          name: 'the final loss <code>0.5</code> is logged',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            const hint = runawayHint(ctx.logs, nums);
            ctx.assert(
              nums.some(v => Math.abs(v - 0.5) <= 1e-3),
              hint ||
                'log the final loss — expected to see 0.5 in the console output (25.5 means nothing moved)'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Drive the learner's own gradient kernel through the same descent,
            // from a start their code never visits.
            const grad = findByOutput(ctx, [64, 2]);
            ctx.assert(grad, 'expected a kernel with output [64, 2]');
            const { xs, ys } = makeFit(4096);
            const [m, c] = driveDescent(grad, xs, ys, 0.1, 60, 10, -5);
            ctx.assertClose(m, M_STAR, 1e-3, 'slope after 60 steps from (10, −5)');
            ctx.assertClose(c, C_STAR, 1e-3, 'intercept after 60 steps from (10, −5)');
            // One step from (0, 0) at rate 0.1 must be exactly (0.6, 0.8).
            const [m1, c1] = driveDescent(grad, xs, ys, 0.1, 1, 0, 0);
            ctx.assertClose(m1, 0.6, 1e-4, 'slope after exactly one step from (0, 0)');
            ctx.assertClose(c1, 0.8, 1e-4, 'intercept after exactly one step from (0, 0)');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The run must have walked down, not up and not off the edge.
            const nums = loggedNumbers(ctx.logs);
            const hint = runawayHint(ctx.logs, nums);
            ctx.assert(!hint, hint || 'the descent ran away');
            // …and the loss kernel it watched itself with must be right away
            // from the origin too, not only at (0, 0).
            const loss = findByOutput(ctx, [64]);
            ctx.assert(loss, 'expected a kernel with output [64]');
            const { xs, ys } = makeFit(4096);
            for (const [m, c] of [[3, 4], [2.5, 4.5], [-1, 1]]) {
              const partials = loss(xs, ys, m, c);
              let total = 0;
              for (let i = 0; i < 64; i++) total += partials[i];
              ctx.assertClose(total / 4096, trueLoss(m, c), 2e-3,
                `the loss at m = ${m}, c = ${c}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'sweep-the-rate',
      title: 'How Big a Step? Ask 256 at Once',
      intro: `<p>Where did <code>η = 0.1</code> come from? Nowhere. It was a guess, and guesses
        are where gradient descent goes wrong: too small and you never arrive, too large and you
        do not merely overshoot — you <strong>diverge</strong>, because the step lands you
        further from the minimum than you started and the next one is longer still.</p>
        <p>There is an exact limit. One step turns the error <code>e = θ − θ*</code> into
        <code>(I − η·H)·e</code>, where <code>H</code> is the matrix of second derivatives, so
        the walk shrinks the error only while <code>η &lt; 2 / λ_max</code>. On this data
        <code>H = 2I</code>, which makes <code>λ_max = 2</code> and the limit exactly
        <code>η &lt; 1</code> — and <code>η = 1/2</code> the rate that lands on the answer in a
        single step. (A step size with a hard ceiling is not a quirk of optimisation. Explicit
        time-stepping of a diffusion has one too, for the same eigenvalue reason.)</p>
        <p>Believing that is optional; measuring it costs one launch. Give each of 256 threads
        its own learning rate, let each run a complete 80-step descent on its own copy of
        <code>m</code> and <code>c</code>, and read all 256 final losses back at once. This is
        the axis flip that makes hyperparameter search a GPU problem: earlier tasks were
        parallel <em>over data points</em>, and here the data is small (64 points, the identical
        loss surface) while the <em>runs</em> are many — so the threads go one per run, and each
        one walks its own slice of the data by itself.</p>`,
      goal: `<strong>Goal:</strong> give each thread its own learning rate and its own descent,
        and return the final loss — so that rates <code>1/8</code> and <code>1/2</code> come
        back at <code>0.5</code>, rate <code>1</code> sits at <code>25.5</code> forever, and
        rate <code>3/2</code> is off the scale.`,
      requirements: [
        'Each thread reads exactly one rate: <code>rates[this.thread.x]</code>',
        'The step is scaled by that rate <em>and</em> by <code>this.constants.gradScale</code> (the 2/n)',
        'Return the mean squared residual after the last step, not <code>m</code> or <code>c</code>',
        'Log the losses at rates <code>1/8</code>, <code>1/2</code>, <code>1</code> and <code>3/2</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — one thread, one rate',
          body: `<p>This is the same "which element is mine?" question every kernel asks, and
            the answer is the same: <code>const rate = rates[this.thread.x];</code>. Leave it as
            <code>rates[0]</code> and all 256 threads run the identical experiment.</p>`,
        },
        {
          title: 'Hint 2 — the step',
          body: `<pre><code>m = m - rate * this.constants.gradScale * gm;
c = c - rate * this.constants.gradScale * gc;</code></pre>
<p><code>gradScale</code> is <code>2 / 64</code>, precomputed in JavaScript — two integer
            constants would divide as integers in the shader and give you zero.</p>`,
        },
        {
          title: 'Hint 3 — reading the result',
          body: `<p>The final losses tell a story in three parts: a band in the middle that
            reached <code>0.5</code>, a few rates at the bottom still crawling towards it, and
            everything past <code>η = 1</code> heading for infinity. The one at exactly
            <code>η = 1</code> is the strangest: it neither converges nor explodes, because
            <code>|1 − η·λ| = 1</code> means the error flips sign and keeps its size.</p>`,
        },
      ],
      transfer: `Sweeping hyperparameters one per thread is the small, cheap version of what a
        tuner does with whole models across a cluster — and when the per-run state fits in
        registers there is no reason to leave the GPU between runs at all. The design decision
        is which axis to parallelise: over data when the data is big, over runs when the runs
        are many. Same kernel language, opposite layout.`,
      starterCode: `// 256 threads. 256 learning rates. One complete descent each.
// The dataset is 64 points from the same line — the same loss surface,
// L(m, c) = (m − 3)² + (c − 4)² + 0.5, at a size that fits in a loop.
const gpu = new GPU({ mode });

const sweep = gpu.createKernel(function (xs, ys, rates) {
  // TODO: this thread owns exactly ONE of the 256 learning rates.
  const rate = rates[0];

  let m = 0;
  let c = 0;
  for (let s = 0; s < this.constants.steps; s++) {
    let gm = 0;
    let gc = 0;
    for (let i = 0; i < this.constants.points; i++) {
      const r = m * xs[i] + c - ys[i];
      gm += r * xs[i];
      gc += r;
    }
    // TODO: the step is missing this thread's rate.
    m = m - this.constants.gradScale * gm;
    c = c - this.constants.gradScale * gc;
  }

  let loss = 0;
  for (let i = 0; i < this.constants.points; i++) {
    const r = m * xs[i] + c - ys[i];
    loss += r * r;
  }
  return loss * this.constants.invPoints;
}, {
  output: [256],
  constants: { points: 64, steps: 80, gradScale: 2 / 64, invPoints: 1 / 64 },
});

const finalLoss = sweep(xs, ys, rates);

let converged = 0;
for (let k = 0; k < finalLoss.length; k++) {
  if (finalLoss[k] < 1) converged++;
}

console.log('rate 1/8 → loss', finalLoss[15]);
console.log('rate 1/2 → loss', finalLoss[63]);
console.log('rate 1   → loss', finalLoss[127]);
console.log('rate 3/2 → loss', finalLoss[191]);
console.log('rates that got under loss 1:', converged, 'of 256');
`,
      solutionCode: `// 256 threads. 256 learning rates. One complete descent each.
// The dataset is 64 points from the same line — the same loss surface,
// L(m, c) = (m − 3)² + (c − 4)² + 0.5, at a size that fits in a loop.
const gpu = new GPU({ mode });

const sweep = gpu.createKernel(function (xs, ys, rates) {
  const rate = rates[this.thread.x];

  let m = 0;
  let c = 0;
  for (let s = 0; s < this.constants.steps; s++) {
    let gm = 0;
    let gc = 0;
    for (let i = 0; i < this.constants.points; i++) {
      const r = m * xs[i] + c - ys[i];
      gm += r * xs[i];
      gc += r;
    }
    m = m - rate * this.constants.gradScale * gm;
    c = c - rate * this.constants.gradScale * gc;
  }

  let loss = 0;
  for (let i = 0; i < this.constants.points; i++) {
    const r = m * xs[i] + c - ys[i];
    loss += r * r;
  }
  return loss * this.constants.invPoints;
}, {
  output: [256],
  constants: { points: 64, steps: 80, gradScale: 2 / 64, invPoints: 1 / 64 },
});

const finalLoss = sweep(xs, ys, rates);

let converged = 0;
for (let k = 0; k < finalLoss.length; k++) {
  if (finalLoss[k] < 1) converged++;
}

console.log('rate 1/8 → loss', finalLoss[15]);
console.log('rate 1/2 → loss', finalLoss[63]);
console.log('rate 1   → loss', finalLoss[127]);
console.log('rate 3/2 → loss', finalLoss[191]);
console.log('rates that got under loss 1:', converged, 'of 256');
`,
      inputs: () => ({ ...makeFit(64), rates: makeRates() }),
      publicTests: [
        {
          name: '256 final losses, one per rate',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const { xs, ys } = makeFit(64);
            const rates = makeRates();
            const out = ctx.kernel(xs, ys, rates);
            ctx.assert(out && out.length === 256, `expected 256 final losses, got ${out && out.length}`);
            const identical = out.every(v => Math.abs(v - out[0]) < 1e-9);
            ctx.assert(
              !identical,
              `all 256 threads returned the same loss (${out[0]}) — each thread has to read its own rate: rates[this.thread.x]`
            );
          },
        },
        {
          name: 'rates from <code>1/8</code> to <code>7/8</code> all reach the floor <code>0.5</code>',
          run: async ctx => {
            const { xs, ys } = makeFit(64);
            const rates = makeRates();
            const out = ctx.kernel(xs, ys, rates);
            const hint = sweepShapeHint(out);
            // measured worst deviation across this band: 1.2e-7 (float32),
            // 2.2e-16 (float64) — asserted three orders of magnitude looser
            for (let k = 0; k < 256; k++) {
              if (rates[k] < 0.125 || rates[k] > 0.875) continue;
              ctx.assertClose(out[k], L_STAR, 1e-4, hint || `final loss at rate ${rates[k]}`);
            }
          },
        },
        {
          name: 'past <code>η = 1</code> it detonates, and at exactly 1 it just orbits',
          run: async ctx => {
            const { xs, ys } = makeFit(64);
            const rates = makeRates();
            const out = ctx.kernel(xs, ys, rates);
            const hint = sweepShapeHint(out);
            // rate exactly 1: |1 − η·λ| = 1, so the error flips sign forever and
            // the loss stays at its starting value, L(0, 0) = 25.5
            ctx.assertClose(out[127], 25.5, 0.05, hint ||
              'the loss at rate 1 should sit exactly where it started, 25.5 — the error flips sign each step without shrinking');
            for (let k = 0; k < 256; k++) {
              if (rates[k] < 1.25) continue;
              ctx.assert(
                !(out[k] < 1e6),
                hint || `rate ${rates[k]} is past the stability limit of 1 and should have blown up, but its loss came back as ${out[k]}`
              );
            }
            // the smallest rate has not arrived yet after 80 steps
            ctx.assert(
              out[0] > 1.5 && out[0] < 4,
              hint || `rate ${rates[0]} is far too small to converge in 80 steps — expected a final loss around 2.5, got ${out[0]}`
            );
          },
        },
        {
          name: 'the four sampled rates are logged',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - 0.5) <= 1e-3),
              'log the loss at rate 1/2 — expected to see 0.5 in the console output'
            );
            ctx.assert(
              nums.some(v => Math.abs(v - 25.5) <= 0.05),
              'log the loss at rate 1 — expected to see 25.5 in the console output'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { xs, ys } = makeFit(64);
            const rates = makeRates();
            const out = ctx.kernel(xs, ys, rates);
            const hint = sweepShapeHint(out);
            // The shape of the whole sweep: a converged band, a blown-up tail,
            // and both counts identical in float32 and float64 (125 / 124).
            let converged = 0;
            let blown = 0;
            for (let k = 0; k < 256; k++) {
              if (out[k] < 1) converged++;
              if (!(out[k] < 1e6)) blown++;
            }
            ctx.assert(
              Math.abs(converged - 125) <= 2,
              hint || `expected about 125 of the 256 rates to finish under loss 1, got ${converged}`
            );
            ctx.assert(
              Math.abs(blown - 124) <= 2,
              hint || `expected about 124 of the 256 rates to blow past loss 1e6, got ${blown}`
            );
            // η = 1/2 kills the error in a single step, so it is the fastest
            ctx.assertClose(out[63], L_STAR, 1e-4, hint || 'final loss at rate 1/2');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'many-starts',
      title: 'A Thousand Starts, Three Answers',
      intro: `<p>Everything so far worked because the loss was a bowl: one minimum, and every
        road leads to it. Most surfaces are not bowls. Here is one that is not — a single
        parameter <code>w</code>, and a slope written out by hand, the way every gradient in
        this module has been:</p>
<pre><code>f'(w) = w⁵ − 5w³ + 4w
      = w · (w² − 1) · (w² − 4)</code></pre>
        <p>Five roots: <code>−2, −1, 0, 1, 2</code>. Three are valleys (<code>−2</code>,
        <code>0</code>, <code>+2</code>) and two are ridges (<code>±1</code>). Gradient descent
        can see none of this. It reads the slope under its feet and steps, so
        <em>where it ends up is decided entirely by where it began</em> — and the borders
        between the three answers sit exactly at <code>w = ±1</code>.</p>
        <p>Which is a question with 1,024 answers and 1,024 threads to answer it: one start
        each, 200 steps each, one launch. The step size is fixed at <code>0.02</code>, well
        under the <code>2/24 ≈ 0.083</code> the curvature at <code>w = ±2</code> allows. Watch
        what comes back: <code>starts[300] = −1.0352</code> finishes at <code>−2</code> and
        <code>starts[320] = −0.9375</code> finishes at <code>0</code> — two runs a tenth of a
        unit apart, ending two whole units apart.</p>`,
      goal: `<strong>Goal:</strong> give each thread its own starting point, step it 200 times
        against the slope, and return where it stopped — <code>308</code> threads should land
        near <code>−2</code>, <code>409</code> near <code>0</code> and <code>307</code> near
        <code>+2</code>.`,
      requirements: [
        'Each thread reads its own start: <code>starts[this.thread.x]</code>',
        'The slope is <code>w⁵ − 5w³ + 4w</code> — write it out with plain multiplications',
        'Step against it: <code>w = w - this.constants.rate * slope</code>',
        'Count how many threads finished in each of the three valleys and log the three counts',
      ],
      hints: [
        {
          title: 'Hint 1 — the slope, spelled out',
          body: `<p>No <code>Math.pow</code> needed, and no autodiff either — the derivative of
            a polynomial is a polynomial:</p>
<pre><code>const slope = w * w * w * w * w - 5 * w * w * w + 4 * w;</code></pre>`,
        },
        {
          title: 'Hint 2 — the update is the same one',
          body: `<p>Identical to the line fit, with one parameter instead of two:
            <code>w = w - this.constants.rate * slope;</code>. That is the entire algorithm; the
            surface changed, not the method.</p>`,
        },
        {
          title: 'Hint 3 — counting the basins',
          body: `<p>Every thread finishes within a rounding error of <code>−2</code>,
            <code>0</code> or <code>+2</code>, so a two-way split on the returned value is
            enough:</p>
<pre><code>if (ends[k] &lt; -1) left++;
else if (ends[k] &lt; 1) middle++;
else right++;</code></pre>`,
        },
      ],
      transfer: `Random-restart optimisation is a working tool, not a curiosity — basin hopping,
        multi-start least squares, ensembles of annealing runs all do this, and every one of
        them is embarrassingly parallel, which is why CUDA and Metal implementations launch
        thousands of restarts at a time. The GPU does not make any single walk faster; it makes
        a thousand of them fit in one launch.`,
      starterCode: `// f'(w) = w⁵ − 5w³ + 4w. Three valleys at −2, 0, +2;
// two ridges at ±1. 1,024 starts, one per thread.
const gpu = new GPU({ mode });

const walk = gpu.createKernel(function (starts) {
  // TODO: this thread owns ONE starting point.
  let w = starts[0];

  for (let s = 0; s < this.constants.steps; s++) {
    // TODO: the slope of the surface at w is w⁵ − 5w³ + 4w.
    const slope = 0;
    w = w - this.constants.rate * slope;
  }
  return w;
}, {
  output: [1024],
  constants: { steps: 200, rate: 0.02 },
});

const ends = walk(starts);

let left = 0;
let middle = 0;
let right = 0;
for (let k = 0; k < ends.length; k++) {
  if (ends[k] < -1) left++;
  else if (ends[k] < 1) middle++;
  else right++;
}

console.log('landed near -2:', left, '| near 0:', middle, '| near +2:', right);
console.log('start', starts[300], '→', ends[300]);
console.log('start', starts[320], '→', ends[320]);
`,
      solutionCode: `// f'(w) = w⁵ − 5w³ + 4w. Three valleys at −2, 0, +2;
// two ridges at ±1. 1,024 starts, one per thread.
const gpu = new GPU({ mode });

const walk = gpu.createKernel(function (starts) {
  let w = starts[this.thread.x];

  for (let s = 0; s < this.constants.steps; s++) {
    const slope = w * w * w * w * w - 5 * w * w * w + 4 * w;
    w = w - this.constants.rate * slope;
  }
  return w;
}, {
  output: [1024],
  constants: { steps: 200, rate: 0.02 },
});

const ends = walk(starts);

let left = 0;
let middle = 0;
let right = 0;
for (let k = 0; k < ends.length; k++) {
  if (ends[k] < -1) left++;
  else if (ends[k] < 1) middle++;
  else right++;
}

console.log('landed near -2:', left, '| near 0:', middle, '| near +2:', right);
console.log('start', starts[300], '→', ends[300]);
console.log('start', starts[320], '→', ends[320]);
`,
      inputs: () => ({ starts: makeStarts() }),
      publicTests: [
        {
          name: '1,024 threads, 1,024 different answers to give',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const starts = makeStarts();
            const out = ctx.kernel(starts);
            ctx.assert(out && out.length === 1024, `expected 1024 results, got ${out && out.length}`);
            const hint = wellsHint(Array.from(out), starts);
            ctx.assert(!hint, hint || 'the walks did not settle');
          },
        },
        {
          name: 'every thread lands in the valley its start was already in',
          run: async ctx => {
            const starts = makeStarts();
            const out = ctx.kernel(starts);
            const hint = wellsHint(Array.from(out), starts);
            // measured worst |w − minimum| over all 1,024 threads: 2.0e−6, in
            // both float32 and float64
            for (let k = 0; k < 1024; k++) {
              ctx.assertClose(out[k], basinOf(starts[k]), 0.01, hint ||
                `the thread starting at w = ${starts[k].toFixed(4)} should finish at ${basinOf(starts[k])}`);
            }
          },
        },
        {
          name: 'the three basin counts <code>308 / 409 / 307</code> are logged',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            for (const want of [308, 409, 307]) {
              ctx.assert(
                nums.some(v => v === want),
                `log the basin counts — expected to see ${want} in the console output`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different grid the learner's code never sees, including starts
            // that sit deliberately close to a ridge on either side.
            const starts = [];
            for (let k = 0; k < 1024; k++) starts.push(-2.4 + k * (4.8 / 1024));
            const out = ctx.kernel(starts);
            const hint = wellsHint(Array.from(out), starts);
            const counts = { left: 0, middle: 0, right: 0 };
            for (let k = 0; k < 1024; k++) {
              ctx.assertClose(out[k], basinOf(starts[k]), 0.01, hint ||
                `the thread starting at w = ${starts[k].toFixed(4)} should finish at ${basinOf(starts[k])}`);
              if (out[k] < -1) counts.left++;
              else if (out[k] < 1) counts.middle++;
              else counts.right++;
            }
            ctx.assert(
              counts.left > 0 && counts.middle > 0 && counts.right > 0,
              `all three valleys should be reached, got ${JSON.stringify(counts)}`
            );
          },
        },
      ],
    },
  ],
};
