// Module: ODE Integrators — uuid 62f4a3ff-b240-4f1a-9f00-2d0f0d2e4dfc (short id 62f4a3ff).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. No legacyId: this module postdates the uuid switch.
//
// Six tasks: the time loop moves INSIDE the kernel → error measured against a
// closed form, and Euler's first-order exponent → midpoint's second → RK4's
// fourth → velocity Verlet, second order but symplectic → the long run where
// RK4 leaks energy and Verlet does not.
//
// The model problem is a unit spring, a = −x, all the way through. N-Body
// Gravity owns interesting forces AND the JavaScript tick loop; this module
// owns the STEP, so the force law is deliberately the dullest one there is and
// is never dwelt on. What it buys is an exact answer to compare against:
//   x(t) = x₀·cos t + v₀·sin t,  E = (x² + v²)/2 constant.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested numeric arrays as inputs, this.thread.* for indexing, and a
// kernel local must never share a name with a constant (gpujs/gpu.js#858 — it
// throws on the CPU backend only). Every task passes in CPU mode; the slowest
// kernel measured 14 ms on WebGL and 8 ms on the CPU backend, so no task needs
// budgetMs.
//
// TWO SIZING TRAPS, both measured rather than guessed:
//
// 1. loopMaxIterations is a HARD CAP, not a hint. A trip count that comes from
//    an argument (levelSteps[this.thread.y]) compiles to a fixed-length loop
//    with an early exit, and a level larger than the cap is silently truncated
//    — a wrong answer with no error. Every task's cap equals its largest level,
//    and every test that swaps the levels stays at or below it.
//
// 2. Each method needs its OWN window of step sizes. Too coarse and the
//    measurement is not reading the leading error term; too fine and it is
//    reading float32 rounding. Euler 50–400 steps, midpoint/Verlet 25–200,
//    RK4 only 5–20 over three levels — at Euler's finest setting RK4's error
//    would be ~1e-9, a thousand times below what float32 can resolve here.
//
// FLOAT DETERMINISM (tests compute in float64, the GL backend in float32).
// Measured on headless Chrome/SwiftShader against the float64 references
// below, worst case per assertion family:
//   • task 1 final x, per thread ........ 2.2e-7 relative
//   • row means, tasks 2 and 4 .......... 1.4e-6 / 7.9e-4 relative
//   • row means, tasks 3 and 5 .......... 4.5e-4 / 1.8e-3 relative (finest level)
//   • task 6 energy ratios .............. 6.3e-6 relative
// Assertions therefore use 5% on row means and 1% on task-6 energies — 25x to
// 1500x the measured drift — and lean on CONVERGENCE BEHAVIOUR (the error
// ratio is near 2 / 4 / 16; the energy fell / never fell) rather than on exact
// values wherever the behaviour is what the task is about. The near-miss
// integrators are 4x to 240x away from the right answer, so the wide margins
// cost no diagnostic power.

// ---- the model problem ----------------------------------------------------

const T_END = 5; // every convergence study integrates t = 0 … 5

// Refinement ladders. Each entry is a step count covering the SAME T_END, so
// the step size is T_END / n and consecutive levels halve it.
const EULER_LEVELS = [50, 100, 200, 400];
const SECOND_LEVELS = [25, 50, 100, 200]; // midpoint and velocity Verlet
const RK4_LEVELS = [5, 10, 20];

const TRAJECTORIES = 256; // trajectories per level in tasks 2–5

// A ring of oscillators: evenly spaced phases, amplitudes 0.6–1.4 (mean 1), all
// rounded to 4 dp so the numbers a learner sees in the inputs panel are the
// numbers the kernel gets. Deterministic — inputs() and the tests share it.
function startStates(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const startX = new Array(n);
  const startV = new Array(n);
  for (let i = 0; i < n; i++) {
    const angle = (2 * Math.PI * i) / n;
    const amplitude = Math.round((0.6 + 0.8 * rand()) * 1e4) / 1e4;
    startX[i] = Math.round(amplitude * Math.cos(angle) * 1e4) / 1e4;
    startV[i] = Math.round(-amplitude * Math.sin(angle) * 1e4) / 1e4;
  }
  return { startX, startV };
}

// The closed-form solution of x'' = −x. This is the whole reason the module
// can talk about error as a number instead of a feeling.
function exactX(x0, v0, t) {
  return x0 * Math.cos(t) + v0 * Math.sin(t);
}

// ---- float64 reference integrators ----------------------------------------
//
// Each mirrors its kernel STATEMENT FOR STATEMENT, in the same order, so the
// only difference left between a reference and a kernel is float64 vs float32.
// Each returns [x, v] after n steps of size dt.

function runEuler(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const a = -x;
    x = x + v * dt;
    v = v + a * dt;
  }
  return [x, v];
}

function runMidpoint(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const half = dt / 2;
    const a = -x;
    const midX = x + v * half;
    const midV = v + a * half;
    x = x + midV * dt;
    v = v + -midX * dt;
  }
  return [x, v];
}

function runRk4(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const half = dt / 2;
    const k1x = v;
    const k1v = -x;
    const k2x = v + k1v * half;
    const k2v = -(x + k1x * half);
    const k3x = v + k2v * half;
    const k3v = -(x + k2x * half);
    const k4x = v + k3v * dt;
    const k4v = -(x + k3x * dt);
    x = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    v = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
  }
  return [x, v];
}

function runVerlet(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const a = -x;
    x = x + v * dt + 0.5 * a * dt * dt;
    const aNext = -x;
    v = v + 0.5 * (a + aNext) * dt;
  }
  return [x, v];
}

// ---- near-miss integrators -------------------------------------------------
//
// Not alternatives — these are the specific wrong things a learner writes, kept
// here so a failing assert can name the mistake instead of reporting two
// numbers. Every one of them was run and its error curve measured; see the
// probe tables in the tests.

// Update v first, then move with the NEW v. A different method entirely —
// semi-implicit Euler, the one N-Body Gravity uses — and a better one.
function runSemiImplicit(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    v = v + -x * dt;
    x = x + v * dt;
  }
  return [x, v];
}

// Midpoint with the start-of-step acceleration kept for the velocity update.
function runMidStartV(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const half = dt / 2;
    const a = -x;
    const midV = v + a * half;
    x = x + midV * dt;
    v = v + a * dt;
  }
  return [x, v];
}

// Midpoint with the start-of-step velocity kept for the position update.
function runMidStartX(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const half = dt / 2;
    const midX = x + v * half;
    x = x + v * dt;
    v = v + -midX * dt;
  }
  return [x, v];
}

// A FULL trial step instead of a half one.
function runMidFullTrial(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const a = -x;
    const midX = x + v * dt;
    const midV = v + a * dt;
    x = x + midV * dt;
    v = v + -midX * dt;
  }
  return [x, v];
}

// RK4 averaging its four slopes evenly instead of weighting the middle pair.
function runRk4EqualWeights(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const half = dt / 2;
    const k1x = v;
    const k1v = -x;
    const k2x = v + k1v * half;
    const k2v = -(x + k1x * half);
    const k3x = v + k2v * half;
    const k3v = -(x + k2x * half);
    const k4x = v + k3v * dt;
    const k4v = -(x + k3x * dt);
    x = x + (dt / 4) * (k1x + k2x + k3x + k4x);
    v = v + (dt / 4) * (k1v + k2v + k3v + k4v);
  }
  return [x, v];
}

// RK4 with full steps where the middle two probes want half steps.
function runRk4FullSteps(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const k1x = v;
    const k1v = -x;
    const k2x = v + k1v * dt;
    const k2v = -(x + k1x * dt);
    const k3x = v + k2v * dt;
    const k3v = -(x + k2x * dt);
    const k4x = v + k3v * dt;
    const k4v = -(x + k3x * dt);
    x = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    v = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
  }
  return [x, v];
}

// Verlet without its ½·a·dt² position term (identical to runMidStartX).
function runVerletNoHalfTerm(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const a = -x;
    x = x + v * dt;
    const aNext = -x;
    v = v + 0.5 * (a + aNext) * dt;
  }
  return [x, v];
}

// Verlet whose velocity update never sees the acceleration at the NEW position.
function runVerletOldAccel(x, v, dt, n) {
  for (let s = 0; s < n; s++) {
    const a = -x;
    x = x + v * dt + 0.5 * a * dt * dt;
    v = v + a * dt;
  }
  return [x, v];
}

// ---- measurement helpers ---------------------------------------------------

function meanOf(row) {
  let total = 0;
  for (let i = 0; i < row.length; i++) total += row[i];
  return total / row.length;
}

// The reference version of what a task's kernel produces: for each refinement
// level, the mean |x_numeric(T_END) − x_exact(T_END)| over the ensemble.
function referenceMeans(run, starts, levels) {
  return levels.map(n => {
    const dt = T_END / n;
    let total = 0;
    for (let i = 0; i < starts.startX.length; i++) {
      const [x] = run(starts.startX[i], starts.startV[i], dt, n);
      total += Math.abs(x - exactX(starts.startX[i], starts.startV[i], T_END));
    }
    return total / starts.startX.length;
  });
}

// coarse ÷ fine at each halving: ≈2 for a first-order method, ≈4 for second,
// ≈16 for fourth. THE number this module is about.
function ratiosOf(means) {
  return means.slice(1).map((value, i) => means[i] / value);
}

function gridMeans(grid) {
  return Array.from(grid).map(row => meanOf(Array.from(row)));
}

// Energy as a multiple of its starting value, for a trajectory released from
// x = 0 with v = 1: E₀ = ½, so E/E₀ is exactly x² + v².
function energyRatio(run, dt, n) {
  const [x, v] = run(0, 1, dt, n);
  return x * x + v * v;
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

// ---- near-miss diagnosis ---------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so a trajectory where two integrators happen to
// agree stays silent, as do observations matching probes that disagree with
// each other. A wrong diagnosis is worse than none, and in this module the
// integrators genuinely do coincide here and there: over 1,024 trajectories the
// closest explicit-Euler and semi-implicit-Euler endpoints sit 1.7e-4 apart,
// well inside the tolerance, so the silence rule is load-bearing rather than
// decorative.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Tolerance that scales with the expected value (GPU float32 vs float64 ref).
function looseEps(scale, expected) {
  return scale * (1 + Math.abs(expected));
}

function closeish(ctx, got, expected, scale, message) {
  ctx.assertClose(got, expected, looseEps(scale, expected), message);
}

// A whole refinement curve compared against one candidate's curve: it may only
// speak if EVERY level matches the candidate and at least one level tells the
// candidate apart from the truth. One matching level is a coincidence; four are
// a diagnosis.
function diagnoseCurve(got, expected, relEps, probes) {
  const hits = probes
    .filter(([curve]) => {
      let differs = false;
      for (let i = 0; i < got.length; i++) {
        if (!(Math.abs(got[i] - curve[i]) <= relEps * (curve[i] + 1e-9))) return false;
        if (Math.abs(expected[i] - curve[i]) > relEps * (curve[i] + 1e-9)) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Every level came back the same → the step size never varied with the level.
function flatCurveHint(means) {
  const spread = Math.max(...means) - Math.min(...means);
  return spread <= 1e-6 * (1 + Math.abs(means[0]))
    ? 'all the levels came back identical — the step size has to come from this thread\'s ' +
      'row: n = levelSteps[this.thread.y], dt = this.constants.tEnd / n'
    : null;
}

// ---- task 6 helpers --------------------------------------------------------

const DT_LONG = 0.5;
const LONG_STEPS = 1024;

// Classify the two energy kernels by what they do, not by the order they were
// created in: one curve ends below its start, the other never dips below it.
function findEnergyKernels(ctx) {
  const rk4End = energyRatio(runRk4, DT_LONG, LONG_STEPS);
  const verletEnd = energyRatio(runVerlet, DT_LONG, LONG_STEPS);
  const found = { rk4: null, verlet: null, strays: [], rk4Count: 0 };
  for (const k of ctx.kernels) {
    let out;
    try {
      out = k(DT_LONG);
    } catch (e) {
      continue;
    }
    if (!out || out.length !== LONG_STEPS) continue;
    const end = out[LONG_STEPS - 1];
    if (Math.abs(end - rk4End) <= 5e-3) {
      found.rk4 = k;
      found.rk4Count++;
    } else if (Math.abs(end - verletEnd) <= 5e-3) found.verlet = k;
    else found.strays.push(end);
  }
  return found;
}

// Why is there no velocity-Verlet curve? Answered from what the OTHER kernel
// actually produced, and silent when nothing recognisable did.
function missingVerletHint(found) {
  if (found.verlet) return null;
  if (!found.strays.length) {
    return found.rk4Count > 1
      ? 'both kernels produced the same RK4 curve — the second one is the task: step it with ' +
        'velocity Verlet instead'
      : 'no kernel produced a 1,024-sample velocity-Verlet energy curve';
  }
  return strayEnergyHint(found.strays[0]) ||
    'no kernel produced the velocity-Verlet energy curve';
}

// What did the kernel that is NOT velocity Verlet actually integrate?
function strayEnergyHint(end) {
  if (!Number.isFinite(end)) {
    return 'that trajectory ran away to infinity — explicit Euler at dt = 0.5 gains energy every ' +
      'single step and is past 1e99 after 1,024 of them';
  }
  return diagnose(end, energyRatio(runVerlet, DT_LONG, LONG_STEPS), 5e-3, [
    [energyRatio(runSemiImplicit, DT_LONG, LONG_STEPS),
      'that is semi-implicit Euler — symplectic too, but only first order, and its energy swings ' +
      'far enough to dip below where it started'],
    [energyRatio(runMidpoint, DT_LONG, LONG_STEPS),
      'that is the midpoint method — second order like Verlet and not symplectic at all, so its ' +
      'energy grows without bound (about 8 million times its start after this run)'],
    [energyRatio(runEuler, DT_LONG, LONG_STEPS),
      'that is explicit Euler, which gains energy every step'],
    [energyRatio(runVerletNoHalfTerm, DT_LONG, LONG_STEPS),
      'the ½·a·dt² term is missing from the position update — without it the step is no longer ' +
      'the Verlet step and no longer symplectic'],
  ]);
}

// ---- the shared driver, shown in tasks 3–5 and written in task 2 ------------

const STUDY_DRIVER = `
// average the trajectories at each step size, then see what each halving bought
const means = [];
for (let level = 0; level < errors.length; level++) {
  let total = 0;
  for (let i = 0; i < errors[level].length; i++) total += errors[level][i];
  means.push(total / errors[level].length);
  console.log(levelSteps[level] + ' steps: mean error ' + means[level].toFixed(6));
}
for (let level = 1; level < means.length; level++) {
  console.log('halving dt divided the error by ' + (means[level - 1] / means[level]).toFixed(3));
}
`;

export default {
  uuid: '62f4a3ff-b240-4f1a-9f00-2d0f0d2e4dfc',
  version: 1,
  slug: 'ode-integrators',
  title: 'ODE Integrators',
  blurb: 'Euler, midpoint, RK4 and velocity Verlet — measured against a closed form, one thread per trajectory.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'euler-trajectory',
      title: 'One Thread, One Whole Trajectory',
      intro: `<p>A mass on a spring, in one line: the acceleration always points back at the
        origin, <code>a = −x</code>. That is the entire physics of this module. It is
        deliberately the dullest force there is, because the subject here is the
        <strong>clock</strong>, not the force — N-Body Gravity owns interesting forces.</p>
        <p>What the spring buys is that we already know the answer, exactly and forever:</p>
<pre><code>x(t) = x₀·cos t + v₀·sin t
E    = (x² + v²) / 2, constant</code></pre>
        <p>So every number a solver produces can be subtracted from the truth, and "how wrong
        is this?" stops being a feeling. And the GPU shape flips. N-Body's tick loop runs in
        JavaScript, one launch per step, because every body needs every other body's
        <em>new</em> position before the next tick — that is a barrier. These trajectories
        never speak to each other, so the loop moves <strong>inside</strong> the kernel: one
        launch, one thread, one entire trajectory, a thousand at a time.</p>`,
      goal: `<strong>Goal:</strong> fill in one explicit-Euler step, so each thread integrates
        its own oscillator for 200 steps and returns the final <code>x</code>.`,
      requirements: [
        'The time loop stays <em>inside</em> the kernel — <code>this.constants.steps</code> iterations, one thread per trajectory',
        'Explicit Euler commits to the start of the step: take <code>a = −x</code> first',
        'Advance <code>x</code> by the step-start <code>v · dt</code>, and <code>v</code> by <code>a · dt</code>',
        'Return this thread\'s final <code>x</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — what one step is',
          body: `<p>Three lines. The acceleration first, because <code>x</code> is about to
            change under it:</p>
<pre><code>const a = -x;
x = x + v * dt;
v = v + a * dt;</code></pre>`,
        },
        {
          title: 'Hint 2 — why that order',
          body: `<p>Explicit Euler uses <em>only</em> step-start values. Writing
            <code>v</code> first and then moving <code>x</code> with the <strong>new</strong>
            <code>v</code> is a different method — semi-implicit Euler, the one N-Body Gravity
            steps with. It is better, and it is task 5's business. Here we want the naive one,
            because its badness is the lesson.</p>`,
        },
        {
          title: 'Hint 3 — no arrays inside the loop',
          body: `<p><code>x</code> and <code>v</code> are plain <code>let</code> locals. The
            arrays are read once, before the loop, to start this thread off — after that the
            whole trajectory lives in two registers.</p>`,
        },
      ],
      transfer: `One thread per independent problem is how GPUs are pointed at ODEs everywhere:
        CUDA and HIP ensemble solvers give each thread one particle's trajectory, WebGPU compute
        does the same for particle systems, and the "batched" families in cuBLAS and cuSOLVER
        exist because thousands of small independent problems are the shape this hardware likes
        best. The kernel is the solver; the launch is the ensemble.`,
      starterCode: `// One thread owns one WHOLE trajectory: the time loop is inside the kernel.
const gpu = new GPU({ mode });
const STEPS = 200;
const DT = 0.025;                 // 200 × 0.025 = t = 5
const T_END = STEPS * DT;

const trajectory = gpu.createKernel(function (startX, startV, dt) {
  let x = startX[this.thread.x];
  let v = startV[this.thread.x];
  for (let s = 0; s < this.constants.steps; s++) {
    // TODO: one explicit-Euler step.
    //   a = -x first, then x moves by v * dt and v changes by a * dt —
    //   both using the values from the START of the step.
  }
  return x;
}, {
  output: [1024],
  constants: { steps: STEPS },
});

const finalX = trajectory(startX, startV, DT);

console.log('trajectory 0 ended at', finalX[0]);
console.log('the exact answer is  ', startX[0] * Math.cos(T_END) + startV[0] * Math.sin(T_END));
`,
      solutionCode: `// One thread owns one WHOLE trajectory: the time loop is inside the kernel.
const gpu = new GPU({ mode });
const STEPS = 200;
const DT = 0.025;                 // 200 × 0.025 = t = 5
const T_END = STEPS * DT;

const trajectory = gpu.createKernel(function (startX, startV, dt) {
  let x = startX[this.thread.x];
  let v = startV[this.thread.x];
  for (let s = 0; s < this.constants.steps; s++) {
    const a = -x;
    x = x + v * dt;
    v = v + a * dt;
  }
  return x;
}, {
  output: [1024],
  constants: { steps: STEPS },
});

const finalX = trajectory(startX, startV, DT);

console.log('trajectory 0 ended at', finalX[0]);
console.log('the exact answer is  ', startX[0] * Math.cos(T_END) + startV[0] * Math.sin(T_END));
`,
      inputs: utils => startStates(utils, 1024, 5107),
      publicTests: [
        {
          name: '1,024 trajectories come back, and they moved',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const starts = startStates(ctx.utils, 1024, 5107);
            const out = ctx.kernel(starts.startX, starts.startV, 0.025);
            ctx.assert(out && out.length === 1024, `expected 1,024 values, got ${out && out.length}`);
            let moved = 0;
            for (let i = 0; i < 1024; i++) {
              ctx.assert(Number.isFinite(out[i]), `trajectory ${i} came back ${out[i]}`);
              if (Math.abs(out[i] - starts.startX[i]) > 1e-4) moved++;
            }
            ctx.assert(moved > 1000, 'almost nothing moved — is the loop body still empty?');
          },
        },
        {
          name: 'each trajectory matches an explicit-Euler reference',
          run: async ctx => {
            const starts = startStates(ctx.utils, 1024, 5107);
            const out = ctx.kernel(starts.startX, starts.startV, 0.025);
            // These four are chosen where explicit Euler, semi-implicit Euler
            // and "never moved" are all far apart, so a probe can speak.
            for (const i of [0, 1, 200, 700, 1023]) {
              const x0 = starts.startX[i];
              const v0 = starts.startV[i];
              const expected = runEuler(x0, v0, 0.025, 200)[0];
              const eps = looseEps(1e-3, expected);
              const hint = diagnose(out[i], expected, eps, [
                [runSemiImplicit(x0, v0, 0.025, 200)[0],
                  'the velocity is being updated first and the position then moved with the NEW ' +
                  'velocity — that is semi-implicit Euler. Explicit Euler uses step-start values ' +
                  'for both: a = −x, then x += v · dt, then v += a · dt'],
                [runVerlet(x0, v0, 0.025, 200)[0],
                  'the velocity is being updated first and the position then moved with the NEW ' +
                  'velocity — that is semi-implicit Euler. Explicit Euler uses step-start values ' +
                  'for both: a = −x, then x += v · dt, then v += a · dt'],
                [x0, 'that value came back unchanged — the loop never touched x'],
                [exactX(x0, v0, 5), 'that is the exact answer, not Euler\'s — Euler is supposed to be wrong here'],
              ]);
              closeish(ctx, out[i], expected, 1e-3, hint || `trajectory ${i} after 200 steps`);
            }
          },
        },
        {
          name: 'Euler spirals outward — the ensemble ends ≈0.0405 from the truth',
          run: async ctx => {
            const starts = startStates(ctx.utils, 1024, 5107);
            const out = ctx.kernel(starts.startX, starts.startV, 0.025);
            let total = 0;
            for (let i = 0; i < 1024; i++) {
              total += Math.abs(out[i] - exactX(starts.startX[i], starts.startV[i], 5));
            }
            const got = total / 1024;
            const expected = 0.0404673; // measured, float64 and float32 agree to 8 dp
            ctx.assertClose(got, expected, 0.05 * expected,
              `the mean distance from the exact solution should be ≈${expected.toFixed(4)} ` +
              `(Euler's orbit swells 6.4% per 200 steps), got ${got.toFixed(4)}`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different step size: 200 steps of 0.01 reaches t = 2, so a
            // hardcoded dt or a hardcoded end time fails here.
            const starts = startStates(ctx.utils, 1024, 5107);
            const out = ctx.kernel(starts.startX, starts.startV, 0.01);
            for (let i = 0; i < 1024; i++) {
              const expected = runEuler(starts.startX[i], starts.startV[i], 0.01, 200)[0];
              const hint = diagnose(out[i], expected, looseEps(1e-3, expected), [
                [runEuler(starts.startX[i], starts.startV[i], 0.025, 200)[0],
                  'the step size is hardcoded — dt arrives as a kernel argument and this run uses a different one'],
              ]);
              closeish(ctx, out[i], expected, 1e-3, hint || `trajectory ${i} at dt = 0.01`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'step-size-scaling',
      title: 'Halve the Step, Halve the Error',
      intro: `<p>Euler is wrong. The useful question is <em>how</em> wrong, and what more steps
        buy you. Explicit Euler is a <strong>first-order</strong> method: its error at a fixed
        end time is proportional to <code>dt</code>. Halve the step, halve the error. Pay
        twice, get twice — which is a terrible exchange rate, and you can only find that out
        by measuring it.</p>
        <p>Measuring needs the same trajectories run at several step sizes, which is what the
        second output axis is for. <code>output: [256, 4]</code> launches a grid:
        <code>this.thread.x</code> picks the trajectory, <code>this.thread.y</code> picks the
        <strong>refinement level</strong> — 50, 100, 200 or 400 steps, every level covering the
        same <code>t = 0…5</code>, so <code>dt</code> is <code>5 / n</code>. One launch, the
        whole convergence study.</p>
        <p>Notice what that does to the loop: the trip count now differs from thread to thread.
        gpu.js compiles a non-constant bound as a fixed <code>loopMaxIterations</code> loop with
        an early exit, so the 50-step threads still march through all 400 iterations alongside
        their neighbours. That is not a gpu.js quirk — lockstep hardware behaves that way
        everywhere.</p>`,
      goal: `<strong>Goal:</strong> return each thread's absolute error against the closed form,
        then average each row in JavaScript and log what each halving of <code>dt</code> bought.`,
      requirements: [
        'Take the step count from the level: <code>levelSteps[this.thread.y]</code>, and <code>dt = this.constants.tEnd / n</code>',
        'Integrate <code>n</code> explicit-Euler steps, exactly as in the last task',
        'Return the <em>absolute</em> error against <code>startX·cos(tEnd) + startV·sin(tEnd)</code>',
        'In JavaScript: average each row, then <code>console.log</code> the three ratios <code>means[level − 1] / means[level]</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — which step size is mine?',
          body: `<p>The row index <em>is</em> the level:</p>
<pre><code>const n = levelSteps[this.thread.y];
const dt = this.constants.tEnd / n;</code></pre>
          <p>Every row ends at the same time; only the number of steps it took to get there
            differs.</p>`,
        },
        {
          title: 'Hint 2 — the truth to subtract',
          body: `<p><code>Math.cos</code> and <code>Math.sin</code> both work inside kernels:</p>
<pre><code>const exact = startX[this.thread.x] * Math.cos(this.constants.tEnd)
            + startV[this.thread.x] * Math.sin(this.constants.tEnd);
return Math.abs(x - exact);</code></pre>
          <p><code>Math.abs</code> matters: without it half the trajectories report a positive
            error and half a negative one, and the row average cancels to nearly nothing.</p>`,
        },
        {
          title: 'Hint 3 — the JavaScript half',
          body: `<p>A 2D kernel returns rows, so <code>errors[level]</code> is a whole row of
            trajectory errors:</p>
<pre><code>const means = [];
for (let level = 0; level &lt; errors.length; level++) {
  let total = 0;
  for (let i = 0; i &lt; errors[level].length; i++) total += errors[level][i];
  means.push(total / errors[level].length);
}
for (let level = 1; level &lt; means.length; level++) {
  console.log((means[level - 1] / means[level]).toFixed(3));
}</code></pre>`,
        },
      ],
      transfer: `Running an entire convergence study as one dispatch is the GPU-native form of a
        parameter sweep, and 2D launch grids are how every platform spells it — CUDA's
        <code>dim3</code> grid, WebGPU's dispatch dimensions, Metal's threadgroup grid. The
        divergence lesson travels too: a warp, a wavefront or a subgroup runs its loop until the
        <em>last</em> lane is finished, so a per-thread trip count is a hint about work, never a
        promise about time.`,
      starterCode: `// x = trajectory, y = refinement level. 256 oscillators × 4 step sizes,
// every level covering the same t = 0…5. One launch, one whole study.
const gpu = new GPU({ mode });

const errorOf = gpu.createKernel(function (startX, startV, levelSteps) {
  // TODO: this row's step count and step size
  const n = 1;
  const dt = 1;

  let x = startX[this.thread.x];
  let v = startV[this.thread.x];
  for (let s = 0; s < n; s++) {
    const a = -x;
    x = x + v * dt;
    v = v + a * dt;
  }

  // TODO: the exact answer is startX·cos(tEnd) + startV·sin(tEnd).
  // Return how far this trajectory ended up from it — an absolute distance.
  return x;
}, {
  output: [256, 4],
  constants: { tEnd: 5 },
  loopMaxIterations: 400,          // the largest level; a bigger one would be truncated
});

const errors = errorOf(startX, startV, levelSteps);
console.log('levels:', errors.length, '× trajectories:', errors[0].length);

// TODO: average each row into means[], log each one, then log the ratios
// means[level - 1] / means[level] — that number is the method's order.
`,
      solutionCode: `// x = trajectory, y = refinement level. 256 oscillators × 4 step sizes,
// every level covering the same t = 0…5. One launch, one whole study.
const gpu = new GPU({ mode });

const errorOf = gpu.createKernel(function (startX, startV, levelSteps) {
  const n = levelSteps[this.thread.y];
  const dt = this.constants.tEnd / n;

  let x = startX[this.thread.x];
  let v = startV[this.thread.x];
  for (let s = 0; s < n; s++) {
    const a = -x;
    x = x + v * dt;
    v = v + a * dt;
  }

  const exact = startX[this.thread.x] * Math.cos(this.constants.tEnd)
              + startV[this.thread.x] * Math.sin(this.constants.tEnd);
  return Math.abs(x - exact);
}, {
  output: [256, 4],
  constants: { tEnd: 5 },
  loopMaxIterations: 400,          // the largest level; a bigger one would be truncated
});

const errors = errorOf(startX, startV, levelSteps);
console.log('levels:', errors.length, '× trajectories:', errors[0].length);
${STUDY_DRIVER}`,
      inputs: utils => ({
        ...startStates(utils, TRAJECTORIES, 8231),
        levelSteps: EULER_LEVELS.slice(),
      }),
      publicTests: [
        {
          name: 'a 4 × 256 grid of non-negative errors',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const grid = ctx.kernel(starts.startX, starts.startV, EULER_LEVELS.slice());
            ctx.assert(grid && grid.length === 4, `expected 4 rows, one per level, got ${grid && grid.length}`);
            ctx.assert(grid[0] && grid[0].length === TRAJECTORIES,
              `each row should hold ${TRAJECTORIES} trajectory errors`);
            let negative = 0;
            for (let level = 0; level < 4; level++) {
              for (let i = 0; i < TRAJECTORIES; i++) {
                ctx.assert(Number.isFinite(grid[level][i]), `level ${level}, trajectory ${i} came back ${grid[level][i]}`);
                if (grid[level][i] < -1e-6) negative++;
              }
            }
            ctx.assert(negative === 0,
              `${negative} cells came back negative — an error is a distance, so wrap the ` +
              'difference in Math.abs (without it the row average cancels to almost nothing)');
          },
        },
        {
          name: 'row means match an explicit-Euler reference',
          run: async ctx => {
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const grid = ctx.kernel(starts.startX, starts.startV, EULER_LEVELS.slice());
            const got = gridMeans(grid);
            const expected = referenceMeans(runEuler, starts, EULER_LEVELS);
            // measured curves for the two integrators a learner may have written
            // instead — 5.8x apart at every level, so the probe is unambiguous
            const hint = flatCurveHint(got) || diagnoseCurve(got, expected, 0.05, [
              [referenceMeans(runSemiImplicit, starts, EULER_LEVELS),
                'those errors are about six times too small — the velocity is being updated ' +
                'first and the position then moved with the NEW velocity. That is semi-implicit ' +
                'Euler; explicit Euler uses step-start values for both'],
              [referenceMeans(runMidpoint, starts, EULER_LEVELS),
                'that is the midpoint method — accurate, but the point of this task is Euler'],
            ]);
            for (let level = 0; level < 4; level++) {
              ctx.assertClose(got[level], expected[level], 0.05 * expected[level],
                hint || `mean error at ${EULER_LEVELS[level]} steps`);
            }
          },
        },
        {
          name: 'first order: each halving divides the error by ≈2',
          run: async ctx => {
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const grid = ctx.kernel(starts.startX, starts.startV, EULER_LEVELS.slice());
            const ratios = ratiosOf(gridMeans(grid));
            ratios.forEach((ratio, i) => {
              ctx.assert(
                ratio > 1.85 && ratio < 2.4,
                `halving dt from ${T_END / EULER_LEVELS[i]} to ${T_END / EULER_LEVELS[i + 1]} ` +
                `divided the error by ${ratio.toFixed(3)} — a first-order method should give ≈2 ` +
                `(4 would mean second order, 16 fourth)`
              );
            });
          },
        },
        {
          name: 'the three ratios are logged',
          run: async ctx => {
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const expected = ratiosOf(referenceMeans(runEuler, starts, EULER_LEVELS));
            const nums = loggedNumbers(ctx.logs);
            expected.forEach((ratio, i) => {
              const eps = Math.max(0.02, 0.01 * ratio);
              const inverted = nums.some(v => Math.abs(v - 1 / ratio) <= 0.01);
              ctx.assert(
                nums.some(v => Math.abs(v - ratio) <= eps),
                inverted
                  ? `halving ${i + 1} was logged upside down — the ratio is coarse ÷ fine, ` +
                    `means[level - 1] / means[level], so it should read ≈${ratio.toFixed(3)}`
                  : `log what halving ${i + 1} bought — expected ≈${ratio.toFixed(3)} in the console output`
              );
            });
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different ladder (still ≤ the 400 loop cap): a hardcoded dt,
            // or a level count baked into the kernel, dies here.
            const alt = [40, 80, 160, 320];
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const got = gridMeans(ctx.kernel(starts.startX, starts.startV, alt));
            const expected = referenceMeans(runEuler, starts, alt);
            const hint = flatCurveHint(got) || diagnoseCurve(got, expected, 0.05, [
              [referenceMeans(runEuler, starts, EULER_LEVELS),
                'those are the errors for 50/100/200/400 steps — the step counts arrive in ' +
                'levelSteps and must be read from it, not hardcoded'],
            ]);
            for (let level = 0; level < 4; level++) {
              ctx.assertClose(got[level], expected[level], 0.05 * expected[level],
                hint || `mean error at ${alt[level]} steps`);
            }
            ratiosOf(got).forEach(ratio => {
              ctx.assert(ratio > 1.85 && ratio < 2.4,
                `error divided by ${ratio.toFixed(3)} per halving — first order should give ≈2`);
            });
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'midpoint',
      title: 'Look Before You Leap',
      intro: `<p>Euler's mistake is committing. It reads the slope at the start of the step and
        then pretends that slope holds all the way across. The <strong>midpoint method</strong>
        (RK2) makes the obvious repair: take a trial <em>half</em>-step, read the slope
        <em>there</em>, throw the trial away, and take the real full step using that better
        slope.</p>
        <p>Two force evaluations per step instead of one, and the error goes from first order to
        <strong>second</strong>: halve <code>dt</code> and the error drops by <em>four</em>.
        Same instrument as the last task; only the number at the bottom changes.</p>
        <p>The step sizes are coarser here — 25 to 200 steps, not 50 to 400 — and that is worth
        knowing rather than glossing. A convergence study only reads true inside a window. Too
        coarse and the leading error term is not yet dominant; too fine and the error sinks into
        the rounding noise of the float32 the WebGL backend computes in, and the ratio starts
        reporting arithmetic instead of mathematics.</p>`,
      goal: `<strong>Goal:</strong> replace the Euler step with a midpoint step, and watch the
        ratio go from 2 to 4.`,
      requirements: [
        'Trial half-step from the start: <code>midX = x + v·(dt/2)</code> and <code>midV = v + a·(dt/2)</code>, with <code>a = −x</code>',
        'Take the real step with the <em>midpoint</em> slopes: <code>x += midV·dt</code> and <code>v += (−midX)·dt</code>',
        'Both updates use midpoint values — leaving either one on the start-of-step slope drops you back to first order',
      ],
      hints: [
        {
          title: 'Hint 1 — what the trial step is for',
          body: `<p>Nothing about <code>midX</code> and <code>midV</code> survives the step. They
            exist only to answer one question — <em>what is the slope halfway across?</em> — and
            the answer is <code>midV</code> for the position and <code>−midX</code> for the
            velocity.</p>`,
        },
        {
          title: 'Hint 2 — which slope goes where',
          body: `<p>The state is <code>(x, v)</code> and its derivative is
            <code>(v, −x)</code>. So the midpoint <em>velocity</em> drives the position, and the
            midpoint <em>position</em> drives the velocity. Crossing those over is the single
            easiest way to get this wrong.</p>`,
        },
        {
          title: 'Hint 3 — the whole step',
          body: `<pre><code>const half = dt / 2;
const a = -x;
const midX = x + v * half;
const midV = v + a * half;
x = x + midV * dt;
v = v + -midX * dt;</code></pre>`,
        },
      ],
      transfer: `A multi-stage integrator has to hold several copies of the state at once, and on a
        GPU that is register pressure — the resource that decides how many threads a streaming
        multiprocessor can keep in flight. CUDA calls it occupancy, WebGPU and Metal have the same
        constraint under different names. It is a real reason production codes sometimes prefer a
        cheap scheme with a smaller step to an elegant one with a bigger footprint.`,
      starterCode: `// The same instrument, a better step. 256 oscillators × 4 step sizes.
const gpu = new GPU({ mode });

const errorOf = gpu.createKernel(function (startX, startV, levelSteps) {
  const n = levelSteps[this.thread.y];
  const dt = this.constants.tEnd / n;

  let x = startX[this.thread.x];
  let v = startV[this.thread.x];
  for (let s = 0; s < n; s++) {
    // TODO: one MIDPOINT step.
    //   half-step to (midX, midV), read the slope there,
    //   then take the full step with THAT slope.
    const a = -x;
    x = x + v * dt;
    v = v + a * dt;
  }

  const exact = startX[this.thread.x] * Math.cos(this.constants.tEnd)
              + startV[this.thread.x] * Math.sin(this.constants.tEnd);
  return Math.abs(x - exact);
}, {
  output: [256, 4],
  constants: { tEnd: 5 },
  loopMaxIterations: 200,
});

const errors = errorOf(startX, startV, levelSteps);
${STUDY_DRIVER}`,
      solutionCode: `// The same instrument, a better step. 256 oscillators × 4 step sizes.
const gpu = new GPU({ mode });

const errorOf = gpu.createKernel(function (startX, startV, levelSteps) {
  const n = levelSteps[this.thread.y];
  const dt = this.constants.tEnd / n;

  let x = startX[this.thread.x];
  let v = startV[this.thread.x];
  for (let s = 0; s < n; s++) {
    const half = dt / 2;
    const a = -x;
    const midX = x + v * half;
    const midV = v + a * half;
    x = x + midV * dt;
    v = v + -midX * dt;
  }

  const exact = startX[this.thread.x] * Math.cos(this.constants.tEnd)
              + startV[this.thread.x] * Math.sin(this.constants.tEnd);
  return Math.abs(x - exact);
}, {
  output: [256, 4],
  constants: { tEnd: 5 },
  loopMaxIterations: 200,
});

const errors = errorOf(startX, startV, levelSteps);
${STUDY_DRIVER}`,
      inputs: utils => ({
        ...startStates(utils, TRAJECTORIES, 8231),
        levelSteps: SECOND_LEVELS.slice(),
      }),
      publicTests: [
        {
          name: 'second order: each halving divides the error by ≈4',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const grid = ctx.kernel(starts.startX, starts.startV, SECOND_LEVELS.slice());
            ctx.assert(grid && grid.length === 4, `expected 4 rows, got ${grid && grid.length}`);
            const ratios = ratiosOf(gridMeans(grid));
            ratios.forEach((ratio, i) => {
              ctx.assert(
                ratio > 3.5 && ratio < 4.6,
                `halving dt from ${T_END / SECOND_LEVELS[i]} to ${T_END / SECOND_LEVELS[i + 1]} ` +
                `divided the error by ${ratio.toFixed(3)} — a second-order method gives ≈4. ` +
                `≈2 means the step is still only first-order accurate: either an update is using ` +
                `a start-of-step value, or the trial step is not half a step`
              );
            });
          },
        },
        {
          name: 'row means match a midpoint reference',
          run: async ctx => {
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const got = gridMeans(ctx.kernel(starts.startX, starts.startV, SECOND_LEVELS.slice()));
            const expected = referenceMeans(runMidpoint, starts, SECOND_LEVELS);
            // measured at 25 steps: midpoint 2.07e-2, start-slope-for-v 1.90e-1,
            // start-slope-for-x 1.77e-1, full trial step 2.55e-1, Euler 4.05e-1
            const hint = flatCurveHint(got) || diagnoseCurve(got, expected, 0.05, [
              [referenceMeans(runMidStartV, starts, SECOND_LEVELS),
                'the position uses the midpoint slope but the velocity is still using the ' +
                'start-of-step acceleration — v += (−midX)·dt, not a·dt'],
              [referenceMeans(runMidStartX, starts, SECOND_LEVELS),
                'the velocity uses the midpoint slope but the position is still moving on the ' +
                'start-of-step velocity — x += midV·dt, not v·dt'],
              [referenceMeans(runMidFullTrial, starts, SECOND_LEVELS),
                'the trial step is a whole step, not a half one — the slope has to be read at ' +
                'the MIDDLE, so the trial uses dt / 2'],
              [referenceMeans(runEuler, starts, SECOND_LEVELS),
                'that is still plain explicit Euler — the trial half-step never happened'],
              [referenceMeans(runSemiImplicit, starts, SECOND_LEVELS),
                'that is semi-implicit Euler — first order, and no midpoint anywhere in it'],
            ]);
            for (let level = 0; level < 4; level++) {
              ctx.assertClose(got[level], expected[level], 0.05 * expected[level],
                hint || `mean error at ${SECOND_LEVELS[level]} steps`);
            }
          },
        },
        {
          name: 'a real gain: 25 midpoint steps beat 400 Euler steps',
          run: async ctx => {
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const got = gridMeans(ctx.kernel(starts.startX, starts.startV, SECOND_LEVELS.slice()));
            // measured: midpoint at 25 steps 2.07e-2 vs Euler at 400 steps 2.01e-2 —
            // 50 force evaluations against 400, for the same accuracy
            const euler400 = referenceMeans(runEuler, starts, [400])[0];
            ctx.assert(
              got[0] < 4 * euler400,
              `at 25 steps the mean error is ${got[0].toFixed(5)}; 400 Euler steps manage ` +
              `${euler400.toFixed(5)}, so a correct midpoint should already be in that league ` +
              `with a sixteenth of the steps`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const alt = [20, 40, 80, 160];
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const got = gridMeans(ctx.kernel(starts.startX, starts.startV, alt));
            const expected = referenceMeans(runMidpoint, starts, alt);
            const hint = flatCurveHint(got) || diagnoseCurve(got, expected, 0.05, [
              [referenceMeans(runMidpoint, starts, SECOND_LEVELS),
                'those are the errors for 25/50/100/200 steps — the step counts arrive in ' +
                'levelSteps and must be read from it'],
            ]);
            for (let level = 0; level < 4; level++) {
              ctx.assertClose(got[level], expected[level], 0.05 * expected[level],
                hint || `mean error at ${alt[level]} steps`);
            }
            ratiosOf(got).forEach(ratio => {
              ctx.assert(ratio > 3.5 && ratio < 4.6,
                `error divided by ${ratio.toFixed(3)} per halving — second order should give ≈4`);
            });
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'rk4',
      title: 'Four Slopes, Weighted',
      intro: `<p>Classical RK4 samples the slope four times per step — once at the start, twice
        at the middle (the second one using the first one's estimate), once at the end — and
        combines them <code>(k₁ + 2·k₂ + 2·k₃ + k₄) / 6</code>. Four evaluations,
        <strong>fourth</strong> order: halving <code>dt</code> divides the error by
        <em>sixteen</em>.</p>
        <p>Which is why this task's step sizes look absurd. The coarsest level crosses
        <code>t = 0…5</code> in <strong>five</strong> steps — a full radian each — and it is
        still more accurate than 400 Euler steps. Run RK4 at Euler's finest setting instead and
        its error would be around 10⁻⁹, roughly a thousand times below what float32 can resolve
        next to a value of order 1; the study would measure rounding, not convergence. Three
        levels is what fits between "not yet asymptotic" and "already noise" — take one more
        halving and the ratio stops being a clean 16.</p>`,
      goal: `<strong>Goal:</strong> write the RK4 step and watch the ratio jump to ≈16.`,
      requirements: [
        'Four slope pairs: <code>k1</code> at the start, <code>k2</code> and <code>k3</code> at the midpoint, <code>k4</code> at the end',
        '<code>k2</code> and <code>k3</code> step out by <code>dt/2</code>; <code>k4</code> by the full <code>dt</code>',
        'Each probe builds on the <em>previous</em> one — <code>k3</code> from <code>k2</code>, <code>k4</code> from <code>k3</code>',
        'Combine with weights <code>1, 2, 2, 1</code> over <code>6</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the derivative of the state',
          body: `<p>The state is <code>(x, v)</code>, so every slope is a pair: the derivative
            of <code>x</code> is <code>v</code>, and the derivative of <code>v</code> is
            <code>−x</code>. That gives <code>k1x = v</code> and <code>k1v = −x</code>, and
            every later probe is the same rule evaluated at a shifted state.</p>`,
        },
        {
          title: 'Hint 2 — the middle two',
          body: `<p><code>k2</code> is the slope half a step along <code>k1</code>;
            <code>k3</code> is the slope half a step along <code>k2</code>:</p>
<pre><code>const k2x = v + k1v * half;
const k2v = -(x + k1x * half);
const k3x = v + k2v * half;
const k3v = -(x + k2x * half);</code></pre>
          <p>Building <code>k3</code> from <code>k1</code> again is the classic slip, and it
            costs you two whole orders.</p>`,
        },
        {
          title: 'Hint 3 — the combination',
          body: `<pre><code>x = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
v = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);</code></pre>
          <p>The middle pair carries twice the weight, and the six is what the weights sum
            to.</p>`,
        },
      ],
      transfer: `RK4 trades memory traffic for arithmetic: four evaluations of the derivative per
        step, all of them on values already sitting in registers. That is exactly the bargain GPUs
        reward — arithmetic is nearly free, and the force evaluation in a real simulation is
        usually memory-bound, so fewer, bigger, better steps often beat more cheap ones. It is the
        same arithmetic-intensity argument that drives kernel fusion in CUDA, WebGPU and Metal.`,
      starterCode: `// Only THREE levels here, and they are brutally coarse: 5, 10, 20 steps
// to cross t = 0…5. Fourth order needs big steps to stay visible above float32.
const gpu = new GPU({ mode });

const errorOf = gpu.createKernel(function (startX, startV, levelSteps) {
  const n = levelSteps[this.thread.y];
  const dt = this.constants.tEnd / n;

  let x = startX[this.thread.x];
  let v = startV[this.thread.x];
  for (let s = 0; s < n; s++) {
    const half = dt / 2;
    // TODO: four slope pairs — k1 at the start, k2 and k3 at the midpoint
    // (each from the previous one), k4 at the end — then combine them
    // with weights 1, 2, 2, 1 over 6.
    const k1x = v;
    const k1v = -x;
    x = x + k1x * dt;
    v = v + k1v * dt;
  }

  const exact = startX[this.thread.x] * Math.cos(this.constants.tEnd)
              + startV[this.thread.x] * Math.sin(this.constants.tEnd);
  return Math.abs(x - exact);
}, {
  output: [256, 3],
  constants: { tEnd: 5 },
  loopMaxIterations: 20,
});

const errors = errorOf(startX, startV, levelSteps);
${STUDY_DRIVER}`,
      solutionCode: `// Only THREE levels here, and they are brutally coarse: 5, 10, 20 steps
// to cross t = 0…5. Fourth order needs big steps to stay visible above float32.
const gpu = new GPU({ mode });

const errorOf = gpu.createKernel(function (startX, startV, levelSteps) {
  const n = levelSteps[this.thread.y];
  const dt = this.constants.tEnd / n;

  let x = startX[this.thread.x];
  let v = startV[this.thread.x];
  for (let s = 0; s < n; s++) {
    const half = dt / 2;
    const k1x = v;
    const k1v = -x;
    const k2x = v + k1v * half;
    const k2v = -(x + k1x * half);
    const k3x = v + k2v * half;
    const k3v = -(x + k2x * half);
    const k4x = v + k3v * dt;
    const k4v = -(x + k3x * dt);
    x = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    v = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
  }

  const exact = startX[this.thread.x] * Math.cos(this.constants.tEnd)
              + startV[this.thread.x] * Math.sin(this.constants.tEnd);
  return Math.abs(x - exact);
}, {
  output: [256, 3],
  constants: { tEnd: 5 },
  loopMaxIterations: 20,
});

const errors = errorOf(startX, startV, levelSteps);
${STUDY_DRIVER}`,
      inputs: utils => ({
        ...startStates(utils, TRAJECTORIES, 8231),
        levelSteps: RK4_LEVELS.slice(),
      }),
      publicTests: [
        {
          name: 'fourth order: each halving divides the error by ≈16',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const grid = ctx.kernel(starts.startX, starts.startV, RK4_LEVELS.slice());
            ctx.assert(grid && grid.length === 3, `expected 3 rows, got ${grid && grid.length}`);
            const ratios = ratiosOf(gridMeans(grid));
            ratios.forEach((ratio, i) => {
              ctx.assert(
                ratio > 13.5 && ratio < 18.5,
                `halving dt from ${T_END / RK4_LEVELS[i]} to ${T_END / RK4_LEVELS[i + 1]} ` +
                `divided the error by ${ratio.toFixed(2)} — fourth order gives ≈16, ≈4 means ` +
                `you are somewhere in second order, ≈2 in first`
              );
            });
          },
        },
        {
          name: 'row means match an RK4 reference',
          run: async ctx => {
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const got = gridMeans(ctx.kernel(starts.startX, starts.startV, RK4_LEVELS.slice()));
            const expected = referenceMeans(runRk4, starts, RK4_LEVELS);
            // measured at 5 steps: rk4 2.55e-2, equal weights 1.10e-1,
            // full steps 5.83e-1, midpoint 6.36e-1, verlet 1.37e-1
            const hint = flatCurveHint(got) || diagnoseCurve(got, expected, 0.06, [
              [referenceMeans(runRk4EqualWeights, starts, RK4_LEVELS),
                'the four slopes are being averaged evenly — the middle two carry weight 2, and ' +
                'the divisor is 6, not 4'],
              [referenceMeans(runRk4FullSteps, starts, RK4_LEVELS),
                'k2 and k3 are stepping out by a whole dt — they are MIDpoint probes, so they ' +
                'step out by dt / 2'],
              [referenceMeans(runMidpoint, starts, RK4_LEVELS),
                'that is the midpoint method from the last task — second order, and at these ' +
                'step sizes it is 25 times worse'],
              [referenceMeans(runEuler, starts, RK4_LEVELS),
                'that is still explicit Euler'],
            ]);
            for (let level = 0; level < 3; level++) {
              ctx.assertClose(got[level], expected[level], 0.05 * expected[level],
                hint || `mean error at ${RK4_LEVELS[level]} steps`);
            }
          },
        },
        {
          name: 'five RK4 steps beat four hundred Euler steps',
          run: async ctx => {
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const got = gridMeans(ctx.kernel(starts.startX, starts.startV, RK4_LEVELS.slice()));
            // measured: RK4 at 5 steps 2.55e-2 (20 force evaluations) against
            // Euler at 400 steps 2.01e-2 (400 evaluations) — the same accuracy
            // for a twentieth of the work, and RK4 pulls away from there
            const euler400 = referenceMeans(runEuler, starts, [400])[0];
            ctx.assert(
              got[2] < euler400 / 100,
              `20 RK4 steps end ${got[2].toExponential(2)} from the truth; 400 Euler steps ` +
              `manage ${euler400.toExponential(2)}. A correct RK4 should be at least a hundred ` +
              `times better than that, on a twentieth of the force evaluations`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const alt = [4, 8, 16];
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const got = gridMeans(ctx.kernel(starts.startX, starts.startV, alt));
            const expected = referenceMeans(runRk4, starts, alt);
            const hint = flatCurveHint(got) || diagnoseCurve(got, expected, 0.06, [
              [referenceMeans(runRk4, starts, RK4_LEVELS),
                'those are the errors for 5/10/20 steps — the step counts arrive in levelSteps ' +
                'and must be read from it'],
            ]);
            for (let level = 0; level < 3; level++) {
              ctx.assertClose(got[level], expected[level], 0.05 * expected[level],
                hint || `mean error at ${alt[level]} steps`);
            }
            ratiosOf(got).forEach(ratio => {
              ctx.assert(ratio > 13.5 && ratio < 18.5,
                `error divided by ${ratio.toFixed(2)} per halving — fourth order should give ≈16`);
            });
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'verlet',
      title: 'Velocity Verlet: the Symplectic Step',
      intro: `<p><strong>Velocity Verlet</strong> reads like Euler with one extra term. Move the
        position with a quadratic instead of a straight line, then update the velocity using the
        <em>average</em> of the accelerations at the two ends of the step:</p>
<pre><code>x ← x + v·dt + ½·a·dt²
v ← v + ½·(a + a_new)·dt        where a_new = −x_new</code></pre>
        <p>It is second order — the same exponent midpoint gave you, so the instrument you have
        been using <em>cannot tell them apart</em> by exponent alone. (It is about four times
        more accurate than midpoint at the same step size, and a real implementation carries
        <code>a_new</code> into the next step as its <code>a</code>, so it costs one force
        evaluation per step against midpoint's two. Here <code>a = −x</code> is free, so the code
        below just recomputes it.)</p>
        <p>What it has that midpoint does not is invisible in a single step and decisive over a
        million. The Verlet step is an exactly area-preserving map of the <code>(x, v)</code>
        plane — <strong>symplectic</strong> — and for this oscillator it conserves
        <code>(1 − dt²/4)·x² + v²</code> exactly, forever, in a way no accumulation of steps can
        erode. Measure the order here. The next task is where that sentence starts to matter.</p>`,
      goal: `<strong>Goal:</strong> write the velocity Verlet step. The ratio should be ≈4
        again — and the errors themselves about four times smaller than midpoint's.`,
      requirements: [
        'Position first, with the quadratic term: <code>x += v·dt + 0.5·a·dt·dt</code> where <code>a = −x</code> at the start of the step',
        'Then recompute the acceleration at the NEW position: <code>aNext = −x</code>',
        'Velocity from the average of the two: <code>v += 0.5·(a + aNext)·dt</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — order of operations',
          body: `<p><code>a</code> is captured before <code>x</code> moves; <code>aNext</code> is
            read after. The velocity update needs both, so it has to come last.</p>`,
        },
        {
          title: 'Hint 2 — the extra term',
          body: `<p><code>0.5 * a * dt * dt</code> is the constant-acceleration formula from
            first-year mechanics. Dropping it leaves a scheme that still looks plausible and is
            back to first order.</p>`,
        },
        {
          title: 'Hint 3 — the whole step',
          body: `<pre><code>const a = -x;
x = x + v * dt + 0.5 * a * dt * dt;
const aNext = -x;
v = v + 0.5 * (a + aNext) * dt;</code></pre>`,
        },
      ],
      transfer: `Velocity Verlet, not RK4, is what every production molecular-dynamics code on a GPU
        actually ships — GROMACS, LAMMPS, HOOMD-blue and OpenMM all step with Verlet or leapfrog on
        CUDA, HIP and Metal. It is not because they cannot afford RK4's four evaluations. It is
        because a run of a hundred million steps is judged on whether the energy stayed put, and
        that is a property of the <em>shape</em> of the update, not of its order.`,
      starterCode: `// Same instrument again. Same order as midpoint — and something else.
const gpu = new GPU({ mode });

const errorOf = gpu.createKernel(function (startX, startV, levelSteps) {
  const n = levelSteps[this.thread.y];
  const dt = this.constants.tEnd / n;

  let x = startX[this.thread.x];
  let v = startV[this.thread.x];
  for (let s = 0; s < n; s++) {
    // TODO: one velocity-Verlet step.
    //   x moves with v·dt AND the ½·a·dt² term,
    //   then v updates on the AVERAGE of a and the new a.
    const a = -x;
    x = x + v * dt;
    v = v + a * dt;
  }

  const exact = startX[this.thread.x] * Math.cos(this.constants.tEnd)
              + startV[this.thread.x] * Math.sin(this.constants.tEnd);
  return Math.abs(x - exact);
}, {
  output: [256, 4],
  constants: { tEnd: 5 },
  loopMaxIterations: 200,
});

const errors = errorOf(startX, startV, levelSteps);
${STUDY_DRIVER}`,
      solutionCode: `// Same instrument again. Same order as midpoint — and something else.
const gpu = new GPU({ mode });

const errorOf = gpu.createKernel(function (startX, startV, levelSteps) {
  const n = levelSteps[this.thread.y];
  const dt = this.constants.tEnd / n;

  let x = startX[this.thread.x];
  let v = startV[this.thread.x];
  for (let s = 0; s < n; s++) {
    const a = -x;
    x = x + v * dt + 0.5 * a * dt * dt;
    const aNext = -x;
    v = v + 0.5 * (a + aNext) * dt;
  }

  const exact = startX[this.thread.x] * Math.cos(this.constants.tEnd)
              + startV[this.thread.x] * Math.sin(this.constants.tEnd);
  return Math.abs(x - exact);
}, {
  output: [256, 4],
  constants: { tEnd: 5 },
  loopMaxIterations: 200,
});

const errors = errorOf(startX, startV, levelSteps);
${STUDY_DRIVER}`,
      inputs: utils => ({
        ...startStates(utils, TRAJECTORIES, 8231),
        levelSteps: SECOND_LEVELS.slice(),
      }),
      publicTests: [
        {
          name: 'second order again: each halving divides the error by ≈4',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const grid = ctx.kernel(starts.startX, starts.startV, SECOND_LEVELS.slice());
            ctx.assert(grid && grid.length === 4, `expected 4 rows, got ${grid && grid.length}`);
            ratiosOf(gridMeans(grid)).forEach((ratio, i) => {
              ctx.assert(
                ratio > 3.5 && ratio < 4.6,
                `halving dt from ${T_END / SECOND_LEVELS[i]} to ${T_END / SECOND_LEVELS[i + 1]} ` +
                `divided the error by ${ratio.toFixed(3)} — velocity Verlet is second order, so ≈4`
              );
            });
          },
        },
        {
          name: 'row means match a velocity-Verlet reference',
          run: async ctx => {
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const got = gridMeans(ctx.kernel(starts.startX, starts.startV, SECOND_LEVELS.slice()));
            const expected = referenceMeans(runVerlet, starts, SECOND_LEVELS);
            // measured at 25 steps: verlet 5.21e-3, midpoint 2.07e-2,
            // no ½·a·dt² term 1.77e-1, old acceleration only 1.90e-1,
            // semi-implicit 6.46e-2
            const hint = flatCurveHint(got) || diagnoseCurve(got, expected, 0.05, [
              [referenceMeans(runMidpoint, starts, SECOND_LEVELS),
                'that is the midpoint method from two tasks ago — second order too, so the ratio ' +
                'looks right, but the errors are four times larger than Verlet\'s'],
              [referenceMeans(runVerletNoHalfTerm, starts, SECOND_LEVELS),
                'the ½·a·dt² term is missing from the position update — without it the scheme ' +
                'drops back to first order'],
              [referenceMeans(runVerletOldAccel, starts, SECOND_LEVELS),
                'the velocity update never sees the acceleration at the NEW position — it needs ' +
                'the average ½·(a + aNext)·dt, and computing aNext is the whole point'],
              [referenceMeans(runSemiImplicit, starts, SECOND_LEVELS),
                'that is semi-implicit Euler — symplectic, but only first order'],
              [referenceMeans(runEuler, starts, SECOND_LEVELS),
                'that is still explicit Euler'],
            ]);
            for (let level = 0; level < 4; level++) {
              ctx.assertClose(got[level], expected[level], 0.05 * expected[level],
                hint || `mean error at ${SECOND_LEVELS[level]} steps`);
            }
          },
        },
        {
          name: 'and it is about four times sharper than midpoint at the same step size',
          run: async ctx => {
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const got = gridMeans(ctx.kernel(starts.startX, starts.startV, SECOND_LEVELS.slice()));
            const midpoint = referenceMeans(runMidpoint, starts, SECOND_LEVELS);
            // measured ratio midpoint/verlet ≈ 3.97 at every level
            for (let level = 0; level < 4; level++) {
              ctx.assert(
                got[level] < midpoint[level] / 2,
                `at ${SECOND_LEVELS[level]} steps the error is ${got[level].toExponential(2)}; ` +
                `midpoint manages ${midpoint[level].toExponential(2)} and velocity Verlet should ` +
                `be roughly four times better than that`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const alt = [20, 40, 80, 160];
            const starts = startStates(ctx.utils, TRAJECTORIES, 8231);
            const got = gridMeans(ctx.kernel(starts.startX, starts.startV, alt));
            const expected = referenceMeans(runVerlet, starts, alt);
            const hint = flatCurveHint(got) || diagnoseCurve(got, expected, 0.05, [
              [referenceMeans(runVerlet, starts, SECOND_LEVELS),
                'those are the errors for 25/50/100/200 steps — the step counts arrive in ' +
                'levelSteps and must be read from it'],
            ]);
            for (let level = 0; level < 4; level++) {
              ctx.assertClose(got[level], expected[level], 0.05 * expected[level],
                hint || `mean error at ${alt[level]} steps`);
            }
            ratiosOf(got).forEach(ratio => {
              ctx.assert(ratio > 3.5 && ratio < 4.6,
                `error divided by ${ratio.toFixed(3)} per halving — second order should give ≈4`);
            });
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'long-orbit',
      title: 'Payoff: RK4 Drifts, Verlet Does Not',
      intro: `<p>Eighty-one orbits at <code>dt = 0.5</code>, every trajectory released from
        <code>x = 0</code> with <code>v = 1</code>. That start makes the bookkeeping free:
        <code>E₀ = ½</code>, so the energy as a multiple of its starting value is exactly
        <code>x² + v²</code>.</p>
        <p>The launch is the interesting part. Thread <em>x</em> integrates
        <code>this.thread.x + 1</code> steps — one thread takes a single step, the last takes all
        1,024, and between them the 1,024 threads trace the <strong>whole energy history</strong>
        of the run. Every prefix computed independently, from scratch, at the same time. Nothing
        is shared, so nothing has to be sequenced.</p>
        <p>RK4 is the most accurate thing you have written: after eighty-one orbits its phase is
        0.24 radians behind the truth, where Verlet's has slipped nearly a whole orbit. Watch
        what its energy does anyway — and remember that midpoint, second order and perfectly
        respectable, ends this same run with about eight million times the energy it started
        with.</p>`,
      goal: `<strong>Goal:</strong> finish the velocity-Verlet energy kernel, then log each
        method's final energy and the smallest value Verlet ever reaches.`,
      requirements: [
        'Thread <em>x</em> runs <code>this.thread.x + 1</code> steps — the loop bound is the thread index',
        'Start every trajectory at <code>x = 0</code>, <code>v = 1</code> and step with velocity Verlet',
        'Return <code>x * x + v * v</code> — the energy as a multiple of where it started',
        '<code>console.log</code> both final energies and the minimum of the Verlet curve',
      ],
      hints: [
        {
          title: 'Hint 1 — the trip count',
          body: `<p><code>for (let s = 0; s &lt; this.thread.x + 1; s++)</code>. gpu.js compiles a
            bound it cannot fold at build time into a <code>loopMaxIterations</code> loop with an
            early exit, which is why the kernel declares
            <code>loopMaxIterations: 1024</code>.</p>`,
        },
        {
          title: 'Hint 2 — why x² + v² is the energy ratio',
          body: `<p><code>E = (x² + v²)/2</code> and every trajectory starts at
            <code>(0, 1)</code>, so <code>E₀ = ½</code> and <code>E/E₀ = x² + v²</code>. A value
            of 1 means the energy is exactly where it began.</p>`,
        },
        {
          title: 'Hint 3 — what to look for in JavaScript',
          body: `<p>Scan for the smallest value on each curve, and read the last one:</p>
<pre><code>let low = Infinity;
for (let i = 0; i &lt; verletCurve.length; i++) {
  if (verletCurve[i] &lt; low) low = verletCurve[i];
}</code></pre>
          <p>RK4's minimum <em>is</em> its final value — it never goes back up.</p>`,
        },
      ],
      transfer: `Two lessons travel. The launch shape — a thread per prefix, wildly unequal work,
        the wavefront running until its slowest lane finishes — is the load-imbalance problem every
        CUDA, ROCm, WebGPU and Metal programmer eventually has to lay out differently. And the
        result is why orbital-mechanics and molecular-dynamics codes on those platforms pick
        symplectic integrators: over a long run you are not choosing the method with the smallest
        error per step, you are choosing the one whose error does not have a direction.`,
      starterCode: `// 1,024 steps of dt = 0.5 — about eighty-one orbits.
// Thread x integrates x + 1 of them, so the 1,024 threads together
// trace the whole energy history, every prefix computed from scratch.
const gpu = new GPU({ mode });
const DT = 0.5;

// RK4, exactly as you wrote it two tasks ago, reporting energy instead of error.
const rk4Energy = gpu.createKernel(function (dt) {
  let x = 0;
  let v = 1;
  for (let s = 0; s < this.thread.x + 1; s++) {
    const half = dt / 2;
    const k1x = v;
    const k1v = -x;
    const k2x = v + k1v * half;
    const k2v = -(x + k1x * half);
    const k3x = v + k2v * half;
    const k3v = -(x + k2x * half);
    const k4x = v + k3v * dt;
    const k4v = -(x + k3x * dt);
    x = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    v = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
  }
  return x * x + v * v;
}, { output: [1024], loopMaxIterations: 1024 });

const verletEnergy = gpu.createKernel(function (dt) {
  let x = 0;
  let v = 1;
  // TODO: run this.thread.x + 1 velocity-Verlet steps and return
  // the energy as a multiple of its starting value: x * x + v * v.
  return 1;
}, { output: [1024], loopMaxIterations: 1024 });

const rk4Curve = rk4Energy(DT);
const verletCurve = verletEnergy(DT);

// TODO: log both final energies, and the smallest value the Verlet curve reaches.
`,
      solutionCode: `// 1,024 steps of dt = 0.5 — about eighty-one orbits.
// Thread x integrates x + 1 of them, so the 1,024 threads together
// trace the whole energy history, every prefix computed from scratch.
const gpu = new GPU({ mode });
const DT = 0.5;

// RK4, exactly as you wrote it two tasks ago, reporting energy instead of error.
const rk4Energy = gpu.createKernel(function (dt) {
  let x = 0;
  let v = 1;
  for (let s = 0; s < this.thread.x + 1; s++) {
    const half = dt / 2;
    const k1x = v;
    const k1v = -x;
    const k2x = v + k1v * half;
    const k2v = -(x + k1x * half);
    const k3x = v + k2v * half;
    const k3v = -(x + k2x * half);
    const k4x = v + k3v * dt;
    const k4v = -(x + k3x * dt);
    x = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x);
    v = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
  }
  return x * x + v * v;
}, { output: [1024], loopMaxIterations: 1024 });

const verletEnergy = gpu.createKernel(function (dt) {
  let x = 0;
  let v = 1;
  for (let s = 0; s < this.thread.x + 1; s++) {
    const a = -x;
    x = x + v * dt + 0.5 * a * dt * dt;
    const aNext = -x;
    v = v + 0.5 * (a + aNext) * dt;
  }
  return x * x + v * v;
}, { output: [1024], loopMaxIterations: 1024 });

const rk4Curve = rk4Energy(DT);
const verletCurve = verletEnergy(DT);

let verletLow = Infinity;
for (let i = 0; i < verletCurve.length; i++) {
  if (verletCurve[i] < verletLow) verletLow = verletCurve[i];
}

console.log('RK4    final energy:', rk4Curve[1023].toFixed(4));
console.log('Verlet final energy:', verletCurve[1023].toFixed(4));
console.log('Verlet never fell below:', verletLow.toFixed(4));
`,
      publicTests: [
        {
          name: 'two energy curves, 1,024 samples each',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2,
              `expected 2 kernels (rk4Energy, verletEnergy), found ${ctx.kernels.length}`);
            const found = findEnergyKernels(ctx);
            ctx.assert(found.rk4, 'no kernel produced the RK4 energy curve');
            ctx.assert(found.verlet, missingVerletHint(found));
            const curve = found.verlet(DT_LONG);
            for (let i = 0; i < LONG_STEPS; i += 97) {
              ctx.assert(Number.isFinite(curve[i]), `Verlet sample ${i} came back ${curve[i]}`);
            }
          },
        },
        {
          name: 'RK4 slides downhill: the energy only ever falls',
          run: async ctx => {
            const { rk4 } = findEnergyKernels(ctx);
            ctx.assert(rk4, 'no kernel produced the RK4 energy curve');
            const curve = rk4(DT_LONG);
            const samples = [0, 255, 511, 767, 1023];
            for (const i of samples) {
              const expected = energyRatio(runRk4, DT_LONG, i + 1);
              ctx.assertClose(curve[i], expected, 0.01 * expected, `RK4 energy after ${i + 1} steps`);
            }
            for (let k = 1; k < samples.length; k++) {
              ctx.assert(
                curve[samples[k]] < curve[samples[k - 1]] - 0.01,
                `RK4's energy after ${samples[k] + 1} steps is not below its value after ` +
                `${samples[k - 1] + 1} steps — this curve should fall monotonically`
              );
            }
            ctx.assert(curve[1023] < 0.85,
              `after 1,024 steps RK4 should have leaked about a fifth of the energy ` +
              `(≈0.806 of the start), got ${curve[1023].toFixed(4)}`);
          },
        },
        {
          name: 'Verlet never dips below the energy it started with',
          run: async ctx => {
            const found = findEnergyKernels(ctx);
            ctx.assert(found.verlet, missingVerletHint(found));
            const curve = found.verlet(DT_LONG);
            for (const i of [0, 255, 511, 767, 1023]) {
              const expected = energyRatio(runVerlet, DT_LONG, i + 1);
              ctx.assertClose(curve[i], expected, 0.01 * expected, `Verlet energy after ${i + 1} steps`);
            }
            let low = Infinity;
            let high = -Infinity;
            for (let i = 0; i < LONG_STEPS; i++) {
              if (curve[i] < low) low = curve[i];
              if (curve[i] > high) high = curve[i];
            }
            // measured on WebGL float32: low 0.9999978, high 1.0666653
            ctx.assert(low >= 0.999,
              `the Verlet curve dipped to ${low.toFixed(6)} — released from x = 0 its energy is ` +
              `at its minimum, so a correct symplectic step can ripple upward but never below 1`);
            ctx.assert(high <= 1.08,
              `the Verlet curve reached ${high.toFixed(4)} — the ripple should stay inside ` +
              `dt²/(4 − dt²) ≈ 6.7% above the start, and stay there forever`);
          },
        },
        {
          name: 'the verdict is logged',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            const rk4End = energyRatio(runRk4, DT_LONG, LONG_STEPS);
            const verletEnd = energyRatio(runVerlet, DT_LONG, LONG_STEPS);
            ctx.assert(nums.some(v => Math.abs(v - rk4End) <= 0.01),
              `log RK4's final energy — expected ≈${rk4End.toFixed(4)} in the console output`);
            ctx.assert(nums.some(v => Math.abs(v - verletEnd) <= 0.01),
              `log Verlet's final energy — expected ≈${verletEnd.toFixed(4)} in the console output`);
            ctx.assert(nums.some(v => Math.abs(v - 1) <= 0.002),
              'log the smallest value the Verlet curve reaches — it should be ≈1.0000, which is ' +
              'the whole point of the task');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different step size: at dt = 0.4 RK4 leaks 5.5% instead of 19%,
            // and Verlet's ripple narrows to 4.2% — both still on their own side
            // of 1, and a hardcoded 0.5 fails here.
            const found = findEnergyKernels(ctx);
            ctx.assert(found.rk4, 'no kernel produced the RK4 energy curve');
            ctx.assert(found.verlet, missingVerletHint(found));
            const rk4Curve = found.rk4(0.4);
            const verletCurve = found.verlet(0.4);
            for (const i of [63, 511, 1023]) {
              const rkRef = energyRatio(runRk4, 0.4, i + 1);
              const vlRef = energyRatio(runVerlet, 0.4, i + 1);
              ctx.assertClose(rk4Curve[i], rkRef, 0.01 * rkRef, `RK4 energy at dt = 0.4 after ${i + 1} steps`);
              ctx.assertClose(verletCurve[i], vlRef, 0.01 * vlRef, `Verlet energy at dt = 0.4 after ${i + 1} steps`);
            }
            let low = Infinity;
            let high = -Infinity;
            for (let i = 0; i < LONG_STEPS; i++) {
              if (verletCurve[i] < low) low = verletCurve[i];
              if (verletCurve[i] > high) high = verletCurve[i];
            }
            ctx.assert(low >= 0.999, `Verlet dipped to ${low.toFixed(6)} at dt = 0.4`);
            ctx.assert(high <= 1.05, `Verlet reached ${high.toFixed(4)} at dt = 0.4`);
            ctx.assert(rk4Curve[1023] < 0.97 && rk4Curve[1023] > 0.90,
              `RK4 should end near 0.945 of its starting energy at dt = 0.4, got ${rk4Curve[1023].toFixed(4)}`);
          },
        },
      ],
    },
  ],
};
