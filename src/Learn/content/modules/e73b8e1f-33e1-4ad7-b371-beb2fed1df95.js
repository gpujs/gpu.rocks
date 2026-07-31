// Module: Iterative Linear Solvers — uuid e73b8e1f-33e1-4ad7-b371-beb2fed1df95
// (short id e73b8e1f). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId — this module postdates
// the uuid migration.
//
// Five tasks: one Jacobi sweep on a heated plate → the residual, and watching it
// fall → the red half of a red-black sweep → chaining both halves into a full
// Gauss-Seidel sweep → the race, counting sweeps to a fixed tolerance.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested number arrays as inputs, this.thread.* for indexing,
// this.constants.* for compile-time values. Every task passes in CPU mode.
//
// TWO GPU.JS TRAPS THIS MODULE DELIBERATELY WALKS PAST
//
//   • A boolean cannot live in a kernel VARIABLE on the GL backend:
//     `const isRed = (x + y) % 2 === 0;` compiles happily on the CPU backend and
//     dies on WebGL with "cannot convert from 'bool' to 'lowp float'" (verified
//     against gpu.js 2.20 on a real GL context). Colouring a grid is exactly the
//     temptation, so tasks 3–5 keep the parity as a NUMBER — and because the
//     failure is a shader compile error rather than a wrong value, no numeric
//     probe can ever see it: boolTrapHint() reads the RUN's error text instead.
//   • A kernel local must never share a name with a constant — `const size =
//     this.constants.size` transpiles to `const constants_size = constants_size`
//     and throws on the CPU backend only. Nothing here aliases a constant.
//
// FLOAT MARGINS ARE A CONTRACT (measured, not guessed). Tests compute in
// float64; the GL backend computes in float32. Every single-sweep assertion in
// tasks 1–4 is one kernel call deep, where 1e-4 is comfortable by three orders
// of magnitude. The only quantity that accumulates is task 5's sweep COUNT,
// which is a float comparison against a tolerance — a decision boundary, and
// therefore the one thing in this file that had to be measured three ways:
//
//     float64 reference          jacobi 275, red-black 170
//     float32-rounded reference  jacobi 275, red-black 170   (Math.fround after
//                                                             every cell update)
//     gpu.js on a real GL ctx    jacobi 275, red-black 170
//
// All three agree exactly. They agree because the residual is only sampled every
// 5 sweeps, which makes the count discrete: at the deciding check the residual
// sits 2.4% below the tolerance and at the check before it 0.4% above, so a
// float32 wobble would have to be enormous to move the answer. The assertions
// still allow ±10 sweeps (two checks) around a reference the test recomputes for
// itself, and lean on the RATIO rather than the exact numbers.
//
// A SWEEP COUNT IS A WEAK PROBE, and task 5's diagnoses are written accordingly.
// Measured on this problem: correct red-black 170, Jacobi 275, red-black with
// half-sweeps counted 340 — and an under-relaxed weighted-Jacobi step (ω = 0.8)
// also 340, exactly. Two quite different mistakes, one number. So every count
// message there states the OBSERVATION ("almost exactly twice the expected
// number") and offers the usual culprit as somewhere to look, rather than
// asserting a cause it cannot know. ω = 0.9 finishes in 305 and matches nothing,
// which is the negative control the browser pass used.
//
// SIZING THE PAYOFF CLAIM. "Red-black converges in meaningfully fewer sweeps" is
// measured, not asserted. At 32×32, from a zero interior guess, to an RMS
// residual below 5e-4: Jacobi 275 sweeps, red-black 170 — 38% fewer, for the
// same 900 cell updates per sweep. The gap widens as the tolerance tightens
// (at 4e-4 it is 315 vs 190; at 3e-4, 370 vs 220) because Gauss-Seidel's error
// factor is asymptotically the SQUARE of Jacobi's; 5e-4 is the tightest finish
// line that still runs fast on both backends. Measured cost of task 5's whole
// program (both solvers, residual checks included): 22 ms on the gpu.js CPU
// backend, 380 ms on a real GL context — so no budgetMs is needed even with the
// pre-flight probe running the program a second time.

const SIZE = 32; // every grid in this module
const TOL = 5e-4; // task 5's finish line: RMS residual below this
const CHECK_EVERY = 5; // sweeps between residual checks in task 5
const MAX_SWEEPS = 400; // task 5's safety cap
const WATCH_SWEEPS = 60; // task 2's fixed run

// ---- the problem ----------------------------------------------------------

// A square plate whose edge is clamped at fixed temperatures: 0 all round
// except a heater across the middle third of the last row. The interior starts
// at 0 — the honest "I have no idea" guess, and the starting point for the two
// tasks that iterate to convergence.
function makePlate(size) {
  const lo = Math.round(size / 3);
  const hi = size - lo;
  const grid = new Array(size);
  for (let y = 0; y < size; y++) grid[y] = new Array(size).fill(0);
  for (let x = lo; x < hi; x++) grid[size - 1][x] = 1;
  return grid;
}

// The same plate part-way through a solve: a deterministic rough guess in the
// interior, the boundary untouched. The single-sweep tasks use this rather than
// the zero plate because a zero interior leaves most cells at exactly 0 after
// one sweep, where a wrong formula and a right one look identical — this makes
// every one of the 900 interior cells move, which is what lets the diagnosis
// probes tell mistakes apart.
function makeGuess(utils, size, seed) {
  const grid = makePlate(size);
  const rand = utils.seededRandom(seed);
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) grid[y][x] = Math.round(rand() * 1000) / 1000;
  }
  return grid;
}

function makeFlat(size, value) {
  const grid = new Array(size);
  for (let y = 0; y < size; y++) grid[y] = new Array(size).fill(value);
  return grid;
}

// Works on a kernel result (rows are Float32Array) as well as a plain grid.
function copyGrid(u) {
  const out = new Array(u.length);
  for (let y = 0; y < u.length; y++) {
    const row = new Array(u[y].length);
    for (let x = 0; x < u[y].length; x++) row[x] = u[y][x];
    out[y] = row;
  }
  return out;
}

// ---- CPU references -------------------------------------------------------

function neighborSum(u, y, x) {
  return u[y][x - 1] + u[y][x + 1] + u[y - 1][x] + u[y + 1][x];
}

// Fill every interior cell from `cell(y, x)`, leaving the boundary as it was.
// Every reference and every whole-grid probe candidate is built through here,
// so they can only ever differ in the one expression under test.
function interiorMap(u, cell) {
  const n = u.length;
  const out = copyGrid(u);
  for (let y = 1; y < n - 1; y++) {
    for (let x = 1; x < n - 1; x++) out[y][x] = cell(y, x);
  }
  return out;
}

// One Jacobi sweep: every interior cell becomes the average of its four
// neighbours in the PREVIOUS grid. The boundary never moves.
function jacobiRef(u) {
  return interiorMap(u, (y, x) => neighborSum(u, y, x) / 4);
}

// One half-sweep. `colour` is 0 (red) or 1 (black): only cells whose (x + y)
// parity matches update, and they read the grid they were handed.
function halfSweepRef(u, colour) {
  return interiorMap(u, (y, x) =>
    (x + y) % 2 === colour ? neighborSum(u, y, x) / 4 : u[y][x]
  );
}

// A full red-black Gauss-Seidel sweep: the red half, then the black half
// reading the grid the red half produced.
function redBlackRef(u) {
  return halfSweepRef(halfSweepRef(u, 0), 1);
}

// The residual: 0 on the boundary (there is no equation to violate where the
// value is given), the 5-point Laplacian inside.
function residualRef(u) {
  const n = u.length;
  const out = makeFlat(n, 0);
  for (let y = 1; y < n - 1; y++) {
    for (let x = 1; x < n - 1; x++) out[y][x] = neighborSum(u, y, x) - 4 * u[y][x];
  }
  return out;
}

function rmsOf(grid) {
  const n = grid.length;
  let sum = 0;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) sum += grid[y][x] * grid[y][x];
  }
  return Math.sqrt(sum / (n * n));
}

// Task 5's race, run exactly the way the starter's loop runs it: check the
// residual, then take CHECK_EVERY sweeps, until it drops below TOL or the cap
// is reached. Sampling every 5 sweeps is what makes the answer discrete enough
// to assert on — see the float-margins note in the header.
function sweepsToTolRef(u0, sweep) {
  let u = u0;
  let sweeps = 0;
  while (sweeps < MAX_SWEEPS) {
    if (rmsOf(residualRef(u)) < TOL) break;
    for (let i = 0; i < CHECK_EVERY; i++) {
      u = sweep(u);
      sweeps++;
    }
  }
  return sweeps;
}

// The residual after k Jacobi sweeps of the plate — task 2's expected console.
function watchTrace(size, sweeps, every) {
  let u = makePlate(size);
  const trace = [];
  for (let k = 0; k <= sweeps; k++) {
    if (k % every === 0) trace.push(rmsOf(residualRef(u)));
    u = jacobiRef(u);
  }
  return trace;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; the helpers below speak
// only when the observation matches a probe AND the correct answer does not —
// so a cell where two candidates coincide (a flat patch, where averaging four
// neighbours and averaging five cells agree exactly) stays silent, as do
// observations matching probes that disagree with each other. A confident wrong
// diagnosis is worse than a plain numeric mismatch.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Every mistake this module can make is grid-WIDE — a wrong divisor, a missing
// parity test, an unchained half-sweep — so a candidate must predict all 1,024
// cells (and disagree with the right answer somewhere) before it may speak. One
// lucky cell is not evidence: on a plate whose corners barely move, "divided by
// 5" and "divided by 4" agree to the eye across whole regions.
function diagnoseGrid(out, expected, eps, alternatives) {
  const n = expected.length;
  const hits = alternatives
    .filter(([candidate]) => {
      let differs = false;
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          if (!(out[y] && Math.abs(out[y][x] - candidate[y][x]) <= eps)) return false;
          if (Math.abs(expected[y][x] - candidate[y][x]) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(a => a[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// gpu.js cannot hold a boolean in a kernel variable on the GL backend, and
// colouring a grid is precisely the temptation. The failure is a shader compile
// error at the kernel's FIRST CALL, so it never reaches a value probe — it
// arrives as a failed run whose error text is 200 characters of GLSL. Read that
// text instead. The match is the exact GLSL wording, so it cannot fire on an
// unrelated failure.
function boolTrapHint(ctx) {
  const message = (ctx.error && ctx.error.message) || '';
  return /cannot convert from 'bool'/i.test(message)
    ? 'gpu.js cannot store a boolean in a kernel variable on the WebGL backend — ' +
        'const isRed = (x + y) % 2 === 0 compiles on the CPU backend and dies here. ' +
        'Keep the parity as a number: const parity = (x + y) % 2; then if (parity === 0).'
    : null;
}

// Every task whose kernels can hit the boolean trap opens with this, because
// gpu.js compiles lazily: the kernel OBJECT exists, so a "no kernel was created"
// check passes and the learner is handed raw GLSL instead of a diagnosis.
function assertRunOk(ctx) {
  ctx.assert(
    ctx.ok,
    boolTrapHint(ctx) ||
      `the program did not finish: ${(ctx.error && ctx.error.message) || 'unknown error'}`
  );
}

// Boundary cells are the given data — a solver that moves them is solving a
// different problem. Nothing else produces this failure, so no probe list is
// needed: any change at all is the diagnosis.
function boundaryHint(got, expected, eps) {
  return Math.abs(got - expected) > eps
    ? 'a boundary cell moved — the edges are the known values that pin the whole ' +
        'solution down, so the kernel has to return u[y][x] unchanged for any cell ' +
        'with x or y equal to 0 or size − 1'
    : null;
}

// ---- probe candidates -----------------------------------------------------

// Task 1: the Jacobi update, mis-averaged.
function jacobiAlternatives(u) {
  return [
    [interiorMap(u, (y, x) => (neighborSum(u, y, x) + u[y][x]) / 5),
      'the centre crept into the average — a Jacobi update reads only the four neighbours, never its own old value'],
    [interiorMap(u, (y, x) => neighborSum(u, y, x) / 5),
      'the neighbour sum was divided by 5 — there are four neighbours, so divide by 4'],
    [interiorMap(u, (y, x) => neighborSum(u, y, x)),
      'the four neighbours were added but never averaged — divide the sum by 4'],
    [interiorMap(u, (y, x) => neighborSum(u, y, x) - 4 * u[y][x]),
      'that is the residual, not the update — the update is the neighbours’ average, (left + right + up + down) / 4'],
    [interiorMap(u, (y, x) => (u[y - 1][x - 1] + u[y - 1][x + 1] + u[y + 1][x - 1] + u[y + 1][x + 1]) / 4),
      'those are the four DIAGONAL neighbours — the 5-point stencil reads left, right, up and down, one coordinate at a time'],
    [copyGrid(u),
      'nothing moved — every cell returned its own value instead of its neighbours’ average'],
  ];
}

// Task 2: the residual, mis-assembled. Candidates keep the boundary at 0 so a
// boundary mistake falls through to boundaryResidualHint instead of being
// mis-diagnosed as an arithmetic one.
function residualAlternatives(u) {
  const n = u.length;
  const build = cell => {
    const out = makeFlat(n, 0);
    for (let y = 1; y < n - 1; y++) {
      for (let x = 1; x < n - 1; x++) out[y][x] = cell(y, x);
    }
    return out;
  };
  return [
    [build((y, x) => 4 * u[y][x] - neighborSum(u, y, x)),
      'the sign is flipped — the residual is (left + right + up + down) − 4·centre, neighbours first'],
    [build((y, x) => neighborSum(u, y, x)),
      'the − 4·centre term is missing — the residual asks how far the cell is from its neighbours, so the centre has to be subtracted'],
    [build((y, x) => neighborSum(u, y, x) / 4 - u[y][x]),
      'that is the residual divided by 4 — the Jacobi update minus the centre. The residual sums the four neighbours and subtracts 4·centre, with no division'],
    [build((y, x) => neighborSum(u, y, x) / 4),
      'that is the Jacobi update, not the residual — subtract 4·centre from the neighbour SUM'],
  ];
}

// Task 2: the reduction, applied to the right grid the wrong way. `sumSq` is
// the residual grid's raw sum of squares, recovered from the reference RMS.
function rmsProbes(sumSq, cells, interiorCells) {
  return [
    [sumSq, 'that is the raw sum of squares — an RMS takes the mean and then the square root'],
    [sumSq / cells, 'the square root is missing — Math.sqrt(sum / (32 × 32))'],
    [Math.sqrt(sumSq), 'the division is missing — an RMS is a MEAN square before the root'],
    [Math.sqrt(sumSq / interiorCells),
      'the mean divides by every cell in the grid — 32 × 32 — not just the interior ones'],
  ];
}

// Task 3: the red half-sweep, half done.
function halfSweepAlternatives(u) {
  return [
    [jacobiRef(u),
      'every interior cell updated — that is a full Jacobi sweep. A half-sweep leaves the cells where (x + y) % 2 is 1 exactly as it found them'],
    [halfSweepRef(u, 1),
      'the colours are swapped — the red half-sweep updates the cells where (x + y) % 2 is 0 and passes the rest through'],
    [copyGrid(u),
      'nothing moved — no cell received its neighbours’ average'],
  ];
}

// Task 4: the two halves, mis-chained. The first entry is the one this task
// exists for: black(u) instead of black(red(u)).
//
// Deliberately NOT listed: a black kernel with no parity test at all. Run over
// the red half's output it produces the RIGHT grid — a red cell's neighbours are
// all black, and the red half changed no black cell, so recomputing a red cell
// lands on the value it already has. The composition is identical, so a probe
// for it could never fire and would only be a lie waiting to happen. The wasted
// half of the work is caught by the "black kernel on its own" test instead.
function fullSweepAlternatives(u) {
  return [
    [halfSweepRef(u, 1),
      'only the black half reached the answer — the halves have to be chained, black(red(u)), so the black cells read the reds the first half just wrote'],
    [halfSweepRef(u, 0),
      'the black half changed nothing — check that its parity test keeps the cells where (x + y) % 2 is 1 and passes the rest through'],
    [halfSweepRef(halfSweepRef(u, 1), 0),
      'the halves ran in the wrong order — red first, then black on the red half’s output'],
    [jacobiRef(u),
      'both halves read the same grid — that is Jacobi in two passes. The whole point of red-black is that the black half reads what the red half just wrote'],
  ];
}

// ---- task 2 / task 5 console readers --------------------------------------

// The residual line the prewired loop prints: "sweep 20 — RMS residual 0.00726".
// The loop is not the learner's code, so the format is fixed; only the NUMBER
// is theirs.
function loggedResiduals(ctx) {
  const values = [];
  for (const line of ctx.logs || []) {
    if (line.type !== 'log' || !line.text) continue;
    const match = /residual\s+(-?[0-9][0-9.eE+-]*)\s*$/.exec(String(line.text).trim());
    if (match && Number.isFinite(Number(match[1]))) values.push(Number(match[1]));
  }
  return values;
}

// "jacobi: converged in 275 sweeps" / "red-black: converged in 170 sweeps".
function loggedCount(ctx, label) {
  const re = new RegExp(`${label}:\\s*converged in\\s*(\\d+)\\s*sweeps`, 'i');
  for (const line of ctx.logs || []) {
    if (line.type !== 'log' || !line.text) continue;
    const match = re.exec(String(line.text));
    if (match) return Number(match[1]);
  }
  return null;
}

// ---- shared kernel source (prewired into later starters) ------------------

const JACOBI_KERNEL = `const sweep = gpu.createKernel(function (u) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.size - 1 || y === this.constants.size - 1) {
    return u[y][x];
  }
  return (u[y][x - 1] + u[y][x + 1] + u[y - 1][x] + u[y + 1][x]) / 4;
}, { output: [32, 32], constants: { size: 32 } });`;

const RESIDUAL_KERNEL = `const residual = gpu.createKernel(function (u) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.size - 1 || y === this.constants.size - 1) {
    return 0;
  }
  return u[y][x - 1] + u[y][x + 1] + u[y - 1][x] + u[y + 1][x] - 4 * u[y][x];
}, { output: [32, 32], constants: { size: 32 } });`;

const RED_KERNEL = `const red = gpu.createKernel(function (u) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.size - 1 || y === this.constants.size - 1) {
    return u[y][x];
  }
  // parity as a NUMBER — gpu.js cannot keep a boolean in a kernel variable
  const parity = (x + y) % 2;
  if (parity !== 0) return u[y][x];
  return (u[y][x - 1] + u[y][x + 1] + u[y - 1][x] + u[y + 1][x]) / 4;
}, { output: [32, 32], constants: { size: 32 } });`;

const BLACK_KERNEL = RED_KERNEL
  .replace('const red =', 'const black =')
  .replace('if (parity !== 0)', 'if (parity !== 1)');

const RMS_HELPER = `function rmsOf(grid) {
  let sum = 0;
  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) sum += grid[y][x] * grid[y][x];
  }
  return Math.sqrt(sum / (32 * 32));
}`;

export default {
  uuid: 'e73b8e1f-33e1-4ad7-b371-beb2fed1df95',
  version: 1,
  slug: 'iterative-solvers',
  title: 'Iterative Linear Solvers',
  blurb: 'Jacobi, Gauss-Seidel, and why colouring a grid like a chessboard turns a sequential algorithm parallel.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'jacobi-sweep',
      title: 'One Sweep of Jacobi',
      intro: `<p>A square metal plate, its edges clamped at fixed temperatures. What does the
        inside settle to? Not "what happens next" — the answer once nothing happens any more.
        That steady state solves <code>∇²u = 0</code>, and the 5-point stencil turns it into one
        tiny equation per interior cell: <strong>every cell equals the average of its four
        neighbours</strong>. Nine hundred equations, nine hundred unknowns, all tangled together.</p>
        <p>Nobody inverts that matrix. You guess, and improve the guess. <strong>Jacobi's
        method</strong> is that made literal: set every interior cell to the average of its
        neighbours <em>as they were before this sweep</em>, and repeat. Because every cell reads
        the previous iterate and nothing else, all 1,024 threads are independent — it is the pure
        <em>gather</em> Thinking in Parallel calls the shape that always parallelises, and one
        whole sweep is one kernel call.</p>
        <p>The edges never move: they are the known values that pin the answer down. (Reaction–Diffusion
        steps this same stencil <em>forward in time</em>; here we are solving for the state where
        time has stopped.)</p>`,
      goal: `<strong>Goal:</strong> finish the sweep kernel — interior cells return the average of
        their four neighbours in <code>u</code>, boundary cells return their own value unchanged.`,
      requirements: [
        'Boundary cells — <code>x</code> or <code>y</code> equal to <code>0</code> or <code>size − 1</code> — return <code>u[y][x]</code> untouched',
        'Interior cells return <code>(left + right + up + down) / 4</code>: four neighbours, no centre',
        'Read only <code>u</code> — no cell may see a value written during this sweep',
      ],
      hints: [
        {
          title: 'Hint 1 — the edges come first',
          body: `<p>Guard the boundary before you do any arithmetic, so the neighbour reads below
            can never leave the grid:</p>
<pre><code>if (x === 0 || y === 0 || x === this.constants.size - 1
    || y === this.constants.size - 1) {
  return u[y][x];
}</code></pre>`,
        },
        {
          title: 'Hint 2 — the four neighbours',
          body: `<p>Only ever vary <em>one</em> coordinate at a time: <code>u[y][x - 1]</code> and
            <code>u[y][x + 1]</code> along the row, <code>u[y - 1][x]</code> and
            <code>u[y + 1][x]</code> down the column. The centre <code>u[y][x]</code> is not part
            of a Jacobi average.</p>`,
        },
        {
          title: 'Hint 3 — the whole return',
          body: `<pre><code>return (u[y][x - 1] + u[y][x + 1]
      + u[y - 1][x] + u[y + 1][x]) / 4;</code></pre>`,
        },
      ],
      transfer: `Jacobi is the starting point of every multigrid solver on every platform — a CUDA
        or WGSL version of this kernel is line-for-line the same gather, with a buffer swap where
        your JavaScript assignment is. It is also why "matrix-free" is a phrase: nobody stores the
        900×900 matrix this stencil stands for, because the kernel <em>is</em> the matrix.`,
      starterCode: `// One Jacobi sweep: every interior cell becomes the average of its four
// neighbours, read from the grid as it was BEFORE this sweep.
const gpu = new GPU({ mode });

const sweep = gpu.createKernel(function (u) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO 1: the boundary is held fixed — return u[y][x] for any cell whose
  //         x or y is 0 or this.constants.size - 1.
  // TODO 2: every other cell returns the average of its four neighbours:
  //         u[y][x - 1], u[y][x + 1], u[y - 1][x], u[y + 1][x].
  return u[y][x];
}, { output: [32, 32], constants: { size: 32 } });

const next = await sweep(guess);
console.log('centre before:', guess[16][16], '→ after:', next[16][16]);
`,
      solutionCode: `// One Jacobi sweep: every interior cell becomes the average of its four
// neighbours, read from the grid as it was BEFORE this sweep.
const gpu = new GPU({ mode });

${JACOBI_KERNEL}

const next = await sweep(guess);
console.log('centre before:', guess[16][16], '→ after:', next[16][16]);
`,
      inputs: utils => ({ guess: makeGuess(utils, SIZE, 4711) }),
      publicTests: [
        {
          name: 'the sweep produces a 32×32 grid',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(makeGuess(ctx.utils, SIZE, 4711));
            ctx.assert(out && out.length === SIZE, `expected 32 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === SIZE, 'each row should hold 32 values');
          },
        },
        {
          name: 'interior cells hold <code>(left + right + up + down) / 4</code>',
          run: async ctx => {
            const u = makeGuess(ctx.utils, SIZE, 4711);
            const out = await ctx.kernel(u);
            const ref = jacobiRef(u);
            const hint = diagnoseGrid(out, ref, 1e-4, jacobiAlternatives(u));
            for (let y = 1; y < SIZE - 1; y++) {
              for (let x = 1; x < SIZE - 1; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'the boundary is held fixed — the heater and the cold edges never move',
          run: async ctx => {
            const u = makeGuess(ctx.utils, SIZE, 4711);
            const out = await ctx.kernel(u);
            for (let i = 0; i < SIZE; i++) {
              const edges = [[0, i], [SIZE - 1, i], [i, 0], [i, SIZE - 1]];
              for (const [y, x] of edges) {
                ctx.assertClose(
                  out[y][x], u[y][x], 1e-5,
                  boundaryHint(out[y][x], u[y][x], 1e-5) || `boundary cell [${y}][${x}]`
                );
              }
            }
          },
        },
        {
          name: 'a flat field is already the answer — the sweep leaves it alone',
          run: async ctx => {
            // Every cell equal to its neighbours' average is exactly what the
            // equation asks for, so a constant grid must be a fixed point. This
            // is the cheapest way to catch a divisor of 5 or a centre term.
            const flat = makeFlat(SIZE, 0.7);
            const out = await ctx.kernel(flat);
            const hint = diagnoseGrid(out, flat, 1e-5, jacobiAlternatives(flat));
            for (let y = 0; y < SIZE; y += 3) {
              for (let x = 0; x < SIZE; x += 3) {
                ctx.assertClose(out[y][x], 0.7, 1e-5, hint || `cell [${y}][${x}] of a flat field`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different rough guess, compared cell for cell including edges.
            const u = makeGuess(ctx.utils, SIZE, 90210);
            const out = await ctx.kernel(u);
            const ref = jacobiRef(u);
            const hint = diagnoseGrid(out, ref, 1e-4, jacobiAlternatives(u));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                const edge = y === 0 || x === 0 || y === SIZE - 1 || x === SIZE - 1;
                const message = edge
                  ? boundaryHint(out[y][x], ref[y][x], 1e-5)
                  : hint;
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, message || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Two sweeps in a row: the kernel has to accept its own output as
            // input (a Float32Array-backed grid) and keep agreeing with the
            // reference — the feedback shape every later task depends on.
            const u = makeGuess(ctx.utils, SIZE, 31337);
            const once = await ctx.kernel(u);
            const twice = await ctx.kernel(once);
            const ref = jacobiRef(jacobiRef(u));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(twice[y][x], ref[y][x], 2e-4, `cell [${y}][${x}] after two sweeps`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'watch-the-residual',
      title: 'Watch the Residual Fall',
      intro: `<p>Iterating is easy; knowing when to stop is the skill. The honest measure is the
        <strong>residual</strong>: take the current guess, put it back into the equation, and see
        how badly it is violated. For "every cell equals the average of its four neighbours" that
        is the 5-point Laplacian — <code>left + right + up + down − 4·centre</code> — the same
        stencil Reaction–Diffusion uses to diffuse, borrowed here as a scorecard. Zero where the
        equation holds, large where it does not.</p>
        <p>Boundary cells have no equation to violate: they are given, not solved. Their residual
        is <code>0</code> by definition rather than by accident, and saying so in the kernel keeps
        the edge from polluting the score forever.</p>
        <p>A grid of numbers is not a progress report, so collapse it to one: the root-mean-square
        over the grid. Totalling a grid <em>on</em> the GPU is the halving ladder Reductions
        builds; at 1,024 cells the read-back is cheaper than the ladder, so this sum happens in
        plain JavaScript.</p>`,
      goal: `<strong>Goal:</strong> complete the <code>residual</code> kernel and the
        <code>rmsOf</code> helper. The sweep loop is already wired and will print the residual
        every 10 sweeps — you should watch it fall by a factor of about 36.`,
      requirements: [
        'The kernel returns <code>0</code> for boundary cells',
        'Interior cells return <code>left + right + up + down − 4·centre</code> — a sum, with no division',
        '<code>rmsOf(grid)</code> returns <code>Math.sqrt(sum of every cell squared / (32 × 32))</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the kernel is the stencil, unaveraged',
          body: `<p>Same five reads as the Jacobi sweep, assembled differently: the update divides
            the neighbour sum by 4, the residual subtracts <code>4 × centre</code> from it.</p>
<pre><code>return u[y][x - 1] + u[y][x + 1] + u[y - 1][x]
     + u[y + 1][x] - 4 * u[y][x];</code></pre>`,
        },
        {
          title: 'Hint 2 — the reduction is ordinary JavaScript',
          body: `<p><code>residual(u)</code> hands back a plain 2D array of numbers, so:</p>
<pre><code>let sum = 0;
for (let y = 0; y &lt; 32; y++) {
  for (let x = 0; x &lt; 32; x++) sum += grid[y][x] * grid[y][x];
}
return Math.sqrt(sum / (32 * 32));</code></pre>
<p>Square, mean, root — in that order. Divide by every cell in the grid, not just the
            interior ones.</p>`,
        },
      ],
      transfer: `Every production solver stops on a residual, not on a sweep count, and every one
        of them argues about how often to measure it: the norm needs a reduction across the whole
        device and then a read-back to the host, which is a synchronisation point in CUDA, WebGPU
        and MPI alike. Checking every iteration can cost more than the iterations do — the usual
        answer is exactly what this task does, sample it every few sweeps.`,
      starterCode: `// How wrong is the current guess? Plug it back into the equation.
const gpu = new GPU({ mode });
const SWEEPS = 60;

${JACOBI_KERNEL}

const residual = gpu.createKernel(function (u) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO 1: a boundary cell has no equation to violate — return 0.
  // TODO 2: every other cell returns
  //         left + right + up + down - 4 * centre.
  return 1;
}, { output: [32, 32], constants: { size: 32 } });

function rmsOf(grid) {
  // TODO 3: square every cell, take the mean over all 32 × 32 of them,
  //         then the square root.
  return 0;
}

let u = plate;
for (let k = 0; k <= SWEEPS; k++) {
  if (k % 10 === 0) console.log('sweep', k, '— RMS residual', rmsOf(await residual(u)));
  u = await sweep(u);
}
`,
      solutionCode: `// How wrong is the current guess? Plug it back into the equation.
const gpu = new GPU({ mode });
const SWEEPS = 60;

${JACOBI_KERNEL}

${RESIDUAL_KERNEL}

${RMS_HELPER}

let u = plate;
for (let k = 0; k <= SWEEPS; k++) {
  if (k % 10 === 0) console.log('sweep', k, '— RMS residual', rmsOf(await residual(u)));
  u = await sweep(u);
}
`,
      inputs: () => ({ plate: makePlate(SIZE) }),
      publicTests: [
        {
          name: 'the residual of a flat field is <code>0</code> everywhere',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2,
              `expected 2 kernels (the prewired sweep, then residual), found ${ctx.kernels.length}`);
            const flat = makeFlat(SIZE, 0.4);
            const out = await ctx.kernels[1](flat);
            ctx.assert(out && out.length === SIZE && out[0].length === SIZE,
              'expected a 32×32 residual grid');
            // A flat field makes several candidates collapse onto the right
            // answer (a flipped sign of zero is still zero), and diagnoseGrid
            // filters exactly those out — what survives are the two that do not.
            const hint = diagnoseGrid(out, residualRef(flat), 1e-5, residualAlternatives(flat));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                const edge = y === 0 || x === 0 || y === SIZE - 1 || x === SIZE - 1;
                const message = edge
                  ? 'a boundary cell scored non-zero — the edges are given, not solved, so their residual is 0 by definition'
                  : hint;
                ctx.assertClose(out[y][x], 0, 1e-5,
                  message || `cell [${y}][${x}] of a flat field — every cell already equals its neighbours' average, so nothing is violated`);
              }
            }
          },
        },
        {
          name: 'interior cells hold <code>left + right + up + down − 4·centre</code>',
          run: async ctx => {
            const u = makeGuess(ctx.utils, SIZE, 6180);
            const out = await ctx.kernels[1](u);
            const ref = residualRef(u);
            const hint = diagnoseGrid(out, ref, 1e-4, residualAlternatives(u));
            for (let y = 1; y < SIZE - 1; y++) {
              for (let x = 1; x < SIZE - 1; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'the boundary scores <code>0</code> — a given value cannot be wrong',
          run: async ctx => {
            // A guess whose edge carries the heater: an unguarded boundary cell
            // would read outside the grid, so this catches both the missing
            // return and a boundary residual that is merely non-zero.
            const u = makeGuess(ctx.utils, SIZE, 6180);
            const out = await ctx.kernels[1](u);
            for (let i = 0; i < SIZE; i++) {
              const edges = [[0, i], [SIZE - 1, i], [i, 0], [i, SIZE - 1]];
              for (const [y, x] of edges) {
                ctx.assert(Number.isFinite(out[y][x]),
                  `boundary cell [${y}][${x}] came back as ${out[y][x]} — it read outside the grid. Return 0 for boundary cells before touching a neighbour`);
                ctx.assertClose(out[y][x], 0, 1e-5,
                  `boundary cell [${y}][${x}] — the edges are given, not solved, so their residual is 0 by definition`);
              }
            }
          },
        },
        {
          name: 'the console shows the residual falling, sweep by sweep',
          run: async ctx => {
            const trace = watchTrace(SIZE, WATCH_SWEEPS, 10);
            const got = loggedResiduals(ctx);
            ctx.assert(got.length >= trace.length,
              `expected ${trace.length} residual lines in the console (sweeps 0, 10, … ${WATCH_SWEEPS}), found ${got.length} — did rmsOf return a number?`);
            const cells = SIZE * SIZE;
            const interior = (SIZE - 2) * (SIZE - 2);
            for (let i = 0; i < trace.length; i++) {
              const expected = trace[i];
              const sumSq = expected * expected * cells;
              const eps = Math.max(2e-5, expected * 0.02);
              const hint = diagnose(got[i], expected, eps, rmsProbes(sumSq, cells, interior));
              ctx.assertClose(got[i], expected, eps,
                hint || `the residual printed at sweep ${i * 10}`);
            }
            for (let i = 1; i < trace.length; i++) {
              ctx.assert(got[i] < got[i - 1],
                `the residual rose between sweep ${(i - 1) * 10} and sweep ${i * 10} — Jacobi never diverges on this problem, so something is off`);
            }
            ctx.assert(got[trace.length - 1] < 0.004,
              `after ${WATCH_SWEEPS} sweeps the residual should be under 0.004, got ${got[trace.length - 1]}`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A fresh guess, full-grid comparison, edges included.
            const u = makeGuess(ctx.utils, SIZE, 27182);
            const out = await ctx.kernels[1](u);
            const ref = residualRef(u);
            const hint = diagnoseGrid(out, ref, 1e-4, residualAlternatives(u));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The residual has to fall when the solver runs: 20 sweeps driven by
            // the test, measured with the learner's own kernel.
            let u = makePlate(SIZE);
            const before = rmsOf(copyGrid(await ctx.kernels[1](u)));
            for (let i = 0; i < 20; i++) u = await ctx.kernels[0](u);
            const after = rmsOf(copyGrid(await ctx.kernels[1](u)));
            ctx.assertClose(before, rmsOf(residualRef(makePlate(SIZE))), 1e-4,
              'the residual of the starting plate');
            ctx.assert(after < before / 5,
              `20 sweeps should cut the residual by more than 5× (${before.toFixed(5)} → ${after.toFixed(5)})`);
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'red-black-halves',
      title: 'Colour the Board',
      intro: `<p>Jacobi throws information away. Halfway through a sweep plenty of neighbours
        already have better values, and Jacobi ignores every one of them because it reads only the
        old grid. <strong>Gauss-Seidel</strong> is the fix a human would reach for: walk the cells
        in order and always use the newest value available. It converges about twice as fast — and
        it is <em>sequential by construction</em>. Cell 500 cannot start until cell 499 has
        finished. There is no thread ordering on a GPU and no way to make one thread wait for
        another, so the textbook algorithm is simply not on the menu.</p>
        <p>The fix is a chessboard. The 5-point stencil only ever reads the four direct
        neighbours, and on a chessboard every direct neighbour of a red square is black. So call a
        cell <strong>red</strong> when <code>(x + y)</code> is even and <strong>black</strong> when
        it is odd, and update all the reds at once: no red cell reads another red cell, so there is
        nothing left to order. Then update all the blacks, reading the reds that were just
        written — which is exactly the "use the newest value" that made Gauss-Seidel fast. One
        sequential pass becomes two data-parallel half-sweeps.</p>
        <p>This task is the red half. Every thread still writes only its own cell, so black cells
        are not "skipped" — they gather <em>themselves</em>, unchanged, ready for the half-sweep
        that is about to need them exactly as they are.</p>`,
      goal: `<strong>Goal:</strong> write the red half-sweep — cells with <code>(x + y) % 2 === 0</code>
        take their neighbours' average, and every other cell (black cells and the whole boundary)
        comes through untouched.`,
      requirements: [
        'Boundary cells return <code>u[y][x]</code>',
        'Keep the parity in a <em>number</em>: <code>const parity = (x + y) % 2;</code> — a boolean in a kernel variable does not compile on WebGL',
        'Cells with parity <code>1</code> (black) return <code>u[y][x]</code> unchanged',
        'Cells with parity <code>0</code> (red) return <code>(left + right + up + down) / 4</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the trap this task is built around',
          body: `<p>The natural spelling is a boolean, and it is the one thing gpu.js cannot do:</p>
<pre><code>const isRed = (x + y) % 2 === 0;   // throws on WebGL</code></pre>
<p>The GL backend has no way to store a <code>bool</code> in a kernel variable, so it fails
            at shader-compile time with <em>cannot convert from 'bool' to 'lowp float'</em> — and
            the CPU backend runs it happily, which is how this reaches production. Keep the number:</p>
<pre><code>const parity = (x + y) % 2;        // 0 or 1
if (parity !== 0) return u[y][x];</code></pre>`,
        },
        {
          title: 'Hint 2 — three exits, one average',
          body: `<p>The kernel is a stack of guards: boundary first, then the wrong colour, then
            the arithmetic. Whichever way a thread leaves, it writes exactly one cell — its own.</p>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<pre><code>if (x === 0 || y === 0 || x === this.constants.size - 1
    || y === this.constants.size - 1) {
  return u[y][x];
}
const parity = (x + y) % 2;
if (parity !== 0) return u[y][x];
return (u[y][x - 1] + u[y][x + 1]
      + u[y - 1][x] + u[y + 1][x]) / 4;</code></pre>`,
        },
      ],
      transfer: `Red-black ordering — and its multi-colour generalisation — is the standard way to
        put a Gauss-Seidel or SOR smoother on a GPU: CUDA and WGSL do exactly this, one dispatch
        per colour, and unstructured meshes get their colours from a graph-colouring pass first.
        The idea generalises past solvers: a colour is simply a set of updates guaranteed not to
        depend on each other, which is the same permission slip a wavefront or a task-graph level
        hands out.`,
      starterCode: `// The red half-sweep: update the cells where (x + y) is even,
// and pass everything else through untouched.
const gpu = new GPU({ mode });

const red = gpu.createKernel(function (u) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.size - 1 || y === this.constants.size - 1) {
    return u[y][x];
  }
  // TODO 1: compute the parity as a NUMBER — const parity = (x + y) % 2;
  //         (a boolean in a kernel variable will not compile on WebGL)
  // TODO 2: parity 1 is black — return u[y][x] unchanged.
  // TODO 3: parity 0 is red — return the four neighbours' average.
  return u[y][x];
}, { output: [32, 32], constants: { size: 32 } });

const afterRed = await red(guess);
console.log('a red cell [16][16]:', guess[16][16], '→', afterRed[16][16]);
console.log('a black cell [16][17]:', guess[16][17], '→', afterRed[16][17]);
`,
      solutionCode: `// The red half-sweep: update the cells where (x + y) is even,
// and pass everything else through untouched.
const gpu = new GPU({ mode });

${RED_KERNEL}

const afterRed = await red(guess);
console.log('a red cell [16][16]:', guess[16][16], '→', afterRed[16][16]);
console.log('a black cell [16][17]:', guess[16][17], '→', afterRed[16][17]);
`,
      inputs: utils => ({ guess: makeGuess(utils, SIZE, 4711) }),
      publicTests: [
        {
          name: 'the half-sweep produces a 32×32 grid',
          run: async ctx => {
            assertRunOk(ctx);
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(makeGuess(ctx.utils, SIZE, 4711));
            ctx.assert(out && out.length === SIZE && out[0].length === SIZE,
              'expected a 32×32 result');
          },
        },
        {
          name: 'red cells take the average; black cells and the boundary do not move',
          run: async ctx => {
            const u = makeGuess(ctx.utils, SIZE, 4711);
            const out = await ctx.kernel(u);
            const ref = halfSweepRef(u, 0);
            const hint = diagnoseGrid(out, ref, 1e-4, halfSweepAlternatives(u));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                const edge = y === 0 || x === 0 || y === SIZE - 1 || x === SIZE - 1;
                const message = edge ? boundaryHint(out[y][x], ref[y][x], 1e-5) : hint;
                ctx.assertClose(out[y][x], ref[y][x], 1e-4,
                  message || `cell [${y}][${x}] (${(x + y) % 2 === 0 ? 'red' : 'black'})`);
              }
            }
          },
        },
        {
          name: 'exactly half the interior moved — 450 cells, not 900 and not 0',
          run: async ctx => {
            // Counting is a different question from comparing values: it tells a
            // learner who coloured the board wrong that the SHAPE is wrong, even
            // where a stray cell happens to land on the right number.
            const u = makeGuess(ctx.utils, SIZE, 4711);
            const out = await ctx.kernel(u);
            let movedRed = 0;
            let movedBlack = 0;
            for (let y = 1; y < SIZE - 1; y++) {
              for (let x = 1; x < SIZE - 1; x++) {
                if (Math.abs(out[y][x] - u[y][x]) <= 1e-6) continue;
                if ((x + y) % 2 === 0) movedRed++;
                else movedBlack++;
              }
            }
            ctx.assert(movedBlack === 0,
              `${movedBlack} black cells changed — a half-sweep must leave every cell with (x + y) % 2 === 1 exactly as it found it`);
            ctx.assert(movedRed > 400,
              `only ${movedRed} of the 450 red interior cells changed — every cell with (x + y) % 2 === 0 should have taken its neighbours' average`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const u = makeGuess(ctx.utils, SIZE, 8675309);
            const out = await ctx.kernel(u);
            const ref = halfSweepRef(u, 0);
            const hint = diagnoseGrid(out, ref, 1e-4, halfSweepAlternatives(u));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A half-sweep is idempotent: the red cells' inputs (the blacks) did
            // not change, so running it twice must land on the same grid. A
            // kernel that quietly reads its own colour fails this and nothing
            // else.
            const u = makeGuess(ctx.utils, SIZE, 12345);
            const once = await ctx.kernel(u);
            const twice = await ctx.kernel(once);
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(twice[y][x], once[y][x], 1e-4,
                  `cell [${y}][${x}] changed on a second red half-sweep — a red cell must read only black neighbours, which this half-sweep never touched`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'full-sweep',
      title: 'Two Halves Make a Sweep',
      intro: `<p>The red half alone is not a sweep — half the grid has not been touched. The black
        half is the same kernel with its parity test flipped, and the ordering that matters is in
        the <em>chaining</em>: <code>black(red(u))</code>. The black cells read the grid the red
        half produced, so they see this sweep's reds, not last sweep's. That single fact is all
        that separates Gauss-Seidel from Jacobi.</p>
        <p>Write <code>black(u)</code> instead and both halves read the same old grid. The code
        still runs, the answer still looks plausible, and you have written Jacobi with an extra
        kernel launch — which is the most expensive way to be wrong in this module, because
        nothing about the output shouts.</p>`,
      goal: `<strong>Goal:</strong> write the black half-sweep and chain the two halves into one
        full red-black Gauss-Seidel sweep.`,
      requirements: [
        'The black kernel is the red kernel with its parity test flipped: cells with <code>(x + y) % 2 === 1</code> update, everything else passes through',
        'Create the red kernel first and the black kernel second',
        'One full sweep is <code>await black(await red(u))</code> — the black half reads what the red half wrote',
        'The boundary is untouched by both halves',
      ],
      hints: [
        {
          title: 'Hint 1 — the black kernel',
          body: `<p>Copy the red kernel and change one character:</p>
<pre><code>const parity = (x + y) % 2;
if (parity !== 1) return u[y][x];</code></pre>
<p>Still a number, never a boolean — the WebGL backend rejects
            <code>const isBlack = …</code> exactly as it rejects <code>isRed</code>.</p>`,
        },
        {
          title: 'Hint 2 — the chain is the lesson',
          body: `<pre><code>const afterRed = await red(guess);
const afterBoth = await black(afterRed);   // NOT black(guess)</code></pre>
<p>Or, in one line: <code>await black(await red(guess))</code> — the inner
            <code>await</code> is not optional, because an un-awaited kernel call hands the next
            kernel a Promise instead of a grid.</p>`,
        },
      ],
      transfer: `Two dispatches with a dependency between them is the ordinary shape of GPU work:
        WebGPU inserts a barrier between compute passes, CUDA orders them on a stream, Vulkan wants
        an explicit pipeline barrier. What you cannot do — on any of them — is order threads
        <em>inside</em> one dispatch, which is exactly why the sequential Gauss-Seidel had to be
        split into two of them in the first place.`,
      starterCode: `// One full red-black sweep = the red half, then the black half
// reading what the red half just wrote.
const gpu = new GPU({ mode });

${RED_KERNEL}

const black = gpu.createKernel(function (u) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.size - 1 || y === this.constants.size - 1) {
    return u[y][x];
  }
  // TODO 1: same shape as the red kernel, with the parity test flipped —
  //         cells where (x + y) % 2 is 1 take the neighbours' average.
  return u[y][x];
}, { output: [32, 32], constants: { size: 32 } });

const afterRed = await red(guess);
// TODO 2: finish the sweep. The black half must read afterRed, not guess.
const afterSweep = afterRed;

console.log('a black cell [16][17]:', guess[16][17], '→', afterSweep[16][17]);
`,
      solutionCode: `// One full red-black sweep = the red half, then the black half
// reading what the red half just wrote.
const gpu = new GPU({ mode });

${RED_KERNEL}

${BLACK_KERNEL}

const afterRed = await red(guess);
const afterSweep = await black(afterRed);

console.log('a black cell [16][17]:', guess[16][17], '→', afterSweep[16][17]);
`,
      inputs: utils => ({ guess: makeGuess(utils, SIZE, 4711) }),
      publicTests: [
        // FIRST, and it has to stay first: it reads ctx.kernels[1].lastArgs, and
        // every later test invokes the kernels itself, which overwrites them.
        {
          name: 'your code chained them — the black half was handed the red half’s output',
          run: async ctx => {
            // The kernels can both be right and the program still wrong: what
            // this checks is the ARGUMENT the black half was actually called
            // with, which is the whole difference between Gauss-Seidel and
            // Jacobi-in-two-passes.
            assertRunOk(ctx);
            ctx.assert(ctx.kernels.length >= 2,
              `expected 2 kernels (red, then black), found ${ctx.kernels.length}`);
            const seen = ctx.kernels[1].lastArgs && ctx.kernels[1].lastArgs[0];
            ctx.assert(seen, 'the black kernel was never called — one full sweep is black(red(u))');
            const u = makeGuess(ctx.utils, SIZE, 4711);
            const red = halfSweepRef(u, 0);
            let matchesRed = 0;
            let matchesOriginal = 0;
            let cells = 0;
            for (let y = 1; y < SIZE - 1; y++) {
              for (let x = 1; x < SIZE - 1; x++) {
                if ((x + y) % 2 !== 0) continue; // only red cells moved
                cells++;
                if (Math.abs(seen[y][x] - red[y][x]) <= 1e-4) matchesRed++;
                if (Math.abs(seen[y][x] - u[y][x]) <= 1e-6) matchesOriginal++;
              }
            }
            ctx.assert(
              matchesRed > cells - 5,
              matchesOriginal > cells - 5
                ? 'the black half was handed the ORIGINAL grid — chain the halves, black(red(guess)), or the black cells read last sweep’s reds and you have written Jacobi in two passes'
                : `the grid handed to the black kernel matches neither the red half’s output nor the original (${matchesRed} of ${cells} red cells agreed with the red half)`
            );
          },
        },
        {
          name: 'the first kernel is still the red half',
          run: async ctx => {
            const u = makeGuess(ctx.utils, SIZE, 4711);
            const red = await ctx.kernels[0](u);
            const refRed = halfSweepRef(u, 0);
            const hint = diagnoseGrid(red, refRed, 1e-4, halfSweepAlternatives(u));
            for (let y = 1; y < SIZE - 1; y += 3) {
              for (let x = 1; x < SIZE - 1; x += 3) {
                ctx.assertClose(red[y][x], refRed[y][x], 1e-4,
                  hint || `the FIRST kernel should be the red half — cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'on its own, the second kernel is a <em>black</em> half-sweep',
          run: async ctx => {
            // Checked in isolation, because in the chain it cannot be checked:
            // a black kernel that forgot its parity test recomputes the red
            // cells to the values they already hold, so black(red(u)) comes out
            // right and only the wasted half of the work gives it away.
            const u = makeGuess(ctx.utils, SIZE, 4711);
            const out = await ctx.kernels[1](u);
            const ref = halfSweepRef(u, 1);
            const hint = diagnoseGrid(out, ref, 1e-4, [
              [jacobiRef(u),
                'the black kernel has no parity test — it updates every interior cell, which doubles the work and only happens to give the right answer because the red half ran first'],
              [halfSweepRef(u, 0),
                'the black kernel is updating the RED cells — flip its parity test to keep the cells where (x + y) % 2 is 1'],
              [copyGrid(u),
                'the black kernel changed nothing — cells where (x + y) % 2 is 1 should take their neighbours’ average'],
            ]);
            for (let y = 1; y < SIZE - 1; y++) {
              for (let x = 1; x < SIZE - 1; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'chained, the two halves are one Gauss-Seidel sweep',
          run: async ctx => {
            const u = makeGuess(ctx.utils, SIZE, 4711);
            const out = await ctx.kernels[1](await ctx.kernels[0](u));
            const ref = redBlackRef(u);
            const hint = diagnoseGrid(out, ref, 1e-4, fullSweepAlternatives(u));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                const edge = y === 0 || x === 0 || y === SIZE - 1 || x === SIZE - 1;
                const message = edge ? boundaryHint(out[y][x], ref[y][x], 1e-5) : hint;
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, message || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const u = makeGuess(ctx.utils, SIZE, 5551212);
            const out = await ctx.kernels[1](await ctx.kernels[0](u));
            const ref = redBlackRef(u);
            const hint = diagnoseGrid(out, ref, 1e-4, fullSweepAlternatives(u));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Five sweeps from the plate, and the residual after them must beat
            // five Jacobi sweeps — the whole reason for the extra kernel.
            let u = makePlate(SIZE);
            let j = makePlate(SIZE);
            for (let i = 0; i < 5; i++) {
              u = await ctx.kernels[1](await ctx.kernels[0](u));
              j = jacobiRef(j);
            }
            const ref = (() => {
              let g = makePlate(SIZE);
              for (let i = 0; i < 5; i++) g = redBlackRef(g);
              return g;
            })();
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(u[y][x], ref[y][x], 2e-4, `cell [${y}][${x}] after 5 sweeps`);
              }
            }
            ctx.assert(rmsOf(residualRef(copyGrid(u))) < rmsOf(residualRef(j)),
              'after 5 sweeps red-black should already be closer than Jacobi — is the black half really reading the red half’s output?');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'count-the-sweeps',
      title: 'The Race',
      intro: `<p>Everything is wired: the Jacobi sweep, both halves of the red-black sweep, and
        the residual. Same plate, same starting guess of zero, same finish line — an RMS residual
        below <code>0.0005</code>. Count sweeps.</p>
        <p>"The same work per sweep" is the claim worth being careful about. One Jacobi sweep
        updates all 900 interior cells once. One red-black sweep updates all 900 once too, in two
        halves — the same averages, split across two kernel launches instead of one, with half the
        threads in each launch copying themselves. Sweeps are what we are comparing; the extra
        launch is what it costs. (A production kernel launches only the cells of one colour and
        pays nothing for the copies.)</p>
        <p>The residual is sampled every 5 sweeps rather than every sweep: the read-back is the
        expensive part of this loop, and five sweeps barely move the number.</p>`,
      goal: `<strong>Goal:</strong> fill in the red-black loop, and read the two sweep counts off
        the console — Jacobi should need about 275 sweeps and red-black about 170.`,
      requirements: [
        'Drive red-black with the same <code>sweepsToTolerance</code> helper the Jacobi baseline uses',
        'One red-black sweep is <code>black(red(u))</code> — both halves, in that order',
        'Count sweeps, not half-sweeps',
      ],
      hints: [
        {
          title: 'Hint 1 — the helper already does the counting',
          body: `<p><code>sweepsToTolerance</code> takes one argument: a function that turns a grid
            into the next grid. Because a kernel call is awaited, that function is
            <code>async</code> — Jacobi's is <code>async u =&gt; await sweep(u)</code>, and the
            helper awaits whatever it returns. Red-black's is one sweep — both halves — expressed
            the same way.</p>`,
        },
        {
          title: 'Hint 2 — the one-liner',
          body: `<pre><code>const redBlackSweeps = await sweepsToTolerance(
  async u =&gt; await black(await red(u))
);</code></pre>
<p>Both halves inside one call, so the helper counts a full sweep each time it runs it. Both
            <code>await</code>s matter: the outer one hands the helper a grid, and the inner one is
            what makes the black half read the red half's output instead of a Promise.</p>`,
        },
      ],
      transfer: `The shape of this measurement is the one that transfers, more than the numbers:
        an iterative solver is judged on iterations-to-tolerance, and a GPU implementation is
        judged on that <em>times</em> the cost of an iteration. Red-black buys fewer sweeps for one
        extra dispatch, which is a trade you make on nearly every platform; the same ledger decides
        whether SOR's relaxation factor, a Chebyshev acceleration or a full multigrid V-cycle is
        worth its complexity. Multigrid is where this ends up — and its inner smoother is the
        red-black sweep you just wrote.`,
      starterCode: `// Both solvers, one finish line. Jacobi's loop is done — write red-black's.
const gpu = new GPU({ mode });

${JACOBI_KERNEL}

${RED_KERNEL}

${BLACK_KERNEL}

${RESIDUAL_KERNEL}

${RMS_HELPER}

const TOL = 0.0005;
const CHECK_EVERY = 5;
const MAX_SWEEPS = 400;

// Sweeps until the RMS residual drops below TOL, checking every 5 sweeps.
// \`step\` turns one grid into the next.
async function sweepsToTolerance(step) {
  let u = plate;
  let sweeps = 0;
  while (sweeps < MAX_SWEEPS) {
    if (rmsOf(await residual(u)) < TOL) break;
    for (let i = 0; i < CHECK_EVERY; i++) {
      u = await step(u);
      sweeps++;
    }
  }
  return sweeps;
}

const jacobiSweeps = await sweepsToTolerance(async u => await sweep(u));
console.log('jacobi: converged in', jacobiSweeps, 'sweeps');

// TODO: one red-black sweep is the red half followed by the black half,
//       and the black half has to read what the red half just wrote.
//       Await both halves — an un-awaited call hands black a Promise.
const redBlackSweeps = await sweepsToTolerance(async u => u);
console.log('red-black: converged in', redBlackSweeps, 'sweeps');
`,
      solutionCode: `// Both solvers, one finish line. Jacobi's loop is done — write red-black's.
const gpu = new GPU({ mode });

${JACOBI_KERNEL}

${RED_KERNEL}

${BLACK_KERNEL}

${RESIDUAL_KERNEL}

${RMS_HELPER}

const TOL = 0.0005;
const CHECK_EVERY = 5;
const MAX_SWEEPS = 400;

// Sweeps until the RMS residual drops below TOL, checking every 5 sweeps.
// \`step\` turns one grid into the next.
async function sweepsToTolerance(step) {
  let u = plate;
  let sweeps = 0;
  while (sweeps < MAX_SWEEPS) {
    if (rmsOf(await residual(u)) < TOL) break;
    for (let i = 0; i < CHECK_EVERY; i++) {
      u = await step(u);
      sweeps++;
    }
  }
  return sweeps;
}

const jacobiSweeps = await sweepsToTolerance(async u => await sweep(u));
console.log('jacobi: converged in', jacobiSweeps, 'sweeps');

const redBlackSweeps = await sweepsToTolerance(async u => await black(await red(u)));
console.log('red-black: converged in', redBlackSweeps, 'sweeps');

console.log('red-black needed', Math.round(100 - (100 * redBlackSweeps) / jacobiSweeps),
  '% fewer sweeps for the same work per sweep');
`,
      inputs: () => ({ plate: makePlate(SIZE) }),
      publicTests: [
        {
          name: 'four kernels, in order: sweep, red, black, residual',
          run: async ctx => {
            assertRunOk(ctx);
            ctx.assert(ctx.kernels.length >= 4,
              `expected 4 kernels (sweep, red, black, residual), found ${ctx.kernels.length}`);
            const u = makeGuess(ctx.utils, SIZE, 4711);
            const oneSweep = await ctx.kernels[1](u);
            const refRed = halfSweepRef(u, 0);
            for (let y = 1; y < SIZE - 1; y += 5) {
              for (let x = 1; x < SIZE - 1; x += 5) {
                ctx.assertClose(oneSweep[y][x], refRed[y][x], 1e-4,
                  `kernel 2 should still be the red half-sweep — cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'Jacobi’s baseline reaches the tolerance in about 275 sweeps',
          run: async ctx => {
            const expected = sweepsToTolRef(makePlate(SIZE), jacobiRef);
            const got = loggedCount(ctx, 'jacobi');
            ctx.assert(got !== null,
              'the console never reported a jacobi sweep count — leave the baseline’s console.log in place');
            ctx.assert(
              Math.abs(got - expected) <= 10,
              got >= MAX_SWEEPS
                ? `jacobi hit the ${MAX_SWEEPS}-sweep cap without converging — is the residual kernel or rmsOf still intact?`
                : `jacobi should reach an RMS residual below ${TOL} in about ${expected} sweeps, the console says ${got}`
            );
          },
        },
        {
          name: 'red-black reaches the same tolerance in about 170 sweeps',
          run: async ctx => {
            const expected = sweepsToTolRef(makePlate(SIZE), redBlackRef);
            const got = loggedCount(ctx, 'red-black');
            ctx.assert(got !== null,
              'the console never reported a red-black sweep count — leave its console.log in place');
            const jacobiCount = sweepsToTolRef(makePlate(SIZE), jacobiRef);
            // A sweep COUNT is one number, and several different mistakes land
            // on the same one — measured here, not assumed: counting half-sweeps
            // and an under-relaxed weighted-Jacobi step (ω = 0.8) both finish in
            // exactly 340. So none of these three messages asserts a cause. Each
            // states the observation, which is certain, and then names the usual
            // culprit as a place to look. Do not "tighten" them into a claim.
            const hint = got >= MAX_SWEEPS
              ? `red-black hit the ${MAX_SWEEPS}-sweep cap: the residual never dropped below ${TOL}. One full sweep is black(red(u)) — check what the step function actually does to the grid`
              : Math.abs(got - jacobiCount) <= 10
                ? 'red-black finished in the same number of sweeps as Jacobi, so the step is not gaining anything from the colouring. The usual cause is a black half handed the ORIGINAL grid instead of the red half’s output — Jacobi in two passes'
                : Math.abs(got - 2 * expected) <= 10
                  ? 'that is almost exactly twice the expected number of sweeps. The usual cause is counting HALF-sweeps: one red-black sweep is the red half and the black half together, and it counts once'
                  : null;
            ctx.assert(
              Math.abs(got - expected) <= 10,
              hint || `red-black should reach an RMS residual below ${TOL} in about ${expected} sweeps, the console says ${got}`
            );
          },
        },
        {
          name: 'red-black wins by a wide margin — meaningfully fewer sweeps, same work per sweep',
          run: async ctx => {
            const jacobiCount = loggedCount(ctx, 'jacobi');
            const redBlackCount = loggedCount(ctx, 'red-black');
            ctx.assert(jacobiCount !== null && redBlackCount !== null,
              'both sweep counts should be printed to the console');
            // Measured at this size and tolerance: 170 vs 275, a ratio of 0.62.
            // 0.75 is the assertion because the ratio drifts toward 0.5 as the
            // tolerance tightens and toward 1 as it loosens — the margin has to
            // cover the measurement, not sit on top of it.
            ctx.assert(
              redBlackCount < jacobiCount * 0.75,
              `red-black took ${redBlackCount} sweeps against Jacobi's ${jacobiCount} — expected it to need under three quarters as many`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The kernels themselves, driven by the test: 20 red-black sweeps
            // from the plate have to match the reference exactly, so a run that
            // logged the right numbers with the wrong kernels cannot pass.
            let u = makePlate(SIZE);
            let ref = makePlate(SIZE);
            for (let i = 0; i < 20; i++) {
              u = await ctx.kernels[2](await ctx.kernels[1](u));
              ref = redBlackRef(ref);
            }
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(u[y][x], ref[y][x], 5e-4, `cell [${y}][${x}] after 20 red-black sweeps`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Same sweep budget, different residual: after 40 sweeps each,
            // red-black must already be measurably ahead. Measured margin at
            // this size: 0.002938 against 0.003957, a 26% gap — asserted at 10%.
            let j = makePlate(SIZE);
            let rb = makePlate(SIZE);
            for (let i = 0; i < 40; i++) {
              j = await ctx.kernels[0](j);
              rb = await ctx.kernels[2](await ctx.kernels[1](rb));
            }
            const jRes = rmsOf(copyGrid(await ctx.kernels[3](j)));
            const rbRes = rmsOf(copyGrid(await ctx.kernels[3](rb)));
            ctx.assert(
              rbRes < jRes * 0.9,
              `after 40 sweeps each, red-black's residual (${rbRes.toFixed(6)}) should be at least 10% below Jacobi's (${jRes.toFixed(6)})`
            );
          },
        },
      ],
    },
  ],
};
