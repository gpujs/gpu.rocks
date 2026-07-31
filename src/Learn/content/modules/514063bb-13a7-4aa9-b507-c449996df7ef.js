// Module: The Heat Equation & Stability — uuid 514063bb-13a7-4aa9-b507-c449996df7ef
// (short id 514063bb). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId — this module is new.
//
// Five tasks: one explicit forward-Euler step written as a weighted average →
// crossing dt = dx²/(2·D·dims) and watching the field detonate → sixteen step
// sizes raced against each other in one kernel to MEASURE the limit → backward
// Euler, which is a linear solve, driven by a self-contained Jacobi sweep →
// the honest capstone: unconditionally stable is not the same as accurate.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested numeric arrays as arguments, this.thread.* for indexing,
// this.constants.* for compile-time values, wrap-around (torus) edges as in
// Reaction–Diffusion. No kernel LOCAL shares a name with a constant
// (gpujs/gpu.js#858 — that throws on the CPU backend only). No booleans are
// stored in kernel variables. Every task passes in CPU mode and in GPU mode.
//
// PHYSICAL CONSTANTS. D = 8 and dx = 4 are not arbitrary: they are the smallest
// tidy pair that makes every classic slip land on a DIFFERENT number, which is
// what lets the probes name a specific mistake instead of shrugging. The limit
// dx²/(2·D·dims) = 0.5 against dx²/(2·D) = 1, dx/(2·D·dims) = 0.125 and
// (2·D·dims)/dx² = 2; the diffusion number D·dt/dx² = 0.1 against dt = 0.2,
// D·dt/dx = 0.4 and D·dt = 1.6. (The obvious D = dx = 2 collapses two of those
// pairs onto each other — "I forgot to square dx" and "I used dt for α" both
// give 0.2 — and a probe that cannot tell two mistakes apart must stay silent.)
// They also make α = D·dt/dx² = dt/2, so the α values below stay round.
//
// FLOAT DETERMINISM. Tests compute in float64; the GL backend computes in
// float32. Measured (Math.fround-per-operation emulation of the whole chain,
// scripts in the authoring session):
//   • 80 explicit steps at α = 0.1 on 32×32 — float32 vs float64: 1.7e-8
//   • 8 implicit steps × 25 Jacobi sweeps at α = 1 — float32 vs float64: 2.6e-8
//   • 90 explicit 1D steps at α ≤ 0.49 — agreed to every digit printed
// Every value assertion here uses a tolerance of 2e-4 or looser, i.e. four
// orders of magnitude of headroom, and no asserted value sits near a decision
// boundary. The BLOW-UP assertions never test a value: they test a magnitude
// threshold six orders of magnitude away from anything a stable run produces,
// written as !(|v| < limit) so that a NaN or an Infinity — which the GL backend
// reaches and float64 does not, at these step counts — counts as exploded
// rather than silently passing a `>` comparison.

// ---- the physics, shared by inputs(), starters, solutions and tests --------

const D = 8; // diffusivity
const DX = 4; // cell spacing
const DX2 = DX * DX; // 16

// The diffusion number. α = D·dt/dx² = dt/2 with these constants.
function alphaFor(dt) {
  return (D * dt) / DX2;
}

// The explicit stability limit: the centre weight of the update stencil is
// 1 − 2·dims·α, and it must not go negative.
function dtLimit(dims) {
  return DX2 / (2 * D * dims);
}

const DT_LIMIT_2D = dtLimit(2); // 0.5
const DT_LIMIT_1D = dtLimit(1); // 1

// Task 1: one step at dt = 0.2 → α = 0.1, comfortably inside the limit.
const T1_DT = 0.2;
const T1_ALPHA = alphaFor(T1_DT); // 0.1

// Task 2: 80 steps on each side of the line.
const T2_STEPS = 80;
const T2_SAFE_DT = 0.4 * DT_LIMIT_2D; // 0.2  → α = 0.1
const T2_WILD_DT = 1.6 * DT_LIMIT_2D; // 0.8  → α = 0.4

// Task 3: sixteen 1D rings, one dt each, 90 steps.
const T3_CELLS = 64;
const T3_ROWS = 16;
const T3_STEPS = 90;

// Task 4 / 5: implicit stepping at dt = 2 → α = 1, four times the explicit limit.
const T4_DT = 2;
const T4_ALPHA = alphaFor(T4_DT); // 1
const SWEEPS = 25;

// Task 5: reach T = 16 by two routes.
const T5_TIME = 16;
const T5_SMALL_DT = 0.2; // 80 steps
const T5_BIG_DT = 2; // 8 steps

function makeGrid(size, value) {
  const grid = new Array(size);
  for (let y = 0; y < size; y++) grid[y] = new Array(size).fill(value);
  return grid;
}

// A hot square in a cold field — the standard initial condition here. Its sharp
// edges carry every spatial frequency the grid can hold, including the
// checkerboard, which is the mode that decides stability.
function hotSquare(size, block, hot) {
  const grid = makeGrid(size, 0);
  const lo = (size - block) >> 1;
  for (let y = lo; y < lo + block; y++) {
    for (let x = lo; x < lo + block; x++) grid[y][x] = hot;
  }
  return grid;
}

function spikeGrid(size, y, x, value) {
  const grid = makeGrid(size, 0);
  grid[y][x] = value;
  return grid;
}

// The sixteen candidate step sizes of task 3: 0.42 … 1.62, straddling the 1D
// limit of 1 without ever landing on it. Rounded to 2 dp so the console table
// reads cleanly and no value sits at a decision boundary.
function makeDts() {
  const dts = new Array(T3_ROWS);
  for (let r = 0; r < T3_ROWS; r++) dts[r] = Math.round((0.42 + 0.08 * r) * 100) / 100;
  return dts;
}

// Sixteen identical 64-cell rings, each with one hot cell in the middle.
function makeRows() {
  const rows = new Array(T3_ROWS);
  for (let r = 0; r < T3_ROWS; r++) {
    rows[r] = new Array(T3_CELLS).fill(0);
    rows[r][T3_CELLS >> 1] = 1;
  }
  return rows;
}

const wrap = (i, n) => (i < 0 ? n - 1 : i > n - 1 ? 0 : i);

// ---- CPU references -------------------------------------------------------

// One explicit (forward-Euler) step on a torus.
function explicitStepRef(u, alpha) {
  const n = u.length;
  const out = makeGrid(n, 0);
  for (let y = 0; y < n; y++) {
    const yd = wrap(y - 1, n);
    const yu = wrap(y + 1, n);
    for (let x = 0; x < n; x++) {
      const c = u[y][x];
      const lap = u[y][wrap(x - 1, n)] + u[y][wrap(x + 1, n)] + u[yd][x] + u[yu][x] - 4 * c;
      out[y][x] = c + alpha * lap;
    }
  }
  return out;
}

function explicitRunRef(u, alpha, steps) {
  let field = u;
  for (let i = 0; i < steps; i++) field = explicitStepRef(field, alpha);
  return field;
}

// One Jacobi sweep of the backward-Euler system (I − α∇²)u' = u.
function jacobiSweepRef(uOld, guess, alpha) {
  const n = uOld.length;
  const out = makeGrid(n, 0);
  for (let y = 0; y < n; y++) {
    const yd = wrap(y - 1, n);
    const yu = wrap(y + 1, n);
    for (let x = 0; x < n; x++) {
      const neighbours =
        guess[y][wrap(x - 1, n)] + guess[y][wrap(x + 1, n)] + guess[yd][x] + guess[yu][x];
      out[y][x] = (uOld[y][x] + alpha * neighbours) / (1 + 4 * alpha);
    }
  }
  return out;
}

function implicitStepRef(u, alpha, sweeps) {
  let guess = u;
  for (let k = 0; k < sweeps; k++) guess = jacobiSweepRef(u, guess, alpha);
  return guess;
}

function implicitRunRef(u, alpha, sweeps, steps) {
  let field = u;
  for (let i = 0; i < steps; i++) field = implicitStepRef(field, alpha, sweeps);
  return field;
}

// The 1D scan of task 3: rows[r] is its own ring, stepped with its own dt.
function scanStepRef(rows, dts) {
  const n = rows[0].length;
  return rows.map((row, r) => {
    const a = alphaFor(dts[r]);
    const out = new Array(n);
    for (let x = 0; x < n; x++) {
      out[x] = row[x] + a * (row[wrap(x - 1, n)] + row[wrap(x + 1, n)] - 2 * row[x]);
    }
    return out;
  });
}

function scanRunRef(rows, dts, steps) {
  let field = rows;
  for (let i = 0; i < steps; i++) field = scanStepRef(field, dts);
  return field;
}

// The mistake task 3 is really guarding against: the 2D 5-point stencil, which
// would let row r read rows r ± 1 — a different simulation, with a different dt.
function scanStep2DRef(rows, dts) {
  const n = rows[0].length;
  const h = rows.length;
  return rows.map((row, r) => {
    const a = alphaFor(dts[r]);
    const out = new Array(n);
    for (let x = 0; x < n; x++) {
      out[x] =
        row[x] +
        a *
          (row[wrap(x - 1, n)] +
            row[wrap(x + 1, n)] +
            rows[wrap(r - 1, h)][x] +
            rows[wrap(r + 1, h)][x] -
            4 * row[x]);
    }
    return out;
  });
}

// ---- measuring a field ----------------------------------------------------
//
// NaN AND INFINITY MUST NOT BE ABLE TO HIDE. `if (v > m) m = v` silently skips
// a NaN, because every comparison with NaN is false — a max written that way
// reports a small number for a field that is already dead. Every magnitude
// helper below therefore asks `!(a <= m)`, which is TRUE for NaN, so a single
// poisoned cell propagates into the answer. This matters because the GL backend
// reaches Infinity where float64 keeps counting: the two backends disagree
// about the exact numbers in an exploded field and must still agree that it
// exploded.

function maxAbsRow(row) {
  let m = 0;
  for (let x = 0; x < row.length; x++) {
    const a = Math.abs(row[x]);
    if (!(a <= m)) m = a;
  }
  return m;
}

function maxAbsGrid(grid) {
  let m = 0;
  for (let y = 0; y < grid.length; y++) {
    const a = maxAbsRow(grid[y]);
    if (!(a <= m)) m = a;
  }
  return m;
}

function sumGrid(grid) {
  let total = 0;
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) total += grid[y][x];
  }
  return total;
}

function maxGrid(grid) {
  let m = -Infinity;
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) if (!(grid[y][x] <= m)) m = grid[y][x];
  }
  return m;
}

function minGrid(grid) {
  let m = Infinity;
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) if (!(grid[y][x] >= m)) m = grid[y][x];
  }
  return m;
}

// "Did it survive?" — true only when EVERY cell is finite and under `limit`.
// A NaN fails `<`, so it lands on the exploded side, which is where it belongs.
function boundedBy(grid, limit) {
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) {
      if (!(Math.abs(grid[y][x]) < limit)) return false;
    }
  }
  return true;
}

// ---- console helpers ------------------------------------------------------

function loggedNumbers(logs) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

// Numbers from the console lines that mention a particular word — so "the
// measured limit" can be checked against the line that claims to report it,
// not against any number the program happened to print.
function numbersInLines(logs, re) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text || !re.test(line.text)) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so cells where two candidates coincide (a flat
// field, where every one of these formulas agrees) stay silent, as do
// observations matching probes that disagree with each other. A wrong diagnosis
// is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Grid form: a probe must predict EVERY cell (and differ from the right answer
// somewhere) before it may speak. One matching cell is weak evidence when the
// candidates are whole formulas over the same field.
function diagnoseGrid(out, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (let y = 0; y < expected.length; y++) {
        for (let x = 0; x < expected[y].length; x++) {
          const c = value[y][x];
          if (!(out[y] && Math.abs(out[y][x] - c) <= eps)) return false;
          if (Math.abs(expected[y][x] - c) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Recover the α a kernel was actually built with, whatever else its body does:
// step a field holding a single 1 and every direct neighbour of that cell gains
// exactly α. One multiply, no accumulation, so it is exact on both backends.
// NOTE FOR CALLERS: invoking a kernel overwrites its .lastArgs — read those
// FIRST if the test also needs the field the learner's own loop finished on.
function alphaOfKernel(k, size) {
  try {
    const out = k(spikeGrid(size, 2, 2, 1));
    return out && out[2] && typeof out[2][3] === 'number' ? out[2][3] : NaN;
  } catch (e) {
    return NaN;
  }
}

// ---- task 1 probes --------------------------------------------------------

// The explicit update with α wrong, missing, or applied to the wrong thing.
// `lap` and `c` come from the reference, so these are exact per cell.
function stepProbes(field, alpha) {
  const c = field;
  const lapOf = grid => {
    const n = grid.length;
    const out = makeGrid(n, 0);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        out[y][x] =
          grid[y][wrap(x - 1, n)] +
          grid[y][wrap(x + 1, n)] +
          grid[wrap(y - 1, n)][x] +
          grid[wrap(y + 1, n)][x] -
          4 * grid[y][x];
      }
    }
    return out;
  };
  const lap = lapOf(field);
  const combine = fn => lap.map((row, y) => row.map((l, x) => fn(c[y][x], l)));
  return [
    [combine((u, l) => u + l), 'the Laplacian went in unscaled — a step is u + α·∇²u, and α is D·dt/dx²'],
    [combine((u, l) => alpha * l), 'that is the CHANGE, not the new value — the step ADDS it to the old field: u + α·∇²u'],
    [combine((u, l) => u - alpha * l), 'the sign is flipped — heat flows toward the neighbours, so the Laplacian is added, not subtracted'],
    [combine((u, l) => u + T1_DT * l), `dt on its own is not the diffusion number — α = D·dt/dx² is ${T1_ALPHA} here, not ${T1_DT}`],
    [combine((u, l) => u + D * T1_DT * l), 'dx² is missing from the denominator — α = D·dt/dx², and dx is squared'],
    [combine((u, l) => u + (D * T1_DT) / DX * l), 'dx is there but not squared — α = D·dt/(dx·dx)'],
    [combine(u => u), 'the field came back unchanged — the update never reached the return value'],
  ];
}

// ---- task 2 probes --------------------------------------------------------

// The stability limit, mis-derived. D = 8 and dx = 4 are picked so these land on
// different numbers (0.5 correct, 1, 0.125, 2, 512); the two ways of losing a
// factor from the denominator both give 1, so they share one sentence rather
// than cancelling each other into silence.
function dtMaxProbes() {
  const missing =
    'a factor is missing from the denominator 2·D·dims — a 2D grid has four neighbours, ' +
    'so dims = 2 and the whole denominator is 2 · 8 · 2 = 32';
  return [
    [DX2 / (2 * D), missing],
    [DX2 / (2 * D * 1), missing],
    [DX / (2 * D * 2), 'that is dx, not dx² — the limit falls with the SQUARE of the cell spacing, which is why refining a mesh is so expensive'],
    [(2 * D * 2) / DX2, 'the fraction is upside down — dt ≤ dx² / (2·D·dims)'],
    [DX2 * 2 * D * 2, 'those factors are multiplied where they should divide — dt ≤ dx² / (2·D·dims)'],
  ];
}

// ---- task 3 probes --------------------------------------------------------

// One step of the scan, done three wrong ways. All three are exact references,
// compared cell for cell by diagnoseGrid.
function scanProbes(rows, dts) {
  const flat = dts.map(() => dts[0]);
  return [
    [scanStep2DRef(rows, dts),
      'that is the 2D five-point stencil — but the rows are sixteen INDEPENDENT simulations, and the row above yours is running a different dt. The Laplacian here is left + right − 2·centre, along x only'],
    [scanStepRef(rows, flat),
      "every row used dts[0] — the row's own step size is dts[this.thread.y]"],
    [scanStepRef(rows, dts.map(dt => dt * DX2 / D)),
      'dts holds dt, not α — the kernel still has to form α = D·dt/dx² before it steps'],
  ];
}

// ---- task 4 probes --------------------------------------------------------

// One Jacobi sweep, mis-assembled. Called with guess ≠ uOld, since when the two
// coincide (the first sweep) several of these agree and must stay silent.
function sweepProbes(uOld, guess, alpha) {
  const n = uOld.length;
  const build = fn => {
    const out = makeGrid(n, 0);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const neighbours =
          guess[y][wrap(x - 1, n)] + guess[y][wrap(x + 1, n)] +
          guess[wrap(y - 1, n)][x] + guess[wrap(y + 1, n)][x];
        out[y][x] = fn(uOld[y][x], guess[y][x], neighbours);
      }
    }
    return out;
  };
  return [
    [build((old, g, s) => (g + alpha * s) / (1 + 4 * alpha)),
      'the centre term came from the current iterate instead of the OLD field. Drop uOld and the iteration forgets which time level it started from — it converges to the steady state (∇²u = 0, a flat field), not to one step of the heat equation'],
    [build((old, g, s) => old + alpha * s),
      'the division is missing — collecting u′ on the left gives (1 + 4α)·u′ = u + α·(neighbours), so the whole right-hand side is divided by 1 + 4α'],
    [build((old, g, s) => (old + alpha * s) / (4 * alpha)),
      'the 1 is missing from the denominator — it is 1 + 4α, and that 1 is the cell’s own coefficient'],
    [build((old, g, s) => (old - alpha * s) / (1 - 4 * alpha)),
      'the signs are those of a forward step — backward Euler moves the Laplacian to the LEFT of the equals sign, so both signs turn over: (1 + 4α)·u′ = u + α·(neighbours)'],
    [build((old, g, s) => old),
      'the old field came back untouched — the sweep never reached the return value'],
  ];
}

// ---- task 5 helpers and probes --------------------------------------------

// Task 5 builds two kernels of different ARITY — explicitStep(u, alpha) and
// sweep(uOld, guess, alpha) — so they are told apart by their signature rather
// than by creation order or by trial invocation. Calling a kernel with the
// wrong number of arguments throws ("arguments are miss-aligned"), and a
// try/catch around that would be a coin flip; argumentNames is exact.
function kernelArity(k) {
  const built = k && k.kernel;
  return built && Array.isArray(built.argumentNames) ? built.argumentNames.length : -1;
}

function findByArity(ctx, arity) {
  return ctx.kernels.find(k => kernelArity(k) === arity) || null;
}

// The two ways the step counts go wrong. Read from the lines that report each
// run, so a number that merely happens to appear elsewhere cannot fire them.
function stepCountHint(logs) {
  const small = numbersInLines(logs, /explicit, dt/i);
  const big = numbersInLines(logs, /implicit, dt/i);
  const has = (list, v) => list.some(n => Math.abs(n - v) < 1e-9);
  if (has(small, T5_TIME / T5_BIG_DT) && has(big, T5_TIME / T5_SMALL_DT)) {
    return 'the two step counts are swapped — the SMALLER step needs MORE of them: 16 / 0.2 = 80 and 16 / 2 = 8';
  }
  if (has(small, T5_TIME * T5_SMALL_DT) || has(big, T5_TIME * T5_BIG_DT)) {
    return 'the step count is T · dt there; it is T / dt — a step of 0.2 needs 80 of them to cover 16 units of time';
  }
  return null;
}

export default {
  uuid: '514063bb-13a7-4aa9-b507-c449996df7ef',
  version: 1,
  slug: 'heat-and-stability',
  title: 'The Heat Equation & Stability',
  blurb: 'Why a correct-looking simulation explodes — the step-size limit, and the implicit step that ignores it.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'explicit-step',
      title: 'One Explicit Step',
      intro: `<p>The heat equation is the simplest interesting PDE there is:
        <code>∂u/∂t = D·∇²u</code>. Every point drifts toward the average of what surrounds
        it, at a rate set by the diffusivity <code>D</code>. Discretise the right-hand side
        with the 5-point stencil — <code>left + right + up + down − 4·centre</code>, divided
        by <code>dx²</code>, which Reaction–Diffusion derives in full — take a plain
        forward-Euler step in time, and the whole solver is one line:</p>
<pre><code>u' = u + α·(left + right + up + down − 4u)
with  α = D·dt/dx²</code></pre>
        <p>That single dimensionless number <code>α</code>, the <strong>diffusion
        number</strong>, is what this module is about. Collect the terms and the step turns
        out not to be an addition at all. It is an <em>average</em>:</p>
<pre><code>u' = (1 − 4α)·u
     + α·left + α·right + α·up + α·down</code></pre>
        <p>Five weights that add to exactly one. Notice the shape of that centre weight,
        <code>1 − 4α</code> — the next task is about what happens when it goes negative.</p>`,
      goal: `<strong>Goal:</strong> compute the diffusion number <code>α</code> from
        <code>D</code>, <code>dt</code> and <code>dx</code>, and return one explicit step of
        <code>field</code>.`,
      requirements: [
        'Compute <code>ALPHA = D * dt / (dx * dx)</code> in JavaScript — it reaches the kernel as a constant',
        'The kernel returns this cell’s new value: the old one plus <code>α</code> times the Laplacian',
        'The stencil and its wrap-around edges are already written — the world is a torus',
      ],
      hints: [
        {
          title: 'Hint 1 — dx is squared',
          body: `<p><code>α = D·dt/dx²</code> — the cell spacing appears <em>squared</em>, because a
            second derivative is a difference of differences. With <code>D = 8</code>,
            <code>dt = 0.2</code> and <code>dx = 4</code> that comes to <code>0.1</code>.</p>`,
        },
        {
          title: 'Hint 2 — the whole return',
          body: `<p><code>lap</code> and <code>c</code> are already in scope, so the body is one line:</p>
<pre><code>return c + this.constants.alpha * lap;</code></pre>`,
        },
      ],
      transfer: `This three-line update is, almost character for character, the innermost loop of
        every explicit finite-difference solver on every platform: a CUDA kernel with one thread
        per cell, a WGSL compute shader reading and writing a storage texture, a Metal kernel
        tiling the grid into threadgroups. What differs between them is memory layout and how the
        halo is exchanged — never the arithmetic.`,
      starterCode: `// One forward-Euler step of the heat equation on a 48×48 torus.
const gpu = new GPU({ mode });

const D = 8;     // diffusivity
const dx = 4;    // cell spacing
const dt = 0.2;  // time step

// TODO: the diffusion number, alpha = D * dt / dx²
const ALPHA = 0;

const step = gpu.createKernel(function (u) {
  const x = this.thread.x;
  const y = this.thread.y;
  // the 5-point stencil, wrapped at the edges (a torus, as in Reaction–Diffusion)
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const c = u[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * c;
  // TODO: return the new value — the old one plus alpha times the Laplacian
  return c;
}, {
  output: [48, 48],
  constants: { size: 48, alpha: ALPHA },
});

const next = step(field);
console.log('alpha:', ALPHA, ' centre weight 1 - 4*alpha:', 1 - 4 * ALPHA);
console.log('a hot cell on the block edge, was 1, is now:', next[24][20]);
`,
      solutionCode: `// One forward-Euler step of the heat equation on a 48×48 torus.
const gpu = new GPU({ mode });

const D = 8;     // diffusivity
const dx = 4;    // cell spacing
const dt = 0.2;  // time step

const ALPHA = D * dt / (dx * dx);

const step = gpu.createKernel(function (u) {
  const x = this.thread.x;
  const y = this.thread.y;
  // the 5-point stencil, wrapped at the edges (a torus, as in Reaction–Diffusion)
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const c = u[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * c;
  return c + this.constants.alpha * lap;
}, {
  output: [48, 48],
  constants: { size: 48, alpha: ALPHA },
});

const next = step(field);
console.log('alpha:', ALPHA, ' centre weight 1 - 4*alpha:', 1 - 4 * ALPHA);
console.log('a hot cell on the block edge, was 1, is now:', next[24][20]);
`,
      inputs: () => ({ field: hotSquare(48, 8, 1) }),
      publicTests: [
        {
          name: 'a flat field is a fixed point — nothing to diffuse, nothing changes',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const flat = makeGrid(48, 0.7);
            const out = ctx.kernel(flat);
            ctx.assert(
              out && out.length === 48 && out[0] && out[0].length === 48,
              `expected a 48×48 result, got ${out && out.length} rows`
            );
            for (let y = 0; y < 48; y += 7) {
              for (let x = 0; x < 48; x += 7) {
                ctx.assertClose(out[y][x], 0.7, 1e-5, `cell [${y}][${x}] of a flat field`);
              }
            }
          },
        },
        {
          name: 'the diffusion number is <code>α = D·dt/dx² = 0.1</code>',
          run: async ctx => {
            // A single hot cell: whatever else the body does, each of its four
            // neighbours gains exactly α and the cell itself keeps 1 − 4α.
            const spike = spikeGrid(48, 10, 10, 1);
            const out = ctx.kernel(spike);
            const hint = diagnose(out[10][11], T1_ALPHA, 2e-4, [
              [T1_DT, `that is dt, not α — the diffusion number is D·dt/dx² = ${T1_ALPHA}`],
              [D * T1_DT, 'dx² never made it into the denominator — α = D·dt/dx²'],
              [(D * T1_DT) / DX, 'dx is there but not squared — α = D·dt/(dx·dx)'],
              [-T1_ALPHA, 'the sign is flipped — a neighbour of a hot cell GAINS α'],
              [0, 'the neighbour of a hot cell gained nothing — is ALPHA still 0?'],
            ]);
            ctx.assertClose(out[10][11], T1_ALPHA, 2e-4, hint || 'the neighbour of a single hot cell gains α');
            ctx.assertClose(out[10][10], 1 - 4 * T1_ALPHA, 2e-4, 'the hot cell itself keeps 1 − 4α');
          },
        },
        {
          name: 'one step of <code>field</code> matches <code>u + α·∇²u</code> everywhere',
          run: async ctx => {
            const field = hotSquare(48, 8, 1);
            const out = ctx.kernel(field);
            const expected = explicitStepRef(field, T1_ALPHA);
            const hint = diagnoseGrid(out, expected, 2e-4, stepProbes(field, T1_ALPHA));
            for (let y = 0; y < 48; y++) {
              for (let x = 0; x < 48; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 2e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different block size, plus the torus conservation law: diffusion
            // moves heat around, it never creates or destroys any.
            const field = hotSquare(48, 14, 0.5);
            const out = ctx.kernel(field);
            const expected = explicitStepRef(field, T1_ALPHA);
            const hint = diagnoseGrid(out, expected, 2e-4, stepProbes(field, T1_ALPHA));
            for (let y = 0; y < 48; y++) {
              for (let x = 0; x < 48; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 2e-4, hint || `cell [${y}][${x}]`);
              }
            }
            ctx.assertClose(sumGrid(out), sumGrid(field), 5e-3, 'total heat on a closed torus');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'blow-it-up',
      title: 'Cross the Line',
      intro: `<p>Nothing about the kernel you just wrote is wrong. Hand it a step size that is
        slightly too big and it will still compute exactly what you asked for — and the field
        will be at 10²⁴ in eighty steps.</p>
        <p>Look again at that centre weight, <code>1 − 4α</code>. At <code>α = 0.1</code> it is
        <code>+0.6</code>, the new value really is an average of five old ones, and an average
        can never leave the range of the things it averaged. At <code>α = 0.4</code> it is
        <code>−0.6</code>, and "average" has become a lie: a cell that sits above its
        neighbours is now pushed <em>further</em> above them every step. In
        <code>dims</code> dimensions the centre weight is <code>1 − 2·dims·α</code>, so the
        tipping point is <code>α = 1/(2·dims)</code> — which, written in the quantities you
        actually control, is the limit every explicit solver lives under:</p>
<pre><code>dt  ≤  dx² / (2 · D · dims)
       dims = 2 on a 2D grid</code></pre>
        <p>Past it, the fastest pattern the grid can hold — a checkerboard of alternating hot
        and cold cells — is multiplied by <code>|1 − 8α|</code> every step instead of damped.
        At <code>α = 0.4</code> that is <strong>2.2× per step</strong>, and there is always
        some checkerboard in there, if only from rounding. Ten steps: ×2,700. Eighty steps:
        the run below.</p>`,
      goal: `<strong>Goal:</strong> work out <code>dtMax</code>, then run the same eighty-step
        simulation twice — once safely inside the limit, once past it — and watch the console.
        The stepping loop is written for you: N-Body already made a meal of <em>how</em> to
        advance a simulation, and the only thing that matters here is how far each step goes.`,
      requirements: [
        'Compute <code>dtMax = dx * dx / (2 * D * DIMS)</code> — it is <code>0.5</code> for this grid',
        'Call <code>run</code> once at <code>0.4 * dtMax</code> and once at <code>1.6 * dtMax</code>',
        'Leave <code>STEPS</code> at 80 — the trace prints the hottest cell every 20 steps',
      ],
      hints: [
        {
          title: 'Hint 1 — where the numbers come from',
          body: `<p>Each of the <code>2 · dims</code> neighbours takes an <code>α</code>-sized bite
            out of the centre, so the centre keeps <code>1 − 2·dims·α</code>. Set that to zero,
            substitute <code>α = D·dt/dx²</code>, and solve for <code>dt</code>.</p>`,
        },
        {
          title: 'Hint 2 — the two runs',
          body: `<pre><code>const dtMax = dx * dx / (2 * D * DIMS);
run('SAFE', 0.4 * dtMax);
run('PAST THE LINE', 1.6 * dtMax);</code></pre>`,
        },
      ],
      transfer: `Every production explicit solver computes this number and refuses to exceed it:
        CFL conditions in fluid codes, the diffusion-number check in a thermal simulation, the
        substepping loop in a cloth or fluid solver on the GPU. It is also why GPU simulations
        so often become <em>launch-bound</em> — halving <code>dx</code> to sharpen a picture
        quarters the legal <code>dt</code>, so the same second of simulated time costs four
        times as many kernel launches.`,
      starterCode: `// Two runs of the same simulation. Only the step size differs.
const gpu = new GPU({ mode });

const D = 8;      // diffusivity
const dx = 4;     // cell spacing
const DIMS = 2;   // a 2D grid: four neighbours
const STEPS = 80;

// TODO: the explicit stability limit — dt ≤ dx² / (2 · D · dims)
const dtMax = 0;

// max |u| over the field, written so a NaN cannot hide: every comparison with
// NaN is false, so \`if (a > m)\` would skip it. \`!(a <= m)\` is true for NaN.
function hottest(u) {
  let m = 0;
  for (let y = 0; y < u.length; y++) {
    for (let x = 0; x < u[y].length; x++) {
      const a = Math.abs(u[y][x]);
      if (!(a <= m)) m = a;
    }
  }
  return m;
}

function run(label, dt) {
  const alpha = D * dt / (dx * dx);
  const step = gpu.createKernel(function (u) {
    const x = this.thread.x;
    const y = this.thread.y;
    let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
    let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
    let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
    let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
    const c = u[y][x];
    const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * c;
    return c + this.constants.alpha * lap;
  }, { output: [48, 48], constants: { size: 48, alpha } });

  console.log(label, '— dt =', dt, ' alpha =', alpha, ' centre weight =', 1 - 4 * alpha);
  let u = seed;
  for (let i = 1; i <= STEPS; i++) {
    u = step(u);
    if (i % 20 === 0) console.log('   step', i, '→ hottest |u| =', hottest(u));
  }
  return u;
}

console.log('stability limit: dt <=', dtMax);

// TODO: run twice — at 0.4 * dtMax, then at 1.6 * dtMax
`,
      solutionCode: `// Two runs of the same simulation. Only the step size differs.
const gpu = new GPU({ mode });

const D = 8;      // diffusivity
const dx = 4;     // cell spacing
const DIMS = 2;   // a 2D grid: four neighbours
const STEPS = 80;

const dtMax = dx * dx / (2 * D * DIMS);

// max |u| over the field, written so a NaN cannot hide: every comparison with
// NaN is false, so \`if (a > m)\` would skip it. \`!(a <= m)\` is true for NaN.
function hottest(u) {
  let m = 0;
  for (let y = 0; y < u.length; y++) {
    for (let x = 0; x < u[y].length; x++) {
      const a = Math.abs(u[y][x]);
      if (!(a <= m)) m = a;
    }
  }
  return m;
}

function run(label, dt) {
  const alpha = D * dt / (dx * dx);
  const step = gpu.createKernel(function (u) {
    const x = this.thread.x;
    const y = this.thread.y;
    let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
    let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
    let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
    let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
    const c = u[y][x];
    const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * c;
    return c + this.constants.alpha * lap;
  }, { output: [48, 48], constants: { size: 48, alpha } });

  console.log(label, '— dt =', dt, ' alpha =', alpha, ' centre weight =', 1 - 4 * alpha);
  let u = seed;
  for (let i = 1; i <= STEPS; i++) {
    u = step(u);
    if (i % 20 === 0) console.log('   step', i, '→ hottest |u| =', hottest(u));
  }
  return u;
}

console.log('stability limit: dt <=', dtMax);

run('SAFE', 0.4 * dtMax);
run('PAST THE LINE', 1.6 * dtMax);
`,
      inputs: () => ({ seed: hotSquare(48, 8, 1) }),
      publicTests: [
        {
          name: 'the limit is <code>dx² / (2·D·dims)</code>, and it is logged',
          run: async ctx => {
            const nums = numbersInLines(ctx.logs, /limit/i);
            const hit = nums.find(v => Math.abs(v - DT_LIMIT_2D) <= 1e-6);
            const near = nums.map(v => diagnose(v, DT_LIMIT_2D, 1e-6, dtMaxProbes())).find(Boolean);
            ctx.assert(
              hit !== undefined,
              near ||
                `the stability limit line should report ${DT_LIMIT_2D} — dt ≤ dx² / (2·D·dims) ` +
                  `with dx = ${DX}, D = ${D}, dims = 2`
            );
          },
        },
        {
          name: 'two runs, one on each side of the line',
          run: async ctx => {
            ctx.assert(
              ctx.kernels.length >= 2,
              `expected two runs (two kernels, one per step size), found ${ctx.kernels.length}` +
                (ctx.kernels.length === 0 ? ' — run() is never called' : '')
            );
            const alphas = ctx.kernels.map(k => alphaOfKernel(k, 48)).filter(a => !Number.isNaN(a));
            ctx.assert(alphas.length >= 2, 'could not read a diffusion number back out of two kernels');
            const lo = Math.min(...alphas);
            const hi = Math.max(...alphas);
            ctx.assertClose(lo, alphaFor(T2_SAFE_DT), 2e-4,
              `the safe run should use dt = 0.4 · dtMax = ${T2_SAFE_DT}, i.e. α = ${alphaFor(T2_SAFE_DT)}`);
            ctx.assertClose(hi, alphaFor(T2_WILD_DT), 2e-4,
              `the reckless run should use dt = 1.6 · dtMax = ${T2_WILD_DT}, i.e. α = ${alphaFor(T2_WILD_DT)}`);
            ctx.assert(hi > 0.25, 'neither run crossed the line — one of them has to be past α = 1/4');
          },
        },
        {
          name: 'below the line the field only ever cools; above it, it detonates',
          run: async ctx => {
            // Both runs are driven HERE rather than read off .lastArgs: the
            // tests share the run's kernels, so by the time this one executes an
            // earlier test has already re-invoked them and .lastArgs is its
            // probe, not the learner's last step.
            const alphas = ctx.kernels.map(k => alphaOfKernel(k, 48));
            const safeK = ctx.kernels.find((k, i) => alphas[i] < 0.25);
            const wildK = ctx.kernels.find((k, i) => alphas[i] > 0.25);
            ctx.assert(safeK, 'no run inside the limit — one of the two step sizes must be below dtMax');
            ctx.assert(wildK, 'no run past the limit — one of the two step sizes must be above dtMax');
            const seed = hotSquare(48, 8, 1);

            let safe = seed;
            for (let i = 0; i < T2_STEPS; i++) safe = safeK(safe);
            ctx.assert(
              boundedBy(safe, 1.001),
              `the stable run left the range of its own initial data (hottest |u| = ${maxAbsGrid(safe)}) — ` +
                'with a positive centre weight every new value is an average of five old ones, so it cannot'
            );

            let wild = seed;
            for (let i = 0; i < T2_STEPS; i++) wild = wildK(wild);
            ctx.assert(
              !boundedBy(wild, 1e6),
              `the run past the limit should have blown up, but its hottest |u| is only ` +
                `${maxAbsGrid(wild)} — is the second run really at 1.6 · dtMax?`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Drive both kernels from a fresh seed, and check the growth is the
            // geometric explosion the theory predicts rather than a one-off spike.
            const alphas = ctx.kernels.map(k => alphaOfKernel(k, 48));
            const safeK = ctx.kernels.find((k, i) => alphas[i] < 0.25);
            const wildK = ctx.kernels.find((k, i) => alphas[i] > 0.25);
            ctx.assert(safeK && wildK, 'expected one kernel inside the limit and one past it');
            const seed = hotSquare(48, 12, 1);

            let u = seed;
            for (let i = 0; i < 40; i++) u = safeK(u);
            const ref = explicitRunRef(seed, alphaFor(T2_SAFE_DT), 40);
            for (let y = 0; y < 48; y += 3) {
              for (let x = 0; x < 48; x += 3) {
                ctx.assertClose(u[y][x], ref[y][x], 2e-4, `stable run, cell [${y}][${x}] after 40 steps`);
              }
            }
            ctx.assert(boundedBy(u, 1.001), 'the stable run must stay inside its own initial range');
            ctx.assertClose(sumGrid(u), sumGrid(seed), 5e-3, 'heat is conserved on a torus');

            let v = seed;
            const trace = [];
            for (let i = 1; i <= 60; i++) {
              v = wildK(v);
              if (i % 20 === 0) trace.push(maxAbsGrid(v));
            }
            ctx.assert(!boundedBy(v, 1e6), 'the run past the limit did not blow up');
            ctx.assert(
              !(trace[0] < 1) && !(trace[1] < trace[0]) && !(trace[2] < trace[1]),
              `an unstable run grows every step; this trace does not: ${trace.join(', ')}`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'stability-scan',
      title: 'Sixteen Step Sizes at Once',
      intro: `<p>You have been told where the line is. Now measure it — and measure it the way
        a GPU makes cheap: not by running sixteen simulations one after another, but by
        running <strong>all sixteen in the same kernel launch</strong>.</p>
        <p>The grid below is 64 columns by 16 rows, and each row is its own universe: a
        64-cell ring, seeded with one hot cell, stepped with <em>its own</em>
        <code>dts[row]</code>. Nothing couples the rows, so the Laplacian here is the 1D one,
        <code>left + right − 2·centre</code>, along <code>x</code> only. Reach for the
        familiar 5-point stencil and row 3's instability leaks into row 4.</p>
        <p>Rings are one-dimensional, so <code>dims = 1</code> and the limit moves:
        <code>dt ≤ dx²/(2·D·1) = 1</code>, twice what it was on the 2D grid. That factor is
        not decoration — it is the number of neighbours taking a bite out of the centre. After
        90 steps the answer is unmissable: the stable rows have flattened to a few hundredths,
        and the row on the other side of the line is at 10³.</p>`,
      goal: `<strong>Goal:</strong> step all sixteen rings 90 times, then report the largest
        <code>dt</code> whose row is still under 1 — and compare it with
        <code>dx²/(2·D·1)</code>.`,
      requirements: [
        'The kernel forms <em>this row’s</em> diffusion number from <code>dts[this.thread.y]</code>',
        'The Laplacian runs along <code>x</code> only: <code>left + right − 2·centre</code> — rows must not read each other',
        'After 90 steps, find each row’s largest <code>|u|</code>; a row survived if that is below 1',
        'Log the largest surviving <code>dt</code> on a line that says <code>measured</code>, and the predicted limit beside it',
      ],
      hints: [
        {
          title: 'Hint 1 — which dt is mine?',
          body: `<p><code>this.thread.y</code> is the row, so <code>dts[this.thread.y]</code> is this
            ring's step size. Turn it into a diffusion number the same way as before:</p>
<pre><code>const a = this.constants.diff * dts[r]
  / (this.constants.dx * this.constants.dx);</code></pre>`,
        },
        {
          title: 'Hint 2 — one dimension, two neighbours',
          body: `<p><code>return u[r][x] + a * (u[r][xl] + u[r][xr] - 2 * u[r][x]);</code> — note the
            <code>2</code>, not <code>4</code>. Only the row index <code>r</code> never varies.</p>`,
        },
        {
          title: 'Hint 3 — scanning the rows afterwards',
          body: `<p>Plain JavaScript on the finished grid:</p>
<pre><code>for (let r = 0; r &lt; ROWS; r++) {
  let m = 0;
  for (let x = 0; x &lt; CELLS; x++) {
    const a = Math.abs(u[r][x]);
    if (!(a &lt;= m)) m = a;   // so a NaN cannot hide
  }
  if (m &lt; 1 &amp;&amp; dts[r] &gt; measured) {
    measured = dts[r];
  }
}</code></pre>`,
        },
      ],
      transfer: `Sweeping a parameter by giving it an axis of the launch grid is the GPU's answer
        to "try them all": CUDA codes run a batch of independent problems as extra blocks, WebGPU
        dispatches a third workgroup dimension over configurations, and every autotuner on every
        platform is this shape. It is also how a solver picks its own step size in production —
        run the candidate, look at what came back, and back off.`,
      starterCode: `// Sixteen simulations in one grid: row r is a 64-cell ring with its own dt.
const gpu = new GPU({ mode });

const D = 8;
const dx = 4;
const CELLS = 64;
const ROWS = 16;
const STEPS = 90;

const step = gpu.createKernel(function (u, dts) {
  const x = this.thread.x;
  const r = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.cells - 1;
  let xr = x + 1; if (xr > this.constants.cells - 1) xr = 0;
  // TODO: this row's diffusion number, then ONE 1D step of ring r:
  //   a = diff * dts[r] / (dx * dx)     (both live in this.constants)
  //   u[r][x] + a * (left + right - 2 * centre)
  return u[r][x];
}, {
  output: [64, 16],
  constants: { cells: 64, diff: 8, dx: 4 },
});

let u = seed;
for (let i = 0; i < STEPS; i++) u = step(u, dts);

// TODO: for each row print the largest |u| left, and keep the largest dt
// whose row stayed below 1.
let measured = 0;

console.log('measured limit: dt <=', measured);
console.log('predicted:      dt <=', dx * dx / (2 * D * 1));
`,
      solutionCode: `// Sixteen simulations in one grid: row r is a 64-cell ring with its own dt.
const gpu = new GPU({ mode });

const D = 8;
const dx = 4;
const CELLS = 64;
const ROWS = 16;
const STEPS = 90;

const step = gpu.createKernel(function (u, dts) {
  const x = this.thread.x;
  const r = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.cells - 1;
  let xr = x + 1; if (xr > this.constants.cells - 1) xr = 0;
  const a = this.constants.diff * dts[r] / (this.constants.dx * this.constants.dx);
  return u[r][x] + a * (u[r][xl] + u[r][xr] - 2 * u[r][x]);
}, {
  output: [64, 16],
  constants: { cells: 64, diff: 8, dx: 4 },
});

let u = seed;
for (let i = 0; i < STEPS; i++) u = step(u, dts);

let measured = 0;
for (let r = 0; r < ROWS; r++) {
  let m = 0;
  for (let x = 0; x < CELLS; x++) {
    const a = Math.abs(u[r][x]);
    if (!(a <= m)) m = a;   // !(a <= m), so a NaN cannot hide
  }
  console.log('  dt =', dts[r].toFixed(2), '→ largest |u| =', m.toExponential(2));
  if (m < 1 && dts[r] > measured) measured = dts[r];
}

console.log('measured limit: dt <=', measured);
console.log('predicted:      dt <=', dx * dx / (2 * D * 1));
`,
      inputs: () => ({ seed: makeRows(), dts: makeDts() }),
      publicTests: [
        {
          name: 'one step: every ring uses its own <code>dt</code>, and only its own row',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const dts = makeDts();
            // Rows deliberately DIFFERENT: with sixteen identical rows the
            // vertical part of a 5-point stencil is zero and the wrong answer
            // would be indistinguishable from the right one.
            const rows = makeRows().map((row, r) =>
              row.map((v, x) => (x === (T3_CELLS >> 1) + r ? 1 : 0))
            );
            const out = ctx.kernel(rows, dts);
            ctx.assert(
              out && out.length === T3_ROWS && out[0] && out[0].length === T3_CELLS,
              `expected a ${T3_CELLS}×${T3_ROWS} result, got ${out && out.length} rows`
            );
            const expected = scanStepRef(rows, dts);
            const hint = diagnoseGrid(out, expected, 2e-4, scanProbes(rows, dts));
            for (let r = 0; r < T3_ROWS; r++) {
              for (let x = 0; x < T3_CELLS; x++) {
                ctx.assertClose(out[r][x], expected[r][x], 2e-4, hint || `row ${r}, cell ${x}`);
              }
            }
          },
        },
        {
          name: '90 steps: the rows split cleanly at <code>dt = 1</code>',
          run: async ctx => {
            const dts = makeDts();
            let u = makeRows();
            for (let i = 0; i < T3_STEPS; i++) u = ctx.kernel(u, dts);
            for (let r = 0; r < T3_ROWS; r++) {
              const m = maxAbsRow(u[r]);
              if (dts[r] < DT_LIMIT_1D) {
                ctx.assert(
                  m < 1,
                  `row ${r} (dt = ${dts[r]}, inside the limit) should have cooled below 1, but its ` +
                    `hottest |u| is ${m}`
                );
              } else {
                ctx.assert(
                  !(m < 1e3),
                  `row ${r} (dt = ${dts[r]}, past the limit of ${DT_LIMIT_1D}) should have blown up, ` +
                    `but its hottest |u| is only ${m}`
                );
              }
            }
          },
        },
        {
          name: 'the measured limit is reported, and it brackets the predicted one',
          run: async ctx => {
            const dts = makeDts();
            const largestStable = dts.filter(dt => dt < DT_LIMIT_1D).pop();
            const measured = numbersInLines(ctx.logs, /measured/i);
            ctx.assert(
              measured.length > 0,
              'log the measured limit on a line containing the word "measured"'
            );
            const hit = measured.find(v => Math.abs(v - largestStable) <= 5e-3);
            const near = measured
              .map(v =>
                diagnose(v, largestStable, 5e-3, [
                  [dts.find(dt => dt > DT_LIMIT_1D),
                    'that is the smallest dt that FAILED — the measured limit is the largest one that survived, the row just below it'],
                  [dts[dts.length - 1],
                    'that is simply the largest dt in the list — the loop is not filtering on the row’s result'],
                  [DT_LIMIT_1D,
                    'that is the predicted limit, not the measured one — the measurement should come from the rows that survived'],
                ])
              )
              .find(Boolean);
            ctx.assert(
              hit !== undefined,
              near || `the largest dt whose row stayed under 1 is ${largestStable}`
            );
            const predicted = numbersInLines(ctx.logs, /predict/i);
            ctx.assert(
              predicted.some(v => Math.abs(v - DT_LIMIT_1D) <= 1e-6),
              `also log the predicted limit dx²/(2·D·1) = ${DT_LIMIT_1D} — with dims = 1 a ring ` +
                `tolerates twice the step a 2D grid does`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Fresh initial data — two hot cells per ring — driven for 30 steps
            // and compared cell for cell against the CPU reference. Every dt in
            // the list is still stable at 30 steps, so this is an exact check on
            // the whole grid, not just on the survivors.
            const dts = makeDts();
            const rows = makeRows().map(row => {
              const copy = row.slice();
              copy[8] = 0.5;
              return copy;
            });
            let u = rows;
            for (let i = 0; i < 30; i++) u = ctx.kernel(u, dts);
            const ref = scanRunRef(rows, dts, 30);
            for (let r = 0; r < T3_ROWS; r++) {
              for (let x = 0; x < T3_CELLS; x++) {
                if (dts[r] < DT_LIMIT_1D) {
                  ctx.assertClose(u[r][x], ref[r][x], 2e-4, `row ${r}, cell ${x} after 30 steps`);
                }
              }
            }
            // Each ring is closed, so a stable row conserves its heat exactly.
            for (let r = 0; r < T3_ROWS; r++) {
              if (dts[r] >= DT_LIMIT_1D) continue;
              let total = 0;
              for (let x = 0; x < T3_CELLS; x++) total += u[r][x];
              ctx.assertClose(total, 1.5, 2e-3, `heat in ring ${r} after 30 steps`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'implicit-step',
      title: 'Solve, Don’t Step',
      intro: `<p>The whole problem is that the explicit step evaluates the Laplacian on the field
        it is leaving. Evaluate it on the field it is <em>arriving</em> at instead — that is
        backward Euler — and the step size limit vanishes entirely. Unconditionally stable, at
        any <code>dt</code>, forever.</p>
        <p>The catch is visible the moment you write it down. The unknown is on both sides:</p>
<pre><code>u' = u + α·∇²u'
⟺  (1 + 4α)·u' − α·(neighbours of u') = u</code></pre>
        <p>That is not a formula you evaluate, it is a <strong>linear system</strong> — one
        equation per cell, 1,024 of them on this grid, all coupled. Solve it the way GPUs
        like: rearrange each equation for its own cell, then iterate.</p>
<pre><code>u'[c] = ( u[c] + α·(neighbours of u') )
        / (1 + 4α)</code></pre>
        <p>Every cell reads only the <em>previous</em> iterate, so all 1,024 can be computed at
        once — that is a <strong>Jacobi sweep</strong>, and the Iterative Solvers module takes it
        much further (red-black ordering, residuals, why it beats Gauss–Seidel on a GPU). Here
        25 sweeps is plenty, because the <code>1</code> in <code>1 + 4α</code> makes this system
        diagonally dominant and easy. One thing must not slip: <code>u</code> on the right is the
        <em>old time level</em> and never changes during the solve. Only the guess moves.</p>`,
      goal: `<strong>Goal:</strong> write the Jacobi sweep, then iterate it 25 times to take a
        single implicit step at <code>dt = 2</code> — four times the explicit limit.`,
      requirements: [
        'The sweep returns <code>(uOld[y][x] + α · (four neighbours of <em>guess</em>)) / (1 + 4α)</code>',
        'The centre term comes from <code>uOld</code>; only the four neighbours come from <code>guess</code>',
        'Iterate <code>SWEEPS</code> times, passing the <em>same</em> <code>seed</code> as <code>uOld</code> every time',
      ],
      hints: [
        {
          title: 'Hint 1 — where the division comes from',
          body: `<p>Collect the unknown cell on the left of
            <code>u' = u + α·(l + r + up + dn − 4u')</code>: the <code>−4α·u'</code> moves over as
            <code>+4α·u'</code>, giving <code>(1 + 4α)·u' = u + α·(l + r + up + dn)</code>. Divide.</p>`,
        },
        {
          title: 'Hint 2 — the sweep body',
          body: `<pre><code>const neighbours = guess[y][xl] + guess[y][xr]
  + guess[yd][x] + guess[yu][x];
return (uOld[y][x] + this.constants.alpha * neighbours)
  / (1 + 4 * this.constants.alpha);</code></pre>`,
        },
        {
          title: 'Hint 3 — why the starter’s loop is wrong',
          body: `<p><code>sweep(guess, guess)</code> replaces the right-hand side with the current
            iterate every sweep, which throws away the one piece of information that makes this a
            <em>time step</em>. It still converges — to <code>∇²u = 0</code>, a flat field. The
            fix is one word: <code>guess = sweep(seed, guess);</code></p>`,
        },
      ],
      transfer: `"The implicit step is a linear solve" is the fork in the road for every
        production simulator: implicit thermal and structural codes hand
        <code>(I − αL)</code> to a Krylov solver with a preconditioner, and GPU fluid solvers
        run exactly this Jacobi (or a multigrid V-cycle) for the pressure projection every
        frame. cuSPARSE, rocSPARSE and every WebGPU fluid demo you have seen are all standing
        on this one rearrangement.`,
      starterCode: `// Backward Euler: the new field appears on BOTH sides of the equation.
// Solve it with Jacobi sweeps — every cell reads the previous iterate.
const gpu = new GPU({ mode });

const D = 8;
const dx = 4;
const dt = 2;                       // 4× the explicit limit of 0.5
const ALPHA = D * dt / (dx * dx);   // = 1
const SWEEPS = 25;

const sweep = gpu.createKernel(function (uOld, guess) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  // TODO: one Jacobi sweep —
  //   (uOld[y][x] + alpha * (the four neighbours of GUESS)) / (1 + 4 * alpha)
  return guess[y][x];
}, {
  output: [32, 32],
  constants: { size: 32, alpha: ALPHA },
});

function hottest(u) {
  let m = 0;
  for (let y = 0; y < u.length; y++) {
    for (let x = 0; x < u[y].length; x++) {
      const a = Math.abs(u[y][x]);
      if (!(a <= m)) m = a;
    }
  }
  return m;
}

function total(u) {
  let s = 0;
  for (let y = 0; y < u.length; y++) for (let x = 0; x < u[y].length; x++) s += u[y][x];
  return s;
}

let guess = seed;
for (let k = 0; k < SWEEPS; k++) {
  // TODO: the right-hand side is the OLD field and never changes during a
  // solve. This passes the current iterate instead, which throws the time
  // step away and converges to a flat field.
  guess = sweep(guess, guess);
}

console.log('hottest after one implicit step:', hottest(guess));
console.log('total heat (unchanged at 36):', total(guess));
`,
      solutionCode: `// Backward Euler: the new field appears on BOTH sides of the equation.
// Solve it with Jacobi sweeps — every cell reads the previous iterate.
const gpu = new GPU({ mode });

const D = 8;
const dx = 4;
const dt = 2;                       // 4× the explicit limit of 0.5
const ALPHA = D * dt / (dx * dx);   // = 1
const SWEEPS = 25;

const sweep = gpu.createKernel(function (uOld, guess) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const neighbours = guess[y][xl] + guess[y][xr] + guess[yd][x] + guess[yu][x];
  return (uOld[y][x] + this.constants.alpha * neighbours)
    / (1 + 4 * this.constants.alpha);
}, {
  output: [32, 32],
  constants: { size: 32, alpha: ALPHA },
});

function hottest(u) {
  let m = 0;
  for (let y = 0; y < u.length; y++) {
    for (let x = 0; x < u[y].length; x++) {
      const a = Math.abs(u[y][x]);
      if (!(a <= m)) m = a;
    }
  }
  return m;
}

function total(u) {
  let s = 0;
  for (let y = 0; y < u.length; y++) for (let x = 0; x < u[y].length; x++) s += u[y][x];
  return s;
}

let guess = seed;
for (let k = 0; k < SWEEPS; k++) {
  // seed is the old time level: fixed for the whole solve. Only guess moves.
  guess = sweep(seed, guess);
}

console.log('hottest after one implicit step:', hottest(guess));
console.log('total heat (unchanged at 36):', total(guess));
`,
      inputs: () => ({ seed: hotSquare(32, 6, 1) }),
      publicTests: [
        {
          // MUST STAY FIRST. Every test shares the run's kernels, so .lastArgs
          // records whichever test invoked the sweep most recently — read them
          // before any other test has had the chance.
          name: 'the driving loop held the old field fixed',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const last = ctx.kernels[0] && ctx.kernels[0].lastArgs;
            ctx.assert(last && last.length >= 2, 'the sweep should be called as sweep(uOld, guess)');
            const seed = hotSquare(32, 6, 1);
            const cells = [[16, 16], [13, 13], [16, 13], [4, 4], [20, 20]];
            for (const [y, x] of cells) {
              ctx.assertClose(
                last[0][y][x], seed[y][x], 2e-4,
                `the last sweep’s uOld no longer matches the seed at [${y}][${x}] — the ` +
                  'right-hand side is the old time level and must be passed unchanged every sweep. ' +
                  'sweep(guess, guess) solves for the steady state instead of taking a time step'
              );
            }
          },
        },
        {
          name: 'a flat field is a fixed point of the solve, at any <code>α</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const flat = makeGrid(32, 0.7);
            const out = ctx.kernel(flat, flat);
            ctx.assert(
              out && out.length === 32 && out[0] && out[0].length === 32,
              `expected a 32×32 result, got ${out && out.length} rows`
            );
            for (let y = 0; y < 32; y += 5) {
              for (let x = 0; x < 32; x += 5) {
                // (0.7 + α·4·0.7) / (1 + 4α) = 0.7 exactly, whatever α is.
                ctx.assertClose(out[y][x], 0.7, 1e-5, `cell [${y}][${x}] of a flat field`);
              }
            }
          },
        },
        {
          name: 'one sweep: the centre from <code>uOld</code>, the neighbours from <code>guess</code>',
          run: async ctx => {
            // uOld and guess deliberately DIFFERENT — where they coincide (the
            // very first sweep of a solve) half of these mistakes are invisible.
            const uOld = hotSquare(32, 6, 1);
            const guess = hotSquare(32, 10, 0.4);
            const out = ctx.kernel(uOld, guess);
            const expected = jacobiSweepRef(uOld, guess, T4_ALPHA);
            const hint = diagnoseGrid(out, expected, 2e-4, sweepProbes(uOld, guess, T4_ALPHA));
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 2e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: '25 sweeps solve the implicit step — and it cannot overshoot',
          run: async ctx => {
            const seed = hotSquare(32, 6, 1);
            let guess = seed;
            for (let k = 0; k < SWEEPS; k++) guess = ctx.kernel(seed, guess);
            const expected = implicitStepRef(seed, T4_ALPHA, SWEEPS);
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                ctx.assertClose(guess[y][x], expected[y][x], 2e-4, `cell [${y}][${x}] after ${SWEEPS} sweeps`);
              }
            }
            // The discrete maximum principle: an implicit step is a weighted
            // average with every weight positive, so no new hot or cold spot
            // can appear no matter how large dt is.
            ctx.assert(
              maxGrid(guess) <= 1 + 2e-4 && minGrid(guess) >= -2e-4,
              `an implicit step may not create a new extreme: got max ${maxGrid(guess)}, min ${minGrid(guess)}`
            );
            ctx.assertClose(sumGrid(guess), sumGrid(seed), 5e-3, 'heat is conserved on a torus');
          },
        },
        {
          name: 'the hottest cell and the total heat are logged',
          run: async ctx => {
            const seed = hotSquare(32, 6, 1);
            const expected = maxGrid(implicitStepRef(seed, T4_ALPHA, SWEEPS));
            const nums = loggedNumbers(ctx.logs);
            const wrongLoop = maxGrid(
              (() => {
                let g = seed;
                for (let k = 0; k < SWEEPS; k++) g = jacobiSweepRef(g, g, T4_ALPHA);
                return g;
              })()
            );
            const hint = nums.some(v => Math.abs(v - wrongLoop) <= 2e-3)
              ? `${wrongLoop.toFixed(4)} is what you get when the old field is replaced by the ` +
                'current guess each sweep — that iteration converges to a flat field, not to a time step'
              : null;
            ctx.assert(
              nums.some(v => Math.abs(v - expected) <= 2e-3),
              hint || `log the hottest cell after the implicit step — expected ≈${expected.toFixed(4)}`
            );
            ctx.assert(
              nums.some(v => Math.abs(v - 36) <= 0.05),
              'log the total heat — an implicit step on a torus conserves it exactly, at 36'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A second implicit step, from different data, checked in full —
            // and then twelve of them in a row, to show the step size really is
            // unbounded: at α = 1 an explicit run would have been dead by step 3.
            const seed = hotSquare(32, 10, 0.75);
            let guess = seed;
            for (let k = 0; k < SWEEPS; k++) guess = ctx.kernel(seed, guess);
            const expected = implicitStepRef(seed, T4_ALPHA, SWEEPS);
            const hint = diagnoseGrid(guess, expected, 2e-4, [
              [implicitStepRef(seed, T4_ALPHA, 1),
                'only one sweep happened — the loop has to iterate SWEEPS times'],
            ]);
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                ctx.assertClose(guess[y][x], expected[y][x], 2e-4, hint || `cell [${y}][${x}]`);
              }
            }

            let u = seed;
            for (let i = 0; i < 12; i++) {
              let g = u;
              for (let k = 0; k < SWEEPS; k++) g = ctx.kernel(u, g);
              u = g;
            }
            const ref = implicitRunRef(seed, T4_ALPHA, SWEEPS, 12);
            ctx.assert(boundedBy(u, 1.001), 'twelve implicit steps at 4× the explicit limit must stay bounded');
            for (let y = 0; y < 32; y += 3) {
              for (let x = 0; x < 32; x += 3) {
                ctx.assertClose(u[y][x], ref[y][x], 5e-4, `cell [${y}][${x}] after 12 implicit steps`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'stable-vs-accurate',
      title: 'Stable Is Not Accurate',
      intro: `<p>Time to put the two schemes on the same clock. Both runs below finish at
        <code>T = 16</code>: the explicit one in small steps of <code>0.2</code>, the implicit
        one in steps of <code>2</code> — ten times larger, and four times past the explicit
        limit. A third run takes the explicit scheme at the implicit step size, for the pleasure
        of watching it fail in eight steps.</p>
        <p>The two survivors will not agree. Backward Euler is <em>first-order</em> accurate,
        just like forward Euler, so a ten-times-larger step carries a ten-times-larger error —
        it damps sharp features harder than the real equation does. Expect the two fields to
        differ by a few percent of the peak. That is the honest trade, and it is worth stating
        plainly: <strong>unconditional stability is not accuracy</strong>. What implicit
        stepping buys you is the right to choose <code>dt</code> for the accuracy you need,
        rather than having it dictated by the smallest cell in your mesh.</p>
        <p>Nor is it free. Each implicit step here costs 25 sweeps, so the coarse run makes
        <code>8 × 25 = 200</code> kernel launches against the fine run's 80 — implicit
        <em>loses</em> the launch count at this size. It wins when the explicit limit gets
        brutal: refine <code>dx</code> by 10× and the explicit run needs 100× the steps, while
        the implicit one needs the same eight and a slightly harder solve.</p>`,
      goal: `<strong>Goal:</strong> finish the Jacobi sweep with <code>α</code> arriving as an
        argument, work out how many steps of each size reach <code>T</code>, and run all three.`,
      requirements: [
        'The sweep is the one from the last task, but <code>alpha</code> is a kernel <em>argument</em>, not a constant',
        '<code>smallSteps</code> and <code>bigSteps</code> are the counts that reach <code>T = 16</code> at <code>dt = 0.2</code> and <code>dt = 2</code>',
        'Run all three: explicit at the small step, implicit at the big one, explicit at the big one',
        'Log the hottest cell of each, and the largest gap between the two survivors',
      ],
      hints: [
        {
          title: 'Hint 1 — alpha as an argument',
          body: `<p>Only the spelling changes: a plain <code>alpha</code> where the constant used to
            be, and the caller passes it. Everything else is last task's body:</p>
<pre><code>return (uOld[y][x] + alpha * neighbours)
  / (1 + 4 * alpha);</code></pre>`,
        },
        {
          title: 'Hint 2 — how many steps?',
          body: `<p>Steps × step size = elapsed time, so it is <code>T / dt</code>:
            <code>16 / 0.2 = 80</code> and <code>16 / 2 = 8</code>.</p>`,
        },
      ],
      transfer: `Choosing a scheme by what limits it — accuracy or stability — is the daily work
        of numerical simulation everywhere: stiff chemistry and implicit thermal solvers pay for
        a linear solve per step because the explicit alternative would need millions of them,
        while explicit codes dominate wave propagation and particle work, where the stability
        step is close to the accuracy step anyway. On a GPU the arithmetic is nearly free, so
        the calculus is really about launches and memory traffic per unit of simulated time.`,
      starterCode: `// Same physics, same finish time, two step sizes.
const gpu = new GPU({ mode });

const D = 8;
const dx = 4;
const T = 16;           // finish time
const DT_SMALL = 0.2;   // 0.4× the explicit limit (0.5)
const DT_BIG = 2;       // 4×   the explicit limit
const SWEEPS = 25;

const explicitStep = gpu.createKernel(function (u, alpha) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const c = u[y][x];
  return c + alpha * (u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * c);
}, { output: [32, 32], constants: { size: 32 } });

const sweep = gpu.createKernel(function (uOld, guess, alpha) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  // TODO: last task's sweep, with alpha coming in as an argument
  return guess[y][x];
}, { output: [32, 32], constants: { size: 32 } });

function hottest(u) {
  let m = 0;
  for (let y = 0; y < u.length; y++) {
    for (let x = 0; x < u[y].length; x++) {
      const a = Math.abs(u[y][x]);
      if (!(a <= m)) m = a;
    }
  }
  return m;
}

function gap(a, b) {
  let m = 0;
  for (let y = 0; y < a.length; y++) {
    for (let x = 0; x < a[y].length; x++) {
      const d = Math.abs(a[y][x] - b[y][x]);
      if (!(d <= m)) m = d;
    }
  }
  return m;
}

function runExplicit(dt, steps) {
  const alpha = D * dt / (dx * dx);
  let u = seed;
  for (let i = 0; i < steps; i++) u = explicitStep(u, alpha);
  return u;
}

function runImplicit(dt, steps) {
  const alpha = D * dt / (dx * dx);
  let u = seed;
  for (let i = 0; i < steps; i++) {
    let guess = u;
    for (let k = 0; k < SWEEPS; k++) guess = sweep(u, guess, alpha);
    u = guess;
  }
  return u;
}

// TODO: how many steps of each size land exactly on time T?
const smallSteps = 0;
const bigSteps = 0;

const fine = runExplicit(DT_SMALL, smallSteps);
const coarse = runImplicit(DT_BIG, bigSteps);
const doomed = runExplicit(DT_BIG, bigSteps);

console.log('explicit, dt =', DT_SMALL, 'x', smallSteps, 'steps → hottest', hottest(fine));
console.log('implicit, dt =', DT_BIG, 'x', bigSteps, 'steps → hottest', hottest(coarse));
console.log('explicit at dt =', DT_BIG, '→ hottest', hottest(doomed));
console.log('largest gap between the two survivors:', gap(fine, coarse));
`,
      solutionCode: `// Same physics, same finish time, two step sizes.
const gpu = new GPU({ mode });

const D = 8;
const dx = 4;
const T = 16;           // finish time
const DT_SMALL = 0.2;   // 0.4× the explicit limit (0.5)
const DT_BIG = 2;       // 4×   the explicit limit
const SWEEPS = 25;

const explicitStep = gpu.createKernel(function (u, alpha) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const c = u[y][x];
  return c + alpha * (u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * c);
}, { output: [32, 32], constants: { size: 32 } });

const sweep = gpu.createKernel(function (uOld, guess, alpha) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const neighbours = guess[y][xl] + guess[y][xr] + guess[yd][x] + guess[yu][x];
  return (uOld[y][x] + alpha * neighbours) / (1 + 4 * alpha);
}, { output: [32, 32], constants: { size: 32 } });

function hottest(u) {
  let m = 0;
  for (let y = 0; y < u.length; y++) {
    for (let x = 0; x < u[y].length; x++) {
      const a = Math.abs(u[y][x]);
      if (!(a <= m)) m = a;
    }
  }
  return m;
}

function gap(a, b) {
  let m = 0;
  for (let y = 0; y < a.length; y++) {
    for (let x = 0; x < a[y].length; x++) {
      const d = Math.abs(a[y][x] - b[y][x]);
      if (!(d <= m)) m = d;
    }
  }
  return m;
}

function runExplicit(dt, steps) {
  const alpha = D * dt / (dx * dx);
  let u = seed;
  for (let i = 0; i < steps; i++) u = explicitStep(u, alpha);
  return u;
}

function runImplicit(dt, steps) {
  const alpha = D * dt / (dx * dx);
  let u = seed;
  for (let i = 0; i < steps; i++) {
    let guess = u;
    for (let k = 0; k < SWEEPS; k++) guess = sweep(u, guess, alpha);
    u = guess;
  }
  return u;
}

const smallSteps = T / DT_SMALL;   // 80
const bigSteps = T / DT_BIG;       // 8

const fine = runExplicit(DT_SMALL, smallSteps);
const coarse = runImplicit(DT_BIG, bigSteps);
const doomed = runExplicit(DT_BIG, bigSteps);

console.log('explicit, dt =', DT_SMALL, 'x', smallSteps, 'steps → hottest', hottest(fine));
console.log('implicit, dt =', DT_BIG, 'x', bigSteps, 'steps → hottest', hottest(coarse));
console.log('explicit at dt =', DT_BIG, '→ hottest', hottest(doomed));
console.log('largest gap between the two survivors:', gap(fine, coarse));
`,
      inputs: () => ({ seed: hotSquare(32, 6, 1) }),
      publicTests: [
        {
          name: 'two kernels: an explicit step and a Jacobi sweep, both taking <code>α</code> as an argument',
          run: async ctx => {
            ctx.assert(
              ctx.kernels.length >= 2,
              `expected two kernels (explicit step and Jacobi sweep), found ${ctx.kernels.length}`
            );
            const stepK = findByArity(ctx, 2);
            const sweepK = findByArity(ctx, 3);
            ctx.assert(stepK, 'no explicit-step kernel found — one kernel should take (u, alpha)');
            ctx.assert(sweepK, 'no Jacobi sweep found — one kernel should take (uOld, guess, alpha)');
            const spike = stepK(spikeGrid(32, 5, 5, 1), 0.1);
            ctx.assertClose(spike[5][6], 0.1, 2e-4,
              'the explicit kernel should give a hot cell’s neighbour exactly α — here α arrives as an argument, not as this.constants.alpha');
            const flat = makeGrid(32, 0.7);
            const still = sweepK(flat, flat, T4_ALPHA);
            ctx.assertClose(still[5][5], 0.7, 2e-4,
              'a flat field is a fixed point of the implicit solve at any α — (u + α·4u) / (1 + 4α) = u');
          },
        },
        {
          name: 'the sweep matches the implicit formula at the big step size',
          run: async ctx => {
            const uOld = hotSquare(32, 6, 1);
            const guess = hotSquare(32, 10, 0.4);
            const sweepK = findByArity(ctx, 3);
            ctx.assert(sweepK, 'no three-argument Jacobi sweep found — it takes (uOld, guess, alpha)');
            const out = sweepK(uOld, guess, T4_ALPHA);
            const expected = jacobiSweepRef(uOld, guess, T4_ALPHA);
            const hint = diagnoseGrid(out, expected, 2e-4, sweepProbes(uOld, guess, T4_ALPHA));
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 2e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'all three runs happened: 80 small explicit steps, 8 implicit, 8 doomed',
          run: async ctx => {
            const seed = hotSquare(32, 6, 1);
            const fine = explicitRunRef(seed, alphaFor(T5_SMALL_DT), T5_TIME / T5_SMALL_DT);
            const coarse = implicitRunRef(seed, alphaFor(T5_BIG_DT), SWEEPS, T5_TIME / T5_BIG_DT);
            const doomed = explicitRunRef(seed, alphaFor(T5_BIG_DT), T5_TIME / T5_BIG_DT);
            const nums = loggedNumbers(ctx.logs);
            const counts = stepCountHint(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - maxGrid(fine)) <= 2e-3),
              counts || `log the hottest cell of the fine explicit run — expected ≈${maxGrid(fine).toFixed(4)} ` +
                `after ${T5_TIME / T5_SMALL_DT} steps of dt = ${T5_SMALL_DT}`
            );
            ctx.assert(
              nums.some(v => Math.abs(v - maxGrid(coarse)) <= 3e-3),
              counts || `log the hottest cell of the implicit run — expected ≈${maxGrid(coarse).toFixed(4)} ` +
                `after ${T5_TIME / T5_BIG_DT} steps of dt = ${T5_BIG_DT}`
            );
            ctx.assert(
              nums.some(v => !(v < 1e3)),
              `the explicit scheme at dt = ${T5_BIG_DT} should be reported at about ` +
                `${maxAbsGrid(doomed).toExponential(2)} — eight steps is all it takes`
            );
          },
        },
        {
          name: 'the two survivors agree to a few percent — and disagree by more than rounding',
          run: async ctx => {
            const seed = hotSquare(32, 6, 1);
            const fine = explicitRunRef(seed, alphaFor(T5_SMALL_DT), T5_TIME / T5_SMALL_DT);
            const coarse = implicitRunRef(seed, alphaFor(T5_BIG_DT), SWEEPS, T5_TIME / T5_BIG_DT);
            let expectedGap = 0;
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                const d = Math.abs(fine[y][x] - coarse[y][x]);
                if (!(d <= expectedGap)) expectedGap = d;
              }
            }
            // Measured: 2.846e-2 (about 9.6% of the peak). The window is wide
            // enough that neither float32 drift (≤ 3e-8) nor a different but
            // reasonable sweep count can move a correct answer out of it.
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - expectedGap) <= 5e-3),
              `log the largest cell-by-cell gap between the two survivors — expected ≈${expectedGap.toFixed(4)}`
            );
            ctx.assert(expectedGap > 5e-3, 'sanity: the two schemes are supposed to disagree at this step ratio');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Re-run both routes from fresh data through the learner's kernels
            // and check the claim end to end: the implicit route survives ten
            // times the step size, and lands within a few percent of the fine
            // explicit answer, while the explicit scheme at that step is dead.
            const seed = hotSquare(32, 8, 0.8);
            const stepK = findByArity(ctx, 2);
            const sweepK = findByArity(ctx, 3);
            ctx.assert(stepK && sweepK, 'expected an explicit-step kernel and a Jacobi sweep');

            const alphaSmall = alphaFor(T5_SMALL_DT);
            const alphaBig = alphaFor(T5_BIG_DT);
            let fine = seed;
            for (let i = 0; i < 40; i++) fine = stepK(fine, alphaSmall);
            let coarse = seed;
            for (let i = 0; i < 4; i++) {
              let g = coarse;
              for (let k = 0; k < SWEEPS; k++) g = sweepK(coarse, g, alphaBig);
              coarse = g;
            }
            let doomed = seed;
            // Eight steps at α = 1 — the same count the task runs. Measured
            // hottest |u| from this seed: 8.0e4, five orders of magnitude past
            // the 1e3 threshold below, on both backends.
            for (let i = 0; i < 8; i++) doomed = stepK(doomed, alphaBig);

            const refFine = explicitRunRef(seed, alphaSmall, 40);
            const refCoarse = implicitRunRef(seed, alphaBig, SWEEPS, 4);
            for (let y = 0; y < 32; y += 3) {
              for (let x = 0; x < 32; x += 3) {
                ctx.assertClose(fine[y][x], refFine[y][x], 2e-4, `explicit route, cell [${y}][${x}]`);
                ctx.assertClose(coarse[y][x], refCoarse[y][x], 5e-4, `implicit route, cell [${y}][${x}]`);
              }
            }
            ctx.assert(boundedBy(coarse, 1.001), 'the implicit route must stay inside its initial range');
            ctx.assert(!boundedBy(doomed, 1e3), 'the explicit scheme at the implicit step size must blow up');
          },
        },
      ],
    },
  ],
};
