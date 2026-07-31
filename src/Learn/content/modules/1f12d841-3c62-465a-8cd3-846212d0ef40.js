// Module: The Ising Model — uuid 1f12d841-3c62-465a-8cd3-846212d0ef40
// (short id 1f12d841). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId — this module postdates
// the uuid migration.
//
// Six tasks: the cost of flipping one spin → a random number per thread with no
// random number generator → the all-at-once update, and watching it drive the
// energy UP → the red half-sweep → both halves chained, and the same run going
// DOWN instead → a temperature slider through the phase transition.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested numeric arrays as inputs, this.thread.* for indexing,
// this.constants.* for compile-time values, every kernel call awaited. Every
// task passes in cpu, webgl and auto.
//
// ---------------------------------------------------------------------------
// THE ONE IDEA, AND WHOSE IT IS
//
// The checkerboard here is the SAME colouring "Iterative Linear Solvers"
// (e73b8e1f) uses for red-black Gauss-Seidel, and this module says so in task 4
// rather than teaching it twice. What differs is what the colouring buys: there
// it makes a sequential-by-construction solver runnable at all; here it repairs
// a Monte Carlo update that is silently WRONG when run in parallel. Same
// permission slip — "these cells cannot see each other, so they may all go at
// once" — issued for two different crimes.
//
// ---------------------------------------------------------------------------
// RANDOMNESS: WHY A HASH, AND WHY THESE PARTICULAR MODULI
//
// gpu.js's WebGPU backend refuses Math.random outright (backend/web-gpu/
// function-node.js: "WebGPU backend does not yet support Math.random"), and a
// stream RNG needs one shared, ordered state — the two things 16,384 threads do
// not have. So every random number here is hash(x, y, seed): a pure function of
// where the thread is and which half-sweep it is, which needs no state, no
// ordering, and replays identically every run.
//
// The hash is arithmetic on INTEGERS, and every intermediate is deliberately
// kept below 2^24 = 16,777,216, the largest integer a 32-bit float holds
// exactly. Largest value each step reaches, measured over every (x, y, seed)
// this module can produce — x, y < 128 and seed ≤ 301, task 6's last half-sweep
// being seed 299: the input fold 1,564,075, the first squaring 4,255,744, the
// byte swap 65,279, the multiply-and-add 16,545,598 (its worst case if the swap
// could return 65,535 would be 16,610,366, still inside), the second squaring
// 4,251,650. All exact — so WebGPU (f32), WebGL (f32) and the CPU backend (f64)
// compute the same 16-bit integers and the same trajectory, which is what lets
// the tests below compare a whole lattice cell for cell instead of settling for
// statistics. Verified: 0 mismatches between float64 and Math.fround-simulated
// float32 over 512 seeds × 128 × 128.
//
// Quality was measured, not assumed. Pooling five seeds per lag (n = 81,920,
// standard error 0.0035) over 750 (dx, dy, dseed) lags, the worst |correlation|
// is 0.011 — ~3σ, i.e. the noise floor of the measurement itself; a genuine
// 32-bit mixer scores the same. Uniformity over 32 bins on a 128×128 field:
// chi-square 24.4 on 31 degrees of freedom, mean 0.49965, 12,808 distinct values
// out of 16,384 cells.
//
// Two rounds are needed, and WHERE the second round earns its place is worth
// being exact about, because the obvious test cannot see it. The numbers below
// are per single field (n = 16,384, standard error 0.0078), which is what a test
// can compute from one kernel call, so they fluctuate more than the pooled 0.011
// above — that is measurement noise, not disagreement.
//
// Within ONE field a single-round hash is already fine: over 63 spatial lags on
// each of 12 seeds its worst |correlation| is 0.021, against 0.032 for the
// two-round hash. The same noise; the spatial test cannot tell them apart. The
// damage is BETWEEN consecutive seeds. One squaring plus one multiply is still
// so nearly affine that hash(x, y, k) and hash(x+1, y+1, k+1) correlate at 0.30
// (six seed pairs, 0.295–0.306), against 0.007–0.021 for the two-round hash.
// That is precisely the pair a red-black sweep puts side by side — a red cell
// drawing at seed 2k and the black neighbour that reads it drawing at seed
// 2k+1 — so task 2's private test #2 measures both lag families, not just the
// spatial one, and only the second family fails a single-round hash.
//
// THE DECISION BOUNDARY IS SAFE BY MEASUREMENT, and the margin is smaller than
// it looks, so here are the real numbers. A Metropolis test compares u = k/65536
// against exp(-dE/T), and dE is only ever 4 or 8 on the uphill side, so there are
// just two thresholds per temperature. At the temperatures the tests actually
// assert on (1.5, 2.0, 2.25, 2.5, 3, 0.125, 40) the closest any threshold comes
// to a representable k/65536 is 0.069 of a step — 1.05e-6 in probability, at
// T = 2.25, dE = 8. An f32 exp() is good to a few ULP, ~3e-7 relative, which is
// ~9e-9 absolute there: a hundredfold margin, so no cell can decide differently
// on one backend than on another and exact-match assertions are legitimate.
// Across every stop the slider can reach the tightest case is T = 1.80, dE = 4,
// where the threshold sits 0.0068 of a step (1.0e-7) above k = 7102 and the
// margin is only a few times the f32 error. Nothing is asserted at T = 1.80 —
// the worst that can happen there is one cell of a painted lattice differing
// between backends — but the claim is "safe where it is measured", not "safe
// everywhere by construction".
//
// ---------------------------------------------------------------------------
// THE NUMBERS THIS MODULE ASSERTS, ALL MEASURED AT 128 x 128
//
//   task 1  dE takes exactly the five values {-8, -4, 0, 4, 8}; an aligned
//           lattice is +8 everywhere, a perfect checkerboard -8 everywhere.
//   task 3  all-at-once at T = 1.5 from the seeded lattice: E/spin goes
//           -0.0085 → +0.580 after ONE sweep → +1.303 after 30, with all 16,384
//           spins flipping every sweep from then on (a period-2 blinker).
//   task 5  the same lattice, same T, same 30 sweeps, coloured: -0.990 after one
//           sweep → -1.837 after 30. It is NOT monotone — 26 of the 29 steps
//           fall, and sweeps 10, 22 and 30 tick back up by 0.0007, 0.0027 and
//           0.0034. That is thermal noise at T = 1.5, not a broken sweep, and
//           the test allows +0.02 per step rather than pretending otherwise.
//   task 4  the red half-sweep moves 5,730 of the 8,192 red cells and 0 black
//           ones; all 5,586 downhill reds flip; at T = 0.125 no uphill flip is
//           ever accepted — exp(-4/0.125) is 1.27e-14 and the smallest u this
//           hash produces anywhere in seeds 0…399 is 2/65536 = 3.1e-5, nine
//           orders of magnitude above it.
//   task 6  cold start, 150 sweeps: |m| = 0.983 at T = 1.5, 0.908 at 2.0,
//           0.816 at 2.25 (the slider default), 0.066 at 2.5, 0.017 at 3.0.
//           Onsager's exact |m| for the infinite lattice: 0.9865, 0.9113,
//           0.6719, 0, 0. The two agree to ~0.005 up to T = 2.0 (the gap is
//           0.0036 at 1.5 and 0.0029 at 2.0) and then part company fast — 0.017
//           at 2.1, 0.061 at 2.2 — which is honest finite-size behaviour and is
//           said out loud in the task rather than smoothed over.
//
// Cost, measured on the gpu.js CPU backend (the slowest path a learner can
// choose): task 6 — 150 full sweeps plus 15 paints at 128×128 — takes 230-240 ms
// end to end, and task 5's 30 sweeps take 50 ms. So no task needs a budgetMs and
// no task is within an order of magnitude of the 10 s run watchdog or the 15 s
// test budget. The pre-flight guard compares its threshold against the LARGEST
// kernel output, which here is 16,384 threads, below the 65,536 it starts
// guarding at, so the guard never engages either.
//
// ---------------------------------------------------------------------------
// TWO GPU.JS TRAPS THIS MODULE WALKS PAST DELIBERATELY
//
//   • `const isRed = (x + y) % 2 === 0;` compiles on the CPU backend and dies on
//     WebGL with "cannot convert from 'bool' to 'lowp float'". Iterative Linear
//     Solvers found this one; the parity here is a NUMBER for the same reason,
//     and boolTrapHint() reads the run's error text because no numeric probe can
//     ever see a shader compile failure.
//   • `(x - 1) % n` is -1 at x = 0 in JavaScript AND in gpu.js's GLSL `modulo`
//     helper, which preserves the sign. The wrap has to be `(x + n - 1) % n`.
//     On the CPU backend that reads `undefined` and every value downstream is
//     NaN; on WebGL the out-of-range fetch comes back as 0. Task 1 probes both.

const SIZE = 128; // every lattice in this module
const CELLS = SIZE * SIZE;
// The exact critical temperature of the 2D Ising model, Onsager 1944:
// 2 / ln(1 + sqrt(2)) = 2.269185…, in units of J / k_B.
const TC = 2 / Math.log(1 + Math.SQRT2);
const RACE_T = 1.5; // tasks 3 and 5: cold enough that the two answers are opposite
const RACE_SWEEPS = 30;
const PAYOFF_T = 2.25; // task 6's slider default — ordered, but not by much
const PAYOFF_SWEEPS = 150;
const FRAME_EVERY = 10; // 15 frames → a scrubber

// ---- lattices --------------------------------------------------------------

// A random ±1 lattice. Magnetisation ≈ 0, energy ≈ 0: the "infinite temperature"
// configuration, and the honest starting point for a quench.
function makeLattice(utils, seed) {
  const rand = utils.seededRandom(seed);
  const grid = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) row[x] = rand() < 0.5 ? -1 : 1;
    grid[y] = row;
  }
  return grid;
}

// Every spin up: the coldest possible start, and the one task 6 melts.
function makeAligned() {
  const grid = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) grid[y] = new Array(SIZE).fill(1);
  return grid;
}

// The perfect antiferromagnet — every spin disagrees with all four neighbours.
// Energy +2 per spin, the highest a lattice can reach.
function makeCheckerboard() {
  const grid = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) row[x] = (x + y) % 2 === 0 ? 1 : -1;
    grid[y] = row;
  }
  return grid;
}

// Works on a kernel result (rows are Float32Array) as well as a plain grid.
function copyGrid(s) {
  const out = new Array(s.length);
  for (let y = 0; y < s.length; y++) {
    const row = new Array(s[y].length);
    for (let x = 0; x < s[y].length; x++) row[x] = s[y][x];
    out[y] = row;
  }
  return out;
}

// ---- CPU references --------------------------------------------------------

// The lattice is a torus: no edges, no special cases, every cell has four
// neighbours. `(x + SIZE - 1)` rather than `(x - 1)` — see the header.
function neighbourSum(s, y, x) {
  return (
    s[y][(x + 1) % SIZE] +
    s[y][(x + SIZE - 1) % SIZE] +
    s[(y + 1) % SIZE][x] +
    s[(y + SIZE - 1) % SIZE][x]
  );
}

// Fill every cell from cell(y, x). Every reference and every whole-grid probe
// candidate is built through here, so two of them can only differ in the one
// expression under test.
function gridMap(s, cell) {
  const out = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) row[x] = cell(y, x);
    out[y] = row;
  }
  return out;
}

// What flipping the spin at (y, x) would do to the total energy.
function flipCostRef(s) {
  return gridMap(s, (y, x) => 2 * s[y][x] * neighbourSum(s, y, x));
}

// This cell's share of the energy. The ½ is because every bond belongs to two
// cells and would otherwise be counted twice.
function bondEnergyRef(s) {
  return gridMap(s, (y, x) => -0.5 * s[y][x] * neighbourSum(s, y, x));
}

function meanOf(grid) {
  let total = 0;
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) total += grid[y][x];
  }
  return total / (grid.length * grid[0].length);
}

const energyRef = s => meanOf(bondEnergyRef(s));
const magnetisationRef = s => meanOf(s);

// THE hash — the JavaScript twin of the kernel body the learner writes in task
// 2. Every intermediate is an exact integer below 2^24; see the header.
function hash01(x, y, seed) {
  let h = (x * 1103 + y * 2749 + seed * 3571) % 65536;
  let q = h % 2048;
  h = (h + q * q) % 65536;
  h = (h % 256) * 256 + Math.floor(h / 256);
  h = (h * 253 + 30011) % 65536;
  q = h % 2048;
  h = (h + q * q) % 65536;
  return h / 65536;
}

// The Metropolis rule, one cell: downhill always, uphill with probability
// exp(-dE / T). (u < 1 always, so the first branch is technically implied by
// the second — see the task 4 note. Keeping it says what it means.)
function accepts(dE, temperature, u) {
  return dE <= 0 || u < Math.exp(-dE / temperature);
}

// One half-sweep. `colour` is 0 (red) or 1 (black): only cells whose (x + y)
// parity matches may move, and they read the lattice they were handed.
function halfSweepRef(s, colour, temperature, seed) {
  return gridMap(s, (y, x) => {
    if ((x + y) % 2 !== colour) return s[y][x];
    const dE = 2 * s[y][x] * neighbourSum(s, y, x);
    return accepts(dE, temperature, hash01(x, y, seed)) ? -s[y][x] : s[y][x];
  });
}

// One full sweep: reds, then blacks reading what the reds just wrote.
function fullSweepRef(s, temperature, sweep) {
  return halfSweepRef(halfSweepRef(s, 0, temperature, sweep * 2), 1, temperature, sweep * 2 + 1);
}

// The wrong one: every cell decides at once, from the same old lattice.
function naiveSweepRef(s, temperature, seed) {
  return gridMap(s, (y, x) => {
    const dE = 2 * s[y][x] * neighbourSum(s, y, x);
    return accepts(dE, temperature, hash01(x, y, seed)) ? -s[y][x] : s[y][x];
  });
}

// The energy trace tasks 3 and 5 print, run exactly the way their loops run it.
function traceRef(start, sweeps, temperature, step) {
  let s = start;
  const trace = [];
  for (let k = 0; k < sweeps; k++) {
    s = step(s, temperature, k);
    trace.push(energyRef(s));
  }
  return trace;
}

// Task 6's magnetisation trace at a given temperature, from the cold start.
function magTraceRef(temperature, sweeps) {
  let s = makeAligned();
  const trace = [];
  for (let k = 0; k < sweeps; k++) {
    s = fullSweepRef(s, temperature, k);
    trace.push(magnetisationRef(s));
  }
  return trace;
}

// ---- near-miss diagnosis ---------------------------------------------------
//
// A failing assert that reports two numbers tells a learner nothing about WHICH
// slip produced them. A probe pairs the value one specific known mistake would
// produce with a sentence naming that mistake; the helpers below speak only when
// the observation matches a probe AND the correct answer does not — so a case
// where two candidates coincide stays silent, as do observations matching probes
// that disagree with each other. A confident wrong diagnosis is worse than a
// plain numeric mismatch.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Every mistake in this module is lattice-WIDE — a missing factor of 2, an
// uncoloured sweep, a hash that ignored its seed — so a candidate has to predict
// all 16,384 cells (and disagree with the right answer somewhere) before it may
// speak. One lucky cell is not evidence: on a lattice where most spins agree
// with their neighbours, several wrong expressions land on the right number over
// whole regions.
function diagnoseGrid(out, expected, eps, alternatives) {
  const hits = alternatives
    .filter(([candidate]) => {
      let differs = false;
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          if (!(out[y] && Math.abs(out[y][x] - candidate[y][x]) <= eps)) return false;
          if (Math.abs(expected[y][x] - candidate[y][x]) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(a => a[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// gpu.js cannot hold a boolean in a kernel variable on the WebGL backend, and
// colouring a lattice is precisely the temptation. The failure is a shader
// compile error on the kernel's FIRST CALL, so it never reaches a value probe —
// it arrives as a failed run whose message is 200 characters of GLSL. Read that
// instead. The match is the exact GLSL wording, so it cannot fire on anything
// else.
function boolTrapHint(ctx) {
  const message = (ctx.error && ctx.error.message) || '';
  return /cannot convert from 'bool'/i.test(message)
    ? 'gpu.js cannot store a boolean in a kernel variable on the WebGL backend — ' +
        'const isRed = (x + y) % 2 === 0 compiles on the CPU backend and dies here. ' +
        'Keep the parity as a number: const parity = (x + y) % 2; then if (parity !== 0).'
    : null;
}

// The other compile-time refusal this module can provoke, and the whole reason
// the hash exists. Only the WebGPU backend says this, so the message explains
// what the learner would see everywhere else too.
function randomTrapHint(ctx) {
  const message = (ctx.error && ctx.error.message) || '';
  return /does not yet support Math\.random/i.test(message)
    ? 'gpu.js\'s WebGPU backend refuses Math.random outright, which is exactly why this ' +
        'task builds a hash instead. Delete the Math.random call and derive the number from ' +
        'this.thread.x, this.thread.y and the seed argument.'
    : null;
}

// Every task whose kernels can hit a compile-time refusal opens with this,
// because gpu.js compiles lazily: the kernel OBJECT exists, so a "no kernel was
// created" check passes and the learner is handed raw GLSL instead of a
// diagnosis.
function assertRunOk(ctx) {
  ctx.assert(
    ctx.ok,
    boolTrapHint(ctx) ||
      randomTrapHint(ctx) ||
      `the program did not finish: ${(ctx.error && ctx.error.message) || 'unknown error'}`
  );
}

// The wrap-around mistake, caught two ways because the two backends fail
// differently: the CPU backend reads undefined (→ NaN), WebGL fetches out of
// range and gets 0.
function wrapHint(out) {
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (!Number.isFinite(out[y] ? out[y][x] : NaN)) {
        return (
          `cell [${y}][${x}] came back as ${out[y] ? out[y][x] : 'undefined'} — a neighbour read ` +
          'outside the lattice. In JavaScript (0 - 1) % 128 is -1, not 127, and gpu.js\'s GLSL ' +
          'modulo keeps the sign too. Add the size first: (x + n - 1) % n.'
        );
      }
    }
  }
  return null;
}

// ---- probe candidates ------------------------------------------------------

// Neighbour sums a wrong wrap would produce, so the grid probes can name them.
function edgeZeroSum(s, y, x) {
  const at = (yy, xx) => (yy < 0 || xx < 0 || yy >= SIZE || xx >= SIZE ? 0 : s[yy][xx]);
  return at(y, x + 1) + at(y, x - 1) + at(y + 1, x) + at(y - 1, x);
}

function diagonalSum(s, y, x) {
  return (
    s[(y + 1) % SIZE][(x + 1) % SIZE] +
    s[(y + 1) % SIZE][(x + SIZE - 1) % SIZE] +
    s[(y + SIZE - 1) % SIZE][(x + 1) % SIZE] +
    s[(y + SIZE - 1) % SIZE][(x + SIZE - 1) % SIZE]
  );
}

// Task 1: the flip cost, mis-assembled.
function flipCostAlternatives(s) {
  return [
    [gridMap(s, (y, x) => s[y][x] * neighbourSum(s, y, x)),
      'the factor of 2 is missing — flipping a spin does not just cancel its bonds, it reverses them, so the change is 2 × s × (neighbour sum)'],
    [gridMap(s, (y, x) => -2 * s[y][x] * neighbourSum(s, y, x)),
      'the sign is inverted — a spin that already agrees with all four neighbours has the most to lose, so its flip cost is +8, not −8'],
    [gridMap(s, (y, x) => 2 * neighbourSum(s, y, x)),
      'the spin\'s own value dropped out — the cost depends on whether THIS spin agrees with the sum, so s[y][x] has to multiply it'],
    [gridMap(s, (y, x) => neighbourSum(s, y, x)),
      'that is the bare neighbour sum — the flip cost is 2 × s[y][x] × that'],
    [gridMap(s, (y, x) => -0.5 * s[y][x] * neighbourSum(s, y, x)),
      'that is this cell\'s share of the ENERGY, not the cost of flipping it — the two differ by a factor of −4'],
    [gridMap(s, (y, x) => 2 * s[y][x] * diagonalSum(s, y, x)),
      'those are the four DIAGONAL neighbours — the Ising bond runs along the axes, so vary one coordinate at a time'],
    [gridMap(s, (y, x) => 2 * s[y][x] * edgeZeroSum(s, y, x)),
      'the lattice edges are reading as 0 — this lattice is a torus, so wrap with (x + n - 1) % n instead of letting the index fall off'],
    [gridMap(s, (y, x) => 2 * s[y][x] * (neighbourSum(s, y, x) + diagonalSum(s, y, x))),
      'all eight neighbours went into the sum — the Ising model only bonds along the four axis directions, not the diagonals'],
  ];
}

// Task 3: the per-cell energy, mis-assembled.
function bondEnergyAlternatives(s) {
  return [
    [gridMap(s, (y, x) => -s[y][x] * neighbourSum(s, y, x)),
      'the ½ is missing — every bond joins two cells, so summing s × (neighbour sum) over the whole lattice counts each bond twice'],
    [gridMap(s, (y, x) => 0.5 * s[y][x] * neighbourSum(s, y, x)),
      'the minus sign is missing — agreeing neighbours LOWER the energy, so an aligned lattice has to come out at −2 per spin, not +2'],
    [gridMap(s, (y, x) => s[y][x] * neighbourSum(s, y, x)),
      'both the minus and the ½ are missing — this cell\'s share is −½ × s × (neighbour sum)'],
    [gridMap(s, (y, x) => 2 * s[y][x] * neighbourSum(s, y, x)),
      'that is task 1\'s flip cost, not the energy — the energy is −½ × s × (neighbour sum), a factor of −4 away'],
    [gridMap(s, (y, x) => -0.5 * s[y][x] * edgeZeroSum(s, y, x)),
      'the lattice edges are reading as 0 — wrap the neighbour indices with (x + n - 1) % n'],
  ];
}

// Task 2: the hash, mis-mixed. Each candidate is a whole 128×128 field.
function hashField(fn, seed) {
  const out = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) row[x] = fn(x, y, seed);
    out[y] = row;
  }
  return out;
}

function hashAlternatives(seed) {
  // The byte swap written the other way round is the identity — h % 256 is the
  // LOW byte, so putting it back in the low position changes nothing. Measured:
  // it agrees with the correct hash in 64 of 16,384 cells, so the probe is
  // genuinely distinguishable.
  const noSwap = (x, y, sd) => {
    let h = (x * 1103 + y * 2749 + sd * 3571) % 65536;
    let q = h % 2048;
    h = (h + q * q) % 65536;
    h = (h * 253 + 30011) % 65536;
    q = h % 2048;
    h = (h + q * q) % 65536;
    return h / 65536;
  };
  const noSeed = (x, y) => hash01(x, y, 0);
  const unscaled = (x, y, sd) => hash01(x, y, sd) * 65536;
  const oneRound = (x, y, sd) => {
    let h = (x * 1103 + y * 2749 + sd * 3571) % 65536;
    const q = h % 2048;
    h = (h + q * q) % 65536;
    h = (h % 256) * 256 + Math.floor(h / 256);
    h = (h * 253 + 30011) % 65536;
    return h / 65536;
  };
  return [
    [hashField(unscaled, seed),
      'the final division is missing — the mixed value is a 16-bit integer, so divide it by 65536 to land in [0, 1)'],
    [hashField(noSeed, seed),
      'the seed never reached the hash — every sweep would draw the SAME 16,384 numbers, and the simulation would freeze into a loop after two sweeps'],
    [hashField(noSwap, seed),
      'the byte swap is the wrong way round: h % 256 is already the LOW byte, so (h % 256) + Math.floor(h / 256) * 256 puts everything back where it was. The swap is (h % 256) * 256 + Math.floor(h / 256)'],
    [hashField(oneRound, seed),
      'the second mixing round is missing — one squaring plus one multiply leaves the hash so nearly affine that a cell\'s draw on one sweep correlates at 0.30 with its neighbour\'s on the next, which is exactly the pair a red-black sweep puts together'],
  ];
}

// Task 4: the red half-sweep, half done.
function halfSweepAlternatives(s, temperature, seed) {
  return [
    [naiveSweepRef(s, temperature, seed),
      'every cell moved — that is task 3\'s all-at-once sweep. A half-sweep leaves every cell with (x + y) % 2 === 1 exactly as it found it'],
    [halfSweepRef(s, 1, temperature, seed),
      'the colours are swapped — the red half-sweep updates the cells where (x + y) % 2 is 0 and passes the rest through'],
    [copyGrid(s),
      'nothing moved — not even the red cells whose flip would LOWER the energy, which are accepted unconditionally'],
    [gridMap(s, (y, x) => {
      if ((x + y) % 2 !== 0) return s[y][x];
      const dE = 2 * s[y][x] * neighbourSum(s, y, x);
      // the Boltzmann factor with its sign inside out
      return dE <= 0 || hash01(x, y, seed) < Math.exp(dE / temperature) ? -s[y][x] : s[y][x];
    }),
      'the minus sign inside the exponential is missing — exp(+dE/T) is bigger than 1 for every costly flip, so every one of them gets accepted. The Boltzmann factor is exp(−dE/T)'],
    [gridMap(s, (y, x) => {
      if ((x + y) % 2 !== 0) return s[y][x];
      const dE = 2 * s[y][x] * neighbourSum(s, y, x);
      // the comparison the wrong way round
      return dE <= 0 || hash01(x, y, seed) > Math.exp(-dE / temperature) ? -s[y][x] : s[y][x];
    }),
      'the comparison is the wrong way round — a costly flip should be accepted when the random number falls BELOW exp(−dE/T), which is the rare case, not above it'],
    [gridMap(s, (y, x) => {
      if ((x + y) % 2 !== 0) return s[y][x];
      const dE = 2 * s[y][x] * neighbourSum(s, y, x);
      return hash01(x, y, seed) < Math.exp(-dE / temperature) && dE > 0 ? -s[y][x] : s[y][x];
    }),
      'the downhill flips are being thrown away — the dE > 0 guard excludes them from the gamble instead of letting them through first. A flip with dE ≤ 0 is accepted every time, unconditionally, and those are most of the moves'],
  ];
}

// Task 5: the two halves, mis-chained.
function fullSweepAlternatives(s, temperature, sweep) {
  return [
    [halfSweepRef(s, 1, temperature, sweep * 2 + 1),
      'only the black half reached the answer — the halves have to be chained, black(red(s)), or the red half\'s work is thrown away'],
    [halfSweepRef(s, 0, temperature, sweep * 2),
      'the black half changed nothing — check that its parity test keeps the cells where (x + y) % 2 is 1 and passes the rest through'],
    [halfSweepRef(halfSweepRef(s, 1, temperature, sweep * 2), 0, temperature, sweep * 2 + 1),
      'the halves ran in the wrong order — red first with seed 2k, then black on the red half\'s output with seed 2k + 1'],
    [naiveSweepRef(naiveSweepRef(s, temperature, sweep * 2), temperature, sweep * 2 + 1),
      'neither kernel is testing the parity — both halves are updating the whole lattice, which is task 3\'s race run twice per sweep'],
  ];
}

// ---- console readers -------------------------------------------------------

// The series a plot() call carried, by name. Tasks 3, 5 and 6 assert on the
// learner's own trace, and the plot payload is where it lands.
function plottedSeries(ctx, name) {
  for (const line of ctx.logs || []) {
    if (line.type !== 'plot' || !line.plot) continue;
    for (const series of line.plot.series || []) {
      if (!name || series.name === name) return series.values;
    }
  }
  return null;
}

function loggedNumber(ctx, pattern) {
  for (const line of ctx.logs || []) {
    if (line.type !== 'log' || !line.text) continue;
    const match = pattern.exec(String(line.text));
    if (match && Number.isFinite(Number(match[1]))) return Number(match[1]);
  }
  return null;
}

function renderedFrames(ctx) {
  return (ctx.logs || []).filter(line => line.type === 'canvas').length;
}

// ---- shared kernel source (prewired into later starters) -------------------

// The hash body, exactly as task 2 asks for it. Reused verbatim by every sweep
// kernel from task 3 on, so a learner reads the same seven lines each time.
const HASH_BODY = `  // a random number for THIS thread, this sweep — no shared state, no ordering
  let h = (x * 1103 + y * 2749 + seed * 3571) % 65536;
  let q = h % 2048;
  h = (h + q * q) % 65536;
  h = (h % 256) * 256 + Math.floor(h / 256);
  h = (h * 253 + 30011) % 65536;
  q = h % 2048;
  h = (h + q * q) % 65536;
  const u = h / 65536;`;

const NEIGHBOUR_SUM = `  const sum = s[y][(x + 1) % n] + s[y][(x + n - 1) % n]
            + s[(y + 1) % n][x] + s[(y + n - 1) % n][x];`;

const COORDS = `  const x = this.thread.x;
  const y = this.thread.y;
  const n = this.constants.size;`;

const OUTPUT = `{ output: [128, 128], constants: { size: 128 } }`;

const FLIP_COST_KERNEL = `const flipCost = gpu.createKernel(function (s) {
${COORDS}
${NEIGHBOUR_SUM}
  return 2 * s[y][x] * sum;
}, ${OUTPUT});`;

const BOND_ENERGY_KERNEL = `const bondEnergy = gpu.createKernel(function (s) {
${COORDS}
${NEIGHBOUR_SUM}
  return -0.5 * s[y][x] * sum;
}, ${OUTPUT});`;

const NAIVE_KERNEL = `const naiveSweep = gpu.createKernel(function (s, temperature, seed) {
${COORDS}
  const spin = s[y][x];
${NEIGHBOUR_SUM}
  const dE = 2 * spin * sum;
${HASH_BODY}
  if (dE <= 0) return -spin;
  if (u < Math.exp(-dE / temperature)) return -spin;
  return spin;
}, ${OUTPUT});`;

function halfKernel(name, parity) {
  return `const ${name} = gpu.createKernel(function (s, temperature, seed) {
${COORDS}
  const spin = s[y][x];
  // parity as a NUMBER — gpu.js cannot keep a boolean in a kernel variable
  const parity = (x + y) % 2;
  if (parity !== ${parity}) return spin;
${NEIGHBOUR_SUM}
  const dE = 2 * spin * sum;
${HASH_BODY}
  if (dE <= 0) return -spin;
  if (u < Math.exp(-dE / temperature)) return -spin;
  return spin;
}, ${OUTPUT});`;
}

const RED_KERNEL = halfKernel('red', 0);
const BLACK_KERNEL = halfKernel('black', 1);

const PAINT_KERNEL = `const paint = gpu.createKernel(function (s) {
  const spin = s[this.thread.y][this.thread.x];
  if (spin > 0) {
    this.color(0.97, 0.58, 0.26, 1);
  } else {
    this.color(0.09, 0.14, 0.28, 1);
  }
}, { output: [128, 128], graphical: true });`;

const MEAN_HELPER = `function meanOf(grid) {
  let total = 0;
  for (let y = 0; y < 128; y++) {
    for (let x = 0; x < 128; x++) total += grid[y][x];
  }
  return total / (128 * 128);
}`;

export default {
  uuid: '1f12d841-3c62-465a-8cd3-846212d0ef40',
  version: 1,
  slug: 'ising-model',
  title: 'The Ising Model: Colour to Break the Race',
  blurb:
    'Metropolis on a lattice of spins, the race that makes an all-at-once update silently wrong, ' +
    'and the checkerboard that repairs it — ending in a temperature slider you can drag through a phase transition.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'flip-energy',
      title: 'What a Flip Would Cost',
      intro: `<p>A magnet, stripped to almost nothing: a grid of arrows, each one either
        <strong>up</strong> (<code>+1</code>) or <strong>down</strong> (<code>−1</code>), and a single
        rule — neighbours would rather agree. Ernst Ising's 1925 model is that sentence and no more,
        and it is still the standard proving ground for phase transitions, because it is simple
        enough to be solved exactly and rich enough to have one.</p>
        <p>Agreement is written as energy. Every neighbouring pair contributes <code>−s·s'</code>:
        <code>−1</code> when the two agree, <code>+1</code> when they do not, so the lattice's total
        energy falls as more spins line up. Now ask what one spin flipping would do. Its four bonds
        all reverse, so the change is</p>
<pre><code>ΔE = 2 · s · (up + down + left + right)</code></pre>
        <p>and because every spin and every neighbour is <code>±1</code>, that has exactly five possible
        values: <code>−8</code> when all four neighbours disagree with you (flipping is a bargain),
        through <code>0</code>, up to <code>+8</code> when all four agree (flipping is the most expensive
        move on the board). This lattice is a <strong>torus</strong> — the right edge is glued to the
        left, the top to the bottom — so there are no boundary cells to special-case, and every one of
        the 16,384 threads does the identical four reads. Pure gather, the shape Thinking in Parallel
        calls the one that always parallelises.</p>`,
      goal: `<strong>Goal:</strong> finish the kernel so cell <code>[y][x]</code> holds
        <code>2 · s[y][x] · (sum of its four neighbours)</code>, with the neighbour indices wrapping
        round the lattice.`,
      requirements: [
        'Sum the four axis neighbours with wrap-around — no diagonals, no boundary special case',
        'Wrap by adding the size first: <code>(x + n - 1) % n</code>, never <code>(x - 1) % n</code>',
        'Return <code>2 * s[y][x] * sum</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — where the 2 comes from',
          body: `<p>This spin's share of the energy is <code>−s · (neighbour sum)</code>. Flip it and the
            share becomes <code>+s · (neighbour sum)</code>. The difference between those two is
            <code>2 · s · (neighbour sum)</code> — the bonds do not just vanish, they reverse, so the
            change is twice the share, not once.</p>`,
        },
        {
          title: 'Hint 2 — the wrap that bites',
          body: `<p><code>(x - 1) % n</code> is <code>-1</code> at <code>x = 0</code>, in JavaScript and in
            gpu.js's GLSL alike — both keep the sign of the left operand. Reading
            <code>s[y][-1]</code> gives you <code>undefined</code> on the CPU backend and a zero out of
            nowhere on WebGL. Add the size before you subtract:</p>
<pre><code>const left  = s[y][(x + n - 1) % n];
const right = s[y][(x + 1) % n];
const up    = s[(y + n - 1) % n][x];
const down  = s[(y + 1) % n][x];</code></pre>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<pre><code>const sum = s[y][(x + 1) % n] + s[y][(x + n - 1) % n]
          + s[(y + 1) % n][x] + s[(y + n - 1) % n][x];
return 2 * s[y][x] * sum;</code></pre>`,
        },
      ],
      transfer: `A four-neighbour gather with periodic wrap is one of the most-written kernels there
        is — CUDA and WGSL spell it the same way, and a graphics API gives you the wrap for free by
        setting a texture's address mode to <code>repeat</code> instead of doing modular arithmetic
        at all. The five-value structure matters too: a stencil whose result comes from a tiny
        discrete set is exactly the case where production Ising codes drop the exponential and look
        the answer up in a five-entry table.`,
      starterCode: `// One thread per spin. Each one works out what flipping ITSELF would cost.
const gpu = new GPU({ mode });

const flipCost = gpu.createKernel(function (s) {
  const x = this.thread.x;
  const y = this.thread.y;
  const n = this.constants.size;
  // TODO 1: sum the four neighbours, wrapping round the torus:
  //         s[y][(x + 1) % n], s[y][(x + n - 1) % n],
  //         s[(y + 1) % n][x], s[(y + n - 1) % n][x]
  // TODO 2: return 2 * (this spin) * (that sum).
  return 0;
}, { output: [128, 128], constants: { size: 128 } });

const cost = await flipCost(lattice);
console.log('the spin at [0][0] is', lattice[0][0], '— flipping it would cost', cost[0][0]);
console.log('row 0, first ten costs:', Array.from(cost[0]).slice(0, 10));
`,
      solutionCode: `// One thread per spin. Each one works out what flipping ITSELF would cost.
const gpu = new GPU({ mode });

${FLIP_COST_KERNEL}

const cost = await flipCost(lattice);
console.log('the spin at [0][0] is', lattice[0][0], '— flipping it would cost', cost[0][0]);
console.log('row 0, first ten costs:', Array.from(cost[0]).slice(0, 10));
`,
      inputs: utils => ({ lattice: makeLattice(utils, 20260801) }),
      publicTests: [
        {
          name: 'the kernel produces a 128×128 grid of flip costs',
          run: async ctx => {
            assertRunOk(ctx);
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(makeLattice(ctx.utils, 20260801));
            ctx.assert(out && out.length === SIZE, `expected 128 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === SIZE, 'each row should hold 128 values');
          },
        },
        {
          name: 'every cost is one of <code>−8, −4, 0, 4, 8</code>',
          run: async ctx => {
            // A structural check before an arithmetic one: it tells a learner
            // whose formula is off by a factor that the SHAPE is wrong, which is
            // more use than a first failing cell.
            const s = makeLattice(ctx.utils, 20260801);
            const out = await ctx.kernel(s);
            const offLattice = wrapHint(out);
            ctx.assert(!offLattice, offLattice || 'every cell holds a finite number');
            const allowed = new Set([-8, -4, 0, 4, 8]);
            for (let y = 0; y < SIZE; y += 7) {
              for (let x = 0; x < SIZE; x += 5) {
                ctx.assert(
                  allowed.has(Math.round(out[y][x] * 1e6) / 1e6),
                  `cell [${y}][${x}] holds ${out[y][x]} — with ±1 spins the four-neighbour sum is one of ` +
                    `−4, −2, 0, 2, 4, so 2 · s · sum can only be −8, −4, 0, 4 or 8`
                );
              }
            }
          },
        },
        {
          name: 'cell [y][x] equals <code>2 · s[y][x] · (neighbour sum)</code>',
          run: async ctx => {
            const s = makeLattice(ctx.utils, 20260801);
            const out = await ctx.kernel(s);
            const ref = flipCostRef(s);
            const hint = wrapHint(out) || diagnoseGrid(out, ref, 1e-4, flipCostAlternatives(s));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'the torus has no edges — row 0 and column 0 wrap round',
          run: async ctx => {
            // The wrap is invisible in the interior and decides every edge cell,
            // so it gets its own test against a lattice built to make the two
            // answers differ: the last row and column are the opposite of the
            // first, so a cell that reads a zero (or nothing) cannot match.
            const s = makeAligned();
            for (let i = 0; i < SIZE; i++) {
              s[SIZE - 1][i] = -1;
              s[i][SIZE - 1] = -1;
            }
            const out = await ctx.kernel(s);
            const ref = flipCostRef(s);
            const hint = wrapHint(out) || diagnoseGrid(out, ref, 1e-4, flipCostAlternatives(s));
            for (let i = 0; i < SIZE; i++) {
              const cells = [[0, i], [i, 0], [SIZE - 1, i], [i, SIZE - 1]];
              for (const [y, x] of cells) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint ||
                  `edge cell [${y}][${x}] — this lattice is a torus, so the neighbour off the right ` +
                  `edge is column 0 and the one above row 0 is row 127. Wrap with (x + 1) % n and ` +
                  `(x + n - 1) % n`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Two lattices whose answer is a single number everywhere: aligned
            // is the most expensive board there is, a perfect checkerboard the
            // cheapest. Nothing about a factor or a sign survives both.
            const aligned = makeAligned();
            const outA = await ctx.kernel(aligned);
            const hintA = diagnoseGrid(outA, flipCostRef(aligned), 1e-4, flipCostAlternatives(aligned));
            for (let y = 0; y < SIZE; y += 3) {
              for (let x = 0; x < SIZE; x += 3) {
                ctx.assertClose(outA[y][x], 8, 1e-4,
                  hintA || `cell [${y}][${x}] of an all-up lattice — all four neighbours agree, so flipping costs +8`);
              }
            }
            const checker = makeCheckerboard();
            const outC = await ctx.kernel(checker);
            const hintC = diagnoseGrid(outC, flipCostRef(checker), 1e-4, flipCostAlternatives(checker));
            for (let y = 0; y < SIZE; y += 3) {
              for (let x = 0; x < SIZE; x += 3) {
                ctx.assertClose(outC[y][x], -8, 1e-4,
                  hintC || `cell [${y}][${x}] of a perfect checkerboard — all four neighbours disagree, so flipping gains 8`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            const s = makeLattice(ctx.utils, 31337);
            const out = await ctx.kernel(s);
            const ref = flipCostRef(s);
            const hint = wrapHint(out) || diagnoseGrid(out, ref, 1e-4, flipCostAlternatives(s));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'thread-local-random',
      title: 'Randomness Without a Random Number Generator',
      intro: `<p>The rule that makes this a <em>model of temperature</em> rather than a downhill slide
        needs a coin toss per spin per sweep. On a CPU you reach for <code>Math.random()</code> without
        thinking. Think about it here: an ordinary generator is a <strong>stream</strong> — one hidden
        state, advanced once per call, so call number 900 depends on call number 899. That is a
        sequential dependency, and it is the same one Thinking in Parallel rules out. There is no
        ordering between 16,384 threads and nothing they can share. gpu.js is blunt about it too: the
        WebGPU backend <em>refuses</em> <code>Math.random</code> at compile time, and on WebGL a kernel
        that calls it is no longer reproducible.</p>
        <p>So stop asking for a stream and ask for a <strong>function</strong>:
        <code>u = hash(x, y, seed)</code>. Same thread, same sweep, same number — every time, on every
        backend. Different thread, unrelated number. No state, no ordering, and a run you can replay
        exactly, which is what makes a stochastic simulation debuggable at all. This is not a
        compromise for gpu.js's benefit; it is what production GPU Monte Carlo does.</p>
        <p>A hash has one job: make the output look nothing like the input. Three moves do it. The
        <em>fold</em> multiplies <code>x</code>, <code>y</code> and <code>seed</code> by unrelated odd
        constants and adds them, which spreads the three inputs across sixteen bits. The
        <em>squaring</em> step is the only nonlinear one, and without it a hash is just an affine map:
        neighbouring cells would come out a fixed distance apart, which is not randomness, it is a
        ramp. The <em>byte swap</em> moves the high eight bits down where the next multiply can reach
        them. Every intermediate stays under <code>2²⁴</code> — the largest integer a 32-bit float
        holds exactly — so WebGPU, WebGL and the CPU compute the identical value and nothing about your
        run depends on which one you got.</p>`,
      goal: `<strong>Goal:</strong> fill a 128×128 grid with <code>hash(x, y, seed)</code> in
        <code>[0, 1)</code>, using no <code>Math.random</code> anywhere.`,
      requirements: [
        'Fold <code>x</code>, <code>y</code> and <code>seed</code> into one value below <code>65536</code>',
        'Mix it twice: square a low slice, swap the two bytes, one multiply-and-add round',
        'Return the mixed value divided by <code>65536</code> so it lands in <code>[0, 1)</code>',
        'No <code>Math.random</code> — the same seed must give the same field every run',
      ],
      hints: [
        {
          title: 'Hint 1 — the fold, and why the seed belongs in it',
          body: `<p>Three odd constants, one modulo:</p>
<pre><code>let h = (x * 1103 + y * 2749 + seed * 3571) % 65536;</code></pre>
<p>Leave <code>seed</code> out and the field is the same 16,384 numbers on every sweep — the
            simulation would take one step and then repeat it forever. The seed is what makes this a
            <em>sequence</em> of fields rather than one field.</p>`,
        },
        {
          title: 'Hint 2 — the two mixing moves',
          body: `<p>Squaring is the nonlinear part. <code>h % 2048</code> keeps the value small enough
            that <code>q * q</code> stays under <code>2²⁴</code> and the arithmetic is still exact:</p>
<pre><code>let q = h % 2048;
h = (h + q * q) % 65536;</code></pre>
<p>The swap exchanges the high and low bytes. Note which side gets the multiply —
            <code>h % 256</code> is the LOW byte, so it is the one that has to move UP:</p>
<pre><code>h = (h % 256) * 256 + Math.floor(h / 256);</code></pre>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<pre><code>let h = (x * 1103 + y * 2749 + seed * 3571) % 65536;
let q = h % 2048;
h = (h + q * q) % 65536;
h = (h % 256) * 256 + Math.floor(h / 256);
h = (h * 253 + 30011) % 65536;
q = h % 2048;
h = (h + q * q) % 65536;
return h / 65536;</code></pre>
<p>Two rounds, not one — and the reason is not where you would look for it. Inside a
            single field one round is already fine: neighbouring cells measure <code>0.021</code>
            against the two-round hash's <code>0.032</code>, which is the same noise. The damage is
            <em>between consecutive seeds</em>. One squaring plus one multiply is still so nearly
            affine that <code>hash(x, y, k)</code> and <code>hash(x+1, y+1, k+1)</code> correlate at
            <code>0.30</code>, against <code>0.007</code>–<code>0.021</code> with two rounds — and that is precisely the
            pair the next task puts side by side, a red cell drawing at seed <code>2k</code> and the
            black neighbour that reads it drawing at <code>2k + 1</code>.</p>`,
        },
      ],
      transfer: `Counter-based randomness — hash a coordinate instead of advancing a stream — is the
        standard on every parallel platform: Random123 / Philox in CUDA, <code>curand</code>'s
        counter-based generators, and essentially every shader that needs noise. The reason is the same
        everywhere: a stream forces an order on things that have none, while a hash gives every thread
        an independent draw for free and makes the whole run reproducible. Keeping the arithmetic
        inside the exactly-representable integer range is the other half of the trick, and it is why
        real GPU hashes are written in integer types rather than floats.`,
      starterCode: `// 16,384 threads, 16,384 random numbers, and no random number generator.
const gpu = new GPU({ mode });

const noise = gpu.createKernel(function (seed) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO 1: fold x, y and seed into one value below 65536.
  // TODO 2: mix it twice — square a low slice, swap the two bytes,
  //         one multiply-and-add round, then square again.
  // TODO 3: return the mixed value / 65536.
  return 0;
}, { output: [128, 128] });

const field = await noise(1);

// A histogram of all 16,384 values, in plain JavaScript.
const bins = new Array(32).fill(0);
let total = 0;
for (let y = 0; y < 128; y++) {
  for (let x = 0; x < 128; x++) {
    bins[Math.min(31, Math.floor(field[y][x] * 32))]++;
    total += field[y][x];
  }
}
console.log('mean of the field:', total / (128 * 128));
console.log('counts per bin:', bins);
plot({ 'cells per bin': bins }, { title: 'hash output, 32 equal bins of [0, 1)' });
`,
      solutionCode: `// 16,384 threads, 16,384 random numbers, and no random number generator.
const gpu = new GPU({ mode });

const noise = gpu.createKernel(function (seed) {
  const x = this.thread.x;
  const y = this.thread.y;
  let h = (x * 1103 + y * 2749 + seed * 3571) % 65536;
  let q = h % 2048;
  h = (h + q * q) % 65536;
  h = (h % 256) * 256 + Math.floor(h / 256);
  h = (h * 253 + 30011) % 65536;
  q = h % 2048;
  h = (h + q * q) % 65536;
  return h / 65536;
}, { output: [128, 128] });

const field = await noise(1);

// A histogram of all 16,384 values, in plain JavaScript.
const bins = new Array(32).fill(0);
let total = 0;
for (let y = 0; y < 128; y++) {
  for (let x = 0; x < 128; x++) {
    bins[Math.min(31, Math.floor(field[y][x] * 32))]++;
    total += field[y][x];
  }
}
console.log('mean of the field:', total / (128 * 128));
console.log('counts per bin:', bins);
plot({ 'cells per bin': bins }, { title: 'hash output, 32 equal bins of [0, 1)' });
`,
      publicTests: [
        {
          name: 'the field is 128×128 and every value lies in <code>[0, 1)</code>',
          run: async ctx => {
            assertRunOk(ctx);
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(1);
            ctx.assert(out && out.length === SIZE && out[0].length === SIZE,
              'expected a 128×128 field');
            let lo = Infinity;
            let hi = -Infinity;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                lo = Math.min(lo, out[y][x]);
                hi = Math.max(hi, out[y][x]);
              }
            }
            ctx.assert(
              lo >= 0 && hi < 1,
              hi >= 65535
                ? `values run up to ${hi} — the final division is missing. The mixed value is a 16-bit integer, so return h / 65536`
                : `values run from ${lo} to ${hi}; a hash in [0, 1) has to stay inside those bounds`
            );
          },
        },
        {
          name: 'the values spread out — thousands of distinct ones, evenly binned',
          run: async ctx => {
            // Uniformity before exactness: it names "this is not noise at all"
            // for a learner whose mixing collapsed, without depending on the
            // exact constants.
            const out = await ctx.kernel(1);
            const distinct = new Set();
            const bins = new Array(32).fill(0);
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                distinct.add(out[y][x]);
                bins[Math.min(31, Math.max(0, Math.floor(out[y][x] * 32)))]++;
              }
            }
            // Measured with the reference hash: 12,808 distinct values.
            ctx.assert(distinct.size > 8000,
              `only ${distinct.size} distinct values across 16,384 cells — the mixing has collapsed. ` +
                `The reference hash produces about 12,800`);
            const expected = CELLS / 32;
            for (let b = 0; b < 32; b++) {
              ctx.assert(
                Math.abs(bins[b] - expected) < expected * 0.35,
                `bin ${b} of 32 holds ${bins[b]} of the 16,384 values, where an even spread would put ` +
                  `about ${expected} in it — the output is not uniform over [0, 1)`
              );
            }
          },
        },
        {
          name: 'the hash matches, cell for cell',
          run: async ctx => {
            const out = await ctx.kernel(1);
            const ref = hashField(hash01, 1);
            const hint = diagnoseGrid(out, ref, 1e-9, hashAlternatives(1));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-9, hint || `cell [${y}][${x}] at seed 1`);
              }
            }
          },
        },
        {
          name: 'a different seed gives a different field — and the same seed repeats exactly',
          run: async ctx => {
            // The two failures this catches are the two that matter: a hash that
            // ignores its seed freezes the simulation after one sweep, and a
            // hash built on Math.random cannot be replayed at all.
            const a = await ctx.kernel(1);
            const b = await ctx.kernel(2);
            const again = await ctx.kernel(1);
            let differ = 0;
            let unstable = 0;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                if (a[y][x] !== b[y][x]) differ++;
                if (a[y][x] !== again[y][x]) unstable++;
              }
            }
            ctx.assert(
              unstable === 0,
              `${unstable} cells changed between two calls with the SAME seed — is Math.random in the ` +
                `kernel? A hash of (x, y, seed) has to replay exactly, and gpu.js's WebGPU backend ` +
                `refuses Math.random outright`
            );
            ctx.assert(
              differ > CELLS * 0.9,
              `only ${differ} of 16,384 cells changed when the seed did — the seed is barely reaching ` +
                `the hash. Every sweep needs its own field, or the simulation repeats one step forever`
            );
          },
        },
        {
          name: 'the console reports a mean near <code>0.5</code>',
          run: async ctx => {
            const got = loggedNumber(ctx, /mean of the field:\s*(-?[0-9][0-9.eE+-]*)/);
            ctx.assert(got !== null,
              'the console never reported a mean — leave the prewired histogram and its console.log in place');
            // A probe list would be dead weight here — the only near-miss worth
            // naming is off by four orders of magnitude, so a threshold says it
            // and a tolerance never could.
            ctx.assert(
              Math.abs(got - 0.5) <= 0.02,
              got > 100
                ? `the mean came back as ${got.toFixed(0)} — the final division is missing. The mixed ` +
                  `value is a 16-bit integer, so return h / 65536`
                : `a uniform [0, 1) field has mean 0.5; this one reported ${got}`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A seed the learner never saw, and one large enough to prove the
            // arithmetic has not silently left the exact-integer range.
            for (const seed of [7, 299]) {
              const out = await ctx.kernel(seed);
              const ref = hashField(hash01, seed);
              const hint = diagnoseGrid(out, ref, 1e-9, hashAlternatives(seed));
              for (let y = 0; y < SIZE; y++) {
                for (let x = 0; x < SIZE; x++) {
                  ctx.assertClose(out[y][x], ref[y][x], 1e-9,
                    hint || `cell [${y}][${x}] at seed ${seed}`);
                }
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // TWO lag families, because they are not the same test and only the
            // second one can see the mistake this task is really guarding:
            //
            //   • within one field — catches a hash whose output is a ramp.
            //   • between consecutive SEEDS — catches a hash that mixes too
            //     little. A single-round hash passes the first comfortably
            //     (0.021, against 0.032 for the reference: identical noise) and
            //     fails the second at 0.30, because one squaring plus one
            //     multiply is still nearly affine in the seed. That is the lag
            //     the next task cares about: a red cell draws at seed 2k and the
            //     black neighbour that reads its result draws at 2k + 1.
            //
            // Thresholds: 0.08, against a reference worst of 0.021 measured over
            // seeds 0…60 on the cross-seed lags and 0.032 on the spatial ones.
            const a = await ctx.kernel(11);
            const b = await ctx.kernel(12);
            const corr = (left, right, dx, dy) => {
              let sa = 0;
              let sb = 0;
              let sab = 0;
              let saa = 0;
              let sbb = 0;
              for (let y = 0; y < SIZE; y++) {
                for (let x = 0; x < SIZE; x++) {
                  const p = left[y][x];
                  const q = right[(y + dy) % SIZE][(x + dx) % SIZE];
                  sa += p; sb += q; sab += p * q; saa += p * p; sbb += q * q;
                }
              }
              const cov = sab / CELLS - (sa / CELLS) * (sb / CELLS);
              const sd = Math.sqrt((saa / CELLS - (sa / CELLS) ** 2) * (sbb / CELLS - (sb / CELLS) ** 2));
              return cov / sd;
            };
            for (const [dx, dy] of [[1, 0], [0, 1], [2, 0], [0, 2], [1, 1]]) {
              const c = corr(a, a, dx, dy);
              ctx.assert(
                Math.abs(c) < 0.08,
                `within one field, cells ${dx} across and ${dy} down correlate at ${c.toFixed(3)} — ` +
                  `that is structure, not noise: the output is still a ramp in x and y rather than a hash`
              );
            }
            for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
              const c = corr(a, b, dx, dy);
              // Two very different failures land here and they must not share a
              // diagnosis: a correlation of 1 is a hash that ignored its seed
              // (the same field twice), while ~0.3 is a hash that mixed once.
              const why = Math.abs(c) > 0.9
                ? `the two fields are the same field — the seed is not reaching the hash at all`
                : `one mixing round measures 0.30 here; the second round is what breaks it`;
              ctx.assert(
                Math.abs(c) < 0.08,
                `the field at seed 11 correlates at ${c.toFixed(3)} with the field at seed 12, ` +
                  `${dx} across and ${dy} down — ${why}. Consecutive seeds are consecutive ` +
                  `half-sweeps, so a red cell's draw and its black neighbour's next draw would move ` +
                  `together`
              );
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'the-race',
      title: 'Everyone at Once Is Wrong',
      intro: `<p>Now the dynamics. The <strong>Metropolis</strong> rule is two lines: work out what
        flipping this spin would cost; if the cost is zero or negative, flip it; if it is positive,
        flip it anyway with probability <code>exp(−ΔE / T)</code>. That exponential is the whole of
        temperature — near <code>T = 0</code> an expensive flip essentially never happens and the
        lattice freezes into agreement; at large <code>T</code> almost everything is accepted and the
        lattice is noise. You have both halves already: task 1's cost and task 2's <code>u</code>.</p>
        <p>The obvious GPU move is to do all 16,384 at once, and it looks watertight. Every thread
        reads the old lattice and writes only its own cell, so there is no memory race — nothing is
        overwritten while something else is reading it. The race is in the <em>physics</em>. Each spin
        computed its cost on the assumption that its neighbours would hold still, and they did not.
        Two neighbours that agree share one bond; each one separately prices the cost of breaking it;
        both flip; the bond is not broken at all, and both of them paid for a change that never
        happened.</p>
        <p>Don't take it on trust — measure it. The energy per spin is
        <code>−½ · mean(s · neighbour sum)</code>, the same five reads as task 1 assembled differently,
        with the <code>½</code> because each bond is counted once from each end. It runs from
        <code>−2</code> (perfectly aligned) through <code>0</code> (random) to <code>+2</code> (a
        perfect checkerboard, every neighbour disagreeing). Metropolis at <code>T = 1.5</code> should
        walk downhill from a random start. Run it all-at-once and watch which way it actually goes.</p>`,
      goal: `<strong>Goal:</strong> write the <code>bondEnergy</code> kernel and the
        <code>meanOf</code> helper, then read the energy the prewired all-at-once loop prints.`,
      requirements: [
        'The kernel returns <code>-0.5 * s[y][x] * (neighbour sum)</code>, wrapping as in task 1',
        '<code>meanOf(grid)</code> averages all 128 × 128 cells',
        'Leave the prewired <code>naiveSweep</code> kernel and its 30-sweep loop as they are',
      ],
      hints: [
        {
          title: 'Hint 1 — why the ½',
          body: `<p>The bond between two neighbours belongs to both of them. If every cell claimed the
            full <code>−s · (neighbour sum)</code>, summing over the lattice would count each bond
            twice, and an aligned lattice would report <code>−4</code> per spin instead of
            <code>−2</code>. Halving each cell's claim fixes it exactly.</p>`,
        },
        {
          title: 'Hint 2 — the same reads as task 1',
          body: `<p>Identical neighbour sum, different assembly: the flip cost multiplies it by
            <code>2 · s</code>, the energy by <code>−½ · s</code>.</p>
<pre><code>return -0.5 * s[y][x] * sum;</code></pre>`,
        },
        {
          title: 'Hint 3 — the mean',
          body: `<p><code>bondEnergy(s)</code> hands back an ordinary 2D grid of numbers, so this is
            plain JavaScript:</p>
<pre><code>let total = 0;
for (let y = 0; y &lt; 128; y++) {
  for (let x = 0; x &lt; 128; x++) total += grid[y][x];
}
return total / (128 * 128);</code></pre>
<p>Totalling a grid <em>on</em> the GPU is the halving ladder Reductions builds; at 16,384
            cells the read-back is cheaper than the ladder, so this one stays in JavaScript.</p>`,
        },
      ],
      transfer: `The failure you are about to watch is not a gpu.js quirk, it is what "synchronous
        Metropolis" does on any platform: the update rule was derived for one spin moving against a
        fixed background, and running it in parallel silently changes the algorithm into a different,
        wrong one. The general lesson is worth more than the physics — a parallel version of a
        sequential algorithm is a <em>different algorithm</em> until you have proved otherwise, and
        "no thread writes another thread's memory" proves nothing about whether the maths still holds.`,
      starterCode: `// Metropolis, applied to every spin at once. Watch the energy.
const gpu = new GPU({ mode });

${NAIVE_KERNEL}

const bondEnergy = gpu.createKernel(function (s) {
  const x = this.thread.x;
  const y = this.thread.y;
  const n = this.constants.size;
  // TODO 1: same four wrapped neighbours as task 1, summed.
  // TODO 2: return this cell's share of the energy: -0.5 * s[y][x] * sum.
  return 0;
}, { output: [128, 128], constants: { size: 128 } });

function meanOf(grid) {
  // TODO 3: average all 128 x 128 cells of grid.
  return 0;
}

let s = lattice;
console.log('energy per spin at the start:', meanOf(await bondEnergy(s)));

const trace = [];
for (let k = 0; k < 30; k++) {
  s = await naiveSweep(s, 1.5, k);
  trace.push(meanOf(await bondEnergy(s)));
}

console.log('energy per spin, sweep by sweep:', trace);
console.log('after 30 all-at-once sweeps: E =', trace[29]);
plot({ 'energy per spin': trace }, { title: 'all-at-once Metropolis at T = 1.5' });
`,
      solutionCode: `// Metropolis, applied to every spin at once. Watch the energy.
const gpu = new GPU({ mode });

${NAIVE_KERNEL}

${BOND_ENERGY_KERNEL}

${MEAN_HELPER}

let s = lattice;
console.log('energy per spin at the start:', meanOf(await bondEnergy(s)));

const trace = [];
for (let k = 0; k < 30; k++) {
  s = await naiveSweep(s, 1.5, k);
  trace.push(meanOf(await bondEnergy(s)));
}

console.log('energy per spin, sweep by sweep:', trace);
console.log('after 30 all-at-once sweeps: E =', trace[29]);
plot({ 'energy per spin': trace }, { title: 'all-at-once Metropolis at T = 1.5' });
`,
      inputs: utils => ({ lattice: makeLattice(utils, 20260801) }),
      publicTests: [
        {
          name: 'the energy kernel returns <code>−½ · s · (neighbour sum)</code>',
          run: async ctx => {
            assertRunOk(ctx);
            ctx.assert(ctx.kernels.length >= 2,
              `expected 2 kernels (the prewired naiveSweep, then bondEnergy), found ${ctx.kernels.length}`);
            const s = makeLattice(ctx.utils, 20260801);
            const out = await ctx.kernels[1](s);
            ctx.assert(out && out.length === SIZE && out[0].length === SIZE,
              'expected a 128×128 energy grid');
            const ref = bondEnergyRef(s);
            const hint = wrapHint(out) || diagnoseGrid(out, ref, 1e-4, bondEnergyAlternatives(s));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'the two extremes come out at <code>−2</code> and <code>+2</code> per spin',
          run: async ctx => {
            const aligned = makeAligned();
            const outA = await ctx.kernels[1](aligned);
            const hintA = diagnoseGrid(outA, bondEnergyRef(aligned), 1e-4, bondEnergyAlternatives(aligned));
            for (let y = 0; y < SIZE; y += 3) {
              for (let x = 0; x < SIZE; x += 3) {
                ctx.assertClose(outA[y][x], -2, 1e-4,
                  hintA || `cell [${y}][${x}] of an all-up lattice — every bond satisfied is −2 per spin`);
              }
            }
            const checker = makeCheckerboard();
            const outC = await ctx.kernels[1](checker);
            const hintC = diagnoseGrid(outC, bondEnergyRef(checker), 1e-4, bondEnergyAlternatives(checker));
            for (let y = 0; y < SIZE; y += 3) {
              for (let x = 0; x < SIZE; x += 3) {
                ctx.assertClose(outC[y][x], 2, 1e-4,
                  hintC || `cell [${y}][${x}] of a perfect checkerboard — every bond broken is +2 per spin`);
              }
            }
          },
        },
        {
          name: 'the plotted trace matches the energy this run really had',
          run: async ctx => {
            // Checks the kernel and meanOf together, through the learner's own
            // numbers: meanOf is theirs, so its only visible effect is the trace.
            const trace = plottedSeries(ctx, 'energy per spin');
            ctx.assert(Array.isArray(trace) && trace.length === RACE_SWEEPS,
              `expected a plotted 'energy per spin' series of ${RACE_SWEEPS} points, got ` +
                `${trace ? trace.length : 'none'} — leave the prewired plot() call in place`);
            const ref = traceRef(makeLattice(ctx.utils, 20260801), RACE_SWEEPS, RACE_T, naiveSweepRef);
            for (let k = 0; k < RACE_SWEEPS; k++) {
              const hint = diagnose(trace[k], ref[k], 2e-3, [
                [ref[k] * 2, 'every value is twice the energy — the ½ is missing, so each bond is being counted from both ends'],
                [-ref[k], 'the whole trace has the wrong sign — agreeing neighbours LOWER the energy'],
                [ref[k] * CELLS, 'that is the TOTAL, not the mean — meanOf has to divide by 128 × 128'],
              ]);
              ctx.assertClose(trace[k], ref[k], 2e-3, hint || `the energy plotted after sweep ${k + 1}`);
            }
          },
        },
        {
          name: 'the energy goes <em>up</em> — the race, in one number',
          run: async ctx => {
            const trace = plottedSeries(ctx, 'energy per spin');
            ctx.assert(Array.isArray(trace) && trace.length === RACE_SWEEPS, 'no energy trace was plotted');
            const start = loggedNumber(ctx, /energy per spin at the start:\s*(-?[0-9][0-9.eE+-]*)/);
            ctx.assert(start !== null, 'the console never reported the starting energy');
            ctx.assertClose(start, energyRef(makeLattice(ctx.utils, 20260801)), 2e-3,
              'the energy of the starting lattice — a random ±1 lattice sits near 0');
            ctx.assert(
              trace[0] > start + 0.3,
              `one all-at-once sweep should push the energy UP, from ${start.toFixed(4)} to about ` +
                `+0.58; this run reported ${trace[0].toFixed(4)}. If it went down, the sweep is not ` +
                `updating every cell`
            );
            ctx.assert(
              trace[RACE_SWEEPS - 1] > 1,
              `after 30 all-at-once sweeps at T = 1.5 the energy should be above +1 — worse than a ` +
                `random lattice, which is 0 — but this run ended at ${trace[RACE_SWEEPS - 1].toFixed(4)}`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const s = makeLattice(ctx.utils, 4711);
            const out = await ctx.kernels[1](s);
            const ref = bondEnergyRef(s);
            const hint = wrapHint(out) || diagnoseGrid(out, ref, 1e-4, bondEnergyAlternatives(s));
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
            // Drive the prewired race from a different lattice with the
            // learner's own energy kernel: the pathology has to be a property of
            // the algorithm, not of one starting configuration. And every spin
            // flipping every sweep is the signature — measured: 16,384 of 16,384.
            let s = makeLattice(ctx.utils, 90210);
            const before = meanOf(copyGrid(await ctx.kernels[1](s)));
            for (let k = 0; k < 20; k++) s = await ctx.kernels[0](s, RACE_T, k);
            const after = meanOf(copyGrid(await ctx.kernels[1](s)));
            ctx.assert(after > before + 1,
              `20 all-at-once sweeps should drive the energy from about 0 up past +1 ` +
                `(${before.toFixed(4)} → ${after.toFixed(4)})`);
            const next = await ctx.kernels[0](s, RACE_T, 20);
            let flipped = 0;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) if (next[y][x] !== s[y][x]) flipped++;
            }
            ctx.assert(flipped > CELLS * 0.95,
              `by now every spin should be flipping every sweep — the lattice has locked into a ` +
                `period-2 blinker — but only ${flipped} of 16,384 moved`);
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'half-sweep',
      title: 'Colour the Lattice',
      intro: `<p>The cure is a chessboard, and if you have been through <strong>Iterative Linear
        Solvers</strong> you have already met it: red-black Gauss-Seidel colours a grid in exactly this
        way, for exactly this reason. One idea in two costumes. There it rescues a solver that is
        sequential by construction; here it repairs a Monte Carlo update that is silently wrong when
        it is parallel. Both times the argument is one sentence — <em>the stencil reads only the four
        direct neighbours, and on a chessboard every direct neighbour of a red square is black</em>.</p>
        <p>So call a cell <strong>red</strong> when <code>(x + y)</code> is even and
        <strong>black</strong> when it is odd, and update all 8,192 reds at once. No red cell reads
        another red cell, so nothing a red thread looked at can move while it is deciding: the
        <code>ΔE</code> it computed is exact, not an estimate, and the flip it accepts is the flip
        Metropolis meant. Then do the blacks, reading the reds that were just written. Two data-parallel
        half-sweeps, and the race is gone — not mitigated, gone.</p>
        <p>Black cells are not "skipped" in the red half. Every thread still writes its own cell;
        a black thread writes back the value it already had, ready for the half-sweep that is about
        to need it exactly as it is.</p>`,
      goal: `<strong>Goal:</strong> write the red half-sweep — cells with
        <code>(x + y) % 2 === 0</code> take the Metropolis decision, every other cell comes through
        untouched.`,
      requirements: [
        'Keep the parity in a <em>number</em>: <code>const parity = (x + y) % 2;</code> — a boolean in a kernel variable does not compile on WebGL',
        'Cells with parity <code>1</code> (black) return their spin unchanged',
        'Red cells with <code>dE ≤ 0</code> flip unconditionally',
        'Red cells with <code>dE &gt; 0</code> flip when <code>u &lt; Math.exp(-dE / temperature)</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the trap this task is built around',
          body: `<p>The natural spelling is a boolean, and it is the one thing gpu.js cannot do:</p>
<pre><code>const isRed = (x + y) % 2 === 0;   // throws on WebGL</code></pre>
<p>The GL backend has no way to store a <code>bool</code> in a kernel variable, so it fails at
            shader-compile time with <em>cannot convert from 'bool' to 'lowp float'</em> — while the CPU
            backend runs it happily, which is how this reaches production. Keep the number:</p>
<pre><code>const parity = (x + y) % 2;        // 0 or 1
if (parity !== 0) return spin;</code></pre>`,
        },
        {
          title: 'Hint 2 — the Metropolis decision',
          body: `<p>Two exits, in this order:</p>
<pre><code>if (dE &lt;= 0) return -spin;
if (u &lt; Math.exp(-dE / temperature)) return -spin;
return spin;</code></pre>
<p>The first line is technically implied by the second — <code>u</code> is always below 1 and
            <code>exp(−dE/T)</code> is at least 1 whenever <code>dE ≤ 0</code>, so the test would pass
            anyway. Write it out regardless: it is the rule, and it says that a flip which lowers the
            energy is never a gamble.</p>`,
        },
        {
          title: 'Hint 3 — the shape of the body',
          body: `<p>The parity guard comes first, so a black thread never computes a neighbour sum it is
            not going to use:</p>
<pre><code>const spin = s[y][x];
const parity = (x + y) % 2;
if (parity !== 0) return spin;
const sum = s[y][(x + 1) % n] + s[y][(x + n - 1) % n]
          + s[(y + 1) % n][x] + s[(y + n - 1) % n][x];
const dE = 2 * spin * sum;</code></pre>
<p>Then the hash (already written for you) and the two exits above.</p>`,
        },
      ],
      transfer: `Red-black ordering, and its multi-colour generalisation, is the standard way to put
        a Gauss-Seidel smoother, a lattice Monte Carlo or a physics solver's constraint pass on a GPU:
        one dispatch per colour, and an unstructured mesh gets its colours from a graph-colouring pass
        first. The idea is bigger than any of them — a colour is simply a set of updates guaranteed
        not to depend on each other, which is the same permission slip a wavefront's anti-diagonal or
        a task graph's level hands out.`,
      starterCode: `// Half the lattice at a time. Reds first — and no red reads a red.
const gpu = new GPU({ mode });

const red = gpu.createKernel(function (s, temperature, seed) {
  const x = this.thread.x;
  const y = this.thread.y;
  const n = this.constants.size;
  const spin = s[y][x];
  // TODO 1: the parity, as a NUMBER — const parity = (x + y) % 2;
  //         (a boolean in a kernel variable will not compile on WebGL)
  // TODO 2: parity 1 is black — return spin unchanged.
  // TODO 3: sum the four wrapped neighbours and form dE = 2 * spin * sum.

${HASH_BODY}

  // TODO 4: dE <= 0 flips unconditionally; otherwise flip when u < exp(-dE / temperature).
  return spin;
}, { output: [128, 128], constants: { size: 128 } });

const after = await red(lattice, 1.5, 0);

let movedRed = 0;
let movedBlack = 0;
for (let y = 0; y < 128; y++) {
  for (let x = 0; x < 128; x++) {
    if (after[y][x] === lattice[y][x]) continue;
    if ((x + y) % 2 === 0) movedRed++;
    else movedBlack++;
  }
}
console.log('red cells that flipped:', movedRed, 'of 8192');
console.log('black cells that flipped:', movedBlack, '(must be 0)');
`,
      solutionCode: `// Half the lattice at a time. Reds first — and no red reads a red.
const gpu = new GPU({ mode });

${RED_KERNEL}

const after = await red(lattice, 1.5, 0);

let movedRed = 0;
let movedBlack = 0;
for (let y = 0; y < 128; y++) {
  for (let x = 0; x < 128; x++) {
    if (after[y][x] === lattice[y][x]) continue;
    if ((x + y) % 2 === 0) movedRed++;
    else movedBlack++;
  }
}
console.log('red cells that flipped:', movedRed, 'of 8192');
console.log('black cells that flipped:', movedBlack, '(must be 0)');
`,
      inputs: utils => ({ lattice: makeLattice(utils, 20260801) }),
      publicTests: [
        {
          name: 'the half-sweep returns a 128×128 lattice of <code>±1</code>',
          run: async ctx => {
            assertRunOk(ctx);
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(makeLattice(ctx.utils, 20260801), RACE_T, 0);
            ctx.assert(out && out.length === SIZE && out[0].length === SIZE,
              'expected a 128×128 result');
            for (let y = 0; y < SIZE; y += 7) {
              for (let x = 0; x < SIZE; x += 5) {
                ctx.assert(Math.abs(Math.abs(out[y][x]) - 1) < 1e-6,
                  `cell [${y}][${x}] holds ${out[y][x]} — a spin is +1 or −1, so return spin or -spin, never a probability or a cost`);
              }
            }
          },
        },
        {
          name: 'only red cells moved — every black cell is exactly as it was',
          run: async ctx => {
            // Counting is a different question from comparing values: it tells a
            // learner who coloured the board wrong that the SHAPE is wrong, even
            // where a stray cell lands on the right number by luck.
            const s = makeLattice(ctx.utils, 20260801);
            const out = await ctx.kernel(s, RACE_T, 0);
            let movedRed = 0;
            let movedBlack = 0;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                if (Math.abs(out[y][x] - s[y][x]) < 1e-6) continue;
                if ((x + y) % 2 === 0) movedRed++;
                else movedBlack++;
              }
            }
            ctx.assert(movedBlack === 0,
              `${movedBlack} black cells changed — a half-sweep must leave every cell with ` +
                `(x + y) % 2 === 1 exactly as it found it, or the reds and blacks are deciding from ` +
                `each other's half-written values all over again`);
            // Measured with the reference kernel: 5,730 of the 8,192 reds move.
            ctx.assert(movedRed > 4000 && movedRed < 7000,
              `${movedRed} of the 8,192 red cells flipped; the reference kernel flips about 5,700 at ` +
                `T = 1.5 on this lattice`);
          },
        },
        {
          name: 'every downhill red flip was taken — those are not a gamble',
          run: async ctx => {
            const s = makeLattice(ctx.utils, 20260801);
            const out = await ctx.kernel(s, RACE_T, 0);
            let missed = 0;
            let first = null;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                if ((x + y) % 2 !== 0) continue;
                const dE = 2 * s[y][x] * neighbourSum(s, y, x);
                if (dE > 0) continue;
                if (Math.abs(out[y][x] + s[y][x]) > 1e-6) {
                  missed++;
                  if (!first) first = [y, x, dE];
                }
              }
            }
            ctx.assert(missed === 0, missed
              ? `${missed} red cells whose flip would not raise the energy stayed put (for example ` +
                `[${first[0]}][${first[1]}], where dE = ${first[2]}). A flip with dE ≤ 0 is accepted ` +
                `unconditionally — the random number only decides the uphill ones`
              : 'every downhill red flip was taken');
          },
        },
        {
          name: 'the half-sweep matches the Metropolis rule, cell for cell',
          run: async ctx => {
            const s = makeLattice(ctx.utils, 20260801);
            const out = await ctx.kernel(s, RACE_T, 0);
            const ref = halfSweepRef(s, 0, RACE_T, 0);
            const hint = diagnoseGrid(out, ref, 1e-6, halfSweepAlternatives(s, RACE_T, 0));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-6,
                  hint || `cell [${y}][${x}] (${(x + y) % 2 === 0 ? 'red' : 'black'})`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different lattice, a different temperature and a different seed:
            // a kernel that hard-coded any of the three fails here and nowhere
            // else.
            const s = makeLattice(ctx.utils, 8675309);
            const out = await ctx.kernel(s, 2.5, 17);
            const ref = halfSweepRef(s, 0, 2.5, 17);
            const hint = diagnoseGrid(out, ref, 1e-6, halfSweepAlternatives(s, 2.5, 17));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-6, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Temperature has to be doing real work. At T = 0.125 the Boltzmann
            // factor for the cheapest uphill move is exp(-32) = 1.27e-14, and the
            // smallest u this hash produces anywhere in seeds 0…399 is
            // 2/65536 = 3.1e-5, so NO uphill flip may be accepted — the margin is
            // nine orders of magnitude, not a coincidence of one seed. At T = 40
            // almost every one should be.
            const s = makeLattice(ctx.utils, 20260801);
            const cold = await ctx.kernel(s, 0.125, 3);
            let coldUphill = 0;
            let uphill = 0;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                if ((x + y) % 2 !== 0) continue;
                const dE = 2 * s[y][x] * neighbourSum(s, y, x);
                if (dE <= 0) continue;
                uphill++;
                if (Math.abs(cold[y][x] + s[y][x]) < 1e-6) coldUphill++;
              }
            }
            ctx.assert(coldUphill === 0,
              `${coldUphill} of the ${uphill} uphill red flips were accepted at T = 0.125, where ` +
                `exp(−4/T) is 1.3e-14 and the smallest number this hash can draw is 3.1e-5. Something ` +
                `upstream of the comparison is wrong: the sign inside the exponential, the direction of ` +
                `the <, the temperature not reaching it at all — or a dE built from the wrong ` +
                `neighbours, since a cell reading off the edge of the lattice gets a different dE from ` +
                `the one this test computed`);
            const hot = await ctx.kernel(s, 40, 3);
            let hotUphill = 0;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                if ((x + y) % 2 !== 0) continue;
                const dE = 2 * s[y][x] * neighbourSum(s, y, x);
                if (dE > 0 && Math.abs(hot[y][x] + s[y][x]) < 1e-6) hotUphill++;
              }
            }
            ctx.assert(hotUphill > uphill * 0.8,
              `at T = 40 nearly every uphill flip should be accepted (exp(−8/40) = 0.82), but only ` +
                `${hotUphill} of ${uphill} were`);
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'full-sweep',
      title: 'Two Halves Make a Sweep',
      intro: `<p>The red half is not a sweep — half the lattice has not been offered a move. The black
        half is the same kernel with its parity test flipped, and the ordering that matters is in the
        <strong>chaining</strong>: <code>black(red(s))</code>. The black cells read the lattice the red
        half produced, so their <code>ΔE</code> accounts for the reds that just moved. That is not an
        optimisation, it is the correctness argument: a black cell's four neighbours are all red, and
        the reds have finished.</p>
        <p>Give the two halves different seeds too — <code>2k</code> and <code>2k + 1</code> — or every
        black cell would draw the same number its red neighbours just used.</p>
        <p>Then run it. Same starting lattice as task 3, same temperature, same thirty sweeps, and
        the energy that climbed from <code>−0.01</code> to <code>+1.30</code> now falls to about
        <code>−1.84</code>, sweep after sweep — 26 of the 29 steps go down, and the three that do not
        tick back up by less than <code>0.004</code>. That wobble is the temperature doing its job:
        <code>T = 1.5</code> is cold, not zero, so a handful of uphill moves are accepted every sweep
        and the energy is allowed to breathe. Nothing about the physics changed and nothing got
        slower: the same 16,384 spins are offered the same moves. All that changed is <em>which of
        them are allowed to move at the same time</em>.</p>`,
      goal: `<strong>Goal:</strong> write the black half-sweep and chain the two halves into one full
        sweep, <code>black(red(s))</code>.`,
      requirements: [
        'The black kernel is the red kernel with its parity test flipped — parity <code>1</code> moves, everything else passes through',
        'Create the red kernel first and the black kernel second',
        'One full sweep is <code>await black(await red(s, T, 2k), T, 2k + 1)</code> — the black half reads what the red half wrote',
      ],
      hints: [
        {
          title: 'Hint 1 — the black kernel',
          body: `<p>Copy the red kernel and change one digit:</p>
<pre><code>const parity = (x + y) % 2;
if (parity !== 1) return spin;</code></pre>
<p>Still a number, never a boolean — the WebGL backend rejects <code>const isBlack = …</code>
            exactly as it rejects <code>isRed</code>.</p>`,
        },
        {
          title: 'Hint 2 — the chain is the lesson',
          body: `<pre><code>const afterRed = await red(s, temperature, k * 2);
s = await black(afterRed, temperature, k * 2 + 1);   // NOT black(s, …)</code></pre>
<p>Or in one line, <code>await black(await red(s, T, 2 * k), T, 2 * k + 1)</code>. The inner
            <code>await</code> is not decoration: an un-awaited kernel call hands the black half a
            Promise instead of a lattice.</p>`,
        },
      ],
      transfer: `Two dispatches with a dependency between them is the ordinary shape of GPU work —
        WebGPU puts a barrier between compute passes, CUDA orders them on a stream, Vulkan wants an
        explicit pipeline barrier. What no platform will do is order threads <em>inside</em> one
        dispatch, which is exactly why the update had to be split in two. Production lattice Monte
        Carlo goes one step further and launches only the cells of one colour, so the half that would
        copy itself is never scheduled at all; the trade is a strided memory access pattern against
        half the threads, and which wins is a benchmark, not an argument.`,
      starterCode: `// One full sweep = the reds, then the blacks reading what the reds just wrote.
const gpu = new GPU({ mode });

${RED_KERNEL}

const black = gpu.createKernel(function (s, temperature, seed) {
  const x = this.thread.x;
  const y = this.thread.y;
  const n = this.constants.size;
  const spin = s[y][x];
  // TODO 1: same body as the red kernel, with the parity test flipped —
  //         the cells where (x + y) % 2 is 1 take the Metropolis decision.
  return spin;
}, { output: [128, 128], constants: { size: 128 } });

${BOND_ENERGY_KERNEL}

${MEAN_HELPER}

let s = lattice;
const trace = [];
for (let k = 0; k < 30; k++) {
  const afterRed = await red(s, 1.5, k * 2);
  // TODO 2: finish the sweep. The black half must read afterRed, not s.
  s = afterRed;
  trace.push(meanOf(await bondEnergy(s)));
}

console.log('energy per spin, sweep by sweep:', trace);
console.log('after 30 red-black sweeps: E =', trace[29]);
plot({ 'energy per spin': trace }, { title: 'checkerboard Metropolis at T = 1.5' });
`,
      solutionCode: `// One full sweep = the reds, then the blacks reading what the reds just wrote.
const gpu = new GPU({ mode });

${RED_KERNEL}

${BLACK_KERNEL}

${BOND_ENERGY_KERNEL}

${MEAN_HELPER}

let s = lattice;
const trace = [];
for (let k = 0; k < 30; k++) {
  const afterRed = await red(s, 1.5, k * 2);
  s = await black(afterRed, 1.5, k * 2 + 1);
  trace.push(meanOf(await bondEnergy(s)));
}

console.log('energy per spin, sweep by sweep:', trace);
console.log('after 30 red-black sweeps: E =', trace[29]);
plot({ 'energy per spin': trace }, { title: 'checkerboard Metropolis at T = 1.5' });
`,
      inputs: utils => ({ lattice: makeLattice(utils, 20260801) }),
      publicTests: [
        // FIRST, and it has to stay first: it reads ctx.kernels[1].lastArgs, and
        // every later test invokes the kernels itself, which overwrites them.
        {
          name: 'your code chained them — the black half was handed the red half’s output',
          run: async ctx => {
            // Both kernels can be right and the program still wrong: what this
            // checks is the ARGUMENT the black half was actually called with,
            // which is the entire difference between a correct sweep and the
            // race from task 3 wearing a chessboard.
            assertRunOk(ctx);
            ctx.assert(ctx.kernels.length >= 2,
              `expected at least 2 kernels (red, then black), found ${ctx.kernels.length}`);
            const redArgs = ctx.kernels[0].lastArgs;
            const blackArgs = ctx.kernels[1].lastArgs;
            ctx.assert(redArgs && redArgs[0], 'the red kernel was never called');
            ctx.assert(blackArgs && blackArgs[0],
              'the black kernel was never called — one full sweep is black(red(s))');
            const seen = blackArgs[0];
            ctx.assert(!ctx.utils.isPromiseLike(seen),
              'the black half was handed a Promise, not a lattice — the inner kernel call needs its own await');
            // Compared against the RED CALL'S OWN arguments, not against a
            // reference trajectory: a program that is already wrong has drifted
            // away from the reference by the last sweep, and this test has to
            // stay able to say WHY rather than reporting a lattice that matches
            // nothing. What red was handed and what black was handed are both
            // recorded, so the relation between them is checkable on its own.
            const redIn = copyGrid(redArgs[0]);
            const afterRed = halfSweepRef(redIn, 0, redArgs[1], redArgs[2]);
            let matchesRed = 0;
            let matchesRedInput = 0;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                if (Math.abs(seen[y][x] - afterRed[y][x]) < 1e-6) matchesRed++;
                if (Math.abs(seen[y][x] - redIn[y][x]) < 1e-6) matchesRedInput++;
              }
            }
            ctx.assert(
              matchesRed > CELLS - 8,
              matchesRedInput > CELLS - 8
                ? 'the black half was handed the same lattice the red half was — chain them, ' +
                  'black(red(s)), or the reds\' moves are thrown away and the blacks decide against a ' +
                  'lattice that no longer exists'
                : `the lattice handed to the black kernel is neither the red half's output nor the ` +
                  `lattice red itself was given (${matchesRed} of 16,384 cells agreed with the red ` +
                  `half's output) — the two halves have to be chained within one sweep`
            );
          },
        },
        {
          name: 'the first kernel is still the red half',
          run: async ctx => {
            const s = makeLattice(ctx.utils, 20260801);
            const out = await ctx.kernels[0](s, RACE_T, 0);
            const ref = halfSweepRef(s, 0, RACE_T, 0);
            const hint = diagnoseGrid(out, ref, 1e-6, halfSweepAlternatives(s, RACE_T, 0));
            for (let y = 0; y < SIZE; y += 3) {
              for (let x = 0; x < SIZE; x += 3) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-6,
                  hint || `the FIRST kernel should still be the red half — cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'on its own, the second kernel is a <em>black</em> half-sweep',
          run: async ctx => {
            // Checked in isolation, because inside the chain it is hard to see:
            // a black kernel with no parity test still produces a plausible
            // lattice, it just re-decides the reds a second time with a
            // different seed.
            const s = makeLattice(ctx.utils, 20260801);
            const out = await ctx.kernels[1](s, RACE_T, 1);
            const ref = halfSweepRef(s, 1, RACE_T, 1);
            const hint = diagnoseGrid(out, ref, 1e-6, [
              [naiveSweepRef(s, RACE_T, 1),
                'the black kernel has no parity test — it offers every cell a move, which is task 3\'s race again. Only the cells where (x + y) % 2 is 1 belong to this half'],
              [halfSweepRef(s, 0, RACE_T, 1),
                'the black kernel is updating the RED cells — flip its parity test to keep the cells where (x + y) % 2 is 1'],
              [copyGrid(s),
                'the black kernel changed nothing — cells where (x + y) % 2 is 1 should take the Metropolis decision'],
            ]);
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-6, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'chained, the two halves are one correct sweep',
          run: async ctx => {
            const s = makeLattice(ctx.utils, 20260801);
            const out = await ctx.kernels[1](await ctx.kernels[0](s, RACE_T, 0), RACE_T, 1);
            const ref = fullSweepRef(s, RACE_T, 0);
            const hint = diagnoseGrid(out, ref, 1e-6, fullSweepAlternatives(s, RACE_T, 0));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-6, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'the energy falls — the same run that climbed in task 3',
          run: async ctx => {
            const trace = plottedSeries(ctx, 'energy per spin');
            ctx.assert(Array.isArray(trace) && trace.length === RACE_SWEEPS,
              `expected a plotted 'energy per spin' series of ${RACE_SWEEPS} points, got ` +
                `${trace ? trace.length : 'none'}`);
            const ref = traceRef(makeLattice(ctx.utils, 20260801), RACE_SWEEPS, RACE_T, fullSweepRef);
            const naive = traceRef(makeLattice(ctx.utils, 20260801), RACE_SWEEPS, RACE_T, naiveSweepRef);
            for (let k = 0; k < RACE_SWEEPS; k++) {
              const hint = diagnose(trace[k], ref[k], 5e-3, [
                [naive[k], 'that is the all-at-once trace from task 3 — neither kernel is testing the parity, so both halves are updating the whole lattice'],
              ]);
              ctx.assertClose(trace[k], ref[k], 5e-3, hint || `the energy plotted after sweep ${k + 1}`);
            }
            // Measured: -0.990 after one sweep, -1.837 after thirty. Not
            // monotone — sweeps 10, 22 and 30 tick up by 0.0007, 0.0027 and
            // 0.0034, which is what an accepted uphill move looks like at
            // T = 1.5. Hence the +0.02 allowance below: six times the largest
            // rise the reference produces, and still nowhere near the 0.3 an
            // uncoloured sweep climbs by on its first step.
            ctx.assert(trace[RACE_SWEEPS - 1] < -1.7,
              `after 30 coloured sweeps at T = 1.5 the energy should be near −1.84, not ` +
                `${trace[RACE_SWEEPS - 1].toFixed(4)}`);
            for (let k = 1; k < RACE_SWEEPS; k++) {
              ctx.assert(trace[k] <= trace[k - 1] + 0.02,
                `the energy rose between sweep ${k} and sweep ${k + 1} ` +
                  `(${trace[k - 1].toFixed(4)} → ${trace[k].toFixed(4)}) — a coloured sweep at this ` +
                  `temperature walks downhill essentially every time`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Twenty sweeps of the learner's own two kernels against the
            // reference, cell for cell: a run that plotted the right curve with
            // the wrong kernels cannot survive this.
            let s = makeLattice(ctx.utils, 4711);
            let ref = makeLattice(ctx.utils, 4711);
            for (let k = 0; k < 20; k++) {
              s = await ctx.kernels[1](await ctx.kernels[0](s, 2.25, k * 2), 2.25, k * 2 + 1);
              ref = fullSweepRef(ref, 2.25, k);
            }
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(s[y][x], ref[y][x], 1e-6, `cell [${y}][${x}] after 20 sweeps at T = 2.25`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The physics, driven by the test: from the coldest possible start
            // at T = 1.5 the lattice must STAY ordered. A coloured sweep holds it
            // at |m| = 0.987 after forty; the all-at-once update from the same
            // all-up start has chewed it down to |m| = 0.738 by sweep thirty and
            // keeps going, so 0.95 separates the two comfortably.
            let s = makeAligned();
            for (let k = 0; k < 40; k++) {
              s = await ctx.kernels[1](await ctx.kernels[0](s, RACE_T, k * 2), RACE_T, k * 2 + 1);
            }
            const m = Math.abs(meanOf(copyGrid(s)));
            ctx.assert(m > 0.95,
              `40 coloured sweeps at T = 1.5 should leave an all-up lattice still ordered ` +
                `(|m| ≈ 0.986), but |m| came out ${m.toFixed(4)} — well below Tc the lattice does not ` +
                `melt, and if it did the halves are still fighting each other`);
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'phase-transition',
      title: 'Drag the Temperature Across Tc',
      intro: `<p>Everything is wired. The payoff is one number you can move with your finger.</p>
        <p>Start from the coldest configuration there is — every spin up — and run 150 coloured
        sweeps at a temperature you choose. Below the <strong>critical temperature</strong> the lattice
        keeps its order: thermal noise chews holes in it, but the holes heal and the magnetisation
        <code>m</code>, the average spin, sits stubbornly near <code>±1</code>. Above it the order
        does not survive at all — the lattice dissolves into salt and pepper and <code>m</code> falls
        to zero. In between there is no gentle slope; the whole thing turns over inside a few tenths
        of a degree. Onsager solved this exactly in 1944 and the answer is
        <code>Tc = 2 / ln(1 + √2) ≈ 2.269</code>, in units of <code>J / k_B</code>.</p>
        <p>Two honest caveats, because a lattice of 16,384 spins is not infinite and 150 sweeps is not
        forever. The crossover you can see sits a little above <code>2.269</code> — finite lattices
        round a transition off, and a cold start clings to its order — and right at <code>Tc</code>
        the model slows to a crawl, which is not a bug in the simulation but the defining symptom of a
        critical point. Drag slowly through <code>2.3</code>–<code>2.5</code> and watch the domains
        grow to the size of the whole box just before they let go.</p>`,
      goal: `<strong>Goal:</strong> declare the temperature slider, paint the lattice, and render a
        frame every ten sweeps so the run becomes something you can scrub through.`,
      requirements: [
        'Declare the control: <code>slider(\'temperature\', { min: 1.5, max: 3.5, value: 2.25, step: 0.05 })</code>',
        'Paint up spins <code>this.color(0.97, 0.58, 0.26, 1)</code> and down spins <code>this.color(0.09, 0.14, 0.28, 1)</code>',
        'Every tenth sweep, <code>await paint(s)</code> and then <code>render(paint.canvas)</code>',
        'Leave the prewired sweep loop and <code>plot()</code> call alone',
      ],
      hints: [
        {
          title: 'Hint 1 — the slider is the program',
          body: `<p><code>slider()</code> returns the value this run is using and puts the control under
            the console; moving it re-runs the whole program from the top. That is the entire model —
            your code is a pure function of its controls, so there is no event loop to write.</p>
<pre><code>const temperature = slider('temperature',
  { min: 1.5, max: 3.5, value: 2.25, step: 0.05 });</code></pre>`,
        },
        {
          title: 'Hint 2 — three renders make a scrubber',
          body: `<p>Consecutive <code>render()</code> calls collapse into a frame strip with a slider
            under it, so rendering inside the loop costs you nothing and gives you the whole history:</p>
<pre><code>if (k % 10 === 0) {
  await paint(s);
  render(paint.canvas);
}</code></pre>
<p>Keep the two lines adjacent and do not <code>console.log</code> between them — a log line in
            the middle breaks the run of frames into separate images.</p>`,
        },
      ],
      transfer: `A control that re-runs the whole computation is the interaction model of every GPU
        toy worth playing with, and it works for the same reason here as in a shader: the frame is
        cheap enough that recomputing it beats maintaining incremental state. The physics travels
        further than the code. Order parameters, critical exponents and finite-size scaling are the
        vocabulary of everything from magnets to percolation thresholds to the training dynamics of
        large models, and the Ising lattice is where all of it was worked out first.`,
      starterCode: `// The whole model, with a dial on it. Drag the temperature past 2.269.
const gpu = new GPU({ mode });

// TODO 1: declare the control. slider() returns the value THIS run is using.
const temperature = 2.25;

${RED_KERNEL}

${BLACK_KERNEL}

const paint = gpu.createKernel(function (s) {
  const spin = s[this.thread.y][this.thread.x];
  // TODO 2: up spins this.color(0.97, 0.58, 0.26, 1),
  //         down spins this.color(0.09, 0.14, 0.28, 1).
  this.color(1, 0, 1, 1);
}, { output: [128, 128], graphical: true });

${MEAN_HELPER}

// The coldest possible start: every spin up. 150 sweeps at your temperature.
let s = alignedLattice;
const trace = [];
for (let k = 0; k < 150; k++) {
  s = await black(await red(s, temperature, k * 2), temperature, k * 2 + 1);
  trace.push(meanOf(s));
  // TODO 3: every tenth sweep, await paint(s) and then render(paint.canvas).
}

console.log('temperature', temperature, '— Tc is 2.269');
console.log('magnetisation after 150 sweeps:', trace[149]);
plot({ magnetisation: trace }, { title: 'magnetisation per sweep' });
`,
      solutionCode: `// The whole model, with a dial on it. Drag the temperature past 2.269.
const gpu = new GPU({ mode });

const temperature = slider('temperature', { min: 1.5, max: 3.5, value: 2.25, step: 0.05 });

${RED_KERNEL}

${BLACK_KERNEL}

${PAINT_KERNEL}

${MEAN_HELPER}

// The coldest possible start: every spin up. 150 sweeps at your temperature.
let s = alignedLattice;
const trace = [];
for (let k = 0; k < 150; k++) {
  s = await black(await red(s, temperature, k * 2), temperature, k * 2 + 1);
  trace.push(meanOf(s));
  if (k % 10 === 0) {
    await paint(s);
    render(paint.canvas);
  }
}

console.log('temperature', temperature, '— Tc is 2.269');
console.log('magnetisation after 150 sweeps:', trace[149]);
plot({ magnetisation: trace }, { title: 'magnetisation per sweep' });
`,
      inputs: () => ({ alignedLattice: makeAligned() }),
      publicTests: [
        {
          name: 'a <code>temperature</code> slider is declared, spanning Tc',
          run: async ctx => {
            assertRunOk(ctx);
            const controls = ctx.controls || [];
            const slider = controls.find(c => c.name === 'temperature');
            ctx.assert(
              slider,
              controls.length
                ? `the run declared ${controls.length} control(s) but none called 'temperature': ` +
                  `${controls.map(c => c.name).join(', ')}`
                : 'no slider was declared — call slider(\'temperature\', { min: 1.5, max: 3.5, value: 2.25, step: 0.05 })'
            );
            ctx.assert(slider.min <= 2 && slider.max >= 2.6,
              `the slider runs ${slider.min}–${slider.max}, which does not straddle Tc = 2.269 with ` +
                `room on both sides; 1.5 to 3.5 does`);
            // Verification runs with defaults, so the default is what every
            // assertion below is measured against.
            ctx.assertClose(slider.value, PAYOFF_T, 1e-9,
              `the slider's default value should be ${PAYOFF_T} — that is the temperature the rest of ` +
                `these tests are measured at`);
          },
        },
        {
          name: 'the run rendered a strip of frames, not a single picture',
          run: async ctx => {
            const frames = renderedFrames(ctx);
            ctx.assert(
              frames >= 3,
              frames === 0
                ? 'nothing was rendered — call render(paint.canvas) inside the loop'
                : `only ${frames} frame(s) were rendered; the console collapses three or more ` +
                  `consecutive render() calls into a scrubber, and 150 sweeps at one frame per ten ` +
                  `gives 15`
            );
            const wanted = Math.ceil(PAYOFF_SWEEPS / FRAME_EVERY);
            ctx.assert(frames >= wanted - 5 && frames <= wanted + 15,
              `${frames} frames were rendered; rendering every ${FRAME_EVERY}th of ${PAYOFF_SWEEPS} ` +
                `sweeps gives ${wanted}`);
            ctx.assert(ctx.canvas && ctx.canvas.width === SIZE && ctx.canvas.height === SIZE,
              `expected a 128×128 canvas, got ${ctx.canvas ? `${ctx.canvas.width}×${ctx.canvas.height}` : 'none'}`);
          },
        },
        {
          name: 'the painting is two-toned, and the two tones are the two spins',
          run: async ctx => {
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(graphical, 'no graphical paint kernel found');
            // A lattice with a known, lopsided mix, so "everything one colour"
            // cannot pass: three quarters up.
            const s = makeAligned();
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) if ((x + 2 * y) % 4 === 0) s[y][x] = -1;
            }
            await graphical(s);
            const pixels = graphical.getPixels();
            let warm = 0;
            let cool = 0;
            for (let i = 0; i < pixels.length; i += 4) {
              const r = pixels[i];
              if (r > 200) warm++;
              else if (r < 60) cool++;
            }
            ctx.assert(warm + cool === pixels.length / 4,
              `${pixels.length / 4 - warm - cool} pixels are neither of the two colours — an up spin is ` +
                `this.color(0.97, 0.58, 0.26, 1) and a down spin this.color(0.09, 0.14, 0.28, 1)`);
            ctx.assert(Math.abs(cool - CELLS / 4) < 200 && Math.abs(warm - (3 * CELLS) / 4) < 200,
              `this lattice is three-quarters up, so about ${(3 * CELLS) / 4} pixels should be warm and ` +
                `${CELLS / 4} cool; got ${warm} and ${cool}. Are the two colours the right way round?`);
          },
        },
        {
          name: 'at the default 2.25 the lattice stays ordered — <code>|m|</code> holds above 0.7',
          run: async ctx => {
            const trace = plottedSeries(ctx, 'magnetisation');
            ctx.assert(Array.isArray(trace) && trace.length === PAYOFF_SWEEPS,
              `expected a plotted 'magnetisation' series of ${PAYOFF_SWEEPS} points, got ` +
                `${trace ? trace.length : 'none'} — leave the prewired plot() call in place`);
            const ref = magTraceRef(PAYOFF_T, PAYOFF_SWEEPS);
            for (const k of [0, 9, 49, 99, PAYOFF_SWEEPS - 1]) {
              ctx.assertClose(trace[k], ref[k], 5e-3,
                diagnose(trace[k], ref[k], 5e-3, [
                  [ref[k] * CELLS, 'that is the total spin, not the mean — meanOf divides by 128 × 128'],
                ]) || `the magnetisation plotted after sweep ${k + 1}`);
            }
            // Measured over the last 50 sweeps at T = 2.25: |m| never drops
            // below 0.80 and never rises above 0.84.
            const tail = trace.slice(100).map(Math.abs);
            ctx.assert(Math.min(...tail) > 0.7,
              `|m| dipped to ${Math.min(...tail).toFixed(3)} over the last 50 sweeps; at T = 2.25, ` +
                `below Tc, an all-up lattice holds |m| near 0.82`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The phase transition itself, asserted: the SAME kernels, the same
            // cold start, two temperatures either side of Tc. Measured after 60
            // sweeps: |m| = 0.986 at T = 1.5, |m| = 0.033 at T = 3.
            const run = async temperature => {
              let s = makeAligned();
              for (let k = 0; k < 60; k++) {
                s = await ctx.kernels[1](
                  await ctx.kernels[0](s, temperature, k * 2), temperature, k * 2 + 1
                );
              }
              return Math.abs(meanOf(copyGrid(s)));
            };
            const cold = await run(1.5);
            const hot = await run(3);
            ctx.assert(cold > 0.95,
              `at T = 1.5, well below Tc = ${TC.toFixed(3)}, 60 sweeps should leave the lattice ordered ` +
                `at |m| ≈ 0.99 — this run gave ${cold.toFixed(4)}`);
            ctx.assert(hot < 0.15,
              `at T = 3, well above Tc = ${TC.toFixed(3)}, 60 sweeps should melt the lattice to |m| ≈ 0.03 ` +
                `— this run gave ${hot.toFixed(4)}. A lattice that stays magnetised at three times the ` +
                `bond energy is not feeling the temperature at all`);
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The kernels are still the sweep from task 5, exactly: 20 sweeps at
            // the slider's default against the reference, cell for cell.
            let s = makeAligned();
            let ref = makeAligned();
            for (let k = 0; k < 20; k++) {
              s = await ctx.kernels[1](
                await ctx.kernels[0](s, PAYOFF_T, k * 2), PAYOFF_T, k * 2 + 1
              );
              ref = fullSweepRef(ref, PAYOFF_T, k);
            }
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(s[y][x], ref[y][x], 1e-6,
                  `cell [${y}][${x}] after 20 sweeps at the slider's default temperature`);
              }
            }
          },
        },
      ],
    },
  ],
};
