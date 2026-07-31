// Module: Jump Flooding — uuid a741a650-84e8-4362-a344-43a4d7018c7f (short id a741a650).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. No legacyId: this module never had a pre-uuid url.
//
// Jump Flooding: Voronoi in log n Passes.
//
// Six tasks: brute-force nearest seed and what a cell must CARRY → one flood
// pass at one stride → the halving ladder, rendered pass by pass → seed field
// to distance field → signed distance field, by flooding inside and outside a
// bitmap → the honesty task: measure how wrong the approximation actually is.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested numeric arrays as inputs, this.thread.* for indexing,
// statically bounded loops, Math.* per gpu.js's whitelist. Every kernel call is
// awaited — kernels return a Promise on every backend. Every task passes in CPU
// mode; the grid is 128×128 = 16,384 threads and a ladder is 7 passes, so the
// whole module is comfortably inside the run budget without asking for more.
//
// THE ONE ENCODING EVERYTHING RESTS ON. A gpu.js cell holds a single number,
// and a jump-flooding cell has to carry a COORDINATE PAIR — so the pair is
// packed: id = sy * 128 + sx, with -1 meaning "no seed yet". Every id in play
// is a whole number below 16,384, which float32 stores exactly, so packing and
// unpacking are lossless on every backend. All distance comparisons inside the
// kernels are on SQUARED distances, which are whole numbers too — so a tie is
// a tie on the CPU and on the GPU alike, and no test has to care which
// candidate a backend happened to visit first.

const N = 128; // grid side. log2(128) = 7 passes.
const CELLS = N * N;
const STRIDES = [64, 32, 16, 8, 4, 2, 1];

// ---- deterministic inputs (shared by inputs() and tests) -------------------

// `count` distinct seed sites on the N×N grid. Same seed → same sites, always.
function makeSites(utils, count, seed) {
  const rand = utils.seededRandom(seed);
  const xs = [];
  const ys = [];
  const used = {};
  while (xs.length < count) {
    const x = Math.floor(rand() * N);
    const y = Math.floor(rand() * N);
    const key = y * N + x;
    if (used[key]) continue;
    used[key] = true;
    xs.push(x);
    ys.push(y);
  }
  return { xs, ys };
}

// The starting field: the packed id at each seed site, -1 everywhere else.
function seedGridOf(xs, ys) {
  const grid = new Array(N);
  for (let y = 0; y < N; y++) grid[y] = new Array(N).fill(-1);
  for (let i = 0; i < xs.length; i++) grid[ys[i]][xs[i]] = ys[i] * N + xs[i];
  return grid;
}

// A five-pointed star as a 0/1 bitmap — concave, sharp-cornered, and utterly
// beyond writing down as a formula, which is the point of task 5.
function makeStarMask() {
  const cx = 63.5;
  const cy = 63.5;
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const radius = i % 2 === 0 ? 54 : 23;
    const a = (i * Math.PI) / 5;
    pts.push([cx + radius * Math.sin(a), cy - radius * Math.cos(a)]);
  }
  const inside = (x, y) => {
    let hit = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i];
      const [xj, yj] = pts[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  };
  const mask = new Array(N);
  for (let y = 0; y < N; y++) {
    const row = new Array(N);
    for (let x = 0; x < N; x++) row[x] = inside(x, y) ? 1 : 0;
    mask[y] = row;
  }
  return mask;
}

// ---- plain-JS mirrors of the kernels (the reference every test compares to)

// Squared distance from cell (x, y) to the seed a packed id names. An
// unassigned cell is farther than anything on the grid, which is what makes
// "did this pass improve me?" answerable without a special case.
function dist2Of(id, x, y) {
  const packed = Math.round(id);
  if (!(packed >= 0)) return CELLS * 2;
  const sy = Math.floor(packed / N);
  const sx = packed - sy * N;
  return (sx - x) * (sx - x) + (sy - y) * (sy - y);
}

// One flood pass at stride k: nine candidates, keep the nearest seed.
// `variant` lets the probe helpers below build the field a specific mistake
// would produce, from the same code the correct answer comes from.
function floodPassJS(grid, k, variant) {
  const v = variant || {};
  const out = new Array(N);
  for (let y = 0; y < N; y++) {
    const row = new Array(N);
    for (let x = 0; x < N; x++) {
      let best = -1;
      let bestD = CELLS * 2;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (v.skipCentre && dx === 0 && dy === 0) continue;
          const nx = x + dx * k;
          const ny = y + dy * k;
          if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
          const id = grid[ny][nx];
          if (!(id >= 0)) continue;
          let d;
          if (v.fromNeighbour) d = dist2Of(id, nx, ny);
          else if (v.toNeighbour) d = (nx - x) * (nx - x) + (ny - y) * (ny - y);
          else if (v.rawId) d = id;
          else d = dist2Of(id, x, y);
          if (v.lastWins || d < bestD) {
            bestD = d;
            best = id;
          }
        }
      }
      row[x] = best;
    }
    out[y] = row;
  }
  return out;
}

// The whole ladder: k = 64, 32, … 1.
function ladderJS(grid) {
  let g = grid;
  for (const k of STRIDES) g = floodPassJS(g, k);
  return g;
}

// Brute force: the exact answer, O(cells × sites).
function exactGridJS(xs, ys) {
  const out = new Array(N);
  for (let y = 0; y < N; y++) {
    const row = new Array(N);
    for (let x = 0; x < N; x++) {
      let best = -1;
      let bestD = CELLS * 2;
      for (let i = 0; i < xs.length; i++) {
        const d = (xs[i] - x) * (xs[i] - x) + (ys[i] - y) * (ys[i] - y);
        if (d < bestD) {
          bestD = d;
          best = ys[i] * N + xs[i];
        }
      }
      row[x] = best;
    }
    out[y] = row;
  }
  return out;
}

function distanceGridJS(grid) {
  const out = new Array(N);
  for (let y = 0; y < N; y++) {
    const row = new Array(N);
    for (let x = 0; x < N; x++) row[x] = Math.sqrt(dist2Of(grid[y][x], x, y));
    out[y] = row;
  }
  return out;
}

function seedWhereJS(mask, want) {
  const out = new Array(N);
  for (let y = 0; y < N; y++) {
    const row = new Array(N);
    for (let x = 0; x < N; x++) row[x] = mask[y][x] === want ? y * N + x : -1;
    out[y] = row;
  }
  return out;
}

// The signed field: distance to the nearest inside pixel minus distance to the
// nearest outside pixel. Inside, the first term is 0 and the answer is
// negative; outside, the second is 0 and it is positive.
function sdfJS(mask) {
  const dIn = distanceGridJS(ladderJS(seedWhereJS(mask, 1)));
  const dOut = distanceGridJS(ladderJS(seedWhereJS(mask, 0)));
  const out = new Array(N);
  for (let y = 0; y < N; y++) {
    const row = new Array(N);
    for (let x = 0; x < N; x++) row[x] = dIn[y][x] - dOut[y][x];
    out[y] = row;
  }
  return out;
}

function countFilledJS(grid) {
  let n = 0;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) if (grid[y][x] >= 0) n++;
  }
  return n;
}

// Cells whose flooded seed is STRICTLY farther than the true nearest one.
// Equal distance is not an error: two sites can tie, and both answers are
// right. Counting id mismatches instead would report roughly twice as many.
function wrongCells(grid, truth) {
  let n = 0;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (dist2Of(grid[y][x], x, y) > dist2Of(truth[y][x], x, y)) n++;
    }
  }
  return n;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports two numbers tells a learner nothing about WHICH
// slip produced them. A probe pairs the whole field one specific mistake would
// produce with a sentence naming that mistake; the helpers below speak only
// when the observed field matches a probe in EVERY cell and the correct field
// disagrees with that probe somewhere. One matching cell is worthless here —
// half the plausible mistakes agree with the right answer over most of the grid
// — so nothing less than a total match may accuse anyone. Probes that disagree
// with each other cancel, and a field matching nothing gets the plain numeric
// message. A wrong diagnosis is worse than none.

function fieldMatches(got, want, eps) {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const a = got[y] && got[y][x];
      if (typeof a !== 'number' || !(Math.abs(a - want[y][x]) <= eps)) return false;
    }
  }
  return true;
}

// probes: [fieldTheMistakeProduces, message]
function diagnoseField(got, expected, eps, probes) {
  const hits = probes
    .filter(p => fieldMatches(got, p[0], eps) && !fieldMatches(expected, p[0], eps))
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Compare a whole 128×128 field, and build the probe fields ONLY once something
// has actually gone wrong. Every probe here is a full re-run of the mirror, so
// a passing test that built them anyway would pay six ladders for nothing.
// `probesFn` is called at most once, and only on the failing path.
function assertField(ctx, got, expected, eps, label, probesFn) {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const a = got[y] && got[y][x];
      if (typeof a === 'number' && Math.abs(a - expected[y][x]) <= eps) continue;
      const hint = probesFn ? diagnoseField(got, expected, eps, probesFn()) : null;
      ctx.assertClose(a, expected[y][x], eps, hint || `${label} — cell [${y}][${x}]`);
    }
  }
}

// The same, but judging a field of packed ids by the DISTANCE each one names.
// Two seeds can tie, and a backend is free to visit the tied candidates in
// either order, so the ids may legitimately differ; the distance may not.
function assertSameDistances(ctx, got, expected, label, probesFn) {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const a = dist2Of(got[y] && got[y][x], x, y);
      const b = dist2Of(expected[y][x], x, y);
      if (a === b) continue;
      const hint = probesFn ? diagnoseField(got, expected, 1e-3, probesFn()) : null;
      ctx.assertClose(
        a,
        b,
        1e-3,
        hint || `${label} — cell [${y}][${x}] holds a seed at squared distance ${a}, ` +
          `but one at ${b} was reachable`
      );
    }
  }
}

// The single numeric kernel a task expects when there is only one.
function soleNumericKernel(ctx) {
  const found = ctx.kernels.filter(k => k && k.kernel && !k.kernel.graphical);
  return found.length ? found[0] : null;
}

// The same idea for a short sequence of numbers (task 3's pass-by-pass counts).
function seqMatches(got, want) {
  if (!Array.isArray(got) || got.length !== want.length) return false;
  return want.every((v, i) => Math.abs(got[i] - v) <= 0.5);
}

function diagnoseSeq(got, expected, probes) {
  const hits = probes
    .filter(p => seqMatches(got, p[0]) && !seqMatches(expected, p[0]))
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// ---- field builders for the probes ----------------------------------------

// Task 1: the three fields you get instead of "where the nearest site is".
function nearestProbes(xs, ys) {
  const index = new Array(N);
  const dist = new Array(N);
  const dist2 = new Array(N);
  const swapped = new Array(N);
  for (let y = 0; y < N; y++) {
    index[y] = new Array(N);
    dist[y] = new Array(N);
    dist2[y] = new Array(N);
    swapped[y] = new Array(N);
    for (let x = 0; x < N; x++) {
      let bi = 0;
      let bd = CELLS * 2;
      for (let i = 0; i < xs.length; i++) {
        const d = (xs[i] - x) * (xs[i] - x) + (ys[i] - y) * (ys[i] - y);
        if (d < bd) {
          bd = d;
          bi = i;
        }
      }
      index[y][x] = bi;
      dist[y][x] = Math.sqrt(bd);
      dist2[y][x] = bd;
      swapped[y][x] = xs[bi] * N + ys[bi];
    }
  }
  return [
    [index,
      'every cell holds WHICH site won — its index in seedX/seedY, 0…15 — not where that site is. ' +
      'The index is useless to the next pass, which has no seed list to look it up in. Pack the ' +
      'coordinates instead: seedY[best] * n + seedX[best]'],
    [dist,
      'every cell holds the DISTANCE to its nearest site. A distance is a dead end: the next pass ' +
      'has to re-measure that site from a different pixel, and a number that has forgotten where ' +
      'the site was cannot answer. Return the packed position, not how far away it is'],
    [dist2,
      'every cell holds the SQUARED distance to its nearest site. Squared distance is the right ' +
      'thing to compare, but it is not what the cell carries — return the packed position of the ' +
      'winner: seedY[best] * n + seedX[best]'],
    [swapped,
      'the pack is the wrong way round — that is seedX * n + seedY. Rows come first, exactly as in ' +
      'grid[y][x]: seedY[best] * n + seedX[best]'],
  ];
}

// Task 2: five ways one flood pass goes wrong, all built from the same mirror.
function passProbes(grid, k) {
  const probes = [
    [floodPassJS(grid, k, { fromNeighbour: true }),
      'each candidate was judged by how far ITS OWN cell is from the seed it carries, not by how ' +
      'far that seed is from YOU. Measure from this thread: (sx − this.thread.x), (sy − this.thread.y)'],
    [floodPassJS(grid, k, { toNeighbour: true }),
      'you compared the distance to the NEIGHBOUR CELL, which is the same nine numbers for every ' +
      'thread. What is being compared is the distance to the SEED each neighbour carries — decode ' +
      'the id first'],
    [floodPassJS(grid, k, { rawId: true }),
      'the smallest packed id won, not the nearest seed — an id is a position, not a distance. ' +
      'Unpack it into (sx, sy) and compare (sx − x)² + (sy − y)²'],
    [floodPassJS(grid, k, { lastWins: true }),
      'the last candidate with a seed won regardless of distance — the assignment needs its guard: ' +
      'only take the candidate when its seed is closer than the best one so far'],
    [floodPassJS(grid, k, { skipCentre: true }),
      'the nine candidates include this pixel itself. dx and dy each run −1, 0, 1, and the 0, 0 ' +
      'case is how a cell keeps the seed it already had when no neighbour beats it'],
  ];
  if (k !== 1) {
    probes.push([floodPassJS(grid, 1),
      'the neighbours are k away, not 1 — the offsets are dx * k and dy * k. A stride of 1 is the ' +
      'last rung of the ladder, not the whole thing']);
  }
  return probes;
}

// A cell holding an id that is neither -1 nor a position on the grid did not
// come from the algorithm. Two causes, and they must not be confused: the
// starting value of `bestD` coming back means the DISTANCE was returned instead
// of the winner, and anything else means a read ran off the edge (undefined on
// the CPU backend, a texel from who-knows-where on WebGL). Before this
// distinction existed, `return bestD` — the exact mistake task 1 spends a whole
// page on — was accused of an out-of-bounds read it had not made.
function outOfRangeHint(got) {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const v = got[y] && got[y][x];
      if (typeof v !== 'number' || Number.isNaN(v) || v < -1.5 || v > CELLS - 0.5) {
        if (Math.abs(v - CELLS * 2) <= 0.5) {
          return `cell [${y}][${x}] came back ${CELLS * 2}, which is the value bestD starts at — so ` +
            'this pass is returning the distance, not the winner. bestD exists only to decide WHICH ' +
            'candidate wins; what a cell carries is best, that candidate\'s packed id';
        }
        return `cell [${y}][${x}] came back ${v}, which is neither −1 nor a position on the grid — ` +
          'a neighbour off the edge got read. Skip a candidate unless 0 ≤ nx < n and 0 ≤ ny < n';
      }
    }
  }
  return null;
}

// ---- kernel finders -------------------------------------------------------
//
// NEVER by creation order — a learner may add or reorder kernels. And never by
// calling a kernel with argument types it was not built for: gpu.js
// re-specialises a kernel when its signature changes, so one exploratory call
// with a grid where a number belongs leaves that kernel broken for every later
// call. (It is also how you get a kernel doing arithmetic on whole rows, which
// on the CPU backend costs minutes.)
//
// So: filter on the SHAPE of the arguments the run itself passed — the sandbox
// records them on every kernel as .lastArgs — and only then make a probe call,
// always with that same shape.

function isArrayish(v) {
  return Array.isArray(v) || ArrayBuffer.isView(v);
}

// '2d,num' for flood(grid, k), '2d,2d' for combine(a, b), '1d,1d' for
// exact(seedX, seedY), '2d' for distance(grid). null if never called.
function argShape(k) {
  const args = k && Array.isArray(k.lastArgs) ? k.lastArgs : null;
  if (!args || !args.length) return null;
  return args
    .map(a => (isArrayish(a) ? (isArrayish(a[0]) ? '2d' : '1d') : 'num'))
    .join(',');
}

function kernelsShaped(ctx, shape) {
  return ctx.kernels.filter(k => k && k.kernel && !k.kernel.graphical && argShape(k) === shape);
}

// A grid with exactly one seed, at (5, 7) → packed id 901.
function probeGrid() {
  const grid = new Array(N);
  for (let y = 0; y < N; y++) grid[y] = new Array(N).fill(-1);
  grid[7][5] = 7 * N + 5;
  return grid;
}

function constantGrid(value) {
  const grid = new Array(N);
  for (let y = 0; y < N; y++) grid[y] = new Array(N).fill(value);
  return grid;
}

// The one-grid numeric kernel: task 4's distance pass.
async function findDistanceKernel(ctx) {
  for (const k of kernelsShaped(ctx, '2d')) {
    let out;
    try {
      out = await k(probeGrid());
    } catch (e) {
      continue;
    }
    if (out && out.length === N && out[0] && out[0].length === N) return k;
  }
  return null;
}

// Task 5's seeding kernel. It shares its (grid, number) signature with the
// flood pass, so it is told apart STRUCTURALLY — by the one property a seeding
// kernel cannot lose however wrong it is: it reads ONLY its own cell. Flip a
// single mask cell and a seeding kernel's answer moves in at most that one
// cell; a flood pass, which reads nine, moves in several.
//
// That test is deliberately blind to WHICH mistake was made. An inverted
// comparison, a swapped pack, 0 where −1 belongs are all wrong seedWheres, not
// missing ones, and each has a probe in the test that names it. The earlier,
// stricter finder — "answers only −1 or the cell's own packed id" — recognised
// the first of those and reported the other two as "no seeding kernel found",
// which is both untrue and useless: a finder that refuses to recognise a broken
// kernel turns a precise diagnosis into a denial that the code exists.
//
// `strict` is kept as the tie-breaker for a learner who added another per-cell
// kernel over the same (grid, number) signature: the one answering only −1 or
// its own id is the seeding kernel.
async function findSeedWhereKernel(ctx) {
  const flat = new Array(N);
  for (let y = 0; y < N; y++) flat[y] = new Array(N).fill(0);
  const bumped = flat.map(row => row.slice());
  bumped[90][60] = 1; // off the diagonal, so a swapped pack is visible too
  let strict = null;
  let local = null;
  for (const k of kernelsShaped(ctx, '2d,num')) {
    let a;
    let b;
    try {
      a = await k(flat, 1);
      b = await k(bumped, 1);
    } catch (e) {
      continue;
    }
    if (!a || a.length !== N || !a[0] || a[0].length !== N) continue;
    if (!b || b.length !== N || !b[0] || b[0].length !== N) continue;
    let perCell = true;
    for (let y = 0; y < N && perCell; y++) {
      for (let x = 0; x < N; x++) {
        if (a[y][x] === b[y][x]) continue;
        if (y !== 90 || x !== 60) {
          perCell = false;
          break;
        }
      }
    }
    if (!perCell) continue;
    if (!local) local = k;
    let own = true;
    for (let y = 0; y < N && own; y++) {
      for (let x = 0; x < N; x++) {
        const v = Math.round(b[y][x]);
        if (v !== -1 && v !== y * N + x) {
          own = false;
          break;
        }
      }
    }
    if (own && !strict) strict = k;
  }
  return strict || local;
}

// Task 5's combine kernel — and task 6's verdict kernel, which has the same
// two-grids-in shape. In both tasks it is the only kernel the run ever handed
// two grids to, so nothing needs probing at all; the behaviour check below is
// the tie-breaker for a learner who added another. Again: a kernel that
// computes the WRONG thing must still be found, or the probes never get to
// speak.
async function findTwoGridKernel(ctx, prefer) {
  const shaped = kernelsShaped(ctx, '2d,2d');
  if (shaped.length <= 1) return shaped[0] || null;
  let fallback = null;
  for (const k of shaped) {
    let out;
    try {
      out = await k(...prefer.args);
    } catch (e) {
      continue;
    }
    if (!out || out.length !== N || !out[0] || out[0].length !== N) continue;
    if (!fallback) fallback = k;
    if (prefer.test(out)) return k;
  }
  return fallback;
}

function findCombineKernel(ctx) {
  return findTwoGridKernel(ctx, {
    args: [constantGrid(3), constantGrid(1)],
    // either direction: a flipped sign is a wrong combine, not a missing one
    test: out => Math.abs(Math.abs(out[64][64]) - 2) <= 1e-3,
  });
}

function findWorseKernel(ctx, a, b) {
  return findTwoGridKernel(ctx, {
    args: [a, b],
    test: out => {
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const v = out[y][x];
          if (!(Math.abs(v) <= 1e-3 || Math.abs(v - 1) <= 1e-3)) return false;
        }
      }
      return true;
    },
  });
}

// ---- console readers ------------------------------------------------------

function loggedNumbers(logs) {
  const out = [];
  for (const line of logs || []) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?/g);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

function renderedFrames(ctx) {
  return (ctx.logs || []).filter(line => line.type === 'canvas').length;
}

// The first series of the first plot() call, as plain numbers.
function plottedSeries(ctx) {
  const entry = (ctx.logs || []).find(line => line.type === 'plot' && line.plot);
  if (!entry || !entry.plot.series || !entry.plot.series.length) return null;
  return entry.plot.series[0].values.map(Number);
}

// ---- shared kernel source (given to the learner, quoted by the prose) -----

const FLOOD_KERNEL = `const flood = gpu.createKernel(function (grid, k) {
  const x = this.thread.x;
  const y = this.thread.y;
  let best = -1;
  let bestD = this.constants.n * this.constants.n * 2;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const nx = x + dx * k;
      const ny = y + dy * k;
      if (nx >= 0 && nx < this.constants.n && ny >= 0 && ny < this.constants.n) {
        const id = grid[ny][nx];
        if (id >= 0) {
          const sy = Math.floor(id / this.constants.n);
          const sx = id - sy * this.constants.n;
          const d = (sx - x) * (sx - x) + (sy - y) * (sy - y);
          if (d < bestD) {
            bestD = d;
            best = id;
          }
        }
      }
    }
  }
  return best;
}, { output: [128, 128], constants: { n: 128 } });`;

const PAINT_KERNEL = `const paint = gpu.createKernel(function (grid) {
  const id = grid[this.thread.y][this.thread.x];
  let r = 0.11;
  let g = 0.12;
  let b = 0.17;
  if (id >= 0) {
    const sy = Math.floor(id / this.constants.n);
    const sx = id - sy * this.constants.n;
    r = 0.22 + 0.68 * (sx / this.constants.n);
    g = 0.30 + 0.55 * (sy / this.constants.n);
    b = 0.88 - 0.6 * (sx / this.constants.n);
  }
  this.color(r, g, b, 1);
}, { output: [128, 128], graphical: true, constants: { n: 128 } });`;

const DISTANCE_KERNEL = `const distance = gpu.createKernel(function (grid) {
  const x = this.thread.x;
  const y = this.thread.y;
  const id = grid[y][x];
  const sy = Math.floor(id / this.constants.n);
  const sx = id - sy * this.constants.n;
  return Math.sqrt((sx - x) * (sx - x) + (sy - y) * (sy - y));
}, { output: [128, 128], constants: { n: 128 } });`;

const LADDER_DRIVER = `async function ladder(seeded) {
  let g = seeded;
  for (let k = 64; k >= 1; k = k / 2) g = await flood(g, k);
  return g;
}`;

export default {
  uuid: 'a741a650-84e8-4362-a344-43a4d7018c7f',
  version: 1,
  slug: 'jump-flooding',
  title: 'Jump Flooding: Voronoi in log n Passes',
  blurb:
    'A Voronoi diagram and a signed distance field in log₂(n) passes — more total work than the CPU algorithm, and faster anyway.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'nearest-seed',
      title: 'What a Cell Has to Carry',
      intro: `<p>Scatter a handful of <strong>seeds</strong> over a grid and colour every cell by
        whichever seed is closest. That is a <strong>Voronoi diagram</strong>, and it is one of the
        most useful pictures in graphics: it is a distance field, a nearest-neighbour lookup, a
        watershed, a shatter pattern and a texture, depending on who is asking.</p>
        <p>A CPU builds one in <em>O(n)</em> in the number of pixels — Felzenszwalb's exact distance
        transform runs a couple of linear scans along every row, then the same down every column,
        and it is done. This module builds the same picture in <em>O(n log n)</em> and wins anyway,
        because each of those scans is a <strong>chain</strong>: the answer at column <em>j</em> is
        read off a running lower envelope that columns <em>0…j−1</em> built, so the exact algorithm
        offers one thread per row and nothing finer. Jump flooding hands all 16,384 cells to their
        own threads, seven times over. That is the whole reason this algorithm exists, and it is a
        different claim from "the GPU is faster": jump flooding does <strong>more total work</strong>
        than the algorithm it beats.</p>
        <p>Start where anyone would: ask each cell to check all 16 seeds. What matters is not the
        loop — it is <strong>what the cell writes down</strong>. Not the distance. The seed's
        <em>position</em>, because the next pass will have to measure that seed again from a
        different pixel. A gpu.js cell holds one number, so the pair is packed:
        <code>id = sy * 128 + sx</code>, and <code>-1</code> for "nothing yet".</p>`,
      goal: `<strong>Goal:</strong> make <code>nearest</code> return the <strong>packed
        position</strong> of the closest seed — <code>seedY[best] * n + seedX[best]</code> — rather
        than the distance the starter hands back.`,
      requirements: [
        'Loop over all <code>this.constants.sites</code> seeds and keep the closest',
        'Compare <strong>squared</strong> distances — no <code>Math.sqrt</code> in the loop',
        'Return the winner packed as <code>seedY[best] * this.constants.n + seedX[best]</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — remember the winner, not just its distance',
          body: `<p>The starter already finds the smallest <code>bestD</code>. Add a second variable
            that remembers <em>which</em> seed produced it, and update both together.</p>
<pre><code>if (d &lt; bestD) {
  bestD = d;
  best = i;
}</code></pre>`,
        },
        {
          title: 'Hint 2 — packing the pair',
          body: `<p>A cell holds one number and you need two. Rows first, exactly as in
            <code>grid[y][x]</code>:</p>
<pre><code>return seedY[best] * this.constants.n + seedX[best];</code></pre>
          <p>Unpacking it later is the same arithmetic backwards:
            <code>sy = Math.floor(id / n)</code>, then <code>sx = id - sy * n</code>.</p>`,
        },
      ],
      transfer: `Packing a payload into the value a thread can write is universal GPGPU
        housekeeping — a CUDA kernel stuffs an index and a key into one <code>uint64</code> so a
        single <code>atomicMin</code> carries both, and a WebGPU jump-flood pass stores its seed in
        an <code>rg32float</code> texture for the same reason. The lesson underneath is the one that
        transfers: a parallel algorithm's state has to be enough to <em>continue</em> from, not just
        enough to display.`,
      starterCode: `// 16 seeds, 16,384 cells, one thread each. Brute force — for now.
const gpu = new GPU({ mode });

const nearest = gpu.createKernel(function (seedX, seedY) {
  const x = this.thread.x;
  const y = this.thread.y;
  let bestD = this.constants.n * this.constants.n * 2;
  for (let i = 0; i < this.constants.sites; i++) {
    const dx = seedX[i] - x;
    const dy = seedY[i] - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
    }
  }
  // TODO: a distance is a dead end — the next pass cannot re-measure from it.
  // Remember WHICH seed won, and return its packed position instead:
  //   seedY[best] * this.constants.n + seedX[best]
  return Math.sqrt(bestD);
}, { output: [128, 128], constants: { n: 128, sites: 16 } });

${PAINT_KERNEL}

const cells = await nearest(seedX, seedY);
await paint(cells);
render(paint.canvas);
console.log('cell (0, 0) carries', cells[0][0]);
`,
      solutionCode: `// 16 seeds, 16,384 cells, one thread each. Brute force — for now.
const gpu = new GPU({ mode });

const nearest = gpu.createKernel(function (seedX, seedY) {
  const x = this.thread.x;
  const y = this.thread.y;
  let best = -1;
  let bestD = this.constants.n * this.constants.n * 2;
  for (let i = 0; i < this.constants.sites; i++) {
    const dx = seedX[i] - x;
    const dy = seedY[i] - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return seedY[best] * this.constants.n + seedX[best];
}, { output: [128, 128], constants: { n: 128, sites: 16 } });

${PAINT_KERNEL}

const cells = await nearest(seedX, seedY);
await paint(cells);
render(paint.canvas);
console.log('cell (0, 0) carries', cells[0][0]);
`,
      inputs: utils => {
        const sites = makeSites(utils, 16, 2120);
        return { seedX: sites.xs, seedY: sites.ys };
      },
      publicTests: [
        {
          name: 'every cell holds the packed position of a real seed',
          run: async ctx => {
            const nearest = soleNumericKernel(ctx);
            ctx.assert(nearest, 'no numeric kernel was created — call gpu.createKernel()');
            const sites = makeSites(ctx.utils, 16, 2120);
            const out = await nearest(sites.xs, sites.ys);
            ctx.assert(out && out.length === N, `expected ${N} rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === N, `each row should hold ${N} values`);
            const valid = new Set(sites.xs.map((x, i) => sites.ys[i] * N + x));
            const expected = exactGridJS(sites.xs, sites.ys);
            for (let y = 0; y < N; y++) {
              for (let x = 0; x < N; x++) {
                if (valid.has(Math.round(out[y][x]))) continue;
                const hint = diagnoseField(out, expected, 1e-3, nearestProbes(sites.xs, sites.ys));
                ctx.assert(
                  false,
                  hint ||
                    `cell [${y}][${x}] holds ${out[y][x]}, which is not the packed position of any ` +
                      'seed — a cell must carry seedY[best] * n + seedX[best]'
                );
              }
            }
          },
        },
        {
          name: 'every cell names a seed no farther than the true nearest one',
          run: async ctx => {
            const nearest = soleNumericKernel(ctx);
            ctx.assert(nearest, 'no numeric kernel was created');
            const sites = makeSites(ctx.utils, 16, 2120);
            const out = await nearest(sites.xs, sites.ys);
            const expected = exactGridJS(sites.xs, sites.ys);
            assertSameDistances(ctx, out, expected, 'nearest seed', () =>
              nearestProbes(sites.xs, sites.ys)
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const nearest = soleNumericKernel(ctx);
            ctx.assert(nearest, 'no numeric kernel was created');
            const sites = makeSites(ctx.utils, 16, 5150);
            const out = await nearest(sites.xs, sites.ys);
            const expected = exactGridJS(sites.xs, sites.ys);
            assertSameDistances(ctx, out, expected, 'nearest seed on a fresh layout', () =>
              nearestProbes(sites.xs, sites.ys)
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'one-pass',
      title: 'One Pass, One Stride',
      intro: `<p>Brute force cost 16 distance tests per cell, and it would cost 16,000 for 16,000
        seeds. Jump flooding never looks at the seed list at all. It looks at
        <strong>nine cells</strong>: itself, and eight neighbours at offset <code>±k</code> — the
        corners and edges of a square of side <code>2k</code>. Each of those nine already carries a
        seed (or <code>-1</code>). Measure every carried seed <em>from here</em>, keep the nearest,
        write it down. That is the entire algorithm.</p>
        <p>Notice the shape of it: every thread <strong>reads</strong> nine cells and writes only
        its own. Nothing is ever pushed outwards to a neighbour. That is the course's gather
        formulation — the one gpu.js forces on you because it has no scatter — and jump flooding is
        the cleanest example of it there is, because the obvious way to describe the algorithm
        ("each seed spreads outwards") is scatter, and the way you actually write it is the exact
        inverse.</p>
        <p>Two traps live in that paragraph. The distance is measured from <em>this</em> pixel to
        the neighbour's seed, never from the neighbour to its own seed. And the neighbour at
        <code>dx = dy = 0</code> is <em>you</em>: keeping what you already had is one of the nine
        cases, not a special one.</p>`,
      goal: `<strong>Goal:</strong> write <code>flood</code> — nine candidates at stride
        <code>k</code>, keep the one whose seed is nearest to this pixel — and run it once at
        <code>k = 64</code>.`,
      requirements: [
        'Visit the nine offsets <code>dx, dy ∈ {−1, 0, 1}</code>, each scaled by <code>k</code>',
        'Skip a candidate that falls off the grid, and one holding <code>-1</code>',
        'Unpack each candidate\'s id and measure that seed from <code>this.thread.x/y</code>',
        'Return the packed id of the nearest — or <code>-1</code> if no candidate had one',
      ],
      hints: [
        {
          title: 'Hint 1 — the nine candidates',
          body: `<p>Two nested loops, each running −1, 0, 1. The offset is scaled by the stride:</p>
<pre><code>const nx = x + dx * k;
const ny = y + dy * k;</code></pre>
          <p>At <code>k = 64</code> that reaches 64 cells away in each direction; at
            <code>k = 1</code> it is the ordinary 3×3 neighbourhood.</p>`,
        },
        {
          title: 'Hint 2 — unpacking a candidate',
          body: `<p><code>id = sy * n + sx</code> comes apart the way it went together:</p>
<pre><code>const sy = Math.floor(id / this.constants.n);
const sx = id - sy * this.constants.n;
const d = (sx - x) * (sx - x) + (sy - y) * (sy - y);</code></pre>
          <p><code>x</code> and <code>y</code> in that last line are <em>this thread's</em>
            coordinates — not <code>nx</code> and <code>ny</code>.</p>`,
        },
        {
          title: 'Hint 3 — the guards',
          body: `<p>Two of them, nested. First that the neighbour exists —
            <code>nx &gt;= 0 &amp;&amp; nx &lt; this.constants.n</code> and the same for
            <code>ny</code> — and then that it carries something,
            <code>id &gt;= 0</code>. Start <code>bestD</code> larger than any distance on the grid
            so the first real candidate always wins.</p>`,
        },
      ],
      transfer: `A fixed nine-tap stencil with a runtime stride is what every platform's jump-flood
        implementation looks like: a WebGPU compute pass sampling a seed texture at
        <code>±k</code>, a CUDA kernel doing the same over global memory, a Metal fragment shader
        with <code>k</code> as a push constant. Nothing about it needs atomics, shared memory or
        scatter — which is precisely why it ports to anything with a texture unit.`,
      starterCode: `// The whole algorithm is nine reads. This is one pass of it.
const gpu = new GPU({ mode });

const flood = gpu.createKernel(function (grid, k) {
  const x = this.thread.x;
  const y = this.thread.y;
  let best = -1;
  let bestD = this.constants.n * this.constants.n * 2;
  // TODO: loop dy and dx over -1, 0, 1.
  //   nx = x + dx * k, ny = y + dy * k
  //   skip it unless it is on the grid AND grid[ny][nx] >= 0
  //   unpack that id, measure the seed FROM THIS PIXEL, keep the nearest
  return best;
}, { output: [128, 128], constants: { n: 128 } });

${PAINT_KERNEL}

await paint(seedGrid);
render(paint.canvas);

const once = await flood(seedGrid, 64);
await paint(once);
render(paint.canvas);

let filled = 0;
for (let y = 0; y < 128; y++) {
  for (let x = 0; x < 128; x++) if (once[y][x] >= 0) filled++;
}
console.log('cells holding a seed: 16 ->', filled);
`,
      solutionCode: `// The whole algorithm is nine reads. This is one pass of it.
const gpu = new GPU({ mode });

${FLOOD_KERNEL}

${PAINT_KERNEL}

await paint(seedGrid);
render(paint.canvas);

const once = await flood(seedGrid, 64);
await paint(once);
render(paint.canvas);

let filled = 0;
for (let y = 0; y < 128; y++) {
  for (let x = 0; x < 128; x++) if (once[y][x] >= 0) filled++;
}
console.log('cells holding a seed: 16 ->', filled);
`,
      inputs: utils => {
        const sites = makeSites(utils, 16, 2120);
        return { seedGrid: seedGridOf(sites.xs, sites.ys) };
      },
      publicTests: [
        {
          name: 'one pass at <code>k = 64</code> spreads 16 seeds to 64 cells',
          run: async ctx => {
            const flood = soleNumericKernel(ctx);
            ctx.assert(flood, 'no numeric kernel was created — call gpu.createKernel()');
            const sites = makeSites(ctx.utils, 16, 2120);
            const seeded = seedGridOf(sites.xs, sites.ys);
            const out = await flood(seeded, 64);
            ctx.assert(out && out.length === N, `expected ${N} rows, got ${out && out.length}`);
            const range = outOfRangeHint(out);
            ctx.assert(!range, range || 'a cell holds something that is not a position');
            const filled = countFilledJS(out);
            if (filled !== 64) {
              const expected = floodPassJS(seeded, 64);
              const hint = diagnoseField(out, expected, 1e-3, passProbes(seeded, 64));
              ctx.assertClose(
                filled,
                64,
                0.5,
                hint ||
                  'cells holding a seed after one stride-64 pass — each of the 16 seeds should have ' +
                    'reached the corners of its 128-wide square that land on the grid'
              );
            }
          },
        },
        {
          name: 'with candidates competing, the nearest seed wins',
          run: async ctx => {
            // A half-flooded field: every cell has a seed and most have several
            // DIFFERENT ones within reach, so every way of choosing wrongly
            // shows up here. On the sparse starting grid it cannot — no cell
            // there ever sees two candidates at once.
            const flood = soleNumericKernel(ctx);
            ctx.assert(flood, 'no numeric kernel was created');
            const sites = makeSites(ctx.utils, 16, 2120);
            let base = seedGridOf(sites.xs, sites.ys);
            for (const k of [64, 32, 16, 8]) base = floodPassJS(base, k);
            const out = await flood(base, 4);
            const range = outOfRangeHint(out);
            ctx.assert(!range, range || 'a cell holds something that is not a position');
            assertSameDistances(ctx, out, floodPassJS(base, 4), 'stride-4 pass', () =>
              passProbes(base, 4)
            );
          },
        },
        {
          name: 'the filled count is reported',
          run: async ctx => {
            ctx.assert(
              loggedNumbers(ctx.logs).some(v => Math.abs(v - 64) < 0.5),
              'log how many cells hold a seed after the pass — expected 64 in the console output'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const flood = soleNumericKernel(ctx);
            ctx.assert(flood, 'no numeric kernel was created');
            const sites = makeSites(ctx.utils, 16, 5150);
            let base = seedGridOf(sites.xs, sites.ys);
            // stride 32 on the bare seeds, then stride 2 on a contended field
            const first = await flood(base, 32);
            assertSameDistances(ctx, first, floodPassJS(base, 32), 'stride-32 pass', () =>
              passProbes(base, 32)
            );
            for (const k of [64, 32, 16, 8, 4]) base = floodPassJS(base, k);
            const second = await flood(base, 2);
            assertSameDistances(ctx, second, floodPassJS(base, 2), 'stride-2 pass', () =>
              passProbes(base, 2)
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'halving-ladder',
      title: 'The Halving Ladder',
      intro: `<p>One pass at <code>k = 64</code> moved 16 seeds into 64 cells. Useless on its own —
        and then you halve the stride and run it again. And again. <code>64, 32, 16, 8, 4, 2, 1</code>:
        seven passes on a 128-wide grid, <code>log₂(n)</code> of them, and the diagram is finished.
        Seven is enough because any distance up to 127 is a sum of those powers of two —
        <code>127 = 64 + 32 + 16 + 8 + 4 + 2 + 1</code>, and a shorter one simply drops the terms it
        does not need — so every seed has a route of jumps to every cell. (Having a route and
        arriving are not quite the same thing, which is what the last task is for.) It is the same
        halving ladder <em>Reductions</em> climbs, run backwards.</p>
        <p>The loop lives in JavaScript; the work stays on the GPU. Seven launches instead of a
        pair of scans, and each launch moves all 16,384 cells at once. Count the work honestly:
        <em>n log n</em> against the exact transform's <em>n</em>. Jump flooding loses that
        comparison and wins the race, because the exact transform spends its <em>n</em> walking
        chains — 128 cells deep along a row, then 128 deep down a column — while jump flooding
        spends its <em>n log n</em> as seven steps of 16,384 independent ones.</p>
        <p>Render inside the loop and you get the best view in this course: a frame scrubber you can
        drag, watching the diagram arrive in seven jumps — sparse dust, then blocks, then the
        boundaries snapping straight on the last pass.</p>`,
      goal: `<strong>Goal:</strong> drive the ladder — start <code>k</code> at 64, halve it to 1,
        feed each pass's output into the next, and <code>render()</code> every pass.`,
      requirements: [
        'Loop <code>k = 64, 32, 16, 8, 4, 2, 1</code> — halve, never double',
        'Each pass floods the <strong>previous pass\'s output</strong>, not <code>seedGrid</code>',
        '<code>await</code> each pass before launching the next',
        'Paint and <code>render()</code> inside the loop, and record <code>countFilled</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the loop',
          body: `<p>Halving is just the update expression:</p>
<pre><code>for (let k = 64; k &gt;= 1; k = k / 2) {
  // …
}</code></pre>
          <p>Seven iterations, and the last one is <code>k = 1</code>.</p>`,
        },
        {
          title: 'Hint 2 — carrying the field forward',
          body: `<p><code>grid</code> has to be reassigned, or every pass re-floods the same sparse
            starting field:</p>
<pre><code>grid = await flood(grid, k);</code></pre>
          <p>Never <code>Promise.all</code> here — pass <em>k</em> + 1 reads pass <em>k</em>'s
            output, so the ladder is sequential by construction.</p>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<pre><code>for (let k = 64; k &gt;= 1; k = k / 2) {
  grid = await flood(grid, k);
  filled.push(countFilled(grid));
  await paint(grid);
  render(paint.canvas);
}</code></pre>`,
        },
      ],
      transfer: `Driving a shrinking sequence of dispatches from the host is the standard shape of
        every multi-pass GPU algorithm: CUDA launches a kernel per rung, WebGPU records repeated
        dispatches ping-ponging between two textures, Metal encodes one compute pass each. The
        stride schedule is data the host owns; the parallelism is what the device owns. Real
        implementations ping-pong between two buffers rather than reading back — here each pass
        already hands JavaScript a plain array, which is what makes <code>countFilled</code> and the
        per-pass render free.`,
      starterCode: `// Seven passes. Halve the stride each time and the picture arrives.
const gpu = new GPU({ mode });

${FLOOD_KERNEL}

${PAINT_KERNEL}

function countFilled(grid) {
  let n = 0;
  for (let y = 0; y < 128; y++) {
    for (let x = 0; x < 128; x++) if (grid[y][x] >= 0) n++;
  }
  return n;
}

let grid = seedGrid;
const filled = [countFilled(grid)];
await paint(grid);
render(paint.canvas);

// TODO: seven passes. Start k at 64 and HALVE it every time, flooding the
// CURRENT grid — then record countFilled(grid), paint it and render() it, so
// the console gives you a scrubber over the whole ladder.

plot(filled, { title: 'cells holding a seed, pass by pass', log: true });
console.log('passes:', filled.length - 1, '- filled:', filled);
`,
      solutionCode: `// Seven passes. Halve the stride each time and the picture arrives.
const gpu = new GPU({ mode });

${FLOOD_KERNEL}

${PAINT_KERNEL}

function countFilled(grid) {
  let n = 0;
  for (let y = 0; y < 128; y++) {
    for (let x = 0; x < 128; x++) if (grid[y][x] >= 0) n++;
  }
  return n;
}

let grid = seedGrid;
const filled = [countFilled(grid)];
await paint(grid);
render(paint.canvas);

for (let k = 64; k >= 1; k = k / 2) {
  grid = await flood(grid, k);
  filled.push(countFilled(grid));
  await paint(grid);
  render(paint.canvas);
}

plot(filled, { title: 'cells holding a seed, pass by pass', log: true });
console.log('passes:', filled.length - 1, '- filled:', filled);
`,
      inputs: utils => {
        const sites = makeSites(utils, 16, 2120);
        return { seedGrid: seedGridOf(sites.xs, sites.ys) };
      },
      publicTests: [
        {
          name: 'the ladder ran seven passes, halving the stride',
          run: async ctx => {
            const sites = makeSites(ctx.utils, 16, 2120);
            const seeded = seedGridOf(sites.xs, sites.ys);
            const expected = [countFilledJS(seeded)];
            let g = seeded;
            for (const k of STRIDES) {
              g = floodPassJS(g, k);
              expected.push(countFilledJS(g));
            }
            const got = plottedSeries(ctx);
            ctx.assert(
              got,
              'no plot() call reached the console — leave the plot(filled, …) line in place so the ' +
                'pass-by-pass counts can be checked'
            );
            if (seqMatches(got, expected)) return;
            // Only now, on the failing path, is it worth running three more
            // mirror ladders to work out WHICH loop was written.
            const upward = [countFilledJS(seeded)];
            let u = seeded;
            for (const k of [...STRIDES].reverse()) {
              u = floodPassJS(u, k);
              upward.push(countFilledJS(u));
            }
            const reseeded = [countFilledJS(seeded)];
            for (const k of STRIDES) reseeded.push(countFilledJS(floodPassJS(seeded, k)));
            const stuck = [countFilledJS(seeded)];
            let s = seeded;
            for (let i = 0; i < STRIDES.length; i++) {
              s = floodPassJS(s, 1);
              stuck.push(countFilledJS(s));
            }
            const hint = diagnoseSeq(got, expected, [
              [upward,
                'the strides went UP, not down. Halving is what makes the long jumps happen while ' +
                'the field is still empty — starting at 1 spends the fine passes first and the ' +
                'coarse ones on a field that is already full'],
              [reseeded,
                'every pass flooded seedGrid again, so nothing accumulated — assign the result back: ' +
                'grid = await flood(grid, k)'],
              [stuck,
                'every pass ran at stride 1, so the field creeps outward one cell at a time. That is ' +
                'the flood fill this algorithm exists to replace: k has to start at 64 and halve'],
            ]);
            ctx.assert(
              got.length === expected.length,
              hint ||
                `expected ${expected.length} entries in filled (one before the loop and one per pass), ` +
                  `got ${got.length}`
            );
            for (let i = 0; i < expected.length; i++) {
              ctx.assertClose(
                got[i],
                expected[i],
                0.5,
                hint || `cells filled after pass ${i} (0 = before the loop)`
              );
            }
          },
        },
        {
          name: 'every pass was rendered — the console shows a frame scrubber',
          run: async ctx => {
            const frames = renderedFrames(ctx);
            ctx.assert(
              frames >= STRIDES.length + 1,
              `expected ${STRIDES.length + 1} rendered frames (the seeds, then one per pass), got ` +
                `${frames} — call render(paint.canvas) inside the loop, not only after it`
            );
          },
        },
        {
          name: 'the finished field is a complete Voronoi diagram',
          run: async ctx => {
            const flood = soleNumericKernel(ctx);
            ctx.assert(flood, 'no numeric kernel found — the flood pass should still be here');
            const sites = makeSites(ctx.utils, 16, 2120);
            let grid = seedGridOf(sites.xs, sites.ys);
            for (const k of STRIDES) grid = await flood(grid, k);
            const truth = exactGridJS(sites.xs, sites.ys);
            ctx.assertClose(countFilledJS(grid), CELLS, 0.5, 'every cell should end up with a seed');
            ctx.assertClose(
              wrongCells(grid, truth),
              0,
              0.5,
              'on this seed layout the ladder reproduces the brute-force diagram exactly'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const flood = soleNumericKernel(ctx);
            ctx.assert(flood, 'no numeric kernel found');
            const sites = makeSites(ctx.utils, 16, 5150);
            let grid = seedGridOf(sites.xs, sites.ys);
            const expected = ladderJS(seedGridOf(sites.xs, sites.ys));
            for (const k of STRIDES) grid = await flood(grid, k);
            assertSameDistances(ctx, grid, expected, 'after the full ladder');
            ctx.assertClose(
              wrongCells(grid, exactGridJS(sites.xs, sites.ys)),
              0,
              0.5,
              'the ladder should reproduce brute force exactly on this layout too'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'distance-field',
      title: 'From Seeds to Distances',
      intro: `<p>The finished field already <em>is</em> a distance field — you just have to ask it.
        Every cell knows where its nearest seed is, so the distance to that seed is one subtraction
        and one square root away, in a single extra pass with no memory of its own.</p>
        <p>This is where carrying the position rather than the distance pays off twice over. Had the
        cells accumulated distances, every pass would have compounded whatever rounding the last one
        introduced. Carrying coordinates means the distance is computed <strong>once, at the
        end</strong>, from two exact integers — so the field is as accurate as the seed assignment
        is, and no more approximate than that.</p>`,
      goal: `<strong>Goal:</strong> write <code>distance</code> — unpack each cell's seed and return
        the Euclidean distance from the cell to it.`,
      requirements: [
        'One kernel, one argument: the finished <code>grid</code> of packed ids',
        'Unpack with <code>Math.floor(id / n)</code> then <code>id − sy * n</code>',
        'Return <code>Math.sqrt(…)</code> — this pass is where the square root belongs',
      ],
      hints: [
        {
          title: 'Hint 1 — the same unpack as the flood pass',
          body: `<p>Nothing new: it is the two lines you already wrote inside the loop, applied
            once.</p>
<pre><code>const sy = Math.floor(id / this.constants.n);
const sx = id - sy * this.constants.n;</code></pre>`,
        },
        {
          title: 'Hint 2 — the whole body',
          body: `<pre><code>return Math.sqrt((sx - x) * (sx - x) + (sy - y) * (sy - y));</code></pre>
          <p>A seed cell measures 0 from itself, which is exactly right — those are the black
            pinpricks in the rendered field.</p>`,
        },
      ],
      transfer: `Distance fields are the workhorse texture of real-time graphics: glyph rendering
        (Valve's signed-distance text), outlines and glows, soft particle collision, path planning
        and morphological dilation are all "threshold a distance field". Every one of them wants the
        field regenerated per frame from changing input, which is why this algorithm — not the
        asymptotically better sweep — is the one that ships.`,
      starterCode: `// The seed field IS a distance field. One pass to read it out.
const gpu = new GPU({ mode });

${FLOOD_KERNEL}

const distance = gpu.createKernel(function (grid) {
  const x = this.thread.x;
  const y = this.thread.y;
  const id = grid[y][x];
  // TODO: unpack id into (sx, sy) and return the distance from (x, y) to it.
  return id;
}, { output: [128, 128], constants: { n: 128 } });

const shade = gpu.createKernel(function (field) {
  const t = Math.min(1, field[this.thread.y][this.thread.x] / this.constants.scale);
  this.color(t, t, t, 1);
}, { output: [128, 128], graphical: true, constants: { scale: 56 } });

let grid = seedGrid;
for (let k = 64; k >= 1; k = k / 2) grid = await flood(grid, k);

const field = await distance(grid);
await shade(field);
render(shade.canvas);

plot(field[64], { title: 'distance to the nearest seed, along row 64' });
console.log('distance at (0, 0):', field[0][0]);
`,
      solutionCode: `// The seed field IS a distance field. One pass to read it out.
const gpu = new GPU({ mode });

${FLOOD_KERNEL}

${DISTANCE_KERNEL}

const shade = gpu.createKernel(function (field) {
  const t = Math.min(1, field[this.thread.y][this.thread.x] / this.constants.scale);
  this.color(t, t, t, 1);
}, { output: [128, 128], graphical: true, constants: { scale: 56 } });

let grid = seedGrid;
for (let k = 64; k >= 1; k = k / 2) grid = await flood(grid, k);

const field = await distance(grid);
await shade(field);
render(shade.canvas);

plot(field[64], { title: 'distance to the nearest seed, along row 64' });
console.log('distance at (0, 0):', field[0][0]);
`,
      inputs: utils => {
        const sites = makeSites(utils, 16, 2120);
        return { seedGrid: seedGridOf(sites.xs, sites.ys) };
      },
      publicTests: [
        {
          name: 'the distance pass reads a hand-made field correctly',
          run: async ctx => {
            const distance = await findDistanceKernel(ctx);
            ctx.assert(
              distance,
              'no one-argument numeric kernel found — distance takes the grid of packed ids and ' +
                'nothing else'
            );
            // Every cell carries the SAME seed, at (5, 7) — so the answer is a
            // clean cone whose every wrong version is a different surface.
            const filled = constantGrid(7 * N + 5);
            const out = await distance(filled);
            const expected = distanceGridJS(filled);
            assertField(ctx, out, expected, 2e-3, 'distance to the seed at (5, 7)', () => {
              const squared = new Array(N);
              const swapped = new Array(N);
              const raw = new Array(N);
              for (let y = 0; y < N; y++) {
                squared[y] = new Array(N);
                swapped[y] = new Array(N);
                raw[y] = new Array(N).fill(7 * N + 5);
                for (let x = 0; x < N; x++) {
                  squared[y][x] = expected[y][x] * expected[y][x];
                  swapped[y][x] = Math.sqrt((7 - x) * (7 - x) + (5 - y) * (5 - y));
                }
              }
              return [
                [squared,
                  'that is the SQUARED distance. Comparing squares is right inside the flood pass — ' +
                  'it keeps every comparison on whole numbers — but this pass is the one that takes ' +
                  'the Math.sqrt'],
                [swapped,
                  'the unpack is the wrong way round. sy = Math.floor(id / n) is the ROW and ' +
                  'sx = id − sy * n is what is left over, the same order as grid[y][x]'],
                [raw,
                  'the packed id came straight back out. It is a position, not a distance — unpack ' +
                  'it into (sx, sy) first'],
              ];
            });
          },
        },
        {
          name: 'the flooded field turns into a distance field',
          run: async ctx => {
            const distance = await findDistanceKernel(ctx);
            ctx.assert(distance, 'no one-argument numeric kernel found');
            const sites = makeSites(ctx.utils, 16, 2120);
            const grid = ladderJS(seedGridOf(sites.xs, sites.ys));
            const out = await distance(grid);
            const expected = distanceGridJS(grid);
            assertField(ctx, out, expected, 2e-3, 'the flooded field as distances');
            for (let i = 0; i < sites.xs.length; i++) {
              ctx.assertClose(
                out[sites.ys[i]][sites.xs[i]],
                0,
                2e-3,
                `seed ${i} should be zero distance from itself`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const distance = await findDistanceKernel(ctx);
            ctx.assert(distance, 'no one-argument numeric kernel found');
            const sites = makeSites(ctx.utils, 16, 5150);
            const grid = ladderJS(seedGridOf(sites.xs, sites.ys));
            const out = await distance(grid);
            const expected = distanceGridJS(grid);
            assertField(ctx, out, expected, 2e-3, 'a second seed layout');
            let maxSeen = 0;
            for (let y = 0; y < N; y++) {
              for (let x = 0; x < N; x++) maxSeen = Math.max(maxSeen, expected[y][x]);
            }
            ctx.assert(maxSeen > 5, 'the reference field should span a real range of distances');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'signed-distance',
      title: 'A Signed Distance Field From a Bitmap',
      intro: `<p>Seeds do not have to be dots. Seed the flood with <em>every pixel inside a
        shape</em> and the finished field answers "how far is the nearest inside pixel?" — which is
        0 inside and grows outside. Seed it with every pixel <em>outside</em> and you get the mirror
        image. Subtract one from the other and the result is a <strong>signed distance field</strong>:
        negative inside, zero on the boundary, positive outside.</p>
        <p><em>Ray-Marched Metaballs</em> marches an SDF that is defined <strong>analytically</strong>
        — a sphere is <code>length(p) − r</code>, and the whole scene is a formula. This is the other
        half of that story: here you <strong>manufacture</strong> one from an arbitrary bitmap.
        Nothing about a five-pointed star wants to be a formula, and it does not have to be. Two
        ladders and a subtraction, and it is marchable, glowable, outlineable — exactly like the
        analytic kind.</p>
        <p>The bookkeeping is the interesting part. <code>dIn</code> is 0 for every inside pixel and
        positive outside; <code>dOut</code> is 0 for every outside pixel and positive inside. So
        <code>dIn − dOut</code> is signed automatically, with no test on the mask at all — one of
        the two terms is always zero.</p>`,
      goal: `<strong>Goal:</strong> write <code>seedWhere(mask, want)</code> — seed the cells where
        the mask equals <code>want</code> — and <code>combine(dIn, dOut)</code>, which returns
        <code>dIn − dOut</code>.`,
      requirements: [
        '<code>seedWhere</code> returns the packed id where <code>mask[y][x] === want</code>, else <code>-1</code>',
        'Flood once with <code>want = 1</code> and once with <code>want = 0</code>, awaiting each ladder',
        '<code>combine</code> returns <code>dIn − dOut</code> — negative inside, positive outside',
      ],
      hints: [
        {
          title: 'Hint 1 — seeding a region',
          body: `<p>The same packed id as ever, gated on the mask:</p>
<pre><code>let id = -1;
if (mask[y][x] === want) id = y * this.constants.n + x;
return id;</code></pre>
          <p>The <code>want</code> argument is what lets one kernel seed both sides.</p>`,
        },
        {
          title: 'Hint 2 — two ladders, one driver',
          body: `<p><code>ladder()</code> takes any seeded field, so it runs twice unchanged:</p>
<pre><code>const dIn = await distance(await ladder(await seedWhere(mask, 1)));
const dOut = await distance(await ladder(await seedWhere(mask, 0)));</code></pre>
          <p>Fourteen passes in total, and every one of them awaited in order.</p>`,
        },
        {
          title: 'Hint 3 — why no sign test is needed',
          body: `<p>Inside a pixel of the shape, the nearest inside pixel is itself, so
            <code>dIn = 0</code> and the answer is <code>−dOut</code>. Outside,
            <code>dOut = 0</code> and the answer is <code>+dIn</code>. The subtraction is the whole
            sign logic.</p>`,
        },
      ],
      transfer: `Manufacturing an SDF from a raster is production practice: Valve's distance-field
        glyphs, Unity and Godot's SDF text, mesh voxelisation into a 3D distance field for collision
        and soft shadows — all of it is "flood a bitmap, subtract two fields". The measurement is a
        pixel-centre one, so this field is quantised to the raster it came from; the usual fix is to
        seed sub-pixel boundary positions rather than pixel centres, which changes the seeding and
        nothing else about the algorithm.`,
      starterCode: `// Seed the inside. Seed the outside. Subtract. That is an SDF.
const gpu = new GPU({ mode });

${FLOOD_KERNEL}

${DISTANCE_KERNEL}

const seedWhere = gpu.createKernel(function (mask, want) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO: return the packed id y * n + x where mask[y][x] === want, else -1.
  return -1;
}, { output: [128, 128], constants: { n: 128 } });

const combine = gpu.createKernel(function (dIn, dOut) {
  // TODO: return dIn - dOut for this cell — negative inside, positive outside.
  return dIn[this.thread.y][this.thread.x];
}, { output: [128, 128] });

const paintSdf = gpu.createKernel(function (field) {
  const s = field[this.thread.y][this.thread.x];
  const band = 0.55 + 0.45 * Math.cos(s * 0.9);
  const t = Math.min(1, Math.abs(s) / 26);
  let r = 0.22 + 0.72 * t * band;
  let g = 0.44 + 0.26 * t * band;
  let b = 0.24 + 0.16 * t * band;
  if (s < 0) {
    r = 0.14 + 0.20 * t * band;
    g = 0.42 + 0.30 * t * band;
    b = 0.55 + 0.44 * t * band;
  }
  this.color(r, g, b, 1);
}, { output: [128, 128], graphical: true });

${LADDER_DRIVER}

const dIn = await distance(await ladder(await seedWhere(mask, 1)));
const dOut = await distance(await ladder(await seedWhere(mask, 0)));
const sdf = await combine(dIn, dOut);

await paintSdf(sdf);
render(paintSdf.canvas);
console.log('sdf at the centre:', sdf[64][64], '- at a corner:', sdf[0][0]);
`,
      solutionCode: `// Seed the inside. Seed the outside. Subtract. That is an SDF.
const gpu = new GPU({ mode });

${FLOOD_KERNEL}

${DISTANCE_KERNEL}

const seedWhere = gpu.createKernel(function (mask, want) {
  const x = this.thread.x;
  const y = this.thread.y;
  let id = -1;
  if (mask[y][x] === want) id = y * this.constants.n + x;
  return id;
}, { output: [128, 128], constants: { n: 128 } });

const combine = gpu.createKernel(function (dIn, dOut) {
  return dIn[this.thread.y][this.thread.x] - dOut[this.thread.y][this.thread.x];
}, { output: [128, 128] });

const paintSdf = gpu.createKernel(function (field) {
  const s = field[this.thread.y][this.thread.x];
  const band = 0.55 + 0.45 * Math.cos(s * 0.9);
  const t = Math.min(1, Math.abs(s) / 26);
  let r = 0.22 + 0.72 * t * band;
  let g = 0.44 + 0.26 * t * band;
  let b = 0.24 + 0.16 * t * band;
  if (s < 0) {
    r = 0.14 + 0.20 * t * band;
    g = 0.42 + 0.30 * t * band;
    b = 0.55 + 0.44 * t * band;
  }
  this.color(r, g, b, 1);
}, { output: [128, 128], graphical: true });

${LADDER_DRIVER}

const dIn = await distance(await ladder(await seedWhere(mask, 1)));
const dOut = await distance(await ladder(await seedWhere(mask, 0)));
const sdf = await combine(dIn, dOut);

await paintSdf(sdf);
render(paintSdf.canvas);
console.log('sdf at the centre:', sdf[64][64], '- at a corner:', sdf[0][0]);
`,
      inputs: () => ({ mask: makeStarMask() }),
      publicTests: [
        {
          name: '<code>seedWhere</code> seeds exactly the cells the mask selects',
          run: async ctx => {
            const seedWhere = await findSeedWhereKernel(ctx);
            ctx.assert(
              seedWhere,
              'no seeding kernel found — seedWhere(mask, want) should answer the packed id ' +
                'y * n + x wherever mask[y][x] === want, and -1 everywhere else'
            );
            const mask = makeStarMask();
            for (const want of [1, 0]) {
              const out = await seedWhere(mask, want);
              const expected = seedWhereJS(mask, want);
              assertField(ctx, out, expected, 0.5, `want = ${want}`, () => {
                const empty = new Array(N);
                const everywhere = new Array(N);
                const swapped = new Array(N);
                const zeroMiss = new Array(N);
                for (let y = 0; y < N; y++) {
                  empty[y] = new Array(N).fill(-1);
                  everywhere[y] = new Array(N);
                  swapped[y] = new Array(N);
                  zeroMiss[y] = new Array(N);
                  for (let x = 0; x < N; x++) {
                    everywhere[y][x] = y * N + x;
                    swapped[y][x] = mask[y][x] === want ? x * N + y : -1;
                    zeroMiss[y][x] = mask[y][x] === want ? y * N + x : 0;
                  }
                }
                return [
                  // Two different slips produce this one field — an inverted
                  // comparison, and a comparison against a hard-coded 0 or 1
                  // instead of `want`. No test can tell them apart, so the
                  // message names both rather than accusing the wrong one.
                  [seedWhereJS(mask, 1 - want),
                    `the cells that got seeded are the ones where the mask is ${1 - want}, but this ` +
                    `call asked for want = ${want}. Either the comparison is inverted — it should be ` +
                    '=== want, not !== want — or the mask is being compared against a hard-coded 0 ' +
                    'or 1, which throws away the argument that lets one kernel seed both sides'],
                  [empty,
                    'every cell came back −1, so the mask test never fired. Compare this cell\'s ' +
                    'mask value with want, and answer y * n + x when they agree'],
                  [everywhere,
                    'every cell was seeded, mask or no mask. Seeding everything makes the flood ' +
                    'a no-op — every cell is already zero distance from "its" seed'],
                  [swapped,
                    'the pack is the wrong way round — that is x * n + y. Rows come first, exactly ' +
                    'as in mask[y][x]: y * this.constants.n + x'],
                  [zeroMiss,
                    'the cells the mask did not select came back 0 rather than −1 — and 0 is a real ' +
                    'packed id, the seed at the top-left corner. Every unselected cell would claim ' +
                    'to be holding it and the flood would have nothing left to do. The miss has to ' +
                    'be a value the flood can recognise as empty: −1'],
                ];
              });
            }
          },
        },
        {
          name: 'the combined field is signed: negative inside, positive outside',
          run: async ctx => {
            const combine = await findCombineKernel(ctx);
            ctx.assert(
              combine,
              'no combine kernel found — it takes two distance fields and returns dIn - dOut'
            );
            const mask = makeStarMask();
            const dIn = distanceGridJS(ladderJS(seedWhereJS(mask, 1)));
            const dOut = distanceGridJS(ladderJS(seedWhereJS(mask, 0)));
            const out = await combine(dIn, dOut);
            const expected = new Array(N);
            for (let y = 0; y < N; y++) {
              expected[y] = new Array(N);
              for (let x = 0; x < N; x++) expected[y][x] = dIn[y][x] - dOut[y][x];
            }
            assertField(ctx, out, expected, 2e-3, 'the signed field', () => {
              const flipped = new Array(N);
              const summed = new Array(N);
              const unsigned = new Array(N);
              for (let y = 0; y < N; y++) {
                flipped[y] = new Array(N);
                summed[y] = new Array(N);
                unsigned[y] = new Array(N);
                for (let x = 0; x < N; x++) {
                  flipped[y][x] = dOut[y][x] - dIn[y][x];
                  summed[y][x] = dIn[y][x] + dOut[y][x];
                  unsigned[y][x] = Math.abs(expected[y][x]);
                }
              }
              // dIn + dOut and |dIn − dOut| are the SAME field — one of the two
              // terms is always zero — so no test could ever tell them apart.
              // They share one message rather than cancelling each other out,
              // and the message says the thing both have in common.
              const unsignedNote =
                'that field has no sign: it is positive everywhere. Adding the two fields and ' +
                'taking |dIn − dOut| produce exactly these numbers, because one of the two terms ' +
                'is always zero. Subtract, in the order dIn − dOut, and the sign comes for free';
              return [
                [flipped,
                  'the sign is inverted — this field is positive inside the shape and negative ' +
                  'outside. An SDF is negative inside, which is dIn − dOut in that order'],
                [summed, unsignedNote],
                [unsigned, unsignedNote],
              ];
            });
          },
        },
        {
          name: 'the finished SDF matches the mask it came from',
          run: async ctx => {
            const seedWhere = await findSeedWhereKernel(ctx);
            const combine = await findCombineKernel(ctx);
            ctx.assert(
              seedWhere && combine,
              'expected a seedWhere kernel and a combine kernel'
            );
            // The ladder and the distance pass belong to earlier tasks, so
            // they run here as the JS mirror: what this test is composing is
            // the learner's TWO kernels, and asking the cpu backend for 14 more
            // full-grid passes would cost more than the rest of the module.
            const mask = makeStarMask();
            const sideOf = async want => distanceGridJS(ladderJS(await seedWhere(mask, want)));
            const got = await combine(await sideOf(1), await sideOf(0));
            const expected = sdfJS(mask);
            assertField(ctx, got, expected, 3e-3, 'the star SDF end to end');
            let insideNegative = 0;
            let outsidePositive = 0;
            for (let y = 0; y < N; y++) {
              for (let x = 0; x < N; x++) {
                if (mask[y][x] === 1 && got[y][x] < 0) insideNegative++;
                if (mask[y][x] === 0 && got[y][x] > 0) outsidePositive++;
              }
            }
            ctx.assert(
              insideNegative > 3000,
              `only ${insideNegative} of the shape's pixels came out negative — an SDF is negative inside`
            );
            ctx.assert(
              outsidePositive > 10000,
              `only ${outsidePositive} background pixels came out positive — an SDF is positive outside`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different bitmap entirely: a hollow ring, so the field has to
            // be negative in an annulus and positive both inside and outside it.
            const mask = new Array(N);
            for (let y = 0; y < N; y++) {
              const row = new Array(N);
              for (let x = 0; x < N; x++) {
                const r = Math.sqrt((x - 63.5) * (x - 63.5) + (y - 63.5) * (y - 63.5));
                row[x] = r > 22 && r < 46 ? 1 : 0;
              }
              mask[y] = row;
            }
            const seedWhere = await findSeedWhereKernel(ctx);
            const combine = await findCombineKernel(ctx);
            ctx.assert(seedWhere && combine, 'expected a seedWhere kernel and a combine kernel');
            const sideOf = async want => distanceGridJS(ladderJS(await seedWhere(mask, want)));
            const got = await combine(await sideOf(1), await sideOf(0));
            assertField(ctx, got, sdfJS(mask), 3e-3, 'a hollow ring');
            ctx.assert(got[64][64] > 10, 'the middle of the ring is outside the shape — sdf > 0');
            ctx.assert(got[64][30] < 0, 'a cell in the band is inside the shape — sdf < 0');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'how-wrong',
      title: 'Payoff: Measure the Lie',
      intro: `<p>Jump flooding is an <strong>approximation</strong>. It is not "exact but for
        rounding": there are seed layouts for which a cell's true nearest seed never reaches it. The
        route of halving jumps the ladder counted on always exists — but a cell part-way along it
        only forwards the seed it is holding at that moment, and it may be holding a different one
        that looked nearer when the pass ran. The chain breaks in the middle. Every honest
        description of this algorithm says so, and the way to believe it is to count.</p>
        <p>Careful about what "wrong" means, though. Two seeds can be exactly the same distance
        away, and then <em>both</em> answers are right — comparing the ids the two methods chose
        would report roughly twice as many failures as there are. A cell is wrong only when the seed
        it holds is <strong>strictly farther</strong> than the true nearest one.</p>
        <p>The layout here is 24 seeds chosen to make the flaw visible. It is not typical: across
        200 random 24-seed layouts on this grid, 95 came out perfect and the average was 1.4 wrong
        cells out of 16,384 — 0.009%. This one manages 67. The standard patch, one extra pass at
        stride 1 ("JFA+1"), takes it to 55 — better, and still not exact. If you need exact, you
        need a different algorithm; what you get here is a fast answer with a bounded, measurable
        error, and for a glow or an outline or a shatter pattern that is the right trade.</p>`,
      goal: `<strong>Goal:</strong> write <code>worse(jfa, truth)</code> — 1 where the flooded seed
        is strictly farther than the true nearest one, 0 otherwise — and <code>countOnes</code> to
        total it.`,
      requirements: [
        'Unpack both ids and compare <strong>squared</strong> distances — whole numbers, so ties are exact',
        'Equal distance is <strong>not</strong> an error: only strictly farther counts',
        'A cell still holding <code>-1</code> counts as wrong',
        '<code>countOnes</code> sums the field in plain JavaScript',
      ],
      hints: [
        {
          title: 'Hint 1 — two unpacks, one comparison',
          body: `<p>Unpack <code>jfa[y][x]</code> and <code>truth[y][x]</code> the usual way, measure
            both seeds from this pixel, and compare the squared distances. Strictly greater:</p>
<pre><code>let bad = 0;
if (dJfa &gt; dTruth) bad = 1;
return bad;</code></pre>`,
        },
        {
          title: 'Hint 2 — the unassigned case',
          body: `<p>An id of <code>-1</code> unpacks to nonsense, so give it a distance larger than
            anything on the grid before the comparison — the same
            <code>n * n * 2</code> the flood pass starts from.</p>`,
        },
        {
          title: 'Hint 3 — counting in JavaScript',
          body: `<p>The field is 0s and 1s, so the count is the sum:</p>
<pre><code>let n = 0;
for (let y = 0; y &lt; 128; y++) {
  for (let x = 0; x &lt; 128; x++) n += grid[y][x];
}
return n;</code></pre>`,
        },
      ],
      transfer: `"Asymptotically worse, measurably approximate, and shipped anyway" is a recurring GPU
        story — screen-space ambient occlusion approximates an integral nobody can afford, temporal
        upscalers approximate frames that were never rendered, and JFA approximates a transform that
        has an exact linear-time algorithm nobody can parallelise. What makes each of them
        defensible is exactly this task: somebody counted the error, published the number, and
        decided it was small enough. An approximation whose error you have not measured is not an
        engineering decision.`,
      starterCode: `// How often is the fast answer the wrong answer? Count it.
const gpu = new GPU({ mode });

${FLOOD_KERNEL}

const exact = gpu.createKernel(function (seedX, seedY) {
  const x = this.thread.x;
  const y = this.thread.y;
  let best = -1;
  let bestD = this.constants.n * this.constants.n * 2;
  for (let i = 0; i < this.constants.sites; i++) {
    const dx = seedX[i] - x;
    const dy = seedY[i] - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = seedY[i] * this.constants.n + seedX[i];
    }
  }
  return best;
}, { output: [128, 128], constants: { n: 128, sites: 24 } });

const worse = gpu.createKernel(function (jfa, truth) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO: unpack both ids, measure both seeds from (x, y), and return 1 only
  // when the flooded one is STRICTLY farther. A tie is not an error.
  // An id of -1 is farther than anything: n * n * 2.
  return 0;
}, { output: [128, 128], constants: { n: 128 } });

const paintErr = gpu.createKernel(function (grid, bad) {
  const id = grid[this.thread.y][this.thread.x];
  const sy = Math.floor(id / this.constants.n);
  const sx = id - sy * this.constants.n;
  let r = 0.22 + 0.68 * (sx / this.constants.n);
  let g = 0.30 + 0.55 * (sy / this.constants.n);
  let b = 0.88 - 0.6 * (sx / this.constants.n);
  if (bad[this.thread.y][this.thread.x] > 0.5) {
    r = 1;
    g = 0.15;
    b = 0.2;
  }
  this.color(r, g, b, 1);
}, { output: [128, 128], graphical: true, constants: { n: 128 } });

function countOnes(grid) {
  // TODO: total the 128x128 field of 0s and 1s and return the count.
  return 0;
}

const truth = await exact(seedX, seedY);
let grid = seedGrid;
const wrong = [countOnes(await worse(grid, truth))];
for (let k = 64; k >= 1; k = k / 2) {
  grid = await flood(grid, k);
  wrong.push(countOnes(await worse(grid, truth)));
}

plot(wrong, { title: 'cells not yet holding their nearest seed', log: true });
console.log('after the ladder:', wrong[wrong.length - 1], 'of 16384 cells are wrong');

const patched = await flood(grid, 1);
console.log('after one extra stride-1 pass:', countOnes(await worse(patched, truth)));

await paintErr(grid, await worse(grid, truth));
render(paintErr.canvas);
`,
      solutionCode: `// How often is the fast answer the wrong answer? Count it.
const gpu = new GPU({ mode });

${FLOOD_KERNEL}

const exact = gpu.createKernel(function (seedX, seedY) {
  const x = this.thread.x;
  const y = this.thread.y;
  let best = -1;
  let bestD = this.constants.n * this.constants.n * 2;
  for (let i = 0; i < this.constants.sites; i++) {
    const dx = seedX[i] - x;
    const dy = seedY[i] - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = seedY[i] * this.constants.n + seedX[i];
    }
  }
  return best;
}, { output: [128, 128], constants: { n: 128, sites: 24 } });

const worse = gpu.createKernel(function (jfa, truth) {
  const x = this.thread.x;
  const y = this.thread.y;
  const a = jfa[y][x];
  const b = truth[y][x];
  const ay = Math.floor(a / this.constants.n);
  const ax = a - ay * this.constants.n;
  const by = Math.floor(b / this.constants.n);
  const bx = b - by * this.constants.n;
  let da = (ax - x) * (ax - x) + (ay - y) * (ay - y);
  if (a < 0) da = this.constants.n * this.constants.n * 2;
  const db = (bx - x) * (bx - x) + (by - y) * (by - y);
  let bad = 0;
  if (da > db) bad = 1;
  return bad;
}, { output: [128, 128], constants: { n: 128 } });

const paintErr = gpu.createKernel(function (grid, bad) {
  const id = grid[this.thread.y][this.thread.x];
  const sy = Math.floor(id / this.constants.n);
  const sx = id - sy * this.constants.n;
  let r = 0.22 + 0.68 * (sx / this.constants.n);
  let g = 0.30 + 0.55 * (sy / this.constants.n);
  let b = 0.88 - 0.6 * (sx / this.constants.n);
  if (bad[this.thread.y][this.thread.x] > 0.5) {
    r = 1;
    g = 0.15;
    b = 0.2;
  }
  this.color(r, g, b, 1);
}, { output: [128, 128], graphical: true, constants: { n: 128 } });

function countOnes(grid) {
  let n = 0;
  for (let y = 0; y < 128; y++) {
    for (let x = 0; x < 128; x++) n += grid[y][x];
  }
  return n;
}

const truth = await exact(seedX, seedY);
let grid = seedGrid;
const wrong = [countOnes(await worse(grid, truth))];
for (let k = 64; k >= 1; k = k / 2) {
  grid = await flood(grid, k);
  wrong.push(countOnes(await worse(grid, truth)));
}

plot(wrong, { title: 'cells not yet holding their nearest seed', log: true });
console.log('after the ladder:', wrong[wrong.length - 1], 'of 16384 cells are wrong');

const patched = await flood(grid, 1);
console.log('after one extra stride-1 pass:', countOnes(await worse(patched, truth)));

await paintErr(grid, await worse(grid, truth));
render(paintErr.canvas);
`,
      inputs: utils => {
        const sites = makeSites(utils, 24, 211);
        return {
          seedX: sites.xs,
          seedY: sites.ys,
          seedGrid: seedGridOf(sites.xs, sites.ys),
        };
      },
      publicTests: [
        {
          name: 'a tie is not an error — only a strictly farther seed is',
          run: async ctx => {
            const sites = makeSites(ctx.utils, 24, 211);
            const jfa = ladderJS(seedGridOf(sites.xs, sites.ys));
            const truth = exactGridJS(sites.xs, sites.ys);
            const worse = await findWorseKernel(ctx, jfa, truth);
            ctx.assert(
              worse,
              'no verdict kernel found — worse(jfa, truth) should return a field of 0s and 1s'
            );
            const out = await worse(jfa, truth);
            const expected = new Array(N);
            for (let y = 0; y < N; y++) {
              expected[y] = new Array(N);
              for (let x = 0; x < N; x++) {
                expected[y][x] = dist2Of(jfa[y][x], x, y) > dist2Of(truth[y][x], x, y) ? 1 : 0;
              }
            }
            assertField(ctx, out, expected, 1e-3, 'the verdict field', () => {
              const idBased = new Array(N);
              const everything = new Array(N);
              const nothing = new Array(N);
              for (let y = 0; y < N; y++) {
                idBased[y] = new Array(N);
                everything[y] = new Array(N).fill(1);
                nothing[y] = new Array(N).fill(0);
                for (let x = 0; x < N; x++) {
                  idBased[y][x] = Math.round(jfa[y][x]) === Math.round(truth[y][x]) ? 0 : 1;
                }
              }
              return [
                [idBased,
                  'the two ids were compared instead of the two distances. Seeds tie: on this ' +
                  'layout 118 cells hold a different id and only 67 of them are actually farther ' +
                  'away, so counting id mismatches nearly doubles the error you report'],
                // An all-1s field and an all-0s field are each reachable by more
                // than one slip, and no test can separate them — so each
                // message describes the field first and offers every mistake
                // that produces it, rather than picking one and being wrong
                // half the time.
                [everything,
                  'every cell came back 1, which would mean the flood got nothing right. The ' +
                  'comparison has to be strictly greater: written >= — or dropped altogether — it ' +
                  'accuses every correct cell of being wrong by exactly zero'],
                [nothing,
                  'every cell came back 0, so no cell is ever accused. Either the comparison has ' +
                  'not been written yet, or it is the wrong way round: unpack both ids, measure ' +
                  'both seeds from this pixel, and return 1 when the FLOODED one is farther — ' +
                  'da > db, not db > da'],
              ];
            });
          },
        },
        {
          name: 'unassigned cells count as wrong',
          run: async ctx => {
            const sites = makeSites(ctx.utils, 24, 211);
            const seeded = seedGridOf(sites.xs, sites.ys);
            const truth = exactGridJS(sites.xs, sites.ys);
            const jfa = ladderJS(seeded);
            const worse = await findWorseKernel(ctx, jfa, truth);
            ctx.assert(worse, 'no verdict kernel found');
            const out = await worse(seeded, truth);
            let ones = 0;
            for (let y = 0; y < N; y++) {
              for (let x = 0; x < N; x++) ones += out[y][x];
            }
            // Only accuse the -1 case when the kernel demonstrably works on a
            // FULL field — a kernel that answers 0 everywhere is a different
            // bug, and the previous test has already named it.
            let hint = null;
            if (ones < 0.5) {
              const onFull = await worse(jfa, truth);
              let live = 0;
              for (let y = 0; y < N; y++) {
                for (let x = 0; x < N; x++) live += onFull[y][x];
              }
              if (live > 0.5) {
                hint =
                  'nothing was counted on a field that is 99.9% empty. An id of −1 unpacks to a ' +
                  'position that does not exist, so it has to be given a distance larger than ' +
                  'anything on the grid before the comparison';
              }
            }
            ctx.assertClose(
              ones,
              CELLS - sites.xs.length,
              0.5,
              hint ||
                'on the bare seed field every cell except the 24 seeds themselves is wrong'
            );
          },
        },
        {
          name: 'the measured error — 67 cells, and 55 after one more pass — is reported',
          run: async ctx => {
            const sites = makeSites(ctx.utils, 24, 211);
            const seeded = seedGridOf(sites.xs, sites.ys);
            const truth = exactGridJS(sites.xs, sites.ys);
            const done = ladderJS(seeded);
            const after = wrongCells(done, truth);
            const patched = wrongCells(floodPassJS(done, 1), truth);
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - after) < 0.5),
              `log how many cells the ladder gets wrong — expected ${after} in the console output`
            );
            ctx.assert(
              nums.some(v => Math.abs(v - patched) < 0.5),
              `log the count after the extra stride-1 pass too — expected ${patched}`
            );
            const series = plottedSeries(ctx);
            ctx.assert(
              series && series.length === STRIDES.length + 1,
              'leave the plot(wrong, …) line in place — it is the pass-by-pass version of the same ' +
                'number, and it is what shows the diagram snapping into place on the last rung'
            );
            ctx.assertClose(series[series.length - 1], after, 0.5, 'the last plotted count');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const sites = makeSites(ctx.utils, 24, 25);
            const seeded = seedGridOf(sites.xs, sites.ys);
            const truth = exactGridJS(sites.xs, sites.ys);
            const jfa = ladderJS(seeded);
            const worse = await findWorseKernel(ctx, jfa, truth);
            ctx.assert(worse, 'no verdict kernel found');
            const out = await worse(jfa, truth);
            const expected = new Array(N);
            let ones = 0;
            for (let y = 0; y < N; y++) {
              expected[y] = new Array(N);
              for (let x = 0; x < N; x++) {
                expected[y][x] = dist2Of(jfa[y][x], x, y) > dist2Of(truth[y][x], x, y) ? 1 : 0;
                ones += out[y][x];
              }
            }
            assertField(ctx, out, expected, 1e-3, 'a second layout');
            ctx.assertClose(ones, wrongCells(jfa, truth), 0.5, 'total wrong cells on a second layout');
            // The same field, judged against itself, has to be all zeros — a
            // field is never strictly farther than itself, and a >= comparison
            // is the one mistake that would say otherwise.
            const self = await worse(jfa, jfa);
            const zeros = new Array(N);
            for (let y = 0; y < N; y++) zeros[y] = new Array(N).fill(0);
            assertField(ctx, self, zeros, 1e-3, 'a field judged against itself');
          },
        },
      ],
    },
  ],
};
