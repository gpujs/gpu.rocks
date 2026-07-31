// Module: Hydraulic Erosion — uuid 07165ca1-0a5d-4c84-9b63-fdd3c9480661
// (short id 07165ca1). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId — this module is new.
//
// Six tasks: hillshading a heightmap so the learner can judge it → the
// downhill drop total (its own pass, because a gather cannot ask a neighbour
// to add up its own drops) → routing every cell's water downhill at once →
// capacity, erosion and deposition, with the SAME exchange computed twice →
// a hundred ping-ponged steps with a rainfall slider and two plots → two
// hundred steps at 96×96, rendered every ten, so the frame scrubber shows a
// drainage network growing out of noise.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested numeric arrays as arguments, this.thread.* for indexing,
// this.constants.* for compile-time values, wrap-around (torus) edges as in
// Reaction–Diffusion, graphical kernels use this.color(). No kernel LOCAL
// shares a name with a constant (gpujs/gpu.js#858 — that throws on the CPU
// backend only). Kernels are asynchronous here, so every call is awaited.
// Every task passes in cpu, webgl and auto.
//
// ---------------------------------------------------------------------------
// THE MODEL, AND WHY IT IS SHAPED LIKE THIS
//
// Everything below is a GATHER. Water moving from cell A to cell B is a
// transfer, and a transfer is the one thing a GPU cannot write directly: A
// cannot scatter into B. The way out is the standard one — B works out what A
// would have sent it, from data B can reach. That needs A's total downhill
// drop, which is an aggregate over A's OWN neighbours, i.e. two rings away
// from B. So the drop total gets its own pass. That extra pass IS the lesson.
//
// Per step, four kernels, all reading the same snapshot of (height, water,
// sediment) plus the drop grid the first one produced:
//
//   drop[c]  = Σ over the four neighbours of max(0, H[c] − H[n])²   where
//              H = height + water (water runs on the SURFACE, not the rock)
//   water'   = water + move·(taken − given) + rain
//   sediment'= sediment + move·(taken − given) + delta
//   height'  = height − delta
//
//   given[c] = drop[c] · field[c] / (drop[c] + soft)
//   taken[c] = Σ over neighbours ABOVE c of  d² · field[n] / (drop[n] + soft)
//   capacity = carry · √drop · water        (stream power: slope × discharge)
//   delta    = load < capacity ? min(pickUp·(capacity − load), maxCut)
//                              : −settle·(load − capacity)
//
// Squared drops rather than plain ones: with plain drops the flow smears
// across every neighbour that is even slightly lower and no channel ever
// forms; squaring makes a cell commit to the steep side. Measured on the
// 96×96 terrain below, 200 steps: plain drops put 53% of the water in the
// wettest 5% of cells and 14% in the wettest 1%, and the picture shows two
// swollen lakes with no channels between them; squared drops put 73% and 31%
// there, and draw the rivers you can see in the payoff task.
//
// `soft` (2e-4) keeps the division safe where a cell has nowhere to go, and
// quiets flat ground: with drop ≈ 0 a cell hands on drop/(drop+soft) ≈ 0 of
// its water instead of all of it, which is what stops a flat lake surface
// checkerboarding.
//
// STABILITY. A cell keeps 1 − move·drop/(drop + soft) of its water. That
// weight is the same object as the centre weight of an explicit heat step, and
// it must not go negative: move ≤ 1. Measured (reference implementation,
// 64×64, 200 steps): move = 1.0 is fine; move = 1.05 puts negative water on
// the grid at step 43; move = 1.6 goes negative at step 9, is past 1e6 by step
// 48 and saturates around 1e108. maxCut plays the same role for the rock — cut
// a cell, and its neighbours are suddenly steeper, so they cut harder. That
// feedback is not marginal: with the clamp removed and everything else left at
// the values below, the 96×96 payoff run detonates at step 96 (and sooner as
// `carry` rises — step 73 at carry = 600).
//
// CONSERVATION. Both routing sums are exactly conservative: Σ taken = Σ given,
// because every d² a receiver divides by drop[n] is one of the terms that made
// drop[n]. So water is created only by rain, and rock turns into sediment
// gram for gram. Two public tests lean on that.
//
// FLOAT DETERMINISM. Tests compute in float64; the WebGL backend computes in
// float32. The model is Lipschitz across all three of its branches (d > 0,
// load < capacity, and the maxCut clamp are each continuous at the boundary),
// so a flipped branch costs nothing. Single-step assertions use 5e-6 on
// quantities of order 1e-2 to 1; the 20-step loop comparison in task 5 uses
// 2e-3 on terrain of order 1 (2e-4 on water and sediment). Measured worst case
// in webgl: 1.1e-7 on a single step (task 4's height), 6.4e-7 after task 5's
// twenty — two to three orders of margin on every assertion.

// ---- the model's constants, shared by starters, solutions and tests --------

const P = {
  move: 0.6, // fraction of a cell's water that moves on, per step (≤ 1)
  soft: 0.0002, // drop floor — safe division, and quiet flat ground
  rain: 0.00002, // depth of rain per cell per step
  carry: 60, // capacity coefficient
  pickUp: 0.3, // how fast under-loaded water bites into the rock
  settle: 0.05, // how fast over-loaded water drops its load
  maxCut: 0.004, // the most rock one step may remove from one cell
};

const LIGHT = 0.5774; // 1/√3 — the light is ℓ = (−1, −1, 1)/√3, from the NW
const ROCK_TINT = 1.15; // height → "how bare is this rock" for the palette

// ---- deterministic terrain ------------------------------------------------

// Periodic value noise on a `cells × cells` lattice, smoothstep-interpolated.
// Periodic so the terrain has no seam: like Reaction–Diffusion's grid, this
// world is a torus, and a cliff at the wrap would make every drainage test a
// test of the seam instead.
function noiseLayer(rand, size, cells) {
  const lattice = [];
  for (let j = 0; j < cells; j++) {
    const row = [];
    for (let i = 0; i < cells; i++) row.push(rand());
    lattice.push(row);
  }
  const smooth = t => t * t * (3 - 2 * t);
  const out = [];
  for (let y = 0; y < size; y++) {
    const row = [];
    const fy = (y / size) * cells;
    const j0 = Math.floor(fy) % cells;
    const j1 = (j0 + 1) % cells;
    const ty = smooth(fy - Math.floor(fy));
    for (let x = 0; x < size; x++) {
      const fx = (x / size) * cells;
      const i0 = Math.floor(fx) % cells;
      const i1 = (i0 + 1) % cells;
      const tx = smooth(fx - Math.floor(fx));
      const a = lattice[j0][i0] * (1 - tx) + lattice[j0][i1] * tx;
      const b = lattice[j1][i0] * (1 - tx) + lattice[j1][i1] * tx;
      row.push(a * (1 - ty) + b * ty);
    }
    out.push(row);
  }
  return out;
}

const OCTAVES = [[4, 1], [8, 0.55], [16, 0.3], [32, 0.16], [64, 0.08]];

// Two hills and two hollows on a wrap-around tile, plus five octaves of noise.
// The smooth part guarantees every cell has somewhere downhill to go; the
// noise is what the water gets to organise into channels.
function makeHills(utils, size, seed = 7, roughness = 0.35) {
  const rand = utils.seededRandom(seed);
  const layers = OCTAVES.map(([cells, amp]) => [noiseLayer(rand, size, cells), amp]);
  const norm = OCTAVES.reduce((sum, [, amp]) => sum + amp, 0);
  const grid = [];
  for (let y = 0; y < size; y++) {
    const row = [];
    for (let x = 0; x < size; x++) {
      let v = 0;
      for (const [layer, amp] of layers) v += layer[y][x] * amp;
      v = v / norm - 0.5;
      const base =
        0.5 +
        0.5 *
          Math.sin((2 * Math.PI * (x + 0.5)) / size) *
          Math.sin((2 * Math.PI * (y + 0.5)) / size);
      row.push(Math.round((base + roughness * v) * 1e6) / 1e6);
    }
    grid.push(row);
  }
  return grid;
}

function makeGrid(size, value) {
  const grid = new Array(size);
  for (let y = 0; y < size; y++) grid[y] = new Array(size).fill(value);
  return grid;
}

// ---- CPU references (they mirror the kernels line for line) ---------------

function wrapLo(i, size) {
  return i < 0 ? size - 1 : i;
}

function wrapHi(i, size) {
  return i > size - 1 ? 0 : i;
}

function neighboursOf(size, y, x) {
  return [
    [y, wrapLo(x - 1, size)],
    [y, wrapHi(x + 1, size)],
    [wrapLo(y - 1, size), x],
    [wrapHi(y + 1, size), x],
  ];
}

function hillshadeRef(height, relief) {
  const size = height.length;
  const out = [];
  for (let y = 0; y < size; y++) {
    const row = [];
    const yd = wrapLo(y - 1, size);
    const yu = wrapHi(y + 1, size);
    for (let x = 0; x < size; x++) {
      const xl = wrapLo(x - 1, size);
      const xr = wrapHi(x + 1, size);
      const gx = (height[y][xr] - height[y][xl]) * 0.5 * relief;
      const gy = (height[yu][x] - height[yd][x]) * 0.5 * relief;
      const inv = 1 / Math.sqrt(gx * gx + gy * gy + 1);
      row.push(Math.max(0, LIGHT * (1 + gx + gy) * inv));
    }
    out.push(row);
  }
  return out;
}

// lit + height → the three 0–1 channels the painter writes.
function shadeColour(lit, h) {
  const rock = Math.min(1, h * ROCK_TINT);
  return [lit * (0.3 + 0.62 * rock), lit * (0.44 + 0.4 * rock), lit * (0.54 + 0.2 * rock)];
}

function dropTotalRef(height, water) {
  const size = height.length;
  const out = [];
  for (let y = 0; y < size; y++) {
    const row = [];
    for (let x = 0; x < size; x++) {
      const here = height[y][x] + water[y][x];
      let total = 0;
      for (const [ny, nx] of neighboursOf(size, y, x)) {
        const d = here - (height[ny][nx] + water[ny][nx]);
        if (d > 0) total += d * d;
      }
      row.push(total);
    }
    out.push(row);
  }
  return out;
}

// One routing pass for any carried field: what this cell keeps plus what its
// uphill neighbours hand down. `move` and `soft` are P's.
function routeRef(height, water, drop, field) {
  const size = height.length;
  const out = [];
  for (let y = 0; y < size; y++) {
    const row = [];
    for (let x = 0; x < size; x++) {
      const here = height[y][x] + water[y][x];
      let taken = 0;
      for (const [ny, nx] of neighboursOf(size, y, x)) {
        const d = height[ny][nx] + water[ny][nx] - here;
        if (d > 0) taken += (d * d * field[ny][nx]) / (drop[ny][nx] + P.soft);
      }
      const given = (drop[y][x] * field[y][x]) / (drop[y][x] + P.soft);
      row.push(field[y][x] + P.move * (taken - given));
    }
    out.push(row);
  }
  return out;
}

function capacityOf(dropHere, waterHere) {
  return P.carry * Math.sqrt(dropHere) * waterHere;
}

function deltaOf(capacity, load) {
  return load < capacity
    ? Math.min(P.pickUp * (capacity - load), P.maxCut)
    : -P.settle * (load - capacity);
}

function moveWaterRef(height, water, drop, rain = P.rain) {
  const routed = routeRef(height, water, drop, water);
  return routed.map(row => row.map(v => v + rain));
}

function moveSedimentRef(height, water, sediment, drop) {
  const size = height.length;
  const routed = routeRef(height, water, drop, sediment);
  return routed.map((row, y) =>
    row.map((v, x) => v + deltaOf(capacityOf(drop[y][x], water[y][x]), sediment[y][x]))
  );
}

function erodeRef(height, water, sediment, drop) {
  return height.map((row, y) =>
    row.map((h, x) => h - deltaOf(capacityOf(drop[y][x], water[y][x]), sediment[y][x]))
  );
}

function stepRef(height, water, sediment, rain = P.rain) {
  const drop = dropTotalRef(height, water);
  return [
    erodeRef(height, water, sediment, drop),
    moveWaterRef(height, water, drop, rain),
    moveSedimentRef(height, water, sediment, drop),
  ];
}

function runRef(height, water, sediment, steps, rain = P.rain) {
  let h = height;
  let w = water;
  let s = sediment;
  for (let i = 0; i < steps; i++) [h, w, s] = stepRef(h, w, s, rain);
  return [h, w, s];
}

// A landscape that has already been rained on for a while — the state the
// middle tasks want as input, so a single step has something to bite into.
function warmUp(utils, size, seed, steps) {
  const height = makeHills(utils, size, seed);
  const [h, w, s] = runRef(height, makeGrid(size, 0), makeGrid(size, 0), steps);
  const round = grid => grid.map(row => row.map(v => Math.round(v * 1e8) / 1e8));
  return { height: round(h), water: round(w), sediment: round(s) };
}

// ---- summaries the tasks and tests both use -------------------------------

function totalOf(grid) {
  let sum = 0;
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) sum += grid[y][x];
  }
  return sum;
}

function maxOf(grid) {
  let top = -Infinity;
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) if (grid[y][x] > top) top = grid[y][x];
  }
  return top;
}

// What share of the water sits in the wettest 5% of cells? 5% for a puddle
// spread evenly, far more once the water has found channels — this is the
// number that says "a drainage network exists" without anyone squinting at a
// picture.
function channelShare(water) {
  const flat = [];
  for (let y = 0; y < water.length; y++) {
    for (let x = 0; x < water[y].length; x++) flat.push(water[y][x]);
  }
  flat.sort((a, b) => b - a);
  const top = Math.max(1, Math.round(flat.length * 0.05));
  let head = 0;
  for (let i = 0; i < top; i++) head += flat[i];
  const all = flat.reduce((a, b) => a + b, 0);
  return all > 0 ? head / all : 0;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so cells where two candidates coincide (a flat
// patch, where "ignored the water" and the right answer agree) stay silent, as
// do observations matching probes that disagree with each other. A wrong
// diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Grid form: a probe must predict EVERY cell (and differ from the right answer
// somewhere) before it may speak. One matching cell is worth nothing here —
// half the grid is nearly flat, and on flat ground most of these mistakes
// produce the right answer anyway. (Task 1 hands this a 3-row grid of colour
// channels rather than a square one, hence the per-row length.)
//
// Every probe in this module is attached to a PUBLIC test. Private test
// failures show the learner a red bar and nothing else — task/TaskPage.jsx
// renders their name but never their message — so a diagnosis buried in one
// would be a diagnosis nobody ever reads.
function diagnoseGrid(out, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (let y = 0; y < expected.length; y++) {
        for (let x = 0; x < expected[y].length; x++) {
          const c = value(y, x);
          if (!(out[y] && Math.abs(out[y][x] - c) <= eps)) return false;
          if (Math.abs(expected[y][x] - c) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// ---- probe families -------------------------------------------------------

// Task 1. `lit` is a scalar per pixel, so these are computed once per grid and
// compared channel by channel after the palette is applied.
function litProbes(height, relief) {
  const size = height.length;
  const build = fn => {
    const out = [];
    for (let y = 0; y < size; y++) {
      const row = [];
      const yd = wrapLo(y - 1, size);
      const yu = wrapHi(y + 1, size);
      for (let x = 0; x < size; x++) {
        const xl = wrapLo(x - 1, size);
        const xr = wrapHi(x + 1, size);
        row.push(Math.max(0, fn(height, y, x, xl, xr, yd, yu)));
      }
      out.push(row);
    }
    return out;
  };
  const norm = (gx, gy) => 1 / Math.sqrt(gx * gx + gy * gy + 1);
  return [
    [
      build((h, y, x, xl, xr, yd, yu) => {
        const gx = (h[y][xr] - h[y][xl]) * relief;
        const gy = (h[yu][x] - h[yd][x]) * relief;
        return LIGHT * (1 + gx + gy) * norm(gx, gy);
      }),
      'the slope is twice what it should be — a central difference spans TWO cells, so it needs the × 0.5',
    ],
    [
      build((h, y, x, xl, xr, yd, yu) => {
        const gx = (h[y][xr] - h[y][xl]) * 0.5 * relief;
        const gy = (h[yu][x] - h[yd][x]) * 0.5 * relief;
        return LIGHT * (1 + gx + gy);
      }),
      'the normal was never normalised — divide by √(gx² + gy² + 1), or steep ground comes out arbitrarily bright',
    ],
    [
      build((h, y, x, xl, xr, yd, yu) => {
        const gx = (h[y][xr] - h[y][xl]) * 0.5 * relief;
        const gy = (h[yu][x] - h[yd][x]) * 0.5 * relief;
        return LIGHT * (1 - gx - gy) * norm(gx, gy);
      }),
      'the light is coming from the wrong side — it is 1 + gx + gy, so slopes that face up-left are the lit ones',
    ],
    [
      build((h, y, x, xl, xr, yd, yu) => {
        const gx = (h[y][xr] - h[y][x]) * 0.5 * relief;
        const gy = (h[yu][x] - h[y][x]) * 0.5 * relief;
        return LIGHT * (1 + gx + gy) * norm(gx, gy);
      }),
      'that is a forward difference — the central one straddles this cell: height[y][xr] − height[y][xl]',
    ],
  ];
}

// Task 2. Every probe is a whole grid, built from the same inputs.
function dropProbes(height, water) {
  const size = height.length;
  const build = fn => {
    const out = [];
    for (let y = 0; y < size; y++) {
      const row = [];
      for (let x = 0; x < size; x++) row.push(fn(y, x));
      out.push(row);
    }
    return out;
  };
  const rockOnly = build((y, x) => {
    let total = 0;
    for (const [ny, nx] of neighboursOf(size, y, x)) {
      const d = height[y][x] - height[ny][nx];
      if (d > 0) total += d * d;
    }
    return total;
  });
  const unsigned = build((y, x) => {
    let total = 0;
    for (const [ny, nx] of neighboursOf(size, y, x)) {
      const d = height[y][x] + water[y][x] - (height[ny][nx] + water[ny][nx]);
      total += d * d;
    }
    return total;
  });
  const unsquared = build((y, x) => {
    let total = 0;
    for (const [ny, nx] of neighboursOf(size, y, x)) {
      const d = height[y][x] + water[y][x] - (height[ny][nx] + water[ny][nx]);
      if (d > 0) total += d;
    }
    return total;
  });
  const uphill = build((y, x) => {
    let total = 0;
    for (const [ny, nx] of neighboursOf(size, y, x)) {
      const d = height[ny][nx] + water[ny][nx] - (height[y][x] + water[y][x]);
      if (d > 0) total += d * d;
    }
    return total;
  });
  return [
    [(y, x) => rockOnly[y][x], 'the water was left out of the surface — flow follows height + water, not the bare rock'],
    [(y, x) => unsigned[y][x], 'the uphill neighbours got counted too — squaring hides their sign, so the `if (d > 0)` has to do it'],
    [(y, x) => unsquared[y][x], 'the drops were added, not their squares — this pass totals d × d'],
    [(y, x) => uphill[y][x], 'the subtraction is the wrong way round — a drop is here MINUS the neighbour'],
  ];
}

// Task 3.
function waterProbes(height, water, drop, rain) {
  const size = height.length;
  const build = fn => {
    const out = [];
    for (let y = 0; y < size; y++) {
      const row = [];
      for (let x = 0; x < size; x++) row.push(fn(y, x));
      out.push(row);
    }
    return out;
  };
  const taken = (y, x, denom, weight) => {
    const here = height[y][x] + water[y][x];
    let sum = 0;
    for (const [ny, nx] of neighboursOf(size, y, x)) {
      const d = height[ny][nx] + water[ny][nx] - here;
      if (d > 0) sum += (weight(d) * water[ny][nx]) / denom(y, x, ny, nx);
    }
    return sum;
  };
  const own = (y, x) => drop[y][x] + P.soft;
  const theirs = (y, x, ny, nx) => drop[ny][nx] + P.soft;
  const given = (y, x) => (drop[y][x] * water[y][x]) / (drop[y][x] + P.soft);
  const sq = d => d * d;
  return [
    [
      (y, x) => water[y][x] + P.move * (taken(y, x, own, sq) - given(y, x)) + rain,
      "each share was divided by THIS cell's drop total — it belongs to the neighbour that is handing the water over: drop[ny][nx]",
    ],
    [
      (y, x) => water[y][x] + P.move * taken(y, x, theirs, sq) + rain,
      'water only ever arrives — the cell also has to hand its own share on, which is the `given` term',
    ],
    [
      (y, x) => water[y][x] + P.move * (taken(y, x, theirs, sq) - given(y, x)),
      'the rain never fell — every cell gains this.constants.rain on top of the routing',
    ],
    [
      (y, x) => water[y][x] + (taken(y, x, theirs, sq) - given(y, x)) + rain,
      'the move factor is missing — only this.constants.move of the water travels in one step',
    ],
    [
      (y, x) => water[y][x] + P.move * (taken(y, x, theirs, d => d) - given(y, x)) + rain,
      'the share weights are plain drops, not squared ones — they have to match the d × d the drop pass added up, or the shares do not sum to one',
    ],
  ];
}

// Task 4, terrain side.
function terrainProbes(height, water, sediment, drop) {
  const cap = (y, x) => capacityOf(drop[y][x], water[y][x]);
  const raw = (y, x) => {
    const c = cap(y, x);
    const load = sediment[y][x];
    return load < c ? P.pickUp * (c - load) : -P.settle * (load - c);
  };
  return [
    [
      (y, x) => height[y][x] + deltaOf(cap(y, x), sediment[y][x]),
      'the sign is inverted — the exchange is SUBTRACTED from the rock: what the water picks up, the ground loses',
    ],
    [
      (y, x) => height[y][x] - deltaOf(P.carry * drop[y][x] * water[y][x], sediment[y][x]),
      'capacity is using the drop total itself — the slope is its square root, because the total is a sum of squared drops',
    ],
    [
      (y, x) => height[y][x] - raw(y, x),
      'the maxCut clamp is missing — one step may not cut more than this.constants.maxCut out of a cell',
    ],
    [
      (y, x) => height[y][x] - deltaOf(P.carry * Math.sqrt(drop[y][x]), sediment[y][x]),
      'capacity forgot the water — a dry steep cell carries nothing, so it is carry × √drop × water',
    ],
  ];
}

// Task 4, sediment side. The routing half is given to the learner, so these
// only describe ways of getting the exchange wrong.
function sedimentProbes(height, water, sediment, drop) {
  const routed = routeRef(height, water, drop, sediment);
  const cap = (y, x) => capacityOf(drop[y][x], water[y][x]);
  const raw = (y, x) => {
    const c = cap(y, x);
    const load = sediment[y][x];
    return load < c ? P.pickUp * (c - load) : -P.settle * (load - c);
  };
  return [
    [(y, x) => routed[y][x], 'the exchange never reached the sediment — the same delta the rock loses is what the water gains'],
    [(y, x) => routed[y][x] - deltaOf(cap(y, x), sediment[y][x]), 'the exchange is subtracted here — the rock loses it, the water gains it, so the two kernels use opposite signs'],
    [(y, x) => routed[y][x] + raw(y, x), 'the maxCut clamp is missing — the two kernels have to compute the SAME delta, clamp included, or rock goes missing'],
  ];
}

export default {
  uuid: '07165ca1-0a5d-4c84-9b63-fdd3c9480661',
  version: 1,
  slug: 'hydraulic-erosion',
  title: 'Hydraulic Erosion: Carving Terrain by Accumulation',
  blurb: 'Rain on a fractal heightmap, one gather at a time — until the noise grows rivers.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'hillshade',
      title: 'Seeing the Ground',
      intro: `<p>A heightmap is a grid of numbers, and nobody can judge a landscape from numbers.
        Before we start moving rock around we need <strong>eyes</strong>: a picture in which a
        valley looks like a valley. The standard one is a <em>hillshade</em> — pretend the terrain
        is a solid surface, work out which way each cell faces, and see how squarely it faces
        the sun.</p>
        <p>Which way a cell faces is its slope, and slope on a grid is a
        <strong>central difference</strong>: the neighbour to the right minus the neighbour to the
        left, halved because that step spans two cells. (The same two-tap derivative the Sobel
        filters use — this is just the one place it gets to be lighting instead of edges.) With
        <code>gx</code> and <code>gy</code> in hand the surface normal is
        <code>(−gx, −gy, 1)</code>; normalise it, dot it with the light
        <code>ℓ = (−1, −1, 1)/√3</code>, and the whole thing collapses to</p>
<pre><code>lit = 0.5774 * (1 + gx + gy)
    / Math.sqrt(gx * gx + gy * gy + 1)</code></pre>
        <p>The grid wraps at the edges, the way Reaction–Diffusion's did: column 0's left
        neighbour is the last column. No special cases, no missing pixels.</p>`,
      goal: `<strong>Goal:</strong> finish the graphical kernel so it hillshades
        <code>terrain</code> — central differences for <code>gx</code> and <code>gy</code>, then
        the lit value above, clamped at zero.`,
      requirements: [
        '<code>gx</code> and <code>gy</code> are central differences × <code>0.5</code> × <code>this.constants.relief</code>',
        'Divide by <code>Math.sqrt(gx * gx + gy * gy + 1)</code> — an un-normalised normal is not a normal',
        'Clamp <code>lit</code> at <code>0</code> so ground facing away from the light goes black, not negative',
      ],
      hints: [
        {
          title: 'Hint 1 — the two slopes',
          body: `<p>The wrapped indexes are already computed for you:</p>
<pre><code>const gx = (height[y][xr] - height[y][xl])
  * 0.5 * this.constants.relief;
const gy = (height[yu][x] - height[yd][x])
  * 0.5 * this.constants.relief;</code></pre>
<p><code>relief</code> is only a vertical exaggeration — terrain this shallow would be almost
            invisible at a true aspect ratio.</p>`,
        },
        {
          title: 'Hint 2 — the rest of it',
          body: `<pre><code>const inv = 1 / Math.sqrt(gx * gx + gy * gy + 1);
let lit = this.constants.light * (1 + gx + gy) * inv;
if (lit &lt; 0) lit = 0;</code></pre>
<p>The palette line underneath is already written; it only wants <code>lit</code>.</p>`,
        },
      ],
      transfer: `Hillshading is a fragment shader that has escaped its renderer: one texture read
        per neighbour, one dot product, one write. Every terrain engine — WebGPU, Metal, Unreal's
        landscape — does exactly this, usually with the normals baked into a texture beside the
        heights so the fetch is one sample instead of four.`,
      starterCode: `// A heightmap you cannot see is a heightmap you cannot debug.
const gpu = new GPU({ mode });

const hillshade = gpu.createKernel(function (height) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;

  // TODO: central differences, scaled by this.constants.relief
  const gx = 0;
  const gy = 0;
  // TODO: normalise, dot with the light, clamp at zero
  let lit = 1;

  const rock = Math.min(1, height[y][x] * 1.15);
  this.color(lit * (0.3 + 0.62 * rock), lit * (0.44 + 0.4 * rock), lit * (0.54 + 0.2 * rock), 1);
}, {
  output: [64, 64],
  graphical: true,
  constants: { size: 64, relief: 12, light: 0.5774 },
});

await hillshade(terrain);
render(hillshade.canvas);
`,
      solutionCode: `// A heightmap you cannot see is a heightmap you cannot debug.
const gpu = new GPU({ mode });

const hillshade = gpu.createKernel(function (height) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;

  const gx = (height[y][xr] - height[y][xl]) * 0.5 * this.constants.relief;
  const gy = (height[yu][x] - height[yd][x]) * 0.5 * this.constants.relief;
  const inv = 1 / Math.sqrt(gx * gx + gy * gy + 1);
  let lit = this.constants.light * (1 + gx + gy) * inv;
  if (lit < 0) lit = 0;

  const rock = Math.min(1, height[y][x] * 1.15);
  this.color(lit * (0.3 + 0.62 * rock), lit * (0.44 + 0.4 * rock), lit * (0.54 + 0.2 * rock), 1);
}, {
  output: [64, 64],
  graphical: true,
  constants: { size: 64, relief: 12, light: 0.5774 },
});

await hillshade(terrain);
render(hillshade.canvas);
`,
      inputs: utils => ({ terrain: makeHills(utils, 64, 7) }),
      publicTests: [
        {
          name: 'a 64×64 canvas is rendered',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(ctx.canvas, 'no canvas — is the kernel graphical: true, and did you call render()?');
            ctx.assert(
              ctx.canvas.width === 64 && ctx.canvas.height === 64,
              `expected a 64×64 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            const pixels = ctx.getPixels();
            ctx.assert(pixels.length === 64 * 64 * 4, 'pixel buffer should hold 64×64 RGBA values');
          },
        },
        {
          name: 'flat ground is lit by exactly <code>0.5774</code>',
          run: async ctx => {
            // Both slopes vanish, so lit = light and the answer is the same in
            // every pixel — no row-order question to answer.
            const flat = makeGrid(64, 0.5);
            await ctx.kernel(flat);
            const pixels = ctx.getPixels();
            const expected = shadeColour(LIGHT, 0.5).map(v => v * 255);
            for (let i = 0; i < pixels.length; i += 331 * 4) {
              ctx.assertClose(pixels[i], expected[0], 3, `red at byte ${i} on flat ground`);
              ctx.assertClose(pixels[i + 1], expected[1], 3, `green at byte ${i} on flat ground`);
              ctx.assertClose(pixels[i + 2], expected[2], 3, `blue at byte ${i} on flat ground`);
            }
          },
        },
        {
          name: 'a ridge running north–south shades correctly, cell by cell',
          run: async ctx => {
            // Height varies with x only, so every row of the picture is
            // identical and the readback's row order cannot confuse the test.
            const ridge = makeGrid(64, 0);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ridge[y][x] = 0.5 + 0.35 * Math.sin((2 * Math.PI * x) / 64);
              }
            }
            await ctx.kernel(ridge);
            const pixels = ctx.getPixels();
            const lit = hillshadeRef(ridge, 12);
            // Shape the row as a 3 × 64 grid of channels so a probe has to
            // predict the WHOLE row before it is allowed to name a mistake:
            // at the crest of the ridge the slope is zero and every candidate
            // agrees, so one matching column proves nothing.
            const got = [0, 1, 2].map(c => Array.from({ length: 64 }, (_, x) => pixels[x * 4 + c] / 255));
            const want = [0, 1, 2].map(c =>
              Array.from({ length: 64 }, (_, x) => shadeColour(lit[0][x], ridge[0][x])[c])
            );
            const hint = diagnoseGrid(
              got,
              want,
              2.5 / 255,
              litProbes(ridge, 12).map(([grid, message]) => [
                (c, x) => shadeColour(grid[0][x], ridge[0][x])[c],
                message,
              ])
            );
            for (const x of [0, 5, 8, 16, 31, 48, 63]) {
              for (let c = 0; c < 3; c++) {
                ctx.assertClose(got[c][x], want[c][x], 2.5 / 255, hint || `channel ${c} at column ${x}`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The real terrain, compared over the whole picture. Row order is
            // unknown, so match the grid both ways up and demand that ONE of
            // them agrees everywhere.
            const terrain = makeHills(ctx.utils, 64, 7);
            await ctx.kernel(terrain);
            const pixels = ctx.getPixels();
            const lit = hillshadeRef(terrain, 12);
            const score = flip => {
              let worst = 0;
              for (let y = 0; y < 64; y += 3) {
                for (let x = 0; x < 64; x += 3) {
                  const row = flip ? 63 - y : y;
                  const at = (row * 64 + x) * 4;
                  const want = shadeColour(lit[y][x], terrain[y][x]);
                  for (let c = 0; c < 3; c++) {
                    worst = Math.max(worst, Math.abs(pixels[at + c] / 255 - want[c]));
                  }
                }
              }
              return worst;
            };
            const worst = Math.min(score(false), score(true));
            ctx.assert(
              worst <= 2.5 / 255,
              `the shaded picture differs from the reference hillshade by up to ${(worst * 255).toFixed(1)}/255`
            );
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A steep east-facing wall must not come out brighter than flat
            // ground: that only happens when the normal is not normalised.
            const wall = makeGrid(64, 0);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) wall[y][x] = 0.5 + 0.02 * Math.min(x, 63 - x);
            }
            await ctx.kernel(wall);
            const pixels = ctx.getPixels();
            const lit = hillshadeRef(wall, 12);
            let brightest = 0;
            for (let i = 0; i < pixels.length; i += 4) brightest = Math.max(brightest, pixels[i]);
            const expected = maxOf(lit.map((row, y) => row.map((v, x) => shadeColour(v, wall[y][x])[0])));
            ctx.assertClose(
              brightest / 255,
              expected,
              3 / 255,
              'the brightest pixel of a slope is off — an un-normalised normal makes steep ground arbitrarily bright'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'downhill',
      title: 'Which Way Is Down',
      intro: `<p>Rain has landed. Where does it go? Downhill — but "downhill" on a grid needs
        spelling out, and two details decide whether this simulation ever grows a river.</p>
        <p><strong>One: the surface is rock plus water.</strong> A puddle sitting in a hollow
        raises the level the next drop has to climb; keep filling it and the puddle spills over
        the rim by itself. Route on <code>height</code> alone and hollows become bottomless
        traps. So every comparison below is between <code>height + water</code> values — call
        that <code>H</code>.</p>
        <p><strong>Two: square the drops.</strong> This pass adds up
        <code>d²</code> over the neighbours that are <em>below</em> this cell, and the next task
        will hand each of them <code>d²/total</code> of the water. Weighting by the plain drop
        smears the flow across every neighbour that is even slightly lower and no channel ever
        forms; squaring makes a cell commit to the steep side. That one exponent is the
        difference between a damp hillside and a river.</p>`,
      goal: `<strong>Goal:</strong> complete <code>dropTotal</code> so each cell returns the sum
        of <code>d²</code> over its four neighbours, counting only the ones the water can
        actually fall to.`,
      requirements: [
        'Build the surface as <code>height[y][x] + water[y][x]</code>, not <code>height</code> alone',
        'For each neighbour, <code>d = here − there</code>; add <code>d * d</code> only when <code>d &gt; 0</code>',
        'All four neighbours, with the same wrap-around the starter already set up',
      ],
      hints: [
        {
          title: 'Hint 1 — one neighbour at a time',
          body: `<p>The starter writes the left neighbour out in full. The other three are the
            same two lines with different indexes:</p>
<pre><code>d = here - (height[y][xr] + water[y][xr]);
if (d &gt; 0) total += d * d;</code></pre>`,
        },
        {
          title: 'Hint 2 — why the if matters',
          body: `<p><code>d * d</code> is positive whether the neighbour is above or below, so the
            square destroys the very information you are selecting on. The <code>if (d &gt; 0)</code>
            has to come first, or an uphill neighbour ends up "receiving" water.</p>`,
        },
      ],
      transfer: `Deciding a flow direction per cell and then normalising by a per-cell total is
        the multiple-flow-direction (MFD) router every terrain and hydrology package ships —
        GRASS, TauDEM, Houdini's erosion nodes. The exponent this task fixes at 2 is a tuning
        knob in all of them; 1.1 gives braided sheets, ∞ gives single-thread rivers.`,
      starterCode: `// How much "downhill" does each cell have to give away?
const gpu = new GPU({ mode });

const dropTotal = gpu.createKernel(function (height, water) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;

  const here = height[y][x] + water[y][x];
  let total = 0;

  let d = here - (height[y][xl] + water[y][xl]);
  if (d > 0) total += d * d;
  // TODO: the other three neighbours — xr, yd, yu

  return total;
}, { output: [64, 64], constants: { size: 64 } });

const drop = await dropTotal(terrain, water);
console.log('steepest cell has a drop total of', Math.max(...drop.map(row => Math.max(...row))));
`,
      solutionCode: `// How much "downhill" does each cell have to give away?
const gpu = new GPU({ mode });

const dropTotal = gpu.createKernel(function (height, water) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;

  const here = height[y][x] + water[y][x];
  let total = 0;

  let d = here - (height[y][xl] + water[y][xl]);
  if (d > 0) total += d * d;
  d = here - (height[y][xr] + water[y][xr]);
  if (d > 0) total += d * d;
  d = here - (height[yd][x] + water[yd][x]);
  if (d > 0) total += d * d;
  d = here - (height[yu][x] + water[yu][x]);
  if (d > 0) total += d * d;

  return total;
}, { output: [64, 64], constants: { size: 64 } });

const drop = await dropTotal(terrain, water);
console.log('steepest cell has a drop total of', Math.max(...drop.map(row => Math.max(...row))));
`,
      inputs: utils => {
        const state = warmUp(utils, 64, 7, 30);
        return { terrain: state.height, water: state.water };
      },
      publicTests: [
        {
          name: 'flat ground has nothing to give: every cell returns <code>0</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const flat = makeGrid(64, 0.4);
            const wet = makeGrid(64, 0.02);
            const out = await ctx.kernel(flat, wet);
            ctx.assert(out && out.length === 64 && out[0].length === 64, 'expected a 64×64 result');
            for (let y = 0; y < 64; y += 5) {
              for (let x = 0; x < 64; x += 5) {
                ctx.assertClose(out[y][x], 0, 1e-6, `cell [${y}][${x}] of a flat surface`);
              }
            }
          },
        },
        {
          name: 'a single peak drops to all four neighbours, even across the wrap',
          run: async ctx => {
            const height = makeGrid(64, 0);
            height[0][0] = 0.5;
            const dry = makeGrid(64, 0);
            const out = await ctx.kernel(height, dry);
            ctx.assertClose(out[0][0], 4 * 0.25, 1e-6, 'the peak itself — four drops of 0.5, squared');
            ctx.assertClose(out[0][1], 0, 1e-6, 'a neighbour of the peak has nowhere lower to go');
            ctx.assertClose(out[0][63], 0, 1e-6, 'the LEFT neighbour, which wraps to column 63');
            ctx.assertClose(out[63][0], 0, 1e-6, 'the neighbour below, which wraps to row 63');
            ctx.assertClose(out[8][8], 0, 1e-6, 'a far-away cell on flat ground');
          },
        },
        {
          name: 'the water counts: a puddle changes where the surface points',
          run: async ctx => {
            // Rock is a uniform ramp; the water is what makes cell [4][4] a
            // high point. A kernel that reads `height` alone cannot see it.
            const height = makeGrid(64, 0.3);
            const wet = makeGrid(64, 0.01);
            wet[4][4] = 0.09;
            const out = await ctx.kernel(height, wet);
            const expected = dropTotalRef(height, wet);
            const hint = diagnoseGrid(out, expected, 1e-6, dropProbes(height, wet));
            ctx.assertClose(out[4][4], expected[4][4], 1e-6, hint || 'the puddled cell [4][4]');
            ctx.assertClose(out[4][5], expected[4][5], 1e-6, hint || 'the cell next to the puddle');
            ctx.assertClose(out[20][20], expected[20][20], 1e-6, hint || 'a cell far from the puddle');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A real rained-on landscape, every cell checked.
            const state = warmUp(ctx.utils, 64, 4242, 40);
            const out = await ctx.kernel(state.height, state.water);
            const expected = dropTotalRef(state.height, state.water);
            const hint = diagnoseGrid(out, expected, 5e-6, dropProbes(state.height, state.water));
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 5e-6, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Nothing may be negative: a drop total is a sum of squares.
            const state = warmUp(ctx.utils, 64, 909, 25);
            const out = await ctx.kernel(state.height, state.water);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assert(
                  out[y][x] >= -1e-9,
                  `cell [${y}][${x}] came back negative (${out[y][x]}) — a total of squares cannot be`
                );
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'route-the-water',
      title: 'Everybody Downhill At Once',
      intro: `<p>Now move the water. On a CPU you would walk each cell and <em>push</em> its water
        into the neighbours below it. You cannot do that here: two cells pushing into the same
        neighbour is a write collision, and a kernel writes exactly one output cell — the whole
        point Thinking in Parallel makes. So invert it. Each cell works out what its uphill
        neighbours would have <strong>sent</strong> it, and adds that up.</p>
        <p>Neighbour <code>n</code> hands this cell the share <code>d² / drop[n]</code> of its
        water. And there is the catch that shapes the whole module: <code>drop[n]</code> is an
        aggregate over <em>n</em>'s neighbours, which are two rings away from us. A gather cannot
        ask a neighbour to add something up on demand — so the previous task exists purely to
        have that total already sitting in a grid we can read. <strong>When a gather needs a
        neighbour's aggregate, the aggregate gets its own pass.</strong></p>
        <p>The cell also hands its own water on, <code>move</code> of it per step, and keeps
        <code>1 − move·drop/(drop + soft)</code>. That keep-weight is the centre weight of an
        explicit diffusion step wearing a different hat, and it obeys the same rule
        The Heat Equation &amp; Stability derives: let it go negative — here, take
        <code>move</code> past <code>1</code> — and the grid detonates. The
        <code>+ soft</code> keeps the division safe where a cell has nowhere to go, and quiets
        flat water, which would otherwise flip back and forth between neighbours forever.</p>`,
      goal: `<strong>Goal:</strong> complete <code>moveWater</code> — gather each uphill
        neighbour's share, subtract what this cell hands on, and add the rain.`,
      requirements: [
        'For each neighbour above this cell, add <code>d * d * water[n] / (drop[n] + soft)</code> — the neighbour\'s drop total, not this cell\'s',
        'Subtract <code>given = drop[here] * water[here] / (drop[here] + soft)</code>',
        'Return <code>water + move * (taken − given) + rain</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — whose total is it?',
          body: `<p>The share is <em>the neighbour's</em>: it is dividing up <em>the neighbour's</em>
            water, so the denominator has to be the neighbour's drop total.</p>
<pre><code>d = height[y][xl] + water[y][xl] - here;
if (d &gt; 0) taken += (d * d * water[y][xl]) / (drop[y][xl] + this.constants.soft);</code></pre>
<p>Divide by <code>drop[y][x]</code> instead and the shares stop summing to one — water
            appears and vanishes.</p>`,
        },
        {
          title: 'Hint 2 — the outgoing half',
          body: `<p>Sum the shares this cell gives away and you get <code>drop[y][x]</code> back on
            top, so the whole outgoing side is one line:</p>
<pre><code>const given = (drop[y][x] * water[y][x]) / (drop[y][x] + this.constants.soft);
return water[y][x] + this.constants.move * (taken - given) + this.constants.rain;</code></pre>`,
        },
        {
          title: 'Hint 3 — the test that catches everything',
          body: `<p>Water is conserved: whatever leaves one cell arrives in another, so after one
            step the grid holds exactly what it held plus one step of rain. If your total drifts,
            the two halves are not using the same shares.</p>`,
        },
      ],
      transfer: `Gather-instead-of-scatter with a pre-computed normaliser is the shape of sparse
        matrix–vector multiply, of graph message passing, and of every particle-to-grid transfer
        written for a GPU: nobody writes to a neighbour, everybody reads from one, and any
        per-source total the readers need is materialised by an earlier pass.`,
      starterCode: `// Nobody pushes. Everybody pulls.
const gpu = new GPU({ mode });

const dropTotal = gpu.createKernel(function (height, water) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let total = 0;
  let d = here - (height[y][xl] + water[y][xl]); if (d > 0) total += d * d;
  d = here - (height[y][xr] + water[y][xr]); if (d > 0) total += d * d;
  d = here - (height[yd][x] + water[yd][x]); if (d > 0) total += d * d;
  d = here - (height[yu][x] + water[yu][x]); if (d > 0) total += d * d;
  return total;
}, { output: [64, 64], constants: { size: 64 } });

const moveWater = gpu.createKernel(function (height, water, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];

  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * water[y][xl]) / (drop[y][xl] + this.constants.soft);
  // TODO: the other three neighbours — xr, yd, yu

  // TODO: how much does this cell hand on? and don't forget the rain.
  return water[y][x] + this.constants.move * taken;
}, {
  output: [64, 64],
  constants: { size: 64, move: 0.6, soft: 0.0002, rain: 0.00002 },
});

const drop = await dropTotal(terrain, water);
const next = await moveWater(terrain, water, drop);

let before = 0;
let after = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) { before += water[y][x]; after += next[y][x]; }
}
console.log('water before:', before, ' after:', after, ' rain added:', 64 * 64 * 0.00002);
`,
      solutionCode: `// Nobody pushes. Everybody pulls.
const gpu = new GPU({ mode });

const dropTotal = gpu.createKernel(function (height, water) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let total = 0;
  let d = here - (height[y][xl] + water[y][xl]); if (d > 0) total += d * d;
  d = here - (height[y][xr] + water[y][xr]); if (d > 0) total += d * d;
  d = here - (height[yd][x] + water[yd][x]); if (d > 0) total += d * d;
  d = here - (height[yu][x] + water[yu][x]); if (d > 0) total += d * d;
  return total;
}, { output: [64, 64], constants: { size: 64 } });

const moveWater = gpu.createKernel(function (height, water, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];

  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * water[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * water[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * water[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * water[yu][x]) / (drop[yu][x] + this.constants.soft);

  const given = (drop[y][x] * water[y][x]) / (drop[y][x] + this.constants.soft);
  return water[y][x] + this.constants.move * (taken - given) + this.constants.rain;
}, {
  output: [64, 64],
  constants: { size: 64, move: 0.6, soft: 0.0002, rain: 0.00002 },
});

const drop = await dropTotal(terrain, water);
const next = await moveWater(terrain, water, drop);

let before = 0;
let after = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) { before += water[y][x]; after += next[y][x]; }
}
console.log('water before:', before, ' after:', after, ' rain added:', 64 * 64 * 0.00002);
`,
      inputs: utils => {
        const state = warmUp(utils, 64, 7, 30);
        return { terrain: state.height, water: state.water };
      },
      publicTests: [
        {
          name: 'water is conserved — the grid gains exactly one step of rain',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels (dropTotal then moveWater), found ${ctx.kernels.length}`);
            const state = warmUp(ctx.utils, 64, 7, 30);
            const drop = await ctx.kernels[0](state.height, state.water);
            const next = await ctx.kernels[1](state.height, state.water, drop);
            const before = totalOf(state.water);
            const after = totalOf(next);
            ctx.assertClose(
              after - before,
              64 * 64 * P.rain,
              2e-4,
              'the grid did not gain exactly one step of rain — if it gained less, water is leaking out of the shares; ' +
                'if it gained none at all, the rain term is missing'
            );
          },
        },
        {
          name: 'on flat ground nothing moves but the rain',
          run: async ctx => {
            const flat = makeGrid(64, 0.4);
            const wet = makeGrid(64, 0.03);
            const drop = await ctx.kernels[0](flat, wet);
            const next = await ctx.kernels[1](flat, wet, drop);
            for (let y = 0; y < 64; y += 7) {
              for (let x = 0; x < 64; x += 7) {
                ctx.assertClose(
                  next[y][x],
                  0.03 + P.rain,
                  1e-6,
                  `cell [${y}][${x}] on a flat lake — with no drop anywhere, every cell keeps what it has and gains the rain`
                );
              }
            }
          },
        },
        {
          name: 'a rained-on landscape, cell for cell',
          run: async ctx => {
            const state = warmUp(ctx.utils, 64, 7, 30);
            const drop = await ctx.kernels[0](state.height, state.water);
            const next = await ctx.kernels[1](state.height, state.water, drop);
            const ref = dropTotalRef(state.height, state.water);
            const expected = moveWaterRef(state.height, state.water, ref);
            const hint = diagnoseGrid(
              next, expected, 5e-6,
              waterProbes(state.height, state.water, ref, P.rain)
            );
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(next[y][x], expected[y][x], 5e-6, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A hand-checkable configuration on a different landscape: [8][8]
            // sits above two of its neighbours by different amounts, so the
            // d²-weighted shares are unequal and have to add to one.
            const height = makeGrid(64, 0.5);
            height[8][8] = 0.7;
            height[8][9] = 0.4; // drop of 0.3
            height[9][8] = 0.6; // drop of 0.1
            const wet = makeGrid(64, 0.02);
            const drop = await ctx.kernels[0](height, wet);
            const next = await ctx.kernels[1](height, wet, drop);
            const ref = dropTotalRef(height, wet);
            const expected = moveWaterRef(height, wet, ref);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(next[y][x], expected[y][x], 5e-6, `cell [${y}][${x}]`);
              }
            }
            // and again on a fresh landscape the public tests never touch
            const state = warmUp(ctx.utils, 64, 31337, 35);
            const drop2 = await ctx.kernels[0](state.height, state.water);
            const next2 = await ctx.kernels[1](state.height, state.water, drop2);
            const want2 = moveWaterRef(state.height, state.water, dropTotalRef(state.height, state.water));
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(next2[y][x], want2[y][x], 5e-6, `cell [${y}][${x}] of a second landscape`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Nothing goes negative, and nothing runs away: with move ≤ 1 a
            // cell can never hand on more water than it holds.
            const state = warmUp(ctx.utils, 64, 5150, 20);
            const drop = await ctx.kernels[0](state.height, state.water);
            const next = await ctx.kernels[1](state.height, state.water, drop);
            const ceiling = maxOf(state.water) * 4 + 1e-3;
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assert(
                  next[y][x] >= -1e-9,
                  `cell [${y}][${x}] has negative water (${next[y][x]}) — a cell handed on more than it had`
                );
                ctx.assert(
                  next[y][x] <= ceiling,
                  `cell [${y}][${x}] holds ${next[y][x]}, far more water than existed anywhere before the step`
                );
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'pick-up-and-drop',
      title: 'Capacity: What the Water Can Carry',
      intro: `<p>Water alone does nothing to rock. What carves is <strong>load</strong>: fast,
        deep water can hold sediment, slow shallow water cannot, and the ground pays the
        difference. That is one number per cell:</p>
<pre><code>capacity = carry * Math.sqrt(drop) * water</code></pre>
        <p>√drop is the size of the downhill gradient — the drop total was a sum of squares, so
        its square root is the slope — and multiplying by depth is the classic
        <em>stream power</em> product: steep × wet. Compare it with what the cell is already
        carrying and you get one signed exchange:</p>
<pre><code>delta = load &lt; capacity
  ? Math.min(pickUp * (capacity - load), maxCut)  // cut
  : -settle * (load - capacity)                   // fill</code></pre>
        <p>Then <code>height − delta</code> and <code>sediment + delta</code>. That
        <code>maxCut</code> ceiling is the same kind of guard as <code>move ≤ 1</code> next door:
        cut a cell too deep in one step and its neighbours are suddenly steeper, so they cut
        harder, and the feedback runs away.</p>
        <p>Two kernels need the same <code>delta</code> and there is no way to hand a value from
        one to the other — so both recompute it from the same snapshot. That is not waste. On a
        GPU, arithmetic is close to free next to a round trip through memory; recomputing beats
        communicating almost every time.</p>`,
      goal: `<strong>Goal:</strong> write the same exchange in both kernels —
        <code>moveSediment</code> adds it to the water's load, <code>erode</code> takes it out of
        the rock. The routing half of <code>moveSediment</code> is already there.`,
      requirements: [
        '<code>capacity = carry * Math.sqrt(drop[y][x]) * water[y][x]</code> in both kernels',
        'Under-loaded water cuts <code>Math.min(pickUp * (capacity − load), maxCut)</code>; over-loaded water settles <code>−settle * (load − capacity)</code>',
        '<code>erode</code> returns <code>height − delta</code>; <code>moveSediment</code> returns the routed sediment <code>+ delta</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the exchange, spelled out',
          body: `<pre><code>const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
const load = sediment[y][x];
let delta = -this.constants.settle * (load - capacity);
if (load &lt; capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);</code></pre>
<p>Paste the identical four lines into both kernels.</p>`,
        },
        {
          title: 'Hint 2 — opposite signs',
          body: `<p><code>erode</code> ends <code>return height[y][x] - delta;</code> and
            <code>moveSediment</code> ends
            <code>return sediment[y][x] + this.constants.move * (taken - given) + delta;</code>.
            Rock loses exactly what the stream gains, which is what the conservation test checks.</p>`,
        },
      ],
      transfer: `"Recompute rather than communicate" is a GPU reflex, not a gpu.js workaround:
        CUDA kernels routinely redo an index calculation in every thread instead of staging it in
        shared memory, and shader authors re-derive a normal per fragment rather than interpolate
        one. Arithmetic units idle while memory is the bottleneck.`,
      starterCode: `// The rock pays for what the water carries.
const gpu = new GPU({ mode });

const moveSediment = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];

  // Sediment rides the water: same shares, same gather, one letter different.
  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * sediment[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * sediment[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * sediment[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * sediment[yu][x]) / (drop[yu][x] + this.constants.soft);
  const given = (drop[y][x] * sediment[y][x]) / (drop[y][x] + this.constants.soft);

  // TODO: capacity, load, delta — then add delta to the routed sediment.
  return sediment[y][x] + this.constants.move * (taken - given);
}, {
  output: [64, 64],
  constants: { size: 64, move: 0.6, soft: 0.0002, carry: 60, pickUp: 0.3, settle: 0.05, maxCut: 0.004 },
});

const erode = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO: the SAME capacity / load / delta as above — then take it out of the rock.
  return height[y][x];
}, {
  output: [64, 64],
  constants: { carry: 60, pickUp: 0.3, settle: 0.05, maxCut: 0.004 },
});

const nextSediment = await moveSediment(terrain, water, sediment, drop);
const nextHeight = await erode(terrain, water, sediment, drop);

let before = 0;
let after = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) {
    before += terrain[y][x] + sediment[y][x];
    after += nextHeight[y][x] + nextSediment[y][x];
  }
}
console.log('rock + sediment before:', before, 'after:', after);
console.log('deepest cut this step:', Math.min(...nextHeight.map((row, y) => Math.min(...row.map((v, x) => v - terrain[y][x])))));
`,
      solutionCode: `// The rock pays for what the water carries.
const gpu = new GPU({ mode });

const moveSediment = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];

  // Sediment rides the water: same shares, same gather, one letter different.
  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * sediment[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * sediment[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * sediment[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * sediment[yu][x]) / (drop[yu][x] + this.constants.soft);
  const given = (drop[y][x] * sediment[y][x]) / (drop[y][x] + this.constants.soft);

  const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
  const load = sediment[y][x];
  let delta = -this.constants.settle * (load - capacity);
  if (load < capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);

  return sediment[y][x] + this.constants.move * (taken - given) + delta;
}, {
  output: [64, 64],
  constants: { size: 64, move: 0.6, soft: 0.0002, carry: 60, pickUp: 0.3, settle: 0.05, maxCut: 0.004 },
});

const erode = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
  const load = sediment[y][x];
  let delta = -this.constants.settle * (load - capacity);
  if (load < capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);

  return height[y][x] - delta;
}, {
  output: [64, 64],
  constants: { carry: 60, pickUp: 0.3, settle: 0.05, maxCut: 0.004 },
});

const nextSediment = await moveSediment(terrain, water, sediment, drop);
const nextHeight = await erode(terrain, water, sediment, drop);

let before = 0;
let after = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) {
    before += terrain[y][x] + sediment[y][x];
    after += nextHeight[y][x] + nextSediment[y][x];
  }
}
console.log('rock + sediment before:', before, 'after:', after);
console.log('deepest cut this step:', Math.min(...nextHeight.map((row, y) => Math.min(...row.map((v, x) => v - terrain[y][x])))));
`,
      inputs: utils => {
        const state = warmUp(utils, 64, 7, 30);
        return {
          terrain: state.height,
          water: state.water,
          sediment: state.sediment,
          drop: dropTotalRef(state.height, state.water).map(row => row.map(v => Math.round(v * 1e9) / 1e9)),
        };
      },
      publicTests: [
        {
          name: 'nothing is created: rock plus sediment is unchanged',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels (moveSediment then erode), found ${ctx.kernels.length}`);
            const state = warmUp(ctx.utils, 64, 7, 30);
            const drop = dropTotalRef(state.height, state.water);
            const nextSediment = await ctx.kernels[0](state.height, state.water, state.sediment, drop);
            const nextHeight = await ctx.kernels[1](state.height, state.water, state.sediment, drop);
            const before = totalOf(state.height) + totalOf(state.sediment);
            const after = totalOf(nextHeight) + totalOf(nextSediment);
            ctx.assertClose(
              after,
              before,
              5e-3,
              'rock + sediment changed — the two kernels are not exchanging the same delta ' +
                '(the rock must lose exactly what the stream gains)'
            );
          },
        },
        {
          name: 'dry ground is left alone; over-loaded water drops its cargo',
          run: async ctx => {
            // No water anywhere → capacity 0 → every cell settles what it
            // carries, and every gram of it lands on the rock.
            const height = makeGrid(64, 0.5);
            const dry = makeGrid(64, 0);
            const load = makeGrid(64, 0.02);
            const drop = dropTotalRef(height, dry);
            const nextSediment = await ctx.kernels[0](height, dry, load, drop);
            const nextHeight = await ctx.kernels[1](height, dry, load, drop);
            const wantHeight = erodeRef(height, dry, load, drop);
            const wantSediment = moveSedimentRef(height, dry, load, drop);
            const hHint = diagnoseGrid(nextHeight, wantHeight, 1e-6, terrainProbes(height, dry, load, drop));
            const sHint = diagnoseGrid(nextSediment, wantSediment, 1e-6, sedimentProbes(height, dry, load, drop));
            for (let y = 0; y < 64; y += 9) {
              for (let x = 0; x < 64; x += 9) {
                ctx.assertClose(
                  nextHeight[y][x],
                  0.5 + P.settle * 0.02,
                  1e-6,
                  hHint || `rock at [${y}][${x}] — with no water the capacity is zero, so settle × load lands on it`
                );
                ctx.assertClose(
                  nextSediment[y][x],
                  0.02 - P.settle * 0.02,
                  1e-6,
                  sHint || `load at [${y}][${x}] — the stream should be lighter by exactly what the rock gained`
                );
              }
            }
          },
        },
        {
          name: 'the <code>maxCut</code> ceiling holds on a steep, wet, empty cell',
          run: async ctx => {
            // capacity here is enormous, so the un-clamped bite would be far
            // more than maxCut; the clamp is the only thing standing in the way.
            const height = makeGrid(64, 0);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) height[y][x] = 0.4;
            }
            height[16][16] = 1;
            const wet = makeGrid(64, 0.05);
            const empty = makeGrid(64, 0);
            const drop = dropTotalRef(height, wet);
            const nextHeight = await ctx.kernels[1](height, wet, empty, drop);
            const raw = P.pickUp * capacityOf(drop[16][16], wet[16][16]);
            ctx.assert(raw > P.maxCut * 3, 'test setup: the un-clamped bite should be much bigger than maxCut');
            ctx.assertClose(
              nextHeight[16][16],
              1 - P.maxCut,
              1e-6,
              `the peak should lose exactly maxCut (${P.maxCut}); un-clamped it would lose ${raw.toFixed(4)}`
            );
          },
        },
        {
          name: 'a rained-on landscape, both kernels, cell for cell',
          run: async ctx => {
            const state = warmUp(ctx.utils, 64, 7, 30);
            const drop = dropTotalRef(state.height, state.water);
            const nextSediment = await ctx.kernels[0](state.height, state.water, state.sediment, drop);
            const nextHeight = await ctx.kernels[1](state.height, state.water, state.sediment, drop);
            const wantSediment = moveSedimentRef(state.height, state.water, state.sediment, drop);
            const wantHeight = erodeRef(state.height, state.water, state.sediment, drop);
            const sHint = diagnoseGrid(
              nextSediment, wantSediment, 5e-6,
              sedimentProbes(state.height, state.water, state.sediment, drop)
            );
            const hHint = diagnoseGrid(
              nextHeight, wantHeight, 5e-6,
              terrainProbes(state.height, state.water, state.sediment, drop)
            );
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(nextHeight[y][x], wantHeight[y][x], 5e-6, hHint || `height at [${y}][${x}]`);
                ctx.assertClose(nextSediment[y][x], wantSediment[y][x], 5e-6, sHint || `sediment at [${y}][${x}]`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Whole-grid comparison on a landscape the public tests never see.
            const state = warmUp(ctx.utils, 64, 8899, 45);
            const drop = dropTotalRef(state.height, state.water);
            const nextSediment = await ctx.kernels[0](state.height, state.water, state.sediment, drop);
            const nextHeight = await ctx.kernels[1](state.height, state.water, state.sediment, drop);
            const wantSediment = moveSedimentRef(state.height, state.water, state.sediment, drop);
            const wantHeight = erodeRef(state.height, state.water, state.sediment, drop);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(nextSediment[y][x], wantSediment[y][x], 5e-6, `sediment at [${y}][${x}]`);
                ctx.assertClose(nextHeight[y][x], wantHeight[y][x], 5e-6, `height at [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Cell by cell, the rock loses precisely what the stream's
            // exchange gained — independent of the routing.
            const state = warmUp(ctx.utils, 64, 246, 30);
            const drop = dropTotalRef(state.height, state.water);
            const nextSediment = await ctx.kernels[0](state.height, state.water, state.sediment, drop);
            const nextHeight = await ctx.kernels[1](state.height, state.water, state.sediment, drop);
            const routed = routeRef(state.height, state.water, drop, state.sediment);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const gained = nextSediment[y][x] - routed[y][x];
                const lost = state.height[y][x] - nextHeight[y][x];
                ctx.assertClose(
                  gained,
                  lost,
                  5e-6,
                  `at [${y}][${x}] the stream gained ${gained.toFixed(6)} while the rock lost ${lost.toFixed(6)} — ` +
                    'both kernels must compute the identical delta, clamp included'
                );
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'one-hundred-steps',
      title: 'Ping-Pong the Whole Landscape',
      intro: `<p>Four kernels, three fields, one hundred steps. The loop lives in JavaScript, the
        grids live on the GPU, and the discipline is exactly the one Reaction–Diffusion and
        Cellular Automata already drilled — with one extra wrinkle worth naming.</p>
        <p>Each step is <em>ordered</em>: <code>dropTotal</code> runs first, because the other
        three read the grid it produces. Then <code>moveWater</code>, <code>moveSediment</code>
        and <code>erode</code> all read the <strong>same snapshot</strong> of height, water and
        sediment — plus that fresh drop grid — and only when all three have returned do you swap.
        Overwrite <code>height</code> early and <code>moveWater</code> starts routing over
        terrain that has already been cut, which is a different simulation and a worse one.</p>
        <p>Nothing here is pipelined: each kernel hands JavaScript an ordinary array and gets one
        back. That works, and it costs a round trip per pass — Pipelines &amp; Textures shows how
        to keep the whole thing resident on the card with <code>pipeline: true</code> and
        <code>immutable: true</code>, which is the production answer once the grid stops being
        64×64.</p>`,
      goal: `<strong>Goal:</strong> run <code>STEPS</code> steps of the whole model, recording the
        sediment in flight and the deepest water each step, then plot both.`,
      requirements: [
        'Call <code>dropTotal</code> first each step, then the other three with the <em>same</em> height / water / sediment',
        'Swap all three fields only after all four kernels have returned',
        'Push <code>totalOf(sediment)</code> and <code>maxOf(water)</code> onto the traces each step and <code>plot</code> them',
      ],
      hints: [
        {
          title: 'Hint 1 — the loop body',
          body: `<pre><code>const drop = await dropTotal(height, water);
const nextWater = await moveWater(height, water, drop);
const nextSediment = await moveSediment(height, water, sediment, drop);
const nextHeight = await erode(height, water, sediment, drop);
height = nextHeight;
water = nextWater;
sediment = nextSediment;</code></pre>
<p>Four <code>await</code>s in order. Never <code>Promise.all</code> them: three of the
            four need the first one's answer.</p>`,
        },
        {
          title: 'Hint 2 — the traces',
          body: `<p><code>totalOf</code> and <code>maxOf</code> are written for you. After the
            swap:</p>
<pre><code>carried.push(totalOf(sediment));
deepest.push(maxOf(water));</code></pre>
<p>and after the loop,
            <code>plot({ 'sediment in flight': carried }, { title: '…' })</code>.</p>`,
        },
        {
          title: 'Hint 3 — what the curves should look like',
          body: `<p>Sediment climbs, bends over around step 60 as pick-up and settling come into
            balance — and then, around step 80, kicks upward again: that second climb is the two
            hollows turning into lakes, and it shows up in the other plot at the same moment as a
            step in the deepest water. From there the deepest cell keeps climbing, with a
            sawtooth on it, because a lake surface trades a little water back and forth with its
            rim every step. Drag the rainfall slider and watch both curves scale.</p>`,
        },
      ],
      transfer: `Buying quality with time instead of work per frame is the same trade a
        progressive path tracer makes, and the same one a physically based fluid sim makes: the
        per-step kernel is cheap and stupid, and the result comes from running it enough times.
        Every one of them ping-pongs two sets of buffers, on every API there is.`,
      starterCode: `// Rain, flow, cut, settle. A hundred times.
const gpu = new GPU({ mode });

const SIZE = 64;
const STEPS = 100;
// Rain per step, in units of 10⁻⁵ of depth. Drag it and the whole run redoes itself.
const rainfall = slider('rain', { min: 0.5, max: 6, value: 2, step: 0.5, label: 'rain per step (×10⁻⁵)' });
const SHARED = { size: SIZE, move: 0.6, soft: 0.0002, carry: 60, pickUp: 0.3, settle: 0.05, maxCut: 0.004 };

const dropTotal = gpu.createKernel(function (height, water) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let total = 0;
  let d = here - (height[y][xl] + water[y][xl]); if (d > 0) total += d * d;
  d = here - (height[y][xr] + water[y][xr]); if (d > 0) total += d * d;
  d = here - (height[yd][x] + water[yd][x]); if (d > 0) total += d * d;
  d = here - (height[yu][x] + water[yu][x]); if (d > 0) total += d * d;
  return total;
}, { output: [SIZE, SIZE], constants: SHARED });

const moveWater = gpu.createKernel(function (height, water, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * water[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * water[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * water[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * water[yu][x]) / (drop[yu][x] + this.constants.soft);
  const given = (drop[y][x] * water[y][x]) / (drop[y][x] + this.constants.soft);
  return water[y][x] + this.constants.move * (taken - given) + this.constants.rain;
}, { output: [SIZE, SIZE], constants: { ...SHARED, rain: rainfall * 0.00001 } });

const moveSediment = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * sediment[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * sediment[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * sediment[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * sediment[yu][x]) / (drop[yu][x] + this.constants.soft);
  const given = (drop[y][x] * sediment[y][x]) / (drop[y][x] + this.constants.soft);
  const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
  const load = sediment[y][x];
  let delta = -this.constants.settle * (load - capacity);
  if (load < capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);
  return sediment[y][x] + this.constants.move * (taken - given) + delta;
}, { output: [SIZE, SIZE], constants: SHARED });

const erode = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
  const load = sediment[y][x];
  let delta = -this.constants.settle * (load - capacity);
  if (load < capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);
  return height[y][x] - delta;
}, { output: [SIZE, SIZE], constants: SHARED });

function blank() {
  const grid = [];
  for (let y = 0; y < SIZE; y++) grid.push(new Array(SIZE).fill(0));
  return grid;
}
function totalOf(grid) {
  let sum = 0;
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) sum += grid[y][x];
  return sum;
}
function maxOf(grid) {
  let top = 0;
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) if (grid[y][x] > top) top = grid[y][x];
  return top;
}

let height = terrain;
let water = blank();
let sediment = blank();
const carried = [];
const deepest = [];

// TODO: run STEPS steps. Each one is dropTotal FIRST, then the other three
// reading the same height / water / sediment, then a swap of all three.
// Push totalOf(sediment) and maxOf(water) after every swap.

plot({ 'sediment in flight': carried }, { title: 'sediment the streams are carrying' });
plot({ 'deepest water': deepest }, { title: 'deepest cell on the map' });
`,
      solutionCode: `// Rain, flow, cut, settle. A hundred times.
const gpu = new GPU({ mode });

const SIZE = 64;
const STEPS = 100;
// Rain per step, in units of 10⁻⁵ of depth. Drag it and the whole run redoes itself.
const rainfall = slider('rain', { min: 0.5, max: 6, value: 2, step: 0.5, label: 'rain per step (×10⁻⁵)' });
const SHARED = { size: SIZE, move: 0.6, soft: 0.0002, carry: 60, pickUp: 0.3, settle: 0.05, maxCut: 0.004 };

const dropTotal = gpu.createKernel(function (height, water) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let total = 0;
  let d = here - (height[y][xl] + water[y][xl]); if (d > 0) total += d * d;
  d = here - (height[y][xr] + water[y][xr]); if (d > 0) total += d * d;
  d = here - (height[yd][x] + water[yd][x]); if (d > 0) total += d * d;
  d = here - (height[yu][x] + water[yu][x]); if (d > 0) total += d * d;
  return total;
}, { output: [SIZE, SIZE], constants: SHARED });

const moveWater = gpu.createKernel(function (height, water, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * water[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * water[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * water[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * water[yu][x]) / (drop[yu][x] + this.constants.soft);
  const given = (drop[y][x] * water[y][x]) / (drop[y][x] + this.constants.soft);
  return water[y][x] + this.constants.move * (taken - given) + this.constants.rain;
}, { output: [SIZE, SIZE], constants: { ...SHARED, rain: rainfall * 0.00001 } });

const moveSediment = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * sediment[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * sediment[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * sediment[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * sediment[yu][x]) / (drop[yu][x] + this.constants.soft);
  const given = (drop[y][x] * sediment[y][x]) / (drop[y][x] + this.constants.soft);
  const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
  const load = sediment[y][x];
  let delta = -this.constants.settle * (load - capacity);
  if (load < capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);
  return sediment[y][x] + this.constants.move * (taken - given) + delta;
}, { output: [SIZE, SIZE], constants: SHARED });

const erode = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
  const load = sediment[y][x];
  let delta = -this.constants.settle * (load - capacity);
  if (load < capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);
  return height[y][x] - delta;
}, { output: [SIZE, SIZE], constants: SHARED });

function blank() {
  const grid = [];
  for (let y = 0; y < SIZE; y++) grid.push(new Array(SIZE).fill(0));
  return grid;
}
function totalOf(grid) {
  let sum = 0;
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) sum += grid[y][x];
  return sum;
}
function maxOf(grid) {
  let top = 0;
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) if (grid[y][x] > top) top = grid[y][x];
  return top;
}

let height = terrain;
let water = blank();
let sediment = blank();
const carried = [];
const deepest = [];

for (let i = 0; i < STEPS; i++) {
  // dropTotal first — the other three all read the grid it produces.
  const drop = await dropTotal(height, water);
  // All three read the SAME snapshot. Nothing is swapped until they are done.
  const nextWater = await moveWater(height, water, drop);
  const nextSediment = await moveSediment(height, water, sediment, drop);
  const nextHeight = await erode(height, water, sediment, drop);
  height = nextHeight;
  water = nextWater;
  sediment = nextSediment;
  carried.push(totalOf(sediment));
  deepest.push(maxOf(water));
}

plot({ 'sediment in flight': carried }, { title: 'sediment the streams are carrying' });
plot({ 'deepest water': deepest }, { title: 'deepest cell on the map' });
`,
      inputs: utils => ({ terrain: makeHills(utils, 64, 7) }),
      publicTests: [
        {
          name: 'four kernels, and the last step saw a landscape, not the seed',
          run: async ctx => {
            ctx.assert(
              ctx.kernels.length >= 4,
              `expected 4 kernels (dropTotal, moveWater, moveSediment, erode), found ${ctx.kernels.length}`
            );
            const seed = makeHills(ctx.utils, 64, 7);
            const args = ctx.kernels[1].lastArgs;
            ctx.assert(args && args.length >= 3, 'moveWater should have been called with (height, water, drop)');
            const [height, water] = args;
            ctx.assert(
              maxOf(water) > 1e-4,
              'the last moveWater call still saw a dry map — did the loop feed each step back into the next?'
            );
            let moved = 0;
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) moved = Math.max(moved, Math.abs(height[y][x] - seed[y][x]));
            }
            ctx.assert(moved > 1e-4, 'the terrain handed to the last step is still the untouched seed');
          },
        },
        {
          name: 'all four kernels read the same snapshot',
          run: async ctx => {
            const fromDrop = ctx.kernels[0].lastArgs;
            const fromWater = ctx.kernels[1].lastArgs;
            const fromSediment = ctx.kernels[2].lastArgs;
            const fromErode = ctx.kernels[3].lastArgs;
            ctx.assert(
              fromDrop && fromWater && fromSediment && fromErode,
              'all four kernels should have been called inside the loop'
            );
            const cells = [[32, 32], [8, 40], [55, 12], [1, 1], [63, 63]];
            for (const [y, x] of cells) {
              ctx.assertClose(
                fromWater[0][y][x],
                fromDrop[0][y][x],
                1e-9,
                `height[${y}][${x}] differs between the dropTotal and moveWater calls — ` +
                  'every kernel in a step reads the same snapshot; swap only after all four have returned'
              );
              ctx.assertClose(
                fromErode[0][y][x],
                fromDrop[0][y][x],
                1e-9,
                `height[${y}][${x}] differs between the dropTotal and erode calls — erode was handed already-cut terrain`
              );
              ctx.assertClose(
                fromSediment[1][y][x],
                fromWater[1][y][x],
                1e-9,
                `water[${y}][${x}] differs between the moveWater and moveSediment calls — ` +
                  'the sediment is riding water from the future'
              );
            }
          },
        },
        {
          name: 'both traces are plotted, one value per step',
          run: async ctx => {
            const plots = ctx.logs.filter(line => line.type === 'plot' && line.plot);
            ctx.assert(plots.length >= 2, `expected two plot() calls, found ${plots.length}`);
            for (const entry of plots) {
              const series = entry.plot.series[0];
              ctx.assert(series, 'a plot came back with no series in it');
              ctx.assert(
                series.total === 100,
                `a plotted trace holds ${series.total} points — one per step means 100`
              );
            }
            const carried = plots[0].plot.series[0];
            ctx.assert(
              carried.values[carried.values.length - 1] > carried.values[0],
              'the sediment trace should climb — the streams start empty and pick rock up'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A hundred steps of this must have channelised the water: an evenly
            // spread puddle puts 5% of itself in 5% of the cells. This runs
            // FIRST of the two, because it reads what the learner's own run
            // left in lastArgs and the next test overwrites that.
            const water = ctx.kernels[1].lastArgs[1];
            const share = channelShare(water);
            ctx.assert(
              share > 0.35,
              `the wettest 5% of cells hold only ${(share * 100).toFixed(1)}% of the water — ` +
                'after 100 steps it should be well over a third, or the water never found channels'
            );
            const height = ctx.kernels[1].lastArgs[0];
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assert(
                  Number.isFinite(height[y][x]) && Math.abs(height[y][x]) < 5,
                  `height at [${y}][${x}] is ${height[y][x]} — the run went unstable`
                );
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Drive the learner's own kernels for 20 steps and compare the
            // whole landscape with the reference. Twenty is enough for an
            // out-of-order loop to have drifted well past the tolerance.
            //
            // The reference has to rain at whatever the RUN rained at, not at
            // the slider's default: the task invites the learner to drag the
            // rainfall control, moving it re-runs the program, and pressing Run
            // afterwards tests the kernels that re-run compiled. `ctx.controls`
            // is what the run actually declared — the program is a pure
            // function of its controls, and so is this test.
            const declared = (ctx.controls || []).find(c => c.name === 'rain');
            const rain = declared ? declared.value * 0.00001 : P.rain;
            const seed = makeHills(ctx.utils, 64, 4242);
            let height = seed;
            let water = makeGrid(64, 0);
            let sediment = makeGrid(64, 0);
            for (let i = 0; i < 20; i++) {
              const drop = await ctx.kernels[0](height, water);
              const nextWater = await ctx.kernels[1](height, water, drop);
              const nextSediment = await ctx.kernels[2](height, water, sediment, drop);
              const nextHeight = await ctx.kernels[3](height, water, sediment, drop);
              height = nextHeight;
              water = nextWater;
              sediment = nextSediment;
            }
            const [refH, refW, refS] = runRef(seed, makeGrid(64, 0), makeGrid(64, 0), 20, rain);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(height[y][x], refH[y][x], 2e-3, `height at [${y}][${x}] after 20 steps`);
                ctx.assertClose(water[y][x], refW[y][x], 2e-4, `water at [${y}][${x}] after 20 steps`);
                ctx.assertClose(sediment[y][x], refS[y][x], 2e-4, `sediment at [${y}][${x}] after 20 steps`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'carve-a-valley',
      title: 'Two Hundred Steps, and a River Appears',
      // Under mode "auto" this task legitimately runs on two backends: the four
      // simulation kernels upgrade to WebGPU and `paint` declines it (gpu.js's
      // WebGPU backend refuses graphical mode outright), so the console reports
      // "webgpu (4 kernels) + webgl (1 kernel)". Deliberately NOT pinned:
      // nothing is pipelined, so every pass already hands JavaScript an
      // ordinary array and `paint` receives one too — no texture crosses
      // backends and the pixels are identical to a pure-WebGL run.
      intro: `<p>The payoff. 96×96, two hundred steps, and the only thing that has changed is
        scale — the physics is the four kernels you already wrote, and the loop is the one from
        last task, wrapped in a <code>step()</code> helper. (It calls kernels, so it is
        <code>async</code>, and every caller has to <code>await</code> it.)</p>
        <p>What is new is that you <strong>render inside the loop</strong>. The console collapses
        consecutive <code>render()</code> calls into a frame scrubber, so painting every tenth
        step costs one line and buys the whole story: drag the handle under the picture and watch
        a plausible-but-lifeless fractal grow a drainage network. Frame 1 is ten steps in — the
        water has barely spread and the map is still raw noise. Frame 20 has rivers.</p>
        <p>The painter is task 1's hillshade with the water laid over it in blue, so the channels
        are visible while they form. And the erosion dial is worth dragging for what it
        <em>doesn't</em> do: sixteen times the pick-up rate — <code>0.05</code> to
        <code>0.8</code> — moves about half as much rock again (44.6 units of height against
        66.8 over the run, deepest cut 0.062 against 0.076), and the drainage pattern is
        essentially the same picture. Two reasons, and both are the module: <em>where</em> the
        water goes was settled by the routing long before any rock shifted, and a stream that
        bites faster also loads up faster and starts settling, so the model throttles itself.
        That is the whole of temporal accumulation: a cheap step, run often, producing structure
        no single pass could have computed.</p>`,
      goal: `<strong>Goal:</strong> run <code>STEPS</code> steps, painting and rendering every
        <code>EVERY</code>th one, so the console shows a scrubber from bare noise to a river
        network.`,
      requirements: [
        '<code>await step()</code> <code>STEPS</code> times — it is async, so the <code>await</code> is not optional',
        'Every <code>EVERY</code> steps, <code>await paint(height, water)</code> then <code>render(paint.canvas)</code>',
        'Keep the <code>render()</code> calls consecutive — a <code>console.log</code> between two of them splits the scrubber',
      ],
      hints: [
        {
          title: 'Hint 1 — the shape of it',
          body: `<pre><code>for (let i = 1; i &lt;= STEPS; i++) {
  await step();
  if (i % EVERY === 0) {
    await paint(height, water);
    render(paint.canvas);
  }
}</code></pre>
<p>Twenty frames out of two hundred steps, and no bookkeeping at all.</p>`,
        },
        {
          title: 'Hint 2 — why step() needs its await',
          body: `<p><code>step()</code> awaits four kernels, so it returns a promise like any
            other async function. Call it without <code>await</code> and the loop fires two
            hundred overlapping steps at grids that are still being written — the picture will
            be nonsense and the console will tell you a promise turned up where a grid should be.</p>`,
        },
        {
          title: 'Hint 3 — reading the scrubber',
          body: `<p>The first frames look like nothing is happening: the water is still spreading
            out. Around frame 3 thin blue threads appear on the slopes, and from there they
            thicken, merge, and cut visible notches into the ridges. The two hollows fill up and
            become lakes.</p>`,
        },
      ],
      transfer: `This is the shape of every offline GPU simulation that ends in a picture:
        a compute loop that never leaves the device, a render pass tapping it every N steps, and
        a result that exists only because the cheap step ran thousands of times. Landscape tools
        (Houdini, World Machine, Gaea) run exactly this loop — just at 4096² and with a few more
        terms in the capacity.`,
      starterCode: `// Everything you wrote, at 96×96, for two hundred steps.
const gpu = new GPU({ mode });

const SIZE = 96;
const STEPS = 200;
const EVERY = 10;
const erosion = slider('erosion', { min: 0.05, max: 0.8, value: 0.3, step: 0.05, label: 'erosion rate (pickUp)' });
const SHARED = { size: SIZE, move: 0.6, soft: 0.0002, rain: 0.00002, carry: 60, pickUp: erosion, settle: 0.05, maxCut: 0.004 };

const dropTotal = gpu.createKernel(function (height, water) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let total = 0;
  let d = here - (height[y][xl] + water[y][xl]); if (d > 0) total += d * d;
  d = here - (height[y][xr] + water[y][xr]); if (d > 0) total += d * d;
  d = here - (height[yd][x] + water[yd][x]); if (d > 0) total += d * d;
  d = here - (height[yu][x] + water[yu][x]); if (d > 0) total += d * d;
  return total;
}, { output: [SIZE, SIZE], constants: SHARED });

const moveWater = gpu.createKernel(function (height, water, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * water[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * water[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * water[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * water[yu][x]) / (drop[yu][x] + this.constants.soft);
  const given = (drop[y][x] * water[y][x]) / (drop[y][x] + this.constants.soft);
  return water[y][x] + this.constants.move * (taken - given) + this.constants.rain;
}, { output: [SIZE, SIZE], constants: SHARED });

const moveSediment = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * sediment[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * sediment[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * sediment[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * sediment[yu][x]) / (drop[yu][x] + this.constants.soft);
  const given = (drop[y][x] * sediment[y][x]) / (drop[y][x] + this.constants.soft);
  const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
  const load = sediment[y][x];
  let delta = -this.constants.settle * (load - capacity);
  if (load < capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);
  return sediment[y][x] + this.constants.move * (taken - given) + delta;
}, { output: [SIZE, SIZE], constants: SHARED });

const erode = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
  const load = sediment[y][x];
  let delta = -this.constants.settle * (load - capacity);
  if (load < capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);
  return height[y][x] - delta;
}, { output: [SIZE, SIZE], constants: SHARED });

// Task 1's hillshade, with the water painted over it in blue.
const paint = gpu.createKernel(function (height, water) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const gx = (height[y][xr] - height[y][xl]) * 0.5 * this.constants.relief;
  const gy = (height[yu][x] - height[yd][x]) * 0.5 * this.constants.relief;
  const inv = 1 / Math.sqrt(gx * gx + gy * gy + 1);
  let lit = this.constants.light * (1 + gx + gy) * inv;
  if (lit < 0) lit = 0;
  const rock = Math.min(1, height[y][x] * 1.15);
  const wet = Math.min(1, Math.sqrt(water[y][x] * 40));
  this.color(
    lit * (0.3 + 0.62 * rock) * (1 - wet),
    lit * (0.44 + 0.4 * rock) * (1 - wet) + 0.26 * wet,
    lit * (0.54 + 0.2 * rock) * (1 - wet) + 0.85 * wet,
    1
  );
}, { output: [SIZE, SIZE], graphical: true, constants: { size: SIZE, relief: 18, light: 0.5774 } });

function blank() {
  const grid = [];
  for (let y = 0; y < SIZE; y++) grid.push(new Array(SIZE).fill(0));
  return grid;
}

let height = terrain;
let water = blank();
let sediment = blank();

// One step of everything. It awaits kernels, so it is async — and so is every
// call to it.
async function step() {
  const drop = await dropTotal(height, water);
  const nextWater = await moveWater(height, water, drop);
  const nextSediment = await moveSediment(height, water, sediment, drop);
  const nextHeight = await erode(height, water, sediment, drop);
  height = nextHeight;
  water = nextWater;
  sediment = nextSediment;
}

// TODO: run STEPS steps, painting and rendering every EVERY of them.
// Consecutive render() calls collapse into one frame scrubber.
`,
      solutionCode: `// Everything you wrote, at 96×96, for two hundred steps.
const gpu = new GPU({ mode });

const SIZE = 96;
const STEPS = 200;
const EVERY = 10;
const erosion = slider('erosion', { min: 0.05, max: 0.8, value: 0.3, step: 0.05, label: 'erosion rate (pickUp)' });
const SHARED = { size: SIZE, move: 0.6, soft: 0.0002, rain: 0.00002, carry: 60, pickUp: erosion, settle: 0.05, maxCut: 0.004 };

const dropTotal = gpu.createKernel(function (height, water) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let total = 0;
  let d = here - (height[y][xl] + water[y][xl]); if (d > 0) total += d * d;
  d = here - (height[y][xr] + water[y][xr]); if (d > 0) total += d * d;
  d = here - (height[yd][x] + water[yd][x]); if (d > 0) total += d * d;
  d = here - (height[yu][x] + water[yu][x]); if (d > 0) total += d * d;
  return total;
}, { output: [SIZE, SIZE], constants: SHARED });

const moveWater = gpu.createKernel(function (height, water, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * water[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * water[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * water[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * water[yu][x]) / (drop[yu][x] + this.constants.soft);
  const given = (drop[y][x] * water[y][x]) / (drop[y][x] + this.constants.soft);
  return water[y][x] + this.constants.move * (taken - given) + this.constants.rain;
}, { output: [SIZE, SIZE], constants: SHARED });

const moveSediment = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const here = height[y][x] + water[y][x];
  let taken = 0;
  let d = height[y][xl] + water[y][xl] - here;
  if (d > 0) taken += (d * d * sediment[y][xl]) / (drop[y][xl] + this.constants.soft);
  d = height[y][xr] + water[y][xr] - here;
  if (d > 0) taken += (d * d * sediment[y][xr]) / (drop[y][xr] + this.constants.soft);
  d = height[yd][x] + water[yd][x] - here;
  if (d > 0) taken += (d * d * sediment[yd][x]) / (drop[yd][x] + this.constants.soft);
  d = height[yu][x] + water[yu][x] - here;
  if (d > 0) taken += (d * d * sediment[yu][x]) / (drop[yu][x] + this.constants.soft);
  const given = (drop[y][x] * sediment[y][x]) / (drop[y][x] + this.constants.soft);
  const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
  const load = sediment[y][x];
  let delta = -this.constants.settle * (load - capacity);
  if (load < capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);
  return sediment[y][x] + this.constants.move * (taken - given) + delta;
}, { output: [SIZE, SIZE], constants: SHARED });

const erode = gpu.createKernel(function (height, water, sediment, drop) {
  const x = this.thread.x;
  const y = this.thread.y;
  const capacity = this.constants.carry * Math.sqrt(drop[y][x]) * water[y][x];
  const load = sediment[y][x];
  let delta = -this.constants.settle * (load - capacity);
  if (load < capacity) delta = Math.min(this.constants.pickUp * (capacity - load), this.constants.maxCut);
  return height[y][x] - delta;
}, { output: [SIZE, SIZE], constants: SHARED });

// Task 1's hillshade, with the water painted over it in blue.
const paint = gpu.createKernel(function (height, water) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const gx = (height[y][xr] - height[y][xl]) * 0.5 * this.constants.relief;
  const gy = (height[yu][x] - height[yd][x]) * 0.5 * this.constants.relief;
  const inv = 1 / Math.sqrt(gx * gx + gy * gy + 1);
  let lit = this.constants.light * (1 + gx + gy) * inv;
  if (lit < 0) lit = 0;
  const rock = Math.min(1, height[y][x] * 1.15);
  const wet = Math.min(1, Math.sqrt(water[y][x] * 40));
  this.color(
    lit * (0.3 + 0.62 * rock) * (1 - wet),
    lit * (0.44 + 0.4 * rock) * (1 - wet) + 0.26 * wet,
    lit * (0.54 + 0.2 * rock) * (1 - wet) + 0.85 * wet,
    1
  );
}, { output: [SIZE, SIZE], graphical: true, constants: { size: SIZE, relief: 18, light: 0.5774 } });

function blank() {
  const grid = [];
  for (let y = 0; y < SIZE; y++) grid.push(new Array(SIZE).fill(0));
  return grid;
}

let height = terrain;
let water = blank();
let sediment = blank();

// One step of everything. It awaits kernels, so it is async — and so is every
// call to it.
async function step() {
  const drop = await dropTotal(height, water);
  const nextWater = await moveWater(height, water, drop);
  const nextSediment = await moveSediment(height, water, sediment, drop);
  const nextHeight = await erode(height, water, sediment, drop);
  height = nextHeight;
  water = nextWater;
  sediment = nextSediment;
}

for (let i = 1; i <= STEPS; i++) {
  await step();
  if (i % EVERY === 0) {
    // Twenty frames. Nothing else may be logged between them, or the console
    // stops seeing one strip and starts seeing twenty pictures.
    await paint(height, water);
    render(paint.canvas);
  }
}
`,
      inputs: utils => ({ terrain: makeHills(utils, 96, 7) }),
      publicTests: [
        {
          name: 'the run leaves a scrubber behind — twenty frames',
          run: async ctx => {
            const frames = ctx.logs.filter(line => line.type === 'canvas');
            ctx.assert(
              frames.length >= 15,
              `found ${frames.length} rendered frames — one every 10 of 200 steps is 20`
            );
            ctx.assert(ctx.canvas, 'no canvas — did you call render(paint.canvas)?');
            ctx.assert(
              ctx.canvas.width === 96 && ctx.canvas.height === 96,
              `expected a 96×96 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
          },
        },
        {
          name: 'the frames are consecutive, so the console groups them',
          run: async ctx => {
            // A console.log between two render()s splits the strip in two.
            let run = 0;
            let longest = 0;
            for (const line of ctx.logs) {
              run = line.type === 'canvas' ? run + 1 : 0;
              if (run > longest) longest = run;
            }
            ctx.assert(
              longest >= 15,
              `the longest unbroken run of frames is ${longest} — the scrubber only forms from consecutive ` +
                'render() calls, so nothing else may be logged between them'
            );
          },
        },
        {
          name: 'two hundred steps actually ran, and the landscape moved',
          run: async ctx => {
            const seed = makeHills(ctx.utils, 96, 7);
            const args = ctx.kernels[1] && ctx.kernels[1].lastArgs;
            ctx.assert(args, 'moveWater was never called — did the loop run?');
            const [height, water] = args;
            let moved = 0;
            for (let y = 0; y < 96; y++) {
              for (let x = 0; x < 96; x++) moved = Math.max(moved, Math.abs(height[y][x] - seed[y][x]));
            }
            ctx.assert(
              moved > 0.03,
              `the deepest change to the terrain is ${moved.toFixed(4)} — two hundred steps should cut ` +
                'several times that, so either the loop is short or nothing is feeding back'
            );
            ctx.assert(
              maxOf(water) > 0.02,
              `the deepest water is only ${maxOf(water).toFixed(4)} — 200 steps of rain should have pooled somewhere`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The point of the whole module: the water is in channels, and the
            // rock the streams removed came out of those channels rather than
            // being shaved evenly off the map.
            const seed = makeHills(ctx.utils, 96, 7);
            const [height, water] = ctx.kernels[1].lastArgs;
            const share = channelShare(water);
            ctx.assert(
              share > 0.5,
              `the wettest 5% of cells hold ${(share * 100).toFixed(1)}% of the water — after 200 steps ` +
                'a real drainage network puts more than half of it there'
            );
            const cells = [];
            for (let y = 0; y < 96; y++) {
              for (let x = 0; x < 96; x++) cells.push([water[y][x], seed[y][x] - height[y][x]]);
            }
            cells.sort((a, b) => b[0] - a[0]);
            const top = Math.round(cells.length * 0.05);
            const wetCut = cells.slice(0, top).reduce((sum, c) => sum + c[1], 0) / top;
            const dryCut = cells.slice(top).reduce((sum, c) => sum + c[1], 0) / (cells.length - top);
            ctx.assert(
              dryCut > 0 && wetCut > dryCut * 3,
              `the wettest cells were cut by ${wetCut.toFixed(4)} on average against ${dryCut.toFixed(4)} ` +
                'elsewhere — a river carves its own bed, so that ratio should be several times over, not level'
            );
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The seed frame really is the seed, and the last frame is not.
            const frames = ctx.logs.filter(line => line.type === 'canvas' && line.snapshot);
            ctx.assert(frames.length >= 2, 'expected at least a first and a last frame');
            const paint = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(paint, 'no graphical kernel found — the painter should be graphical: true');
            const seed = makeHills(ctx.utils, 96, 7);
            const dry = makeGrid(96, 0);
            await paint(seed, dry);
            const before = Array.from(paint.getPixels());
            const [height, water] = ctx.kernels[1].lastArgs;
            await paint(height, water);
            const after = paint.getPixels();
            let changed = 0;
            for (let i = 0; i < after.length; i += 4) {
              if (Math.abs(after[i] - before[i]) > 8 || Math.abs(after[i + 2] - before[i + 2]) > 8) changed++;
            }
            ctx.assert(
              changed > 96 * 96 * 0.25,
              `only ${changed} of ${96 * 96} pixels differ between the first and last frame — ` +
                'the scrubber should have something to show'
            );
          },
        },
      ],
    },
  ],
};
