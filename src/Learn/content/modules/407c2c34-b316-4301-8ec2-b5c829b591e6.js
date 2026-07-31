// Module: Cellular Automata — uuid 407c2c34-b316-4301-8ec2-b5c829b591e6 (short id 407c2c34).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. Legacy id (pre-uuid URLs, localStorage migration): 3-3.
//
// Module 3.3 — Cellular Automata.
//
// Five tasks: counting live neighbors on a wrapped grid → one tick of
// Conway's Life as a numeric 2D kernel → generations via a JS feed-back
// loop → the glider, painted and flying → rule tables as data, one kernel
// for every Life-like universe.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// only numbers / nested number arrays as inputs, this.thread.* for indexing,
// statically bounded loops, single-return bodies for maximum backend safety.
// World convention: grid[y][x] is 0 (dead) or 1 (alive), 16×16, toroidal.

const SIZE = 16;

// ---------------------------------------------------------------- helpers

function emptyGrid(size = SIZE) {
  const grid = new Array(size);
  for (let y = 0; y < size; y++) grid[y] = new Array(size).fill(0);
  return grid;
}

function withCells(cells, size = SIZE) {
  const grid = emptyGrid(size);
  for (const [y, x] of cells) grid[y][x] = 1;
  return grid;
}

function randomGrid(utils, seed, size = SIZE, density = 0.35) {
  const rand = utils.seededRandom(seed);
  const grid = emptyGrid(size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) grid[y][x] = rand() < density ? 1 : 0;
  }
  return grid;
}

function neighborsOf(grid, y, x) {
  const size = grid.length;
  let count = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      if (dy === 0 && dx === 0) continue;
      count += grid[(y + dy + size) % size][(x + dx + size) % size];
    }
  }
  return count;
}

// Outer-totalistic reference step: born[n] / stay[n] are 9-entry 0/1 tables.
function refStep(grid, born = B3, stay = S23) {
  const size = grid.length;
  const next = emptyGrid(size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const n = neighborsOf(grid, y, x);
      next[y][x] = grid[y][x] === 1 ? stay[n] : born[n];
    }
  }
  return next;
}

function refSteps(grid, n, born = B3, stay = S23) {
  let current = grid;
  for (let i = 0; i < n; i++) current = refStep(current, born, stay);
  return current;
}

function population(grid) {
  let alive = 0;
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) alive += grid[y][x];
  }
  return alive;
}

// `hint` (optional) replaces the label when a whole-grid diagnosis applies —
// see ruleHint below. The comparison itself is untouched by it.
function expectGrid(ctx, got, want, label, hint) {
  ctx.assert(got && got.length === want.length, `${label} — expected ${want.length} rows`);
  const prefix = hint || label;
  for (let y = 0; y < want.length; y++) {
    for (let x = 0; x < want.length; x++) {
      ctx.assertClose(got[y][x], want[y][x], 1e-3, `${prefix} — cell [${y}][${x}]`);
    }
  }
}

// ---- near-miss diagnosis --------------------------------------------------
//
// Cells here are 0 or 1, so one mismatched cell is a coin flip's worth of
// evidence: any diagnosis has to fit the ENTIRE grid. ruleHint replays the
// tick under the most plausible mis-readings of the rulebook and names the one
// that reproduces the learner's result exactly — otherwise it stays quiet,
// because a wrong diagnosis is worse than none.
function sameGrid(got, want) {
  for (let y = 0; y < want.length; y++) {
    if (!got[y]) return false;
    for (let x = 0; x < want.length; x++) {
      if (!(Math.abs(got[y][x] - want[y][x]) <= 1e-3)) return false;
    }
  }
  return true;
}

function ruleHint(got, grid, born = B3, stay = S23) {
  const alternatives = [
    [refStep(grid, stay, stay),
      'a dead cell with 2 neighbors came alive — birth is on exactly 3; 2 is what lets an already-live cell survive'],
    [refStep(grid, born, born),
      'live cells with 2 neighbors died — survival covers 2 or 3, and only birth is limited to exactly 3'],
    [refStep(grid, stay, born),
      'the two rules are swapped — a dead cell follows the birth rule, a live cell the survival rule'],
    [grid,
      'the world came back unchanged — the rule never reached the return value'],
  ];
  for (const [candidate, message] of alternatives) {
    if (sameGrid(got, candidate)) return message;
  }
  return null;
}

// Task 1: the 3×3 sum still carrying the cell itself.
function censusProbes(grid, y, x) {
  return [
    [neighborsOf(grid, y, x) + grid[y][x],
      'your own cell is still inside the 3×3 sum — subtract grid[this.thread.y][this.thread.x] at the end'],
  ];
}

// Task 1, wrapped cells only: the first test already proved the census counts
// ordinary neighbors, so a 0 where a wrapped neighbor belongs can only be the
// missing + 16 — JavaScript's % yields −1 there, and the GPU's is worse.
function wrapHint(got) {
  return Math.abs(got) <= 1e-3
    ? 'the edge did not wrap — add the width before the modulo, (this.thread.y + dy + 16) % 16, because a bare % can go negative'
    : null;
}

// Task 3: logging the population BEFORE the step shifts the whole history by a
// generation. `before` is the population at the start of generation g.
function generationHint(ctx, g, before) {
  const stale = `gen ${g}: ${before} alive`;
  return ctx.logs.some(line => line.type === 'log' && line.text && line.text.includes(stale))
    ? 'that generation logged the population from BEFORE its step — count the live cells after current = step(current)'
    : null;
}

// Task 4: exactly the pixels that should be dark are lit and vice versa.
function litHint(lit, alive, total) {
  return lit === total - alive
    ? 'the two colors are swapped — live cells take the green, dead cells the dark background'
    : null;
}

function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Rule tables, indexed by live-neighbor count 0–8.
const B3 = [0, 0, 0, 1, 0, 0, 0, 0, 0]; //            Life births on 3
const S23 = [0, 0, 1, 1, 0, 0, 0, 0, 0]; //           Life survives on 2 or 3
const B36 = [0, 0, 0, 1, 0, 0, 1, 0, 0]; //           HighLife births on 3 or 6
const B3678 = [0, 0, 0, 1, 0, 0, 1, 1, 1]; //         Day & Night
const S34678 = [0, 0, 0, 1, 1, 0, 1, 1, 1];

// Classic patterns as [y, x] cell lists (grids built fresh per use).
const BLINKER = [[7, 6], [7, 7], [7, 8]]; //          horizontal, flips vertical
const BLOCK = [[3, 3], [3, 4], [4, 3], [4, 4]]; //    2×2 still life
const GLIDER = [[1, 2], [2, 3], [3, 1], [3, 2], [3, 3]]; // flies down-right
const R_PENTOMINO = [[6, 8], [6, 9], [7, 7], [7, 8], [8, 8]];

function shiftCells(cells, dy, dx, size = SIZE) {
  return cells.map(([y, x]) => [(y + dy + size) % size, (x + dx + size) % size]);
}

export default {
  uuid: '407c2c34-b316-4301-8ec2-b5c829b591e6',
  version: 1,
  slug: 'cellular-automata',
  legacyId: '3-3',
  title: 'Cellular Automata',
  blurb: "Conway's Life and friends: feed a kernel's output back in and watch worlds evolve.",
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'neighbor-census',
      title: 'The Neighbor Census',
      intro: `<p>A cellular automaton is a world of cells, each one dead (<code>0</code>) or alive
        (<code>1</code>), where every cell's next state depends only on its immediate neighborhood.
        That makes it embarrassingly parallel: 256 cells, 256 threads, and no thread needs to know
        what any other thread is doing — only what the grid looked like.</p>
        <p>Every rule in this module starts with the same question: <strong>how many of my eight
        neighbors are alive?</strong> This world is a torus — walk off the right edge, reappear on
        the left — and wrapping costs one modulo: <code>(x + dx + 16) % 16</code>. The
        <code>+ 16</code> is not decoration: JavaScript's <code>%</code> can go negative while the
        GPU's cannot, and adding the width first keeps both operands positive so CPU mode and GPU
        mode tell the same story.</p>`,
      goal: `<strong>Goal:</strong> make the kernel return, for every cell, the number of live
        cells among its eight neighbors — with the edges wrapped around.`,
      requirements: [
        'Visit the 3×3 block around this cell with nested <code>dy</code>/<code>dx</code> loops from −1 to 1',
        'Wrap every coordinate: <code>(this.thread.x + dx + 16) % 16</code> (and the same for y)',
        'Don\'t count yourself — a cell is not its own neighbor',
      ],
      hints: [
        {
          title: 'Hint 1 — the loop bounds',
          body: `<p>Two statically bounded loops: <code>for (let dy = -1; dy &lt; 2; dy++)</code>
            around <code>for (let dx = -1; dx &lt; 2; dx++)</code>. Nine visits per cell.</p>`,
        },
        {
          title: 'Hint 2 — the subtract-self trick',
          body: `<p>Skipping the middle of the 3×3 block needs no <code>if</code>: sum all nine
            cells, then subtract <code>grid[this.thread.y][this.thread.x]</code> at the end. If
            you're dead you subtract 0; if you're alive you take yourself back out.</p>`,
        },
        {
          title: 'Hint 3 — the whole loop body',
          body: `<pre><code>const yy = (this.thread.y + dy + 16) % 16;
const xx = (this.thread.x + dx + 16) % 16;
count += grid[yy][xx];</code></pre>
<p>— then</p>
<pre><code>return count - grid[this.thread.y][this.thread.x];</code></pre>`,
        },
      ],
      transfer: `Reading a fixed window around your own coordinate is the <em>stencil</em> pattern,
        and it dominates real GPU workloads: CUDA stencil kernels tile the grid into shared memory
        with a one-cell "halo" so neighbors are read once, and WebGPU compute shaders do the same
        with workgroup memory.`,
      starterCode: `// Every cell asks the same question, all at once:
// how many of my eight neighbors are alive?
// The world wraps — leave one edge, come back on the other.
const gpu = new GPU({ mode });

const census = gpu.createKernel(function (grid) {
  let count = 0;
  // TODO: sum the 3x3 block around this cell (dy and dx from -1 to 1),
  // wrapping each coordinate with (coord + d + 16) % 16.
  // Careful: a cell is not its own neighbor.
  return count;
}, { output: [16, 16] });

const counts = census(grid);
console.log('cell (8, 8) sees', counts[8][8], 'live neighbors');
`,
      solutionCode: `// Every cell asks the same question, all at once:
// how many of my eight neighbors are alive?
const gpu = new GPU({ mode });

const census = gpu.createKernel(function (grid) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += grid[yy][xx];
    }
  }
  // The 3x3 sum counted this cell too — take it back out.
  return count - grid[this.thread.y][this.thread.x];
}, { output: [16, 16] });

const counts = census(grid);
console.log('cell (8, 8) sees', counts[8][8], 'live neighbors');
`,
      inputs: utils => ({ grid: randomGrid(utils, 1101) }),
      publicTests: [
        {
          name: 'a lone cell has zero neighbors — each of its eight neighbors sees one',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const lone = withCells([[5, 5]]);
            const counts = ctx.kernel(lone);
            ctx.assertClose(counts[5][5], 0, 1e-3,
              diagnose(counts[5][5], 0, 1e-3, censusProbes(lone, 5, 5)) ||
                'the live cell itself (it is not its own neighbor)');
            const ring = [[4, 4], [4, 5], [4, 6], [5, 4], [5, 6], [6, 4], [6, 5], [6, 6]];
            for (const [y, x] of ring) {
              const hint = diagnose(counts[y][x], 1, 1e-3, censusProbes(lone, y, x));
              ctx.assertClose(counts[y][x], 1, 1e-3, hint || `neighbor cell [${y}][${x}]`);
            }
            ctx.assertClose(counts[10][10], 0, 1e-3, 'a far-away cell');
          },
        },
        {
          name: 'the world wraps: a corner cell is seen across all four edges',
          run: async ctx => {
            const counts = ctx.kernel(withCells([[0, 0]]));
            ctx.assertClose(counts[15][15], 1, 1e-3,
              wrapHint(counts[15][15]) || 'diagonal wrap — cell [15][15]');
            ctx.assertClose(counts[0][15], 1, 1e-3,
              wrapHint(counts[0][15]) || 'horizontal wrap — cell [0][15]');
            ctx.assertClose(counts[15][0], 1, 1e-3,
              wrapHint(counts[15][0]) || 'vertical wrap — cell [15][0]');
            ctx.assertClose(counts[1][1], 1, 1e-3, 'ordinary diagonal — cell [1][1]');
            ctx.assertClose(counts[0][0], 0, 1e-3, 'the corner cell itself');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const grid = randomGrid(ctx.utils, 2202);
            const counts = ctx.kernel(grid);
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                const expected = neighborsOf(grid, y, x);
                const hint = diagnose(counts[y][x], expected, 1e-3, censusProbes(grid, y, x));
                ctx.assertClose(counts[y][x], expected, 1e-3, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'one-tick',
      title: 'One Tick of Life',
      intro: `<p>In 1970 John Conway picked the simplest rules he could find that make a world worth
        watching. <strong>Birth:</strong> a dead cell with exactly 3 live neighbors comes alive.
        <strong>Survival:</strong> a live cell with 2 or 3 neighbors stays alive. Everything else —
        lonely or overcrowded — dies. That's the whole game (the notation is B3/S23).</p>
        <p>There's a classic bug in CPU implementations: update the grid <em>in place</em> and
        cells start reading half-new, half-old neighbors. A kernel is immune by construction —
        every thread reads the old <code>world</code> argument and writes into a brand-new output.
        The double buffer isn't a technique here; it's what a kernel <em>is</em>.</p>`,
      goal: `<strong>Goal:</strong> finish the kernel so it computes one full generation of
        Conway's Life — birth on 3, survival on 2 or 3, death otherwise.`,
      requirements: [
        'Keep the wrapped neighbor census from the last task (already in place)',
        'A dead cell returns <code>1</code> exactly when <code>count === 3</code>',
        'A live cell returns <code>1</code> exactly when <code>count === 2 || count === 3</code>',
        'Everything else returns <code>0</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — start dead',
          body: `<p>Declare <code>let next = 0;</code>, flip it to <code>1</code> in the cases
            that live, and <code>return next;</code> once at the end. Two <code>if</code>s cover
            the whole rulebook.</p>`,
        },
        {
          title: 'Hint 2 — the two ifs',
          body: `<pre><code>if (self === 1 &amp;&amp; (count === 2 || count === 3)) next = 1;
if (self === 0 &amp;&amp; count === 3) next = 1;</code></pre>`,
        },
      ],
      transfer: `Reading one buffer while writing another is <em>ping-ponging</em>, and every
        platform institutionalizes it: WebGPU simulations bind two storage buffers and swap their
        roles each dispatch, and CUDA solvers keep <code>d_old</code>/<code>d_new</code> device
        pointers and trade them every launch.`,
      starterCode: `// B3/S23: birth on 3 neighbors, survival on 2 or 3, death otherwise.
// The census below is task 1's answer — the rulebook is yours.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  // TODO: apply Conway's rules to \`self\` and \`count\`.
  return self;
}, { output: [16, 16] });

const next = step(world);
console.log('before:', world[7].join(''));
console.log('after :', Array.from(next[7]).join(''));
`,
      solutionCode: `// B3/S23: birth on 3 neighbors, survival on 2 or 3, death otherwise.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  let next = 0;
  if (self === 1 && (count === 2 || count === 3)) next = 1;
  if (self === 0 && count === 3) next = 1;
  return next;
}, { output: [16, 16] });

const next = step(world);
console.log('before:', world[7].join(''));
console.log('after :', Array.from(next[7]).join(''));
`,
      inputs: () => ({ world: withCells(BLINKER) }),
      publicTests: [
        {
          name: 'the blinker: three-in-a-row flips to three-in-a-column',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const start = withCells(BLINKER);
            const next = ctx.kernel(start);
            expectGrid(ctx, next, withCells([[6, 7], [7, 7], [8, 7]]), 'blinker after one tick',
              ruleHint(next, start));
          },
        },
        {
          name: 'the block: a 2×2 square is a still life — nothing moves',
          run: async ctx => {
            const start = withCells(BLOCK);
            const next = ctx.kernel(start);
            expectGrid(ctx, next, withCells(BLOCK), 'block after one tick', ruleHint(next, start));
          },
        },
        {
          name: 'an empty world stays empty — no spontaneous generation',
          run: async ctx => {
            const next = ctx.kernel(emptyGrid());
            expectGrid(ctx, next, emptyGrid(), 'empty world after one tick');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const grid = randomGrid(ctx.utils, 4404);
            const next = ctx.kernel(grid);
            expectGrid(ctx, next, refStep(grid), 'random world, one tick', ruleHint(next, grid));
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            const grid = randomGrid(ctx.utils, 5505, SIZE, 0.6);
            const next = ctx.kernel(grid);
            expectGrid(ctx, next, refStep(grid), 'crowded world, one tick', ruleHint(next, grid));
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'generations',
      title: 'Generations: Feed It Back',
      intro: `<p>One tick is a snapshot; a world is a movie. A kernel has no memory of the previous
        frame — <strong>time lives in JavaScript</strong>. The result of a 2D kernel is an array of
        rows, which is exactly the shape the kernel accepts as input, so
        <code>current = step(current)</code> is the whole time machine: output becomes input,
        forever.</p>
        <p>Your test subject is the <strong>R-pentomino</strong> — five innocent-looking cells that
        erupt into chaos (on an infinite grid they don't settle down for 1,103 generations; Conway's
        group tracked it by hand). You'll run six generations and log the population after each, so
        you can watch the explosion begin.</p>`,
      goal: `<strong>Goal:</strong> restore the B3/S23 rule inside the kernel, then run 6
        generations by feeding each output back in — logging
        <code>'gen ' + g + ': ' + alive + ' alive'</code> after every step.`,
      requirements: [
        'Complete the kernel: the same B3/S23 rule you wrote last task',
        'Loop 6 times in plain JavaScript, reassigning: <code>current = step(current)</code>',
        'After each step, total the live cells in plain JS (kernel output rows are ordinary arrays)',
        `Log each generation exactly as <code>'gen ' + g + ': ' + alive + ' alive'</code>, g from 1 to 6`,
      ],
      hints: [
        {
          title: 'Hint 1 — the feed-back loop',
          body: `<p><code>let current = world;</code> then</p>
<pre><code>for (let g = 1; g &lt;= 6; g++) {
  current = step(current);
  // …
}</code></pre>
<p>No copying, no bookkeeping — the kernel's output is already a valid input.</p>`,
        },
        {
          title: 'Hint 2 — counting the living',
          body: `<p>Inside the loop, after stepping: <code>let alive = 0;</code> and two nested
            loops adding <code>current[y][x]</code>. Cells are 0 or 1, so the sum <em>is</em> the
            population.</p>`,
        },
      ],
      transfer: `The frame loop lives on the host everywhere: a CUDA fluid sim launches its kernel
        thousands of times from an ordinary CPU <code>for</code> loop, and a Metal app encodes one
        compute dispatch per frame — the GPU computes each tick, but the CPU decides that time
        passes.`,
      starterCode: `// The kernel computes one tick. Time itself is a JavaScript loop:
// whatever comes out goes straight back in.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  // TODO: B3/S23 — you wrote this rule last task. Own it.
  return self;
}, { output: [16, 16] });

// world starts as the R-pentomino: five cells, endless trouble.
// TODO: run 6 generations. Each time around: current = step(current),
// count the live cells in plain JS, then log exactly:
//   console.log('gen ' + g + ': ' + alive + ' alive');
let current = world;
`,
      solutionCode: `// The kernel computes one tick. Time itself is a JavaScript loop:
// whatever comes out goes straight back in.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  let next = 0;
  if (self === 1 && (count === 2 || count === 3)) next = 1;
  if (self === 0 && count === 3) next = 1;
  return next;
}, { output: [16, 16] });

let current = world;
for (let g = 1; g <= 6; g++) {
  current = step(current);
  let alive = 0;
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) alive += current[y][x];
  }
  console.log('gen ' + g + ': ' + alive + ' alive');
}
`,
      inputs: () => ({ world: withCells(R_PENTOMINO) }),
      publicTests: [
        {
          name: "six generations logged, matching the R-pentomino's true population history",
          run: async ctx => {
            let grid = withCells(R_PENTOMINO);
            for (let g = 1; g <= 6; g++) {
              const before = population(grid);
              grid = refStep(grid);
              const expected = 'gen ' + g + ': ' + population(grid) + ' alive';
              const found = ctx.logs.some(
                line => line.type === 'log' && line.text && line.text.includes(expected)
              );
              ctx.assert(found, generationHint(ctx, g, before) ||
                `expected a log line containing "${expected}"`);
            }
          },
        },
        {
          name: 'the step kernel is still a faithful B3/S23 tick',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const start = withCells(BLINKER);
            const next = ctx.kernel(start);
            expectGrid(ctx, next, withCells([[6, 7], [7, 7], [8, 7]]), 'blinker after one tick',
              ruleHint(next, start));
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Feed the kernel's own output back in for three ticks on a
            // different world — the loop pattern must actually work.
            const start = randomGrid(ctx.utils, 6606);
            let grid = start;
            for (let i = 0; i < 3; i++) grid = ctx.kernel(grid);
            expectGrid(ctx, grid, refSteps(start, 3), 'random world after three ticks');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'glider-on-screen',
      title: 'Watch the Glider Fly',
      intro: `<p>The <strong>glider</strong> is five cells that <em>travel</em>. No individual cell
        moves — each one just dies or is born in place, like every other cell — yet after four
        ticks an identical copy of the pattern stands one cell down and one cell right. Motion as
        pure side effect. When it was discovered in 1970 it changed the game: Life could transmit
        information.</p>
        <p>Time to see it. You already know both halves from earlier tracks: a numeric
        <code>step</code> kernel computes generations, and a <strong>graphical</strong> kernel
        turns the final grid into pixels. Simulation pass, then render pass — the fundamental
        division of labor in every real-time visualization.</p>`,
      goal: `<strong>Goal:</strong> complete the <code>paint</code> kernel so live cells glow
        green and dead cells stay near-black, then watch the glider that started in the top-left
        arrive further down the board.`,
      requirements: [
        'Read this thread\'s cell from <code>cells</code> — same indexing as every task so far',
        'Live cells: <code>this.color(0.2, 1, 0.4, 1)</code>; dead cells: <code>this.color(0.05, 0.06, 0.09, 1)</code>',
        'Leave the 8-generation loop and <code>render()</code> call as they are',
      ],
      hints: [
        {
          title: 'Hint 1 — numbers in, colors out',
          body: `<p><code>cells</code> is the plain 2D grid the step kernel produced. Read
            <code>cells[this.thread.y][this.thread.x]</code> into a variable — it's 0 or 1.</p>`,
        },
        {
          title: 'Hint 2 — the branch',
          body: `<pre><code>if (alive === 1) {
  this.color(0.2, 1, 0.4, 1);
} else {
  this.color(0.05, 0.06, 0.09, 1);
}</code></pre>`,
        },
      ],
      transfer: `Sim pass feeding a render pass is the standard split in every API: a WebGPU compute
        shader writes the state a fragment shader then draws, and CUDA–OpenGL interop exists purely
        so simulation buffers can be displayed without a round trip through the CPU.`,
      starterCode: `// Two kernels, two jobs: step computes the world, paint shows it.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  let next = 0;
  if (self === 1 && (count === 2 || count === 3)) next = 1;
  if (self === 0 && count === 3) next = 1;
  return next;
}, { output: [16, 16] });

const paint = gpu.createKernel(function (cells) {
  // TODO: live cells glow this.color(0.2, 1, 0.4, 1),
  // dead cells stay this.color(0.05, 0.06, 0.09, 1).
  this.color(1, 0, 1, 1);
}, { output: [16, 16], graphical: true });

// world starts as a glider in the top-left. Fly, little guy.
let current = world;
for (let g = 0; g < 8; g++) {
  current = step(current);
}
paint(current);
render(paint.canvas);
`,
      solutionCode: `// Two kernels, two jobs: step computes the world, paint shows it.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  let next = 0;
  if (self === 1 && (count === 2 || count === 3)) next = 1;
  if (self === 0 && count === 3) next = 1;
  return next;
}, { output: [16, 16] });

const paint = gpu.createKernel(function (cells) {
  const alive = cells[this.thread.y][this.thread.x];
  if (alive === 1) {
    this.color(0.2, 1, 0.4, 1);
  } else {
    this.color(0.05, 0.06, 0.09, 1);
  }
}, { output: [16, 16], graphical: true });

// world starts as a glider in the top-left. Fly, little guy.
let current = world;
for (let g = 0; g < 8; g++) {
  current = step(current);
}
paint(current);
render(paint.canvas);
`,
      inputs: () => ({ world: withCells(GLIDER) }),
      publicTests: [
        {
          name: 'the glider translates: four ticks move the whole pattern down-right by one',
          run: async ctx => {
            const numeric = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            ctx.assert(numeric, 'no numeric (non-graphical) step kernel found');
            let grid = withCells(GLIDER);
            for (let i = 0; i < 4; i++) grid = numeric(grid);
            expectGrid(ctx, grid, withCells(shiftCells(GLIDER, 1, 1)), 'glider after four ticks');
          },
        },
        {
          name: 'canvas is 16×16 and shows exactly the 5 glider cells lit green',
          run: async ctx => {
            ctx.assert(ctx.canvas, 'no canvas — did you call render(paint.canvas)?');
            ctx.assert(
              ctx.canvas.width === 16 && ctx.canvas.height === 16,
              `expected a 16×16 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(graphical, 'no graphical paint kernel found');
            // Repaint the 8-ticks-later world so the readback is fresh.
            graphical(refSteps(withCells(GLIDER), 8));
            const pixels = graphical.getPixels();
            let lit = 0;
            for (let i = 0; i < pixels.length; i += 4) {
              const g = pixels[i + 1];
              ctx.assert(
                g > 200 || g < 40,
                `pixel at byte ${i} is neither live-green nor dead-dark (green = ${g})`
              );
              if (g > 200) lit++;
            }
            ctx.assert(lit === 5, litHint(lit, 5, 16 * 16) ||
              `a glider is always 5 cells — found ${lit} lit pixels`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Step a blinker once with the numeric kernel, paint it, and count
            // exactly 3 lit pixels — then paint a block and count 4.
            const numeric = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(numeric && graphical, 'expected a numeric and a graphical kernel');
            const litCount = pixels => {
              let lit = 0;
              for (let i = 0; i < pixels.length; i += 4) {
                if (pixels[i + 1] > 200) lit++;
              }
              return lit;
            };
            graphical(numeric(withCells(BLINKER)));
            const blinkerLit = litCount(graphical.getPixels());
            ctx.assert(blinkerLit === 3, litHint(blinkerLit, 3, 16 * 16) ||
              'stepped blinker should light 3 pixels');
            graphical(withCells(BLOCK));
            const blockLit = litCount(graphical.getPixels());
            ctx.assert(blockLit === 4, litHint(blockLit, 4, 16 * 16) ||
              'block should light 4 pixels');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'any-rule',
      title: 'One Kernel, Every Universe',
      intro: `<p>B3/S23 is one point in a whole family. Any <em>outer-totalistic</em> rule is fully
        described by two 9-entry tables: <code>born[n]</code> — does a dead cell with n live
        neighbors come alive? — and <code>stay[n]</code> — does a live cell with n survive? Conway
        is <code>born[3] = 1</code>, <code>stay[2] = stay[3] = 1</code>, zeros everywhere else.
        <strong>HighLife</strong> adds <code>born[6] = 1</code> and suddenly the world contains a
        pattern that builds copies of itself.</p>
        <p>Here's the move that matters: pass the tables <strong>as kernel arguments</strong>.
        The rulebook stops being code and becomes data — one compiled kernel runs every universe
        in the family, and switching physics is just passing different arrays. No <code>if</code>
        per rule, no recompile: alive cells look up <code>stay[count]</code>, dead cells look up
        <code>born[count]</code>.</p>`,
      goal: `<strong>Goal:</strong> finish the <code>evolve</code> kernel so it applies whatever
        rule tables it's handed — then let the wired-up code count where Life and HighLife disagree
        about the same world's next tick.`,
      requirements: [
        'The kernel takes <code>world</code>, <code>born</code> and <code>stay</code> — don\'t hard-code any rule',
        'Dead cells (<code>self === 0</code>) return <code>born[count]</code>',
        'Live cells (<code>self === 1</code>) return <code>stay[count]</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — arrays index like anywhere else',
          body: `<p><code>count</code> is a number from 0 to 8, and <code>born</code> is a 9-entry
            array — <code>born[count]</code> is already the answer for a dead cell. The lookup
            <em>is</em> the rule.</p>`,
        },
        {
          title: 'Hint 2 — a single return',
          body: `<pre><code>let fate = born[count];
if (self === 1) fate = stay[count];
return fate;</code></pre>`,
        },
      ],
      transfer: `Shipping small lookup tables to a fixed kernel instead of recompiling is how GPUs
        stay fast when behavior changes: CUDA and ROCm keep them in <code>__constant__</code>
        memory, WebGPU and Metal bind them as uniform buffers — same shader, new physics, zero
        pipeline rebuilds.`,
      starterCode: `// The rulebook as data: born[n] and stay[n] answer every question
// a cell can ask. One kernel, any Life-like universe.
const gpu = new GPU({ mode });

const evolve = gpu.createKernel(function (world, born, stay) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  // TODO: no rule logic — just look the answer up.
  // Dead cells consult born[count]; live cells consult stay[count].
  return self;
}, { output: [16, 16] });

// The same world, two different laws of physics:
const life = evolve(world, lifeBorn, lifeStay);
const high = evolve(world, highlifeBorn, lifeStay);

let differ = 0;
for (let y = 0; y < 16; y++) {
  for (let x = 0; x < 16; x++) {
    if (life[y][x] !== high[y][x]) differ++;
  }
}
console.log('Life and HighLife disagree on ' + differ + ' cells after one tick');
`,
      solutionCode: `// The rulebook as data: born[n] and stay[n] answer every question
// a cell can ask. One kernel, any Life-like universe.
const gpu = new GPU({ mode });

const evolve = gpu.createKernel(function (world, born, stay) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  let fate = born[count];
  if (self === 1) fate = stay[count];
  return fate;
}, { output: [16, 16] });

// The same world, two different laws of physics:
const life = evolve(world, lifeBorn, lifeStay);
const high = evolve(world, highlifeBorn, lifeStay);

let differ = 0;
for (let y = 0; y < 16; y++) {
  for (let x = 0; x < 16; x++) {
    if (life[y][x] !== high[y][x]) differ++;
  }
}
console.log('Life and HighLife disagree on ' + differ + ' cells after one tick');
`,
      inputs: utils => ({
        world: randomGrid(utils, 7707),
        lifeBorn: B3.slice(),
        lifeStay: S23.slice(),
        highlifeBorn: B36.slice(),
      }),
      publicTests: [
        {
          name: 'fed the Life tables, it is still Life: the blinker spins',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const start = withCells(BLINKER);
            const next = ctx.kernel(start, B3, S23);
            expectGrid(ctx, next, withCells([[6, 7], [7, 7], [8, 7]]), 'blinker under B3/S23',
              ruleHint(next, start, B3, S23));
          },
        },
        {
          name: "HighLife's B6: six neighbors ignite a dead cell that Life leaves dark",
          run: async ctx => {
            // A dead cell at (5,5) ringed by exactly 6 live neighbors.
            const ring = withCells([[4, 4], [4, 5], [4, 6], [5, 4], [5, 6], [6, 4]]);
            const life = ctx.kernel(ring, B3, S23);
            const high = ctx.kernel(ring, B36, S23);
            ctx.assertClose(life[5][5], 0, 1e-3, ruleHint(life, ring, B3, S23) ||
              'under Life (B3), 6 neighbors do not give birth');
            ctx.assertClose(high[5][5], 1, 1e-3, ruleHint(high, ring, B36, S23) ||
              'under HighLife (B36), 6 neighbors do');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Day & Night (B3678/S34678) on a random world — the kernel must
            // honor tables it has never seen before.
            const grid = randomGrid(ctx.utils, 8808, SIZE, 0.5);
            const next = ctx.kernel(grid, B3678, S34678);
            expectGrid(ctx, next, refStep(grid, B3678, S34678), 'Day & Night, one tick',
              ruleHint(next, grid, B3678, S34678));
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            const grid = randomGrid(ctx.utils, 9909);
            const next = ctx.kernel(grid, B36, S23);
            expectGrid(ctx, next, refStep(grid, B36, S23), 'HighLife, one tick',
              ruleHint(next, grid, B36, S23));
          },
        },
      ],
    },
  ],
};
