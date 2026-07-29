// Module 3.4 — Reaction–Diffusion.
//
// Four tasks: the 5-point Laplacian as a wrap-around gather kernel → one
// Gray–Scott update step for the U and V grids (two kernels sharing a
// snapshot) → iterating steps JS-side with a ping-pong loop → painting the
// V field with a graphical kernel as the payoff.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// only numbers / nested number arrays as inputs, this.thread.* for indexing,
// this.constants.* for compile-time values, graphical kernels use
// this.color(). Every task passes in CPU mode.

// Gray–Scott parameters used by every kernel and every reference below.
// Chosen (and numerically checked) so an explicit-Euler step with a 5-point
// Laplacian stays stable and the pattern visibly grows within ~200 steps.
const GS = { du: 0.2, dv: 0.1, f: 0.035, k: 0.06, dt: 1.0 };

function makeGrid(size, value) {
  const grid = new Array(size);
  for (let y = 0; y < size; y++) grid[y] = new Array(size).fill(value);
  return grid;
}

// Deterministic bumpy field for the Laplacian task (shared by inputs/tests).
function randomGrid(utils, size, seed) {
  const rand = utils.seededRandom(seed);
  const grid = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    for (let x = 0; x < size; x++) row[x] = Math.round(rand() * 1000) / 1000;
    grid[y] = row;
  }
  return grid;
}

// CPU reference: 5-point Laplacian on a torus (wrap-around edges).
function laplacianRef(grid) {
  const size = grid.length;
  const out = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    const yu = y === 0 ? size - 1 : y - 1;
    const yd = y === size - 1 ? 0 : y + 1;
    for (let x = 0; x < size; x++) {
      const xl = x === 0 ? size - 1 : x - 1;
      const xr = x === size - 1 ? 0 : x + 1;
      row[x] = grid[y][xl] + grid[y][xr] + grid[yu][x] + grid[yd][x] - 4 * grid[y][x];
    }
    out[y] = row;
  }
  return out;
}

// CPU reference: one full Gray–Scott step → [newU, newV].
function gsStepRef(u, v) {
  const size = u.length;
  const lapU = laplacianRef(u);
  const lapV = laplacianRef(v);
  const nu = new Array(size);
  const nv = new Array(size);
  for (let y = 0; y < size; y++) {
    nu[y] = new Array(size);
    nv[y] = new Array(size);
    for (let x = 0; x < size; x++) {
      const uc = u[y][x];
      const vc = v[y][x];
      const uvv = uc * vc * vc;
      nu[y][x] = uc + (GS.du * lapU[y][x] - uvv + GS.f * (1 - uc)) * GS.dt;
      nv[y][x] = vc + (GS.dv * lapV[y][x] + uvv - (GS.f + GS.k) * vc) * GS.dt;
    }
  }
  return [nu, nv];
}

function gsRunRef(u, v, steps) {
  for (let i = 0; i < steps; i++) [u, v] = gsStepRef(u, v);
  return [u, v];
}

// Standard seed: calm ocean (U=1, V=0) with a center square of chemical V.
function makeSeedPair(size, block) {
  const u = makeGrid(size, 1);
  const v = makeGrid(size, 0);
  const lo = (size - block) >> 1;
  for (let y = lo; y < lo + block; y++) {
    for (let x = lo; x < lo + block; x++) {
      u[y][x] = 0.5;
      v[y][x] = 0.25;
    }
  }
  return { u, v };
}

function spikeGrid(size, y, x, value) {
  const grid = makeGrid(size, 0);
  grid[y][x] = value;
  return grid;
}

export default {
  id: '3-4',
  track: 3,
  title: 'Reaction–Diffusion',
  blurb: 'Two chemicals, two equations, and suddenly: coral, fingerprints, leopard spots.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'laplacian',
      title: 'The Laplacian: Ask Your Neighbors',
      intro: `<p>Diffusion is gossip: every cell drifts toward the average of its neighbors.
        The operator that measures "how far am I from my neighbors' average" is the
        <strong>Laplacian</strong>, and on a grid it's a five-read gather:
        <code>left + right + up + down − 4·center</code>. Positive means the neighbors are
        higher and stuff will flow in; negative means it flows out.</p>
        <p>One wrinkle: simulations hate edges. Instead of clamping like the convolution
        filters in track 2, we <strong>wrap around</strong> — the left neighbor of column 0 is
        column 31. The world becomes a torus and every cell has exactly four neighbors,
        no special cases.</p>`,
      goal: `<strong>Goal:</strong> complete the gather kernel so it returns the 5-point
        Laplacian of <code>field</code> with wrap-around edges.`,
      requirements: [
        'Wrap all four neighbor indexes — below <code>0</code> becomes <code>size − 1</code>, past <code>size − 1</code> becomes <code>0</code>',
        'Read exactly five cells: the four direct neighbors and the center',
        'Return <code>left + right + up + down − 4·center</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the wrap is an if',
          body: `<p>Same trick as clamping, different else:
            <code>let xr = this.thread.x + 1; if (xr &gt; this.constants.size - 1) xr = 0;</code>
            The starter already wrote <code>xl</code> for you — mirror it three times.</p>`,
        },
        {
          title: 'Hint 2 — five reads',
          body: `<p>The neighbors sit at <code>field[y][xl]</code>, <code>field[y][xr]</code>,
            <code>field[yd][x]</code> and <code>field[yu][x]</code> — only ever vary
            <em>one</em> coordinate at a time. The center is <code>field[y][x]</code>.</p>`,
        },
        {
          title: 'Hint 3 — the whole return',
          body: `<p><code>return field[y][xl] + field[y][xr] + field[yd][x] + field[yu][x]
            - 4 * field[y][x];</code></p>`,
        },
      ],
      transfer: `The 5-point Laplacian stencil is the beating heart of PDE solvers on every
        platform — heat, waves, pressure projection in fluids. On big CUDA/ROCm clusters the
        wrap you just wrote becomes a <em>halo exchange</em>: each GPU ships its border rows to
        the neighbor that needs them before every step.`,
      starterCode: `// The Laplacian: how far is each cell from its neighbors' average?
// The world is a torus — indexes wrap around the edges.
const gpu = new GPU({ mode });

const laplacian = gpu.createKernel(function (field) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1;
  if (xl < 0) xl = this.constants.size - 1;
  // TODO: wrap xr (right), yu (up) and yd (down) the same way,
  // then return left + right + up + down - 4 * center.
  return 0;
}, { output: [32, 32], constants: { size: 32 } });

const result = laplacian(field);
console.log('at a bump:', result[16][16]);
`,
      solutionCode: `// The Laplacian: how far is each cell from its neighbors' average?
// The world is a torus — indexes wrap around the edges.
const gpu = new GPU({ mode });

const laplacian = gpu.createKernel(function (field) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1;
  if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1;
  if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1;
  if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1;
  if (yu > this.constants.size - 1) yu = 0;
  return field[y][xl] + field[y][xr] + field[yd][x] + field[yu][x] - 4 * field[y][x];
}, { output: [32, 32], constants: { size: 32 } });

const result = laplacian(field);
console.log('at a bump:', result[16][16]);
`,
      inputs: utils => ({ field: randomGrid(utils, 32, 3401) }),
      publicTests: [
        {
          name: 'a uniform field has zero Laplacian everywhere',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(makeGrid(32, 0.7));
            ctx.assert(out && out.length === 32 && out[0].length === 32, 'expected a 32×32 result');
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                ctx.assertClose(out[y][x], 0, 1e-4, `cell [${y}][${x}] of a flat field`);
              }
            }
          },
        },
        {
          name: 'a single spike: <code>−4·s</code> at the peak, <code>+s</code> on each neighbor — even across the wrap',
          run: async ctx => {
            const out = ctx.kernel(spikeGrid(32, 0, 0, 2));
            ctx.assertClose(out[0][0], -8, 1e-4, 'the peak itself (−4 × 2)');
            ctx.assertClose(out[0][1], 2, 1e-4, 'right neighbor');
            ctx.assertClose(out[1][0], 2, 1e-4, 'neighbor above');
            ctx.assertClose(out[0][31], 2, 1e-4, 'LEFT neighbor — wraps to column 31');
            ctx.assertClose(out[31][0], 2, 1e-4, 'neighbor below — wraps to row 31');
            ctx.assertClose(out[5][5], 0, 1e-4, 'a far-away cell');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Full comparison against a CPU reference on a fresh random field,
            // plus the torus conservation law: the Laplacian sums to zero.
            const field = randomGrid(ctx.utils, 32, 909);
            const out = ctx.kernel(field);
            const ref = laplacianRef(field);
            let sum = 0;
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-3, `cell [${y}][${x}]`);
                sum += out[y][x];
              }
            }
            ctx.assertClose(sum, 0, 1e-2, 'on a torus, gains and losses cancel exactly');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'gray-scott-step',
      title: 'One Step of Gray–Scott',
      intro: `<p>Now the chemistry. Gray–Scott tracks two chemicals on the same grid:
        <code>U</code> (food, fed in everywhere) and <code>V</code> (the eater —
        <code>U + 2V → 3V</code>, so V converts U into more V, and is itself slowly removed).
        Per cell, per step:</p>
        <p><code>u' = u + (Du·∇²u − u·v² + F·(1 − u))·dt</code><br>
        <code>v' = v + (Dv·∇²v + u·v² − (F + K)·v)·dt</code></p>
        <p>Each equation is your task-1 Laplacian plus three pointwise terms — diffusion,
        reaction, feed/kill. One kernel per chemical: both are gathers over the <em>old</em>
        grids, so all 1,024 cells of a step can run in parallel.</p>`,
      goal: `<strong>Goal:</strong> finish the two update kernels — <code>stepU</code> and
        <code>stepV</code> each return their chemical's next value. The Laplacians are already
        gathered for you.`,
      requirements: [
        'Keep the kernel order as wired: <code>stepU</code> first, then <code>stepV</code>',
        'The reaction term is <code>u·v·v</code> — U loses it, V gains it',
        '<code>stepU</code> returns <code>uc + (du·lap − uc·vc·vc + f·(1 − uc))·dt</code>',
        '<code>stepV</code> returns <code>vc + (dv·lap + uc·vc·vc − (f + k)·vc)·dt</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — everything is already in scope',
          body: `<p><code>lap</code>, <code>uc</code> and <code>vc</code> are computed for you;
            the parameters live in <code>this.constants</code> (<code>du</code>, <code>f</code>,
            <code>dt</code> in stepU; <code>dv</code>, <code>f</code>, <code>k</code>,
            <code>dt</code> in stepV). The TODO is one <code>return</code> per kernel.</p>`,
        },
        {
          title: 'Hint 2 — stepU, spelled out',
          body: `<p><code>return uc + (this.constants.du * lap - uc * vc * vc +
            this.constants.f * (1 - uc)) * this.constants.dt;</code> — stepV is the same shape
            with <code>+ uc·vc·vc</code> and <code>− (f + k)·vc</code>.</p>`,
        },
      ],
      transfer: `Fusing the stencil and the pointwise chemistry into one kernel is a classic
        GPU move — in CUDA or a WGSL compute shader you'd do exactly this to touch each grid
        cell's memory once per step instead of once per term. Separate passes per term would
        triple the bandwidth bill.`,
      starterCode: `// Two chemicals, two kernels. Both read the OLD u and v grids.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  // TODO: return uc + (du * lap - uc*vc*vc + f * (1 - uc)) * dt
  //       (parameters live in this.constants)
  return uc;
}, { output: [32, 32], constants: { size: 32, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  // TODO: return vc + (dv * lap + uc*vc*vc - (f + k) * vc) * dt
  return vc;
}, { output: [32, 32], constants: { size: 32, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const newU = stepU(u0, v0);
const newV = stepV(u0, v0);
console.log('center after one step — U:', newU[16][16], ' V:', newV[16][16]);
`,
      solutionCode: `// Two chemicals, two kernels. Both read the OLD u and v grids.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  return uc + (this.constants.du * lap - uc * vc * vc
    + this.constants.f * (1 - uc)) * this.constants.dt;
}, { output: [32, 32], constants: { size: 32, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  return vc + (this.constants.dv * lap + uc * vc * vc
    - (this.constants.f + this.constants.k) * vc) * this.constants.dt;
}, { output: [32, 32], constants: { size: 32, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const newU = stepU(u0, v0);
const newV = stepV(u0, v0);
console.log('center after one step — U:', newU[16][16], ' V:', newV[16][16]);
`,
      inputs: () => {
        const seed = makeSeedPair(32, 6);
        return { u0: seed.u, v0: seed.v };
      },
      publicTests: [
        {
          name: 'the calm ocean is a fixed point: U=1, V=0 stays exactly put',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels (stepU then stepV), found ${ctx.kernels.length}`);
            const u = makeGrid(32, 1);
            const v = makeGrid(32, 0);
            const newU = ctx.kernels[0](u, v);
            const newV = ctx.kernels[1](u, v);
            for (let y = 0; y < 32; y += 5) {
              for (let x = 0; x < 32; x += 5) {
                ctx.assertClose(newU[y][x], 1, 1e-5, `U at [${y}][${x}] — nothing to react, nothing to feed`);
                ctx.assertClose(newV[y][x], 0, 1e-5, `V at [${y}][${x}] — no V, no reaction`);
              }
            }
          },
        },
        {
          name: 'a well-mixed beaker (u=0.6, v=0.3) follows the equations exactly',
          run: async ctx => {
            // Uniform grids → the Laplacian vanishes, so the answer is analytic.
            const u = makeGrid(32, 0.6);
            const v = makeGrid(32, 0.3);
            const uvv = 0.6 * 0.3 * 0.3;
            const expectedU = 0.6 + (-uvv + GS.f * (1 - 0.6)) * GS.dt;
            const expectedV = 0.3 + (uvv - (GS.f + GS.k) * 0.3) * GS.dt;
            const newU = ctx.kernels[0](u, v);
            const newV = ctx.kernels[1](u, v);
            for (let y = 0; y < 32; y += 7) {
              for (let x = 0; x < 32; x += 7) {
                ctx.assertClose(newU[y][x], expectedU, 1e-4, `U at [${y}][${x}]`);
                ctx.assertClose(newV[y][x], expectedV, 1e-4, `V at [${y}][${x}]`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Another uniform pair, analytic again.
            const u = makeGrid(32, 0.8);
            const v = makeGrid(32, 0.1);
            const uvv = 0.8 * 0.1 * 0.1;
            const expectedU = 0.8 + (-uvv + GS.f * (1 - 0.8)) * GS.dt;
            const expectedV = 0.1 + (uvv - (GS.f + GS.k) * 0.1) * GS.dt;
            const newU = ctx.kernels[0](u, v);
            const newV = ctx.kernels[1](u, v);
            for (let y = 0; y < 32; y += 3) {
              for (let x = 0; x < 32; x += 3) {
                ctx.assertClose(newU[y][x], expectedU, 1e-4, `U at [${y}][${x}]`);
                ctx.assertClose(newV[y][x], expectedV, 1e-4, `V at [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A seeded (non-uniform) grid: diffusion and reaction together,
            // checked cell-for-cell against the CPU reference.
            const seed = makeSeedPair(32, 10);
            const [refU, refV] = gsStepRef(seed.u, seed.v);
            const newU = ctx.kernels[0](seed.u, seed.v);
            const newV = ctx.kernels[1](seed.u, seed.v);
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                ctx.assertClose(newU[y][x], refU[y][x], 1e-4, `U at [${y}][${x}]`);
                ctx.assertClose(newV[y][x], refV[y][x], 1e-4, `V at [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'feed-it-back',
      title: 'Feed It Back: 100 Steps',
      intro: `<p>One step is chemistry; a hundred steps is <em>morphogenesis</em>. The kernels
        stay on the GPU — the loop lives in JavaScript: call both step kernels, take their
        outputs, feed them back in as next step's inputs. This is the same feedback move as
        the cellular automata in 3.3, just with two grids in flight instead of one.</p>
        <p>The trap: both kernels must read the <strong>same snapshot</strong>. If you
        overwrite <code>u</code> before calling <code>stepV</code>, chemical V reacts with food
        from the <em>future</em> — the simulation drifts and the tests will know. Hold both new
        grids, <em>then</em> swap. Graphics folk call this ping-pong buffering.</p>`,
      goal: `<strong>Goal:</strong> run 100 Gray–Scott steps from the seeded grids
        <code>seedU</code> / <code>seedV</code>, feeding each step's outputs into the next —
        both kernels always reading the same snapshot.`,
      requirements: [
        'Loop exactly <code>STEPS</code> (100) times in plain JavaScript',
        'Call <code>stepU(u, v)</code> and <code>stepV(u, v)</code> with the <em>same</em> <code>u</code> and <code>v</code>',
        'Only after both calls, replace <code>u</code> and <code>v</code> with the new grids',
      ],
      hints: [
        {
          title: 'Hint 1 — why the starter is wrong',
          body: `<p>The starter does <code>u = stepU(u, v); v = stepV(u, v);</code> — by the
            second call, <code>u</code> is already next step's grid. Stash both results in
            temporaries before assigning either.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<p><code>const nextU = stepU(u, v);<br>const nextV = stepV(u, v);<br>
            u = nextU;<br>v = nextV;</code> — four lines, inside
            <code>for (let i = 0; i &lt; STEPS; i++)</code>.</p>`,
        },
      ],
      transfer: `This snapshot discipline is double buffering, and GPUs institutionalize it:
        a WebGPU or Metal simulation binds texture A for reading and texture B for writing,
        then swaps the bindings each frame — you never write the buffer you're reading. CUDA
        codes do the same by swapping two device pointers between kernel launches.`,
      starterCode: `// The kernels from last task, prewired at 48×48. Your job: the time loop.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  return uc + (this.constants.du * lap - uc * vc * vc
    + this.constants.f * (1 - uc)) * this.constants.dt;
}, { output: [48, 48], constants: { size: 48, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  return vc + (this.constants.dv * lap + uc * vc * vc
    - (this.constants.f + this.constants.k) * vc) * this.constants.dt;
}, { output: [48, 48], constants: { size: 48, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const STEPS = 100;
let u = seedU;
let v = seedV;

// TODO: run STEPS steps. This single "step" has TWO bugs: it only runs
// once, and stepV reads the u we just overwrote — future food!
u = stepU(u, v);
v = stepV(u, v);

console.log('center V after', STEPS, 'steps:', v[24][24]);
`,
      solutionCode: `// The kernels from last task, prewired at 48×48. Your job: the time loop.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  return uc + (this.constants.du * lap - uc * vc * vc
    + this.constants.f * (1 - uc)) * this.constants.dt;
}, { output: [48, 48], constants: { size: 48, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  return vc + (this.constants.dv * lap + uc * vc * vc
    - (this.constants.f + this.constants.k) * vc) * this.constants.dt;
}, { output: [48, 48], constants: { size: 48, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const STEPS = 100;
let u = seedU;
let v = seedV;

for (let i = 0; i < STEPS; i++) {
  // Both kernels read the same snapshot; swap only after both are done.
  const nextU = stepU(u, v);
  const nextV = stepV(u, v);
  u = nextU;
  v = nextV;
}

console.log('center V after', STEPS, 'steps:', v[24][24]);
`,
      inputs: () => {
        const seed = makeSeedPair(48, 8);
        return { seedU: seed.u, seedV: seed.v };
      },
      publicTests: [
        {
          name: 'after 100 steps the kernels are seeing step 99, not the seed',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels (stepU then stepV), found ${ctx.kernels.length}`);
            const seed = makeSeedPair(48, 8);
            const [refU, refV] = gsRunRef(seed.u, seed.v, 99);
            const lastArgs = ctx.kernels[0].lastArgs;
            ctx.assert(lastArgs && lastArgs.length >= 2, 'stepU should have been called with (u, v)');
            const [u, v] = lastArgs;
            ctx.assert(
              Math.abs(u[24][24] - seed.u[24][24]) > 0.05,
              'the last stepU call still saw the seed — did the loop actually feed results back?'
            );
            const cells = [[24, 24], [24, 20], [20, 28], [24, 12], [4, 4]];
            for (const [y, x] of cells) {
              ctx.assertClose(u[y][x], refU[y][x], 2e-3, `U at [${y}][${x}] after 99 steps`);
              ctx.assertClose(v[y][x], refV[y][x], 2e-3, `V at [${y}][${x}] after 99 steps`);
            }
          },
        },
        {
          name: 'stepU and stepV read the same snapshot — no future food',
          run: async ctx => {
            const uSeenByU = ctx.kernels[0].lastArgs[0];
            const uSeenByV = ctx.kernels[1].lastArgs[0];
            ctx.assert(uSeenByU && uSeenByV, 'both kernels should have been called with (u, v)');
            const cells = [[24, 24], [24, 21], [27, 24], [21, 27], [24, 16]];
            for (const [y, x] of cells) {
              ctx.assertClose(
                uSeenByV[y][x], uSeenByU[y][x], 1e-4,
                `u[${y}][${x}] differs between the stepU and stepV calls — swap only after BOTH kernels ran`
              );
            }
          },
        },
        {
          name: 'V has escaped the seed square, and both fields stay in [0, 1]',
          run: async ctx => {
            const [u, v] = ctx.kernels[0].lastArgs;
            ctx.assert(
              v[24][12] > 1e-4,
              'V should have diffused well outside the 8×8 seed by step 99'
            );
            for (let y = 0; y < 48; y += 3) {
              for (let x = 0; x < 48; x += 3) {
                ctx.assert(u[y][x] >= -1e-6 && u[y][x] <= 1 + 1e-6, `U at [${y}][${x}] left [0, 1] — unstable loop?`);
                ctx.assert(v[y][x] >= -1e-6 && v[y][x] <= 1 + 1e-6, `V at [${y}][${x}] left [0, 1] — unstable loop?`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Fresh seed (different block size), 40 steps driven by the test,
            // full-grid comparison against the CPU reference.
            const seed = makeSeedPair(48, 12);
            let u = seed.u;
            let v = seed.v;
            for (let i = 0; i < 40; i++) {
              const nextU = ctx.kernels[0](u, v);
              const nextV = ctx.kernels[1](u, v);
              u = nextU;
              v = nextV;
            }
            const [refU, refV] = gsRunRef(seed.u, seed.v, 40);
            for (let y = 0; y < 48; y++) {
              for (let x = 0; x < 48; x++) {
                ctx.assertClose(u[y][x], refU[y][x], 1e-3, `U at [${y}][${x}] after 40 steps`);
                ctx.assertClose(v[y][x], refV[y][x], 1e-3, `V at [${y}][${x}] after 40 steps`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'paint-the-pattern',
      title: 'Paint the Pattern',
      intro: `<p>Payoff time. The whole simulation is wired below — 64×64 grid, 200 steps —
        and it ends holding <code>v</code>, a grid of numbers with coral growing in it.
        Numbers deserve pixels: one graphical kernel, exactly like the painters in 3.1,
        turns the V field into the picture the module cover promised.</p>
        <p>The palette is fixed so we can test it: brightness
        <code>t = min(1, v·2.5)</code>, painted as
        <code>color(t, t·t, 0.25 + 0.75·t)</code> — a deep-blue ocean at <code>v = 0</code>
        rising through violet to white-hot at the pattern's crest.</p>`,
      goal: `<strong>Goal:</strong> complete the <code>paint</code> kernel — map this thread's
        <code>v</code> value through the palette and put it on screen.`,
      requirements: [
        'Read this thread\'s value: <code>v[this.thread.y][this.thread.x]</code>',
        'Brightness <code>t = Math.min(1, value * 2.5)</code>',
        'Paint <code>this.color(t, t * t, 0.25 + 0.75 * t, 1)</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — same move as the luminance painter',
          body: `<p>This is the paint kernel from track 1 with a fancier ramp: read one number
            from the grid, compute the channels, call <code>this.color()</code>.
            <code>Math.min</code> works inside kernels.</p>`,
        },
        {
          title: 'Hint 2 — the whole body',
          body: `<p><code>const t = Math.min(1, v[this.thread.y][this.thread.x] * 2.5);<br>
            this.color(t, t * t, 0.25 + 0.75 * t, 1);</code></p>`,
        },
      ],
      transfer: `Compute passes that end in a draw are the shape of every GPU simulation you've
        seen on the web: WebGPU chains compute pipelines into a render pipeline whose fragment
        shader is your <code>paint</code>; Metal apps do the same with a compute encoder feeding
        a fragment function. The data never has to leave the card.`,
      starterCode: `// 200 steps of Gray–Scott, then paint the V field. The sim is done —
// the painter is yours.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  return uc + (this.constants.du * lap - uc * vc * vc
    + this.constants.f * (1 - uc)) * this.constants.dt;
}, { output: [64, 64], constants: { size: 64, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  return vc + (this.constants.dv * lap + uc * vc * vc
    - (this.constants.f + this.constants.k) * vc) * this.constants.dt;
}, { output: [64, 64], constants: { size: 64, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const paint = gpu.createKernel(function (v) {
  // TODO: t = Math.min(1, value * 2.5), then
  // this.color(t, t * t, 0.25 + 0.75 * t, 1)
  this.color(1, 0, 1, 1);
}, { output: [64, 64], graphical: true });

let u = seedU;
let v = seedV;
for (let i = 0; i < 200; i++) {
  const nextU = stepU(u, v);
  const nextV = stepV(u, v);
  u = nextU;
  v = nextV;
}

paint(v);
render(paint.canvas);
`,
      solutionCode: `// 200 steps of Gray–Scott, then paint the V field. The sim is done —
// the painter is yours.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  return uc + (this.constants.du * lap - uc * vc * vc
    + this.constants.f * (1 - uc)) * this.constants.dt;
}, { output: [64, 64], constants: { size: 64, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  return vc + (this.constants.dv * lap + uc * vc * vc
    - (this.constants.f + this.constants.k) * vc) * this.constants.dt;
}, { output: [64, 64], constants: { size: 64, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const paint = gpu.createKernel(function (v) {
  const t = Math.min(1, v[this.thread.y][this.thread.x] * 2.5);
  this.color(t, t * t, 0.25 + 0.75 * t, 1);
}, { output: [64, 64], graphical: true });

let u = seedU;
let v = seedV;
for (let i = 0; i < 200; i++) {
  const nextU = stepU(u, v);
  const nextV = stepV(u, v);
  u = nextU;
  v = nextV;
}

paint(v);
render(paint.canvas);
`,
      inputs: () => {
        const seed = makeSeedPair(64, 8);
        return { seedU: seed.u, seedV: seed.v };
      },
      publicTests: [
        {
          name: 'a 64×64 canvas is rendered',
          run: async ctx => {
            ctx.assert(ctx.canvas, 'no canvas — is paint graphical: true, and did you call render()?');
            ctx.assert(
              ctx.canvas.width === 64 && ctx.canvas.height === 64,
              `expected a 64×64 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            const pixels = ctx.getPixels();
            ctx.assert(pixels.length === 64 * 64 * 4, 'pixel buffer should hold 64×64 RGBA values');
          },
        },
        {
          name: 'the palette is exact: still water is deep blue, <code>v = 0.2</code> is half-lit violet',
          run: async ctx => {
            const paint = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(paint, 'no graphical kernel found');
            paint(makeGrid(64, 0));
            let pixels = paint.getPixels();
            for (let i = 0; i < pixels.length; i += 331 * 4) {
              ctx.assertClose(pixels[i], 0, 3, `red at byte ${i} for v = 0`);
              ctx.assertClose(pixels[i + 1], 0, 3, `green at byte ${i} for v = 0`);
              ctx.assertClose(pixels[i + 2], 0.25 * 255, 3, `blue at byte ${i} for v = 0`);
            }
            paint(makeGrid(64, 0.2)); // t = 0.5
            pixels = paint.getPixels();
            for (let i = 0; i < pixels.length; i += 331 * 4) {
              ctx.assertClose(pixels[i], 0.5 * 255, 3, `red at byte ${i} for v = 0.2`);
              ctx.assertClose(pixels[i + 1], 0.25 * 255, 3, `green at byte ${i} for v = 0.2`);
              ctx.assertClose(pixels[i + 2], 0.625 * 255, 3, `blue at byte ${i} for v = 0.2`);
            }
          },
        },
        {
          name: 'the picture is alive — bright coral on a dark ocean',
          run: async ctx => {
            // Repaint from a reference 200-step simulation so this test does
            // not depend on what other tests left on the canvas.
            const paint = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(paint, 'no graphical kernel found');
            const seed = makeSeedPair(64, 8);
            const [, refV] = gsRunRef(seed.u, seed.v, 200);
            paint(refV);
            const pixels = paint.getPixels();
            let bright = 0;
            let dark = 0;
            for (let i = 0; i < pixels.length; i += 4) {
              if (pixels[i] > 150) bright++;
              if (pixels[i] < 20) dark++;
            }
            ctx.assert(
              bright >= 50,
              `expected at least 50 bright pattern pixels after 200 steps, found ${bright}`
            );
            ctx.assert(
              dark >= 1000,
              `expected a mostly-dark ocean around the pattern, found only ${dark} dark pixels`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Palette spot-checks at values the public tests never use,
            // including the clamp at t = 1.
            const paint = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(paint, 'no graphical kernel found');
            paint(makeGrid(64, 0.6)); // t clamps to 1 → pure white
            let pixels = paint.getPixels();
            for (let i = 0; i < pixels.length; i += 449 * 4) {
              ctx.assertClose(pixels[i], 255, 3, `red at byte ${i} for v = 0.6`);
              ctx.assertClose(pixels[i + 1], 255, 3, `green at byte ${i} for v = 0.6`);
              ctx.assertClose(pixels[i + 2], 255, 3, `blue at byte ${i} for v = 0.6`);
            }
            paint(makeGrid(64, 0.1)); // t = 0.25
            pixels = paint.getPixels();
            for (let i = 0; i < pixels.length; i += 449 * 4) {
              ctx.assertClose(pixels[i], 0.25 * 255, 3, `red at byte ${i} for v = 0.1`);
              ctx.assertClose(pixels[i + 1], 0.0625 * 255, 3, `green at byte ${i} for v = 0.1`);
              ctx.assertClose(pixels[i + 2], 0.4375 * 255, 3, `blue at byte ${i} for v = 0.1`);
            }
          },
        },
      ],
    },
  ],
};
