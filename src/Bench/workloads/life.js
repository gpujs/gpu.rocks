/**
 * Conway's Life on a torus — the plainest possible iterated stencil.
 *
 * Every cell reads its eight neighbours and writes one bit, which makes this
 * the cleanest thing in the table to reason about: no reduction, no ordering
 * inside a generation, no floating point. What it prices is the SHAPE of an
 * iterated simulation rather than its arithmetic.
 *
 * A generation cannot be done in place — a cell's new value depends on its
 * neighbours' old values — so the state has to live in two buffers that swap
 * roles each generation. That is the ping-pong, and it is the reason this row
 * exists next to matmul. Matmul is one dispatch of enormous arithmetic
 * intensity; this is a hundred dispatches of nine loads and a compare, with a
 * dependency between every pair of them. Per byte moved there is almost
 * nothing to compute, so the row measures memory bandwidth and the cost of a
 * dependent dispatch chain, which is what most real simulation kernels
 * actually run into.
 *
 * WHICH DISPATCH COST, AND WHY THE gpu.js COLUMN IS NOW FUSED. There are two
 * different prices hiding in the phrase "dispatch overhead", and this row only
 * ever wanted one of them. The first is the HOST-side cost of asking: a
 * runtime validating arguments, rebinding textures and resolving a promise in
 * JavaScript, 96 times. The second is what the device charges to run 96
 * dependent passes back to back — the barrier between each pair, and the fact
 * that no generation can start until the last one's 16 MB has landed. Only the
 * second is a property of the simulation; the first is a property of whichever
 * runtime you happened to call.
 *
 * The bare-WebGPU column has never paid the first — it has always recorded all
 * 96 dispatches into ONE command buffer (see its header) — so for as long as
 * the gpu.js column awaited each generation separately, the gap between the
 * two columns was a gpu.js call-overhead measurement wearing a stencil as a
 * hat. The gpu.js column now traces the whole simulation with
 * `gpu.createPipeline`, which records the 96 generations into one plan and
 * launches it once, so both GPU columns are finally answering the same
 * question and the row is left measuring bandwidth and the dependency chain.
 *
 * NOTHING ABOUT THE WORK CHANGED. Same kernel body, same nine loads, same 96
 * generations, same two buffers alternating, same single read-back at the end.
 * The plan is a static unroll of the loop that was already there, not a
 * cleverer Life. If you want the host-side per-call price on its own it is
 * launch-overhead.js, which exists for exactly that and must never be fused —
 * one add per thread has nothing left in it once the launches are free.
 *
 * The keeping-honest details:
 *
 *   - the state is 0/1 integers, so every backend must agree EXACTLY. There is
 *     no rounding anywhere in Life; if two columns' checksums differ at all,
 *     one of them computed the wrong cells, and the runner's 1e-4 tolerance is
 *     slack this row never uses.
 *   - the torus wrap is done with comparisons rather than %, because a modulo
 *     on gpu.js's WebGL1 backend is a float division whose rounding is a
 *     driver's business, and an off-by-one on the wrap row would be a silent
 *     wrong answer rather than a loud one.
 *   - every run starts from the same seeded soup. A run that carried state
 *     forward would do different work each repetition and the median would be
 *     meaningless.
 */

const N = 2048;

// 96, not the round 256 you might expect. 2048 x 2048 x 256 generations is a
// six-second plain-JS baseline, twice the top of the sizing band, and a cell
// nobody would sit through. The grid is what makes this row interesting — 16 MB
// of state per buffer is well past any cache, which is exactly the regime a
// bandwidth-bound stencil should be measured in — so the generation count is
// what gives, not the grid. 96 generations is still far more dispatches than it
// takes to show the per-dispatch cost.
const GENS = 96;

// Live-cell density of the initial soup. 0.3 is high enough that the first few
// generations are dense and interesting and low enough that it settles into
// the usual still-lifes and blinkers rather than dying out.
const DENSITY = 0.3;

export default {
  id: 'life',
  name: "Conway's Life",
  params: `${N} × ${N} torus · ${GENS} generations`,
  tag: 'ping-pong stencil',
  group: 'sim',
  size: { n: N, gens: GENS },

  make({ n }) {
    // Deterministic, so that two columns are handed the same soup. Math.random
    // would make the row uncheckable rather than merely unreproducible.
    let s = 0x9e3779b9 >>> 0;
    const grid = new Uint8Array(n * n);
    const cut = DENSITY * 0x1000000;
    for (let i = 0; i < grid.length; i++) {
      s = (s * 1664525 + 1013904223) >>> 0;
      grid[i] = s >>> 8 < cut ? 1 : 0;
    }

    // gpu.js and WebGPU both want floats; the CPU oracle is happier with bytes,
    // which are a quarter of the cache footprint and hold 0/1 just as exactly.
    // Built here, once, so that no column pays for the conversion (rule 2).
    const flat = Float32Array.from(grid);
    const rows = [];
    for (let y = 0; y < n; y++) rows.push(flat.subarray(y * n, y * n + n));

    return { grid, flat, rows };
  },

  /**
   * The oracle, and a fair baseline. Two flat Uint8Arrays swapped by reference,
   * rows walked in order so the three row bases are hoisted out of the inner
   * loop, and the wrap handled by two comparisons per cell.
   *
   * Bytes rather than floats on purpose: 4 MB of state instead of 16 MB is the
   * version a JS programmer would actually write, and writing the slow one
   * would hand every GPU column a speed-up it had not earned.
   */
  js({ n, gens }, { grid }) {
    let a = new Uint8Array(grid); // fresh soup per run
    let b = new Uint8Array(n * n);

    for (let g = 0; g < gens; g++) {
      for (let y = 0; y < n; y++) {
        const up = (y === 0 ? n - 1 : y - 1) * n;
        const row = y * n;
        const down = (y === n - 1 ? 0 : y + 1) * n;
        for (let x = 0; x < n; x++) {
          const left = x === 0 ? n - 1 : x - 1;
          const right = x === n - 1 ? 0 : x + 1;
          const live =
            a[up + left] + a[up + x] + a[up + right] +
            a[row + left] + a[row + right] +
            a[down + left] + a[down + x] + a[down + right];
          b[row + x] = live === 3 || (live === 2 && a[row + x] === 1) ? 1 : 0;
        }
      }
      const t = a;
      a = b;
      b = t;
    }
    return a;
  },

  gpujs(gpu, { n, gens }, { rows }) {
    // ONE step kernel, where this row used to carry two.
    //
    // The two existed for a real reason: a pipelined kernel owns one output
    // texture and reuses it, so calling a single kernel twice in a row asks it
    // to read the texture it is about to overwrite, which gpu.js refuses
    // (correctly). Alternating between two instances was the ping-pong.
    //
    // Inside a traced pipeline that hand-rolling is not just unnecessary, it is
    // the wrong shape. The tracer sees `state = step(state)` 96 times, works out
    // statically that each generation's output dies as soon as the next has read
    // it, and assigns two alternating plan buffers to one kernel. That is the
    // same ping-pong — two buffers, no allocation per generation — decided by
    // liveness analysis rather than by us remembering to say `g % 2`.
    //
    // No `.setPipeline(true)` either. That flag is how a kernel keeps its result
    // on the device when YOU are chaining the calls; inside a plan, residency of
    // every intermediate is the pipeline's job, and the executor configures its
    // own clones. Setting it here would be a claim that the hand-rolled
    // residency management is still load-bearing, and it is not.
    const step = gpu
      .createKernel(function (a) {
        // `side`, not `n`. The CPU backend translates every identifier that
        // MATCHES A CONSTANT'S NAME into `constants_<name>`, kernel locals
        // included, so a local called `n` would be emitted as
        // `const constants_n = …` and shadow the real constant for the rest
        // of the thread body — silently, and only on that one column.
        //
        // Naming the local `dim` is not enough on its own, because the local
        // names in the shipped bundle are the MINIFIER's, not ours, and a
        // one-character constant is exactly what a minifier hands out. (This
        // is not hypothetical: it is what went wrong in nbody.js, where a
        // local minified to `g` collided with the constant `g`.) A constant
        // whose name is longer than two characters cannot be collided with.
        const dim = this.constants.side;
        const x = this.thread.x;
        const y = this.thread.y;

        let left = x - 1;
        if (left < 0) left = dim - 1;
        let right = x + 1;
        if (right >= dim) right = 0;
        let up = y - 1;
        if (up < 0) up = dim - 1;
        let down = y + 1;
        if (down >= dim) down = 0;

        const live =
          a[up][left] + a[up][x] + a[up][right] +
          a[y][left] + a[y][right] +
          a[down][left] + a[down][x] + a[down][right];

        // live is a whole number, so these equality tests are exact even
        // though the backend is carrying it in a float.
        if (live === 3) return 1;
        if (live === 2) return a[y][x];
        return 0;
      })
      .setConstants({ side: n })
      .setPrecision('single')
      .setOutput([n, n]);

    // Copies the soup into a plan buffer as the plan's first step. It no longer
    // exists for the reason it used to — keeping the step kernels' argument type
    // constant so gpu.js would not recompile between generation 0 and
    // generation 1 — because a plan is traced once at build and every step's
    // argument shape is fixed there; there is no mid-run compiler left to time.
    //
    // It stays because of what it does to the RUN. The soup is what makes each
    // repetition identical, and this is the step that re-lays it down: without
    // it, generation 0 would read the uploaded pipeline argument and the reset
    // would move outside the plan. Keeping it also makes all 96 generations
    // literally the same step reading the same kind of buffer, which is what
    // lets the tracer collapse them onto one kernel, and it matches the
    // bare-WebGPU column beat for beat — its run() opens with a device-side
    // copyBufferToBuffer from the pristine soup for exactly this reason.
    const seed = gpu
      .createKernel(function (a) {
        return a[this.thread.y][this.thread.x];
      })
      .setPrecision('single')
      .setOutput([n, n]);

    // The whole simulation as one traced plan. This function runs ONCE, at the
    // first call, against an opaque handle; the loop is unrolled at trace time
    // into 96 recorded steps, and every later call replays the plan in a single
    // launch with the state never leaving the device.
    //
    // `gens` comes through `this.constants` rather than the closure to say what
    // is true: it is a trace-time fact. Changing it means re-tracing, not
    // re-running, and the constant is where the tracer looks.
    const simulate = gpu.createPipeline(
      function (soup) {
        let state = seed(soup);
        for (let g = 0; g < this.constants.gens; g++) state = step(state);
        return state;
      },
      { constants: { gens }  }
    );

    return {
      async run() {
        // One call, one read-back. Awaiting the pipeline is still the thing
        // that proves every generation actually ran rather than merely being
        // queued — the promise resolves only after the plan's last step has
        // landed and its result has come home. What is gone is the 96 separate
        // awaits, which proved the same thing 96 times over at host prices.
        return await simulate(rows);
      },
      // The pipeline's OWN backend, not a kernel's. Under a plan the user's
      // kernel shortcut is not what executes, so asking it reports the mode we
      // requested no matter what ran — which silently disables this suite's
      // guard against gpu.js degrading to CPU. Reading plan internals was no
      // better: an accessor built on plan.kernels[0].clone went stale one
      // commit later without erroring. `pipeline.backend` is supported API and
      // derives from the executor that actually ran (gpujs/gpu.js#871).
      backend: () => simulate.backend,
      // which lowering actually ran, so a cell that could not reach the
      // fused or threaded path says so instead of being read as one that did
      executor: () => simulate.executorKind,
      destroy() {
        // The pipeline first: it owns cloned kernel instances and the plan
        // buffers, and releasing it while the kernels it cloned from are
        // already gone is a teardown order nobody should have to think about.
        if (simulate.destroy) simulate.destroy();
        [step, seed].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU, borrowing nothing from gpu.js. Same nine loads and
   * same rule; the differences are the ones a runtime cannot help with:
   *
   *   - cells are u32. WGSL has real integers, so the neighbour count is an
   *     integer add and the rule is an integer compare. gpu.js's GL backends
   *     have no integer type at all and carry the same counts as floats.
   *   - every generation goes into ONE compute pass with two alternating bind
   *     groups. WebGPU orders dispatches within a pass and makes each one's
   *     writes visible to the next, so there is one submit and one read-back
   *     for the whole simulation.
   */
  async webgpu(device, { n, gens }, { grid }) {
    const cells = n * n;
    const bytes = cells * 4;
    const S = GPUBufferUsage.STORAGE;

    // The soup as u32, kept GPU-resident so each run resets with a device-side
    // copy instead of a 16 MB upload inside the timed region.
    const seed = Uint32Array.from(grid);
    const bufInit = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
    new Uint32Array(bufInit.getMappedRange()).set(seed);
    bufInit.unmap();

    const mk = () => device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const bufA = mk();
    const bufB = mk();
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

const N: u32 = ${n}u;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= N || y >= N) { return; }

  let left  = select(x - 1u, N - 1u, x == 0u);
  let right = select(x + 1u, 0u, x == N - 1u);
  let up    = select(y - 1u, N - 1u, y == 0u) * N;
  let down  = select(y + 1u, 0u, y == N - 1u) * N;
  let row   = y * N;

  let live = src[up + left] + src[up + x] + src[up + right]
           + src[row + left] + src[row + right]
           + src[down + left] + src[down + x] + src[down + right];

  // NOT named "self": that is a WGSL reserved keyword, and a shader that fails to
  // compile does not fail loudly — createShaderModule reports asynchronously,
  // the pipeline is invalid, every dispatch is dropped, and the read-back is a
  // buffer of zeros that looks like an answer.
  let alive = src[row + x];
  dst[row + x] = select(0u, 1u, live == 3u || (live == 2u && alive == 1u));
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const layout = pipeline.getBindGroupLayout(0);
    const bindGroup = (src, dst) =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: src } },
          { binding: 1, resource: { buffer: dst } },
        ],
      });
    const aToB = bindGroup(bufA, bufB);
    const bToA = bindGroup(bufB, bufA);
    const groups = Math.ceil(n / TILE);
    // Where the final generation lands: an even generation count leaves it back
    // in A, an odd one in B.
    const final = gens % 2 === 0 ? bufA : bufB;

    return {
      async run() {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(bufInit, 0, bufA, 0, bytes);
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (let g = 0; g < gens; g++) {
          pass.setBindGroup(0, g % 2 === 0 ? aToB : bToA);
          pass.dispatchWorkgroups(groups, groups);
        }
        pass.end();
        enc.copyBufferToBuffer(final, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Uint32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufInit, bufA, bufB, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Index-weighted live-cell count. Every cell is visited, and the weight means
   * a grid that is correct in one region and stale in another cannot land on
   * the same total as the correct one by having the right population.
   *
   * Cells are 0 or 1 and the weights are small integers, so this is an exact
   * integer sum in every backend's arithmetic and the columns must agree to the
   * last bit. gpu.js returns an array of rows; the other two return a flat
   * array. Both are walked here in the same order.
   */
  reduce(out, { n }) {
    let acc = 0;
    let i = 0;
    if (ArrayBuffer.isView(out)) {
      for (; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    } else {
      for (let y = 0; y < out.length; y++) {
        const row = out[y];
        for (let x = 0; x < row.length; x++, i++) acc += row[x] * (1 + (i % 17));
      }
    }
    return acc / (n * n);
  },
};
