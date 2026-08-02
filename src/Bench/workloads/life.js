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
 * nothing to compute, so the row measures memory bandwidth and dispatch
 * overhead, which is what most real simulation kernels actually run into.
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
    // Two identical kernels, not one called twice. Each pipelined kernel owns
    // one output texture and reuses it, so `even` reads `odd`'s texture and
    // writes its own and vice versa — a ping-pong with no allocation per
    // generation. One kernel called twice would be asked to read the texture it
    // was about to overwrite, which gpu.js refuses (correctly).
    const step = () =>
      gpu
        .createKernel(function (a) {
          // `dim`, not `n`: a kernel-local named after one of the kernel's own
          // constants collides with the CPU backend's generated `constants_n`.
          const dim = this.constants.n;
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
        .setConstants({ n })
        .setPipeline(true)
        .setPrecision('single')
        .setOutput([n, n]);

    const even = step();
    const odd = step();

    // Uploads the soup into a texture once per run. Its real job is to keep the
    // two step kernels' argument type constant: handing them a plain array on
    // generation 0 and a Texture afterwards makes gpu.js recompile, and the row
    // would be timing a shader compiler.
    const seed = gpu
      .createKernel(function (a) {
        return a[this.thread.y][this.thread.x];
      })
      .setPipeline(true)
      .setPrecision('single')
      .setOutput([n, n]);

    return {
      async run() {
        let state = await seed(rows);
        for (let g = 0; g < gens; g++) {
          state = await (g % 2 === 0 ? even : odd)(state);
        }
        // The read-back, which is the only thing that proves every generation
        // actually ran rather than merely being queued.
        return state.toArray ? await state.toArray() : state;
      },
      backend: () => even.kernel && even.kernel.constructor.mode,
      destroy() {
        [even, odd, seed].forEach(k => k.destroy && k.destroy());
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
