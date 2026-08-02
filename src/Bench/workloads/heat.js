/**
 * Explicit heat diffusion — 1024 time steps on a 1024x1024 grid.
 *
 *     u'(x,y) = u(x,y) + a * (u(x-1,y) + u(x+1,y) + u(x,y-1) + u(x,y+1) - 4u(x,y))
 *
 * Forward Euler on the 2-D Laplacian: the five-point stencil, the oldest
 * finite-difference scheme there is, run for a thousand steps. There is almost
 * no arithmetic per cell — four adds, a subtract, a multiply-add — and five
 * loads, four of them to neighbours. Which is the point. Matmul reuses every
 * value it loads n times and is limited by how fast the machine multiplies;
 * this reuses each value five times at best and is limited by how fast the
 * machine can move a 4 MB grid past the ALUs, a thousand times over. Most real
 * simulation code looks far more like this row than like matmul, and the two
 * numbers are usually not close.
 *
 * SAME STENCIL AS jacobi.js, ON PURPOSE — see the note there. In short: that
 * row runs 512 passes and reads a fifth array; this one runs 1024 and reads
 * four. Twice the passes over the same grid separates per-dispatch cost from
 * per-cell cost, which neither row can do by itself.
 *
 * STABILITY, WHICH IS ALSO WHAT MAKES THE CHECKSUM POSSIBLE. The amplification
 * factor of this scheme for a mode with wavenumbers (kx, ky) is
 * 1 - 4a*(sin^2(kx/2) + sin^2(ky/2)), whose extreme value is 1 - 8a. So a <= 1/4
 * is stable and a = 1/4 is only MARGINALLY so — the checkerboard mode sits at
 * -1 and never decays, which would keep any fp32-vs-fp64 disagreement alive for
 * all 1024 steps. a = 0.2 puts that mode at -0.6, so every mode is strictly
 * damped and rounding injected at step k is smaller by step k+1 rather than
 * larger. Measured against an operation-by-operation fp32-emulated baseline the
 * checksum moves by 5.5e-7, with the worst single cell at 9.2e-6. That is the
 * narrowest margin of any row here — 180x, not the thousands the others have —
 * because a thousand steps is a long time to accumulate rounding and because
 * diffusion correlates neighbouring cells' errors over about twenty cells, so
 * the checksum's million-cell average only cancels about fifty independent
 * regions' worth of it. Still comfortable, but this is the row to look at first
 * if a GPU column ever reads WRONG by a hair.
 *
 * 0.2 IS NOT A FP32 NUMBER, so it is rounded to one here, once, and that single
 * value is what all three implementations multiply by. Letting JS use the fp64
 * 0.2 and the shaders use the fp32 one would be a small, invisible way of
 * giving two columns different arithmetic and then comparing them.
 *
 * WHY THE CHECKSUM IS A SUM OF SQUARES. Diffusion CONSERVES the mean — that is
 * what makes it diffusion — so a checksum built from the mean would be nearly
 * the same number whether the thousand steps ran or not, and would be a poor
 * test of the one thing the row is timing. The sum of squares is the field's
 * energy, and diffusion strictly decreases it: here it falls from 2.708 to
 * 2.250 over the run, so a backend that did nothing at all is out by 17% rather
 * than by a fraction of a percent. It also stays strictly positive, which keeps
 * the runner's relative tolerance away from a division by nearly nothing.
 *
 * The initial field carries most of its energy at a 32-cell wavelength, which
 * this scheme damps by e^-16 over 1024 steps — annihilated — while the smooth
 * part survives. The run therefore genuinely changes the answer, instead of
 * nudging it.
 */

const N = 1024;
const STEPS = 1024;
const HI = N - 2; // last interior index
const ALPHA = Math.fround(0.2); // rounded to fp32 once, shared by every column
const WAVE = 32; // initial wavelength in cells

function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s >>> 8) / 0x1000000;
  };
}

// Rows of a flat grid as a 2-D array, which is what a gpu.js kernel indexes.
// subarray, not slice: these are views, so nothing is copied.
function rows(flat, n) {
  const out = [];
  for (let y = 0; y < n; y++) out.push(flat.subarray(y * n, y * n + n));
  return out;
}

export default {
  id: 'heat',
  name: 'Heat diffusion',
  params: `${N} × ${N}, ${STEPS} steps, fp32`,
  tag: '5-point stencil',
  group: 'sim',
  size: { n: N, steps: STEPS, hi: HI, alpha: ALPHA },

  // Temperatures in [0, 1]: a strong 32-cell oscillation the run will destroy,
  // on a smooth background it will not, plus a little noise. The boundary keeps
  // whatever value it starts with, so the interior relaxes towards the harmonic
  // extension of an edge that is itself not flat.
  make({ n }) {
    const rnd = lcg(0x1b873593);
    const u0 = new Float32Array(n * n);
    const k = (2 * Math.PI) / WAVE;
    for (let y = 0; y < n; y++) {
      const sy = Math.sin(k * y);
      for (let x = 0; x < n; x++) {
        u0[y * n + x] = 0.5 + 0.45 * Math.sin(k * x) * sy + 0.05 * (rnd() - 0.5);
      }
    }
    return { u0 };
  },

  /**
   * The oracle. Two flat buffers swapped by reference, the row offset hoisted,
   * and the neighbours addressed as i-n, i+n, i-1, i+1 so the inner loop walks
   * three cache lines forwards in lockstep. This is what a competent hand-
   * written stencil looks like; a version that recomputed y*n+x four times per
   * cell, or that indexed a nested array, would be several times slower and
   * would hand every GPU column on this row a speedup it did not earn.
   */
  js({ n, steps, hi, alpha }, { u0 }) {
    // Copied, not aliased: make() is called once and shared with the other
    // columns. Both buffers get the boundary, because both take a turn as the
    // source and the boundary is never recomputed.
    let src = new Float32Array(u0);
    let dst = new Float32Array(u0);
    for (let s = 0; s < steps; s++) {
      for (let y = 1; y <= hi; y++) {
        const row = y * n;
        for (let x = 1; x <= hi; x++) {
          const i = row + x;
          const c = src[i];
          dst[i] = c + alpha * (src[i - n] + src[i + n] + src[i - 1] + src[i + 1] - 4 * c);
        }
      }
      const t = src;
      src = dst;
      dst = t;
    }
    return src;
  },

  gpujs(gpu, { n, steps, hi, alpha }, { u0 }) {
    // Two instances of one kernel body: with gpu.js's default immutable:false a
    // kernel reuses its own output texture, so one kernel cannot both read the
    // previous step and overwrite it. They alternate instead.
    const step = function (u) {
      const x = this.thread.x;
      const y = this.thread.y;
      // Dirichlet edge, copied through. Uniform across every workgroup except
      // the four on the border, so it costs the GPU essentially nothing — and
      // it is what keeps both ping-pong textures holding a correct edge without
      // either having to be seeded.
      if (x < 1 || y < 1 || x > this.constants.hi || y > this.constants.hi) {
        return u[y][x];
      }
      const c = u[y][x];
      return c + this.constants.alpha * (u[y - 1][x] + u[y + 1][x] + u[y][x - 1] + u[y][x + 1] - 4 * c);
    };
    const settings = { constants: { hi, alpha }, output: [n, n], pipeline: true };
    const kA = gpu.createKernel(step, settings);
    const kB = gpu.createKernel(step, settings);

    // The initial grid goes to the GPU once, at build time, not once per run:
    // a thousand steps read it and nothing writes it. Timing an upload that a
    // real solver performs once would be timing the wrong thing, and the JS
    // baseline pays the matching cost by copying u0 into its two buffers at the
    // top of every run.
    const upload = gpu.createKernel(
      function (v) {
        return v[this.thread.y][this.thread.x];
      },
      { output: [n, n], pipeline: true }
    );
    const u0Tex = upload(rows(u0, n));

    return {
      async run() {
        // Step 0 reads the pristine texture and writes kA's own, so u0 survives
        // every repetition the runner asks for.
        let t = u0Tex;
        for (let s = 0; s < steps; s++) t = (s % 2 === 0 ? kA : kB)(t);
        // The read-back, and awaiting it is the only thing that proves all 1024
        // dispatches finished. The CPU backend's pipeline result is already a
        // plain array of rows and has no toArray to call.
        return t.toArray ? await t.toArray() : t;
      },
      backend: () => kA.kernel && kA.kernel.constructor.mode,
      destroy() {
        [kA, kB, upload].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU: three storage buffers, three bind groups, and all 1024
   * dispatches recorded into ONE compute pass before anything is submitted.
   * Dispatches inside a pass run in order with WebGPU's own barrier between
   * them, so the ping-pong needs no synchronisation and the CPU pays for one
   * submit rather than a thousand. On a row this dispatch-heavy that is most of
   * what separates this cell from the gpu.js cell beside it.
   *
   * Same stencil, same order of operations, no workgroup-memory halo. A tiled
   * version would read each cell once per workgroup instead of once per thread
   * and would be meaningfully faster — and would turn this column into a
   * measure of how well the kernel was written rather than of what the runtime
   * costs.
   */
  async webgpu(device, { n, steps, hi, alpha }, { u0 }) {
    const bytes = n * n * 4;
    const S = GPUBufferUsage.STORAGE;
    const mk = (data, usage) => {
      const buf = device.createBuffer({ size: bytes, usage, mappedAtCreation: Boolean(data) });
      if (data) {
        new Float32Array(buf.getMappedRange()).set(data);
        buf.unmap();
      }
      return buf;
    };
    const bufInit = mk(u0, S);
    const bufA = mk(null, S | GPUBufferUsage.COPY_SRC);
    const bufB = mk(null, S | GPUBufferUsage.COPY_SRC);
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // alpha is emitted from the fp32 value computed at the top of this file, so
    // the shader multiplies by the same bits the baseline multiplies by.
    const TILE = 16;
    const module = device.createShaderModule({
      code: `
const N: i32 = ${n};
const HI: i32 = ${hi};
const ALPHA: f32 = ${alpha};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= N || y >= N) { return; }
  let i = y * N + x;
  if (x < 1 || y < 1 || x > HI || y > HI) {
    dst[i] = src[i];
    return;
  }
  let c = src[i];
  dst[i] = c + ALPHA * (src[i - N] + src[i + N] + src[i - 1] + src[i + 1] - 4.0 * c);
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bindGroup = (from, to) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: from } },
          { binding: 1, resource: { buffer: to } },
        ],
      });
    // Three buffers, not two: the first step reads the pristine grid, so A and
    // B are free to alternate afterwards and the initial state is never lost.
    const first = bindGroup(bufInit, bufA);
    const ab = bindGroup(bufA, bufB);
    const ba = bindGroup(bufB, bufA);
    const groups = Math.ceil(n / TILE);
    // Step 0 lands in A, then they alternate: an even step count ends in B.
    const last = steps % 2 === 0 ? bufB : bufA;

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (let s = 0; s < steps; s++) {
          pass.setBindGroup(0, s === 0 ? first : s % 2 === 1 ? ab : ba);
          pass.dispatchWorkgroups(groups, groups);
        }
        pass.end();
        enc.copyBufferToBuffer(last, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        // The map is the only thing that proves the pass finished.
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufInit, bufA, bufB, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  // Field energy: every cell squared, index-weighted so a backend that stepped
  // only part of the grid cannot match by luck, and averaged so fp32 and fp64
  // totals stay close enough for a relative tolerance to mean something. See
  // the note at the top for why it is squares and not a mean. gpu.js hands back
  // rows and the other two hand back one flat array, so both shapes are walked
  // rather than flattened — concatenating a million floats to compute a
  // checksum would cost more than some of the cells being checked.
  reduce(out, { n }) {
    let acc = 0;
    if (ArrayBuffer.isView(out)) {
      for (let i = 0; i < out.length; i++) acc += out[i] * out[i] * (1 + (i % 17));
    } else {
      for (let y = 0; y < out.length; y++) {
        const row = out[y];
        for (let x = 0; x < row.length; x++) acc += row[x] * row[x] * (1 + ((y * n + x) % 17));
      }
    }
    return acc / (n * n);
  },
};
