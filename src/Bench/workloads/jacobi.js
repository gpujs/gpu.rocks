/**
 * Jacobi relaxation — 512 cheap passes over one 1024x1024 grid.
 *
 * Matmul is one dispatch that runs for a long time. This is 512 dispatches that
 * each run for almost no time, and that is the whole reason the row is here.
 * One sweep of a 1024^2 grid is about a megacell of work: a GPU finishes it in
 * well under a millisecond, which is the same order as the cost of asking it to
 * start. So this row measures per-dispatch overhead as much as it measures
 * arithmetic, and the four gpu.js columns will not agree with each other about
 * what that overhead costs.
 *
 *     u'(x,y) = (u(x-1,y) + u(x+1,y) + u(x,y-1) + u(x,y+1)) / 4  +  q(x,y)
 *
 * which is Jacobi's method for the Poisson equation, with q = -h^2*f/4 folded
 * into one stored array. Boundary cells are Dirichlet: every pass copies them
 * through unchanged, which is also what keeps both ping-pong buffers carrying a
 * correct edge without either of them having to be pre-filled.
 *
 * THIS IS THE SAME STENCIL AS heat.js, AND THAT IS DELIBERATE. Explicit
 * diffusion with alpha = 1/4 and undamped Jacobi are algebraically the same
 * update; pretending otherwise would be worse than saying so. What differs is
 * everything around it: this row reads a fifth array (the source) on every cell
 * of every pass and runs 512 passes; heat.js reads four, runs 1024, and has no
 * source at all. Between them they separate "cost per pass" from "cost per
 * cell", which one row cannot do alone.
 *
 * NO READ-BACK BETWEEN SWEEPS. Every GPU column ping-pongs GPU-resident
 * buffers — pipeline textures under gpu.js, two storage buffers under bare
 * WebGPU — and reads back exactly once, at the end. Pulling 4 MB to the host
 * 512 times would be a transfer benchmark with a stencil attached to it, and it
 * would make the GPU columns lose to plain JS for reasons that have nothing to
 * do with either.
 *
 * WHY BOTH INPUT ARRAYS ARE UPLOADED AT BUILD TIME. u0 and q are 4 MB each and
 * are uploaded once, outside run(), because 512 sweeps read them and a real
 * solver uploads its problem once too. u0 is never written by the ping-pong —
 * the first sweep reads it and writes elsewhere — so every run genuinely starts
 * from the same initial guess. The JS baseline pays the matching cost: it
 * copies u0 into both of its working buffers at the top of every run.
 *
 * FP32 VS FP64. Jacobi is a CONTRACTION: the iteration matrix has spectral
 * radius cos(pi/n) < 1, so no error mode grows, and rounding injected at sweep
 * k is never amplified by sweep k+1 — the low-frequency modes that Jacobi is
 * famously bad at are exactly the ones with eigenvalue nearest 1, and nearest 1
 * from BELOW. What is left is a random walk of 512 independent fp32 roundings,
 * about eps32*sqrt(512) ~ 1.4e-6 relative, averaged over a million cells in the
 * checksum. Measured against an operation-by-operation fp32-emulated baseline
 * the checksum moves by 2.6e-8, with the worst single cell at 1.2e-6. Note also
 * that q >= 0 and u0 > 0 and the update is a positive combination of positive
 * numbers, so u can never change sign: the checksum stays a long way from zero,
 * which is what makes the runner's RELATIVE tolerance a real test rather than a
 * division by nearly nothing. For scale, the checksum of the initial grid is
 * 4.50 and of the relaxed grid 10.38 — a backend that skipped the sweeps would
 * be out by 57%, not by something a tolerance has to squint at.
 */

const N = 1024;
const SWEEPS = 512;
const HI = N - 2; // last interior index
const C = (N - 1) / 2; // grid centre, 511.5 — exact in fp32
const INV = Math.fround(2 / (N - 1)); // maps an index to [-1, 1]; rounded once
const QS = 1 / 1024; // source scale, a power of two so it is exact everywhere

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
  id: 'jacobi',
  name: 'Jacobi relaxation',
  params: `${N} × ${N}, ${SWEEPS} sweeps, fp32`,
  tag: 'iterative sweep',
  group: 'sim',
  size: { n: N, sweeps: SWEEPS, hi: HI },

  make({ n }) {
    const rnd = lcg(0x27d4eb2f);
    const u0 = new Float32Array(n * n);
    const q = new Float32Array(n * n);
    for (let y = 0; y < n; y++) {
      const sy = (y - C) * INV;
      for (let x = 0; x < n; x++) {
        const sx = (x - C) * INV;
        const i = y * n + x;
        // A smooth initial guess plus a little noise. The smooth part survives
        // 512 sweeps (Jacobi barely touches low frequencies), the noise does
        // not — so the checksum is sensitive both to the sweeps having happened
        // and to the initial data having arrived.
        u0[i] = 0.5 + 0.25 * Math.sin(3 * Math.PI * sx) * Math.sin(2 * Math.PI * sy) + 0.1 * (rnd() - 0.5);
        // A smooth non-negative source: a paraboloid over the grid, zero at the
        // corners. Stored rather than recomputed per cell per sweep — see the
        // note on the js() baseline.
        q[i] = QS * (2 - sx * sx - sy * sy);
      }
    }
    return { u0, q };
  },

  /**
   * The oracle.
   *
   * Two flat buffers swapped by reference, row offsets hoisted, and the four
   * neighbours read as src[row-n+x], src[row+n+x], src[row+x-1], src[row+x+1] —
   * three of which the hardware prefetcher already has. This is the version
   * anyone would write, and writing anything slower would be the cheapest way
   * to make every GPU column on this row look better than it is.
   *
   * The source term q is READ, not computed. It could have been computed from
   * the indices instead — (x-c)*inv and so on — and that would have cost about
   * four flops per cell, on the GPU. In JS most of those flops hoist out of the
   * inner loop, so the baseline would quietly be doing less arithmetic per cell
   * than the kernels it is being compared against. An array both sides read is
   * the only version where the two are doing the same work.
   */
  js({ n, sweeps, hi }, { u0, q }) {
    // Copied, not aliased: make() is called once and shared, so js() must not
    // write through to the inputs the other columns are still going to use.
    // Both buffers get the boundary, because both take a turn as the source.
    let src = new Float32Array(u0);
    let dst = new Float32Array(u0);
    for (let s = 0; s < sweeps; s++) {
      for (let y = 1; y <= hi; y++) {
        const row = y * n;
        for (let x = 1; x <= hi; x++) {
          const i = row + x;
          dst[i] = 0.25 * (src[i - n] + src[i + n] + src[i - 1] + src[i + 1]) + q[i];
        }
      }
      const t = src;
      src = dst;
      dst = t;
    }
    return src;
  },

  gpujs(gpu, { n, sweeps, hi }, { u0, q }) {
    // One kernel body, three instances of it. Two are needed because with
    // gpu.js's default immutable:false a kernel reuses its own output texture,
    // so a kernel cannot both read the previous result and overwrite it; the
    // third is the identity kernel that puts the two inputs on the GPU once.
    const sweep = function (u, src) {
      const x = this.thread.x;
      const y = this.thread.y;
      // Dirichlet edge, copied through. The branch is uniform across all but
      // the four boundary workgroups, so it costs the GPU essentially nothing,
      // and it is what keeps both ping-pong buffers holding a correct edge.
      if (x < 1 || y < 1 || x > this.constants.hi || y > this.constants.hi) {
        return u[y][x];
      }
      return 0.25 * (u[y - 1][x] + u[y + 1][x] + u[y][x - 1] + u[y][x + 1]) + src[y][x];
    };
    const settings = { constants: { hi }, output: [n, n], pipeline: true };
    const kA = gpu.createKernel(sweep, settings);
    const kB = gpu.createKernel(sweep, settings);

    // Two identity kernels rather than one called twice: with immutable:false
    // the second call would hand back the same texture it filled the first
    // time, and q would silently become a copy of u0.
    const identity = function (v) {
      return v[this.thread.y][this.thread.x];
    };
    const upU = gpu.createKernel(identity, { output: [n, n], pipeline: true });
    const upQ = gpu.createKernel(identity, { output: [n, n], pipeline: true });
    const u0Tex = upU(rows(u0, n));
    const qTex = upQ(rows(q, n));

    return {
      async run() {
        // Sweep 0 reads the pristine u0 texture and writes kA's own texture, so
        // u0 is never overwritten and every run starts from the same grid.
        let t = u0Tex;
        for (let s = 0; s < sweeps; s++) t = (s % 2 === 0 ? kA : kB)(t, qTex);
        // toArray is the read-back, and awaiting it is the only thing that
        // proves all 512 dispatches finished; on the synchronous backends the
        // await is harmless. The CPU backend's pipeline result is already a
        // plain array of rows and has no toArray to call.
        return t.toArray ? await t.toArray() : t;
      },
      backend: () => kA.kernel && kA.kernel.constructor.mode,
      destroy() {
        [kA, kB, upU, upQ].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU: three storage buffers, three bind groups, and all 512
   * dispatches recorded into ONE compute pass before anything is submitted.
   *
   * Dispatches inside a pass execute in order and WebGPU inserts the memory
   * barrier between them, so the ping-pong needs no synchronisation of its own —
   * and the CPU pays for one submit instead of 512. That is exactly the cost
   * the gpu.js columns cannot avoid, which is what makes this cell worth having
   * on this row in particular.
   *
   * Same stencil, same order of operations, no tiling and no workgroup memory:
   * a shared-memory halo would be a better kernel, and would fold "we wrote a
   * better kernel" into a number that is supposed to be the runtime's price.
   */
  async webgpu(device, { n, sweeps, hi }, { u0, q }) {
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
    const bufQ = mk(q, S);
    const bufA = mk(null, S | GPUBufferUsage.COPY_SRC);
    const bufB = mk(null, S | GPUBufferUsage.COPY_SRC);
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
const N: i32 = ${n};
const HI: i32 = ${hi};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> q: array<f32>;

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
  dst[i] = 0.25 * (src[i - N] + src[i + N] + src[i - 1] + src[i + 1]) + q[i];
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bindGroup = (from, to) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: from } },
          { binding: 1, resource: { buffer: to } },
          { binding: 2, resource: { buffer: bufQ } },
        ],
      });
    // Three, not two: the first sweep reads the pristine initial grid, so bufA
    // and bufB are free to alternate afterwards and the initial state survives
    // every repetition the runner asks for.
    const first = bindGroup(bufInit, bufA);
    const ab = bindGroup(bufA, bufB);
    const ba = bindGroup(bufB, bufA);
    const groups = Math.ceil(n / TILE);
    // Sweep 0 lands in A, then they alternate: an even sweep count ends in B.
    const last = sweeps % 2 === 0 ? bufB : bufA;

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (let s = 0; s < sweeps; s++) {
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
        [bufInit, bufQ, bufA, bufB, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  // Every cell. gpu.js hands back rows, bare WebGPU and plain JS hand back one
  // flat array, so both shapes are walked rather than flattened — concatenating
  // a million floats to compute a checksum would cost more than some of the
  // cells being checked. Index-weighted so a backend that relaxed only part of
  // the grid cannot match by luck, and averaged so fp32 and fp64 totals stay
  // close enough for the runner's relative tolerance to mean something.
  reduce(out, { n }) {
    let acc = 0;
    if (ArrayBuffer.isView(out)) {
      for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    } else {
      for (let y = 0; y < out.length; y++) {
        const row = out[y];
        for (let x = 0; x < row.length; x++) acc += row[x] * (1 + ((y * n + x) % 17));
      }
    }
    return acc / (n * n);
  },
};
