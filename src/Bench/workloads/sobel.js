/**
 * Sobel gradient magnitude — the workhorse image stencil.
 *
 * Nine loads, twelve adds, four multiplies and one square root per pixel, with
 * an access pattern every image kernel ever written shares: read a small
 * neighbourhood, write one value, never look at a pixel you did not need. If a
 * GPU is worth anything at all for imaging, it is worth it here.
 *
 * The arithmetic intensity is roughly two flops per byte moved, which is three
 * orders of magnitude below `matmul`. That is the point of putting it in the
 * table: the dense row and this row are the two ends of the axis every real
 * kernel sits somewhere along, and where a kernel sits on that axis predicts
 * its speed-up far better than how clever the code is.
 *
 * ── WHY SIXTEEN PASSES AND NOT ONE ──────────────────────────────────────────
 *
 * One Sobel pass over 2048 x 2048 is about 20 ms of plain JS on the machine
 * this was written on. That is a tenth of the sizing script's floor, so a
 * single-pass row would be measuring the clock and the dispatch, not the
 * stencil. The obvious fix — a bigger image — does not survive contact with the
 * backends: gpu.js's WebGL output texture is RGBA32F with one channel used, so
 * a 6144 x 6144 output is a 604 MB allocation, and that is the smallest image
 * that gets a single pass into the band. Sweeping the same 2048 x 2048 buffer
 * sixteen times keeps the footprint at two 16 MB planes, keeps the arithmetic
 * identical in every column, and is a shape real code has anyway (a stencil
 * swept to convergence, a pyramid level refined in place).
 *
 * THE CAVEAT A READER IS OWED, and it is a real one: sixteen passes amortise
 * the upload and the read-back over sixteen dispatches. A single Sobel call
 * from JS pays that transfer once for one pass and looks far worse than this
 * row does. `blur-separable` is the row that shows the transfer-dominated case
 * honestly; this row shows the case where you have already decided to keep the
 * image on the GPU. Both are true and they are not the same number.
 *
 * ── WHY THE 1/8 ────────────────────────────────────────────────────────────
 *
 * The Sobel kernel estimates the derivative scaled by 8 (a unit ramp gives
 * gx = 8), so each pass is normalised by 1/8 to be an actual gradient rather
 * than eight of one. It matters here because the passes compose: unnormalised,
 * the magnitudes grow about 3.3x per pass and the sixteenth pass would be
 * carrying values around 1e7. 1/8 is exactly representable, so the
 * normalisation itself introduces no rounding, and because the magnitude is
 * homogeneous — mag(c*p) = c*mag(p) for c > 0 — scaling every pass by a
 * constant cannot change the relative error by a single ulp.
 *
 * ── CHECKSUM ────────────────────────────────────────────────────────────────
 *
 * There is not a single branch or comparison in the arithmetic, so there is
 * nothing here for fp32 and fp64 to break a tie on differently. The one thing
 * worth checking was whether sixteen chained passes compound their rounding:
 * they do not, because the state is re-rounded to fp32 at the end of every pass
 * in both the baseline and the kernels, so neither drifts away from the other.
 * Simulating fp32 rounding on every intermediate of the baseline moves the
 * checksum by about 1e-8 relative after sixteen passes, four orders inside the
 * runner's 1e-4.
 */

const N = 2048;
const PASSES = 16;

// 1/8, written as the reciprocal so it is visibly a power of two: exact in
// fp32, exact in fp64, and the same bits in WGSL, GLSL and JS.
const SCALE = 0.125;

/**
 * A structured image rather than pure noise. Sobel on white noise is white
 * noise, which is a perfectly good timing input but a useless one to look at
 * and a poor test of anything: real images have long smooth regions where the
 * gradient is near zero and sharp ones where it is not, and those are where
 * cancellation in `gx` and `gy` actually happens.
 */
function image(n, seed) {
  const a = new Float32Array(n * n);
  let s = seed >>> 0;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      s = (s * 1664525 + 1013904223) >>> 0;
      const r = (s >>> 8) / 0x1000000 - 0.5;
      a[y * n + x] =
        0.5 +
        0.25 * Math.sin(x * 0.013 + Math.cos(y * 0.007) * 3) +
        0.15 * Math.cos(y * 0.021) +
        0.1 * r;
    }
  }
  return a;
}

export default {
  id: 'sobel',
  name: 'Sobel gradient magnitude',
  params: `${N} × ${N} fp32, 3×3, ${PASSES} passes`,
  tag: 'image stencil',
  group: 'image',
  size: { n: N, passes: PASSES, scale: SCALE, nm1: N - 1 },

  make({ n }) {
    const src = image(n, 0x2545f491);
    // 2-D rows, built once and shared: reshaping inside a column would time the
    // reshape, and handing two columns different bytes would not be a benchmark.
    const rows = [];
    for (let y = 0; y < n; y++) rows.push(src.subarray(y * n, y * n + n));
    return { src, rows };
  },

  /**
   * The oracle, and a fair baseline. Two scratch planes swapped by reference so
   * no pass allocates; the three row bases hoisted out of the inner loop; the
   * border clamp done with two predictable comparisons per pixel rather than a
   * min/max pair, which is what the kernels do too. The input plane is never
   * written, so `make`'s bytes survive for the columns that run after this one.
   */
  js({ n, passes, scale }, { src }) {
    const t0 = new Float32Array(n * n);
    const t1 = new Float32Array(n * n);
    let read = src;
    let write = t0;

    for (let p = 0; p < passes; p++) {
      for (let y = 0; y < n; y++) {
        const up = (y > 0 ? y - 1 : 0) * n;
        const row = y * n;
        const down = (y < n - 1 ? y + 1 : n - 1) * n;
        for (let x = 0; x < n; x++) {
          const xl = x > 0 ? x - 1 : 0;
          const xr = x < n - 1 ? x + 1 : n - 1;
          const p00 = read[up + xl];
          const p01 = read[up + x];
          const p02 = read[up + xr];
          const p10 = read[row + xl];
          const p12 = read[row + xr];
          const p20 = read[down + xl];
          const p21 = read[down + x];
          const p22 = read[down + xr];
          const gx = p02 + 2 * p12 + p22 - (p00 + 2 * p10 + p20);
          const gy = p20 + 2 * p21 + p22 - (p00 + 2 * p01 + p02);
          write[row + x] = Math.sqrt(gx * gx + gy * gy) * scale;
        }
      }
      read = write;
      write = write === t0 ? t1 : t0;
    }
    return read;
  },

  gpujs(gpu, { n, passes, scale, nm1 }, { rows }) {
    const consts = { nm1, scale };

    // One body, three instances. `even` and `odd` each own a pipeline texture
    // and reuse it, so one reads the other's while writing its own — a
    // ping-pong with no allocation per pass. Handing a single kernel its own
    // output to read would be asking it to sample the texture it is about to
    // overwrite, which gpu.js correctly refuses.
    const body = function (a) {
      const x = this.thread.x;
      const y = this.thread.y;
      let xl = x - 1;
      if (xl < 0) xl = 0;
      let xr = x + 1;
      if (xr > this.constants.nm1) xr = this.constants.nm1;
      let yu = y - 1;
      if (yu < 0) yu = 0;
      let yd = y + 1;
      if (yd > this.constants.nm1) yd = this.constants.nm1;

      const p00 = a[yu][xl];
      const p01 = a[yu][x];
      const p02 = a[yu][xr];
      const p10 = a[y][xl];
      const p12 = a[y][xr];
      const p20 = a[yd][xl];
      const p21 = a[yd][x];
      const p22 = a[yd][xr];

      const gx = p02 + 2 * p12 + p22 - (p00 + 2 * p10 + p20);
      const gy = p20 + 2 * p21 + p22 - (p00 + 2 * p01 + p02);
      return Math.sqrt(gx * gx + gy * gy) * this.constants.scale;
    };

    const mk = pipeline =>
      gpu
        .createKernel(body)
        .setConstants(consts)
        .setPipeline(pipeline)
        .setPrecision('single')
        .setTactic('precision')
        .setOutput([n, n]);

    // The first pass takes the plain 2-D array; every later pass takes a
    // texture. They have to be different kernel objects, because changing an
    // argument's type between calls makes gpu.js recompile and the row would be
    // timing a shader compiler.
    const first = mk(true);
    const even = mk(true);
    const odd = mk(true);
    // The last pass is not pipelined, so `run` resolves on a real array rather
    // than a handle to work that may still be queued.
    const last = mk(false);

    return {
      async run() {
        let state = await first(rows);
        for (let p = 1; p < passes - 1; p++) {
          state = await (p % 2 === 1 ? even : odd)(state);
        }
        return await last(state);
      },
      backend: () => first.kernel && first.kernel.constructor.mode,
      destroy() {
        [first, even, odd, last].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU with no gpu.js anywhere in it. Three storage buffers —
   * the untouched source and two scratch planes — and all sixteen dispatches
   * recorded into ONE compute pass, because WebGPU orders dispatches within a
   * pass and makes each one's writes visible to the next. That is one submit
   * and one read-back for the whole sweep; the difference between this cell and
   * the WebGPU cell to its left is what the runtime costs.
   */
  async webgpu(device, { n, passes, scale, nm1 }, { src }) {
    const bytes = n * n * 4;
    const S = GPUBufferUsage.STORAGE;

    const bufSrc = device.createBuffer({
      size: bytes,
      usage: S | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(bufSrc.getMappedRange()).set(src);
    bufSrc.unmap();

    const mk = () => device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_SRC });
    const bufX = mk();
    const bufY = mk();
    const read = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;

const N: i32 = ${n};
const NM1: i32 = ${nm1};
const SCALE: f32 = ${scale};

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= N || y >= N) { return; }

  let xl = max(x - 1, 0);
  let xr = min(x + 1, NM1);
  let yu = max(y - 1, 0) * N;
  let yd = min(y + 1, NM1) * N;
  let row = y * N;

  let p00 = a[yu + xl];
  let p01 = a[yu + x];
  let p02 = a[yu + xr];
  let p10 = a[row + xl];
  let p12 = a[row + xr];
  let p20 = a[yd + xl];
  let p21 = a[yd + x];
  let p22 = a[yd + xr];

  let gx = p02 + 2.0 * p12 + p22 - (p00 + 2.0 * p10 + p20);
  let gy = p20 + 2.0 * p21 + p22 - (p00 + 2.0 * p01 + p02);
  o[row + x] = sqrt(gx * gx + gy * gy) * SCALE;
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const layout = pipeline.getBindGroupLayout(0);
    const bind = (from, to) =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: from } },
          { binding: 1, resource: { buffer: to } },
        ],
      });
    const srcToX = bind(bufSrc, bufX);
    const xToY = bind(bufX, bufY);
    const yToX = bind(bufY, bufX);
    const groups = Math.ceil(n / TILE);
    // Pass 1 lands in X; passes alternate from there, so an even pass count
    // finishes in Y and an odd one in X.
    const final = passes % 2 === 0 ? bufY : bufX;

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (let p = 0; p < passes; p++) {
          pass.setBindGroup(0, p === 0 ? srcToX : p % 2 === 1 ? xToY : yToX);
          pass.dispatchWorkgroups(groups, groups);
        }
        pass.end();
        enc.copyBufferToBuffer(final, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        // The read-back is the only thing that proves the sweep actually ran
        // rather than merely being queued.
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufSrc, bufX, bufY, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Index-weighted mean over every pixel. The weight is what stops a backend
   * that computed one tile — or one pass — from landing on the right total by
   * accident: the same values in the wrong places give a different sum. gpu.js
   * hands back an array of rows, the other two a flat array; both are walked in
   * the same order here.
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
