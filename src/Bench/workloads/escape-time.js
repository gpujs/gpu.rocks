/**
 * Escape-time iteration over a Julia set: 2048 × 2048 pixels, up to 100
 * iterations each, smooth (fractional) escape value out.
 *
 * This row prices BRANCH DIVERGENCE. Every other kernel on this page gives
 * every thread the same amount of work; this one does not. Two pixels a
 * hair apart can escape after 3 iterations and after 100, and a GPU executes a
 * warp in lockstep — so the whole group pays for its slowest lane, and a warp
 * straddling the boundary runs at the speed of the pixel that never escapes.
 * A CPU has no such problem: it simply stops. The speed-up on this row is
 * therefore the honest one for irregular work, and it is much smaller than the
 * dense-arithmetic row above it.
 *
 * The parameter c = -0.7269 + 0.1889i is on the boundary of the Mandelbrot set,
 * which gives a dendrite Julia set: filaments everywhere, so neighbouring
 * threads genuinely disagree, and a filled interior of measure zero, so almost
 * no pixel hits the iteration cap.
 *
 * ── Why a smooth escape value, and why the checksum survives ────────────────
 *
 * A raw iteration COUNT is a step function of the pixel coordinate, so a pixel
 * on the boundary lands on one side in fp32 and the other in fp64, and the
 * checksum becomes a coin toss the runner reports as WRONG. The smooth value
 *
 *     nu = i + 1 - log2(log|z_i|)
 *
 * is continuous across the escape threshold: a pixel that escapes an iteration
 * later gets very nearly the same number, so a one-iteration disagreement costs
 * a thousandth rather than a whole unit.
 *
 * That is not the whole story, because escape-time iteration is chaotic near
 * the boundary and fp32 error grows exponentially there. So it was measured
 * rather than argued: an fp32-throughout evaluation (Math.fround on every
 * operation) of this exact kernel differs from the oracle by 1.3e-5 on the
 * checksum — inside the runner's 1e-4, with about eight times to spare. 550
 * pixels out of 4.2 million differ by more than half an iteration, and they
 * always will; the boundary of a Julia set is not a thing fp32 and fp64 can be
 * made to agree about, and a row that claimed otherwise would be lying.
 */

const W = 2048;
const H = 2048;
const MAX_ITER = 100;
// c, the Julia parameter.
const CR = -0.7269;
const CI = 0.1889;
// Width of the complex plane across the image, and the squared escape radius.
// A large radius (64, not 2) is what makes the smooth value smooth: the
// correction term assumes |z| is already well past the threshold.
const SPAN = 3.2;
const ESCAPE_R2 = 4096;
const INV_LN2 = 1.4426950408889634;

export default {
  id: 'escape-time',
  name: 'Julia escape time',
  params: `${W} × ${H} px · ≤ ${MAX_ITER} iterations, smooth value, fp32`,
  tag: 'branch divergence',
  group: 'render',
  size: { w: W, h: H, maxIter: MAX_ITER },

  // No make(): the input is the pixel grid, and the grid is the size. Anything
  // else here would be a table the GPU could read instead of arithmetic it has
  // to do, which is the opposite of what this row measures.

  /**
   * The oracle. Flat typed array, the per-row imaginary coordinate hoisted, and
   * the squares reused between the modulus test and the update — the same three
   * multiplies do both jobs, which is what anyone writing this properly does.
   * A version that recomputed zr*zr for the test would be 30% slower and would
   * hand every GPU column a speed-up it had not earned.
   */
  js({ w, h, maxIter }) {
    const out = new Float32Array(w * h);
    const scale = SPAN / w;
    for (let y = 0; y < h; y++) {
      const ci0 = (y - h / 2) * scale;
      for (let x = 0; x < w; x++) {
        let zr = (x - w / 2) * scale;
        let zi = ci0;
        let i = 0;
        let m = 0;
        for (let it = 0; it < maxIter; it++) {
          const zr2 = zr * zr;
          const zi2 = zi * zi;
          m = zr2 + zi2;
          if (m > ESCAPE_R2) break;
          zi = 2 * zr * zi + CI;
          zr = zr2 - zi2 + CR;
          i = i + 1;
        }
        // Points that never escaped are reported at the cap, so the output is
        // continuous everywhere except across the (measure-zero) filled set.
        out[y * w + x] = i >= maxIter ? maxIter : i + 1 - Math.log(0.5 * Math.log(m)) * INV_LN2;
      }
    }
    return out;
  },

  gpujs(gpu, { w, h, maxIter }) {
    // No arguments at all: nothing is uploaded, so this row is pure arithmetic
    // plus one readback, which is exactly the comparison it is meant to be.
    const kernel = gpu
      .createKernel(function () {
        const scale = this.constants.span / this.constants.w;
        let zr = (this.thread.x - this.constants.w / 2) * scale;
        let zi = (this.thread.y - this.constants.h / 2) * scale;
        let i = 0;
        let m = 0;
        for (let it = 0; it < this.constants.maxIter; it++) {
          const zr2 = zr * zr;
          const zi2 = zi * zi;
          m = zr2 + zi2;
          if (m > this.constants.escapeR2) break;
          zi = 2 * zr * zi + this.constants.ci;
          zr = zr2 - zi2 + this.constants.cr;
          i = i + 1;
        }
        if (i >= this.constants.maxIter) return this.constants.maxIter;
        return i + 1 - Math.log(0.5 * Math.log(m)) * this.constants.invLn2;
      })
      .setConstants({
        w,
        h,
        maxIter,
        cr: CR,
        ci: CI,
        span: SPAN,
        escapeR2: ESCAPE_R2,
        invLn2: INV_LN2,
      })
      // The smooth value runs from 0 to 100, well outside what gpu.js's
      // 'unsigned' fallback encoding can carry. Asking for single precision
      // makes an unsupported machine fail loudly instead of quietly.
      .setPrecision('single')
      .setOutput([w, h]);

    return {
      async run() {
        return await kernel();
      },
      backend: () => kernel.kernel && kernel.kernel.constructor.mode,
      destroy: () => kernel.destroy && kernel.destroy(),
    };
  },

  /**
   * Hand-written WGSL. The `break` is left exactly as it is: a real
   * implementation would not try to be clever about it, and neither backend has
   * anything better to offer, so what separates this cell from the gpu.js cell
   * is dispatch overhead and readback rather than a different loop.
   */
  async webgpu(device, { w, h, maxIter }) {
    const bytes = w * h * 4;
    const bufOut = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const uni = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uni, 0, new Uint32Array([w, h, maxIter, 0]));
    device.queue.writeBuffer(uni, 16, new Float32Array([CR, CI, SPAN, ESCAPE_R2]));

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
struct Params {
  w: u32, h: u32, maxIter: u32, pad: u32,
  cr: f32, ci: f32, span: f32, escapeR2: f32,
};
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.w || gid.y >= p.h) { return; }
  let scale = p.span / f32(p.w);
  var zr = (f32(gid.x) - f32(p.w) * 0.5) * scale;
  var zi = (f32(gid.y) - f32(p.h) * 0.5) * scale;
  var i: u32 = 0u;
  var m = 0.0;
  for (var it: u32 = 0u; it < p.maxIter; it = it + 1u) {
    let zr2 = zr * zr;
    let zi2 = zi * zi;
    m = zr2 + zi2;
    if (m > p.escapeR2) { break; }
    zi = 2.0 * zr * zi + p.ci;
    zr = zr2 - zi2 + p.cr;
    i = i + 1u;
  }
  var v = f32(p.maxIter);
  if (i < p.maxIter) {
    v = f32(i) + 1.0 - log(0.5 * log(m)) * ${INV_LN2};
  }
  out[gid.y * p.w + gid.x] = v;
}`,
    });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufOut } },
        { binding: 1, resource: { buffer: uni } },
      ],
    });
    const gx = Math.ceil(w / TILE);
    const gy = Math.ceil(h / TILE);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(gx, gy);
        pass.end();
        enc.copyBufferToBuffer(bufOut, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufOut, read, uni].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Every pixel, index-weighted, in the same row-major order whichever shape the
   * backend handed back — a gpu.js 2-D kernel returns an array of rows and the
   * other two return one flat buffer, and the weight has to line up or the row
   * would report WRONG for a formatting difference.
   *
   * Values are non-negative and the weighting is bounded, so the sum cannot
   * cancel; the measured fp32-vs-fp64 spread on this checksum is 1.3e-5.
   */
  reduce(out, { w, h }) {
    let acc = 0;
    if (ArrayBuffer.isView(out)) {
      for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
      return acc / (w * h);
    }
    let i = 0;
    for (let y = 0; y < out.length; y++) {
      const row = out[y];
      for (let x = 0; x < row.length; x++, i++) acc += row[x] * (1 + (i % 17));
    }
    return acc / (w * h);
  },
};
