/**
 * A million independent descents — the embarrassingly parallel case.
 *
 * 2^20 starting points, each rolling downhill on the same fixed 2-D surface for
 * 256 steps. No thread reads another thread's data, at any point, ever. There
 * is no reduction at the end, no shared grid, no neighbour, no barrier: the
 * only memory traffic is one pair of floats in and one float out per thread,
 * and everything between is registers.
 *
 * That makes this row the CEILING. Every other row here is limited by something
 * other than arithmetic — bandwidth, dispatch count, a serial dependency, a
 * gather. This one is limited by nothing at all, so the ratio it shows is about
 * as good as the ratio gets, and every other row in the table should be read
 * against it. If a workload's speedup is much lower than this row's, the
 * difference is what that workload's structure costs; if a workload's speedup
 * is HIGHER than this row's, something is wrong with the workload.
 *
 * THE SURFACE.
 *
 *     f(x,y) = (a*x^2 + b*y^2)/2 + c*x^2*y^2
 *     df/dx  = x * (a + 2c*y^2)
 *     df/dy  = y * (b + 2c*x^2)
 *
 * with a = 1, b = 2, c = 1/4 — an anisotropic bowl with a quartic coupling that
 * makes the descent genuinely non-linear without making it interesting enough
 * to be unstable. Every constant, including the learning rate 1/256, is a power
 * of two or a small integer, so all of them are exact in fp32 and JS and the
 * shaders multiply by identical bits.
 *
 * SIZE. 256 steps rather than 512: at 512 the plain-JS baseline is 3.7 s on the
 * development machine, past the 3 s ceiling the sizing rule sets. The learning
 * rate was doubled to 1/256 at the same time, so the trajectories travel
 * exactly as far as they did before — half the steps, each twice as long — and
 * the row still ends mid-descent rather than at the minimum.
 *
 * WHY THIS SURFACE AND NOT ROSENBROCK. Rosenbrock is the famous one and would
 * be the wrong choice twice over. Its valley is ill-conditioned, so gradient
 * descent either diverges or crawls depending on the fifth decimal place of the
 * step size — and a step size that sits near the edge of stability is exactly
 * the case where fp32 and fp64 stop agreeing, because the trajectory amplifies
 * the difference between them instead of forgetting it. Here the largest
 * Hessian eigenvalue over the sampled region is about 7.5, so lr*L is about
 * 0.03: nearly two orders of magnitude inside the stability limit of 2, monotone
 * descent everywhere, and a CONTRACTION. Rounding injected at step k is damped
 * by every step after it. Measured against an operation-by-operation
 * fp32-emulated baseline the checksum moves by 5e-11 — the largest margin of
 * any row here, which is what a contraction with no memory traffic buys you.
 *
 * WHY 256 STEPS AND NOT ENOUGH TO CONVERGE. Everything here descends to (0,0),
 * so a run long enough to converge would drive every output to zero and the
 * checksum with it — and a checksum of nearly-zero makes a RELATIVE tolerance
 * meaningless, quite apart from being satisfiable by a backend that returned
 * zeroes. With lr*a = 1/256 the x coordinate shrinks by e^-1 over the run and y
 * by e^-2, so the trajectories are still mid-flight when they are read: the
 * mean objective falls from 2.44 to 0.089 and the coordinates keep most of
 * their spread. The checksum lands near 0.80 — small, but positive and nowhere
 * near zero, and 0.089 against 2.44 means a backend that returned its input
 * untouched would be out by a factor of 27.
 */

const STARTS = 1 << 20;
const STEPS = 256;
const A = 1; // curvature in x
const B = 2; // curvature in y — anisotropic, so the two coordinates decay differently
const C2 = 0.5; // 2c, the coupling coefficient as it appears in the gradient
const C = 0.25; // c, as it appears in the objective
const LR = 1 / 256; // exact in fp32, and lr*L ~ 0.03 — far inside stability

// Deterministic and cheap. Not Math.random: two columns must be handed the same
// bytes, and a seeded generator is the only way to say that and mean it.
function fill(a, seed, lo, hi) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = lo + ((s >>> 8) / 0x1000000) * (hi - lo);
  }
  return a;
}

export default {
  id: 'gradient-descent',
  name: 'Gradient descent',
  params: `2^20 starts × ${STEPS} steps, fp32`,
  tag: 'embarrassingly parallel',
  group: 'sim',
  size: { starts: STARTS, steps: STEPS, a: A, b: B, c: C, c2: C2, lr: LR },

  // Starts spread over [-2, 2]^2, so the coupling term c*x^2*y^2 is worth as
  // much as 4 near the corners and nothing near the axes: the trajectories are
  // not all the same trajectory scaled, which is what stops a broken backend
  // matching the checksum by computing one of them.
  make({ starts }) {
    return {
      x0: fill(new Float32Array(starts), 0x9e3779b9, -2, 2),
      y0: fill(new Float32Array(starts), 0xc2b2ae35, -2, 2),
    };
  },

  // The oracle. Both coordinates live in locals for the whole descent, x*x and
  // y*y are each computed once per step and used twice, and the two gradient
  // components are factored as x*(a + 2c*y^2) rather than a*x + 2c*x*y^2 —
  // one multiply cheaper, and the form any careful implementation would reach
  // for. This is the same expression, in the same order, that both GPU columns
  // evaluate; on a row whose whole point is that nothing gets in the way of the
  // arithmetic, a sloppy baseline would be the entire result.
  js({ starts, steps, a, b, c, c2, lr }, { x0, y0 }) {
    const out = new Float32Array(starts);
    for (let i = 0; i < starts; i++) {
      let x = x0[i];
      let y = y0[i];
      for (let s = 0; s < steps; s++) {
        const xx = x * x;
        const yy = y * y;
        x = x - lr * (x * (a + c2 * yy));
        y = y - lr * (y * (b + c2 * xx));
      }
      const xx = x * x;
      const yy = y * y;
      out[i] = 0.5 * (a * xx + b * yy) + c * xx * yy;
    }
    return out;
  },

  gpujs(gpu, { starts, steps, a, b, c, c2, lr }, { x0, y0 }) {
    const kernel = gpu
      .createKernel(function (sx, sy) {
        let x = sx[this.thread.x];
        let y = sy[this.thread.x];
        for (let s = 0; s < this.constants.steps; s++) {
          const xx = x * x;
          const yy = y * y;
          x = x - this.constants.lr * (x * (this.constants.a + this.constants.c2 * yy));
          y = y - this.constants.lr * (y * (this.constants.b + this.constants.c2 * xx));
        }
        const fx = x * x;
        const fy = y * y;
        return 0.5 * (this.constants.a * fx + this.constants.b * fy) + this.constants.c * fx * fy;
      })
      .setConstants({ steps, a, b, c, c2, lr })
      .setOutput([starts]);

    return {
      // await, not fire-and-forget: on the WebGPU backend the result is a
      // promise, and returning before it settles would time the dispatch call
      // rather than 268 million descent steps.
      async run() {
        return await kernel(x0, y0);
      },
      backend: () => kernel.kernel && kernel.kernel.constructor.mode,
      destroy: () => kernel.destroy && kernel.destroy(),
    };
  },

  /**
   * Hand-written WebGPU, borrowing nothing from gpu.js. The same loop, the same
   * constants, the same order of operations — there is nothing here anyone
   * could write more cleverly, which is what makes the gap between this cell
   * and the WebGPU cell beside it a clean reading of the runtime's price.
   */
  async webgpu(device, { starts, steps, a, b, c, c2, lr }, { x0, y0 }) {
    const bytes = starts * 4;
    const upload = data => {
      const buf = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
      });
      new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return buf;
    };
    const bufX = upload(x0);
    const bufY = upload(y0);
    const bufOut = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Every constant baked in: they are constants of the workload, they are all
    // exactly representable, and the compiler gets to see the trip count.
    const WG = 64;
    const module = device.createShaderModule({
      code: `
const STARTS: u32 = ${starts}u;
const STEPS: u32 = ${steps}u;
const A: f32 = ${a};
const B: f32 = ${b};
const C: f32 = ${c};
const C2: f32 = ${c2};
const LR: f32 = ${lr};

@group(0) @binding(0) var<storage, read> x0: array<f32>;
@group(0) @binding(1) var<storage, read> y0: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= STARTS) { return; }
  var x = x0[i];
  var y = y0[i];
  for (var s: u32 = 0u; s < STEPS; s = s + 1u) {
    let xx = x * x;
    let yy = y * y;
    x = x - LR * (x * (A + C2 * yy));
    y = y - LR * (y * (B + C2 * xx));
  }
  let fx = x * x;
  let fy = y * y;
  out[i] = 0.5 * (A * fx + B * fy) + C * fx * fy;
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufX } },
        { binding: 1, resource: { buffer: bufY } },
        { binding: 2, resource: { buffer: bufOut } },
      ],
    });
    const groups = Math.ceil(starts / WG);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(groups);
        pass.end();
        enc.copyBufferToBuffer(bufOut, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        // The read-back is the only thing that proves the descent actually ran.
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufX, bufY, bufOut, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  // Every start, index-weighted so a backend that descended a prefix of the
  // array cannot match by luck, and averaged so the value stays O(1). The
  // objective is a sum of squares with positive coefficients, so it cannot go
  // negative and the checksum sits near 0.80, well away from the zero that
  // would make a relative tolerance meaningless.
  reduce(out, { starts }) {
    const flat = ArrayBuffer.isView(out) ? out : Float32Array.from(out);
    let acc = 0;
    for (let i = 0; i < flat.length; i++) acc += flat[i] * (1 + (i % 17));
    return acc / starts;
  },
};
