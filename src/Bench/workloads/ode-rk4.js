/**
 * Half a million independent RK4 trajectories — the long inner loop.
 *
 * Every other row here gets wider when it gets bigger. This one gets DEEPER.
 * The 2^19 trajectories never talk to each other and never will, so the only
 * way to spend more time is to take more steps, and all 256 of those steps live
 * inside the kernel. One dispatch, one read-back, and between them a loop no
 * scheduler can help with: step k+1 cannot start until step k has finished, in
 * every thread, for the whole kernel.
 *
 * That makes it the counterweight to matmul. Matmul is latency-tolerant — a
 * thread stalled on memory is replaced by another with work to do. Here every
 * thread is doing dependent scalar arithmetic at the same time, so occupancy
 * buys nothing and what is actually being measured is a GPU's per-lane
 * instruction throughput on a serial chain. That is the number people are
 * usually most surprised by.
 *
 * THE SYSTEM. Classical RK4 on the logistic equation
 *
 *     y' = r * y * (1 - y)
 *
 * with r and y0 varying per trajectory. Non-linear, one state variable, and
 * about three flops per derivative evaluation — deliberately cheap, so that the
 * row times the loop structure rather than a fat right-hand side. A damped
 * pendulum would be a better advert and a worse benchmark: two sin() calls per
 * stage, four stages, 134 million steps, and the row would silently become a
 * transcendental-function benchmark.
 *
 * WHY THIS SYSTEM AND NOT A PRETTIER ONE. The checksum has to survive fp32 on
 * the GPU against fp64 in JS across 256 dependent steps, and for a chaotic
 * system it simply would not: two trajectories that differ in the last mantissa
 * bit of y0 separate exponentially, and a Lorenz or a stiff oscillator would
 * make every GPU column read WRONG for reasons that have nothing to do with the
 * GPU. The logistic equation is CONTRACTIVE towards y = 1 for r > 0: the
 * derivative of y(T) with respect to y(0) is exp(int r*(1-2y) dt), which is
 * bounded above by 1 once y passes 1/2 and decays like exp(-r*T) after that.
 * Rounding injected at step k is therefore damped by every step after it
 * instead of amplified. With r*h <= 0.012 the RK4 stability region is not even
 * in sight. Measured against an operation-by-operation fp32-emulated baseline,
 * the checksum moves by 2e-9 — see the note at the bottom of this comment.
 *
 * EXACT CONSTANTS. h = 2/256 is a power of two and therefore exact in fp32, but
 * h/6 is not, and "compute h/6 in fp64 on the CPU and in fp32 on the GPU" is a
 * quiet way to give the two columns different arithmetic. So h/6 is rounded to
 * fp32 ONCE, here, and that single value is what every backend multiplies by.
 *
 * SIZE. 2^19 trajectories rather than 2^20: at 2^20 the plain-JS baseline is
 * 5.3 s on the development machine, well past the 3 s ceiling the sizing rule
 * sets, and a row nobody waits for is a row nobody reads. The steps were kept
 * and the width was halved rather than the other way round, because the depth
 * of the inner loop is the whole point of this row and 524288 threads still
 * saturate any GPU that can run the page.
 */

const PATHS = 1 << 19;
const STEPS = 256;
const T = 2;
const H = T / STEPS; // 0.0078125 — exact in fp32
const H2 = H / 2; // also exact
const H6 = Math.fround(H / 6); // NOT exact: rounded once, shared by all columns

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
  id: 'ode-rk4',
  name: 'RK4 trajectories',
  params: `2^19 systems × ${STEPS} steps, fp32`,
  tag: 'long inner loop',
  group: 'sim',
  size: { paths: PATHS, steps: STEPS, h: H, h2: H2, h6: H6 },

  // y0 spans both sides of the equilibrium at y = 1, so the trajectories are not
  // all approaching from below, and r spans a 3x range so they are not all
  // travelling at the same speed. Both matter: identical trajectories would let
  // a broken backend match the checksum by getting one of them right.
  make({ paths }) {
    return {
      y0: fill(new Float32Array(paths), 0x9e3779b9, 0.05, 1.95),
      r: fill(new Float32Array(paths), 0x85ebca6b, 0.5, 1.5),
    };
  },

  // The oracle. One flat pass, both inputs walked forwards, the whole state of
  // a trajectory held in one local — which is exactly what the GPU columns do
  // too, so the comparison is between the same program on different hardware
  // rather than between two different programs. Nothing here is held back: an
  // array-of-state baseline that wrote y back to memory every step would be
  // several times slower and would flatter every column to its right.
  js({ paths, steps, h, h2, h6 }, { y0, r }) {
    const out = new Float32Array(paths);
    for (let i = 0; i < paths; i++) {
      const ri = r[i];
      let y = y0[i];
      for (let s = 0; s < steps; s++) {
        const k1 = ri * y * (1 - y);
        const y2 = y + h2 * k1;
        const k2 = ri * y2 * (1 - y2);
        const y3 = y + h2 * k2;
        const k3 = ri * y3 * (1 - y3);
        const y4 = y + h * k3;
        const k4 = ri * y4 * (1 - y4);
        y = y + h6 * (k1 + 2 * k2 + 2 * k3 + k4);
      }
      out[i] = y;
    }
    return out;
  },

  gpujs(gpu, { paths, steps, h, h2, h6 }, { y0, r }) {
    const kernel = gpu
      .createKernel(function (start, rate) {
        const ri = rate[this.thread.x];
        let y = start[this.thread.x];
        for (let s = 0; s < this.constants.steps; s++) {
          const k1 = ri * y * (1 - y);
          const y2 = y + this.constants.h2 * k1;
          const k2 = ri * y2 * (1 - y2);
          const y3 = y + this.constants.h2 * k2;
          const k3 = ri * y3 * (1 - y3);
          const y4 = y + this.constants.h * k3;
          const k4 = ri * y4 * (1 - y4);
          y = y + this.constants.h6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
        return y;
      })
      .setConstants({ steps, h, h2, h6 })
      .setOutput([paths]);

    return {
      // await, not fire-and-forget: on the WebGPU backend the result is a
      // promise, and returning before it settles would time the dispatch call
      // rather than 134 million integration steps.
      async run() {
        return await kernel(y0, r);
      },
      backend: () => kernel.kernel && kernel.kernel.constructor.mode,
      destroy: () => kernel.destroy && kernel.destroy(),
    };
  },

  /**
   * Hand-written WebGPU, with no gpu.js anywhere in it — the same loop, the
   * same constants, the same order of operations. There is nothing to optimise
   * differently here and that is the point: with the algorithm pinned, the gap
   * between this cell and the WebGPU cell to its left is the runtime's price
   * and nothing else.
   */
  async webgpu(device, { paths, steps, h, h2, h6 }, { y0, r }) {
    const bytes = paths * 4;
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
    const bufY0 = upload(y0);
    const bufR = upload(r);
    const bufOut = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // The step count and the three step constants are baked in rather than
    // bound as a uniform. H6 in particular is emitted from the fp32 value
    // computed at the top of this file, so the shader multiplies by the same
    // number the baseline multiplies by, bit for bit.
    const WG = 64;
    const module = device.createShaderModule({
      code: `
const STEPS: u32 = ${steps}u;
const H: f32 = ${h};
const H2: f32 = ${h2};
const H6: f32 = ${h6};
const PATHS: u32 = ${paths}u;

@group(0) @binding(0) var<storage, read> y0: array<f32>;
@group(0) @binding(1) var<storage, read> rate: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= PATHS) { return; }
  let ri = rate[i];
  var y = y0[i];
  for (var s: u32 = 0u; s < STEPS; s = s + 1u) {
    let k1 = ri * y * (1.0 - y);
    let y2 = y + H2 * k1;
    let k2 = ri * y2 * (1.0 - y2);
    let y3 = y + H2 * k2;
    let k3 = ri * y3 * (1.0 - y3);
    let y4 = y + H * k3;
    let k4 = ri * y4 * (1.0 - y4);
    y = y + H6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  }
  out[i] = y;
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufY0 } },
        { binding: 1, resource: { buffer: bufR } },
        { binding: 2, resource: { buffer: bufOut } },
      ],
    });
    const groups = Math.ceil(paths / WG);

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
        // The read-back is the only thing that proves the loop actually ran.
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufY0, bufR, bufOut, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  // Every trajectory, index-weighted so a backend that integrated a prefix of
  // the array cannot match by luck, and averaged so the value stays O(1). Every
  // y stays strictly positive (the logistic equation cannot cross zero from
  // above), so the checksum is nowhere near zero and the runner's RELATIVE
  // tolerance is a real one.
  reduce(out, { paths }) {
    const flat = ArrayBuffer.isView(out) ? out : Float32Array.from(out);
    let acc = 0;
    for (let i = 0; i < flat.length; i++) acc += flat[i] * (1 + (i % 17));
    return acc / paths;
  },
};
