/**
 * Undersized kernel — the row that is supposed to lose.
 *
 * Four thousand threads, one multiply each. On paper that is embarrassingly
 * parallel work and a GPU has thousands of lanes sitting idle waiting for it.
 * In practice the answer is already in the CPU's L1 cache, the whole job is
 * 4096 multiplies, and getting it to the GPU and back costs more than doing it.
 *
 * DO NOT "FIX" THIS ROW. Every honest benchmark page needs the case where the
 * accelerator is the wrong tool, because that case is extremely common in real
 * code and it is the one nobody publishes. Making this row win would mean
 * either shrinking the transfers (there is nothing to shrink — it is one
 * upload and one read-back) or growing the arithmetic until the row stops being
 * about small kernels, at which point it is just a worse copy of `matmul`.
 *
 * WHAT IT MEASURES. A dispatch plus two transfers. The arithmetic is 4096
 * multiplies, roughly a microsecond of one CPU core, so essentially the whole
 * GPU number is round-trip latency: the driver call, the queue submission, the
 * host-to-device copy of 16 KB, the wait, and the device-to-host copy of 16 KB
 * back. Read the row as a floor. Any GPU kernel you write pays at least this
 * much before it does anything useful, so if the useful part is smaller than
 * this number, do it on the CPU.
 *
 * WHY THIS ONE IS EXEMPT FROM THE SIZING BAND. Every other row is sized so its
 * plain-JS baseline lands between 0.2 s and 3 s, because a number the clock
 * cannot resolve is not a measurement. This row's baseline is about 3
 * MICROseconds — five orders of magnitude below the floor — and that is the
 * entire point of it: the row exists to show what happens when the work is far
 * too small for the machinery around it. Scaling it up to satisfy the band
 * would delete the finding. Hence `sizeExempt`. The runner's own adaptive
 * repetition (up to 25 timed runs per cell) is what makes the numbers on this
 * row readable despite being small; treat the ratio, not the absolute
 * microseconds, as the thing to read.
 */

const N = 4096;

function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s >>> 8) / 0x1000000 - 0.5;
  }
  return a;
}

export default {
  id: 'undersized',
  name: 'Undersized kernel',
  params: `${N} threads, 1 multiply each, fp32`,
  tag: 'transfer-bound',
  group: 'movement',
  size: { n: N },

  // Deliberately outside the 0.2-3 s baseline band; see the header. This is the
  // only row in the table that is allowed to be, and the reason is the finding
  // itself rather than a convenience.
  sizeExempt: true,

  make({ n }) {
    return { a: fill(new Float32Array(n), 0x27d4eb2f) };
  },

  // The oracle, and about as fast as JavaScript gets: one flat typed array, one
  // pass, one multiply, no allocation the JIT cannot see through.
  js({ n }, { a }) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = a[i] * a[i];
    return out;
  },

  gpujs(gpu, { n }, { a }) {
    // No pipeline mode here, on purpose. The other movement rows keep their data
    // GPU-resident so they can measure compute; this row exists to measure the
    // round trip, so the array goes up and the answer comes back, every run,
    // exactly as it would in code that calls a kernel once.
    const kernel = gpu
      .createKernel(function (x) {
        const v = x[this.thread.x];
        return v * v;
      })
      .setOutput([n]);

    return {
      async run() {
        return await kernel(a);
      },
      backend: () => kernel.kernel && kernel.kernel.constructor.mode,
      destroy: () => kernel.destroy && kernel.destroy(),
    };
  },

  /**
   * Hand-written WebGPU. Same shape, no runtime: writeBuffer up, one dispatch,
   * copyBufferToBuffer and mapAsync down. The gap between this cell and the
   * gpu.js WebGPU cell is what the runtime charges to arrange those four calls,
   * and on a job this small that charge is most of the number.
   */
  async webgpu(device, { n }, { a }) {
    const bytes = n * 4;
    const src = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const dst = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const WG = 64;
    const module = device.createShaderModule({
      code: `
struct P { n: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let v = src[i];
  dst[i] = v * v;
}`,
    });

    const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uni, 0, new Uint32Array([n, 0, 0, 0]));

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: uni } },
      ],
    });
    const groups = Math.ceil(n / WG);

    return {
      async run() {
        device.queue.writeBuffer(src, 0, a);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(groups);
        pass.end();
        enc.copyBufferToBuffer(dst, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [src, dst, read, uni].forEach(x => x.destroy && x.destroy());
      },
    };
  },

  // Squares are all positive, so an index weight is the only thing stopping a
  // backend that filled a prefix of the array from matching by accident.
  reduce(out, { n }) {
    let acc = 0;
    for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    return acc / n;
  },
};
