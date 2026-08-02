/**
 * Kernel launch overhead — what a dispatch costs before any arithmetic happens.
 *
 * Every thread does exactly one floating-point add. There is nothing to be
 * clever about and nothing to amortise: the only interesting quantity in this
 * row is the fixed price of asking a GPU to do something. matmul says the GPU
 * is worth 100×; this row says what you pay each time you ask.
 *
 * WHY 512 PASSES. One pass of 2^20 adds is 0.8 ms of plain JavaScript — below
 * the floor where a wall clock is worth reading, and far below the point where
 * the sizing script will accept a row. So one run is 512 dependent passes over
 * the same buffer. Every column does the same 512 passes, so the ratio is
 * untouched; what the repetition buys is a number the clock can actually
 * resolve, and a dispatch count (512) large enough that per-launch cost stops
 * hiding in the noise and becomes the thing being measured.
 *
 * WHY THE DATA STAYS RESIDENT. The array is handed to the GPU once per run and
 * read back once per run; the 512 passes ping-pong between two GPU-side
 * buffers. Uploading 4 MB before each pass would be a perfectly good benchmark
 * of PCIe, and a poor one of dispatch — the transfer would dominate and the
 * launches would be invisible underneath it. This row is about dispatch, so the
 * transfer is paid once and the launches are what is left.
 *
 * EXPECT THE GPU COLUMNS TO LOOK BAD. That is the finding, not a defect. One
 * add per thread over 4 MB is arithmetic intensity of essentially zero, so the
 * GPU has nothing to be good at, and 512 launches of a runtime that validates
 * arguments and rebinds textures in JavaScript on every call is a cost plain JS
 * simply does not have. The bare-WebGPU column is the control: it records the
 * same 512 dispatches into ONE command buffer, so the gap between it and the
 * gpu.js WebGPU column to its left is the runtime's per-call price, in
 * milliseconds, isolated from everything else.
 */

const N = 1 << 20; // 1,048,576 threads
const PASSES = 512;

// Exactly representable in fp32, and small enough that 512 of them added to a
// value near 0 stay in a range where every add is exact. That matters: with an
// exact step the plain-JS column and the GPU columns agree bit for bit, so a
// checksum mismatch on this row can only mean a backend skipped a pass.
const STEP = 1 / 1024;

function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s >>> 8) / 0x1000000 - 0.5;
  }
  return a;
}

export default {
  id: 'launch-overhead',
  name: 'Kernel launch overhead',
  params: `${PASSES} × 2^20 threads, 1 add each, fp32`,
  tag: 'dispatch cost',
  group: 'movement',
  size: { n: N, passes: PASSES, step: STEP },

  make({ n }) {
    return { a: fill(new Float32Array(n), 0xc2b2ae35) };
  },

  // The oracle. Two scratch buffers ping-ponged, exactly as the GPU columns do
  // it — no copy of the shared input, because the first pass reads it directly.
  js({ n, passes, step }, { a }) {
    let src = new Float32Array(n);
    let dst = new Float32Array(n);
    for (let i = 0; i < n; i++) src[i] = a[i] + step;
    for (let p = 1; p < passes; p++) {
      for (let i = 0; i < n; i++) dst[i] = src[i] + step;
      const t = src;
      src = dst;
      dst = t;
    }
    return src;
  },

  gpujs(gpu, { n, passes, step }, { a }) {
    // Hands the array to the GPU once and returns a GPU-resident handle. On the
    // CPU backend pipeline mode is a no-op and this is just a copy, which is
    // the right answer there: on a CPU the data is already where it needs to be.
    const upload = gpu
      .createKernel(function (x) {
        return x[this.thread.x];
      })
      .setOutput([n])
      .setPipeline(true);

    // Two kernels, not one called twice. A pipeline kernel reuses its own
    // output texture, so feeding a kernel its own last result would have it
    // reading and writing the same texture. Ping-ponging between two kernels is
    // the cheap way to say "these are different buffers" and mean it.
    const mk = () =>
      gpu
        .createKernel(function (x) {
          return x[this.thread.x] + this.constants.step;
        })
        .setConstants({ step })
        .setOutput([n])
        .setPipeline(true);
    const even = mk();
    const odd = mk();

    return {
      async run() {
        let cur = await upload(a);
        for (let p = 0; p < passes; p++) {
          // eslint-disable-next-line no-await-in-loop
          cur = await (p % 2 === 0 ? even : odd)(cur);
        }
        // The read-back is what proves all 512 passes actually ran.
        return cur.toArray ? await cur.toArray() : cur;
      },
      backend: () => even.kernel && even.kernel.constructor.mode,
      destroy() {
        [upload, even, odd].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU. The whole point of this column on this row: all 512
   * dispatches go into a single command buffer with one submit, so what it
   * measures is the GPU's own per-dispatch cost with no JavaScript in the loop
   * at all. WebGPU orders and synchronises dispatches within a compute pass, so
   * pass p genuinely sees pass p-1's writes.
   */
  async webgpu(device, { n, passes, step }, { a }) {
    const bytes = n * 4;
    const S = GPUBufferUsage.STORAGE;
    const bufA = device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const bufB = device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const WG = 256;
    const module = device.createShaderModule({
      code: `
struct P { n: u32, step: f32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  dst[i] = src[i] + p.step;
}`,
    });

    const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uni, 0, new Uint32Array([n, 0, 0, 0]));
    device.queue.writeBuffer(uni, 4, new Float32Array([step]));

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = (from, to) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: from } },
          { binding: 1, resource: { buffer: to } },
          { binding: 2, resource: { buffer: uni } },
        ],
      });
    const ab = bind(bufA, bufB);
    const ba = bind(bufB, bufA);
    const groups = Math.ceil(n / WG);
    // pass p writes B when p is even, A when p is odd, so an even pass count
    // leaves the answer in A.
    const finalBuf = passes % 2 === 0 ? bufA : bufB;

    return {
      async run() {
        // One upload per run, matching every other column on this row.
        device.queue.writeBuffer(bufA, 0, a);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (let p = 0; p < passes; p++) {
          pass.setBindGroup(0, p % 2 === 0 ? ab : ba);
          pass.dispatchWorkgroups(groups);
        }
        pass.end();
        enc.copyBufferToBuffer(finalBuf, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufA, bufB, read, uni].forEach(x => x.destroy && x.destroy());
      },
    };
  },

  // Index-weighted so a backend that wrote only part of the buffer cannot match
  // by luck, and averaged so the magnitude stays near 1 where fp32 and fp64
  // agree far better than the runner's 1e-4.
  reduce(out, { n }) {
    let acc = 0;
    for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    return acc / n;
  },
};
