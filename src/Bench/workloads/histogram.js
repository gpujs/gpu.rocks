/**
 * Orientation histogram — many threads, 256 destinations.
 *
 * Every sample is a gradient direction, and every sample has to land in one of
 * 256 counters. That is the whole point of the row: the arithmetic is trivial
 * and the destinations are few, so what is being priced is the WRITE, not the
 * work. This is the shape of every histogram, every accumulator, every
 * "count things into buckets" kernel, and it is the first place the GPU's
 * you-write-your-own-cell model stops being free.
 *
 * The three columns solve it three genuinely different ways, and the row is
 * worth reading precisely because of that difference:
 *
 *   plain JS   one pass, one counter array, `h[b]++`. No contention at all —
 *              a single thread cannot contend with itself.
 *   gpu.js     no atomics, and a kernel thread may only write its own output
 *              cell. So the loop nest is TRANSPOSED: one thread per (bin, row),
 *              each scanning a row and counting the samples that match its bin.
 *              Same counts, same rule, 256× the reads. That factor is not a
 *              strawman — it is what "no scatter-add" costs, and pretending
 *              otherwise by giving each thread a private 256-entry histogram is
 *              not possible here anyway: a gpu.js thread returns ONE number.
 *   WGSL       `atomicAdd` into a workgroup-local 256-bin array, then one
 *              flush per workgroup into global memory. One pass over the data,
 *              which is exactly the primitive gpu.js does not expose.
 *
 * So the gap between the two right-hand GPU columns is not tuning. It is a
 * missing instruction.
 *
 * WHY THE BIN COSTS SOMETHING. A histogram of pre-computed bin indices runs at
 * about a nanosecond a sample in JS — 4 ms for a whole image — which is far too
 * short to time and would make the row a report on the harness. The bin here is
 * derived the way a real orientation histogram derives it (central differences,
 * then atan2), so every column pays the same per-sample arithmetic and the
 * baseline lands where it can be measured. The neighbourhood wraps toroidally
 * so that every one of the n² samples is a real sample and no column needs a
 * border special case.
 *
 * MEMORY. gpu.js stores one float per RGBA32F texel, so the image is ~151 MB of
 * VRAM and the bin texture another ~151 MB. That is the real cost of this size
 * on the gpu.js path and it is worth knowing before the tab is opened.
 */

const N = 3072;
const BINS = 256;

// Deterministic and cheap, as in matmul: two columns must be handed the same
// bytes, and a seeded generator is the only way to say that and mean it.
function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s >>> 8) / 0x1000000;
  }
  return a;
}

export default {
  id: 'histogram',
  name: 'Orientation histogram',
  params: `${N} × ${N} samples → ${BINS} bins, fp32`,
  tag: 'write contention',
  group: 'movement',
  size: { n: N, bins: BINS },

  make({ n }) {
    return { img: fill(new Float32Array(n * n), 0xc2b2ae35) };
  },

  /**
   * The oracle. One sequential pass, one counter array small enough to sit in
   * L1 forever. This is the fair baseline: there is no faster way to write it
   * in plain JS, and writing it any slower (a bin array per row, say) would
   * hand every GPU column a speed-up it did not earn.
   *
   * The bin is `floor(...) % bins` rather than a clamp because the quantity is
   * an ANGLE: an atan2 of exactly +pi lands on bin 256, and 256 is bin 0. Every
   * column can then use the identical expression with no edge case to disagree
   * about.
   */
  js({ n, bins }, { img }) {
    const h = new Float32Array(bins);
    const scale = bins / (2 * Math.PI);
    for (let y = 0; y < n; y++) {
      const rowUp = ((y + n - 1) % n) * n;
      const rowDn = ((y + 1) % n) * n;
      const row = y * n;
      for (let x = 0; x < n; x++) {
        const gx = img[row + ((x + 1) % n)] - img[row + ((x + n - 1) % n)];
        const gy = img[rowDn + x] - img[rowUp + x];
        h[Math.floor((Math.atan2(gy, gx) + Math.PI) * scale) % bins]++;
      }
    }
    return h;
  },

  gpujs(gpu, { n, bins }, { img }) {
    // 2-D rows, built once. Reshaping inside run() would time the reshape.
    const rows = [];
    for (let y = 0; y < n; y++) rows.push(img.subarray(y * n, y * n + n));

    // Stage 1: the bin index of every sample, kept on the GPU as a texture.
    // It has to be materialised rather than recomputed inside the count kernel:
    // the count kernel visits every sample 256 times, and 256 × n² atan2 calls
    // would be measuring the transcendental, not the histogram.
    const code = gpu
      // `w` rather than `n`: a kernel-local name that collides with a constant
      // name compiles to `const constants_n = constants_n` and throws.
      .createKernel(function (im) {
        const w = this.constants.n;
        const x = this.thread.x;
        const y = this.thread.y;
        const gx = im[y][(x + 1) % w] - im[y][(x + w - 1) % w];
        const gy = im[(y + 1) % w][x] - im[(y + w - 1) % w][x];
        return Math.floor((Math.atan2(gy, gx) + this.constants.pi) * this.constants.scale) % this.constants.bins;
      })
      .setConstants({ n, bins, pi: Math.PI, scale: bins / (2 * Math.PI) })
      .setOutput([n, n])
      .setPipeline(true);

    // Stage 2: the transposed loop nest. Thread (row, bin) scans one image row
    // and counts the samples in its bin — a gather, because a gpu.js thread can
    // only write its own cell. Comparison is |code - bin| < 0.5 rather than ==
    // so that it does not depend on a float texture round-tripping bit-exactly.
    const partial = gpu
      .createKernel(function (codes) {
        const b = this.thread.y;
        const y = this.thread.x;
        let c = 0;
        for (let x = 0; x < this.constants.n; x++) {
          if (Math.abs(codes[y][x] - b) < 0.5) c = c + 1;
        }
        return c;
      })
      .setConstants({ n })
      .setOutput([n, bins])
      .setPipeline(true);

    // Stage 3: n partial counts per bin, summed. Cheap — bins × n, not bins × n².
    const total = gpu
      .createKernel(function (part) {
        let c = 0;
        for (let t = 0; t < this.constants.n; t++) c = c + part[this.thread.x][t];
        return c;
      })
      .setConstants({ n })
      .setOutput([bins]);

    return {
      // Nothing here returns before the counts are in a JS array: `total` is a
      // plain (non-pipeline) kernel, so its result is already read back.
      async run() {
        const codes = await code(rows);
        const part = await partial(codes);
        const out = await total(part);
        return out.toArray ? out.toArray() : out;
      },
      backend: () => total.kernel && total.kernel.constructor.mode,
      destroy() {
        [code, partial, total].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU, with no gpu.js in it. This is the column that shows
   * what the missing instruction is worth: 256 `atomic<u32>` counters in
   * workgroup memory, one atomicAdd per sample, and a single flush of 256
   * atomicAdds per workgroup into the global histogram. One pass over the
   * image, against gpu.js's 256.
   *
   * Each workgroup chews through 256 × PER_THREAD samples so that the flush is
   * amortised; the stride is 256 so that neighbouring invocations read
   * neighbouring floats.
   */
  async webgpu(device, { n, bins }, { img }) {
    const total = n * n;
    const PER_THREAD = 64;
    const perGroup = bins * PER_THREAD;
    const groups = Math.ceil(total / perGroup);

    const imgBuf = device.createBuffer({
      size: total * 4,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Float32Array(imgBuf.getMappedRange()).set(img);
    imgBuf.unmap();

    const histBuf = device.createBuffer({
      size: bins * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const read = device.createBuffer({
      size: bins * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const dim = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dim, 0, new Uint32Array([n, bins, PER_THREAD, total]));
    const zeros = new Uint32Array(bins);

    const module = device.createShaderModule({
      code: `
struct Dim { n: u32, bins: u32, perThread: u32, total: u32 };
@group(0) @binding(0) var<storage, read> img: array<f32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> dim: Dim;

var<workgroup> local: array<atomic<u32>, ${bins}>;

@compute @workgroup_size(${bins})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  atomicStore(&local[lid.x], 0u);
  workgroupBarrier();

  let n = dim.n;
  let base = wid.x * ${bins}u * dim.perThread + lid.x;
  for (var s: u32 = 0u; s < dim.perThread; s = s + 1u) {
    let i = base + s * ${bins}u;
    if (i < dim.total) {
      let y = i / n;
      let x = i - y * n;
      let xr = select(x + 1u, 0u, x + 1u == n);
      let xl = select(x - 1u, n - 1u, x == 0u);
      let yd = select(y + 1u, 0u, y + 1u == n);
      let yu = select(y - 1u, n - 1u, y == 0u);
      let gx = img[y * n + xr] - img[y * n + xl];
      let gy = img[yd * n + x] - img[yu * n + x];
      let a = (atan2(gy, gx) + ${Math.PI}) * ${BINS / (2 * Math.PI)};
      atomicAdd(&local[u32(floor(a)) % dim.bins], 1u);
    }
  }

  workgroupBarrier();
  atomicAdd(&hist[lid.x], atomicLoad(&local[lid.x]));
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: imgBuf } },
        { binding: 1, resource: { buffer: histBuf } },
        { binding: 2, resource: { buffer: dim } },
      ],
    });

    return {
      async run() {
        // The histogram accumulates, so it is zeroed per run. 1 KB on the queue
        // ahead of the submit — real work, and far too small to move the number.
        device.queue.writeBuffer(histBuf, 0, zeros);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(groups);
        pass.end();
        enc.copyBufferToBuffer(histBuf, 0, read, 0, bins * 4);
        device.queue.submit([enc.finish()]);
        // The read-back is the only thing that proves the dispatch finished.
        await read.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(read.getMappedRange()).slice();
        read.unmap();
        return Float32Array.from(counts);
      },
      destroy() {
        [imgBuf, histBuf, read, dim].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Every bin, with a weight that rises smoothly across them, divided by the
   * sample count. Smoothly on purpose: atan2 is fp64 in JS and fp32 on a GPU,
   * so a few hundred of the nine million samples sit close enough to a bin edge
   * to land one bin either side, and a checksum that jumped by 16 for each of
   * those would report arithmetic noise as WRONG. A weight that differs by
   * 1/256 between neighbours absorbs that and still separates a backend that
   * filled the wrong bins, counted part of the image, or dropped samples.
   */
  reduce(h, { n, bins }) {
    const flat = ArrayBuffer.isView(h) ? h : Float32Array.from(h);
    let acc = 0;
    for (let b = 0; b < flat.length; b++) acc += flat[b] * (1 + b / bins);
    return acc / (n * n);
  },
};
