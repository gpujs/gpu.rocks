/**
 * Naive O(n²) discrete Fourier transform.
 *
 * This row exists to be read next to the FFT row. Both compute a Fourier
 * transform of the same kind of signal, both use the same precomputed twiddle
 * table, and both are written as well as the author could write them — the ONLY
 * difference between them is the algorithm. That is the comparison worth
 * printing: the FFT row beats every GPU column on this row by a margin no
 * hardware can close, and it does it in plain JavaScript.
 *
 * Two decisions that keep the pair honest:
 *
 *   1. The twiddles come from a table on BOTH sides. A textbook naive DFT calls
 *      cos() and sin() 2n² times; a GPU has a transcendental unit and JavaScript
 *      does not, so that version would measure the sin unit rather than the
 *      algorithm, and would flatter every GPU column by an order of magnitude
 *      for a reason that has nothing to do with the DFT. Anybody writing this
 *      for real would hoist the table anyway.
 *   2. The twiddle index is advanced by addition — m += k, wrapped — rather than
 *      computed as (k*j) mod n. k*j reaches n² = 2.7e8, which is past the 24-bit
 *      integer range of an fp32 GPU register, so the multiply-and-mod form would
 *      silently give a different (wrong) angle on the GPU columns. The running
 *      sum never exceeds 2n and is exact everywhere.
 *
 * Sizing note: n is 16384 rather than the "few thousand" the shape suggests,
 * because with a twiddle table the inner loop is cheap enough that a few
 * thousand points lands at 30 ms — under the floor where a measurement means
 * anything. n² is 2.7e8 inner iterations, which is 0.54 s of plain JS.
 */

const N = 16384;
const TWO_PI = Math.PI * 2;

// Deterministic, so every column is handed the same bytes: two tones, a third
// tone in quadrature, and a seeded noise floor so the spectrum is not so sparse
// that most of the output is zero (a checksum over mostly-zeros is a weak one).
function makeSignal(n) {
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  let s = 0x9e3779b9 >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const noise = (s >>> 8) / 0x1000000 - 0.5;
    const t = i / n;
    re[i] = Math.sin(TWO_PI * 5 * t) + 0.5 * Math.sin(TWO_PI * 37 * t + 0.4) + 0.25 * noise;
    im[i] = 0.3 * Math.cos(TWO_PI * 11 * t) + 0.25 * noise;
  }
  return { re, im };
}

export default {
  id: 'dft-naive',
  name: 'Naive DFT',
  params: `${N.toLocaleString()} complex points, O(n²), fp32`,
  tag: 'quadratic transform',
  group: 'transform',
  size: { n: N },

  make({ n }) {
    const { re, im } = makeSignal(n);
    // W[m] = e^(-2πi·m/n), the full circle. The FFT row only needs half of it;
    // a quadratic transform touches every angle.
    const twRe = new Float32Array(n);
    const twIm = new Float32Array(n);
    for (let m = 0; m < n; m++) {
      twRe[m] = Math.cos(-TWO_PI * m / n);
      twIm[m] = Math.sin(-TWO_PI * m / n);
    }
    return { re, im, twRe, twIm };
  },

  /**
   * The oracle. Flat typed arrays, one pass over the input per output bin, and
   * the twiddle index carried in a register instead of recomputed — this is
   * what a careful person writes, and a slower baseline here would inflate
   * every other column on the row.
   *
   * The accumulators are plain (fp64) numbers. Every GPU column accumulates
   * 16384 terms in fp32 instead; measured against this oracle that costs up to
   * 1e-3 of relative error on an individual bin with heavy cancellation, but
   * only 8e-8 on the checksum, because reduce() sums magnitudes and the errors
   * are independent across bins. See the note on reduce().
   */
  js({ n }, { re, im, twRe, twIm }) {
    const out = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      let sumRe = 0;
      let sumIm = 0;
      let m = 0;
      for (let j = 0; j < n; j++) {
        const wr = twRe[m];
        const wi = twIm[m];
        const xr = re[j];
        const xi = im[j];
        sumRe += xr * wr - xi * wi;
        sumIm += xr * wi + xi * wr;
        m += k;
        if (m >= n) m -= n;
      }
      out[k] = Math.sqrt(sumRe * sumRe + sumIm * sumIm);
    }
    return out;
  },

  gpujs(gpu, { n }, { re, im, twRe, twIm }) {
    // The twiddles are CONSTANTS, not arguments: gpu.js re-uploads an argument
    // on every call, and re-uploading 128 KB of table per run would be timing a
    // transfer that no real caller would repeat. The signal stays an argument,
    // so getting the problem data onto the device is still inside the measured
    // run — the same bargain matmul.js strikes.
    const kernel = gpu
      .createKernel(function (signalRe, signalIm) {
        const k = this.thread.x;
        let sumRe = 0;
        let sumIm = 0;
        let m = 0;
        for (let j = 0; j < this.constants.n; j++) {
          const wr = this.constants.twRe[m];
          const wi = this.constants.twIm[m];
          const xr = signalRe[j];
          const xi = signalIm[j];
          sumRe = sumRe + (xr * wr - xi * wi);
          sumIm = sumIm + (xr * wi + xi * wr);
          m += k;
          if (m >= this.constants.n) m -= this.constants.n;
        }
        return Math.sqrt(sumRe * sumRe + sumIm * sumIm);
      })
      .setConstants({ n, twRe, twIm })
      // Explicit, because gpu.js defaults to 'unsigned' when float textures are
      // unavailable, and an 8-bit-per-channel encoding of a spectrum would be
      // wrong by far more than the runner's tolerance. Failing loudly beats a
      // cell that reads fast and is quietly nonsense.
      .setPrecision('single')
      .setOutput([n]);

    return {
      async run() {
        return await kernel(re, im);
      },
      backend: () => kernel.kernel && kernel.kernel.constructor.mode,
      destroy: () => kernel.destroy && kernel.destroy(),
    };
  },

  /**
   * Hand-written WGSL, sharing nothing with gpu.js. One thread per output bin,
   * one linear pass over the signal per thread. There is no clever tiling here
   * on purpose: the gpu.js kernel to the left cannot express one either, so the
   * gap between the two cells is the runtime's overhead and not a different
   * algorithm.
   */
  async webgpu(device, { n }, { re, im, twRe, twIm }) {
    // vec2 twiddles, interleaved once at build time. The interleave is a real
    // cost, but it is a cost of setting up, not of transforming, and make() is
    // shared so it cannot live there.
    const tw = new Float32Array(n * 2);
    for (let m = 0; m < n; m++) {
      tw[2 * m] = twRe[m];
      tw[2 * m + 1] = twIm[m];
    }

    const storage = (data, extra = 0) => {
      const buf = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra,
        mappedAtCreation: true,
      });
      new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return buf;
    };
    const bufRe = storage(re);
    const bufIm = storage(im);
    const bufTw = storage(tw);
    const bufOut = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const dim = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dim, 0, new Uint32Array([n, 0, 0, 0]));

    const WG = 64;
    const module = device.createShaderModule({
      code: `
struct Dim { n: u32 };
@group(0) @binding(0) var<storage, read> sigRe: array<f32>;
@group(0) @binding(1) var<storage, read> sigIm: array<f32>;
@group(0) @binding(2) var<storage, read> tw: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> mag: array<f32>;
@group(0) @binding(4) var<uniform> dim: Dim;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = dim.n;
  let k = gid.x;
  if (k >= n) { return; }
  var sumRe = 0.0;
  var sumIm = 0.0;
  // The same running twiddle index as the JS oracle. u32 here, so the wrap is
  // exact for any n; the fp32 kernel to the left needs it for a stronger
  // reason, and both use it so the two agree bit for bit on the angle.
  var m: u32 = 0u;
  for (var j: u32 = 0u; j < n; j = j + 1u) {
    let w = tw[m];
    let xr = sigRe[j];
    let xi = sigIm[j];
    sumRe = sumRe + (xr * w.x - xi * w.y);
    sumIm = sumIm + (xr * w.y + xi * w.x);
    m = m + k;
    if (m >= n) { m = m - n; }
  }
  mag[k] = sqrt(sumRe * sumRe + sumIm * sumIm);
}`,
    });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufRe } },
        { binding: 1, resource: { buffer: bufIm } },
        { binding: 2, resource: { buffer: bufTw } },
        { binding: 3, resource: { buffer: bufOut } },
        { binding: 4, resource: { buffer: dim } },
      ],
    });
    const groups = Math.ceil(n / WG);

    return {
      async run() {
        // Re-uploaded per run, to match what gpu.js does with an argument: the
        // cell to the left pays this, so this cell pays it too.
        device.queue.writeBuffer(bufRe, 0, re);
        device.queue.writeBuffer(bufIm, 0, im);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(groups);
        pass.end();
        enc.copyBufferToBuffer(bufOut, 0, read, 0, n * 4);
        device.queue.submit([enc.finish()]);
        // Only the mapped read proves the dispatch finished.
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufRe, bufIm, bufTw, bufOut, read, dim].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Magnitudes are non-negative, so this sum cannot cancel and its relative
   * error stays at the level of a single term — which is what makes an fp32
   * column agree with an fp64 oracle to 8e-8 even though one bin in the
   * spectrum disagrees by 1e-3. The index weight is there so a backend that
   * filled only part of the output cannot match by luck.
   */
  reduce(out, { n }) {
    let acc = 0;
    for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    return acc / n;
  },
};
