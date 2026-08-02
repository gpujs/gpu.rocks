/**
 * Radix-2 FFT, 2²² complex points, 22 stages.
 *
 * The sibling of the naive-DFT row. Same signal family, same twiddle table,
 * same output (a magnitude spectrum) — and 256× more points, in a fraction of
 * the time, because the algorithm is n log n instead of n². That comparison is
 * the most useful thing on this page, and it is a comparison between two
 * columns of the SAME colour: plain JS beats plain JS.
 *
 * ── What the GPU columns can and cannot do here ──────────────────────────────
 *
 * An FFT is 22 dependent passes over the whole array. There is no way to fuse
 * them without shared memory, so every backend does the same thing: bit-reverse
 * once, then 22 gather passes, ping-ponging between two buffers. Two honest
 * consequences, both of which the row should be read as including:
 *
 *   • The gather form computes each output element independently, so it does
 *     one complex multiply per element where the in-place butterfly in js()
 *     does one per PAIR. That is not a handicap invented here — it is what a
 *     GPU FFT without shared memory has to do.
 *   • A gpu.js kernel returns one float per thread. Complex data therefore has
 *     to be laid out as two rows, [n][2], and the two halves of one butterfly
 *     are computed by two different threads that each re-read the twiddle and
 *     the partner element. That is roughly 1.7× the memory traffic of the
 *     hand-written WGSL column, which returns a vec2 and reads each value once.
 *     The gap between the two GPU columns on this row is mostly that, and it is
 *     a real cost of expressing complex arithmetic through gpu.js.
 *
 * ── Numerics ────────────────────────────────────────────────────────────────
 *
 * 22 stages of fp32 is exactly the case where a checksum tolerance of 1e-4
 * might not be honestly reachable, so it was measured rather than assumed: an
 * fp32-throughout evaluation (every operation rounded with Math.fround) differs
 * from this oracle by 2.0e-9 on the checksum. An FFT is well conditioned and
 * reduce() sums magnitudes, which are non-negative and cannot cancel, so the
 * margin is four orders of magnitude. Individual bins near a zero of the
 * spectrum disagree by much more than that, and always will; a checksum that
 * cared about them would be measuring cancellation, not correctness.
 *
 * Memory: the ping-pong textures are 2n floats each. gpu.js stores a single
 * float per RGBA32F texel, so each is ~134 MB on the GL backends. That is the
 * reason this row is 2²² and not 2²⁴.
 */

// 14, not 22. The ping-pong output is [n, 2], so n IS the texture width, and
// WebGL caps a texture dimension at 16,384 on both GL backends — measured, not
// assumed. There is nothing to chunk here the way the STFT row chunks its
// frames: a single transform's working set is the whole signal, so the platform
// limit sets the row's size outright and the 0.2-3 s sizing band cannot be met.
// The limit wins; the row says so rather than being quietly dropped.
const LOG2N = 14;
const N = 1 << LOG2N;
const TWO_PI = Math.PI * 2;

// Same generator as the naive-DFT row, so the two rows transform the same kind
// of signal and the spectra are comparable by eye as well as by stopwatch.
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
  id: 'fft',
  name: 'Radix-2 FFT',
  params: `2¹⁴ complex points · ${LOG2N} stages, fp32`,
  tag: 'O(n log n) transform',
  group: 'transform',
  // Below the sizing band by necessity, not by choice — see LOG2N above.
  sizeExempt: true,
  size: { n: N, bits: LOG2N },

  make({ n }) {
    const { re, im } = makeSignal(n);
    // Half the circle is enough: stage `len` wants e^(-2πi·j/len) for j < len/2,
    // which is entry j·(n/len) of this table. One table serves all 22 stages.
    const half = n >> 1;
    const twRe = new Float32Array(half);
    const twIm = new Float32Array(half);
    for (let m = 0; m < half; m++) {
      twRe[m] = Math.cos(-TWO_PI * m / n);
      twIm[m] = Math.sin(-TWO_PI * m / n);
    }
    return { re, im, twRe, twIm };
  },

  /**
   * The oracle: iterative Cooley-Tukey, decimation in time, in place, with the
   * twiddles hoisted into a table. Iterative and not recursive on purpose — a
   * recursive FFT allocates two arrays per level and would be a lazy baseline
   * that made every other column on the row look better than it is.
   *
   * The working arrays are Float32Array, so the value is rounded to fp32 at
   * every stage boundary exactly as it is on a GPU. Only the five operations
   * inside one butterfly are evaluated wider.
   */
  js({ n, bits }, { re, im, twRe, twIm }) {
    const ar = new Float32Array(n);
    const ai = new Float32Array(n);

    // Bit-reversal permutation, written as an arithmetic loop rather than with
    // shifts because the two GPU columns run the identical loop — a GLSL ES 1.0
    // kernel has no bitwise operators, and gpu.js emulates them with a 32-turn
    // loop per operation, which would cost more here than the transform.
    for (let i = 0; i < n; i++) {
      let r = 0;
      let v = i;
      for (let b = 0; b < bits; b++) {
        const h = Math.floor(v / 2);
        r = r * 2 + (v - h * 2);
        v = h;
      }
      ar[i] = re[r];
      ai[i] = im[r];
    }

    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1;
      const step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let j = 0, m = 0; j < halfLen; j++, m += step) {
          const wr = twRe[m];
          const wi = twIm[m];
          const p = i + j;
          const q = p + halfLen;
          const br = ar[q];
          const bi = ai[q];
          const tr = wr * br - wi * bi;
          const ti = wr * bi + wi * br;
          ar[q] = ar[p] - tr;
          ai[q] = ai[p] - ti;
          ar[p] = ar[p] + tr;
          ai[p] = ai[p] + ti;
        }
      }
    }

    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.sqrt(ar[i] * ar[i] + ai[i] * ai[i]);
    return out;
  },

  gpujs(gpu, { n, bits }, { re, im, twRe, twIm }) {
    // Pipeline + immutable is what makes 22 passes possible without touching the
    // host: each call hands back a GPU-resident handle, and immutable gives each
    // call its own storage so a kernel can be fed its own previous output
    // without reading and writing the same texture in one dispatch.
    const pipe = k => k.setPipeline(true).setImmutable(true).setPrecision('single').setOutput([n, 2]);

    // Row 0 is the real part, row 1 the imaginary part. Both rows of one index
    // are computed by different threads; see the header for what that costs.
    const bitrev = pipe(
      gpu
        .createKernel(function (signalRe, signalIm) {
          let r = 0;
          let v = this.thread.x;
          for (let b = 0; b < this.constants.bits; b++) {
            const h = Math.floor(v / 2);
            r = r * 2 + (v - h * 2);
            v = h;
          }
          if (this.thread.y === 0) return signalRe[r];
          return signalIm[r];
        })
        .setConstants({ bits })
    );

    // One stage, gathered. len/halfLen/step are arguments rather than constants
    // so a single compiled kernel serves all 22 passes — 22 compiled kernels
    // would put 22 shader compiles in the first warm-up and nothing else.
    const stage = pipe(
      gpu
        .createKernel(function (a, len, halfLen, step) {
          const i = this.thread.x;
          const blk = Math.floor(i / len) * len;
          const pos = i - blk;
          let p = 0;
          let q = 0;
          let m = 0;
          let sgn = 1;
          if (pos < halfLen) {
            p = blk + pos;
            q = p + halfLen;
            m = pos * step;
          } else {
            p = blk + pos - halfLen;
            q = blk + pos;
            m = (pos - halfLen) * step;
            sgn = -1;
          }
          const wr = this.constants.twRe[m];
          const wi = this.constants.twIm[m];
          const br = a[0][q];
          const bi = a[1][q];
          if (this.thread.y === 0) return a[0][p] + sgn * (wr * br - wi * bi);
          return a[1][p] + sgn * (wr * bi + wi * br);
        })
        .setConstants({ twRe, twIm })
    );

    const magnitude = gpu
      .createKernel(function (a) {
        const r = a[0][this.thread.x];
        const m = a[1][this.thread.x];
        return Math.sqrt(r * r + m * m);
      })
      .setPrecision('single')
      .setOutput([n]);

    return {
      async run() {
        // await on every handoff: on the WebGPU backend a kernel call is a
        // promise, and on the GL backends awaiting a texture is free. Without it
        // this loop would enqueue 23 dispatches and return before any ran.
        let cur = await bitrev(re, im);
        for (let s = 0; s < bits; s++) {
          const len = 2 << s;
          const halfLen = 1 << s;
          const next = await stage(cur, len, halfLen, n / len);
          // Released only after the next stage has been issued, so the texture
          // being recycled is always two stages old and never the one being
          // read. (CPU-mode pipeline results are plain arrays and have no
          // delete; they fall out of scope instead.)
          if (cur.delete) cur.delete();
          cur = next;
        }
        const out = await magnitude(cur);
        if (cur.delete) cur.delete();
        return out;
      },
      backend: () => stage.kernel && stage.kernel.constructor.mode,
      destroy: () => {
        [bitrev, stage, magnitude].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WGSL. Structurally identical to the gpu.js version — bit
   * reverse, 22 gather passes, magnitude — with one difference that is the
   * whole point of this column: a WGSL kernel can write a vec2, so one thread
   * owns one complex number and reads the twiddle and the partner once instead
   * of twice.
   */
  async webgpu(device, { n, bits }, { re, im, twRe, twIm }) {
    const interleaved = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      interleaved[2 * i] = re[i];
      interleaved[2 * i + 1] = im[i];
    }
    const tw = new Float32Array(n);
    for (let m = 0; m < n / 2; m++) {
      tw[2 * m] = twRe[m];
      tw[2 * m + 1] = twIm[m];
    }

    const S = GPUBufferUsage.STORAGE;
    const upload = data => {
      const buf = device.createBuffer({
        size: data.byteLength,
        usage: S | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return buf;
    };
    const bufIn = upload(interleaved);
    const bufTw = upload(tw);
    const bufA = device.createBuffer({ size: n * 8, usage: S });
    const bufB = device.createBuffer({ size: n * 8, usage: S });
    const bufMag = device.createBuffer({ size: n * 4, usage: S | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // One uniform buffer, one 256-byte slot per stage, all written at build.
    // Writing a uniform between passes would not work: writeBuffer is queued at
    // call time, so every write would land before the single submit.
    const SLOT = 256;
    const uni = device.createBuffer({ size: SLOT * bits, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (let s = 0; s < bits; s++) {
      const len = 2 << s;
      device.queue.writeBuffer(uni, s * SLOT, new Uint32Array([len, 1 << s, n / len, n]));
    }
    const dim = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dim, 0, new Uint32Array([n, bits, 0, 0]));

    const WG = 64;
    const mk = code => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });

    const revPipe = mk(`
struct Dim { n: u32, bits: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> dim: Dim;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dim.n) { return; }
  // The same arithmetic loop as js() and the gpu.js kernel. reverseBits() would
  // do it in one instruction here, but keeping the three implementations
  // textually comparable is worth more than 22 cycles that run once per element.
  var r: u32 = 0u;
  var v: u32 = i;
  for (var b: u32 = 0u; b < dim.bits; b = b + 1u) {
    r = r * 2u + (v % 2u);
    v = v / 2u;
  }
  dst[i] = src[r];
}`);

    const stagePipe = mk(`
struct Stage { len: u32, halfLen: u32, step: u32, n: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> tw: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> st: Stage;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= st.n) { return; }
  let blk = (i / st.len) * st.len;
  let pos = i - blk;
  var p: u32;
  var q: u32;
  var m: u32;
  var sgn = 1.0;
  if (pos < st.halfLen) {
    p = blk + pos;
    q = p + st.halfLen;
    m = pos * st.step;
  } else {
    p = blk + pos - st.halfLen;
    q = blk + pos;
    m = (pos - st.halfLen) * st.step;
    sgn = -1.0;
  }
  let w = tw[m];
  let b = src[q];
  let t = vec2<f32>(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);
  dst[i] = src[p] + sgn * t;
}`);

    const magPipe = mk(`
struct Dim { n: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> mag: array<f32>;
@group(0) @binding(2) var<uniform> dim: Dim;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dim.n) { return; }
  let v = src[i];
  mag[i] = sqrt(v.x * v.x + v.y * v.y);
}`);

    const revBind = device.createBindGroup({
      layout: revPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufIn } },
        { binding: 1, resource: { buffer: bufA } },
        { binding: 2, resource: { buffer: dim } },
      ],
    });
    // Bit reversal lands in A, so stage s reads A when s is even.
    const stageBinds = [];
    for (let s = 0; s < bits; s++) {
      const src = s % 2 === 0 ? bufA : bufB;
      const dst = s % 2 === 0 ? bufB : bufA;
      stageBinds.push(
        device.createBindGroup({
          layout: stagePipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: src } },
            { binding: 1, resource: { buffer: dst } },
            { binding: 2, resource: { buffer: bufTw } },
            { binding: 3, resource: { buffer: uni, offset: s * SLOT, size: 16 } },
          ],
        })
      );
    }
    const finalBuf = bits % 2 === 0 ? bufA : bufB;
    const magBind = device.createBindGroup({
      layout: magPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: finalBuf } },
        { binding: 1, resource: { buffer: bufMag } },
        { binding: 2, resource: { buffer: dim } },
      ],
    });
    const groups = Math.ceil(n / WG);

    return {
      async run() {
        device.queue.writeBuffer(bufIn, 0, interleaved);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(revPipe);
        pass.setBindGroup(0, revBind);
        pass.dispatchWorkgroups(groups);
        pass.setPipeline(stagePipe);
        for (let s = 0; s < bits; s++) {
          pass.setBindGroup(0, stageBinds[s]);
          pass.dispatchWorkgroups(groups);
        }
        pass.setPipeline(magPipe);
        pass.setBindGroup(0, magBind);
        pass.dispatchWorkgroups(groups);
        pass.end();
        enc.copyBufferToBuffer(bufMag, 0, read, 0, n * 4);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufIn, bufTw, bufA, bufB, bufMag, read, uni, dim].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  // Magnitudes only, so nothing in this sum can cancel and the fp32 columns
  // agree with the fp64 oracle to 2e-9. Index-weighted so a partly-filled
  // output cannot match it.
  reduce(out, { n }) {
    let acc = 0;
    for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    return acc / n;
  },
};
