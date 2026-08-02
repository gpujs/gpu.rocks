/**
 * Radix-2 FFT, 2¹⁴ complex points, 14 stages.
 *
 * The sibling of the naive-DFT row. Same signal family, same twiddle table,
 * same output (a magnitude spectrum) and — since both rows are capped at the
 * same 16,384 points — exactly the same transform, computed in a fraction of
 * the time because the algorithm is n log n instead of n². Nothing but the
 * algorithm differs between the two rows, which is what makes the comparison
 * worth having, and it is a comparison between two columns of the SAME colour:
 * plain JS beats plain JS.
 *
 * ── What the GPU columns can and cannot do here ──────────────────────────────
 *
 * An FFT is 14 dependent passes over the whole array. There is no way to fuse
 * them without shared memory, so every backend does the same thing: bit-reverse
 * once, then 14 gather passes, ping-ponging between two buffers. Two honest
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
 * 14 stages of fp32 is exactly the case where a checksum tolerance of 1e-4
 * might not be honestly reachable, so it was measured rather than assumed: an
 * fp32-throughout evaluation (every operation rounded with Math.fround) differs
 * from this oracle by 7.0e-9 on the checksum, and the GL backends come in at
 * 7.2e-9. An FFT is well conditioned and reduce() sums magnitudes, which are
 * non-negative and cannot cancel, so the margin is four orders of magnitude.
 * Individual bins near a zero of the spectrum disagree by much more than that,
 * and always will; a checksum that cared about them would be measuring
 * cancellation, not correctness.
 *
 * Worth stating plainly because it was once wrongly blamed for this row: fp32
 * has never been anywhere near the reason a column on it disagreed.
 *
 * Memory: the ping-pong textures are 2 * n * BATCH floats each. gpu.js stores a
 * single float per RGBA32F texel, so each is 0.5 MB per batched transform —
 * around 100 MB here. That, and not the texture-width limit, is what now caps
 * the row; see LOG2N below.
 */

// 14, not 22. The ping-pong output is [n, ...], so n IS the texture width, and
// WebGL caps a texture dimension at 16,384 on both GL backends — measured, not
// assumed. A single transform therefore cannot grow past 2^14.
//
// So the row grows the OTHER way. One transform is about 1.1 ms of plain JS,
// which is not a measurement — it is the clock and one dispatch. BATCH
// independent transforms run per call, stacked into the texture's HEIGHT where
// there is room to spare, and the row lands in the sizing band without any
// single transform exceeding what WebGL can hold.
//
// This is also how an FFT is actually used. Nobody transforms one buffer and
// stops: a spectrogram, a convolution and a channel filter bank all issue many
// independent transforms of the same length, which is exactly this shape.
//
// WHAT IT COSTS THE dft-naive COMPARISON. That row is the sibling of this one —
// same signal, same twiddles, same output, n^2 against n log n — and it is NOT
// batched, because 224 naive DFTs would be two minutes. The two rows are
// still comparable, but per transform rather than per row: divide this row's
// time by BATCH. The header used to promise the times could be read side by
// side, and that is no longer true, so it says this instead.
const LOG2N = 14;
const N = 1 << LOG2N;
// 224 x 1.1 ms clears the 200 ms floor. Each ping-pong texture is n * 2 * BATCH texels
// and gpu.js stores one float per RGBA32F texel at single precision, so the
// working set is 0.5 MB * BATCH — about 100 MB a texture here, two or three
// live at once. That is the real ceiling on this number, not the sizing band.
const BATCH = 224;
const TWO_PI = Math.PI * 2;

// Same generator as the naive-DFT row, so the two rows transform the same kind
// of signal and the spectra are comparable by eye as well as by stopwatch.
function makeSignal(n, seed) {
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  let s = seed >>> 0;
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
  params: `${BATCH} × 2¹⁴ complex points · ${LOG2N} stages, fp32`,
  tag: 'O(n log n) transform',
  group: 'transform',
  size: { n: N, bits: LOG2N, batch: BATCH },

  make({ n, batch }) {
    // An array of typed arrays, which is what gpu.js wants for a 2D input and
    // what js() indexes directly. A different seed per transform: 192 copies of
    // 224 copies of one signal would let a cache serve most of the batch and would price the
    // memory system rather than the transform.
    const re = [];
    const im = [];
    for (let b = 0; b < batch; b++) {
      const sig = makeSignal(n, 0x9e3779b9 + b * 0x85ebca6b);
      re.push(sig.re);
      im.push(sig.im);
    }
    // Half the circle is enough: stage `len` wants e^(-2πi·j/len) for j < len/2,
    // which is entry j·(n/len) of this table. One table serves all 14 stages.
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
  js({ n, bits, batch }, { re, im, twRe, twIm }) {
    // Flat and batch-major, so the index of element i of transform b is
    // b * n + i — the same order the two GPU columns hand back, which is what
    // lets reduce() be one function rather than three.
    const out = new Float32Array(n * batch);
    const ar = new Float32Array(n);
    const ai = new Float32Array(n);

    for (let b = 0; b < batch; b++) {
      const sigRe = re[b];
      const sigIm = im[b];

    // Bit-reversal permutation, written as an arithmetic loop rather than with
    // shifts because the two GPU columns run the identical loop — a GLSL ES 1.0
    // kernel has no bitwise operators, and gpu.js emulates them with a 32-turn
    // loop per operation, which would cost more here than the transform.
    for (let i = 0; i < n; i++) {
      let r = 0;
      let v = i;
      // `bit`, not `b`: b is the batch index in the enclosing loop now.
      for (let bit = 0; bit < bits; bit++) {
        const h = Math.floor(v / 2);
        r = r * 2 + (v - h * 2);
        v = h;
      }
      ar[i] = sigRe[r];
      ai[i] = sigIm[r];
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

    const base = b * n;
      for (let i = 0; i < n; i++) out[base + i] = Math.sqrt(ar[i] * ar[i] + ai[i] * ai[i]);
    }
    return out;
  },

  // async because the twiddle tables are uploaded here, once, and a gpu.js
  // kernel call resolves to a promise on the WebGPU backend. runner.js awaits
  // this builder; the uploads are outside the timed region either way.
  async gpujs(gpu, { n, bits, batch }, { re, im, twRe, twIm }) {
    // Pipeline + immutable is what makes 14 passes possible without touching the
    // host: each call hands back a GPU-resident handle, and immutable gives each
    // call its own storage so a kernel can be fed its own previous output
    // without reading and writing the same texture in one dispatch.
    // [n, 2 * batch]: x still walks one transform, and the height carries the
    // batch. Row 2b is transform b's real part and row 2b+1 its imaginary part,
    // which keeps a complex pair adjacent and leaves x — the axis the stage
    // arithmetic works on — exactly as it was before the row was batched.
    const pipe = k => k.setPipeline(true).setImmutable(true).setPrecision('single').setOutput([n, 2 * batch]);

    // Row 0 is the real part, row 1 the imaginary part. Both rows of one index
    // are computed by different threads; see the header for what that costs.
    const bitrev = pipe(
      gpu
        .createKernel(function (signalRe, signalIm) {
          const b = Math.floor(this.thread.y / 2);
          const part = this.thread.y - b * 2;
          let r = 0;
          let v = this.thread.x;
          for (let s = 0; s < this.constants.bits; s++) {
            const h = Math.floor(v / 2);
            r = r * 2 + (v - h * 2);
            v = h;
          }
          if (part === 0) return signalRe[b][r];
          return signalIm[b][r];
        })
        .setConstants({ bits })
    );

    // The twiddle table, uploaded to the GPU once at build time. Two kernels
    // and not one called twice: a pipelined kernel reuses its own single output
    // texture, so one uploader would hand back the same texture for both tables
    // and the imaginary part would silently become the real one.
    const uploadTable = () =>
      gpu
        .createKernel(function (v) {
          return v[this.thread.x];
        })
        .setPipeline(true)
        .setPrecision('single')
        .setOutput([n >> 1]);
    const twReUp = uploadTable();
    const twImUp = uploadTable();
    const twReTex = await twReUp(twRe);
    const twImTex = await twImUp(twIm);

    // One stage, gathered. len/halfLen/step are arguments rather than constants
    // so a single compiled kernel serves all 14 passes — 14 compiled kernels
    // would put 14 shader compiles in the first warm-up and nothing else.
    //
    // The twiddle tables are ARGUMENTS too, and that is a correctness
    // requirement rather than a preference. gpu.js binds a kernel's array
    // CONSTANTS to texture units once, when the kernel is first called, and
    // never rebinds them; arguments are rebound on every dispatch. Texture
    // units are global to the GL context, and `bitrev`'s two signal arguments
    // sit on units 0 and 1 — exactly where twRe/twIm lived as constants. The
    // first run was therefore correct (this kernel is set up after bitrev's
    // first call) and every run after it read the input signal as its twiddle
    // table. The runner takes its checksum from the third call, so the row
    // reported WRONG. Handing the tables in as arguments costs one texture bind
    // per dispatch and nothing else: they are uploaded once, above.
    const stage = pipe(
      gpu
        .createKernel(function (a, twRe, twIm, len, halfLen, step) {
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
          // The block arithmetic above needed no change for the batch: every
          // stage length divides n, so blocks tile each transform exactly and
          // never straddle two of them.
          const rowRe = Math.floor(this.thread.y / 2) * 2;
          const rowIm = rowRe + 1;
          const wr = twRe[m];
          const wi = twIm[m];
          const br = a[rowRe][q];
          const bi = a[rowIm][q];
          if (this.thread.y - rowRe === 0) return a[rowRe][p] + sgn * (wr * br - wi * bi);
          return a[rowIm][p] + sgn * (wr * bi + wi * br);
        })
    );

    // [n, batch] — one row of magnitudes per transform. Read row by row that is
    // the same batch-major order the other two columns return flat.
    const magnitude = gpu
      .createKernel(function (a) {
        const rowRe = this.thread.y * 2;
        const r = a[rowRe][this.thread.x];
        const m = a[rowRe + 1][this.thread.x];
        return Math.sqrt(r * r + m * m);
      })
      .setPrecision('single')
      .setOutput([n, batch]);

    return {
      async run() {
        // await on every handoff: on the WebGPU backend a kernel call is a
        // promise, and on the GL backends awaiting a texture is free. Without it
        // this loop would enqueue 16 dispatches and return before any ran.
        let cur = await bitrev(re, im);
        for (let s = 0; s < bits; s++) {
          const len = 2 << s;
          const halfLen = 1 << s;
          const next = await stage(cur, twReTex, twImTex, len, halfLen, n / len);
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
        [bitrev, stage, magnitude, twReUp, twImUp].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WGSL. Structurally identical to the gpu.js version — bit
   * reverse, 14 gather passes, magnitude — with one difference that is the
   * whole point of this column: a WGSL kernel can write a vec2, so one thread
   * owns one complex number and reads the twiddle and the partner once instead
   * of twice.
   */
  async webgpu(device, { n, bits, batch }, { re, im, twRe, twIm }) {
    // Flat, batch-major, complex interleaved. 8 bytes a complex number here
    // against gpu.js's 32 — one float per RGBA32F texel is a 4x tax this column
    // does not pay, and at this batch size that is the difference between 29 MB
    // and 112 MB of working set.
    const total = n * batch;
    const interleaved = new Float32Array(total * 2);
    for (let b = 0; b < batch; b++) {
      const sigRe = re[b];
      const sigIm = im[b];
      for (let i = 0; i < n; i++) {
        interleaved[2 * (b * n + i)] = sigRe[i];
        interleaved[2 * (b * n + i) + 1] = sigIm[i];
      }
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
    const bufA = device.createBuffer({ size: total * 8, usage: S });
    const bufB = device.createBuffer({ size: total * 8, usage: S });
    const bufMag = device.createBuffer({ size: total * 4, usage: S | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: total * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // One uniform buffer, one 256-byte slot per stage, all written at build.
    // Writing a uniform between passes would not work: writeBuffer is queued at
    // call time, so every write would land before the single submit.
    const SLOT = 256;
    const uni = device.createBuffer({ size: SLOT * bits, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (let s = 0; s < bits; s++) {
      const len = 2 << s;
      // The 4th word is the dispatch bound, so it counts the whole batch. len,
      // halfLen and step stay per transform — every stage length divides n, so
      // a block never straddles two transforms and the arithmetic is unchanged.
      device.queue.writeBuffer(uni, s * SLOT, new Uint32Array([len, 1 << s, n / len, total]));
    }
    const dim = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dim, 0, new Uint32Array([n, bits, total, 0]));

    const WG = 64;
    const mk = code => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });

    const revPipe = mk(`
struct Dim { n: u32, bits: u32, total: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> dim: Dim;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dim.total) { return; }
  // Reversal is within one transform, so it works on the offset inside the
  // batch element and reads back from that element's own base.
  let base = i - (i % dim.n);
  let k = i % dim.n;
  // The same arithmetic loop as js() and the gpu.js kernel. reverseBits() would
  // do it in one instruction here, but keeping the three implementations
  // textually comparable is worth more than 14 cycles that run once per element.
  var r: u32 = 0u;
  var v: u32 = k;
  for (var b: u32 = 0u; b < dim.bits; b = b + 1u) {
    r = r * 2u + (v % 2u);
    v = v / 2u;
  }
  dst[i] = src[base + r];
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
struct Dim { n: u32, bits: u32, total: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> mag: array<f32>;
@group(0) @binding(2) var<uniform> dim: Dim;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dim.total) { return; }
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
    const groups = Math.ceil(total / WG);

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
        enc.copyBufferToBuffer(bufMag, 0, read, 0, total * 4);
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
  // output cannot match it, and averaged over every element of every transform
  // so the value stays O(1) rather than growing with the batch.
  //
  // Two shapes arrive here: a flat Float32Array from js() and from the bare
  // WebGPU column, and an array of rows from gpu.js, whose magnitude kernel has
  // a 2D output. Walking the rows in order visits exactly the flat batch-major
  // order, so the weighting lines up without a copy.
  reduce(out, { n, batch }) {
    let acc = 0;
    let i = 0;
    if (ArrayBuffer.isView(out)) {
      for (; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    } else {
      for (const row of out) {
        for (let x = 0; x < row.length; x++, i++) acc += row[x] * (1 + (i % 17));
      }
    }
    return acc / (n * batch);
  },
};
