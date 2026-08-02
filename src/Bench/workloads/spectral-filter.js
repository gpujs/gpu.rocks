/**
 * Convolution by way of the frequency domain: forward FFT, multiply by a
 * kernel spectrum, inverse FFT. 128 signals of 2¹⁴ points, 14 stages each way.
 *
 * The plain FFT row measures a transform. This row measures the thing people
 * actually build transforms for, and the difference matters on a GPU: it is
 * 30 dependent passes rather than 15, with a pointwise multiply and a second
 * bit-reversal wedged in the middle, and every one of those passes is a full
 * round trip through memory that the arithmetic cannot hide. If the FFT row
 * says the GPU is worth n×, this row says what n× survives contact with a
 * pipeline.
 *
 * The filter is a Gaussian low-pass with an integer group delay, so the kernel
 * spectrum is a genuine complex number per bin — a real-valued spectrum would
 * reduce the middle pass to a scale and would not be a convolution worth the
 * name. The delay is an integer so the spectrum stays conjugate-symmetric and
 * the inverse transform lands back on the real axis; the imaginary residue is
 * fp32 noise, and js() discards it, as any filter implementation does.
 *
 * Numerics: an fp32-throughout evaluation of this whole round trip differs from
 * the oracle by 4.9e-8 on the checksum (measured, with Math.fround on every
 * operation). 42 stages of fp32 sounds like it should be worse than that; it is
 * not, because an FFT is well conditioned and reduce() sums magnitudes.
 *
 * See fft.js for why complex data costs gpu.js roughly 1.7× the memory traffic
 * of the hand-written WGSL column — the same two-rows-of-floats layout is used
 * here, for the same reason.
 */

// 14, not 22: the ping-pong output is [n, ...], so n IS the texture width, and
// WebGL caps a texture dimension at 16,384 on both GL backends — measured, not
// assumed. One round trip cannot grow past that.
//
// So the row grows in the batch instead, exactly as fft.js does. One signal was
// 2.0 ms of plain JS and the row reported 0.9x-1.0x across every gpu.js column:
// at that size the whole table was dispatch overhead and clock granularity, and
// sizeExempt hid it from the sizing gate. 128 signals put the baseline at a
// quarter of a second, where the numbers mean something.
//
// One filter over many signals is what this operation IS — a filter bank, a
// channel strip, an image convolved row by row. The kernel spectrum is built
// once and shared across the batch, which is both the honest shape and the
// reason the batch costs no extra bandwidth for the filter itself.
const LOG2N = 14;
const N = 1 << LOG2N;
// 128 x 2.0 ms lands mid-band. Each ping-pong texture is n * 2 * BATCH texels at
// one float per RGBA32F texel, so 0.5 MB per batched signal — 64 MB a texture,
// which is what caps this number rather than the sizing band.
const BATCH = 128;
const TWO_PI = Math.PI * 2;
// The Gaussian low-pass keeps the lowest 1/256th of the spectrum, so its width
// is derived from n rather than fixed — size is the only place a dimension
// lives, and a filter whose bandwidth did not scale would quietly become an
// all-pass at any other n. The group delay is in samples and does not scale.
const SIGMA_FRACTION = 1 / 256;
const DELAY = 5;

export default {
  id: 'spectral-filter',
  name: 'FFT convolution',
  params: `${BATCH} × 2¹⁴ points · forward + multiply + inverse, fp32`,
  tag: 'frequency-domain multiply',
  group: 'transform',
  size: { n: N, bits: LOG2N, batch: BATCH },

  make({ n, batch }) {
    // Real signals: two tones and a seeded noise floor, so the low-pass has
    // something to remove and the output is visibly not the input. One seed per
    // signal — a batch of identical inputs would let a cache serve all of it and
    // would price the memory system rather than the pipeline.
    const signal = [];
    for (let b = 0; b < batch; b++) {
      const one = new Float32Array(n);
      let s = (0x9e3779b9 + b * 0x85ebca6b) >>> 0;
      for (let i = 0; i < n; i++) {
        s = (s * 1664525 + 1013904223) >>> 0;
        const noise = (s >>> 8) / 0x1000000 - 0.5;
        const t = i / n;
        one[i] = Math.sin(TWO_PI * 5 * t) + 0.5 * Math.sin(TWO_PI * 37 * t + 0.4) + 0.5 * noise;
      }
      signal.push(one);
    }

    const half = n >> 1;
    const twRe = new Float32Array(half);
    const twIm = new Float32Array(half);
    for (let m = 0; m < half; m++) {
      twRe[m] = Math.cos(-TWO_PI * m / n);
      twIm[m] = Math.sin(-TWO_PI * m / n);
    }

    // K[i] = gaussian(|f|) · e^(-2πi·i·DELAY/n). Folding the frequency at n/2
    // and using an integer delay together make K conjugate-symmetric, which is
    // what keeps the filtered signal real.
    const kernRe = new Float32Array(n);
    const kernIm = new Float32Array(n);
    const sigma = n * SIGMA_FRACTION;
    for (let i = 0; i < n; i++) {
      const f = i <= n / 2 ? i : n - i;
      const gain = Math.exp(-(f * f) / (2 * sigma * sigma));
      const phase = -TWO_PI * ((i * DELAY) % n) / n;
      kernRe[i] = gain * Math.cos(phase);
      kernIm[i] = gain * Math.sin(phase);
    }
    return { signal, twRe, twIm, kernRe, kernIm };
  },

  /**
   * The oracle. One iterative in-place FFT written once and run twice — the
   * inverse transform is the same code with the twiddles conjugated, which is
   * both the standard identity and the only way to be sure the two directions
   * cannot drift apart.
   */
  js({ n, bits, batch }, { signal, twRe, twIm, kernRe, kernIm }) {
    // Flat and batch-major: element i of signal b is out[b * n + i], which is
    // the order both GPU columns hand back, so reduce() stays one function.
    const out = new Float32Array(n * batch);
    const ar = new Float32Array(n);
    const ai = new Float32Array(n);
    const br = new Float32Array(n);
    const bi = new Float32Array(n);

    const bitReverseInto = (dstRe, dstIm, srcRe, srcIm) => {
      for (let i = 0; i < n; i++) {
        let r = 0;
        let v = i;
        for (let b = 0; b < bits; b++) {
          const h = Math.floor(v / 2);
          r = r * 2 + (v - h * 2);
          v = h;
        }
        dstRe[i] = srcRe[r];
        dstIm[i] = srcIm ? srcIm[r] : 0;
      }
    };

    const transform = (re, im, sign) => {
      for (let len = 2; len <= n; len <<= 1) {
        const halfLen = len >> 1;
        const step = n / len;
        for (let i = 0; i < n; i += len) {
          for (let j = 0, m = 0; j < halfLen; j++, m += step) {
            const wr = twRe[m];
            const wi = sign * twIm[m];
            const p = i + j;
            const q = p + halfLen;
            const br = re[q];
            const bi = im[q];
            const tr = wr * br - wi * bi;
            const ti = wr * bi + wi * br;
            re[q] = re[p] - tr;
            im[q] = im[p] - ti;
            re[p] = re[p] + tr;
            im[p] = im[p] + ti;
          }
        }
      }
    };

    for (let b = 0; b < batch; b++) {
      bitReverseInto(ar, ai, signal[b], null);
      transform(ar, ai, 1);

      for (let i = 0; i < n; i++) {
        const r = ar[i];
        const m = ai[i];
        ar[i] = r * kernRe[i] - m * kernIm[i];
        ai[i] = r * kernIm[i] + m * kernRe[i];
      }

      bitReverseInto(br, bi, ar, ai);
      transform(br, bi, -1);

      const base = b * n;
      for (let i = 0; i < n; i++) out[base + i] = br[i] / n;
    }
    return out;
  },

  gpujs(gpu, { n, bits, batch }, { signal, twRe, twIm, kernRe, kernIm }) {
    // [n, 2 * batch]: x walks one signal and the height carries the batch, so
    // row 2b is signal b's real part and 2b+1 its imaginary part. The stage
    // arithmetic works on x and is therefore untouched by batching.
    const pipe = k => k.setPipeline(true).setImmutable(true).setPrecision('single').setOutput([n, 2 * batch]);

    // Row 0 real, row 1 imaginary. The signal is real, so row 1 starts at zero.
    const bitrevIn = pipe(
      gpu
        .createKernel(function (x) {
          let r = 0;
          let v = this.thread.x;
          for (let b = 0; b < this.constants.bits; b++) {
            const h = Math.floor(v / 2);
            r = r * 2 + (v - h * 2);
            v = h;
          }
          const b2 = Math.floor(this.thread.y / 2);
          if (this.thread.y - b2 * 2 === 0) return x[b2][r];
          return 0;
        })
        .setConstants({ bits })
    );

    // The second bit-reversal reads a texture rather than a host array, so it
    // needs its own kernel — but it can permute both rows with one expression.
    // Batching costs it nothing: y is just a row index and the permutation runs
    // along x, inside one signal, so this kernel is unchanged.
    const bitrevTex = pipe(
      gpu
        .createKernel(function (a) {
          let r = 0;
          let v = this.thread.x;
          for (let b = 0; b < this.constants.bits; b++) {
            const h = Math.floor(v / 2);
            r = r * 2 + (v - h * 2);
            v = h;
          }
          return a[this.thread.y][r];
        })
        .setConstants({ bits })
    );

    // `sign` flips the twiddle's imaginary part, so one compiled kernel does
    // both the forward and the inverse transform — 42 passes, one shader.
    const stage = pipe(
      gpu
        .createKernel(function (a, len, halfLen, step, sign) {
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
          // Every stage length divides n, so a block tiles one signal exactly
          // and never straddles two of them — the block arithmetic above needed
          // no change for the batch.
          const rowRe = Math.floor(this.thread.y / 2) * 2;
          const rowIm = rowRe + 1;
          const wr = this.constants.twRe[m];
          const wi = sign * this.constants.twIm[m];
          const br = a[rowRe][q];
          const bi = a[rowIm][q];
          if (this.thread.y - rowRe === 0) return a[rowRe][p] + sgn * (wr * br - wi * bi);
          return a[rowIm][p] + sgn * (wr * bi + wi * br);
        })
        .setConstants({ twRe, twIm })
    );

    const multiply = pipe(
      gpu
        .createKernel(function (a) {
          const i = this.thread.x;
          // one filter, every signal: the kernel spectrum is indexed by bin
          // alone and is shared across the batch
          const rowRe = Math.floor(this.thread.y / 2) * 2;
          const r = a[rowRe][i];
          const m = a[rowRe + 1][i];
          const kr = this.constants.kernRe[i];
          const ki = this.constants.kernIm[i];
          if (this.thread.y - rowRe === 0) return r * kr - m * ki;
          return r * ki + m * kr;
        })
        .setConstants({ kernRe, kernIm })
    );

    // [n, batch] — one row of filtered samples per signal, which read in order
    // is the same batch-major sequence the other two columns return flat.
    const extract = gpu
      .createKernel(function (a) {
        return a[this.thread.y * 2][this.thread.x] / this.constants.n;
      })
      .setConstants({ n })
      .setPrecision('single')
      .setOutput([n, batch]);

    return {
      async run() {
        const sweep = async (cur, sign) => {
          for (let s = 0; s < bits; s++) {
            const len = 2 << s;
            const next = await stage(cur, len, 1 << s, n / len, sign);
            if (cur.delete) cur.delete();
            cur = next;
          }
          return cur;
        };
        let cur = await bitrevIn(signal);
        cur = await sweep(cur, 1);
        let next = await multiply(cur);
        if (cur.delete) cur.delete();
        cur = next;
        next = await bitrevTex(cur);
        if (cur.delete) cur.delete();
        cur = await sweep(next, -1);
        const out = await extract(cur);
        if (cur.delete) cur.delete();
        return out;
      },
      backend: () => stage.kernel && stage.kernel.constructor.mode,
      destroy: () => {
        [bitrevIn, bitrevTex, stage, multiply, extract].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WGSL. The whole round trip is recorded into ONE command
   * encoder — 44 dispatches, one submit, one readback — which is the shape a
   * real pipeline has and the shape gpu.js cannot quite reach, because every
   * gpu.js kernel call is its own submit.
   */
  async webgpu(device, { n, bits, batch }, { signal, twRe, twIm, kernRe, kernIm }) {
    // Flat and batch-major. 8 bytes a complex number against gpu.js's 32.
    const total = n * batch;
    const flatSignal = new Float32Array(total);
    for (let b = 0; b < batch; b++) flatSignal.set(signal[b], b * n);
    const tw = new Float32Array(n);
    for (let m = 0; m < n / 2; m++) {
      tw[2 * m] = twRe[m];
      tw[2 * m + 1] = twIm[m];
    }
    const kern = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      kern[2 * i] = kernRe[i];
      kern[2 * i + 1] = kernIm[i];
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
    const bufIn = upload(flatSignal);
    const bufTw = upload(tw);
    const bufKern = upload(kern);
    const bufA = device.createBuffer({ size: total * 8, usage: S });
    const bufB = device.createBuffer({ size: total * 8, usage: S });
    const bufOut = device.createBuffer({ size: total * 4, usage: S | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: total * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // 2·bits uniform slots: the forward sweep, then the inverse sweep with the
    // twiddle sign flipped. All written at build, because writeBuffer is queued
    // at call time and a write between passes would land before the submit.
    const SLOT = 256;
    const uni = device.createBuffer({ size: SLOT * bits * 2, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (let d = 0; d < 2; d++) {
      for (let s = 0; s < bits; s++) {
        const len = 2 << s;
        const slot = (d * bits + s) * SLOT;
        // 4th word is the dispatch bound, so it counts the whole batch; len,
        // halfLen and step stay per signal.
        device.queue.writeBuffer(uni, slot, new Uint32Array([len, 1 << s, n / len, total]));
        device.queue.writeBuffer(uni, slot + 16, new Float32Array([d === 0 ? 1 : -1]));
      }
    }
    const dim = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dim, 0, new Uint32Array([n, bits, total, 0]));

    const WG = 64;
    const mk = code => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });

    const revInPipe = mk(`
struct Dim { n: u32, bits: u32, total: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> dim: Dim;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dim.total) { return; }
  // reversal is within one signal, so it works on the offset inside the batch
  // element and reads back from that element's own base
  let base = i - (i % dim.n);
  var r: u32 = 0u;
  var v: u32 = i % dim.n;
  for (var b: u32 = 0u; b < dim.bits; b = b + 1u) {
    r = r * 2u + (v % 2u);
    v = v / 2u;
  }
  dst[i] = vec2<f32>(src[base + r], 0.0);
}`);

    const revTexPipe = mk(`
struct Dim { n: u32, bits: u32, total: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> dim: Dim;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dim.total) { return; }
  let base = i - (i % dim.n);
  var r: u32 = 0u;
  var v: u32 = i % dim.n;
  for (var b: u32 = 0u; b < dim.bits; b = b + 1u) {
    r = r * 2u + (v % 2u);
    v = v / 2u;
  }
  dst[i] = src[base + r];
}`);

    const stagePipe = mk(`
struct Stage { len: u32, halfLen: u32, step: u32, n: u32, sign: f32 };
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
  let raw = tw[m];
  let w = vec2<f32>(raw.x, st.sign * raw.y);
  let b = src[q];
  let t = vec2<f32>(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);
  dst[i] = src[p] + sgn * t;
}`);

    const mulPipe = mk(`
struct Dim { n: u32, bits: u32, total: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> kern: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> dim: Dim;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dim.total) { return; }
  let a = src[i];
  // one filter over every signal: the spectrum is indexed by bin alone
  let k = kern[i % dim.n];
  dst[i] = vec2<f32>(a.x * k.x - a.y * k.y, a.x * k.y + a.y * k.x);
}`);

    const outPipe = mk(`
struct Dim { n: u32, bits: u32, total: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> dim: Dim;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dim.total) { return; }
  dst[i] = src[i].x / f32(dim.n);
}`);

    // Build the whole ping-pong schedule up front: which buffer each of the 44
    // passes reads and writes is fixed, so none of it has to be worked out
    // inside run().
    const buf = [bufA, bufB];
    let cur = 0;
    const bind2 = (pipeline, src, dst) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: src } },
          { binding: 1, resource: { buffer: dst } },
          { binding: 2, resource: { buffer: dim } },
        ],
      });

    const revInBind = bind2(revInPipe, bufIn, buf[cur]);
    const schedule = [];
    for (let d = 0; d < 2; d++) {
      if (d === 1) {
        // multiply, then re-permute, each swapping buffers
        schedule.push({
          pipeline: mulPipe,
          bind: device.createBindGroup({
            layout: mulPipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: buf[cur] } },
              { binding: 1, resource: { buffer: buf[1 - cur] } },
              { binding: 2, resource: { buffer: bufKern } },
              { binding: 3, resource: { buffer: dim } },
            ],
          }),
        });
        cur = 1 - cur;
        schedule.push({ pipeline: revTexPipe, bind: bind2(revTexPipe, buf[cur], buf[1 - cur]) });
        cur = 1 - cur;
      }
      for (let s = 0; s < bits; s++) {
        schedule.push({
          pipeline: stagePipe,
          bind: device.createBindGroup({
            layout: stagePipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: buf[cur] } },
              { binding: 1, resource: { buffer: buf[1 - cur] } },
              { binding: 2, resource: { buffer: bufTw } },
              { binding: 3, resource: { buffer: uni, offset: (d * bits + s) * SLOT, size: 32 } },
            ],
          }),
        });
        cur = 1 - cur;
      }
    }
    const outBind = bind2(outPipe, buf[cur], bufOut);
    const groups = Math.ceil(total / WG);

    return {
      async run() {
        device.queue.writeBuffer(bufIn, 0, flatSignal);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(revInPipe);
        pass.setBindGroup(0, revInBind);
        pass.dispatchWorkgroups(groups);
        for (const step of schedule) {
          pass.setPipeline(step.pipeline);
          pass.setBindGroup(0, step.bind);
          pass.dispatchWorkgroups(groups);
        }
        pass.setPipeline(outPipe);
        pass.setBindGroup(0, outBind);
        pass.dispatchWorkgroups(groups);
        pass.end();
        enc.copyBufferToBuffer(bufOut, 0, read, 0, total * 4);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufIn, bufTw, bufKern, bufA, bufB, bufOut, read, uni, dim].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * The output is a signed waveform, so this sums |y| rather than y: a signed
   * sum over two million samples of a low-passed signal cancels almost
   * perfectly, and a checksum that lands near zero has no relative tolerance
   * left to speak of. Index-weighted, so a backend that produced the right
   * samples in the wrong order — a time-reversed inverse transform, say — is
   * still caught.
   */
  reduce(out, { n, batch }) {
    // Flat from js() and from the bare WebGPU column; an array of rows from
    // gpu.js, whose extract kernel has a 2D output. Walking the rows in order
    // is exactly the flat batch-major order, so the weighting lines up.
    let acc = 0;
    let i = 0;
    if (ArrayBuffer.isView(out)) {
      for (; i < out.length; i++) acc += Math.abs(out[i]) * (1 + (i % 17));
    } else {
      for (const row of out) {
        for (let x = 0; x < row.length; x++, i++) acc += Math.abs(row[x]) * (1 + (i % 17));
      }
    }
    return acc / (n * batch);
  },
};
