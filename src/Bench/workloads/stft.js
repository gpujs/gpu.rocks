/**
 * Short-time Fourier transform: 6144 windowed frames of 2048 points, hop 1024.
 *
 * The FFT row asks how fast one big transform is. This row asks the question a
 * GPU is actually good at: thousands of small INDEPENDENT transforms, none of
 * which is worth a dispatch on its own. Every stage here is one dispatch over a
 * whole chunk of frames at once, so the per-frame transform never has to be
 * large enough to fill the machine — the batch fills it. That is the entire
 * point of the row, and it is the reason a 2048-point transform, which a CPU
 * does in ten microseconds, is worth putting on a GPU at all.
 *
 * ── Two sizing decisions worth defending ────────────────────────────────────
 *
 * The obvious shape for this row is a few hundred frames of 1024 bins. That is
 * 11 ms of plain JavaScript — under the floor where a measurement means
 * anything, because a 1024-point frame fits in L1 and the plain-JS baseline is
 * genuinely, legitimately quick at it. Reaching the sizing band takes about
 * 12.6 million samples of signal, hence 6144 frames.
 *
 * That much complex data does not fit comfortably in gpu.js's ping-pong
 * textures — a gpu.js single-precision texel holds one float in an RGBA32F
 * slot, so the working set would be over half a gigabyte. So the batch is
 * processed in chunks of 1024 frames: six chunks, 13 dispatches each. Chunking
 * a batch is what a real streaming spectrogram does anyway, and it keeps the
 * working set at 67 MB per buffer.
 *
 * Only the lower half of each spectrum is returned. The input is real, so the
 * upper half is its mirror; every STFT implementation in the world returns
 * bins/2, and returning the redundant half would double a 25 MB readback to
 * measure nothing.
 *
 * See fft.js for why complex data through gpu.js costs roughly 1.7× the memory
 * traffic of the hand-written WGSL column. The same [n][2] layout is used here.
 */

const FRAMES = 6144;
const BINS = 2048;
const LOG2_BINS = 11;
const HOP = BINS / 2;
// 8, not 1024. The ping-pong output is [chunkFrames * bins, 2], so the texture
// is that wide, and WebGL caps a dimension at 16,384 on both GL backends
// (measured). 8 x 2048 = 16,384 exactly. My first cut used 16 and assumed the
// half-spectrum; the kernel ping-pongs the FULL bins, so it asked for 32,768.
// Only the chunk shrinks: all 6144 frames are still transformed, in more passes.
const CHUNK_FRAMES = 8;
const TWO_PI = Math.PI * 2;

export default {
  id: 'stft',
  name: 'Short-time Fourier transform',
  params: `${FRAMES} frames × ${BINS} bins, hop ${HOP}, fp32`,
  tag: 'batched transform',
  group: 'transform',
  size: { frames: FRAMES, bins: BINS, bits: LOG2_BINS, hop: HOP, chunkFrames: CHUNK_FRAMES },

  make({ frames, bins, hop }) {
    // A chirp plus a seeded noise floor: the tone sweeps across the spectrum, so
    // no frame's spectrum looks like its neighbour's and a backend that
    // transformed one frame and copied it cannot pass the checksum. The tone is
    // advanced by complex rotation rather than by calling sin() per sample —
    // make() is shared by every column and should not cost more than the work.
    const length = (frames - 1) * hop + bins;
    const signal = new Float32Array(length);
    let s = 0x9e3779b9 >>> 0;
    let pr = 1;
    let pi = 0;
    let w = TWO_PI * 0.002;
    for (let i = 0; i < length; i++) {
      s = (s * 1664525 + 1013904223) >>> 0;
      const c = Math.cos(w);
      const sn = Math.sin(w);
      const nr = pr * c - pi * sn;
      pi = pr * sn + pi * c;
      pr = nr;
      signal[i] = pr + 0.3 * ((s >>> 8) / 0x1000000 - 0.5);
      w += TWO_PI * 3e-9;
    }

    // Periodic Hann window, and half a twiddle circle for a `bins`-point FFT.
    const window = new Float32Array(bins);
    for (let i = 0; i < bins; i++) window[i] = 0.5 - 0.5 * Math.cos(TWO_PI * i / bins);
    const half = bins >> 1;
    const twRe = new Float32Array(half);
    const twIm = new Float32Array(half);
    for (let m = 0; m < half; m++) {
      twRe[m] = Math.cos(-TWO_PI * m / bins);
      twIm[m] = Math.sin(-TWO_PI * m / bins);
    }
    return { signal, window, twRe, twIm };
  },

  /**
   * The oracle: one frame at a time, iterative in-place radix-2, twiddles and
   * the bit-reversal permutation hoisted out of the frame loop. Frame-at-a-time
   * is not a handicap — it is the fast way to do this on a CPU, because a
   * 2048-point frame stays in cache for all 11 stages, where the GPU has to
   * stream the whole chunk through memory 11 times. If this row's speed-up
   * looks smaller than the plain FFT row's, that cache advantage is why.
   */
  js({ frames, bins, bits, hop }, { signal, window, twRe, twIm }) {
    const halfBins = bins >> 1;
    const out = new Float32Array(frames * halfBins);
    const ar = new Float32Array(bins);
    const ai = new Float32Array(bins);

    const rev = new Uint32Array(bins);
    for (let i = 0; i < bins; i++) {
      let r = 0;
      let v = i;
      for (let b = 0; b < bits; b++) {
        const h = Math.floor(v / 2);
        r = r * 2 + (v - h * 2);
        v = h;
      }
      rev[i] = r;
    }

    for (let f = 0; f < frames; f++) {
      const base = f * hop;
      for (let i = 0; i < bins; i++) {
        const r = rev[i];
        ar[i] = signal[base + r] * window[r];
        ai[i] = 0;
      }
      for (let len = 2; len <= bins; len <<= 1) {
        const halfLen = len >> 1;
        const step = bins / len;
        for (let i = 0; i < bins; i += len) {
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
      const o = f * halfBins;
      for (let i = 0; i < halfBins; i++) out[o + i] = Math.sqrt(ar[i] * ar[i] + ai[i] * ai[i]);
    }
    return out;
  },

  gpujs(gpu, { frames, bins, bits, hop, chunkFrames }, { signal, window, twRe, twIm }) {
    const halfBins = bins >> 1;
    const chunks = frames / chunkFrames;
    const perChunk = chunkFrames * bins;
    const chunkSamples = (chunkFrames - 1) * hop + bins;

    // Real copies rather than subarray views, built once: a view carries a byte
    // offset that a texture upload has no reason to respect, and finding that
    // out as silently wrong numbers is not worth saving 25 MB.
    const chunkData = [];
    for (let c = 0; c < chunks; c++) {
      const base = c * chunkFrames * hop;
      chunkData.push(signal.slice(base, base + chunkSamples));
    }
    const out = new Float32Array(frames * halfBins);

    const pipe = k => k.setPipeline(true).setImmutable(true).setPrecision('single').setOutput([perChunk, 2]);

    // Window, permute and lift to complex in one pass. Thread x walks the whole
    // chunk; the frame it belongs to is x / bins, which is exact because bins is
    // a power of two.
    const load = pipe(
      gpu
        .createKernel(function (x) {
          const gi = this.thread.x;
          const frame = Math.floor(gi / this.constants.bins);
          const i = gi - frame * this.constants.bins;
          let r = 0;
          let v = i;
          for (let b = 0; b < this.constants.bits; b++) {
            const h = Math.floor(v / 2);
            r = r * 2 + (v - h * 2);
            v = h;
          }
          if (this.thread.y === 0) return x[frame * this.constants.hop + r] * this.constants.window[r];
          return 0;
        })
        .setConstants({ bins, bits, hop, window })
    );

    // One butterfly stage, for every frame in the chunk at once. The frame
    // offset is added back at the end so the butterfly indices stay local — a
    // frame never reads another frame's samples, which is what makes this a
    // batch and not one big transform.
    const stage = pipe(
      gpu
        .createKernel(function (a, len, halfLen, step) {
          const gi = this.thread.x;
          const frame = Math.floor(gi / this.constants.bins);
          const local = gi - frame * this.constants.bins;
          const blk = Math.floor(local / len) * len;
          const pos = local - blk;
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
          const off = frame * this.constants.bins;
          const wr = this.constants.twRe[m];
          const wi = this.constants.twIm[m];
          const br = a[0][off + q];
          const bi = a[1][off + q];
          if (this.thread.y === 0) return a[0][off + p] + sgn * (wr * br - wi * bi);
          return a[1][off + p] + sgn * (wr * bi + wi * br);
        })
        .setConstants({ bins, twRe, twIm })
    );

    const magnitude = gpu
      .createKernel(function (a) {
        const gi = this.thread.x;
        const frame = Math.floor(gi / this.constants.halfBins);
        const bin = gi - frame * this.constants.halfBins;
        const idx = frame * this.constants.bins + bin;
        const r = a[0][idx];
        const m = a[1][idx];
        return Math.sqrt(r * r + m * m);
      })
      .setConstants({ bins, halfBins })
      .setPrecision('single')
      .setOutput([chunkFrames * halfBins]);

    return {
      async run() {
        for (let c = 0; c < chunks; c++) {
          let cur = await load(chunkData[c]);
          for (let s = 0; s < bits; s++) {
            const len = 2 << s;
            const next = await stage(cur, len, 1 << s, bins / len);
            if (cur.delete) cur.delete();
            cur = next;
          }
          const part = await magnitude(cur);
          if (cur.delete) cur.delete();
          out.set(part, c * chunkFrames * halfBins);
        }
        return out;
      },
      backend: () => stage.kernel && stage.kernel.constructor.mode,
      destroy: () => {
        [load, stage, magnitude].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WGSL. The signal lives in one storage buffer and each chunk is
   * addressed with an offset, so the six chunks cost six offsets rather than six
   * uploads. All 78 dispatches go into one command encoder and one submit — the
   * gpu.js column pays a submit per kernel call, and that gap is what this
   * column is for.
   */
  async webgpu(device, { frames, bins, bits, hop, chunkFrames }, { signal, window, twRe, twIm }) {
    const halfBins = bins >> 1;
    const chunks = frames / chunkFrames;
    const perChunk = chunkFrames * bins;

    const tw = new Float32Array(bins);
    for (let m = 0; m < halfBins; m++) {
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
    const bufSignal = upload(signal);
    const bufWindow = upload(window);
    const bufTw = upload(tw);
    const bufA = device.createBuffer({ size: perChunk * 8, usage: S });
    const bufB = device.createBuffer({ size: perChunk * 8, usage: S });
    const outBytes = frames * halfBins * 4;
    const bufOut = device.createBuffer({ size: outBytes, usage: S | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const dim = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dim, 0, new Uint32Array([perChunk, bins, bits, hop, halfBins, 0, 0, 0]));

    const SLOT = 256;
    const chunkUni = device.createBuffer({ size: SLOT * chunks, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (let c = 0; c < chunks; c++) {
      device.queue.writeBuffer(
        chunkUni,
        c * SLOT,
        new Uint32Array([c * chunkFrames * hop, c * chunkFrames * halfBins, 0, 0])
      );
    }
    const stageUni = device.createBuffer({ size: SLOT * bits, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (let s = 0; s < bits; s++) {
      const len = 2 << s;
      device.queue.writeBuffer(stageUni, s * SLOT, new Uint32Array([len, 1 << s, bins / len, 0]));
    }

    const WG = 64;
    const mk = code => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    const DIM = `struct Dim { perChunk: u32, bins: u32, bits: u32, hop: u32, halfBins: u32 };`;

    const loadPipe = mk(`
${DIM}
struct Chunk { sampleBase: u32, outBase: u32 };
@group(0) @binding(0) var<storage, read> signal: array<f32>;
@group(0) @binding(1) var<storage, read> win: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> dim: Dim;
@group(0) @binding(4) var<uniform> ck: Chunk;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gi = gid.x;
  if (gi >= dim.perChunk) { return; }
  let frame = gi / dim.bins;
  let i = gi - frame * dim.bins;
  var r: u32 = 0u;
  var v: u32 = i;
  for (var b: u32 = 0u; b < dim.bits; b = b + 1u) {
    r = r * 2u + (v % 2u);
    v = v / 2u;
  }
  dst[gi] = vec2<f32>(signal[ck.sampleBase + frame * dim.hop + r] * win[r], 0.0);
}`);

    const stagePipe = mk(`
${DIM}
struct Stage { len: u32, halfLen: u32, step: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> tw: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> dim: Dim;
@group(0) @binding(4) var<uniform> st: Stage;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gi = gid.x;
  if (gi >= dim.perChunk) { return; }
  let frame = gi / dim.bins;
  let local = gi - frame * dim.bins;
  let blk = (local / st.len) * st.len;
  let pos = local - blk;
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
  let off = frame * dim.bins;
  let w = tw[m];
  let b = src[off + q];
  let t = vec2<f32>(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);
  dst[gi] = src[off + p] + sgn * t;
}`);

    const magPipe = mk(`
${DIM}
struct Chunk { sampleBase: u32, outBase: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> mag: array<f32>;
@group(0) @binding(2) var<uniform> dim: Dim;
@group(0) @binding(3) var<uniform> ck: Chunk;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gi = gid.x;
  let total = (dim.perChunk / dim.bins) * dim.halfBins;
  if (gi >= total) { return; }
  let frame = gi / dim.halfBins;
  let bin = gi - frame * dim.halfBins;
  let v = src[frame * dim.bins + bin];
  mag[ck.outBase + gi] = sqrt(v.x * v.x + v.y * v.y);
}`);

    const buf = [bufA, bufB];
    const loadBinds = [];
    const magBinds = [];
    for (let c = 0; c < chunks; c++) {
      loadBinds.push(
        device.createBindGroup({
          layout: loadPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: bufSignal } },
            { binding: 1, resource: { buffer: bufWindow } },
            { binding: 2, resource: { buffer: buf[0] } },
            { binding: 3, resource: { buffer: dim } },
            { binding: 4, resource: { buffer: chunkUni, offset: c * SLOT, size: 16 } },
          ],
        })
      );
    }
    // The ping-pong schedule is the same for every chunk, so the stage bind
    // groups are built once and reused across all six.
    const stageBinds = [];
    for (let s = 0; s < bits; s++) {
      stageBinds.push(
        device.createBindGroup({
          layout: stagePipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: buf[s % 2] } },
            { binding: 1, resource: { buffer: buf[1 - (s % 2)] } },
            { binding: 2, resource: { buffer: bufTw } },
            { binding: 3, resource: { buffer: dim } },
            { binding: 4, resource: { buffer: stageUni, offset: s * SLOT, size: 16 } },
          ],
        })
      );
    }
    const finalBuf = buf[bits % 2];
    for (let c = 0; c < chunks; c++) {
      magBinds.push(
        device.createBindGroup({
          layout: magPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: finalBuf } },
            { binding: 1, resource: { buffer: bufOut } },
            { binding: 2, resource: { buffer: dim } },
            { binding: 3, resource: { buffer: chunkUni, offset: c * SLOT, size: 16 } },
          ],
        })
      );
    }
    const groups = Math.ceil(perChunk / WG);
    const magGroups = Math.ceil((chunkFrames * halfBins) / WG);

    return {
      async run() {
        device.queue.writeBuffer(bufSignal, 0, signal);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let c = 0; c < chunks; c++) {
          pass.setPipeline(loadPipe);
          pass.setBindGroup(0, loadBinds[c]);
          pass.dispatchWorkgroups(groups);
          pass.setPipeline(stagePipe);
          for (let s = 0; s < bits; s++) {
            pass.setBindGroup(0, stageBinds[s]);
            pass.dispatchWorkgroups(groups);
          }
          pass.setPipeline(magPipe);
          pass.setBindGroup(0, magBinds[c]);
          pass.dispatchWorkgroups(magGroups);
        }
        pass.end();
        enc.copyBufferToBuffer(bufOut, 0, read, 0, outBytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufSignal, bufWindow, bufTw, bufA, bufB, bufOut, read, dim, chunkUni, stageUni].forEach(
          b => b.destroy && b.destroy()
        );
      },
    };
  },

  // Every magnitude in the spectrogram, index-weighted. Non-negative terms, so
  // nothing cancels; the weight walks across bins AND frames, so a backend that
  // transformed one chunk and left the rest cannot match it.
  reduce(out, { frames, bins }) {
    let acc = 0;
    for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    return acc / (frames * (bins >> 1));
  },
};
