/**
 * Smith–Waterman dynamic programming, computed anti-diagonal by anti-diagonal.
 *
 * THIS ROW IS ALLOWED TO LOSE, and it is here because it is allowed to lose.
 *
 * Every other row in the table is embarrassingly parallel: a million threads,
 * no two of which need anything from each other. Local sequence alignment is
 * not. Cell (i, j) needs (i-1, j-1), (i-1, j) and (i, j-1), so the only set of
 * cells that can be computed at once is an anti-diagonal — everything with
 * i + j equal. A 4096 × 4096 score matrix therefore has 8191 dependent steps,
 * and no amount of hardware collapses them, because the last cell genuinely
 * depends on the first through a chain that long.
 *
 * That gives the GPU an awkward shape: 8191 dispatches with an average of about
 * 2700 useful lanes each, of arithmetic so trivial (three adds, three maxima)
 * that a dispatch costs more than the work inside it. Plain JS, meanwhile, gets
 * to walk the matrix row by row with perfect cache behaviour and no
 * per-diagonal overhead at all, and will very likely win this row outright.
 *
 * That is a real result, not a bug to tune away. A table where the GPU wins
 * every row teaches a reader nothing except that the author chose the rows. The
 * useful thing to know about a GPU is the shape of problem it is bad at, and
 * this is the canonical one: a long dependency chain with a small amount of
 * work per link. Nothing here is handicapped to produce that answer — the
 * kernel is the ordinary wavefront formulation, the diagonals are kept in
 * registers-worth of GPU-resident buffers, and there is no read-back until the
 * end. It loses on structure.
 *
 * ── THE LAYOUT ─────────────────────────────────────────────────────────────
 *
 * Diagonal d holds the cells with i + j = d; lane k on that diagonal is cell
 * (i = k, j = d - k). A cell's three predecessors then live at fixed places:
 * (i-1, j-1) is lane k-1 of diagonal d-2, and (i-1, j) and (i, j-1) are lanes
 * k-1 and k of diagonal d-1. So only three diagonals ever need to exist at
 * once, and they rotate.
 *
 * Anything off the edge of the matrix is stored as zero, which makes every
 * boundary case fall out for free: Smith–Waterman's max(0, …) means a gap step
 * from a nonexistent cell scores -1 and loses to the zero anyway.
 *
 * ── WHY THE OUTPUT IS 4096 NUMBERS AND NOT 16 MILLION ──────────────────────
 *
 * The full score matrix is 16.7 million cells — 67 MB. Materialising it would
 * make this row a memory benchmark with a dynamic program attached: the
 * read-back alone would dwarf the arithmetic, and every column's number would
 * mostly be PCIe. So each lane carries a running weighted sum of every cell it
 * has ever computed, alongside the score, in the upper half of the same buffer.
 * The output is those 4096 accumulators.
 *
 * reduce() therefore touches all 4096 outputs and each of them is the folded-up
 * total of a whole row of 4096 real cells — every one of the 16.7 million is in
 * there, at its own weight, which is what rule 3 is actually asking for. A
 * backend that computed a single cell, or a single diagonal, or every diagonal
 * but the last, cannot land on this number.
 *
 * Every value in this row is a small integer: the scores are integers by
 * construction and the accumulators stay well under 2^24, which fp32 carries
 * exactly. So the columns must agree EXACTLY — the runner's 1e-4 tolerance is
 * slack this row never uses — whether a backend holds them in f32 (gpu.js, which
 * has no integer type on its GL backends) or in i32 (the hand-written WGSL).
 */

// 6144, not 4096: the anti-diagonal sweep at 4096 measured 122 ms, under the
// band where the clock stops mattering. Cost is quadratic in N.
const N = 6144;

// Match, mismatch, gap. Over a 4-letter alphabet the expected score of a random
// aligned pair is 0.25*2 + 0.75*(-1) = -0.25, i.e. negative, which puts this in
// the logarithmic regime: local alignments between two random sequences stay
// short and the scores stay small. That is what keeps the accumulators inside
// the exactly-representable range — see the header.
const MATCH = 2;
const MISMATCH = -1;
const GAP = -1;

// Accumulator weight period. Coprime with everything else in sight, so a
// backend that shifted a diagonal by one, or transposed a pair of lanes, gets a
// different total rather than the same one.
const PERIOD = 7;

export default {
  id: 'wavefront',
  name: 'Smith–Waterman alignment',
  params: `${N} × ${N} score matrix · ${2 * N - 1} anti-diagonals`,
  tag: 'diagonal dependency',
  group: 'sim',
  size: { n: N },

  make({ n }) {
    // Two seeded 4-letter sequences. Seeded, because two columns have to align
    // the same strings for their checksums to mean anything.
    //
    // The letters come from the TOP two bits of the generator, which matters
    // more than it looks. Bit k of a power-of-two-modulus LCG repeats with
    // period 2^(k+1), so taking the letters from low bits gives a sequence that
    // repeats every few hundred characters — and then the second sequence,
    // drawn from the same stream at a multiple of that period, comes out
    // IDENTICAL to the first. Aligning a string with itself scores 2 per
    // character, the accumulators overflow the exactly-representable range, and
    // the row quietly stops being checkable. The top bits have the full period.
    let s = 0x27d4eb2f >>> 0;
    const seq = len => {
      const out = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        s = (s * 1664525 + 1013904223) >>> 0;
        out[i] = s >>> 30;
      }
      return out;
    };
    const a = seq(n);
    const b = seq(n);
    // The same letters in the shapes the three backends want. Built once here,
    // never inside a column (rule 2).
    return {
      a,
      b,
      aF: Float32Array.from(a),
      bF: Float32Array.from(b),
      aU: Uint32Array.from(a),
      bU: Uint32Array.from(b),
    };
  },

  /**
   * The oracle, and a fair baseline — which here means it does NOT walk
   * anti-diagonals. Nothing about the problem forces a serial implementation to
   * traverse it that way; the ordinary row-major recurrence produces the same
   * matrix with one row of history, sequential memory access and no
   * per-diagonal bookkeeping, and that is what an honest plain-JS baseline
   * looks like. Making the baseline zig-zag along diagonals to "match" the GPU
   * would be slowing it down to flatter the other columns, which is precisely
   * the failure this table is built to avoid.
   *
   * Two Int32Arrays of history, swapped by reference. The accumulator is
   * summed per row, and a row of this matrix is exactly the set of cells that
   * lane i owns on the GPU — so the two orders produce the same 4096 numbers.
   */
  js({ n }, { a, b }) {
    const acc = new Float32Array(n);
    let prev = new Int32Array(n + 1); // prev[j + 1] = H(i - 1, j)
    let cur = new Int32Array(n + 1);

    for (let i = 0; i < n; i++) {
      const ai = a[i];
      // The 2*i is not decoration: the GPU weights cell (i, j) by its diagonal
      // index i + j and its lane i, and i + (i + j) is 2i + j.
      const wBase = 2 * i;
      cur[0] = 0;
      let rowAcc = 0;
      for (let j = 0; j < n; j++) {
        const diag = prev[j] + (ai === b[j] ? MATCH : MISMATCH);
        const up = prev[j + 1] + GAP;
        const left = cur[j] + GAP;
        let h = diag > up ? diag : up;
        if (left > h) h = left;
        if (h < 0) h = 0;
        cur[j + 1] = h;
        rowAcc += h * (1 + ((wBase + j) % PERIOD));
      }
      acc[i] = rowAcc;
      const t = prev;
      prev = cur;
      cur = t;
    }
    return acc;
  },

  gpujs(gpu, { n }, { aF, bF }) {
    const width = 2 * n; // [0, n) scores, [n, 2n) accumulators
    const diagonals = 2 * n; // 2n-1 real diagonals, plus one pass to flush the accumulator

    // One kernel body, three kernel objects. A pipelined mutable kernel reuses
    // one output texture, and this recurrence needs the two PREVIOUS diagonals
    // live while it writes the current one — so the three rotate, and no kernel
    // is ever asked to write a texture it is reading.
    const body = function (prev1, prev2, d) {
      // `dim`, not `n`: a kernel-local named after one of the kernel's own
      // constants collides with the CPU backend's generated `constants_n`.
      const dim = this.constants.n;
      const t = this.thread.x;

      if (t >= dim) {
        // Accumulator lane. It folds in the PREVIOUS diagonal's scores, which
        // are already sitting in prev1, so nothing is recomputed and the whole
        // diagonal still costs one dispatch.
        const k = t - dim;
        let m = k + d + this.constants.period - 1;
        m = m - this.constants.period * Math.floor(m / this.constants.period);
        return prev1[t] + (1 + m) * prev1[k];
      }

      // Score lane. Off the end of this diagonal, or off the matrix, is zero —
      // which is what makes every boundary case below correct without a
      // special case for it.
      if (t > d) return 0;
      const j = d - t;
      if (j >= dim) return 0;

      let sub = this.constants.mismatch;
      if (this.constants.seqA[t] === this.constants.seqB[j]) sub = this.constants.match;

      // t === 0 is the top row of the matrix: H(-1, j-1) and H(-1, j) are both
      // outside it and count as zero.
      let best = sub;
      let up = this.constants.gap;
      if (t > 0) {
        best = prev2[t - 1] + sub;
        up = prev1[t - 1] + this.constants.gap;
      }
      if (up > best) best = up;
      const left = prev1[t] + this.constants.gap;
      if (left > best) best = left;
      if (best < 0) return 0;
      return best;
    };

    const constants = {
      n,
      seqA: aF,
      seqB: bF,
      match: MATCH,
      mismatch: MISMATCH,
      gap: GAP,
      period: PERIOD,
    };
    const build = () =>
      gpu.createKernel(body).setConstants(constants).setPipeline(true).setPrecision('single').setOutput([width]);
    const rotation = [build(), build(), build()];

    // Diagonals -1 and -2 are all zeros. One kernel, called once per run, and
    // its single texture can serve as both because they are only ever read.
    const zeros = gpu
      .createKernel(function () {
        return 0;
      })
      .setPipeline(true)
      .setPrecision('single')
      .setOutput([width]);

    return {
      async run() {
        const zero = await zeros();
        let prev1 = zero;
        let prev2 = zero;
        for (let d = 0; d < diagonals; d++) {
          const cur = await rotation[d % 3](prev1, prev2, d);
          prev2 = prev1;
          prev1 = cur;
        }
        // The read-back. Without it this would return with 8192 dispatches
        // still queued and the row would report the cost of queueing them.
        const flat = prev1.toArray ? await prev1.toArray() : prev1;
        return flat.slice(n); // the accumulator half
      },
      backend: () => rotation[0].kernel && rotation[0].kernel.constructor.mode,
      destroy() {
        [...rotation, zeros].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU, borrowing nothing from gpu.js.
   *
   * Two things this cell can do that the gpu.js column cannot:
   *
   *   - the three rotating diagonals live in ONE storage buffer at three
   *     offsets, so rotating them is arithmetic on an index rather than three
   *     kernel objects taking turns.
   *   - all 8192 dispatches go into a single compute pass, with the diagonal
   *     index read through a dynamic uniform offset out of a pre-filled buffer.
   *     Rewriting a uniform between dispatches would force a submit per
   *     diagonal, and the row would be measuring the queue.
   *
   * Scores are i32 here rather than f32. WGSL has real integers and this
   * recurrence is integer arithmetic; the values are small enough that f32
   * carries them exactly too, which is why this column and the gpu.js ones must
   * still agree to the last bit.
   */
  async webgpu(device, { n }, { aU, bU }) {
    const width = 2 * n;
    const diagonals = 2 * n;
    const S = GPUBufferUsage.STORAGE;

    const upload = data => {
      const buf = device.createBuffer({ size: data.byteLength, usage: S, mappedAtCreation: true });
      new Uint32Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return buf;
    };
    const bufA = upload(aU);
    const bufB = upload(bU);
    // Three diagonals of 2n slots each. 96 KB in total — the whole working set
    // of this dynamic program, which is why none of it needs to be a texture.
    const bufRing = device.createBuffer({ size: 3 * width * 4, usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const read = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // One 256-byte slot per diagonal — 256 is the dynamic-offset alignment.
    const STRIDE = 256;
    const plan = new Uint32Array((diagonals * STRIDE) / 4);
    for (let d = 0; d < diagonals; d++) plan[(d * STRIDE) / 4] = d;
    const bufPlan = device.createBuffer({
      size: plan.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufPlan, 0, plan);

    const GROUP = 64;
    const module = device.createShaderModule({
      code: `
struct Diag { d: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read_write> ring: array<i32>;
@group(0) @binding(1) var<storage, read> seqA: array<u32>;
@group(0) @binding(2) var<storage, read> seqB: array<u32>;
@group(0) @binding(3) var<uniform> p: Diag;

const N: u32 = ${n}u;
const W: u32 = ${width}u;
const MATCH: i32 = ${MATCH};
const MISMATCH: i32 = ${MISMATCH};
const GAP: i32 = ${GAP};
const PERIOD: u32 = ${PERIOD}u;

@compute @workgroup_size(${GROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= W) { return; }
  let d = p.d;
  // cur = d, prev1 = d-1, prev2 = d-2, modulo three, written as additions so
  // the u32 arithmetic never goes negative at d = 0.
  let cur = (d % 3u) * W;
  let p1  = ((d + 2u) % 3u) * W;
  let p2  = ((d + 1u) % 3u) * W;

  if (t >= N) {
    let k = t - N;
    let w = i32(1u + ((k + d + PERIOD - 1u) % PERIOD));
    ring[cur + t] = ring[p1 + t] + w * ring[p1 + k];
    return;
  }

  var h: i32 = 0;
  if (t <= d && (d - t) < N) {
    let j = d - t;
    var sub = MISMATCH;
    if (seqA[t] == seqB[j]) { sub = MATCH; }
    var best = sub;   // t == 0: H(-1, j-1) is off the matrix, so zero
    var up = GAP;     // t == 0: H(-1, j)   is off the matrix, so zero
    if (t > 0u) {
      best = ring[p2 + t - 1u] + sub;
      up = ring[p1 + t - 1u] + GAP;
    }
    best = max(best, up);
    best = max(best, ring[p1 + t] + GAP);
    h = max(best, 0);
  }
  ring[cur + t] = h;
}`,
    });

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: 'main' },
    });
    const bind = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: bufRing } },
        { binding: 1, resource: { buffer: bufA } },
        { binding: 2, resource: { buffer: bufB } },
        { binding: 3, resource: { buffer: bufPlan, offset: 0, size: 16 } },
      ],
    });
    const groups = Math.ceil(width / GROUP);
    // Where the last diagonal lands, and where its accumulator half starts.
    const finalSlot = (diagonals - 1) % 3;
    const accOffset = (finalSlot * width + n) * 4;

    return {
      async run() {
        const enc = device.createCommandEncoder();
        enc.clearBuffer(bufRing); // diagonals -1 and -2 are zero
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (let d = 0; d < diagonals; d++) {
          // Dispatches inside one pass are ordered and each one's writes are
          // visible to the next, which is exactly the guarantee a wavefront
          // needs and the reason this is one submit rather than 8192.
          pass.setBindGroup(0, bind, [d * STRIDE]);
          pass.dispatchWorkgroups(groups);
        }
        pass.end();
        enc.copyBufferToBuffer(bufRing, accOffset, read, 0, n * 4);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Int32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufA, bufB, bufRing, bufPlan, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * The 4096 per-lane accumulators, each of which already carries a whole row
   * of the score matrix at per-cell weights. Weighted again by lane here, so
   * that two rows swapped, or one row missing, changes the answer.
   *
   * Everything in this sum is a whole number, so the result is exact in fp64
   * and identical across backends to the last bit.
   */
  reduce(out, { n }) {
    let acc = 0;
    for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 13));
    return acc / n;
  },
};
