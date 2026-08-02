/**
 * Stream compaction — keep the elements that pass a test, in order, packed.
 *
 * Four million floats, a predicate that about half of them satisfy, and an
 * output that must contain the survivors contiguously and in their original
 * order. In JavaScript this is six lines and one write cursor. On a GPU there
 * is no write cursor, because there is no "next": every thread has to work out,
 * on its own, where its element lands. That answer is the exclusive prefix sum
 * of the predicate, so compaction is flag, scan, then a write to an address
 * that depends on the data — which is the only reason this row exists.
 *
 * THE THING THIS ROW IS ACTUALLY ABOUT. A gpu.js kernel is a pure gather:
 * thread i computes out[i] and cannot write anywhere else. Compaction is a
 * scatter — element i wants to write to position scan[i], which it does not
 * own. The two are not interchangeable, and there is no flag to make gpu.js
 * scatter. So the gpu.js columns invert the problem instead: every OUTPUT slot
 * binary-searches the scanned counts to find which input element belongs to it,
 * 22 probes into a 16 MB array, four million times. It is a real technique and
 * it gives exactly the right answer, and it is also strictly more work than the
 * scatter it is standing in for.
 *
 * The bare-WebGPU column does the scatter, because WGSL has storage buffers and
 * a thread may write wherever it likes: `dst[u32(incl[i]) - 1] = src[i]`. One
 * line, one write, no search. The gap between that cell and the gpu.js cells is
 * not the GPU being fast — it is the cost of a programming model that only
 * offers gather, measured in milliseconds, on a problem that wants scatter.
 * That is the most useful number on this row and possibly on this page.
 *
 * WHAT MIGHT MISLEAD. Do not read the gpu.js number as "gpu.js is bad at
 * compaction" and stop there. Read it as "if your problem needs scatter, the
 * gather model will cost you a log factor" — which tells you something you can
 * act on, and which is true of every gather-only abstraction, not just this one.
 *
 * EXACTNESS. The values are multiples of 2^-12 and the thresholds are odd
 * multiples of 2^-13, so no input can ever sit exactly on a threshold and the
 * comparison is identical in fp32 and fp64. That matters more here than
 * anywhere else in the table: one element deciding differently on one backend
 * would shift every survivor after it by one slot, and the checksum would not
 * be slightly off, it would be unrecognisable. The counts scanned are integers
 * below 2^22, exact in fp32, so this row's columns agree bit for bit or not at
 * all — there is no rounding for a bug to hide behind.
 */

const N = 1 << 22; // 4,194,304
const B = 64; // block width for the scan: 2^22 -> 2^16 -> 2^10 -> 2^4
const ROUNDS = 24;
const STEPS = 22; // log2(N): probes per binary search
const QUANT = 4096; // input values are k / 4096

// Round r keeps everything at or above this. Odd multiples of 2^-13, so never
// equal to an input, and every round compacts a slightly different set.
const threshold = r => (QUANT - 1 + 2 * r) / (2 * QUANT);

function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s >>> 20) / QUANT; // k / 4096, exact in fp32
  }
  return a;
}

// [2^22, 2^16, 2^10, 2^4]
function levelsOf(n, b) {
  const out = [n];
  let m = n;
  while (m > b) {
    m /= b;
    out.push(m);
  }
  return out;
}

export default {
  id: 'compaction',
  name: 'Stream compaction',
  params: `2^22 fp32, ~50% kept, × ${ROUNDS} rounds`,
  tag: 'data-dependent write',
  group: 'movement',
  size: { n: N, b: B, rounds: ROUNDS, steps: STEPS },

  make({ n }) {
    return { a: fill(new Float32Array(n), 0xcc9e2d51) };
  },

  // The oracle, and the whole reason the GPU has to work so hard: one pass, one
  // cursor, one branch. The tail is zeroed from the cursor rather than the
  // whole array being cleared first, because the survivors have already
  // overwritten everything before it.
  js({ n, rounds }, { a }) {
    const out = new Float32Array(n);
    for (let r = 0; r < rounds; r++) {
      const t = threshold(r);
      let j = 0;
      for (let i = 0; i < n; i++) {
        const v = a[i];
        if (v >= t) out[j++] = v;
      }
      out.fill(0, j);
    }
    return out;
  },

  gpujs(gpu, { n, b, rounds, steps }, { a }) {
    const levels = levelsOf(n, b);
    const L = levels.length;

    const upload = gpu
      .createKernel(function (x) {
        return x[this.thread.x];
      })
      .setOutput([n])
      .setPipeline(true);

    // flag → scan → place. Materialising the flags costs one pass over 16 MB
    // and buys a scan that is exactly the textbook one, with the predicate in
    // one place instead of smeared through three kernels.
    const flags = gpu
      .createKernel(function (x, t) {
        return x[this.thread.x] >= t ? 1 : 0;
      })
      .setOutput([n])
      .setPipeline(true);

    const up = levels.slice(1).map(m =>
      gpu
        .createKernel(function (x) {
          const start = this.thread.x * this.constants.b;
          let s = 0;
          for (let k = 0; k < this.constants.b; k++) s += x[start + k];
          return s;
        })
        .setConstants({ b })
        .setOutput([m])
        .setPipeline(true)
    );

    const top = gpu
      .createKernel(function (x) {
        let s = 0;
        for (let k = 0; k < this.constants.m; k++) {
          if (k < this.thread.x) s += x[k];
        }
        return s;
      })
      .setConstants({ m: levels[L - 1] })
      .setOutput([levels[L - 1]])
      .setPipeline(true);

    // Levels above 0 want the exclusive scan; level 0 wants the INCLUSIVE one,
    // because incl[i] is "how many survivors up to and including i", which is
    // exactly what the search below is inverting.
    const mkDown = (m, inclusive) =>
      gpu
        .createKernel(
          inclusive
            ? function (x, off) {
                const blk = Math.floor(this.thread.x * this.constants.invb);
                const start = blk * this.constants.b;
                let s = off[blk];
                for (let j = 0; j < this.constants.b; j++) {
                  if (start + j <= this.thread.x) s += x[start + j];
                }
                return s;
              }
            : function (x, off) {
                const blk = Math.floor(this.thread.x * this.constants.invb);
                const start = blk * this.constants.b;
                let s = off[blk];
                for (let j = 0; j < this.constants.b; j++) {
                  if (start + j < this.thread.x) s += x[start + j];
                }
                return s;
              }
        )
        .setConstants({ b, invb: 1 / b })
        .setOutput([m])
        .setPipeline(true);
    const down = levels.map((m, i) => mkDown(m, i === 0));

    /**
     * The gather that stands in for a scatter. Output slot j wants the input
     * index of the (j+1)-th survivor; `incl` is non-decreasing, so that is the
     * first index where incl reaches j+1, found by binary lifting — 22 probes,
     * no branches that change the trip count, no `break` (several gpu.js
     * backends will not compile one).
     */
    const gather = gpu
      .createKernel(function (x, incl) {
        const target = this.thread.x + 1;
        let p = -1;
        let step = this.constants.n;
        for (let s = 0; s < this.constants.steps; s++) {
          step = step * 0.5;
          const q = p + step;
          if (incl[q] < target) p = q;
        }
        // p + 1 is always inside the array, so this read is safe even when the
        // slot is past the last survivor and the answer is the zero pad.
        const v = x[p + 1];
        return target <= incl[this.constants.nm1] ? v : 0;
      })
      .setConstants({ n, steps, nm1: n - 1 })
      .setOutput([n]);

    return {
      async run() {
        const resident = await upload(a);
        let out = null;
        for (let r = 0; r < rounds; r++) {
          // eslint-disable-next-line no-await-in-loop
          const f = await flags(resident, threshold(r));
          const lvl = [f];
          for (let i = 0; i < up.length; i++) {
            // eslint-disable-next-line no-await-in-loop
            lvl.push(await up[i](lvl[i]));
          }
          // eslint-disable-next-line no-await-in-loop
          let scan = await top(lvl[L - 1]);
          for (let i = L - 2; i >= 0; i--) {
            // eslint-disable-next-line no-await-in-loop
            scan = await down[i](lvl[i], scan);
          }
          // eslint-disable-next-line no-await-in-loop
          out = await gather(resident, scan);
        }
        return out.toArray ? await out.toArray() : out;
      },
      backend: () => gather.kernel && gather.kernel.constructor.mode,
      destroy() {
        [upload, flags, top, gather, ...up, ...down].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU. Identical flag-and-scan front end — same block width,
   * same tree, same arithmetic — and then a scatter instead of a search, which
   * is the one thing the runtime above cannot express. Everything for every
   * round goes into one command buffer with one submit.
   */
  async webgpu(device, { n, b, rounds }, { a }) {
    const levels = levelsOf(n, b);
    const L = levels.length;
    const S = GPUBufferUsage.STORAGE;
    const mk = (m, extra = 0) => device.createBuffer({ size: Math.max(16, m * 4), usage: S | extra });

    const bufA = mk(n, GPUBufferUsage.COPY_DST);
    const lvl = levels.map(m => mk(m)); // lvl[0] is the flags
    const scan = levels.map(m => mk(m));
    const out = mk(n, GPUBufferUsage.COPY_SRC);
    const read = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // 256 and not 64: the full-width dispatches are 2^22 threads, and at 64 per
    // workgroup that is 65,536 of them — one over WebGPU's default
    // maxComputeWorkgroupsPerDimension of 65,535, which would fail validation
    // on every machine rather than being a performance question.
    const WG = 256;
    const shader = code => device.createShaderModule({ code });
    const HEAD = 'struct P { m: u32, b: u32, t: f32 };';

    // One module per kernel: `layout: "auto"` builds a bind group layout from
    // the bindings an entry point actually touches, so sharing a module between
    // entry points with different binding sets is a quiet way to get layouts
    // that do not match the bind groups written against them.
    const pFlags = shader(`${HEAD}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.m) { return; }
  if (src[i] >= p.t) { dst[i] = 1.0; } else { dst[i] = 0.0; }
}`);

    const pUp = shader(`${HEAD}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.m) { return; }
  let start = i * p.b;
  var s = 0.0;
  for (var k: u32 = 0u; k < p.b; k = k + 1u) { s = s + src[start + k]; }
  dst[i] = s;
}`);

    const pTop = shader(`${HEAD}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.m) { return; }
  var s = 0.0;
  for (var k: u32 = 0u; k < i; k = k + 1u) { s = s + src[k]; }
  dst[i] = s;
}`);

    // `incl` picks exclusive (k < i) or inclusive (k <= i) with one character.
    const downCode = last => `${HEAD}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> off: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.m) { return; }
  let blk = i / p.b;
  var s = off[blk];
  for (var k: u32 = blk * p.b; k ${last ? '<=' : '<'} i; k = k + 1u) { s = s + src[k]; }
  dst[i] = s;
}`;
    const pDown = shader(downCode(false));
    const pDown0 = shader(downCode(true));

    const pScatter = shader(`${HEAD}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> incl: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.m) { return; }
  let total = u32(incl[p.m - 1u]);
  // Slots past the last survivor are the zero pad. They are never scatter
  // targets — every target is below total — so these writes cannot race the
  // ones underneath.
  if (i >= total) { dst[i] = 0.0; }
  if (flags[i] != 0.0) { dst[u32(incl[i]) - 1u] = src[i]; }
}`);

    const pipe = module => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const [cFlags, cUp, cTop, cDown, cDown0, cScatter] = [pFlags, pUp, pTop, pDown, pDown0, pScatter].map(pipe);

    const unis = [];
    const mkUni = (m, t) => {
      const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, new Uint32Array([m, b, 0, 0]));
      device.queue.writeBuffer(buf, 8, new Float32Array([t]));
      unis.push(buf);
      return buf;
    };
    const bg = (pipeline, buffers) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });

    // The per-round threshold lives in `rounds` uniform buffers built once:
    // queue.writeBuffer cannot be interleaved between dispatches in a command
    // buffer, and WebGPU has no push constants.
    const flagBinds = [];
    for (let r = 0; r < rounds; r++) flagBinds.push(bg(cFlags, [bufA, lvl[0], mkUni(n, threshold(r))]));
    const upSteps = levels.slice(1).map((m, i) => ({
      bind: bg(cUp, [lvl[i], lvl[i + 1], mkUni(m, 0)]),
      groups: Math.ceil(m / WG),
    }));
    const topBind = bg(cTop, [lvl[L - 1], scan[L - 1], mkUni(levels[L - 1], 0)]);
    const downSteps = [];
    for (let i = L - 2; i >= 1; i--) {
      downSteps.push({ bind: bg(cDown, [lvl[i], scan[i + 1], scan[i], mkUni(levels[i], 0)]), groups: Math.ceil(levels[i] / WG) });
    }
    const down0Bind = bg(cDown0, [lvl[0], scan[1], scan[0], mkUni(n, 0)]);
    const scatterBind = bg(cScatter, [bufA, scan[0], lvl[0], out, mkUni(n, 0)]);
    const nGroups = Math.ceil(n / WG);

    return {
      async run() {
        device.queue.writeBuffer(bufA, 0, a);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let r = 0; r < rounds; r++) {
          pass.setPipeline(cFlags);
          pass.setBindGroup(0, flagBinds[r]);
          pass.dispatchWorkgroups(nGroups);
          pass.setPipeline(cUp);
          for (const s of upSteps) {
            pass.setBindGroup(0, s.bind);
            pass.dispatchWorkgroups(s.groups);
          }
          pass.setPipeline(cTop);
          pass.setBindGroup(0, topBind);
          pass.dispatchWorkgroups(Math.ceil(levels[L - 1] / WG));
          pass.setPipeline(cDown);
          for (const s of downSteps) {
            pass.setBindGroup(0, s.bind);
            pass.dispatchWorkgroups(s.groups);
          }
          pass.setPipeline(cDown0);
          pass.setBindGroup(0, down0Bind);
          pass.dispatchWorkgroups(nGroups);
          pass.setPipeline(cScatter);
          pass.setBindGroup(0, scatterBind);
          pass.dispatchWorkgroups(nGroups);
        }
        pass.end();
        enc.copyBufferToBuffer(out, 0, read, 0, n * 4);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return result;
      },
      destroy() {
        [bufA, ...lvl, ...scan, out, read, ...unis].forEach(x => x.destroy && x.destroy());
      },
    };
  },

  // Index-weighted over the whole padded output, which is the only checksum
  // that catches the failure this row is most likely to have: survivors that
  // are all present and all one slot out.
  reduce(out, { n }) {
    let acc = 0;
    for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    return acc / n;
  },
};
