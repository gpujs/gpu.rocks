/**
 * Full reduction — 16.7 million floats down to two numbers.
 *
 * A mean and a root-mean-square over 2^24 fp32 values. In JavaScript that is
 * one loop and two accumulators. On a GPU there is no such thing as "one
 * accumulator": every lane would have to fight for it, so the reduction becomes
 * a tree — 2^24 values folded 256:1 into 2^16, again into 2^8, again into 1,
 * three dispatches deep. Same answer, three times the passes over memory, and a
 * shape that only makes sense because the machine is wide.
 *
 * WHY ROUNDS. One reduction of 2^24 floats is about 16 ms of plain JavaScript —
 * it is a single streaming pass over 64 MB and there is nothing in it to be
 * slow at. That is well under the 200 ms floor the sizing script enforces, so
 * one run is 32 reductions rather than one. Every column does the same 32, so
 * no ratio moves; what changes is that the clock has something to resolve.
 * Round r reduces the data biased by r/4096 so no two rounds are the same
 * computation, which keeps the row honest about actually doing 32 of them.
 *
 * WHY THE gpu.js COLUMNS READ THE INPUT TWICE AND THE BARE COLUMN DOES NOT.
 * This is the most interesting thing on the row, so it is worth being blunt
 * about. A gpu.js kernel returns ONE value per thread. The mean needs a sum and
 * the RMS needs a sum of squares, and one thread cannot emit both, so the
 * gpu.js columns run two independent reduction trees and touch all 64 MB twice
 * per round. Hand-written WGSL has storage buffers and can write two floats
 * from one thread, so the bare column fuses them and touches the data once.
 * On a reduction — which is bandwidth-bound and nothing else — that is close to
 * a factor of two, and it is bought entirely by the shape of the API rather
 * than by the hardware. Separating that from "the GPU is fast" is the whole
 * reason the bare column exists.
 *
 * PRECISION. Summing 16.7 million fp32 values in fp32 order is a classic way to
 * lose four digits, so every level here averages rather than sums: each thread
 * divides by its own group size, and because the groups are equal the mean of
 * the means is exactly the mean. Magnitudes stay near 0.5 at every level of the
 * tree instead of climbing to 8 x 10^6, and the tree agrees with the sequential
 * fp64 baseline to about 1e-6 — comfortably inside the runner's 1e-4.
 */

const N = 1 << 24; // 16,777,216
const ROUNDS = 32;
const GROUP = 256; // fold factor per level: 2^24 -> 2^16 -> 2^8 -> 1
const BIAS = 1 / 4096;

function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s >>> 8) / 0x1000000;
  }
  return a;
}

// 2^24 -> [2^16, 2^8, 1]. Derived rather than written down so a smaller `n`
// still produces a valid tree.
function levelsOf(n, group) {
  const out = [];
  let m = n;
  while (m > 1) {
    m /= group;
    out.push(m);
  }
  return out;
}

export default {
  id: 'reduction',
  name: 'Full reduction to mean + RMS',
  params: `2^24 fp32 → 2 values, ${GROUP}:1 tree, × ${ROUNDS} rounds`,
  tag: 'reduction tree',
  group: 'movement',
  size: { n: N, rounds: ROUNDS, group: GROUP, bias: BIAS },

  make({ n }) {
    return { a: fill(new Float32Array(n), 0x2545f491) };
  },

  // The oracle: one pass, two accumulators, both in fp64 because that is what a
  // JavaScript number is. Nothing here is handicapped for the GPU's benefit —
  // this is exactly how you would write it if the GPU did not exist.
  js({ n, rounds, bias }, { a }) {
    let mean = 0;
    let meansq = 0;
    for (let r = 0; r < rounds; r++) {
      const b = r * bias;
      let s = 0;
      let q = 0;
      for (let i = 0; i < n; i++) {
        const v = a[i] + b;
        s += v;
        q += v * v;
      }
      mean = s / n;
      meansq = q / n;
    }
    return new Float32Array([mean, Math.sqrt(meansq)]);
  },

  gpujs(gpu, { n, rounds, group, bias }, { a }) {
    const levels = levelsOf(n, group);
    const invg = 1 / group;

    // One upload per run, then everything stays on the GPU. Without this the
    // level-1 kernels would take the raw 64 MB array as an argument and gpu.js
    // would re-upload it on every call — twice per round, 64 times per run.
    // That would be a perfectly good measurement of an upload path, and a
    // useless one of a reduction.
    const upload = gpu
      .createKernel(function (x) {
        return x[this.thread.x];
      })
      .setOutput([n])
      .setPipeline(true);

    // Level 1 reads the raw values; `square` decides which of the two trees this
    // is. Thread j takes src[j], src[j + m], src[j + 2m], ... so neighbouring
    // threads read neighbouring addresses — the strided form is the coalesced
    // one, and a contiguous chunk per thread would be the slow way round.
    const head = square =>
      gpu
        .createKernel(
          square
            ? function (x, b) {
                let s = 0;
                for (let t = 0; t < this.constants.g; t++) {
                  const v = x[this.thread.x + t * this.constants.m] + b;
                  s += v * v;
                }
                return s * this.constants.invg;
              }
            : function (x, b) {
                let s = 0;
                for (let t = 0; t < this.constants.g; t++) {
                  s += x[this.thread.x + t * this.constants.m] + b;
                }
                return s * this.constants.invg;
              }
        )
        .setConstants({ g: group, m: levels[0], invg })
        .setOutput([levels[0]])
        .setPipeline(true);

    // Every level after the first is the same fold, so it is the same code with
    // a different output size.
    const fold = (m, last) => {
      const k = gpu
        .createKernel(function (x) {
          let s = 0;
          for (let t = 0; t < this.constants.g; t++) {
            s += x[this.thread.x + t * this.constants.m];
          }
          return s * this.constants.invg;
        })
        .setConstants({ g: group, m, invg })
        .setOutput([m]);
      return last ? k : k.setPipeline(true);
    };

    // Two independent trees — see the header. They cannot share a kernel object
    // because a pipeline kernel owns one output texture, and the second call
    // would overwrite the first tree's result before it was read.
    const build = square => [head(square), ...levels.slice(1).map((m, i) => fold(m, i === levels.length - 2))];
    const sumTree = build(false);
    const sqTree = build(true);

    const runTree = async (tree, resident, b) => {
      let cur = await tree[0](resident, b);
      for (let i = 1; i < tree.length; i++) {
        // eslint-disable-next-line no-await-in-loop
        cur = await tree[i](cur);
      }
      return cur;
    };

    return {
      async run() {
        const resident = await upload(a);
        let mean = null;
        let meansq = null;
        for (let r = 0; r < rounds; r++) {
          const b = r * bias;
          // eslint-disable-next-line no-await-in-loop
          mean = await runTree(sumTree, resident, b);
          // eslint-disable-next-line no-await-in-loop
          meansq = await runTree(sqTree, resident, b);
        }
        // The last level is not a pipeline kernel, so these are already arrays:
        // the work is finished by the time this returns.
        return new Float32Array([mean[0], Math.sqrt(meansq[0])]);
      },
      backend: () => sumTree[0].kernel && sumTree[0].kernel.constructor.mode,
      destroy() {
        [upload, ...sumTree, ...sqTree].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU. Same 256:1 tree, same strided access, but `first`
   * writes a sum AND a sum of squares from one thread into an interleaved pair,
   * so the 64 MB is read once per round instead of twice. Every dispatch for
   * every round goes into one command buffer and one submit, and the only
   * synchronisation with the host is the eight-byte read-back at the end.
   */
  async webgpu(device, { n, rounds, group, bias }, { a }) {
    const levels = levelsOf(n, group);
    const S = GPUBufferUsage.STORAGE;
    const bufIn = device.createBuffer({ size: n * 4, usage: S | GPUBufferUsage.COPY_DST });
    // Each level holds (mean, mean-of-squares) pairs; 16 bytes minimum keeps the
    // one-element top level a legal binding.
    const bufs = levels.map(m =>
      device.createBuffer({ size: Math.max(16, m * 2 * 4), usage: S | GPUBufferUsage.COPY_SRC })
    );
    const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const WG = 64;
    const module = device.createShaderModule({
      code: `
struct P { m: u32, k: u32, bias: f32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

// Level 1: raw scalars in, (mean, mean of squares) pairs out. One pass.
@compute @workgroup_size(${WG})
fn first(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= p.m) { return; }
  var s = 0.0;
  var q = 0.0;
  for (var t: u32 = 0u; t < p.k; t = t + 1u) {
    let v = src[j + t * p.m] + p.bias;
    s = s + v;
    q = q + v * v;
  }
  let inv = 1.0 / f32(p.k);
  dst[j * 2u] = s * inv;
  dst[j * 2u + 1u] = q * inv;
}

// Every level after that: pairs in, pairs out.
@compute @workgroup_size(${WG})
fn fold(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= p.m) { return; }
  var s = 0.0;
  var q = 0.0;
  for (var t: u32 = 0u; t < p.k; t = t + 1u) {
    let i = (j + t * p.m) * 2u;
    s = s + src[i];
    q = q + src[i + 1u];
  }
  let inv = 1.0 / f32(p.k);
  dst[j * 2u] = s * inv;
  dst[j * 2u + 1u] = q * inv;
}`,
    });

    const mkUni = (m, k, b) => {
      const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, new Uint32Array([m, k, 0, 0]));
      device.queue.writeBuffer(buf, 8, new Float32Array([b]));
      return buf;
    };
    const pipeFirst = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'first' } });
    const pipeFold = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'fold' } });
    const bind = (pipeline, src, dst, uni) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: src } },
          { binding: 1, resource: { buffer: dst } },
          { binding: 2, resource: { buffer: uni } },
        ],
      });

    // WebGPU has no push constants and queue.writeBuffer cannot be interleaved
    // between dispatches inside one command buffer, so the per-round bias lives
    // in `rounds` tiny uniform buffers built once, here, at zero per-run cost.
    const unis = [];
    const firstBinds = [];
    for (let r = 0; r < rounds; r++) {
      const u = mkUni(levels[0], group, r * bias);
      unis.push(u);
      firstBinds.push(bind(pipeFirst, bufIn, bufs[0], u));
    }
    const foldBinds = [];
    for (let i = 1; i < levels.length; i++) {
      const u = mkUni(levels[i], group, 0);
      unis.push(u);
      foldBinds.push({ bind: bind(pipeFold, bufs[i - 1], bufs[i], u), groups: Math.ceil(levels[i] / WG) });
    }
    const firstGroups = Math.ceil(levels[0] / WG);
    const top = bufs[levels.length - 1];

    return {
      async run() {
        device.queue.writeBuffer(bufIn, 0, a);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let r = 0; r < rounds; r++) {
          pass.setPipeline(pipeFirst);
          pass.setBindGroup(0, firstBinds[r]);
          pass.dispatchWorkgroups(firstGroups);
          pass.setPipeline(pipeFold);
          for (const f of foldBinds) {
            pass.setBindGroup(0, f.bind);
            pass.dispatchWorkgroups(f.groups);
          }
        }
        pass.end();
        enc.copyBufferToBuffer(top, 0, read, 0, 8);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const pair = new Float32Array(read.getMappedRange(), 0, 2).slice();
        read.unmap();
        return new Float32Array([pair[0], Math.sqrt(pair[1])]);
      },
      destroy() {
        [bufIn, ...bufs, read, ...unis].forEach(x => x.destroy && x.destroy());
      },
    };
  },

  // The output is two numbers, so "touch every element" is easy — but note that
  // both of them are functions of all 2^24 inputs, which is a stronger check
  // than any index weight could be: a backend that reduced half the array gets
  // a visibly different mean. The weight is still there so that a backend which
  // produced the right mean and no RMS cannot pass.
  reduce(out) {
    let acc = 0;
    for (let i = 0; i < out.length; i++) acc += out[i] * (1 + i);
    return acc;
  },
};
