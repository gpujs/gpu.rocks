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
 * WHY THE gpu.js COLUMNS ARE A PIPELINE. That comparison only says what it
 * claims to say if the two sides are submitted the same way. The bare column
 * puts all 32 rounds — 192 dispatches — into one command buffer and one submit;
 * the gpu.js column used to issue those same 192 dispatches one call at a time
 * and await each, so part of the gap it showed was scheduling rather than
 * bandwidth. `gpu.createPipeline` traces the orchestration once and executes
 * the whole plan as one launch, which puts the two columns in the same shape
 * and leaves the doubled read as very nearly the only thing between them. What
 * it deliberately does NOT change: the same two trees, the same kernels, the
 * same 256:1 fold, the same 64 MB read twice per round. This row prices moving
 * memory, not dispatching, and fusing does not remove one byte of the traffic —
 * `launch-overhead` is where dispatch cost is the subject, and this is not it.
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
    // useless one of a reduction. Inside a plan the same trap is set by a raw
    // array ARGUMENT: the generic executor hands the argument itself to every
    // step that names it, and 64 of them do. So the upload is the plan's first
    // step and the 64 head steps read its resident output. A fused executor
    // stages an argument once per call and would not have needed that, which
    // costs it one extra 64 MB copy per run — cheap next to the 4 GB the tree
    // reads, and the price of the backends that cannot fuse not re-uploading
    // 64 MB sixty-four times.
    //
    // None of the kernels below say `pipeline: true` any more. Residency is the
    // plan's business — it runs private clones with pipeline and immutable
    // forced on — and the flag on the user's kernel would only describe a
    // direct call that never happens.
    const upload = gpu
      .createKernel(function (x) {
        return x[this.thread.x];
      })
      .setOutput([n]);

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
        .setOutput([levels[0]]);

    // Every level after the first is the same fold, so it is the same code with
    // a different output size.
    const fold = m =>
      gpu
        .createKernel(function (x) {
          let s = 0;
          for (let t = 0; t < this.constants.g; t++) {
            s += x[this.thread.x + t * this.constants.m];
          }
          return s * this.constants.invg;
        })
        .setConstants({ g: group, m, invg })
        .setOutput([m]);

    // Two independent trees — see the header — but only their bottom level is
    // two kernels, because only down there does one square and the other not.
    // Level for level, the folds above are the same code at the same size, and
    // ONE instance serves both trees. What used to force a second copy was that
    // a resident kernel owns its output texture, so the sq tree's fold would
    // overwrite the sum tree's answer before anyone read it; the plan assigns
    // buffers from static liveness instead, which reuses a slot the moment its
    // last reader has run and hands the two final folds separate slots
    // precisely because both of those are still live when the plan ends.
    const sumHead = head(false);
    const sqHead = head(true);
    const folds = levels.slice(1).map(fold);

    const climb = (headKernel, resident, b) => {
      let cur = headKernel(resident, b);
      for (let i = 0; i < folds.length; i++) cur = folds[i](cur);
      return cur;
    };

    // Traced once, at the first call, against opaque handles: the rounds loop
    // unrolls into a static plan — 193 steps at these constants — that every
    // later call executes as one launch, and the warm-up runs the runner does
    // before it starts timing are what keep that trace out of the numbers.
    // Unrolling is why `r * bias` may be an ordinary number here — it is
    // a trace-time fact frozen into its step, which is the same thing the bare
    // column does with its `rounds` prebuilt uniform buffers, and for the same
    // reason: the bias cannot change between dispatches of one submission.
    const solve = gpu.createPipeline(
      function (x) {
        const resident = upload(x);
        let mean = null;
        let meansq = null;
        for (let r = 0; r < this.constants.rounds; r++) {
          const b = r * this.constants.bias;
          mean = climb(sumHead, resident, b);
          meansq = climb(sqHead, resident, b);
        }
        // Handles, not values — nothing in here is allowed to look at a number.
        // The two that survive the loop are the plan's results, and they are
        // read back once, together, when it finishes.
        return [mean, meansq];
      },
      { constants: { rounds, bias }  }
    );

    return {
      async run() {
        // `a` is an argument rather than something the orchestration closes
        // over, because a captured array freezes into the plan and would be
        // uploaded once for the whole benchmark. The bare column writes its
        // input buffer on every run; a column that quietly stopped doing so
        // would be measuring a different thing. The price of that honesty is
        // that the pipeline samples its arguments at the call, so the 64 MB is
        // copied host-side once per run on top of the upload — the one cost
        // this shape adds that the bare column does not pay.
        const [mean, meansq] = await solve(a);
        // Results, not handles: the plan has finished and read them back by the
        // time this resolves.
        return new Float32Array([mean[0], Math.sqrt(meansq[0])]);
      },
      // Ask the plan's own kernel, not the one created up there. gpu.js answers
      // a kernel it cannot compile by swapping in a CPU one, and it is the
      // plan's private clones that get built and so the clones that would be
      // swapped — the user-facing objects here are never run directly and would
      // still be reporting the backend they were asked for.
      // The pipeline's OWN backend, not a kernel's. Under a plan the user's
      // kernel shortcut is not what executes, so asking it reports the mode we
      // requested no matter what ran — which silently disables this suite's
      // guard against gpu.js degrading to CPU. Reading plan internals was no
      // better: an accessor built on plan.kernels[0].clone went stale one
      // commit later without erroring. `pipeline.backend` is supported API and
      // derives from the executor that actually ran (gpujs/gpu.js#871).
      backend() {
        return solve.backend;
      },
      // which lowering actually ran, so a cell that could not reach the
      // fused or threaded path says so instead of being read as one that did
      executor: () => solve.executorKind,
      destroy() {
        // The pipeline first: it owns the clones and their buffers, and its
        // release queues behind any call still in flight.
        if (solve.destroy) solve.destroy();
        [upload, sumHead, sqHead, ...folds].forEach(k => k.destroy && k.destroy());
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
