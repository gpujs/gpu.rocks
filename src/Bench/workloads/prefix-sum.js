/**
 * Exclusive prefix sum — a dependency chain that parallelises anyway.
 *
 * out[i] is the sum of everything before i. Read literally that is the most
 * sequential statement in this whole table: element i cannot be known until
 * element i-1 is. In JavaScript it really is one accumulator and one pass, four
 * milliseconds for four million elements, and it is hard to imagine improving.
 *
 * The GPU does not get to run that loop. What it runs instead is the standard
 * block scan: chop the array into blocks of 64, sum each block (up-sweep), scan
 * the 65,536 block sums the same way, then again, until the top is small enough
 * to scan in one kernel — and then walk back down, where every element adds its
 * block's offset to a local scan of its own block. Seven dispatches, a tree
 * four levels deep, and the answer comes out in the same order the sequential
 * loop would have produced it.
 *
 * THE COST OF PARALLELISM, STATED PLAINLY. The sequential scan does n additions
 * — one per element. This one does about 33n. A gpu.js kernel is a gather: a
 * thread produces its own output and cannot write to a neighbour's, so the
 * down-sweep cannot carry a running total sideways and each thread re-adds the
 * elements before it inside its own 64-wide block, 32 of them on average. That
 * is a real 33x more arithmetic to reach the same answer, and the GPU still
 * wins, because 33x more work spread over thousands of lanes is a good trade
 * against 1x on one lane. The row is worth reading precisely because the naive
 * "GPU does the same work, faster" model is wrong here and the GPU is still
 * ahead.
 *
 * WHAT MIGHT MISLEAD. Seven dispatches per round is a lot for four million
 * elements, and on the gpu.js columns the per-call cost of the runtime is a
 * large share of the number — the same effect `launch-overhead` isolates. The
 * bare-WebGPU column records all 896 dispatches into one command buffer, so the
 * gap between it and the gpu.js WebGPU column beside it is mostly launch price,
 * not scan price. Read the two together.
 *
 * WHY ROUNDS, AND WHY THE VALUES ARE SO SMALL. One scan is 4 ms of plain JS, so
 * a run is 128 of them; round r starts its accumulator at r so no two rounds
 * compute the same array, at exactly zero cost per element. And the inputs are
 * scaled to about 3.7e-4 so the running total ends near 1536 rather than near
 * 2^21: at 2 million, an fp32 ulp is 0.125, the blocked GPU tree and the
 * sequential fp64 baseline would disagree in the fourth digit, and the runner
 * would correctly report the GPU columns as WRONG. Keeping the magnitude down
 * is what makes the columns comparable at all.
 */

const N = 1 << 22; // 4,194,304
const B = 64; // block width: 2^22 -> 2^16 -> 2^10 -> 2^4
const ROUNDS = 128;
const SCALE = 1 / 2048;

function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    // Positive and narrow: a monotonic ramp with no cancellation, so the
    // checksum's magnitude is set by the data rather than by luck.
    a[i] = (0.5 + 0.5 * ((s >>> 8) / 0x1000000)) * SCALE;
  }
  return a;
}

// [2^22, 2^16, 2^10, 2^4] — the last entry is the top, small enough for one
// kernel to scan directly.
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
  id: 'prefix-sum',
  name: 'Exclusive prefix sum',
  params: `2^22 fp32, block ${B}, × ${ROUNDS} rounds`,
  tag: 'scan',
  group: 'movement',
  size: { n: N, b: B, rounds: ROUNDS },

  make({ n }) {
    return { a: fill(new Float32Array(n), 0x1b873593) };
  },

  // The oracle. One accumulator, one pass, in fp64 because that is what a
  // JavaScript number is — and it is the exact answer the tree is checked
  // against.
  js({ n, rounds }, { a }) {
    const out = new Float32Array(n);
    for (let r = 0; r < rounds; r++) {
      let acc = r;
      for (let i = 0; i < n; i++) {
        out[i] = acc;
        acc += a[i];
      }
    }
    return out;
  },

  gpujs(gpu, { n, b, rounds }, { a }) {
    const levels = levelsOf(n, b);
    const L = levels.length;

    // One upload per run; every level after that lives on the GPU. Handing the
    // 16 MB array to a kernel argument instead would re-upload it twice per
    // round, 256 times per run, and this row would be measuring a bus.
    const upload = gpu
      .createKernel(function (x) {
        return x[this.thread.x];
      })
      .setOutput([n])
      .setPipeline(true);

    // Up-sweep: one thread per block, summing its 64 elements.
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

    // The top of the tree, scanned in one kernel. `base` is the round's starting
    // accumulator, and folding it in here means the per-round variation costs
    // nothing at any level below.
    const top = gpu
      .createKernel(function (x, base) {
        let s = base;
        for (let k = 0; k < this.constants.m; k++) {
          if (k < this.thread.x) s += x[k];
        }
        return s;
      })
      .setConstants({ m: levels[L - 1] })
      .setOutput([levels[L - 1]])
      .setPipeline(true);

    // Down-sweep: this block's offset, plus a local exclusive scan of the block.
    // The `if` is the 32-adds-per-element the header talks about.
    const down = levels.map((m, i) => {
      const k = gpu
        .createKernel(function (x, off) {
          const blk = Math.floor(this.thread.x * this.constants.invb);
          const start = blk * this.constants.b;
          let s = off[blk];
          for (let j = 0; j < this.constants.b; j++) {
            if (start + j < this.thread.x) s += x[start + j];
          }
          return s;
        })
        .setConstants({ b, invb: 1 / b })
        .setOutput([m]);
      return i === 0 ? k : k.setPipeline(true);
    });

    return {
      async run() {
        const resident = await upload(a);
        let out = null;
        for (let r = 0; r < rounds; r++) {
          const lvl = [resident];
          for (let i = 0; i < up.length; i++) {
            // eslint-disable-next-line no-await-in-loop
            lvl.push(await up[i](lvl[i]));
          }
          // eslint-disable-next-line no-await-in-loop
          let scan = await top(lvl[L - 1], r);
          for (let i = L - 2; i >= 0; i--) {
            // eslint-disable-next-line no-await-in-loop
            scan = await down[i](lvl[i], scan);
          }
          out = scan;
        }
        // down[0] is not a pipeline kernel, so this is already the array: the
        // last dispatch has landed before run() resolves.
        return out;
      },
      backend: () => down[0].kernel && down[0].kernel.constructor.mode,
      destroy() {
        [upload, top, ...up, ...down].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU. Same tree, same block width, same 33n of arithmetic —
   * the only difference from the column to its left is that there is no
   * JavaScript between the dispatches. All 896 of them are recorded into one
   * command buffer and submitted once.
   */
  async webgpu(device, { n, b, rounds }, { a }) {
    const levels = levelsOf(n, b);
    const L = levels.length;
    const S = GPUBufferUsage.STORAGE;
    const mk = (m, extra = 0) => device.createBuffer({ size: Math.max(16, m * 4), usage: S | extra });

    // lvl[0] is the input; lvl[i] is the up-sweep's block sums. scan[i] is the
    // exclusive scan of lvl[i]. scan[0] is the answer.
    const lvl = levels.map((m, i) => mk(m, i === 0 ? GPUBufferUsage.COPY_DST : 0));
    const scan = levels.map((m, i) => mk(m, i === 0 ? GPUBufferUsage.COPY_SRC : 0));
    const read = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // 256 and not 64: the level-0 dispatch is 2^22 threads, and at 64 per
    // workgroup that is 65,536 of them — one over WebGPU's default
    // maxComputeWorkgroupsPerDimension of 65,535, which would fail validation
    // on every machine rather than being a performance question.
    const WG = 256;
    // Three modules rather than three entry points in one: `layout: "auto"`
    // derives a bind group layout from the bindings an entry point actually
    // uses, so entry points with different binding sets in one module produce
    // layouts that quietly disagree with the bind groups written for them.
    const upModule = device.createShaderModule({
      code: `
struct P { m: u32, b: u32, base: f32 };
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
}`,
    });
    const topModule = device.createShaderModule({
      code: `
struct P { m: u32, b: u32, base: f32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.m) { return; }
  var s = p.base;
  for (var k: u32 = 0u; k < i; k = k + 1u) { s = s + src[k]; }
  dst[i] = s;
}`,
    });
    const downModule = device.createShaderModule({
      code: `
struct P { m: u32, b: u32, base: f32 };
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
  for (var k: u32 = blk * p.b; k < i; k = k + 1u) { s = s + src[k]; }
  dst[i] = s;
}`,
    });

    const pipe = module => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const pUp = pipe(upModule);
    const pTop = pipe(topModule);
    const pDown = pipe(downModule);

    const unis = [];
    const mkUni = (m, base) => {
      const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, new Uint32Array([m, b, 0, 0]));
      device.queue.writeBuffer(buf, 8, new Float32Array([base]));
      unis.push(buf);
      return buf;
    };
    const bg = (pipeline, entries) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });

    const upSteps = levels.slice(1).map((m, i) => ({
      bind: bg(pUp, [lvl[i], lvl[i + 1], mkUni(m, 0)]),
      groups: Math.ceil(m / WG),
    }));
    // One uniform buffer per round: writeBuffer cannot be interleaved between
    // dispatches inside a command buffer, so the per-round base is baked in here.
    const topBinds = [];
    for (let r = 0; r < rounds; r++) topBinds.push(bg(pTop, [lvl[L - 1], scan[L - 1], mkUni(levels[L - 1], r)]));
    const topGroups = Math.ceil(levels[L - 1] / WG);
    const downSteps = [];
    for (let i = L - 2; i >= 0; i--) {
      downSteps.push({
        bind: bg(pDown, [lvl[i], scan[i + 1], scan[i], mkUni(levels[i], 0)]),
        groups: Math.ceil(levels[i] / WG),
      });
    }

    return {
      async run() {
        device.queue.writeBuffer(lvl[0], 0, a);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let r = 0; r < rounds; r++) {
          pass.setPipeline(pUp);
          for (const s of upSteps) {
            pass.setBindGroup(0, s.bind);
            pass.dispatchWorkgroups(s.groups);
          }
          pass.setPipeline(pTop);
          pass.setBindGroup(0, topBinds[r]);
          pass.dispatchWorkgroups(topGroups);
          pass.setPipeline(pDown);
          for (const s of downSteps) {
            pass.setBindGroup(0, s.bind);
            pass.dispatchWorkgroups(s.groups);
          }
        }
        pass.end();
        enc.copyBufferToBuffer(scan[0], 0, read, 0, n * 4);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [...lvl, ...scan, read, ...unis].forEach(x => x.destroy && x.destroy());
      },
    };
  },

  // Index-weighted over all 2^22 outputs. A scan is exactly the kind of kernel
  // that can be right for the first block and wrong everywhere after it, so
  // every element has to be in the checksum — and since the ramp is monotonic
  // and positive there is no cancellation to hide an error inside.
  reduce(out, { n }) {
    let acc = 0;
    for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    return acc / n;
  },
};
