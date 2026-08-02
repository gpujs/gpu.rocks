/**
 * LSD radix sort — the row where gpu.js runs out of primitives.
 *
 * The companion to the bitonic row: a bitonic network knows every comparison
 * before it starts, and a radix pass does not. A radix pass counts the keys in each of 16 digit
 * buckets, prefix-sums those counts into starting offsets, and then moves every
 * key to an address it computes from its own value. Count, scan, SCATTER —
 * eight times over, four bits at a time.
 *
 * The scatter is the whole story. A gpu.js kernel thread returns one number and
 * that number lands in the thread's own output cell; there is no way to say
 * "write this value over there". So a radix pass cannot be written in gpu.js —
 * not slowly, not at all — and this row reports four N/A cells rather than
 * pretending otherwise.
 *
 * WHAT THE ALTERNATIVE WOULD HAVE BEEN, and why it is not on this page. A
 * scatter can be turned inside out into a gather: for each OUTPUT slot p, work
 * out which input key belongs there. That needs, per pass, a per-tile count of
 * every digit, a prefix sum over the whole 16 × tiles table, and then a binary
 * search plus a linear scan per output element to find the k-th key with a
 * given digit. It is a different algorithm with roughly fifty times the memory
 * traffic, and putting its number in the gpu.js column would say "gpu.js can do
 * radix sort, slowly" when the truth is "gpu.js cannot express this pass".
 * Rule 6 of the contract exists for exactly this case.
 *
 * Note which cells go N/A: all four, including gpu.js on the CPU. That is the
 * tell. The obstacle is not the hardware — a CPU can obviously scatter — it is
 * the kernel model, one output cell per thread, and that model is the same on
 * every backend gpu.js compiles to.
 *
 * The hand-written WebGPU column does the real thing, and shows what the
 * missing primitive is worth against the same 2^22 keys the bitonic row sorts.
 */

// 2^24, not 2^22: at four million keys the plain-JS baseline was 74 ms, which
// is below the band where the clock and the harness stop mattering. Sixteen
// million keys is still a realistic sort and lands mid-band.
const LOG = 24;
const N = 1 << LOG;
const RADIX_BITS = 4;
const DIGITS = 1 << RADIX_BITS; // 16
const PASSES = 32 / RADIX_BITS; // 8, for a full 32-bit key
const TILE = 256; // keys per thread in the WebGPU column
const TILES = N / TILE;

// splitmix32, not the LCG the other rows use. An LCG's low bits are almost
// periodic, and a radix sort starts at the low bits: pass 1 would have been
// handed a distribution no real key set has.
function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    a[i] = (z ^ (z >>> 15)) >>> 0;
  }
  return a;
}

export default {
  id: 'radix-sort',
  name: 'Radix sort',
  params: `2^${LOG} keys · ${RADIX_BITS}-bit digits · ${PASSES} passes, u32`,
  tag: 'scan + scatter',
  group: 'movement',
  size: { n: N, digits: DIGITS, passes: PASSES, bits: RADIX_BITS, tile: TILE },

  // Every gpu.js backend, and the reason is the same on all of them.
  declines: ['webgpu', 'webgl2', 'webgl', 'cpu'],
  declinesReason:
    'a radix pass ends in a scatter — each key writes to an address it computes — and a gpu.js kernel thread can only write its own output cell. Rewriting the scatter as a per-output search is a different algorithm with ~50× the memory traffic, so this cell is N/A rather than misleading.',

  make({ n }) {
    return { keys: fill(new Uint32Array(n), 0x6c078965) };
  },

  /**
   * The oracle: textbook LSD radix, counting once per pass. Two touches of the
   * data per pass — one to count, one to move — with the sixteen bucket cursors
   * living in L1 the whole time. This is the fast way to write it in JS.
   *
   * It counts inside each pass rather than computing all eight digit histograms
   * in a single sweep, which is a real optimisation worth about 30% and which
   * the WebGPU column does not do either. Both columns therefore run the same
   * eight count-scan-scatter passes, which is the point of a row.
   */
  js({ n, digits, passes, bits }, { keys }) {
    let src = keys.slice();
    let dst = new Uint32Array(n);
    const cursor = new Uint32Array(digits);
    const mask = digits - 1;
    for (let p = 0; p < passes; p++) {
      const shift = p * bits;
      cursor.fill(0);
      for (let i = 0; i < n; i++) cursor[(src[i] >>> shift) & mask]++;
      let acc = 0;
      for (let d = 0; d < digits; d++) {
        const c = cursor[d];
        cursor[d] = acc;
        acc += c;
      }
      for (let i = 0; i < n; i++) dst[cursor[(src[i] >>> shift) & mask]++] = src[i];
      const swap = src;
      src = dst;
      dst = swap;
    }
    return src;
  },

  // Never called: `declines` covers every gpu.js mode. It exists because the
  // registry requires the key, and it throws rather than returning something
  // that could quietly be timed.
  gpujs() {
    throw new Error('radix-sort declines every gpu.js backend; see declinesReason');
  },

  /**
   * Hand-written WebGPU. Three dispatches per pass, eight passes, ping-ponging
   * between two key buffers, and only the last result read back.
   *
   *   count    one invocation per 256-key tile, sixteen counters in registers,
   *            written out digit-major as counts[digit * tiles + tile].
   *   scan     one exclusive prefix sum over that whole 16 × tiles table. Digit-
   *            major is what makes the sort stable and the scan a single sweep:
   *            reading it in order visits every digit-0 group in tile order,
   *            then every digit-1 group, which is exactly the output order.
   *   scatter  the same invocation walks its tile again and writes each key to
   *            its cursor, incrementing as it goes. Sequential within a tile,
   *            so the sort stays stable — and LSD radix is wrong, not merely
   *            unordered, if a pass loses stability.
   *
   * One invocation per tile rather than a cooperative workgroup split is a
   * deliberate simplification: it mirrors the JS loop line for line and is easy
   * to check by eye. A production radix sort would rank keys with a workgroup
   * ballot and read coalesced; this one trades some bandwidth for a kernel a
   * reader can verify.
   */
  async webgpu(device, { n, digits, passes, bits, tile }, { keys }) {
    const tiles = n / tile;
    const table = digits * tiles;
    const bytes = n * 4;
    const WG = 64;

    const mkBuf = (size, usage, data) => {
      const b = device.createBuffer({ size, usage, mappedAtCreation: Boolean(data) });
      if (data) {
        new Uint32Array(b.getMappedRange()).set(data);
        b.unmap();
      }
      return b;
    };
    const S = GPUBufferUsage.STORAGE;
    const pristine = mkBuf(bytes, GPUBufferUsage.COPY_SRC, keys);
    const keyA = mkBuf(bytes, S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const keyB = mkBuf(bytes, S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const counts = mkBuf(table * 4, S);
    const offsets = mkBuf(table * 4, S);
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // One uniform slot per pass, 256-byte aligned, all written once at build
    // time: a queue.writeBuffer between dispatches is ordered against the whole
    // submit, not between them, and every pass would see the last pass's shift.
    const STRIDE = 256;
    const params = mkBuf(passes * STRIDE, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const paramData = new Uint32Array((passes * STRIDE) / 4);
    for (let p = 0; p < passes; p++) paramData[(p * STRIDE) / 4] = p * bits;
    device.queue.writeBuffer(params, 0, paramData);

    const common = `
struct P { shift: u32 };
const DIGITS: u32 = ${digits}u;
const MASK: u32 = ${digits - 1}u;
const TILE: u32 = ${tile}u;
const TILES: u32 = ${tiles}u;
const TABLE: u32 = ${table}u;`;

    // Three modules rather than three entry points in one, so that each
    // pipeline's automatic bind group layout is unambiguous.
    const countMod = device.createShaderModule({
      code: `${common}
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= TILES) { return; }
  var local: array<u32, ${digits}>;
  for (var d: u32 = 0u; d < DIGITS; d = d + 1u) { local[d] = 0u; }
  let base = t * TILE;
  for (var i: u32 = 0u; i < TILE; i = i + 1u) {
    let d = (src[base + i] >> p.shift) & MASK;
    local[d] = local[d] + 1u;
  }
  for (var d: u32 = 0u; d < DIGITS; d = d + 1u) { counts[d * TILES + t] = local[d]; }
}`,
    });

    // A single workgroup scans the whole table: 256 threads take a contiguous
    // chunk each, one barrier, then each adds the chunks before it. The table
    // is 16 × tiles entries — small enough that one workgroup is faster than
    // the launch overhead of a proper multi-level scan.
    const scanMod = device.createShaderModule({
      code: `${common}
@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;

var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let chunk = TABLE / 256u;
  let start = lid.x * chunk;
  var sum: u32 = 0u;
  for (var i: u32 = 0u; i < chunk; i = i + 1u) { sum = sum + counts[start + i]; }
  partial[lid.x] = sum;
  workgroupBarrier();
  var acc: u32 = 0u;
  for (var i: u32 = 0u; i < lid.x; i = i + 1u) { acc = acc + partial[i]; }
  for (var i: u32 = 0u; i < chunk; i = i + 1u) {
    let c = counts[start + i];
    offsets[start + i] = acc;
    acc = acc + c;
  }
}`,
    });

    const scatterMod = device.createShaderModule({
      code: `${common}
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= TILES) { return; }
  var cursor: array<u32, ${digits}>;
  for (var d: u32 = 0u; d < DIGITS; d = d + 1u) { cursor[d] = offsets[d * TILES + t]; }
  let base = t * TILE;
  for (var i: u32 = 0u; i < TILE; i = i + 1u) {
    let k = src[base + i];
    let d = (k >> p.shift) & MASK;
    dst[cursor[d]] = k;
    cursor[d] = cursor[d] + 1u;
  }
}`,
    });

    const mkPipe = module => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const countPipe = mkPipe(countMod);
    const scanPipe = mkPipe(scanMod);
    const scatterPipe = mkPipe(scatterMod);

    const scanBind = device.createBindGroup({
      layout: scanPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: counts } },
        { binding: 1, resource: { buffer: offsets } },
      ],
    });

    // Pass p reads keyA and writes keyB when p is even, and the other way when
    // it is odd. PASSES is even, so the answer ends up back in keyA.
    const countBinds = [];
    const scatterBinds = [];
    for (let p = 0; p < passes; p++) {
      const from = p % 2 === 0 ? keyA : keyB;
      const to = p % 2 === 0 ? keyB : keyA;
      const slot = { buffer: params, offset: p * STRIDE, size: 16 };
      countBinds.push(
        device.createBindGroup({
          layout: countPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: from } },
            { binding: 1, resource: { buffer: counts } },
            { binding: 2, resource: slot },
          ],
        })
      );
      scatterBinds.push(
        device.createBindGroup({
          layout: scatterPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: from } },
            { binding: 1, resource: { buffer: to } },
            { binding: 2, resource: { buffer: offsets } },
            { binding: 3, resource: slot },
          ],
        })
      );
    }
    const tileGroups = Math.ceil(tiles / WG);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        // Sorting is destructive, so every run starts from the pristine keys —
        // the same copy the JS column makes.
        enc.copyBufferToBuffer(pristine, 0, keyA, 0, bytes);
        const pass = enc.beginComputePass();
        for (let p = 0; p < passes; p++) {
          pass.setPipeline(countPipe);
          pass.setBindGroup(0, countBinds[p]);
          pass.dispatchWorkgroups(tileGroups);
          pass.setPipeline(scanPipe);
          pass.setBindGroup(0, scanBind);
          pass.dispatchWorkgroups(1);
          pass.setPipeline(scatterPipe);
          pass.setBindGroup(0, scatterBinds[p]);
          pass.dispatchWorkgroups(tileGroups);
        }
        pass.end();
        enc.copyBufferToBuffer(keyA, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        // Nothing before this line proves a single pass ran.
        await read.mapAsync(GPUMapMode.READ);
        const out = new Uint32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [pristine, keyA, keyB, counts, offsets, read, params].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Sortedness plus a rank-weighted sum, the same pair the bitonic row uses.
   * Keys are u32 and are only ever moved, never computed on, so every correct
   * column agrees exactly; the tolerance is there for the other rows.
   *
   * Keys are scaled to [0, 1) before summing so the total stays in a range
   * where the sum of four million of them is still meaningful — and a single
   * inverted pair adds a whole 1 to a number of order 0.75.
   */
  reduce(out, { n }) {
    const a = out;
    let acc = 0;
    let inversions = 0;
    for (let i = 0; i < a.length; i++) {
      acc += (a[i] / 4294967296) * (1 + i / n);
      if (i > 0 && a[i] < a[i - 1]) inversions++;
    }
    return acc / n + inversions;
  },
};
