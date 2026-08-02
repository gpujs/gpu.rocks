/**
 * Top-k selection — the 512 largest of 8.4 million, without sorting 8.4 million.
 *
 * Selection is the operation a CPU is quietly brilliant at. One pass, one
 * comparison per element, and a 512-slot min-heap that is disturbed 4,958 times
 * in 8.4 million iterations — 0.06% — because after the first few thousand
 * elements the running k-th largest is already so high that almost nothing beats
 * it. The branch is predicted, the heap stays in L1, and the whole thing costs
 * barely more than reading the array once. That early-out is not a trick; it is
 * the natural shape of the problem on a machine that has one cursor and a memory
 * of what it has already seen.
 *
 * A GPU has neither. There is no shared "current k-th largest" that thousands of
 * lanes can consult, and no early-out worth having when every lane in a
 * workgroup executes the same instruction anyway. So a GPU cannot select — it
 * can only COUNT, and then work out where the count crosses k. That is what this
 * row prices.
 *
 * ── THE TWO ALGORITHMS, AND WHY THEY DIFFER ────────────────────────────────
 *
 * This is one of the rows where the columns do not run the same algorithm, and
 * like `compaction` and `reduction` it says so up front rather than hiding it.
 * Every column computes the SAME defined answer — the k largest values of the
 * array, as a multiset, in descending order — and each uses the formulation its
 * execution model actually admits:
 *
 *   plain JS   one streaming pass, a k-slot min-heap, then sort the 512.
 *   GPU        binary-search the VALUE above which exactly k elements lie, by
 *              counting; then gather those k; then sort the 512.
 *
 * The GPU form is textbook distribution-free selection: hold a bracket [lo, hi)
 * with count(x >= lo) >= k > count(x >= hi), halve it 22 times, and it closes on
 * the exact k-th largest value. Twenty-two counting passes to learn one number
 * that the CPU learned for free as a side effect of walking the array once. That
 * is the finding, and it is a real one — selection is the array primitive where
 * a sequential machine's memory of its own past is worth more than three
 * thousand lanes.
 *
 * It is emphatically NOT a sort, and the difference is the whole point. Sorting
 * these keys with the bitonic network on the `bitonic-sort` row would be 276
 * passes; `Float32Array.prototype.sort` on this input is 868 ms against the heap
 * select's 11.9 ms, 73x. What gets sorted here is 512 survivors, by a rank
 * matrix of 262,144 reads — one thirty-second of a single pass over the input.
 * Read this row beside `bitonic-sort`: that ratio is the entire practical
 * argument for selecting instead of sorting.
 *
 * ── TILE MAXIMA, WHICH ARE WHAT MAKE 22 PASSES AFFORDABLE ──────────────────
 *
 * A naive bisection would read all 32 MB twenty-two times over. Instead one pass
 * reduces the array to 131,072 tile maxima (64 elements each), and the counting
 * kernel checks its tile's maximum first: a tile that cannot contain anything at
 * or above the current threshold contributes zero without a single element being
 * read. The threshold climbs monotonically towards the top 0.006% of the
 * distribution, so after the first handful of halvings nearly every tile is dead
 * and a pass costs 131,072 reads rather than 8,388,608. Measured over the whole
 * bisection the traffic is 7.8 passes' worth, not 24 — still several times what
 * the CPU spends, which is the point, but an honest implementation rather than a
 * strawman built to lose.
 *
 * The prune is a branch, and a branch that a whole workgroup takes together is
 * free. This is one of the few places in the table where control flow works in
 * the GPU's favour, and it only does so because a tile is a contiguous run of
 * threads.
 *
 * ── THE GATHER, AND WHY IT IS NOT A COMPACTION ─────────────────────────────
 *
 * Once the threshold is known the survivors are scattered one in 16,384 through
 * the array, and pulling them into a dense 512-slot buffer is a compaction — the
 * operation `compaction` exists to show gpu.js cannot express. Here it is small
 * enough to dodge the problem entirely: each of the 512 output threads walks the
 * 512 block sums to find its block, the 256 tile counts inside that block to
 * find its tile, then that tile's 64 elements. 832 reads per thread, 512
 * threads, no prefix sum and no scatter. Both GPU columns do exactly this, so
 * what the bare column measures on this row is dispatch cost and nothing else.
 *
 * ── EVERY NUMBER IN THIS ROW IS EXACT IN fp32 ──────────────────────────────
 *
 * Selection is made of comparisons, and a comparison that goes one way in fp64
 * and the other in fp32 swaps an element into or out of the answer. So nothing
 * here is allowed to round. Values are integers over 2^22; the per-round bias is
 * 256/2^22; every bracket endpoint the bisection forms is an integer over 2^22
 * and the one half-grid threshold at the end is an integer over 2^23 — all under
 * fp32's exact-integer ceiling of 2^24, all identical in fp64. The counts are
 * integers below 2^23, also exact. Every backend on this row agrees with the
 * oracle BIT FOR BIT; the runner's 1e-4 tolerance is slack this row never
 * spends, and a checksum that is off at all means something is wrong.
 *
 * ── WHY ROUNDS ─────────────────────────────────────────────────────────────
 *
 * One selection over 2^23 floats is 11.9 ms of plain JavaScript, well under the
 * 200 ms floor the sizing script enforces, so one run is 24 selections. Every
 * column does the same 24, so no ratio moves. Round r biases the data by r/16384
 * — the same device `reduction` uses — so no two rounds are the same
 * computation, and a column that ran round 0 and stopped is off by 0.0014 in
 * every one of its 512 outputs, which is fourteen times the runner's tolerance.
 */

const LOG_N = 23;
const N = 1 << LOG_N; // 8,388,608
const K = 512;

// Tiles for the prune, and blocks of tiles for the two-level walk the gather
// does. 64 * 131072 = N and 256 * 512 = T, so nothing here needs a remainder.
const G = 64; // elements per tile
const T = N / G; // 131,072 tiles
const BLK = 256; // tiles per block
const NB = T / BLK; // 512 blocks

// The value grid. Every input is an integer multiple of 1/QUANT, which is what
// lets the bisection terminate exactly rather than approximately. 2^22 and not
// 2^23: the very last threshold the bisection forms sits half a grid step above
// t*, i.e. an integer over 2^23, and 2^23 + a little is still under fp32's exact
// integer ceiling of 2^24. A finer grid would put that one number outside it.
const QUANT = 1 << 22;

// Halvings of a width-1 bracket needed to close on the grid: 2^-22.
const STEPS = 22;

// Per-round bias, an exact multiple of 1/QUANT (256/2^22).
const BIAS = 1 / 16384;
const ROUNDS = 24;

// Deterministic and cheap: two columns must be handed the same bytes. 22 bits
// of the LCG state over 2^22, so every value is exactly a grid point and about
// one element lands on each — enough ties to be realistic, few enough that the
// 512 answers are mostly distinct.
function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s >>> 10) / QUANT;
  }
  return a;
}

// Standard binary min-heap sift-down. The heap holds the k largest seen so far
// with the SMALLEST of them at the root, so one comparison against the root
// decides whether an element is worth any work at all.
function sift(h, i, k) {
  const v = h[i];
  for (;;) {
    let c = 2 * i + 1;
    if (c >= k) break;
    if (c + 1 < k && h[c + 1] < h[c]) c++;
    if (h[c] >= v) break;
    h[i] = h[c];
    i = c;
  }
  h[i] = v;
}

export default {
  id: 'topk',
  name: 'Top-k selection',
  params: `2^${LOG_N} fp32 → top ${K}, ${STEPS}-step select, × ${ROUNDS} rounds`,
  tag: 'partial order',
  group: 'movement',
  size: { n: N, k: K, g: G, t: T, blk: BLK, nb: NB, steps: STEPS, quant: QUANT, bias: BIAS, rounds: ROUNDS },

  make({ n }) {
    return { a: fill(new Float32Array(n), 0x2f6f2b79) };
  },

  /**
   * The oracle, and a genuinely fast baseline — this is what a JavaScript
   * programmer writes and it is very hard to beat. One pass, a k-slot min-heap,
   * and a single comparison against the root for the 99.9% of elements that
   * cannot possibly qualify. Sorting the whole array with the built-in sort and
   * slicing 512 off the end is 73x slower on this input, so the heap is not a
   * handicap chosen to flatter the GPU; it is the right answer.
   *
   * The final sort is of 512 values, not 2^22 — the same k-sized sort the GPU
   * columns pay for at the end of their pipeline.
   */
  js({ n, k, rounds, bias }, { a }) {
    const heap = new Float32Array(k);
    for (let r = 0; r < rounds; r++) {
      const b = r * bias;
      for (let i = 0; i < k; i++) heap[i] = a[i] + b;
      for (let i = (k >> 1) - 1; i >= 0; i--) sift(heap, i, k);
      let top = heap[0];
      for (let i = k; i < n; i++) {
        const v = a[i] + b;
        if (v > top) {
          heap[0] = v;
          sift(heap, 0, k);
          top = heap[0];
        }
      }
    }
    const out = heap.slice().sort();
    out.reverse();
    return out;
  },

  gpujs(gpu, { n, k, g, t, blk, nb, steps, bias, rounds }, { a }) {
    // One upload per run, then the 16 MB never leaves the GPU. Without this the
    // tile-maximum kernel would take the raw array as an argument and gpu.js
    // would re-upload it 32 times per run — a fine measurement of an upload
    // path and a useless one of a selection.
    const upload = gpu
      .createKernel(function (x) {
        return x[this.thread.x];
      })
      .setOutput([n])
      .setPipeline(true);

    // Tiles are STRIDE classes — tile j owns j, j+T, j+2T, ... — so
    // neighbouring threads read neighbouring addresses. A tile of 64 contiguous
    // elements would be the same arithmetic with every read uncoalesced. The
    // gather below walks the same layout, which is the only thing that has to
    // be true about it.
    const tileMax = gpu
      .createKernel(function (x, b) {
        let m = x[this.thread.x] + b;
        for (let e = 1; e < this.constants.g; e++) {
          const v = x[this.thread.x + e * this.constants.t] + b;
          if (v > m) m = v;
        }
        return m;
      })
      .setConstants({ g, t })
      .setOutput([t])
      .setPipeline(true);

    // The counting pass, and the prune that makes twenty-two of them tolerable:
    // a tile whose maximum is below the threshold contributes nothing and is
    // never read. Threshold comes from the resident bracket rather than from an
    // argument, so the whole bisection stays on the GPU and the host never sees
    // an intermediate count.
    const tileCount = gpu
      .createKernel(function (x, m, bnd, b) {
        const theta = (bnd[0] + bnd[1]) * 0.5;
        if (m[this.thread.x] < theta) return 0;
        let c = 0;
        for (let e = 0; e < this.constants.g; e++) {
          if (x[this.thread.x + e * this.constants.t] + b >= theta) c += 1;
        }
        return c;
      })
      .setConstants({ g, t })
      .setOutput([t])
      .setPipeline(true);

    // 131,072 tile counts folded to 512 block sums. Contiguous, not strided:
    // the gather walks blocks and then tiles-within-a-block, and a strided fold
    // would make "the tiles of block b" a scattered set for no gain on an array
    // this small.
    const blockSum = gpu
      .createKernel(function (c) {
        const start = this.thread.x * this.constants.blk;
        let s = 0;
        for (let i = 0; i < this.constants.blk; i++) s += c[start + i];
        return s;
      })
      .setConstants({ blk })
      .setOutput([nb])
      .setPipeline(true);

    // The bracket, as two GPU-resident floats. It starts at [b, 1+b), which
    // brackets the data by construction: every key is >= b and every key < 1+b.
    const initBounds = gpu
      .createKernel(function (b) {
        if (this.thread.x < 0.5) return b;
        return b + 1;
      })
      .setOutput([2])
      .setPipeline(true);

    /**
     * One halving, and the only place the count is ever compared against k. Two
     * threads, so the new bracket comes out as one texture instead of two; both
     * of them re-add the 512 block sums, which is 512 wasted adds against the
     * cost of a whole extra kernel in the chain to compute the total once.
     *
     * Two instances because a gpu.js pipeline kernel owns one output texture and
     * cannot read the bracket it is about to overwrite; they alternate.
     */
    const mkRefine = () =>
      gpu
        .createKernel(function (bs, bnd) {
          let total = 0;
          for (let i = 0; i < this.constants.nb; i++) total += bs[i];
          const lo = bnd[0];
          const hi = bnd[1];
          const mid = (lo + hi) * 0.5;
          // count(x >= mid) >= k keeps the answer in the upper half.
          if (this.thread.x < 0.5) {
            if (total >= this.constants.k) return mid;
            return lo;
          }
          if (total >= this.constants.k) return hi;
          return mid;
        })
        .setConstants({ nb, k })
        .setOutput([2])
        .setPipeline(true);
    const refineA = mkRefine();
    const refineB = mkRefine();

    /**
     * Pull the survivors into a dense 512-slot buffer. Thread j wants the j-th
     * element strictly above the k-th largest value, in tile-major order: it
     * accumulates the 512 block sums to find its block, the 256 tile counts
     * inside that block to find its tile, then reads that tile's 64 elements.
     * 832 reads, no prefix sum, no scatter.
     *
     * No `break` anywhere: the loops run to completion and the running prefix
     * is non-decreasing, so "the last index whose prefix is still <= j" is
     * exactly the block (or tile) before the one that contains j. Several gpu.js
     * backends will not compile a `break`, and a loop whose trip count depends
     * on the data is the wrong shape for a GPU regardless.
     *
     * Slots past the last survivor get the k-th largest value itself. That is
     * not padding: count(x >= t*) >= k by construction, so the multiset of the k
     * largest values genuinely ends in (k - m) copies of t*.
     */
    const gather = gpu
      .createKernel(function (x, c, bs, bnd, b) {
        const j = this.thread.x;
        const tstar = bnd[0];
        const theta = (bnd[0] + bnd[1]) * 0.5;

        let prefix = 0;
        let block = 0;
        let base = 0;
        for (let i = 0; i < this.constants.nb; i++) {
          prefix += bs[i];
          if (prefix <= j) {
            block = i + 1;
            base = prefix;
          }
        }
        if (j >= prefix) return tstar;

        let p2 = base;
        let tile = block * this.constants.blk;
        let base2 = base;
        for (let i = 0; i < this.constants.blk; i++) {
          p2 += c[block * this.constants.blk + i];
          if (p2 <= j) {
            tile = block * this.constants.blk + i + 1;
            base2 = p2;
          }
        }

        const want = j - base2;
        let seen = 0;
        let v = tstar;
        for (let e = 0; e < this.constants.g; e++) {
          const val = x[tile + e * this.constants.t] + b;
          if (val >= theta) {
            if (seen === want) v = val;
            seen += 1;
          }
        }
        return v;
      })
      .setConstants({ nb, blk, g, t })
      .setOutput([k])
      .setPipeline(true);

    // Sort the 512. A rank matrix is the gather-only way to sort, and at k=512
    // it is 262,144 reads — one thirty-second of a single pass over the input,
    // which is why nobody should read this stage as "the GPU sorted something".
    // Ties break on index so the rank is a bijection and the inversion below
    // finds exactly one source for every slot.
    const rank = gpu
      .createKernel(function (c) {
        const me = c[this.thread.x];
        let r = 0;
        for (let i = 0; i < this.constants.k; i++) {
          const o = c[i];
          if (o > me) r += 1;
          else if (o === me && i < this.thread.x) r += 1;
        }
        return r;
      })
      .setConstants({ k })
      .setOutput([k])
      .setPipeline(true);

    // Not a pipeline kernel: this is the run's result, so its resolution is
    // what proves every dispatch behind it finished.
    const invert = gpu
      .createKernel(function (c, rk) {
        let v = 0;
        for (let i = 0; i < this.constants.k; i++) {
          if (rk[i] === this.thread.x) v = c[i];
        }
        return v;
      })
      .setConstants({ k })
      .setOutput([k]);

    return {
      async run() {
        const keys = await upload(a);
        let out = null;
        for (let r = 0; r < rounds; r++) {
          const b = r * bias;
          // eslint-disable-next-line no-await-in-loop
          const m = await tileMax(keys, b);
          // eslint-disable-next-line no-await-in-loop
          let bnd = await initBounds(b);
          for (let s = 0; s < steps; s++) {
            // eslint-disable-next-line no-await-in-loop
            const c = await tileCount(keys, m, bnd, b);
            // eslint-disable-next-line no-await-in-loop
            const bs = await blockSum(c);
            // eslint-disable-next-line no-await-in-loop
            bnd = await (s % 2 === 0 ? refineA : refineB)(bs, bnd);
          }
          // The bracket is now [t*, t* + 2^-22), so this last count uses the
          // midpoint t* + 2^-23 and therefore counts x > t* strictly — which is
          // exactly the set the gather wants.
          // eslint-disable-next-line no-await-in-loop
          const c = await tileCount(keys, m, bnd, b);
          // eslint-disable-next-line no-await-in-loop
          const bs = await blockSum(c);
          // eslint-disable-next-line no-await-in-loop
          const cand = await gather(keys, c, bs, bnd, b);
          // eslint-disable-next-line no-await-in-loop
          const rk = await rank(cand);
          // eslint-disable-next-line no-await-in-loop
          out = await invert(cand, rk);
        }
        return out;
      },
      backend: () => tileCount.kernel && tileCount.kernel.constructor.mode,
      destroy() {
        [upload, tileMax, tileCount, blockSum, initBounds, refineA, refineB, gather, rank, invert].forEach(
          x => x.destroy && x.destroy()
        );
      },
    };
  },

  /**
   * Hand-written WebGPU. Identical algorithm, identical tile layout, identical
   * arithmetic — the only difference is that all 24 rounds' worth of dispatches
   * (1,752 of them) are recorded into ONE compute pass and submitted once, and
   * the bracket lives in a storage buffer that the host never reads. WebGPU
   * orders dispatches within a pass and inserts the barrier itself, so the
   * counting kernel genuinely sees the refinement kernel's write from the
   * dispatch before it.
   *
   * On a row with this many small dispatches that single fact is most of what
   * separates this cell from the gpu.js WebGPU cell beside it, which is exactly
   * what the bare column is here to isolate.
   */
  async webgpu(device, { n, k, g, t, blk, nb, steps, bias, rounds }, { a }) {
    const S = GPUBufferUsage.STORAGE;
    const mk = (count, extra = 0) => device.createBuffer({ size: Math.max(16, count * 4), usage: S | extra });

    const keys = mk(n, GPUBufferUsage.COPY_DST);
    const tmax = mk(t);
    const tcnt = mk(t);
    const bsum = mk(nb);
    const bnd = mk(2);
    const cand = mk(k);
    const rk = mk(k);
    const out = mk(k, GPUBufferUsage.COPY_SRC);
    const read = device.createBuffer({ size: k * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // 256 and not 64: the widest dispatch here is 131,072 threads, and 256 both
    // keeps that to 512 workgroups and matches the block width the gather walks,
    // so one number describes the whole shader.
    const WG = 256;
    const CONST = `
const G: u32 = ${g}u;
const T: u32 = ${t}u;
const BLK: u32 = ${blk}u;
const NB: u32 = ${nb}u;
const K: u32 = ${k}u;
struct P { b: f32 };`;

    // One module per entry point. `layout: "auto"` derives a bind group layout
    // from the bindings an entry point actually touches, so sharing a module
    // between entry points with different binding sets is a quiet way to get
    // layouts that do not match the bind groups written against them.
    const shader = code => device.createShaderModule({ code });

    const mInit = shader(`${CONST}
@group(0) @binding(0) var<storage, read_write> bnd: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(1)
fn main() {
  bnd[0] = p.b;
  bnd[1] = p.b + 1.0;
}`);

    const mTileMax = shader(`${CONST}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> tmax: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= T) { return; }
  var m = x[j] + p.b;
  for (var e: u32 = 1u; e < G; e = e + 1u) {
    let v = x[j + e * T] + p.b;
    if (v > m) { m = v; }
  }
  tmax[j] = m;
}`);

    const mTileCount = shader(`${CONST}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> tmax: array<f32>;
@group(0) @binding(2) var<storage, read> bnd: array<f32>;
@group(0) @binding(3) var<storage, read_write> tcnt: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= T) { return; }
  let theta = (bnd[0] + bnd[1]) * 0.5;
  // The prune. Whole workgroups take this together once the threshold has
  // climbed, and the 64 reads below never happen.
  if (tmax[j] < theta) { tcnt[j] = 0.0; return; }
  var c = 0.0;
  for (var e: u32 = 0u; e < G; e = e + 1u) {
    if (x[j + e * T] + p.b >= theta) { c = c + 1.0; }
  }
  tcnt[j] = c;
}`);

    const mBlockSum = shader(`${CONST}
@group(0) @binding(0) var<storage, read> tcnt: array<f32>;
@group(0) @binding(1) var<storage, read_write> bsum: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= NB) { return; }
  var s = 0.0;
  for (var q: u32 = 0u; q < BLK; q = q + 1u) { s = s + tcnt[i * BLK + q]; }
  bsum[i] = s;
}`);

    const mRefine = shader(`${CONST}
@group(0) @binding(0) var<storage, read> bsum: array<f32>;
@group(0) @binding(1) var<storage, read_write> bnd: array<f32>;
@compute @workgroup_size(1)
fn main() {
  var total = 0.0;
  for (var i: u32 = 0u; i < NB; i = i + 1u) { total = total + bsum[i]; }
  let lo = bnd[0];
  let hi = bnd[1];
  let mid = (lo + hi) * 0.5;
  if (total >= f32(K)) { bnd[0] = mid; } else { bnd[1] = mid; }
}`);

    const mGather = shader(`${CONST}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> tcnt: array<f32>;
@group(0) @binding(2) var<storage, read> bsum: array<f32>;
@group(0) @binding(3) var<storage, read> bnd: array<f32>;
@group(0) @binding(4) var<storage, read_write> cand: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let jj = gid.x;
  if (jj >= K) { return; }
  let j = f32(jj);
  let tstar = bnd[0];
  let theta = (bnd[0] + bnd[1]) * 0.5;

  var prefix = 0.0;
  var block: u32 = 0u;
  var base = 0.0;
  for (var i: u32 = 0u; i < NB; i = i + 1u) {
    prefix = prefix + bsum[i];
    if (prefix <= j) { block = i + 1u; base = prefix; }
  }
  if (j >= prefix) { cand[jj] = tstar; return; }

  var p2 = base;
  var tile = block * BLK;
  var base2 = base;
  for (var i: u32 = 0u; i < BLK; i = i + 1u) {
    p2 = p2 + tcnt[block * BLK + i];
    if (p2 <= j) { tile = block * BLK + i + 1u; base2 = p2; }
  }

  let want = j - base2;
  var seen = 0.0;
  var v = tstar;
  for (var e: u32 = 0u; e < G; e = e + 1u) {
    let val = x[tile + e * T] + p.b;
    if (val >= theta) {
      if (seen == want) { v = val; }
      seen = seen + 1.0;
    }
  }
  cand[jj] = v;
}`);

    const mRank = shader(`${CONST}
@group(0) @binding(0) var<storage, read> cand: array<f32>;
@group(0) @binding(1) var<storage, read_write> rk: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= K) { return; }
  let me = cand[i];
  var r = 0.0;
  for (var q: u32 = 0u; q < K; q = q + 1u) {
    let o = cand[q];
    if (o > me) { r = r + 1.0; }
    else if (o == me && q < i) { r = r + 1.0; }
  }
  rk[i] = r;
}`);

    const mInvert = shader(`${CONST}
@group(0) @binding(0) var<storage, read> cand: array<f32>;
@group(0) @binding(1) var<storage, read> rk: array<f32>;
@group(0) @binding(2) var<storage, read_write> outb: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= K) { return; }
  var v = 0.0;
  for (var q: u32 = 0u; q < K; q = q + 1u) {
    if (rk[q] == f32(j)) { v = cand[q]; }
  }
  outb[j] = v;
}`);

    const pipe = module => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const cInit = pipe(mInit);
    const cTileMax = pipe(mTileMax);
    const cTileCount = pipe(mTileCount);
    const cBlockSum = pipe(mBlockSum);
    const cRefine = pipe(mRefine);
    const cGather = pipe(mGather);
    const cRank = pipe(mRank);
    const cInvert = pipe(mInvert);

    // WebGPU has no push constants and queue.writeBuffer cannot be interleaved
    // between dispatches inside one command buffer, so the per-round bias lives
    // in `rounds` slices of one uniform buffer at the 256-byte alignment, with a
    // bind group per round built once here at zero per-run cost.
    const STRIDE = 256;
    const params = device.createBuffer({
      size: rounds * STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const staging = new Float32Array((rounds * STRIDE) / 4);
    for (let r = 0; r < rounds; r++) staging[(r * STRIDE) / 4] = r * bias;
    device.queue.writeBuffer(params, 0, staging);

    const bg = (pipeline, buffers, round) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers
          .map((buffer, binding) => ({ binding, resource: { buffer } }))
          .concat(
            round === undefined
              ? []
              : [
                  {
                    binding: buffers.length,
                    resource: { buffer: params, offset: round * STRIDE, size: 16 },
                  },
                ]
          ),
      });

    const perRound = [];
    for (let r = 0; r < rounds; r++) {
      perRound.push({
        init: bg(cInit, [bnd], r),
        tileMax: bg(cTileMax, [keys, tmax], r),
        tileCount: bg(cTileCount, [keys, tmax, bnd, tcnt], r),
        gather: bg(cGather, [keys, tcnt, bsum, bnd, cand], r),
      });
    }
    const bBlockSum = bg(cBlockSum, [tcnt, bsum]);
    const bRefine = bg(cRefine, [bsum, bnd]);
    const bRank = bg(cRank, [cand, rk]);
    const bInvert = bg(cInvert, [cand, rk, out]);

    const gT = Math.ceil(t / WG);
    const gNB = Math.ceil(nb / WG);
    const gK = Math.ceil(k / WG);

    return {
      async run() {
        // One upload per run, matching the gpu.js column's single upload.
        device.queue.writeBuffer(keys, 0, a);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let r = 0; r < rounds; r++) {
          const b = perRound[r];
          pass.setPipeline(cInit);
          pass.setBindGroup(0, b.init);
          pass.dispatchWorkgroups(1);
          pass.setPipeline(cTileMax);
          pass.setBindGroup(0, b.tileMax);
          pass.dispatchWorkgroups(gT);
          for (let s = 0; s <= steps; s++) {
            pass.setPipeline(cTileCount);
            pass.setBindGroup(0, b.tileCount);
            pass.dispatchWorkgroups(gT);
            pass.setPipeline(cBlockSum);
            pass.setBindGroup(0, bBlockSum);
            pass.dispatchWorkgroups(gNB);
            // The extra trip round this loop (s === steps) is the strict count
            // at t* + 2^-23 that the gather needs; refining after it would move
            // a bracket nobody reads again, so it is skipped.
            if (s < steps) {
              pass.setPipeline(cRefine);
              pass.setBindGroup(0, bRefine);
              pass.dispatchWorkgroups(1);
            }
          }
          pass.setPipeline(cGather);
          pass.setBindGroup(0, b.gather);
          pass.dispatchWorkgroups(gK);
          pass.setPipeline(cRank);
          pass.setBindGroup(0, bRank);
          pass.dispatchWorkgroups(gK);
          pass.setPipeline(cInvert);
          pass.setBindGroup(0, bInvert);
          pass.dispatchWorkgroups(gK);
        }
        pass.end();
        enc.copyBufferToBuffer(out, 0, read, 0, k * 4);
        device.queue.submit([enc.finish()]);
        // The map is the only thing that proves the pass finished.
        await read.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return result;
      },
      destroy() {
        [keys, tmax, tcnt, bsum, bnd, cand, rk, out, read, params].forEach(x => x.destroy && x.destroy());
      },
    };
  },

  /**
   * Two things at once, because a selection can be wrong in two ways.
   *
   * The weighted sum uses a strictly increasing weight, so any transposition of
   * two different values changes it — membership and position are both covered,
   * and a column that found the right 512 values but emitted them in the wrong
   * order does not pass. The monotonicity count then catches what a weighted sum
   * cannot: the output must be non-increasing, and each violation adds a whole 1
   * to a number of order 1.5, which is four orders of magnitude outside the
   * runner's tolerance.
   *
   * Selection moves values without doing arithmetic on them and every value here
   * is exact in fp32, so a correct backend agrees to the last bit.
   */
  reduce(out, { k }) {
    const a = ArrayBuffer.isView(out) ? out : Float32Array.from(out);
    let acc = 0;
    let bad = 0;
    for (let j = 0; j < a.length; j++) {
      acc += a[j] * (1 + j / k);
      if (j > 0 && a[j] > a[j - 1]) bad++;
    }
    return acc / k + bad;
  },
};
