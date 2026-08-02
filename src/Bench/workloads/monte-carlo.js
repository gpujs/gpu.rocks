/**
 * Monte Carlo option pricing — geometric Brownian motion, European call.
 *
 * Every other row on this page reads its numbers from memory. This one makes
 * them up, four million paths at a time, and that is the whole point: the
 * arithmetic below is about 80% random-number generation and 20% finance. A GPU
 * is supposed to be extraordinary at this — the paths are perfectly independent,
 * there is no communication, no neighbourhood, no reduction until the very end —
 * so if the speed-up is ever going to be large it is going to be large here.
 *
 * There is a second thing this row prices, which is less obvious. A PRNG stream
 * is a chain: each draw depends on the previous state, so a single path is a
 * long sequence of dependent multiply-floor-subtract steps and a CPU core runs
 * it at the speed of its own latency, not its throughput. A GPU has thousands of
 * paths in flight and hides that chain completely. The gap on this row is
 * therefore wider than the flop counts suggest, and it is wider for a reason
 * worth naming.
 *
 * ── WHY NOT Math.random ─────────────────────────────────────────────────────
 *
 * Because then the row could not be checked, and an unchecked benchmark row is
 * decoration. Six columns have to agree on one number to 1e-4; if each of them
 * draws its own randomness they produce six different (perfectly valid) prices
 * and there is no way left to tell a broken kernel from a differently-seeded
 * one. Math.random also has no WebGPU equivalent — a compute shader has no
 * global mutable state to keep a generator in — so the GPU columns would have to
 * do something else entirely, and a row where two columns do different work is
 * not a benchmark.
 *
 * So the generator is a pure function: path p draws its whole stream from a
 * counter-based seed, every backend evaluates the same function, and the six
 * columns produce the SAME price. That constraint is not a workaround. It is how
 * production GPU Monte Carlo is actually written, for exactly this reason:
 * reproducibility across a machine that reorders everything.
 *
 * ── WHY THE GENERATOR LOOKS LIKE THAT (fp32, and gpu.js has no integers) ─────
 *
 * The obvious counter-based PRNG is a 32-bit integer hash: xor, shift, multiply
 * by an odd constant, let it wrap. That cannot be used here.
 *
 *   - gpu.js has no integer pipe. Every kernel value is an f32; `Math.imul`
 *     compiles to `float(int(a) * int(b))` on WebGL, WebGL2 AND WebGPU, so the
 *     low bits of a 32-bit product are rounded away before you ever see them.
 *   - WebGL's GLSL has no integer bitwise operators at all, so gpu.js emulates
 *     `^`, `&` and `<<` with 32-iteration loops over floats. A hash would cost
 *     more than the finance and the row would be measuring a polyfill.
 *
 * fp32 does hold every integer below 2^24 exactly, though, and that is enough to
 * build a real generator out of nothing but multiply, add, subtract and floor.
 * Every intermediate below is bounded by 2^24 BY CONSTRUCTION (the bounds are
 * spelled out at each constant), so the arithmetic is exact rather than
 * approximate, and the JS, gpu.js and WGSL columns produce bit-identical draws
 * on every backend — no tolerance needed, and no dependence on how a driver
 * rounds anything.
 *
 * The generator is: a 24-bit LCG held as two 12-bit lanes, rotated 12 bits per
 * step so the multiply's carries reach the top, with a per-path Weyl counter as
 * the increment. Each round yields the LCG's top 12 bits — the only bits an LCG
 * modulo a power of two is entitled to.
 *
 * ── HOW THE STREAMS WERE VERIFIED TO BE INDEPENDENT ─────────────────────────
 *
 * This is the part that is easy to get wrong, and the first three designs did.
 *
 * The failure is not that a stream looks non-random; it is that two DIFFERENT
 * paths produce the SAME stream, which no amount of testing one stream will
 * ever reveal. A plain 24-bit LCG-plus-rotate has it: for a fixed increment the
 * round is affine, so a difference between two states can be a fixed point of
 * its own propagation, and the two paths then emit identical draws until a carry
 * finally breaks the tie. Hunting exhaustively over all 2^22 streams found 121
 * such pairs, agreeing for a median of 15 and a maximum of 157 draws. Rare, but
 * structural, and it does not go away with more warm-up.
 *
 * The fix is the Weyl counter: each path's increment ADVANCES by its own odd
 * stride (2p+1, distinct for every path), so two paths can never share a
 * constant difference for more than a moment and the fixed point cannot exist.
 * The same exhaustive hunt over all 2^22 streams then finds ZERO pairs with even
 * their first 12 draws in common.
 *
 * The rest of the evidence, all measured, none assumed:
 *
 *   - fp32 exactness: re-running seeding and 60 draws with Math.fround applied
 *     to every single operation — which is what a GPU does — reproduced the
 *     fp64 draws exactly, 0 mismatches over 120 000 streams. Largest
 *     intermediate ever seen: 16 768 551, under the 2^24 ceiling.
 *   - between streams: draws 0, 1, 7 and 23 of three million consecutive paths,
 *     binned 1-D and 2-D at counter lags of 1, 2, 64, 512, 4096, 65 536,
 *     262 144 and 1 048 576. All 36 reduced chi-squares landed in [0.96, 1.04];
 *     unity is the ideal. (The power-of-two lags are the ones that catch a weak
 *     seed: an earlier seed that used the LOW 24 bits of p² scored 68 there
 *     rather than 1. This one uses the middle lanes of the full 44-bit square.)
 *   - within a stream: 2-D and 3-D bin counts over 300 000 streams × 48 draws,
 *     chi-square 1.020 and 1.011.
 *   - bit level: all 12 output bits, autocorrelation at lags 1 to 12, worst of
 *     the 144 tests 2.5 sigma. (This is the specific trap — the low bit of a
 *     power-of-two LCG has period 2. It is not in the output at all: the output
 *     is the top lane.)
 *   - and the answer itself: the price below lands within 0.05% of the
 *     Black–Scholes closed form, which is inside the Monte Carlo standard error
 *     of 0.06% at this path count. A broken generator does not do that.
 *
 * ── WHY THE PATH IS EULER–MARUYAMA AND NOT THE EXACT SOLUTION ───────────────
 *
 * A European call is path-INDEPENDENT, so the textbook exact scheme —
 * log S_T = log S_0 + (r - s²/2)T + s√T·Z — collapses the whole path into one
 * normal draw. Written that way the step loop below would be arithmetic theatre:
 * summing the increments and exponentiating once gives the identical answer, and
 * a reader would be right to ask what the steps were for.
 *
 * Euler–Maruyama on the SDE keeps them honest. S is advanced multiplicatively,
 * S <- S(1 + r·dt + s√dt·z), and a product of eight factors is NOT the
 * exponential of a sum, so the discretisation genuinely happens. It is also the
 * first scheme in every textbook and the one you are forced back to the moment
 * the payoff stops being European. Its cost is a discretisation bias, and that
 * bias is measured rather than hoped for: 0.05% against Black–Scholes at eight
 * steps, smaller than the sampling error.
 *
 * ── AND WHY THE NORMAL VARIATE IS A SUM OF THREE UNIFORMS ───────────────────
 *
 * Box–Muller would put a log, a sqrt and a cos in the inner loop, and then this
 * row would be a transcendental-function benchmark: GPUs have dedicated hardware
 * for those and libm does not, so the row would report the special-function
 * units rather than the generator. The Irwin–Hall sum of three uniforms is pure
 * arithmetic, has variance exactly 1 - 2^-24, and keeps the row on the thing it
 * claims to measure. Its tails are truncated at ±3 per step — which does not
 * reach the answer, because the payoff sees the TERMINAL value, and that is a
 * sum of 24 uniforms whose distribution is normal to well beyond the accuracy
 * this row needs. The 0.05% agreement with Black–Scholes is the proof.
 */

const PATHS = 1 << 22;
const STEPS = 8;
const DRAWS_PER_STEP = 3; // Irwin-Hall: three uniforms make one N(0,1)

// Contract terms. At-the-money, one year, the textbook numbers.
const SPOT = 100;
const STRIKE = 100;
const RATE = 0.05;
const VOL = 0.2;
const TERM = 1;

// ── The generator, and the bound on every value it computes ─────────────────
//
// State: two 12-bit lanes (h, l) holding a 24-bit LCG state v = h·4096 + l, plus
// a 23-bit Weyl counter c that advances by the path's own odd stride s.
//
// One round is  v <- rot12((MUL·v + c) mod 2^24),  emitted as the top 12 bits.
// Expanded into lanes that is exactly the four lines in every column below, and
// the reason it can be done in floats is that nothing exceeds 2^24:
//
//     MUL·l   <= 1133 · 4095    =  4 639 635
//     c       <  2^23           =  8 388 608     ← Weyl counter, always reduced
//     B       =  MUL·l + c      <= 13 028 243    ← the largest value in the loop
//     Bh      =  floor(B/4096)  <=      3 180
//     MUL·h + Bh                <=  4 642 815
//     c + s   <  2^23 + 2^23    =  2^24          ← exact, hence the reduce below
//
// MUL is 1133 because it is the 12-bit multiplier that scored best on the
// cross-stream chi-square sweep of all 512 candidates congruent to 5 mod 8.
const MUL = 1133;
const LANE = 4096;
const INV_LANE = 1 / 4096; // a power of two, so multiplying by it is exact
const WEYL = 8388608; // 2^23
const INV_WEYL = 1 / 8388608;

// Four rounds discarded after seeding. Measured, not guessed: the cross-stream
// chi-squares are already at unity after two, and go bad fast below that.
const WARMUP = 4;

// Precomputed so the inner loop is one multiply-add. z = 2(u1+u2+u3) - 3 with
// u = (t + 0.5)/4096, so S <- S·(base + zscale·(t1+t2+t3)).
const DT = TERM / STEPS;
const SIGMA = VOL * Math.sqrt(DT);
const ZSCALE = 2 * INV_LANE * SIGMA;
const BASE = 1 + RATE * DT - 3 * SIGMA + 1.5 * ZSCALE;
const DISCOUNT = Math.exp(-RATE * TERM);

export default {
  id: 'monte-carlo',
  name: 'Monte Carlo option pricing',
  params: `2^22 paths · ${STEPS} steps · European call, fp32`,
  tag: 'RNG-bound',
  group: 'sim',
  size: { paths: PATHS, steps: STEPS },

  // No make(). The only input is the path index, and the path index is the
  // thread id. Handing the columns a table of random numbers instead would turn
  // the row into a bandwidth test — 2^22 paths × 24 draws is 400 MB of table —
  // and delete the thing it is here to measure.

  /**
   * The oracle, and a fair baseline: flat typed array out, the generator state
   * in three local variables so it stays in registers, the whole per-step
   * constant folded into BASE and ZSCALE so the inner loop is one multiply-add,
   * and the three uniforms summed as INTEGERS and scaled once rather than
   * divided three times.
   *
   * S is a plain JS number, so this column carries the path in fp64 while every
   * GPU column carries it in fp32 — the oracle should be the accurate one. The
   * draws themselves are identical everywhere (they are exact), so the only
   * divergence is eight roundings of one multiply. Re-running this function with
   * Math.fround on every operation of the path arithmetic moves the checksum by
   * 1.4e-6 relative — seventy times inside the runner's tolerance.
   */
  js({ paths, steps }) {
    const out = new Float32Array(paths);
    for (let p = 0; p < paths; p++) {
      // Seed: the middle 24 bits of the full 44-bit p², assembled lane by lane.
      // The low 24 bits would be cheaper and are much worse — see the header.
      const ph = Math.floor(p * INV_LANE);
      const pl = p - ph * LANE;
      const t0 = pl * pl;
      const k0 = Math.floor(t0 * INV_LANE);
      const lane0 = t0 - k0 * LANE;
      const t1 = 2 * ph * pl + k0;
      const k1 = Math.floor(t1 * INV_LANE);
      const lane1 = t1 - k1 * LANE;
      const t2 = ph * ph + k1;
      const lane2 = t2 - Math.floor(t2 * INV_LANE) * LANE;

      let h = lane2;
      let l = lane1;
      const q = lane2 * LANE + lane0;
      let c = q - WEYL * Math.floor(q * INV_WEYL);
      // The stride. Odd and distinct for every path, which is what makes two
      // streams unable to stay in lockstep. See the header.
      const s = 2 * p + 1;

      for (let i = 0; i < WARMUP; i++) {
        c += s;
        if (c >= WEYL) c -= WEYL;
        const b = MUL * l + c;
        const bh = Math.floor(b * INV_LANE);
        const top = MUL * h + bh;
        h = b - bh * LANE;
        l = top - LANE * Math.floor(top * INV_LANE);
      }

      let spot = SPOT;
      for (let k = 0; k < steps; k++) {
        let tsum = 0;
        for (let j = 0; j < DRAWS_PER_STEP; j++) {
          c += s;
          if (c >= WEYL) c -= WEYL;
          const b = MUL * l + c;
          const bh = Math.floor(b * INV_LANE);
          const top = MUL * h + bh;
          h = b - bh * LANE;
          l = top - LANE * Math.floor(top * INV_LANE);
          tsum += l;
        }
        spot *= BASE + ZSCALE * tsum;
      }
      out[p] = spot > STRIKE ? (spot - STRIKE) * DISCOUNT : 0;
    }
    return out;
  },

  gpujs(gpu, { paths, steps }) {
    // No arguments at all — nothing is uploaded, so what this column measures is
    // arithmetic plus one 16 MB readback, which is the comparison it should be.
    const kernel = gpu
      .createKernel(function () {
        const lane = this.constants.lane;
        const invLane = this.constants.invLane;
        const weyl = this.constants.weyl;
        const mul = this.constants.mul;
        const idx = this.thread.x;

        const ph = Math.floor(idx * invLane);
        const pl = idx - ph * lane;
        const a0 = pl * pl;
        const k0 = Math.floor(a0 * invLane);
        const lane0 = a0 - k0 * lane;
        const a1 = 2 * ph * pl + k0;
        const k1 = Math.floor(a1 * invLane);
        const lane1 = a1 - k1 * lane;
        const a2 = ph * ph + k1;
        const lane2 = a2 - Math.floor(a2 * invLane) * lane;

        let h = lane2;
        let l = lane1;
        const q = lane2 * lane + lane0;
        let c = q - weyl * Math.floor(q * this.constants.invWeyl);
        const s = 2 * idx + 1;

        for (let i = 0; i < this.constants.warmup; i++) {
          c = c + s;
          if (c >= weyl) c = c - weyl;
          const b = mul * l + c;
          const bh = Math.floor(b * invLane);
          const top = mul * h + bh;
          h = b - bh * lane;
          l = top - lane * Math.floor(top * invLane);
        }

        let spot = this.constants.spot;
        for (let k = 0; k < this.constants.steps; k++) {
          let tsum = 0;
          for (let j = 0; j < 3; j++) {
            c = c + s;
            if (c >= weyl) c = c - weyl;
            const b = mul * l + c;
            const bh = Math.floor(b * invLane);
            const top = mul * h + bh;
            h = b - bh * lane;
            l = top - lane * Math.floor(top * invLane);
            tsum = tsum + l;
          }
          spot = spot * (this.constants.base + this.constants.zscale * tsum);
        }
        if (spot > this.constants.strike) {
          return (spot - this.constants.strike) * this.constants.discount;
        }
        return 0;
      })
      .setConstants({
        mul: MUL,
        lane: LANE,
        invLane: INV_LANE,
        weyl: WEYL,
        invWeyl: INV_WEYL,
        warmup: WARMUP,
        steps,
        spot: SPOT,
        strike: STRIKE,
        base: BASE,
        zscale: ZSCALE,
        discount: DISCOUNT,
      })
      // Not optional. The generator needs every integer below 2^24 to survive
      // exactly; gpu.js's default 'unsigned' encoding packs a value into RGBA8
      // and would round the payoff to nothing like the right number. 'single'
      // asks for float32 and fails loudly on a machine that cannot provide it.
      .setPrecision('single')
      .setOutput([paths]);

    return {
      // await, not fire-and-forget: on the WebGPU path the result is a promise
      // and returning early would time the dispatch call rather than the paths.
      async run() {
        return await kernel();
      },
      backend: () => kernel.kernel && kernel.kernel.constructor.mode,
      destroy: () => kernel.destroy && kernel.destroy(),
    };
  },

  /**
   * Hand-written WGSL, with nothing borrowed from gpu.js — the gap between this
   * cell and the WebGPU cell to its left is the runtime's price.
   *
   * The kernel is deliberately line-for-line the gpu.js one. There is no clever
   * WGSL version of this available: WGSL does have u32 and could run a proper
   * 32-bit integer hash three times faster than the float generator below, but
   * then this column would be running a DIFFERENT generator, would produce a
   * different price, and the row would be a lie that happened to be fast. Same
   * arithmetic, same draws, same answer — the difference is dispatch and
   * readback, which is exactly what is being asked.
   *
   * workgroup_size is 256 rather than 64 because 2^22 threads at 64 would need
   * 65 536 workgroups and the limit is 65 535.
   */
  async webgpu(device, { paths, steps }) {
    const bytes = paths * 4;
    const out = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const GROUP = 256;
    const module = device.createShaderModule({
      code: `
@group(0) @binding(0) var<storage, read_write> payoff: array<f32>;

const N: u32 = ${paths}u;
const MUL: f32 = ${MUL}.0;
const LANE: f32 = ${LANE}.0;
const INV_LANE: f32 = ${INV_LANE};
const WEYL: f32 = ${WEYL}.0;
const INV_WEYL: f32 = ${INV_WEYL};
const SPOT: f32 = ${SPOT}.0;
const STRIKE: f32 = ${STRIKE}.0;
const BASE: f32 = ${BASE};
const ZSCALE: f32 = ${ZSCALE};
const DISCOUNT: f32 = ${DISCOUNT};

@compute @workgroup_size(${GROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= N) { return; }
  let idx = f32(gid.x);

  // Seed: middle lanes of the full 44-bit idx².
  let ph = floor(idx * INV_LANE);
  let pl = idx - ph * LANE;
  let a0 = pl * pl;
  let k0 = floor(a0 * INV_LANE);
  let lane0 = a0 - k0 * LANE;
  let a1 = 2.0 * ph * pl + k0;
  let k1 = floor(a1 * INV_LANE);
  let lane1 = a1 - k1 * LANE;
  let a2 = ph * ph + k1;
  let lane2 = a2 - floor(a2 * INV_LANE) * LANE;

  var h = lane2;
  var l = lane1;
  let q = lane2 * LANE + lane0;
  var c = q - WEYL * floor(q * INV_WEYL);
  let s = 2.0 * idx + 1.0;

  for (var i: u32 = 0u; i < ${WARMUP}u; i = i + 1u) {
    c = c + s;
    c = select(c, c - WEYL, c >= WEYL);
    let b = MUL * l + c;
    let bh = floor(b * INV_LANE);
    let top = MUL * h + bh;
    h = b - bh * LANE;
    l = top - LANE * floor(top * INV_LANE);
  }

  var spot = SPOT;
  for (var k: u32 = 0u; k < ${steps}u; k = k + 1u) {
    var tsum = 0.0;
    for (var j: u32 = 0u; j < 3u; j = j + 1u) {
      c = c + s;
      c = select(c, c - WEYL, c >= WEYL);
      let b = MUL * l + c;
      let bh = floor(b * INV_LANE);
      let top = MUL * h + bh;
      h = b - bh * LANE;
      l = top - LANE * floor(top * INV_LANE);
      tsum = tsum + l;
    }
    spot = spot * (BASE + ZSCALE * tsum);
  }
  payoff[gid.x] = select(0.0, (spot - STRIKE) * DISCOUNT, spot > STRIKE);
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: out } }],
    });
    const groups = Math.ceil(paths / GROUP);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(groups);
        pass.end();
        enc.copyBufferToBuffer(out, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        // The read-back is the only thing that proves the dispatch finished.
        await read.mapAsync(GPUMapMode.READ);
        const v = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return v;
      },
      destroy() {
        [out, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Every path, index-weighted. The plain mean of the payoffs IS the price, and
   * it is the number this row is nominally computing — but a mean is a terrible
   * checksum, because a backend that priced half the paths and left the rest at
   * their initial value could still land on it. The weight makes each path's
   * position matter, so a partly-filled output cannot pass.
   *
   * Payoffs are non-negative and bounded, the weights are small integers, and
   * the total is divided by the path count, so the sum of four million terms
   * stays in a range where fp32 and fp64 agree far inside the runner's 1e-4.
   * The measured fp32-throughout spread on this checksum is 1.4e-6.
   */
  reduce(out, { paths }) {
    const flat = ArrayBuffer.isView(out) ? out : Float32Array.from(out);
    let acc = 0;
    for (let i = 0; i < flat.length; i++) acc += flat[i] * (1 + (i % 17));
    return acc / paths;
  },
};
