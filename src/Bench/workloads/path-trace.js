/**
 * Progressive path tracing — two spheres, a ground plane, and a sky.
 *
 * This is the row where the GPU is doing the thing it was invented for, and it
 * is also the row where the two GPU columns stop being the same shape. A path
 * tracer is not one kernel: it is the SAME kernel run once per sample, each run
 * adding its contribution to an image that is already there. That accumulate is
 * the whole structure of progressive rendering — it is why a renderer can show
 * you a noisy frame immediately and a clean one a minute later — and it is
 * exactly the operation gpu.js cannot express.
 *
 *   - A gpu.js kernel returns one fresh value per thread into a fresh texture.
 *     It cannot read and write the same texture, so accumulating means
 *     ping-ponging two full images and copying the running total through the
 *     kernel every pass: 8 passes, 8 image-sized reads and 8 image-sized
 *     writes that exist only to move a number that never left the pixel.
 *   - A WGSL kernel writes `acc[i] = acc[i] + L`. One buffer, in place, and the
 *     read-modify-write is free because the invocation that reads a pixel is the
 *     only one that writes it.
 *
 * Same rays, same arithmetic, same image. The gap between those two cells is the
 * price of "one output per kernel", measured rather than asserted, and this is
 * the workload where that price is most obviously worth knowing.
 *
 * The tracer itself is deliberately the plain one: brute force, no light
 * sampling, no Russian roulette, no importance sampling beyond a cosine lobe. It
 * is monochrome — surfaces have a scalar reflectance — because a gpu.js kernel
 * returns one number per thread and a three-channel image would need three
 * kernels or a packed return type, which would change the memory traffic of one
 * column relative to the others for no benefit. A grey path tracer runs exactly
 * the same intersection and sampling code as a colour one.
 *
 * ── WHY NOT Math.random, AND WHY NOT sin/cos ────────────────────────────────
 *
 * Randomness is what makes a path tracer work and what makes it hard to put in
 * a benchmark table. Math.random cannot be used, though not for want of
 * support — gpu.js 2.21 implements it on WebGPU too. The problem is that if
 * every column drew its own randomness the six of them would produce six
 * different — equally valid — images, and the checksum could no longer tell a
 * broken kernel from a differently-seeded one. So every random number here is a
 * pure function of (pixel, pass): the same generator as the monte-carlo row,
 * built out of multiply/add/floor on integers below 2^24 so that fp32 carries
 * it exactly and all six columns draw bit-identical numbers. Its independence
 * testing is written up in full in monte-carlo.js and is not repeated here; the
 * short version is that streams are seeded from the middle lanes of the counter
 * squared, advanced by a per-stream odd Weyl stride, and an exhaustive hunt over
 * all 2^22 streams finds no two that share even their first 12 draws.
 *
 * The direction sampling is trig-free on purpose, and that is not a
 * micro-optimisation. The textbook cosine-weighted hemisphere sample is
 * (√u·cos 2πv, √u·sin 2πv, √(1-u)), and WGSL only guarantees `sin` and `cos` to
 * an ABSOLUTE error of 2^-11 — about 5e-4. A bounce direction wrong in the
 * fourth decimal is a different bounce direction: it lands somewhere else on the
 * sphere, collects different light, and does so systematically across every
 * sample rather than averaging out. That is a checksum error of the same order
 * as the runner's entire 1e-4 tolerance, and the row would report WRONG for a
 * reason that is not a bug.
 *
 * So the disk sample is drawn by REJECTION instead: take a point in [-1,1]²,
 * keep it if it is inside the unit circle, retry up to eight times. A uniform
 * point on the disk lifted to the hemisphere is exactly the cosine-weighted
 * distribution — this is not an approximation of the trig version, it is the
 * same distribution obtained a different way — and it uses nothing but multiply,
 * compare and one `sqrt`, which WGSL does pin to 2.5 ULP. Acceptance is π/4, so
 * it costs 1.27 tries on average, and the chance of exhausting all eight is
 * 4.5e-6 (those fall back to the surface normal, which is a legal direction).
 *
 * ── HOW MUCH fp32 COSTS ─────────────────────────────────────────────────────
 *
 * The remaining precision risk is real and worth stating plainly: a ray that
 * passes within an ulp of a sphere's silhouette can hit in fp64 and miss in
 * fp32, and a miss is worth a whole skyful of radiance rather than a slightly
 * different one. That cannot be designed away — it is what a silhouette is.
 * So it was measured. Re-running the oracle with Math.fround on every operation
 * — which is what a GPU does — moves the checksum by 5.0e-7 relative, two
 * hundred times inside the runner's tolerance. Exactly 4 pixels out of 262 144
 * move by more than 5%: those are the silhouette flips, one sample out of 32
 * landing on a sphere instead of the sky. They are real, they are permanent, and
 * they are four pixels.
 *
 * All scene and camera constants are rounded to fp32 where they are defined, so
 * that is the only source of disagreement between the columns: no column starts
 * from a slightly different sphere.
 */

const W = 512;
const H = 512;
// 8 accumulation passes of 4 samples each. The pass count is what this row is
// about, so it is the one that is not allowed to collapse to 1; the samples
// inside a pass are what keeps the per-pass RNG seeding from dominating.
const PASSES = 8;
const PER_PASS = 4;
const BOUNCES = 3;
const TRIES = 8; // rejection attempts per hemisphere sample

const fp32 = Math.fround;

// ── Camera ──────────────────────────────────────────────────────────────────
// Everything rounded to fp32 at definition, so the oracle and the shaders are
// looking at exactly the same camera rather than one that differs in the last
// bit. right = (1,0,0) exactly, because the camera only tilts in the y-z plane.
const CAM_X = fp32(0);
const CAM_Y = fp32(1.15);
const CAM_Z = fp32(4);
const _fl = Math.hypot(0 - CAM_X, 0.72 - CAM_Y, 0 - CAM_Z);
const FWD_X = fp32((0 - CAM_X) / _fl);
const FWD_Y = fp32((0.72 - CAM_Y) / _fl);
const FWD_Z = fp32((0 - CAM_Z) / _fl);
const UP_X = fp32(0);
const UP_Y = fp32(-FWD_Z); // cross((1,0,0), forward)
const UP_Z = fp32(FWD_Y);
const TAN_FOV = fp32(0.42);

// ── Scene: a plane at y = 0 and two spheres, all with scalar reflectance ─────
const SA_X = fp32(-0.95), SA_Y = fp32(0.9), SA_Z = fp32(-0.15), SA_R = fp32(0.9);
const SB_X = fp32(0.95), SB_Y = fp32(0.55), SB_Z = fp32(0.8), SB_R = fp32(0.55);
const SA_R2 = fp32(SA_R * SA_R), SB_R2 = fp32(SB_R * SB_R);
// Reciprocals, not divisions: every column must scale the hit offset by the
// identical number, and a division would be evaluated in fp64 by the oracle.
const SA_RINV = fp32(1 / SA_R), SB_RINV = fp32(1 / SB_R);
const SA_ALB = fp32(0.8), SB_ALB = fp32(0.32), PLANE_ALB = fp32(0.58);

// Sky: ambient + a gradient with height + a broad sun lobe. The sun is a
// dot-product raised to the 8th by three squarings — no `pow`, which is another
// loosely-specified builtin, and a wide lobe rather than a point light so the
// image converges without fireflies.
const SUN_X = fp32(0.3612), SUN_Y = fp32(0.8028), SUN_Z = fp32(0.4737);
const SKY_AMBIENT = fp32(0.18);
const SKY_GRADIENT = fp32(0.42);
const SUN_AMPLITUDE = fp32(3);

const T_MIN = fp32(1e-4); // ignore hits at the ray origin
// 1e9 and not 1e30: gpu.js appends '.0' to any integer-valued number it emits
// into GLSL, and JS renders anything from 1e21 up in exponential form, so 1e30
// became the literal `1e+30.0` and neither WebGL backend would compile the
// shader (gpujs/gpu.js#864). Any sentinel below 1e21 renders as digits and is
// safe; 1e9 is nine orders of magnitude past the far wall of a unit-scale
// scene, so nothing that is actually hit can reach it.
const T_MISS = fp32(1e9);
const OFFSET = fp32(1e-3); // push the next origin off the surface

// ── The generator (see monte-carlo.js for the full write-up and its testing) ─
// 24-bit LCG in two 12-bit lanes, rotated 12 bits a round, with a per-stream
// odd Weyl stride as the increment. Nothing it computes exceeds 2^24, so fp32
// carries every value exactly and every backend draws the same numbers.
const MUL = 1133;
const LANE = 4096;
const INV_LANE = 1 / 4096;
const WEYL = 8388608;
const INV_WEYL = 1 / 8388608;
const WARMUP = 4;

export default {
  id: 'path-trace',
  name: 'Progressive path tracing',
  params: `${W} × ${H} · ${PASSES} × ${PER_PASS} spp · ${BOUNCES} bounces, fp32`,
  tag: 'accumulation',
  group: 'render',
  size: { w: W, h: H, passes: PASSES, per: PER_PASS },

  // No make(). The scene is six constants and the randomness is computed, so
  // there is nothing to upload — which is the point: this row is arithmetic and
  // an accumulator, not a bandwidth test.

  /**
   * The oracle, and a fair baseline.
   *
   * The pass loop is on the OUTSIDE, exactly as the GPU columns run it. A JS
   * renderer left to itself would nest the samples inside the pixel and keep the
   * accumulator in a register, which is faster — but then this column would be
   * doing 8 image-sized reads and writes fewer than the others, and the row
   * would be comparing two different amounts of memory traffic. Progressive is
   * what every column does here. It costs 16 MB of accumulator traffic against
   * 25 million rays, so it is not what makes this column slow; it is just what
   * makes it the same.
   *
   * Everything else is written the way you would write it: flat typed array,
   * the generator state in three locals, the ray in scalars rather than objects,
   * and the two sphere tests unrolled. An object-per-ray version is four times
   * slower and would hand every GPU column a speed-up it had not earned.
   */
  js({ w, h, passes, per }) {
    const acc = new Float32Array(w * h);

    for (let pass = 0; pass < passes; pass++) {
      for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
          const idx = py * w + px;

          // Seed the stream for this (pixel, pass) from the middle lanes of the
          // counter squared; see monte-carlo.js for why the low lanes are wrong.
          const ctr = idx * passes + pass;
          const ch = Math.floor(ctr * INV_LANE);
          const cl = ctr - ch * LANE;
          const q0 = cl * cl;
          const k0 = Math.floor(q0 * INV_LANE);
          const w0 = q0 - k0 * LANE;
          const q1 = 2 * ch * cl + k0;
          const k1 = Math.floor(q1 * INV_LANE);
          const w1 = q1 - k1 * LANE;
          const q2 = ch * ch + k1;
          const w2 = q2 - Math.floor(q2 * INV_LANE) * LANE;

          let hi = w2;
          let lo = w1;
          const qq = w2 * LANE + w0;
          let weyl = qq - WEYL * Math.floor(qq * INV_WEYL);
          const stride = 2 * ctr + 1;
          for (let i = 0; i < WARMUP; i++) {
            weyl += stride;
            if (weyl >= WEYL) weyl -= WEYL;
            const b = MUL * lo + weyl;
            const bh = Math.floor(b * INV_LANE);
            const tp = MUL * hi + bh;
            hi = b - bh * LANE;
            lo = tp - LANE * Math.floor(tp * INV_LANE);
          }

          let total = 0;
          for (let s = 0; s < per; s++) {
            // Jitter, one draw each. `lo` after a round IS the draw.
            weyl += stride;
            if (weyl >= WEYL) weyl -= WEYL;
            let b = MUL * lo + weyl;
            let bh = Math.floor(b * INV_LANE);
            let tp = MUL * hi + bh;
            hi = b - bh * LANE;
            lo = tp - LANE * Math.floor(tp * INV_LANE);
            const jx = (lo + 0.5) * INV_LANE;

            weyl += stride;
            if (weyl >= WEYL) weyl -= WEYL;
            b = MUL * lo + weyl;
            bh = Math.floor(b * INV_LANE);
            tp = MUL * hi + bh;
            hi = b - bh * LANE;
            lo = tp - LANE * Math.floor(tp * INV_LANE);
            const jy = (lo + 0.5) * INV_LANE;

            const sx = ((px + jx) / w * 2 - 1) * TAN_FOV;
            const sy = (1 - (py + jy) / h * 2) * TAN_FOV;
            let dx = FWD_X + sx + sy * UP_X; // right = (1, 0, 0)
            let dy = FWD_Y + sy * UP_Y;
            let dz = FWD_Z + sy * UP_Z;
            const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz);
            dx *= inv;
            dy *= inv;
            dz *= inv;

            let ox = CAM_X;
            let oy = CAM_Y;
            let oz = CAM_Z;
            let tput = 1;

            for (let bounce = 0; bounce < BOUNCES; bounce++) {
              let best = T_MISS;
              let nx = 0;
              let ny = 0;
              let nz = 0;
              let alb = 0;

              // Ground plane. Only tested for downward rays, which is both
              // correct (the camera and every bounce origin are above it) and
              // the reason there is no division by zero here.
              if (dy < -T_MIN) {
                const t = -oy / dy;
                if (t > T_MIN) {
                  best = t;
                  nx = 0;
                  ny = 1;
                  nz = 0;
                  alb = PLANE_ALB;
                }
              }

              let ocx = ox - SA_X;
              let ocy = oy - SA_Y;
              let ocz = oz - SA_Z;
              let bq = ocx * dx + ocy * dy + ocz * dz;
              let cq = ocx * ocx + ocy * ocy + ocz * ocz - SA_R2;
              let disc = bq * bq - cq;
              if (disc > 0) {
                const sd = Math.sqrt(disc);
                let t = -bq - sd;
                if (t <= T_MIN) t = -bq + sd;
                if (t > T_MIN && t < best) {
                  best = t;
                  nx = (ox + t * dx - SA_X) * SA_RINV;
                  ny = (oy + t * dy - SA_Y) * SA_RINV;
                  nz = (oz + t * dz - SA_Z) * SA_RINV;
                  alb = SA_ALB;
                }
              }

              ocx = ox - SB_X;
              ocy = oy - SB_Y;
              ocz = oz - SB_Z;
              bq = ocx * dx + ocy * dy + ocz * dz;
              cq = ocx * ocx + ocy * ocy + ocz * ocz - SB_R2;
              disc = bq * bq - cq;
              if (disc > 0) {
                const sd = Math.sqrt(disc);
                let t = -bq - sd;
                if (t <= T_MIN) t = -bq + sd;
                if (t > T_MIN && t < best) {
                  best = t;
                  nx = (ox + t * dx - SB_X) * SB_RINV;
                  ny = (oy + t * dy - SB_Y) * SB_RINV;
                  nz = (oz + t * dz - SB_Z) * SB_RINV;
                  alb = SB_ALB;
                }
              }

              if (best === T_MISS) {
                // Escaped. The sky is the only light in the scene.
                let sun = dx * SUN_X + dy * SUN_Y + dz * SUN_Z;
                if (sun < 0) sun = 0;
                const s2 = sun * sun;
                const s4 = s2 * s2;
                total += tput * (SKY_AMBIENT + SKY_GRADIENT * (dy > 0 ? dy : 0) + SUN_AMPLITUDE * s4 * s4);
                break;
              }

              const hx = ox + best * dx;
              const hy = oy + best * dy;
              const hz = oz + best * dz;
              tput *= alb;

              // Cosine-weighted hemisphere by rejection on the unit disk. See
              // the header: this is the trig-free route to the same
              // distribution, and the reason this row is checkable at all.
              let ddx = 0;
              let ddy = 0;
              for (let k = 0; k < TRIES; k++) {
                weyl += stride;
                if (weyl >= WEYL) weyl -= WEYL;
                b = MUL * lo + weyl;
                bh = Math.floor(b * INV_LANE);
                tp = MUL * hi + bh;
                hi = b - bh * LANE;
                lo = tp - LANE * Math.floor(tp * INV_LANE);
                const u = (lo + 0.5) * INV_LANE * 2 - 1;

                weyl += stride;
                if (weyl >= WEYL) weyl -= WEYL;
                b = MUL * lo + weyl;
                bh = Math.floor(b * INV_LANE);
                tp = MUL * hi + bh;
                hi = b - bh * LANE;
                lo = tp - LANE * Math.floor(tp * INV_LANE);
                const v = (lo + 0.5) * INV_LANE * 2 - 1;

                if (u * u + v * v < 1) {
                  ddx = u;
                  ddy = v;
                  break;
                }
              }
              const rad = 1 - ddx * ddx - ddy * ddy;
              const ddz = Math.sqrt(rad > 0 ? rad : 0);

              // Branchless orthonormal basis around the normal (Duff et al.).
              // sign is chosen so that sg + nz is never zero.
              const sg = nz >= 0 ? 1 : -1;
              const a = -1 / (sg + nz);
              const bxy = nx * ny * a;
              const t1x = 1 + sg * nx * nx * a;
              const t1y = sg * bxy;
              const t1z = -sg * nx;
              const t2x = bxy;
              const t2y = sg + ny * ny * a;
              const t2z = -ny;

              dx = ddx * t1x + ddy * t2x + ddz * nx;
              dy = ddx * t1y + ddy * t2y + ddz * ny;
              dz = ddx * t1z + ddy * t2z + ddz * nz;
              ox = hx + OFFSET * nx;
              oy = hy + OFFSET * ny;
              oz = hz + OFFSET * nz;
            }
          }
          acc[idx] += total;
        }
      }
    }
    return acc;
  },

  gpujs(gpu, { w, h, passes, per }) {
    const constants = {
      wid: w,
      hgt: h,
      spp: per,
      npass: passes,
      tanf: TAN_FOV,
      camx: CAM_X, camy: CAM_Y, camz: CAM_Z,
      fwdx: FWD_X, fwdy: FWD_Y, fwdz: FWD_Z,
      upx: UP_X, upy: UP_Y, upz: UP_Z,
      sax: SA_X, say: SA_Y, saz: SA_Z, sar2: SA_R2, sarinv: SA_RINV, salb: SA_ALB,
      sbx: SB_X, sby: SB_Y, sbz: SB_Z, sbr2: SB_R2, sbrinv: SB_RINV, sbalb: SB_ALB,
      palb: PLANE_ALB,
      sunx: SUN_X, suny: SUN_Y, sunz: SUN_Z,
      skyamb: SKY_AMBIENT, skygrad: SKY_GRADIENT, sunamp: SUN_AMPLITUDE,
      tmin: T_MIN, tmiss: T_MISS, offs: OFFSET, tries: TRIES, bounces: BOUNCES,
      gmul: MUL, glane: LANE, ginv: INV_LANE, gweyl: WEYL, gwinv: INV_WEYL, gwarm: WARMUP,
    };

    /**
     * One accumulation pass: read the running image, add this pass's samples,
     * return the new image. That copy of `prev` through the kernel is the thing
     * the WGSL column does not have to do, and it is why there are two of these.
     *
     * Every constant is read through `this.constants.` at its use site. A local
     * named after one of the kernel's own constants collides with the generated
     * `constants_<name>` on the CPU backend and the kernel fails to build.
     */
    const bodyFn = function (prev, pass) {
      const idx = this.thread.y * this.constants.wid + this.thread.x;
      const ctr = idx * this.constants.npass + pass;

      const ch = Math.floor(ctr * this.constants.ginv);
      const cl = ctr - ch * this.constants.glane;
      const q0 = cl * cl;
      const k0 = Math.floor(q0 * this.constants.ginv);
      const g0 = q0 - k0 * this.constants.glane;
      const q1 = 2 * ch * cl + k0;
      const k1 = Math.floor(q1 * this.constants.ginv);
      const g1 = q1 - k1 * this.constants.glane;
      const q2 = ch * ch + k1;
      const g2 = q2 - Math.floor(q2 * this.constants.ginv) * this.constants.glane;

      let hi = g2;
      let lo = g1;
      const qq = g2 * this.constants.glane + g0;
      let wy = qq - this.constants.gweyl * Math.floor(qq * this.constants.gwinv);
      const stride = 2 * ctr + 1;

      for (let i = 0; i < this.constants.gwarm; i++) {
        wy = wy + stride;
        if (wy >= this.constants.gweyl) wy = wy - this.constants.gweyl;
        const bb = this.constants.gmul * lo + wy;
        const bh = Math.floor(bb * this.constants.ginv);
        const tp = this.constants.gmul * hi + bh;
        hi = bb - bh * this.constants.glane;
        lo = tp - this.constants.glane * Math.floor(tp * this.constants.ginv);
      }

      let total = 0;
      for (let s = 0; s < this.constants.spp; s++) {
        wy = wy + stride;
        if (wy >= this.constants.gweyl) wy = wy - this.constants.gweyl;
        let bb = this.constants.gmul * lo + wy;
        let bh = Math.floor(bb * this.constants.ginv);
        let tp = this.constants.gmul * hi + bh;
        hi = bb - bh * this.constants.glane;
        lo = tp - this.constants.glane * Math.floor(tp * this.constants.ginv);
        const jx = (lo + 0.5) * this.constants.ginv;

        wy = wy + stride;
        if (wy >= this.constants.gweyl) wy = wy - this.constants.gweyl;
        bb = this.constants.gmul * lo + wy;
        bh = Math.floor(bb * this.constants.ginv);
        tp = this.constants.gmul * hi + bh;
        hi = bb - bh * this.constants.glane;
        lo = tp - this.constants.glane * Math.floor(tp * this.constants.ginv);
        const jy = (lo + 0.5) * this.constants.ginv;

        const sx = ((this.thread.x + jx) / this.constants.wid * 2 - 1) * this.constants.tanf;
        const sy = (1 - (this.thread.y + jy) / this.constants.hgt * 2) * this.constants.tanf;
        let dx = this.constants.fwdx + sx + sy * this.constants.upx;
        let dy = this.constants.fwdy + sy * this.constants.upy;
        let dz = this.constants.fwdz + sy * this.constants.upz;
        const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz);
        dx = dx * inv;
        dy = dy * inv;
        dz = dz * inv;

        let ox = this.constants.camx;
        let oy = this.constants.camy;
        let oz = this.constants.camz;
        let tput = 1;

        for (let bounce = 0; bounce < this.constants.bounces; bounce++) {
          let best = this.constants.tmiss;
          let nx = 0;
          let ny = 0;
          let nz = 0;
          let alb = 0;

          if (dy < -this.constants.tmin) {
            const tpl = -oy / dy;
            if (tpl > this.constants.tmin) {
              best = tpl;
              nx = 0;
              ny = 1;
              nz = 0;
              alb = this.constants.palb;
            }
          }

          let ocx = ox - this.constants.sax;
          let ocy = oy - this.constants.say;
          let ocz = oz - this.constants.saz;
          let bq = ocx * dx + ocy * dy + ocz * dz;
          let cq = ocx * ocx + ocy * ocy + ocz * ocz - this.constants.sar2;
          let disc = bq * bq - cq;
          if (disc > 0) {
            const sd = Math.sqrt(disc);
            let ta = -bq - sd;
            if (ta <= this.constants.tmin) ta = -bq + sd;
            if (ta > this.constants.tmin && ta < best) {
              best = ta;
              nx = (ox + ta * dx - this.constants.sax) * this.constants.sarinv;
              ny = (oy + ta * dy - this.constants.say) * this.constants.sarinv;
              nz = (oz + ta * dz - this.constants.saz) * this.constants.sarinv;
              alb = this.constants.salb;
            }
          }

          ocx = ox - this.constants.sbx;
          ocy = oy - this.constants.sby;
          ocz = oz - this.constants.sbz;
          bq = ocx * dx + ocy * dy + ocz * dz;
          cq = ocx * ocx + ocy * ocy + ocz * ocz - this.constants.sbr2;
          disc = bq * bq - cq;
          if (disc > 0) {
            const sd = Math.sqrt(disc);
            let tb = -bq - sd;
            if (tb <= this.constants.tmin) tb = -bq + sd;
            if (tb > this.constants.tmin && tb < best) {
              best = tb;
              nx = (ox + tb * dx - this.constants.sbx) * this.constants.sbrinv;
              ny = (oy + tb * dy - this.constants.sby) * this.constants.sbrinv;
              nz = (oz + tb * dz - this.constants.sbz) * this.constants.sbrinv;
              alb = this.constants.sbalb;
            }
          }

          if (best === this.constants.tmiss) {
            let sun = dx * this.constants.sunx + dy * this.constants.suny + dz * this.constants.sunz;
            if (sun < 0) sun = 0;
            const s2 = sun * sun;
            const s4 = s2 * s2;
            let up = dy;
            if (up < 0) up = 0;
            total = total + tput * (this.constants.skyamb + this.constants.skygrad * up + this.constants.sunamp * s4 * s4);
            break;
          }

          const hx = ox + best * dx;
          const hy = oy + best * dy;
          const hz = oz + best * dz;
          tput = tput * alb;

          let ddx = 0;
          let ddy = 0;
          for (let k = 0; k < this.constants.tries; k++) {
            wy = wy + stride;
            if (wy >= this.constants.gweyl) wy = wy - this.constants.gweyl;
            bb = this.constants.gmul * lo + wy;
            bh = Math.floor(bb * this.constants.ginv);
            tp = this.constants.gmul * hi + bh;
            hi = bb - bh * this.constants.glane;
            lo = tp - this.constants.glane * Math.floor(tp * this.constants.ginv);
            const u = (lo + 0.5) * this.constants.ginv * 2 - 1;

            wy = wy + stride;
            if (wy >= this.constants.gweyl) wy = wy - this.constants.gweyl;
            bb = this.constants.gmul * lo + wy;
            bh = Math.floor(bb * this.constants.ginv);
            tp = this.constants.gmul * hi + bh;
            hi = bb - bh * this.constants.glane;
            lo = tp - this.constants.glane * Math.floor(tp * this.constants.ginv);
            const v = (lo + 0.5) * this.constants.ginv * 2 - 1;

            if (u * u + v * v < 1) {
              ddx = u;
              ddy = v;
              break;
            }
          }
          let rad = 1 - ddx * ddx - ddy * ddy;
          if (rad < 0) rad = 0;
          const ddz = Math.sqrt(rad);

          let sg = 1;
          if (nz < 0) sg = -1;
          const aa = -1 / (sg + nz);
          const bxy = nx * ny * aa;
          const t1x = 1 + sg * nx * nx * aa;
          const t1y = sg * bxy;
          const t1z = -sg * nx;
          const t2x = bxy;
          const t2y = sg + ny * ny * aa;
          const t2z = -ny;

          dx = ddx * t1x + ddy * t2x + ddz * nx;
          dy = ddx * t1y + ddy * t2y + ddz * ny;
          dz = ddx * t1z + ddy * t2z + ddz * nz;
          ox = hx + this.constants.offs * nx;
          oy = hy + this.constants.offs * ny;
          oz = hz + this.constants.offs * nz;
        }
      }
      return prev[this.thread.y][this.thread.x] + total;
    };

    // Two of them, ping-ponged. A pipelined mutable kernel owns one output
    // texture and reuses it, so a single kernel would be asked to write the
    // texture it is reading.
    const mk = () =>
      gpu
        .createKernel(bodyFn)
        .setConstants(constants)
        .setPipeline(true)
        // Radiance runs past 1, well outside what gpu.js's default 'unsigned'
        // RGBA8 encoding can carry, and the generator needs exact integers up to
        // 2^24. 'single' asks for float32 and fails loudly if it cannot have it.
        .setPrecision('single')
        .setOutput([w, h]);
    const even = mk();
    const odd = mk();

    // Produces the zeroed accumulator. Its real job is to keep the pass
    // kernels' argument type constant: a plain array on pass 0 and a Texture
    // afterwards would make gpu.js recompile inside the timed region.
    const clear = gpu
      .createKernel(function () {
        return 0;
      })
      .setPipeline(true)
      .setPrecision('single')
      .setOutput([w, h]);

    return {
      async run() {
        let img = await clear();
        for (let pass = 0; pass < passes; pass++) {
          img = await (pass % 2 === 0 ? even : odd)(img, pass);
        }
        // The read-back. Until this resolves the passes are only queued, and
        // the row would be reporting the cost of filling a command buffer.
        return img.toArray ? await img.toArray() : img;
      },
      backend: () => even.kernel && even.kernel.constructor.mode,
      destroy() {
        [even, odd, clear].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WGSL, with nothing borrowed from gpu.js.
   *
   * Two differences from the column to its left, and both are the runtime's
   * doing rather than the algorithm's:
   *
   *   - the accumulator is updated IN PLACE. `acc[i] = acc[i] + total` needs no
   *     second image and no copy, because the invocation that reads a pixel is
   *     the only one that writes it. That is the whole point of this row.
   *   - all 8 passes go into ONE compute pass. Dispatches inside a pass are
   *     ordered and each one's writes are visible to the next, so there is one
   *     submit for the whole render rather than one per pass. The pass index
   *     arrives through a pre-filled uniform read at a dynamic offset; rewriting
   *     a uniform between dispatches would force a submit each time and the row
   *     would be measuring the queue.
   *
   * The tracing is line-for-line the gpu.js kernel — same rejection loop, same
   * basis, same intersection order, so the two cells differ by the runtime and
   * nothing else.
   */
  async webgpu(device, { w, h, passes, per }) {
    const cells = w * h;
    const bytes = cells * 4;

    // A zeroed buffer to reset from, so a run costs a device-side copy rather
    // than a 1 MB upload inside the timed region.
    const zero = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
    new Float32Array(zero.getMappedRange()).fill(0);
    zero.unmap();
    const acc = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const STRIDE = 256; // minUniformBufferOffsetAlignment
    const plan = new Uint32Array((passes * STRIDE) / 4);
    for (let p = 0; p < passes; p++) plan[(p * STRIDE) / 4] = p;
    const planBuf = device.createBuffer({
      size: plan.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(planBuf, 0, plan);

    // WGSL wants float literals to look like floats. `${0}` would emit an
    // abstract int and a strict compiler is within its rights to reject
    // `vec3<f32>(0, 1.15, 4)`, so every constant goes through this.
    const wf = v => (Number.isInteger(v) ? v.toFixed(1) : String(v));

    const TILE = 8;
    const module = device.createShaderModule({
      code: `
struct Pass { index: u32 };
@group(0) @binding(0) var<storage, read_write> acc: array<f32>;
@group(0) @binding(1) var<uniform> p: Pass;

const WIDTH: u32 = ${w}u;
const HEIGHT: u32 = ${h}u;
const NPASS: f32 = ${wf(passes)};
const SPP: u32 = ${per}u;

const MUL: f32 = ${wf(MUL)};
const LANE: f32 = ${wf(LANE)};
const INV_LANE: f32 = ${wf(INV_LANE)};
const WEYL: f32 = ${wf(WEYL)};
const INV_WEYL: f32 = ${wf(INV_WEYL)};

const CAM = vec3<f32>(${wf(CAM_X)}, ${wf(CAM_Y)}, ${wf(CAM_Z)});
const FWD = vec3<f32>(${wf(FWD_X)}, ${wf(FWD_Y)}, ${wf(FWD_Z)});
const UPV = vec3<f32>(${wf(UP_X)}, ${wf(UP_Y)}, ${wf(UP_Z)});
const TAN_FOV: f32 = ${wf(TAN_FOV)};
const SA = vec3<f32>(${wf(SA_X)}, ${wf(SA_Y)}, ${wf(SA_Z)});
const SB = vec3<f32>(${wf(SB_X)}, ${wf(SB_Y)}, ${wf(SB_Z)});
const SUN = vec3<f32>(${wf(SUN_X)}, ${wf(SUN_Y)}, ${wf(SUN_Z)});
const SA_R2: f32 = ${wf(SA_R2)};
const SB_R2: f32 = ${wf(SB_R2)};
const SA_RINV: f32 = ${wf(SA_RINV)};
const SB_RINV: f32 = ${wf(SB_RINV)};
const SA_ALB: f32 = ${wf(SA_ALB)};
const SB_ALB: f32 = ${wf(SB_ALB)};
const PLANE_ALB: f32 = ${wf(PLANE_ALB)};
const SKY_AMBIENT: f32 = ${wf(SKY_AMBIENT)};
const SKY_GRADIENT: f32 = ${wf(SKY_GRADIENT)};
const SUN_AMPLITUDE: f32 = ${wf(SUN_AMPLITUDE)};
const T_MIN: f32 = ${wf(T_MIN)};
const T_MISS: f32 = ${wf(T_MISS)};
const OFFSET: f32 = ${wf(OFFSET)};

// Generator state, threaded through by hand so the arithmetic is identical to
// the other columns rather than merely equivalent.
var<private> gHi: f32;
var<private> gLo: f32;
var<private> gWeyl: f32;
var<private> gStride: f32;

fn draw() -> f32 {
  gWeyl = gWeyl + gStride;
  gWeyl = select(gWeyl, gWeyl - WEYL, gWeyl >= WEYL);
  let b = MUL * gLo + gWeyl;
  let bh = floor(b * INV_LANE);
  let tp = MUL * gHi + bh;
  gHi = b - bh * LANE;
  gLo = tp - LANE * floor(tp * INV_LANE);
  return gLo;
}

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= WIDTH || gid.y >= HEIGHT) { return; }
  let pix = gid.y * WIDTH + gid.x;
  let ctr = f32(pix) * NPASS + f32(p.index);

  // Seed: middle lanes of the full 44-bit ctr².
  let ch = floor(ctr * INV_LANE);
  let cl = ctr - ch * LANE;
  let q0 = cl * cl;
  let k0 = floor(q0 * INV_LANE);
  let g0 = q0 - k0 * LANE;
  let q1 = 2.0 * ch * cl + k0;
  let k1 = floor(q1 * INV_LANE);
  let g1 = q1 - k1 * LANE;
  let q2 = ch * ch + k1;
  let g2 = q2 - floor(q2 * INV_LANE) * LANE;

  gHi = g2;
  gLo = g1;
  let qq = g2 * LANE + g0;
  gWeyl = qq - WEYL * floor(qq * INV_WEYL);
  gStride = 2.0 * ctr + 1.0;
  for (var i: u32 = 0u; i < ${WARMUP}u; i = i + 1u) { _ = draw(); }

  var total = 0.0;
  for (var s: u32 = 0u; s < SPP; s = s + 1u) {
    let jx = (draw() + 0.5) * INV_LANE;
    let jy = (draw() + 0.5) * INV_LANE;
    let sx = ((f32(gid.x) + jx) / f32(WIDTH) * 2.0 - 1.0) * TAN_FOV;
    let sy = (1.0 - (f32(gid.y) + jy) / f32(HEIGHT) * 2.0) * TAN_FOV;
    var d = vec3<f32>(FWD.x + sx + sy * UPV.x, FWD.y + sy * UPV.y, FWD.z + sy * UPV.z);
    d = d * (1.0 / sqrt(d.x * d.x + d.y * d.y + d.z * d.z));

    var o = CAM;
    var tput = 1.0;

    for (var bounce: u32 = 0u; bounce < ${BOUNCES}u; bounce = bounce + 1u) {
      var best = T_MISS;
      var n = vec3<f32>(0.0, 0.0, 0.0);
      var alb = 0.0;

      if (d.y < -T_MIN) {
        let tpl = -o.y / d.y;
        if (tpl > T_MIN) { best = tpl; n = vec3<f32>(0.0, 1.0, 0.0); alb = PLANE_ALB; }
      }

      var oc = o - SA;
      var bq = dot(oc, d);
      var cq = dot(oc, oc) - SA_R2;
      var disc = bq * bq - cq;
      if (disc > 0.0) {
        let sd = sqrt(disc);
        var ta = -bq - sd;
        if (ta <= T_MIN) { ta = -bq + sd; }
        if (ta > T_MIN && ta < best) { best = ta; n = (o + ta * d - SA) * SA_RINV; alb = SA_ALB; }
      }

      oc = o - SB;
      bq = dot(oc, d);
      cq = dot(oc, oc) - SB_R2;
      disc = bq * bq - cq;
      if (disc > 0.0) {
        let sd = sqrt(disc);
        var tb = -bq - sd;
        if (tb <= T_MIN) { tb = -bq + sd; }
        if (tb > T_MIN && tb < best) { best = tb; n = (o + tb * d - SB) * SB_RINV; alb = SB_ALB; }
      }

      if (best == T_MISS) {
        let sun = max(dot(d, SUN), 0.0);
        let s2 = sun * sun;
        let s4 = s2 * s2;
        total = total + tput * (SKY_AMBIENT + SKY_GRADIENT * max(d.y, 0.0) + SUN_AMPLITUDE * s4 * s4);
        break;
      }

      let hit = o + best * d;
      tput = tput * alb;

      // Cosine-weighted hemisphere by rejection — no sin/cos anywhere, because
      // WGSL only pins those to 2^-11 absolute and a bounce direction wrong in
      // the fourth decimal is a different bounce. See the header.
      var dd = vec2<f32>(0.0, 0.0);
      for (var k: u32 = 0u; k < ${TRIES}u; k = k + 1u) {
        let u = (draw() + 0.5) * INV_LANE * 2.0 - 1.0;
        let v = (draw() + 0.5) * INV_LANE * 2.0 - 1.0;
        if (u * u + v * v < 1.0) { dd = vec2<f32>(u, v); break; }
      }
      let ddz = sqrt(max(1.0 - dd.x * dd.x - dd.y * dd.y, 0.0));

      let sg = select(-1.0, 1.0, n.z >= 0.0);
      let a = -1.0 / (sg + n.z);
      let bxy = n.x * n.y * a;
      let t1 = vec3<f32>(1.0 + sg * n.x * n.x * a, sg * bxy, -sg * n.x);
      let t2 = vec3<f32>(bxy, sg + n.y * n.y * a, -n.y);

      d = dd.x * t1 + dd.y * t2 + ddz * n;
      o = hit + OFFSET * n;
    }
  }
  // In place. No ping-pong, no copy of the running image through the kernel.
  acc[pix] = acc[pix] + total;
}`,
    });

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: 'main' },
    });
    const bind = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: acc } },
        { binding: 1, resource: { buffer: planBuf, offset: 0, size: 16 } },
      ],
    });
    const gx = Math.ceil(w / TILE);
    const gy = Math.ceil(h / TILE);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(zero, 0, acc, 0, bytes);
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (let i = 0; i < passes; i++) {
          pass.setBindGroup(0, bind, [i * STRIDE]);
          pass.dispatchWorkgroups(gx, gy);
        }
        pass.end();
        enc.copyBufferToBuffer(acc, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [zero, acc, read, planBuf].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Every pixel, index-weighted, in row-major order whichever shape the backend
   * handed back — a gpu.js 2-D kernel returns an array of rows, the other two
   * return one flat buffer, and the weight has to line up or the row would
   * report WRONG for a formatting difference.
   *
   * The weight is what makes this a checksum rather than an average. A mean
   * radiance would be passed by a backend that rendered half the image and left
   * the rest at zero if it happened to be the brighter half; a positional weight
   * would not. Values are non-negative and bounded, and the total is divided by
   * both the pixel count and the sample count, so it lands near the mean pixel
   * value and a million-term sum stays where fp32 and fp64 agree. The measured
   * fp32-throughout spread on this checksum is 5.0e-7, two hundred times inside
   * the runner's 1e-4.
   */
  reduce(out, { w, h, passes, per }) {
    let acc = 0;
    let i = 0;
    if (ArrayBuffer.isView(out)) {
      for (; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    } else {
      for (let y = 0; y < out.length; y++) {
        const row = out[y];
        for (let x = 0; x < row.length; x++, i++) acc += row[x] * (1 + (i % 17));
      }
    }
    return acc / (w * h * passes * per);
  },
};
