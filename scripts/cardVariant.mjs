/**
 * scripts/cardVariant.mjs — the card's copy of a payoff task, and why it differs.
 *
 * Shared by capture-module-renders.mjs (the stills) and capture-module-gifs.mjs
 * (the animated ones), because both need the SAME rewrite: a GIF frame that
 * disagreed with the still would be a second source of truth for what a card
 * looks like.
 */
// ---- card resolution -------------------------------------------------------
//
// A lesson picks its output size FOR THE LESSON: 16x16 so a cellular automaton
// is legible cell by cell, 64x64 so a path tracer converges in a few seconds on
// a laptop. The card then shows that at roughly 300 CSS px — 600 device px on a
// phone — so a 64px render is blown up tenfold. Hence the mush.
//
// The obvious fix does not work, and it was measured rather than reasoned
// about: multiplying the `output:` literal alone made all eight renders it was
// tried on WRONG, not bigger. Resolution is baked into the kernel arithmetic —
// metaballs map pixels to world space with `(this.thread.x - 32) / 16`, the
// Julia set walks the complex plane in `step: 0.025`, the ripples measure
// radius from `63.5`. Four times the threads with the same constants is a 4x
// ZOOM: the fractal goes flat blue, the metaballs march off screen into black.
// Kernels that index an input (photo[y][x], a seeded grid) are worse off still,
// since task.inputs() built that input at the lesson's size inside the sandbox,
// where nothing here can reach it.
//
// So each entry below is a hand-checked rewrite that moves the constants WITH
// the resolution, keeping the framing identical and spending the extra threads
// on detail. Only modules that generate their picture from nothing can be done
// this way; everything driven by an input keeps its lesson size and falls back
// to a plain upscale on the card.
//
// A `from` that no longer appears is a hard error, not a skip: content drifts,
// and a silently-unapplied edit would re-capture the lesson-size picture at
// four times the resolution — which is the zoomed, broken one.
export const CARD_SCALE = {
  // same complex-plane window (xMin/yMin unchanged), quarter-size steps
  'escape-time-fractals': [
    ['output: [128, 128]', 'output: [512, 512]'],
    ['step: 0.025', 'step: 0.00625'],
  ],
  // centre, wavelength and falloff radius all scale with the grid
  'pixels-from-scratch': [
    ['output: [128, 128]', 'output: [512, 512]'],
    ['this.thread.x / 1 - 63.5', 'this.thread.x / 1 - 255.5'],
    ['this.thread.y / 1 - 63.5', 'this.thread.y / 1 - 255.5'],
    ['Math.cos(r * 0.35)', 'Math.cos(r * 0.0875)'],
    ['1 - r / 96', '1 - r / 384'],
  ],
  // the photo comes from cardInputs at 4x; the kernels just index it
  'colour-spaces': [[/output: \[64, 64\]/g, 'output: [256, 256]']],
  // pixel -> world mapping is the only place resolution appears
  'ray-marched-metaballs': [
    ['output: [64, 64]', 'output: [256, 256]'],
    ['(this.thread.x - 32) / 16', '(this.thread.x - 128) / 64'],
    ['(this.thread.y - 32) / 16', '(this.thread.y - 128) / 64'],
  ],
  // every kernel in the chain moves together — they share the accumulation
  // buffers — and the camera's normalised device coords divide by the width
  'path-tracing': [
    [/output: \[64, 64\]/g, 'output: [256, 256]'],
    ['((this.thread.x + seed) / 64)', '((this.thread.x + seed) / 256)'],
    ['((this.thread.y + seed) / 64)', '((this.thread.y + seed) / 256)'],
    ['for (let y = 0; y < 64; y++)', 'for (let y = 0; y < 256; y++)'],
    ['for (let x = 0; x < 64; x++)', 'for (let x = 0; x < 256; x++)'],
    ['sum / 4096', 'sum / 65536'],
  ],

  // the same 24 sites four times as wide (cardInputs), so the ladder needs two
  // more rungs — log2(512) = 9 — or the corners never hear about their seed.
  // Measured: 7 rungs at 512 leaves 7.6% of cells wrong, with red corner blocks.
  'jump-flooding': [
    [/output: \[128, 128\]/g, 'output: [512, 512]'],
    [/constants: \{ n: 128/g, 'constants: { n: 512'],
    ['for (let k = 64; k >= 1; k = k / 2)', 'for (let k = 256; k >= 1; k = k / 2)'],
    ['for (let y = 0; y < 128; y++)', 'for (let y = 0; y < 512; y++)'],
    ['for (let x = 0; x < 128; x++)', 'for (let x = 0; x < 512; x++)'],
    ["'of 16384 cells are wrong'", "'of 262144 cells are wrong'"],
  ],

  // The canvas was already 256px but the DATA was 64 frames x 128 bins painted
  // as 4x2 blocks, so the real detail was 64 columns wide. Time gets a finer
  // hop; frequency gets a zero-padded DFT — same window, so the time resolution
  // it costs nothing. A longer window would buy real bins by smearing time,
  // which is the trade the module itself teaches, and would change the picture.
  spectrograms: [
    ['const HOP = 64;', 'const HOP = 8;'],
    ['const BINS = 128;', 'const BINS = 512;'],
    [
      'const angle = (-2 * Math.PI * bin * t) / this.constants.win;',
      'const angle = (-2 * Math.PI * bin * t) / (4 * this.constants.win);',
    ],
    ['const frame = Math.floor(this.thread.x / 4);', 'const frame = this.thread.x;'],
    ['const bin = Math.floor(this.thread.y / 2);', 'const bin = this.thread.y;'],
    ['output: [256, 256],', 'output: [512, 512],'],
  ],

  // frames come from cardInputs at 4x, and both blur radii are SPATIAL: at 4x
  // a radius of 2 is not a blurred backdrop, it is a sharp one with speckle.
  'video-filters': [
    [/output: \[64, 64\]/g, 'output: [256, 256]'],
    [/last: 63/g, 'last: 255'],
    ['for (let dy = -1; dy <= 1; dy++)', 'for (let dy = -4; dy <= 4; dy++)'],
    ['for (let dx = -1; dx <= 1; dx++)', 'for (let dx = -4; dx <= 4; dx++)'],
    ['return sum / 9;', 'return sum / 81;'],
    ['for (let dy = -2; dy <= 2; dy++)', 'for (let dy = -8; dy <= 8; dy++)'],
    ['for (let dx = -2; dx <= 2; dx++)', 'for (let dx = -8; dx <= 8; dx++)'],
    ['const backR = sr / 25;', 'const backR = sr / 289;'],
    ['const backG = sg / 25;', 'const backG = sg / 289;'],
    ['const backB = sb / 25;', 'const backB = sb / 289;'],
  ],

  // cardInputs hands this a ROTATION field rather than the lesson's sliding
  // band (see cardRotationFrames): every direction at once, so the card is the
  // hue wheel the module is about. maxFlow is set from that field — measured
  // mean |flow| is 1.31 at r=100 and 3.5 at the corners — so the picture is
  // vivid away from the centre and the motionless centre stays white.
  'optical-flow': [
    ['output: [64, 64]', 'output: [256, 256]'],
    ['maxFlow: 1.5', 'maxFlow: 1.6'],
  ],

  // A diffusion lattice, so the rates are not decoration: du = D·dt/h², and
  // explicit Euler is stable only while du·dt <= 1/4 — twice the cells costs 4x
  // du, a quarter of dt and 4x the steps. Hence 2x and not 4x: 4x is 256x the
  // work and blows the 10 s run watchdog.
  'reaction-diffusion': [
    [/output: \[64, 64\]/g, 'output: [128, 128]'],
    [/size: 64,/g, 'size: 128,'],
    ['du: 0.2', 'du: 0.8'],
    ['dv: 0.1', 'dv: 0.4'],
    [/dt: 1 }/g, 'dt: 0.25 }'],
    // the step COUNT is the learner's dial now, so the card scales its default
    // and its ceiling rather than a loop literal — 4x the steps for a lattice
    // half the cell size, same as du/dv/dt above
    ['value: 200, step: 10', 'value: 800, step: 40'],
    ['max: 600', 'max: 2400'],
  ],

  // 96 is the LESSON's grid and the model's constants are measured in it, so
  // they move with it. Water advects one cell per step, so time scales too:
  // STEPS x2.67 and the per-step rates x0.375. 256 rather than 384 because the
  // pre-flight guard exempts exactly 65,536 threads and no more.
  'hydraulic-erosion': [
    ['const SIZE = 96;', 'const SIZE = 256;'],
    ['const STEPS = 200;', 'const STEPS = 540;'],
    ['const EVERY = 10;', 'const EVERY = 27;'],
    [
      'soft: 0.0002, rain: 0.00002, carry: 60, pickUp: erosion, settle: 0.05, maxCut: 0.004',
      'soft: 0.000028125, rain: 0.0000075, carry: 160, pickUp: erosion * 0.375, settle: 0.01875, maxCut: 0.0015',
    ],
    ['relief: 18', 'relief: 48'],
  ],

  // A 64x36 torus of famous Life shapes (cardInputs) instead of the lesson's
  // lone glider on 16x16 — the card is allowed to be a better picture than the
  // exercise. 16:9 rather than square because the catalogue crops to 16:9, and
  // a square world loses its top and bottom rows to that crop. The torus wrap
  // is written as literal 16s, so the world size lives in the kernel and has to
  // move with it — and the two axes now differ. The painter magnifies 8x,
  // nearest neighbour, so a cell is a hard-edged square, not a blurry pixel.
  'cellular-automata': [
    [/\+ dy \+ 16\) % 16/g, '+ dy + 36) % 36'],
    [/\+ dx \+ 16\) % 16/g, '+ dx + 64) % 64'],
    ['output: [16, 16] }', 'output: [64, 36] }'],
    [
      'const alive = cells[this.thread.y][this.thread.x];',
      'const alive = cells[Math.floor(this.thread.y / 8)][Math.floor(this.thread.x / 8)];',
    ],
    ['output: [16, 16], graphical: true', 'output: [512, 288], graphical: true'],
  ],

  // The picture, the mask and the carve all come from cardInputs at 2x, and the
  // seam COUNT scales with it: the card is the run's LAST FRAME, and a picture
  // that narrowed by an eighth instead of a quarter is a different card. Only
  // 2x — the cumulative map is one awaited launch per row, so cost is quadratic
  // in the scale, and 4x does not finish inside the 10 s watchdog. A run the
  // watchdog kills yields no canvas at all, which is the earlier MISS.
  'seam-carving': [
    [/output: \[128, 72\]/g, 'output: [256, 144]'],
    ['output: [128]', 'output: [256]'],
    ['output: [127, 72]', 'output: [255, 144]'],
    ['maskedEnergy.setOutput([w, 72])', 'maskedEnergy.setOutput([w, 144])'],
    ['for (let y = 1; y < 72; y++)', 'for (let y = 1; y < 144; y++)'],
    ['rows[71][seam[71]]', 'rows[143][seam[143]]'],
    ['carve.setOutput([w - 1, 72])', 'carve.setOutput([w - 1, 144])'],
    [
      'for (let y = 0; y < 72; y++) for (let x = 0; x < 128; x++)',
      'for (let y = 0; y < 144; y++) for (let x = 0; x < 256; x++)',
    ],
    ['for (let k = 0; k < 32; k++)', 'for (let k = 0; k < 64; k++)'],
    ['for (let y = 0; y < 72; y++) {', 'for (let y = 0; y < 144; y++) {'],
  ],

  // The matrix IS the picture, so its resolution is the two sequence lengths:
  // cardInputs aligns a longer pair built to the same recipe. `top` is that
  // pair's exact best local score — it turns a score into a colour, so it is
  // coupled to cardPair's seed and would fail quietly, not loudly, if either
  // moved. Noted next to cardInputs in the module too.
  'sequence-alignment': [
    [/output: \[37, 33\]/g, 'output: [145, 129]'],
    ['constants: { top: 40 }', 'constants: { top: 203 }'],
    [
      'for (let i = 0; i <= 32; i++) H.push(new Array(37).fill(0));',
      'for (let i = 0; i <= 128; i++) H.push(new Array(145).fill(0));',
    ],
    [/for \(let d = 2; d <= 68; d\+\+\) \{/g, 'for (let d = 2; d <= 272; d++) {'],
    ['for (let i = 0; i <= 32; i++) {', 'for (let i = 0; i <= 128; i++) {'],
    ['for (let j = 0; j <= 36; j++)', 'for (let j = 0; j <= 144; j++)'],
    ['Math.max(1, d - 36)', 'Math.max(1, d - 144)'],
    ['Math.min(32, d - 1)', 'Math.min(128, d - 1)'],
    ['lengths.map(() => 33 * 37)', 'lengths.map(() => 129 * 145)'],
  ],
};

// Two colours is a broken render everywhere except where it IS the picture:
// Life paints live and dead, Ising paints spin up and spin down, and nothing
// faithful to either lesson can add a third. Both were rejected by the default
// floor while rendering perfectly correctly. Named here so the guard stays
// strict for the modules where flat output really does mean something broke.
export const COLOUR_FLOOR = { 'cellular-automata': 2, 'ising-model': 2 };

// Applies a module's card rewrite, or returns the lesson code untouched.
export function cardVariant(code, slug) {
  const edits = CARD_SCALE[slug];
  if (!edits) return code;
  let out = code;
  for (const [from, to] of edits) {
    const before = out;
    out = out.replace(from, to);
    if (out === before) {
      throw new Error(
        `capture: ${slug} card rewrite is stale — "${from}" no longer appears in the ` +
          'solution. Re-check the constants against the task before re-capturing.'
      );
    }
  }
  return out;
}
