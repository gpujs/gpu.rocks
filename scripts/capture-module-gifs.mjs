/**
 * scripts/capture-module-gifs.mjs — the moving version of a module card.
 *
 * Three payoff tasks hand the learner a slider: the Julia set's constant, the
 * Ising lattice's temperature, the erosion rate. A still card has to pick one
 * value off that slider and throw the rest away — which is a shame, because on
 * all three the INTERESTING thing is what happens as you drag it.
 *
 * So this runs the payoff with the slider replaced by a loop and one render()
 * per value, collects the frames the console already keeps for its scrubber,
 * and encodes them.
 *
 *   node scripts/capture-module-gifs.mjs [slug ...]
 *
 * NOT part of `yarn build`: like the stills, it needs a browser and a real GPU.
 * Run it when an animated module changes and commit the GIFs. Needs ffmpeg,
 * gifsicle and cwebp on PATH. GIF_DIR=... redirects the output for previewing.
 *
 * GIF, despite appearances. It also writes an animated WebP for comparison and
 * WebP loses every time here — 3-10x the size — because these frames are
 * high-entropy (a spin lattice is close to noise) and GIF's shared 32-96 colour
 * palette beats per-frame lossy compression on exactly that kind of content.
 * The WebPs are not shipped; they exist so the next person does not have to
 * re-derive that the obvious modernisation is a regression.
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { loadContent } from './contentLoader.mjs';
import { launch } from './browser.mjs';
import { cardVariant } from './cardVariant.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = process.env.GIF_DIR || join(ROOT, 'public/img/modules');
const MANIFEST = join(ROOT, 'src/Learn/content/moduleAnims.js');

// Each entry replaces a slider with a sweep over its own range. The sweeps are
// built to LOOP: the Julia constant walks a closed circle, and the Ising
// temperature ramps down through Tc and back up, so the last frame meets the
// first and there is no jump cut.
//
// Applied AFTER the still card's rewrite (cardVariant), so an animation frame
// is the same picture as the still — same resolution, same constants — with
// only the swept variable moving.
const CARD_ANIM = {
  // c walks a circle around the lesson's dendrite value, so the set breathes
  // through the shapes either side of it and returns to where it started.
  'escape-time-fractals': {
    width: 288,
    fps: 8,
    colors: 64,
    lossy: 150,
    edits: [
      [
        /const cRe = slider\([\s\S]*?\n/,
        'const FRAMES = 36;\n',
      ],
      [/const cIm = slider\([^\n]*\n/, ''],
      [
        'await julia(cRe, cIm);\nrender(julia.canvas);',
        'for (let f = 0; f < FRAMES; f++) {\n' +
          '  const th = (2 * Math.PI * f) / FRAMES;\n' +
          '  await julia(-0.7269 + 0.16 * Math.cos(th), 0.1889 + 0.16 * Math.sin(th));\n' +
          '  render(julia.canvas);\n' +
          '}',
      ],
    ],
  },

  // Temperature ramps 3.5 -> 1.5 -> 3.5 across one continuous run rather than
  // re-equilibrating per frame: 150 sweeps x 40 temperatures would never finish
  // inside the run watchdog. Slow cooling through Tc = 2.269 and back is also
  // the more honest picture — order appearing and dissolving, not a slideshow
  // of independent equilibria.
  'ising-model': {
    width: 240,
    fps: 8,
    colors: 32,
    lossy: 200,
    edits: [
      [/const temperature = slider\([^\n]*\n/, 'const SWEEPS = 240;\n'],
      [
        /let s = alignedLattice;[\s\S]*?^}$/m,
        'let s = alignedLattice;\n' +
          'const trace = [];\n' +
          'for (let k = 0; k < SWEEPS; k++) {\n' +
          '  // 3.5 -> 1.5 -> 3.5, so the loop closes where it opened\n' +
          '  const ramp = Math.abs(1 - (2 * k) / SWEEPS);\n' +
          '  const temperature = 1.5 + 2 * ramp;\n' +
          '  s = await black(await red(s, temperature, k * 2), temperature, k * 2 + 1);\n' +
          '  trace.push(meanOf(s));\n' +
          '  if (k % 12 === 0) {\n' +
          '    await paint(s);\n' +
          '    render(paint.canvas);\n' +
          '  }\n' +
          '}',
      ],
      [/console\.log\('temperature'[^\n]*\n/, ''],
      [/console\.log\('magnetisation[^\n]*\n/, ''],
    ],
  },

  // The task now hands the learner a generations dial, so the card sweeps it
  // end to end: the whole menagerie evolving from its seed, oscillators
  // beating and ships crossing. Not a seamless loop — Life is not periodic
  // over a world this size — but it starts from a clean seed each cycle.
  'cellular-automata': {
    width: 320,
    fps: 6,
    edits: [
      [/const generations = slider\([^\n]*\n/, 'const GENS = 33;\n'],
      [
        /let current = world;\nfor \(let g = 0; g < generations; g\+\+\) \{\n  current = await step\(current\);\n\}\nawait paint\(current\);\nrender\(paint\.canvas\);/,
        'let current = world;\n' +
          'for (let g = 0; g < GENS; g++) {\n' +
          '  await paint(current);\n' +
          '  render(paint.canvas);\n' +
          '  current = await step(current);\n' +
          '}',
      ],
    ],
  },

  // ---- swept dials the lesson now offers -------------------------------------

  // Only the last four rungs. Until then every remaining jump is a multiple of
  // two, so three quarters of the grid cannot reach its nearest seed and the
  // diagram is a flat field of a few ids — thirteen near-identical frames for
  // one good one. The tail is where it actually resolves: coarse cells sharpen
  // into the exact diagram, and the last frame IS the still.
  //
  // Painted as the DIAGRAM, not the error field: paintErr reddens every wrong
  // cell, which early on is most of them. Handed an all-zero mask
  // (worse(truth, truth)) the same kernel paints the plain Voronoi.
  //
  // Per-frame palettes because the colours change wholesale across those four
  // frames; one shared palette spent itself on the early field and tinted the
  // NOT jump-flooding. Its sweep was built and measured at fourteen frames,
  // then four, three and two, and every rung before the last is still a flat
  // field of a few seed ids: jump flooding resolves ABRUPTLY, because the
  // stride-1 pass is not a polish pass, it is the one that reaches odd-offset
  // cells at all. Two frames is a flicker, not an animation, and the still is
  // the best picture this module makes. The lesson keeps its passes dial.

  // Progressive rendering IS the module, so the card renders inside the
  // accumulation loop: the same picture getting quieter. The two accumulators
  // and the RMS plot are left alone, so an animation frame is exactly the still
  // at that sample count. Held frames at the end — the last ten frames of a
  // 1/sqrt(n) fall are nearly identical anyway.
  'path-tracing': {
    width: 224,
    fps: 6,
    colors: 48,
    lossy: 200,
    edits: [
      [/const FRAMES = slider\([^\n]*\n/, 'const FRAMES = 24;\n'],
      [
        '  ideal.push(errs[0] / Math.sqrt(f + 1));\n}',
        '  ideal.push(errs[0] / Math.sqrt(f + 1));\n  await paint(accA);\n  render(paint.canvas);\n}',
      ],
      [
        'await paint(accA);\nrender(paint.canvas);\nconsole.log(',
        'for (let h = 0; h < 8; h++) render(paint.canvas);\nconsole.log(',
      ],
    ],
  },

  // The window length the lesson now puts on a dial — as TAPER, not WIN, so the
  // transform length, the frame count and the signal length hold still and only
  // the Hann bell inside each frame shortens. Frequency resolution moves, the
  // frequency AXIS does not, which is the trade the module measures. Longest
  // first, so frame 0 is the still; the way back down re-paints cached
  // spectrograms, so each length is transformed once and the loop still closes.
  spectrograms: {
    width: 288,
    fps: 5,
    colors: 96,
    lossy: 120,
    edits: [
      [/const TAPER = slider\([^\n]*\n/, 'const TAPERS = [256, 192, 128, 96, 64, 32];\n'],
      [
        'const spectrogram = gpu.createKernel(function (signal) {',
        'const makeSpectrogram = TAPER => gpu.createKernel(function (signal) {',
      ],
      [
        /const spec = await spectrogram\(signal\);[\s\S]*?console\.log\('ratio:[^\n]*\n/,
        '// One STFT per window length, longest first. The way back down the\n' +
          '// sweep re-paints these, so each length is transformed once, not twice.\n' +
          'const specs = [];\n' +
          'const peaks = [];\n' +
          'for (const taper of TAPERS) {\n' +
          '  const s = await makeSpectrogram(taper)(signal);\n' +
          '  let p = 0;\n' +
          '  for (let b = 0; b < BINS; b++) {\n' +
          '    for (let f = 0; f < FRAMES; f++) if (s[b][f] > p) p = s[b][f];\n' +
          '  }\n' +
          '  specs.push(s);\n' +
          '  peaks.push(p);\n' +
          '}\n',
      ],
      [
        'await paint(spec, peak);\nrender(paint.canvas);',
        'for (let i = 0; i < specs.length; i++) {\n' +
          '  await paint(specs[i], peaks[i]);\n' +
          '  render(paint.canvas);\n' +
          '}\n' +
          '// back up the sweep, endpoints not repeated, so the loop closes\n' +
          'for (let i = specs.length - 2; i >= 1; i--) {\n' +
          '  await paint(specs[i], peaks[i]);\n' +
          '  render(paint.canvas);\n' +
          '}',
      ],
    ],
  },

  // The lobes drift apart until the neck snaps, then smin welds them into one
  // blob and back. That IS the module. The ramp is a palindrome, so the last
  // frame meets the first.
  'ray-marched-metaballs': {
    width: 288,
    fps: 10,
    colors: 64,
    lossy: 150,
    edits: [
      [/const separation = slider\([^\n]*\n/, 'const FRAMES = 40;\n'],
      [
        'await finalScene(separation, 0.5, 0.3, -0.86, 0, -0.51, 1);\nrender(finalScene.canvas);',
        'for (let f = 0; f < FRAMES; f++) {\n' +
          '  // 1.0 -> 0.05 -> 1.0, so the last frame meets the first\n' +
          '  const ramp = Math.abs(1 - (2 * f) / FRAMES);\n' +
          '  await finalScene(0.05 + 0.95 * ramp, 0.5, 0.3, -0.86, 0, -0.51, 1);\n' +
          '  render(finalScene.canvas);\n' +
          '}',
      ],
    ],
  },

  // The seed square dissolving into coral, one paint every 16 of the 800 steps
  // the card's finer lattice takes. Not a loop that closes — nothing in
  // Gray-Scott comes back to its seed — but it starts from one.
  'reaction-diffusion': {
    width: 256,
    fps: 10,
    colors: 64,
    lossy: 130,
    edits: [
      [/const steps = slider\([^\n]*\n/, 'const EVERY = 16;\n'],
      [
        'for (let i = 0; i < steps; i++) {\n' +
          '  const nextU = await stepU(u, v);\n' +
          '  const nextV = await stepV(u, v);\n' +
          '  u = nextU;\n' +
          '  v = nextV;\n' +
          '}',
        'for (let i = 0; i < 800; i++) {\n' +
          '  if (i % EVERY === 0) {\n' +
          '    await paint(v);\n' +
          '    render(paint.canvas);\n' +
          '  }\n' +
          '  const nextU = await stepU(u, v);\n' +
          '  const nextV = await stepV(u, v);\n' +
          '  u = nextU;\n' +
          '  v = nextV;\n' +
          '}',
      ],
    ],
  },

  // The input IS a clip, so the sweep is the clip: one composite per frame
  // rather than only the last. The model is trained once, so the backdrop does
  // not flicker under the object. Out and back, so the object crosses, returns,
  // and the loop closes — there are only eight distinct pictures.
  'video-filters': {
    width: 256,
    fps: 5,
    colors: 96,
    lossy: 120,
    edits: [
      [/const frame = slider\([^\n]*\n/, ''],
      [
        'for (let i = 1; i < frame; i++) {\n' +
          '  model = await learn(frames[i], model);\n' +
          '}\n\n' +
          'const live = frames[frame];\n' +
          'await compose(live, await feather(await softMask(live, model)));\n' +
          'render(compose.canvas);',
        'for (let i = 1; i < frames.length - 1; i++) {\n' +
          '  model = await learn(frames[i], model);\n' +
          '}\n\n' +
          'for (let f = 0; f < 2 * frames.length - 2; f++) {\n' +
          '  const live = frames[f < frames.length ? f : 2 * frames.length - 2 - f];\n' +
          '  await compose(live, await feather(await softMask(live, model)));\n' +
          '  render(compose.canvas);\n' +
          '}',
      ],
    ],
  },

  // Phase is exactly what a still throws away: rings standing still. Swept over
  // one full period the crests travel outward and the last frame IS the first.
  // The first edit is not optional — cardVariant scales the ring spacing for the
  // 512px card but leaves the radians-to-pixels conversion alone, and a phase
  // that only walked a quarter of a ring would not loop.
  // smooth concentric gradients are the worst case for a 256-colour format:
  // every ring edge dithers. Fewer colours and a heavier lossy pass cost almost
  // nothing visible here and a third of the file.
  'pixels-from-scratch': {
    width: 256,
    fps: 12,
    colors: 32,
    lossy: 200,
    edits: [
      ['r = r - phase / 0.35', 'r = r - phase / 0.0875'],
      [/const phase = slider\([^\n]*\n/, 'const FRAMES = 36;\n'],
      [
        'await ripples(phase);\nrender(ripples.canvas);',
        'for (let f = 0; f < FRAMES; f++) {\n' +
          '  await ripples((2 * Math.PI * f) / FRAMES);\n' +
          '  render(ripples.canvas);\n' +
          '}',
      ],
    ],
  },

  // ---- already animated, no rewrite needed -----------------------------------
  //
  // These three already render inside their driver loop, so the run the STILL
  // capture takes its last frame from is a whole animation the catalogue was
  // throwing away. `edits: []` means "capture what the lesson already does".

  // 20 paints across the erosion run: rain falling, channels cutting, lakes
  // filling. The module's own slider stays where the lesson leaves it — the
  // interesting axis here is time, not the erosion rate.
  'hydraulic-erosion': { width: 256, fps: 6, colors: 64, lossy: 150, edits: [] },

  // one paint per carved seam: the picture narrows and the face does not.
  'seam-carving': { width: 288, fps: 8, colors: 48, lossy: 170, stride: 3, edits: [] },

  // the DP wavefront sweeping the matrix — 271 frames of it, which is more
  // than a card needs; every ninth keeps the sweep legible and the file small.
  'sequence-alignment': { width: 288, fps: 10, colors: 64, stride: 9, edits: [] },
};

function animVariant(code, slug) {
  const spec = CARD_ANIM[slug];
  if (!spec) return null;
  let out = cardVariant(code, slug);
  for (const [from, to] of spec.edits) {
    const before = out;
    out = out.replace(from, to);
    if (out === before) {
      throw new Error(`gif: ${slug} sweep is stale — ${from} no longer matches`);
    }
  }
  return out;
}

const only = process.argv.slice(2).filter(a => !a.startsWith('--'));
const slugs = Object.keys(CARD_ANIM).filter(s => !only.length || only.includes(s));
if (!slugs.length) {
  console.error('gif: nothing to do');
  process.exit(1);
}

const reg = await loadContent(ROOT);
mkdirSync(OUT_DIR, { recursive: true });

const vite = spawn('yarn', ['vite', '--port', '0'], { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
const port = await new Promise((res, rej) => {
  let buf = '';
  vite.stdout.on('data', d => {
    buf += d;
    const m = buf.match(/localhost:(\d+)/);
    if (m) res(m[1]);
  });
  setTimeout(() => rej(new Error('vite did not start')), 40000);
});

const browser = await launch();

for (const slug of slugs) {
  const mod = reg.modules.find(m => m.slug === slug);
  const gfx = mod.tasks
    .map((task, i) => ({ task, step: i + 1 }))
    .filter(({ task }) => /graphical:\s*true/.test(task.solutionCode || ''));
  const { task, step } = gfx[gfx.length - 1];
  const code = animVariant(task.solutionCode, slug);

  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(String(e.message).slice(0, 140)));
  let frames = [];
  try {
    await page.goto(`http://localhost:${port}${mod.url}/${step}`, { waitUntil: 'networkidle2' });
    await new Promise(r => setTimeout(r, 1300));
    await page.evaluate(() => {
      window.__learnCardCapture = true;
    });
    await page.evaluate(src => {
      const view = window.__learnEditorView;
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: src } });
    }, code);
    await page.click('.tb-run');
    // Consecutive render() calls become ONE scrubber, not one <img> each: the
    // frames live in React state and only the selected one is in the DOM. So
    // wait for the scrubber's range to stop growing, then walk it.
    let last = -1;
    for (let i = 0; i < 60; i++) {
      await new Promise(r => setTimeout(r, 1000));
      const n = await page.evaluate(() => {
        const rs = [...document.querySelectorAll('.console .imgout.frames input[type=range]')];
        return rs.length ? Number(rs[rs.length - 1].max) + 1 : 0;
      });
      if (n > 0 && n === last) break;
      last = n;
    }
    frames = await page.evaluate(async () => {
      const strips = [...document.querySelectorAll('.console .imgout.frames')];
      if (!strips.length) {
        const one = document.querySelector('.console .imgout img');
        return one ? [one.src] : [];
      }
      const strip = strips[strips.length - 1];
      const range = strip.querySelector('input[type=range]');
      const img = strip.querySelector('img');
      // a React-controlled input ignores .value =; go through the native setter
      // so its onChange actually fires
      const setValue = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,
        'value'
      ).set;
      const out = [];
      for (let i = 0; i <= Number(range.max); i++) {
        setValue.call(range, String(i));
        range.dispatchEvent(new Event('input', { bubbles: true }));
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        out.push(img.src);
      }
      return out;
    });
  } catch (e) {
    errors.push(String(e.message).slice(0, 140));
  }
  await page.close();

  if (frames.length < 4) {
    console.log(`  MISS  ${slug} — ${frames.length} frames${errors.length ? ` (${errors[0]})` : ''}`);
    continue;
  }

  const W = CARD_ANIM[slug].width || 320;
  const FPS = (CARD_ANIM[slug].ramp && CARD_ANIM[slug].ramp.fps) || CARD_ANIM[slug].fps || 8;
  const COLORS = CARD_ANIM[slug].colors || 96;
  const LOSSY = String(CARD_ANIM[slug].lossy === undefined ? 100 : CARD_ANIM[slug].lossy);
  // bayer's crosshatch is cheap and fine on flat art; a smooth gradient needs an
  // error-diffusion dither or it breaks into visible speckle
  const DITHER = CARD_ANIM[slug].dither || 'bayer:bayer_scale=3';
  const tmp = join(OUT_DIR, `frames-${slug}`);
  rmSync(tmp, { recursive: true, force: true });
  mkdirSync(tmp, { recursive: true });
  const stride = CARD_ANIM[slug].stride || 1;
  let picked = frames.filter((_, i) => i % stride === 0);
  // `tail` keeps only the last n frames. Some runs are only worth watching at
  // the end — see jump-flooding, whose diagram is a flat field of a few seed
  // ids until the last couple of rungs and then resolves.
  if (CARD_ANIM[slug].tail) picked = picked.slice(-CARD_ANIM[slug].tail);

  // Ease-out: hold each frame longer than the last, so the run opens quickly
  // and settles into its final state instead of plodding at a constant rate.
  // Done by REPEATING frames at a higher base rate rather than by per-frame
  // delays, because gifsicle -O3 merges identical neighbours back into one
  // frame with the summed delay — the same result, and it survives the
  // optimiser rather than being undone by it.
  const ramp = CARD_ANIM[slug].ramp;
  const kept = [];
  picked.forEach((src, i) => {
    const t = picked.length > 1 ? i / (picked.length - 1) : 0;
    const reps = ramp ? Math.max(1, Math.round(ramp.from + (ramp.to - ramp.from) * t * t)) : 1;
    for (let r = 0; r < reps; r++) kept.push(src);
  });

  kept.forEach((src, i) => {
    writeFileSync(join(tmp, `f${String(i).padStart(3, '0')}.png`), Buffer.from(src.split(',')[1], 'base64'));
  });

  const gif = join(OUT_DIR, `${slug}.gif`);
  const webp = join(tmp, `${slug}.webp`);
  const pattern = join(tmp, 'f%03d.png');
  // one palette for the whole animation, not per frame: a per-frame palette
  // makes flat regions crawl, which on a card reads as noise
  const palette = join(tmp, 'palette.png');
  // `single` gives each frame its own palette. Costs bytes, and is the only
  // correct choice when an animation changes colour WHOLESALE: jump-flooding
  // opens on a near-solid red error field and ends on a pastel Voronoi, and one
  // global palette spent itself on the reds — the finished diagram came out
  // tinted red and did not match its own still.
  const perFrame = CARD_ANIM[slug].palette === 'single';
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', pattern,
    '-vf', `scale=${W}:-1:flags=lanczos,palettegen=max_colors=${COLORS}:stats_mode=${perFrame ? 'single' : 'diff'}`,
    perFrame ? palette.replace('.png', '%03d.png') : palette]);
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', pattern,
    '-i', perFrame ? palette.replace('.png', '%03d.png') : palette,
    '-lavfi', `scale=${W}:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=${DITHER}${perFrame ? ':new=1' : ''}`,
    gif]);
  execFileSync('gifsicle', LOSSY === '0' ? ['-O3', '-b', gif] : ['-O3', `--lossy=${LOSSY}`, '-b', gif]);
  // img2webp, not ffmpeg: the homebrew ffmpeg is built without libwebp, and
  // WebP is the whole point of the comparison — it is usually a fraction of the
  // GIF, which is what decides whether this can ship.
  let webpOk = true;
  try {
    const scaled = kept.map((_, i) => join(tmp, `w${String(i).padStart(3, '0')}.webp`));
    kept.forEach((_, i) => {
      execFileSync('ffmpeg', ['-y', '-loglevel', 'error',
        '-i', join(tmp, `f${String(i).padStart(3, '0')}.png`),
        '-vf', `scale=${W}:-1:flags=lanczos`, join(tmp, `s${String(i).padStart(3, '0')}.png`)]);
      execFileSync('cwebp', ['-quiet', '-q', '55', join(tmp, `s${String(i).padStart(3, '0')}.png`),
        '-o', scaled[i]]);
    });
    execFileSync('img2webp', ['-loop', '0', '-d', String(Math.round(1000 / FPS)), ...scaled, '-o', webp]);
  } catch (e) {
    webpOk = false;
  }

  const kb = f => (statSync(f).size / 1024).toFixed(0);
  console.log(
    `  ok    ${slug.padEnd(24)} ${picked.length}/${frames.length} frames   gif ${kb(gif)} KB` +
      (webpOk ? `   webp ${kb(webp)} KB` : '   webp unavailable')
  );
  rmSync(tmp, { recursive: true, force: true });
}

await browser.close();
vite.kill();

// Same content-hash trick as the stills: <slug>.gif is a stable URL whose bytes
// change on re-capture, and neither the CDN nor a phone would notice otherwise.
if (OUT_DIR === join(ROOT, 'public/img/modules')) {
  const entries = readdirSync(OUT_DIR)
    .filter(f => f.endsWith('.gif'))
    .sort()
    .map(f => [
      f.replace(/\.gif$/, ''),
      createHash('sha256').update(readFileSync(join(OUT_DIR, f))).digest('hex').slice(0, 8),
    ]);
  writeFileSync(
    MANIFEST,
    '// GENERATED by scripts/capture-module-gifs.mjs — do not edit by hand.\n' +
      '//\n' +
      '// Slug -> content hash of its animated card, for the modules whose payoff\n' +
      '// has a dial worth turning. The catalogue shows the still by default and\n' +
      '// swaps to this on hover; see ModuleThumb for why it is not simply always\n' +
      '// animated.\n' +
      `export default new Map(${JSON.stringify(entries, null, 2)});\n`
  );
  console.log(`gif: wrote manifest with ${entries.length} slugs`);
}
console.log(`gif: wrote to ${OUT_DIR}`);
