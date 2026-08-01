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
  const FPS = CARD_ANIM[slug].fps || 8;
  const COLORS = CARD_ANIM[slug].colors || 96;
  const LOSSY = String(CARD_ANIM[slug].lossy || 100);
  const tmp = join(OUT_DIR, `frames-${slug}`);
  rmSync(tmp, { recursive: true, force: true });
  mkdirSync(tmp, { recursive: true });
  const stride = CARD_ANIM[slug].stride || 1;
  const kept = frames.filter((_, i) => i % stride === 0);
  kept.forEach((src, i) => {
    writeFileSync(join(tmp, `f${String(i).padStart(3, '0')}.png`), Buffer.from(src.split(',')[1], 'base64'));
  });

  const gif = join(OUT_DIR, `${slug}.gif`);
  const webp = join(tmp, `${slug}.webp`);
  const pattern = join(tmp, 'f%03d.png');
  // one palette for the whole animation, not per frame: a per-frame palette
  // makes flat regions crawl, which on a card reads as noise
  const palette = join(tmp, 'palette.png');
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', pattern,
    '-vf', `scale=${W}:-1:flags=lanczos,palettegen=max_colors=${COLORS}:stats_mode=diff`, palette]);
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', pattern, '-i', palette,
    '-lavfi', `scale=${W}:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3`, gif]);
  execFileSync('gifsicle', ['-O3', `--lossy=${LOSSY}`, '-b', gif]);
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
    `  ok    ${slug.padEnd(24)} ${kept.length}/${frames.length} frames   gif ${kb(gif)} KB` +
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
