/**
 * scripts/capture-module-renders.mjs — thumbnails that are the real thing.
 *
 * Eighteen modules end in a picture: a Mandelbrot, a Voronoi diagram, carved
 * terrain, a path-traced scene. This runs each of those modules' payoff task —
 * its actual solution, in the actual sandbox, on the actual GPU — and saves the
 * canvas it paints to public/img/modules/<slug>.png, which the learn catalogue
 * uses as that module's card art.
 *
 * So the thumbnail is the module's output rather than an illustration of it,
 * and it cannot drift: re-run this and every card is whatever the course now
 * produces.
 *
 *   node scripts/capture-module-renders.mjs [slug ...]
 *   node scripts/capture-module-renders.mjs --manifest   (rebuild the list only)
 *   node scripts/capture-module-renders.mjs --check      (report staleness)
 *
 * NOT part of `yarn build`: it needs a browser and a working GPU, and CI has
 * neither reliably. Run it when a painting module's output changes, and commit
 * the PNGs. `--check` reports what would change without writing, which is what
 * to run if you want to know whether the committed art is stale.
 *
 * Modules with no graphical task — and the handful in NO_RENDER below, whose
 * real output is technically correct but unreadable at thumbnail size — fall
 * back to their first figure, which the catalogue inlines directly from
 * content/figures. Nothing to capture for those.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { loadContent } from './contentLoader.mjs';
import { launch } from './browser.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = join(ROOT, 'public/img/modules');
const MANIFEST = join(ROOT, 'src/Learn/content/moduleRenders.js');

const args = process.argv.slice(2);
const check = args.includes('--check');
const only = args.filter(a => !a.startsWith('--'));

const reg = await loadContent(ROOT);

// Modules whose real output is technically correct and useless as card art.
// Kept as an explicit list rather than by deleting the PNGs, so the reason
// survives the next re-capture instead of quietly coming back.
//
//   data-in-data-out / pipelines-and-textures — both end on a grayscale
//   luminance map, which at 16:9 and 300px wide is indistinguishable from
//   static. Their figures show the actual idea (output shapes; the chain that
//   never leaves the card), so the fallback is strictly better here.
//   convolution-and-filters — same problem one step further on: its payoff is a
//   Sobel edge-magnitude map, which is grey noise at this size.
const NO_RENDER = new Set([
  'data-in-data-out',
  'pipelines-and-textures',
  'convolution-and-filters',
]);

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
const CARD_SCALE = {
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
};

// Applies a module's card rewrite, or returns the lesson code untouched.
function cardVariant(code, slug) {
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

// Did scaling produce a PICTURE, or the black rectangle you get when every
// thread reads past the end of its input? Cheap proxy: how many distinct bytes
// the PNG's pixel payload has. A real render has hundreds; a flat fill has one.
function pngSize(png) {
  return `${png.readUInt32BE(16)}x${png.readUInt32BE(20)}`;
}

// Did scaling produce a PICTURE, or the black rectangle you get when every
// thread reads past the end of its input?
//
// Measured on real pixels, in the page. The first version of this counted
// distinct bytes in the PNG payload, which is worthless: that payload is
// DEFLATE output and looks like noise whatever the image was, so a half-black
// render scored exactly as well as a good one. Decoding back to pixels costs
// one canvas and answers the question actually being asked.
async function pixelStats(page, dataUrl) {
  return page.evaluate(
    src =>
      new Promise(resolve => {
        const img = new Image();
        img.onload = () => {
          const c = document.createElement('canvas');
          c.width = img.width;
          c.height = img.height;
          const ctx = c.getContext('2d');
          ctx.drawImage(img, 0, 0);
          const d = ctx.getImageData(0, 0, c.width, c.height).data;
          let black = 0;
          let sum = 0;
          const hues = new Set();
          for (let i = 0; i < d.length; i += 4) {
            sum += (d[i] + d[i + 1] + d[i + 2]) / 3;
            if (d[i] < 8 && d[i + 1] < 8 && d[i + 2] < 8) black++;
            hues.add((d[i] >> 4) * 256 + (d[i + 1] >> 4) * 16 + (d[i + 2] >> 4));
          }
          const n = d.length / 4;
          resolve({ black: black / n, mean: sum / n, colours: hues.size });
        };
        img.onerror = () => resolve(null);
        img.src = src;
      }),
    dataUrl
  );
}

// The LAST graphical task: the payoff, the picture the module exists to make.
function payloadTask(mod) {
  const gfx = mod.tasks
    .map((t, i) => ({ task: t, step: i + 1 }))
    .filter(({ task }) => /graphical:\s*true/.test(task.solutionCode || ''));
  return gfx.length ? gfx[gfx.length - 1] : null;
}

const targets = reg.modules
  .map(mod => ({ mod, ...(payloadTask(mod) || {}) }))
  .filter(t => t.task)
  .filter(t => !NO_RENDER.has(t.mod.slug))
  .filter(t => !only.length || only.includes(t.mod.slug));

// The manifest is what the catalogue reads, and it is simply "which slugs have
// art on disk" — so it can be rebuilt without a browser after removing one.
function writeManifest() {
  const slugs = readdirSync(OUT_DIR)
    .filter(f => f.endsWith('.png'))
    .map(f => f.replace(/\.png$/, ''))
    .sort();
  writeFileSync(
    MANIFEST,
    '// GENERATED by scripts/capture-module-renders.mjs — do not edit by hand.\n' +
      '//\n' +
      '// Slugs whose module paints something worth showing, and therefore have a\n' +
      '// captured render at /img/modules/<slug>.png. Everything else falls back to\n' +
      "// the module's first figure. Rebuild with `--manifest` (no browser needed).\n" +
      `export default new Set(${JSON.stringify(slugs, null, 2)});\n`
  );
  return slugs;
}

if (args.includes('--manifest')) {
  const slugs = writeManifest();
  console.log(`capture: manifest rebuilt from disk — ${slugs.length} slugs`);
  process.exit(0);
}

if (!targets.length) {
  console.error('capture: nothing to do');
  process.exit(1);
}

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
const captured = [];
let changed = 0;
let missed = 0;

for (const { mod, task, step } of targets) {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(String(e.message).slice(0, 110)));
  let url = null;
  let stats = null;
  try {
    await page.goto(`http://localhost:${port}${mod.url}/${step}`, { waitUntil: 'networkidle2' });
    await new Promise(r => setTimeout(r, 1300));
    await page.evaluate(code => {
      const view = window.__learnEditorView;
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: code } });
    }, cardVariant(task.solutionCode, mod.slug));
    // Asks the sandbox for task.cardInputs instead of task.inputs, where the
    // module declares one: the same data at card resolution. Set before Run,
    // read when the run request is built.
    await page.evaluate(() => {
      window.__learnCardCapture = true;
    });
    await page.click('.tb-run');
    // Some of these accumulate for tens of frames before the picture exists.
    for (let i = 0; i < 45 && !url; i++) {
      await new Promise(r => setTimeout(r, 1000));
      url = await page.evaluate(() => {
        const imgs = [...document.querySelectorAll('.console .imgout img')];
        return imgs.length ? imgs[imgs.length - 1].src : null;
      });
    }
    if (url) stats = await pixelStats(page, url);
  } catch (e) {
    errors.push(String(e.message).slice(0, 110));
  }
  await page.close();

  if (!url) {
    missed++;
    console.log(`  MISS  ${mod.slug} — no canvas${errors.length ? ` (${errors[0]})` : ''}`);
    continue;
  }
  const png = Buffer.from(url.split(',')[1], 'base64');

  // A card rewrite that got the constants wrong does not throw — it paints a
  // zoomed-in flat colour or an all-black frame, which would sail through as a
  // perfectly valid PNG. Refuse to overwrite good art with that.
  if (stats && (stats.black > 0.6 || stats.colours < 6)) {
    missed++;
    console.log(
      `  BAD   ${mod.slug.padEnd(24)} ${pngSize(png)} is ${(stats.black * 100).toFixed(0)}% black, ` +
        `${stats.colours} colours — kept the existing art`
    );
    continue;
  }

  const file = join(OUT_DIR, `${mod.slug}.png`);
  const before = existsSync(file) ? createHash('sha256').update(readFileSync(file)).digest('hex') : null;
  const after = createHash('sha256').update(png).digest('hex');
  captured.push(mod.slug);
  if (before !== after) {
    changed++;
    if (!check) writeFileSync(file, png);
    console.log(
      `  ${check ? 'STALE' : 'write'} ${mod.slug.padEnd(24)} task ${step}  ` +
        `${pngSize(png)}  ${(png.length / 1024).toFixed(0)} KB${CARD_SCALE[mod.slug] ? '  (card variant)' : ''}`
    );
  } else {
    console.log(`  same  ${mod.slug.padEnd(24)} task ${step}`);
  }
}

await browser.close();
vite.kill();

// The catalogue needs to know which slugs have art WITHOUT probing the network
// for a 404 per card, so the list is generated here and imported by the app.
if (!check && !only.length) {
  const slugs = writeManifest();
  console.log(`\ncapture: wrote manifest with ${slugs.length} slugs`);
}

console.log(
  `capture: ${captured.length} captured, ${changed} ${check ? 'stale' : 'written'}, ${missed} missed`
);
if (check && changed) process.exit(1);
