/**
 * scripts/bench-record.mjs — capture a saved run.
 *
 * Drives the real benchmark page in a real browser, waits for a full pass, and
 * writes the table to src/Bench/saved/<id>.json. Nothing here re-measures
 * anything: the numbers are lifted from the page that just displayed them, so a
 * saved run is exactly what a visitor with this machine would have seen.
 *
 *   node scripts/bench-record.mjs --label "M4 Max · Chrome 141"
 *
 * Saved runs are read-only in the UI. The page's whole purpose is comparison,
 * and a table half-measured here and half on someone's laptop compares nothing.
 */
import http from 'node:http';
import { existsSync, readFileSync, statSync, writeFileSync, readdirSync } from 'node:fs';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launch } from './browser.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIST = join(ROOT, 'dist');
const SAVED = join(ROOT, 'src/Bench/saved');

const argv = process.argv.slice(2);
const arg = n => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : null; };
const label = arg('--label') || 'unlabelled run';
const stamp = arg('--date') || new Date().toISOString().slice(0, 10);
const id = arg('--id') || `${stamp}-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')}`;

if (!existsSync(join(DIST, 'benchmark', 'index.html')) && !existsSync(join(DIST, 'benchmark.html'))) {
  console.error('bench-record: build first — dist/benchmark is missing');
  process.exit(1);
}

const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.png': 'image/png', '.gif': 'image/gif', '.svg': 'image/svg+xml' };
const srv = http.createServer((q, r) => {
  const u = decodeURIComponent(q.url.split('?')[0]);
  for (const p of [join(DIST, u), join(DIST, `${u}.html`), join(DIST, u, 'index.html')]) {
    if (existsSync(p) && statSync(p).isFile()) {
      r.writeHead(200, { 'content-type': TYPES[extname(p)] || 'application/octet-stream' });
      return r.end(readFileSync(p));
    }
  }
  r.writeHead(404);
  return r.end('not found');
});
await new Promise(r => srv.listen(0, r));
const base = `http://localhost:${srv.address().port}`;

const browser = await launch({ real: true, headed: argv.includes('--headed') });
const page = await browser.newPage();
page.on('pageerror', e => console.log('  page error:', String(e.message).slice(0, 120)));
await page.setViewport({ width: 1400, height: 1000 });
await page.goto(`${base}/benchmark`, { waitUntil: 'networkidle2' });
await new Promise(r => setTimeout(r, 1200));

// A saved run is presented as what a real machine did. Chrome will happily
// hand back a SwiftShader adapter and a software WebGL renderer and never say
// so, and those numbers would be a CPU wearing a GPU's label — the one lie
// this page cannot afford. Check before spending half an hour, not after.
const rendering = await page.evaluate(async () => {
  const gl = document.createElement('canvas').getContext('webgl2');
  const dbg = gl && gl.getExtension('WEBGL_debug_renderer_info');
  let adapter = 'none';
  try {
    const a = await navigator.gpu.requestAdapter();
    const info = a && (a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : null));
    adapter = info ? [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') : 'unknown';
  } catch (e) { /* left as none */ }
  return {
    adapter,
    webgl: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : (gl ? 'unknown' : 'none'),
  };
});
console.log(`bench-record: WebGPU ${rendering.adapter}`);
console.log(`bench-record: WebGL  ${rendering.webgl}`);
const software = /swiftshader|llvmpipe|software|angle \(google/i;
if (!argv.includes('--allow-software')) {
  const bad = Object.entries(rendering).filter(([, v]) => software.test(v));
  if (bad.length) {
    console.error(`bench-record: software renderer (${bad.map(([k]) => k).join(', ')}) — refusing to save`);
    console.error('  these numbers would be a CPU labelled as a GPU. --allow-software to override.');
    process.exit(1);
  }
}

// A saved run is a WHOLE table — every workload, every column — so the two run
// options the page offers a visitor are both turned off here: brief mode would
// store ten rows, and an unchecked column would store a hole. The full-run
// warning is for someone about to spend half an hour by accident; this script
// is that half hour on purpose, so it answers its own dialog.
await page.evaluate(() => {
  const set = (input, want) => {
    if (!input || input.disabled || input.checked === want) return;
    input.click();
  };
  set(document.querySelector('.opt.brief input'), false);
  document.querySelectorAll('.opt.cols input').forEach(i => set(i, true));
});
await new Promise(r => setTimeout(r, 200));

const rows = await page.evaluate(() => document.querySelectorAll('#root tbody tr').length);
console.log(`bench-record: ${rows} workload(s), running…`);
await page.click('.toolbar .btn.primary');
await page.waitForSelector('.runwarn', { timeout: 5000 });
await page.click('.runwarn .btn.primary');

// A full pass is half an hour and longer on a slow machine, so the cap is
// wall-clock and generous rather than a poll count that quietly expired first.
const deadline = Date.now() + 150 * 60 * 1000;
let done = false;
for (let i = 0; !done; i++) {
  if (Date.now() > deadline) break;
  // eslint-disable-next-line no-await-in-loop
  await new Promise(r => setTimeout(r, 3000));
  // eslint-disable-next-line no-await-in-loop
  const s = await page.evaluate(() => window.__benchStatus || '');
  if (/^done|^stopped/.test(s)) done = true;
  else if (i % 20 === 0) console.log(`  [${new Date().toTimeString().slice(0, 5)}] ${s}`);
}
if (!done) {
  console.error('bench-record: timed out; nothing written');
  process.exit(1);
}

const results = await page.evaluate(() => window.__benchResults);
const ua = await page.evaluate(() => navigator.userAgent);
const adapter = rendering.adapter;
await browser.close();
srv.close();

const chrome = (ua.match(/Chrome\/(\d+)/) || [])[1];
const run = {
  id,
  label,
  machine: `${adapter}${chrome ? ` · Chrome ${chrome}` : ''}`,
  date: stamp,
  gpujs: JSON.parse(readFileSync(join(ROOT, 'node_modules/gpu.js/package.json'), 'utf8')).version,
  results,
};
writeFileSync(join(SAVED, `${id}.json`), `${JSON.stringify(run, null, 2)}\n`);

// regenerate the index from whatever is on disk, newest first
const files = readdirSync(SAVED).filter(f => f.endsWith('.json')).sort().reverse();
writeFileSync(join(SAVED, 'index.js'),
`/**
 * src/Bench/saved/index.js — GENERATED by scripts/bench-record.mjs.
 *
 * A saved run is a whole table captured on one machine at one moment. The page
 * exists to compare backends, and a table half-measured here and half on
 * someone else's laptop compares nothing — so saved runs are read-only in the
 * UI and the picker chooses WHICH machine you are reading.
 *
 * Never hand-edited: a number in this directory has to have been measured.
 */
${files.map((f, i) => `import r${i} from './${f}';`).join('\n')}

export default [${files.map((_, i) => `r${i}`).join(', ')}];
`);
console.log(`bench-record: wrote ${id}.json (${Object.keys(results).length} rows) and rebuilt the index`);
