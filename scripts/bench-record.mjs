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
// --out sends the recorded run somewhere other than the repo, which is how the
// container hands its result back through a mounted volume. Writing outside
// src/Bench/saved deliberately does NOT regenerate the index: that file is a
// manifest of the runs this repo ships, and a run measured on someone else's
// machine is not one of them until a human puts it there.
const outDirArgIndex = process.argv.indexOf('--out');

const argv = process.argv.slice(2);
const arg = n => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : null; };
const label = arg('--label') || 'unlabelled run';
// --only re-measures named rows and MERGES them into the newest saved run,
// rather than spending a quarter of an hour to change one number. The merge is
// only honest while the rest of the table still describes the same machine and
// the same library, so both are checked below and a mismatch refuses.
const only = (arg('--only') || '').split(',').map(x => x.trim()).filter(Boolean);
// --columns is the other axis: measure named COLUMNS across every row and
// merge just those cells. Adding a backend to the table needs 30 new cells,
// not 30 re-measured rows. The plain-JS baseline is measured alongside no
// matter what — the runner needs it for the checksum comparison that decides
// whether a new column is right, not merely fast — but it is not merged; it is
// compared against the stored one, and a large drift means the machine is not
// the machine the rest of the table came from.
const columnsOnly = (arg('--columns') || '').split(',').map(x => x.trim()).filter(Boolean);
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
// A renderer that dies mid-run leaves puppeteer talking to a detached frame,
// and the stack for that says nothing about what was being measured. Record
// the last known status so a crash names the workload it happened on.
let lastStatus = '(before the first row)';
let crashed = null;
page.on('error', e => {
  crashed = String((e && e.message) || e);
  console.error(`bench-record: RENDERER CRASH during "${lastStatus}" — ${crashed}`);
});
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
if (columnsOnly.length) {
  const found = await page.evaluate(wanted => {
    const boxes = [...document.querySelectorAll('.opt.cols input')];
    const seen = [];
    boxes.forEach(i => {
      const id = i.getAttribute('data-col');
      if (!id) return;
      seen.push(id);
      const want = wanted.includes(id);
      if (!i.disabled && i.checked !== want) i.click();
    });
    return seen;
  }, columnsOnly);
  const missing = columnsOnly.filter(c => !found.includes(c));
  if (missing.length) {
    console.error(`bench-record: no such column(s): ${missing.join(', ')} (have: ${found.join(', ')})`);
    process.exit(1);
  }
  console.log(`bench-record: columns ${columnsOnly.join(', ')} (+ baseline) across every row`);
}
await new Promise(r => setTimeout(r, 200));

if (only.length) {
  const missing = await page.evaluate(
    ids => ids.filter(id => !document.querySelector(`tr[data-workload="${id}"] .run-one`)),
    only
  );
  if (missing.length) {
    console.error(`bench-record: no such row(s): ${missing.join(', ')}`);
    process.exit(1);
  }
  console.log(`bench-record: ${only.length} row(s) — ${only.join(', ')} — running…`);
  for (const id of only) {
    // eslint-disable-next-line no-await-in-loop
    await page.evaluate(rid => document.querySelector(`tr[data-workload="${rid}"] .run-one`).click(), id);
    // eslint-disable-next-line no-await-in-loop
    await new Promise(r => setTimeout(r, 500));
  }
} else {
  const rows = await page.evaluate(() => document.querySelectorAll('#root tbody tr').length);
  console.log(`bench-record: ${rows} workload(s), running…`);
  await page.click('.toolbar .btn.primary');
  await page.waitForSelector('.runwarn', { timeout: 5000 });
  await page.click('.runwarn .btn.primary');
}

// A full pass is half an hour and longer on a slow machine, so the cap is
// wall-clock and generous rather than a poll count that quietly expired first.
const deadline = Date.now() + 150 * 60 * 1000;
let done = false;
for (let i = 0; !done; i++) {
  if (Date.now() > deadline) break;
  // eslint-disable-next-line no-await-in-loop
  await new Promise(r => setTimeout(r, 3000));
  if (crashed) break;
  let s;
  try {
    // eslint-disable-next-line no-await-in-loop
    s = await page.evaluate(() => {
      const m = performance.memory;
      return `${window.__benchStatus || ''}\u0000${m ? Math.round(m.usedJSHeapSize / 1048576) : -1}`;
    });
  } catch (e) {
    // the frame went away between polls — the crash handler above has the why
    crashed = crashed || String((e && e.message) || e);
    break;
  }
  const [statusText, heapMb] = s.split('\u0000');
  lastStatus = statusText;
  if (/^done|^stopped/.test(statusText)) done = true;
  else if (i % 20 === 0) console.log(`  [${new Date().toTimeString().slice(0, 5)}] ${statusText}  heap ${heapMb}MB`);
}
if (!done) {
  console.error(crashed
    ? `bench-record: renderer died during "${lastStatus}"; nothing written`
    : 'bench-record: timed out; nothing written');
  process.exit(1);
}

const results = await page.evaluate(() => window.__benchResults);
const ua = await page.evaluate(() => navigator.userAgent);
const adapter = rendering.adapter;
await browser.close();
srv.close();

const chrome = (ua.match(/Chrome\/(\d+)/) || [])[1];
const machine = `${adapter}${chrome ? ` · Chrome ${chrome}` : ''}`;
const gpujs = JSON.parse(readFileSync(join(ROOT, 'node_modules/gpu.js/package.json'), 'utf8')).version;

let outId = id;
let run = { id, label, machine, date: stamp, gpujs, results };
let driftReport = null;
const outIdGuess = (readdirSync(SAVED).filter(f => f.endsWith('.json')).sort().reverse()[0] || 'run').replace(/\.json$/, '');

if (only.length) {
  const existing = readdirSync(SAVED).filter(f => f.endsWith('.json')).sort().reverse()[0];
  if (!existing) {
    console.error('bench-record: --only needs a saved run to merge into; record a full one first');
    process.exit(1);
  }
  const prev = JSON.parse(readFileSync(join(SAVED, existing), 'utf8'));
  // A saved run is one table from one machine. Patching a row into a table
  // measured on different hardware or a different gpu.js would produce a
  // column of numbers that never coexisted.
  if (prev.machine !== machine || prev.gpujs !== gpujs) {
    console.error('bench-record: refusing to merge into a run from elsewhere');
    console.error(`  saved:   ${prev.machine} · gpu.js ${prev.gpujs}`);
    console.error(`  current: ${machine} · gpu.js ${gpujs}`);
    process.exit(1);
  }
  const measured = Object.fromEntries(only.filter(k => results[k]).map(k => [k, results[k]]));
  const absent = only.filter(k => !results[k]);
  if (absent.length) {
    console.error(`bench-record: no results came back for ${absent.join(', ')}; nothing written`);
    process.exit(1);
  }
  outId = prev.id;
  run = {
    ...prev,
    results: { ...prev.results, ...measured },
    // say so in the file: the table is no longer one continuous sitting
    patched: [...(prev.patched || []), { rows: only, date: stamp }],
  };
  console.log(`bench-record: merged ${only.join(', ')} into ${prev.id}`);
}

if (columnsOnly.length) {
  const existing = readdirSync(SAVED).filter(f => f.endsWith('.json')).sort().reverse()[0];
  if (!existing) {
    console.error('bench-record: --columns needs a saved run to merge into; record a full one first');
    process.exit(1);
  }
  const prev = JSON.parse(readFileSync(join(SAVED, existing), 'utf8'));
  if (prev.machine !== machine) {
    console.error('bench-record: refusing to merge a column measured on other hardware');
    console.error(`  saved:   ${prev.machine}`);
    console.error(`  current: ${machine}`);
    process.exit(1);
  }

  // The baseline was re-measured on this pass. It is not merged, but a big
  // move in it means the machine is busier or slower than when the rest of the
  // table was taken, and the new column would be unfairly scaled against
  // numbers it never ran beside.
  const drift = [];
  for (const [id, cells] of Object.entries(results)) {
    const before = prev.results[id] && prev.results[id]['bare-js'];
    const after = cells['bare-js'];
    if (before && after && before.ms > 0 && after.ms > 0) {
      drift.push({ id, pct: (100 * (after.ms - before.ms)) / before.ms });
    }
  }
  drift.sort((a, b) => Math.abs(b.pct) - Math.abs(a.pct));

  // A refused merge must not cost the measurement. Anything that came back is
  // written aside first, so the decision can be reviewed without re-running.
  const pendingPath = join(SAVED, `.pending-${outIdGuess}.json`);
  writeFileSync(pendingPath, `${JSON.stringify({ machine, gpujs, date: stamp, columns: columnsOnly, results }, null, 2)}\n`);

  if (drift.length) {
    // MEDIAN, not mean: the mean is dragged by a single anomalous row, and one
    // row behaving differently is a fact about that row, not evidence the
    // machine changed underneath the whole table. The median says whether
    // conditions moved; the outlier list says which rows to distrust.
    const sortedAbs = drift.map(d => Math.abs(d.pct)).sort((a, b) => a - b);
    const mid = sortedAbs.length >> 1;
    const median = sortedAbs.length % 2 ? sortedAbs[mid] : (sortedAbs[mid - 1] + sortedAbs[mid]) / 2;
    const outliers = drift.filter(d => Math.abs(d.pct) > 50);
    console.log(`bench-record: baseline drift — median ${median.toFixed(1)}%, worst ${drift[0].id} ${drift[0].pct.toFixed(1)}%`);
    if (outliers.length) {
      console.log(`bench-record: ${outliers.length} row(s) over 50%: ${outliers.map(d => `${d.id} ${d.pct.toFixed(0)}%`).join(', ')}`);
    }
    if (median > 15 && !argv.includes('--force')) {
      console.error('bench-record: the machine itself moved; the merge would compare numbers taken under different conditions');
      console.error(`  fresh results kept at ${pendingPath} — --force to merge anyway`);
      process.exit(1);
    }
    driftReport = { median: +median.toFixed(1), outliers: outliers.map(d => ({ id: d.id, pct: +d.pct.toFixed(0) })) };
  }

  const merged = { ...prev.results };
  let cellCount = 0;
  const fellBack = [];
  for (const [id, cells] of Object.entries(results)) {
    if (!merged[id]) continue; // measured a row this table does not carry
    const add = {};
    for (const c of columnsOnly) {
      if (!cells[c]) continue;
      add[c] = cells[c];
      cellCount++;
      if (cells[c].fellBackTo) fellBack.push(`${id}->${cells[c].fellBackTo}`);
      if (cells[c].wrong) fellBack.push(`${id}:WRONG`);
    }
    merged[id] = { ...merged[id], ...add };
  }
  outId = prev.id;
  run = {
    ...prev,
    results: merged,
    // gpujs is stamped per patch: this column came from a branch build, and a
    // reader comparing it against the rest of the table should be able to see
    // that from the file rather than from a commit message.
    patched: [...(prev.patched || []), { columns: columnsOnly, date: stamp, gpujs, baselineDrift: driftReport }],
  };
  console.log(`bench-record: merged ${cellCount} cell(s) of ${columnsOnly.join(', ')} into ${prev.id}`);
  if (fellBack.length) console.log(`bench-record: note — ${fellBack.length} did not run natively: ${fellBack.join(', ')}`);
}

const outDir = outDirArgIndex >= 0 ? process.argv[outDirArgIndex + 1] : SAVED;
const outPath = join(outDir, `${outId}.json`);
writeFileSync(outPath, `${JSON.stringify(run, null, 2)}\n`);
if (outDir !== SAVED) {
  console.log(`bench-record: wrote ${outPath} (${Object.keys(run.results).length} rows)`);
  process.exit(0);
}

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
console.log(`bench-record: wrote ${outId}.json (${Object.keys(run.results).length} rows) and rebuilt the index`);
