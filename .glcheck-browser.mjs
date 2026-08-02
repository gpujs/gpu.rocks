// Drive the two workloads across all gpu.js backends in a real Chrome, with no
// site build: serve the workload modules and the gpu.js browser bundle raw.
import http from 'node:http';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { launch } from './scripts/browser.mjs';

const ROOT = '/Users/fuzzie/Documents/gpu.rocks';
const ids = (process.argv[2] || 'fft,ising').split(',');
const overrides = process.argv[3] || '{}';

const PAGE = `<!doctype html><meta charset=utf-8><body>
<script src="/gpu.js"></script>
<script type="module">
const ids = ${JSON.stringify(ids)};
const over = ${overrides};
window.__out = [];
const log = o => { window.__out.push(o); console.log(JSON.stringify(o)); };
(async () => {
  for (const id of ids) {
    const w = (await import('/w/' + id + '.js')).default;
    const size = { ...w.size, ...(over[id] || {}) };
    const inputs = w.make(size);
    const ref = w.js(size, inputs);
    const refCheck = w.reduce(ref, size);
    log({ id, size, col: 'bare-js', check: refCheck });
    for (const mode of ['webgl', 'webgl2', 'webgpu', 'cpu']) {
      let gpu = null, built = null;
      try {
        gpu = new GPU.GPU({ mode });
        built = await w.gpujs(gpu, size, inputs);
        const out = await built.run();
        const check = w.reduce(out, size);
        // element diff
        const flat = v => (ArrayBuffer.isView(v) ? Array.from(v)
          : Array.isArray(v) && (ArrayBuffer.isView(v[0]) || Array.isArray(v[0])) ? [].concat(...v.map(r => Array.from(r)))
          : Array.from(v));
        const a = flat(ref), b = flat(out);
        let nbad = 0, worst = 0, firstBad = -1;
        for (let i = 0; i < a.length; i++) {
          const d = Math.abs(a[i] - b[i]) / Math.max(Math.abs(a[i]), 1e-6);
          if (d > 1e-3) { nbad++; if (firstBad < 0) firstBad = i; }
          if (d > worst) worst = d;
        }
        log({ id, col: mode, check, rel: Math.abs(check - refCheck) / Math.max(Math.abs(refCheck), 1e-9),
              backend: built.backend && built.backend(), nbad, len: a.length, worst, firstBad,
              sample: firstBad >= 0 ? [a[firstBad], b[firstBad]] : null });
      } catch (e) {
        log({ id, col: mode, error: String(e && e.message || e).slice(0, 300) });
      } finally {
        if (built && built.destroy) built.destroy();
        if (gpu && gpu.destroy) await gpu.destroy();
      }
    }
  }
  window.__done = true;
})().catch(e => { log({ fatal: String(e && e.stack || e).slice(0, 500) }); window.__done = true; });
</script>`;

const srv = http.createServer((q, r) => {
  const u = q.url.split('?')[0];
  if (u === '/') { r.writeHead(200, { 'content-type': 'text/html' }); return r.end(PAGE); }
  if (u === '/gpu.js') {
    r.writeHead(200, { 'content-type': 'text/javascript' });
    return r.end(readFileSync(join(ROOT, 'node_modules/gpu.js/dist/gpu-browser.js')));
  }
  if (u.startsWith('/w/')) {
    const p = join(ROOT, 'src/Bench/workloads', u.slice(3));
    if (existsSync(p)) { r.writeHead(200, { 'content-type': 'text/javascript' }); return r.end(readFileSync(p)); }
  }
  r.writeHead(404); r.end('nope');
});
await new Promise(r => srv.listen(0, r));
const base = `http://localhost:${srv.address().port}`;

const browser = await launch();
const page = await browser.newPage();
page.on('pageerror', e => console.log('PAGEERROR:', String(e.message).slice(0, 300)));
page.on('console', m => { const t = m.text(); if (!t.startsWith('{')) console.log('CONSOLE:', t.slice(0, 300)); });
await page.goto(base, { waitUntil: 'domcontentloaded' });

const deadline = Date.now() + 900000;
while (Date.now() < deadline) {
  // eslint-disable-next-line no-await-in-loop
  if (await page.evaluate(() => window.__done)) break;
  // eslint-disable-next-line no-await-in-loop
  await new Promise(r => setTimeout(r, 2000));
}
const out = await page.evaluate(() => window.__out);
console.log('\n=== RESULTS ===');
for (const o of out) console.log(JSON.stringify(o));
await browser.close();
srv.close();
process.exit(0);
