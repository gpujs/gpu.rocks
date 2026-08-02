// Run the REAL runner.js protocol (2 warm-ups, then timed reps) against the
// two workloads, in a real Chrome on the real GPU.
import http from 'node:http';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { join, extname } from 'node:path';
import { launch } from './scripts/browser.mjs';

const ROOT = '/Users/fuzzie/Documents/gpu.rocks';
const ids = (process.argv[2] || 'fft,ising').split(',');
const runs = Number(process.argv[3] || 4);

const PAGE = `<!doctype html><meta charset=utf-8><body>
<script src="/gpu.js"></script>
<script type="module">
const ids = ${JSON.stringify(ids)};
const RUNS = ${runs};
window.__out = [];
const log = o => { window.__out.push(o); };
(async () => {
  for (const id of ids) {
    const w = (await import(id.startsWith('.') ? '/' + id : '/src/Bench/workloads/' + id + '.js')).default;
    const size = w.size;
    const inputs = w.make(size);
    const ref = w.js(size, inputs);
    const refCheck = w.reduce(ref, size);
    log({ id, col: 'bare-js', checks: [refCheck] });
    for (const mode of ['webgl', 'webgl2', 'webgpu', 'cpu']) {
      let gpu = null, built = null;
      try {
        gpu = new GPU.GPU({ mode });
        built = await w.gpujs(gpu, size, inputs);
        const checks = [];
        for (let i = 0; i < RUNS; i++) {
          const out = await built.run();
          checks.push(w.reduce(out, size));
        }
        log({ id, col: mode, checks, ref: refCheck,
              rels: checks.map(c => Math.abs(c - refCheck) / Math.max(Math.abs(refCheck), 1e-9)),
              backend: built.backend && built.backend() });
      } catch (e) {
        log({ id, col: mode, error: String(e && e.message || e).slice(0, 300) });
      } finally {
        if (built && built.destroy) built.destroy();
        if (gpu && gpu.destroy) await gpu.destroy();
      }
    }
  }
  window.__done = true;
})().catch(e => { log({ fatal: String(e && e.stack || e).slice(0, 600) }); window.__done = true; });
</script>`;

const TYPES = { '.js': 'text/javascript', '.mjs': 'text/javascript' };
const srv = http.createServer((q, r) => {
  const u = decodeURIComponent(q.url.split('?')[0]);
  if (u === '/') { r.writeHead(200, { 'content-type': 'text/html' }); return r.end(PAGE); }
  if (u === '/gpu.js') {
    r.writeHead(200, { 'content-type': 'text/javascript' });
    return r.end(readFileSync(join(ROOT, 'node_modules/gpu.js/dist/gpu-browser.js')));
  }
  const p = join(ROOT, u);
  if (p.startsWith(ROOT) && existsSync(p) && statSync(p).isFile()) {
    r.writeHead(200, { 'content-type': TYPES[extname(p)] || 'application/octet-stream' });
    return r.end(readFileSync(p));
  }
  r.writeHead(404); r.end('nope');
});
await new Promise(r => srv.listen(0, r));
const base = `http://localhost:${srv.address().port}`;

const browser = await launch();
const page = await browser.newPage();
page.on('pageerror', e => console.log('PAGEERROR:', String(e.message).slice(0, 400)));
page.on('console', m => console.log('CONSOLE:', m.text().slice(0, 300)));
await page.goto(base, { waitUntil: 'domcontentloaded' });

const deadline = Date.now() + 1200000;
while (Date.now() < deadline) {
  // eslint-disable-next-line no-await-in-loop
  if (await page.evaluate(() => window.__done)) break;
  // eslint-disable-next-line no-await-in-loop
  await new Promise(r => setTimeout(r, 2000));
}
const out = await page.evaluate(() => window.__out);
const probe = await page.evaluate(() => window.__probe || []);
console.log('=== PROBE ==='); for (const p of probe) console.log(JSON.stringify(p));
console.log('\n=== RESULTS ===');
for (const o of out) console.log(JSON.stringify(o));
await browser.close();
srv.close();
process.exit(0);
