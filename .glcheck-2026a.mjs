// Reproduce the WebGL1 column in node via gpu.js's headlessgl backend.
import { GPU } from 'gpu.js';

const ROOT = '/Users/fuzzie/Documents/gpu.rocks';
const id = process.argv[2];
const w = (await import(`${ROOT}/src/Bench/workloads/${id}.js`)).default;

const size = { ...w.size };
for (const a of process.argv.slice(3)) {
  const [k, v] = a.split('=');
  size[k] = Number(v);
}
console.log('size', size);

const inputs = w.make(size);
const ref = w.js(size, inputs);
const refCheck = w.reduce(ref, size);
console.log('js check      ', refCheck);

const gpu = new GPU({ mode: 'headlessgl' });
const built = await w.gpujs(gpu, size, inputs);
const out = await built.run();
const gotCheck = w.reduce(out, size);
console.log('headlessgl    ', gotCheck, 'backend:', built.backend && built.backend());
console.log('rel diff      ', Math.abs(gotCheck - refCheck) / Math.max(Math.abs(refCheck), 1e-9));

const flat = v => (ArrayBuffer.isView(v) ? Array.from(v) : Array.isArray(v) && ArrayBuffer.isView(v[0]) ? v.flatMap(r => Array.from(r)) : Array.isArray(v) && Array.isArray(v[0]) ? v.flat() : Array.from(v));
const a = flat(ref);
const b = flat(out);
console.log('lengths', a.length, b.length);
let nbad = 0;
let worst = 0;
const first = [];
for (let i = 0; i < a.length; i++) {
  const d = Math.abs(a[i] - b[i]) / Math.max(Math.abs(a[i]), 1e-6);
  if (d > 1e-3) {
    nbad++;
    if (first.length < 12) first.push([i, a[i], b[i]]);
  }
  if (d > worst) worst = d;
}
console.log(`elements differing >1e-3 rel: ${nbad} / ${a.length}, worst rel ${worst}`);
console.log('first offenders:', JSON.stringify(first));

if (built.destroy) built.destroy();
process.exit(0);
