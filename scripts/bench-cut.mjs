/**
 * scripts/bench-cut.mjs — which rows is the table better off without?
 *
 * The suite grew to thirty-four workloads by asking "is this a real thing a GPU
 * does?", which is the wrong question for a table someone reads. The right one
 * is "does this row say something no other row says?" — and that is answerable
 * from a recorded run rather than from opinion, which is what this script does.
 *
 *   node scripts/bench-cut.mjs [--run <id>] [--verbose]
 *
 * It reports, per row, three things a cut can be argued from:
 *
 *   BROKEN     a column that is WRONG or errored. The page's claim is that
 *              every answer is checked; a row that cannot pass its own check
 *              is not a measurement, it is an open bug on display.
 *
 *   COST       the wall-clock the row spends, as a share of the whole run.
 *              This is what a cut actually buys, and it is wildly uneven —
 *              a handful of rows own most of the half hour.
 *
 *   ECHO       its nearest neighbour in speed-up space. Each row becomes the
 *              vector of log10 speed-ups over the plain-JS baseline across the
 *              GPU columns; that vector IS the row's finding, because it is
 *              the shape of the bars a reader sees. Two rows whose vectors sit
 *              within ~0.1 of each other (a 25% difference in every column)
 *              teach the same lesson twice, and the second one is only costing
 *              the reader time.
 *
 * Nothing here deletes anything. It ranks candidates and shows the working;
 * which rows go is a judgement about what the table is for.
 */
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SAVED = join(ROOT, 'src/Bench/saved');

const argv = process.argv.slice(2);
const arg = n => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : null; };
const verbose = argv.includes('--verbose');

const files = readdirSync(SAVED).filter(f => f.endsWith('.json')).sort().reverse();
const wanted = arg('--run');
const file = wanted ? files.find(f => f.includes(wanted)) : files[0];
if (!file) {
  console.error(`bench-cut: no saved run${wanted ? ` matching "${wanted}"` : ''} in ${SAVED}`);
  process.exit(1);
}
const run = JSON.parse(readFileSync(join(SAVED, file), 'utf8'));

const { default: workloads } = await import(`${ROOT}/src/Bench/workloads/index.js`);
const { SIGNATURE_IDS } = await import(`${ROOT}/src/Bench/signature.js`);
const meta = new Map(workloads.map(w => [w.id, w]));

const BASELINE = 'bare-js';
// The GPU columns only. 'cpu' (gpu.js transpiling to a JS loop) is excluded
// from the fingerprint on purpose: it measures the transpiler, and including
// it would let two rows look different because one confuses gpu.js's CPU
// backend, which is not a fact about GPUs.
const AXES = ['webgpu', 'webgl2', 'webgl', 'bare-webgpu'];

const rows = [];
for (const [id, cells] of Object.entries(run.results)) {
  const base = cells[BASELINE];
  // > 0, not just a number: `undersized` has a 0.0 ms baseline, and a falsy
  // check there erased the whole row rather than reporting it.
  const baseMs = base && typeof base.ms === 'number' ? base.ms : null;
  const hasBase = typeof baseMs === 'number' && baseMs > 0;

  const broken = [];
  for (const [col, cell] of Object.entries(cells)) {
    if (!cell) continue;
    if (cell.wrong) broken.push(`${col}:WRONG`);
    else if (cell.error) broken.push(`${col}:err`);
  }

  // Wall clock: every column pays warm-ups plus reps, and the two JS columns
  // are where the half hour actually goes.
  let cost = 0;
  for (const cell of Object.values(cells)) {
    if (cell && typeof cell.ms === 'number') cost += cell.ms * ((cell.reps || 1) + 2);
  }

  const speedups = {};
  const vec = [];
  for (const col of AXES) {
    const c = cells[col];
    const ok = hasBase && c && typeof c.ms === 'number' && c.ms > 0 && !c.wrong && !c.error;
    speedups[col] = ok ? baseMs / c.ms : null;
    vec.push(ok ? Math.log10(baseMs / c.ms) : null);
  }

  rows.push({
    id,
    name: (meta.get(id) || {}).name || id,
    signature: SIGNATURE_IDS.includes(id),
    baseMs,
    broken,
    cost,
    speedups,
    vec,
  });
}

// Distance over the axes both rows actually measured, so a row that declines
// one backend is still comparable on the rest. Fewer than two shared axes is
// not a comparison, so it reports no neighbour rather than a flattering one.
function distance(a, b) {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < a.vec.length; i++) {
    if (a.vec[i] === null || b.vec[i] === null) continue;
    sum += (a.vec[i] - b.vec[i]) ** 2;
    n++;
  }
  return n < 2 ? null : Math.sqrt(sum / n);
}

for (const r of rows) {
  let best = null;
  for (const o of rows) {
    if (o.id === r.id) continue;
    const d = distance(r, o);
    if (d === null) continue;
    if (!best || d < best.d) best = { d, id: o.id, signature: o.signature };
  }
  r.echo = best;
}

const ECHO = Number(arg('--echo') || 0.12);

// Mutual echoes have to be resolved, not listed. erosion echoes jacobi AND
// jacobi echoes erosion, so a flat candidate list proposed cutting both — which
// deletes the finding instead of de-duplicating it. Union-find groups every row
// that is within ECHO of another, and exactly one survives each group.
const parent = new Map(rows.map(r => [r.id, r.id]));
const find = x => (parent.get(x) === x ? x : (parent.set(x, find(parent.get(x))), parent.get(x)));
for (let i = 0; i < rows.length; i++) {
  for (let j = i + 1; j < rows.length; j++) {
    const d = distance(rows[i], rows[j]);
    if (d !== null && d < ECHO) parent.set(find(rows[i].id), find(rows[j].id));
  }
}
const clusters = new Map();
for (const r of rows) {
  const k = find(r.id);
  if (!clusters.has(k)) clusters.set(k, []);
  clusters.get(k).push(r);
}

// A tiny baseline means the row is timing the harness. sizeExempt suppresses
// the sizing gate for rows whose smallness IS the finding — but it cannot tell
// those apart from rows that are simply too small to say anything.
const TINY_MS = 50;

const totalCost = rows.reduce((a, r) => a + r.cost, 0);
const fmt = (x, d = 1) => (x === null || x === undefined ? '\u2014' : x.toFixed(d));

console.log(`saved run: ${run.label} \u00b7 ${run.machine} \u00b7 gpu.js ${run.gpujs}`);
console.log(`${rows.length} rows, ${(totalCost / 1000 / 60).toFixed(1)} min of measured work\n`);

console.log('  row                        base     WebGPU  WebGL2   WebGL    bare   share   nearest echo');
for (const r of [...rows].sort((a, b) => b.cost - a.cost)) {
  const s = r.speedups;
  console.log(
    `  ${(r.signature ? '*' : ' ') + r.id.padEnd(25)}` +
    `${fmt(r.baseMs, 0).padStart(6)}ms` +
    `${fmt(s.webgpu).padStart(9)}x${fmt(s.webgl2).padStart(8)}x${fmt(s.webgl).padStart(8)}x${fmt(s['bare-webgpu']).padStart(8)}x` +
    `${(100 * r.cost / totalCost).toFixed(1).padStart(7)}%  ` +
    `${r.echo ? `${r.echo.id} (${r.echo.d.toFixed(2)})` : '\u2014'}`
  );
  if (r.broken.length) console.log(`   ${' '.repeat(25)}^ ${r.broken.join('  ')}`);
  if (r.baseMs !== null && r.baseMs < TINY_MS) {
    console.log(`   ${' '.repeat(25)}^ baseline ${fmt(r.baseMs, 1)}ms \u2014 timing the harness, not the kernel` +
      `${(meta.get(r.id) || {}).sizeExempt ? ' (sizeExempt hides this)' : ''}`);
  }
}

console.log('\n* = signature row (brief mode). Cut candidates, most arguable first:\n');

const candidates = [];
for (const r of rows) {
  if (r.broken.length) candidates.push({ r, why: `fails its own check \u2014 ${r.broken.join(', ')}`, rank: 0 });
}
// Single-linkage chains: a—b and b—c merge even when a and c are far apart, so
// the group's DIAMETER is reported next to it. A tight group is a real echo; a
// wide one is a chain and the cut needs looking at rather than applying.
const diameter = group => {
  let d = 0;
  for (let i = 0; i < group.length; i++) {
    for (let j = i + 1; j < group.length; j++) {
      const x = distance(group[i], group[j]);
      if (x !== null && x > d) d = x;
    }
  }
  return d;
};
for (const group of clusters.values()) {
  if (group.length < 2) continue;
  const dia = diameter(group);
  // Nothing is protected. Signature membership is derived from these same
  // numbers now, so it is not independent evidence that a row earns its place —
  // a row can top the flattery ranking and still be an echo of the row above it.
  // The survivor is simply the cheapest, since they say the same thing.
  const keep = [...group].sort((a, b) => a.cost - b.cost)[0];
  for (const r of group) {
    if (r.id === keep.id || r.broken.length) continue;
    candidates.push({
      r,
      why: `echoes ${keep.id} (group \u00f8${dia.toFixed(2)}: ${group.map(g => g.id).join(', ')})`,
      rank: 1,
    });
  }
}
for (const r of rows) {
  if (candidates.some(c => c.r.id === r.id)) continue;
  if (r.baseMs !== null && r.baseMs < TINY_MS) {
    candidates.push({ r, why: `baseline ${fmt(r.baseMs, 1)}ms \u2014 measures the harness`, rank: 2 });
  } else if (r.cost / totalCost > 0.1) {
    candidates.push({ r, why: `${(100 * r.cost / totalCost).toFixed(0)}% of the run for one row`, rank: 3 });
  }
}
candidates.sort((a, b) => a.rank - b.rank || b.r.cost - a.r.cost);
if (!candidates.length) console.log('  none \u2014 every row is checked, distinct and affordable');
for (const c of candidates) {
  console.log(`  ${c.r.id.padEnd(24)} ${c.why}`);
  if (verbose) console.log(`  ${' '.repeat(24)} ${c.r.name} \u00b7 ${(c.r.cost / 1000).toFixed(1)}s`);
}

// The signature list promises that no two of its ten say the same thing. That
// is a claim about the numbers, so check it against them: two signature rows
// inside ECHO of each other means brief mode spends two of its ten slots on one
// lesson, which is a fact about signature.js rather than a row to cut.
const sigEcho = [];
for (let i = 0; i < rows.length; i++) {
  for (let j = i + 1; j < rows.length; j++) {
    if (!rows[i].signature || !rows[j].signature) continue;
    const d = distance(rows[i], rows[j]);
    if (d !== null && d < ECHO) sigEcho.push([rows[i].id, rows[j].id, d]);
  }
}
if (sigEcho.length) {
  console.log('\nSIGNATURE ROWS THAT ECHO EACH OTHER (brief mode teaches one lesson twice):');
  for (const [a, b, d] of sigEcho.sort((x, y) => x[2] - y[2])) {
    console.log(`  ${a} \u2194 ${b}  (${d.toFixed(2)})`);
  }
}

// Signature rows are never proposed above, so a broken or harness-timing one
// would pass silently. Say so instead.
const sigTrouble = rows.filter(r => r.signature && (r.broken.length || (r.baseMs !== null && r.baseMs < TINY_MS)));
if (sigTrouble.length) {
  console.log('\nSIGNATURE ROWS IN TROUBLE (brief mode shows these):');
  for (const r of sigTrouble) {
    console.log(`  ${r.id.padEnd(24)} ${r.broken.length ? r.broken.join(', ') : `baseline ${fmt(r.baseMs, 1)}ms`}`);
  }
}

console.log(
  `\ncutting all ${candidates.length} would save ` +
  `${(candidates.reduce((a, c) => a + c.r.cost, 0) / 1000 / 60).toFixed(1)} min of the ` +
  `${(totalCost / 1000 / 60).toFixed(1)} min run`
);
