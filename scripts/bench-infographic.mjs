/**
 * scripts/bench-infographic.mjs — turn a saved run into a poster.
 *
 *   node scripts/bench-infographic.mjs                     # newest saved run
 *   node scripts/bench-infographic.mjs --run apple-m1-max
 *   node scripts/bench-infographic.mjs --file out/run.json --out poster.png
 *
 * Writes a PNG and the HTML it was rendered from, so the layout can be edited
 * and re-rendered without touching this file.
 *
 * ── WHY THIS CHART AND NOT A BIG NUMBER ─────────────────────────────────────
 *
 * The obvious infographic for benchmark data is one enormous multiple over a
 * gradient. That is precisely the claim the page exists to argue against: the
 * best row here is 1073x and the worst is 0.58x, and any single headline figure
 * is a choice of workload wearing the costume of a fact.
 *
 * So the SPREAD is the subject. Every workload gets a row on a log axis, sorted
 * by what gpu.js achieved, and the 1x baseline is drawn as a full-height spine.
 * Rows reaching right of the spine are wins; rows that fall short of it sit to
 * its left and are drawn in the losing colour rather than cropped, because a
 * benchmark poster that only shows its wins is an advertisement.
 *
 * Two marks per row, which is the fewest that can carry both findings:
 *
 *   • a filled dot at the best gpu.js backend — what you get for writing a
 *     kernel as a JavaScript function
 *   • a hollow ring at hand-written WebGPU — what the same GPU does without
 *     the library
 *
 * The gap between them IS the cost of the abstraction, made visible as distance
 * rather than asserted as a ratio. On a log axis that gap is the same width
 * everywhere it means the same thing, which is the whole reason for a log axis.
 */
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launch } from './browser.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SAVED = join(ROOT, 'src/Bench/saved');

const argv = process.argv.slice(2);
const arg = n => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : null; };

// ── the run ────────────────────────────────────────────────────────────────
let run;
const fileArg = arg('--file');
if (fileArg) {
  run = JSON.parse(readFileSync(resolve(ROOT, fileArg), 'utf8'));
} else {
  const files = readdirSync(SAVED).filter(f => f.endsWith('.json') && !f.startsWith('.')).sort().reverse();
  const wanted = arg('--run');
  const file = wanted ? files.find(f => f.includes(wanted)) : files[0];
  if (!file) {
    console.error('bench-infographic: no saved run to draw');
    process.exit(1);
  }
  run = JSON.parse(readFileSync(join(SAVED, file), 'utf8'));
}

const { default: workloads } = await import(`${ROOT}/src/Bench/workloads/index.js`);
const meta = new Map(workloads.map(w => [w.id, w]));

const BASE = 'bare-js';
const GPUJS = [
  { id: 'webgpu', label: 'WebGPU' },
  { id: 'webgl2', label: 'WebGL2' },
  { id: 'webgl', label: 'WebGL' },
  { id: 'webasm', label: 'WebASM' },
];

const usable = c => c && typeof c.ms === 'number' && c.ms > 0 && !c.wrong && !c.error;

const rows = [];
for (const [id, cells] of Object.entries(run.results)) {
  const base = cells[BASE];
  if (!usable(base)) continue;
  let best = null;
  for (const col of GPUJS) {
    const c = cells[col.id];
    if (!usable(c)) continue;
    // a backend that quietly degraded is not that backend's result
    if (c.fellBackTo) continue;
    const factor = base.ms / c.ms;
    if (!best || factor > best.factor) best = { factor, label: col.label };
  }
  if (!best) continue;
  const bare = usable(cells['bare-webgpu']) ? base.ms / cells['bare-webgpu'].ms : null;
  rows.push({
    id,
    name: (meta.get(id) || {}).name || id,
    tag: (meta.get(id) || {}).tag || '',
    best: best.factor,
    bestLabel: best.label,
    bare,
  });
}
rows.sort((a, b) => b.best - a.best);
if (!rows.length) {
  console.error('bench-infographic: the run has no usable rows');
  process.exit(1);
}

// ── the numbers the footer quotes ──────────────────────────────────────────
const median = xs => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const tax = rows.filter(r => r.bare).map(r => r.bare / r.best);
const stats = {
  medianBest: median(rows.map(r => r.best)),
  medianTax: tax.length ? median(tax) : null,
  top: rows[0],
  bottom: rows[rows.length - 1],
  losses: rows.filter(r => r.best < 1).length,
};

// ── geometry ───────────────────────────────────────────────────────────────
// A log axis, because the data spans three orders of magnitude and a linear one
// would render everything below 50x as a smear against the left edge.
const W = 1240;
const PAD = 64;
const LABEL_W = 330;
const ROW_H = 30;
const HEAD_H = 250;
const FOOT_H = 250;
const CHART_X = PAD + LABEL_W;
const CHART_W = W - CHART_X - PAD - 70; // room for the value at the right
const H = HEAD_H + rows.length * ROW_H + FOOT_H;

const allFactors = rows.flatMap(r => [r.best, r.bare].filter(Boolean));
const lo = Math.min(0.5, Math.min(...allFactors) * 0.8);
const hi = Math.max(...allFactors) * 1.35;
const l10 = Math.log10;
const x = f => CHART_X + ((l10(f) - l10(lo)) / (l10(hi) - l10(lo))) * CHART_W;

// decade gridlines inside the domain
const ticks = [];
for (let e = Math.floor(l10(lo)); e <= Math.ceil(l10(hi)); e++) {
  const v = 10 ** e;
  if (v >= lo && v <= hi) ticks.push(v);
}

// A decade tick should read "1×", not "1.0×"; a measurement should keep the
// digit that distinguishes 9.7 from 10.
const fmtX = f => {
  if (f >= 10) return f.toFixed(0);
  if (f < 1) return f.toFixed(2);
  return Number.isInteger(f) ? String(f) : f.toFixed(1);
};

// SVG text does not wrap or ellipsise, so the label column has to be budgeted
// by hand. The name gets whatever the tag does not need, and when that leaves
// too little to read, the tag is dropped rather than the name truncated into
// uselessness — the workload's name is the thing a reader is looking for.
const NAME_PX = 7.0;   // 'Avenir Next' at 13.5px, measured against the render
const TAG_PX = 6.3;    // monospace at 10.5px
const fit = (text, px, perChar) => {
  const max = Math.floor(px / perChar);
  return text.length <= max ? text : `${text.slice(0, Math.max(1, max - 1))}…`;
};
const labelFor = row => {
  const tagW = row.tag ? row.tag.length * TAG_PX : 0;
  const nameW = LABEL_W - tagW - 22;
  if (row.tag && nameW >= 120) return { name: fit(row.name, nameW, NAME_PX), tag: row.tag };
  return { name: fit(row.name, LABEL_W - 12, NAME_PX), tag: '' };
};
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// ── the drawing ────────────────────────────────────────────────────────────
const INK = '#f6f7f8';
const MUTED = '#9b94c0';
const TEAL = '#18bc9c';
const AMBER = '#e2b04a';
const PINK = '#ff79c6';
const GRID = 'rgba(158,140,220,.16)';

let svg = '';

// decade gridlines, drawn first so everything sits on top of them
for (const t of ticks) {
  svg += `<line x1="${x(t).toFixed(1)}" y1="${HEAD_H - 22}" x2="${x(t).toFixed(1)}" y2="${HEAD_H + rows.length * ROW_H + 6}" stroke="${GRID}" stroke-width="1"/>`;
  svg += `<text x="${x(t).toFixed(1)}" y="${HEAD_H - 32}" fill="${MUTED}" font-size="12" text-anchor="middle" font-family="ui-monospace, monospace">${fmtX(t)}×</text>`;
}

// The spine: plain JavaScript. Everything is measured from here.
//
// Its label sits BELOW the last row rather than above the first. Above, it ran
// into the machine line in the header — and the header is fixed-height while
// the chart grows with the number of workloads, so the collision would come
// and go with the size of the run rather than staying fixed.
const spine = x(1);
const chartBottom = HEAD_H + rows.length * ROW_H;
svg += `<line x1="${spine.toFixed(1)}" y1="${HEAD_H - 26}" x2="${spine.toFixed(1)}" y2="${chartBottom + 10}" stroke="${PINK}" stroke-width="1.5" stroke-dasharray="3 4" opacity=".75"/>`;
svg += `<text x="${(spine + 8).toFixed(1)}" y="${chartBottom + 28}" fill="${PINK}" font-size="12" font-weight="700" font-family="ui-monospace, monospace">1× — plain JavaScript. Left of this line the GPU lost.</text>`;

rows.forEach((r, i) => {
  const y = HEAD_H + i * ROW_H + ROW_H / 2;
  const win = r.best >= 1;
  const dot = win ? TEAL : PINK;

  const lab = labelFor(r);
  svg += `<text x="${PAD}" y="${(y + 4).toFixed(1)}" fill="${INK}" font-size="13.5" font-family="'Avenir Next','Montserrat',system-ui,sans-serif">${esc(lab.name)}</text>`;
  if (lab.tag) {
    svg += `<text x="${PAD + LABEL_W - 22}" y="${(y + 4).toFixed(1)}" fill="${MUTED}" font-size="10.5" text-anchor="end" font-family="ui-monospace, monospace">${esc(lab.tag)}</text>`;
  }

  // the connector runs from the baseline to the result, so its LENGTH is the
  // speed-up and a reader can compare rows without reading a single number
  const from = Math.min(spine, x(r.best));
  const to = Math.max(spine, x(r.best));
  svg += `<line x1="${from.toFixed(1)}" y1="${y.toFixed(1)}" x2="${to.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${dot}" stroke-width="2" opacity=".35"/>`;

  // hand-written WebGPU: the same hardware without the library
  if (r.bare) {
    svg += `<circle cx="${x(r.bare).toFixed(1)}" cy="${y.toFixed(1)}" r="4.5" fill="none" stroke="${AMBER}" stroke-width="1.6" opacity=".95"/>`;
  }
  svg += `<circle cx="${x(r.best).toFixed(1)}" cy="${y.toFixed(1)}" r="5" fill="${dot}"/>`;
  svg += `<text x="${(W - PAD).toFixed(1)}" y="${(y + 4).toFixed(1)}" fill="${win ? INK : PINK}" font-size="12.5" text-anchor="end" font-weight="700" font-family="ui-monospace, monospace">${fmtX(r.best)}×</text>`;
});

const html = `<div class="poster">
<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="glow1" cx="80%" cy="0%" r="70%">
      <stop offset="0%" stop-color="#2e0741" stop-opacity=".9"/>
      <stop offset="60%" stop-color="#2e0741" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="glow2" cx="8%" cy="12%" r="55%">
      <stop offset="0%" stop-color="#20a4f3" stop-opacity=".13"/>
      <stop offset="60%" stop-color="#20a4f3" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="${W}" height="${H}" fill="#050218"/>
  <rect width="${W}" height="${H}" fill="url(#glow1)"/>
  <rect width="${W}" height="${H}" fill="url(#glow2)"/>

  <text x="${PAD}" y="78" fill="${PINK}" font-size="13" font-weight="800" letter-spacing="3" font-family="ui-monospace, monospace">GPU.JS</text>
  <text x="${PAD}" y="132" fill="${INK}" font-size="44" font-weight="800" letter-spacing="-1" font-family="'Montserrat','Avenir Next',system-ui,sans-serif">The Benchmark Gauntlet</text>
  <text x="${PAD}" y="166" fill="${MUTED}" font-size="15" font-family="'Avenir Next','Montserrat',system-ui,sans-serif">${esc(rows.length)} GPGPU workloads, every one checked against a plain-JavaScript oracle before it was timed.</text>
  <text x="${PAD}" y="192" fill="${MUTED}" font-size="13" font-family="ui-monospace, monospace">${esc(run.machine || '')}${run.date ? ` · ${esc(run.date)}` : ''} · gpu.js ${esc(run.gpujs || '?')}</text>

  <!-- legend, placed where the eye lands before the first row -->
  <circle cx="${W - PAD - 250}" cy="${HEAD_H - 78}" r="5" fill="${TEAL}"/>
  <text x="${W - PAD - 238}" y="${HEAD_H - 74}" fill="${MUTED}" font-size="12" font-family="'Avenir Next',system-ui,sans-serif">best via gpu.js</text>
  <circle cx="${W - PAD - 120}" cy="${HEAD_H - 78}" r="4.5" fill="none" stroke="${AMBER}" stroke-width="1.6"/>
  <text x="${W - PAD - 108}" y="${HEAD_H - 74}" fill="${MUTED}" font-size="12" font-family="'Avenir Next',system-ui,sans-serif">hand-written WebGPU</text>

  ${svg}

  <line x1="${PAD}" y1="${H - FOOT_H + 46}" x2="${W - PAD}" y2="${H - FOOT_H + 46}" stroke="${GRID}"/>

  <text x="${PAD}" y="${H - FOOT_H + 92}" fill="${MUTED}" font-size="11" letter-spacing="2" font-family="ui-monospace, monospace">MEDIAN BEST</text>
  <text x="${PAD}" y="${H - FOOT_H + 128}" fill="${TEAL}" font-size="30" font-weight="800" font-family="'Montserrat',system-ui,sans-serif">${fmtX(stats.medianBest)}×</text>
  <text x="${PAD}" y="${H - FOOT_H + 152}" fill="${MUTED}" font-size="12" font-family="'Avenir Next',system-ui,sans-serif">over plain JavaScript</text>

  <text x="${PAD + 260}" y="${H - FOOT_H + 92}" fill="${MUTED}" font-size="11" letter-spacing="2" font-family="ui-monospace, monospace">SPREAD</text>
  <text x="${PAD + 260}" y="${H - FOOT_H + 128}" fill="${INK}" font-size="30" font-weight="800" font-family="'Montserrat',system-ui,sans-serif">${fmtX(stats.bottom.best)}×–${fmtX(stats.top.best)}×</text>
  <text x="${PAD + 260}" y="${H - FOOT_H + 152}" fill="${MUTED}" font-size="12" font-family="'Avenir Next',system-ui,sans-serif">${esc(stats.bottom.name)} → ${esc(stats.top.name)}</text>

  ${stats.medianTax ? `<text x="${PAD + 620}" y="${H - FOOT_H + 92}" fill="${MUTED}" font-size="11" letter-spacing="2" font-family="ui-monospace, monospace">COST OF THE LIBRARY</text>
  <text x="${PAD + 620}" y="${H - FOOT_H + 128}" fill="${AMBER}" font-size="30" font-weight="800" font-family="'Montserrat',system-ui,sans-serif">${stats.medianTax.toFixed(1)}×</text>
  <text x="${PAD + 620}" y="${H - FOOT_H + 152}" fill="${MUTED}" font-size="12" font-family="'Avenir Next',system-ui,sans-serif">hand-written WebGPU over gpu.js, median</text>` : ''}

  <text x="${PAD + 940}" y="${H - FOOT_H + 92}" fill="${MUTED}" font-size="11" letter-spacing="2" font-family="ui-monospace, monospace">GPU LOSES</text>
  <text x="${PAD + 940}" y="${H - FOOT_H + 128}" fill="${PINK}" font-size="30" font-weight="800" font-family="'Montserrat',system-ui,sans-serif">${stats.losses}</text>
  <text x="${PAD + 940}" y="${H - FOOT_H + 152}" fill="${MUTED}" font-size="12" font-family="'Avenir Next',system-ui,sans-serif">of ${rows.length} workloads</text>

  <text x="${PAD}" y="${H - 34}" fill="${MUTED}" font-size="12" font-family="'Avenir Next',system-ui,sans-serif">Every number here was measured in a browser and can be reproduced in one — gpu.rocks/benchmark</text>
</svg>
</div>`;

const page = `<style>
  html, body { margin: 0; background: #050218; }
  .poster { width: ${W}px; }
  svg { display: block; }
</style>
${html}`;

// ── render ─────────────────────────────────────────────────────────────────
const outArg = arg('--out') || join(ROOT, 'out', `gauntlet-${run.id || 'run'}.png`);
mkdirSync(dirname(resolve(ROOT, outArg)), { recursive: true });
const htmlPath = resolve(ROOT, outArg).replace(/\.png$/, '.html');
writeFileSync(htmlPath, page);

const browser = await launch();
const p = await browser.newPage();
await p.setViewport({ width: W, height: Math.min(H, 4000), deviceScaleFactor: 2 });
await p.setContent(page, { waitUntil: 'load' });
// let the font stack settle before the shot, or the first paint measures a
// fallback face and the labels shift after capture
await p.evaluateHandle('document.fonts.ready');
const el = await p.$('.poster');
await el.screenshot({ path: resolve(ROOT, outArg) });
await browser.close();

console.log(`bench-infographic: ${resolve(ROOT, outArg)}  (${W}×${H} at 2x)`);
console.log(`                   ${htmlPath}  — edit and re-render`);
console.log(`  ${rows.length} rows · median ${fmtX(stats.medianBest)}× · spread ${fmtX(stats.bottom.best)}×–${fmtX(stats.top.best)}×${stats.medianTax ? ` · library costs ${stats.medianTax.toFixed(1)}×` : ''}`);
