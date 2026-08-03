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
// Three NAMED series, not a best-of. A "best available backend" mark is a
// number without an author: two rows sitting side by side could be WebGPU and
// WebGL2 and the poster would not say so, and the reader cannot tell which
// backend to reach for. Naming them costs nothing and answers that.
//
//   WebGPU via gpu.js   — the backend you would actually target today
//   WebASM via gpu.js   — the same kernel with no GPU underneath it at all
//   WebGPU hand-written — the same hardware without the library

const usable = c => c && typeof c.ms === 'number' && c.ms > 0 && !c.wrong && !c.error;

const rows = [];
for (const [id, cells] of Object.entries(run.results)) {
  const base = cells[BASE];
  if (!usable(base)) continue;
  // a backend that quietly degraded is not that backend's result
  const gp = cells.webgpu;
  if (!usable(gp) || gp.fellBackTo) continue;
  const best = { factor: base.ms / gp.ms };
  const bare = usable(cells['bare-webgpu']) ? base.ms / cells['bare-webgpu'].ms : null;
  // Only where WebAssembly actually compiled the kernel. A degraded cell holds
  // the cpu backend's time, and plotting that as a WebASM result would credit
  // the mark for work it did not do.
  const wa = cells.webasm;
  const webasm = usable(wa) && !wa.fellBackTo ? base.ms / wa.ms : null;
  rows.push({
    id,
    name: (meta.get(id) || {}).name || id,
    tag: (meta.get(id) || {}).tag || '',
    group: (meta.get(id) || {}).group || 'other',
    baseMs: base.ms,
    best: best.factor,
    bare,
    webasm,
    // what writing the kernel in JavaScript instead of WGSL cost on this row
    tax: bare ? gp.ms / cells['bare-webgpu'].ms : null,
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
const wasmRows = rows.filter(r => r.webasm);
const stats = {
  medianGpu: median(rows.map(r => r.best)),
  medianTax: tax.length ? median(tax) : null,
  medianWasm: wasmRows.length ? median(wasmRows.map(r => r.webasm)) : null,
  wasmBest: wasmRows.length ? Math.max(...wasmRows.map(r => r.webasm)) : null,
  wasmCount: wasmRows.length,
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
// the three panels below the main chart
const PANEL_H = 340;
const CHART_X = PAD + LABEL_W;
const CHART_W = W - CHART_X - PAD - 70; // room for the value at the right
const H = HEAD_H + rows.length * ROW_H + PANEL_H + FOOT_H;

const allFactors = rows.flatMap(r => [r.best, r.bare, r.webasm].filter(Boolean));
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
const fmtMs = ms => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(0)} ms`);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// ── the drawing ────────────────────────────────────────────────────────────
const INK = '#f6f7f8';
const MUTED = '#9b94c0';
const TEAL = '#18bc9c';
const AMBER = '#e2b04a';
const PINK = '#ff79c6';
const BLUE = '#20a4f3';
const GRID = 'rgba(158,140,220,.16)';

// The footer is generated rather than hand-placed, so a block can be added or
// dropped without four x-coordinates needing to be recomputed by hand.
const foot = [
  { k: 'WEBGPU VIA GPU.JS', v: `${fmtX(stats.medianGpu)}×`, c: TEAL, s: 'median over plain JavaScript' },
  { k: 'SPREAD', v: `${fmtX(stats.bottom.best)}×–${fmtX(stats.top.best)}×`, c: INK, s: `${stats.bottom.name} → ${stats.top.name}` },
  stats.medianWasm
    ? { k: 'WEBASM VIA GPU.JS', v: `${fmtX(stats.medianWasm)}×`, c: BLUE, s: `median with no GPU · best ${fmtX(stats.wasmBest)}×` }
    : null,
  stats.medianTax
    ? { k: 'COST OF THE LIBRARY', v: `${stats.medianTax.toFixed(1)}×`, c: AMBER, s: 'hand-written WebGPU over gpu.js' }
    : null,
].filter(Boolean);

let footBlocks = '';
foot.forEach((f, i) => {
  const bx = PAD + (i * (W - PAD * 2)) / foot.length;
  footBlocks += `<text x="${bx.toFixed(0)}" y="${H - FOOT_H + 92}" fill="${MUTED}" font-size="11" letter-spacing="2" font-family="ui-monospace, monospace">${esc(f.k)}</text>`;
  footBlocks += `<text x="${bx.toFixed(0)}" y="${H - FOOT_H + 128}" fill="${f.c}" font-size="27" font-weight="800" font-family="'Montserrat',system-ui,sans-serif">${esc(f.v)}</text>`;
  footBlocks += `<text x="${bx.toFixed(0)}" y="${H - FOOT_H + 152}" fill="${MUTED}" font-size="11.5" font-family="'Avenir Next',system-ui,sans-serif">${esc(f.s)}</text>`;
});

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
// "gpu.js lost", not "the GPU lost". The two marks that can fall left of this
// line are both gpu.js — the WebGPU dot and the WebASM diamond — and the
// hand-written ring can sit RIGHT of it on the very same row, which is exactly
// what Smith-Waterman does. Saying the GPU lost there would blame the hardware
// for something the programming model did.
svg += `<text x="${(spine + 8).toFixed(1)}" y="${chartBottom + 28}" fill="${PINK}" font-size="12" font-weight="700" font-family="ui-monospace, monospace">1× — plain JavaScript. Left of this line gpu.js lost to it.</text>`;

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

  // WebAssembly: the same kernel with no GPU underneath it at all. A diamond
  // rather than a third circle — the marks should be told apart by shape as
  // well as by hue, which also survives being printed or read by someone who
  // does not separate teal from blue.
  if (r.webasm) {
    const wx = x(r.webasm);
    svg += `<path d="M ${(wx).toFixed(1)} ${(y - 4.6).toFixed(1)} L ${(wx + 4.6).toFixed(1)} ${y.toFixed(1)} L ${(wx).toFixed(1)} ${(y + 4.6).toFixed(1)} L ${(wx - 4.6).toFixed(1)} ${y.toFixed(1)} Z" fill="${BLUE}" opacity=".95"/>`;
  }

  // hand-written WebGPU: the same hardware without the library
  if (r.bare) {
    svg += `<circle cx="${x(r.bare).toFixed(1)}" cy="${y.toFixed(1)}" r="4.5" fill="none" stroke="${AMBER}" stroke-width="1.6" opacity=".95"/>`;
  }
  svg += `<circle cx="${x(r.best).toFixed(1)}" cy="${y.toFixed(1)}" r="5" fill="${dot}"/>`;
  svg += `<text x="${(W - PAD).toFixed(1)}" y="${(y + 4).toFixed(1)}" fill="${win ? INK : PINK}" font-size="12.5" text-anchor="end" font-weight="700" font-family="ui-monospace, monospace">${fmtX(r.best)}×</text>`;
});


// ── three panels: the questions the main chart cannot answer ───────────────
// It ranks workloads. It cannot say whether YOUR workload is big enough to
// bother, what KIND of work the ranking reflects, or whether the single
// library-tax figure in the footer is a constant or an average of extremes.
const PANEL_Y = HEAD_H + rows.length * ROW_H + 92;
const PANEL_W = (W - PAD * 2 - 64) / 3;
const PANEL_BODY = 205;
let panels = '';

const panelHead = (px, title, sub) =>
  `<text x="${px.toFixed(0)}" y="${PANEL_Y}" fill="${INK}" font-size="15" font-weight="800" font-family="'Montserrat',system-ui,sans-serif">${esc(title)}</text>` +
  `<text x="${px.toFixed(0)}" y="${PANEL_Y + 20}" fill="${MUTED}" font-size="11.5" font-family="'Avenir Next',system-ui,sans-serif">${esc(sub)}</text>`;

// ── A. is it worth it? duration against speed-up, both log ─────────────────
{
  const px = PAD;
  const top = PANEL_Y + 40;
  const pts = rows.map(r => ({ x: r.baseMs, y: r.best }));
  const xs = pts.map(p => p.x); const ys = pts.map(p => p.y);
  const xlo = Math.min(...xs) * 0.7, xhi = Math.max(...xs) * 1.4;
  const ylo = Math.min(...ys) * 0.7, yhi = Math.max(...ys) * 1.4;
  const sx = v => px + ((l10(v) - l10(xlo)) / (l10(xhi) - l10(xlo))) * PANEL_W;
  const sy = v => top + PANEL_BODY - ((l10(v) - l10(ylo)) / (l10(yhi) - l10(ylo))) * PANEL_BODY;

  // Pearson on the logs, which is what a straight line on a log-log plot means
  const lx = xs.map(l10), ly = ys.map(l10);
  const mx = lx.reduce((a, b) => a + b) / lx.length, my = ly.reduce((a, b) => a + b) / ly.length;
  const cov = lx.map((v, i) => (v - mx) * (ly[i] - my)).reduce((a, b) => a + b);
  const vx = lx.map(v => (v - mx) ** 2).reduce((a, b) => a + b);
  const r = cov / Math.sqrt(vx * ly.map(v => (v - my) ** 2).reduce((a, b) => a + b));
  const slope = cov / vx;
  const fit = v => 10 ** (my + slope * (l10(v) - mx));

  panels += panelHead(px, 'Is the work big enough?', `plain-JS duration against WebGPU speed-up · r = ${r.toFixed(2)}`);
  panels += `<rect x="${px}" y="${top}" width="${PANEL_W}" height="${PANEL_BODY}" fill="rgba(158,140,220,.04)" rx="6"/>`;
  panels += `<line x1="${sx(xlo).toFixed(1)}" y1="${sy(fit(xlo)).toFixed(1)}" x2="${sx(xhi).toFixed(1)}" y2="${sy(fit(xhi)).toFixed(1)}" stroke="${TEAL}" stroke-width="1.5" stroke-dasharray="5 4" opacity=".55"/>`;
  panels += `<line x1="${px}" y1="${sy(1).toFixed(1)}" x2="${(px + PANEL_W).toFixed(1)}" y2="${sy(1).toFixed(1)}" stroke="${PINK}" stroke-width="1" opacity=".5"/>`;
  for (const p of pts) {
    panels += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="3.6" fill="${p.y >= 1 ? TEAL : PINK}" opacity=".85"/>`;
  }
  panels += `<text x="${px}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" font-family="ui-monospace,monospace">${fmtMs(xlo)}</text>`;
  panels += `<text x="${(px + PANEL_W).toFixed(0)}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" text-anchor="end" font-family="ui-monospace,monospace">${fmtMs(xhi)} of JavaScript →</text>`;
  panels += `<text x="${px}" y="${top - 6}" fill="${MUTED}" font-size="10.5" font-family="ui-monospace,monospace">↑ ${fmtX(yhi)}×</text>`;
}

// ── B. what kind of work wins ──────────────────────────────────────────────
// Both gpu.js series per family, because the interesting thing is not either
// curve alone but how differently they respond to the SAME axis: WebGPU swings
// three orders of magnitude across families, WebAssembly barely leaves 1x. A
// GPU cares enormously what shape the work is; one core mostly does not.
{
  const px = PAD + PANEL_W + 32;
  const top = PANEL_Y + 40;
  const byGroup = new Map();
  for (const r of rows) {
    if (!byGroup.has(r.group)) byGroup.set(r.group, { gpu: [], wasm: [] });
    byGroup.get(r.group).gpu.push(r.best);
    if (r.webasm) byGroup.get(r.group).wasm.push(r.webasm);
  }
  const fams = [...byGroup.entries()]
    .map(([g, v]) => ({ g, ...v, med: median(v.gpu) }))
    .sort((a, b) => b.med - a.med);

  // Family names get a gutter of their own. Inside the plot they collided with
  // any dot that landed near the left edge — which is exactly the interesting
  // case, a family with a member gpu.js lost on.
  const GUT = 82;
  const plotX = px + GUT;
  const plotW = PANEL_W - GUT;
  const all2 = rows.flatMap(r => [r.best, r.webasm].filter(Boolean));
  const lo2 = Math.min(...all2) * 0.7;
  const hi2 = Math.max(...all2) * 1.4;
  const sx2 = v => plotX + ((l10(v) - l10(lo2)) / (l10(hi2) - l10(lo2))) * plotW;
  const step = PANEL_BODY / fams.length;

  panels += panelHead(px, 'What kind of work wins', 'by family · WebGPU above, WebAssembly below, medians marked');
  panels += `<rect x="${plotX.toFixed(1)}" y="${top}" width="${plotW.toFixed(1)}" height="${PANEL_BODY}" fill="rgba(158,140,220,.04)" rx="6"/>`;
  panels += `<line x1="${sx2(1).toFixed(1)}" y1="${top}" x2="${sx2(1).toFixed(1)}" y2="${(top + PANEL_BODY).toFixed(1)}" stroke="${PINK}" stroke-width="1" opacity=".5"/>`;
  fams.forEach((f, i) => {
    const band = top + i * step;
    const yA = band + step * 0.33;
    const yB = band + step * 0.72;
    panels += `<text x="${(plotX - 10).toFixed(1)}" y="${(band + step * 0.5 + 1).toFixed(1)}" fill="${MUTED}" font-size="10.5" text-anchor="end" font-family="ui-monospace,monospace">${esc(f.g)}</text>`;
    panels += `<text x="${(plotX - 10).toFixed(1)}" y="${(band + step * 0.5 + 12).toFixed(1)}" fill="${MUTED}" font-size="9" text-anchor="end" opacity=".7" font-family="ui-monospace,monospace">n=${f.gpu.length}</text>`;
    if (i) panels += `<line x1="${plotX.toFixed(1)}" y1="${band.toFixed(1)}" x2="${(plotX + plotW).toFixed(1)}" y2="${band.toFixed(1)}" stroke="${GRID}" stroke-width="1" opacity=".5"/>`;

    for (const v of f.gpu) {
      panels += `<circle cx="${sx2(v).toFixed(1)}" cy="${yA.toFixed(1)}" r="2.8" fill="${v >= 1 ? TEAL : PINK}" opacity=".55"/>`;
    }
    panels += `<line x1="${sx2(median(f.gpu)).toFixed(1)}" y1="${(yA - 6).toFixed(1)}" x2="${sx2(median(f.gpu)).toFixed(1)}" y2="${(yA + 6).toFixed(1)}" stroke="${TEAL}" stroke-width="2.5"/>`;

    for (const v of f.wasm) {
      const d = 2.9;
      panels += `<path d="M ${sx2(v).toFixed(1)} ${(yB - d).toFixed(1)} L ${(sx2(v) + d).toFixed(1)} ${yB.toFixed(1)} L ${sx2(v).toFixed(1)} ${(yB + d).toFixed(1)} L ${(sx2(v) - d).toFixed(1)} ${yB.toFixed(1)} Z" fill="${BLUE}" opacity=".6"/>`;
    }
    if (f.wasm.length) {
      panels += `<line x1="${sx2(median(f.wasm)).toFixed(1)}" y1="${(yB - 6).toFixed(1)}" x2="${sx2(median(f.wasm)).toFixed(1)}" y2="${(yB + 6).toFixed(1)}" stroke="${BLUE}" stroke-width="2.5"/>`;
    }
  });
  panels += `<text x="${(plotX + plotW).toFixed(0)}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" text-anchor="end" font-family="ui-monospace,monospace">faster →</text>`;
  panels += `<text x="${plotX.toFixed(0)}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" font-family="ui-monospace,monospace">← slower than plain JS</text>`;
}

// ── C. the tax, distributed ────────────────────────────────────────────────
{
  const px = PAD + (PANEL_W + 32) * 2;
  const top = PANEL_Y + 40;
  const taxes = rows.filter(r => r.tax).map(r => r.tax).sort((a, b) => a - b);
  const lo3 = 1, hi3 = Math.max(...taxes) * 1.25;
  const sx3 = v => px + ((l10(v) - l10(lo3)) / (l10(hi3) - l10(lo3))) * PANEL_W;
  const medTax = median(taxes);

  panels += panelHead(px, 'What the library costs', `every workload · median ${medTax.toFixed(1)}×, worst ${taxes.at(-1).toFixed(0)}×`);
  panels += `<rect x="${px}" y="${top}" width="${PANEL_W}" height="${PANEL_BODY}" fill="rgba(158,140,220,.04)" rx="6"/>`;
  // a beeswarm-ish column per workload: one bar each, sorted, so the shape of
  // the distribution is the picture rather than a summary of it
  const bw = PANEL_BODY / taxes.length;
  taxes.forEach((t, i) => {
    const y = top + i * bw;
    panels += `<rect x="${sx3(lo3).toFixed(1)}" y="${(y + 1).toFixed(1)}" width="${Math.max(1.5, sx3(t) - sx3(lo3)).toFixed(1)}" height="${Math.max(1.5, bw - 2).toFixed(1)}" fill="${AMBER}" opacity="${t >= 3 ? '.95' : '.55'}" rx="1"/>`;
  });
  panels += `<line x1="${sx3(medTax).toFixed(1)}" y1="${top}" x2="${sx3(medTax).toFixed(1)}" y2="${(top + PANEL_BODY).toFixed(1)}" stroke="${INK}" stroke-width="1.5" stroke-dasharray="3 3"/>`;
  panels += `<text x="${(sx3(medTax) + 6).toFixed(1)}" y="${top + 14}" fill="${INK}" font-size="10.5" font-family="ui-monospace,monospace">median</text>`;
  panels += `<text x="${px}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" font-family="ui-monospace,monospace">1× — free</text>`;
  panels += `<text x="${(px + PANEL_W).toFixed(0)}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" text-anchor="end" font-family="ui-monospace,monospace">${taxes.at(-1).toFixed(0)}× slower than WGSL →</text>`;
}

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
  <circle cx="${W - PAD - 470}" cy="${HEAD_H - 78}" r="5" fill="${TEAL}"/>
  <text x="${W - PAD - 458}" y="${HEAD_H - 74}" fill="${MUTED}" font-size="12" font-family="'Avenir Next',system-ui,sans-serif">WebGPU via gpu.js</text>
  <path d="M ${W - PAD - 300} ${HEAD_H - 82.6} L ${W - PAD - 295.4} ${HEAD_H - 78} L ${W - PAD - 300} ${HEAD_H - 73.4} L ${W - PAD - 304.6} ${HEAD_H - 78} Z" fill="${BLUE}"/>
  <text x="${W - PAD - 288}" y="${HEAD_H - 74}" fill="${MUTED}" font-size="12" font-family="'Avenir Next',system-ui,sans-serif">WebAssembly, no GPU</text>
  <circle cx="${W - PAD - 130}" cy="${HEAD_H - 78}" r="4.5" fill="none" stroke="${AMBER}" stroke-width="1.6"/>
  <text x="${W - PAD - 118}" y="${HEAD_H - 74}" fill="${MUTED}" font-size="12" font-family="'Avenir Next',system-ui,sans-serif">hand-written WebGPU</text>

  ${svg}

  ${panels}

  <line x1="${PAD}" y1="${H - FOOT_H + 46}" x2="${W - PAD}" y2="${H - FOOT_H + 46}" stroke="${GRID}"/>

  ${footBlocks}

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
console.log(`  ${rows.length} rows · WebGPU median ${fmtX(stats.medianGpu)}× · spread ${fmtX(stats.bottom.best)}×–${fmtX(stats.top.best)}×`);
console.log(`  WebASM on ${stats.wasmCount} of ${rows.length} rows${stats.medianWasm ? `, median ${fmtX(stats.medianWasm)}×` : ''}${stats.medianTax ? ` · library costs ${stats.medianTax.toFixed(1)}×` : ''}`);
