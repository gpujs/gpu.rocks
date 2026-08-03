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
import { createHash } from 'node:crypto';
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

const { default: workloads, GROUPS } = await import(`${ROOT}/src/Bench/workloads/index.js`);
// The registry already exports display names for the groups, and the poster's
// row labels already use display names. Panel B was printing the raw source
// slug — "movement" for data-movement primitives, which reads as physics on
// the one panel whose whole job is to say what kind of work wins.
const GROUP_LABEL = new Map(GROUPS || []);
// || f.g because a workload added without a group falls back to 'other', which
// has no GROUPS entry and would otherwise print "undefined".
const groupLabel = g => GROUP_LABEL.get(g) || g;
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
// 286, not 250. Three things shared a 60px band: the deck's final period sat
// tangent to the legend's first dot (at 4x it reads as one mark), the legend
// floated on a baseline belonging to neither neighbour, and the first axis tick
// sat 13px under the machine line with overlapping x. All three are relative to
// HEAD_H, so one number opens all three gaps.
const HEAD_H = 286;
const LEGEND_Y = 218;
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

// ── the drawing ────────────────────────────────────────────────────────────
const INK = '#f6f7f8';
const MUTED = '#9b94c0';
const TEAL = '#18bc9c';
const AMBER = '#e2b04a';
const PINK = '#ff79c6';
const BLUE = '#20a4f3';
const GRID = 'rgba(158,140,220,.16)';

// Three faces named once, because the file carried six literal stacks and two
// of them omitted 'Avenir Next'. Montserrat is not installed here, so those two
// fell through to system-ui (SF Pro) while the rest fell to Avenir Next — the
// poster was set in three faces where it meant to use two, and the split ran
// between a panel heading and the subtitle 20px under it.
const DISPLAY = "'Montserrat','Avenir Next',system-ui,sans-serif";
const TEXT = "'Avenir Next','Montserrat',system-ui,sans-serif";
const MONO = 'ui-monospace, monospace';

// SVG text does not wrap or ellipsise, so the label column has to be budgeted
// by hand. The name gets whatever the tag does not need, and when that leaves
// too little to read, the tag is dropped rather than the name truncated into
// uselessness — the workload's name is the thing a reader is looking for.
//
// MEASURED, not estimated. This budgeted with a px-per-character constant of
// 7.0, and a proportional face has no such constant: across these thirty names
// the real figure runs 5.60 ("All-pairs gravity") to 7.06 ("Smith-Waterman
// alignment"). Over-estimating truncated "Gray-Scott reaction-diffusion" with
// 23px to spare; under-estimating is worse, because a name of capitals — an
// "AMD MI300X GEMM (WMMA)" — measures 8.5 px/char, would be judged to fit, and
// would silently overlap its tag. getComputedTextLength() on a real text node
// is exact and honours letter-spacing, which a canvas measure would not.
const browser = await launch();
const page = await browser.newPage();
await page.setContent(
  '<style>html,body{margin:0}</style><svg id="m" width="10" height="10"></svg>',
  { waitUntil: 'load' }
);
await page.evaluateHandle('document.fonts.ready');

const measure = specs =>
  page.evaluate(list => {
    const svg = document.getElementById('m');
    return list.map(([text, family, size, spacing]) => {
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('font-family', family);
      t.setAttribute('font-size', String(size));
      if (spacing) t.setAttribute('letter-spacing', String(spacing));
      t.textContent = text;
      svg.appendChild(t);
      const w = t.getComputedTextLength();
      t.remove();
      return w;
    });
  }, specs);

// Trimming happens in the browser too: the ellipsis has its own width, and
// slicing by character count is the monospace assumption being removed.
const fitAll = specs =>
  page.evaluate(list => {
    const svg = document.getElementById('m');
    return list.map(({ text, budget, family, size }) => {
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('font-family', family);
      t.setAttribute('font-size', String(size));
      svg.appendChild(t);
      const width = v => { t.textContent = v; return t.getComputedTextLength(); };
      let out = text;
      if (width(out) > budget) {
        let n = text.length;
        while (n > 1 && width(`${text.slice(0, n)}\u2026`) > budget) n--;
        out = `${text.slice(0, n)}\u2026`;
      }
      t.remove();
      return out;
    });
  }, specs);

// 12px of air between a name and its tag. The old estimate's 8% over-prediction
// had been acting as an accidental gutter; removing it without an explicit one
// lets a fitted name butt right against the tag.
const NAME_GUTTER = 12;
const tagWidths = await measure(rows.map(r => [r.tag || '', MONO, 10.5]));
const nameSpecs = rows.map((r, i) => {
  const tagW = r.tag ? tagWidths[i] : 0;
  const roomy = LABEL_W - tagW - NAME_GUTTER - 10;
  const keepTag = Boolean(r.tag) && roomy >= 120;
  return { text: r.name, budget: keepTag ? roomy : LABEL_W - 12, family: TEXT, size: 13.5, keepTag };
});
const fittedNames = await fitAll(nameSpecs.map(({ text, budget, family, size }) => ({ text, budget, family, size })));
const LABELS = new Map(rows.map((r, i) => [r.id, { name: fittedNames[i], tag: nameSpecs[i].keepTag ? r.tag : '' }]));
const labelFor = row => LABELS.get(row.id);
const fmtMs = ms => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(0)} ms`);
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');


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
  footBlocks += `<text x="${bx.toFixed(0)}" y="${H - FOOT_H + 92}" fill="${MUTED}" font-size="11" letter-spacing="2" font-family="${MONO}">${esc(f.k)}</text>`;
  footBlocks += `<text x="${bx.toFixed(0)}" y="${H - FOOT_H + 128}" fill="${f.c}" font-size="27" font-weight="800" font-family="${DISPLAY}">${esc(f.v)}</text>`;
  footBlocks += `<text x="${bx.toFixed(0)}" y="${H - FOOT_H + 152}" fill="${MUTED}" font-size="11.5" font-family="${TEXT}">${esc(f.s)}</text>`;
});

const groupLabelW = await measure(
  [...new Set(rows.map(r => r.group))].map(g => [groupLabel(g), MONO, 10.5])
);

// ── the legend ─────────────────────────────────────────────────────────────
// Generated rather than three hand-placed pairs, for three reasons. It sat on
// no baseline of its own and collided with the deck's final period. It
// advertised all three series unconditionally, where the footer already drops
// its WebASM and library-cost blocks when those series are absent. And its
// three x offsets were tuned by eye — two of the labels are both 19 characters
// and differ by 4.5px, which no per-character estimate can see.
const legendItems = [
  { c: TEAL, shape: 'dot', label: 'WebGPU via gpu.js' },
  wasmRows.length ? { c: BLUE, shape: 'diamond', label: 'WebAssembly, no GPU' } : null,
  rows.some(r => r.bare) ? { c: AMBER, shape: 'ring', label: 'hand-written WebGPU' } : null,
].filter(Boolean);

const legendW = await measure(legendItems.map(i => [i.label, TEXT, 12]));
let legendSvg = '';
{
  const GAP = 34;
  // 17, not 12: the mark is centred at lx+5 with r=5, so at 12 the label began
  // 2px from the mark's edge and read as attached to it.
  const MARK = 17;
  const total = legendW.reduce((a, w) => a + w + MARK, 0) + GAP * (legendItems.length - 1);
  let lx = W - PAD - total;
  legendItems.forEach((it, i) => {
    const cy = LEGEND_Y - 4;
    if (it.shape === 'dot') legendSvg += `<circle cx="${(lx + 5).toFixed(1)}" cy="${cy}" r="5" fill="${it.c}"/>`;
    else if (it.shape === 'ring') legendSvg += `<circle cx="${(lx + 5).toFixed(1)}" cy="${cy}" r="4.5" fill="none" stroke="${it.c}" stroke-width="1.6"/>`;
    else legendSvg += `<path d="M ${(lx + 5).toFixed(1)} ${cy - 4.6} L ${(lx + 9.6).toFixed(1)} ${cy} L ${(lx + 5).toFixed(1)} ${cy + 4.6} L ${(lx + 0.4).toFixed(1)} ${cy} Z" fill="${it.c}"/>`;
    legendSvg += `<text x="${(lx + MARK).toFixed(1)}" y="${LEGEND_Y}" fill="${MUTED}" font-size="12" font-family="${TEXT}">${esc(it.label)}</text>`;
    lx += MARK + legendW[i] + GAP;
  });
}

let svg = '';

// decade gridlines, drawn first so everything sits on top of them
for (const t of ticks) {
  svg += `<line x1="${x(t).toFixed(1)}" y1="${HEAD_H - 22}" x2="${x(t).toFixed(1)}" y2="${HEAD_H + rows.length * ROW_H + 6}" stroke="${GRID}" stroke-width="1"/>`;
  svg += `<text x="${x(t).toFixed(1)}" y="${HEAD_H - 32}" fill="${MUTED}" font-size="12" text-anchor="middle" font-family="${MONO}">${fmtX(t)}×</text>`;
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
svg += `<text x="${(spine + 8).toFixed(1)}" y="${chartBottom + 28}" fill="${PINK}" font-size="12" font-weight="700" font-family="${MONO}">1× — plain JavaScript. Left of this line gpu.js lost to it.</text>`;

rows.forEach((r, i) => {
  const y = HEAD_H + i * ROW_H + ROW_H / 2;
  const win = r.best >= 1;
  const dot = win ? TEAL : PINK;

  const lab = labelFor(r);
  svg += `<text x="${PAD}" y="${(y + 4).toFixed(1)}" fill="${INK}" font-size="13.5" font-family="${TEXT}">${esc(lab.name)}</text>`;
  if (lab.tag) {
    svg += `<text x="${PAD + LABEL_W - 22}" y="${(y + 4).toFixed(1)}" fill="${MUTED}" font-size="10.5" text-anchor="end" font-family="${MONO}">${esc(lab.tag)}</text>`;
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
  svg += `<text x="${(W - PAD).toFixed(1)}" y="${(y + 4).toFixed(1)}" fill="${win ? INK : PINK}" font-size="12.5" text-anchor="end" font-weight="700" font-family="${MONO}">${fmtX(r.best)}×</text>`;
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
  `<text x="${px.toFixed(0)}" y="${PANEL_Y}" fill="${INK}" font-size="15" font-weight="800" font-family="${DISPLAY}">${esc(title)}</text>` +
  `<text x="${px.toFixed(0)}" y="${PANEL_Y + 20}" fill="${MUTED}" font-size="11.5" font-family="${TEXT}">${esc(sub)}</text>`;

// ── A. is it worth it? duration against speed-up, both log ─────────────────
{
  const px = PAD;
  const top = PANEL_Y + 40;
  const pts = rows.map(r => ({ x: r.baseMs, y: r.best }));
  const xs = pts.map(p => p.x); const ys = pts.map(p => p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymax = Math.max(...ys);
  const xlo = xmin * 0.7, xhi = xmax * 1.4;
  const ylo = Math.min(...ys) * 0.7, yhi = ymax * 1.4;
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
  // The EXTREMES, not the bounds. These three captions printed xlo/xhi/yhi,
  // which are min*0.7 and max*1.4 — breathing room — in the same muted mono as
  // every measured figure on the sheet. On this run that meant 110 ms, 6.1 s
  // and 1214x, none of which occurs anywhere in the data, and the 1214x
  // contradicted the footer and the top row, both of which print 867x for the
  // same quantity. On a poster closing with "every number here was measured",
  // that is the one defect that costs a reader their trust in the other 30.
  //
  // The words matter as much as the values: the extremes are inset from the
  // edges by the padding, so a bare "157 ms" in the corner would relabel the
  // edge instead of the data. "shortest" and "longest" say what is being named.
  panels += `<text x="${px}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" font-family="${MONO}">shortest ${fmtMs(xmin)}</text>`;
  panels += `<text x="${(px + PANEL_W).toFixed(0)}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" text-anchor="end" font-family="${MONO}">longest ${fmtMs(xmax)} of JavaScript →</text>`;
  panels += `<text x="${px}" y="${top - 6}" fill="${MUTED}" font-size="10.5" font-family="${MONO}">↑ WebGPU speed-up · max ${fmtX(ymax)}×</text>`;
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
  // Measured, not 82. "Dense arithmetic" is 101px where the slug "dense" was
  // 30, so a constant tuned to the old slugs would clip the new labels — and a
  // constant tuned to today's GROUPS would be the same brittleness one list
  // later. This costs panel B some plot width; the marks already fill 94% of
  // it, so it is a real trade for bands that are not misread, not free space.
  const GUT = Math.ceil(Math.max(...groupLabelW) + 18);
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
    panels += `<text x="${(plotX - 10).toFixed(1)}" y="${(band + step * 0.5 + 1).toFixed(1)}" fill="${MUTED}" font-size="10.5" text-anchor="end" font-family="${MONO}">${esc(groupLabel(f.g))}</text>`;
    // .85, not .7: muted at .7 composites to rgb(110,104,142) on this ground,
    // 3.92:1, the only text on the poster below AA — and it is the label
    // carrying the sample-size caveat, at the smallest size on the sheet.
    // .85 gives 5.38:1 and is still visibly subordinate.
    //
    // The offset is a fraction of the band rather than a flat 12px, which at
    // nine or more families would have pushed this baseline into the next row.
    panels += `<text x="${(plotX - 10).toFixed(1)}" y="${(band + step * 0.5 + Math.min(12, step * 0.35)).toFixed(1)}" fill="${MUTED}" font-size="9" text-anchor="end" opacity=".85" font-family="${MONO}">n=${f.gpu.length}</text>`;
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
  panels += `<text x="${(plotX + plotW).toFixed(0)}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" text-anchor="end" font-family="${MONO}">faster →</text>`;
  panels += `<text x="${plotX.toFixed(0)}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" font-family="${MONO}">← slower than plain JS</text>`;
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
  panels += `<text x="${(sx3(medTax) + 6).toFixed(1)}" y="${top + 14}" fill="${INK}" font-size="10.5" font-family="${MONO}">median</text>`;
  panels += `<text x="${px}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" font-family="${MONO}">1× — free</text>`;
  panels += `<text x="${(px + PANEL_W).toFixed(0)}" y="${top + PANEL_BODY + 16}" fill="${MUTED}" font-size="10.5" text-anchor="end" font-family="${MONO}">${taxes.at(-1).toFixed(0)}× slower than WGSL →</text>`;
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

  <text x="${PAD}" y="78" fill="${PINK}" font-size="13" font-weight="800" letter-spacing="3" font-family="${MONO}">GPU.JS</text>
  <text x="${PAD}" y="132" fill="${INK}" font-size="44" font-weight="800" letter-spacing="-1" font-family="'Montserrat','Avenir Next',system-ui,sans-serif">The Benchmark Gauntlet</text>
  <text x="${PAD}" y="166" fill="${MUTED}" font-size="15" font-family="${TEXT}">${esc(rows.length)} GPGPU workloads, every one checked against a plain-JavaScript oracle before it was timed.</text>
  <text x="${PAD}" y="192" fill="${MUTED}" font-size="13" font-family="${MONO}">${esc(run.machine || '')}${run.date ? ` · ${esc(run.date)}` : ''} · gpu.js ${esc(run.gpujs || '?')}</text>

  ${legendSvg}

  ${svg}

  ${panels}

  <line x1="${PAD}" y1="${H - FOOT_H + 46}" x2="${W - PAD}" y2="${H - FOOT_H + 46}" stroke="${GRID}"/>

  ${footBlocks}

  <text x="${PAD}" y="${H - 34}" fill="${MUTED}" font-size="12" font-family="${TEXT}">Every number here was measured in a browser and can be reproduced in one — gpu.rocks/benchmark</text>
</svg>
</div>`;

const pageHtml = `<style>
  html, body { margin: 0; background: #050218; }
  .poster { width: ${W}px; }
  svg { display: block; }
</style>
${html}`;

// ── the link-preview card ──────────────────────────────────────────────────
// Read at about 300px wide in a feed, so it cannot be a shrunken poster — the
// thirty labels and four panels become texture. It gets one idea, and it is
// the page's thesis rather than its best number: how UNEVEN the win is.
//
// Two treatments, --og-style=top (default) and =stair. The top variant labels
// the ten biggest rows, each with its own bar and figure, trading the overall
// shape for names a reader might recognise. The stair variant draws all thirty
// from the 1x baseline so the silhouette itself carries the unevenness. Both
// are generated from the run rather than drawn.
const OG_W = 1200;
const OG_H = 630;
const OG_STYLE = arg('--og-style') || 'top';

function ogHead() {
  return `<rect width="${OG_W}" height="${OG_H}" fill="#050218"/>
  <rect width="${OG_W}" height="${OG_H}" fill="url(#g1)"/>
  <rect width="${OG_W}" height="${OG_H}" fill="url(#g2)"/>
  <text x="72" y="88" fill="${PINK}" font-size="18" font-weight="800" letter-spacing="5" font-family="${MONO}">GPU.JS</text>
  <text x="72" y="152" fill="${INK}" font-size="58" font-weight="800" letter-spacing="-1.4" font-family="${DISPLAY}">The Benchmark Gauntlet</text>`;
}
const OG_DEFS = `<defs>
    <radialGradient id="g1" cx="84%" cy="0%" r="72%">
      <stop offset="0%" stop-color="#2e0741" stop-opacity=".95"/><stop offset="62%" stop-color="#2e0741" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="g2" cx="4%" cy="98%" r="62%">
      <stop offset="0%" stop-color="#20a4f3" stop-opacity=".16"/><stop offset="60%" stop-color="#20a4f3" stop-opacity="0"/>
    </radialGradient>
  </defs>`;

function ogStair() {
  const sorted = [...rows].sort((a, b) => b.best - a.best);
  const lo = Math.min(...sorted.map(r => r.best)) * 0.75;
  const hi = Math.max(...sorted.map(r => r.best)) * 1.15;
  const x0 = 72;
  const x1 = OG_W - 250;
  const sx = v => x0 + ((l10(v) - l10(lo)) / (l10(hi) - l10(lo))) * (x1 - x0);
  const top = 232;
  const rowH = (OG_H - top - 96) / sorted.length;
  const spine = sx(1);

  let g = '';
  for (let e = Math.ceil(l10(lo)); e <= Math.floor(l10(hi)); e++) {
    const v = 10 ** e;
    g += `<line x1="${sx(v).toFixed(1)}" y1="${top - 12}" x2="${sx(v).toFixed(1)}" y2="${OG_H - 84}" stroke="rgba(158,140,220,.14)"/>`;
    g += `<text x="${sx(v).toFixed(1)}" y="${OG_H - 62}" fill="${MUTED}" font-size="15" text-anchor="middle" font-family="${MONO}">${fmtX(v)}×</text>`;
  }
  sorted.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    const win = r.best >= 1;
    const a = Math.min(spine, sx(r.best));
    const b = Math.max(spine, sx(r.best));
    g += `<rect x="${a.toFixed(1)}" y="${(y - rowH * 0.34).toFixed(1)}" width="${Math.max(2, b - a).toFixed(1)}" height="${(rowH * 0.68).toFixed(1)}" rx="${(rowH * 0.34).toFixed(1)}" fill="${win ? TEAL : PINK}" opacity=".92"/>`;
  });
  g += `<line x1="${spine.toFixed(1)}" y1="${top - 26}" x2="${spine.toFixed(1)}" y2="${OG_H - 84}" stroke="${PINK}" stroke-width="2" stroke-dasharray="4 5"/>`;
  g += `<text x="${(spine - 8).toFixed(1)}" y="${top - 34}" fill="${PINK}" font-size="15" font-weight="700" text-anchor="end" font-family="${MONO}">plain JavaScript</text>`;

  return `<svg width="${OG_W}" height="${OG_H}" viewBox="0 0 ${OG_W} ${OG_H}" xmlns="http://www.w3.org/2000/svg">
  ${OG_DEFS}
  ${ogHead()}
  <text x="72" y="196" fill="${MUTED}" font-size="21" font-family="${TEXT}">${rows.length} GPGPU workloads, every answer checked before it was timed.</text>
  ${g}
  <text x="${OG_W - 200}" y="${top + 78}" fill="${INK}" font-size="52" font-weight="800" font-family="${DISPLAY}">${fmtX(sorted[0].best)}×</text>
  <text x="${OG_W - 200}" y="${top + 108}" fill="${MUTED}" font-size="17" font-family="${TEXT}">fastest row</text>
  <text x="${OG_W - 200}" y="${top + 186}" fill="${INK}" font-size="52" font-weight="800" font-family="${DISPLAY}">${fmtX(stats.medianGpu)}×</text>
  <text x="${OG_W - 200}" y="${top + 216}" fill="${MUTED}" font-size="17" font-family="${TEXT}">median</text>
  <text x="${OG_W - 200}" y="${top + 294}" fill="${PINK}" font-size="52" font-weight="800" font-family="${DISPLAY}">${fmtX(sorted[sorted.length - 1].best)}×</text>
  <text x="${OG_W - 200}" y="${top + 324}" fill="${MUTED}" font-size="17" font-family="${TEXT}">slowest row</text>
  <text x="72" y="${OG_H - 26}" fill="${MUTED}" font-size="18" font-family="${MONO}">gpu.rocks/benchmark</text>
</svg>`;
}

function ogTop() {
  const top10 = [...rows].sort((a, b) => b.best - a.best).slice(0, 10);
  const lo = 1;
  const hi = Math.max(...top10.map(r => r.best)) * 1.5;
  const x0 = 430;
  const x1 = OG_W - 150;
  const sx = v => x0 + ((l10(v) - l10(lo)) / (l10(hi) - l10(lo))) * (x1 - x0);
  const top = 224;
  const rowH = (OG_H - top - 70) / top10.length;
  let g = '';
  top10.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    g += `<text x="${x0 - 16}" y="${(y + 5).toFixed(1)}" fill="${MUTED}" font-size="16" text-anchor="end" font-family="${TEXT}">${esc(r.name.length > 26 ? `${r.name.slice(0, 25)}…` : r.name)}</text>`;
    g += `<rect x="${x0}" y="${(y - rowH * 0.3).toFixed(1)}" width="${Math.max(3, sx(r.best) - x0).toFixed(1)}" height="${(rowH * 0.6).toFixed(1)}" rx="${(rowH * 0.3).toFixed(1)}" fill="${TEAL}" opacity=".9"/>`;
    g += `<text x="${(sx(r.best) + 12).toFixed(1)}" y="${(y + 5).toFixed(1)}" fill="${INK}" font-size="16" font-weight="700" font-family="${MONO}">${fmtX(r.best)}×</text>`;
  });
  return `<svg width="${OG_W}" height="${OG_H}" viewBox="0 0 ${OG_W} ${OG_H}" xmlns="http://www.w3.org/2000/svg">
  ${OG_DEFS}
  ${ogHead()}
  <text x="72" y="196" fill="${MUTED}" font-size="21" font-family="${TEXT}">WebGPU through gpu.js, against plain JavaScript. ${rows.length} workloads, every answer checked.</text>
  ${g}
  <text x="72" y="${OG_H - 26}" fill="${MUTED}" font-size="18" font-family="${MONO}">gpu.rocks/benchmark</text>
</svg>`;
}

const ogCard = () => (OG_STYLE === 'top' ? ogTop() : ogStair());

// ── render ─────────────────────────────────────────────────────────────────
const outArg = arg('--out') || join(ROOT, 'out', `gauntlet-${run.id || 'run'}.png`);
mkdirSync(dirname(resolve(ROOT, outArg)), { recursive: true });
const htmlPath = resolve(ROOT, outArg).replace(/\.png$/, '.html');
writeFileSync(htmlPath, pageHtml);

// the same page the measurements were taken on — a second browser would be a
// second font environment, and the layout is now measured against this one
const p = page;
await p.setViewport({ width: W, height: Math.min(H, 4000), deviceScaleFactor: 2 });
await p.setContent(pageHtml, { waitUntil: 'load' });
// let the font stack settle before the shot, or the first paint measures a
// fallback face and the labels shift after capture
await p.evaluateHandle('document.fonts.ready');
const el = await p.$('.poster');
await el.screenshot({ path: resolve(ROOT, outArg) });

// --site also publishes it: the full poster and a thumbnail into public/img,
// plus a manifest of content hashes. /img/... is a stable URL whose BYTES
// change every time the poster is regenerated, and the deploy purges HTML
// only — so without a ?v= key a returning visitor keeps the old picture, and
// a purge cannot reach their phone's cache at all. Same reasoning, and the
// same solution, as the module card art.
if (argv.includes('--site')) {
  const IMG = join(ROOT, 'public/img/bench');
  mkdirSync(IMG, { recursive: true });
  writeFileSync(join(IMG, 'gauntlet.png'), readFileSync(resolve(ROOT, outArg)));
  // A second raster rather than a CSS-scaled copy of the first: the page shows
  // a preview a few hundred pixels wide, and shipping a 2480px image to be
  // displayed at 260 is most of a megabyte spent on nothing.
  const THUMB_SCALE = 0.36;
  await p.setViewport({ width: W, height: Math.min(H, 4000), deviceScaleFactor: THUMB_SCALE });
  await p.evaluateHandle('document.fonts.ready');
  await (await p.$('.poster')).screenshot({ path: join(IMG, 'gauntlet-thumb.png') });

  // the link-preview card, same data, its own aspect ratio
  await p.setViewport({ width: OG_W, height: OG_H, deviceScaleFactor: 1 });
  await p.setContent(`<style>html,body{margin:0}</style>${ogCard()}`, { waitUntil: 'load' });
  await p.evaluateHandle('document.fonts.ready');
  await p.screenshot({ path: join(IMG, 'ogbench.png'), clip: { x: 0, y: 0, width: OG_W, height: OG_H } });

  const hash = f => createHash('sha256').update(readFileSync(join(IMG, f))).digest('hex').slice(0, 8);
  writeFileSync(join(ROOT, 'src/Bench/poster.js'),
`/**
 * src/Bench/poster.js — GENERATED by scripts/bench-infographic.mjs --site.
 *
 * Content hashes for the poster art. /img/bench/gauntlet.png is a stable URL
 * whose bytes change whenever the poster is regenerated, and the deploy purges
 * HTML rather than assets — so the hash is hung off the URL as ?v= to make new
 * art a new URL. See scripts/capture-module-renders.mjs for the same problem
 * and the same fix on the module card art.
 */
export default {
  full: '/img/bench/gauntlet.png?v=${hash('gauntlet.png')}',
  thumb: '/img/bench/gauntlet-thumb.png?v=${hash('gauntlet-thumb.png')}',
  og: '/img/bench/ogbench.png?v=${hash('ogbench.png')}',
  ogWidth: ${OG_W},
  ogHeight: ${OG_H},
  width: ${W},
  height: ${H},
  rows: ${rows.length},
  machine: ${JSON.stringify(run.machine || '')},
  gpujs: ${JSON.stringify(run.gpujs || '')},
  date: ${JSON.stringify(run.date || '')},
};
`);
  console.log(`bench-infographic: published public/img/bench/ + src/Bench/poster.js`);
}

await browser.close();

console.log(`bench-infographic: ${resolve(ROOT, outArg)}  (${W}×${H} at 2x)`);
console.log(`                   ${htmlPath}  — edit and re-render`);
console.log(`  ${rows.length} rows · WebGPU median ${fmtX(stats.medianGpu)}× · spread ${fmtX(stats.bottom.best)}×–${fmtX(stats.top.best)}×`);
console.log(`  WebASM on ${stats.wasmCount} of ${rows.length} rows${stats.medianWasm ? `, median ${fmtX(stats.medianWasm)}×` : ''}${stats.medianTax ? ` · library costs ${stats.medianTax.toFixed(1)}×` : ''}`);
