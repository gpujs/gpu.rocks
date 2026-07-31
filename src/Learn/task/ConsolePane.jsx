import React, { useEffect, useMemo, useRef, useState } from 'react';

// Console output pane — renders the engine's log stream in the mockup's
// style: timestamps, italic system lines, teal ok lines, red errors, and
// inline canvas snapshots with a "canvas · W×H" badge.
//
// It also renders the rich entries the sandbox can emit:
//   • plot(...)            an explicit chart          (log.plot)
//   • a logged numeric array gets a free sparkline    (log.spark)
//   • consecutive render() frames become a scrubber   (grouped here)
//   • slider(...) declarations become real sliders    (controls)
//
// All of it is plain JSON off the worker — see engine/utils.js — so nothing in
// here has to know about kernels, canvases or the sandbox.

function lineClass(type) {
  switch (type) {
    case 'system':
      return 'sys';
    case 'ok':
      return 'ok';
    case 'error':
      return 'err';
    case 'warn':
      return 'warn';
    default:
      return undefined;
  }
}

// The figure palette, so a chart drawn by the console and a chart drawn by hand
// in an SVG figure use the same colours.
const SERIES_COLORS = ['var(--blue)', 'var(--pink)', 'var(--teal)', 'var(--amber)', 'var(--lred)'];

// ---- charts ----------------------------------------------------------------

function extent(series, useLog) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    for (const v of s.values) {
      if (!Number.isFinite(v)) continue;
      if (useLog && v <= 0) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (lo === hi) return [lo - Math.abs(lo || 1) * 0.5, hi + Math.abs(hi || 1) * 0.5];
  return [lo, hi];
}

function fmtTick(v) {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5 || a < 1e-3) return v.toExponential(1).replace('e+', 'e');
  return String(Number(v.toPrecision(3)));
}

function Chart({ plot }) {
  const W = 440;
  const H = 150;
  const PAD = { l: 46, r: 8, t: 10, b: 20 };
  const useLog = plot.log && plot.series.some(s => s.values.some(v => v > 0));
  const [lo, hi] = extent(plot.series, useLog);
  const tx = useLog ? v => Math.log10(Math.max(v, Number.MIN_VALUE)) : v => v;
  const [tlo, thi] = [tx(lo), tx(hi)];
  const span = thi - tlo || 1;
  const plotW = W - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;
  const yOf = v => PAD.t + plotH - ((tx(v) - tlo) / span) * plotH;
  const maxLen = Math.max(...plot.series.map(s => s.values.length));
  const xOf = i => PAD.l + (maxLen <= 1 ? 0 : (i / (maxLen - 1)) * plotW);

  const ticks = [lo, useLog ? Math.sqrt(lo * hi) : (lo + hi) / 2, hi];

  return (
    <div className="plotout">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={plotAria(plot)}>
        <g className="plot-axis">
          {ticks.map((t, i) => (
            <g key={i}>
              <line x1={PAD.l} y1={yOf(t)} x2={W - PAD.r} y2={yOf(t)} />
              <text x={PAD.l - 5} y={yOf(t) + 3} textAnchor="end">
                {fmtTick(t)}
              </text>
            </g>
          ))}
        </g>
        {plot.series.map((s, si) => {
          const pts = s.values
            .map((v, i) => {
              if (!Number.isFinite(v) || (useLog && v <= 0)) return null;
              return `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`;
            })
            .filter(Boolean);
          if (!pts.length) return null;
          return (
            <polyline
              key={si}
              points={pts.join(' ')}
              fill="none"
              stroke={SERIES_COLORS[si % SERIES_COLORS.length]}
              strokeWidth="1.6"
              strokeLinejoin="round"
            />
          );
        })}
      </svg>
      <div className="plot-legend">
        {plot.series.map((s, si) => (
          <span key={si}>
            <i style={{ background: SERIES_COLORS[si % SERIES_COLORS.length] }} aria-hidden="true" />
            {s.name || `series ${si + 1}`}
            {s.total > s.values.length ? ` (${s.total.toLocaleString('en-US')} pts)` : ''}
          </span>
        ))}
        {useLog && <span className="plot-note">log scale</span>}
      </div>
    </div>
  );
}

function plotAria(plot) {
  const parts = plot.series.map(s => {
    const v = s.values.filter(Number.isFinite);
    if (!v.length) return s.name || 'empty series';
    return `${s.name || 'series'}: ${v.length} points from ${fmtTick(v[0])} to ${fmtTick(v[v.length - 1])}`;
  });
  return `${plot.title || 'Plot'}. ${parts.join('; ')}.`;
}

function Sparkline({ spark }) {
  const { values } = spark;
  const W = 120;
  const H = 18;
  const finite = values.filter(Number.isFinite);
  if (finite.length < 2) return null;
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const span = hi - lo || 1;
  const pts = values
    .map((v, i) =>
      Number.isFinite(v)
        ? `${((i / (values.length - 1)) * W).toFixed(1)},${(H - ((v - lo) / span) * H).toFixed(1)}`
        : null
    )
    .filter(Boolean)
    .join(' ');
  return (
    <svg
      className="spark"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`sparkline, ${spark.total} values from ${fmtTick(finite[0])} to ${fmtTick(finite[finite.length - 1])}, minimum ${fmtTick(lo)}, maximum ${fmtTick(hi)}`}
    >
      <polyline points={pts} fill="none" stroke="var(--blue)" strokeWidth="1.2" />
    </svg>
  );
}

// ---- frame scrubber --------------------------------------------------------
//
// render() in a loop already produces one canvas entry per call, each with its
// pixels captured at log time so a later frame cannot retroactively change an
// earlier one. So the frames are already recorded; showing them as a strip
// rather than N stills is purely a rendering decision.

function FrameStrip({ frames }) {
  const [i, setI] = useState(frames.length - 1);
  const idx = Math.min(i, frames.length - 1);
  const shot = frames[idx].snapshot;
  return (
    <div className="imgout frames">
      <img src={shot.url} alt={`frame ${idx + 1} of ${frames.length}, ${shot.w} by ${shot.h} pixels`} />
      <div className="frame-ctl">
        <input
          type="range"
          min={0}
          max={frames.length - 1}
          value={idx}
          onChange={e => setI(Number(e.target.value))}
          aria-label={`frame ${idx + 1} of ${frames.length}`}
        />
        <span className="cbadge">
          frame {idx + 1}/{frames.length} · {shot.w}×{shot.h}
        </span>
      </div>
    </div>
  );
}

// ---- controls --------------------------------------------------------------

function Controls({ controls, values, onChange, busy }) {
  return (
    <div className="controls" role="group" aria-label="Program controls">
      {controls.map(c => {
        const v = values[c.name] !== undefined ? values[c.name] : c.value;
        return (
          <label key={c.name} className="ctl">
            <span className="ctl-name">{c.label}</span>
            <input
              type="range"
              min={c.min}
              max={c.max}
              step={c.step}
              value={v}
              disabled={busy}
              onChange={e => onChange(c.name, Number(e.target.value))}
            />
            <output>{Number(v.toPrecision ? v.toPrecision(4) : v)}</output>
          </label>
        );
      })}
    </div>
  );
}

// ---- pane ------------------------------------------------------------------

function ConsoleLine({ log }) {
  return (
    <>
      <div className="ln">
        <span className="t">{log.time}</span>
        <span className={lineClass(log.type)}>{log.text}</span>
      </div>
      {/* Its own row rather than trailing the text: a formatted array wraps to
          two or three lines, and an inline sparkline ends up pushed off the
          right edge of the pane where it is clipped and useless. */}
      {log.spark && (
        <div className="sparkrow">
          <Sparkline spark={log.spark} />
        </div>
      )}
      {log.plot && <Chart plot={log.plot} />}
      {log.snapshot && (
        <div className="imgout">
          <img
            src={log.snapshot.url}
            alt={`canvas output, ${log.snapshot.w} by ${log.snapshot.h} pixels`}
          />
          <span className="cbadge" aria-hidden="true">
            canvas · {log.snapshot.w}×{log.snapshot.h}
          </span>
        </div>
      )}
    </>
  );
}

// Consecutive canvas entries collapse into one scrubber. Three is the
// threshold: two stills side by side are a comparison and read fine, but a
// 60-generation run should not be 60 images down the pane.
const FRAME_GROUP_MIN = 3;

function groupLogs(logs) {
  const out = [];
  let run = [];
  const flush = () => {
    if (!run.length) return;
    if (run.length >= FRAME_GROUP_MIN) out.push({ frames: run });
    else run.forEach(l => out.push({ log: l }));
    run = [];
  };
  for (const log of logs) {
    if (log.type === 'canvas' && log.snapshot) run.push(log);
    else {
      flush();
      out.push({ log });
    }
  }
  flush();
  return out;
}

function ConsolePane({ logs, active, controls, controlValues, onControlChange, controlsBusy }) {
  const paneRef = useRef(null);
  const grouped = useMemo(() => groupLogs(logs), [logs]);

  // keep the newest line in view as output streams in
  useEffect(() => {
    const pane = paneRef.current;
    if (pane) pane.scrollTop = pane.scrollHeight;
  }, [logs, active]);

  return (
    <div
      ref={paneRef}
      className={`bpanel console${active ? ' on' : ''}`}
      role="tabpanel"
      id="bpanel-console"
      aria-labelledby="btab-console"
    >
      {controls && controls.length > 0 && (
        <Controls
          controls={controls}
          values={controlValues || {}}
          onChange={onControlChange}
          busy={controlsBusy}
        />
      )}
      {logs.length === 0 ? (
        <div className="ln">
          <span className="sys">Console is empty — press ▶ Run (or ⌘/Ctrl + Enter).</span>
        </div>
      ) : (
        grouped.map((item, i) =>
          item.frames ? (
            <FrameStrip key={`f${i}`} frames={item.frames} />
          ) : (
            <ConsoleLine key={i} log={item.log} />
          )
        )
      )}
    </div>
  );
}

export default ConsolePane;
