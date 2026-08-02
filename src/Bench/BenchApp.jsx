import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GPU } from 'gpu.js';
import { ThemeProvider, useTheme } from '../Learn/ThemeContext';
import workloads, { GROUPS } from './workloads/index.js';
import savedRuns from './saved/index.js';
import { BASELINE, COLUMNS, runWorkload, webgpuStatus } from './runner.js';
import './scss/bench.scss';

// The benchmark page. Same visual language as the course — same tokens, same
// nav, same theme switch — because it is the same site, but none of the
// course's vocabulary: a row here is an operation and its parameters, legible
// to somebody who has never opened a lesson.

function capitalize(w) {
  return w.charAt(0).toUpperCase() + w.slice(1);
}

function Jelly() {
  return (
    <svg className="jelly" viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
      <path
        d="M12 3c-4.4 0-7 3.2-7 6.6 0 1.1.9 1.9 2 1.9h10c1.1 0 2-.8 2-1.9C19 6.2 16.4 3 12 3z"
        fill="#ff79c6"
      />
      <path
        d="M7.5 13.5c.3 2-.8 3.2-.5 5M11 14c.2 2.4-.6 3.6-.3 6M14.5 13.8c.3 2-.7 3.4-.4 5.4M17 13.5c.2 1.6-.6 2.6-.4 4.2"
        stroke="#20a4f3"
        strokeWidth="1.4"
        fill="none"
        strokeLinecap="round"
      />
    </svg>
  );
}

function BenchNav() {
  const { pref, cycleTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const navRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const onKey = e => e.key === 'Escape' && setOpen(false);
    const onDown = e => {
      if (navRef.current && !navRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onDown);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onDown);
    };
  }, [open]);

  return (
    <>
      {open && <div className="nav-scrim" aria-hidden="true" onClick={() => setOpen(false)} />}
      <nav className={open ? 'nav open' : 'nav'} ref={navRef}>
        <a className="brand" href="/" aria-label="GPU.js benchmark — home">
          <Jelly />
          <span className="brand-word">GPU.js</span> <span className="tag-pill">benchmark</span>
        </a>
        <button
          type="button"
          className="nav-burger"
          aria-expanded={open}
          aria-controls="bench-menu"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen(o => !o)}
        >
          <span aria-hidden="true">{open ? '✕' : '☰'}</span>
        </button>
        <div className="links" id="bench-menu" onClick={() => setOpen(false)}>
          <a href="/">Home</a>
          <a href="/learn/">Learn</a>
          <a href="/api/">API</a>
          <a href="/examples">Examples</a>
          <a href="/benchmark" className="active">Benchmark</a>
          <a href="https://github.com/gpujs/gpu.js">GitHub</a>
        </div>
        <button
          type="button"
          className="btn theme-btn"
          onClick={cycleTheme}
          aria-label={`Theme: ${capitalize(pref)} — tap to change`}
        >
          <span aria-hidden="true">{pref === 'dark' ? '☾' : pref === 'light' ? '☀' : '◐'}</span>
        </button>
      </nav>
    </>
  );
}

function fmtMs(ms) {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 10) return `${ms.toFixed(0)} ms`;
  if (ms >= 1) return `${ms.toFixed(1)} ms`;
  if (ms >= 0.1) return `${ms.toFixed(2)} ms`;
  return `${(ms * 1000).toFixed(0)} µs`;
}

function fmtX(x) {
  return x < 1 ? `${x.toFixed(2)}×` : `${x.toFixed(1)}×`;
}

function Cell({ cell, base, isBaseline, bare }) {
  const cls = ['num', bare ? 'bare' : '', isBaseline ? 'baseline' : ''].filter(Boolean).join(' ');
  if (!cell) return <td className={cls}><span className="dash">—</span></td>;
  if (cell.running) {
    return (
      <td className={`${cls} working`}>
        <span className="spin" role="status" aria-label="measuring">
          <i />
          <i />
          <i />
        </span>
      </td>
    );
  }
  if (cell.na) return <td className={`${cls} na`} title={cell.reason}><span className="dash">N/A</span></td>;
  if (cell.error) return <td className={`${cls} err`} title={cell.error}><span className="dash">error</span></td>;
  if (cell.tooSlow) return <td className={`${cls} na`} title="one run exceeded the ceiling"><span className="dash">too slow</span></td>;

  const x = base && cell.ms ? base / cell.ms : null;
  const tone = isBaseline ? 'base' : !x ? 'flat' : x >= 1.5 ? 'win' : x < 0.95 ? 'lose' : 'flat';
  // The factor leads. It is the answer to the question the page asks; the
  // millisecond figure is the evidence for it, and reads better underneath.
  return (
    <td className={cls} title={cell.reps ? `median of ${cell.reps}` : undefined}>
      <span className={`x ${tone}`}>
        {cell.wrong ? 'WRONG' : isBaseline ? '1.0×' : x ? fmtX(x) : ''}
      </span>
      <span className="t">{fmtMs(cell.ms)}</span>
      {cell.fellBackTo && <span className="fell" title={`gpu.js actually ran this on ${cell.fellBackTo}`}>↘{cell.fellBackTo}</span>}
    </td>
  );
}

function BenchTable({ rows, results, onRun, readOnly }) {
  return (
    <div className="tablecard">
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Work</th>
              {COLUMNS.map(c => (
                <th key={c.id} className={c.bare ? 'bare' : undefined}>
                  {c.label}
                  <span className="g">{c.sub}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(w => {
              const cells = results[w.id] || {};
              const base = cells[BASELINE] && cells[BASELINE].ms;
              return (
                <tr key={w.id} className={cells.__running ? 'running' : undefined}>
                  {/* the grid lives on an inner div, never on the th: a th with
                      display:grid is no longer a table-cell, and position:sticky
                      stops tracking the scroll container — the header stayed put
                      while these scrolled away underneath it */}
                  <th scope="row" className="work">
                    <div className="workinner">
                      {/* disabled, not removed: pulling the button out of the
                          row would reflow every name the moment you switch
                          source, and the table would look like it changed */}
                      <button
                        type="button"
                        className="run-one"
                        onClick={() => onRun([w])}
                        disabled={readOnly}
                        aria-label={
                          readOnly ? `Benchmark ${w.name} — disabled while a saved run is shown` : `Benchmark ${w.name}`
                        }
                      >
                        <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
                          <polygon points="6 4 20 12 6 20" fill="currentColor" />
                        </svg>
                      </button>
                      <span className="wname">
                        {w.name}
                        <span className="tag">{w.tag}</span>
                      </span>
                      <span className="wdesc">{w.params}</span>
                    </div>
                  </th>
                  {COLUMNS.map(c => (
                    <Cell
                      key={c.id}
                      cell={cells[c.id]}
                      base={base}
                      isBaseline={c.id === BASELINE}
                      bare={c.bare}
                    />
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Reading order, not filename order. Dense matmul leads because it is the case
// every GPU claim starts from and the one a reader already has an intuition
// for; everything after it is grouped by the primitive it stresses, which is
// what the footer promises.
const ORDERED = (() => {
  const rank = new Map(GROUPS.map(([g], i) => [g, i]));
  return [...workloads].sort((a, b) => {
    if (a.id === 'matmul') return -1;
    if (b.id === 'matmul') return 1;
    const byGroup = (rank.has(a.group) ? rank.get(a.group) : 99) - (rank.has(b.group) ? rank.get(b.group) : 99);
    return byGroup || a.name.localeCompare(b.name);
  });
})();

function BenchPage() {
  // The effective theme ('light' | 'dark'), never the preference: 'auto' is
  // resolved against prefers-color-scheme by the provider, and the stylesheet
  // only knows [data-theme='dark']. Without this the toggle changed its own
  // glyph and nothing else, and auto never resolved at all.
  const { theme } = useTheme();
  const [source, setSource] = useState('live');
  const [savedId, setSavedId] = useState(savedRuns.length ? savedRuns[0].id : '');
  const [live, setLive] = useState({});
  const [status, setStatus] = useState('idle');
  const [gpuInfo, setGpuInfo] = useState('');
  const abortRef = useRef(null);
  const workerRef = useRef(null);

  // The suite runs off the main thread: the plain-JS baseline and the gpu.js
  // CPU backend are synchronous multi-second loops, and on the main thread one
  // row blocked for 23 s and Stop could not be clicked.
  useEffect(() => {
    let w = null;
    try {
      w = new Worker(new URL('./bench.worker.js', import.meta.url), { type: 'module' });
      w.onerror = () => {
        workerRef.current = null;
      };
    } catch (e) {
      w = null;
    }
    workerRef.current = w;
    return () => {
      if (w) w.terminate();
      workerRef.current = null;
    };
  }, []);

  useEffect(() => {
    let dead = false;
    webgpuStatus().then(s => {
      if (!dead) setGpuInfo(s.ok ? s.reason : `${s.reason}`);
    });
    return () => {
      dead = true;
    };
  }, []);

  const saved = useMemo(() => savedRuns.find(r => r.id === savedId), [savedId]);
  const readOnly = source === 'saved';
  const results = readOnly ? (saved ? saved.results : {}) : live;

  // One workload at a time, in the worker. Serial on purpose: two workloads
  // measured at once would be measuring each other.
  const runOne = useCallback((w, ctl) => new Promise(resolve => {
    const worker = workerRef.current;
    if (!worker) {
      // no Worker (or it failed to construct): run here and accept the freeze,
      // which is still better than the page not working at all
      runWorkload(w, {
        GPU,
        signal: ctl,
        onCell: (colId, cell) =>
          setLive(prev => ({ ...prev, [w.id]: { ...(prev[w.id] || {}), __running: true, [colId]: cell } })),
      }).then(cells => {
        setLive(prev => ({ ...prev, [w.id]: cells }));
        resolve();
      });
      return;
    }
    const id = `${w.id}:${Date.now()}`;
    const onMessage = e => {
      const m = e.data || {};
      if (m.id !== id) return;
      if (m.cellId) {
        setLive(prev => ({ ...prev, [w.id]: { ...(prev[w.id] || {}), __running: true, [m.cellId]: m.cell } }));
        return;
      }
      worker.removeEventListener('message', onMessage);
      setLive(prev => ({ ...prev, [w.id]: m.failed ? { __error: m.error } : m.cells }));
      resolve();
    };
    worker.addEventListener('message', onMessage);
    worker.postMessage({ id, workloadId: w.id });
  }), []);

  const run = useCallback(async list => {
    const ctl = { aborted: false };
    abortRef.current = ctl;
    for (let i = 0; i < list.length; i++) {
      if (ctl.aborted) break;
      const w = list[i];
      setStatus(`running ${w.name}${list.length > 1 ? ` — ${i + 1} of ${list.length}` : ''}`);
      setLive(prev => ({ ...prev, [w.id]: { __running: true } }));
      // eslint-disable-next-line no-await-in-loop
      await runOne(w, ctl);
    }
    abortRef.current = null;
    setStatus(ctl.aborted ? 'stopped' : 'done');
  }, [runOne]);

  // How scripts/bench-record.mjs lifts a finished table out of the page. A
  // recorded run has to be the SAME numbers the page just showed, so it is read
  // from the page rather than re-measured somewhere else.
  useEffect(() => {
    window.__benchResults = live;
    window.__benchStatus = status;
  }, [live, status]);

  const running = Boolean(abortRef.current) || status.startsWith('running');

  return (
    <div className="bench-root" data-theme={theme}>
      <BenchNav />
      <div className="wrap">
        <p className="eyebrow">gpu.js · benchmark</p>
        <h1>What the GPU is actually worth</h1>
        <p className="sub">
          {workloads.length} GPGPU workload{workloads.length === 1 ? '' : 's'}, timed on every backend
          gpu.js can reach and against hand-written baselines. The rightmost column is plain
          JavaScript with no gpu.js in the picture — every speed-up on this page is measured
          against it. Each column runs the best implementation for its platform, not the same
          code six times; what is held identical is the problem, and every answer is checked.
        </p>

        <div className="toolbar">
          {/* Both stay put and grey out. A saved run is read-only, but hiding
              the controls moved every other control in the bar, which reads as
              the page having changed rather than the source having. */}
          <button
            type="button"
            className="btn primary"
            disabled={running || readOnly}
            title={readOnly ? 'Showing a saved run — switch to Live to measure on this machine' : undefined}
            onClick={() => run(ORDERED)}
          >
            <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
              <polygon points="6 4 20 12 6 20" fill="currentColor" />
            </svg>
            Run all {workloads.length}
          </button>
          <button
            type="button"
            className="btn"
            disabled={!running || readOnly}
            onClick={() => {
              if (abortRef.current) abortRef.current.aborted = true;
            }}
          >
            Stop
          </button>
          <div className="seg" role="group" aria-label="Result source">
            <button type="button" aria-pressed={source === 'live'} onClick={() => setSource('live')}>
              Live
            </button>
            <button
              type="button"
              aria-pressed={source === 'saved'}
              onClick={() => setSource('saved')}
              disabled={!savedRuns.length}
            >
              Saved run
            </button>
          </div>
          {readOnly && (
            <label className="pick">
              <span className="lbl">Run</span>
              <select value={savedId} onChange={e => setSavedId(e.target.value)}>
                {savedRuns.map(r => (
                  <option key={r.id} value={r.id}>
                    {r.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          <span className="grow" />
          <span className="stat">
            {readOnly ? (saved ? saved.machine : 'no saved runs') : `${status} · ${gpuInfo}`}
          </span>
        </div>

        {readOnly && saved && (
          <div className="ro">
            <b>Read-only.</b>
            <span>
              Showing a saved run — <code>{saved.machine}</code>, {saved.date}. Running is disabled
              so a stored result can never be half-overwritten by this machine.
            </span>
          </div>
        )}

        <BenchTable rows={ORDERED} results={results} onRun={run} readOnly={readOnly} />

        <div className="legend">
          <div>
            <b>The baseline is the last column</b>
            <p>Plain JavaScript, no gpu.js. Every × on the row is that column divided by the cell.</p>
          </div>
          <div>
            <b>Two columns have no gpu.js in them</b>
            <p>Hand-written WebGPU and plain JS, tinted amber. They separate “the GPU is fast” from
              “gpu.js is fast”.</p>
          </div>
          <div>
            <b>N/A means this machine, not this workload</b>
            <p>Every workload here runs on every backend. An N/A is about where you are reading
              from — no WebGPU adapter, or a page served over plain http, which does not expose
              <code> navigator.gpu</code> at all. Hover it for the reason.</p>
          </div>
          <div>
            <b>Each column is the best that platform can do</b>
            <p>Not one algorithm compiled six ways. Selecting the top 512 is a heap in plain JS
              and a threshold bisection on a GPU, because that is what each is good at. What is
              held identical is the <em>problem</em> and the answer — every column is checksummed
              against the same oracle.</p>
          </div>
          <div>
            <b>Median of ≥3, after 2 warm-ups</b>
            <p>The first call compiles. Every column is also checksummed against the plain-JS
              result, and a mismatch reads WRONG rather than fast.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function BenchApp() {
  return (
    <ThemeProvider>
      <BenchPage />
    </ThemeProvider>
  );
}
