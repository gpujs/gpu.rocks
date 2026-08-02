import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GPU } from 'gpu.js';
import { ThemeProvider, useTheme } from '../Learn/ThemeContext';
import workloads from './workloads/index.js';
import savedRuns from './saved/index.js';
import { BASELINE, COLUMNS, runWorkload, webgpuAvailable } from './runner.js';
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
  if (cell.running) return <td className={cls}><span className="spin" aria-label="running">•••</span></td>;
  if (cell.na) return <td className={`${cls} na`} title={cell.reason}><span className="dash">N/A</span></td>;
  if (cell.error) return <td className={`${cls} err`} title={cell.error}><span className="dash">error</span></td>;
  if (cell.tooSlow) return <td className={`${cls} na`} title="one run exceeded the ceiling"><span className="dash">too slow</span></td>;

  const x = base && cell.ms ? base / cell.ms : null;
  const tone = isBaseline ? 'base' : !x ? 'flat' : x >= 1.5 ? 'win' : x < 0.95 ? 'lose' : 'flat';
  return (
    <td className={cls} title={cell.reps ? `median of ${cell.reps}` : undefined}>
      <span className="t">{fmtMs(cell.ms)}</span>
      <span className={`x ${tone}`}>
        {cell.wrong ? 'WRONG' : isBaseline ? '1.0×' : x ? fmtX(x) : ''}
      </span>
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
                  <th scope="row" className="work">
                    {!readOnly && (
                      <button
                        type="button"
                        className="run-one"
                        onClick={() => onRun([w])}
                        aria-label={`Benchmark ${w.name}`}
                      >
                        <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
                          <polygon points="6 4 20 12 6 20" fill="currentColor" />
                        </svg>
                      </button>
                    )}
                    <span className="wname">
                      {w.name}
                      <span className="tag">{w.tag}</span>
                    </span>
                    <span className="wdesc">{w.params}</span>
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

function BenchPage() {
  const [source, setSource] = useState('live');
  const [savedId, setSavedId] = useState(savedRuns.length ? savedRuns[0].id : '');
  const [live, setLive] = useState({});
  const [status, setStatus] = useState('idle');
  const [gpuInfo, setGpuInfo] = useState('');
  const abortRef = useRef(null);

  useEffect(() => {
    let dead = false;
    webgpuAvailable().then(ok => {
      if (!dead) setGpuInfo(ok ? 'WebGPU adapter present' : 'no WebGPU adapter — those columns will read N/A');
    });
    return () => {
      dead = true;
    };
  }, []);

  const saved = useMemo(() => savedRuns.find(r => r.id === savedId), [savedId]);
  const readOnly = source === 'saved';
  const results = readOnly ? (saved ? saved.results : {}) : live;

  const run = useCallback(async list => {
    const ctl = { aborted: false };
    abortRef.current = ctl;
    for (let i = 0; i < list.length; i++) {
      if (ctl.aborted) break;
      const w = list[i];
      setStatus(`running ${w.name}${list.length > 1 ? ` — ${i + 1} of ${list.length}` : ''}`);
      setLive(prev => ({ ...prev, [w.id]: { __running: true } }));
      // eslint-disable-next-line no-await-in-loop
      const cells = await runWorkload(w, {
        GPU,
        signal: ctl,
        onCell: (colId, cell) =>
          setLive(prev => ({ ...prev, [w.id]: { ...(prev[w.id] || {}), __running: true, [colId]: cell } })),
      });
      setLive(prev => ({ ...prev, [w.id]: cells }));
    }
    abortRef.current = null;
    setStatus(ctl.aborted ? 'stopped' : 'done');
  }, []);

  // How scripts/bench-record.mjs lifts a finished table out of the page. A
  // recorded run has to be the SAME numbers the page just showed, so it is read
  // from the page rather than re-measured somewhere else.
  useEffect(() => {
    window.__benchResults = live;
    window.__benchStatus = status;
  }, [live, status]);

  const running = Boolean(abortRef.current) || status.startsWith('running');

  return (
    <div className="bench-root">
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
          {!readOnly && (
            <>
              <button type="button" className="btn primary" disabled={running} onClick={() => run(workloads)}>
                <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                  <polygon points="6 4 20 12 6 20" fill="currentColor" />
                </svg>
                Run all {workloads.length}
              </button>
              <button
                type="button"
                className="btn"
                disabled={!running}
                onClick={() => {
                  if (abortRef.current) abortRef.current.aborted = true;
                }}
              >
                Stop
              </button>
            </>
          )}
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

        <BenchTable rows={workloads} results={results} onRun={run} readOnly={readOnly} />

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
            <b>N/A is a fact, not a gap</b>
            <p>Hover it for the reason — a graphical kernel pins unsigned precision, and
              <code> Math.random</code> has no WebGPU equivalent.</p>
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
