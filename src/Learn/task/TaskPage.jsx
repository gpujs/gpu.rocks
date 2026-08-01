import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Navigate, useNavigate, useParams } from 'react-router';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { EditorView, keymap } from '@codemirror/view';
import { indentUnit } from '@codemirror/language';
import { indentWithTab } from '@codemirror/commands';
import { Prec } from '@codemirror/state';
import LearnNav from '../components/LearnNav';
import TaskDots from '../components/TaskDots';
import { useTheme } from '../ThemeContext';
import {
  tracks,
  getTask,
  moduleNumber,
  nextModule as nextModuleOf,
  parseModulePath,
  taskUrl,
} from '../content/index';
import { getFigures } from '../content/figures/index';
import { runUserCode, runTests, snapshotCanvas, utils, warmUpSandbox } from '../engine/runner';
import { runBenchmark } from '../engine/benchmark';
import {
  getSavedCode,
  saveCode,
  clearCode,
  markTaskDone,
  isTaskDone,
  moduleProgress,
} from '../engine/storage';
import { lightEditorTheme, darkEditorTheme } from './editorThemes';
import { moduleTaskMeta } from '../../routeMeta';
import { setPageMeta } from '../../pageMeta';
import { learnCompletions } from './completion/completions';
import { signatureHelp } from './completion/signatureHelp';
import { inputHover } from './completion/inputHover';
import { taskInputDocs } from './inputDoc';
import { starterCodeFor } from './starterPreamble';
import ConsolePane from './ConsolePane';
import CompletionModal from './CompletionModal';
import { highlightCodeBlocks } from './highlightCode';

// ---- helpers ---------------------------------------------------------------

// render() snapshots at log time (see engine/sandbox.js); this is the fallback
// for a canvas whose pixels couldn't be captured then — retried right after the
// run, in the same microtask chain, before the browser can composite + clear
// it. Only the main-thread fallback path can ever hit it: a worker run has no
// live canvas to retry, it ships an ImageBitmap that runner.js has already
// turned into a data URL.
function decorateLog(log) {
  if (log.type === 'canvas' && log.canvas && !log.snapshot) {
    const snapshot = snapshotCanvas(log.canvas);
    if (snapshot) return { ...log, snapshot };
  }
  return log;
}

function fmtMs(ms) {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) return '—';
  return ms >= 100 ? ms.toFixed(0) : ms.toFixed(1);
}

function fmtRatio(ratio) {
  if (!Number.isFinite(ratio)) return '∞';
  return ratio >= 10 ? ratio.toFixed(0) : ratio.toFixed(1);
}

// ---- benchmark chip --------------------------------------------------------

// below Chrome's coarsened performance.now() resolution a median is noise
const TIMER_FLOOR_MS = 0.05;

function BenchChip({ bench }) {
  if (!bench) return null;
  if (bench.gpuUnavailable) {
    return (
      <span className="bench-chip note">
        GPU unavailable here
        {typeof bench.cpuMs === 'number' ? ` · CPU ${fmtMs(bench.cpuMs)} ms` : ''}
      </span>
    );
  }
  if (bench.gpuFailed) {
    return (
      <span className="bench-chip note">
        code failed in GPU mode
        {bench.error ? ` · ${bench.error.message}` : ''}
        {typeof bench.cpuMs === 'number' ? ` · CPU ${fmtMs(bench.cpuMs)} ms` : ''}
      </span>
    );
  }
  if (bench.error) {
    return <span className="bench-chip note">benchmark failed · {bench.error.message}</span>;
  }
  if (bench.gpuRanOnCpu) {
    return (
      <span className="bench-chip note">
        no GPU comparison — gpu.js ran this kernel on the CPU in both modes ·{' '}
        {fmtMs(bench.cpuMs)} ms vs {fmtMs(bench.gpuMs)} ms
      </span>
    );
  }
  if (bench.cpuMs < TIMER_FLOOR_MS || bench.gpuMs < TIMER_FLOOR_MS) {
    return (
      <span className="bench-chip note">
        too fast to compare · CPU {fmtMs(bench.cpuMs)} ms vs GPU {fmtMs(bench.gpuMs)} ms
      </span>
    );
  }
  if (bench.fasterOn === 'cpu') {
    const ratio = bench.cpuMs > 0 ? bench.gpuMs / bench.cpuMs : Infinity;
    return (
      <span className="bench-chip cpu">
        CPU {fmtRatio(ratio)}× faster{' '}
        <span className="vs">· {fmtMs(bench.cpuMs)} ms vs {fmtMs(bench.gpuMs)} ms</span>
      </span>
    );
  }
  return (
    <span className="bench-chip">
      GPU {fmtRatio(bench.ratio)}× faster{' '}
      <span className="vs">· {fmtMs(bench.gpuMs)} ms vs {fmtMs(bench.cpuMs)} ms</span>
    </span>
  );
}

// screen-reader summary of a benchmark result (visually the chip shows it)
function benchStatusMessage(b) {
  if (!b) return 'Benchmark finished.';
  if (b.gpuUnavailable) return 'Benchmark finished — GPU unavailable here.';
  if (b.gpuFailed) {
    return `Benchmark finished — code failed in GPU mode${b.error ? `: ${b.error.message}` : ''}`;
  }
  if (b.error) return `Benchmark failed — ${b.error.message}`;
  if (b.gpuRanOnCpu) return 'Benchmark finished — no GPU comparison, this kernel ran on the CPU backend.';
  if (b.cpuMs < TIMER_FLOOR_MS || b.gpuMs < TIMER_FLOOR_MS) {
    return 'Benchmark finished — both runs too fast to compare.';
  }
  if (b.fasterOn === 'cpu') {
    const ratio = b.cpuMs > 0 ? b.gpuMs / b.cpuMs : Infinity;
    return `Benchmark finished — CPU ${fmtRatio(ratio)} times faster.`;
  }
  return `Benchmark finished — GPU ${fmtRatio(b.ratio)} times faster.`;
}

// ---- tests panel -----------------------------------------------------------

function TestRow({ name, result }) {
  const status = result ? (result.passed ? 'PASS' : 'FAIL') : '—';
  const stClass = result ? (result.passed ? 'pass' : 'fail') : 'lock';
  return (
    <div className="test">
      <span className={`st ${stClass}`}>{status}</span>
      <span className="desc">
        {/* trusted HTML: test names are authored in-repo and may contain <code> */}
        <span dangerouslySetInnerHTML={{ __html: name }} />
        {result && !result.passed && result.error && (
          <span className="fail-msg">{result.error}</span>
        )}
      </span>
      <span className="ms">{result ? `${fmtMs(result.ms)} ms` : ''}</span>
    </div>
  );
}

function TestsPanel({ task, report, active }) {
  const publicTests = task.publicTests || [];
  const privateTests = task.privateTests || [];
  const pub = report ? report.results.filter(r => !r.private) : null;
  const priv = report ? report.results.filter(r => r.private) : null;
  return (
    <div
      className={`bpanel tests${active ? ' on' : ''}`}
      role="tabpanel"
      id="bpanel-tests"
      aria-labelledby="btab-tests"
    >
      <div className="testlist">
        {publicTests.map((t, i) => (
          <TestRow key={i} name={t.name} result={pub ? pub[i] : null} />
        ))}
        {privateTests.length > 0 && (
          <div className="privnote">
            🔒 {privateTests.length} private test{privateTests.length === 1 ? '' : 's'} — hidden
            inputs, same rules
          </div>
        )}
        {priv &&
          priv.map((r, i) => (
            <div className="test" key={`p${i}`}>
              <span className={`st ${r.passed ? 'pass' : 'fail'}`}>
                {r.passed ? 'PASS' : 'FAIL'}
              </span>
              <span className="desc">private test #{i + 1}</span>
              <span className="ms">{fmtMs(r.ms)} ms</span>
            </div>
          ))}
      </div>
      {report && report.allPassed && report.total > 0 && (
        <div className="testsum">
          ✓ {report.passed} of {report.total} passed — Next task unlocked
        </div>
      )}
    </div>
  );
}

// ---- brief pane ------------------------------------------------------------

// All HTML fragments here are trusted: course content authored in this repo.
// That includes the figure SVGs — in-repo markup, injected verbatim.
function Figures({ figures }) {
  return figures.map((fig, i) => (
    <figure className="diagram" key={i}>
      <div dangerouslySetInnerHTML={{ __html: fig.svg }} />
      <figcaption>{fig.caption}</figcaption>
    </figure>
  ));
}

// The task's injected globals, described from the values themselves (see
// task/inputDoc.js). Discoverable without knowing to type anything — the
// editor's completion popup and hover tooltip show the same descriptors.
// Renders nothing for a task with no inputs.
function TaskInputs({ docs }) {
  if (!docs || docs.length === 0) return null;
  return (
    <div className="taskinputs">
      <b>Task inputs</b>
      <p className="lead">Already defined as globals — use them directly.</p>
      <dl>
        {docs.map(doc => (
          <React.Fragment key={doc.name}>
            <dt>
              <code className="iname">{doc.name}</code>
              <code className="itype">{doc.type}</code>
            </dt>
            <dd>
              <span className="isum">{doc.summary}</span>
              {doc.sample && <code className="isample">{doc.sample}</code>}
              {doc.note && <span className="inote">{doc.note}</span>}
            </dd>
          </React.Fragment>
        ))}
      </dl>
    </div>
  );
}

function BriefPane({ module, task, taskNum, total, inputDocs, full, onToggleFull }) {
  const figures = getFigures(module, task);
  const briefRef = useRef(null);
  // Static syntax highlighting for authored <pre><code> blocks. Layout effect
  // so the spans exist before paint (no flash of unhighlighted code, no
  // post-paint layout thrash); idempotent, re-runs when the task changes.
  useLayoutEffect(() => {
    highlightCodeBlocks(briefRef.current);
  }, [task]);
  return (
    <aside className={full ? 'brief brief-full' : 'brief'} ref={briefRef}>
      {/* Phones only (hidden by media query): the brief is capped at a fraction
          of a phone screen so the editor and output have room, which makes a
          long task description a keyhole. */}
      <button
        type="button"
        className="brief-fullbtn"
        onClick={onToggleFull}
        aria-label={full ? 'Exit full screen task description' : 'Full screen task description'}
      >
        {full ? '✕ Close' : '⛶'}
      </button>
      <p className="eyebrow">Task {taskNum} of {total}</p>
      <h2>{task.title}</h2>
      <div dangerouslySetInnerHTML={{ __html: task.intro || '' }} />
      <Figures figures={figures.filter(f => f.placement === 'intro')} />
      {task.goal && (
        <div className="goal" dangerouslySetInnerHTML={{ __html: task.goal }} />
      )}
      <Figures figures={figures.filter(f => f.placement === 'goal')} />
      {Array.isArray(task.requirements) && task.requirements.length > 0 && (
        <ul className="reqs">
          {task.requirements.map((req, i) => (
            <li key={i} dangerouslySetInnerHTML={{ __html: req }} />
          ))}
        </ul>
      )}
      <TaskInputs docs={inputDocs} />
      {(task.hints || []).map((hint, i) => (
        <details key={i}>
          <summary>{hint.title}</summary>
          <div dangerouslySetInnerHTML={{ __html: hint.body }} />
        </details>
      ))}
      {task.transfer && (
        <div className="transfer">
          <b>Same idea elsewhere</b>
          <span dangerouslySetInnerHTML={{ __html: task.transfer }} />
        </div>
      )}
    </aside>
  );
}

// ---- workspace (keyed by taskId so state resets per task) ------------------

function TaskWorkspace({ module, task, taskNum, taskIndex, taskId, total }) {
  const navigate = useNavigate();
  const { theme } = useTheme();

  // The task's injected globals, described from their real values. One shared
  // set of descriptors feeds the brief pane, the starter preamble, the
  // completion popup and the editor hover — and taskInputDocs memoizes per task
  // object, so building the inputs (a 512×512 test image, for one task) happens
  // once, never per keystroke and not again on remount.
  const inputDocs = useMemo(() => taskInputDocs(task, utils), [task]);

  // THE starter code, comment preamble and all. Both the initial document and
  // Reset read this one value, so they cannot drift apart.
  const starterCode = useMemo(() => starterCodeFor(task, inputDocs), [task, inputDocs]);

  // The editor is intentionally UNcontrolled after mount (stable `value` prop):
  // @uiw/react-codemirror's controlled-value sync can arm a stale forceUpdate
  // while its "typing latch" is active (its 1 ms-interval scheduler is clamped
  // /throttled by browsers), which later REPLACES the doc with an old value —
  // deleting the user's last keystrokes. TaskWorkspace remounts per task
  // (key={taskId}), so the initial doc is all the value prop ever needs to
  // carry; edits flow through onChange → codeRef, resets dispatch to the view.
  // Saved code always wins: a learner who has written anything never has the
  // preamble pushed back into their document.
  const [initialCode] = useState(() => {
    const saved = getSavedCode(taskId);
    return saved != null ? saved : starterCode;
  });
  const [mode, setMode] = useState('auto');
  // WebGPU has no graphical mode — gpu.js refuses `graphical: true` on that
  // backend outright. Auto handles it silently (the kernel declines the upgrade
  // and paints on WebGL), but offering an explicit WebGPU option that can only
  // throw would be a trap, so it is disabled here and says why. Judged from the
  // task rather than the editor's current text so the option does not flicker
  // in and out while someone types.
  const graphicalTask = useMemo(() => {
    const source = `${task.starterCode || ''}\n${task.solutionCode || ''}`;
    return /graphical:\s*true|this\.color\s*\(/.test(source);
  }, [task]);
  const [logs, setLogs] = useState([]);
  // Controls the last run DECLARED (via slider()), and the values to run with
  // next. The program is a pure function of its controls: moving one re-runs it.
  const [controls, setControls] = useState([]);
  const [controlValues, setControlValues] = useState({});
  const controlValuesRef = useRef({});
  const [report, setReport] = useState(null);
  const [bench, setBench] = useState(null);
  const [running, setRunning] = useState(false);
  const [benching, setBenching] = useState(false);
  const [tab, setTab] = useState('console');
  const [done, setDone] = useState(() => isTaskDone(taskId));
  const [fullscreen, setFullscreen] = useState(false);
  // Phones only (the button is hidden above 720px). Mutually exclusive with the
  // editor's full screen — two fixed-inset panels would stack.
  const [panelFull, setPanelFull] = useState(false);
  const [briefFull, setBriefFull] = useState(false);
  const [progressTick, setProgressTick] = useState(0);
  const [status, setStatus] = useState(''); // announced via the role=status region
  const [celebration, setCelebration] = useState(null); // null | { kind: 'module' | 'track' }

  const codeRef = useRef(initialCode);
  const viewRef = useRef(null);
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const busyRef = useRef(false);
  const saveTimer = useRef(null);
  const nextBtnRef = useRef(null);
  const consoleTabRef = useRef(null);
  const testsTabRef = useRef(null);

  // title/description/canonical for this task — same values the prerender
  // bakes into the static HTML for this route
  useEffect(() => {
    setPageMeta(moduleTaskMeta(module, task, taskNum));
  }, [module, task, taskNum]);

  // Spawn the sandbox worker (gpu.js + the content registry evaluate in it)
  // while the learner reads the brief, so the first ▶ Run does not pay for it.
  useEffect(() => {
    warmUpSandbox();
  }, []);

  const track = tracks.find(t => t.number === module.track);
  const progress = useMemo(
    () => moduleProgress(module),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [module, progressTick, done]
  );

  // debounced (~500 ms) save of the editor content
  const handleChange = useCallback(
    value => {
      codeRef.current = value;
      clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(() => {
        saveTimer.current = null;
        saveCode(taskId, value);
      }, 500);
    },
    [taskId]
  );
  // on unmount, FLUSH a pending save (don't discard it) — navigating away
  // within 500 ms of the last keystroke must not lose the user's edits
  useEffect(
    () => () => {
      if (saveTimer.current != null) {
        clearTimeout(saveTimer.current);
        saveTimer.current = null;
        saveCode(taskId, codeRef.current);
      }
    },
    [taskId]
  );

  // Moving a slider re-runs the program with the new value. Deliberately NOT a
  // full Run: no tests, no progress marking, no status text — dragging a zoom
  // slider must not be able to mark a task complete, and re-running the test
  // suite on every input event would be absurd. Only the console output is
  // replaced. Debounced, because a range input fires continuously while dragged
  // and each re-run recompiles the kernel.
  const controlTimer = useRef(null);
  const [controlsBusy, setControlsBusy] = useState(false);
  const handleControlChange = useCallback(
    (name, value) => {
      setControlValues(prev => {
        const next = { ...prev, [name]: value };
        controlValuesRef.current = next;
        return next;
      });
      if (controlTimer.current) clearTimeout(controlTimer.current);
      controlTimer.current = setTimeout(async () => {
        if (busyRef.current) return;
        busyRef.current = true;
        setControlsBusy(true);
        try {
          const result = await runUserCode(codeRef.current, {
            mode: modeRef.current,
            task,
            controls: controlValuesRef.current,
          });
          setLogs(result.logs.map(decorateLog));
          if (result.controls && result.controls.length) setControls(result.controls);
        } finally {
          busyRef.current = false;
          setControlsBusy(false);
        }
      }, 120);
    },
    [task]
  );

  useEffect(() => () => { if (controlTimer.current) clearTimeout(controlTimer.current); }, []);

  const handleRun = useCallback(async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    setRunning(true);
    try {
      const result = await runUserCode(codeRef.current, {
        mode: modeRef.current,
        task,
        controls: controlValuesRef.current,
      });
      // snapshot canvases immediately, before the browser composites the frame
      setLogs(result.logs.map(decorateLog));
      setControls(result.controls || []);
      // Seed the sliders from what the program actually ran with, so the first
      // render shows them in the right position rather than at their minimum.
      if (result.controls && result.controls.length) {
        setControlValues(prev => {
          const next = { ...prev };
          for (const c of result.controls) if (next[c.name] === undefined) next[c.name] = c.value;
          controlValuesRef.current = next;
          return next;
        });
      }
      const rep = await runTests(task, result);
      setReport(rep);
      if (rep.total > 0 && rep.allPassed) {
        markTaskDone(taskId);
        setDone(true);
        setProgressTick(t => t + 1);
      }
      setStatus(
        !result.ok
          ? `Run failed — ${result.error.message}`
          : rep.total > 0
            ? rep.allPassed
              ? `Run complete — all ${rep.total} tests passed. Next task unlocked.`
              : `Run complete — ${rep.passed} of ${rep.total} tests passed, ${rep.total - rep.passed} failed.`
            : 'Run complete.'
      );
    } finally {
      setRunning(false);
      busyRef.current = false;
    }
  }, [task, taskId]);

  const handleBenchmark = useCallback(async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBenching(true);
    try {
      const result = await runBenchmark(codeRef.current, task);
      setBench(result);
      setStatus(benchStatusMessage(result));
    } finally {
      setBenching(false);
      busyRef.current = false;
    }
  }, [task]);

  const handleReset = useCallback(() => {
    if (!window.confirm('Reset code to the starter? Your changes to this task will be lost.')) {
      return;
    }
    clearTimeout(saveTimer.current);
    saveTimer.current = null; // an explicit Reset must not be flushed back on unmount
    clearCode(taskId);
    codeRef.current = starterCode;
    const view = viewRef.current;
    if (view) {
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: starterCode },
      });
    }
    setReport(null);
    setBench(null);
  }, [starterCode, taskId]);

  // The next module WITHIN this module's track, or null — the registry already
  // refuses to cross a track boundary, and always returns null for an orphan.
  const nextModule = useMemo(() => nextModuleOf(module), [module]);

  // Which completion this module can reach:
  //   'standalone' — no track. Finishing it offers only the way out: an
  //                  unordered module has nothing to be "next".
  //   'track'      — last module of its track. The track celebration.
  //   'module'     — anything else. Offers the next module in the track.
  const completionKind = module.track == null ? 'standalone' : nextModule ? 'module' : 'track';

  // Prev walks back within the module only — it never crosses into the
  // previous module (Next does, but backwards that would drop you at the far
  // end of work you may not have started). At task 1 the way out is the
  // course list, which is its own button.
  const prevEnabled = taskNum > 1;

  const handlePrev = useCallback(() => {
    if (taskNum > 1) navigate(taskUrl(module, taskNum - 1));
  }, [navigate, module, taskNum]);

  const handleAllModules = useCallback(() => navigate('/learn'), [navigate]);

  const handleNext = useCallback(() => {
    if (taskNum < total) {
      navigate(taskUrl(module, taskNum + 1));
      return;
    }
    // last task of the module AND every task done (including this one):
    // celebrate instead of navigating
    if (moduleProgress(module).done === total) {
      setCelebration({ kind: completionKind });
      setStatus(completionKind === 'track' ? 'Track complete' : 'Module complete');
      return;
    }
    // earlier tasks were skipped — keep the plain navigation behavior
    navigate(nextModule ? taskUrl(nextModule, 1) : '/learn');
  }, [navigate, module, taskNum, total, nextModule, completionKind]);

  // Esc / backdrop close: stay on the task, hand focus back to the Next button
  const closeCelebration = useCallback(() => {
    setCelebration(null);
    if (nextBtnRef.current) nextBtnRef.current.focus();
  }, []);

  // ⌘/Ctrl + Enter anywhere on the page runs the code. The CodeMirror keymap
  // below preventDefaults first, so check defaultPrevented to avoid a double run.
  useEffect(() => {
    const onKey = e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !e.defaultPrevented) {
        e.preventDefault();
        handleRun();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleRun]);

  const extensions = useMemo(
    () => [
      javascript(),
      indentUnit.of('  '),
      keymap.of([indentWithTab]),
      Prec.highest(
        keymap.of([
          {
            key: 'Mod-Enter',
            run: () => {
              handleRun();
              return true;
            },
          },
        ])
      ),
      ...learnCompletions(inputDocs),
      ...signatureHelp(),
      ...inputHover(inputDocs),
      EditorView.contentAttributes.of({ 'aria-label': 'kernel.js code editor' }),
    ],
    [handleRun, inputDocs]
  );

  // ARIA tabs pattern: arrow keys move + activate within the tablist
  const handleTablistKey = useCallback(
    e => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const next = tab === 'console' ? 'tests' : 'console';
      setTab(next);
      const ref = next === 'console' ? consoleTabRef : testsTabRef;
      if (ref.current) ref.current.focus();
    },
    [tab]
  );

  const busy = running || benching;
  const nextEnabled = done || Boolean(report && report.total > 0 && report.allPassed);
  const passedNow = Boolean(report && report.total > 0 && report.allPassed);
  const testBadge = report ? `${report.passed}/${report.total}` : null;

  return (
    <div className="taskscreen">
      <LearnNav />

      <div className="crumbbar">
        {/* an orphan module has no track and no number: it says what it is */}
        <span>{track ? `Track ${track.number} · ${track.title}` : 'Standalone module'}</span>
        <span className="sep">/</span>
        <b>
          {moduleNumber(module) ? `Module ${moduleNumber(module)} — ` : ''}
          {module.title}
        </b>
        <span className="sep">/</span>
        <span className="tcount">Task {taskNum} of {total}</span>
        <TaskDots total={total} doneCount={progress.done} currentIndex={taskIndex} />
      </div>

      <div className="toolbar">
        <button type="button" className="tb-run" onClick={handleRun} disabled={busy}>
          {running ? '… Running' : '▶ Run'}
        </button>
        <label className="tb-select">
          <span className="lbl">Mode</span>
          <select
            value={mode}
            onChange={e => setMode(e.target.value)}
            aria-label="Execution mode"
          >
            <option value="auto">Auto</option>
            <option value="webgpu" disabled={graphicalTask}>
              {graphicalTask ? 'WebGPU (not for graphical)' : 'WebGPU'}
            </option>
            <option value="webgl">WebGL</option>
            <option value="cpu">CPU</option>
          </select>
          <span aria-hidden="true">▾</span>
        </label>
        <button type="button" className="tb-btn" onClick={handleBenchmark} disabled={busy}>
          {benching ? '⏱ Benchmarking…' : '⏱ Benchmark'}
        </button>
        <BenchChip bench={bench} />
        <div className="tb-right">
          {passedNow && <span className="pass-note">✓ All tests passed</span>}
          <button type="button" className="tb-btn" onClick={handleReset} aria-label="Reset code">
            Reset<span className="tb-word"> code</span>
          </button>
          {/* same wording as the completion modal's way out, so leaving a
              module reads the same whether you finished it or not */}
          <button type="button" className="tb-btn" onClick={handleAllModules} aria-label="Exit module">
            Exit<span className="tb-word"> module</span>
          </button>
          <button
            type="button"
            className="tb-btn"
            onClick={handlePrev}
            disabled={!prevEnabled}
            aria-label="Previous task"
          >
            ← Prev<span className="tb-word"> task</span>
          </button>
          <button
            type="button"
            className="tb-next"
            ref={nextBtnRef}
            onClick={handleNext}
            disabled={!nextEnabled}
            aria-label="Next task"
          >
            Next<span className="tb-word"> task</span> →
          </button>
        </div>
      </div>

      {/* visually hidden — announces run/benchmark outcomes to screen readers */}
      <div className="sr-status" role="status" aria-live="polite">
        {status}
      </div>

      <div className="workspace">
        <BriefPane
          module={module}
          task={task}
          taskNum={taskNum}
          total={total}
          inputDocs={inputDocs}
          full={briefFull}
          onToggleFull={() => { setFullscreen(false); setPanelFull(false); setBriefFull(f => !f); }}
        />

        <div className="editpane">
          <div className={fullscreen ? 'editor editor-fullscreen' : 'editor'}>
            <div className="edtabs">
              <span className="edtab on">kernel.js</span>
              {/* WCAG 2.1.2: Tab indents in the editor, so advertise the way out */}
              <span className="ed-hint">Esc then Tab exits the editor</span>
              <button
                type="button"
                className="ed-fullbtn"
                onClick={() => { setPanelFull(false); setBriefFull(false); setFullscreen(f => !f); }}
                aria-label={fullscreen ? 'Exit full screen editor' : 'Full screen editor'}
              >
                {fullscreen ? '✕ Close' : '⛶ Full screen'}
              </button>
            </div>
            <CodeMirror
              className="cm-host"
              value={initialCode}
              onChange={handleChange}
              onCreateEditor={view => {
                viewRef.current = view;
                // exposes the EditorView for the headless verification harness
                window.__learnEditorView = view;
              }}
              theme={theme === 'dark' ? darkEditorTheme : lightEditorTheme}
              extensions={extensions}
              basicSetup={{
                foldGutter: false,
                autocompletion: false,
                highlightSelectionMatches: false,
              }}
            />
          </div>

          <div className={panelFull ? 'bottompane bottompane-full' : 'bottompane'}>
            <div
              className="btabs"
              role="tablist"
              aria-label="Output panes"
              onKeyDown={handleTablistKey}
            >
              <button
                type="button"
                role="tab"
                id="btab-console"
                ref={consoleTabRef}
                className={`btab${tab === 'console' ? ' on' : ''}`}
                aria-selected={tab === 'console'}
                aria-controls="bpanel-console"
                tabIndex={tab === 'console' ? 0 : -1}
                onClick={() => setTab('console')}
              >
                Console
              </button>
              <button
                type="button"
                role="tab"
                id="btab-tests"
                ref={testsTabRef}
                className={`btab${tab === 'tests' ? ' on' : ''}`}
                aria-selected={tab === 'tests'}
                aria-controls="bpanel-tests"
                tabIndex={tab === 'tests' ? 0 : -1}
                onClick={() => setTab('tests')}
              >
                Tests {testBadge && <span className="cnt">{testBadge}</span>}
              </button>
              <button
                type="button"
                className="panel-fullbtn"
                onClick={() => { setFullscreen(false); setBriefFull(false); setPanelFull(f => !f); }}
                aria-label={panelFull ? 'Exit full screen output' : 'Full screen output'}
              >
                {panelFull ? '✕ Close' : '⛶'}
              </button>
              <button type="button" className="clearbtn" onClick={() => setLogs([])}>
                clear
              </button>
            </div>
            <ConsolePane
              logs={logs}
              active={tab === 'console'}
              controls={controls}
              controlValues={controlValues}
              onControlChange={handleControlChange}
              controlsBusy={controlsBusy}
            />
            <TestsPanel task={task} report={report} active={tab === 'tests'} />
          </div>
        </div>
      </div>

      {celebration && (
        <CompletionModal
          kind={celebration.kind}
          module={module}
          track={track}
          totalTasks={total}
          courseEnd={
            celebration.kind === 'track' &&
            Boolean(track) &&
            tracks.length > 0 &&
            track.number === tracks[tracks.length - 1].number
          }
          onClose={closeCelebration}
          onEnd={() => navigate('/learn')}
          // only ever called for kind 'module', which is the only kind that
          // has a next module to go to
          onNextModule={() => nextModule && navigate(taskUrl(nextModule, 1))}
        />
      )}
    </div>
  );
}

// ---- route component -------------------------------------------------------

function TaskPage() {
  const { moduleParam, step } = useParams();
  const resolved = parseModulePath(moduleParam);
  if (!resolved) return <Navigate to="/learn" replace />;
  const found = getTask(resolved.module, step);
  if (!found) return <Navigate to="/learn" replace />;
  // The short id resolved but the slug is stale (the module was renamed):
  // land on the canonical url instead of 404ing. Replace, never push.
  if (!resolved.canonical) return <Navigate to={found.url} replace />;
  // key by taskKey so all workspace state (code, logs, tests…) resets per task
  return <TaskWorkspace key={found.taskKey} {...found} />;
}

export default TaskPage;
