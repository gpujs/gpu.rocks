import React, { useRef } from 'react';
import { Link } from 'react-router';
import LearnNav from '../components/LearnNav';
import KernelGrid from '../components/KernelGrid';
import TaskDots from '../components/TaskDots';
import { tracks } from '../content/index.js';
import { moduleProgress } from '../engine/storage.js';

// /learn landing page: hero + highlights + track/module catalogue.
// Layout, spacing and copy come from the approved mockup; module/track data
// comes from content/index.js and per-module progress from engine/storage.js.

// Defensive wrapper around engine/storage.js moduleProgress() — the engine
// module is written by another agent; fall back to an untouched-module shape
// if it is absent or throws (contract: { done, total, currentIndex, state }).
function progressOf(module) {
  const total = Array.isArray(module.tasks) ? module.tasks.length : 0;
  const fallback = { done: 0, total, currentIndex: 0, state: 'todo' };
  try {
    if (typeof moduleProgress === 'function') {
      const p = moduleProgress(module);
      if (p && typeof p === 'object') return { ...fallback, ...p };
    }
  } catch (e) {
    // private browsing / storage unavailable — treat as fresh
  }
  return fallback;
}

function stateLabel(progress) {
  if (progress.state === 'done') return 'Completed';
  if (progress.state === 'now') return `Continue · ${progress.done}/${progress.total}`;
  return 'Start';
}

function ModuleCard({ module }) {
  const progress = progressOf(module);
  const taskCount = Array.isArray(module.tasks) ? module.tasks.length : 0;
  const isCurrent = progress.state === 'now';
  return (
    <Link
      to={`/learn/${module.id}`}
      className={isCurrent ? 'module current' : 'module'}
    >
      <span className="mno">
        MODULE {String(module.id).replace('-', '.')} · {taskCount} TASK{taskCount === 1 ? '' : 'S'}
      </span>
      <h3>{module.title}</h3>
      {/* blurbs are trusted course copy authored in-repo; some contain <code> */}
      <p dangerouslySetInnerHTML={{ __html: module.blurb || '' }} />
      <div className="foot">
        <TaskDots
          total={progress.total}
          doneCount={progress.done}
          currentIndex={isCurrent ? progress.currentIndex : -1}
        />
        <span className={`mstate ${progress.state}`}>{stateLabel(progress)}</span>
      </div>
    </Link>
  );
}

function Track({ track }) {
  const number = track.number != null ? track.number : track.id;
  const modules = Array.isArray(track.modules) ? track.modules : [];
  return (
    <div className="track">
      <div className="track-head">
        <span className="tno">TRACK {number}</span>
        <h2>{track.title}</h2>
        <span className="tagline">{track.tagline}</span>
      </div>
      <div className="modules">
        {modules.map(module => <ModuleCard key={module.id} module={module} />)}
      </div>
    </div>
  );
}

function LearnHome() {
  const tracksRef = useRef(null);
  const trackList = Array.isArray(tracks) ? tracks : [];

  const scrollToTracks = () => {
    const el = tracksRef.current;
    if (!el) return;
    let reduced = false;
    try {
      reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (e) {
      reduced = false;
    }
    el.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
  };

  return (
    <div className="screen">
      <LearnNav />

      <section className="hero">
        <div>
          <p className="eyebrow">gpu.js · interactive course</p>
          <h1>
            Stop writing loops.<br />
            <span className="accent">Launch threads.</span>
          </h1>
          <p className="sub">
            A hands-on course in <strong>GPGPU</strong> — general-purpose computing on graphics
            hardware — built on <strong>gpu.js</strong>. Write real kernels, run them on{' '}
            <strong>your own GPU</strong>, and learn the mental model behind every parallel
            computing platform.
          </p>
          <div className="cta-row">
            <Link to="/learn/1-1" className="btn btn-primary">Start GPGPU 101</Link>
            <button type="button" className="btn btn-ghost" onClick={scrollToTracks}>
              Browse all modules ↓
            </button>
          </div>
        </div>
        <KernelGrid />
      </section>

      <section className="highlights">
        <div className="hl">
          <h3>Real GPGPU concepts</h3>
          <p>
            Kernels, threads, memory layout, reductions, pipelines — learned by writing and
            running them with gpu.js, not by reading slides.
          </p>
        </div>
        <div className="hl">
          <h3>Zero installation</h3>
          <p>
            Everything runs in your browser, on your actual graphics card. No toolchain,
            no drivers, no signup. Open a module and start.
          </p>
        </div>
        <div className="hl">
          <h3>Skills that transfer</h3>
          <p>The concepts here carry straight over to every serious GPU platform.</p>
          <div className="chips">
            <span className="chip">WebGPU</span>
            <span className="chip">CUDA</span>
            <span className="chip">ROCm</span>
            <span className="chip">Metal</span>
          </div>
        </div>
      </section>

      <section className="tracks" ref={tracksRef}>
        {trackList.map(track => (
          <Track key={track.number != null ? track.number : track.id} track={track} />
        ))}
      </section>

      <footer className="pagefoot">
        GPU.js is MIT licensed · learn.gpu.rocks runs entirely in your browser — nothing to
        install, nothing uploaded.
      </footer>
    </div>
  );
}

export default LearnHome;
