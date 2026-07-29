import React, { useEffect, useRef } from 'react';

// Hero CPU-vs-GPU animation: a 16×10 grid of cells lit one at a time on the
// "CPU pass", then all at once on the "GPU pass". Ported from the approved
// mockup. Honors prefers-reduced-motion by rendering the static GPU state.
const COLS = 16;
const ROWS = 10;
const N = COLS * ROWS;

function KernelGrid() {
  const gridRef = useRef(null);
  const modeRef = useRef(null);
  const statRef = useRef(null);

  useEffect(() => {
    const cells = Array.from(gridRef.current.children);
    const modeEl = modeRef.current;
    const statEl = statRef.current;

    const showGpuLit = () => {
      cells.forEach(cell => { cell.className = 'lit'; });
      modeEl.textContent = `GPU · ${N} threads`;
      modeEl.className = 'mode gpu';
      statEl.textContent = `all ${N} cells · 1 tick`;
    };

    let reduced = false;
    try {
      reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (e) {
      reduced = false;
    }
    if (reduced) {
      showGpuLit();
      return undefined;
    }

    let cancelled = false;
    let timer = null;
    const delay = ms => new Promise(resolve => {
      timer = setTimeout(resolve, ms);
    });
    const clearCells = () => cells.forEach(cell => { cell.className = ''; });

    (async () => {
      while (!cancelled) {
        // CPU pass: sequential, one cell per tick
        clearCells();
        modeEl.textContent = 'CPU · 1 thread';
        modeEl.className = 'mode cpu';
        for (let i = 0; i < N && !cancelled; i++) {
          cells[i].className = 'lit cpu';
          statEl.textContent = `cell ${i + 1} / ${N}`;
          await delay(26);
        }
        if (cancelled) break;
        await delay(900);
        if (cancelled) break;
        // GPU pass: all at once
        clearCells();
        modeEl.textContent = `GPU · ${N} threads`;
        modeEl.className = 'mode gpu';
        statEl.textContent = `all ${N} cells · 1 tick`;
        await delay(350);
        if (cancelled) break;
        cells.forEach(cell => { cell.className = 'lit'; });
        await delay(2200);
      }
    })();

    return () => {
      cancelled = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, []);

  const cells = [];
  for (let i = 0; i < N; i++) cells.push(<span key={i} />);

  return (
    <div className="gridcard" aria-label="CPU versus GPU animation">
      <div className="modehead">
        <span className="mode cpu" ref={modeRef}>CPU · 1 thread</span>
        <span className="stat" ref={statRef}>cell 0 / {N}</span>
      </div>
      <div className="kgrid" ref={gridRef} aria-hidden="true">{cells}</div>
      <p className="caption">
        The same job, two ways: one cell at a time — or every cell in a single tick.
      </p>
    </div>
  );
}

export default KernelGrid;
