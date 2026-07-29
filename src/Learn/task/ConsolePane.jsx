import React, { useEffect, useRef } from 'react';

// Console output pane — renders the engine's log stream in the mockup's
// style: timestamps, italic system lines, teal ok lines, red errors, and
// inline canvas snapshots with a "canvas · W×H" badge.

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

function ConsoleLine({ log }) {
  return (
    <>
      <div className="ln">
        <span className="t">{log.time}</span>
        <span className={lineClass(log.type)}>{log.text}</span>
      </div>
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

function ConsolePane({ logs, active }) {
  const paneRef = useRef(null);

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
      {logs.length === 0 ? (
        <div className="ln">
          <span className="sys">Console is empty — press ▶ Run (or ⌘/Ctrl + Enter).</span>
        </div>
      ) : (
        logs.map((log, i) => <ConsoleLine key={i} log={log} />)
      )}
    </div>
  );
}

export default ConsolePane;
