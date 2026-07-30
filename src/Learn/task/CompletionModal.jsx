import React, { useCallback, useEffect, useRef } from 'react';
import TaskDots from '../components/TaskDots';
import Confetti from './Confetti';
import { FEEDBACK_URL } from '../feedback';

// Completion celebration dialog — opened by TaskWorkspace when Next is clicked
// on the last task of a fully-done module. kind 'module' congratulates the
// module and offers the next one; kind 'track' closes out the whole track
// (and, for the final track, the course) with a single exit.
//
// A11y: role=dialog + aria-modal, labelled by the headline; focus moves to the
// primary action on open; Tab cycles inside (simple trap over the dialog's
// focusables); Esc and backdrop click both call onClose, and the CALLER
// restores focus to the Next button.
function CompletionModal({
  kind,
  module,
  track,
  totalTasks,
  courseEnd,
  onClose,
  onEnd,
  onNextModule,
}) {
  const panelRef = useRef(null);
  const primaryRef = useRef(null);

  useEffect(() => {
    if (primaryRef.current) primaryRef.current.focus();
  }, []);

  const handleKeyDown = useCallback(
    e => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
        return;
      }
      if (e.key !== 'Tab' || !panelRef.current) return;
      const focusables = panelRef.current.querySelectorAll(
        'a[href], button:not([disabled])'
      );
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    },
    [onClose]
  );

  // mousedown (not click) so a drag that starts on the panel and ends on the
  // backdrop doesn't dismiss the dialog
  const handleBackdrop = useCallback(
    e => {
      if (e.target !== e.currentTarget) return;
      // preventDefault: the mousedown's default focus change would otherwise
      // land on <body>, clobbering the caller's focus restore to Next
      e.preventDefault();
      onClose();
    },
    [onClose]
  );

  const isTrack = kind === 'track';
  const headline = isTrack
    ? `Track ${track.number} — ${track.title}`
    : `Module ${String(module.id).replace('-', '.')} — ${module.title}`;
  const congrats = isTrack
    ? `Every module in Track ${track.number} is complete — the full track, ` +
      `start to finish.${courseEnd ? ' And that was the final track: you’ve finished the whole course.' : ''}`
    : `All ${totalTasks} tasks passed. That’s the whole module — nicely done.`;

  return (
    <div className="cmodal-overlay" onMouseDown={handleBackdrop} onKeyDown={handleKeyDown}>
      {/* fires from the panel's top corners; aria-hidden and click-through, so
          it stays out of the focus trap and off the buttons underneath */}
      <Confetti
        milestone={courseEnd ? 'course' : kind}
        anchorRef={panelRef}
      />
      <div
        className="cmodal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="cmodal-title"
        ref={panelRef}
      >
        <p className="eyebrow">{isTrack ? 'Track complete' : 'Module complete'}</p>
        <h2 id="cmodal-title">{headline}</h2>
        <TaskDots total={totalTasks} doneCount={totalTasks} currentIndex={-1} />
        <p className="cmodal-line">{congrats}</p>
        <p className="cmodal-feedback">
          Something confusing along the way?{' '}
          <a href={FEEDBACK_URL} target="_blank" rel="noopener noreferrer">
            Tell us
          </a>{' '}
          — it makes the course better.
        </p>
        <div className="cmodal-actions">
          {isTrack ? (
            <button type="button" className="tb-next" ref={primaryRef} onClick={onEnd}>
              Exit track
            </button>
          ) : (
            <>
              <button type="button" className="tb-btn" onClick={onEnd}>
                Exit module
              </button>
              <button type="button" className="tb-next" ref={primaryRef} onClick={onNextModule}>
                Next module →
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default CompletionModal;
