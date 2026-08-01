import React, { useCallback, useEffect, useRef } from 'react';

// A yes/no dialog for an action that throws work away.
//
// Replaces window.confirm, which looked adequate on desktop and is not: on a
// phone the native sheet lands under the thumb that just tapped, so the reflex
// second tap confirms it. Reset code sits in a row of 40px icon buttons next to
// Prev/Next, which makes a mis-tap plausible in the first place.
//
// Two deliberate choices, both about making the safe path the easy one:
//   - focus opens on CANCEL, not the destructive button, so Enter or a stray
//     tap dismisses rather than destroys;
//   - the destructive button says what it does ("Reset code"), never "OK".
//
// A11y matches CompletionModal: role=dialog + aria-modal, labelled by the
// headline, Tab cycles inside the panel, Esc and backdrop both cancel, and the
// CALLER restores focus to the control that opened it.
function ConfirmModal({ eyebrow, title, body, confirmLabel, onConfirm, onCancel }) {
  const panelRef = useRef(null);
  const cancelRef = useRef(null);

  useEffect(() => {
    if (cancelRef.current) cancelRef.current.focus();
  }, []);

  const handleKeyDown = useCallback(
    e => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onCancel();
        return;
      }
      if (e.key !== 'Tab' || !panelRef.current) return;
      const focusables = panelRef.current.querySelectorAll('button:not([disabled])');
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
    [onCancel]
  );

  // mousedown (not click) so a drag that starts on the panel and ends on the
  // backdrop doesn't dismiss the dialog
  const handleBackdrop = useCallback(
    e => {
      if (e.target !== e.currentTarget) return;
      e.preventDefault();
      onCancel();
    },
    [onCancel]
  );

  return (
    <div className="cmodal-overlay" onMouseDown={handleBackdrop} onKeyDown={handleKeyDown}>
      <div
        className="cmodal cmodal-confirm"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        ref={panelRef}
      >
        <p className="eyebrow">{eyebrow}</p>
        <h2 id="confirm-title">{title}</h2>
        <p className="cmodal-line">{body}</p>
        <div className="cmodal-actions">
          <button type="button" className="tb-btn" ref={cancelRef} onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="tb-danger" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ConfirmModal;
