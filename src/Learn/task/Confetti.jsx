import React, { useEffect, useRef } from 'react';
import confetti from 'canvas-confetti';

// One celebratory burst, fired when a completion dialog opens.
//
// It renders its OWN canvas rather than calling the library's global helper:
// that one appends a canvas at z-index 100, which would put the confetti
// behind the dialog (z-index 200) where the panel would hide most of it. Ours
// sits above both, so pieces fall past the dialog the way they would in the
// room — hence also pointer-events: none, since Exit module and Next module
// are directly underneath and an overlay eating those clicks would be a nasty
// bug.
//
// The burst comes from the dialog's own top corners, not the screen edges, so
// the celebration reads as coming from the thing being celebrated.

// how much of a party each milestone earns
const INTENSITY = {
  module: { count: 45, waves: 1 },
  track: { count: 80, waves: 2 },
  course: { count: 120, waves: 3 },
};

// brand colors, read live so the burst matches the active theme
function paletteFrom(el) {
  const fallback = ['#20a4f3', '#ff79c6', '#18bc9c', '#e2b04a'];
  if (!el || typeof getComputedStyle !== 'function') return fallback;
  const style = getComputedStyle(el);
  const colors = ['--blue', '--pink', '--teal', '--amber']
    .map(name => style.getPropertyValue(name).trim())
    .filter(Boolean);
  return colors.length ? colors : fallback;
}

// viewport-normalized origin for a corner of the dialog, nudged inward so the
// pieces appear to leave the panel rather than a point floating beside it
function cornerOrigin(rect, side) {
  const x = side === 'left' ? rect.left + rect.width * 0.1 : rect.right - rect.width * 0.1;
  return {
    x: Math.min(Math.max(x / window.innerWidth, 0), 1),
    y: Math.min(Math.max((rect.top + rect.height * 0.15) / window.innerHeight, 0), 1),
  };
}

function Confetti({ milestone = 'module', anchorRef }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    // Motion this decorative is exactly what the preference is for.
    const reduced =
      typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced || !canvasRef.current) return undefined;

    const fire = confetti.create(canvasRef.current, { resize: true, useWorker: true });
    const { count, waves } = INTENSITY[milestone] || INTENSITY.module;
    const rect = anchorRef && anchorRef.current
      ? anchorRef.current.getBoundingClientRect()
      : { left: window.innerWidth * 0.35, right: window.innerWidth * 0.65, top: window.innerHeight * 0.35, width: window.innerWidth * 0.3, height: window.innerHeight * 0.3 };
    const colors = paletteFrom(canvasRef.current.closest('.learn-root'));

    const timers = [];
    const burst = wave => {
      const scale = 1 - wave * 0.25; // later waves are smaller echoes
      ['left', 'right'].forEach(side => {
        fire({
          particleCount: Math.round((count / 2) * scale),
          angle: side === 'left' ? 60 : 120,
          spread: 70,
          startVelocity: 45,
          decay: 0.9,
          scalar: 0.9,
          origin: cornerOrigin(rect, side),
          colors,
          disableForReducedMotion: true,
        });
      });
    };

    burst(0);
    for (let wave = 1; wave < waves; wave++) {
      timers.push(setTimeout(() => burst(wave), wave * 320));
    }

    return () => {
      timers.forEach(clearTimeout);
      fire.reset();
    };
  }, [milestone, anchorRef]);

  return <canvas className="cmodal-confetti" ref={canvasRef} aria-hidden="true" />;
}

export default Confetti;
