import React from 'react';

// Toolbar icons, all on one 24-unit grid at one stroke weight and one rendered
// size.
//
// These replace text glyphs (⟲ ⏱ ⇤ ← →). A glyph's drawn size is whatever
// fraction of the em box its designer chose, and that fraction differs per
// character AND per font — so in a row of identical 40px buttons the stopwatch
// came out tiny next to the arrows however the font-size was set. There is no
// font-size that fixes that, because the buttons were never the problem.
//
// Drawn on currentColor so they inherit hover and :disabled from the button,
// and drawn rather than typed so a phone missing a glyph cannot fall back to a
// blank box. Phones only: above 720px the toolbar shows words instead.
const ICON = {
  viewBox: '0 0 24 24',
  width: 18,
  height: 18,
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
};

// counter-clockwise arrow — undo, back to how it started
export function IconReset() {
  return (
    <svg {...ICON}>
      <polyline points="2 5 2 11 8 11" />
      <path d="M4.6 15.5a9 9 0 1 0 2-9.3L2 11" />
    </svg>
  );
}

// stopwatch, not a clock: this times a run, it does not tell the time
export function IconBench() {
  return (
    <svg {...ICON}>
      <circle cx="12" cy="14" r="7.5" />
      <line x1="12" y1="14" x2="12" y2="10" />
      <line x1="9.5" y1="2.5" x2="14.5" y2="2.5" />
      <line x1="12" y1="2.5" x2="12" y2="6.5" />
    </svg>
  );
}

// a door with an arrow leaving through it — the action, not the destination
export function IconExit() {
  return (
    <svg {...ICON}>
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <polyline points="16 17 21 12 16 7" />
      <line x1="21" y1="12" x2="9" y2="12" />
    </svg>
  );
}

export function IconPrev() {
  return (
    <svg {...ICON}>
      <line x1="20" y1="12" x2="4" y2="12" />
      <polyline points="11 19 4 12 11 5" />
    </svg>
  );
}

export function IconNext() {
  return (
    <svg {...ICON}>
      <line x1="4" y1="12" x2="20" y2="12" />
      <polyline points="13 5 20 12 13 19" />
    </svg>
  );
}
