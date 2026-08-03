import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router';
import { useTheme } from '../ThemeContext';
import { FEEDBACK_URL } from '../feedback';

function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

function JellyLogo() {
  return (
    <svg className="jelly" viewBox="0 0 24 24" aria-hidden="true">
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

function LearnNav() {
  const { pref, cycleTheme } = useTheme();
  // Phones collapse the links behind a menu button; above 720px the menu state
  // is inert because the links are always shown and the button is hidden.
  const [menuOpen, setMenuOpen] = useState(false);
  const navRef = useRef(null);

  useEffect(() => {
    if (!menuOpen) return undefined;
    const onKey = e => {
      if (e.key === 'Escape') setMenuOpen(false);
    };
    // a tap anywhere else dismisses it, which is what every native menu does
    const onDown = e => {
      if (navRef.current && !navRef.current.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onDown);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onDown);
    };
  }, [menuOpen]);

  return (
    <>
      {/* A real element, not a ::before on the nav: a pseudo-element's clicks
          target the nav itself, so the tap-outside-to-close check above would
          count the scrim as "inside" and the menu could only be closed by the
          button. This one dims the page AND dismisses. */}
      {menuOpen && (
        <div className="nav-scrim" aria-hidden="true" onClick={() => setMenuOpen(false)} />
      )}
      <nav className={menuOpen ? 'nav open' : 'nav'} ref={navRef}>
      <Link className="brand" to="/learn" aria-label="GPU.js Learn — course home">
        <JellyLogo />
        <span className="brand-word">GPU.js</span>{' '}
        <span className="learn-tag">learn</span>
      </Link>
      <div className="links" id="nav-menu" onClick={() => setMenuOpen(false)}>
        <Link to="/">Home</Link>
        <Link to="/learn" className="active">Learn</Link>
        {/* the API reference is a real static directory, not an SPA route */}
        <a href="/api/">API</a>
        <Link to="/examples">Examples</Link>
        {/* was missing entirely: from the course there was no way to reach the
            benchmark at all, and its absence was also what made this nav's
            order differ from the other two. See HeaderLinks.jsx for the shared
            order these three navs follow. */}
        <Link to="/benchmark">Benchmark</Link>
        <a href="https://github.com/gpujs/gpu.js">GitHub</a>
        <a
          className="feedback"
          href={FEEDBACK_URL}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span aria-hidden="true">💬 </span>Feedback
        </a>
      </div>
      <button
        type="button"
        className="nav-burger"
        aria-expanded={menuOpen}
        aria-controls="nav-menu"
        aria-label={menuOpen ? 'Close menu' : 'Open menu'}
        onClick={() => setMenuOpen(o => !o)}
      >
        <span aria-hidden="true">{menuOpen ? '✕' : '☰'}</span>
      </button>
      <button
        type="button"
        className="theme-btn"
        onClick={cycleTheme}
        aria-label={`Theme: ${capitalize(pref)} — tap to change`}
      >
        {/* phones show only the glyph; the word costs a whole nav row */}
        <span className="theme-glyph" aria-hidden="true">
          {pref === 'dark' ? '☾' : pref === 'light' ? '☀' : '◐'}
        </span>
        <span className="theme-word">Theme: {capitalize(pref)}</span>
      </button>
      </nav>
    </>
  );
}

export default LearnNav;
