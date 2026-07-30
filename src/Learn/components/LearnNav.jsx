import React from 'react';
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
  return (
    <nav className="nav">
      <Link className="brand" to="/learn">
        <JellyLogo />
        GPU.js <span className="learn-tag">learn</span>
      </Link>
      <div className="links">
        <Link to="/">Home</Link>
        <Link to="/learn" className="active">Learn</Link>
        {/* the API reference is a real static directory, not an SPA route */}
        <a href="/api/">API</a>
        <Link to="/examples">Examples</Link>
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
      <button type="button" className="theme-btn" onClick={cycleTheme}>
        Theme: {capitalize(pref)}
      </button>
    </nav>
  );
}

export default LearnNav;
