import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';

import { getThemePref, setThemePref } from './engine/storage';

// Theme preference for the learn app: 'auto' | 'light' | 'dark'.
// 'auto' resolves to the OS preference via matchMedia and tracks changes live.
// Persistence goes through engine/storage.js — the only module allowed to
// touch localStorage (key: gpujs-learn:theme).
const ORDER = ['auto', 'light', 'dark'];

function systemTheme() {
  try {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  } catch (e) {
    return 'light';
  }
}

const ThemeContext = createContext({
  pref: 'auto',
  theme: 'light',
  setPref: () => {},
  cycleTheme: () => {},
});

export function ThemeProvider({ children }) {
  const [pref, setPrefState] = useState(getThemePref);
  const [system, setSystem] = useState(systemTheme);

  useEffect(() => {
    let mq;
    try {
      mq = window.matchMedia('(prefers-color-scheme: dark)');
    } catch (e) {
      return undefined;
    }
    const onChange = event => setSystem(event.matches ? 'dark' : 'light');
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const setPref = useCallback(next => {
    const value = ORDER.includes(next) ? next : 'auto';
    setPrefState(value);
    setThemePref(value);
  }, []);

  const cycleTheme = useCallback(() => {
    setPrefState(current => {
      const next = ORDER[(ORDER.indexOf(current) + 1) % ORDER.length];
      setThemePref(next);
      return next;
    });
  }, []);

  const theme = pref === 'auto' ? system : pref;

  // Mirror the resolved theme onto <html>. The tokens are scoped to
  // .bench-root / .learn-root, but the element whose background paints the
  // CANVAS — the area a rubber-band overscroll reveals, and what the browser
  // samples for its own chrome — is <html>, which those scopes never reach. In
  // light mode the canvas stayed at the legacy #020024, so overscrolling on a
  // phone showed a dark band under a light page and made a working auto mode
  // look broken. Removed on unmount so the unthemed pages keep their own.
  useEffect(() => {
    const el = document.documentElement;
    const previous = el.getAttribute('data-theme');
    el.setAttribute('data-theme', theme);
    return () => {
      if (previous === null) el.removeAttribute('data-theme');
      else el.setAttribute('data-theme', previous);
    };
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ pref, theme, setPref, cycleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}

export default ThemeContext;
