import React, { useCallback, useEffect, useRef, useState } from 'react';
import { clearStale, isStale, looksLikeChunkFailure, markStale, onStaleChange } from './updateState';

// Tells a long-open tab that the site has been redeployed under it.
//
// This is not about showing fresh content — it is about not breaking. Every
// deploy replaces the whole gh-pages tree, and the app is code-split, so a tab
// opened before a deploy is holding chunk filenames that no longer exist. It
// looks fine until you navigate to /learn, and then the lazy import 404s and
// the page dies. Catching that after the fact is too late, so we watch for the
// deploy instead.
//
// prerender writes /version.json with an id derived from the built asset
// names. We take whatever it says on first load as "the build this tab is
// running" — no build-time constant needed, and no way for the baked value and
// the served one to disagree — then re-check occasionally and whenever the tab
// comes back to the foreground, which is when a stale tab is about to be used
// again.
//
// It never reloads by itself. A learner may be mid-run with unsaved thinking
// on screen; taking the page out from under them to fix a problem they have
// not hit yet would be a worse bug than the one it prevents.

const VERSION_URL = '/version.json';
const POLL_MS = 10 * 60 * 1000;
const MIN_RECHECK_MS = 60 * 1000; // floor for focus-triggered checks

async function fetchBuildId() {
  // Cache-busted: GitHub Pages serves max-age=600 and Cloudflare caches .json
  // by extension, so a plain fetch could keep answering with the old build for
  // as long as the staleness it is meant to detect.
  const res = await fetch(`${VERSION_URL}?t=${Date.now()}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`version.json ${res.status}`);
  const data = await res.json();
  if (!data || typeof data.build !== 'string') throw new Error('version.json has no build id');
  return data.build;
}

function UpdateBanner() {
  const [stale, setStale] = useState(isStale());
  const currentBuild = useRef(null);
  const latestBuild = useRef(null);   // newest id seen, dismissed or not
  const dismissedBuild = useRef(null);
  const lastCheck = useRef(0);

  const check = useCallback(async () => {
    if (stale) return; // already asking; nothing more to learn
    const now = Date.now();
    if (now - lastCheck.current < MIN_RECHECK_MS) return;
    lastCheck.current = now;
    try {
      const build = await fetchBuildId();
      latestBuild.current = build;
      if (currentBuild.current === null) currentBuild.current = build;
      // Dismissing means "I know, not now" — it must not come back every poll.
      // Another deploy is a new fact, so that one speaks up again.
      else if (build !== currentBuild.current && build !== dismissedBuild.current) markStale();
    } catch (e) {
      // offline, or the file is briefly missing mid-deploy — try again later
    }
  }, [stale]);

  // the error boundary can discover staleness before any poll does
  useEffect(() => onStaleChange(setStale), []);

  useEffect(() => {
    check();
    const timer = setInterval(check, POLL_MS);
    const onVisible = () => {
      if (document.visibilityState === 'visible') check();
    };
    document.addEventListener('visibilitychange', onVisible);
    window.addEventListener('focus', check);

    // A chunk that 404s is proof this tab is stale — no polling required, and
    // by then the user is looking at something broken. Two shapes: a <script>
    // or <link> element failing (non-bubbling, hence capture phase), and a
    // dynamic import() rejecting, which is how a lazy route actually fails.
    const onResourceError = event => {
      const el = event.target;
      if (el && (el.tagName === 'SCRIPT' || el.tagName === 'LINK')) markStale();
    };
    const onRejection = event => {
      if (looksLikeChunkFailure(event.reason)) markStale();
    };
    window.addEventListener('error', onResourceError, true);
    window.addEventListener('unhandledrejection', onRejection);

    return () => {
      clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisible);
      window.removeEventListener('focus', check);
      window.removeEventListener('error', onResourceError, true);
      window.removeEventListener('unhandledrejection', onRejection);
    };
  }, [check]);

  if (!stale) return null;

  return (
    <div className="update-banner" role="status" aria-live="polite">
      <span className="update-banner-text">
        A new version of gpu.rocks is out. Reload to pick it up — your saved
        progress and code are kept.
      </span>
      <button
        type="button"
        className="update-banner-reload"
        onClick={() => window.location.reload()}
      >
        Reload
      </button>
      <button
        type="button"
        className="update-banner-dismiss"
        onClick={() => {
          dismissedBuild.current = latestBuild.current;
          clearStale();
        }}
        aria-label="Dismiss"
      >
        ✕
      </button>
    </div>
  );
}

export default UpdateBanner;
