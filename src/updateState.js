// Shared "this tab is running a build that no longer exists" flag.
//
// Two things can discover it and they are far apart in the tree: UpdateBanner
// polling /version.json, and the error boundary around the lazily-loaded learn
// app catching a chunk that 404s. Both funnel through here so the banner is
// the single place that says it.

let stale = false;
const listeners = new Set();

export function isStale() {
  return stale;
}

export function markStale() {
  if (stale) return;
  stale = true;
  listeners.forEach(fn => {
    try {
      fn(true);
    } catch (e) {
      // a broken listener must not stop the others hearing about it
    }
  });
}

// Dismissal clears the flag so the banner can reappear if ANOTHER deploy lands;
// the banner separately remembers which build the user waved away.
export function clearStale() {
  stale = false;
  listeners.forEach(fn => {
    try {
      fn(false);
    } catch (e) {
      /* as above */
    }
  });
}

export function onStaleChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

// A failed dynamic import is the loudest possible evidence that this tab is
// stale: the chunk was there when the page loaded and is not there now.
const CHUNK_FAILURE = /dynamically imported module|Importing a module script failed|error loading dynamically imported module/i;

export function looksLikeChunkFailure(error) {
  const message = error && (error.message || error.reason || error);
  return typeof message === 'string' ? CHUNK_FAILURE.test(message) : CHUNK_FAILURE.test(String(message));
}
