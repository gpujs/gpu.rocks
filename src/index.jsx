import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

// Upgrade plain-http visits (e.g. arriving via a CDN/origin downgrade
// redirect). Never assign location.protocol = 'https://' — the slashes make
// WebKit throw at module top level, which blanked the whole site for any
// webview that landed on http (Chrome tolerates it, so it hid for years).
let upgradingToHttps = false;
if (window.location.protocol === 'http:' && window.location.hostname === 'gpu.rocks') {
  try {
    upgradingToHttps = true;
    window.location.replace(window.location.href.replace(/^http:/, 'https:'));
  } catch (e) {
    upgradingToHttps = false; // render over http rather than a blank page
  }
}
if (!upgradingToHttps) {
  createRoot(document.getElementById('root')).render(<App />);

  // The site used to ship a create-react-app service worker whose navigation
  // fallback served the app shell for every extension-less path, so real files
  // like /api/ (the gpu.js API reference) rendered the homepage instead
  // (gpujs/gpu.js#852). This site is static and hash-routed, so a service worker
  // buys it nothing; /service-worker.js is now a no-op that tears itself down,
  // and this unregisters any worker still installed from an earlier visit.
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.getRegistrations()
      .then(registrations => registrations.forEach(registration => registration.unregister()))
      .catch(() => {});
  }
}
