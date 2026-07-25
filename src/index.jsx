import React from 'react'
import ReactDOM from 'react-dom'
import App from './App'

if (window.location.protocol === 'http:' && window.location.hostname === 'gpu.rocks') window.location.protocol = 'https://';
else {
  ReactDOM.render(<App />, document.getElementById('root'));

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
