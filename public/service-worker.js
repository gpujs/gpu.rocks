// Tombstone for the create-react-app service worker this site used to ship.
//
// Its navigation fallback served the precached app shell for every
// extension-less path, so real files such as /api/ (the gpu.js API reference)
// and deep links like /benchmark rendered the homepage for anyone who had
// visited before — see gpujs/gpu.js#852. The site is static and hash-routed, so
// it does not need a service worker at all.
//
// Browsers that still have the old worker installed fetch this file on their
// next update check; it unregisters itself, drops the stale precache, and
// reloads open tabs so they pick up the real pages. Keep it here — deleting it
// would leave those browsers on the old worker.

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    await self.registration.unregister();
    const cacheNames = await caches.keys();
    await Promise.all(cacheNames.map(cacheName => caches.delete(cacheName)));
    const clients = await self.clients.matchAll({ type: 'window' });
    clients.forEach(client => client.navigate(client.url));
  })());
});
