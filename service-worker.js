/**
 * Welcome to your Workbox-powered service worker!
 *
 * You'll need to register this file in your web app and you should
 * disable HTTP caching for this file too.
 * See https://goo.gl/nhQhGp
 *
 * The rest of the code is auto-generated. Please don't update this file
 * directly; instead, make changes to your Workbox build configuration
 * and re-run your build process.
 * See https://goo.gl/2aRDsh
 */

importScripts("https://storage.googleapis.com/workbox-cdn/releases/4.3.1/workbox-sw.js");

importScripts(
  "/precache-manifest.a748150979fd58f0c772c08e85930ed5.js"
);

self.skipWaiting();

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

workbox.core.clientsClaim();

/**
 * The workboxSW.precacheAndRoute() method efficiently caches and responds to
 * requests for URLs in the manifest.
 * See https://goo.gl/S9QRab
 */
self.__precacheManifest = [].concat(self.__precacheManifest || []);
workbox.precaching.precacheAndRoute(self.__precacheManifest, {});

// The site is hash-routed (HashRouter), so the app shell is only ever needed
// at the root. Without this whitelist the fallback swallows every extension-less
// path — /api/ (the API reference) and /benchmark rendered the homepage instead
// of the real page for anyone with the service worker installed.
// See gpujs/gpu.js#852. Re-applied automatically by scripts/patch-sw.js.
workbox.routing.registerNavigationRoute(workbox.precaching.getCacheKeyForURL("/index.html"), {
  whitelist: [/^\/$/],
  blacklist: [/^\/_/,/\/[^/]+\.[^/]+$/],
});
