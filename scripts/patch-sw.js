/**
 * Restricts the generated service worker's navigation fallback to the root path.
 *
 * create-react-app hardcodes its workbox options, so this cannot be configured
 * without ejecting. The default fallback serves the precached app shell for
 * every extension-less path, which for this site meant /api/ (the gpu.js API
 * reference, a real static file) and deep links like /benchmark rendered the
 * homepage for anyone with the service worker installed — see gpujs/gpu.js#852.
 *
 * The site uses HashRouter, so the shell is only ever needed at "/".
 *
 * Runs automatically after `yarn build`. Fails the build if the generated
 * service worker no longer matches what it expects, so an upgrade of
 * react-scripts cannot silently reintroduce the bug.
 */
const fs = require('fs');
const path = require('path');

const swPath = path.join(__dirname, '..', 'build', 'service-worker.js');

if (!fs.existsSync(swPath)) {
  console.error(`patch-sw: ${swPath} not found — did the build run?`);
  process.exit(1);
}

let source = fs.readFileSync(swPath, 'utf8');

if (source.includes('whitelist: [/^\\/$/]')) {
  console.log('patch-sw: already patched');
  process.exit(0);
}

const navigationRoute = /(workbox\.routing\.registerNavigationRoute\([^;]*?\{\s*)(blacklist:)/;
if (!navigationRoute.test(source)) {
  console.error('patch-sw: could not find the navigation route to patch. The generated service worker changed shape — re-check that /api/ and deep links are not served the app shell, then update this script.');
  process.exit(1);
}
source = source.replace(navigationRoute, '$1whitelist: [/^\\/$/],\n  $2');

// take over as soon as the updated worker is fetched, so returning visitors get
// fixes on their next load instead of after closing every tab
const messageListener = "self.addEventListener('message', (event) => {";
if (source.includes(messageListener) && !source.includes('self.skipWaiting();\n')) {
  source = source.replace(messageListener, `self.skipWaiting();\n\n${messageListener}`);
}

fs.writeFileSync(swPath, source);
console.log('patch-sw: navigation fallback restricted to "/"');
