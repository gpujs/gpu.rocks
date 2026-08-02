import { existsSync } from 'fs';
import puppeteer from 'puppeteer-core';

// puppeteer-core does not download a browser, so find one that is already here.
// CHROME_PATH wins; otherwise try the usual locations on CI runners and macOS.
const CANDIDATES = [
  process.env.CHROME_PATH,
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium-browser',
  '/usr/bin/chromium',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
].filter(Boolean);

export function findBrowser() {
  const found = CANDIDATES.find(path => existsSync(path));
  if (!found) {
    throw new Error(
      `No Chrome or Chromium found. Looked in:\n  ${CANDIDATES.join('\n  ')}\n` +
      'Set CHROME_PATH to a browser executable.'
    );
  }
  return found;
}

// { real: true } is for measuring rather than looking. It does NOT mean headed:
// measured on this machine, headless Chrome reports the same apple metal-3
// adapter and the same Metal WebGL renderer as a headed window, so a window on
// someone's screen buys nothing and costs them their screen for half an hour.
// What it does mean is no permission to fall back to software, and none of
// Chrome's background throttling. Whether the GPU is genuinely hardware is then
// checked by looking at the renderer strings — see bench-record.mjs — rather
// than inferred from how the browser was launched.
//
// { headed: true } is for watching a run go wrong, and is orthogonal.
export function launch({ real = false, headed = false } = {}) {
  return puppeteer.launch({
    executablePath: findBrowser(),
    headless: !headed,
    args: [
      '--no-sandbox',
      '--disable-dev-shm-usage',
      ...(real
        ? [
          // a benchmark in a tab nobody is watching is still a benchmark
          '--disable-background-timer-throttling',
          '--disable-backgrounding-occluded-windows',
          '--disable-renderer-backgrounding',
        ]
        : [
          // CI runners have no GPU; let WebGL fall back to software rendering
          // so the benchmark page still renders
          '--enable-unsafe-swiftshader',
        ]),
    ],
  });
}

// the path routes the site serves; the static API reference (/api/, which must
// not be swallowed by the SPA — gpujs/gpu.js#852) is checked separately
export const ROUTES = ['/', '/benchmark', '/install', '/examples'];
