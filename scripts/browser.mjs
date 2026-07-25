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

export function launch() {
  return puppeteer.launch({
    executablePath: findBrowser(),
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-dev-shm-usage',
      // CI runners have no GPU; let WebGL fall back to software rendering so
      // the benchmark page still renders
      '--enable-unsafe-swiftshader',
    ],
  });
}

// the hash routes the site serves, plus the static API reference that must not
// be swallowed by the SPA (gpujs/gpu.js#852)
export const ROUTES = ['/#/', '/#/benchmark', '/#/install', '/#/examples'];
