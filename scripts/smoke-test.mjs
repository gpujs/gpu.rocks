/**
 * Loads every route in a real browser and checks the page actually rendered.
 *
 * A production build can succeed and still ship a blank site — migrating off
 * create-react-app hit exactly that, where a missing `global` shim left every
 * route throwing before React mounted. `vite build` cannot catch that; this can.
 *
 * Usage: node scripts/smoke-test.mjs [baseUrl]      (default http://localhost:4173)
 *        CHROME_PATH=/path/to/chrome node scripts/smoke-test.mjs https://gpu.rocks
 */
import { launch, ROUTES } from './browser.mjs';

const base = (process.argv[2] || 'http://localhost:4173').replace(/\/$/, '');
const MIN_NODES = 50;

const browser = await launch();
let failures = 0;

for (const route of ROUTES) {
  const page = await browser.newPage();
  // uncaught exceptions and dead requests are always failures; console.error is
  // reported but tolerated, since software rendering on CI is chatty about WebGL
  const fatal = [];
  const noise = [];
  page.on('pageerror', e => fatal.push(String(e).split('\n')[0]));
  page.on('requestfailed', r => fatal.push(`request failed: ${r.url().slice(0, 100)}`));
  page.on('console', m => { if (m.type() === 'error') noise.push(m.text().slice(0, 140)); });

  await page.goto(base + route, { waitUntil: 'networkidle2', timeout: 60000 });
  await new Promise(resolve => setTimeout(resolve, 1500));

  // Walk the page before judging its images. Anything with loading="lazy"
  // below the fold has not STARTED loading, and a check that treats "not
  // loaded" as "broken" fails an image that is perfectly fine — while also
  // never verifying it, since it was never fetched. Scrolling both triggers
  // the fetch and makes the assertion mean something.
  await page.evaluate(async () => {
    const step = window.innerHeight * 0.8;
    for (let y = 0; y < document.body.scrollHeight; y += step) {
      window.scrollTo(0, y);
      await new Promise(r => setTimeout(r, 120));
    }
    window.scrollTo(0, 0);
    await Promise.all(
      Array.from(document.images)
        .filter(i => !i.complete)
        .map(i => new Promise(res => {
          const done = () => res();
          i.addEventListener('load', done, { once: true });
          i.addEventListener('error', done, { once: true });
          setTimeout(done, 8000);
        }))
    );
  });

  const rendered = await page.evaluate(() => ({
    nodes: document.querySelectorAll('#root *').length,
    // `complete && naturalWidth === 0` is what a FAILED load looks like. A
    // still-in-flight image is `!complete`, which is not the same thing and
    // used to be counted as a failure too.
    brokenImages: Array.from(document.images)
      .filter(i => i.complete && i.naturalWidth === 0)
      .map(i => i.currentSrc || i.src),
    stillLoading: Array.from(document.images).filter(i => !i.complete).length,
    excerpt: document.body.innerText.replace(/\s+/g, ' ').trim().slice(0, 90),
  }));
  await page.close();

  const problems = [
    ...fatal,
    rendered.nodes < MIN_NODES ? `only ${rendered.nodes} nodes rendered` : null,
    rendered.brokenImages.length
      ? `${rendered.brokenImages.length} broken images: ${rendered.brokenImages.map(u => u.split('/').pop()).join(', ')}`
      : null,
    rendered.stillLoading ? `${rendered.stillLoading} images never finished loading` : null,
  ].filter(Boolean);

  if (problems.length) failures++;
  console.log(`${problems.length ? 'FAIL' : 'PASS'} ${route.padEnd(14)} nodes=${rendered.nodes}`);
  console.log(`       "${rendered.excerpt}"`);
  problems.forEach(p => console.log(`       ✗ ${p}`));
  noise.slice(0, 3).forEach(n => console.log(`       (console) ${n}`));
}

// the API reference is a static file living under public/api
const page = await browser.newPage();
const response = await page.goto(`${base}/api/`, { waitUntil: 'domcontentloaded', timeout: 30000 });
const title = await page.title();
const apiOk = response.ok() && /^gpu\.js \d+\.\d+\.\d+/.test(title);
if (!apiOk) failures++;
console.log(`${apiOk ? 'PASS' : 'FAIL'} ${'/api/'.padEnd(14)} "${title}"`);
if (!apiOk) console.log('       ✗ the API reference is not being served — is the SPA swallowing it?');

await browser.close();
console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
