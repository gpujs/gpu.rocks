/**
 * Renders every route on two deployments and diffs what came out.
 *
 * Useful when a change should not alter the site — a build tool swap, a
 * dependency bump — where "it still builds" says nothing about the result.
 * Asset filenames are normalised, since content hashes legitimately differ.
 *
 * Usage: node scripts/compare.mjs <baseUrlA> <baseUrlB>
 *   e.g. node scripts/compare.mjs https://gpu.rocks http://localhost:4173
 */
import { launch, ROUTES } from './browser.mjs';

const [a, b] = process.argv.slice(2).map(url => url && url.replace(/\/$/, ''));
if (!a || !b) {
  console.error('Usage: node scripts/compare.mjs <baseUrlA> <baseUrlB>');
  process.exit(2);
}

const browser = await launch();

// strip the content hash bundlers add, so image-1a2b3c4d.png === image.abcd1234.png
const normaliseAsset = src => src.split('/').pop()
  .replace(/[-.][A-Za-z0-9_-]{8,}\.(png|jpe?g|gif|svg|webp)$/i, '.$1');

async function snapshot(base, route) {
  const page = await browser.newPage();
  await page.goto(base + route, { waitUntil: 'networkidle2', timeout: 60000 });
  await new Promise(resolve => setTimeout(resolve, 2500));
  const data = await page.evaluate(() => ({
    text: document.body.innerText.replace(/\s+/g, ' ').trim(),
    links: Array.from(document.querySelectorAll('a')).map(el => el.getAttribute('href')).filter(Boolean).sort(),
    images: Array.from(document.images).map(img => img.src),
  }));
  await page.close();
  return {
    ...data,
    // data: URIs mean the bundler inlined a small asset; count them rather than
    // comparing their contents
    images: data.images.map(src => src.startsWith('data:') ? '<inlined>' : normaliseAsset(src)).sort(),
  };
}

let differences = 0;
for (const route of ROUTES) {
  const [x, y] = [await snapshot(a, route), await snapshot(b, route)];
  const compare = (name, left, right) => {
    const same = JSON.stringify(left) === JSON.stringify(right);
    if (!same) differences++;
    return { name, same, left, right };
  };
  const results = [
    compare('text', x.text, y.text),
    compare('links', x.links, y.links),
    compare('images', x.images, y.images),
  ];

  console.log(route);
  for (const r of results) {
    const size = Array.isArray(r.left) ? `${r.left.length} vs ${r.right.length}` : `${r.left.length} vs ${r.right.length} chars`;
    console.log(`  ${r.name}: ${r.same ? 'identical' : 'DIFFERS'} (${size})`);
    if (r.same) continue;
    if (Array.isArray(r.left)) {
      const onlyA = r.left.filter(v => !r.right.includes(v));
      const onlyB = r.right.filter(v => !r.left.includes(v));
      if (onlyA.length) console.log(`    only in A: ${onlyA.slice(0, 8).join(', ')}`);
      if (onlyB.length) console.log(`    only in B: ${onlyB.slice(0, 8).join(', ')}`);
    } else {
      const words = (p, q) => p.split(' ').filter(w => !q.includes(w)).slice(0, 20).join(' ');
      console.log(`    only in A: "${words(r.left, r.right).slice(0, 150)}"`);
      console.log(`    only in B: "${words(r.right, r.left).slice(0, 150)}"`);
    }
  }
}

await browser.close();
console.log(differences ? `\n${differences} difference(s)` : '\nA and B render identically');
process.exit(differences ? 1 : 0);
