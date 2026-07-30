// Shared callout for tasks that index 2D image data. One definition so the
// wording (and the gotcha it names) stays identical across modules; imported
// by content module files, which are loaded both by vite and by plain node
// (scripts/prerender.mjs), so keep this file dependency-free ESM.
//
// Source of truth for the convention: gpu.js README's dimensions table —
// output [width, height, depth] is indexed value[z][y][x].
export const ARRAY_LAYOUT = `<div class="layout-note">
  <b>Array layout in gpu.js</b>
  <p>Image data comes in row-major: <code>image[y][x]</code> is the pixel in row <em>y</em>,
    column <em>x</em>, and each pixel is an <code>[r, g, b, a]</code> array with channels from
    0 to 1. Mind the inversion that catches everyone — sizes are given width-first
    (<code>output: [width, height]</code>), but indexing runs row-first, so this thread's own
    pixel is <code>image[this.thread.y][this.thread.x]</code>. Swap those two and you read the
    transpose of your image. Three-dimensional data follows the same rule:
    <code>output: [w, h, d]</code> is indexed <code>[z][y][x]</code>.</p>
</div>`;
