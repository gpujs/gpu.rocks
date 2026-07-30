// Figures for module 3-2 — Escape-Time Fractals.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./3-2/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '1': [
    {
      slug: 'pixel-to-plane',
      caption: 'xMin, yMin, step: three numbers turn a pixel count into a place',
      placement: 'intro',
    },
  ],
  '2': [
    {
      slug: 'escape-orbit',
      caption: 'iterate z² + c and watch: settle in, or fly off — the count is the answer',
      placement: 'intro',
    },
  ],
  '5': [
    {
      slug: 'julia-flip',
      caption: 'same loop, roles swapped — mandelbrot varies c, julia varies z₀',
      placement: 'intro',
    },
  ],
};
