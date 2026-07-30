// Figures for module 3-3 — Cellular Automata.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./3-3/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '1': [
    {
      slug: 'census',
      caption: 'sum the 3×3 block, subtract yourself — and the torus has no edges',
      placement: 'intro',
    },
  ],
  '3': [
    {
      slug: 'feedback',
      caption: 'the kernel ticks; javascript turns the crank — output straight back in',
      placement: 'intro',
    },
  ],
};
