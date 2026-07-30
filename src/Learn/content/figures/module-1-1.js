// Figures for module 1-1 — Hello, Kernel.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./1-1/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '1': [
    {
      slug: 'fan-out',
      caption: 'one function, sixteen launches — the loop you never wrote',
      placement: 'intro',
    },
  ],
  '3': [
    {
      slug: 'loop-unroll',
      caption: 'same body, new address — the loop unrolls into threads',
      placement: 'intro',
    },
  ],
  '4': [
    {
      slug: 'thread-grid',
      caption: 'two coordinates per thread — the grid is the picture',
      placement: 'intro',
    },
  ],
};
