// Figures for module 1-3 — Thinking in Parallel.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./1-3/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '3': [
    {
      slug: 'scatter-gather',
      caption: "you can't push results to neighbours — pull what you need instead",
      placement: 'intro',
    },
  ],
  '4': [
    {
      slug: 'edge-clamp',
      caption: "signal[64] doesn't exist — clamp before you knock",
      placement: 'intro',
    },
  ],
  '5': [
    {
      slug: 'stencil-window',
      caption: 'read five, write one — always your own cell',
      placement: 'intro',
    },
  ],
};
