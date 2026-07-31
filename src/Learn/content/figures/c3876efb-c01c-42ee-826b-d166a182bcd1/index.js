// Figures for "Thinking in Parallel" — uuid c3876efb-c01c-42ee-826b-d166a182bcd1.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'invert-the-scatter': [
    {
      name: 'scatter-gather',
      caption: "you can't push results to neighbours — pull what you need instead",
      placement: 'intro',
    },
  ],
  'edges-and-clamps': [
    {
      name: 'edge-clamp',
      caption: "signal[64] doesn't exist — clamp before you knock",
      placement: 'intro',
    },
  ],
  'moving-average': [
    {
      name: 'stencil-window',
      caption: "read five, write one — always your own cell",
      placement: 'intro',
    },
  ],
};
