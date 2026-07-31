// Figures for "Bitonic Sort" — uuid 84e0728e-6dbd-4f06-8c76-14b708a55b47.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'compare-exchange': [
    {
      name: 'gather-not-swap',
      caption: 'nobody swaps — both threads look at both values and keep their own',
      placement: 'intro',
    },
  ],
  'direction-bit': [
    {
      name: 'two-blocks',
      caption: 'neighbouring blocks sort opposite ways, and your index already knows which',
      placement: 'intro',
    },
  ],
  'the-network': [
    {
      name: 'network8',
      caption: 'six passes, twenty-four comparators, and not one of them depends on a value',
      placement: 'intro',
    },
  ],
};
