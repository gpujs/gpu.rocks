// Figures for "Cellular Automata" — uuid 407c2c34-b316-4301-8ec2-b5c829b591e6.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'neighbor-census': [
    {
      name: 'census',
      caption: "sum the 3×3 block, subtract yourself — and the torus has no edges",
      placement: 'intro',
    },
  ],
  'generations': [
    {
      name: 'feedback',
      caption: "the kernel ticks; javascript turns the crank — output straight back in",
      placement: 'intro',
    },
  ],
};
