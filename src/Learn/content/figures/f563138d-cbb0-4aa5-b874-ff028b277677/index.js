// Figures for "Windowing & Spectral Leakage" — uuid f563138d-cbb0-4aa5-b874-ff028b277677.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'periodic-extension': [
    {
      name: 'tiled-seam',
      caption: 'half a cycle left over, and every join is a cliff the transform has to describe',
      placement: 'intro',
    },
  ],
  'window-trade': [
    {
      name: 'lobes',
      caption: 'the whole trade in one picture — nobody gets both',
      placement: 'intro',
    },
  ],
};
