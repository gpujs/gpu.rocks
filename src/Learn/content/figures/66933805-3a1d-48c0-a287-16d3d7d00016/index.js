// Figures for "Convolution & Filters" — uuid 66933805-3a1d-48c0-a287-16d3d7d00016.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'smooth-a-signal': [
    {
      name: 'window',
      caption: "nothing slides — thread x just reads its own three samples",
      placement: 'intro',
    },
  ],
  'box-blur': [
    {
      name: 'clamp-2d',
      caption: "nine reads and an average; off the edge, the border pixel answers twice",
      placement: 'intro',
    },
  ],
};
