// Figures for module 2-3 — Convolution & Filters.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./2-3/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '1': [
    {
      slug: 'window',
      caption: 'nothing slides — thread x just reads its own three samples',
      placement: 'intro',
    },
  ],
  '3': [
    {
      slug: 'clamp-2d',
      caption: 'nine reads and an average; off the edge, the border pixel answers twice',
      placement: 'intro',
    },
  ],
};
