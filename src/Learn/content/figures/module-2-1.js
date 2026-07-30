// Figures for module 2-1 — Matrix Multiply.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./2-1/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '2': [
    {
      slug: 'row-times-col',
      caption: 'row y across, column x down — 256 threads, each owning one dot product',
      placement: 'intro',
    },
  ],
  '3': [
    {
      slug: 'shared-dim',
      caption: '8 and 12 shape the launch; 32 is the loop’s whole world',
      placement: 'intro',
    },
  ],
};
