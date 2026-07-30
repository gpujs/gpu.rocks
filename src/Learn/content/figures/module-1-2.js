// Figures for module 1-2 — Data In, Data Out.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./1-2/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '2': [
    {
      slug: 'output-shapes',
      caption: 'output is a shape — [6] is a line, [4, 4] is rows of rows',
      placement: 'intro',
    },
  ],
  '3': [
    {
      slug: 'per-pixel',
      caption: 'one pixel in → one thread → one gray pixel out, for every pixel at once',
      placement: 'intro',
    },
  ],
  '5': [
    {
      slug: 'image-as-data',
      caption: 'drop graphical: true and a pixel is just four numbers',
      placement: 'intro',
    },
  ],
};
