// Figures for "Data In, Data Out" — uuid 42b68d01-e46e-455d-9321-2425db8b9eb6.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'output-shapes': [
    {
      name: 'output-shapes',
      caption: "output is a shape — [6] is a line, [4, 4] is rows of rows",
      placement: 'intro',
    },
  ],
  'grayscale': [
    {
      name: 'per-pixel',
      caption: "one pixel in → one thread → one gray pixel out, for every pixel at once",
      placement: 'intro',
    },
  ],
  'image-as-data': [
    {
      name: 'image-as-data',
      caption: "drop graphical: true and a pixel is just four numbers",
      placement: 'intro',
    },
  ],
};
