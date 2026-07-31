// Figures for "Matrix Multiply" — uuid 972e080b-a2a9-4151-ac98-d1d9caf7b6b9.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'full-matmul': [
    {
      name: 'row-times-col',
      caption: "row y across, column x down — 256 threads, each owning one dot product",
      placement: 'intro',
    },
  ],
  'rectangular': [
    {
      name: 'shared-dim',
      caption: "8 and 12 shape the launch; 32 is the loop’s whole world",
      placement: 'intro',
    },
  ],
};
