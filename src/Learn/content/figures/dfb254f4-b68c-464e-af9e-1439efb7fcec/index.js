// Figures for "Histograms & Binning" — uuid dfb254f4-b68c-464e-af9e-1439efb7fcec.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'lost-increment': [
    {
      name: 'race-vs-gather',
      caption: "nobody can increment your bin but you — so go and count it yourself",
      placement: 'intro',
    },
  ],
  'binning-values': [
    {
      name: 'bin-edges',
      caption: "bins are half-open, and the last one only closes because you clamped it",
      placement: 'intro',
    },
  ],
  'partial-histograms': [
    {
      name: 'merge',
      caption: "one row per chunk, one column per bin, one reduction down each column",
      placement: 'intro',
    },
  ],
};
