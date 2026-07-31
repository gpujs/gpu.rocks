// Figures for "The DFT, Honestly" — uuid 7b1e3f9b-baf5-4b75-9ad1-3c05f445a3db.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'one-bin': [
    {
      name: 'correlate',
      caption: 'a transform is a pile of dot products — this is one of them',
      placement: 'intro',
    },
  ],
  'why-complex': [
    {
      name: 'two-planes',
      caption: 'the delay swings the pair around; the radius is the answer you wanted',
      placement: 'intro',
    },
  ],
  'magnitude-and-mirror': [
    {
      name: 'mirror',
      caption: 'half of every real spectrum is news you already have',
      placement: 'intro',
    },
  ],
};
