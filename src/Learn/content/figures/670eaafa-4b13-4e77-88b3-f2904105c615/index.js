// Figures for "Thresholding & Morphology" — uuid 670eaafa-4b13-4e77-88b3-f2904105c615.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'global-threshold': [
    {
      name: 'one-line',
      caption: "one number cannot serve both ends of a lit scene; a number per neighbourhood can",
      placement: 'intro',
    },
  ],
  'erode-and-dilate': [
    {
      name: 'min-max',
      caption: "gather the same nine samples, then decide what to do with them",
      placement: 'intro',
    },
  ],
  'opening-and-closing': [
    {
      name: 'speck-hole',
      caption: "run them the other way round and you repair the other defect",
      placement: 'intro',
    },
  ],
};
