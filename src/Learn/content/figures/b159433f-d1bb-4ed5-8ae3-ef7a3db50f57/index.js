// Figures for "Autocorrelation & Pitch" — uuid b159433f-d1bb-4ed5-8ae3-ef7a3db50f57.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'shift-and-compare': [
    {
      name: 'shifted-copy',
      caption: 'slide it half a period and it argues with itself; slide it a whole one and it agrees',
      placement: 'intro',
    },
  ],
  'lag-to-pitch': [
    {
      name: 'parabola',
      caption: 'the tallest sample is not the peak — the peak is between two of them',
      placement: 'intro',
    },
  ],
  'zero-lag-and-octaves': [
    {
      name: 'peaks',
      caption: 'every trap in one picture: the free peak, the right peak, and the tall wrong one',
      placement: 'intro',
    },
  ],
};
