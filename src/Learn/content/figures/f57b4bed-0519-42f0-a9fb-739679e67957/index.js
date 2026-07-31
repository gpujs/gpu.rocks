// Figures for "Template Matching" — uuid f57b4bed-0519-42f0-a9fb-739679e67957.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'ssd-score-map': [
    {
      name: 'sliding-window',
      caption: 'one thread per position — and a map that comes out smaller than the scene',
      placement: 'intro',
    },
  ],
  'brightness-breaks-it': [
    {
      name: 'brightness-shift',
      caption: 'the same two windows, before and after somebody turned the lights up',
      placement: 'intro',
    },
  ],
  'normalized-correlation': [
    {
      name: 'two-score-maps',
      caption: 'same scene, same patch, two scores — only one of them is looking at the shape',
      placement: 'intro',
    },
  ],
};
