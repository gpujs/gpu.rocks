// Figures for "Gradient Descent" — uuid c94c3f22-2b9b-46ef-bad3-4a62fcd1935a.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'the-gradient': [
    {
      name: 'fan',
      caption: 'read each residual once, spend it twice',
      placement: 'intro',
    },
  ],
  'sweep-the-rate': [
    {
      name: 'cliff',
      caption: 'too short to arrive, exactly right, or thrown out of the bowl',
      placement: 'intro',
    },
  ],
  'many-starts': [
    {
      name: 'wells',
      caption: 'the answer you get is the valley you started in',
      placement: 'intro',
    },
  ],
};
