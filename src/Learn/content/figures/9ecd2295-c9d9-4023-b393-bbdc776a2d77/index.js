// Figures for "Spectrograms" — uuid 9ecd2295-c9d9-4023-b393-bbdc776a2d77.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'one-spectrum': [
    {
      name: 'two-orders',
      caption: 'two signals, one spectrum — the transform never asks what happened first',
      placement: 'intro',
    },
  ],
  stft: [
    {
      name: 'sliding-window',
      caption: 'one window position, one column; slide by the hop and do it again',
      placement: 'intro',
    },
  ],
  'resolution-trade': [
    {
      name: 'short-vs-long',
      caption: 'sharpen one axis and you have spent the other — there is no third option',
      placement: 'intro',
    },
  ],
};
