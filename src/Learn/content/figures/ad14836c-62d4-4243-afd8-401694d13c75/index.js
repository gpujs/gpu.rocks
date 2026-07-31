// Figures for "Sampling & Aliasing" — uuid ad14836c-62d4-4243-afd8-401694d13c75.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'one-thread-one-sample': [
    {
      name: 'index-to-time',
      caption: 'sample 64 is not second 64 — the whole field runs on one division',
      placement: 'intro',
    },
  ],
  'above-nyquist': [
    {
      name: 'same-samples',
      caption: 'two tones, one set of samples — nothing downstream can tell them apart',
      placement: 'intro',
    },
  ],
  'sinc-reconstruction': [
    {
      name: 'sinc-sum',
      caption: 'every sample gets a sinc; the sum threads all of them',
      placement: 'intro',
    },
  ],
};
