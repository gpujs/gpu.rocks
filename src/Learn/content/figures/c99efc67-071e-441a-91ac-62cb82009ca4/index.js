// Figures for "Progressive Path Tracing" — uuid c99efc67-071e-441a-91ac-62cb82009ca4.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three tasks get one: the ones whose idea has a shape. Ray/sphere
// intersection does not — Ray-Marched Metaballs already owns that picture —
// and the √n task draws its own chart from the learner's own run, which beats
// any diagram of it.

export default {
  'per-thread-dice': [
    {
      name: 'streams',
      caption: 'no shared dice on a GPU — so the thread hashes its own address into its own',
      placement: 'intro',
    },
  ],
  'one-noisy-frame': [
    {
      name: 'one-path',
      caption: 'right on average, wrong every single time',
      placement: 'intro',
    },
  ],
  accumulate: [
    {
      name: 'votes',
      caption: 'the new frame gets 1/(n+1) of the vote; everything before it keeps the rest',
      placement: 'intro',
    },
  ],
};
