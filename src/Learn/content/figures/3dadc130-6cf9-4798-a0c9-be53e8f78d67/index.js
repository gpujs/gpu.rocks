// Figures for "Reductions" — uuid 3dadc130-6cf9-4798-a0c9-be53e8f78d67.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'partial-sums': [
    {
      name: 'strided',
      caption: "thread x takes every 64th element — neighbours read neighbours at every step",
      placement: 'intro',
    },
  ],
  'halving-step': [
    {
      name: 'one-rung',
      caption: "your partner lives one output-width away",
      placement: 'intro',
    },
  ],
  'ladder-to-scalar': [
    {
      name: 'ladder',
      caption: "halve, halve, halve — the ladder every platform climbs",
      placement: 'intro',
    },
  ],
};
