// Figures for "Radix Sort" — uuid fd3ff796-daed-4036-9202-987b374bb4d3.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'one-digit-pass': [
    {
      name: 'stability',
      caption: 'equal digits keep the order they arrived in — cross those arrows and the previous pass was wasted',
      placement: 'intro',
    },
  ],
  'digit-histogram': [
    {
      name: 'one-pass',
      caption: 'count, scan, and every key knows its slot without comparing itself to anything (four buckets here, sixteen in the code)',
      placement: 'intro',
    },
  ],
  'full-sort': [
    {
      name: 'passes',
      caption: 'low digit first, every pass stable — shown in base 10; the code counts in base 16',
      placement: 'intro',
    },
  ],
};
