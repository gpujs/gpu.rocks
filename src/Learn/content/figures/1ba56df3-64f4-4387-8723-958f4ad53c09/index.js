// Figures for "Top-K Selection" — uuid 1ba56df3-64f4-4387-8723-958f4ad53c09.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'rank-by-counting': [
    {
      name: 'rank-count',
      caption: 'a rank is a count, and a count is something every element can do alone',
      alt:
        'Eight scores in a row. One of them, highlighted, counts the scores that outrank it: ' +
        'two strictly larger scores count, and an equal score at a lower index counts, while an ' +
        'equal score at a higher index does not. The total, three, is its rank and its output slot.',
      placement: 'intro',
    },
  ],
  'threshold-bisection': [
    {
      name: 'bisect',
      caption: 'eighteen guesses, each one a single counting pass, instead of four billion comparisons',
      alt:
        'A bisection on the value axis. Each step marks a guess in the middle of the live bracket ' +
        'and labels it with how many scores exceed it — 197, then 11, then 2. The half that cannot ' +
        'contain the cutoff is discarded each time, until the bracket is narrow enough that exactly ' +
        'ten scores clear it.',
      placement: 'intro',
    },
  ],
};
