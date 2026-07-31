// Figures for "Measuring Speed Honestly" — uuid b9188894-0ae1-4e75-8538-4348f6fc61ae.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'first-call-lie': [
    {
      name: 'warm-up',
      caption: "the first call buys the compiler — time the calls after it",
      placement: 'intro',
    },
  ],
  'transfer-tax': [
    {
      name: 'transfer-tax',
      caption: "same +1 either way — the bill tracks bytes, not math",
      placement: 'intro',
    },
  ],
};
