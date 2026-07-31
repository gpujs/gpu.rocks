// Figures for "Pipelines & Textures" — uuid 9f4aeaa5-71d4-4b0d-afc4-b08c24b7e08e.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'chain-two-kernels': [
    {
      name: 'texture-handoff',
      caption: "kernel to kernel by texture — javascript never sees the middle",
      placement: 'intro',
    },
  ],
  'iterate-immutable': [
    {
      name: 'ping-pong',
      caption: "immutable: true — a fresh texture per step makes feedback legal",
      placement: 'intro',
    },
  ],
};
