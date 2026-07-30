// Figures for module 1-4 — Pipelines & Textures.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./1-4/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '2': [
    {
      slug: 'texture-handoff',
      caption: 'kernel to kernel by texture — javascript never sees the middle',
      placement: 'intro',
    },
  ],
  '4': [
    {
      slug: 'ping-pong',
      caption: 'immutable: true — a fresh texture per step makes feedback legal',
      placement: 'intro',
    },
  ],
};
