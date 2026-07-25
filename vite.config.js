import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// @gpujs/benchmark's CLI colour table writes escapes as legacy octal ('\033'),
// which webpack accepted but Rollup rejects because ES modules are strict mode.
// The file is only pulled in because it sits behind the package entry point;
// rewriting the escapes to their '' equivalent keeps it byte-identical in
// behaviour. Remove once @gpujs/benchmark ships the fix upstream.
const legacyOctalEscapes = {
  name: 'gpujs-benchmark-legacy-octal-escapes',
  transform(code, id) {
    if (id.includes('@gpujs/benchmark') && code.includes('\\033')) {
      return { code: code.replace(/\\033/g, '\\u001b'), map: null };
    }
  },
};

export default defineConfig({
  // React 16 predates the automatic JSX runtime
  plugins: [react({ jsxRuntime: 'classic' }), legacyOctalEscapes],
  define: {
    // webpack 4 shimmed `global` for browser builds and some of the older
    // dependencies rely on it — @nivo/line reads `global.window.devicePixelRatio`
    // unguarded, which throws before the app renders anything.
    global: 'globalThis',
  },
  css: {
    preprocessorOptions: {
      scss: {
        // the stylesheets still use @import and the global colour functions.
        // Both work today and are only removed in Dart Sass 3.0 — migrating
        // them is a separate change from swapping the build tool.
        silenceDeprecations: ['import', 'global-builtin', 'color-functions'],
      },
    },
  },
  build: {
    outDir: 'dist',
    // the benchmark legitimately pulls in gpu.js and @nivo
    chunkSizeWarningLimit: 1500,
  },
  server: {
    port: 3000,
  },
});
