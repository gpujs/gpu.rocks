import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  // React 16 predates the automatic JSX runtime
  plugins: [react({ jsxRuntime: 'classic' })],
  define: {
    // webpack 4 shimmed `global` for browser builds and some of the older
    // dependencies rely on it — @nivo/line reads `global.window.devicePixelRatio`
    // unguarded, which throws before the app renders anything.
    global: 'globalThis',
  },
  build: {
    outDir: 'dist',
    // the benchmark legitimately pulls in gpu.js and @nivo
    chunkSizeWarningLimit: 1500,
  },
  worker: {
    // src/Learn/engine/sandbox.worker.js is spawned as `{ type: 'module' }`, so
    // the built worker bundle has to be ESM too — the default ('iife') would
    // ship a script the module worker cannot load in the production build.
    format: 'es',
  },
  server: {
    port: 3000,
  },
});
