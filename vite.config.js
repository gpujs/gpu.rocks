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
  // Minify names and whitespace, but NOT syntax.
  //
  // gpu.js compiles a kernel by parsing `fn.toString()`, so whatever the
  // minifier leaves behind is the language its transpiler has to accept. The
  // default esbuild pass rewrites ordinary statements into forms it cannot
  // parse — measured on a real kernel, not assumed:
  //
  //   sum += v; if (v > hi) hi = v;   ->   t+=s,s>n&&(n=s)
  //   if (sum < 0) sum = 0; return …  ->   return t<0&&(t=0),t+n
  //
  // which is where "does not yet support the comma operator", "assignment used
  // as an expression" and GLSL's "'&&' : wrong operand types" all came from.
  // Turning off syntax rewriting keeps every kernel parseable and still mangles
  // identifiers and strips whitespace, so the size cost is small.
  //
  // This applies to any bundled gpu.js kernel, not just ours.
  esbuild: {
    minifySyntax: false,
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
