// gpujsApi.js — structured API documentation dataset for the learn editor's
// completion and signature help. Plain JS module, zero dependencies.
//
// Every fact here was checked against the gpu.js source (~/Documents/gpu.js)
// and src/Learn/engine/sandbox.js (the execution core, where the sandbox globals
// are injected) — do not "fix" docs from memory.
//
// Entry shape:
//   {
//     name:       string   completion label within its context (bare member
//                          name — 'createKernel', 'color', 'abs', …; globals
//                          may be dotted: 'console.log', 'Date.now'),
//     kind:       'method' | 'property' | 'option' | 'global' | 'kernel-api',
//     context:    one of CONTEXTS below,
//     signature:  VS Code-style signature string, e.g.
//                 "createKernel(kernelFunction, settings?): Kernel",
//     params:     [{ name, type, doc, optional?, options? }] — `options` names
//                 the CONTEXT whose entries document the keys of an
//                 options-object parameter (e.g. createKernel's settings →
//                 'kernel-settings'); signature help drills into it when the
//                 cursor sits inside a property of that argument.
//     doc:        1-3 plain-text sentences (may contain <code> inline HTML),
//     insertText: only when it differs from name. Placeholder convention:
//                 ${label} marks a tab-stop the consumer converts to its own
//                 snippet-field syntax (order of appearance = tab order).
//     kernelSafe: boolean, set on 'math-member' and native 'global' entries —
//                 true when gpu.js can transpile the call inside a kernel.
//   }
//
// Exports: CONTEXTS, entries, byName (per-context name → entry), getSignature.

export const CONTEXTS = [
  'gpu-instance', // methods on a `new GPU()` instance
  'gpu-settings', // keys of the `new GPU(settings)` object
  'kernel-settings', // keys of the createKernel settings object
  'kernel-method', // methods/properties on a created kernel (and Texture.toArray)
  'kernel-inside', // this.* API inside kernel functions
  'global', // sandbox globals + general JS natives
  'utils-member', // members of the injected `utils` object
  'math-member', // Math.* members (kernelSafe flags which work in kernels)
];

// ---- GPU instance methods (src/gpu.js) ------------------------------------

const gpuInstance = [
  {
    name: 'createKernel',
    kind: 'method',
    context: 'gpu-instance',
    signature: 'createKernel(kernelFunction, settings?): Kernel',
    params: [
      {
        name: 'kernelFunction',
        type: 'function',
        doc: 'Runs once per output cell; use this.thread.x/y/z to know which cell, and return its number (or call this.color when graphical).',
      },
      {
        name: 'settings',
        type: 'object',
        optional: true,
        options: 'kernel-settings',
        doc: 'Kernel options: output, graphical, pipeline, constants, immutable, precision, …',
      },
    ],
    doc: 'Compiles a JavaScript function into a kernel that runs once per element of <code>output</code> — a shader on the GPU backend, a generated loop on CPU. The returned kernel is called like a normal function and also exposes chainable setters (setOutput, setGraphical, …).',
    insertText: 'createKernel(function () {\n  ${body}\n}, { output: [${size}] })',
  },
  {
    name: 'createKernelMap',
    kind: 'method',
    context: 'gpu-instance',
    signature: 'createKernelMap(subKernels, kernelFunction, settings?): Kernel',
    params: [
      {
        name: 'subKernels',
        type: 'object | function[]',
        doc: 'Named functions ({ name: fn }) or an array of named functions, each callable from the kernel; every one’s output is saved.',
      },
      { name: 'kernelFunction', type: 'function', doc: 'Root kernel; may call the sub-kernel functions.' },
      { name: 'settings', type: 'object', optional: true, options: 'kernel-settings', doc: 'Same options as createKernel.' },
    ],
    doc: 'Like createKernel, but also captures the output of every sub-function: calling the kernel returns <code>{ result, …one entry per sub-kernel }</code> (keyed by property name, or index for the array form). Not available in ’dev’ mode.',
    insertText: 'createKernelMap({ ${name}: ${fn} }, ${kernelFunction}, { output: [${size}] })',
  },
  {
    name: 'combineKernels',
    kind: 'method',
    context: 'gpu-instance',
    signature: 'combineKernels(...kernels, combinerFunction): Function',
    params: [
      { name: '...kernels', type: 'Kernel', doc: 'Already-created kernels to chain (they may have different output sizes).' },
      {
        name: 'combinerFunction',
        type: 'function',
        doc: 'Ordinary function that calls the kernels, e.g. (a, b, c) => multiply(add(a, b), c).',
      },
    ],
    doc: 'Chains kernels into one call without CPU↔GPU round trips: on the GL backend every sub-kernel is switched to pipeline mode so intermediates stay on the GPU as textures and only the final result is read back as arrays; on CPU it simply returns the combiner function unchanged.',
    insertText: 'combineKernels(${kernelA}, ${kernelB}, function (${a}, ${b}) {\n  return ${kernelB}(${kernelA}(${a}, ${b}));\n})',
  },
  {
    name: 'addFunction',
    kind: 'method',
    context: 'gpu-instance',
    signature: 'addFunction(source, settings?): GPU',
    params: [
      { name: 'source', type: 'function | string', doc: 'A named function to make callable from inside every kernel.' },
      {
        name: 'settings',
        type: '{ argumentTypes?, returnType? }',
        optional: true,
        doc: "Optional strong typing, e.g. { argumentTypes: { a: 'Number' }, returnType: 'Array(2)' } — required for Array(2/3/4) returns, otherwise inferred.",
      },
    ],
    doc: 'Registers a helper function that all kernels of this GPU instance can call. Must be called before <code>createKernel</code>; the helper obeys the same kernel-language rules (no closures, whitelisted Math only).',
  },
  {
    name: 'addNativeFunction',
    kind: 'method',
    context: 'gpu-instance',
    signature: 'addNativeFunction(name, source, settings?): GPU',
    params: [
      { name: 'name', type: 'string', doc: 'Name kernels use to call it.' },
      { name: 'source', type: 'string', doc: 'The complete native (GLSL on GPU) implementation as a string.' },
      { name: 'settings', type: 'object', optional: true, doc: 'Optional typing information.' },
    ],
    doc: 'Registers a function written directly in the backend’s native language (GLSL for WebGL) — it is not portable to the CPU backend. Throws if called after any kernel has been created on this instance.',
  },
  {
    name: 'destroy',
    kind: 'method',
    context: 'gpu-instance',
    signature: 'destroy(): Promise<void>',
    params: [],
    doc: 'Destroys every kernel made by this instance and releases its WebGL context (deferred one tick, hence the Promise). In this editor, kernels from the previous run are destroyed automatically at the start of the next run.',
  },
];

// ---- new GPU(settings) options (README "GPU Settings"; the constructor in
// gpu.js src/gpu.js reads settings.mode / .canvas / .context — the installed
// 2.19.9 supports modes 'gpu' | 'cpu' | 'dev' plus the internal kernels
// 'webgl' | 'webgl2' | 'headlessgl', the latter Node-only) ------------------

const gpuSettings = [
  {
    name: 'mode',
    kind: 'option',
    context: 'gpu-settings',
    signature: "mode: 'gpu' | 'cpu' | 'webgl' | 'webgl2' | 'dev'",
    params: [],
    doc: 'Default ’gpu’ — best supported GL backend (webgl2, then webgl), falling back to cpu. ’webgl’/’webgl2’ force one backend; ’dev’ runs the kernel as plain un-transpiled JavaScript so breakpoints work. In this editor the Run-mode dropdown overrides this setting, so <code>new GPU()</code> is all you need.',
    insertText: "mode: '${gpu}'",
  },
  {
    name: 'canvas',
    kind: 'option',
    context: 'gpu-settings',
    signature: 'canvas: HTMLCanvasElement',
    params: [],
    doc: 'Optional existing canvas for gpu.js to render into — for sharing one canvas with another library (e.g. THREE.js). Without it, graphical kernels create their own <code>kernel.canvas</code>.',
  },
  {
    name: 'context',
    kind: 'option',
    context: 'gpu-settings',
    signature: 'context: WebGLRenderingContext | WebGL2RenderingContext',
    params: [],
    doc: 'Optional existing rendering context to share with another library. The backend is chosen to match the context (webgl2 vs webgl), overriding <code>mode</code> — also the way to get custom context attributes like <code>premultipliedAlpha</code>.',
  },
];

// ---- createKernel settings options (README "gpu.createKernel Settings",
// defaults verified in src/backend/kernel.js) --------------------------------

const kernelSettings = [
  {
    name: 'output',
    kind: 'option',
    context: 'kernel-settings',
    signature: 'output: [w] | [w, h] | [w, h, d] | { x, y?, z? }',
    params: [],
    doc: 'Size and dimensionality of the result — one kernel thread runs per cell. Required before the first call (here or via setOutput). 2D graphical kernels use [width, height].',
    insertText: 'output: [${size}]',
  },
  {
    name: 'graphical',
    kind: 'option',
    context: 'kernel-settings',
    signature: 'graphical: boolean',
    params: [],
    doc: 'Default false. Renders to a canvas instead of returning numbers: output must be [width, height], the kernel calls <code>this.color(r, g, b, a?)</code>, its return value is ignored, and the image is read via <code>kernel.canvas</code> / <code>kernel.getPixels()</code>. Forces precision ’unsigned’, which has no 2D pixel-array type — so an image handed to a graphical kernel as a nested <code>image[y][x] = [r, g, b, a]</code> array makes gpu.js silently swap in a CPU kernel. This course’s images are <code>ImageData</code> (<code>utils.makeTestImage</code>), which reads as the same per-pixel array and does run on the GPU.',
    insertText: 'graphical: true',
  },
  {
    name: 'pipeline',
    kind: 'option',
    context: 'kernel-settings',
    signature: 'pipeline: boolean',
    params: [],
    doc: 'Default false. Keeps results on the GPU: on the GL backend the kernel returns a <code>Texture</code> you can pass straight into another kernel (call <code>.toArray()</code> to read values); on CPU results stay plain arrays with no .toArray, so guard with <code>r.toArray ? r.toArray() : r</code>.',
    insertText: 'pipeline: true',
  },
  {
    name: 'immutable',
    kind: 'option',
    context: 'kernel-settings',
    signature: 'immutable: boolean',
    params: [],
    doc: 'Default false — gpu.js recycles the kernel’s output texture between calls, so a feedback loop like <code>state = k(state)</code> would read and write the same texture on the GL backend. Set true so each call allocates a fresh output; free old ones with <code>texture.delete()</code>.',
    insertText: 'immutable: true',
  },
  {
    name: 'constants',
    kind: 'option',
    context: 'kernel-settings',
    signature: 'constants: { [name]: number | number[] }',
    params: [],
    doc: 'Values baked into the kernel and read as <code>this.constants.name</code>. A for-loop bounded by a constant counts as fixed-size, so it needs no loopMaxIterations.',
    insertText: 'constants: { ${name}: ${value} }',
  },
  {
    name: 'loopMaxIterations',
    kind: 'option',
    context: 'kernel-settings',
    signature: 'loopMaxIterations: number',
    params: [],
    doc: 'Default 1000. Safety cap for loops whose bound is not statically known (e.g. bounded by an argument) — iterations beyond the cap are cut off on the GPU backend.',
  },
  {
    name: 'dynamicOutput',
    kind: 'option',
    context: 'kernel-settings',
    signature: 'dynamicOutput: boolean',
    params: [],
    doc: 'Default false. When true, <code>kernel.setOutput([…])</code> may be called after the kernel has compiled to resize its output between calls.',
    insertText: 'dynamicOutput: true',
  },
  {
    name: 'dynamicArguments',
    kind: 'option',
    context: 'kernel-settings',
    signature: 'dynamicArguments: boolean',
    params: [],
    doc: 'Default false. When true, successive calls may pass arrays/textures of different sizes; otherwise argument shapes are baked in at first compile.',
    insertText: 'dynamicArguments: true',
  },
  {
    name: 'precision',
    kind: 'option',
    context: 'kernel-settings',
    signature: "precision: 'single' | 'unsigned'",
    params: [],
    doc: '’single’ stores real float32 values per channel (exact numeric results on the GL backend); ’unsigned’ packs floats into 8-bit RGBA for older hardware, and supports fewer argument types (no 2D pixel-array). Default depends on device support; graphical kernels always use ’unsigned’, and asking for ’single’ alongside <code>graphical: true</code> does not override that.',
    insertText: "precision: '${single}'",
  },
  {
    name: 'argumentTypes',
    kind: 'option',
    context: 'kernel-settings',
    signature: "argumentTypes: { [paramName]: type } | type[]",
    params: [],
    doc: "Overrides type inference per kernel parameter. The course's images arrive as <code>ImageData</code> and need no override. You only need one for an image you build yourself as a nested <code>image[y][x] = [r, g, b, a]</code> array — <code>{ image: 'Array2D(4)' }</code>, because inference would guess a flat 3D 'Array' the GL backend cannot partially index. That type exists only at 'single' precision, so it cannot rescue a <code>graphical: true</code> kernel; pass an ImageData there.",
    insertText: "argumentTypes: { ${param}: '${type}' }",
  },
  {
    name: 'tactic',
    kind: 'option',
    context: 'kernel-settings',
    signature: "tactic: 'speed' | 'balanced' | 'precision'",
    params: [],
    doc: 'GL shader precision hint: ’speed’ = lowp, ’balanced’ = mediump, ’precision’ = highp. Default is the lowest resolution the output supports; no effect on CPU.',
    insertText: "tactic: '${precision}'",
  },
  {
    name: 'useLegacyEncoder',
    kind: 'option',
    context: 'kernel-settings',
    signature: 'useLegacyEncoder: boolean',
    params: [],
    doc: 'Default false. Switches the GL float encoder to the older implementation for the rare drivers where the current one misbehaves. GL backend only.',
  },
];

// ---- Kernel instance API (src/backend/kernel.js setters, chainable via
// src/kernel-run-shortcut.js; getPixels in backend/gl/kernel.js:853 and
// backend/cpu/kernel.js:403) ------------------------------------------------

function setter(name, signature, doc, params = []) {
  return { name, kind: 'method', context: 'kernel-method', signature, params, doc };
}

const kernelMethods = [
  setter(
    'setOutput',
    'setOutput([w] | [w, h] | [w, h, d] | { x, y?, z? }): Kernel',
    'Chainable. Sets the output size. After the kernel has compiled it only works when <code>dynamicOutput</code> is true.',
    [{ name: 'output', type: 'number[] | object', doc: 'e.g. [512, 512]' }]
  ),
  setter(
    'setGraphical',
    'setGraphical(flag): Kernel',
    'Chainable. Toggles canvas output (see the <code>graphical</code> option). Also forces precision ’unsigned’.',
    [{ name: 'flag', type: 'boolean', doc: 'true to render pixels via this.color.' }]
  ),
  setter(
    'setPipeline',
    'setPipeline(flag): Kernel',
    'Chainable. Toggles pipeline mode: GL-backend results become Textures you pass to other kernels (read with .toArray()); CPU results remain plain arrays.',
    [{ name: 'flag', type: 'boolean', doc: 'true to keep results on the GPU.' }]
  ),
  setter(
    'setImmutable',
    'setImmutable(flag): Kernel',
    'Chainable. When true, each call allocates a fresh output instead of recycling — required for texture feedback loops; clean up with texture.delete().',
    [{ name: 'flag', type: 'boolean', doc: 'true for fresh output per call (default false).' }]
  ),
  setter(
    'setConstants',
    'setConstants(constants): Kernel',
    'Chainable. Sets the values readable as <code>this.constants.name</code> inside the kernel.',
    [{ name: 'constants', type: 'object', doc: '{ name: number | array }' }]
  ),
  setter(
    'setLoopMaxIterations',
    'setLoopMaxIterations(max): Kernel',
    'Chainable. Caps loops whose bound is not statically known (default 1000).',
    [{ name: 'max', type: 'number', doc: 'Maximum iterations per loop.' }]
  ),
  setter(
    'setDynamicOutput',
    'setDynamicOutput(flag): Kernel',
    'Chainable. Allows setOutput() to resize the output after compilation.',
    [{ name: 'flag', type: 'boolean', doc: 'true to allow post-compile resizing.' }]
  ),
  setter(
    'setDynamicArguments',
    'setDynamicArguments(flag): Kernel',
    'Chainable. Allows argument arrays/textures to change size between calls.',
    [{ name: 'flag', type: 'boolean', doc: 'true to allow varying argument sizes.' }]
  ),
  setter(
    'setPrecision',
    "setPrecision('unsigned' | 'single'): Kernel",
    'Chainable. ’single’ = real float32 storage (exact values on GL); ’unsigned’ = 8-bit packed (required for graphical kernels).',
    [{ name: 'precision', type: "'unsigned' | 'single'", doc: 'Storage precision.' }]
  ),
  setter(
    'setArgumentTypes',
    'setArgumentTypes({ [paramName]: type } | type[]): Kernel',
    "Chainable. Overrides argument type inference; the object form is keyed by kernel parameter name (e.g. { image: 'Array2D(4)' }), the array form is positional.",
    [{ name: 'argumentTypes', type: 'object | string[]', doc: 'Per-parameter gpu.js type strings.' }]
  ),
  {
    name: 'canvas',
    kind: 'property',
    context: 'kernel-method',
    signature: 'canvas: HTMLCanvasElement',
    params: [],
    doc: 'The canvas a graphical kernel draws into. Call the kernel first, then hand it to the preview: <code>render(kernel.canvas)</code>.',
  },
  {
    name: 'getPixels',
    kind: 'method',
    context: 'kernel-method',
    signature: 'getPixels(flip?): Uint8ClampedArray',
    params: [
      {
        name: 'flip',
        type: 'boolean',
        optional: true,
        doc: 'true returns raw WebGL readPixels order (bottom row first); default returns rows top-to-bottom.',
      },
    ],
    doc: 'Reads a graphical kernel’s pixels as flat RGBA bytes [r,g,b,a, r,g,b,a, …]. By default rows come top-to-bottom like getImageData on both backends (gpu.js un-flips the GL readback for you); pass <code>flip: true</code> for the raw bottom-up WebGL order.',
  },
  {
    name: 'toArray',
    kind: 'method',
    context: 'kernel-method',
    signature: 'toArray(): Float32Array | Float32Array[] | Float32Array[][]',
    params: [],
    doc: 'On a <code>Texture</code> returned by a pipeline kernel (GL backend): downloads the texture into JavaScript arrays shaped like the kernel’s output. CPU pipeline results are already plain arrays and have no toArray.',
  },
  {
    name: 'destroy',
    kind: 'method',
    context: 'kernel-method',
    signature: 'destroy(removeCanvasReferences?): void',
    params: [
      { name: 'removeCanvasReferences', type: 'boolean', optional: true, doc: 'Also drop the canvas reference.' },
    ],
    doc: 'Frees this kernel’s GPU resources. Usually <code>gpu.destroy()</code> is enough — and this editor destroys the previous run’s kernels automatically.',
  },
];

// ---- Inside kernel functions (kind 'kernel-api') --------------------------

const kernelInside = [
  {
    name: 'thread',
    kind: 'kernel-api',
    context: 'kernel-inside',
    signature: 'this.thread.x | this.thread.y | this.thread.z: number',
    params: [],
    doc: 'Which output cell this invocation is computing. For output [w, h]: <code>this.thread.x</code> is the column, <code>this.thread.y</code> the row; unused dimensions read 0.',
    insertText: 'thread.${x}',
  },
  {
    name: 'output',
    kind: 'kernel-api',
    context: 'kernel-inside',
    signature: 'this.output.x | this.output.y | this.output.z: number',
    params: [],
    doc: 'The kernel’s output size as set via the <code>output</code> setting or setOutput — handy for normalizing coordinates, e.g. <code>this.thread.x / this.output.x</code>.',
    insertText: 'output.${x}',
  },
  {
    name: 'constants',
    kind: 'kernel-api',
    context: 'kernel-inside',
    signature: 'this.constants.<name>: number | number[]',
    params: [],
    doc: 'Read-only values passed in the <code>constants</code> setting. A for-loop bounded by a constant is treated as fixed-size.',
    insertText: 'constants.${name}',
  },
  {
    name: 'color',
    kind: 'kernel-api',
    context: 'kernel-inside',
    signature: 'this.color(r, g, b, a?): void',
    params: [
      { name: 'r', type: 'number', doc: 'Red, 0–1.' },
      { name: 'g', type: 'number', doc: 'Green, 0–1.' },
      { name: 'b', type: 'number', doc: 'Blue, 0–1.' },
      { name: 'a', type: 'number', optional: true, doc: 'Alpha, 0–1.' },
    ],
    doc: 'Graphical kernels only: sets this pixel’s color (all channels 0–1). The kernel’s return value is ignored when graphical is true.',
    insertText: 'color(${r}, ${g}, ${b}, ${a})',
  },
];

// ---- Sandbox globals (docs from src/Learn/engine/sandbox.js) --------------

const globals = [
  {
    name: 'GPU',
    kind: 'global',
    context: 'global',
    signature: 'new GPU(settings?): GPU',
    params: [
      {
        name: 'settings',
        type: '{ mode?, canvas?, context? }',
        optional: true,
        options: 'gpu-settings',
        doc: "mode: 'gpu' (default), 'cpu', 'webgl', 'webgl2', 'dev'; canvas/context let gpu.js share a rendering surface with another library.",
      },
    ],
    doc: 'Creates a gpu.js context whose createKernel compiles functions to GPU shaders (or CPU loops). In this editor the Run-mode dropdown overrides <code>settings.mode</code>, so <code>new GPU()</code> is all you need.',
    insertText: 'new GPU()',
  },
  {
    name: 'render',
    kind: 'global',
    context: 'global',
    signature: 'render(canvas): void',
    params: [
      { name: 'canvas', type: 'HTMLCanvasElement', doc: 'Usually a graphical kernel’s <code>kernel.canvas</code>.' },
    ],
    doc: 'Shows a canvas in the console pane. The pixels are snapshotted at call time, so rendering the same canvas twice captures two frames. Call the graphical kernel first, then <code>render(kernel.canvas)</code>.',
    insertText: 'render(${canvas})',
  },
  {
    name: 'utils',
    kind: 'global',
    context: 'global',
    signature: 'utils: { seededRandom, makeTestImage, flatten, assert, assertClose }',
    params: [],
    doc: 'Deterministic helper toolbox shared with the task’s tests and inputs — see the utils.* members.',
  },
  {
    name: 'mode',
    kind: 'global',
    context: 'global',
    signature: "mode: 'gpu' | 'cpu'",
    params: [],
    doc: 'The resolved backend for this run (Run-mode dropdown + WebGL availability). ’auto’ picks gpu when WebGL is supported, else cpu.',
  },
  {
    name: 'console.log',
    kind: 'global',
    context: 'global',
    signature: 'console.log(...values): void',
    params: [{ name: '...values', type: 'any', doc: 'Values to print; arrays and typed arrays are summarized.' }],
    doc: 'Prints to the console pane (and mirrors to the browser console). <code>console.info</code> and <code>console.debug</code> are captured the same way.',
    insertText: 'console.log(${value})',
  },
  {
    name: 'console.warn',
    kind: 'global',
    context: 'global',
    signature: 'console.warn(...values): void',
    params: [{ name: '...values', type: 'any', doc: 'Values to print.' }],
    doc: 'Prints a warning line (amber) to the console pane.',
    insertText: 'console.warn(${value})',
  },
  {
    name: 'console.error',
    kind: 'global',
    context: 'global',
    signature: 'console.error(...values): void',
    params: [{ name: '...values', type: 'any', doc: 'Values to print.' }],
    doc: 'Prints an error line (red) to the console pane without stopping the run.',
    insertText: 'console.error(${value})',
  },
  // ---- general-editor JS natives (kernelSafe marks kernel-language support)
  {
    name: 'Math',
    kind: 'global',
    context: 'global',
    signature: 'Math',
    params: [],
    doc: 'Math constants and functions. Inside kernel functions only the gpu.js whitelist works (abs, sin, floor, pow, min/max, … — see Math.* completions); everything works in ordinary code.',
    kernelSafe: true,
  },
  {
    name: 'Array',
    kind: 'global',
    context: 'global',
    signature: 'Array(length?): any[]',
    params: [{ name: 'length', type: 'number', optional: true, doc: 'Initial length.' }],
    doc: 'Standard array constructor for host code. Inside kernels only small literals ([a, b], up to 4 elements) exist — no Array(), .map, or .forEach.',
    kernelSafe: false,
  },
  {
    name: 'Float32Array',
    kind: 'global',
    context: 'global',
    signature: 'new Float32Array(length | array): Float32Array',
    params: [{ name: 'source', type: 'number | ArrayLike<number>', doc: 'Length or values to copy.' }],
    doc: 'The typed array gpu.js kernels return their results in. Host code only.',
    kernelSafe: false,
  },
  {
    name: 'Date.now',
    kind: 'global',
    context: 'global',
    signature: 'Date.now(): number',
    params: [],
    doc: 'Milliseconds since the Unix epoch — handy for quick timing in host code. Not available inside kernels.',
    kernelSafe: false,
  },
  {
    name: 'JSON.stringify',
    kind: 'global',
    context: 'global',
    signature: 'JSON.stringify(value, replacer?, space?): string',
    params: [
      { name: 'value', type: 'any', doc: 'Value to serialize.' },
      { name: 'replacer', type: 'function | array', optional: true, doc: 'Filters/transforms entries.' },
      { name: 'space', type: 'number | string', optional: true, doc: 'Pretty-print indentation.' },
    ],
    doc: 'Serializes a value to JSON — useful for logging small structures. Host code only.',
    kernelSafe: false,
  },
  {
    name: 'Promise',
    kind: 'global',
    context: 'global',
    signature: 'new Promise(executor): Promise',
    params: [{ name: 'executor', type: '(resolve, reject) => void', doc: 'Started immediately.' }],
    doc: 'Your code runs inside an async wrapper, so top-level <code>await</code> works. Host code only.',
    kernelSafe: false,
  },
  {
    name: 'parseFloat',
    kind: 'global',
    context: 'global',
    signature: 'parseFloat(string): number',
    params: [{ name: 'string', type: 'string', doc: 'Text to parse.' }],
    doc: 'Parses a string into a floating-point number. Host code only.',
    kernelSafe: false,
  },
  {
    name: 'parseInt',
    kind: 'global',
    context: 'global',
    signature: 'parseInt(string, radix?): number',
    params: [
      { name: 'string', type: 'string', doc: 'Text to parse.' },
      { name: 'radix', type: 'number', optional: true, doc: 'Base, e.g. 10 or 16.' },
    ],
    doc: 'Parses a string into an integer. Host code only.',
    kernelSafe: false,
  },
];

// ---- utils members (docs from src/Learn/engine/utils.js) ------------------

const utilsMembers = [
  {
    name: 'seededRandom',
    kind: 'method',
    context: 'utils-member',
    signature: 'utils.seededRandom(seed): () => number',
    params: [{ name: 'seed', type: 'number', doc: 'Any integer; the same seed always yields the same sequence.' }],
    doc: 'Returns a deterministic PRNG (mulberry32) producing numbers in [0, 1). Use it instead of Math.random when tests need reproducible data.',
    insertText: 'seededRandom(${seed})',
  },
  {
    name: 'makeTestImage',
    kind: 'method',
    context: 'utils-member',
    signature: 'utils.makeTestImage(size): ImageData',
    params: [{ name: 'size', type: 'number', doc: 'Width and height in pixels.' }],
    doc: 'Deterministic seeded RGBA test image, ready to pass straight into a kernel: inside one, <code>image[this.thread.y][this.thread.x]</code> is an <code>[r, g, b, a]</code> pixel with channels 0–1 on every backend. It is an <code>ImageData</code> rather than a nested array because a <code>graphical: true</code> kernel can only run on the GPU with one (see the <code>graphical</code> option). Host-side, <code>image.plain[y][x]</code> and <code>image.at(x, y)</code> give you the same pixel as a plain array. Channels are quantized to 8-bit steps, so those two views agree exactly. The same size always produces the exact same image.',
    insertText: 'makeTestImage(${size})',
  },
  {
    name: 'flatten',
    kind: 'method',
    context: 'utils-member',
    signature: 'utils.flatten(arr): number[]',
    params: [{ name: 'arr', type: 'array | TypedArray', doc: 'Arbitrarily nested arrays and/or typed arrays.' }],
    doc: 'Flattens nested arrays and typed arrays (any depth) into one plain Array — useful for comparing kernel outputs elementwise.',
    insertText: 'flatten(${arr})',
  },
  {
    name: 'assert',
    kind: 'method',
    context: 'utils-member',
    signature: 'utils.assert(cond, message?): void',
    params: [
      { name: 'cond', type: 'any', doc: 'Truthy passes.' },
      { name: 'message', type: 'string', optional: true, doc: "Error message (default 'assertion failed')." },
    ],
    doc: 'Throws Error(message) when the condition is falsy. A test passes unless it throws.',
    insertText: 'assert(${cond}, ${message})',
  },
  {
    name: 'assertClose',
    kind: 'method',
    context: 'utils-member',
    signature: 'utils.assertClose(a, b, eps?, message?): void',
    params: [
      { name: 'a', type: 'number', doc: 'Actual value — must be a real number (NaN fails).' },
      { name: 'b', type: 'number', doc: 'Expected value.' },
      { name: 'eps', type: 'number', optional: true, doc: 'Allowed difference, default 1e-4.' },
      { name: 'message', type: 'string', optional: true, doc: 'Prefix for the failure message.' },
    ],
    doc: 'Throws unless <code>|a - b| ≤ eps</code> (default 1e-4). Use it for float comparisons — GPU math is not bit-exact with CPU math.',
    insertText: 'assertClose(${a}, ${b}, ${eps})',
  },
];

// ---- Math members ----------------------------------------------------------
// Kernel whitelist source of truth: gpu.js src/backend/function-node.js —
// mathProperties (lines 5-14) and mathFunctions (lines 16-51). Math.clz32 is
// in the parse whitelist but has no WebGL implementation (README: "bits
// directly are hard"; only the WebGPU backend implements it), so it is marked
// kernel-unsafe for this course's cpu/webgl backends. Math.hypot is not
// transpilable at all.

const MATH_PROPS = [
  ['E', "Euler's number, ≈ 2.718."],
  ['PI', 'π, ≈ 3.14159.'],
  ['SQRT2', '√2, ≈ 1.414.'],
  ['SQRT1_2', '√½, ≈ 0.707.'],
  ['LN2', 'Natural log of 2.'],
  ['LN10', 'Natural log of 10.'],
  ['LOG2E', 'Base-2 log of e.'],
  ['LOG10E', 'Base-10 log of e.'],
];

// [name, params, doc, kernelSafe (default true), extraNote]
const MATH_FNS = [
  ['abs', ['x'], 'Absolute value.'],
  ['acos', ['x'], 'Arccosine (radians).'],
  ['acosh', ['x'], 'Inverse hyperbolic cosine.'],
  ['asin', ['x'], 'Arcsine (radians).'],
  ['asinh', ['x'], 'Inverse hyperbolic sine.'],
  ['atan', ['x'], 'Arctangent (radians).'],
  ['atan2', ['y', 'x'], 'Angle of the point (x, y) from the positive x-axis, in radians. Note the argument order: y first.'],
  ['atanh', ['x'], 'Inverse hyperbolic tangent.'],
  ['cbrt', ['x'], 'Cube root.'],
  ['ceil', ['x'], 'Rounds up to the nearest integer.'],
  [
    'clz32',
    ['x'],
    'Count of leading zero bits in the 32-bit representation. In the gpu.js parse whitelist but not implemented for the WebGL backends — works on CPU mode only, so avoid it in kernels here.',
    false,
  ],
  ['cos', ['x'], 'Cosine (radians).'],
  ['cosh', ['x'], 'Hyperbolic cosine.'],
  ['expm1', ['x'], 'e^x − 1.'],
  ['exp', ['x'], 'e raised to the power x.'],
  ['floor', ['x'], 'Rounds down to the nearest integer. Also the idiom for integer division in kernels: Math.floor(a / b).'],
  ['fround', ['x'], 'Nearest float32 representation.'],
  ['imul', ['a', 'b'], '32-bit integer multiplication.'],
  ['log', ['x'], 'Natural logarithm (base e) — not base 10.'],
  ['log2', ['x'], 'Base-2 logarithm.'],
  ['log10', ['x'], 'Base-10 logarithm.'],
  ['log1p', ['x'], 'ln(1 + x).'],
  ['max', ['a', 'b'], 'Larger of the values. Variadic in host code; inside kernels use exactly two arguments (GLSL max is binary).'],
  ['min', ['a', 'b'], 'Smaller of the values. Variadic in host code; inside kernels use exactly two arguments.'],
  ['pow', ['x', 'y'], 'x raised to the power y.'],
  [
    'random',
    [],
    'Random number in [0, 1). Works inside kernels: on the GL backend gpu.js generates it on the GPU via a plugin (seeded from the CPU) — fine for visuals, but use utils.seededRandom in host code when tests need reproducibility.',
  ],
  ['round', ['x'], 'Rounds to the nearest integer.'],
  ['sign', ['x'], '−1, 0, or 1 by sign.'],
  ['sin', ['x'], 'Sine (radians).'],
  ['sinh', ['x'], 'Hyperbolic sine.'],
  ['sqrt', ['x'], 'Square root.'],
  ['tan', ['x'], 'Tangent (radians).'],
  ['tanh', ['x'], 'Hyperbolic tangent.'],
  ['trunc', ['x'], 'Integer part (truncates toward zero).'],
  [
    'hypot',
    ['...values'],
    '√(sum of squares). NOT transpilable inside kernels (dynamically sized in GLSL) — write Math.sqrt(x * x + y * y) instead.',
    false,
  ],
];

const mathMembers = [
  ...MATH_PROPS.map(([name, doc]) => ({
    name,
    kind: 'property',
    context: 'math-member',
    signature: `Math.${name}: number`,
    params: [],
    doc,
    kernelSafe: true,
  })),
  ...MATH_FNS.map(([name, paramNames, doc, kernelSafe = true]) => ({
    name,
    kind: 'method',
    context: 'math-member',
    signature: `Math.${name}(${paramNames.join(', ')}): number`,
    params: paramNames.map(p => ({ name: p, type: 'number', doc: '' })),
    doc,
    kernelSafe,
    insertText: `${name}(${paramNames.map(p => `\${${p.replace(/^\.+/, '')}}`).join(', ')})`,
  })),
];

// ---- assembled dataset -----------------------------------------------------

export const entries = [
  ...globals,
  ...gpuInstance,
  ...gpuSettings,
  ...kernelSettings,
  ...kernelMethods,
  ...kernelInside,
  ...utilsMembers,
  ...mathMembers,
];

function index(list) {
  const map = Object.create(null);
  for (const entry of list) map[entry.name] = entry;
  return map;
}

export const byName = {
  'gpu-instance': index(gpuInstance),
  'gpu-settings': index(gpuSettings),
  'kernel-settings': index(kernelSettings),
  'kernel-method': index(kernelMethods),
  'kernel-inside': index(kernelInside),
  global: index(globals),
  'utils-member': index(utilsMembers),
  'math-member': index(mathMembers),
};

/**
 * Resolve a callee path to its entry, e.g.
 *   getSignature(['gpu', 'createKernel'])  → the createKernel entry
 *   getSignature(['utils', 'assertClose']) → the assertClose entry
 *   getSignature(['this', 'color'])        → the this.color entry
 *   getSignature(['Math', 'atan2'])        → the Math.atan2 entry
 *   getSignature(['render'])               → the render entry
 * Unknown receivers (any variable name) fall back to GPU-instance methods,
 * then kernel methods — the two APIs user variables typically hold.
 * @param {string[]} calleePath property chain, leftmost receiver first
 * @returns {object | null}
 */
export function getSignature(calleePath) {
  if (!Array.isArray(calleePath) || calleePath.length === 0) return null;
  const head = calleePath[0];
  const last = calleePath[calleePath.length - 1];

  if (calleePath.length === 1) {
    return byName.global[head] || byName['gpu-instance'][head] || null;
  }
  if (head === 'Math') return byName['math-member'][last] || null;
  if (head === 'utils') return byName['utils-member'][last] || null;
  if (head === 'this') {
    // ['this', 'color'] or ['this', 'thread', 'x']
    return byName['kernel-inside'][calleePath[1]] || null;
  }
  // dotted globals: console.log, Date.now, JSON.stringify
  const dotted = byName.global[calleePath.join('.')];
  if (dotted) return dotted;
  // unknown receiver — likely a GPU instance or a kernel
  return byName['gpu-instance'][last] || byName['kernel-method'][last] || null;
}
