const Ua=Math.PI*2;function Gi(e=64){const t=new Array(e);for(let s=0;s<e;s++)t[s]=Math.sin(s/e*Ua);return t}function qs(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Va(e,t,s,n,i){const a=i.filter(([f])=>{let m=!1;for(let A=0;A<e;A++){if(!(Math.abs(t(A)-f(A))<=n))return!1;Math.abs(s(A)-f(A))>n&&(m=!0)}return m}).map(f=>f[1]);return a.length&&a.every(f=>f===a[0])?a[0]:null}function bl(e,t,s,n,i){const a=i.filter(([f])=>{let m=!1;for(let A=0;A<e;A++)for(let N=0;N<e;N++){const H=f(A,N);if(!(t[A]&&Math.abs(t[A][N]-H)<=n))return!1;Math.abs(s(A,N)-H)>n&&(m=!0)}return m}).map(f=>f[1]);return a.length&&a.every(f=>f===a[0])?a[0]:null}function Ws(e,t=64){return[[Math.sin(e),"you sampled Math.sin(this.thread.x) directly — the index counts samples, not radians"],[Math.sin(e/t*Math.PI),"that is half a cycle — a full turn is 2 * Math.PI, not Math.PI"],[Math.sin(e/t*360),"Math.sin takes radians, not degrees — a full turn is 2 * Math.PI, not 360"],[Math.sin(e/t),"the 2π factor is missing — x / 64 on its own spans about one radian, not a full cycle"],[Math.sin(e/(t-1)*Ua),"you divided by 63 instead of 64 — the cycle spans all 64 samples, so sample 64 is where it would repeat"]]}function Li(e){return bl(8,e,(t,s)=>(s+t)%2,.001,[[(t,s)=>s%2,"the whole board is the parity of this.thread.x alone — vertical stripes; a checkerboard flips on both axes, so add this.thread.y"],[(t,s)=>t%2,"the whole board is the parity of this.thread.y alone — horizontal stripes; a checkerboard flips on both axes, so add this.thread.x"]])}function On(e,t){return Va(64,s=>e[s],s=>s*t,.01,[[s=>s,"the scale never reached the result — every cell is the bare this.thread.x"]])}var wl={id:"1-1",track:1,title:"Hello, Kernel",blurb:"What a kernel is, what a thread is, and why <code>this.thread.x</code> replaces your for-loop.",tasks:[{slug:"first-kernel",title:"Your First Kernel",intro:`<p>A <strong>kernel</strong> is an ordinary-looking JavaScript function with one twist:
        it doesn't run once. gpu.js compiles it and launches it <strong>once per output cell</strong>,
        all in parallel — each launch is called a <strong>thread</strong>. You never call the function
        in a loop; you tell the GPU how many cells you want, and it runs that many copies.</p>
        <p>That cell count is the <code>output</code> option: <code>output: [16]</code> means
        &ldquo;give me 16 cells&rdquo;, so 16 threads run and their 16 return values come back to you
        collected into one array.</p>`,goal:`<strong>Goal:</strong> finish the kernel so that <strong>16 threads</strong> each return
        the number <code>42</code> — your first parallel program.`,requirements:["Set <code>output</code> to <code>[16]</code> so 16 threads run","Return <code>42</code> from the kernel body","Call the kernel and log the result (already wired up)"],hints:[{title:"Hint 1 — where does the 16 go?",body:`<p><code>output</code> lives in the options object — the second argument to
            <code>createKernel</code>. It's an array because output can have more than one
            dimension (that's task 4).</p>`},{title:"Hint 2 — the whole thing",body:`<p>The whole call:</p>
<pre><code>gpu.createKernel(function () {
  return 42;
}, {
  output: [16],
})</code></pre>
<p>Calling it returns an array of sixteen 42s.</p>`}],transfer:`Launching N copies of one function is <em>the</em> primitive of every GPU API:
        CUDA spells it <code>kernel&lt;&lt;&lt;blocks, threads&gt;&gt;&gt;()</code>, WebGPU calls it
        a compute <code>dispatch</code>, Metal dispatches threadgroups. gpu.js just hides the
        ceremony behind <code>output</code>.`,starterCode:`// A kernel runs once per output cell — in parallel, not in a loop.
const gpu = new GPU({ mode });

const answer = gpu.createKernel(function () {
  // TODO: every thread should return the same number: 42
  return 0;
}, {
  // TODO: give the kernel 16 output cells, not 1
  output: [1],
});

const result = answer();
console.log(result);
`,solutionCode:`// A kernel runs once per output cell — in parallel, not in a loop.
const gpu = new GPU({ mode });

const answer = gpu.createKernel(function () {
  return 42;
}, {
  output: [16],
});

const result = answer();
console.log(result);
`,publicTests:[{name:"kernel runs 16 threads — the result has 16 values",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===16,`expected 16 output values, got ${t&&t.length}`)}},{name:"every thread returns <code>42</code>",run:async e=>{const t=e.kernel();for(let s=0;s<16;s++)e.assertClose(t[s],42,.001,`value from thread ${s}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel();e.assert(t.length===16,"expected 16 output values");let s=0;for(let n=0;n<t.length;n++)s+=t[n];e.assertClose(s,672,.01,"the 16 values should total 672")}}]},{slug:"thread-identity",title:"Who Am I? this.thread.x",intro:`<p>Sixteen identical 42s prove the launch works, but parallel code is only useful if
        each thread can do something <em>different</em>. The trick: every thread knows which output
        cell it owns. That number is <code>this.thread.x</code> — 0 for the first cell, 1 for the
        next, up to <code>output − 1</code>.</p>
        <p>Same function, same arguments, different <code>this.thread.x</code> — that one number is
        the only thing telling the threads apart, and it's how each one finds its own work.</p>`,goal:`<strong>Goal:</strong> make each of the 32 threads return <strong>its own index</strong>,
        so the result counts <code>0, 1, 2, … 31</code>.`,requirements:["Keep <code>output: [32]</code> — 32 threads","Return <code>this.thread.x</code> from the kernel body","No loops, no counters — the index is handed to you"],hints:[{title:"Hint 1 — it’s already there",body:`<p>You don't compute the index and you don't pass it in. Inside the kernel body,
            <code>this.thread.x</code> is simply available — gpu.js fills it in per thread.</p>`},{title:"Hint 2 — the one-liner",body:"<p>The entire kernel body: <code>return this.thread.x;</code></p>"}],transfer:`Every platform hands threads this same self-identity, just under a different name:
        <code>threadIdx</code>/<code>blockIdx</code> in CUDA and ROCm/HIP,
        <code>global_invocation_id</code> in WebGPU's WGSL,
        <code>thread_position_in_grid</code> in Metal.`,starterCode:`// Every thread runs the same body — this.thread.x is what differs.
const gpu = new GPU({ mode });

const whoAmI = gpu.createKernel(function () {
  // TODO: return this thread's own index
  return 0;
}, { output: [32] });

const result = whoAmI();
console.log(result);
`,solutionCode:`// Every thread runs the same body — this.thread.x is what differs.
const gpu = new GPU({ mode });

const whoAmI = gpu.createKernel(function () {
  return this.thread.x;
}, { output: [32] });

const result = whoAmI();
console.log(result);
`,publicTests:[{name:"result holds 32 values",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===32,`expected 32 output values, got ${t&&t.length}`)}},{name:"cell <code>i</code> holds <code>i</code> — each thread reports its index",run:async e=>{const t=e.kernel(),s=Va(32,n=>t[n],n=>n,.001,[[n=>n+1,"every cell is one more than its index — this.thread.x already counts from 0, so the first cell holds 0"]]);for(let n=0;n<32;n++)e.assertClose(t[n],n,.001,s||`cell ${n}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel();let s=0;for(let n=0;n<t.length;n++)s+=t[n];e.assertClose(s,992/2,.01,"the indices should total 496"),e.assertClose(t[0],0,.001,"first thread is index 0"),e.assertClose(t[17],17,.001,"thread 17"),e.assertClose(t[31],31,.001,"last thread is index 31")}}]},{slug:"index-formula",title:"From For-Loop to Formula",intro:`<p>Here's the payoff of the thread index. On the CPU you'd sample a sine wave like
        this:</p>
<pre><code>for (let i = 0; i &lt; 64; i++) {
  wave[i] = Math.sin(i / 64 * 2 * Math.PI);
}</code></pre>
        <p>On the GPU, the loop <strong>disappears</strong> — the 64 iterations become 64 threads,
        and the loop variable <code>i</code> becomes <code>this.thread.x</code>. The body of the loop
        is your kernel body, unchanged. (<code>Math.sin</code> and <code>Math.PI</code> work inside
        kernels, along with most of <code>Math</code>.)</p>`,goal:`<strong>Goal:</strong> sample one full sine cycle across 64 threads — thread
        <code>x</code> returns <code>Math.sin(x / 64 * 2 * Math.PI)</code>.`,requirements:["Keep <code>output: [64]</code> — one thread per sample","Use <code>this.thread.x</code> where the CPU loop used <code>i</code>","Return one sine sample per thread — the CPU loop body, unchanged except for the index"],hints:[{title:"Hint 1 — the translation rule",body:`<p>Take the CPU loop body, delete the loop, and substitute
            <code>this.thread.x</code> for <code>i</code>. That mechanical rewrite is how most
            for-loops become kernels.</p>`},{title:"Hint 2 — the body",body:"<pre><code>return Math.sin(this.thread.x / 64 * 2 * Math.PI);</code></pre>"}],transfer:`This loop-body-becomes-kernel-body rewrite is called an <em>embarrassingly
        parallel map</em>, and it's the bread and butter of GPGPU: the same move turns a pixel loop
        into a Metal fragment shader, a physics update into a CUDA kernel, or an array transform
        into a WebGPU compute pass.`,starterCode:`// The for-loop is gone — 64 threads each compute one sample.
const gpu = new GPU({ mode });

// CPU version, for reference:
//   for (let i = 0; i < 64; i++) wave[i] = Math.sin(i / 64 * 2 * Math.PI);

const wave = gpu.createKernel(function () {
  // TODO: one sample of a sine wave — i is this.thread.x now
  return 0;
}, { output: [64] });

const samples = wave();
console.log(samples);
`,solutionCode:`// The for-loop is gone — 64 threads each compute one sample.
const gpu = new GPU({ mode });

const wave = gpu.createKernel(function () {
  return Math.sin(this.thread.x / 64 * 2 * Math.PI);
}, { output: [64] });

const samples = wave();
console.log(samples);
`,publicTests:[{name:"kernel produces 64 samples",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===64,`expected 64 samples, got ${t&&t.length}`)}},{name:"samples trace <code>sin(x / 64 · 2π)</code> — starts at 0, peaks at thread 16",run:async e=>{const t=e.kernel(),s=Gi(64);e.assertClose(t[0],0,.001,"thread 0: sin(0) = 0");const n=qs(t[16],1,.001,Ws(16));e.assertClose(t[16],1,.001,n||"thread 16: quarter cycle, sin = 1");const i=qs(t[48],-1,.001,Ws(48));e.assertClose(t[48],-1,.001,i||"thread 48: three-quarter cycle, sin = -1");for(let a=0;a<64;a+=7){const f=qs(t[a],s[a],.001,Ws(a));e.assertClose(t[a],s[a],.001,f||`sample ${a}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(),s=Gi(64);for(let n=0;n<64;n++){const i=qs(t[n],s[n],.001,Ws(n));e.assertClose(t[n],s[n],.001,i||`sample ${n}`)}}}]},{slug:"checkerboard",title:"A Second Dimension: this.thread.y",intro:`<p>Threads don't have to line up in a row. Give <code>output</code> two numbers —
        <code>output: [8, 8]</code> — and gpu.js launches an 8×8 <strong>grid</strong> of 64
        threads. Each one now has two coordinates: <code>this.thread.x</code> is its column and
        <code>this.thread.y</code> is its row, and the result comes back as an array of rows you
        read as <code>result[y][x]</code>.</p>
        <p>To prove both coordinates are live, paint a classic: a checkerboard. A cell is
        &ldquo;black&rdquo; or &ldquo;white&rdquo; depending on whether <code>x + y</code> is even
        or odd — which is just <code>(x + y) % 2</code>.</p>`,goal:`<strong>Goal:</strong> launch an 8×8 grid where each cell holds
        <code>(x + y) % 2</code> — an alternating pattern of 0s and 1s.`,requirements:["Change <code>output</code> to a grid: <code>[8, 8]</code>","Use <code>this.thread.x</code> <em>and</em> <code>this.thread.y</code>","Return 0 or 1 in a checkerboard — the parity of the two coordinates added together"],hints:[{title:"Hint 1 — what changes with 2D?",body:`<p>Two things: <code>output</code> gets a second number
            (<code>[width, height]</code>), and <code>this.thread.y</code> starts meaning
            something. Nothing else about the kernel changes.</p>`},{title:"Hint 2 — the pattern",body:`<pre><code>return (this.thread.x + this.thread.y) % 2;</code></pre>
<p>Neighbours differ by one in <code>x</code> or <code>y</code>, so the parity flips
            checkerboard-style.</p>`}],transfer:`GPUs are built around 2D grids because images are 2D: ROCm and CUDA launch
        <code>dim3</code>-shaped blocks, WebGPU dispatches workgroups across x/y/z, and Metal's
        grids are up to three-dimensional. One thread per pixel — the idea module 1.2 runs with —
        starts exactly here.`,starterCode:`// output: [width, height] launches a whole grid of threads.
const gpu = new GPU({ mode });

const board = gpu.createKernel(function () {
  // TODO: return (x + y) % 2 using BOTH thread coordinates
  return this.thread.x % 2;
}, {
  // TODO: make this an 8×8 grid, not an 8-cell line
  output: [8],
});

const result = board();
console.log(result);
`,solutionCode:`// output: [width, height] launches a whole grid of threads.
const gpu = new GPU({ mode });

const board = gpu.createKernel(function () {
  return (this.thread.x + this.thread.y) % 2;
}, {
  output: [8, 8],
});

const result = board();
console.log(result);
`,publicTests:[{name:"result is an 8×8 grid — 8 rows of 8 values",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===8,`expected 8 rows, got ${t&&t.length}`),e.assert(t[0]&&typeof t[0]!="number"&&t[0].length===8,"each row should hold 8 values — is your output still 1D?")}},{name:"cells alternate like a checkerboard: <code>(x + y) % 2</code>",run:async e=>{const t=e.kernel(),s=Li(t);e.assertClose(t[0][0],0,.001,s||"corner [0][0] is 0"),e.assertClose(t[0][1],1,.001,s||"its neighbour [0][1] is 1"),e.assertClose(t[1][0],1,.001,s||"its neighbour [1][0] is 1"),e.assertClose(t[7][7],0,.001,"far corner [7][7] is 0 (7 + 7 is even)")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(),s=Li(t);for(let n=0;n<8;n++)for(let i=0;i<8;i++)e.assertClose(t[n][i],(i+n)%2,.001,s||`cell [${n}][${i}]`)}}]},{slug:"first-argument",title:"Pass Something In",intro:`<p>So far every kernel has conjured its output from thread coordinates alone. Real
        kernels also take <strong>arguments</strong> — declare a parameter on the kernel function,
        pass a value when you call it, and every thread sees that same value. Combine it with
        <code>this.thread.x</code> and each thread computes something different from shared
        input.</p>
        <p>Here's the payoff: a compiled kernel is <strong>reusable</strong>. Build it once, call it
        with <code>2.5</code>, call it again with <code>0.5</code> — two parallel launches, zero
        recompiles. That build-once/call-many rhythm is how all real GPU code is structured.</p>`,goal:`<strong>Goal:</strong> make <code>ramp</code> return <code>scale * this.thread.x</code>,
        then call it twice — once with <code>2.5</code>, once with <code>0.5</code>.`,requirements:["Give the kernel function a <code>scale</code> parameter","Multiply the shared argument by this thread's index — shared argument × thread identity","Call the kernel twice with different scales (already wired up)"],hints:[{title:"Hint 1 — where arguments come from",body:`<p>Kernel arguments are ordinary function parameters:
            <code>function (scale) { … }</code>, called as <code>ramp(3)</code>. Every one of the
            64 threads receives the same <code>3</code>.</p>`},{title:"Hint 2 — the body",body:`<p><code>return scale * this.thread.x;</code> — the argument is shared, the
            index is per-thread, the product is different in every cell.</p>`}],transfer:`A value shared by all threads is a <em>uniform</em>: WebGPU binds it as a uniform
        buffer, CUDA and ROCm pass it as a kernel launch parameter, Metal hands it over with
        <code>setBytes</code>. And build-once/dispatch-many is universal too — shader and kernel
        compilation is expensive everywhere, so it's paid once up front.`,starterCode:`// Arguments are shared by all threads; this.thread.x stays per-thread.
const gpu = new GPU({ mode });

const ramp = gpu.createKernel(function (scale) {
  // TODO: scale this thread's index by the argument
  // (keep the float argument on the LEFT of the multiply)
  return this.thread.x;
}, { output: [64] });

// One kernel, two launches — no recompilation between calls.
console.log('scale 2.5:', ramp(2.5));
console.log('scale 0.5:', ramp(0.5));
`,solutionCode:`// Arguments are shared by all threads; this.thread.x stays per-thread.
const gpu = new GPU({ mode });

const ramp = gpu.createKernel(function (scale) {
  // float on the left → float math on the GPU
  return scale * this.thread.x;
}, { output: [64] });

// One kernel, two launches — no recompilation between calls.
console.log('scale 2.5:', ramp(2.5));
console.log('scale 0.5:', ramp(0.5));
`,publicTests:[{name:"called with <code>2.5</code>, cell <code>i</code> holds <code>i * 2.5</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(2.5);e.assert(t&&t.length===64,`expected 64 output values, got ${t&&t.length}`);const s=On(t,2.5);for(let n=0;n<64;n++)e.assertClose(t[n],n*2.5,.01,s||`cell ${n} with scale 2.5`)}},{name:"the same kernel re-launches with <code>0.5</code> — no rebuild needed",run:async e=>{const t=e.kernel(.5),s=On(t,.5);for(let n=0;n<64;n++)e.assertClose(t[n],n*.5,.01,s||`cell ${n} with scale 0.5`)}}],privateTests:[{name:"private test #1",run:async e=>{const s=e.kernel(-2.25);e.assert(s.length===64,"expected 64 output values");const n=On(s,-2.25);for(let i=0;i<64;i++)e.assertClose(s[i],i*-2.25,.01,n||`cell ${i} with scale ${-2.25}`)}}]}]},vl=Object.freeze({__proto__:null,default:wl});const $t=`<div class="layout-note">
  <b>Array layout in gpu.js</b>
  <p>Image data comes in row-major: <code>image[y][x]</code> is the pixel in row <em>y</em>,
    column <em>x</em>, and each pixel is an <code>[r, g, b, a]</code> array with channels from
    0 to 1. Mind the inversion that catches everyone — sizes are given width-first
    (<code>output: [width, height]</code>), but indexing runs row-first, so this thread's own
    pixel is <code>image[this.thread.y][this.thread.x]</code>. Swap those two and you read the
    transpose of your image. Three-dimensional data follows the same rule:
    <code>output: [w, h, d]</code> is indexed <code>[z][y][x]</code>.</p>
</div>`;function Ka(e){let t=e>>>0;return function(){t=t+1831565813>>>0;let n=t;return n=Math.imul(n^n>>>15,n|1),n^=n+Math.imul(n^n>>>7,n|61),((n^n>>>14)>>>0)/4294967296}}function _s(e){return e<0?0:e>1?1:e}function Kt(e){return Math.round(_s(e)*255)/255}function Bt(e){return[Kt(e[0]),Kt(e[1]),Kt(e[2]),Kt(e[3]===void 0?1:e[3])]}function us(e){const t=e.length,s=e[0].length,n=new Uint8ClampedArray(s*t*4);let i=0;for(let f=t-1;f>=0;f--){const m=e[f];for(let A=0;A<s;A++){const N=m[A];n[i++]=Math.round(_s(N[0])*255),n[i++]=Math.round(_s(N[1])*255),n[i++]=Math.round(_s(N[2])*255),n[i++]=Math.round(_s(N[3]===void 0?1:N[3])*255)}}const a=new ImageData(n,s,t);return Object.defineProperties(a,{plain:{value:e,enumerable:!1},at:{value:(f,m)=>e[m][f],enumerable:!1}}),a}function kl(e){const t=Ka(1735423278^e*2654435761),s=new Array(e);for(let n=0;n<e;n++){const i=new Array(e),a=n/e;for(let f=0;f<e;f++){const m=f/e;i[f]=[Kt(.2+.55*m+.25*t()),Kt(.2+.55*a+.25*t()),Kt(.15+.6*Math.abs(Math.sin(3.1*(m+a)))+.25*t()),1]}s[n]=i}return us(s)}function Tl(e){const t=[],s=[e];for(;s.length;){const n=s.pop();if(Array.isArray(n)||ArrayBuffer.isView(n))for(let i=n.length-1;i>=0;i--)s.push(n[i]);else t.push(n)}return t}function Na(e,t){if(!e)throw new Error(t||"assertion failed")}function Ba(e,t,s=1e-4,n){const i=n?`${n} — `:"";if(typeof e!="number"||Number.isNaN(e))throw new Error(`${i}expected a number close to ${t}, got ${e}`);if(Math.abs(e-t)>s)throw new Error(`${i}expected ${t} ± ${s}, got ${e}`)}const kr={seededRandom:Ka,makeTestImage:kl,flatten:Tl,assert:Na,assertClose:Ba};function Je(){const e=new Date,t=(s,n)=>String(s).padStart(n,"0");return`${t(e.getHours(),2)}:${t(e.getMinutes(),2)}:${t(e.getSeconds(),2)}.${t(e.getMilliseconds(),3)}`}function Tr(e,t=0){if(e===null)return"null";if(e===void 0)return"undefined";const s=typeof e;if(s==="string")return t===0?e:JSON.stringify(e);if(s==="number"||s==="boolean"||s==="bigint")return String(e);if(s==="function")return`ƒ ${e.name||"(anonymous)"}`;if(e instanceof Error)return`${e.name||"Error"}: ${e.message}`;if(ArrayBuffer.isView(e)){const n=Array.from(e.slice(0,8),a=>Tr(a,t+1)),i=e.length>8?", …":"";return`${e.constructor.name}(${e.length}) [${n.join(", ")}${i}]`}if(typeof ImageData<"u"&&e instanceof ImageData)return`ImageData(${e.width}×${e.height})`;if(Array.isArray(e)){if(t>=2)return`Array(${e.length})`;const n=e.slice(0,8).map(a=>Tr(a,t+1)),i=e.length>8?", …":"";return`[${n.join(", ")}${i}]`}if(typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement)return"HTMLCanvasElement";if(typeof OffscreenCanvas<"u"&&e instanceof OffscreenCanvas)return`OffscreenCanvas(${e.width}×${e.height})`;try{const n=JSON.stringify(e);return n&&n.length>200?`${n.slice(0,200)}…`:n||String(e)}catch{try{return String(e)}catch{return"[unprintable object]"}}}function mn(e){try{const t=e&&e.message;return String(t||e)}catch{return"unprintable error"}}function Sl(e){try{if(!e)return null;const t=e.width,s=e.height;if(!t||!s)return null;if(typeof document<"u"){const n=document.createElement("canvas");return n.width=t,n.height=s,n.getContext("2d").drawImage(e,0,0),{url:n.toDataURL(),w:t,h:s}}if(typeof OffscreenCanvas<"u"){const n=new OffscreenCanvas(t,s);return n.getContext("2d").drawImage(e,0,0),{bitmap:n.transferToImageBitmap(),w:t,h:s}}return null}catch{return null}}function _l(e){if(!e||!e.width)return null;let t;if(typeof document<"u")t=document.createElement("canvas"),t.width=e.width,t.height=e.height;else if(typeof OffscreenCanvas<"u")t=new OffscreenCanvas(e.width,e.height);else return null;const s=t.getContext("2d");return s.drawImage(e,0,0),s.getImageData(0,0,t.width,t.height).data}const as=[.299,.587,.114];function pt(e){return as[0]*e[0]+as[1]*e[1]+as[2]*e[2]}function Rn(e,t=4201){const s=e.seededRandom(t),n=new Array(64);for(let i=0;i<64;i++)n[i]=Math.round(s()*1e3)/100;return n}function Cl(e,t,s,n){return[t[n][0],t[n][s]].some(f=>Math.abs(e-pt(f))<=2/255)?"that value is the luminance of the transposed pixel — looks like this.thread.x and this.thread.y are swapped. Rows come first: image[this.thread.y][this.thread.x].":null}function Fn(e,t,s,n,i){return Math.abs(e-t)<=s?`that is the value for cell [${i}][${n}] — this.thread.x and this.thread.y are swapped. Rows come first: image[this.thread.y][this.thread.x]`:null}function Ui(e,t){const s=new Array(e).fill(Bt(t));return us(new Array(e).fill(s))}function Hs(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function ja(e,t,s,n,i){const a=i.filter(([f])=>{let m=!1;for(let A=0;A<e;A++){if(!(Math.abs(t(A)-f(A))<=n))return!1;Math.abs(s(A)-f(A))>n&&(m=!0)}return m}).map(f=>f[1]);return a.length&&a.every(f=>f===a[0])?a[0]:null}function El(e,t,s,n,i){const a=i.filter(([f])=>{let m=!1;for(let A=0;A<e;A++)for(let N=0;N<e;N++){const H=f(A,N);if(!(t[A]&&Math.abs(t[A][N]-H)<=n))return!1;Math.abs(s(A,N)-H)>n&&(m=!0)}return m}).map(f=>f[1]);return a.length&&a.every(f=>f===a[0])?a[0]:null}function Vi(e,t){return ja(64,s=>e[s],s=>t[s]*2,.001,[[s=>t[s],"every cell is the element itself — the doubling never happened"],[s=>2*s,"every cell is twice the thread index, not twice the element — index the array with it: data[this.thread.x]"]])}function Ki(e){const t="a + 1 is missing — this.thread.x and this.thread.y both start at 0, so cell [y][x] holds (x + 1) * (y + 1)";return El(16,e,(s,n)=>(n+1)*(s+1),.001,[[(s,n)=>n*s,t],[(s,n)=>(n+1)*s,t],[(s,n)=>n*(s+1),t],[(s,n)=>n+1+(s+1),"the coordinates were added, not multiplied — the cell is (x + 1) * (y + 1)"]])}function Il(e){return ja(128,t=>e[t],t=>t*t,.01,[[t=>t,"you returned the thread index itself, not its square — every cell is exactly this.thread.x"],[t=>2*t,"every cell is twice the index, not the index squared — x * x, not x * 2"]])}function Ni(e){return[[pt(e),"that is the weighted luminance — this map wants the plain average (r + g + b) / 3"],[e[0]+e[1]+e[2],"the three channels were summed but never divided by 3"],[(e[0]+e[1]+e[2]+e[3])/4,"alpha crept into the average — only r, g and b belong in it"]]}function Gn(e){return[[(e[0]+e[1]+e[2])/3,"that is the plain channel average — luminance weights the channels 0.299 R + 0.587 G + 0.114 B"],[as[2]*e[0]+as[1]*e[1]+as[0]*e[2],"the weights are in the wrong order — 0.299 belongs on red and 0.114 on blue"]]}function Ml(e,t,s,n){const i=.00784313725490196,a=t.filter(f=>Math.abs(e-f[0])<=i&&Math.abs(s-f[0])>i&&Math.abs(n-f[0])>i).map(f=>f[1]);return a.length&&a.every(f=>f===a[0])?a[0]:null}function $l(e,t,s){return e>=254/255&&Math.max(t,s)<.9?"that pixel is clamped to full white — this.color() takes 0–1 channels and the image already is 0–1, so scaling by 255 saturates everything":null}var Al={id:"1-2",track:1,title:"Data In, Data Out",blurb:"Feeding arrays and images into kernels, shaping 1D/2D/3D output, and reading results back.",tasks:[{slug:"pass-an-array",title:"Pass an Array In",intro:`<p>Kernels don't reach out and grab data — data is <strong>handed to them</strong>
        as arguments, and every thread sees the same arguments. What differs between threads is
        exactly one thing: <code>this.thread.x</code>, the index of the output cell this thread owns.</p>
        <p>Here <code>data</code> is a 64-number array. The kernel below runs 64 times — once per
        output cell — and each run should pick out <em>its own</em> element.</p>`,goal:`<strong>Goal:</strong> make the kernel return <strong>double</strong> the element of
        <code>data</code> that belongs to this thread.`,requirements:["Pass <code>data</code> into the kernel as an argument (already wired up)","Index it with <code>this.thread.x</code> — no loops over the array","Return the element multiplied by <code>2</code>"],hints:[{title:"Hint 1 — which element is mine?",body:`<p>With <code>output: [64]</code> there are 64 threads, numbered
            <code>this.thread.x</code> = 0…63. Thread 7 should read <code>data[7]</code>.</p>`},{title:"Hint 2 — the one-liner",body:`<p>The whole kernel body is a single statement:
            <code>return data[this.thread.x] * 2;</code></p>`}],transfer:`Arguments-in, index-by-thread-id is the universal GPGPU calling convention:
        CUDA kernels get device pointers plus <code>threadIdx</code>, WebGPU compute shaders get
        bound buffers plus <code>global_invocation_id</code>. Same shape, different spelling.`,starterCode:`// A kernel runs once per output cell — 64 cells here, 64 threads.
const gpu = new GPU({ mode });

const double = gpu.createKernel(function (data) {
  // TODO: return double the value that belongs to THIS thread.
  // Which element is yours? this.thread.x knows.
  return 0;
}, { output: [64] });

const result = double(data);
console.log(result);
`,solutionCode:`// A kernel runs once per output cell — 64 cells here, 64 threads.
const gpu = new GPU({ mode });

const double = gpu.createKernel(function (data) {
  return data[this.thread.x] * 2;
}, { output: [64] });

const result = double(data);
console.log(result);
`,inputs:e=>({data:Rn(e)}),publicTests:[{name:"kernel returns 64 values — one per thread",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(Rn(e.utils));e.assert(t&&t.length===64,`expected 64 output values, got ${t&&t.length}`)}},{name:"every element is doubled: <code>data[i] * 2</code>",run:async e=>{const t=new Array(64);for(let i=0;i<64;i++)t[i]=i*1.5-10;const s=e.kernel(t),n=Vi(s,t);for(let i=0;i<64;i++)e.assertClose(s[i],t[i]*2,.001,n||`element ${i}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=Rn(e.utils,777),s=e.kernel(t);e.assert(s.length===64,"expected 64 output values");const n=Vi(s,t);for(let i=0;i<64;i++)e.assertClose(s[i],t[i]*2,.001,n||`element ${i}`)}}]},{slug:"output-shapes",title:"Shape the Output: 2D",intro:`<p><code>output</code> is not just a size — it's a <strong>shape</strong>.
        <code>output: [16]</code> launches a line of 16 threads; <code>output: [16, 16]</code>
        launches a 16×16 <em>grid</em> of 256 threads, and each one gets two coordinates:
        <code>this.thread.x</code> (column) and <code>this.thread.y</code> (row).</p>
        <p>The result comes back with the same shape: a 2D kernel returns an array of rows,
        indexed <code>result[y][x]</code>.</p>`,goal:`<strong>Goal:</strong> turn the kernel into a 16×16 grid that computes a
        multiplication table — cell <code>[y][x]</code> holds <code>(x + 1) * (y + 1)</code>.`,requirements:["Change <code>output</code> to a 16×16 grid: <code>[16, 16]</code>","Use both <code>this.thread.x</code> and <code>this.thread.y</code>","Return <code>(x + 1) * (y + 1)</code> so row 1 counts 1…16, row 2 counts 2…32, …"],hints:[{title:"Hint 1 — reading the shape",body:`<p><code>output: [width, height]</code> — x runs over <code>width</code>,
            y over <code>height</code>. The returned value lands in <code>result[y][x]</code>.</p>`}],transfer:`2D and 3D launch grids are first-class everywhere: CUDA's <code>dim3</code>
        grid/block sizes, WebGPU's <code>workgroup_size</code> and dispatch dimensions. Choosing
        the launch shape to match the output shape is the same design move on every platform.`,starterCode:`// output: [width, height] — gpu.js hands you a whole grid of threads.
const gpu = new GPU({ mode });

const table = gpu.createKernel(function () {
  // TODO: use BOTH this.thread.x and this.thread.y
  // and return (x + 1) * (y + 1).
  return this.thread.x + 1;
}, {
  // TODO: make this a 16×16 grid, not a 16-cell line
  output: [16],
});

const result = table();
console.log('rows:', result.length);
console.log('row 0:', result[0]);
`,solutionCode:`// output: [width, height] — gpu.js hands you a whole grid of threads.
const gpu = new GPU({ mode });

const table = gpu.createKernel(function () {
  return (this.thread.x + 1) * (this.thread.y + 1);
}, {
  output: [16, 16],
});

const result = table();
console.log('rows:', result.length);
console.log('row 0:', result[0]);
`,publicTests:[{name:"result is a 16×16 grid — 16 rows of 16 values",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===16,`expected 16 rows, got ${t&&t.length}`),e.assert(t[0]&&typeof t[0]!="number"&&t[0].length===16,"each row should hold 16 values — is your output still 1D?")}},{name:"cell [y][x] equals <code>(x + 1) * (y + 1)</code>",run:async e=>{const t=e.kernel(),s=[[0,0,1],[2,3,12],[7,0,8],[0,7,8],[15,15,256]],n=Ki(t);for(const[i,a,f]of s)e.assertClose(t[i][a],f,.001,n||`cell [${i}][${a}]`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(),s=Ki(t);for(let n=0;n<16;n++)for(let i=0;i<16;i++)e.assertClose(t[n][i],(i+1)*(n+1),.001,s||`cell [${n}][${i}]`)}}]},{slug:"grayscale",title:"Grayscale, the GPU way",intro:`<p>On the CPU you'd loop over 262,144 pixels one by one. On the GPU, every pixel gets
        <strong>its own thread</strong> — the kernel body runs once per pixel, all at the same time.</p>
        ${$t}`,goal:`<strong>Goal:</strong> write a graphical kernel that converts <code>image</code> to
        grayscale using perceptual luminance.`,requirements:["Create the kernel with <code>graphical: true</code> and <code>output: [512, 512]</code>","Read the pixel for <em>this</em> thread from <code>image</code>","Weight the channels <code>0.299 R + 0.587 G + 0.114 B</code>","Write the result with <code>this.color()</code>"],hints:[{title:"Hint 1 — which pixel is mine?",body:`<p>Inside a kernel, <code>this.thread.x</code> and <code>this.thread.y</code> tell you which
            output cell this thread owns. Use them to index into <code>image</code>.</p>`},{title:"Hint 2 — reading a pixel",body:`<p><code>image[this.thread.y][this.thread.x]</code> gives you an <code>[r, g, b, a]</code>
            array with channels in the 0–1 range.</p>`}],transfer:`This is exactly a fragment shader in WebGPU/Metal, or a 2D thread block in CUDA and
        ROCm — one thread per output element.`,starterCode:`// One thread per pixel. No loops over pixels — ever.
const gpu = new GPU({ mode });

const grayscale = gpu.createKernel(function (image) {
  // TODO: read this thread's pixel from image, weight the channels
  // 0.299 R + 0.587 G + 0.114 B, and write it with this.color()
  this.color(1, 0, 1, 1);
}, {
  output: [512, 512],
  graphical: true,
});

grayscale(inputImage);
render(grayscale.canvas);
`,solutionCode:`// One thread per pixel. No loops over pixels — ever.
const gpu = new GPU({ mode });

const grayscale = gpu.createKernel(function (image) {
  const pixel = image[this.thread.y][this.thread.x];
  const l = 0.299 * pixel[0]
          + 0.587 * pixel[1]
          + 0.114 * pixel[2];
  this.color(l, l, l, 1);
}, {
  output: [512, 512],
  graphical: true,
});

grayscale(inputImage);
render(grayscale.canvas);
`,inputs:e=>({inputImage:e.makeTestImage(512)}),publicTests:[{name:"returns a graphical canvas of size <code>512×512</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===512&&e.canvas.height===512,`expected a 512×512 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();e.assert(t.length===512*512*4,"pixel buffer should hold 512×512 RGBA values")}},{name:"pixel (0,0) matches <code>0.299r + 0.587g + 0.114b</code> within ±1/255",run:async e=>{const t=e.utils.makeTestImage(512).plain,n=e.getPixels()[0]/255,i=pt(t[0][0]),a=pt(t[511][0]),f=Math.abs(n-i)<=2/255||Math.abs(n-a)<=2/255,m=Ml(n,[...Gn(t[0][0]),...Gn(t[511][0])],i,a);e.assert(f,Cl(n,t,511,0)||$l(n,i,a)||m||`corner pixel should be its luminance (got ${n.toFixed(3)}, expected ≈${i.toFixed(3)})`)}},{name:"output is monochrome — <code>r == g == b</code> for sampled pixels",run:async e=>{const t=e.getPixels();for(let s=0;s<t.length;s+=997*4){const n=t[s],i=t[s+1],a=t[s+2];e.assert(Math.abs(n-i)<=1&&Math.abs(i-a)<=1,`pixel at byte ${s} is not gray: rgb(${n}, ${i}, ${a})`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ui(512,[.2,.4,.6,1]),s=pt(t.at(0,0))*255;e.kernel(t);const n=e.getPixels();for(let i=0;i<n.length;i+=4999*4)e.assertClose(n[i],s,2,`red at byte ${i}`),e.assertClose(n[i+1],s,2,`green at byte ${i}`),e.assertClose(n[i+2],s,2,`blue at byte ${i}`)}},{name:"private test #2",run:async e=>{const t=e.utils.makeTestImage(512);e.kernel(t);const s=e.getPixels();let n=0;for(let a=0;a<s.length;a+=4)n+=s[a];n/=s.length/4*255;let i=0;for(let a=0;a<512;a++)for(let f=0;f<512;f++)i+=pt(t.plain[a][f]);i/=512*512,e.assertClose(n,i,1.5/255,"mean luminance")}}]},{slug:"read-it-back",title:"Read the Results Back",intro:`<p>A kernel's return value doesn't stay on the GPU — invoking the kernel hands you the
        finished result as an ordinary (typed) array. From there it's plain JavaScript: loop over it,
        sum it, feed it to a chart, whatever you like.</p>
        <p>This round trip is the heartbeat of GPGPU: <strong>upload → compute in parallel →
        read back</strong>. Here the parallel part computes 128 squares; the read-back part totals them.</p>`,goal:`<strong>Goal:</strong> make the kernel return <code>x²</code> for each thread, then sum
        the returned array in plain JavaScript and log the total with <code>console.log</code>.`,requirements:["Kernel returns <code>this.thread.x * this.thread.x</code> for all 128 threads","Sum the returned <code>result</code> array in ordinary JavaScript — outside the kernel","Log the total (it should come to <code>690880</code>)"],hints:[{title:"Hint 1 — what comes back?",body:`<p>With <code>output: [128]</code> the call <code>squares()</code> returns a
            <code>Float32Array</code> of 128 numbers. It's indexable and loopable like any array.</p>`},{title:"Hint 2 — the sum",body:`<p>A plain <code>for</code> loop after the kernel call:</p>
<pre><code>let total = 0;
for (let i = 0; i &lt; result.length; i++) {
  total += result[i];
}</code></pre>`}],transfer:`Read-back is never free: CUDA's <code>cudaMemcpy</code> device→host and WebGPU's
        <code>mapAsync</code> staging buffers exist for exactly this step — and minimizing round trips
        is rule one of real GPU performance (module 1.4 makes a whole meal of it).`,starterCode:`// Kernel output comes back to JavaScript as a typed array.
const gpu = new GPU({ mode });

const squares = gpu.createKernel(function () {
  // TODO: return this thread's index, squared
  return this.thread.x;
}, { output: [128] });

const result = squares();
console.log(result);

// TODO: total up \`result\` in plain JavaScript, then:
// console.log('sum of squares:', total);
`,solutionCode:`// Kernel output comes back to JavaScript as a typed array.
const gpu = new GPU({ mode });

const squares = gpu.createKernel(function () {
  return this.thread.x * this.thread.x;
}, { output: [128] });

const result = squares();
console.log(result);

let total = 0;
for (let i = 0; i < result.length; i++) total += result[i];
console.log('sum of squares:', total);
`,publicTests:[{name:"kernel returns <code>x²</code> for each of 128 threads",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===128,`expected 128 values, got ${t&&t.length}`);const s=Il(t);for(let n=0;n<128;n++)e.assertClose(t[n],n*n,.01,s||`element ${n}`)}},{name:"the total <code>690880</code> is computed and logged",run:async e=>{const t=e.logs.some(s=>s.type==="log"&&s.text&&s.text.includes("690880"));e.assert(t,"log the sum with console.log — expected to see 690880 in the console output")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel();let s=0;for(let i=0;i<t.length;i++)s+=t[i];const n=Hs(s,690880,1,[[8128,"that total is the sum of the indices themselves — the kernel is returning this.thread.x, not its square"],[2*8128,"that total is the sum of twice each index — the kernel is doubling where it should square"]]);e.assertClose(s,690880,1,n||"sum of the kernel output")}}]},{slug:"image-as-data",title:"Images Are Just Arrays",intro:`<p>Task 3 painted pixels. But an image doesn't have to <em>stay</em> an image: in this
        course an image is a nested array — <code>photo[y][x]</code> is an <code>[r, g, b, a]</code>
        pixel with channels 0–1 — and a kernel can read it like any other array argument.</p>
        <p>Drop <code>graphical: true</code>, and the same per-pixel indexing produces
        <strong>numbers</strong> instead of colors: a measurement per pixel, ready for JavaScript.</p>
        ${$t}`,goal:`<strong>Goal:</strong> compute a 64×64 brightness map of <code>photo</code> — each cell
        the average of that pixel's red, green and blue channels.`,requirements:["Keep the kernel numeric — no <code>graphical: true</code>, output <code>[64, 64]</code>","Read this thread's pixel: <code>photo[this.thread.y][this.thread.x]</code>","Return <code>(r + g + b) / 3</code>"],hints:[{title:"Hint 1 — same indexing as task 3",body:`<p>The pixel lookup is identical to the grayscale task — only the ending changes:
            <code>return</code> a number instead of calling <code>this.color()</code>.</p>`},{title:"Hint 2 — the average",body:`<pre><code>const pixel = photo[this.thread.y][this.thread.x];
return (pixel[0] + pixel[1] + pixel[2]) / 3;</code></pre>`}],transfer:`Treating an image as a data grid is how real pipelines work: computer-vision
        pre-processing, depth-map filtering, scientific imaging. In CUDA/WebGPU this is a compute
        pass sampling a texture and writing to a plain buffer.`,starterCode:`// An image is a nested array: photo[y][x] → [r, g, b, a], all 0–1.
const gpu = new GPU({ mode });

const brightness = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  // TODO: return the average of the red, green and blue channels
  return pixel[0];
}, { output: [64, 64] });

const map = brightness(photo);
console.log('top-left brightness:', map[0][0]);
`,solutionCode:`// An image is a nested array: photo[y][x] → [r, g, b, a], all 0–1.
const gpu = new GPU({ mode });

const brightness = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return (pixel[0] + pixel[1] + pixel[2]) / 3;
}, { output: [64, 64] });

const map = brightness(photo);
console.log('top-left brightness:', map[0][0]);
`,inputs:e=>({photo:e.makeTestImage(64)}),publicTests:[{name:"produces a 64×64 brightness map",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(e.utils.makeTestImage(64));e.assert(t&&t.length===64,`expected 64 rows, got ${t&&t.length}`),e.assert(t[0]&&t[0].length===64,"each row should hold 64 values")}},{name:"each cell averages the channels — <code>(r + g + b) / 3</code>",run:async e=>{const t=e.utils.makeTestImage(64),s=e.kernel(t),n=t.plain,i=[[0,0],[10,3],[31,40],[63,63]];for(const[a,f]of i){const m=n[a][f],A=n[f][a],N=(m[0]+m[1]+m[2])/3,H=Fn(s[a][f],(A[0]+A[1]+A[2])/3,.002,a,f)||Hs(s[a][f],N,.002,Ni(m));e.assertClose(s[a][f],N,.002,H||`cell [${a}][${f}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.utils.makeTestImage(64),s=e.kernel(t),n=t.plain;for(let i=0;i<64;i++)for(let a=0;a<64;a++){const f=n[i][a],m=n[a][i],A=(f[0]+f[1]+f[2])/3,N=Fn(s[i][a],(m[0]+m[1]+m[2])/3,.002,i,a)||Hs(s[i][a],A,.002,Ni(f));e.assertClose(s[i][a],A,.002,N||`cell [${i}][${a}]`)}}}]},{slug:"two-kernels",title:"Put It Together: Two Kernels",intro:`<p>Everything from this module in one pipeline. Kernel one reads the
        <code>photo</code> and produces a 64×64 <strong>luminance map</strong> — pure numbers.
        That result comes back to JavaScript, and you pass it straight into kernel two, a
        <strong>graphical</strong> kernel that paints the map as a grayscale picture.</p>
        <p>Array in → numbers out → array in again → pixels out. Data flowing <em>through</em>
        kernels is the whole game (and module 1.4 shows how to keep that flow on the GPU).</p>
        ${$t}`,goal:`<strong>Goal:</strong> finish both kernels — <code>luminance</code> returns
        <code>0.299r + 0.587g + 0.114b</code> per pixel, and <code>paint</code> renders the map
        as gray pixels with <code>this.color()</code>.`,requirements:["Numeric kernel: read <code>photo[this.thread.y][this.thread.x]</code>, return the weighted luminance","Graphical kernel: read this thread's value from <code>map</code>","Paint it gray: <code>this.color(l, l, l, 1)</code>","Feed the first kernel's result into the second (already wired up)"],hints:[{title:"Hint 1 — the luminance pass",body:`<p>Same lookup as before, but return a number:</p>
<pre><code>return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];</code></pre>`},{title:"Hint 2 — the paint pass",body:`<p><code>map</code> is a plain 2D array of numbers, so</p>
<pre><code>const l = map[this.thread.y][this.thread.x];
this.color(l, l, l, 1);</code></pre>`}],transfer:`Multi-pass pipelines are the backbone of GPU work: render passes in graphics,
        kernel launch chains in CUDA, encoder passes in WebGPU. The handoff you just did through
        JavaScript is the slow version — pipelines (module 1.4) keep it on-device.`,starterCode:`// Kernel 1 turns the photo into numbers. Kernel 2 turns numbers into pixels.
const gpu = new GPU({ mode });

const luminance = gpu.createKernel(function (photo) {
  // TODO: return perceptual luminance — 0.299 R + 0.587 G + 0.114 B
  return 0;
}, { output: [64, 64] });

const paint = gpu.createKernel(function (map) {
  // TODO: read this thread's value from map and paint it gray
  this.color(1, 0, 1, 1);
}, { output: [64, 64], graphical: true });

const map = luminance(photo);
paint(map);
render(paint.canvas);
`,solutionCode:`// Kernel 1 turns the photo into numbers. Kernel 2 turns numbers into pixels.
const gpu = new GPU({ mode });

const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64] });

const paint = gpu.createKernel(function (map) {
  const l = map[this.thread.y][this.thread.x];
  this.color(l, l, l, 1);
}, { output: [64, 64], graphical: true });

const map = luminance(photo);
paint(map);
render(paint.canvas);
`,inputs:e=>({photo:e.makeTestImage(64)}),publicTests:[{name:"two kernels: a numeric pass and a graphical pass",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=e.kernels.find(n=>n.kernel&&n.kernel.graphical),s=e.kernels.find(n=>n.kernel&&!n.kernel.graphical);e.assert(s,"no numeric (non-graphical) kernel found"),e.assert(t,"no graphical kernel found")}},{name:"luminance pass: cell [y][x] = <code>0.299r + 0.587g + 0.114b</code>",run:async e=>{const t=e.kernels.find(f=>f.kernel&&!f.kernel.graphical);e.assert(t,"no numeric kernel found");const s=e.utils.makeTestImage(64),n=t(s),i=s.plain,a=[[0,0],[5,50],[33,12],[63,63]];for(const[f,m]of a){const A=Fn(n[f][m],pt(i[m][f]),.002,f,m)||Hs(n[f][m],pt(i[f][m]),.002,Gn(i[f][m]));e.assertClose(n[f][m],pt(i[f][m]),.002,A||`cell [${f}][${m}]`)}}},{name:"painted canvas is monochrome and <code>64×64</code>",run:async e=>{e.assert(e.canvas,"no canvas — did you call render(paint.canvas)?"),e.assert(e.canvas.width===64&&e.canvas.height===64,`expected a 64×64 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();for(let s=0;s<t.length;s+=244){const n=t[s],i=t[s+1],a=t[s+2];e.assert(Math.abs(n-i)<=1&&Math.abs(i-a)<=1,`pixel at byte ${s} is not gray: rgb(${n}, ${i}, ${a})`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(m=>m.kernel&&!m.kernel.graphical),s=e.kernels.find(m=>m.kernel&&m.kernel.graphical);e.assert(t&&s,"expected a numeric and a graphical kernel");const n=Ui(64,[.6,.2,.4,1]),i=pt(n.at(0,0))*255,a=t(n);s(a);const f=s.getPixels();for(let m=0;m<f.length;m+=596)e.assertClose(f[m],i,2,`red at byte ${m}`),e.assertClose(f[m+1],i,2,`green at byte ${m}`),e.assertClose(f[m+2],i,2,`blue at byte ${m}`)}}]}]},Dl=Object.freeze({__proto__:null,default:Al});function Ln(e,t=1303){const s=e.seededRandom(t),n=new Array(64);for(let i=0;i<64;i++)n[i]=Math.round((s()*50-10)*100)/100;return n}function We(e,t,s=2718){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++)i[a]=Math.round((1+n()*8)*100)/100;return i}function xs(e,t,s=917){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++){const f=new Array(t);for(let m=0;m<t;m++)f[m]=Math.round(n()*1e3)/1e3;i[a]=f}return i}function ls(e,t){return Math.max(0,Math.min(t-1,e))}function Un(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){let i=0;for(let a=-2;a<=2;a++)i+=e[ls(n+a,t)];s[n]=i/5}return s}function Is(e){const t=e.length;return e.map(s=>{const n=new Array(t);for(let i=0;i<t;i++)n[i]=(s[ls(i-1,t)]+s[i]+s[ls(i+1,t)])/3;return n})}function Ms(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){const i=new Array(t);for(let a=0;a<t;a++)i[a]=(e[ls(n-1,t)][a]+e[n][a]+e[ls(n+1,t)][a])/3;s[n]=i}return s}function Xs(e,t){const s=e.kernels[0](t);return e.kernels[1](s)}function ze(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Bi(e){return[[e*9/5,"the + 32 offset is missing — °F = °C × 9/5 + 32"],[e*5/9+32,"that is the °F → °C ratio — this direction multiplies by 9/5, not 5/9"],[(e+32)*9/5,"you added 32 before scaling — the formula scales first, then adds"],[e,"that reading came back unconverted — the formula never ran on it"]]}function Ys(e,t){const s=[[e[t],"that is your own element — a gather reads the mirrored index, data[n − 1 − this.thread.x]"]],n=e[e.length-t];return Number.isFinite(n)&&s.push([n,"that is data[n − this.thread.x] — the last valid index is n − 1, so the mirror of i is n − 1 − i"]),s}function Vn(e,t){const s=e.length;return[[e[(t+1)%s],"that value came from your right — rotating right means pulling from the left, index this.thread.x − 1"],[e[t],"that is your own element — the shift has to be expressed as a read from the neighbor"]]}function ji(e){return Number.isFinite(e)&&e!==0?null:"thread 0 read index −1 — % keeps its left operand's sign, so add n first: (this.thread.x − 1 + n) % n"}function Kn(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){let i=0;for(let a=0;a<5;a++)i+=e[ls(n+a,t)];s[n]=i/5}return s}function Js(e,t,s,n){return[[e[n],"every tap read your own element — the offset d − 2 never reached the index, so you averaged five copies of yourself"],[t[n],"the window is shifted right — the offset d − 2 is what centers the five taps on this.thread.x"],[5*s[n],"that is the window sum — a mean divides by 5"]]}function Nn(e){const t=Is(e),s=Ms(e);return[[Is(t),"both passes blurred along x — the second one has to walk the column: clamp this.thread.y and read grid[j][this.thread.x]"],[Ms(s),"both passes blurred along y — the first one has to walk the row: clamp this.thread.x and read grid[this.thread.y][j]"],[s,"the x pass is a passthrough — only the y blur reached this cell"],[t,"the y pass is a passthrough — only the x blur reached this cell"]]}function Bn(e,t,s,n,i,a){if(!Number.isFinite(e))return"that cell read past the edge of the grid — clamp the index with Math.max(0, Math.min(n − 1, …))";const f=s.map(A=>[A[0][i][a],A[1]]),m="a pass returned the sum of its three taps without dividing by 3";return f.push([3*t[i][a],m],[9*t[i][a],m]),ze(e,t[i][a],n,f)}var Pl={id:"1-3",track:1,title:"Thinking in Parallel",blurb:"Map and gather patterns, why kernels write only their own cell, and how to design around it.",tasks:[{slug:"map-pattern",title:"Map: One Thread, One Value",intro:`<p>Nearly every GPU program you'll ever write is built from a handful of patterns,
        and the first one has a name: <strong>map</strong>. Each output cell is a pure function of
        the input cell <em>at the same index</em> — no neighbors, no shared state, no
        "first do cell 3, then cell 4". That independence is exactly what lets the GPU run all
        the cells at once.</p>
        <p>Here <code>celsius</code> holds 64 temperature readings. Converting them to Fahrenheit
        is a textbook map: reading 7 becomes output 7, and nothing else matters to thread 7.</p>`,goal:`<strong>Goal:</strong> map every Celsius reading to Fahrenheit —
        <code>°F = °C × 9/5 + 32</code> — one thread per reading.`,requirements:["Read only <em>this thread's</em> element: <code>celsius[this.thread.x]</code>","Apply the formula <code>c * 9 / 5 + 32</code>","No loops over the array — the grid of threads <em>is</em> the loop"],hints:[{title:"Hint 1 — the shape of a map",body:`<p>A map kernel touches exactly one input cell and one output cell, both at
            index <code>this.thread.x</code>. If you find yourself reading any other index,
            it's not a map any more.</p>`},{title:"Hint 2 — the one-liner",body:"<pre><code>return celsius[this.thread.x] * 9 / 5 + 32;</code></pre>"}],transfer:`Map is the hello-world of every GPU API: a CUDA grid where each thread transforms
        one element of a device array, a WebGPU compute shader dispatched once per buffer entry, a
        Metal compute encoder doing the same. If a problem is a pure map, it parallelizes for free.`,starterCode:`// Map: output[i] depends ONLY on input[i]. One thread per reading.
const gpu = new GPU({ mode });

const toFahrenheit = gpu.createKernel(function (celsius) {
  // TODO: convert THIS thread's reading — °F = °C × 9/5 + 32
  return celsius[this.thread.x];
}, { output: [64] });

const result = toFahrenheit(celsius);
console.log('first reading:', celsius[0], '°C →', result[0], '°F');
`,solutionCode:`// Map: output[i] depends ONLY on input[i]. One thread per reading.
const gpu = new GPU({ mode });

const toFahrenheit = gpu.createKernel(function (celsius) {
  return celsius[this.thread.x] * 9 / 5 + 32;
}, { output: [64] });

const result = toFahrenheit(celsius);
console.log('first reading:', celsius[0], '°C →', result[0], '°F');
`,inputs:e=>({celsius:Ln(e)}),publicTests:[{name:"converts all 64 readings — one output per thread",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(Ln(e.utils));e.assert(t&&t.length===64,`expected 64 output values, got ${t&&t.length}`)}},{name:"each cell is <code>c × 9/5 + 32</code> — 0 °C → 32 °F, 100 °C → 212 °F",run:async e=>{const t=new Array(64);for(let n=0;n<64;n++)t[n]=n*2-20;const s=e.kernel(t);for(let n=0;n<64;n++){const i=t[n]*9/5+32,a=ze(s[n],i,.01,Bi(t[n]));e.assertClose(s[n],i,.01,a||`reading ${n} (${t[n]} °C)`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ln(e.utils,999),s=e.kernel(t);e.assert(s.length===64,"expected 64 output values");for(let n=0;n<64;n++){const i=t[n]*9/5+32,a=ze(s[n],i,.01,Bi(t[n]));e.assertClose(s[n],i,.01,a||`reading ${n}`)}}}]},{slug:"gather-pattern",title:"Gather: Read Anywhere",intro:`<p>A map reads its own cell. A <strong>gather</strong> reads <em>any</em> cell —
        the thread computes <strong>where to read from</strong> using its own index. Reads are
        random-access and cheap; it's only <em>writes</em> that are pinned to your own cell
        (the next task is all about that).</p>
        <p>The cleanest possible gather: reverse an array. Thread 0 pulls the last element,
        thread 63 pulls the first — every thread reads exactly one cell, just not its own.
        The array length is wired in as <code>this.constants.n</code>, so the kernel doesn't
        hardcode 64.</p>`,goal:`<strong>Goal:</strong> make the kernel return the element from the
        <em>mirrored</em> position, so the output is <code>data</code> reversed.`,requirements:["Compute the read index from <code>this.thread.x</code> and <code>this.constants.n</code>","Thread <code>i</code> reads <code>data[n − 1 − i]</code>","No loops, no temporary arrays — one read per thread"],hints:[{title:"Hint 1 — mirror arithmetic",body:`<p>The mirror of index <code>i</code> in an <code>n</code>-element array is
            <code>n − 1 − i</code>: index 0 ↔ index 63, index 1 ↔ index 62, …</p>`},{title:"Hint 2 — the one-liner",body:"<pre><code>return data[this.constants.n - 1 - this.thread.x];</code></pre>"}],transfer:`Gather is why GPUs have texture units: shaders sample textures at arbitrary
        coordinates, CUDA routes scattered reads through <code>__ldg</code> and texture memory,
        WebGPU compute shaders index storage buffers freely. Hardware is built to make "read from
        anywhere" fast.`,starterCode:`// A gather kernel computes WHERE to read from its own thread id.
const gpu = new GPU({ mode });

const reverse = gpu.createKernel(function (data) {
  // TODO: read the element from the OTHER end of the array.
  // The array length is available as this.constants.n.
  return data[this.thread.x];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = reverse(data);
console.log('first:', result[0], '(should be the old last:', data[63] + ')');
`,solutionCode:`// A gather kernel computes WHERE to read from its own thread id.
const gpu = new GPU({ mode });

const reverse = gpu.createKernel(function (data) {
  return data[this.constants.n - 1 - this.thread.x];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = reverse(data);
console.log('first:', result[0], '(should be the old last:', data[63] + ')');
`,inputs:e=>({data:We(e,64,1101)}),publicTests:[{name:"the ends swap places — <code>out[0] = data[63]</code>, <code>out[63] = data[0]</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(64);for(let a=0;a<64;a++)t[a]=a+1;const s=e.kernel(t);e.assert(s&&s.length===64,`expected 64 output values, got ${s&&s.length}`);const n=ze(s[0],64,.001,Ys(t,0));e.assertClose(s[0],64,.001,n||"out[0] should hold the last input value");const i=ze(s[63],1,.001,Ys(t,63));e.assertClose(s[63],1,.001,i||"out[63] should hold the first input value")}},{name:"every cell mirrors: <code>out[i] = data[63 − i]</code>",run:async e=>{const t=new Array(64);for(let n=0;n<64;n++)t[n]=n*1.5+3;const s=e.kernel(t);for(let n=0;n<64;n++){const i=ze(s[n],t[63-n],.001,Ys(t,n));e.assertClose(s[n],t[63-n],.001,i||`element ${n}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=We(e.utils,64,4242),s=e.kernel(t);e.assert(s.length===64,"expected 64 output values");for(let n=0;n<64;n++){const i=ze(s[n],t[63-n],.001,Ys(t,n));e.assertClose(s[n],t[63-n],.001,i||`element ${n}`)}}}]},{slug:"invert-the-scatter",title:"No Scatter Allowed",intro:`<p>Here's the rule that shapes gpu.js kernels (and any fragment shader): a thread
        can read anywhere but can only write <strong>one place — its own cell</strong>, via
        <code>return</code>. There is no <code>out[i + 1] = value</code> here, because 4096
        simultaneous writers into shared cells would be chaos (who wins? in what order?).</p>
        <p>So the "push my value over there" plan — a <strong>scatter</strong> — must be turned
        inside out. Don't ask <em>"where does my value go?"</em>; ask
        <em>"whose value lands in <strong>my</strong> cell?"</em> — a gather. Try it on a rotation:
        every value moves one slot to the <em>right</em>, and the last wraps around to slot 0.</p>`,goal:`<strong>Goal:</strong> rotate <code>ring</code> one slot to the right by gathering:
        each thread pulls the value that belongs in its cell.`,requirements:["No writes to other cells — express the shift purely as a read","Thread <code>i</code> pulls from index <code>i − 1</code>","Thread 0 wraps around and pulls the <em>last</em> element"],hints:[{title:"Hint 1 — invert the direction",body:`<p>If every value moves right by one, then the value in <em>my</em> cell came
            from my <em>left</em>: index <code>this.thread.x - 1</code>. The starter currently
            pulls from the right — that rotates the wrong way.</p>`},{title:"Hint 2 — wrapping without an if",body:`<p>Adding <code>n</code> before the modulo keeps the index positive:</p>
<pre><code>(this.thread.x - 1 + this.constants.n) % this.constants.n</code></pre>
<p>That turns <code>-1</code> into <code>63</code> and leaves 1…63 alone.</p>`}],transfer:`Compute APIs relax this ban: CUDA, WebGPU and ROCm threads <em>can</em> store to
        any buffer address (scatter), and neighbours in a block cooperate through workgroup
        memory. But two threads storing to the <em>same</em> address is still a data race, and
        the escape hatch — atomics like <code>atomicAdd</code> — serializes threads and costs
        dearly. That's why GPU folklore compresses this lesson into four words:
        <em>turn scatter into gather</em>.`,starterCode:`// There is no out[i + 1] = value on a GPU. Threads only fill their OWN cell.
const gpu = new GPU({ mode });

// Wanted: every value moves one slot RIGHT, the last wraps to slot 0.
// You can't push your value right — so pull the right value in.
const rotate = gpu.createKernel(function (ring) {
  // TODO: this pulls from the wrong side — it rotates LEFT. Fix the
  // gather so each thread pulls the value that belongs in its cell.
  return ring[(this.thread.x + 1) % this.constants.n];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = rotate(ring);
console.log('ring[0] was', ring[0], '— it should now sit at result[1]:', result[1]);
`,solutionCode:`// There is no out[i + 1] = value on a GPU. Threads only fill their OWN cell.
const gpu = new GPU({ mode });

// "Whose value lands in MY cell?" — the one from my left, wrapping at 0.
const rotate = gpu.createKernel(function (ring) {
  return ring[(this.thread.x - 1 + this.constants.n) % this.constants.n];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = rotate(ring);
console.log('ring[0] was', ring[0], '— it should now sit at result[1]:', result[1]);
`,inputs:e=>({ring:We(e,64,3301)}),publicTests:[{name:"values move one slot right: <code>out[i] = ring[i − 1]</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(64);for(let n=0;n<64;n++)t[n]=n*2+5;const s=e.kernel(t);e.assert(s&&s.length===64,`expected 64 output values, got ${s&&s.length}`);for(let n=1;n<64;n++){const i=ze(s[n],t[n-1],.001,Vn(t,n));e.assertClose(s[n],t[n-1],.001,i||`element ${n} should hold ring[${n-1}]`)}}},{name:"the first cell wraps around: <code>out[0] = ring[63]</code>",run:async e=>{const t=new Array(64);for(let i=0;i<64;i++)t[i]=i+10;const s=e.kernel(t),n=ji(s[0])||ze(s[0],t[63],.001,Vn(t,0));e.assertClose(s[0],t[63],.001,n||"out[0] should hold the last input value")}}],privateTests:[{name:"private test #1",run:async e=>{const t=We(e.utils,64,8088),s=e.kernel(t);e.assert(s.length===64,"expected 64 output values");for(let n=0;n<64;n++){const i=t[(n-1+64)%64],a=(n===0?ji(s[0]):null)||ze(s[n],i,.001,Vn(t,n));e.assertClose(s[n],i,.001,a||`element ${n}`)}}}]},{slug:"edges-and-clamps",title:"Life on the Edge",intro:`<p>The moment a gather reads a <em>neighbor</em>, the edges bite. Take the forward
        difference — <code>out[i] = signal[i+1] − signal[i]</code>, "how much does the signal jump
        here?". Thread 63 asks for <code>signal[64]</code>, which doesn't exist. On the CPU that
        read is <code>NaN</code>; on the GPU it's whatever the hardware feels like. Either way,
        garbage.</p>
        <p>The standard fix is <strong>clamping</strong>: pin the index inside
        <code>0 … n−1</code> with <code>Math.min</code>/<code>Math.max</code> before reading.
        The clamped edge cell reads itself, so its difference is exactly 0 — a defined,
        deliberate answer instead of an accident.</p>`,goal:`<strong>Goal:</strong> compute the forward difference with a clamped read, so the
        last cell returns exactly <code>0</code> instead of garbage.`,requirements:["Clamp the neighbor index: <code>Math.min(this.thread.x + 1, this.constants.n - 1)</code>","Interior cells still return <code>signal[i+1] − signal[i]</code>","The last cell returns exactly <code>0</code>"],hints:[{title:"Hint 1 — where the garbage comes from",body:`<p>Only thread 63 misbehaves: <code>this.thread.x + 1</code> is 64, one past the
            end. Every other thread's read is fine — the fix should change <em>only</em> that
            one thread's read.</p>`},{title:"Hint 2 — the clamped read",body:`<pre><code>const j = Math.min(this.thread.x + 1, this.constants.n - 1);
return signal[j] - signal[this.thread.x];</code></pre>
<p>For thread 63, <code>j</code> is 63, and <code>signal[63] - signal[63]</code> is 0.</p>`}],transfer:`Graphics hardware ships this as a sampler setting — <code>clamp-to-edge</code>
        address mode in WebGPU and Metal, <code>cudaAddressModeClamp</code> on CUDA texture
        objects. Reading a raw buffer instead of a texture? Then you clamp by hand, exactly like
        here.`,starterCode:`// Forward difference: out[i] = signal[i + 1] - signal[i].
const gpu = new GPU({ mode });

const delta = gpu.createKernel(function (signal) {
  // TODO: thread 63 reads signal[64] — one past the end. Clamp the
  // neighbor index so the last cell returns 0 instead of garbage.
  return signal[this.thread.x + 1] - signal[this.thread.x];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = delta(signal);
console.log('last delta (should be exactly 0):', result[63]);
`,solutionCode:`// Forward difference: out[i] = signal[i + 1] - signal[i].
const gpu = new GPU({ mode });

const delta = gpu.createKernel(function (signal) {
  const j = Math.min(this.thread.x + 1, this.constants.n - 1);
  return signal[j] - signal[this.thread.x];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = delta(signal);
console.log('last delta (should be exactly 0):', result[63]);
`,inputs:e=>({signal:We(e,64,5150)}),publicTests:[{name:"interior cells hold the jump: <code>out[i] = signal[i+1] − signal[i]</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=We(e.utils,64,5150),s=e.kernel(t);e.assert(s&&s.length===64,`expected 64 output values, got ${s&&s.length}`);for(let n=0;n<63;n++){const i=t[n+1]-t[n],a=ze(s[n],i,.001,[[-i,"the sign is flipped — a forward difference is signal[i + 1] − signal[i], next minus current"],[t[n+1],"that is the neighbor's value, not the jump — subtract your own signal[this.thread.x]"]]);e.assertClose(s[n],i,.001,a||`element ${n}`)}}},{name:"the last cell clamps to <code>0</code> — no read past the end",run:async e=>{const t=We(e.utils,64,5150),s=e.kernel(t);e.assert(Number.isFinite(s[63]),`out[63] is ${s[63]} — an out-of-bounds read`),e.assertClose(s[63],0,1e-4,"the clamped edge cell should be exactly 0")}}],privateTests:[{name:"private test #1",run:async e=>{const t=We(e.utils,64,6006),s=e.kernel(t);for(let n=0;n<63;n++){const i=t[n+1]-t[n],a=ze(s[n],i,.001,[[-i,"the sign is flipped — a forward difference is signal[i + 1] − signal[i], next minus current"],[t[n+1],"that is the neighbor's value, not the jump — subtract your own signal[this.thread.x]"]]);e.assertClose(s[n],i,.001,a||`element ${n}`)}e.assertClose(s[63],0,1e-4,"last cell should clamp to 0")}}]},{slug:"moving-average",title:"Smooth a Signal",intro:`<p>Time to combine everything: a <strong>5-tap moving average</strong>. Each output
        cell is the mean of <code>signal[x−2 … x+2]</code> — a gather over a small
        <em>window</em> of neighbors, with clamping where the window hangs off either end. This
        shape — loop over a fixed window, clamp, accumulate — is called a
        <strong>stencil</strong>, and it powers blurs, edge detectors, and physics simulations
        alike.</p>
        <p>Yes, a loop <em>inside</em> the kernel is fine: it's 5 iterations of private
        arithmetic per thread, not a loop over the data. 128 threads each averaging 5 numbers
        is still one parallel pass.</p>`,goal:`<strong>Goal:</strong> each cell returns the average of the five values centered on
        it, with window indexes clamped to <code>0 … n−1</code>.`,requirements:["Loop over the window: <code>for (let d = 0; d < 5; d++)</code> with offset <code>d − 2</code>","Clamp every read with <code>Math.max(0, Math.min(n − 1, …))</code>","Return the sum divided by <code>5</code>"],hints:[{title:"Hint 1 — the window",body:`<p>The five indexes are <code>this.thread.x + d - 2</code> for
            <code>d = 0…4</code>: two to the left, itself, two to the right.</p>`},{title:"Hint 2 — clamp inside the loop",body:`<p>Each iteration:</p>
<pre><code>const j = Math.max(0, Math.min(this.constants.n - 1, this.thread.x + d - 2));
sum += signal[j];</code></pre>`},{title:"Hint 3 — sanity-check the edge",body:`<p>Cell 0's clamped window reads indexes <code>0, 0, 0, 1, 2</code> — so
            <code>out[0]</code> should equal <code>(3·signal[0] + signal[1] + signal[2]) / 5</code>.</p>`}],transfer:`Windowed sums over neighbors are stencil computations — the bread and butter of
        scientific codes on CUDA and ROCm, where entire papers are devoted to tiling stencils into
        shared memory so the window reads come from fast on-chip storage instead of DRAM.`,starterCode:`// A 5-tap stencil: mean of signal[x-2 ... x+2], edges clamped.
const gpu = new GPU({ mode });

const smooth = gpu.createKernel(function (signal) {
  let sum = 0;
  for (let d = 0; d < 5; d++) {
    // TODO: read the window neighbor at offset d - 2,
    // clamped to 0 ... this.constants.n - 1
    sum += signal[this.thread.x];
  }
  return sum / 5;
}, {
  output: [128],
  constants: { n: 128 },
});

const result = smooth(signal);
console.log('raw:', signal[64], '→ smoothed:', result[64]);
`,solutionCode:`// A 5-tap stencil: mean of signal[x-2 ... x+2], edges clamped.
const gpu = new GPU({ mode });

const smooth = gpu.createKernel(function (signal) {
  let sum = 0;
  for (let d = 0; d < 5; d++) {
    const j = Math.max(0, Math.min(this.constants.n - 1, this.thread.x + d - 2));
    sum += signal[j];
  }
  return sum / 5;
}, {
  output: [128],
  constants: { n: 128 },
});

const result = smooth(signal);
console.log('raw:', signal[64], '→ smoothed:', result[64]);
`,inputs:e=>({signal:We(e,128,7203)}),publicTests:[{name:"mid-signal cells average their five neighbors",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=We(e.utils,128,7203),s=e.kernel(t);e.assert(s&&s.length===128,`expected 128 output values, got ${s&&s.length}`);const n=Un(t),i=Kn(t);for(const a of[2,17,64,99,125]){const f=ze(s[a],n[a],.001,Js(t,i,n,a));e.assertClose(s[a],n[a],.001,f||`element ${a}`)}}},{name:"edge cells clamp — <code>out[0]</code> averages indexes 0, 0, 0, 1, 2",run:async e=>{const t=We(e.utils,128,7203),s=e.kernel(t),n=t.length,i=Un(t),a=Kn(t),f=ze(s[0],i[0],.001,Js(t,a,i,0));e.assertClose(s[0],(3*t[0]+t[1]+t[2])/5,.001,f||"left edge");const m=ze(s[n-1],i[n-1],.001,Js(t,a,i,n-1));e.assertClose(s[n-1],(3*t[n-1]+t[n-2]+t[n-3])/5,.001,m||"right edge")}}],privateTests:[{name:"private test #1",run:async e=>{const t=We(e.utils,128,9090),s=e.kernel(t),n=Un(t),i=Kn(t);for(let a=0;a<128;a++){const f=ze(s[a],n[a],.001,Js(t,i,n,a));e.assertClose(s[a],n[a],.001,f||`element ${a}`)}}}]},{slug:"two-pass-blur",title:"The Two-Pass Blur",intro:`<p>The payoff. A 3×3 box blur of a 2D grid needs nine reads per cell — but the box
        blur is <strong>separable</strong>: blurring horizontally and then blurring that result
        vertically gives the <em>identical</em> answer with just three reads per cell per pass.
        Bigger blurs win bigger: a 9×9 blur drops from 81 reads to 18.</p>
        <p>This is also how you design around the no-communication rule at scale: since threads
        can't share work <em>within</em> a pass, you split the algorithm into passes — each pass
        a clean parallel gather, each handoff a finished grid. Kernel one blurs along
        <code>x</code>; its output feeds kernel two, which blurs along <code>y</code>. Both are
        3-tap clamped stencils — task 5, twice, at right angles.</p>`,goal:`<strong>Goal:</strong> finish both kernels — <code>blurX</code> averages each cell
        with its left/right neighbors, <code>blurY</code> with its up/down neighbors — edges
        clamped, so the composition equals a full 3×3 box blur.`,requirements:["<code>blurX</code>: 3-tap average along the row — clamp <code>x + d − 1</code>, read <code>grid[this.thread.y][j]</code>","<code>blurY</code>: 3-tap average down the column — clamp <code>y + d − 1</code>, read <code>grid[j][this.thread.x]</code>","Both kernels divide their sum by <code>3</code>","Feed <code>blurX</code>'s output into <code>blurY</code> (already wired up)"],hints:[{title:"Hint 1 — task 5, rotated",body:`<p>Each kernel is the moving-average pattern with a 3-wide window. The only new
            move: in 2D you clamp the coordinate along the blur axis and keep the other
            coordinate fixed.</p>`},{title:"Hint 2 — the x pass",body:`<pre><code>for (let d = 0; d &lt; 3; d++) {
  const j = Math.max(0, Math.min(this.constants.n - 1, this.thread.x + d - 1));
  sum += grid[this.thread.y][j];
}
return sum / 3;</code></pre>
<p>The y pass swaps which coordinate is clamped: <code>grid[j][this.thread.x]</code>.</p>`}],transfer:`Separable filtering is a classic GPU optimization you'll meet everywhere: game
        engines render Gaussian blurs as two fullscreen passes, WebGPU and Metal chain compute
        encoder passes the same way, and CUDA image pipelines launch one kernel per axis. Two
        cheap 1D passes beating one fat 2D pass — O(k) taps instead of O(k²) — never stops
        being true.`,starterCode:`// Two passes at right angles = one 3×3 box blur, for 6 reads instead of 9.
const gpu = new GPU({ mode });

const blurX = gpu.createKernel(function (grid) {
  // TODO: average grid[y][x-1], grid[y][x], grid[y][x+1] — clamp x
  return grid[this.thread.y][this.thread.x];
}, { output: [48, 48], constants: { n: 48 } });

const blurY = gpu.createKernel(function (grid) {
  // TODO: average grid[y-1][x], grid[y][x], grid[y+1][x] — clamp y
  return grid[this.thread.y][this.thread.x];
}, { output: [48, 48], constants: { n: 48 } });

const pass1 = blurX(heightmap);
const smooth = blurY(pass1);
console.log('corner before → after:', heightmap[0][0], '→', smooth[0][0]);
`,solutionCode:`// Two passes at right angles = one 3×3 box blur, for 6 reads instead of 9.
const gpu = new GPU({ mode });

const blurX = gpu.createKernel(function (grid) {
  let sum = 0;
  for (let d = 0; d < 3; d++) {
    const j = Math.max(0, Math.min(this.constants.n - 1, this.thread.x + d - 1));
    sum += grid[this.thread.y][j];
  }
  return sum / 3;
}, { output: [48, 48], constants: { n: 48 } });

const blurY = gpu.createKernel(function (grid) {
  let sum = 0;
  for (let d = 0; d < 3; d++) {
    const j = Math.max(0, Math.min(this.constants.n - 1, this.thread.y + d - 1));
    sum += grid[j][this.thread.x];
  }
  return sum / 3;
}, { output: [48, 48], constants: { n: 48 } });

const pass1 = blurX(heightmap);
const smooth = blurY(pass1);
console.log('corner before → after:', heightmap[0][0], '→', smooth[0][0]);
`,inputs:e=>({heightmap:xs(e,48)}),publicTests:[{name:"two passes compose into a 48×48 grid",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=Xs(e,xs(e.utils,48));e.assert(t&&t.length===48,`expected 48 rows, got ${t&&t.length}`),e.assert(t[0]&&t[0].length===48,"each row should hold 48 values")}},{name:"interior cells equal the full 3×3 box average",run:async e=>{const t=xs(e.utils,48),s=Xs(e,t),n=Ms(Is(t)),i=Nn(t);for(const[a,f]of[[1,1],[10,30],[24,24],[40,7],[46,46]]){const m=Bn(s[a][f],n,i,.002,a,f);e.assertClose(s[a][f],n[a][f],.002,m||`cell [${a}][${f}]`)}}},{name:"edges and corners clamp — no zero-padding creeping in",run:async e=>{const t=xs(e.utils,48),s=Xs(e,t),n=Ms(Is(t)),i=Nn(t);for(const[a,f]of[[0,0],[0,47],[47,0],[47,47],[0,20],[20,0]]){const m=Bn(s[a][f],n,i,.002,a,f);e.assertClose(s[a][f],n[a][f],.002,m||`cell [${a}][${f}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=xs(e.utils,48,555),s=Xs(e,t),n=Ms(Is(t)),i=Nn(t);for(let a=0;a<48;a++)for(let f=0;f<48;f++){const m=Bn(s[a][f],n,i,.002,a,f);e.assertClose(s[a][f],n[a][f],.002,m||`cell [${a}][${f}]`)}}}]}]},zl=Object.freeze({__proto__:null,default:Pl});const jn=[.299,.587,.114];function Cs(e){return jn[0]*e[0]+jn[1]*e[1]+jn[2]*e[2]}function qn(e,t,s,n,i){return Math.abs(e-t)<=s?`that is the value for cell [${i}][${n}] — this.thread.x and this.thread.y are swapped. Rows come first: image[this.thread.y][this.thread.x]`:null}function Ot(e){return e&&typeof e.toArray=="function"?e.toArray():e}function qi(e,t=1701){const s=e.seededRandom(t),n=new Array(256);for(let i=0;i<256;i++)n[i]=Math.round(s()*100)/100;return n}function Wn(e,t=2026){const s=e.seededRandom(t),n=new Array(256);for(let i=0;i<256;i++)n[i]=Math.round(s()*1e3)/100;return n}function Zs(e,t,s){const n=new Array(e).fill(0);return n[t]=s,n}function Wi(e,t){const s=new Array(e).fill(Bt(t));return us(new Array(e).fill(s))}function Hi(e){const t=e.map(n=>{const i=n/10;return i*i}),s=new Array(t.length);for(let n=0;n<t.length;n++){const i=Math.max(n-1,0),a=Math.min(n+1,t.length-1);s[n]=(t[i]+t[n]+t[a])/3}return s}function Xi(e,t){let s=e.slice();const n=s.length;for(let i=0;i<t;i++){const a=s.slice();for(let f=1;f<n-1;f++)a[f]=.25*s[f-1]+.5*s[f]+.25*s[f+1];s=a}return s}function Yi(e){return e.plain.map(t=>t.map(Cs))}function Ji(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){s[n]=new Array(t);for(let i=0;i<t;i++){let a=0;for(let f=-1;f<=1;f++)for(let m=-1;m<=1;m++){const A=Math.min(Math.max(n+f,0),t-1),N=Math.min(Math.max(i+m,0),t-1);a+=e[A][N]}s[n][i]=a/9}}return s}function Qs(e){return Math.min(Math.max((e-.5)*2+.5,0),1)}function Sr(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Zi(e){return[[(e-.5)*2+.5,"that is the stretch without its clamp — Math.min / Math.max keep the result inside 0–1"],[e,"that is the luminance unchanged — the stretch never reached the return value"],[Math.min(Math.max(e*2,0),1),"the midpoint is missing — subtract 0.5 before doubling and add it back afterwards"]]}function Qi(e,t,s,n,i,a){return Number.isFinite(e)?Sr(e,t[i][a],n,[[s[i][a],"that is the unblurred luminance — the 3×3 average never happened"],[9*t[i][a],"that is the sum of the nine samples — a mean divides by 9"]]):"that cell read past the edge of the map — clamp yy and xx into 0…63 before indexing"}var Ol={id:"1-4",track:1,title:"Pipelines & Textures",blurb:"Chaining kernels so data stays on the GPU — the single biggest real-world speedup.",tasks:[{slug:"pipeline-on",title:"Flip On the Pipeline",intro:`<p>Until now, every kernel call ended the same way: the GPU finished computing,
        then the whole result was <strong>downloaded back to JavaScript</strong> as a typed array.
        That download is the expensive part — for a 512×512 grid it's a megabyte crossing the
        bus on every single call.</p>
        <p><code>pipeline: true</code> changes the ending. The kernel still runs the same, but the
        result <em>stays in GPU memory</em>, and what you get back is a <strong>texture</strong> —
        a lightweight handle to data that never left the card. Log one and you'll see an object,
        not numbers. When you actually want the values, you ask for the download explicitly with
        <code>.toArray()</code>.</p>
        <p>One backend wrinkle to know: the CPU backend has no textures, so there a pipeline
        kernel hands back a plain array. Mode-safe code uses the same guard gpu.js uses
        internally: <code>result.toArray ? result.toArray() : result</code>.</p>`,goal:`<strong>Goal:</strong> make the <code>boost</code> kernel a pipeline kernel, then
        download its result explicitly and log the first sample.`,requirements:["Add <code>pipeline: true</code> to the kernel settings","Log the raw result — see what a texture looks like in the console","Download the values with <code>.toArray()</code>, using the mode-safe guard","Log the first value as <code>console.log('first sample:', values[0])</code>"],hints:[{title:"Hint 1 — where does the flag go?",body:`<p><code>pipeline: true</code> sits in the settings object, right next to
            <code>output</code>. Nothing about the kernel function itself changes.</p>`},{title:"Hint 2 — the mode-safe download",body:`<pre><code>const values = result.toArray ? result.toArray() : result;</code></pre>
<p>On the GL backend this calls <code>toArray()</code>; on the CPU backend
            <code>result</code> is already an array and passes through untouched.</p>`}],transfer:`A gpu.js texture is the same idea as a <code>GPUBuffer</code> you never map in
        WebGPU, or device memory behind a pointer in CUDA and ROCm: the data has an address on
        the card, and JavaScript only holds the ticket stub. <code>.toArray()</code> is the
        explicit "map it back to the host" step.`,starterCode:`// Run this as-is first: the kernel returns plain numbers, which means
// every call ships the whole result back to JavaScript. Let's stop that.
const gpu = new GPU({ mode });

const boost = gpu.createKernel(function (signal) {
  return Math.min(signal[this.thread.x] * 1.5, 1);
}, {
  output: [256],
  // TODO: keep the result on the GPU
});

const result = boost(signal);
console.log(result);

// TODO: \`result\` is about to become a texture. Download the values
// explicitly (mode-safe: result.toArray ? result.toArray() : result)
// and log the first one as:  console.log('first sample:', values[0]);
`,solutionCode:`// pipeline: true — the result stays in GPU memory as a texture.
const gpu = new GPU({ mode });

const boost = gpu.createKernel(function (signal) {
  return Math.min(signal[this.thread.x] * 1.5, 1);
}, {
  output: [256],
  pipeline: true,
});

const result = boost(signal);
console.log(result); // GL backend: a Texture object — no numbers in sight

// The explicit download. CPU backend already returns a plain array,
// so guard the call — the same trick gpu.js uses internally.
const values = result.toArray ? result.toArray() : result;
console.log('first sample:', values[0]);
`,inputs:e=>({signal:qi(e)}),publicTests:[{name:"kernel is created with <code>pipeline: true</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.kernel.kernel&&e.kernel.kernel.pipeline===!0,"the kernel is not a pipeline kernel — add pipeline: true to its settings")}},{name:"boosted values read back correctly through <code>toArray()</code>",run:async e=>{const t=new Array(256);for(let n=0;n<256;n++)t[n]=n%100/100;const s=Ot(e.kernel(t));e.assert(s&&s.length===256,`expected 256 values, got ${s&&s.length}`);for(let n=0;n<256;n++)e.assertClose(s[n],Math.min(t[n]*1.5,1),.002,`sample ${n}`)}},{name:"the downloaded first sample is logged as 'first sample:'",run:async e=>{const t=e.logs.some(s=>s.type==="log"&&s.text&&s.text.includes("first sample"));e.assert(t,"expected a console.log('first sample:', values[0]) after the download")}}],privateTests:[{name:"private test #1",run:async e=>{const t=qi(e.utils,8842),s=Ot(e.kernel(t));e.assert(s.length===256,"expected 256 values");for(let n=0;n<256;n++)e.assertClose(s[n],Math.min(t[n]*1.5,1),.002,`sample ${n}`)}}]},{slug:"chain-two-kernels",title:"Chain Kernels, Skip the Round Trip",intro:`<p>Here's the payoff of textures: a texture returned by one kernel can be passed
        <strong>straight into the next kernel</strong> as an argument. gpu.js binds the texture
        as the input — no download, no re-upload, no JavaScript in the middle. The data makes
        the whole trip without ever leaving the card.</p>
        <p>In module 1.2 you chained two kernels through JavaScript: the luminance map came
        back as arrays, then went up again for the second pass. Same chain below — except this
        time <code>luminance</code> is a pipeline kernel, and the second pass eats its texture
        directly.</p>
        ${$t}`,goal:`<strong>Goal:</strong> finish the <code>contrast</code> kernel — stretch each
        luminance value around the midpoint with <code>(l − 0.5) × 2 + 0.5</code>, clamped
        to 0–1 — and keep the texture handoff intact.`,requirements:["Keep <code>luminance</code> a pipeline kernel — its result never touches JavaScript","Pass the returned texture directly into <code>contrast</code> (already wired up)","In <code>contrast</code>, return <code>(l - 0.5) * 2 + 0.5</code> clamped with <code>Math.min</code> / <code>Math.max</code>"],hints:[{title:"Hint 1 — textures index like arrays",body:`<p>Inside <code>contrast</code>, the texture argument reads exactly like the
            2D arrays you already know: <code>map[this.thread.y][this.thread.x]</code>.
            The kernel doesn't care where the data lives.</p>`},{title:"Hint 2 — the clamp",body:"<pre><code>return Math.min(Math.max((l - 0.5) * 2 + 0.5, 0), 1);</code></pre>"}],transfer:`Handing a texture from kernel to kernel is what CUDA does when consecutive
        launches read and write the same device pointers, and what a WebGPU compute pass does
        when one dispatch's storage buffer becomes the next dispatch's binding. On Metal it's
        two encoders sharing an <code>MTLBuffer</code>. Nobody copies to the CPU in between.`,starterCode:`const gpu = new GPU({ mode });

// Pass 1 — luminance map, kept on the GPU as a texture.
const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64], pipeline: true });

// Pass 2 — contrast stretch. Final stage, so it returns plain numbers.
const contrast = gpu.createKernel(function (map) {
  const l = map[this.thread.y][this.thread.x];
  // TODO: stretch around the midpoint — (l - 0.5) * 2 + 0.5 —
  // clamped to 0–1 with Math.min / Math.max
  return l;
}, { output: [64, 64] });

const mapTexture = luminance(photo); // a texture — still on the GPU
const result = contrast(mapTexture); // and straight back in it goes
console.log('center cell:', result[32][32]);
`,solutionCode:`const gpu = new GPU({ mode });

// Pass 1 — luminance map, kept on the GPU as a texture.
const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64], pipeline: true });

// Pass 2 — contrast stretch. Final stage, so it returns plain numbers.
const contrast = gpu.createKernel(function (map) {
  const l = map[this.thread.y][this.thread.x];
  return Math.min(Math.max((l - 0.5) * 2 + 0.5, 0), 1);
}, { output: [64, 64] });

const mapTexture = luminance(photo); // a texture — still on the GPU
const result = contrast(mapTexture); // and straight back in it goes
console.log('center cell:', result[32][32]);
`,inputs:e=>({photo:e.makeTestImage(64)}),publicTests:[{name:"two kernels: a pipeline pass feeding a plain pass",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=e.kernels.find(n=>n.kernel&&n.kernel.pipeline),s=e.kernels.find(n=>n.kernel&&!n.kernel.pipeline);e.assert(t,"no pipeline kernel found — keep pipeline: true on luminance"),e.assert(s,"no plain kernel found — contrast should NOT be a pipeline kernel"),e.resolvedMode==="gpu"&&e.assert(s.lastArgs&&s.lastArgs[0]&&typeof s.lastArgs[0].toArray=="function","contrast should be fed the texture itself — no .toArray() in between")}},{name:"chained result: clamped <code>(l - 0.5) * 2 + 0.5</code> per cell",run:async e=>{const t=e.kernels.find(m=>m.kernel&&m.kernel.pipeline),s=e.kernels.find(m=>m.kernel&&!m.kernel.pipeline);e.assert(t&&s,"expected a pipeline kernel and a plain kernel");const n=e.utils.makeTestImage(64),i=s(t(n)),a=n.plain,f=[[0,0],[7,41],[32,32],[63,63]];for(const[m,A]of f){const N=Cs(a[m][A]),H=qn(i[m][A],Qs(Cs(a[A][m])),.003,m,A)||Sr(i[m][A],Qs(N),.003,Zi(N));e.assertClose(i[m][A],Qs(N),.003,H||`cell [${m}][${A}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(A=>A.kernel&&A.kernel.pipeline),s=e.kernels.find(A=>A.kernel&&!A.kernel.pipeline);e.assert(t&&s,"expected a pipeline kernel and a plain kernel");const n=Wi(64,[.8,.3,.5,1]),i=Cs(n.at(0,0)),a=Qs(i),f=s(t(n)),m=Zi(i);for(let A=0;A<64;A++)for(let N=0;N<64;N++){const H=Sr(f[A][N],a,.003,m);e.assertClose(f[A][N],a,.003,H||`cell [${A}][${N}]`)}}}]},{slug:"tollbooth",title:"toArray() Is a Tollbooth",intro:`<p>Here's the mental model that makes GPU code fast: computation on the card is
        nearly free — it's the <strong>transfers</strong> that cost. Every kernel that is
        <em>not</em> <code>pipeline: true</code> ends with an implicit download, and passing
        that array to the next kernel triggers a re-upload. A three-stage chain without
        pipelines pays the toll <strong>four times</strong> for one result.</p>
        <p>The starter below is a fully working three-stage audio chain — normalize, gamma,
        smooth — and every hop goes through JavaScript. Your job isn't to fix the math.
        It's to fix the traffic: intermediates become pipeline kernels, and only the
        <em>final</em> stage returns plain numbers. The chain call itself shouldn't change
        by a single character.</p>`,goal:`<strong>Goal:</strong> refactor the chain so stages 1 and 2 keep their results on
        the GPU, the final stage returns numbers, and the output is bit-for-bit the same idea —
        just without the round trips.`,requirements:["Make <code>normalize</code> and <code>gamma</code> pipeline kernels","Leave <code>smooth</code> as a plain kernel — the one download you actually want","Do not change the chain: <code>smooth(gamma(normalize(signal)))</code> stays as-is"],hints:[{title:"Hint 1 — where is the readback hiding?",body:`<p>There's no <code>.toArray()</code> in the starter, but the readbacks are
            still there: a non-pipeline kernel's <em>return value</em> is the readback.
            Count them: normalize downloads, gamma re-uploads and downloads, smooth re-uploads.</p>`},{title:"Hint 2 — a two-line diff",body:`<p>Add <code>pipeline: true</code> to the settings of <code>normalize</code>
            and <code>gamma</code>. That's the entire refactor — the chain line already does
            the right thing once textures flow through it.</p>`}],transfer:`Profile any real CUDA or ROCm app and the widest bars are often
        <code>cudaMemcpy</code> DtoH/HtoD, not kernels; in WebGPU the same toll is
        <code>mapAsync</code> plus staging-buffer copies. "Keep data resident, read back once
        at the end" is performance rule number one on every GPU platform.`,starterCode:`const gpu = new GPU({ mode });

// Stage 1 — scale the raw 0–10 signal down to 0–1.
const normalize = gpu.createKernel(function (signal) {
  return signal[this.thread.x] / 10;
}, { output: [256] }); // TODO: this intermediate should stay on the GPU

// Stage 2 — gamma curve to tame the loud parts.
const gamma = gpu.createKernel(function (v) {
  return v[this.thread.x] * v[this.thread.x];
}, { output: [256] }); // TODO: so should this one

// Stage 3 — 3-tap smoothing. Final stage: plain numbers out, on purpose.
const smooth = gpu.createKernel(function (v) {
  let left = this.thread.x - 1;
  let right = this.thread.x + 1;
  if (left < 0) left = 0;
  if (right > 255) right = 255;
  return (v[left] + v[this.thread.x] + v[right]) / 3;
}, { output: [256] });

// This chain is CORRECT — and slow. Each non-pipeline return is a full
// GPU → JS download, and the next call re-uploads it. Four transfers.
const out = smooth(gamma(normalize(signal)));
console.log('smoothed[0]:', out[0]);
`,solutionCode:`const gpu = new GPU({ mode });

// Stage 1 — scale the raw 0–10 signal down to 0–1.
const normalize = gpu.createKernel(function (signal) {
  return signal[this.thread.x] / 10;
}, { output: [256], pipeline: true });

// Stage 2 — gamma curve to tame the loud parts.
const gamma = gpu.createKernel(function (v) {
  return v[this.thread.x] * v[this.thread.x];
}, { output: [256], pipeline: true });

// Stage 3 — 3-tap smoothing. Final stage: plain numbers out, on purpose.
const smooth = gpu.createKernel(function (v) {
  let left = this.thread.x - 1;
  let right = this.thread.x + 1;
  if (left < 0) left = 0;
  if (right > 255) right = 255;
  return (v[left] + v[this.thread.x] + v[right]) / 3;
}, { output: [256] });

// Identical chain, one transfer in, one out. The code didn't change —
// the data's home address did.
const out = smooth(gamma(normalize(signal)));
console.log('smoothed[0]:', out[0]);
`,inputs:e=>({signal:Wn(e)}),publicTests:[{name:"stages 1–2 are pipeline kernels; the final stage is not",run:async e=>{e.assert(e.kernels.length>=3,`expected 3 kernels, found ${e.kernels.length}`);const[t,s,n]=e.kernels;e.assert(t.kernel&&t.kernel.pipeline===!0,"normalize should have pipeline: true"),e.assert(s.kernel&&s.kernel.pipeline===!0,"gamma should have pipeline: true"),e.assert(n.kernel&&!n.kernel.pipeline,"smooth should stay a plain kernel — its return IS the readback you want"),e.resolvedMode==="gpu"&&e.assert(s.lastArgs&&s.lastArgs[0]&&typeof s.lastArgs[0].toArray=="function","gamma should receive a texture from normalize, not an array")}},{name:"the numbers survive the refactor — chain output is unchanged",run:async e=>{const[t,s,n]=e.kernels,i=Wn(e.utils),a=n(s(t(i))),f=Hi(i);e.assert(a&&a.length===256,`expected 256 values, got ${a&&a.length}`);for(let m=0;m<256;m++)e.assertClose(a[m],f[m],.003,`sample ${m}`)}}],privateTests:[{name:"private test #1",run:async e=>{const[t,s,n]=e.kernels,i=Wn(e.utils,5150),a=n(s(t(i))),f=Hi(i);for(let m=0;m<256;m++)e.assertClose(a[m],f[m],.003,`sample ${m}`)}}]},{slug:"iterate-immutable",title:"Feedback Loops: immutable Textures",intro:`<p>Simulations don't run once — they <strong>step</strong>: the output of step
        <em>n</em> is the input of step <em>n</em>+1. With pipelines that means feeding a
        kernel its own texture back. Try it naively and gpu.js stops you cold:
        <em>"Source and destination … are the same. Use immutable = true"</em> — the kernel
        would be reading the very texture it's writing to.</p>
        <p><code>immutable: true</code> is the fix: each call renders to a <em>fresh</em>
        texture instead of recycling one, so last step's output is safe to read while this
        step writes. (In long-running sims you'd call <code>texture.delete()</code> on old
        steps to recycle their memory — at 128 cells here, we'll let them slide.)</p>
        <p>Below is a 1D heat field: 128 cells, all cold except one hot spike. One diffusion
        step moves each cell toward its neighbours. Twelve steps stay entirely on the GPU —
        one upload at the start, one download at the end.</p>`,goal:`<strong>Goal:</strong> make the feedback loop legal — the <code>step</code> kernel
        needs <code>immutable: true</code> — and run 12 diffusion steps without the heat ever
        visiting JavaScript.`,requirements:["Add <code>immutable: true</code> to the <code>step</code> kernel (keep <code>pipeline: true</code>)","Keep the loop feeding <code>step</code>'s output straight back in — no readbacks inside it","After 12 steps, download once and log the peak at cell 64"],hints:[{title:"Hint 1 — read the error message",body:`<p>Run the starter as-is. The error names both the crime and the sentence:
            the kernel's input and output are the same storage, and <code>immutable = true</code>
            is the fix. gpu.js error messages are unusually honest.</p>`},{title:"Hint 2 — why upload() exists",body:`<p>The tiny <code>upload</code> kernel copies the seed array into a texture
            once, so <code>step</code> always sees texture inputs from its very first call.
            Keeping argument types stable means the kernel compiles exactly once.</p>`},{title:"Hint 3 — the one-word diff",body:`<p>In <code>step</code>'s settings:</p>
<pre><code>{ output: [128], pipeline: true, immutable: true }</code></pre>
<p>The loop is already correct.</p>`}],transfer:`Every GPU API solves read-write hazards the same way gpu.js just made you do:
        ping-pong buffering. WebGPU compute passes swap two storage buffers each dispatch,
        CUDA stencil codes swap <code>in</code>/<code>out</code> device pointers, Metal
        simulations flip between two textures. <code>immutable: true</code> is ping-ponging
        with the bookkeeping done for you.`,starterCode:`const gpu = new GPU({ mode });

// Upload pass — copies the seed array into a texture, once.
const upload = gpu.createKernel(function (seed) {
  return seed[this.thread.x];
}, { output: [128], pipeline: true });

// One diffusion step: each cell relaxes toward its neighbours.
// Edge cells hold their value.
const step = gpu.createKernel(function (heat) {
  const x = this.thread.x;
  if (x === 0 || x === 127) {
    return heat[x];
  }
  return 0.25 * heat[x - 1] + 0.5 * heat[x] + 0.25 * heat[x + 1];
}, {
  output: [128],
  pipeline: true,
  // TODO: this kernel reads its own previous output — run it and
  // let the error message tell you the missing setting.
});

let state = upload(field);
for (let i = 0; i < 12; i++) {
  state = step(state); // output straight back in — a feedback loop
}

const heat = state.toArray ? state.toArray() : state;
console.log('peak after 12 steps:', heat[64]);
`,solutionCode:`const gpu = new GPU({ mode });

// Upload pass — copies the seed array into a texture, once.
const upload = gpu.createKernel(function (seed) {
  return seed[this.thread.x];
}, { output: [128], pipeline: true });

// One diffusion step: each cell relaxes toward its neighbours.
// Edge cells hold their value.
const step = gpu.createKernel(function (heat) {
  const x = this.thread.x;
  if (x === 0 || x === 127) {
    return heat[x];
  }
  return 0.25 * heat[x - 1] + 0.5 * heat[x] + 0.25 * heat[x + 1];
}, {
  output: [128],
  pipeline: true,
  immutable: true, // fresh output texture per call — feedback is now safe
});

let state = upload(field);
for (let i = 0; i < 12; i++) {
  state = step(state); // output straight back in — a feedback loop
}

const heat = state.toArray ? state.toArray() : state;
console.log('peak after 12 steps:', heat[64]);
`,inputs:()=>({field:Zs(128,64,1)}),publicTests:[{name:"the stepping kernel opts into <code>immutable: true</code>",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=e.kernels.find(s=>s.kernel&&s.kernel.immutable);e.assert(t,"no immutable kernel found — the feedback loop needs immutable: true on step"),e.assert(t.kernel.pipeline===!0,"step should keep pipeline: true too")}},{name:"twelve steps match a reference diffusion of the spike",run:async e=>{const t=e.kernels.find(m=>m.kernel&&m.kernel.pipeline&&!m.kernel.immutable),s=e.kernels.find(m=>m.kernel&&m.kernel.immutable);e.assert(t&&s,"expected an upload kernel and an immutable step kernel");const n=Zs(128,64,1);let i=t(n);for(let m=0;m<12;m++)i=s(i);const a=Ot(i),f=Xi(n,12);for(let m=0;m<128;m++)e.assertClose(a[m],f[m],.002,`cell ${m}`)}},{name:"heat is conserved — the field still sums to 1.0",run:async e=>{const t=e.kernels.find(f=>f.kernel&&f.kernel.pipeline&&!f.kernel.immutable),s=e.kernels.find(f=>f.kernel&&f.kernel.immutable);e.assert(t&&s,"expected an upload kernel and an immutable step kernel");let n=t(Zs(128,64,1));for(let f=0;f<12;f++)n=s(n);const i=Ot(n);let a=0;for(let f=0;f<128;f++)a+=i[f];e.assertClose(a,1,.01,"total heat in the field")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(A=>A.kernel&&A.kernel.pipeline&&!A.kernel.immutable),s=e.kernels.find(A=>A.kernel&&A.kernel.immutable);e.assert(t&&s,"expected an upload kernel and an immutable step kernel");const n=Zs(128,40,.75);let i=t(n);for(let A=0;A<12;A++)i=s(i);const a=Ot(i),f=Xi(n,12);let m=0;for(let A=0;A<128;A++)e.assertClose(a[A],f[A],.002,`cell ${A}`),m+=a[A];e.assertClose(m,.75,.01,"total heat in the field")}}]},{slug:"photo-to-screen",title:"The Payoff: Photo to Screen, Zero Readbacks",intro:`<p>Time to cash in the whole module. In module 1.2's finale, a two-kernel chain
        hauled the luminance map down to JavaScript and back up again — two transfers it didn't
        need. This pipeline does more work with <em>fewer</em> transfers: photo →
        <strong>luminance</strong> → <strong>3×3 blur</strong> → <strong>painted canvas</strong>,
        and after the photo is uploaded, nothing comes back. The graphical kernel eats the blur
        texture and writes pixels; readbacks: zero.</p>
        <p>The missing piece is the blur. Each cell averages its 3×3 neighbourhood — two little
        loops over <code>dy</code>/<code>dx</code>, indices clamped to 0…63 so the edges don't
        read out of bounds. When it works, hit <strong>Benchmark</strong> and watch what
        keeping data on the card does to the gap.</p>
        ${$t}`,goal:`<strong>Goal:</strong> implement the 3×3 box blur so the full three-pass pipeline —
        two texture passes and a graphical finale — runs with zero readbacks.`,requirements:["Blur: average the 3×3 neighbourhood, clamping indices to 0…63 at the edges","Both <code>luminance</code> and <code>blur</code> stay <code>pipeline: true</code>","The graphical pass is fed the blur <em>texture</em> — nothing is downloaded","Render the result with <code>render(paint.canvas)</code>"],hints:[{title:"Hint 1 — the neighbourhood loops",body:`<p>Two nested loops with fixed bounds are fine in a kernel:
            <code>for (let dy = -1; dy &lt;= 1; dy++)</code> and the same for <code>dx</code>.
            Accumulate into a <code>sum</code>, return <code>sum / 9</code>.</p>`},{title:"Hint 2 — clamping the edges",body:`<p>Compute <code>let yy = this.thread.y + dy;</code> then push it back in
            range:</p>
<pre><code>if (yy &lt; 0) yy = 0;
if (yy &gt; 63) yy = 63;</code></pre>
<p>Same for <code>xx</code>. Corner cells just count some neighbours twice.</p>`},{title:"Hint 3 — the whole body",body:`<p><code>let sum = 0;</code> then inside the loops
            <code>sum += map[yy][xx];</code> and finally <code>return sum / 9;</code> —
            the clamped <code>yy</code>/<code>xx</code> from hint 2 do the rest.</p>`}],transfer:`You just built what engine programmers call a render graph: named passes,
        explicit dependencies, all resources resident on the GPU — the architecture behind
        Frostbite's frame graph, CUDA Graphs' pre-recorded launch chains, and a Metal command
        buffer full of encoder passes. Real engines are this task with more boxes.`,starterCode:`const gpu = new GPU({ mode });

// Pass 1 — luminance map. You've written this one twice already.
const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64], pipeline: true });

// Pass 2 — 3×3 box blur. Currently a do-nothing passthrough.
const blur = gpu.createKernel(function (map) {
  // TODO: average the 3×3 neighbourhood around this cell.
  // Clamp indices to 0…63 so edges don't read out of bounds.
  return map[this.thread.y][this.thread.x];
}, { output: [64, 64], pipeline: true });

// Pass 3 — paint the blurred map. Texture in, pixels out.
const paint = gpu.createKernel(function (map) {
  const l = map[this.thread.y][this.thread.x];
  this.color(l, l, l, 1);
}, { output: [64, 64], graphical: true });

// The whole pipeline: after \`photo\` goes up, nothing comes back down.
paint(blur(luminance(photo)));
render(paint.canvas);
`,solutionCode:`const gpu = new GPU({ mode });

// Pass 1 — luminance map. You've written this one twice already.
const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64], pipeline: true });

// Pass 2 — 3×3 box blur over the luminance texture.
const blur = gpu.createKernel(function (map) {
  let sum = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      let yy = this.thread.y + dy;
      let xx = this.thread.x + dx;
      if (yy < 0) yy = 0;
      if (yy > 63) yy = 63;
      if (xx < 0) xx = 0;
      if (xx > 63) xx = 63;
      sum += map[yy][xx];
    }
  }
  return sum / 9;
}, { output: [64, 64], pipeline: true });

// Pass 3 — paint the blurred map. Texture in, pixels out.
const paint = gpu.createKernel(function (map) {
  const l = map[this.thread.y][this.thread.x];
  this.color(l, l, l, 1);
}, { output: [64, 64], graphical: true });

// The whole pipeline: after \`photo\` goes up, nothing comes back down.
paint(blur(luminance(photo)));
render(paint.canvas);
`,inputs:e=>({photo:e.makeTestImage(64)}),publicTests:[{name:"three passes: two texture kernels feeding a graphical finale",run:async e=>{e.assert(e.kernels.length>=3,`expected 3 kernels, found ${e.kernels.length}`);const[t,s,n]=e.kernels;e.assert(t.kernel&&t.kernel.pipeline===!0,"luminance should have pipeline: true"),e.assert(s.kernel&&s.kernel.pipeline===!0,"blur should have pipeline: true"),e.assert(n.kernel&&n.kernel.graphical,"the third kernel should be graphical"),e.assert(e.canvas,"no canvas — did you call render(paint.canvas)?"),e.assert(e.canvas.width===64&&e.canvas.height===64,`expected a 64×64 canvas, got ${e.canvas.width}×${e.canvas.height}`),e.resolvedMode==="gpu"&&e.assert(n.lastArgs&&n.lastArgs[0]&&typeof n.lastArgs[0].toArray=="function","paint should be fed the blur texture directly — zero readbacks")}},{name:"blur pass: each cell is the mean of its 3×3 neighbourhood",run:async e=>{const[t,s]=e.kernels,n=e.utils.makeTestImage(64),i=Ot(s(t(n))),a=Yi(n),f=Ji(a),m=[[32,32],[0,20],[63,20],[20,0],[20,63],[0,0],[10,47]];for(const[A,N]of m){const H=qn(i[A][N],f[N][A],.003,A,N)||Qi(i[A][N],f,a,.003,A,N);e.assertClose(i[A][N],f[A][N],.003,H||`cell [${A}][${N}]`)}}},{name:"painted canvas is monochrome",run:async e=>{const t=e.getPixels();e.assert(t.length===4096*4,"pixel buffer should hold 64×64 RGBA values");for(let s=0;s<t.length;s+=244){const n=t[s],i=t[s+1],a=t[s+2];e.assert(Math.abs(n-i)<=1&&Math.abs(i-a)<=1,`pixel at byte ${s} is not gray: rgb(${n}, ${i}, ${a})`)}}}],privateTests:[{name:"private test #1",run:async e=>{const[t,s,n]=e.kernels,i=Wi(64,[.35,.65,.15,1]),a=Cs(i.at(0,0))*255;n(s(t(i)));const f=n.getPixels();for(let m=0;m<f.length;m+=596)e.assertClose(f[m],a,2,`red at byte ${m}`),e.assertClose(f[m+1],a,2,`green at byte ${m}`),e.assertClose(f[m+2],a,2,`blue at byte ${m}`)}},{name:"private test #2",run:async e=>{const[t,s]=e.kernels,n=e.utils.makeTestImage(64),i=Ot(s(t(n))),a=Yi(n),f=Ji(a);for(let m=0;m<64;m++)for(let A=0;A<64;A++){const N=qn(i[m][A],f[A][m],.004,m,A)||Qi(i[m][A],f,a,.004,m,A);e.assertClose(i[m][A],f[m][A],.004,N||`cell [${m}][${A}]`)}}}]}]},Rl=Object.freeze({__proto__:null,default:Ol});function Tt(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++)i[a]=Math.round(n()*1e3)/100;return i}function ns(e){let t=0;for(let s=1;s<=1e3;s++)t+=1/(s+e);return t}function He(e,t){return e.logs.some(s=>s.type==="log"&&s.text&&s.text.includes(t))}function ot(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function ea(e){return[[Math.sin(e/100),"the amplitude is missing — the sample is Math.sin(x / 100) * 100"],[Math.sin(e)*100,"you sampled Math.sin(this.thread.x) — the index has to be divided by 100 first"],[Math.sin(e*100)*100,"the index is multiplied by 100 where it should be divided by it"]]}function Yt(e,t,s){return[e[t],`that is the element unchanged — the ${s} never happened`]}function ta(e,t){return Fl(16,s=>e[s],s=>t[s]*2,.001,[[s=>2*s,"every cell is twice the thread index, not twice the element — index the array with it: data[this.thread.x]"]])}function Fl(e,t,s,n,i){const a=i.filter(([f])=>{let m=!1;for(let A=0;A<e;A++){if(!(Math.abs(t(A)-f(A))<=n))return!1;Math.abs(s(A)-f(A))>n&&(m=!0)}return m}).map(f=>f[1]);return a.length&&a.every(f=>f===a[0])?a[0]:null}function sa(e){const t=[[1/(1+e),"each pass overwrote the running total — accumulate it with sum += inside the loop"],[ns(0)+1e3*e,"the parentheses are missing — each term is 1 / (k + this.thread.x), not 1 / k + this.thread.x"]];return e>0&&t.push([ns(e)+1/e,"the loop started at k = 0 — that extra 1 / this.thread.x term does not belong to the sum"]),t}var Gl={id:"1-5",track:1,title:"Measuring Speed Honestly",blurb:"Warm-up, transfer costs, and precision — when the GPU wins, and when the CPU quietly beats it.",tasks:[{slug:"first-call-lie",title:"The First Call Is a Lie",intro:`<p>The first time you invoke a kernel, gpu.js does far more than run it: it
        <strong>transpiles</strong> your JavaScript function to shader code, hands it to the GPU
        driver to <strong>compile and link</strong>, allocates textures — and <em>then</em> runs it.
        The second call skips straight to the run.</p>
        <p>So timing the first call measures the compiler, not your kernel. It can be 100× slower
        than the steady state, and it happens exactly once. Every honest benchmark
        <strong>warms up first</strong> and throws that first measurement away.</p>`,goal:`<strong>Goal:</strong> finish the kernel, then use <code>Date.now()</code> to time the
        <em>first</em> call and the <em>warmed-up</em> average separately — and log both.`,requirements:["Finish the kernel: return <code>Math.sin(x / 100) * 100</code> where <code>x</code> is this thread's index","Time the first call with <code>Date.now()</code> and log it: <code>first call: N ms</code>","Call the kernel 10 more times in one timed block and log the average: <code>warm call: N ms</code>"],hints:[{title:"Hint 1 — the stopwatch pattern",body:`<p>Snapshot the clock, do the work, subtract:</p>
<pre><code>const t0 = Date.now();
// … the work …
console.log('first call:', Date.now() - t0, 'ms');</code></pre>`},{title:"Hint 2 — averaging the warm calls",body:`<p>One stopwatch around a loop of 10 calls, then divide:</p>
<pre><code>t0 = Date.now();
for (let i = 0; i &lt; 10; i++) wave();
console.log('warm call:', (Date.now() - t0) / 10, 'ms');</code></pre>`}],transfer:`Every platform has a version of this pause: CUDA JIT-compiles PTX at first launch
        (then caches it), WebGPU builds the shader in <code>createComputePipeline</code>, Metal
        compiles MSL when the pipeline state is created. Benchmarking guides on all of them open
        with the same rule — discard the first iteration.`,starterCode:`// The first call compiles. The rest just run. Prove it.
const gpu = new GPU({ mode });

const wave = gpu.createKernel(function () {
  // TODO: return Math.sin(x / 100) * 100, where x is this thread's index
  return 0;
}, { output: [2048] });

// TODO: time the FIRST call with Date.now():
//   const t0 = Date.now();  ...call wave()...
//   console.log('first call:', Date.now() - t0, 'ms');
const result = wave();

// TODO: call wave() 10 more times inside one timed block, then log the
// average as:  console.log('warm call:', totalMs / 10, 'ms');

console.log('sample value:', result[100]);
`,solutionCode:`// The first call compiles. The rest just run. Prove it.
const gpu = new GPU({ mode });

const wave = gpu.createKernel(function () {
  return Math.sin(this.thread.x / 100) * 100;
}, { output: [2048] });

// First call: transpile + compile + allocate + run.
let t0 = Date.now();
const result = wave();
console.log('first call:', Date.now() - t0, 'ms');

// Steady state: the compiled program just runs.
t0 = Date.now();
for (let i = 0; i < 10; i++) wave();
console.log('warm call:', (Date.now() - t0) / 10, 'ms');

console.log('sample value:', result[100]);
`,publicTests:[{name:"kernel computes <code>sin(x / 100) · 100</code> for all 2048 threads",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===2048,`expected 2048 output values, got ${t&&t.length}`);for(const s of[0,1,100,777,1023,2047]){const n=Math.sin(s/100)*100,i=ot(t[s],n,.05,ea(s));e.assertClose(t[s],n,.05,i||`element ${s}`)}}},{name:"both timings are logged: <code>first call</code> and <code>warm call</code>",run:async e=>{e.assert(He(e,"first call"),"time the first call and log it — console.log('first call:', ms, 'ms')"),e.assert(He(e,"warm call"),"time 10 warmed-up calls and log the average — console.log('warm call:', avg, 'ms')")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel();e.assert(t.length===2048,"expected 2048 output values");for(let s=0;s<2048;s++){const n=Math.sin(s/100)*100,i=ot(t[s],n,.05,ea(s));e.assertClose(t[s],n,.05,i||`element ${s}`)}}}]},{slug:"transfer-tax",title:"Pay the Transfer Tax",intro:`<p>A kernel call isn't just compute. Every invocation ships your input array from
        JavaScript to GPU memory, runs, then ships the result back. For a one-instruction kernel
        like <code>value + 1</code>, the arithmetic is nearly free — <strong>the ride is the whole
        bill</strong>.</p>
        <p>Below, the same trivial kernel runs on 1,024 values and on 65,536 values — 64× the data,
        identical math per thread. If compute were the cost, both would time about the same. Warm
        up first (task 1!), then measure: the per-call cost tracks <strong>bytes moved</strong>,
        not operations performed.</p>`,goal:`<strong>Goal:</strong> finish the <code>+ 1</code> kernel and the
        <code>timeKernel</code> helper — warm up, then average 20 timed calls — and log the
        per-call cost for both payload sizes.`,requirements:["Kernel returns <code>data[this.thread.x] + 1</code> — one instruction, on purpose","In <code>timeKernel</code>: call the kernel once <em>untimed</em> to warm it up","Then time 20 calls with <code>Date.now()</code> and return the average ms per call","Log both costs (the <code>small:</code>/<code>big:</code> lines are already wired up)"],hints:[{title:"Hint 1 — why warm up here too?",body:`<p><code>makePlusOne</code> builds <em>two separate kernels</em>, and each one
            compiles on its own first call. Without the warm-up, the big kernel's timing would
            include a compile — task 1's lie all over again.</p>`},{title:"Hint 2 — the helper body",body:`<pre><code>kernel(arg);
const t0 = Date.now();
for (let i = 0; i &lt; 20; i++) kernel(arg);
return (Date.now() - t0) / 20;</code></pre>`}],transfer:`The bus is the bottleneck everywhere: <code>cudaMemcpy</code> across PCIe is the
        classic hot spot in CUDA and ROCm profiles, WebGPU makes you stage the copies explicitly
        with <code>writeBuffer</code> and <code>mapAsync</code>, and Apple's unified memory exists
        precisely to shrink this tax. Arithmetic is cheap; moving bytes is not.`,starterCode:`// One-instruction kernel, two payload sizes. Cost tracks bytes, not math.
const gpu = new GPU({ mode });

function makePlusOne(n) {
  return gpu.createKernel(function (data) {
    // TODO: return this thread's element, plus one
    return data[this.thread.x];
  }, { output: [n] });
}

const smallKernel = makePlusOne(1024);   // small = 1,024 values
const bigKernel = makePlusOne(65536);    // big = 65,536 values

function timeKernel(kernel, arg) {
  // TODO: warm up with one untimed call (task 1!),
  // then time 20 calls and return the average ms per call
  return 0;
}

console.log('small:', timeKernel(smallKernel, small), 'ms/call');
console.log('big:', timeKernel(bigKernel, big), 'ms/call');
`,solutionCode:`// One-instruction kernel, two payload sizes. Cost tracks bytes, not math.
const gpu = new GPU({ mode });

function makePlusOne(n) {
  return gpu.createKernel(function (data) {
    return data[this.thread.x] + 1;
  }, { output: [n] });
}

const smallKernel = makePlusOne(1024);   // small = 1,024 values
const bigKernel = makePlusOne(65536);    // big = 65,536 values

function timeKernel(kernel, arg) {
  kernel(arg); // warm up — never time the compile (task 1)
  const t0 = Date.now();
  for (let i = 0; i < 20; i++) kernel(arg);
  return (Date.now() - t0) / 20;
}

console.log('small:', timeKernel(smallKernel, small), 'ms/call');
console.log('big:', timeKernel(bigKernel, big), 'ms/call');
`,inputs:e=>({small:Tt(e,1024,1101),big:Tt(e,65536,1102)}),publicTests:[{name:"two kernels exist: output sizes <code>1024</code> and <code>65536</code>",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=e.kernels.find(n=>n.kernel&&n.kernel.output&&n.kernel.output[0]===1024),s=e.kernels.find(n=>n.kernel&&n.kernel.output&&n.kernel.output[0]===65536);e.assert(t,"no kernel with output [1024] found"),e.assert(s,"no kernel with output [65536] found")}},{name:"every element comes back as <code>value + 1</code>",run:async e=>{const t=e.kernels.find(m=>m.kernel&&m.kernel.output&&m.kernel.output[0]===1024),s=e.kernels.find(m=>m.kernel&&m.kernel.output&&m.kernel.output[0]===65536);e.assert(t&&s,"expected kernels with outputs [1024] and [65536]");const n=Tt(e.utils,1024,2201),i=t(n);for(let m=0;m<1024;m++){const A=ot(i[m],n[m]+1,.001,[Yt(n,m,"+ 1")]);e.assertClose(i[m],n[m]+1,.001,A||`small element ${m}`)}const a=Tt(e.utils,65536,2202),f=s(a);for(let m=0;m<65536;m+=271){const A=ot(f[m],a[m]+1,.001,[Yt(a,m,"+ 1")]);e.assertClose(f[m],a[m]+1,.001,A||`big element ${m}`)}}},{name:"per-call cost logged for both payloads (<code>ms/call</code>)",run:async e=>{e.assert(He(e,"small:"),"log the small kernel's cost — the console.log is in the starter"),e.assert(He(e,"big:"),"log the big kernel's cost — the console.log is in the starter"),e.assert(He(e,"ms/call"),"timeKernel should return ms per call (did it return 0 forever?)")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(m=>m.kernel&&m.kernel.output&&m.kernel.output[0]===1024),s=e.kernels.find(m=>m.kernel&&m.kernel.output&&m.kernel.output[0]===65536);e.assert(t&&s,"expected kernels with outputs [1024] and [65536]");const n=Tt(e.utils,1024,3301),i=t(n);e.assert(i.length===1024,"small kernel should produce 1024 values");for(let m=0;m<1024;m++){const A=ot(i[m],n[m]+1,.001,[Yt(n,m,"+ 1")]);e.assertClose(i[m],n[m]+1,.001,A||`small element ${m}`)}const a=Tt(e.utils,65536,3302),f=s(a);e.assert(f.length===65536,"big kernel should produce 65536 values");for(let m=0;m<65536;m+=97){const A=ot(f[m],a[m]+1,.001,[Yt(a,m,"+ 1")]);e.assertClose(f[m],a[m]+1,.001,A||`big element ${m}`)}}}]},{slug:"two-answers",title:"Two Machines, Two Answers",intro:`<p>JavaScript numbers are 64-bit floats — about 16 significant digits. GPU shaders
        compute in <strong>32-bit floats</strong> — about 7. Run the <em>same</em> arithmetic on
        both machines and the answers drift apart, a little more with every operation.</p>
        <p>The kernel below adds 1,000 fractions per thread; a plain JavaScript loop computes the
        identical sum in float64. The two results will disagree somewhere around the sixth decimal
        place — which means <code>===</code> is the wrong question. The right question is:
        <strong>are they within a tolerance that matters for your problem?</strong></p>`,goal:`<strong>Goal:</strong> finish the kernel — each thread sums
        <code>1 / (k + this.thread.x)</code> for <code>k = 1…1000</code> — then fix the final
        comparison to use a tolerance instead of <code>===</code>.`,requirements:["Kernel: accumulate <code>1 / (k + this.thread.x)</code> over <code>k = 1…1000</code> in a loop","Keep the float64 reference sum for thread 0 (already wired up)","Log the verdict with a tolerance: <code>Math.abs(result[0] - ref) &lt; 1e-3</code>, not <code>===</code>"],hints:[{title:"Hint 1 — loops inside kernels",body:`<p>Fixed-bound loops are fine in kernel code:</p>
<pre><code>for (let k = 1; k &lt;= 1000; k++) {
  sum += 1 / (k + this.thread.x);
}</code></pre>`},{title:"Hint 2 — the tolerant verdict",body:`<p>Replace the <code>===</code> comparison in the last line with
            <code>Math.abs(result[0] - ref) &lt; 1e-3</code>. Exact equality across float32 and
            float64 is a coin you will almost never win.</p>`}],transfer:`float32-by-default is universal shader behavior — and production GPU code often
        trades away <em>more</em> precision on purpose: CUDA's <code>--use_fast_math</code>, TF32
        on tensor cores, half-precision inference. That's why numerical toolkits ship
        <code>allclose</code>-style comparisons, and why this course's tests use
        <code>assertClose</code> instead of <code>==</code>.`,starterCode:`// Same math, two machines: your GPU adds in float32, JavaScript in float64.
const gpu = new GPU({ mode });

const partialSums = gpu.createKernel(function () {
  let sum = 0;
  // TODO: add up 1 / (k + this.thread.x) for k = 1 ... 1000
  sum = 1 / (1 + this.thread.x);
  return sum;
}, { output: [64] });

const result = partialSums();

// The same sum for thread 0, computed in float64 JavaScript:
let ref = 0;
for (let k = 1; k <= 1000; k++) ref += 1 / k;

console.log('kernel says:', result[0]);
console.log('js says:    ', ref);
console.log('difference:', Math.abs(result[0] - ref));
// TODO: '===' is the wrong question — compare with a tolerance instead:
console.log('close enough:', result[0] === ref);
`,solutionCode:`// Same math, two machines: your GPU adds in float32, JavaScript in float64.
const gpu = new GPU({ mode });

const partialSums = gpu.createKernel(function () {
  let sum = 0;
  for (let k = 1; k <= 1000; k++) {
    sum += 1 / (k + this.thread.x);
  }
  return sum;
}, { output: [64] });

const result = partialSums();

// The same sum for thread 0, computed in float64 JavaScript:
let ref = 0;
for (let k = 1; k <= 1000; k++) ref += 1 / k;

console.log('kernel says:', result[0]);
console.log('js says:    ', ref);
console.log('difference:', Math.abs(result[0] - ref));
// The right question: within a tolerance that matters for this problem?
console.log('close enough:', Math.abs(result[0] - ref) < 1e-3);
`,publicTests:[{name:"all 64 partial sums match the float64 reference within <code>1e-3</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===64,`expected 64 output values, got ${t&&t.length}`);for(let s=0;s<64;s++){const n=ot(t[s],ns(s),.001,sa(s));e.assertClose(t[s],ns(s),.001,n||`partial sum for thread ${s}`)}}},{name:"verdict uses a tolerance — <code>close enough: true</code> is logged",run:async e=>{e.assert(He(e,"difference:"),"keep the difference log — it shows the float32/float64 drift"),e.assert(He(e,"close enough: true"),"compare with Math.abs(result[0] - ref) < 1e-3, not === — the verdict should log true")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel();for(let s=0;s<64;s++){const n=ot(t[s],ns(s),.001,sa(s));e.assertClose(t[s],ns(s),.001,n||`partial sum for thread ${s}`)}for(let s=0;s<63;s++)e.assert(t[s]>t[s+1],`sum for thread ${s} should exceed thread ${s+1}`)}}]},{slug:"when-cpu-wins",title:"When the CPU Wins",intro:`<p>Sixteen numbers, doubled. The GPU <em>can</em> do it — but every kernel call pays
        a fixed toll before any math happens: dispatch through the graphics API, upload 16 values,
        read 16 back. A plain JavaScript loop finishes the whole job in nanoseconds, before the
        GPU has cleared its throat.</p>
        <p>This is the module's payoff — the full honest-measurement checklist in one run:
        <strong>warm up first</strong> (task 1), <strong>remember the transfer toll</strong>
        (task 2), <strong>compare results with a tolerance</strong> (task 3), and then
        <strong>declare the true winner</strong> — even when it isn't the GPU. Parallel hardware
        pays off on big workloads; on tiny ones, the honest answer is a for-loop.</p>`,goal:`<strong>Goal:</strong> double <code>tiny</code> both ways — kernel and plain loop —
        verify they agree within a tolerance, time both fairly, and log the winner.`,requirements:["Kernel returns <code>data[this.thread.x] * 2</code> for all 16 threads","Compare <code>fromKernel</code> to <code>fromLoop</code> element-wise with tolerance <code>1e-4</code> and log <code>match: true</code>","Time 200 warmed-up rounds of each contender and log both as <code>ms/round</code>","Log <code>winner:</code> with whichever contender was faster"],hints:[{title:"Hint 1 — the tolerant match",body:`<p>Task 3's move, in a loop: start with <code>let allMatch = true;</code> and flip
            it to <code>false</code> whenever <code>Math.abs(fromKernel[i] - fromLoop[i]) &gt; 1e-4</code>.</p>`},{title:"Hint 2 — a fair fight",body:`<p>The first <code>doubleTiny(tiny)</code> call already warmed the kernel up, so
            both timed loops measure steady state. Time 200 rounds of <code>doubleTiny(tiny)</code>,
            then 200 rounds of the JS loop, and divide each total by 200.</p>`},{title:"Hint 3 — declaring the winner",body:`<pre><code>console.log('winner:', kernelMs &lt; loopMs ? 'gpu kernel' : 'plain js');</code></pre>
<p>On a job this small, expect the loop to take it. That's the honest answer.</p>`}],transfer:`Kernel-launch overhead runs to microseconds on CUDA and ROCm — thousands of
        CPU instructions' worth per launch. It's why serious frameworks batch and fuse tiny
        operations instead of dispatching them one at a time, and why "is this workload big
        enough?" is the first question asked in any GPU port.`,starterCode:`// 16 numbers. The GPU CAN double them — but should it?
const gpu = new GPU({ mode });

const doubleTiny = gpu.createKernel(function (data) {
  // TODO: return double this thread's element
  return data[this.thread.x];
}, { output: [16] });

const fromKernel = doubleTiny(tiny); // also serves as the warm-up call

// The same job, plain JavaScript:
const fromLoop = new Array(16);
for (let i = 0; i < 16; i++) fromLoop[i] = tiny[i] * 2;

// TODO: compare fromKernel and fromLoop element-wise with tolerance 1e-4
// (task 3!) and log:  console.log('match:', allMatch);

// TODO: time 200 rounds of each contender with Date.now(), then log:
//   console.log('kernel:  ', kernelMs, 'ms/round');
//   console.log('plain js:', loopMs, 'ms/round');
//   console.log('winner:', kernelMs < loopMs ? 'gpu kernel' : 'plain js');
`,solutionCode:`// 16 numbers. The GPU CAN double them — but should it?
const gpu = new GPU({ mode });

const doubleTiny = gpu.createKernel(function (data) {
  return data[this.thread.x] * 2;
}, { output: [16] });

const fromKernel = doubleTiny(tiny); // also serves as the warm-up call

// The same job, plain JavaScript:
const fromLoop = new Array(16);
for (let i = 0; i < 16; i++) fromLoop[i] = tiny[i] * 2;

// Same answer? Tolerance, not === (task 3).
let allMatch = true;
for (let i = 0; i < 16; i++) {
  if (Math.abs(fromKernel[i] - fromLoop[i]) > 1e-4) allMatch = false;
}
console.log('match:', allMatch);

// A fair fight: both warmed up, both averaged over many rounds.
const ROUNDS = 200;
let t0 = Date.now();
for (let r = 0; r < ROUNDS; r++) doubleTiny(tiny);
const kernelMs = (Date.now() - t0) / ROUNDS;

t0 = Date.now();
for (let r = 0; r < ROUNDS; r++) {
  for (let i = 0; i < 16; i++) fromLoop[i] = tiny[i] * 2;
}
const loopMs = (Date.now() - t0) / ROUNDS;

console.log('kernel:  ', kernelMs, 'ms/round');
console.log('plain js:', loopMs, 'ms/round');
console.log('winner:', kernelMs < loopMs ? 'gpu kernel' : 'plain js');
`,inputs:e=>({tiny:Tt(e,16,4404)}),publicTests:[{name:"kernel doubles all 16 values",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(16);for(let i=0;i<16;i++)t[i]=i*1.25-3;const s=e.kernel(t);e.assert(s&&s.length===16,`expected 16 output values, got ${s&&s.length}`);const n=ta(s,t);for(let i=0;i<16;i++){const a=n||ot(s[i],t[i]*2,.001,[Yt(t,i,"doubling")]);e.assertClose(s[i],t[i]*2,.001,a||`element ${i}`)}}},{name:"results agree within tolerance — <code>match: true</code> is logged",run:async e=>{e.assert(He(e,"match: true"),"compare fromKernel and fromLoop with Math.abs(a - b) <= 1e-4 and log the verdict — expected 'match: true'")}},{name:"both contenders timed and a winner declared",run:async e=>{e.assert(He(e,"ms/round"),"time both contenders and log each as ms/round"),e.assert(He(e,"kernel:"),"log the kernel's time — console.log('kernel:  ', kernelMs, 'ms/round')"),e.assert(He(e,"plain js:"),"log the loop's time — console.log('plain js:', loopMs, 'ms/round')"),e.assert(He(e,"winner:"),"declare the faster contender with a 'winner:' log")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Tt(e.utils,16,5505),s=e.kernel(t);e.assert(s.length===16,"expected exactly 16 output values");const n=ta(s,t);for(let i=0;i<16;i++){const a=n||ot(s[i],t[i]*2,.001,[Yt(t,i,"doubling")]);e.assertClose(s[i],t[i]*2,.001,a||`element ${i}`)}}}]}]},Ll=Object.freeze({__proto__:null,default:Gl});function Jt(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++)i[a]=Math.round(n()*100-50)/10;return i}function ge(e,t,s,n){const i=e.seededRandom(n),a=new Array(t);for(let f=0;f<t;f++){const m=new Array(s);for(let A=0;A<s;A++)m[A]=Math.round(i()*100-50)/10;a[f]=m}return a}function na(e){const t=new Array(e);for(let s=0;s<e;s++){const n=new Array(e).fill(0);n[s]=1,t[s]=n}return t}function Es(e,t){let s=0;for(let n=0;n<e.length;n++)s+=e[n]*t[n];return s}function yt(e,t){const s=e.length,n=t.length,i=t[0].length,a=new Array(s);for(let f=0;f<s;f++){const m=new Array(i);for(let A=0;A<i;A++){let N=0;for(let H=0;H<n;H++)N+=e[f][H]*t[H][A];m[A]=N}a[f]=m}return a}function Ul(e,t,s){const n=e.length,i=t[0].length,a=new Array(n);for(let f=0;f<n;f++){const m=new Array(i);for(let A=0;A<i;A++){let N=0;for(let H=0;H<s;H++)N+=e[f][H]*t[H][A];m[A]=N}a[f]=m}return a}function Xe(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Hn(e,t){const s=e.length-1;return[[e[0]*t[0],"that is only the first product — a dot product accumulates all 16 pairs in the loop"],[Es(e,t)-e[s]*t[s],`the loop stopped one pair short — with k < ${s} the pair a[${s}]·b[${s}] never gets added`]]}function ra(e,t,s,n,i,a){return[[e[i][a]*t[i][a],"that is the elementwise product of the two cells — C[y][x] is the whole dot product of row y of A with column x of B"],[s[a][i],`that is cell [${a}][${i}] of the product — this.thread.y picks A's row and this.thread.x picks B's column`],[n[a][i],"both matrices were read with their indices swapped — the walks are a[this.thread.y][k] across the row and b[k][this.thread.x] down the column"]]}function bs(e,t,s,n,i){return n.map(a=>[Ul(e,t,a),`only the first ${a} of the ${s} shared terms were summed — ${i}`])}function ws(e,t,s){return e.map(n=>[n[0][t][s],n[1]])}function ia(e,t,s){return t<e.length?[[e[t][s],`that is the input cell [${t}][${s}] — a transpose reads with the indices swapped: m[this.thread.x][this.thread.y]`]]:[]}var Vl={id:"2-1",track:2,title:"Matrix Multiply",blurb:"The canonical GPGPU workload: from naive triple loop to a kernel that scales.",tasks:[{slug:"dot-product",title:"One Cell, One Dot Product",intro:`<p>Matrix multiply is the workload GPUs were born for, and every cell of the result
        is the same small machine: a <strong>dot product</strong>. Multiply matching elements of
        two vectors, add the products up, one number comes out. Get one cell right before
        launching a grid of them.</p>
        <p>Notice what is parallel and what is not. The loop over <code>k</code> runs
        <em>sequentially inside one thread</em> — GPUs don't parallelize the sum, they parallelize
        the thousands of <em>independent</em> sums a full matrix needs. This task needs exactly
        one, so the launch is a single thread: <code>output: [1]</code>.</p>`,goal:`<strong>Goal:</strong> make the kernel return the dot product of the 16-vectors
        <code>a</code> and <code>b</code> — one output cell holding
        <code>a[0]·b[0] + a[1]·b[1] + … + a[15]·b[15]</code>.`,requirements:["Change <code>output</code> to a single cell: <code>[1]</code>","Loop <code>k</code> from 0 to 15 <em>inside</em> the kernel — statically bounded loops are allowed","Accumulate <code>a[k] * b[k]</code> into a running sum and return it"],hints:[{title:"Hint 1 — a loop? inside a kernel?",body:`<p>Yes — as long as the bound is a compile-time constant:
            <code>for (let k = 0; k &lt; 16; k++) { … }</code>. The loop belongs to one thread;
            the parallelism (next task) comes from launching many threads that each own a loop.</p>`},{title:"Hint 2 — the whole body",body:`<pre><code>let sum = 0;
for (let k = 0; k &lt; 16; k++) {
  sum += a[k] * b[k];
}
return sum;</code></pre>
<p>— and <code>output: [1]</code> so only one thread runs it.</p>`}],transfer:`Every GPU linear-algebra library — cuBLAS on CUDA, rocBLAS on ROCm, Metal
        Performance Shaders — bottoms out in this exact shape: one output element, one
        multiply-accumulate loop. All their sophistication goes into feeding that loop faster.`,starterCode:`// A dot product folds two 16-vectors into ONE number.
const gpu = new GPU({ mode });

const dot = gpu.createKernel(function (a, b) {
  // TODO: one thread owns the whole sum. Loop k = 0..15,
  // multiply matching elements, add them up, return the total.
  return a[this.thread.x] * b[this.thread.x];
}, {
  // TODO: how many output cells does a dot product have?
  output: [16],
});

console.log(dot(a, b));
`,solutionCode:`// A dot product folds two 16-vectors into ONE number.
const gpu = new GPU({ mode });

const dot = gpu.createKernel(function (a, b) {
  let sum = 0;
  for (let k = 0; k < 16; k++) {
    sum += a[k] * b[k];
  }
  return sum;
}, { output: [1] });

console.log(dot(a, b));
`,inputs:e=>({a:Jt(e,16,1101),b:Jt(e,16,1102)}),publicTests:[{name:"output is a single cell — 1 value, not 16",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(16).fill(1),s=e.kernel(t,t);e.assert(s&&s.length===1,`expected 1 output value, got ${s&&s.length} — a dot product is one number`);const n=Xe(s[0],16,.01,Hn(t,t));e.assertClose(s[0],16,.01,n||"dot of two all-ones vectors should be 16")}},{name:"the sum is right: <code>Σ a[k]·b[k]</code>",run:async e=>{const t=Jt(e.utils,16,1101),s=Jt(e.utils,16,1102),n=e.kernel(t,s),i=Xe(n[0],Es(t,s),.01,Hn(t,s));e.assertClose(n[0],Es(t,s),.01,i||"dot product of the provided vectors")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Jt(e.utils,16,1177),s=Jt(e.utils,16,1178),n=e.kernel(t,s)[0],i=Xe(n,Es(t,s),.01,Hn(t,s));e.assertClose(n,Es(t,s),.01,i||"dot of fresh vectors");const a=new Array(16).fill(0);a[11]=1,e.assertClose(e.kernel(t,a)[0],t[11],.01,"dot with a basis vector picks a[11]")}}]},{slug:"full-matmul",title:"The Full Grid: Matrix × Matrix",intro:`<p>On the CPU, <code>C = A × B</code> is the classic triple loop: over rows, over
        columns, over <code>k</code>. On the GPU the outer two loops <strong>vanish into the
        launch</strong> — <code>output: [16, 16]</code> starts 256 threads, one per cell of
        <code>C</code>, and only the innermost loop survives inside the kernel.</p>
        <p>Cell <code>C[y][x]</code> is the dot product of <strong>row y of A</strong> with
        <strong>column x of B</strong>: walk <code>k</code> across the row
        <code>a[y][k]</code> and down the column <code>b[k][x]</code>. Same loop as task 1 —
        now every thread aims it at its own row/column pair.</p>`,goal:`<strong>Goal:</strong> compute the 16×16 product <code>matA × matB</code> — each
        thread returns the dot product of its row of <code>a</code> with its column of
        <code>b</code>.`,requirements:["Keep <code>output: [16, 16]</code> — one thread per cell of C","Loop <code>k</code> over the 16 shared elements","Accumulate <code>a[this.thread.y][k] * b[k][this.thread.x]</code> and return the sum"],hints:[{title:"Hint 1 — row and column",body:`<p><code>this.thread.y</code> picks the row of <code>a</code>,
            <code>this.thread.x</code> picks the column of <code>b</code>, and <code>k</code> is
            the only index that moves during the loop.</p>`},{title:"Hint 2 — the inner loop",body:`<pre><code>let sum = 0;
for (let k = 0; k &lt; 16; k++) {
  sum += a[this.thread.y][k] * b[k][this.thread.x];
}
return sum;</code></pre>`}],transfer:`This one-thread-per-output-cell matmul is the "naive kernel" every WebGPU and
        CUDA tutorial starts from — and the baseline that tiled, shared-memory versions are
        measured against. The structure you just wrote is their starting point too.`,starterCode:`// output: [16, 16] launches 256 threads — one per cell of C.
const gpu = new GPU({ mode });

const multiply = gpu.createKernel(function (a, b) {
  // TODO: this is the ELEMENTWISE product — one term, no loop.
  // C[y][x] needs the whole dot product: loop k over the 16
  // shared elements, walking a's row and b's column.
  return a[this.thread.y][this.thread.x] * b[this.thread.y][this.thread.x];
}, { output: [16, 16] });

const c = multiply(matA, matB);
console.log('C[0][0] =', c[0][0]);
`,solutionCode:`// output: [16, 16] launches 256 threads — one per cell of C.
const gpu = new GPU({ mode });

const multiply = gpu.createKernel(function (a, b) {
  let sum = 0;
  for (let k = 0; k < 16; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}, { output: [16, 16] });

const c = multiply(matA, matB);
console.log('C[0][0] =', c[0][0]);
`,inputs:e=>({matA:ge(e,16,16,2101),matB:ge(e,16,16,2102)}),publicTests:[{name:"result is a 16×16 grid",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=ge(e.utils,16,16,2101),s=ge(e.utils,16,16,2102),n=e.kernel(t,s);e.assert(n&&n.length===16,`expected 16 rows, got ${n&&n.length}`),e.assert(n[0]&&n[0].length===16,"each row should hold 16 values")}},{name:"cells match the dot product of row × column",run:async e=>{const t=ge(e.utils,16,16,2101),s=ge(e.utils,16,16,2102),n=e.kernel(t,s),i=yt(t,s),a=yt(s,t),f=[[0,0],[3,12],[8,8],[15,1],[15,15]];for(const[m,A]of f){const N=Xe(n[m][A],i[m][A],.01,ra(t,s,i,a,m,A));e.assertClose(n[m][A],i[m][A],.01,N||`cell [${m}][${A}]`)}}},{name:"multiplying by the identity gives A back",run:async e=>{const t=ge(e.utils,16,16,2101),s=e.kernel(t,na(16));for(let n=0;n<16;n++)for(let i=0;i<16;i++)e.assertClose(s[n][i],t[n][i],.01,`cell [${n}][${i}] of A × I`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=ge(e.utils,16,16,2777),s=ge(e.utils,16,16,2778),n=e.kernel(t,s),i=yt(t,s),a=yt(s,t);for(let f=0;f<16;f++)for(let m=0;m<16;m++){const A=Xe(n[f][m],i[f][m],.01,ra(t,s,i,a,f,m));e.assertClose(n[f][m],i[f][m],.01,A||`cell [${f}][${m}]`)}}}]},{slug:"rectangular",title:"Rectangular: Three Different Sizes",intro:`<p>Square matrices hide a trap: every dimension is 16, so any loop bound "works".
        Real matmuls are rectangular — here <code>rectA</code> is 8×32 (8 rows, 32 columns) and
        <code>rectB</code> is 32×12, so the product is <strong>8×12</strong>. Suddenly there are
        three different sizes and each belongs somewhere specific.</p>
        <p>Two of them shape the launch: <code>output: [width, height]</code> = [columns of B,
        rows of A] = <code>[12, 8]</code> — already set up below. The third, 32, is the
        <strong>shared dimension</strong>: A's columns must equal B's rows, and that's the only
        dimension the loop is allowed to run over.</p>`,goal:`<strong>Goal:</strong> compute the 8×12 product <code>rectA × rectB</code> — fix the
        inner loop so it covers the full shared dimension of 32.`,requirements:["Keep <code>output: [12, 8]</code> — columns of B across, rows of A down","Loop <code>k</code> over the <em>shared</em> dimension: all 32 of it","Sum <code>a[this.thread.y][k] * b[k][this.thread.x]</code> as before"],hints:[{title:"Hint 1 — which size does the loop get?",body:`<p>The loop walks <em>across</em> a row of A (32 long) and <em>down</em> a column
            of B (also 32 long — that's why the shapes are compatible). Neither 8 nor 12 appears
            in the loop at all.</p>`},{title:"Hint 2 — the fix",body:`<p>The starter loop stops at 12 — it sums only the first 12 of 32 terms. Change
            the bound: <code>for (let k = 0; k &lt; 32; k++)</code>.</p>`}],transfer:`BLAS calls this M, N, K — <code>sgemm(M, N, K, …)</code> in cuBLAS and rocBLAS
        keeps the three sizes as separate parameters for exactly this reason. Mixing them up is
        the classic GEMM bug on every platform, not just here.`,starterCode:`// (8×32) times (32×12) → 8×12. Three sizes, three different jobs.
const gpu = new GPU({ mode });

const multiply = gpu.createKernel(function (a, b) {
  let sum = 0;
  // TODO: this loop stops too early — it covers 12 of the 32
  // shared elements. Which of the three sizes does the loop own?
  for (let k = 0; k < 12; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}, {
  // [width, height] = [columns of B, rows of A]
  output: [12, 8],
});

const c = multiply(rectA, rectB);
console.log('rows:', c.length, 'cols:', c[0].length);
`,solutionCode:`// (8×32) times (32×12) → 8×12. Three sizes, three different jobs.
const gpu = new GPU({ mode });

const multiply = gpu.createKernel(function (a, b) {
  let sum = 0;
  // k runs over the SHARED dimension: A's columns = B's rows = 32.
  for (let k = 0; k < 32; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}, {
  // [width, height] = [columns of B, rows of A]
  output: [12, 8],
});

const c = multiply(rectA, rectB);
console.log('rows:', c.length, 'cols:', c[0].length);
`,inputs:e=>({rectA:ge(e,8,32,3101),rectB:ge(e,32,12,3102)}),publicTests:[{name:"result is 8 rows × 12 columns",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=ge(e.utils,8,32,3101),s=ge(e.utils,32,12,3102),n=e.kernel(t,s);e.assert(n&&n.length===8,`expected 8 rows, got ${n&&n.length}`),e.assert(n[0]&&n[0].length===12,`expected 12 columns, got ${n[0]&&n[0].length}`)}},{name:"every term counted — all 32 of the shared dimension",run:async e=>{const t=ge(e.utils,8,32,3101),s=ge(e.utils,32,12,3102),n=e.kernel(t,s),i=yt(t,s),a=bs(t,s,32,[12,8],"k has to run over the shared dimension — A's columns and B's rows, not the output's width or height"),f=[[0,0],[2,11],[5,6],[7,0],[7,11]];for(const[m,A]of f){const N=Xe(n[m][A],i[m][A],.02,ws(a,m,A));e.assertClose(n[m][A],i[m][A],.02,N||`cell [${m}][${A}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=ge(e.utils,8,32,3777),s=ge(e.utils,32,12,3778),n=e.kernel(t,s),i=yt(t,s),a=bs(t,s,32,[12,8],"k has to run over the shared dimension — A's columns and B's rows, not the output's width or height");for(let f=0;f<8;f++)for(let m=0;m<12;m++){const A=Xe(n[f][m],i[f][m],.02,ws(a,f,m));e.assertClose(n[f][m],i[f][m],.02,A||`cell [${f}][${m}]`)}}}]},{slug:"transpose",title:"Transpose: Swap the Axes",intro:`<p>Look back at the matmul loop: <code>b[k][x]</code> walks <em>down a column</em> —
        each step jumps a whole row of memory. GPUs hate that; neighbouring threads reading
        neighbouring addresses is where their bandwidth comes from. The standard fix is to
        <strong>transpose</strong> B first, turning column walks into row walks.</p>
        <p>A transpose kernel is one line of insight: the thread that owns output cell
        <code>[y][x]</code> reads input cell <code>[x][y]</code>. With a rectangular 24×40 input
        the flip is visible in the shapes too — the result is 40×24, so
        <code>output: [24, 40]</code>.</p>`,goal:`<strong>Goal:</strong> transpose the 24×40 matrix <code>matWide</code> — output cell
        <code>[y][x]</code> holds <code>matWide[x][y]</code>, giving a 40×24 result.`,requirements:["Keep <code>output: [24, 40]</code> — the transposed width and height","Each thread reads exactly one input cell: indices <em>swapped</em>","No loops — a transpose moves data, it computes nothing"],hints:[{title:"Hint 1 — who reads what",body:`<p>The thread writing output cell <code>[y][x]</code> must read the input cell
            whose row and column are swapped. Both <code>this.thread.x</code> and
            <code>this.thread.y</code> appear — just not in their usual seats.</p>`},{title:"Hint 2 — the one-liner",body:"<p><code>return m[this.thread.x][this.thread.y];</code></p>"}],transfer:`Memory-coalescing is why cuBLAS and rocBLAS pick a different tiled kernel for
        each setting of GEMM's <code>transA/transB</code> flags — whichever layout you pass,
        threads must still read side by side — and why Metal and WebGPU matmul kernels
        pre-stage tiles in threadgroup memory. Reordering data for coalesced access is half of
        GPU performance work.`,starterCode:`// The thread for output [y][x] reads input... where?
const gpu = new GPU({ mode });

const transpose = gpu.createKernel(function (m) {
  // TODO: return the input cell with row and column swapped.
  return 0;
}, {
  // input is 24 rows × 40 cols → output is 40 rows × 24 cols
  output: [24, 40],
});

const t = transpose(matWide);
console.log('rows:', t.length, 'cols:', t[0].length);
`,solutionCode:`// The thread for output [y][x] reads input... where?
const gpu = new GPU({ mode });

const transpose = gpu.createKernel(function (m) {
  return m[this.thread.x][this.thread.y];
}, {
  // input is 24 rows × 40 cols → output is 40 rows × 24 cols
  output: [24, 40],
});

const t = transpose(matWide);
console.log('rows:', t.length, 'cols:', t[0].length);
`,inputs:e=>({matWide:ge(e,24,40,4101)}),publicTests:[{name:"shape flips: 24×40 in, 40×24 out",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=ge(e.utils,24,40,4101),s=e.kernel(t);e.assert(s&&s.length===40,`expected 40 rows, got ${s&&s.length}`),e.assert(s[0]&&s[0].length===24,`expected 24 columns, got ${s[0]&&s[0].length}`)}},{name:"cell [y][x] equals input [x][y]",run:async e=>{const t=ge(e.utils,24,40,4101),s=e.kernel(t),n=[[0,0],[0,23],[39,0],[17,5],[39,23]];for(const[i,a]of n){const f=Xe(s[i][a],t[a][i],.001,ia(t,i,a));e.assertClose(s[i][a],t[a][i],.001,f||`cell [${i}][${a}] should hold input [${a}][${i}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=ge(e.utils,24,40,4777),s=e.kernel(t);for(let n=0;n<40;n++)for(let i=0;i<24;i++){const a=Xe(s[n][i],t[i][n],.001,ia(t,n,i));e.assertClose(s[n][i],t[i][n],.001,a||`cell [${n}][${i}]`)}}}]},{slug:"any-size",title:"One Kernel, Any Size",intro:`<p>Every kernel so far had its size welded on: <code>output: [16, 16]</code>, loop
        to 16. Real code multiplies whatever matrices show up. gpu.js has three switches for
        that: <code>dynamicOutput: true</code> lets you call <code>kernel.setOutput([n, n])</code>
        before each run, <code>dynamicArguments: true</code> lets the input arrays change size
        between calls, and <code>loopMaxIterations</code> raises the safety cap so the loop bound
        can be a <em>runtime argument</em> instead of a constant.</p>
        <p>Pass the size in as a plain number, loop <code>k &lt; size</code>, and one kernel
        object serves an 8×8 and a 48×48 multiply back to back. This is the payoff of the module:
        the naive triple loop from task 2, now packaged as a function that scales.</p>`,goal:`<strong>Goal:</strong> make <code>multiply(a, b)</code> work for any square size up
        to 64 using a <em>single</em> kernel — verify it on the 8×8 and 48×48 pairs provided.`,requirements:["Kernel options: <code>dynamicOutput</code>, <code>dynamicArguments</code>, and <code>loopMaxIterations: 64</code>","Take <code>size</code> as a third kernel argument and loop <code>k &lt; size</code>","In <code>multiply</code>, call <code>matmul.setOutput([n, n])</code> before invoking","Exactly one <code>createKernel</code> call serves both sizes"],hints:[{title:"Hint 1 — why the cap?",body:`<p>On the GPU backend a loop bound that isn't a compile-time constant becomes</p>
<pre><code>for (i = 0; i &lt; LOOP_MAX; i++) {
  if (!(i &lt; size)) break;
  // …
}</code></pre>
<p>in the shader — <code>loopMaxIterations</code> <em>is</em> that LOOP_MAX. Set it to the
            largest size you'll ever pass: 64 here.</p>`},{title:"Hint 2 — sizing per call",body:`<p>Inside <code>multiply</code>:</p>
<pre><code>const n = a.length;
matmul.setOutput([n, n]);
return matmul(a, b, n);</code></pre>
<p>— set the launch shape first,
            then invoke with the size as the last argument.</p>`},{title:"Hint 3 — the kernel",body:`<pre><code>function (a, b, size) {
  let sum = 0;
  for (let k = 0; k &lt; size; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}</code></pre>
<p>with options</p>
<pre><code>{
  dynamicOutput: true,
  dynamicArguments: true,
  loopMaxIterations: 64,
}</code></pre>`}],transfer:`Shipping one kernel that covers a size range is standard practice everywhere:
        CUDA kernels take M, N, K as launch parameters and pick grid dimensions at call time,
        WebGPU dispatches a runtime-computed number of workgroups, and Metal binds sizes through
        a constant buffer. Compile once, launch at any size — exactly what you just built.`,starterCode:`// One kernel, any size — no rebuilding between calls.
const gpu = new GPU({ mode });

// TODO: this kernel is welded to 8×8. Free it: dynamicOutput,
// dynamicArguments, loopMaxIterations: 64, and a size argument.
const matmul = gpu.createKernel(function (a, b) {
  let sum = 0;
  for (let k = 0; k < 8; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}, { output: [8, 8] });

function multiply(a, b) {
  const n = a.length;
  // TODO: point the kernel at an n×n launch before invoking,
  // and pass n in so the loop knows where to stop.
  return matmul(a, b);
}

console.log('8×8  C[0][0] =', multiply(smallA, smallB)[0][0]);
console.log('48×48 C[0][0] =', multiply(bigA, bigB)[0][0]);
`,solutionCode:`// One kernel, any size — no rebuilding between calls.
const gpu = new GPU({ mode });

const matmul = gpu.createKernel(function (a, b, size) {
  let sum = 0;
  for (let k = 0; k < size; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}, {
  dynamicOutput: true,
  dynamicArguments: true,
  loopMaxIterations: 64,
});

function multiply(a, b) {
  const n = a.length;
  matmul.setOutput([n, n]);
  return matmul(a, b, n);
}

console.log('8×8  C[0][0] =', multiply(smallA, smallB)[0][0]);
console.log('48×48 C[0][0] =', multiply(bigA, bigB)[0][0]);
`,inputs:e=>({smallA:ge(e,8,8,5101),smallB:ge(e,8,8,5102),bigA:ge(e,48,48,5103),bigB:ge(e,48,48,5104)}),publicTests:[{name:"one kernel serves both sizes",run:async e=>{e.assert(e.kernels.length===1,`expected exactly 1 kernel to handle every size, found ${e.kernels.length}`)}},{name:"8×8 product is correct",run:async e=>{const t=ge(e.utils,8,8,5101),s=ge(e.utils,8,8,5102);e.kernel.setOutput([8,8]);const n=e.kernel(t,s,8),i=yt(t,s);for(let a=0;a<8;a++)for(let f=0;f<8;f++)e.assertClose(n[a][f],i[a][f],.02,`cell [${a}][${f}]`)}},{name:"48×48 product is correct — same kernel, bigger launch",run:async e=>{const t=ge(e.utils,48,48,5103),s=ge(e.utils,48,48,5104);e.kernel.setOutput([48,48]);const n=e.kernel(t,s,48);e.assert(n.length===48&&n[0].length===48,"expected a 48×48 result");const i=yt(t,s),a=bs(t,s,48,[8],"the loop bound has to be the size argument, not the literal the kernel was born with"),f=[[0,0],[7,33],[24,24],[40,3],[47,47]];for(const[m,A]of f){const N=Xe(n[m][A],i[m][A],.05,ws(a,m,A));e.assertClose(n[m][A],i[m][A],.05,N||`cell [${m}][${A}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=ge(e.utils,32,32,5777),s=ge(e.utils,32,32,5778);e.kernel.setOutput([32,32]);const n=e.kernel(t,s,32),i=yt(t,s),a=bs(t,s,32,[8],"the loop bound has to be the size argument, not the literal the kernel was born with");for(let f=0;f<32;f++)for(let m=0;m<32;m++){const A=Xe(n[f][m],i[f][m],.05,ws(a,f,m));e.assertClose(n[f][m],i[f][m],.05,A||`cell [${f}][${m}]`)}}},{name:"private test #2",run:async e=>{const t=ge(e.utils,16,16,5888),s=na(16);e.kernel.setOutput([16,16]);const n=e.kernel(t,s,16),i=bs(t,s,16,[8],"the loop bound has to be the size argument, not the literal the kernel was born with");for(let a=0;a<16;a++)for(let f=0;f<16;f++){const m=Xe(n[a][f],t[a][f],.02,ws(i,a,f));e.assertClose(n[a][f],t[a][f],.02,m||`cell [${a}][${f}] of A × I`)}}}]}]},Kl=Object.freeze({__proto__:null,default:Vl});function Pe(e,t,s=2207){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++)i[a]=Math.round(n()*2e3)/1e3;return i}function Ye(e){let t=0;for(let s=0;s<e.length;s++)t+=e[s];return t}function aa(e){let t=e[0];for(let s=1;s<e.length;s++)e[s]<t&&(t=e[s]);return t}function oa(e){let t=e[0];for(let s=1;s<e.length;s++)e[s]>t&&(t=e[s]);return t}function qa(e,t,s,n){let i=0;for(let a=0;a<n;a++)i+=e[a*s+t];return i}function lt(e,t){let s=t instanceof Float32Array?t:Float32Array.from(t),n=s.length;for(;n>1;)n=n/2,e.setOutput([n]),s=e(s);return s[0]}function Nl(e,t,s){let n=0;for(let i=0;i<s;i++)n+=e[t*s+i];return n}function St(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function la(e,t,s,n){return[[Nl(e,t,n),"that is the sum of a contiguous block — the strided walk is data[i * this.constants.threads + this.thread.x], so neighbouring threads touch neighbouring elements"],[qa(e,0,s,n),"every thread summed thread 0's slice — this.thread.x has to appear in the index"]]}function en(e,t){return[[e[t]+e[t+1],"you paired with your immediate neighbour — the partner sits one output width away: data[this.thread.x + this.output.x]"],[e[t],"only your own element came back — the partner in the top half never got added"]]}function Xn(e){const t=new Array(4096).fill(2);for(const s of e.kernels){if(!s.kernel||s.kernel.dynamicOutput)continue;let n;try{n=s(t)}catch{continue}if(n&&n.length===64&&Math.abs(n[0]-16384)<=.01)return"one kernel squared its finished partial sum — the square belongs on each value as it is read, inside the loop"}return null}function Yn(e){const t=[];for(const s of e){if(s.type!=="log"||!s.text)continue;const n=s.text.match(/-?\d+(?:\.\d+)?/g);if(n)for(const i of n)t.push(parseFloat(i))}return t}function Zt(e){return e.kernels.find(t=>t.kernel&&t.kernel.dynamicOutput)||null}function Jn(e){const t=new Array(4096).fill(2);let s=null,n=null;for(const i of e.kernels){if(!i.kernel||i.kernel.dynamicOutput)continue;let a;try{a=i(t)}catch{continue}!a||a.length!==64||(Math.abs(a[0]-128)<=.01?s=i:Math.abs(a[0]-256)<=.01&&(n=i))}return{sums:s,squares:n}}var Bl={id:"2-2",track:2,title:"Reductions",blurb:"Sum, min, max and mean over millions of values — the ladder pattern every platform uses.",tasks:[{slug:"one-thread-sum",title:"The One-Thread Trap",intro:`<p>Meet the <strong>reduction</strong>: many values in, one value out — sum, min,
        max, mean. It's the awkward case in GPU land, because a kernel thread writes exactly
        <em>one</em> output cell. 4,096 inputs collapsing to 1 output means
        <code>output: [1]</code>… a single thread.</p>
        <p>You <em>can</em> do it — kernels may loop, as long as the bound is known at compile
        time, which is exactly what <code>this.constants</code> is for. But one thread grinding
        through 4,096 additions while thousands of its neighbours sit idle is the slowest
        possible way to use a GPU. Write it anyway: it's the baseline the rest of this module
        tears down.</p>`,goal:`<strong>Goal:</strong> make the single thread loop over all of <code>data</code>
        (bound: <code>this.constants.n</code>) and return the total.`,requirements:["Keep <code>output: [1]</code> — one thread owns the one output cell","Loop <code>for (let i = 0; i &lt; this.constants.n; i++)</code> — in gpu.js's WebGL backend, loop bounds must be compile-time constants","Accumulate into a local <code>let sum</code> and return it"],hints:[{title:"Hint 1 — an accumulator",body:`<p>Declare <code>let sum = 0;</code> before the loop, add to it inside the loop,
            and <code>return sum;</code> after. Plain JavaScript — the transpiler handles it.</p>`},{title:"Hint 2 — the loop body",body:"<p>One statement: <code>sum += data[i];</code></p>"}],transfer:`This wall exists on every platform: a single CUDA thread summing a whole buffer
        is the textbook example of what <em>not</em> to do, and a naive WebGPU compute shader
        with one invocation hits it just the same. Everyone's escape route is the trick you
        build next — split the work, then combine.`,starterCode:`// 4096 values, ONE output cell — so exactly one thread does everything.
const gpu = new GPU({ mode });

const sumAll = gpu.createKernel(function (data) {
  // TODO: loop i from 0 to this.constants.n, accumulate data[i]
  // into a local sum, and return it.
  return 0;
}, {
  output: [1],
  constants: { n: 4096 },
});

console.log('total:', sumAll(data)[0]);
`,solutionCode:`// 4096 values, ONE output cell — so exactly one thread does everything.
const gpu = new GPU({ mode });

const sumAll = gpu.createKernel(function (data) {
  let sum = 0;
  for (let i = 0; i < this.constants.n; i++) {
    sum += data[i];
  }
  return sum;
}, {
  output: [1],
  constants: { n: 4096 },
});

console.log('total:', sumAll(data)[0]);
`,inputs:e=>({data:Pe(e,4096)}),publicTests:[{name:"one output cell holds the sum of 4096 ones",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(new Array(4096).fill(1));e.assert(t&&t.length===1,`expected 1 output value, got ${t&&t.length}`),e.assertClose(t[0],4096,.5,"sum of 4096 ones")}},{name:"the total matches on fresh data",run:async e=>{const t=new Array(4096);for(let i=0;i<4096;i++)t[i]=i*7%13*.125;const s=e.kernel(t),n=St(s[0],Ye(t),2,[[4096*t[0],"the loop added the same element 4096 times — the accumulation has to index with the loop variable: data[i]"]]);e.assertClose(s[0],Ye(t),2,n||"the total")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Pe(e.utils,4096,4242),s=e.kernel(t);e.assert(s&&s.length===1,"expected 1 output value");const n=St(s[0],Ye(t),2,[[4096*t[0],"the loop added the same element 4096 times — the accumulation has to index with the loop variable: data[i]"]]);e.assertClose(s[0],Ye(t),2,n||"the total")}}]},{slug:"partial-sums",title:"Partial Sums: Divide the Work",intro:`<p>The fix: give <em>every</em> thread a slice. 64 threads, each summing 64 of the
        4,096 values, produce 64 <strong>partial sums</strong> — and 64 leftover numbers are
        cheap to finish off in plain JavaScript.</p>
        <p>Watch the reading pattern, though. Thread <code>x</code> does <em>not</em> take a
        contiguous block; it reads <code>data[x]</code>, <code>data[x + 64]</code>,
        <code>data[x + 128]</code>, … — a <strong>strided</strong> walk. At every step of the
        loop, neighbouring threads touch neighbouring elements, which is exactly the access
        pattern GPU memory hardware is built to serve in one go.</p>`,goal:`<strong>Goal:</strong> compute 64 strided partial sums on the GPU, then total the
        64 partials in JavaScript and log the grand total.`,requirements:["Each of the 64 threads loops <code>this.constants.chunk</code> times","Strided reads: element <code>i</code> of thread <code>x</code> is <code>data[i * this.constants.threads + this.thread.x]</code>","Sum the 64 returned partials in plain JavaScript and <code>console.log</code> the total"],hints:[{title:"Hint 1 — which elements are mine?",body:`<p>Thread <code>x</code> owns elements <code>x</code>, <code>x + 64</code>,
            <code>x + 128</code>, … so its <code>i</code>-th element sits at index
            <code>i * 64 + x</code>.</p>`},{title:"Hint 2 — the loop body",body:"<pre><code>sum += data[i * this.constants.threads + this.thread.x];</code></pre>"},{title:"Hint 3 — finishing in JS",body:`<p>After <code>const partial = partials(data);</code> a plain loop does it:</p>
<pre><code>let total = 0;
for (let i = 0; i &lt; partial.length; i++) {
  total += partial[i];
}</code></pre>`}],transfer:`This is CUDA's <em>grid-stride loop</em>, almost line for line — every serious
        reduction in CUB and Thrust starts with per-thread partials accumulated in registers,
        and coalesced (strided-by-thread-count) reads are the whole reason for the pattern.
        WebGPU and Metal compute kernels stage the same partials into workgroup/threadgroup
        memory.`,starterCode:`// 64 threads, 64 values each. Strided reads keep the memory hardware happy.
const gpu = new GPU({ mode });

const partials = gpu.createKernel(function (data) {
  // TODO: loop this.constants.chunk times and accumulate this thread's
  // strided slice: data[i * this.constants.threads + this.thread.x]
  return 0;
}, {
  output: [64],
  constants: { threads: 64, chunk: 64 },
});

const partial = partials(data);
console.log('partials:', partial.length);

let total = 0;
for (let i = 0; i < partial.length; i++) total += partial[i];
console.log('total:', total);
`,solutionCode:`// 64 threads, 64 values each. Strided reads keep the memory hardware happy.
const gpu = new GPU({ mode });

const partials = gpu.createKernel(function (data) {
  let sum = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    sum += data[i * this.constants.threads + this.thread.x];
  }
  return sum;
}, {
  output: [64],
  constants: { threads: 64, chunk: 64 },
});

const partial = partials(data);
console.log('partials:', partial.length);

let total = 0;
for (let i = 0; i < partial.length; i++) total += partial[i];
console.log('total:', total);
`,inputs:e=>({data:Pe(e,4096,707)}),publicTests:[{name:"64 partial sums — all-ones input gives 64 everywhere",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(new Array(4096).fill(1));e.assert(t&&t.length===64,`expected 64 partial sums, got ${t&&t.length}`);for(let s=0;s<64;s++)e.assertClose(t[s],64,.001,`partial ${s} should sum 64 ones`)}},{name:"partials are strided — thread x sums <code>data[x], data[x + 64], …</code>",run:async e=>{const t=new Array(4096);for(let n=0;n<4096;n++)t[n]=n;const s=e.kernel(t);for(const n of[0,1,31,63]){const i=St(s[n],129024+64*n,.5,la(t,n,64,64));e.assertClose(s[n],129024+64*n,.5,i||`partial ${n} should sum data[${n}], data[${n} + 64], data[${n} + 128], …`)}}},{name:"the grand total is computed and logged",run:async e=>{const t=Ye(Pe(e.utils,4096,707)),s=Yn(e.logs);e.assert(s.some(n=>Math.abs(n-t)<=.5),`log the total of the partials — expected ≈${t.toFixed(2)} in the console output`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=Pe(e.utils,4096,555),s=e.kernel(t);e.assert(s&&s.length===64,"expected 64 partial sums");for(let n=0;n<64;n++){const i=qa(t,n,64,64),a=St(s[n],i,.02,la(t,n,64,64));e.assertClose(s[n],i,.02,a||`partial ${n}`)}e.assertClose(Ye(Array.from(s)),Ye(t),.5,"total of the partials")}}]},{slug:"halving-step",title:"One Rung of the Ladder",intro:`<p>Sixty-four partials finished in JavaScript is fine. A million wouldn't be. To
        stay parallel all the way down, GPUs fold an array onto itself: add each element in the
        <em>top half</em> to its partner in the <em>bottom half</em>, and 512 values become 256
        in a single parallel step. That's one rung of the <strong>halving ladder</strong> —
        every reduction library on every platform is built from this move.</p>
        <p>One kernel invocation = one rung. Each thread adds exactly one pair:
        <code>data[x] + data[x + half]</code>. And <code>half</code> comes for free — the fold
        distance is just the output length, <code>this.output.x</code>.</p>`,goal:`<strong>Goal:</strong> write the rung kernel — fold 512 values into 256 pair sums,
        preserving the total.`,requirements:["<code>output: [256]</code> — one thread per pair","Each thread adds its own element to its partner one output-width away","The fold preserves the total: the 256 outputs sum to the same value as the 512 inputs"],hints:[{title:"Hint 1 — how far away is my partner?",body:`<p>With 512 inputs and 256 outputs, thread <code>x</code> pairs with element
            <code>x + 256</code> — and 256 is exactly <code>this.output.x</code>, the width of
            the output.</p>`},{title:"Hint 2 — the one-liner",body:"<pre><code>return data[this.thread.x] + data[this.thread.x + this.output.x];</code></pre>"}],transfer:`The halving fold is the heart of every tree reduction: CUDA's classic
        shared-memory reduction halves its stride once per barrier, and WGSL subgroup ops or
        Metal's <code>simd_sum</code> are the same fold executed inside the hardware. One rung
        here equals one barrier-separated step there.`,starterCode:`// Fold the top half onto the bottom half: 512 values in, 256 out.
const gpu = new GPU({ mode });

const halve = gpu.createKernel(function (data) {
  // TODO: add this thread's element to its partner in the top half.
  // The fold distance is this.output.x.
  return data[this.thread.x];
}, {
  output: [256],
});

const folded = halve(data);
console.log('folded length:', folded.length);
console.log('first pair sum:', folded[0]);
`,solutionCode:`// Fold the top half onto the bottom half: 512 values in, 256 out.
const gpu = new GPU({ mode });

const halve = gpu.createKernel(function (data) {
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, {
  output: [256],
});

const folded = halve(data);
console.log('folded length:', folded.length);
console.log('first pair sum:', folded[0]);
`,inputs:e=>({data:Pe(e,512,1131)}),publicTests:[{name:"one rung: 512 values fold to 256",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(Pe(e.utils,512,1131));e.assert(t&&t.length===256,`expected 256 values after the fold, got ${t&&t.length}`)}},{name:"cell x holds <code>data[x] + data[x + 256]</code>",run:async e=>{const t=new Array(512);for(let n=0;n<512;n++)t[n]=n;const s=e.kernel(t);for(let n=0;n<256;n++){const i=St(s[n],2*n+256,.001,en(t,n));e.assertClose(s[n],2*n+256,.001,i||`cell ${n}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Pe(e.utils,512,9091),s=e.kernel(t);e.assert(s&&s.length===256,"expected 256 values after the fold");for(let n=0;n<256;n++){const i=t[n]+t[n+256],a=St(s[n],i,.001,en(t,n));e.assertClose(s[n],i,.001,a||`cell ${n}`)}e.assertClose(Ye(Array.from(s)),Ye(t),.05,"the fold must preserve the total")}}]},{slug:"ladder-to-scalar",title:"Ride the Ladder Down",intro:`<p>Now ride it all the way: 1,024 → 512 → 256 → … → 1. Ten rungs and the array is
        a scalar. That means the <em>same</em> kernel has to run at a different size on every
        call — two options make that legal: <code>dynamicOutput: true</code> lets
        <code>setOutput()</code> shrink the thread grid between calls, and
        <code>dynamicArguments: true</code> lets the input shrink with it.</p>
        <p>The driving loop lives in JavaScript, but every rung of actual work stays parallel
        on the GPU: log₂(1024) = 10 launches instead of 1,023 serial additions. One real-world
        wrinkle, already wired into the driver: gpu.js locks an argument's <em>type</em> on the
        kernel's first call, so the ladder starts from a <code>Float32Array</code> — the same
        type every rung's output comes back as.</p>`,goal:`<strong>Goal:</strong> reduce the 1,024 values of <code>data</code> to a single
        total by iterating the halving rung, and log the result.`,requirements:["Create the rung kernel with <code>dynamicOutput: true</code> and <code>dynamicArguments: true</code>","Fold pairs with <code>this.output.x</code>, exactly like the last task","Loop in JS: while <code>n &gt; 1</code>, halve <code>n</code>, <code>setOutput([n])</code>, re-invoke","<code>console.log</code> the final scalar"],hints:[{title:"Hint 1 — resizing a kernel",body:`<p><code>halve.setOutput([n])</code> takes the new output shape as an array.
            Call it before each invocation, with <code>n</code> already halved.</p>`},{title:"Hint 2 — the driver skeleton",body:`<pre><code>let n = values.length;
while (n &gt; 1) {
  n = n / 2;
  // …
}</code></pre>
<p>— inside the loop, resize, re-invoke, and keep the returned array for the next
            rung.</p>`},{title:"Hint 3 — the full driver",body:`<pre><code>while (n &gt; 1) {
  n = n / 2;
  halve.setOutput([n]);
  values = halve(values);
}</code></pre>
<p>— then the answer is <code>values[0]</code>.</p>`}],transfer:`Multi-pass reduction is the production pattern everywhere: CUDA launches a
        shrinking sequence of grids (or grid-syncs with cooperative groups), WebGPU records
        repeated dispatches ping-ponging between two buffers, Metal encodes one compute pass
        per rung. The log₂(n) staircase is identical on all of them.`,starterCode:`// Same rung as before — but dynamic, so it can shrink call by call.
const gpu = new GPU({ mode });

const halve = gpu.createKernel(function (data) {
  // TODO: fold this thread's pair, exactly like the last task
  return data[this.thread.x];
}, {
  dynamicOutput: true,
  dynamicArguments: true,
});

// Start from a Float32Array: gpu.js locks an argument's type on the first
// call, and every rung's output comes back as a Float32Array.
let values = Float32Array.from(data);
let n = values.length;
while (n > 1) {
  n = n / 2;
  halve.setOutput([n]);
  values = halve(values);
}
console.log('total:', values[0]);
`,solutionCode:`// Same rung as before — but dynamic, so it can shrink call by call.
const gpu = new GPU({ mode });

const halve = gpu.createKernel(function (data) {
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, {
  dynamicOutput: true,
  dynamicArguments: true,
});

// Start from a Float32Array: gpu.js locks an argument's type on the first
// call, and every rung's output comes back as a Float32Array.
let values = Float32Array.from(data);
let n = values.length;
while (n > 1) {
  n = n / 2;
  halve.setOutput([n]);
  values = halve(values);
}
console.log('total:', values[0]);
`,inputs:e=>({data:Pe(e,1024,2024)}),publicTests:[{name:"the rung is dynamic and folds pairs",run:async e=>{const t=Zt(e);e.assert(t,"no kernel with dynamicOutput: true found — pass it in the kernel options"),t.setOutput([2]);const s=[1,2,3,4],n=t(Float32Array.from(s));e.assert(n&&n.length===2,`expected 2 values after one rung, got ${n&&n.length}`);const i=St(n[0],4,.001,en(s,0));e.assertClose(n[0],4,.001,i||"cell 0 should fold data[0] + data[2]");const a=St(n[1],6,.001,en(s,1));e.assertClose(n[1],6,.001,a||"cell 1 should fold data[1] + data[3]")}},{name:"the ladder reduces 1024 fresh values to their sum",run:async e=>{const t=Zt(e);e.assert(t,"no kernel with dynamicOutput: true found");const s=new Array(1024);for(let n=0;n<1024;n++)s[n]=n%10*.25;e.assertClose(lt(t,s),Ye(s),.1,"the ladder total")}},{name:"the final scalar is logged",run:async e=>{const t=Ye(Pe(e.utils,1024,2024)),s=Yn(e.logs);e.assert(s.some(n=>Math.abs(n-t)<=.5),`log the final total — expected ≈${t.toFixed(2)} in the console output`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=Zt(e);e.assert(t,"expected a dynamicOutput kernel");const s=Pe(e.utils,256,40961);e.assertClose(lt(t,s),Ye(s),.1,"ladder total on 256 values")}}]},{slug:"min-max",title:"Min and Max: Change the Operator",intro:`<p>Here's the secret hiding inside the ladder: nothing about it is really about
        <em>addition</em>. Any operation that combines two values and doesn't care about order
        or grouping — associative and commutative — can ride the same ladder. Swap
        <code>+</code> for <code>Math.min</code> and the scalar at the bottom is the smallest
        value in the array. <code>Math.max</code> gives the largest.</p>
        <p>Two kernels, one driver. The structure doesn't change at all — only the fold
        rule.</p>`,goal:`<strong>Goal:</strong> find both the minimum and the maximum of <code>data</code>
        with two halving-ladder kernels, and log both.`,requirements:["<code>minStep</code> folds with <code>Math.min</code>, <code>maxStep</code> with <code>Math.max</code>","Both kernels use <code>dynamicOutput: true</code> and <code>dynamicArguments: true</code>","Ride each ladder down to a scalar and <code>console.log</code> both results"],hints:[{title:"Hint 1 — Math inside kernels",body:`<p><code>Math.min(a, b)</code> and <code>Math.max(a, b)</code> both work inside
            kernel functions. The fold becomes</p>
<pre><code>Math.min(data[this.thread.x], data[this.thread.x + this.output.x])</code></pre>`},{title:"Hint 2 — one driver, two ladders",body:`<p>Wrap last task's while-loop in a plain JS function that takes the kernel as
            a parameter — <code>reduce(minStep, data)</code>, <code>reduce(maxStep, data)</code>
            — instead of writing it twice.</p>`}],transfer:`Pluggable operators are why every library ships reduce as a higher-order
        function: <code>thrust::reduce</code> and ROCm's rocPRIM accept any binary op plus an
        identity value, Metal Performance Shaders sells min/max reductions pre-built, and
        WGSL's <code>subgroupMin</code>/<code>subgroupMax</code> are this exact ladder burned
        into silicon.`,starterCode:`// Same ladder, new fold rule. Only the operator changes.
const gpu = new GPU({ mode });

const minStep = gpu.createKernel(function (data) {
  // TODO: keep the SMALLER of the pair, not the sum
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, { dynamicOutput: true, dynamicArguments: true });

const maxStep = gpu.createKernel(function (data) {
  // TODO: keep the LARGER of the pair
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, { dynamicOutput: true, dynamicArguments: true });

function reduce(step, values) {
  // Float32Array from the start — an argument's type is locked on first call.
  let v = Float32Array.from(values);
  let n = v.length;
  while (n > 1) {
    n = n / 2;
    step.setOutput([n]);
    v = step(v);
  }
  return v[0];
}

console.log('min:', reduce(minStep, data));
console.log('max:', reduce(maxStep, data));
`,solutionCode:`// Same ladder, new fold rule. Only the operator changes.
const gpu = new GPU({ mode });

const minStep = gpu.createKernel(function (data) {
  return Math.min(data[this.thread.x], data[this.thread.x + this.output.x]);
}, { dynamicOutput: true, dynamicArguments: true });

const maxStep = gpu.createKernel(function (data) {
  return Math.max(data[this.thread.x], data[this.thread.x + this.output.x]);
}, { dynamicOutput: true, dynamicArguments: true });

function reduce(step, values) {
  // Float32Array from the start — an argument's type is locked on first call.
  let v = Float32Array.from(values);
  let n = v.length;
  while (n > 1) {
    n = n / 2;
    step.setOutput([n]);
    v = step(v);
  }
  return v[0];
}

console.log('min:', reduce(minStep, data));
console.log('max:', reduce(maxStep, data));
`,inputs:e=>({data:Pe(e,1024,5150)}),publicTests:[{name:"one ladder keeps the smaller value, one the larger",run:async e=>{let t=null,s=null;for(const n of e.kernels){if(!n.kernel||!n.kernel.dynamicOutput)continue;n.setOutput([1]);const i=n(Float32Array.from([3,5]))[0],a=n(Float32Array.from([8,2]))[0];Math.abs(i-3)<.001&&Math.abs(a-2)<.001&&(t=n),Math.abs(i-5)<.001&&Math.abs(a-8)<.001&&(s=n)}e.assert(t,"no min ladder found — one kernel should fold with Math.min"),e.assert(s,"no max ladder found — one kernel should fold with Math.max")}},{name:"min and max of a fresh 512-value array",run:async e=>{const t=e.utils.seededRandom(88),s=new Array(512);for(let a=0;a<512;a++)s[a]=Math.round(t()*4e3)/1e3-2;let n=null,i=null;for(const a of e.kernels){if(!a.kernel||!a.kernel.dynamicOutput)continue;a.setOutput([1]);const f=a(Float32Array.from([3,5]))[0];Math.abs(f-3)<.001?n=a:Math.abs(f-5)<.001&&(i=a)}e.assert(n&&i,"expected a Math.min ladder and a Math.max ladder"),e.assertClose(lt(n,s),aa(s),.001,"the minimum"),e.assertClose(lt(i,s),oa(s),.001,"the maximum")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Pe(e.utils,1024,31337);let s=null,n=null;for(const i of e.kernels){if(!i.kernel||!i.kernel.dynamicOutput)continue;i.setOutput([1]);const a=i(Float32Array.from([-4,9]))[0];Math.abs(a- -4)<.001?s=i:Math.abs(a-9)<.001&&(n=i)}e.assert(s&&n,"expected a Math.min ladder and a Math.max ladder"),e.assertClose(lt(s,t),aa(t),.001,"the minimum"),e.assertClose(lt(n,t),oa(t),.001,"the maximum")}}]},{slug:"fused-mean-rms",title:"Payoff: Mean and RMS, Fused",intro:`<p>The payoff. Two statistics over 4,096 values: the <strong>mean</strong>
        (sum ÷ n) and the <strong>RMS</strong> — root-mean-square,
        √(sum&nbsp;of&nbsp;squares&nbsp;÷&nbsp;n) — the standard "how big is this signal"
        measure in audio and physics.</p>
        <p>RMS needs every value squared first. The rookie move is a separate squaring kernel —
        a whole extra pass over memory. The pro move is <strong>fusion</strong>: square each
        value in the same statement that reads it, inside the partial-sum kernel. Map and
        reduce, one pass over the data.</p>
        <p>Stack the whole module: strided partials (task 2) shrink 4,096 values to 64, then a
        single shared halving ladder (task 4) finishes <em>both</em> totals.</p>`,goal:`<strong>Goal:</strong> compute and log the mean and the RMS of <code>data</code> —
        two partial-sum kernels (one fused with squaring) plus one shared dynamic halving
        ladder.`,requirements:["<code>partialSums</code>: 64 strided partial sums of <code>data</code>, as in task 2","<code>partialSquares</code>: same shape, but square each value <em>as it is read</em> — no separate squaring pass","One dynamic halving-ladder kernel rides both 64-value arrays down to scalars","<code>mean = total / 4096</code>, <code>rms = Math.sqrt(totalSq / 4096)</code> — log both"],hints:[{title:"Hint 1 — the fused body",body:`<p>Read once, use twice:</p>
<pre><code>const v = data[i * this.constants.threads + this.thread.x];
sum += v * v;</code></pre>`},{title:"Hint 2 — one ladder, two rides",body:`<p>The ladder kernel doesn't care what its 64 inputs mean. Wrap the driver loop
            in a function and call it once with each partials array.</p>`},{title:"Hint 3 — the whole shape",body:`<pre><code>const total = ladder(partialSums(data));
const totalSq = ladder(partialSquares(data));</code></pre>
<p>then divide, square-root,
            and log.</p>`}],transfer:`Fusing the map into the reduce is a marquee optimization on every platform:
        <code>thrust::transform_reduce</code> exists precisely for it, CUDA programmers
        hand-fuse to halve their memory traffic, and WebGPU/Metal kernels bake the transform
        into the accumulation loop. Memory bandwidth is the budget — fusion is the
        discount.`,starterCode:`// Everything in one pipeline: partials → shared ladder → two statistics.
const gpu = new GPU({ mode });

const partialSums = gpu.createKernel(function (data) {
  // TODO: strided partial sums, exactly like task 2
  return 0;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

const partialSquares = gpu.createKernel(function (data) {
  // TODO: same walk, but square each value AS you read it (fusion!)
  return 0;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

// One rung, reused for both reductions.
const halve = gpu.createKernel(function (data) {
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, { dynamicOutput: true, dynamicArguments: true });

function ladder(values) {
  let v = values;
  let n = v.length;
  while (n > 1) {
    n = n / 2;
    halve.setOutput([n]);
    v = halve(v);
  }
  return v[0];
}

const total = ladder(partialSums(data));
const totalSq = ladder(partialSquares(data));

const mean = total / 4096;
const rms = Math.sqrt(totalSq / 4096);
console.log('mean:', mean);
console.log('rms:', rms);
`,solutionCode:`// Everything in one pipeline: partials → shared ladder → two statistics.
const gpu = new GPU({ mode });

const partialSums = gpu.createKernel(function (data) {
  let sum = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    sum += data[i * this.constants.threads + this.thread.x];
  }
  return sum;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

// Fused map + reduce: the square happens in the same statement as the read.
const partialSquares = gpu.createKernel(function (data) {
  let sum = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    const v = data[i * this.constants.threads + this.thread.x];
    sum += v * v;
  }
  return sum;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

// One rung, reused for both reductions.
const halve = gpu.createKernel(function (data) {
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, { dynamicOutput: true, dynamicArguments: true });

function ladder(values) {
  let v = values;
  let n = v.length;
  while (n > 1) {
    n = n / 2;
    halve.setOutput([n]);
    v = halve(v);
  }
  return v[0];
}

const total = ladder(partialSums(data));
const totalSq = ladder(partialSquares(data));

const mean = total / 4096;
const rms = Math.sqrt(totalSq / 4096);
console.log('mean:', mean);
console.log('rms:', rms);
`,inputs:e=>({data:Pe(e,4096,6001)}),publicTests:[{name:"three kernels: plain partials, fused squared partials, dynamic ladder",run:async e=>{const{sums:t,squares:s}=Jn(e);e.assert(t,"no kernel producing 64 partial sums found (all-2s input should give 128 per thread)");const n=s?null:Xn(e);e.assert(s,n||"no fused kernel producing 64 partial sums of squares found (all-2s input should give 256 per thread)"),e.assert(Zt(e),"no dynamicOutput halving-ladder kernel found")}},{name:"full pipeline: mean and RMS of a fresh array",run:async e=>{const{sums:t,squares:s}=Jn(e),n=Zt(e),i=s?null:Xn(e);e.assert(t&&s&&n,i||"expected partialSums, partialSquares and a dynamic ladder kernel");const a=new Array(4096);for(let H=0;H<4096;H++)a[H]=(H%8+1)/4;let f=0,m=0;for(let H=0;H<4096;H++)f+=a[H],m+=a[H]*a[H];const A=lt(n,t(a)),N=lt(n,s(a));e.assertClose(A/4096,f/4096,.001,"the mean"),e.assertClose(Math.sqrt(N/4096),Math.sqrt(m/4096),.001,"the RMS")}},{name:"mean and RMS of <code>data</code> are logged",run:async e=>{const t=Pe(e.utils,4096,6001);let s=0,n=0;for(let m=0;m<t.length;m++)s+=t[m],n+=t[m]*t[m];const i=s/4096,a=Math.sqrt(n/4096),f=Yn(e.logs);e.assert(f.some(m=>Math.abs(m-i)<=.01),`log the mean — expected ≈${i.toFixed(3)} in the console output`),e.assert(f.some(m=>Math.abs(m-a)<=.01),`log the RMS — expected ≈${a.toFixed(3)} in the console output`)}}],privateTests:[{name:"private test #1",run:async e=>{const{sums:t,squares:s}=Jn(e),n=Zt(e),i=s?null:Xn(e);e.assert(t&&s&&n,i||"expected partialSums, partialSquares and a dynamic ladder kernel");const a=Pe(e.utils,4096,909);let f=0,m=0;for(let H=0;H<a.length;H++)f+=a[H],m+=a[H]*a[H];const A=lt(n,t(a)),N=lt(n,s(a));e.assertClose(A/4096,f/4096,.01,"the mean"),e.assertClose(Math.sqrt(N/4096),Math.sqrt(m/4096),.01,"the RMS")}}]}]},jl=Object.freeze({__proto__:null,default:Bl});function ua(e,t,s,n,i){return Math.abs(e-t)<=s?`that is the value for cell [${i}][${n}] — this.thread.x and this.thread.y are swapped. Rows come first: image[this.thread.y][this.thread.x]`:null}function vs(e,t,s){return e?`the picture is transposed — the value for row ${t}, col ${s} turned up at row ${s}, col ${t}. this.thread.x and this.thread.y are swapped; rows come first: image[this.thread.y][this.thread.x].`:null}function ut(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Zn(e,t){const s=e.length,n=m=>Math.max(0,Math.min(s-1,m)),i=e[n(t-1)],a=e[t],f=e[n(t+1)];return[[(i+a+f)/3,"that is the plain 3-tap mean — this filter weights the taps 0.25 / 0.5 / 0.25"],[a,"that is the sample itself — the weighted average of its neighborhood never happened"],[.25*(i+a+f),"the center tap carries 0.5, not 0.25 — the three weights have to sum to 1"]]}function ca(e){return Number.isFinite(e)?null:"that sample read outside the signal — clamp the neighbor indexes into 0…127 before reading"}function Qn(e,t){const s="the window is not centered on this thread — tap i belongs at x + i − this.constants.radius";return e.map(n=>[n[t],s])}function ql(e,t,s,n){const i=(t*128+s)*4;return e[i]>=253&&e[i+1]>=253&&e[i+2]>=253&&Math.max(n[0],n[1],n[2])<.9?"every channel is clamped to white — that is the sum of the nine samples; divide each one by 9":null}function er(e,t,s){const n=e.length,i=m=>Math.max(0,Math.min(n-1,m)),a=e[t][s],f=e[t][i(s-1)]+e[t][i(s+1)]+e[i(t-1)][s]+e[i(t+1)][s];return[[4*a-f,"the center weight is 4, not 5 — the five weights have to sum to 1 so flat areas pass through unchanged"],[5*a+f,"the four neighbors are being added — a sharpen subtracts them: 5·center − left − right − up − down"],[a,"that is the value unchanged — none of the five weights reached the return value"],[f/4,"that is the average of the four neighbors — the 5·center term is missing"]]}function Qt(e,t=2301){const s=e.seededRandom(t),n=new Array(128);for(let i=0;i<128;i++)n[i]=Math.round((Math.sin(i/6)*3+s()*4)*100)/100;return n}function et(e,t,s){const n=e.length,i=new Array(n);for(let a=0;a<n;a++){let f=0;for(let m=0;m<t.length;m++){let A=a+m-s;A<0&&(A=0),A>n-1&&(A=n-1),f+=t[m]*e[A]}i[a]=f}return i}function Wl(e){const t=e.plain,s=t.length,n=new Array(s);for(let i=0;i<s;i++){const a=new Array(s);for(let f=0;f<s;f++){let m=0,A=0,N=0;for(let H=-1;H<=1;H++)for(let Y=-1;Y<=1;Y++){const Ae=Math.min(s-1,Math.max(0,i+H)),Ze=Math.min(s-1,Math.max(0,f+Y)),rt=t[Ae][Ze];m+=rt[0],A+=rt[1],N+=rt[2]}a[f]=[m/9,A/9,N/9]}n[i]=a}return n}function ha(e){const t=e.plain,s=t.length,n=new Array(s);for(let i=0;i<s;i++){const a=new Array(s);for(let f=0;f<s;f++){const m=t[i][f];a[f]=.299*m[0]+.587*m[1]+.114*m[2]}n[i]=a}return n}function Hl(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++){const f=new Array(t);for(let m=0;m<t;m++)f[m]=Math.round(n()*1e3)/1e3;i[a]=f}return i}function da(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){const i=new Array(t),a=Math.max(0,n-1),f=Math.min(t-1,n+1);for(let m=0;m<t;m++){const A=Math.max(0,m-1),N=Math.min(t-1,m+1);i[m]=5*e[n][m]-e[n][A]-e[n][N]-e[a][m]-e[f][m]}s[n]=i}return s}function pa(e,t){const s=new Array(e).fill(Bt(t));return us(new Array(e).fill(s))}function fa(e,t,s){const n=Bt([t,t,t,1]),i=Bt([s,s,s,1]),a=new Array(e);for(let f=0;f<e;f++){const m=new Array(e);for(let A=0;A<e;A++)m[A]=A<e/2?n:i;a[f]=m}return us(a)}function Xl(e,t,s){const n=Bt([t,t,t,1]),i=Bt([s,s,s,1]),a=new Array(e);for(let f=0;f<e;f++)a[f]=new Array(e).fill(f<e/2?n:i);return us(a)}var Yl={id:"2-3",track:2,title:"Convolution & Filters",blurb:"Sliding-window math on signals and images: blur, sharpen, edge detection.",tasks:[{slug:"smooth-a-signal",title:"Slide a Window: 1D Convolution",intro:`<p>A <strong>convolution</strong> slides a small window of weights along a signal:
        each output sample is a weighted average of the input around it. With weights
        <code>[0.25, 0.5, 0.25]</code> the window <em>smooths</em> — every sample leans toward
        its neighbors and jitter cancels out.</p>
        <p>On the GPU nothing actually slides. Every output sample gets its own thread, and each
        thread reads its <em>own</em> three inputs, all at the same time. The only wrinkle is the
        ends: sample 0 has no left neighbor, so we <strong>clamp</strong> — reuse the nearest
        in-bounds sample instead of reading past the edge.</p>`,goal:`<strong>Goal:</strong> smooth the 128-sample <code>signal</code> — each output is
        <code>0.25·left + 0.5·center + 0.25·right</code>, with indexes clamped at both ends.`,requirements:["Read this thread's neighbors: <code>signal[x - 1]</code> and <code>signal[x + 1]</code>","Clamp the indexes — below <code>0</code> becomes <code>0</code>, above <code>127</code> becomes <code>127</code>","Return <code>0.25·left + 0.5·center + 0.25·right</code>"],hints:[{title:"Hint 1 — nothing slides",body:`<p>Thread <code>x</code> only ever touches <code>signal[x - 1]</code>,
            <code>signal[x]</code> and <code>signal[x + 1]</code>. Three reads, one weighted sum,
            done — the "sliding" is 128 threads doing this at once.</p>`},{title:"Hint 2 — clamping with an if",body:`<pre><code>let left = x - 1;
if (left &lt; 0) left = 0;</code></pre>
<p>and the mirror image
            for <code>right</code> against <code>127</code>. Plain <code>if</code> statements work
            fine inside kernels.</p>`},{title:"Hint 3 — the whole body",body:`<pre><code>let left = x - 1;
if (left &lt; 0) left = 0;
let right = x + 1;
if (right &gt; 127) right = 127;
return 0.25 * signal[left] + 0.5 * signal[x] + 0.25 * signal[right];</code></pre>`}],transfer:`Neighborhood reads like this are called <em>stencil</em> patterns in CUDA and
        ROCm — the classic optimization is staging the window in shared memory. A WebGPU compute
        shader does the same thing with neighboring buffer reads inside a workgroup.`,starterCode:`// Convolution: each output sample is a weighted average of its neighborhood.
const gpu = new GPU({ mode });

const smooth = gpu.createKernel(function (signal) {
  const x = this.thread.x;
  // TODO: return 0.25 * left + 0.5 * center + 0.25 * right,
  // clamping the neighbor indexes so x = 0 and x = 127 stay in bounds.
  return signal[x];
}, { output: [128] });

const result = smooth(signal);
console.log('before:', signal[63], ' after:', result[63]);
`,solutionCode:`// Convolution: each output sample is a weighted average of its neighborhood.
const gpu = new GPU({ mode });

const smooth = gpu.createKernel(function (signal) {
  const x = this.thread.x;
  let left = x - 1;
  if (left < 0) left = 0;
  let right = x + 1;
  if (right > 127) right = 127;
  return 0.25 * signal[left] + 0.5 * signal[x] + 0.25 * signal[right];
}, { output: [128] });

const result = smooth(signal);
console.log('before:', signal[63], ' after:', result[63]);
`,inputs:e=>({signal:Qt(e)}),publicTests:[{name:"returns 128 samples, each a <code>[0.25, 0.5, 0.25]</code> weighted average",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Qt(e.utils),s=e.kernel(t);e.assert(s&&s.length===128,`expected 128 output samples, got ${s&&s.length}`);const n=et(t,[.25,.5,.25],1);for(const i of[1,17,42,63,100,126]){const a=ut(s[i],n[i],.001,Zn(t,i));e.assertClose(s[i],n[i],.001,a||`sample ${i}`)}}},{name:"edges clamp: sample 0 is <code>0.75·s[0] + 0.25·s[1]</code>",run:async e=>{const t=new Array(128);for(let a=0;a<128;a++)t[a]=a*37%23-11;const s=e.kernel(t),n=et(t,[.25,.5,.25],1),i=a=>ca(s[a])||ut(s[a],n[a],.001,Zn(t,a));e.assertClose(s[0],.75*t[0]+.25*t[1],.001,i(0)||"sample 0"),e.assertClose(s[127],.25*t[126]+.75*t[127],.001,i(127)||"sample 127");for(let a=0;a<128;a++)e.assertClose(s[a],n[a],.001,i(a)||`sample ${a}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=Qt(e.utils,909),s=e.kernel(t),n=et(t,[.25,.5,.25],1);e.assert(s.length===128,"expected 128 output samples");for(let i=0;i<128;i++){const a=ca(s[i])||ut(s[i],n[i],.001,Zn(t,i));e.assertClose(s[i],n[i],.001,a||`sample ${i}`)}}}]},{slug:"filter-as-data",title:"Any Filter, One Kernel",intro:`<p>Hardcoded weights mean writing a new kernel for every filter. The fix: pass the
        <code>filter</code> in as an ordinary array argument and loop over its taps. But a GPU
        loop wants bounds it can see <em>at compile time</em> — and that is exactly what
        <code>this.constants</code> is for: values baked into the kernel when it compiles,
        perfectly legal as loop bounds.</p>
        <p>This kernel is built with <code>constants: { size: 5, radius: 2 }</code>. Tap
        <code>i</code> of the filter lines up with input sample
        <code>x + i - radius</code> — clamp that index like before and accumulate
        <code>filter[i] * signal[tap]</code>.</p>`,goal:`<strong>Goal:</strong> finish the generic convolution — loop over
        <code>this.constants.size</code> taps, clamp each tap index, and return the accumulated
        weighted sum. One kernel, any 5-tap filter.`,requirements:["Loop <code>for (let i = 0; i &lt; this.constants.size; i++)</code> — a constant is a legal bound","Tap index: <code>x + i - this.constants.radius</code>, clamped to <code>0…127</code>","Accumulate <code>filter[i] * signal[tap]</code> into <code>sum</code> and return it"],hints:[{title:"Hint 1 — why constants?",body:`<p>Kernel arguments change per call; constants are frozen into the compiled
            kernel. That is why <code>this.constants.size</code> can bound a loop when a plain
            argument could not.</p>`},{title:"Hint 2 — the loop body",body:`<pre><code>let tap = x + i - this.constants.radius;
if (tap &lt; 0) tap = 0;
if (tap &gt; 127) tap = 127;
sum += filter[i] * signal[tap];</code></pre>`}],transfer:`Baked-in constants are a first-class idea everywhere: WGSL has
        pipeline-overridable constants, CUDA kernels take template parameters and
        <code>__constant__</code> memory, Metal has function constants — all so the compiler
        knows your loop bounds and can unroll the filter loop.`,starterCode:`// One kernel, any 5-tap filter: weights come in as data, size as constants.
const gpu = new GPU({ mode });

const convolve = gpu.createKernel(function (signal, filter) {
  const x = this.thread.x;
  let sum = 0;
  // TODO: loop i from 0 to this.constants.size,
  //   tap index = x + i - this.constants.radius (clamped to 0…127),
  //   accumulate filter[i] * signal[tap].
  return sum;
}, {
  output: [128],
  constants: { size: 5, radius: 2 },
});

const gauss = [0.06, 0.24, 0.4, 0.24, 0.06];
const result = convolve(signal, gauss);
console.log('smoothed sample 64:', result[64]);
`,solutionCode:`// One kernel, any 5-tap filter: weights come in as data, size as constants.
const gpu = new GPU({ mode });

const convolve = gpu.createKernel(function (signal, filter) {
  const x = this.thread.x;
  let sum = 0;
  for (let i = 0; i < this.constants.size; i++) {
    let tap = x + i - this.constants.radius;
    if (tap < 0) tap = 0;
    if (tap > 127) tap = 127;
    sum += filter[i] * signal[tap];
  }
  return sum;
}, {
  output: [128],
  constants: { size: 5, radius: 2 },
});

const gauss = [0.06, 0.24, 0.4, 0.24, 0.06];
const result = convolve(signal, gauss);
console.log('smoothed sample 64:', result[64]);
`,inputs:e=>({signal:Qt(e)}),publicTests:[{name:"the identity filter <code>[0, 0, 1, 0, 0]</code> returns the signal untouched",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Qt(e.utils),s=[0,0,1,0,0],n=e.kernel(t,s);e.assert(n&&n.length===128,`expected 128 output samples, got ${n&&n.length}`);const i=[et(t,s,0),et(t,s,-2)];for(let a=0;a<128;a++){const f=ut(n[a],t[a],.001,Qn(i,a));e.assertClose(n[a],t[a],.001,f||`sample ${a}`)}}},{name:"a box filter matches the clamped-edge reference everywhere",run:async e=>{const t=new Array(128);for(let f=0;f<128;f++)t[f]=f*29%17-8;const s=[.2,.2,.2,.2,.2],n=e.kernel(t,s),i=et(t,s,2),a=[et(t,s,0),et(t,s,-2)];for(let f=0;f<128;f++){const m=ut(n[f],i[f],.002,Qn(a,f));e.assertClose(n[f],i[f],.002,m||`sample ${f}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Qt(e.utils,777),s=e.utils.seededRandom(31),n=new Array(5);for(let m=0;m<5;m++)n[m]=Math.round((s()*.6-.1)*100)/100;const i=e.kernel(t,n),a=et(t,n,2),f=[et(t,n,0),et(t,n,-2)];for(let m=0;m<128;m++){const A=ut(i[m],a[m],.002,Qn(f,m));e.assertClose(i[m],a[m],.002,A||`sample ${m}`)}}}]},{slug:"box-blur",title:"Box Blur: the Window Goes 2D",intro:`<p>Take the sliding window into two dimensions and you have image filtering. A
        <strong>3×3 box blur</strong> is the simplest case: every output pixel is the plain
        average of the 3×3 patch centered on it — nine reads, per color channel, per pixel.
        131,072 threads each do their nine reads at once.</p>
        <p>Same edge problem, now on four sides: clamp <em>both</em> coordinates into
        <code>0…this.constants.last</code> before indexing. Average red, green and blue
        separately and hand the result to <code>this.color()</code>.</p>
        ${$t}`,goal:`<strong>Goal:</strong> blur <code>inputImage</code> with a 3×3 box filter — each
        painted pixel is the average of its 3×3 neighborhood, edges clamped.`,requirements:["Loop over the 3×3 neighborhood (a double <code>for</code> loop over <code>dy</code>, <code>dx</code>)","Clamp both sample coordinates to <code>0…this.constants.last</code>","Accumulate red, green and blue separately, then paint <code>this.color(r/9, g/9, b/9, 1)</code>"],hints:[{title:"Hint 1 — the neighborhood loop",body:`<p><code>for (let dy = 0; dy &lt; 3; dy++)</code> nested with
            <code>dx</code>, and the sample position is
            <code>this.thread.y + dy - 1</code>, <code>this.thread.x + dx - 1</code> — the
            <code>- 1</code> centers the window on this thread's pixel.</p>`},{title:"Hint 2 — clamp, then read",body:`<pre><code>let sy = this.thread.y + dy - 1;
if (sy &lt; 0) sy = 0;
if (sy &gt; this.constants.last) sy = this.constants.last;</code></pre>
<p>— same for
            <code>sx</code> — then <code>const pixel = image[sy][sx];</code> and add
            <code>pixel[0]</code>, <code>pixel[1]</code>, <code>pixel[2]</code> into three
            running sums.</p>`},{title:"Hint 3 — the finish",body:`<p>After the loops: <code>this.color(r / 9, g / 9, b / 9, 1);</code> —
            nine samples went in, so divide by nine on the way out.</p>`}],transfer:`Blur passes ship in every production toolkit — Metal Performance Shaders'
        <code>MPSImageBox</code>, NVIDIA's NPP filtering routines, WebGPU post-processing
        chains. The fast ones exploit that a box blur is <em>separable</em>: a horizontal pass
        then a vertical pass — six reads per pixel instead of nine.`,starterCode:`// Nine reads per pixel, averaged per channel. 131,072 threads at once.
const gpu = new GPU({ mode });

const blur = gpu.createKernel(function (image) {
  // TODO: average the 3×3 neighborhood around this pixel.
  // Clamp sample coordinates to 0…this.constants.last on both axes.
  const pixel = image[this.thread.y][this.thread.x];
  this.color(pixel[0], pixel[1], pixel[2], 1);
}, {
  output: [128, 128],
  graphical: true,
  constants: { last: 127 },
});

blur(inputImage);
render(blur.canvas);
`,solutionCode:`// Nine reads per pixel, averaged per channel. 131,072 threads at once.
const gpu = new GPU({ mode });

const blur = gpu.createKernel(function (image) {
  let r = 0;
  let g = 0;
  let b = 0;
  for (let dy = 0; dy < 3; dy++) {
    for (let dx = 0; dx < 3; dx++) {
      let sy = this.thread.y + dy - 1;
      let sx = this.thread.x + dx - 1;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      const pixel = image[sy][sx];
      r += pixel[0];
      g += pixel[1];
      b += pixel[2];
    }
  }
  this.color(r / 9, g / 9, b / 9, 1);
}, {
  output: [128, 128],
  graphical: true,
  constants: { last: 127 },
});

blur(inputImage);
render(blur.canvas);
`,inputs:e=>({inputImage:e.makeTestImage(128)}),publicTests:[{name:"produces a <code>128×128</code> graphical canvas",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===128&&e.canvas.height===128,`expected a 128×128 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();e.assert(t.length===16384*4,"pixel buffer should hold 128×128 RGBA values")}},{name:"blurring a flat color changes nothing",run:async e=>{const t=pa(128,[.3,.5,.7,1]),s=t.at(0,0);e.kernel(t);const n=e.getPixels();for(let i=0;i<n.length;i+=331*4)e.assertClose(n[i],s[0]*255,2,`red at byte ${i}`),e.assertClose(n[i+1],s[1]*255,2,`green at byte ${i}`),e.assertClose(n[i+2],s[2]*255,2,`blue at byte ${i}`)}},{name:"each pixel is the average of its 3×3 neighborhood",run:async e=>{const t=e.utils.makeTestImage(128);e.kernel(t);const s=e.getPixels(),n=Wl(t),i=(a,f,m)=>{const A=(a*128+f)*4;return Math.abs(s[A]-m[0]*255)<=3&&Math.abs(s[A+1]-m[1]*255)<=3&&Math.abs(s[A+2]-m[2]*255)<=3};for(const a of[3,17,40,64,90,121])for(const f of[5,33,64,101,124]){const m=i(a,f,n[f][a])||i(a,f,n[f][127-a]);e.assert(i(a,f,n[a][f])||i(a,f,n[127-a][f]),vs(m,a,f)||ql(s,a,f,n[a][f])||`pixel at row ${a}, col ${f} is not the 3×3 average of its neighborhood`)}}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(fa(128,.2,.8));const t=e.getPixels(),s=n=>n<=62?.2:n===63?.4:n===64?.6:.8;for(const n of[8,60,119])for(const i of[0,20,63,64,90,127]){const a=(n*128+i)*4,f=s(i)*255,m=(i<64?.2:.8)*255,A=ut(t[a],f,2,[[m,"that is the original pixel — the 3×3 average never happened, so the seam did not soften"]]);e.assertClose(t[a],f,2,A||`red at row ${n}, col ${i}`),e.assertClose(t[a+1],f,2,A||`green at row ${n}, col ${i}`),e.assertClose(t[a+2],f,2,A||`blue at row ${n}, col ${i}`)}}}]},{slug:"sharpen",title:"Sharpen: Negative Weights",intro:`<p>Filters are not all averages. Give the window <strong>negative weights</strong>
        and it starts measuring <em>differences</em>. The classic sharpen filter is a cross:
        <code>5</code> at the center, <code>−1</code> at each direct neighbor. Where the image is
        flat, the terms cancel to exactly the original value; where it changes, the difference
        gets amplified — edges pop.</p>
        <p>Sharpened values can overshoot right out of the 0–1 range, so this task computes on a
        numeric <strong>luminance map</strong> (<code>gray[y][x]</code>, one number per pixel)
        and returns raw numbers you can inspect — no color clamping hiding the math.</p>
        ${$t}`,goal:`<strong>Goal:</strong> sharpen the 96×96 <code>gray</code> map — each cell becomes
        <code>5·center − left − right − up − down</code>, with neighbor indexes clamped.`,requirements:["Clamp all four neighbor indexes to <code>0…this.constants.last</code>","Return <code>5 * gray[y][x]</code> minus the four clamped neighbor samples","Keep the kernel numeric — no <code>graphical: true</code>, values may leave 0–1"],hints:[{title:"Hint 1 — why 5 and −1?",body:`<p>The weights sum to 1, so flat regions pass through unchanged:
            <code>5c − 4c = c</code>. Everything the filter adds comes purely from
            center-vs-neighbor <em>differences</em>.</p>`},{title:"Hint 2 — four clamps, one return",body:`<pre><code>let left = x - 1;
if (left &lt; 0) left = 0;</code></pre>
<p>— repeat for
            <code>right</code>, <code>up</code>, <code>down</code> against
            <code>this.constants.last</code>, then a single return with the five terms:</p>
<pre><code>return 5 * gray[y][x] - gray[y][left] - gray[y][right]
  - gray[up][x] - gray[down][x];</code></pre>`}],transfer:`A convolution with learned weights is a CNN layer — cuDNN (CUDA) and MIOpen
        (ROCm) are entire libraries for running this exact multiply-accumulate window fast.
        Your sharpen filter is the same arithmetic with the weights picked by hand instead of
        by gradient descent.`,starterCode:`// Sharpen = identity + edge boost: 5×center − the 4 direct neighbors.
const gpu = new GPU({ mode });

const sharpen = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO: clamp left/right/up/down to 0…this.constants.last, then
  // return 5 * center − left − right − up − down.
  return gray[y][x];
}, {
  output: [96, 96],
  constants: { last: 95 },
});

const result = sharpen(gray);
console.log('center before:', gray[48][48], ' after:', result[48][48]);
`,solutionCode:`// Sharpen = identity + edge boost: 5×center − the 4 direct neighbors.
const gpu = new GPU({ mode });

const sharpen = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  let left = x - 1;
  if (left < 0) left = 0;
  let right = x + 1;
  if (right > this.constants.last) right = this.constants.last;
  let up = y - 1;
  if (up < 0) up = 0;
  let down = y + 1;
  if (down > this.constants.last) down = this.constants.last;
  return 5 * gray[y][x] - gray[y][left] - gray[y][right] - gray[up][x] - gray[down][x];
}, {
  output: [96, 96],
  constants: { last: 95 },
});

const result = sharpen(gray);
console.log('center before:', gray[48][48], ' after:', result[48][48]);
`,inputs:e=>({gray:ha(e.makeTestImage(96))}),publicTests:[{name:"flat regions are a fixed point — sharpening a constant map returns it unchanged",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(96).fill(new Array(96).fill(.5)),s=e.kernel(t);e.assert(s&&s.length===96,`expected 96 rows, got ${s&&s.length}`),e.assert(s[0]&&s[0].length===96,"each row should hold 96 values");for(let n=0;n<96;n+=7)for(let i=0;i<96;i+=7){const a=ut(s[n][i],.5,.001,er(t,n,i));e.assertClose(s[n][i],.5,.001,a||`cell [${n}][${i}]`)}}},{name:"cell [y][x] equals <code>5·center − left − right − up − down</code>",run:async e=>{const t=ha(e.utils.makeTestImage(96)),s=e.kernel(t),n=da(t),i=[[0,0],[0,48],[11,60],[48,48],[77,3],[95,95]];for(const[a,f]of i){const m=ua(s[a][f],n[f][a],.002,a,f)||ut(s[a][f],n[a][f],.002,er(t,a,f));e.assertClose(s[a][f],n[a][f],.002,m||`cell [${a}][${f}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Hl(e.utils,96,4242),s=e.kernel(t),n=da(t);for(let i=0;i<96;i++)for(let a=0;a<96;a++){const f=ua(s[i][a],n[a][i],.002,i,a)||ut(s[i][a],n[i][a],.002,er(t,i,a));e.assertClose(s[i][a],n[i][a],.002,f||`cell [${i}][${a}]`)}}}]},{slug:"sobel",title:"Sobel Edge Detection",intro:`<p>The payoff: run <strong>two convolutions at once</strong>. Sobel's
        <code>Gx</code> filter responds to horizontal change, <code>Gy</code> to vertical change,
        and the length of that gradient vector — <code>√(gx² + gy²)</code> — is how
        <em>edge-like</em> the pixel is, whatever the edge's direction.</p>
        <p>This is a two-kernel pipeline like module 1.2's finale: a numeric pass turns the image
        into a luminance map (written for you), then the Sobel pass reads each map cell's eight
        neighbors, applies both weight grids, and paints the magnitude. Border pixels have no
        full neighborhood, so the starter already paints them black — your work lives in the
        <code>else</code> branch.</p>
        ${$t}`,goal:`<strong>Goal:</strong> finish the Sobel kernel — read the 3×3 neighborhood of
        <code>gray</code>, compute <code>gx</code> and <code>gy</code> with the weights shown in
        the starter, and paint <code>Math.sqrt(gx * gx + gy * gy)</code> as a gray value.`,requirements:["Read the eight neighbors of <code>gray[y][x]</code> (no clamping needed — the border branch already ran)","Apply both weight grids: <code>gx</code> from the right column minus the left, <code>gy</code> from the bottom row minus the top","Paint the magnitude <code>Math.sqrt(gx * gx + gy * gy)</code> as gray via <code>this.color(m, m, m, 1)</code>"],hints:[{title:"Hint 1 — name the neighborhood",body:`<p>Pull the nine cells into locals first —
            <code>const tl = gray[y - 1][x - 1];</code> through
            <code>const br = gray[y + 1][x + 1];</code> — then the two weighted sums are easy to
            read off the grids.</p>`},{title:"Hint 2 — the two sums",body:`<pre><code>const gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);</code></pre>
<p>— right column minus left column, middle counted double. <code>gy</code> is the same
            with rows: <code>(bl + 2 * bm + br) - (tl + 2 * tm + tr)</code>.</p>`},{title:"Hint 3 — the finish",body:`<pre><code>const m = Math.sqrt(gx * gx + gy * gy);
this.color(m, m, m, 1);</code></pre>
<p>— flat areas give 0 (black), sharp edges overshoot 1 and clamp to white.</p>`}],transfer:`Sobel is the hello-world of GPU vision: it opens the OpenCL and CUDA imaging
        tutorials, camera ISPs run it in silicon, and edge maps feed feature detectors
        everywhere. Fusing two directional filters into one pass is exactly how you would write
        it in WGSL or Metal, too.`,starterCode:`// Two directional convolutions, one kernel, magnitude out.
const gpu = new GPU({ mode });

// Pass 1 — luminance map (module 1.2 déjà vu; already done for you).
const luminance = gpu.createKernel(function (image) {
  const pixel = image[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [128, 128] });

// Pass 2 — Sobel. Gx and Gy weigh the same 3×3 neighborhood:
//
//        Gx              Gy
//    -1   0  +1      -1  -2  -1
//    -2   0  +2       0   0   0
//    -1   0  +1      +1  +2  +1
//
const sobel = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    this.color(0, 0, 0, 1); // border: no full neighborhood — paint it black
  } else {
    // TODO: read the 8 neighbors, compute gx and gy with the grids above,
    // then paint the magnitude Math.sqrt(gx * gx + gy * gy).
    const l = gray[y][x];
    this.color(l, l, l, 1);
  }
}, {
  output: [128, 128],
  graphical: true,
  constants: { last: 127 },
});

const grayMap = luminance(inputImage);
sobel(grayMap);
render(sobel.canvas);
`,solutionCode:`// Two directional convolutions, one kernel, magnitude out.
const gpu = new GPU({ mode });

// Pass 1 — luminance map.
const luminance = gpu.createKernel(function (image) {
  const pixel = image[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [128, 128] });

// Pass 2 — Sobel magnitude.
const sobel = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    this.color(0, 0, 0, 1);
  } else {
    const tl = gray[y - 1][x - 1];
    const tm = gray[y - 1][x];
    const tr = gray[y - 1][x + 1];
    const ml = gray[y][x - 1];
    const mr = gray[y][x + 1];
    const bl = gray[y + 1][x - 1];
    const bm = gray[y + 1][x];
    const br = gray[y + 1][x + 1];
    const gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
    const gy = (bl + 2 * bm + br) - (tl + 2 * tm + tr);
    const m = Math.sqrt(gx * gx + gy * gy);
    this.color(m, m, m, 1);
  }
}, {
  output: [128, 128],
  graphical: true,
  constants: { last: 127 },
});

const grayMap = luminance(inputImage);
sobel(grayMap);
render(sobel.canvas);
`,inputs:e=>({inputImage:e.makeTestImage(128)}),publicTests:[{name:"a numeric luminance pass feeding a graphical Sobel pass",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=e.kernels.find(n=>n.kernel&&!n.kernel.graphical),s=e.kernels.find(n=>n.kernel&&n.kernel.graphical);e.assert(t,"no numeric (non-graphical) kernel found"),e.assert(s,"no graphical kernel found"),e.assert(e.canvas,"no canvas — did you call render(sobel.canvas)?"),e.assert(e.canvas.width===128&&e.canvas.height===128,`expected a 128×128 canvas, got ${e.canvas.width}×${e.canvas.height}`)}},{name:"a flat image has no edges — constant in, all black out",run:async e=>{const t=e.kernels.find(i=>i.kernel&&!i.kernel.graphical),s=e.kernels.find(i=>i.kernel&&i.kernel.graphical);e.assert(t&&s,"expected a numeric and a graphical kernel"),s(t(pa(128,[.4,.6,.2,1])));const n=s.getPixels();for(let i=0;i<n.length;i+=1004)e.assert(n[i]<=1&&n[i+1]<=1&&n[i+2]<=1,`pixel at byte ${i} should be black, got rgb(${n[i]}, ${n[i+1]}, ${n[i+2]})`),e.assert(n[i+3]===255,`alpha at byte ${i} should be 255`)}},{name:"a vertical brightness step lights up exactly the step columns",run:async e=>{const t=e.kernels.find(f=>f.kernel&&!f.kernel.graphical),s=e.kernels.find(f=>f.kernel&&f.kernel.graphical);e.assert(t&&s,"expected a numeric and a graphical kernel"),s(t(fa(128,.1,.9)));const n=s.getPixels(),i=(f,m)=>n[(f*128+m)*4],a=(f,m,A)=>A(i(m,f))||A(i(127-m,f));for(const f of[10,64,100]){for(const m of[63,64]){const A=(f*128+m)*4,N=a(f,m,H=>H>=253);e.assert(n[A]>=253,vs(N,f,m)||`the step at col ${m} should saturate white, got ${n[A]} (row ${f})`)}for(const m of[30,96]){const A=(f*128+m)*4,N=a(f,m,H=>H<=1);e.assert(n[A]<=1,vs(N,f,m)||`flat area at col ${m} should be black, got ${n[A]} (row ${f})`)}}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(f=>f.kernel&&!f.kernel.graphical),s=e.kernels.find(f=>f.kernel&&f.kernel.graphical);e.assert(t&&s,"expected a numeric and a graphical kernel"),s(t(Xl(128,.15,.85)));const n=s.getPixels(),i=(f,m)=>n[(f*128+m)*4],a=(f,m,A)=>A(i(m,f))||A(i(127-m,f));for(const f of[10,64,120]){for(const m of[63,64]){const A=(m*128+f)*4,N=a(m,f,H=>H>=253);e.assert(n[A]>=253,vs(N,m,f)||`the step at row ${m} should saturate white, got ${n[A]} (col ${f})`)}for(const m of[20,100]){const A=(m*128+f)*4,N=a(m,f,H=>H<=1);e.assert(n[A]<=1,vs(N,m,f)||`flat area at row ${m} should be black, got ${n[A]} (col ${f})`)}}}}]}]},Jl=Object.freeze({__proto__:null,default:Yl});function tr(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++)i[a]=n();return i}function es(e,t,s){const n=e.seededRandom(s),i=new Array(t),a=new Array(t);for(let f=0;f<t;f++)i[f]=n(),a[f]=n();return{xs:i,ys:a}}function tn(e,t){let s=0;for(let n=0;n<e.length;n++)e[n]*e[n]+t[n]*t[n]<=1&&s++;return s}function sn(e){let t=0;for(let s=0;s<e.length;s++)t+=Math.exp(-e[s]*e[s]);return t}function sr(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a+=2){const f=1-n(),m=n(),A=Math.sqrt(-2*Math.log(f));i[a]=A*Math.cos(2*Math.PI*m),a+1<t&&(i[a+1]=A*Math.sin(2*Math.PI*m))}return i}const Te={s0:100,strike:105,rate:.03,sigma:.2,t:1},rs=(Te.rate-Te.sigma*Te.sigma/2)*Te.t,is=Te.sigma*Math.sqrt(Te.t);function ma(e){const t=1/(1+.2316419*Math.abs(e)),n=.3989422804014327*Math.exp(-e*e/2)*t*(.31938153+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));return e>=0?1-n:n}function ga(){const{s0:e,strike:t,rate:s,sigma:n,t:i}=Te,a=(Math.log(e/t)+(s+n*n/2)*i)/(n*Math.sqrt(i)),f=a-n*Math.sqrt(i);return e*ma(a)-t*Math.exp(-s*i)*ma(f)}function nn(e){let t=0;for(let s=0;s<e.length;s++){const n=Te.s0*Math.exp(rs+is*e[s]);t+=Math.max(n-Te.strike,0)}return Math.exp(-.03*Te.t)*(t/e.length)}function ya(e){let t=0;for(let s=0;s<e.length;s++){const n=Te.s0*Math.exp(rs+is*e[s]);t+=n-Te.strike}return Math.exp(-.03*Te.t)*(t/e.length)}function Rt(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Zl(e,t,s,n){const i=a=>n.every((f,m)=>Math.abs(e[m]-a(m))<=1e-6);return i(a=>t[a]+s[a]<=1?1:0)?"those verdicts are x + y ≤ 1 — the inside test compares squared distance: x * x + y * y <= 1":i(a=>1-n[a])?"the verdicts are inverted — return 1 when the dart lands inside, 0 when it misses":null}function hn(e,t,s,n){let i=0;for(let a=0;a<s;a++){const f=e[t+a];i+=n?n(f):f}return i}function Ql(e,t,s){const n=[[e[t],"that is a single verdict — this thread has to total all 256 in its own slice"]];return t+s<=e.length&&n.push([hn(e,t,s),"the slice starts at this.thread.x * 256 — with this.thread.x alone every thread walks an overlapping window"]),n}function nr(e,t,s){return[[hn(e,t,s),"that is the sum of the samples themselves — the accumulator wants Math.exp(-x * x)"],[hn(e,t,s,n=>Math.exp(-n)),"that is e^(−x), not e^(−x²) — square x inside the exponent"],[hn(e,t,s,n=>Math.exp(n*n)),"the exponent is missing its minus sign — e^(−x²) falls off as x grows"]]}function eu(e,t){return[[e-t,"that is st − strike with no floor — a losing path pays exactly 0, never a negative amount"],[Math.max(t-e,0),"that is the put payoff — a call pays max(st − strike, 0)"]]}var tu={id:"2-4",track:2,title:"Monte Carlo Methods",blurb:"Estimate π, price an option, integrate the un-integrable — with a million random samples.",tasks:[{slug:"darts-at-a-circle",title:"Darts at a Quarter Circle",intro:`<p>Monte Carlo is statistics as a weapon: throw random darts at a square, and the
        <em>fraction</em> that lands inside the quarter circle inscribed in it approaches its area —
        π/4. No geometry beyond the Pythagorean check <code>x² + y² ≤ 1</code>.</p>
        <p>The method is embarrassingly parallel: every dart is judged independently, so every dart
        gets its own thread. One rule, though — the randomness is made <strong>outside</strong> the
        kernel. <code>xs</code> and <code>ys</code> hold 4,096 seeded dart positions; the kernel's
        job is only the verdict. Deterministic data in, deterministic verdicts out — that's what
        makes GPU Monte Carlo debuggable.</p>`,goal:`<strong>Goal:</strong> make each thread return <code>1</code> if its dart
        <code>(xs[x], ys[x])</code> lands inside the unit quarter circle, else <code>0</code>.`,requirements:["Read this thread's dart: <code>xs[this.thread.x]</code> and <code>ys[this.thread.x]</code>","Inside means <code>x² + y² ≤ 1</code> — no <code>Math.sqrt</code> needed","Return exactly <code>1</code> or <code>0</code>, nothing in between"],hints:[{title:"Hint 1 — skip the square root",body:`<p>The dart is inside when its distance to the origin is ≤ 1 — and distances
            compare the same way squared: <code>x * x + y * y &lt;= 1</code> is the whole test.</p>`},{title:"Hint 2 — the verdict",body:`<pre><code>if (x * x + y * y &lt;= 1) {
  return 1;
}
return 0;</code></pre>
<p>— a branch is
            fine in a kernel as long as every path returns.</p>`}],transfer:`Real GPU Monte Carlo keeps the random numbers on-device — CUDA ships cuRAND, and
        WebGPU/Metal compute shaders run counter-based generators like Philox per thread — but the
        shape is exactly this: one thread, one sample, one verdict.`,starterCode:`// 4,096 seeded darts. One thread judges one dart.
const gpu = new GPU({ mode });

const inside = gpu.createKernel(function (xs, ys) {
  const x = xs[this.thread.x];
  const y = ys[this.thread.x];
  // TODO: return 1 if this dart lands inside the unit quarter
  // circle (x² + y² ≤ 1), otherwise 0.
  return 0;
}, { output: [4096] });

const hits = inside(xs, ys);

let count = 0;
for (let i = 0; i < hits.length; i++) count += hits[i];
console.log(count, 'of 4096 darts hit — π ≈', (4 * count) / 4096);
`,solutionCode:`// 4,096 seeded darts. One thread judges one dart.
const gpu = new GPU({ mode });

const inside = gpu.createKernel(function (xs, ys) {
  const x = xs[this.thread.x];
  const y = ys[this.thread.x];
  if (x * x + y * y <= 1) {
    return 1;
  }
  return 0;
}, { output: [4096] });

const hits = inside(xs, ys);

let count = 0;
for (let i = 0; i < hits.length; i++) count += hits[i];
console.log(count, 'of 4096 darts hit — π ≈', (4 * count) / 4096);
`,inputs:e=>es(e,4096,9001),publicTests:[{name:"clearly-inside darts return 1, clearly-outside darts return 0",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(4096).fill(.5),s=new Array(4096).fill(.5);t[0]=.1,s[0]=.1,t[1]=.9,s[1]=.9,t[2]=0,s[2]=0,t[3]=.99,s[3]=.3,t[4]=.6,s[4]=.6;const n=e.kernel(t,s);e.assert(n&&n.length===4096,`expected 4096 verdicts, got ${n&&n.length}`);const i=[1,0,1,0,1],a=Zl(n,t,s,i);for(let f=0;f<i.length;f++)e.assertClose(n[f],i[f],1e-6,a||`dart ${f} at (${t[f]}, ${s[f]})`)}},{name:"hit fraction over the seeded darts approaches <code>π/4</code>",run:async e=>{const{xs:t,ys:s}=es(e.utils,4096,9001),n=e.kernel(t,s);let i=0;for(let a=0;a<n.length;a++)e.assert(n[a]===0||n[a]===1,`verdict ${a} is ${n[a]} — return exactly 1 or 0`),i+=n[a];e.assertClose(i,tn(t,s),2,"hit count over the seeded darts"),e.assertClose(4*i/4096,Math.PI,.06,"π estimate from 4096 darts")}}],privateTests:[{name:"private test #1",run:async e=>{const{xs:t,ys:s}=es(e.utils,4096,4242),n=e.kernel(t,s);let i=0;for(let a=0;a<n.length;a++)i+=n[a];e.assertClose(i,tn(t,s),2,"hit count on unseen darts")}}]},{slug:"reduce-to-pi",title:"Reduce 65,536 Hits to π",intro:`<p>Last task summed the verdicts with a JavaScript loop — fine for 4,096 darts,
        wasteful for 65,536 and absurd for a billion. The GPU answer is a
        <strong>parallel reduction</strong>: don't ship every verdict home, ship
        <em>partial sums</em>. A second kernel with 256 threads gives each thread its own
        256-verdict slice to total, collapsing 65,536 numbers to 256 in one launch.</p>
        <p>Thread <code>t</code> owns the slice starting at <code>t * 256</code> — a statically
        bounded <code>for</code> loop walks it. JavaScript then folds the 256 partials into the
        final count, and <code>4 × hits / 65536</code> is your π.</p>`,goal:`<strong>Goal:</strong> complete the <code>partialSums</code> kernel so each of its
        256 threads returns the sum of its own 256-element slice of <code>hits</code>, then log
        the π estimate.`,requirements:["Kernel 1 (<code>inside</code>) is last task's dart test — leave it as is","In <code>partialSums</code>, thread <code>x</code> starts at <code>this.thread.x * 256</code>","Loop <code>i = 0…255</code> and accumulate <code>hits[base + i]</code>","Total the 256 partials in JavaScript and log <code>4 * total / 65536</code>"],hints:[{title:"Hint 1 — who sums what",body:`<p>Thread 0 sums <code>hits[0…255]</code>, thread 1 sums <code>hits[256…511]</code>,
            and so on. The starting offset is <code>this.thread.x * 256</code>.</p>`},{title:"Hint 2 — the loop",body:`<pre><code>const base = this.thread.x * 256;
let sum = 0;
for (let i = 0; i &lt; 256; i++) {
  sum += hits[base + i];
}
return sum;</code></pre>
<p>The bound is a literal, so gpu.js can unroll it safely.</p>`}],transfer:`Reduction is <em>the</em> fundamental pattern of GPU computing — CUDA has warp
        shuffles and the CUB library for it, Metal has SIMD-group reductions, WebGPU builds them
        from workgroup shared memory. Chunked partial sums like yours are always the first rung.`,starterCode:`// 65,536 darts. Kernel 1 judges them; kernel 2 sums them — in parallel.
const gpu = new GPU({ mode });

const inside = gpu.createKernel(function (xs, ys) {
  const x = xs[this.thread.x];
  const y = ys[this.thread.x];
  if (x * x + y * y <= 1) {
    return 1;
  }
  return 0;
}, { output: [65536] });

const partialSums = gpu.createKernel(function (hits) {
  // TODO: sum THIS thread's 256-element slice of hits.
  // Slice start: this.thread.x * 256.
  return hits[this.thread.x];
}, { output: [256] });

const hits = inside(xs, ys);
const partials = partialSums(hits);

let total = 0;
for (let i = 0; i < partials.length; i++) total += partials[i];
console.log('π ≈', (4 * total) / 65536);
`,solutionCode:`// 65,536 darts. Kernel 1 judges them; kernel 2 sums them — in parallel.
const gpu = new GPU({ mode });

const inside = gpu.createKernel(function (xs, ys) {
  const x = xs[this.thread.x];
  const y = ys[this.thread.x];
  if (x * x + y * y <= 1) {
    return 1;
  }
  return 0;
}, { output: [65536] });

const partialSums = gpu.createKernel(function (hits) {
  const base = this.thread.x * 256;
  let sum = 0;
  for (let i = 0; i < 256; i++) {
    sum += hits[base + i];
  }
  return sum;
}, { output: [256] });

const hits = inside(xs, ys);
const partials = partialSums(hits);

let total = 0;
for (let i = 0; i < partials.length; i++) total += partials[i];
console.log('π ≈', (4 * total) / 65536);
`,inputs:e=>es(e,65536,1337),publicTests:[{name:"reduction kernel collapses a known array to correct partial sums",run:async e=>{const t=e.kernels.find(i=>i.kernel&&Array.isArray(i.kernel.output)&&i.kernel.output[0]===256);e.assert(t,"no kernel with output [256] found — keep the partialSums kernel");const s=new Array(65536);for(let i=0;i<65536;i++)s[i]=i%3;const n=t(new Float32Array(s));e.assert(n&&n.length===256,`expected 256 partials, got ${n&&n.length}`);for(const i of[0,1,17,128,255]){let a=0;for(let m=0;m<256;m++)a+=(i*256+m)%3;const f=Rt(n[i],a,.5,Ql(s,i,256));e.assertClose(n[i],a,.5,f||`partial sum for thread ${i}`)}}},{name:"π comes out within <code>±0.05</code> over the 65,536 seeded darts",run:async e=>{const t=e.kernels.find(A=>A.kernel&&Array.isArray(A.kernel.output)&&A.kernel.output[0]===65536),s=e.kernels.find(A=>A.kernel&&Array.isArray(A.kernel.output)&&A.kernel.output[0]===256);e.assert(t&&s,"expected the inside kernel [65536] and the partialSums kernel [256]");const{xs:n,ys:i}=es(e.utils,65536,1337),a=t(n,i),f=s(new Float32Array(a));let m=0;for(let A=0;A<f.length;A++)m+=f[A];e.assertClose(m,tn(n,i),4,"total hit count after reduction"),e.assertClose(4*m/65536,Math.PI,.05,"π estimate")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(A=>A.kernel&&Array.isArray(A.kernel.output)&&A.kernel.output[0]===65536),s=e.kernels.find(A=>A.kernel&&Array.isArray(A.kernel.output)&&A.kernel.output[0]===256);e.assert(t&&s,"expected the inside kernel [65536] and the partialSums kernel [256]");const{xs:n,ys:i}=es(e.utils,65536,2718),a=t(n,i),f=s(new Float32Array(a));let m=0;for(let A=0;A<f.length;A++)m+=f[A];e.assertClose(m,tn(n,i),4,"hit count on unseen darts")}}]},{slug:"integrate-the-unintegrable",title:"Integrate the Un-integrable",intro:`<p><code>e^(−x²)</code> — the bell curve — famously has <strong>no elementary
        antiderivative</strong>. No substitution, no parts, no closed form. Monte Carlo doesn't
        care: for uniform samples on [0, 1], the <em>average</em> of <code>f(x)</code> converges
        to <code>∫₀¹ f(x) dx</code>. Sampling beats symbolic calculus.</p>
        <p>And here's the efficiency move over last task: instead of one kernel to evaluate and
        another to reduce, <strong>fuse them</strong>. Each of 256 threads walks its own 64-sample
        slice, evaluating <code>e^(−x²)</code> and accumulating in one pass — 16,384 evaluations,
        one launch, 256 numbers back.</p>`,goal:`<strong>Goal:</strong> make each thread return the sum of <code>e^(−x²)</code> over
        its 64-sample slice of <code>samples</code>, so the logged mean lands on
        <code>≈ 0.7468</code>.`,requirements:["Thread <code>x</code> owns the slice starting at <code>this.thread.x * 64</code>","Evaluate <code>Math.exp(-x * x)</code> for each sample — inside the loop, inside the kernel","Return the slice sum; JavaScript divides the grand total by 16384"],hints:[{title:"Hint 1 — mean value, not area sampling",body:`<p>No darts this time: the estimator is just the average height of the curve,
            <code>(1/N) Σ f(xᵢ)</code>, times the interval width (here 1). You only need
            <code>f</code>, not a hit test.</p>`},{title:"Hint 2 — one line changes",body:`<p>The loop skeleton is last task's reduction. Swap what you accumulate:</p>
<pre><code>const x = xs[base + i];
sum += Math.exp(-x * x);</code></pre>`}],transfer:`Fusing the map into the reduction halves the memory traffic — the same reasoning
        behind kernel fusion in CUDA and ROCm, and behind doing per-workgroup sums in a single
        WebGPU compute pass instead of two. Bandwidth, not arithmetic, is usually the bill.`,starterCode:`// ∫₀¹ e^(−x²) dx has no closed form. Estimate it: average f over
// 16,384 seeded samples — 256 threads × 64 samples each, fused map+reduce.
const gpu = new GPU({ mode });

const partials = gpu.createKernel(function (xs) {
  const base = this.thread.x * 64;
  let sum = 0;
  for (let i = 0; i < 64; i++) {
    const x = xs[base + i];
    // TODO: accumulate f(x) = e^(−x²) — not x itself.
    sum += x;
  }
  return sum;
}, { output: [256] });

const sums = partials(samples);

let total = 0;
for (let i = 0; i < sums.length; i++) total += sums[i];
console.log('∫₀¹ e^(−x²) dx ≈', total / 16384, '(truth ≈ 0.746824)');
`,solutionCode:`// ∫₀¹ e^(−x²) dx has no closed form. Estimate it: average f over
// 16,384 seeded samples — 256 threads × 64 samples each, fused map+reduce.
const gpu = new GPU({ mode });

const partials = gpu.createKernel(function (xs) {
  const base = this.thread.x * 64;
  let sum = 0;
  for (let i = 0; i < 64; i++) {
    const x = xs[base + i];
    sum += Math.exp(-x * x);
  }
  return sum;
}, { output: [256] });

const sums = partials(samples);

let total = 0;
for (let i = 0; i < sums.length; i++) total += sums[i];
console.log('∫₀¹ e^(−x²) dx ≈', total / 16384, '(truth ≈ 0.746824)');
`,inputs:e=>({samples:tr(e,16384,6077)}),publicTests:[{name:"each thread sums <code>e^(−x²)</code> over its own 64-sample slice",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(16384);for(let n=0;n<16384;n++)t[n]=n/16384;const s=e.kernel(t);e.assert(s&&s.length===256,`expected 256 partial sums, got ${s&&s.length}`);for(const n of[0,3,100,255]){const i=t.slice(n*64,n*64+64),a=Rt(s[n],sn(i),.05,nr(t,n*64,64));e.assertClose(s[n],sn(i),.05,a||`partial sum for thread ${n}`)}}},{name:"estimate lands within <code>±0.01</code> of the true value <code>0.746824</code>",run:async e=>{const t=tr(e.utils,16384,6077),s=e.kernel(t);let n=0;for(let a=0;a<s.length;a++)n+=s[a];const i=Rt(n/16384,.7468241328124271,.01,nr(t,0,16384).map(a=>[a[0]/16384,a[1]]));e.assertClose(n/16384,.7468241328124271,.01,i||"Monte Carlo integral estimate")}}],privateTests:[{name:"private test #1",run:async e=>{const t=tr(e.utils,16384,1912),s=e.kernel(t);let n=0;for(let a=0;a<s.length;a++)n+=s[a];const i=Rt(n/16384,sn(t)/16384,.002,nr(t,0,16384).map(a=>[a[0]/16384,a[1]]));e.assertClose(n/16384,sn(t)/16384,.002,i||"estimate vs float64 reference"),e.assertClose(n/16384,.7468241328124271,.01,"estimate vs the true integral")}}]},{slug:"price-an-option",title:"Price an Option",intro:`<p>The payoff. A <strong>European call option</strong> is the right to buy a stock at
        a fixed strike price K on a future date — worth <code>max(S_T − K, 0)</code> when the stock
        finishes at <code>S_T</code>, and its fair price today is the <em>discounted expected
        payoff</em>. Expectations are integrals, and you just learned to integrate by sampling.</p>
        <p>Each thread simulates one possible market: under the standard log-normal model, a
        pre-drawn normal shock <code>z</code> gives
        <code>S_T = S0 · e^(drift + volT · z)</code>. Your kernel turns 16,384 shocks into
        16,384 payoffs; JavaScript averages and discounts. Stock at 100, strike 105, one year out —
        the Black–Scholes formula says the answer is ≈ 7.13. Your simulation should agree.</p>`,goal:`<strong>Goal:</strong> complete the payoff kernel — simulate this thread's final stock
        price and return the option payoff <code>max(S_T − strike, 0)</code>.`,requirements:["Simulate the final price: <code>s0 * Math.exp(drift + volT * z)</code> (already wired)","Return the call payoff: <code>Math.max(st - strike, 0)</code> — an option never goes negative","Average the payoffs and discount by <code>Math.exp(-RATE * T)</code> in JavaScript"],hints:[{title:"Hint 1 — why the max?",body:`<p>If the stock ends below the strike you simply don't exercise — the option
            expires worthless, payoff 0, never negative. Forgetting the <code>max</code> drags the
            average down by every losing path (the price comes out near −1.9 instead of ≈ 7.1).</p>`},{title:"Hint 2 — the kernel body",body:`<p><code>return Math.max(st - strike, 0);</code> — <code>Math.max</code> works
            inside kernels, and beats an <code>if</code> here.</p>`}],transfer:`This is production reality: quant desks run exactly this workload on CUDA and ROCm
        — millions of simulated paths per pricing call, one thread per path, then a reduction —
        because exotic options have no closed form at all. You now hold the whole recipe.`,starterCode:`// Fair price = discounted average payoff over simulated futures.
// Stock at 100, strike 105, 3% rate, 20% volatility, 1 year to expiry.
const S0 = 100, STRIKE = 105, RATE = 0.03, SIGMA = 0.2, T = 1;

const gpu = new GPU({ mode });

const payoff = gpu.createKernel(function (normals, s0, strike, drift, volT) {
  const z = normals[this.thread.x];
  const st = s0 * Math.exp(drift + volT * z); // this thread's final stock price
  // TODO: return the call payoff — st minus strike, but never below zero.
  return st - strike;
}, { output: [16384] });

const payoffs = payoff(normals, S0, STRIKE, (RATE - SIGMA * SIGMA / 2) * T, SIGMA * Math.sqrt(T));

let sum = 0;
for (let i = 0; i < payoffs.length; i++) sum += payoffs[i];
const price = Math.exp(-RATE * T) * (sum / payoffs.length);
console.log('Monte Carlo price:', price, '— Black–Scholes says ≈ 7.13');
`,solutionCode:`// Fair price = discounted average payoff over simulated futures.
// Stock at 100, strike 105, 3% rate, 20% volatility, 1 year to expiry.
const S0 = 100, STRIKE = 105, RATE = 0.03, SIGMA = 0.2, T = 1;

const gpu = new GPU({ mode });

const payoff = gpu.createKernel(function (normals, s0, strike, drift, volT) {
  const z = normals[this.thread.x];
  const st = s0 * Math.exp(drift + volT * z); // this thread's final stock price
  return Math.max(st - strike, 0);
}, { output: [16384] });

const payoffs = payoff(normals, S0, STRIKE, (RATE - SIGMA * SIGMA / 2) * T, SIGMA * Math.sqrt(T));

let sum = 0;
for (let i = 0; i < payoffs.length; i++) sum += payoffs[i];
const price = Math.exp(-RATE * T) * (sum / payoffs.length);
console.log('Monte Carlo price:', price, '— Black–Scholes says ≈ 7.13');
`,inputs:e=>({normals:sr(e,16384,8128)}),publicTests:[{name:"payoffs are <code>max(S_T − K, 0)</code> — losing paths pay exactly zero",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(16384).fill(0);t[0]=-3,t[1]=2,t[2]=.5,t[3]=-.5;const s=e.kernel(t,Te.s0,Te.strike,rs,is);e.assert(s&&s.length===16384,`expected 16384 payoffs, got ${s&&s.length}`);for(const n of[0,1,2,3]){const i=Te.s0*Math.exp(rs+is*t[n]),a=Math.max(i-Te.strike,0),f=Rt(s[n],a,.05,eu(i,Te.strike));e.assertClose(s[n],a,.05,f||`payoff for shock z = ${t[n]}`),e.assert(s[n]>=0,`payoff for z = ${t[n]} is negative (${s[n]}) — options never go below zero`)}}},{name:"simulated price agrees with Black–Scholes (<code>≈ 7.13</code>) within <code>±0.4</code>",run:async e=>{const t=sr(e.utils,16384,8128),s=e.kernel(t,Te.s0,Te.strike,rs,is);let n=0;for(let f=0;f<s.length;f++)n+=s[f];const i=Math.exp(-.03*Te.t)*(n/s.length),a=Rt(i,nn(t),.05,[[ya(t),"that is the average of st − strike with no floor — every losing path dragged the mean below zero"]]);e.assertClose(i,nn(t),.05,a||"price vs float64 reference simulation"),e.assertClose(i,ga(),.4,"price vs the Black–Scholes closed form")}}],privateTests:[{name:"private test #1",run:async e=>{const t=sr(e.utils,16384,6174),s=e.kernel(t,Te.s0,Te.strike,rs,is);let n=0;for(let f=0;f<s.length;f++)n+=s[f];const i=Math.exp(-.03*Te.t)*(n/s.length),a=Rt(i,nn(t),.05,[[ya(t),"that is the average of st − strike with no floor — every losing path dragged the mean below zero"]]);e.assertClose(i,nn(t),.05,a||"price vs float64 reference on unseen shocks"),e.assertClose(i,ga(),.5,"price vs Black–Scholes on unseen shocks")}}]}]},su=Object.freeze({__proto__:null,default:tu});function Ft(e,t,s){const n=e.seededRandom(s),i=new Array(t),a=new Array(t),f=new Array(t);for(let m=0;m<t;m++){const A=2*Math.PI*m/t,N=.7+.6*n();i[m]=Math.round(N*Math.cos(A)*1e4)/1e4,a[m]=Math.round(N*Math.sin(A)*1e4)/1e4,f[m]=Math.round((.5+n())*100)/100}return{posX:i,posY:a,mass:f}}function tt(e,t,s){const n=e.seededRandom(s),i=new Array(t),a=new Array(t),f=new Array(t),m=new Array(t),A=new Array(t);for(let N=0;N<t;N++)i[N]=Math.round((n()*2-1)*1e4)/1e4,a[N]=Math.round((n()*2-1)*1e4)/1e4,f[N]=Math.round((n()-.5)*.2*1e4)/1e4,m[N]=Math.round((n()-.5)*.2*1e4)/1e4,A[N]=Math.round((.5+n())*100)/100;return i[1]=i[0]+.001,a[1]=a[0],{posX:i,posY:a,velX:f,velY:m,mass:A}}function Nt(e,t,s,n,i){let a=0,f=0;for(let m=0;m<t.length;m++){if(i===0&&m===e)continue;const A=t[m]-t[e],N=s[m]-s[e],H=A*A+N*N+i*i;if(H===0)continue;const Y=n[m]/(H*Math.sqrt(H));a+=A*Y,f+=N*Y}return[a,f]}function _r(e,t){const s=new Array(e.posX.length),n=new Array(e.posX.length);for(let i=0;i<e.posX.length;i++){const a=Nt(i,e.posX,e.posY,e.mass,t);s[i]=a[0],n[i]=a[1]}return{accX:s,accY:n}}function rr(e,t,s,n){const i={posX:e.posX.slice(),posY:e.posY.slice(),velX:e.velX.slice(),velY:e.velY.slice()};for(let a=0;a<t;a++){const f=_r({posX:i.posX,posY:i.posY,mass:e.mass},n);for(let m=0;m<i.posX.length;m++)i.velX[m]+=f.accX[m]*s,i.velY[m]+=f.accY[m]*s,i.posX[m]+=i.velX[m]*s,i.posY[m]+=i.velY[m]*s}return i}function ir(e){const t=new Array(e.length),s=new Array(e.length);for(let n=0;n<e.length;n++)t[n]=e[n][0],s[n]=e[n][1];return[t,s]}function xa(e,t,s,n,i,a,f){const[m,A]=ir(e(n.posX,n.posY,i,f)),[N,H]=ir(t(n.velX,n.velY,m,A,a)),[Y,Ae]=ir(s(n.posX,n.posY,N,H,a));return{posX:Y,posY:Ae,velX:N,velY:H}}function ve(e,t,s,n,i){e.assertClose(t,s,n*(1+Math.abs(s)),i)}function Et(e,t){return e*(1+Math.abs(t))}function bt(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function ba(e,t){return[[e/Math.sqrt(t),"that is M / r — dx * dx + dy * dy is already r², so there is no square root to take"],[1/t,"the star's mass never entered the result — the pull is starMass / r²"],[e*t,"that multiplies by r² where the law divides by it"]]}function nu(e,t,s,n){let i=0,a=0;for(let f=0;f<t.length;f++){if(f===e)continue;const m=t[f]-t[e],A=s[f]-s[e],N=m*m+A*A;i+=n[f]*m/N,a+=n[f]*A/N}return[i,a]}function wa(e,t,s,n){return[[nu(e,t.posX,t.posY,t.mass)[n],"that is mass[j]·d / r², one factor of r short — the unit direction is d / r, so each term is mass[j]·d / r³"],[-s,"the offset points the wrong way — dx is posX[j] minus your OWN x, so the pull points at the other body"]]}function rn(e,t,s,n){return[[Nt(e,t.posX,t.posY,t.mass,Math.sqrt(s))[n],"that adds soft where it should add soft · soft — Plummer softening replaces r² with r² + ε²"],[Nt(e,t.posX,t.posY,t.mass,0)[n],"that is the unsoftened sum — ε never reached the denominator"]]}function _t(e,t,s,n,i){const a=t+s*n;return bt(e,a,Et(i,a),[[t+s,"the time step is missing — the update is value + rate · dt"],[t,"that value came back unchanged — nothing was added to it"]])}var ru={id:"2-5",track:2,title:"N-Body Gravity",blurb:"Every particle pulls on every other: an O(n²) problem the GPU eats for breakfast.",tasks:[{slug:"one-star-pull",title:"The Pull of One Star",intro:`<p>Newton, in one line: the gravitational pull between two bodies is
        <code>G · m₁ · m₂ / r²</code>. Divide out the mass being pulled and you get its
        <strong>acceleration</strong> — <code>a = G · M / r²</code> — which only depends on the
        <em>other</em> body. In this course <code>G = 1</code> (astrophysicists rescale units to
        do exactly this, so you're in good company).</p>
        <p>Here 64 bodies drift around one star. Each thread owns one body — its position is
        <code>posX[this.thread.x]</code>, <code>posY[this.thread.x]</code> — and answers a single
        question: <em>how hard does the star pull on me?</em> No loops yet; that's next.</p>`,goal:`<strong>Goal:</strong> make the kernel return the strength of the star's pull on this
        thread's body: <code>starMass / r²</code>.`,requirements:["Use the <code>dx</code>, <code>dy</code> offsets to the star (already wired up)","Compute the squared distance: <code>r² = dx·dx + dy·dy</code>","Return <code>starMass / r²</code> — inverse-square, with <code>G = 1</code>"],hints:[{title:"Hint 1 — no square root needed",body:`<p>The law wants <code>r²</code>, and <code>dx*dx + dy*dy</code> <em>is</em>
            <code>r²</code>. Taking <code>Math.sqrt</code> just to square it again is the most
            popular way to waste GPU cycles.</p>`},{title:"Hint 2 — the one-liner",body:"<p><code>return starMass / (dx * dx + dy * dy);</code></p>"}],transfer:`One-thread-per-body is the opening move of GPU physics everywhere: the CUDA SDK's
        classic <code>nbody</code> sample assigns body <em>i</em> to thread <em>i</em> exactly like
        this, and its HIP port runs the identical mapping on ROCm.`,starterCode:`// 64 bodies, one star. Each thread owns one body and asks:
// how hard does the star pull on ME?
const gpu = new GPU({ mode });

const pull = gpu.createKernel(function (posX, posY, starX, starY, starMass) {
  const dx = starX - posX[this.thread.x];
  const dy = starY - posY[this.thread.x];
  // TODO: inverse-square law — return starMass / r²,
  // where r² = dx·dx + dy·dy. (G = 1 here.)
  return 0;
}, { output: [64] });

const strength = pull(posX, posY, 0, 0, 100);
console.log('pull on body 0:', strength[0]);
`,solutionCode:`// 64 bodies, one star. Each thread owns one body and asks:
// how hard does the star pull on ME?
const gpu = new GPU({ mode });

const pull = gpu.createKernel(function (posX, posY, starX, starY, starMass) {
  const dx = starX - posX[this.thread.x];
  const dy = starY - posY[this.thread.x];
  return starMass / (dx * dx + dy * dy);
}, { output: [64] });

const strength = pull(posX, posY, 0, 0, 100);
console.log('pull on body 0:', strength[0]);
`,inputs:e=>{const t=Ft(e,64,901);return{posX:t.posX,posY:t.posY}},publicTests:[{name:"one pull strength per body — 64 positive numbers",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ft(e.utils,64,901),s=e.kernel(t.posX,t.posY,0,0,100);e.assert(s&&s.length===64,`expected 64 values, got ${s&&s.length}`);for(let n=0;n<64;n++)e.assert(Number.isFinite(s[n])&&s[n]>0,`body ${n}: a star of mass 100 should pull with positive strength, got ${s[n]}`)}},{name:"doubling the distance quarters the pull — <code>M / r²</code>",run:async e=>{const t=new Array(64),s=new Array(64);for(let a=0;a<64;a++)t[a]=a+1,s[a]=0;const n=e.kernel(t,s,0,0,100),i=(a,f)=>bt(n[a],f,.01,ba(100,(a+1)*(a+1)));e.assertClose(n[0],100,.01,"body at distance 1"),e.assertClose(n[1],25,.01,i(1,25)||"body at distance 2 (quarter the pull)"),e.assertClose(n[3],6.25,.01,i(3,6.25)||"body at distance 4 (a sixteenth)"),e.assertClose(n[9],1,.01,i(9,1)||"body at distance 10")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ft(e.utils,64,4242),s=e.kernel(t.posX,t.posY,-1.5,2.5,77);for(let n=0;n<64;n++){const i=-1.5-t.posX[n],a=2.5-t.posY[n],f=i*i+a*a,m=77/f,A=bt(s[n],m,Et(.001,m),ba(77,f));ve(e,s[n],m,.001,A||`body ${n}`)}}}]},{slug:"sum-the-sky",title:"Every Body Pulls on Every Body",intro:`<p>Real gravity has no star at the center — <strong>every body pulls on every
        other</strong>. For 64 bodies that's 64 × 63 interactions; for a million, half a trillion.
        On the GPU the shape is beautiful: the <em>outer</em> loop over bodies becomes 64 parallel
        threads, and each thread keeps a small <em>inner</em> loop over the other 63. O(n²) work,
        O(n) time per thread, all at once.</p>
        <p>One wrinkle: pulls are <strong>vectors</strong> now, not strengths. The unit direction
        from you to body <em>j</em> is <code>(dx / r, dy / r)</code>, and the strength is
        <code>mass[j] / r²</code> — multiply them and the x-component of each contribution is
        <code>mass[j] · dx / r³</code>. This kernel sums just the x-components; skip yourself, or
        you'll divide by zero.</p>`,goal:`<strong>Goal:</strong> complete the inner loop so each thread returns the net
        x-acceleration on its body: the sum of <code>mass[j] · dx / r³</code> over every other body.`,requirements:["Loop <code>j</code> over all <code>this.constants.n</code> bodies","Skip yourself — the <code>j !== this.thread.x</code> guard is already there","Accumulate <code>mass[j] * dx / (r² · r)</code> into <code>ax</code>"],hints:[{title:"Hint 1 — where does r³ come from?",body:`<p>Direction <code>dx / r</code> times strength <code>1 / r²</code> is
            <code>dx / r³</code>. With <code>r2 = dx*dx + dy*dy</code> in hand, that's
            <code>r2 * Math.sqrt(r2)</code> — one square root per pair.</p>`},{title:"Hint 2 — the loop body",body:`<pre><code>const dx = posX[j] - myX;
const dy = posY[j] - myY;
const r2 = dx * dx + dy * dy;
ax += mass[j] * dx / (r2 * Math.sqrt(r2));</code></pre>`}],transfer:`This loop-inside-a-thread is the canonical O(n²) GPU pattern. Fast CUDA and ROCm
        n-body codes keep exactly this loop but <em>tile</em> it: a thread block stages a chunk of
        bodies in shared memory so all threads reuse the loads — WebGPU's
        <code>var&lt;workgroup&gt;</code> and Metal's threadgroup memory exist for the same trick.`,starterCode:`// Newton, vectorised: this thread's body feels EVERY other body.
// The inner loop is O(n) — but all 64 of them run at once.
const gpu = new GPU({ mode });

const accelX = gpu.createKernel(function (posX, posY, mass) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  for (let j = 0; j < this.constants.n; j++) {
    if (j !== this.thread.x) {
      // TODO: dx, dy → r² → accumulate mass[j] * dx / r³
      // (dx / r is the direction, 1 / r² is the strength.)
      ax += 0;
    }
  }
  return ax;
}, { output: [64], constants: { n: 64 } });

const ax = accelX(posX, posY, mass);
console.log('net x-pull on body 0:', ax[0]);
`,solutionCode:`// Newton, vectorised: this thread's body feels EVERY other body.
// The inner loop is O(n) — but all 64 of them run at once.
const gpu = new GPU({ mode });

const accelX = gpu.createKernel(function (posX, posY, mass) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  for (let j = 0; j < this.constants.n; j++) {
    if (j !== this.thread.x) {
      const dx = posX[j] - myX;
      const dy = posY[j] - myY;
      const r2 = dx * dx + dy * dy;
      ax += mass[j] * dx / (r2 * Math.sqrt(r2));
    }
  }
  return ax;
}, { output: [64], constants: { n: 64 } });

const ax = accelX(posX, posY, mass);
console.log('net x-pull on body 0:', ax[0]);
`,inputs:e=>Ft(e,64,1702),publicTests:[{name:"pulls are real — and Newton's third law holds",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ft(e.utils,64,1702),s=e.kernel(t.posX,t.posY,t.mass);e.assert(s&&s.length===64,`expected 64 values, got ${s&&s.length}`);let n=!1,i=0;for(let a=0;a<64;a++)Math.abs(s[a])>.001&&(n=!0),i+=t.mass[a]*s[a];e.assert(n,"every net pull came out ~0 — is the loop body still empty?"),e.assertClose(i,0,.05,"Σ mass[i]·ax[i] should cancel to ~0")}},{name:"body-by-body against a reference O(n²) loop",run:async e=>{const t=Ft(e.utils,64,1702),s=e.kernel(t.posX,t.posY,t.mass);for(const n of[0,17,40,63]){const i=Nt(n,t.posX,t.posY,t.mass,0),a=bt(s[n],i[0],Et(.002,i[0]),wa(n,t,i[0],0));ve(e,s[n],i[0],.002,a||`net x-acceleration on body ${n}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ft(e.utils,64,555),s=e.kernel(t.posX,t.posY,t.mass);for(let n=0;n<64;n++){const i=Nt(n,t.posX,t.posY,t.mass,0),a=bt(s[n],i[0],Et(.002,i[0]),wa(n,t,i[0],0));ve(e,s[n],i[0],.002,a||`net x-acceleration on body ${n}`)}}}]},{slug:"softening",title:"Softening the Singularity",intro:`<p>Two of this task's bodies sit <code>0.001</code> apart. Plug that into
        <code>1 / r²</code> and their mutual pull is about a <em>million</em> — one tick of the
        clock later they're flung out of the galaxy. That's not physics; it's what happens when a
        point-mass model meets a finite time step.</p>
        <p>The standard fix is <strong>Plummer softening</strong>: replace <code>r²</code> with
        <code>r² + ε²</code>. Far away, <code>ε</code> changes nothing; up close, the force
        flattens out instead of diverging. Bonus: the <code>j !== i</code> self-check becomes dead
        weight — your own term has <code>dx = dy = 0</code>, so it contributes exactly zero. Drop
        the branch; GPUs run happiest when every thread takes the same path.</p>`,goal:`<strong>Goal:</strong> soften the kernel — use <code>r² + soft²</code>, drop the
        self-check, and return the full <code>[ax, ay]</code> pair.`,requirements:["Squared distance becomes <code>dx·dx + dy·dy + soft·soft</code>","Remove the <code>j !== this.thread.x</code> guard — the self term is now zero","Accumulate <em>both</em> components and return <code>[ax, ay]</code>"],hints:[{title:"Hint 1 — why the guard can go",body:`<p>For <code>j === i</code>: <code>dx</code> and <code>dy</code> are 0, so the
            contribution is <code>0 · something</code>. With <code>soft² &gt; 0</code> the
            denominator is never zero, so that something is a plain finite number.</p>`},{title:"Hint 2 — share the weight",body:`<p>Compute <code>const w = mass[j] / (r2 * Math.sqrt(r2));</code> once, then
            <code>ax += dx * w; ay += dy * w;</code> — one denominator, two components.</p>`}],transfer:`Softening appears verbatim in production astrophysics codes (GADGET, Bonsai) on
        CUDA and ROCm clusters. It's also a lesson in GPU numerics generally: shader float math
        never throws — a divide-by-zero silently mints <code>Infinity</code> and then
        <code>NaN</code>s spread through every sum they touch, on Metal and WebGPU alike.`,starterCode:`// Bodies 0 and 1 sit 0.001 apart. Unsoftened, their mutual pull
// is ~a million — one bad pair and the whole simulation explodes.
const gpu = new GPU({ mode });

const accel = gpu.createKernel(function (posX, posY, mass, soft) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < this.constants.n; j++) {
    if (j !== this.thread.x) {
      const dx = posX[j] - myX;
      const dy = posY[j] - myY;
      // TODO: soften — add soft·soft to r² so close encounters stay
      // finite. Then the j !== i guard above is dead weight: delete it.
      const r2 = dx * dx + dy * dy;
      const w = mass[j] / (r2 * Math.sqrt(r2));
      ax += dx * w;
      ay += dy * w;
    }
  }
  return [ax, ay];
}, { output: [64], constants: { n: 64 } });

const acc = accel(posX, posY, mass, 0.1);
console.log('acceleration on body 0:', acc[0][0], acc[0][1]);
`,solutionCode:`// Plummer softening: r² → r² + ε². Close encounters flatten out
// instead of diverging, and the self term is exactly zero — no branch.
const gpu = new GPU({ mode });

const accel = gpu.createKernel(function (posX, posY, mass, soft) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const dx = posX[j] - myX;
    const dy = posY[j] - myY;
    const r2 = dx * dx + dy * dy + soft * soft;
    const w = mass[j] / (r2 * Math.sqrt(r2));
    ax += dx * w;
    ay += dy * w;
  }
  return [ax, ay];
}, { output: [64], constants: { n: 64 } });

const acc = accel(posX, posY, mass, 0.1);
console.log('acceleration on body 0:', acc[0][0], acc[0][1]);
`,inputs:e=>{const t=tt(e,64,33);return{posX:t.posX,posY:t.posY,mass:t.mass}},publicTests:[{name:"the close pair no longer explodes — every value stays finite and small",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=tt(e.utils,64,33),s=e.kernel(t.posX,t.posY,t.mass,.1);e.assert(s&&s.length===64,`expected 64 [ax, ay] pairs, got ${s&&s.length}`);let n=!1;for(let i=0;i<64;i++){e.assert(s[i]&&s[i].length===2,`body ${i}: expected an [ax, ay] pair`);const a=Math.abs(s[i][0])+Math.abs(s[i][1]);e.assert(Number.isFinite(a)&&a<1e3,`body ${i}: |acceleration| ≈ ${a.toFixed(1)} — the 0.001-apart pair is still unsoftened`),a>.001&&(n=!0)}e.assert(n,"every acceleration came out ~0 — did the loop body survive?")}},{name:"matches the softened reference — <code>mass · d / (r² + ε²)^{3/2}</code>",run:async e=>{const t=tt(e.utils,64,33),s=e.kernel(t.posX,t.posY,t.mass,.1);for(const n of[0,1,7,63]){const i=Nt(n,t.posX,t.posY,t.mass,.1),a=bt(s[n][0],i[0],Et(.002,i[0]),rn(n,t,.1,0)),f=bt(s[n][1],i[1],Et(.002,i[1]),rn(n,t,.1,1));ve(e,s[n][0],i[0],.002,a||`ax on body ${n}`),ve(e,s[n][1],i[1],.002,f||`ay on body ${n}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=tt(e.utils,64,909),s=e.kernel(t.posX,t.posY,t.mass,.25);for(let n=0;n<64;n++){const i=Nt(n,t.posX,t.posY,t.mass,.25),a=bt(s[n][0],i[0],Et(.002,i[0]),rn(n,t,.25,0)),f=bt(s[n][1],i[1],Et(.002,i[1]),rn(n,t,.25,1));ve(e,s[n][0],i[0],.002,a||`ax on body ${n}`),ve(e,s[n][1],i[1],.002,f||`ay on body ${n}`)}}}]},{slug:"euler-step",title:"One Tick of the Clock",intro:`<p>Accelerations are just numbers until an integrator turns them into motion. The
        simplest scheme that doesn't wreck orbits is <strong>semi-implicit Euler</strong>: update
        the velocity <em>first</em>, then move the body with the <em>new</em> velocity —
        <code>v′ = v + a·dt</code>, then <code>x′ = x + v′·dt</code>. Do it in the other order
        (plain Euler) and orbits visibly spiral outward, gaining energy from nowhere.</p>
        <p>Both updates are embarrassingly parallel — body <em>i</em> never looks at body
        <em>j</em> — so they're two tiny kernels. Between them, the <code>[vx, vy]</code> pairs
        come back to JavaScript and get unpacked into plain arrays for the next kernel. Clunky?
        Yes. Instructive? Also yes — and track 2's pipeline module shows how to skip the round
        trip.</p>`,goal:`<strong>Goal:</strong> finish both kernels — <code>stepVel</code> returns
        <code>[v + a·dt]</code> per component, <code>stepPos</code> returns
        <code>[x + v·dt]</code> — and feed the position step the <em>new</em> velocities.`,requirements:["<code>stepVel</code> returns <code>[vx + ax·dt, vy + ay·dt]</code> for its body","<code>stepPos</code> returns <code>[x + vx·dt, y + vy·dt]</code> for its body","The position step must receive the <em>updated</em> velocities (semi-implicit, already wired up)"],hints:[{title:"Hint 1 — the same index four times",body:`<p>Everything in both kernels is indexed by <code>this.thread.x</code>:
            this body's velocity, this body's acceleration, this body's position.</p>`},{title:"Hint 2 — the velocity kernel",body:`<pre><code>return [velX[this.thread.x] + accX[this.thread.x] * dt,
        velY[this.thread.x] + accY[this.thread.x] * dt];</code></pre>
<p>— the position kernel is the
            same shape with <code>pos</code> and <code>vel</code>.</p>`}],transfer:`Splitting an integrator into per-buffer passes is exactly how GPU engines ship it:
        WebGPU dispatches one compute pass per update with position/velocity buffers ping-ponging
        between bind groups, and Metal encodes the same thing as back-to-back compute command
        encoders. The math stays this small; the choreography is the product.`,starterCode:`// Numbers → motion. Semi-implicit Euler: update velocity FIRST,
// then move with the NEW velocity — it keeps orbits stable.
const gpu = new GPU({ mode });

const stepVel = gpu.createKernel(function (velX, velY, accX, accY, dt) {
  // TODO: return [new vx, new vy] — old velocity plus acceleration · dt
  return [velX[this.thread.x], velY[this.thread.x]];
}, { output: [64] });

const stepPos = gpu.createKernel(function (posX, posY, velX, velY, dt) {
  // TODO: return [new x, new y] — old position plus velocity · dt
  return [posX[this.thread.x], posY[this.thread.x]];
}, { output: [64] });

const DT = 0.01;
const newVel = stepVel(velX, velY, accX, accY, DT);

// unpack the [vx, vy] pairs so the position kernel gets plain arrays
const newVelX = [];
const newVelY = [];
for (let i = 0; i < 64; i++) {
  newVelX.push(newVel[i][0]);
  newVelY.push(newVel[i][1]);
}

const newPos = stepPos(posX, posY, newVelX, newVelY, DT);
console.log('body 0 moved to', newPos[0][0], newPos[0][1]);
`,solutionCode:`// Numbers → motion. Semi-implicit Euler: update velocity FIRST,
// then move with the NEW velocity — it keeps orbits stable.
const gpu = new GPU({ mode });

const stepVel = gpu.createKernel(function (velX, velY, accX, accY, dt) {
  return [velX[this.thread.x] + accX[this.thread.x] * dt,
          velY[this.thread.x] + accY[this.thread.x] * dt];
}, { output: [64] });

const stepPos = gpu.createKernel(function (posX, posY, velX, velY, dt) {
  return [posX[this.thread.x] + velX[this.thread.x] * dt,
          posY[this.thread.x] + velY[this.thread.x] * dt];
}, { output: [64] });

const DT = 0.01;
const newVel = stepVel(velX, velY, accX, accY, DT);

// unpack the [vx, vy] pairs so the position kernel gets plain arrays
const newVelX = [];
const newVelY = [];
for (let i = 0; i < 64; i++) {
  newVelX.push(newVel[i][0]);
  newVelY.push(newVel[i][1]);
}

const newPos = stepPos(posX, posY, newVelX, newVelY, DT);
console.log('body 0 moved to', newPos[0][0], newPos[0][1]);
`,inputs:e=>{const t=tt(e,64,74),s=_r(t,.1);return{posX:t.posX,posY:t.posY,velX:t.velX,velY:t.velY,accX:s.accX,accY:s.accY}},publicTests:[{name:"the position step consumed the NEW velocities (semi-implicit)",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels (stepVel, stepPos), found ${e.kernels.length}`);const t=e.kernels[1];e.assert(Array.isArray(t.lastArgs),"stepPos was never called");const s=tt(e.utils,64,74),n=_r(s,.1),i=t.lastArgs[2],a=t.lastArgs[3];for(let f=0;f<64;f++)ve(e,i[f],s.velX[f]+n.accX[f]*.01,.001,`stepPos got a stale vx for body ${f} — did stepVel add a·dt?`),ve(e,a[f],s.velY[f]+n.accY[f]*.01,.001,`stepPos got a stale vy for body ${f}`)}},{name:"velocity kernel: <code>v' = v + a·dt</code>",run:async e=>{const t=e.kernels[0],s=new Array(64),n=new Array(64),i=new Array(64),a=new Array(64);for(let m=0;m<64;m++)s[m]=m*.1-3,n[m]=2-m*.05,i[m]=Math.sin(m)*4,a[m]=Math.cos(m)*4;const f=t(s,n,i,a,.5);for(let m=0;m<64;m++)ve(e,f[m][0],s[m]+i[m]*.5,.001,_t(f[m][0],s[m],i[m],.5,.001)||`vx of body ${m}`),ve(e,f[m][1],n[m]+a[m]*.5,.001,_t(f[m][1],n[m],a[m],.5,.001)||`vy of body ${m}`)}},{name:"position kernel: <code>x' = x + v·dt</code>",run:async e=>{const t=e.kernels[1],s=new Array(64),n=new Array(64),i=new Array(64),a=new Array(64);for(let m=0;m<64;m++)s[m]=m*.25,n[m]=-m*.125,i[m]=1+m*.02,a[m]=-2+m*.03;const f=t(s,n,i,a,.2);for(let m=0;m<64;m++)ve(e,f[m][0],s[m]+i[m]*.2,.001,_t(f[m][0],s[m],i[m],.2,.001)||`x of body ${m}`),ve(e,f[m][1],n[m]+a[m]*.2,.001,_t(f[m][1],n[m],a[m],.2,.001)||`y of body ${m}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.utils.seededRandom(31),s=[];for(let Y=0;Y<6;Y++){const Ae=new Array(64);for(let Ze=0;Ze<64;Ze++)Ae[Ze]=t()*4-2;s.push(Ae)}const[n,i,a,f,m,A]=s,N=e.kernels[0](n,i,a,f,.025),H=e.kernels[1](m,A,n,i,.025);for(let Y=0;Y<64;Y++)ve(e,N[Y][0],n[Y]+a[Y]*.025,.001,_t(N[Y][0],n[Y],a[Y],.025,.001)||`vx of body ${Y}`),ve(e,N[Y][1],i[Y]+f[Y]*.025,.001,_t(N[Y][1],i[Y],f[Y],.025,.001)||`vy of body ${Y}`),ve(e,H[Y][0],m[Y]+n[Y]*.025,.001,_t(H[Y][0],m[Y],n[Y],.025,.001)||`x of body ${Y}`),ve(e,H[Y][1],A[Y]+i[Y]*.025,.001,_t(H[Y][1],A[Y],i[Y],.025,.001)||`y of body ${Y}`)}}]},{slug:"full-simulation",title:"Put It Together: 128 Bodies",intro:`<p>Everything from this module, running as one machine. The three kernels below are
        your last three tasks — softened O(n²) acceleration, the velocity tick, the position tick.
        What's missing is the <strong>conductor</strong>: a JavaScript loop that runs ten full
        ticks, feeding each kernel's output into the next and carrying the new state into the next
        tick.</p>
        <p>Notice who does what: JavaScript never touches a single interaction — it just passes
        arrays around. The GPU grinds through 128 × 128 = 16,384 interactions per tick, 163,840
        across the run. Swap 128 for 100,000 and this exact structure is a galaxy simulator; the
        loop you're about to write wouldn't change by a character.</p>`,goal:`<strong>Goal:</strong> write the simulation loop — ten ticks of
        <code>accel → stepVel → stepPos</code>, carrying the new arrays forward each time.`,requirements:["Each tick: accelerations first — <code>accel(px, py, mass, SOFT)</code>","Unpack the pairs, then <code>stepVel(vx, vy, ax, ay, DT)</code>, then <code>stepPos</code> with the <em>new</em> velocities","Reassign <code>px, py, vx, vy</code> so the next tick starts from the new state","Run exactly <code>STEPS</code> ticks, then log body 0's final position"],hints:[{title:"Hint 1 — the shape of one tick",body:`<p>Inside the loop: call <code>accel</code>, unpack its pairs into
            <code>ax, ay</code> arrays (the <code>unpack</code> helper is right there), call
            <code>stepVel</code>, unpack, call <code>stepPos</code>, unpack.</p>`},{title:"Hint 2 — carrying the state",body:`<p>End every tick by overwriting the state:</p>
<pre><code>vx = newVx;
vy = newVy;
px = newPx;
py = newPy;</code></pre>
<p>— next tick's
            <code>accel</code> must see the moved bodies, or time never advances.</p>`},{title:"Hint 3 — the whole loop",body:`<pre><code>for (let step = 0; step &lt; STEPS; step++) {
  const [ax, ay] = unpack(accel(px, py, mass, SOFT));
  const [nvx, nvy] = unpack(stepVel(vx, vy, ax, ay, DT));
  const [npx, npy] = unpack(stepPos(px, py, nvx, nvy, DT));
  px = npx; py = npy; vx = nvx; vy = nvy;
}</code></pre>`}],transfer:`A host loop launching device kernels in sequence is the universal skeleton of GPU
        simulation: CUDA streams queueing kernel after kernel per timestep, WebGPU building one
        command encoder per frame, Metal committing a command buffer per tick. Production codes
        differ mainly in never reading the arrays back between passes — that's what track 2's
        pipeline textures are for.`,starterCode:`// Three kernels from the last three tasks — and a conductor's podium.
const gpu = new GPU({ mode });
const N = 128;
const DT = 0.01;
const SOFT = 0.1;
const STEPS = 10;

const accel = gpu.createKernel(function (posX, posY, mass, soft) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const dx = posX[j] - myX;
    const dy = posY[j] - myY;
    const r2 = dx * dx + dy * dy + soft * soft;
    const w = mass[j] / (r2 * Math.sqrt(r2));
    ax += dx * w;
    ay += dy * w;
  }
  return [ax, ay];
}, { output: [N], constants: { n: N } });

const stepVel = gpu.createKernel(function (velX, velY, accX, accY, dt) {
  return [velX[this.thread.x] + accX[this.thread.x] * dt,
          velY[this.thread.x] + accY[this.thread.x] * dt];
}, { output: [N] });

const stepPos = gpu.createKernel(function (posX, posY, velX, velY, dt) {
  return [posX[this.thread.x] + velX[this.thread.x] * dt,
          posY[this.thread.x] + velY[this.thread.x] * dt];
}, { output: [N] });

// [x, y] pairs → two plain arrays
function unpack(pairs) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < pairs.length; i++) {
    xs.push(pairs[i][0]);
    ys.push(pairs[i][1]);
  }
  return [xs, ys];
}

let px = posX;
let py = posY;
let vx = velX;
let vy = velY;

for (let step = 0; step < STEPS; step++) {
  // TODO — one full tick:
  //   1. pairs = accel(px, py, mass, SOFT), unpack into ax, ay
  //   2. stepVel with DT → unpack into the NEW vx, vy
  //   3. stepPos with the NEW velocities → unpack into the new px, py
  //   4. reassign px, py, vx, vy for the next tick
}

console.log('after', STEPS, 'ticks, body 0 is at', px[0], py[0]);
`,solutionCode:`// Three kernels from the last three tasks — and a conductor's podium.
const gpu = new GPU({ mode });
const N = 128;
const DT = 0.01;
const SOFT = 0.1;
const STEPS = 10;

const accel = gpu.createKernel(function (posX, posY, mass, soft) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const dx = posX[j] - myX;
    const dy = posY[j] - myY;
    const r2 = dx * dx + dy * dy + soft * soft;
    const w = mass[j] / (r2 * Math.sqrt(r2));
    ax += dx * w;
    ay += dy * w;
  }
  return [ax, ay];
}, { output: [N], constants: { n: N } });

const stepVel = gpu.createKernel(function (velX, velY, accX, accY, dt) {
  return [velX[this.thread.x] + accX[this.thread.x] * dt,
          velY[this.thread.x] + accY[this.thread.x] * dt];
}, { output: [N] });

const stepPos = gpu.createKernel(function (posX, posY, velX, velY, dt) {
  return [posX[this.thread.x] + velX[this.thread.x] * dt,
          posY[this.thread.x] + velY[this.thread.x] * dt];
}, { output: [N] });

// [x, y] pairs → two plain arrays
function unpack(pairs) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < pairs.length; i++) {
    xs.push(pairs[i][0]);
    ys.push(pairs[i][1]);
  }
  return [xs, ys];
}

let px = posX;
let py = posY;
let vx = velX;
let vy = velY;

for (let step = 0; step < STEPS; step++) {
  const [ax, ay] = unpack(accel(px, py, mass, SOFT));
  const [nvx, nvy] = unpack(stepVel(vx, vy, ax, ay, DT));
  const [npx, npy] = unpack(stepPos(px, py, nvx, nvy, DT));
  px = npx;
  py = npy;
  vx = nvx;
  vy = nvy;
}

console.log('after', STEPS, 'ticks, body 0 is at', px[0], py[0]);
`,inputs:e=>tt(e,128,55),publicTests:[{name:"all ten ticks ran — the final tick saw step-nine positions",run:async e=>{e.assert(e.kernels.length>=3,`expected 3 kernels (accel, stepVel, stepPos), found ${e.kernels.length}`);const t=e.kernels[0];e.assert(Array.isArray(t.lastArgs),"the accel kernel was never called — is the loop wired up?");const s=tt(e.utils,128,55),n=rr(s,9,.01,.1),i=t.lastArgs[0],a=t.lastArgs[1];for(let f=0;f<128;f+=7)ve(e,i[f],n.posX[f],.005,`tick 10 saw a wrong x for body ${f} — is the new state carried between ticks?`),ve(e,a[f],n.posY[f],.005,`tick 10 saw a wrong y for body ${f}`)}},{name:"momentum is conserved across the whole run",run:async e=>{const t=e.kernels[1];e.assert(Array.isArray(t.lastArgs),"stepVel was never called — is the loop wired up?");const s=t(...t.lastArgs),n=tt(e.utils,128,55);let i=0,a=0,f=0,m=0;for(let A=0;A<128;A++)i+=n.mass[A]*n.velX[A],a+=n.mass[A]*n.velY[A],f+=n.mass[A]*s[A][0],m+=n.mass[A]*s[A][1];e.assertClose(f,i,.05,"total x-momentum drifted — forces should cancel pairwise"),e.assertClose(m,a,.05,"total y-momentum drifted")}},{name:"one tick, rebuilt from scratch, matches the physics",run:async e=>{const t=tt(e.utils,128,55),s=xa(e.kernels[0],e.kernels[1],e.kernels[2],t,t.mass,.01,.1),n=rr(t,1,.01,.1);for(const i of[0,1,42,127])ve(e,s.posX[i],n.posX[i],.002,`x of body ${i} after one tick`),ve(e,s.posY[i],n.posY[i],.002,`y of body ${i} after one tick`),ve(e,s.velX[i],n.velX[i],.002,`vx of body ${i} after one tick`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=tt(e.utils,128,991);let s=t;for(let i=0;i<5;i++)s=xa(e.kernels[0],e.kernels[1],e.kernels[2],s,t.mass,.01,.1);const n=rr(t,5,.01,.1);for(let i=0;i<128;i++)ve(e,s.posX[i],n.posX[i],.005,`x of body ${i} after five ticks`),ve(e,s.posY[i],n.posY[i],.005,`y of body ${i} after five ticks`)}}]}]},iu=Object.freeze({__proto__:null,default:ru});function xe(e,t,s){return(s*e+t)*4}function ts(e){return 64+40*Math.sin(e*2*Math.PI/128)}function ks(e,t,s){const n=[];for(let i=0;i<t;i++)e[xe(t,s,i)]>128&&n.push(i);return n}function Ts(e){let t=0;for(let s=0;s<e.length;s++)t+=e[s];return t/e.length}function an(e,t){const s=e-63.5,n=t-63.5,i=Math.sqrt(s*s+n*n),a=.5+.5*Math.cos(i*.35),f=Math.max(0,1-i/96),m=a*f;return[.4*m*255,.75*m*255,m*255]}function st(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function va(e,t){const s="red is following the row instead of the column — the horizontal ramp is this.thread.x / 128";return[[255*t/128,s],[255*(127-t)/128,s],[Math.min(255,e*255),"color channels run 0–1, so an undivided this.thread.x saturates every column past the first — divide it by 128"]]}function au(e,t,s){return Math.abs(e-t)<=2&&Math.abs(e-255*s/128)<=3?"green is constant down the canvas and matches this column's x ramp — this.thread.x and this.thread.y are swapped; green rises with this.thread.y":null}function ka(e,t){const s=e[xe(128,0,t)];let n=1;for(;n<128&&e[xe(128,n,t)]===s;)n++;return n===16||n===128?null:`your cells are ${n} pixels wide, not 16 — the cell index is Math.floor(coordinate / 16)`}function Ta(e,t){return[[t(64+40*Math.sin(e)),"that is Math.sin() of the raw pixel count — one period across 128 px needs x * 2 * Math.PI / 128"],[t(64+40*Math.sin(e*2*Math.PI/64)),"two periods fit the canvas — divide x by 128, the full width, for one"],[t(64+40*Math.sin(e*360/128)),"Math.sin takes radians, not degrees — the scale is 2 * Math.PI / 128"],[t(64),"the curve is still the constant 64 — it never became a function of x"]]}function Gt(e,t,s){const n=[.4,.75,1][s],i=e-63.5,a=t-63.5,f=Math.sqrt(i*i+a*a),m=.5+.5*Math.cos(f*.35),A=Math.max(0,1-f/96);return[[m,"the fade never got multiplied in — v = wave * fade"],[A,"that is the bare fade — the cosine ripple is missing from v"],[Math.max(0,Math.cos(f*.35))*A,"the cosine still swings negative — remap it with 0.5 + 0.5 * Math.cos(r * 0.35)"]].map(N=>[Math.min(1,N[0])*n*255,N[1]])}var ou={id:"3-1",track:3,title:"Pixels from Scratch",blurb:"Graphical kernels and <code>this.color()</code>: gradients, patterns and plots, one thread per pixel.",tasks:[{slug:"coordinate-gradient",title:"Paint with Coordinates",intro:`<p>Set <code>graphical: true</code> and a kernel stops returning numbers — instead
        every thread paints <strong>exactly one pixel</strong> by calling
        <code>this.color(r, g, b, a)</code>, channels 0–1. The output shape becomes the canvas:
        <code>output: [128, 128]</code> is a 128×128 picture, 16,384 threads, one per pixel.</p>
        <p>A solid color is one line — and the starter already paints one. The interesting part is
        that each thread knows <em>where</em> it is: <code>this.thread.x</code> counts columns from
        the left, <code>this.thread.y</code> counts rows from the <strong>bottom</strong> (GL
        convention). Divide either by the canvas size and you get a smooth 0–1 ramp, ready to feed
        straight into a color channel.</p>`,goal:`<strong>Goal:</strong> turn the flat gray into a two-axis gradient — red rising with
        <code>x</code>, green rising with <code>y</code>, blue fixed at <code>0.5</code>.`,requirements:["Keep <code>graphical: true</code> and <code>output: [128, 128]</code>","Red channel = <code>this.thread.x / 128</code>","Green channel = <code>this.thread.y / 128</code>","Blue stays <code>0.5</code>, alpha stays <code>1</code>"],hints:[{title:"Hint 1 — where am I?",body:`<p><code>this.thread.x</code> runs 0…127 here, so
            <code>this.thread.x / 128</code> runs 0…0.992 — a ready-made red ramp.
            Same move with <code>this.thread.y</code> for green.</p>`},{title:"Hint 2 — the one-liner",body:`<p>The whole kernel body:</p>
<pre><code>this.color(this.thread.x / 128, this.thread.y / 128, 0.5, 1);</code></pre>`}],transfer:`Normalized pixel coordinates are the <em>uv</em> every shader language starts
        from: WebGPU and Metal fragment shaders derive them from the fragment position, and CUDA
        image kernels divide thread indices by the image width the same way. The famous red-green
        "uv debug gradient" is exactly this kernel.`,starterCode:`// graphical: true turns a kernel into a painter — one thread per pixel.
const gpu = new GPU({ mode });

const gradient = gpu.createKernel(function () {
  // Right now all 16,384 threads paint the SAME color.
  // TODO: mix this thread's coordinates into the color —
  //   red   = this.thread.x / 128
  //   green = this.thread.y / 128
  //   blue  = 0.5
  this.color(0.2, 0.2, 0.2, 1);
}, {
  output: [128, 128],
  graphical: true,
});

gradient();
render(gradient.canvas);
`,solutionCode:`// graphical: true turns a kernel into a painter — one thread per pixel.
const gpu = new GPU({ mode });

const gradient = gpu.createKernel(function () {
  this.color(this.thread.x / 128, this.thread.y / 128, 0.5, 1);
}, {
  output: [128, 128],
  graphical: true,
});

gradient();
render(gradient.canvas);
`,publicTests:[{name:"paints a graphical <code>128×128</code> canvas",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===128&&e.canvas.height===128,`expected a 128×128 canvas, got ${e.canvas.width}×${e.canvas.height}`),e.assert(e.getPixels().length===16384*4,"pixel buffer should hold 128×128 RGBA values")}},{name:"red rises left to right: <code>this.thread.x / 128</code>",run:async e=>{const t=e.getPixels();for(const s of[3,64,124])for(let n=0;n<128;n+=7){const i=t[xe(128,n,s)],a=255*n/128,f=st(i,a,2.5,va(n,s));e.assertClose(i,a,2.5,f||`red at column ${n} (buffer row ${s})`)}}},{name:"green rises with <code>this.thread.y</code>; blue holds at 0.5",run:async e=>{const t=e.getPixels(),s=t[xe(128,20,0)+1],n=t[xe(128,20,127)+1];e.assert(Math.min(s,n)<=4&&Math.max(s,n)>=248,au(s,n,20)||`green should ramp 0 → 252 across the canvas, got edge values ${s} and ${n}`);for(const[i,a]of[[10,10],[90,40],[64,100]])e.assertClose(t[xe(128,i,a)+2],127.5,2.5,`blue at (${i}, row ${a})`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.getPixels(),s=t[xe(128,5,0)+1]<t[xe(128,5,127)+1];for(let n=0;n<128;n+=5){const i=s?n:127-n;for(let a=0;a<128;a+=5){const f=xe(128,a,n),m=st(t[f],255*a/128,2.5,va(a,n));e.assertClose(t[f],255*a/128,2.5,m||`red at (${a}, y=${i})`),e.assertClose(t[f+1],255*i/128,2.5,`green at (${a}, y=${i})`),e.assertClose(t[f+2],127.5,2.5,`blue at (${a}, y=${i})`),e.assert(t[f+3]===255,`alpha at (${a}, y=${i}) should be 255`)}}}}]},{slug:"checkerboard",title:"Checkerboard Logic",intro:`<p>Smooth ramps become hard-edged patterns with two tools:
        <code>Math.floor</code> to chop coordinates into cells, and the remainder operator
        <code>%</code> to make the cells repeat. <code>Math.floor(this.thread.x / 16)</code> asks
        <em>"which 16-pixel band am I in?"</em> — and <code>% 2</code> answers
        <em>"odd or even?"</em>.</p>
        <p>The starter already draws vertical stripes with exactly that trick. A checkerboard is
        the same idea in both axes at once: compute a cell index for x <em>and</em> y, add them,
        and take the parity of the sum — cells that touch on an edge always disagree.</p>`,goal:`<strong>Goal:</strong> upgrade the stripes to an 8×8 checkerboard of 16-pixel cells —
        paint <code>(cellX + cellY) % 2</code> into all three color channels.`,requirements:["Keep the cells 16 pixels: <code>Math.floor(coordinate / 16)</code>","Combine both axes: parity of <code>cellX + cellY</code>","Pure black and white only — the parity (0 or 1) is the color"],hints:[{title:"Hint 1 — the second axis",body:`<p>Mirror the existing line for y:</p>
<pre><code>const cellY = Math.floor(this.thread.y / 16);</code></pre>`},{title:"Hint 2 — why the sum?",body:`<p>Moving one cell right changes <code>cellX</code> by 1; moving one cell up
            changes <code>cellY</code> by 1. Either move flips the parity of
            <code>cellX + cellY</code> — which is exactly what a checkerboard does. So:
            <code>const v = (cellX + cellY) % 2;</code></p>`}],transfer:`Procedural patterns are a GPU staple: GLSL and WGSL shaders build checkers,
        stripes and grids from <code>floor()</code> and <code>mod()</code> with no texture in
        sight, and CUDA kernels lean on the same modular arithmetic on thread ids to stripe work
        across blocks.`,starterCode:`// Modular arithmetic turns smooth coordinates into repeating patterns.
const gpu = new GPU({ mode });

const board = gpu.createKernel(function () {
  // Stripes: which 16-pixel column band is this thread in — odd or even?
  const cellX = Math.floor(this.thread.x / 16);
  const v = cellX % 2;
  // TODO: bring this.thread.y into it. A checkerboard flips parity every
  // 16 pixels vertically too — (cellX + cellY) is the trick.
  this.color(v, v, v, 1);
}, {
  output: [128, 128],
  graphical: true,
});

board();
render(board.canvas);
`,solutionCode:`// Modular arithmetic turns smooth coordinates into repeating patterns.
const gpu = new GPU({ mode });

const board = gpu.createKernel(function () {
  const cellX = Math.floor(this.thread.x / 16);
  const cellY = Math.floor(this.thread.y / 16);
  const v = (cellX + cellY) % 2;
  this.color(v, v, v, 1);
}, {
  output: [128, 128],
  graphical: true,
});

board();
render(board.canvas);
`,publicTests:[{name:"every pixel is pure black or pure white",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.getPixels();e.assert(t.length===16384*4,"expected a 128×128 canvas");for(let s=0;s<t.length;s+=4){const n=t[s];e.assert(n<=1||n>=254,`pixel at byte ${s} is gray (${n}) — the parity should be exactly 0 or 1`),e.assert(Math.abs(t[s+1]-n)<=1&&Math.abs(t[s+2]-n)<=1,`pixel at byte ${s} is tinted — use the same value for r, g and b`)}}},{name:"cells are 16 pixels wide and alternate along x",run:async e=>{const t=e.getPixels();for(const s of[8,40,100]){const n=t[xe(128,2,s)],i=t[xe(128,13,s)];e.assert(Math.abs(n-i)<=1,ka(t,s)||`columns 2 and 13 share a 16-px cell but differ on buffer row ${s}`);const a=t[xe(128,8,s)],f=t[xe(128,24,s)];e.assert(Math.abs(a-f)>=250,ka(t,s)||`columns 8 and 24 are in adjacent cells but match on buffer row ${s} — still stripes?`)}}},{name:"cells alternate along y too — that's what makes it a checkerboard",run:async e=>{const t=e.getPixels();for(const s of[8,40,100]){const n=t[xe(128,s,2)],i=t[xe(128,s,13)];e.assert(Math.abs(n-i)<=1,`rows 2 and 13 share a 16-px cell but differ in column ${s}`);const a=t[xe(128,s,8)],f=t[xe(128,s,24)];e.assert(Math.abs(a-f)>=250,`rows 8 and 24 are in adjacent cells but match in column ${s} — did you use this.thread.y?`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.getPixels(),s=t[0]>=254?255:0;let n=0;for(let i=0;i<128;i++)for(let a=0;a<128;a++){const m=(Math.floor(a/16)+Math.floor(i/16))%2===0?s:255-s,A=t[xe(128,a,i)];e.assert(Math.abs(A-m)<=1,`pixel (${a}, row ${i}) breaks the checkerboard: got ${A}`),A>=254&&n++}e.assert(n===16384/2,`expected exactly half the pixels white, got ${n}`)}}]},{slug:"plot-a-wave",title:"Plot a Function",intro:`<p>How do you plot <code>y = f(x)</code> when no thread can draw a line? Flip the
        question: every pixel decides <em>for itself</em> whether it lies on the curve. Thread
        <code>(x, y)</code> evaluates the function at its own x, measures the vertical distance to
        that height, and paints amber if the distance is under 2 pixels — background otherwise.</p>
        <p>This per-pixel <em>"how far am I from the shape?"</em> question is one of the great
        tricks of computer graphics. Today it draws a sine wave; the same idea, pushed further,
        draws the fractals of module 3.2 and the ray-marched scenes of module 3.5.</p>`,goal:`<strong>Goal:</strong> plot one full period of
        <code>y = 64 + 40 · sin(2πx / 128)</code> as a thin amber curve on the dark background.`,requirements:["Compute the curve height for this thread's x: <code>64 + 40 * Math.sin(x * 2 * Math.PI / 128)</code>","Light the pixel when <code>Math.abs(this.thread.y - curveY) &lt; 2</code>","Keep the amber-on-dark colors from the starter"],hints:[{title:"Hint 1 — one line changes",body:`<p>The distance test and both colors are already written. Only
            <code>curveY</code> is wrong: it's a constant, so you get a flat line instead of a
            wave.</p>`},{title:"Hint 2 — the curve",body:`<pre><code>const curveY = 64 + 40 * Math.sin(x * 2 * Math.PI / 128);</code></pre>
<p><code>Math.sin</code> and <code>Math.PI</code> both work inside kernels.</p>`}],transfer:`Distance-to-shape rendering is how GPUs draw crisp text and vector art at any
        zoom (signed distance fields), and it's the engine behind every Shadertoy graph you've
        seen: a WGSL or Metal fragment shader evaluating <code>f(x)</code> per fragment, exactly
        as here.`,starterCode:`// A plot is a per-pixel question: how far am I from the curve?
const gpu = new GPU({ mode });

const plot = gpu.createKernel(function () {
  const x = this.thread.x;
  // TODO: make this a real curve —
  //   y = 64 + 40 * Math.sin(x * 2 * Math.PI / 128)
  const curveY = 64;
  if (Math.abs(this.thread.y - curveY) < 2) {
    this.color(1, 0.85, 0.3, 1);      // on the curve — amber
  } else {
    this.color(0.06, 0.07, 0.1, 1);   // background — near black
  }
}, {
  output: [128, 128],
  graphical: true,
});

plot();
render(plot.canvas);
`,solutionCode:`// A plot is a per-pixel question: how far am I from the curve?
const gpu = new GPU({ mode });

const plot = gpu.createKernel(function () {
  const x = this.thread.x;
  const curveY = 64 + 40 * Math.sin(x * 2 * Math.PI / 128);
  if (Math.abs(this.thread.y - curveY) < 2) {
    this.color(1, 0.85, 0.3, 1);      // on the curve — amber
  } else {
    this.color(0.06, 0.07, 0.1, 1);   // background — near black
  }
}, {
  output: [128, 128],
  graphical: true,
});

plot();
render(plot.canvas);
`,publicTests:[{name:"a thin curve on a dark background",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.getPixels();e.assert(t.length===16384*4,"expected a 128×128 canvas");let s=0;for(let i=0;i<t.length;i+=4)t[i]>128?s++:e.assert(t[i]<40,`background pixel at byte ${i} is not dark (red ${t[i]})`);const n=s/16384;e.assert(n>.01&&n<.15,`expected a thin curve (1–15% of pixels lit), got ${(n*100).toFixed(1)}%`)}},{name:"every column crosses the curve exactly once",run:async e=>{const t=e.getPixels();for(let s=0;s<128;s+=4){const n=ks(t,128,s);e.assert(n.length>=1,`column ${s} has no lit pixels`),e.assert(n.length<=14,`column ${s} has ${n.length} lit pixels — the band should stay thin`),e.assert(n[n.length-1]-n[0]===n.length-1,`column ${s} lights two separate bands — the curve should cross it once`)}}},{name:"the curve follows <code>64 + 40·sin(2πx/128)</code>",run:async e=>{const t=e.getPixels(),s=Ts(ks(t,128,32)),n=Math.abs(s-ts(32))<=4,i=Math.abs(s-(127-ts(32)))<=4;e.assert(n||i,`at x=32 the curve should sit ~40 px from the middle (y≈104), found its center at buffer row ${s.toFixed(1)}`);const a=f=>n?f:127-f;for(const f of[0,8,16,32,48,64,80,96,112,120]){const m=Ts(ks(t,128,f)),A=a(ts(f)),N=st(m,A,3,Ta(f,a));e.assertClose(m,A,3,N||`curve center in column ${f}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.getPixels(),s=Ts(ks(t,128,32)),n=Math.abs(s-ts(32))<=4;e.assert(n||Math.abs(s-(127-ts(32)))<=4,"curve peak is not where sin() puts it");for(let i=0;i<128;i++){const a=ks(t,128,i);e.assert(a.length>=1&&a.length<=14,`column ${i}: ${a.length} lit pixels`);const f=N=>n?N:127-N,m=f(ts(i)),A=st(Ts(a),m,3,Ta(i,f));e.assertClose(Ts(a),m,3,A||`curve center in column ${i}`)}}}]},{slug:"radial-ripples",title:"Ripples: Think in Polar",intro:`<p>Gradients, cells and curves all thought in x and y. The last move of this module
        is to change coordinate systems <em>inside the kernel</em>: subtract the canvas center
        (63.5, 63.5 — halfway between the two middle rows and columns), and Pythagoras turns the
        thread's position into a <strong>radius</strong>. Anything you compute from that radius is
        automatically a perfect circle.</p>
        <p>Feed the radius into a cosine and you get concentric ripples; multiply by a fade so
        they die out toward the edge; tint the channels and the flat canvas turns into water.
        One gotcha, handled in the starter: <code>this.thread.x</code> is an <em>integer</em> on
        the GPU, so promote it to a float (<code>this.thread.x / 1</code>) before subtracting the
        fractional center — otherwise the GPU rounds your 63.5 away.</p>`,goal:`<strong>Goal:</strong> finish the ripple kernel — a cosine wave over the radius,
        faded toward the edge, tinted blue: <code>this.color(0.4v, 0.75v, v, 1)</code>.`,requirements:["Radius from the center: <code>Math.sqrt(dx*dx + dy*dy)</code> with dx, dy relative to (63.5, 63.5)","Ripple: <code>wave = 0.5 + 0.5 * Math.cos(r * 0.35)</code>","Fade: <code>fade = Math.max(0, 1 - r / 96)</code>, then <code>v = wave * fade</code>","Blue tint: channels <code>0.4*v</code>, <code>0.75*v</code>, <code>v</code>"],hints:[{title:"Hint 1 — the ripple",body:`<p><code>Math.cos(r * 0.35)</code> swings between −1 and 1 as r grows —
            <code>0.5 + 0.5 * cos</code> remaps that to 0…1, a crest roughly every 18 pixels.</p>`},{title:"Hint 2 — the last three lines",body:`<pre><code>const wave = 0.5 + 0.5 * Math.cos(r * 0.35);
const v = wave * Math.max(0, 1 - r / 96);
this.color(0.4 * v, 0.75 * v, v, 1);</code></pre>`}],transfer:`Radius-and-angle reasoning is everywhere in GPU code: vignette and
        lens-distortion passes in Metal and WebGPU post-processing, CUDA and ROCm image warps
        that resample in polar space, every "tunnel" demo ever shipped. Center the coordinates,
        transform them, color by the result — that opening move never changes.`,starterCode:`// Change coordinates INSIDE the kernel: position → radius from center.
const gpu = new GPU({ mode });

const ripples = gpu.createKernel(function () {
  // thread ids are integers — "/ 1" promotes them to floats so the
  // half-pixel center stays exact on the GPU
  const dx = this.thread.x / 1 - 63.5;
  const dy = this.thread.y / 1 - 63.5;
  const r = Math.sqrt(dx * dx + dy * dy);
  const fade = Math.max(0, 1 - r / 96);
  // TODO: 1) ripple — wave = 0.5 + 0.5 * Math.cos(r * 0.35)
  //       2) combine — v = wave * fade
  //       3) tint    — this.color(0.4 * v, 0.75 * v, v, 1)
  this.color(fade, fade, fade, 1);
}, {
  output: [128, 128],
  graphical: true,
});

ripples();
render(ripples.canvas);
`,solutionCode:`// Change coordinates INSIDE the kernel: position → radius from center.
const gpu = new GPU({ mode });

const ripples = gpu.createKernel(function () {
  // thread ids are integers — "/ 1" promotes them to floats so the
  // half-pixel center stays exact on the GPU
  const dx = this.thread.x / 1 - 63.5;
  const dy = this.thread.y / 1 - 63.5;
  const r = Math.sqrt(dx * dx + dy * dy);
  const wave = 0.5 + 0.5 * Math.cos(r * 0.35);  // crest every ~18 px
  const fade = Math.max(0, 1 - r / 96);         // dim toward the edge
  const v = wave * fade;
  this.color(0.4 * v, 0.75 * v, v, 1);
}, {
  output: [128, 128],
  graphical: true,
});

ripples();
render(ripples.canvas);
`,publicTests:[{name:"the picture is radially symmetric about the center",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.getPixels();e.assert(t.length===16384*4,"expected a 128×128 canvas");for(const[s,n]of[[10,30],[45,8],[70,100],[120,60],[33,33]]){const i=xe(128,s,n),a=xe(128,127-s,n),f=xe(128,s,127-n);for(let m=0;m<3;m++)e.assert(Math.abs(t[i+m]-t[a+m])<=2,`pixel (${s}, row ${n}) and its horizontal mirror disagree — is the center at (63.5, 63.5)?`),e.assert(Math.abs(t[i+m]-t[f+m])<=2,`pixel (${s}, row ${n}) and its vertical mirror disagree — is the center at (63.5, 63.5)?`)}}},{name:"crests and troughs land where <code>cos(0.35r)</code> puts them",run:async e=>{const t=e.getPixels();for(const i of[64,73,81,99,120]){const a=xe(128,i,63),f=an(i,63.5+.5),m=st(t[a+2],f[2],3,Gt(i,64,2));e.assertClose(t[a+2],f[2],3,m||`blue in column ${i} of the center row`)}const s=t[xe(128,64,63)+2],n=t[xe(128,73,63)+2];e.assert(s-n>200,`the first trough should be nearly black next to the bright center (got ${s} vs ${n})`)}},{name:"blue tint and edge fade: <code>b &gt; g &gt; r</code>, corners dark",run:async e=>{const t=e.getPixels(),s=xe(128,81,63),[n,i,a]=an(81,64);e.assertClose(t[s],n,3,st(t[s],n,3,Gt(81,64,0))||"red on the first ring"),e.assertClose(t[s+1],i,3,st(t[s+1],i,3,Gt(81,64,1))||"green on the first ring"),e.assertClose(t[s+2],a,3,st(t[s+2],a,3,Gt(81,64,2))||"blue on the first ring"),e.assert(t[s+2]>t[s+1]&&t[s+1]>t[s],"ring pixels should be tinted blue: b > g > r");for(const[f,m]of[[0,0],[127,0],[0,127],[127,127]]){const A=xe(128,f,m),[N,H,Y]=an(f<64?0:127,m<64?0:127);e.assertClose(t[A],N,3,`red in corner (${f}, row ${m})`),e.assertClose(t[A+1],H,3,`green in corner (${f}, row ${m})`),e.assertClose(t[A+2],Y,3,`blue in corner (${f}, row ${m})`),e.assert(t[s+2]-t[A+2]>150,`corner (${f}, row ${m}) should be far dimmer than the first ring`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.getPixels();for(let s=0;s<128;s+=3)for(let n=0;n<128;n+=3){const i=xe(128,n,s),[a,f,m]=an(n,s);e.assertClose(t[i],a,3,st(t[i],a,3,Gt(n,s,0))||`red at (${n}, row ${s})`),e.assertClose(t[i+1],f,3,st(t[i+1],f,3,Gt(n,s,1))||`green at (${n}, row ${s})`),e.assertClose(t[i+2],m,3,st(t[i+2],m,3,Gt(n,s,2))||`blue at (${n}, row ${s})`)}}}]}]},lu=Object.freeze({__proto__:null,default:ou});const os=100;function Wa(e,t,s,n,i=os){let a=e,f=t,m=0;for(let A=0;A<i;A++)if(a*a+f*f<4){const N=a*a-f*f+s;f=2*a*f+n,a=N,m+=1}return{count:m,zr:a,zi:f}}function on(e,t,s,n,i=os){const{count:a,zr:f,zi:m}=Wa(e,t,s,n,i);return a>=i?i:a+1-Math.log2(.5*Math.log2(f*f+m*m))}function ar(e){return[Math.round(e*255),Math.round(e*e*255),Math.round((.5+.5*e)*255)]}function ct(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function or(e,t,s,n,i){const a=e+n*s,f=t+i*s;return[[Math.sqrt(a*a+f*f),"that is |c|, not |c|² — return cr * cr + ci * ci without the square root"],[n*s*(n*s)+i*s*(i*s),"the view offsets never got added — cr is xMin + x * step, ci is yMin + y * step"]]}function ln(){return[[os,"every point reached the 100 cap — counting continued after z escaped, so the guard zr * zr + zi * zi < 4 is either missing or not wrapping the count"],[0,"count came back 0 — no guarded pass ever ran; the guard admits z while zr * zr + zi * zi is BELOW 4"]]}function Sa(e,t,s){return e>=253&&t>=253&&s>=253?"the interior came out white — count = 100 pixels are falling into the shade branch; shade only when count < 100 and paint the rest black":null}function lr(e,t){const s=Wa(e,t),n=s.zr*s.zr+s.zi*s.zi;return[[s.count,"that is the raw integer count — the fractional correction 1 − log2(0.5 · log2|z|²) is missing"],[s.count+1-Math.log(.5*Math.log(n)),"Math.log is the natural logarithm — the normalized iteration count takes log2 twice"],[s.count+1-Math.log2(Math.log2(n)),"the halving is missing — log2|z| is 0.5 * Math.log2(zr * zr + zi * zi)"]]}var uu={id:"3-2",track:3,title:"Escape-Time Fractals",blurb:"Mandelbrot and Julia sets with smooth coloring — infinite detail from a ten-line kernel.",tasks:[{slug:"pixel-to-plane",title:"Map Pixels to the Complex Plane",intro:`<p>A fractal isn't drawn — it's <strong>evaluated</strong>. There is a function
        defined on the complex plane, and every pixel asks: what does that function do
        <em>at my point</em>? So before any fractal math, each thread must know which complex
        number it owns.</p>
        <p>Three numbers describe the camera: <code>xMin</code> and <code>yMin</code> pin the
        bottom-left corner of the view, and <code>step</code> is the width of one pixel in plane
        units. Thread <code>(x, y)</code> then sits at <code>c = (xMin + x·step) +
        (yMin + y·step)·i</code>. Change the three numbers and the same kernel becomes a zoom lens.</p>`,goal:`<strong>Goal:</strong> map each thread to its point <code>(cr, ci)</code> on the
        complex plane and return the squared magnitude <code>cr² + ci²</code> — a distance field
        we can sanity-check before iterating anything.`,requirements:["Hoist the thread ids into consts: <code>const x = this.thread.x</code> — as a const it becomes a float you can scale","Map to the plane: <code>cr = xMin + x * step</code>, <code>ci = yMin + y * step</code>","Return <code>cr * cr + ci * ci</code> — the squared distance from the origin"],hints:[{title:"Hint 1 — pixels are integers, planes are not",body:`<p><code>this.thread.x</code> counts 0…63 — and it's an <em>integer</em>. Assign it
            to a const first (<code>const x = this.thread.x;</code>) so the GPU treats it as a float;
            then <code>x * step</code> turns pixel counts into plane distance, and adding
            <code>xMin</code> slides the view into place. Same story for y.</p>`},{title:"Hint 2 — the whole body",body:`<pre><code>const x = this.thread.x;
const y = this.thread.y;
const cr = xMin + x * step;
const ci = yMin + y * step;
return cr * cr + ci * ci;</code></pre>`}],transfer:`Index-to-domain mapping is step one of nearly every GPU program: fragment
        shaders scale normalized uv coordinates into world space, CUDA turns
        <code>blockIdx * blockDim + threadIdx</code> into a grid coordinate, WebGPU does the same
        with <code>global_invocation_id</code>. Integer id in, domain point out.`,starterCode:`// Which complex number does THIS pixel own?
const gpu = new GPU({ mode });

const distanceField = gpu.createKernel(function (xMin, yMin, step) {
  // TODO: map this thread onto the complex plane:
  //   const x = this.thread.x;   ← hoisting makes it a float
  //   cr = xMin + x * step   (and the same for ci with y)
  // then return the squared magnitude cr² + ci².
  return this.thread.x;
}, { output: [64, 64] });

const field = distanceField(-2, -2, 4 / 64);
console.log('cell [32][32] sits at the origin:', field[32][32]);
console.log('corner cell [0][0]:', field[0][0]);
`,solutionCode:`// Which complex number does THIS pixel own?
const gpu = new GPU({ mode });

const distanceField = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  return cr * cr + ci * ci;
}, { output: [64, 64] });

const field = distanceField(-2, -2, 4 / 64);
console.log('cell [32][32] sits at the origin:', field[32][32]);
console.log('corner cell [0][0]:', field[0][0]);
`,publicTests:[{name:"the view <code>(-2, -2, 4/64)</code> puts the origin at cell [32][32]",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(-2,-2,4/64);e.assert(t&&t.length===64,`expected 64 rows, got ${t&&t.length}`),e.assert(t[0]&&t[0].length===64,"each row should hold 64 values");const s=4/64,n=(i,a,f,m)=>ct(t[i][a],f,m,or(-2,-2,s,a,i));e.assertClose(t[32][32],0,.001,n(32,32,0,.001)||"cell [32][32] should be the origin, |c|² = 0"),e.assertClose(t[32][48],1,.001,n(32,48,1,.001)||"cell [32][48] sits at c = 1 + 0i, so |c|² = 1"),e.assertClose(t[0][0],8,.01,n(0,0,8,.01)||"cell [0][0] sits at c = -2 - 2i, so |c|² = 8")}},{name:"a different camera — <code>(0, 0, 0.5)</code> — moves every cell",run:async e=>{const t=e.kernel(0,0,.5),s=[[0,0],[2,3],[7,7],[63,1]];for(const[n,i]of s){const a=.25*(i*i+n*n),f=ct(t[n][i],a,.01,or(0,0,.5,i,n));e.assertClose(t[n][i],a,.01,f||`cell [${n}][${i}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(-1,2,.25);for(let s=0;s<64;s+=3)for(let n=0;n<64;n+=3){const i=-1+n*.25,a=2+s*.25,f=ct(t[s][n],i*i+a*a,.01,or(-1,2,.25,n,s));e.assertClose(t[s][n],i*i+a*a,.01,f||`cell [${s}][${n}]`)}}}]},{slug:"escape-time",title:"The Escape-Time Loop",intro:`<p>The Mandelbrot set asks one question at every point <code>c</code>: start
        <code>z = 0</code> and repeat <code>z → z² + c</code> — does <code>z</code> stay near the
        origin forever, or fly off to infinity? Points that stay bounded are <em>in</em> the set;
        for the rest, the interesting number is <strong>how many iterations</strong> they survived.</p>
        <p>Two facts make this computable. Once <code>|z| &gt; 2</code>, escape is guaranteed — so
        we can stop watching. And we cap the loop at 100 passes: anything still bounded by then we
        declare "inside". With <code>z = zr + zi·i</code>, one step is
        <code>zr² − zi² + cr</code> for the new real part and <code>2·zr·zi + ci</code> for the new
        imaginary part.</p>`,goal:`<strong>Goal:</strong> iterate <code>z → z² + c</code> up to 100 times, but only
        while <code>zr² + zi² &lt; 4</code>, and return how many iterations actually ran.`,requirements:["Start at <code>zr = 0, zi = 0, count = 0</code> (already wired up)","Loop a fixed 100 times, guarding each pass with <code>zr² + zi² &lt; 4</code>","Inside the guard: update z via a temporary — <code>zr</code> is read by both formulas",'Return <code>count</code>: 100 means "never escaped", small means "escaped fast"'],hints:[{title:"Hint 1 — the shape of the loop",body:`<p>gpu.js's WebGL backend needs a fixed loop bound, so instead of breaking out
            we guard the body:</p>
<pre><code>for (let i = 0; i &lt; 100; i++) {
  if (zr * zr + zi * zi &lt; 4) {
    // …step and count…
  }
}</code></pre>
            <p>After escape the guard fails on every remaining pass, so z freezes and count stops.</p>`},{title:"Hint 2 — don't clobber zr",body:`<p>Both formulas read the <em>old</em> <code>zr</code>, so stash the new real part
            first:</p>
<pre><code>const zrNext = zr * zr - zi * zi + cr;
zi = 2 * zr * zi + ci;
zr = zrNext;
count = count + 1;</code></pre>`}],transfer:`Data-dependent loops like this are where <em>divergence</em> lives: in CUDA and
        ROCm, threads of a warp that escape early still march in lockstep with their slowest
        neighbor, so a tile renders at the speed of its deepest pixel. WGSL and Metal shading
        language allow exactly this kind of bounded loop in fragment and compute stages.`,starterCode:`// z → z² + c, over and over. Count how long z stays near the origin.
const gpu = new GPU({ mode });

const mandelbrot = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  // TODO: loop 100 times; on each pass, ONLY while zr² + zi² < 4:
  //   new real part:      zr² - zi² + cr   (stash it in a temporary!)
  //   new imaginary part: 2 * zr * zi + ci
  //   and add 1 to count.
  return count;
}, { output: [64, 64] });

const counts = mandelbrot(-2.2, -1.6, 3.2 / 64);
console.log('c = 0, deep inside the set:', counts[32][44]);
console.log('far corner, escapes at once:', counts[0][0]);
`,solutionCode:`// z → z² + c, over and over. Count how long z stays near the origin.
const gpu = new GPU({ mode });

const mandelbrot = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    // guarded instead of break: after escape, z and count just freeze
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count = count + 1;
    }
  }
  return count;
}, { output: [64, 64] });

const counts = mandelbrot(-2.2, -1.6, 3.2 / 64);
console.log('c = 0, deep inside the set:', counts[32][44]);
console.log('far corner, escapes at once:', counts[0][0]);
`,publicTests:[{name:"interior points never escape — <code>c = 0</code> and <code>c = -1</code> hit the 100 cap",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(-2.2,-1.6,.05);e.assert(t&&t.length===64&&t[0].length===64,"expected a 64×64 grid");const s=(n,i,a)=>ct(t[n][i],a,.001,ln());e.assertClose(t[32][44],100,.001,s(32,44,100)||"cell [32][44] is c = 0 — it never escapes"),e.assertClose(t[32][24],100,.001,s(32,24,100)||"cell [32][24] is c = -1 — a stable 2-cycle"),e.assertClose(t[0][0],1,.001,s(0,0,1)||"cell [0][0] is c = -2.2 - 1.6i, |c| > 2 — gone in one step")}},{name:"everything with <code>|c| &gt; 2</code> escapes on the very first pass",run:async e=>{const t=e.kernel(2.5,.5,.01);for(let s=0;s<64;s++)for(let n=0;n<64;n++){const i=ct(t[s][n],1,.001,ln());e.assertClose(t[s][n],1,.001,i||`cell [${s}][${n}] has |c| > 2 — count must be 1`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(-.2,-.1,.003);for(let n=0;n<64;n+=5)for(let i=0;i<64;i+=5){const a=ct(t[n][i],100,.001,ln());e.assertClose(t[n][i],100,.001,a||`cardioid cell [${n}][${i}]`)}const s=e.kernel(-.32,2.5,.01);for(let n=0;n<64;n+=5)for(let i=0;i<64;i+=5){const a=ct(s[n][i],1,.001,ln());e.assertClose(s[n][i],1,.001,a||`exterior cell [${n}][${i}]`)}}}]},{slug:"paint-by-count",title:"Paint by Iteration Count",intro:`<p>Those counts <em>are</em> the picture. Make the kernel graphical and let every
        thread color its own pixel: points that hit the 100 cap are inside the set — paint them
        <strong>black</strong> — and everything else gets a shade from its count. That's the whole
        recipe behind every Mandelbrot poster ever printed.</p>
        <p>This module's palette maps <code>t = count / 100</code> to
        <code>this.color(t, t·t, 0.5 + 0.5·t, 1)</code> — fast escapes glow deep blue, slow ones
        burn toward white near the boundary, where all the detail hides.</p>`,goal:`<strong>Goal:</strong> same escape-time loop, but <code>graphical: true</code> —
        interior pixels black, escaped pixels shaded <code>this.color(t, t*t, 0.5 + 0.5*t, 1)</code>
        with <code>t = count / 100</code>.`,requirements:["Keep the guarded 100-pass loop from the last task (already in place)","If <code>count</code> reached 100, paint black: <code>this.color(0, 0, 0, 1)</code>","Otherwise compute <code>t = count / 100</code> and paint <code>this.color(t, t * t, 0.5 + 0.5 * t, 1)</code>"],hints:[{title:"Hint 1 — two kinds of pixel",body:`<p>Branch on the cap: <code>if (count &lt; 100) { …shade… } else { …black… }</code>.
            Both branches must call <code>this.color()</code> — a graphical thread always paints
            exactly one pixel.</p>`},{title:"Hint 2 — the shade branch",body:`<pre><code>const t = count / 100;
this.color(t, t * t, 0.5 + 0.5 * t, 1);</code></pre>`}],transfer:`Mapping a scalar to a color is a <em>transfer function</em> — in scientific
        visualization and medical imaging it's usually a 1D texture the fragment shader samples
        by value; here the colormap is three inline formulas. Same trick, WebGPU to Metal.`,starterCode:`// The counts become the picture: one thread paints one pixel.
const gpu = new GPU({ mode });

const paint = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count = count + 1;
    }
  }
  // TODO: paint this pixel.
  //   count reached 100  → inside the set → black
  //   escaped            → t = count / 100 → this.color(t, t*t, 0.5 + 0.5*t, 1)
  this.color(1, 0, 1, 1);
}, { output: [128, 128], graphical: true });

paint(-2.2, -1.6, 3.2 / 128);
render(paint.canvas);
`,solutionCode:`// The counts become the picture: one thread paints one pixel.
const gpu = new GPU({ mode });

const paint = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count = count + 1;
    }
  }
  if (count < 100) {
    const t = count / 100;
    this.color(t, t * t, 0.5 + 0.5 * t, 1);
  } else {
    this.color(0, 0, 0, 1);
  }
}, { output: [128, 128], graphical: true });

paint(-2.2, -1.6, 3.2 / 128);
render(paint.canvas);
`,publicTests:[{name:"the classic view shows both worlds — black interior AND shaded exterior",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===128&&e.canvas.height===128,`expected a 128×128 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();let s=0,n=0;for(let i=0;i<t.length;i+=4)t[i]+t[i+1]+t[i+2]<=3?s++:t[i+2]>100&&n++;e.assert(s>300,`expected a black interior — found only ${s} black pixels`),e.assert(n>300,`expected a shaded exterior — found only ${n} blue-ish pixels`)}},{name:"a window inside the set is pure black",run:async e=>{e.kernel(-.2,-.05,.001);const t=e.getPixels();for(let s=0;s<t.length;s+=401*4)e.assert(t[s]<=2&&t[s+1]<=2&&t[s+2]<=2,Sa(t[s],t[s+1],t[s+2])||`interior pixel at byte ${s} should be black, got rgb(${t[s]}, ${t[s+1]}, ${t[s+2]})`)}},{name:"far outside, every pixel wears the count-1 shade",run:async e=>{e.kernel(2.5,2.5,.001);const t=e.getPixels(),[s,n,i]=ar(1/os);for(let a=0;a<t.length;a+=401*4)e.assertClose(t[a],s,2,`red at byte ${a}`),e.assertClose(t[a+1],n,2,`green at byte ${a}`),e.assertClose(t[a+2],i,2,`blue at byte ${a}`)}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(-1.05,-.05,8e-4);let t=e.getPixels();for(let a=0;a<t.length;a+=293*4)e.assert(t[a]<=2&&t[a+1]<=2&&t[a+2]<=2,Sa(t[a],t[a+1],t[a+2])||`bulb pixel at byte ${a} should be black, got rgb(${t[a]}, ${t[a+1]}, ${t[a+2]})`);e.kernel(-9,0,.001),t=e.getPixels();const[s,n,i]=ar(1/os);for(let a=0;a<t.length;a+=293*4)e.assertClose(t[a],s,2,`red at byte ${a}`),e.assertClose(t[a+1],n,2,`green at byte ${a}`),e.assertClose(t[a+2],i,2,`blue at byte ${a}`)}}]},{slug:"smooth-coloring",title:"Smooth Out the Bands",intro:`<p>Look closely at task 3's exterior and you'll see hard rings: iteration counts are
        integers, so neighboring pixels jump from shade 6 straight to shade 7. But the kernel knows
        more than the count — it knows <strong>how far past the escape radius</strong> z flew on its
        final step. A barely-escaped z and one that rocketed to |z| = 50 both count the same pass;
        that overshoot is the missing fraction.</p>
        <p>The classic fix is the <em>normalized iteration count</em>:
        <code>count + 1 − log2(log2|z|)</code>. When z barely clears the radius the correction is
        near 1, when it overshoots hugely it's near 0 — and the bands blend into a continuous ramp.
        (With <code>|z|² = zr² + zi²</code> in hand, use <code>log2|z| = 0.5 · log2(zr² + zi²)</code>
        and skip the square root.)</p>`,goal:`<strong>Goal:</strong> return a <em>fractional</em> escape value — interior points
        return exactly 100, escaped points return
        <code>count + 1 − Math.log2(0.5 * Math.log2(zr² + zi²))</code>.`,requirements:["Keep the guarded loop; after it, branch on <code>count &lt; 100</code>","Escaped: return the fractional escape value — the normalized iteration count from the intro","Interior: return <code>100</code> exactly — no correction for points that never escaped"],hints:[{title:"Hint 1 — why z is still usable after the loop",body:`<p>The guard freezes z the moment it escapes, so after the loop <code>zr, zi</code>
            hold the <em>first</em> value with <code>|z|² ≥ 4</code> — exactly the overshoot the
            formula needs. Math.log2 works inside kernels on both backends.</p>`},{title:"Hint 2 — the ending",body:`<pre><code>if (count &lt; 100) {
  return count + 1
    - Math.log2(0.5 * Math.log2(zr * zr + zi * zi));
}
return 100;</code></pre>`}],transfer:`Fighting quantization with a fractional correction is a graphics evergreen:
        trilinear blending between mipmap levels, <code>smoothstep</code> edges, ordered dithering.
        The same normalized-iteration formula runs unchanged in a CUDA kernel or a WGSL fragment
        shader — it's pure float math.`,starterCode:`// Counts are integers — that's why the shading shows rings.
// Return a FRACTIONAL escape value instead.
const gpu = new GPU({ mode });

const smoothField = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count = count + 1;
    }
  }
  // TODO: escaped pixels (count < 100) should return
  //   count + 1 - Math.log2(0.5 * Math.log2(zr² + zi²))
  // interior pixels return 100 exactly.
  return count;
}, { output: [64, 64] });

const field = smoothField(3, 1, 0.01);
console.log('a fractional escape value:', field[0][0]);
`,solutionCode:`// Counts are integers — that's why the shading shows rings.
// Return a FRACTIONAL escape value instead.
const gpu = new GPU({ mode });

const smoothField = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count = count + 1;
    }
  }
  if (count < 100) {
    // z froze at its first escaped value — its overshoot is the fraction
    return count + 1 - Math.log2(0.5 * Math.log2(zr * zr + zi * zi));
  }
  return 100;
}, { output: [64, 64] });

const field = smoothField(3, 1, 0.01);
console.log('a fractional escape value:', field[0][0]);
`,publicTests:[{name:"interior cells still return exactly 100",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(-.2,-.05,.002);e.assert(t&&t.length===64&&t[0].length===64,"expected a 64×64 grid");for(let s=0;s<64;s+=7)for(let n=0;n<64;n+=7)e.assertClose(t[s][n],100,.001,`interior cell [${s}][${n}]`)}},{name:"escaped cells carry a fraction that matches the formula",run:async e=>{const t=e.kernel(3,1,.01);let s=!1;const n=[[0,0],[10,20],[33,7],[63,63]];for(const[i,a]of n){const f=on(0,0,3+a*.01,1+i*.01),m=ct(t[i][a],f,.02,lr(3+a*.01,1+i*.01));e.assertClose(t[i][a],f,.02,m||`cell [${i}][${a}]`),Math.abs(t[i][a]-Math.round(t[i][a]))>.05&&(s=!0)}e.assert(s,"every sampled value is a whole number — are you still returning the raw count?")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(-4,2,.005);for(let n=0;n<64;n+=9)for(let i=0;i<64;i+=9){const a=on(0,0,-4+i*.005,2+n*.005),f=ct(t[n][i],a,.02,lr(-4+i*.005,2+n*.005));e.assertClose(t[n][i],a,.02,f||`far cell [${n}][${i}]`)}const s=e.kernel(1.5,-.032,.001);for(let n=0;n<64;n+=9)for(let i=0;i<64;i+=9){const a=on(0,0,1.5+i*.001,-.032+n*.001),f=ct(s[n][i],a,.02,lr(1.5+i*.001,-.032+n*.001));e.assertClose(s[n][i],a,.02,f||`near cell [${n}][${i}]`)}}}]},{slug:"julia-dial",title:"Julia Sets: Turn the Dial",intro:`<p>Here's the payoff. Take the exact loop you've built and <strong>flip the
        roles</strong>: in a Julia set, z starts <em>at the pixel</em> and <code>c</code> is one
        fixed complex number shared by every thread. Each choice of c is a different fractal —
        <code>c = 0</code> gives a plain disk, <code>−0.7269 + 0.1889i</code> a galaxy of spirals —
        and the Mandelbrot set turns out to be the map of which c values give connected Julias.</p>
        <p>Because c arrives as <strong>kernel arguments</strong>, changing it costs one function
        call — no recompiling. That's what makes those mesmerizing morphing-Julia animations:
        nudge c, redraw, repeat.</p>`,goal:`<strong>Goal:</strong> a graphical Julia kernel over the fixed view −1.6…1.6: seed
        <code>z</code> from the pixel, add the arguments <code>cRe, cIm</code> each step, and keep
        task 4's smooth shading (interior black).`,requirements:["Seed z from the pixel: <code>zr = xMin + x·step</code>, <code>zi = yMin + y·step</code> (constants are wired up)","Inside the loop, add <code>cRe</code> and <code>cIm</code> — not the pixel coordinates","Escaped: shade with <code>t = smooth / 100</code> via <code>this.color(t, t * t, 0.5 + 0.5 * t, 1)</code>; interior: black","Call the kernel with a c of your choice and <code>render()</code> it"],hints:[{title:"Hint 1 — what actually changes",body:`<p>Two lines. Mandelbrot: z starts at 0 and c is the pixel. Julia: z starts at the
            pixel and c is the argument pair. The loop body, the guard, the shading — all identical.</p>`},{title:"Hint 2 — the exact edits",body:`<p>Seed with</p>
<pre><code>let zr = this.constants.xMin + x * this.constants.step;</code></pre>
<p>(and likewise <code>zi</code> from y), then inside the loop use
            <code>… + cRe</code> and <code>… + cIm</code> instead of <code>px</code> / <code>py</code>.</p>`}],transfer:`A per-launch value broadcast to every thread is what other APIs call a
        <em>uniform</em>: a WGSL uniform buffer, a Metal constant buffer, a plain CUDA kernel
        parameter. Animating one uniform per frame — exactly your c — is how every shader-toy
        Julia morph is driven.`,starterCode:`// Same loop, roles flipped: the pixel is z₀, and c is a knob you turn.
const gpu = new GPU({ mode });

const julia = gpu.createKernel(function (cRe, cIm) {
  const x = this.thread.x;
  const y = this.thread.y;
  const px = this.constants.xMin + x * this.constants.step;
  const py = this.constants.yMin + y * this.constants.step;
  // TODO: this is still the Mandelbrot arrangement — z from 0, pixel as c.
  // Flip it: seed z from (px, py), and add cRe / cIm inside the loop.
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + px;
      zi = 2 * zr * zi + py;
      zr = zrNext;
      count = count + 1;
    }
  }
  if (count < 100) {
    const smooth = count + 1 - Math.log2(0.5 * Math.log2(zr * zr + zi * zi));
    const t = smooth / 100;
    this.color(t, t * t, 0.5 + 0.5 * t, 1);
  } else {
    this.color(0, 0, 0, 1);
  }
}, {
  output: [128, 128],
  graphical: true,
  constants: { xMin: -1.6, yMin: -1.6, step: 0.025 },
});

julia(-0.7269, 0.1889); // try 0.285 + 0.01i, or -0.8 + 0.156i
render(julia.canvas);
`,solutionCode:`// Same loop, roles flipped: the pixel is z₀, and c is a knob you turn.
const gpu = new GPU({ mode });

const julia = gpu.createKernel(function (cRe, cIm) {
  // z starts AT the pixel; c is shared by every thread
  const x = this.thread.x;
  const y = this.thread.y;
  let zr = this.constants.xMin + x * this.constants.step;
  let zi = this.constants.yMin + y * this.constants.step;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cRe;
      zi = 2 * zr * zi + cIm;
      zr = zrNext;
      count = count + 1;
    }
  }
  if (count < 100) {
    const smooth = count + 1 - Math.log2(0.5 * Math.log2(zr * zr + zi * zi));
    const t = smooth / 100;
    this.color(t, t * t, 0.5 + 0.5 * t, 1);
  } else {
    this.color(0, 0, 0, 1);
  }
}, {
  output: [128, 128],
  graphical: true,
  constants: { xMin: -1.6, yMin: -1.6, step: 0.025 },
});

julia(-0.7269, 0.1889); // try 0.285 + 0.01i, or -0.8 + 0.156i
render(julia.canvas);
`,publicTests:[{name:"with <code>c = 0</code> the Julia set is the unit disk — inside black, outside shaded",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===128&&e.canvas.height===128,`expected a 128×128 canvas, got ${e.canvas.width}×${e.canvas.height}`),e.kernel(0,0);const t=e.getPixels();for(let s=4;s<128;s+=8)for(let n=4;n<128;n+=8){const i=-1.6+n*.025,a=-1.6+s*.025,f=i*i+a*a,m=(s*128+n)*4;f<.9?e.assert(t[m]+t[m+1]+t[m+2]<=3,`pixel (${n}, ${s}) is inside the unit disk — expected black, got rgb(${t[m]}, ${t[m+1]}, ${t[m+2]})`):f>4.25&&e.assert(t[m+2]>100,`pixel (${n}, ${s}) is far outside the disk — expected a blue-ish shade, got rgb(${t[m]}, ${t[m+1]}, ${t[m+2]})`)}}},{name:"c is a live argument — turn the dial and the center pixel flips",run:async e=>{e.kernel(0,0);let t=e.getPixels();const s=8256*4;e.assert(t[s]+t[s+1]+t[s+2]<=3,"with c = 0 the center pixel (z₀ = 0) never escapes — it should be black"),e.kernel(-2.5,0),t=e.getPixels(),e.assert(t[s+2]>100,"with c = -2.5 the center pixel escapes in one step — it should be shaded, not black. Is c actually used in the loop?")}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(0,0);const t=e.getPixels();let s=10836*4;e.assert(t[s]+t[s+1]+t[s+2]<=3,`pixel (84, 84) lies inside the unit disk — expected black, got rgb(${t[s]}, ${t[s+1]}, ${t[s+2]})`),s=8196*4;const n=on(-1.5,0,0,0)/os,[i,a,f]=ar(n);e.assertClose(t[s],i,4,"red at pixel (4, 64)"),e.assertClose(t[s+1],a,4,"green at pixel (4, 64)"),e.assertClose(t[s+2],f,4,"blue at pixel (4, 64)")}}]}]},cu=Object.freeze({__proto__:null,default:uu});const Mt=16;function As(e=Mt){const t=new Array(e);for(let s=0;s<e;s++)t[s]=new Array(e).fill(0);return t}function Ie(e,t=Mt){const s=As(t);for(const[n,i]of e)s[n][i]=1;return s}function Ct(e,t,s=Mt,n=.35){const i=e.seededRandom(t),a=As(s);for(let f=0;f<s;f++)for(let m=0;m<s;m++)a[f][m]=i()<n?1:0;return a}function Ir(e,t,s){const n=e.length;let i=0;for(let a=-1;a<=1;a++)for(let f=-1;f<=1;f++)a===0&&f===0||(i+=e[(t+a+n)%n][(s+f+n)%n]);return i}function wt(e,t=It,s=Ke){const n=e.length,i=As(n);for(let a=0;a<n;a++)for(let f=0;f<n;f++){const m=Ir(e,a,f);i[a][f]=e[a][f]===1?s[m]:t[m]}return i}function _a(e,t,s=It,n=Ke){let i=e;for(let a=0;a<t;a++)i=wt(i,s,n);return i}function Ca(e){let t=0;for(let s=0;s<e.length;s++)for(let n=0;n<e[s].length;n++)t+=e[s][n];return t}function nt(e,t,s,n,i){e.assert(t&&t.length===s.length,`${n} — expected ${s.length} rows`);const a=i||n;for(let f=0;f<s.length;f++)for(let m=0;m<s.length;m++)e.assertClose(t[f][m],s[f][m],.001,`${a} — cell [${f}][${m}]`)}function hu(e,t){for(let s=0;s<t.length;s++){if(!e[s])return!1;for(let n=0;n<t.length;n++)if(!(Math.abs(e[s][n]-t[s][n])<=.001))return!1}return!0}function ht(e,t,s=It,n=Ke){const i=[[wt(t,n,n),"a dead cell with 2 neighbors came alive — birth is on exactly 3; 2 is what lets an already-live cell survive"],[wt(t,s,s),"live cells with 2 neighbors died — survival covers 2 or 3, and only birth is limited to exactly 3"],[wt(t,n,s),"the two rules are swapped — a dead cell follows the birth rule, a live cell the survival rule"],[t,"the world came back unchanged — the rule never reached the return value"]];for(const[a,f]of i)if(hu(e,a))return f;return null}function ur(e,t,s){return[[Ir(e,t,s)+e[t][s],"your own cell is still inside the 3×3 sum — subtract grid[this.thread.y][this.thread.x] at the end"]]}function cr(e){return Math.abs(e)<=.001?"the edge did not wrap — add the width before the modulo, (this.thread.y + dy + 16) % 16, because a bare % can go negative":null}function du(e,t,s){const n=`gen ${t}: ${s} alive`;return e.logs.some(i=>i.type==="log"&&i.text&&i.text.includes(n))?"that generation logged the population from BEFORE its step — count the live cells after current = step(current)":null}function hr(e,t,s){return e===s-t?"the two colors are swapped — live cells take the green, dead cells the dark background":null}function dr(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}const It=[0,0,0,1,0,0,0,0,0],Ke=[0,0,1,1,0,0,0,0,0],ss=[0,0,0,1,0,0,1,0,0],pr=[0,0,0,1,0,0,1,1,1],fr=[0,0,0,1,1,0,1,1,1],Ss=[[7,6],[7,7],[7,8]],mr=[[3,3],[3,4],[4,3],[4,4]],un=[[1,2],[2,3],[3,1],[3,2],[3,3]],Ea=[[6,8],[6,9],[7,7],[7,8],[8,8]];function pu(e,t,s,n=Mt){return e.map(([i,a])=>[(i+t+n)%n,(a+s+n)%n])}var fu={id:"3-3",track:3,title:"Cellular Automata",blurb:"Conway's Life and friends: feed a kernel's output back in and watch worlds evolve.",tasks:[{slug:"neighbor-census",title:"The Neighbor Census",intro:`<p>A cellular automaton is a world of cells, each one dead (<code>0</code>) or alive
        (<code>1</code>), where every cell's next state depends only on its immediate neighborhood.
        That makes it embarrassingly parallel: 256 cells, 256 threads, and no thread needs to know
        what any other thread is doing — only what the grid looked like.</p>
        <p>Every rule in this module starts with the same question: <strong>how many of my eight
        neighbors are alive?</strong> This world is a torus — walk off the right edge, reappear on
        the left — and wrapping costs one modulo: <code>(x + dx + 16) % 16</code>. The
        <code>+ 16</code> is not decoration: JavaScript's <code>%</code> can go negative while the
        GPU's cannot, and adding the width first keeps both operands positive so CPU mode and GPU
        mode tell the same story.</p>`,goal:`<strong>Goal:</strong> make the kernel return, for every cell, the number of live
        cells among its eight neighbors — with the edges wrapped around.`,requirements:["Visit the 3×3 block around this cell with nested <code>dy</code>/<code>dx</code> loops from −1 to 1","Wrap every coordinate: <code>(this.thread.x + dx + 16) % 16</code> (and the same for y)","Don't count yourself — a cell is not its own neighbor"],hints:[{title:"Hint 1 — the loop bounds",body:`<p>Two statically bounded loops: <code>for (let dy = -1; dy &lt; 2; dy++)</code>
            around <code>for (let dx = -1; dx &lt; 2; dx++)</code>. Nine visits per cell.</p>`},{title:"Hint 2 — the subtract-self trick",body:`<p>Skipping the middle of the 3×3 block needs no <code>if</code>: sum all nine
            cells, then subtract <code>grid[this.thread.y][this.thread.x]</code> at the end. If
            you're dead you subtract 0; if you're alive you take yourself back out.</p>`},{title:"Hint 3 — the whole loop body",body:`<pre><code>const yy = (this.thread.y + dy + 16) % 16;
const xx = (this.thread.x + dx + 16) % 16;
count += grid[yy][xx];</code></pre>
<p>— then</p>
<pre><code>return count - grid[this.thread.y][this.thread.x];</code></pre>`}],transfer:`Reading a fixed window around your own coordinate is the <em>stencil</em> pattern,
        and it dominates real GPU workloads: CUDA stencil kernels tile the grid into shared memory
        with a one-cell "halo" so neighbors are read once, and WebGPU compute shaders do the same
        with workgroup memory.`,starterCode:`// Every cell asks the same question, all at once:
// how many of my eight neighbors are alive?
// The world wraps — leave one edge, come back on the other.
const gpu = new GPU({ mode });

const census = gpu.createKernel(function (grid) {
  let count = 0;
  // TODO: sum the 3x3 block around this cell (dy and dx from -1 to 1),
  // wrapping each coordinate with (coord + d + 16) % 16.
  // Careful: a cell is not its own neighbor.
  return count;
}, { output: [16, 16] });

const counts = census(grid);
console.log('cell (8, 8) sees', counts[8][8], 'live neighbors');
`,solutionCode:`// Every cell asks the same question, all at once:
// how many of my eight neighbors are alive?
const gpu = new GPU({ mode });

const census = gpu.createKernel(function (grid) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += grid[yy][xx];
    }
  }
  // The 3x3 sum counted this cell too — take it back out.
  return count - grid[this.thread.y][this.thread.x];
}, { output: [16, 16] });

const counts = census(grid);
console.log('cell (8, 8) sees', counts[8][8], 'live neighbors');
`,inputs:e=>({grid:Ct(e,1101)}),publicTests:[{name:"a lone cell has zero neighbors — each of its eight neighbors sees one",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ie([[5,5]]),s=e.kernel(t);e.assertClose(s[5][5],0,.001,dr(s[5][5],0,.001,ur(t,5,5))||"the live cell itself (it is not its own neighbor)");const n=[[4,4],[4,5],[4,6],[5,4],[5,6],[6,4],[6,5],[6,6]];for(const[i,a]of n){const f=dr(s[i][a],1,.001,ur(t,i,a));e.assertClose(s[i][a],1,.001,f||`neighbor cell [${i}][${a}]`)}e.assertClose(s[10][10],0,.001,"a far-away cell")}},{name:"the world wraps: a corner cell is seen across all four edges",run:async e=>{const t=e.kernel(Ie([[0,0]]));e.assertClose(t[15][15],1,.001,cr(t[15][15])||"diagonal wrap — cell [15][15]"),e.assertClose(t[0][15],1,.001,cr(t[0][15])||"horizontal wrap — cell [0][15]"),e.assertClose(t[15][0],1,.001,cr(t[15][0])||"vertical wrap — cell [15][0]"),e.assertClose(t[1][1],1,.001,"ordinary diagonal — cell [1][1]"),e.assertClose(t[0][0],0,.001,"the corner cell itself")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ct(e.utils,2202),s=e.kernel(t);for(let n=0;n<Mt;n++)for(let i=0;i<Mt;i++){const a=Ir(t,n,i),f=dr(s[n][i],a,.001,ur(t,n,i));e.assertClose(s[n][i],a,.001,f||`cell [${n}][${i}]`)}}}]},{slug:"one-tick",title:"One Tick of Life",intro:`<p>In 1970 John Conway picked the simplest rules he could find that make a world worth
        watching. <strong>Birth:</strong> a dead cell with exactly 3 live neighbors comes alive.
        <strong>Survival:</strong> a live cell with 2 or 3 neighbors stays alive. Everything else —
        lonely or overcrowded — dies. That's the whole game (the notation is B3/S23).</p>
        <p>There's a classic bug in CPU implementations: update the grid <em>in place</em> and
        cells start reading half-new, half-old neighbors. A kernel is immune by construction —
        every thread reads the old <code>world</code> argument and writes into a brand-new output.
        The double buffer isn't a technique here; it's what a kernel <em>is</em>.</p>`,goal:`<strong>Goal:</strong> finish the kernel so it computes one full generation of
        Conway's Life — birth on 3, survival on 2 or 3, death otherwise.`,requirements:["Keep the wrapped neighbor census from the last task (already in place)","A dead cell returns <code>1</code> exactly when <code>count === 3</code>","A live cell returns <code>1</code> exactly when <code>count === 2 || count === 3</code>","Everything else returns <code>0</code>"],hints:[{title:"Hint 1 — start dead",body:`<p>Declare <code>let next = 0;</code>, flip it to <code>1</code> in the cases
            that live, and <code>return next;</code> once at the end. Two <code>if</code>s cover
            the whole rulebook.</p>`},{title:"Hint 2 — the two ifs",body:`<pre><code>if (self === 1 &amp;&amp; (count === 2 || count === 3)) next = 1;
if (self === 0 &amp;&amp; count === 3) next = 1;</code></pre>`}],transfer:`Reading one buffer while writing another is <em>ping-ponging</em>, and every
        platform institutionalizes it: WebGPU simulations bind two storage buffers and swap their
        roles each dispatch, and CUDA solvers keep <code>d_old</code>/<code>d_new</code> device
        pointers and trade them every launch.`,starterCode:`// B3/S23: birth on 3 neighbors, survival on 2 or 3, death otherwise.
// The census below is task 1's answer — the rulebook is yours.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  // TODO: apply Conway's rules to \`self\` and \`count\`.
  return self;
}, { output: [16, 16] });

const next = step(world);
console.log('before:', world[7].join(''));
console.log('after :', Array.from(next[7]).join(''));
`,solutionCode:`// B3/S23: birth on 3 neighbors, survival on 2 or 3, death otherwise.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  let next = 0;
  if (self === 1 && (count === 2 || count === 3)) next = 1;
  if (self === 0 && count === 3) next = 1;
  return next;
}, { output: [16, 16] });

const next = step(world);
console.log('before:', world[7].join(''));
console.log('after :', Array.from(next[7]).join(''));
`,inputs:()=>({world:Ie(Ss)}),publicTests:[{name:"the blinker: three-in-a-row flips to three-in-a-column",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ie(Ss),s=e.kernel(t);nt(e,s,Ie([[6,7],[7,7],[8,7]]),"blinker after one tick",ht(s,t))}},{name:"the block: a 2×2 square is a still life — nothing moves",run:async e=>{const t=Ie(mr),s=e.kernel(t);nt(e,s,Ie(mr),"block after one tick",ht(s,t))}},{name:"an empty world stays empty — no spontaneous generation",run:async e=>{const t=e.kernel(As());nt(e,t,As(),"empty world after one tick")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ct(e.utils,4404),s=e.kernel(t);nt(e,s,wt(t),"random world, one tick",ht(s,t))}},{name:"private test #2",run:async e=>{const t=Ct(e.utils,5505,Mt,.6),s=e.kernel(t);nt(e,s,wt(t),"crowded world, one tick",ht(s,t))}}]},{slug:"generations",title:"Generations: Feed It Back",intro:`<p>One tick is a snapshot; a world is a movie. A kernel has no memory of the previous
        frame — <strong>time lives in JavaScript</strong>. The result of a 2D kernel is an array of
        rows, which is exactly the shape the kernel accepts as input, so
        <code>current = step(current)</code> is the whole time machine: output becomes input,
        forever.</p>
        <p>Your test subject is the <strong>R-pentomino</strong> — five innocent-looking cells that
        erupt into chaos (on an infinite grid they don't settle down for 1,103 generations; Conway's
        group tracked it by hand). You'll run six generations and log the population after each, so
        you can watch the explosion begin.</p>`,goal:`<strong>Goal:</strong> restore the B3/S23 rule inside the kernel, then run 6
        generations by feeding each output back in — logging
        <code>'gen ' + g + ': ' + alive + ' alive'</code> after every step.`,requirements:["Complete the kernel: the same B3/S23 rule you wrote last task","Loop 6 times in plain JavaScript, reassigning: <code>current = step(current)</code>","After each step, total the live cells in plain JS (kernel output rows are ordinary arrays)","Log each generation exactly as <code>'gen ' + g + ': ' + alive + ' alive'</code>, g from 1 to 6"],hints:[{title:"Hint 1 — the feed-back loop",body:`<p><code>let current = world;</code> then</p>
<pre><code>for (let g = 1; g &lt;= 6; g++) {
  current = step(current);
  // …
}</code></pre>
<p>No copying, no bookkeeping — the kernel's output is already a valid input.</p>`},{title:"Hint 2 — counting the living",body:`<p>Inside the loop, after stepping: <code>let alive = 0;</code> and two nested
            loops adding <code>current[y][x]</code>. Cells are 0 or 1, so the sum <em>is</em> the
            population.</p>`}],transfer:`The frame loop lives on the host everywhere: a CUDA fluid sim launches its kernel
        thousands of times from an ordinary CPU <code>for</code> loop, and a Metal app encodes one
        compute dispatch per frame — the GPU computes each tick, but the CPU decides that time
        passes.`,starterCode:`// The kernel computes one tick. Time itself is a JavaScript loop:
// whatever comes out goes straight back in.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  // TODO: B3/S23 — you wrote this rule last task. Own it.
  return self;
}, { output: [16, 16] });

// world starts as the R-pentomino: five cells, endless trouble.
// TODO: run 6 generations. Each time around: current = step(current),
// count the live cells in plain JS, then log exactly:
//   console.log('gen ' + g + ': ' + alive + ' alive');
let current = world;
`,solutionCode:`// The kernel computes one tick. Time itself is a JavaScript loop:
// whatever comes out goes straight back in.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  let next = 0;
  if (self === 1 && (count === 2 || count === 3)) next = 1;
  if (self === 0 && count === 3) next = 1;
  return next;
}, { output: [16, 16] });

let current = world;
for (let g = 1; g <= 6; g++) {
  current = step(current);
  let alive = 0;
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) alive += current[y][x];
  }
  console.log('gen ' + g + ': ' + alive + ' alive');
}
`,inputs:()=>({world:Ie(Ea)}),publicTests:[{name:"six generations logged, matching the R-pentomino's true population history",run:async e=>{let t=Ie(Ea);for(let s=1;s<=6;s++){const n=Ca(t);t=wt(t);const i="gen "+s+": "+Ca(t)+" alive",a=e.logs.some(f=>f.type==="log"&&f.text&&f.text.includes(i));e.assert(a,du(e,s,n)||`expected a log line containing "${i}"`)}}},{name:"the step kernel is still a faithful B3/S23 tick",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ie(Ss),s=e.kernel(t);nt(e,s,Ie([[6,7],[7,7],[8,7]]),"blinker after one tick",ht(s,t))}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ct(e.utils,6606);let s=t;for(let n=0;n<3;n++)s=e.kernel(s);nt(e,s,_a(t,3),"random world after three ticks")}}]},{slug:"glider-on-screen",title:"Watch the Glider Fly",intro:`<p>The <strong>glider</strong> is five cells that <em>travel</em>. No individual cell
        moves — each one just dies or is born in place, like every other cell — yet after four
        ticks an identical copy of the pattern stands one cell down and one cell right. Motion as
        pure side effect. When it was discovered in 1970 it changed the game: Life could transmit
        information.</p>
        <p>Time to see it. You already know both halves from earlier tracks: a numeric
        <code>step</code> kernel computes generations, and a <strong>graphical</strong> kernel
        turns the final grid into pixels. Simulation pass, then render pass — the fundamental
        division of labor in every real-time visualization.</p>`,goal:`<strong>Goal:</strong> complete the <code>paint</code> kernel so live cells glow
        green and dead cells stay near-black, then watch the glider that started in the top-left
        arrive further down the board.`,requirements:["Read this thread's cell from <code>cells</code> — same indexing as every task so far","Live cells: <code>this.color(0.2, 1, 0.4, 1)</code>; dead cells: <code>this.color(0.05, 0.06, 0.09, 1)</code>","Leave the 8-generation loop and <code>render()</code> call as they are"],hints:[{title:"Hint 1 — numbers in, colors out",body:`<p><code>cells</code> is the plain 2D grid the step kernel produced. Read
            <code>cells[this.thread.y][this.thread.x]</code> into a variable — it's 0 or 1.</p>`},{title:"Hint 2 — the branch",body:`<pre><code>if (alive === 1) {
  this.color(0.2, 1, 0.4, 1);
} else {
  this.color(0.05, 0.06, 0.09, 1);
}</code></pre>`}],transfer:`Sim pass feeding a render pass is the standard split in every API: a WebGPU compute
        shader writes the state a fragment shader then draws, and CUDA–OpenGL interop exists purely
        so simulation buffers can be displayed without a round trip through the CPU.`,starterCode:`// Two kernels, two jobs: step computes the world, paint shows it.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  let next = 0;
  if (self === 1 && (count === 2 || count === 3)) next = 1;
  if (self === 0 && count === 3) next = 1;
  return next;
}, { output: [16, 16] });

const paint = gpu.createKernel(function (cells) {
  // TODO: live cells glow this.color(0.2, 1, 0.4, 1),
  // dead cells stay this.color(0.05, 0.06, 0.09, 1).
  this.color(1, 0, 1, 1);
}, { output: [16, 16], graphical: true });

// world starts as a glider in the top-left. Fly, little guy.
let current = world;
for (let g = 0; g < 8; g++) {
  current = step(current);
}
paint(current);
render(paint.canvas);
`,solutionCode:`// Two kernels, two jobs: step computes the world, paint shows it.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (world) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  let next = 0;
  if (self === 1 && (count === 2 || count === 3)) next = 1;
  if (self === 0 && count === 3) next = 1;
  return next;
}, { output: [16, 16] });

const paint = gpu.createKernel(function (cells) {
  const alive = cells[this.thread.y][this.thread.x];
  if (alive === 1) {
    this.color(0.2, 1, 0.4, 1);
  } else {
    this.color(0.05, 0.06, 0.09, 1);
  }
}, { output: [16, 16], graphical: true });

// world starts as a glider in the top-left. Fly, little guy.
let current = world;
for (let g = 0; g < 8; g++) {
  current = step(current);
}
paint(current);
render(paint.canvas);
`,inputs:()=>({world:Ie(un)}),publicTests:[{name:"the glider translates: four ticks move the whole pattern down-right by one",run:async e=>{const t=e.kernels.find(n=>n.kernel&&!n.kernel.graphical);e.assert(t,"no numeric (non-graphical) step kernel found");let s=Ie(un);for(let n=0;n<4;n++)s=t(s);nt(e,s,Ie(pu(un,1,1)),"glider after four ticks")}},{name:"canvas is 16×16 and shows exactly the 5 glider cells lit green",run:async e=>{e.assert(e.canvas,"no canvas — did you call render(paint.canvas)?"),e.assert(e.canvas.width===16&&e.canvas.height===16,`expected a 16×16 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.kernels.find(i=>i.kernel&&i.kernel.graphical);e.assert(t,"no graphical paint kernel found"),t(_a(Ie(un),8));const s=t.getPixels();let n=0;for(let i=0;i<s.length;i+=4){const a=s[i+1];e.assert(a>200||a<40,`pixel at byte ${i} is neither live-green nor dead-dark (green = ${a})`),a>200&&n++}e.assert(n===5,hr(n,5,256)||`a glider is always 5 cells — found ${n} lit pixels`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(f=>f.kernel&&!f.kernel.graphical),s=e.kernels.find(f=>f.kernel&&f.kernel.graphical);e.assert(t&&s,"expected a numeric and a graphical kernel");const n=f=>{let m=0;for(let A=0;A<f.length;A+=4)f[A+1]>200&&m++;return m};s(t(Ie(Ss)));const i=n(s.getPixels());e.assert(i===3,hr(i,3,256)||"stepped blinker should light 3 pixels"),s(Ie(mr));const a=n(s.getPixels());e.assert(a===4,hr(a,4,256)||"block should light 4 pixels")}}]},{slug:"any-rule",title:"One Kernel, Every Universe",intro:`<p>B3/S23 is one point in a whole family. Any <em>outer-totalistic</em> rule is fully
        described by two 9-entry tables: <code>born[n]</code> — does a dead cell with n live
        neighbors come alive? — and <code>stay[n]</code> — does a live cell with n survive? Conway
        is <code>born[3] = 1</code>, <code>stay[2] = stay[3] = 1</code>, zeros everywhere else.
        <strong>HighLife</strong> adds <code>born[6] = 1</code> and suddenly the world contains a
        pattern that builds copies of itself.</p>
        <p>Here's the move that matters: pass the tables <strong>as kernel arguments</strong>.
        The rulebook stops being code and becomes data — one compiled kernel runs every universe
        in the family, and switching physics is just passing different arrays. No <code>if</code>
        per rule, no recompile: alive cells look up <code>stay[count]</code>, dead cells look up
        <code>born[count]</code>.</p>`,goal:`<strong>Goal:</strong> finish the <code>evolve</code> kernel so it applies whatever
        rule tables it's handed — then let the wired-up code count where Life and HighLife disagree
        about the same world's next tick.`,requirements:["The kernel takes <code>world</code>, <code>born</code> and <code>stay</code> — don't hard-code any rule","Dead cells (<code>self === 0</code>) return <code>born[count]</code>","Live cells (<code>self === 1</code>) return <code>stay[count]</code>"],hints:[{title:"Hint 1 — arrays index like anywhere else",body:`<p><code>count</code> is a number from 0 to 8, and <code>born</code> is a 9-entry
            array — <code>born[count]</code> is already the answer for a dead cell. The lookup
            <em>is</em> the rule.</p>`},{title:"Hint 2 — a single return",body:`<pre><code>let fate = born[count];
if (self === 1) fate = stay[count];
return fate;</code></pre>`}],transfer:`Shipping small lookup tables to a fixed kernel instead of recompiling is how GPUs
        stay fast when behavior changes: CUDA and ROCm keep them in <code>__constant__</code>
        memory, WebGPU and Metal bind them as uniform buffers — same shader, new physics, zero
        pipeline rebuilds.`,starterCode:`// The rulebook as data: born[n] and stay[n] answer every question
// a cell can ask. One kernel, any Life-like universe.
const gpu = new GPU({ mode });

const evolve = gpu.createKernel(function (world, born, stay) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  // TODO: no rule logic — just look the answer up.
  // Dead cells consult born[count]; live cells consult stay[count].
  return self;
}, { output: [16, 16] });

// The same world, two different laws of physics:
const life = evolve(world, lifeBorn, lifeStay);
const high = evolve(world, highlifeBorn, lifeStay);

let differ = 0;
for (let y = 0; y < 16; y++) {
  for (let x = 0; x < 16; x++) {
    if (life[y][x] !== high[y][x]) differ++;
  }
}
console.log('Life and HighLife disagree on ' + differ + ' cells after one tick');
`,solutionCode:`// The rulebook as data: born[n] and stay[n] answer every question
// a cell can ask. One kernel, any Life-like universe.
const gpu = new GPU({ mode });

const evolve = gpu.createKernel(function (world, born, stay) {
  let count = 0;
  for (let dy = -1; dy < 2; dy++) {
    for (let dx = -1; dx < 2; dx++) {
      const yy = (this.thread.y + dy + 16) % 16;
      const xx = (this.thread.x + dx + 16) % 16;
      count += world[yy][xx];
    }
  }
  const self = world[this.thread.y][this.thread.x];
  count -= self;
  let fate = born[count];
  if (self === 1) fate = stay[count];
  return fate;
}, { output: [16, 16] });

// The same world, two different laws of physics:
const life = evolve(world, lifeBorn, lifeStay);
const high = evolve(world, highlifeBorn, lifeStay);

let differ = 0;
for (let y = 0; y < 16; y++) {
  for (let x = 0; x < 16; x++) {
    if (life[y][x] !== high[y][x]) differ++;
  }
}
console.log('Life and HighLife disagree on ' + differ + ' cells after one tick');
`,inputs:e=>({world:Ct(e,7707),lifeBorn:It.slice(),lifeStay:Ke.slice(),highlifeBorn:ss.slice()}),publicTests:[{name:"fed the Life tables, it is still Life: the blinker spins",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ie(Ss),s=e.kernel(t,It,Ke);nt(e,s,Ie([[6,7],[7,7],[8,7]]),"blinker under B3/S23",ht(s,t,It,Ke))}},{name:"HighLife's B6: six neighbors ignite a dead cell that Life leaves dark",run:async e=>{const t=Ie([[4,4],[4,5],[4,6],[5,4],[5,6],[6,4]]),s=e.kernel(t,It,Ke),n=e.kernel(t,ss,Ke);e.assertClose(s[5][5],0,.001,ht(s,t,It,Ke)||"under Life (B3), 6 neighbors do not give birth"),e.assertClose(n[5][5],1,.001,ht(n,t,ss,Ke)||"under HighLife (B36), 6 neighbors do")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ct(e.utils,8808,Mt,.5),s=e.kernel(t,pr,fr);nt(e,s,wt(t,pr,fr),"Day & Night, one tick",ht(s,t,pr,fr))}},{name:"private test #2",run:async e=>{const t=Ct(e.utils,9909),s=e.kernel(t,ss,Ke);nt(e,s,wt(t,ss,Ke),"HighLife, one tick",ht(s,t,ss,Ke))}}]}]},mu=Object.freeze({__proto__:null,default:fu});const fe={du:.2,dv:.1,f:.035,k:.06,dt:1};function Le(e,t){const s=new Array(e);for(let n=0;n<e;n++)s[n]=new Array(e).fill(t);return s}function Ia(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++){const f=new Array(t);for(let m=0;m<t;m++)f[m]=Math.round(n()*1e3)/1e3;i[a]=f}return i}function Vt(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){const i=new Array(t),a=n===0?t-1:n-1,f=n===t-1?0:n+1;for(let m=0;m<t;m++){const A=m===0?t-1:m-1,N=m===t-1?0:m+1;i[m]=e[n][A]+e[n][N]+e[a][m]+e[f][m]-4*e[n][m]}s[n]=i}return s}function Ha(e,t){const s=e.length,n=Vt(e),i=Vt(t),a=new Array(s),f=new Array(s);for(let m=0;m<s;m++){a[m]=new Array(s),f[m]=new Array(s);for(let A=0;A<s;A++){const N=e[m][A],H=t[m][A],Y=N*H*H;a[m][A]=N+(fe.du*n[m][A]-Y+fe.f*(1-N))*fe.dt,f[m][A]=H+(fe.dv*i[m][A]+Y-(fe.f+fe.k)*H)*fe.dt}}return[a,f]}function gr(e,t,s){for(let n=0;n<s;n++)[e,t]=Ha(e,t);return[e,t]}function Lt(e,t){const s=Le(e,1),n=Le(e,0),i=e-t>>1;for(let a=i;a<i+t;a++)for(let f=i;f<i+t;f++)s[a][f]=.5,n[a][f]=.25;return{u:s,v:n}}function gu(e,t,s,n){const i=Le(e,0);return i[t][s]=n,i}function yr(e){const t=e.length,s=i=>Math.max(0,Math.min(t-1,i)),n=new Array(t);for(let i=0;i<t;i++){const a=new Array(t);for(let f=0;f<t;f++)a[f]=e[i][s(f-1)]+e[i][s(f+1)]+e[s(i-1)][f]+e[s(i+1)][f]-4*e[i][f];n[i]=a}return n}function Oe(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function xr(e,t,s,n,i,a,f){return Number.isFinite(e)?Oe(e,s[a][f],i,[[s[a][f]+4*t[a][f],"the −4·center term is missing — the Laplacian is left + right + up + down − 4·center"],[-s[a][f],"the sign is flipped — it is the four neighbors minus 4·center, not the other way round"],[n[a][f],"that edge clamped instead of wrapping — column 0's left neighbor is the last column, not itself"]]):"that cell read outside the grid — wrap the index instead: below 0 becomes size − 1, past size − 1 becomes 0"}function br(e,t,s){const n=e*t*t;return[[e+(fe.du*s+n+fe.f*(1-e))*fe.dt,"the reaction term is added to U — V eats U, so u·v·v is subtracted here and added in stepV"],[e+(fe.du*s-n)*fe.dt,"the feed term is missing — U is replenished everywhere by f · (1 − u)"],[e+(fe.du*s+fe.f*(1-e))*fe.dt,"the reaction term u·v·v never got subtracted"],[e,"U came back unchanged — none of the three terms reached the return value"]]}function wr(e,t,s){const n=e*t*t;return[[t+(fe.dv*s-n-(fe.f+fe.k)*t)*fe.dt,"the reaction term is subtracted from V — V is what grows on it, so u·v·v is added here"],[t+(fe.dv*s+n)*fe.dt,"the kill term is missing — V is removed at (f + k) · v"],[t+(fe.dv*s-(fe.f+fe.k)*t)*fe.dt,"the reaction term u·v·v never got added"],[t,"V came back unchanged — none of the three terms reached the return value"]]}function xt(e,t){const s=Math.min(1,e*2.5),n=Math.min(1,e);return[[[n,n*n,.25+.75*n],"the brightness scale is missing — t = Math.min(1, value * 2.5)"],[[s,s,.25+.75*s],"green is t · t, not t — the square is what holds the mid-tones back"],[[.25+.75*s,s*s,s],"the 0.25 floor belongs on blue — the order is this.color(t, t * t, 0.25 + 0.75 * t, 1)"],[[s,s*s,s],"blue is missing its 0.25 floor — still water should be deep blue, not black"]].map(i=>[i[0][t]*255,i[1]])}var yu={id:"3-4",track:3,title:"Reaction–Diffusion",blurb:"Two chemicals, two equations, and suddenly: coral, fingerprints, leopard spots.",tasks:[{slug:"laplacian",title:"The Laplacian: Ask Your Neighbors",intro:`<p>Diffusion is gossip: every cell drifts toward the average of its neighbors.
        The operator that measures "how far am I from my neighbors' average" is the
        <strong>Laplacian</strong>, and on a grid it's a five-read gather:
        <code>left + right + up + down − 4·center</code>. Positive means the neighbors are
        higher and stuff will flow in; negative means it flows out.</p>
        <p>One wrinkle: simulations hate edges. Instead of clamping like the convolution
        filters in track 2, we <strong>wrap around</strong> — the left neighbor of column 0 is
        column 31. The world becomes a torus and every cell has exactly four neighbors,
        no special cases.</p>`,goal:`<strong>Goal:</strong> complete the gather kernel so it returns the 5-point
        Laplacian of <code>field</code> with wrap-around edges.`,requirements:["Wrap all four neighbor indexes — below <code>0</code> becomes <code>size − 1</code>, past <code>size − 1</code> becomes <code>0</code>","Read exactly five cells: the four direct neighbors and the center","Return <code>left + right + up + down − 4·center</code>"],hints:[{title:"Hint 1 — the wrap is an if",body:`<p>Same trick as clamping, different else:</p>
<pre><code>let xr = this.thread.x + 1;
if (xr &gt; this.constants.size - 1) xr = 0;</code></pre>
<p>The starter already wrote <code>xl</code> for you — mirror it three times.</p>`},{title:"Hint 2 — five reads",body:`<p>The neighbors sit at <code>field[y][xl]</code>, <code>field[y][xr]</code>,
            <code>field[yd][x]</code> and <code>field[yu][x]</code> — only ever vary
            <em>one</em> coordinate at a time. The center is <code>field[y][x]</code>.</p>`},{title:"Hint 3 — the whole return",body:`<pre><code>return field[y][xl] + field[y][xr] + field[yd][x]
  + field[yu][x] - 4 * field[y][x];</code></pre>`}],transfer:`The 5-point Laplacian stencil is the beating heart of PDE solvers on every
        platform — heat, waves, pressure projection in fluids. On big CUDA/ROCm clusters the
        wrap you just wrote becomes a <em>halo exchange</em>: each GPU ships its border rows to
        the neighbor that needs them before every step.`,starterCode:`// The Laplacian: how far is each cell from its neighbors' average?
// The world is a torus — indexes wrap around the edges.
const gpu = new GPU({ mode });

const laplacian = gpu.createKernel(function (field) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1;
  if (xl < 0) xl = this.constants.size - 1;
  // TODO: wrap xr (right), yu (up) and yd (down) the same way,
  // then return left + right + up + down - 4 * center.
  return 0;
}, { output: [32, 32], constants: { size: 32 } });

const result = laplacian(field);
console.log('at a bump:', result[16][16]);
`,solutionCode:`// The Laplacian: how far is each cell from its neighbors' average?
// The world is a torus — indexes wrap around the edges.
const gpu = new GPU({ mode });

const laplacian = gpu.createKernel(function (field) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1;
  if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1;
  if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1;
  if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1;
  if (yu > this.constants.size - 1) yu = 0;
  return field[y][xl] + field[y][xr] + field[yd][x] + field[yu][x] - 4 * field[y][x];
}, { output: [32, 32], constants: { size: 32 } });

const result = laplacian(field);
console.log('at a bump:', result[16][16]);
`,inputs:e=>({field:Ia(e,32,3401)}),publicTests:[{name:"a uniform field has zero Laplacian everywhere",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Le(32,.7),s=e.kernel(t);e.assert(s&&s.length===32&&s[0].length===32,"expected a 32×32 result");const n=Vt(t),i=yr(t);for(let a=0;a<32;a++)for(let f=0;f<32;f++){const m=xr(s[a][f],t,n,i,1e-4,a,f);e.assertClose(s[a][f],0,1e-4,m||`cell [${a}][${f}] of a flat field`)}}},{name:"a single spike: <code>−4·s</code> at the peak, <code>+s</code> on each neighbor — even across the wrap",run:async e=>{const t=gu(32,0,0,2),s=e.kernel(t),n=Vt(t),i=yr(t),a=(f,m)=>xr(s[f][m],t,n,i,1e-4,f,m);e.assertClose(s[0][0],-8,1e-4,a(0,0)||"the peak itself (−4 × 2)"),e.assertClose(s[0][1],2,1e-4,a(0,1)||"right neighbor"),e.assertClose(s[1][0],2,1e-4,a(1,0)||"neighbor above"),e.assertClose(s[0][31],2,1e-4,a(0,31)||"LEFT neighbor — wraps to column 31"),e.assertClose(s[31][0],2,1e-4,a(31,0)||"neighbor below — wraps to row 31"),e.assertClose(s[5][5],0,1e-4,a(5,5)||"a far-away cell")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ia(e.utils,32,909),s=e.kernel(t),n=Vt(t),i=yr(t);let a=0;for(let f=0;f<32;f++)for(let m=0;m<32;m++){const A=xr(s[f][m],t,n,i,.001,f,m);e.assertClose(s[f][m],n[f][m],.001,A||`cell [${f}][${m}]`),a+=s[f][m]}e.assertClose(a,0,.01,"on a torus, gains and losses cancel exactly")}}]},{slug:"gray-scott-step",title:"One Step of Gray–Scott",intro:`<p>Now the chemistry. Gray–Scott tracks two chemicals on the same grid:
        <code>U</code> (food, fed in everywhere) and <code>V</code> (the eater —
        <code>U + 2V → 3V</code>, so V converts U into more V, and is itself slowly removed).
        Per cell, per step:</p>
        <p><code>u' = u + (Du·∇²u − u·v² + F·(1 − u))·dt</code><br>
        <code>v' = v + (Dv·∇²v + u·v² − (F + K)·v)·dt</code></p>
        <p>Each equation is your task-1 Laplacian plus three pointwise terms — diffusion,
        reaction, feed/kill. One kernel per chemical: both are gathers over the <em>old</em>
        grids, so all 1,024 cells of a step can run in parallel.</p>`,goal:`<strong>Goal:</strong> finish the two update kernels — <code>stepU</code> and
        <code>stepV</code> each return their chemical's next value. The Laplacians are already
        gathered for you.`,requirements:["Keep the kernel order as wired: <code>stepU</code> first, then <code>stepV</code>","The reaction term is <code>u·v·v</code> — U loses it, V gains it","<code>stepU</code> returns <code>uc + (du·lap − uc·vc·vc + f·(1 − uc))·dt</code>","<code>stepV</code> returns <code>vc + (dv·lap + uc·vc·vc − (f + k)·vc)·dt</code>"],hints:[{title:"Hint 1 — everything is already in scope",body:`<p><code>lap</code>, <code>uc</code> and <code>vc</code> are computed for you;
            the parameters live in <code>this.constants</code> (<code>du</code>, <code>f</code>,
            <code>dt</code> in stepU; <code>dv</code>, <code>f</code>, <code>k</code>,
            <code>dt</code> in stepV). The TODO is one <code>return</code> per kernel.</p>`},{title:"Hint 2 — stepU, spelled out",body:`<pre><code>return uc + (this.constants.du * lap - uc * vc * vc
  + this.constants.f * (1 - uc)) * this.constants.dt;</code></pre>
<p>stepV is the same shape
            with <code>+ uc·vc·vc</code> and <code>− (f + k)·vc</code>.</p>`}],transfer:`Fusing the stencil and the pointwise chemistry into one kernel is a classic
        GPU move — in CUDA or a WGSL compute shader you'd do exactly this to touch each grid
        cell's memory once per step instead of once per term. Separate passes per term would
        triple the bandwidth bill.`,starterCode:`// Two chemicals, two kernels. Both read the OLD u and v grids.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  // TODO: return uc + (du * lap - uc*vc*vc + f * (1 - uc)) * dt
  //       (parameters live in this.constants)
  return uc;
}, { output: [32, 32], constants: { size: 32, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  // TODO: return vc + (dv * lap + uc*vc*vc - (f + k) * vc) * dt
  return vc;
}, { output: [32, 32], constants: { size: 32, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const newU = stepU(u0, v0);
const newV = stepV(u0, v0);
console.log('center after one step — U:', newU[16][16], ' V:', newV[16][16]);
`,solutionCode:`// Two chemicals, two kernels. Both read the OLD u and v grids.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  return uc + (this.constants.du * lap - uc * vc * vc
    + this.constants.f * (1 - uc)) * this.constants.dt;
}, { output: [32, 32], constants: { size: 32, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  return vc + (this.constants.dv * lap + uc * vc * vc
    - (this.constants.f + this.constants.k) * vc) * this.constants.dt;
}, { output: [32, 32], constants: { size: 32, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const newU = stepU(u0, v0);
const newV = stepV(u0, v0);
console.log('center after one step — U:', newU[16][16], ' V:', newV[16][16]);
`,inputs:()=>{const e=Lt(32,6);return{u0:e.u,v0:e.v}},publicTests:[{name:"the calm ocean is a fixed point: U=1, V=0 stays exactly put",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels (stepU then stepV), found ${e.kernels.length}`);const t=Le(32,1),s=Le(32,0),n=e.kernels[0](t,s),i=e.kernels[1](t,s);for(let a=0;a<32;a+=5)for(let f=0;f<32;f+=5)e.assertClose(n[a][f],1,1e-5,`U at [${a}][${f}] — nothing to react, nothing to feed`),e.assertClose(i[a][f],0,1e-5,`V at [${a}][${f}] — no V, no reaction`)}},{name:"a well-mixed beaker (u=0.6, v=0.3) follows the equations exactly",run:async e=>{const t=Le(32,.6),s=Le(32,.3),n=.6*.3*.3,i=.6+(-n+fe.f*(1-.6))*fe.dt,a=.3+(n-(fe.f+fe.k)*.3)*fe.dt,f=e.kernels[0](t,s),m=e.kernels[1](t,s),A=br(.6,.3,0),N=wr(.6,.3,0);for(let H=0;H<32;H+=7)for(let Y=0;Y<32;Y+=7)e.assertClose(f[H][Y],i,1e-4,Oe(f[H][Y],i,1e-4,A)||`U at [${H}][${Y}]`),e.assertClose(m[H][Y],a,1e-4,Oe(m[H][Y],a,1e-4,N)||`V at [${H}][${Y}]`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=Le(32,.8),s=Le(32,.1),n=.8*.1*.1,i=.8+(-n+fe.f*(1-.8))*fe.dt,a=.1+(n-(fe.f+fe.k)*.1)*fe.dt,f=e.kernels[0](t,s),m=e.kernels[1](t,s),A=br(.8,.1,0),N=wr(.8,.1,0);for(let H=0;H<32;H+=3)for(let Y=0;Y<32;Y+=3)e.assertClose(f[H][Y],i,1e-4,Oe(f[H][Y],i,1e-4,A)||`U at [${H}][${Y}]`),e.assertClose(m[H][Y],a,1e-4,Oe(m[H][Y],a,1e-4,N)||`V at [${H}][${Y}]`)}},{name:"private test #2",run:async e=>{const t=Lt(32,10),[s,n]=Ha(t.u,t.v),i=e.kernels[0](t.u,t.v),a=e.kernels[1](t.u,t.v),f=Vt(t.u),m=Vt(t.v);for(let A=0;A<32;A++)for(let N=0;N<32;N++){const H=t.u[A][N],Y=t.v[A][N];e.assertClose(i[A][N],s[A][N],1e-4,Oe(i[A][N],s[A][N],1e-4,br(H,Y,f[A][N]))||`U at [${A}][${N}]`),e.assertClose(a[A][N],n[A][N],1e-4,Oe(a[A][N],n[A][N],1e-4,wr(H,Y,m[A][N]))||`V at [${A}][${N}]`)}}}]},{slug:"feed-it-back",title:"Feed It Back: 100 Steps",intro:`<p>One step is chemistry; a hundred steps is <em>morphogenesis</em>. The kernels
        stay on the GPU — the loop lives in JavaScript: call both step kernels, take their
        outputs, feed them back in as next step's inputs. This is the same feedback move as
        the cellular automata in 3.3, just with two grids in flight instead of one.</p>
        <p>The trap: both kernels must read the <strong>same snapshot</strong>. If you
        overwrite <code>u</code> before calling <code>stepV</code>, chemical V reacts with food
        from the <em>future</em> — the simulation drifts and the tests will know. Hold both new
        grids, <em>then</em> swap. Graphics folk call this ping-pong buffering.</p>`,goal:`<strong>Goal:</strong> run 100 Gray–Scott steps from the seeded grids
        <code>seedU</code> / <code>seedV</code>, feeding each step's outputs into the next —
        both kernels always reading the same snapshot.`,requirements:["Loop exactly <code>STEPS</code> (100) times in plain JavaScript","Call <code>stepU(u, v)</code> and <code>stepV(u, v)</code> with the <em>same</em> <code>u</code> and <code>v</code>","Only after both calls, replace <code>u</code> and <code>v</code> with the new grids"],hints:[{title:"Hint 1 — why the starter is wrong",body:`<p>The starter does</p>
<pre><code>u = stepU(u, v);
v = stepV(u, v);</code></pre>
<p>— by the
            second call, <code>u</code> is already next step's grid. Stash both results in
            temporaries before assigning either.</p>`},{title:"Hint 2 — the loop body",body:`<pre><code>const nextU = stepU(u, v);
const nextV = stepV(u, v);
u = nextU;
v = nextV;</code></pre>
<p>Four lines, inside
            <code>for (let i = 0; i &lt; STEPS; i++)</code>.</p>`}],transfer:`This snapshot discipline is double buffering, and GPUs institutionalize it:
        a WebGPU or Metal simulation binds texture A for reading and texture B for writing,
        then swaps the bindings each frame — you never write the buffer you're reading. CUDA
        codes do the same by swapping two device pointers between kernel launches.`,starterCode:`// The kernels from last task, prewired at 48×48. Your job: the time loop.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  return uc + (this.constants.du * lap - uc * vc * vc
    + this.constants.f * (1 - uc)) * this.constants.dt;
}, { output: [48, 48], constants: { size: 48, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  return vc + (this.constants.dv * lap + uc * vc * vc
    - (this.constants.f + this.constants.k) * vc) * this.constants.dt;
}, { output: [48, 48], constants: { size: 48, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const STEPS = 100;
let u = seedU;
let v = seedV;

// TODO: run STEPS steps. This single "step" has TWO bugs: it only runs
// once, and stepV reads the u we just overwrote — future food!
u = stepU(u, v);
v = stepV(u, v);

console.log('center V after', STEPS, 'steps:', v[24][24]);
`,solutionCode:`// The kernels from last task, prewired at 48×48. Your job: the time loop.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  return uc + (this.constants.du * lap - uc * vc * vc
    + this.constants.f * (1 - uc)) * this.constants.dt;
}, { output: [48, 48], constants: { size: 48, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  return vc + (this.constants.dv * lap + uc * vc * vc
    - (this.constants.f + this.constants.k) * vc) * this.constants.dt;
}, { output: [48, 48], constants: { size: 48, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const STEPS = 100;
let u = seedU;
let v = seedV;

for (let i = 0; i < STEPS; i++) {
  // Both kernels read the same snapshot; swap only after both are done.
  const nextU = stepU(u, v);
  const nextV = stepV(u, v);
  u = nextU;
  v = nextV;
}

console.log('center V after', STEPS, 'steps:', v[24][24]);
`,inputs:()=>{const e=Lt(48,8);return{seedU:e.u,seedV:e.v}},publicTests:[{name:"after 100 steps the kernels are seeing step 99, not the seed",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels (stepU then stepV), found ${e.kernels.length}`);const t=Lt(48,8),[s,n]=gr(t.u,t.v,99),i=e.kernels[0].lastArgs;e.assert(i&&i.length>=2,"stepU should have been called with (u, v)");const[a,f]=i;e.assert(Math.abs(a[24][24]-t.u[24][24])>.05,"the last stepU call still saw the seed — did the loop actually feed results back?");const m=[[24,24],[24,20],[20,28],[24,12],[4,4]];for(const[A,N]of m)e.assertClose(a[A][N],s[A][N],.002,`U at [${A}][${N}] after 99 steps`),e.assertClose(f[A][N],n[A][N],.002,`V at [${A}][${N}] after 99 steps`)}},{name:"stepU and stepV read the same snapshot — no future food",run:async e=>{const t=e.kernels[0].lastArgs[0],s=e.kernels[1].lastArgs[0];e.assert(t&&s,"both kernels should have been called with (u, v)");const n=[[24,24],[24,21],[27,24],[21,27],[24,16]];for(const[i,a]of n)e.assertClose(s[i][a],t[i][a],1e-4,`u[${i}][${a}] differs between the stepU and stepV calls — swap only after BOTH kernels ran`)}},{name:"V has escaped the seed square, and both fields stay in [0, 1]",run:async e=>{const[t,s]=e.kernels[0].lastArgs;e.assert(s[24][12]>1e-4,"V should have diffused well outside the 8×8 seed by step 99");for(let n=0;n<48;n+=3)for(let i=0;i<48;i+=3)e.assert(t[n][i]>=-1e-6&&t[n][i]<=1+1e-6,`U at [${n}][${i}] left [0, 1] — unstable loop?`),e.assert(s[n][i]>=-1e-6&&s[n][i]<=1+1e-6,`V at [${n}][${i}] left [0, 1] — unstable loop?`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=Lt(48,12);let s=t.u,n=t.v;for(let f=0;f<40;f++){const m=e.kernels[0](s,n),A=e.kernels[1](s,n);s=m,n=A}const[i,a]=gr(t.u,t.v,40);for(let f=0;f<48;f++)for(let m=0;m<48;m++)e.assertClose(s[f][m],i[f][m],.001,`U at [${f}][${m}] after 40 steps`),e.assertClose(n[f][m],a[f][m],.001,`V at [${f}][${m}] after 40 steps`)}}]},{slug:"paint-the-pattern",title:"Paint the Pattern",intro:`<p>Payoff time. The whole simulation is wired below — 64×64 grid, 200 steps —
        and it ends holding <code>v</code>, a grid of numbers with coral growing in it.
        Numbers deserve pixels: one graphical kernel, exactly like the painters in 3.1,
        turns the V field into the picture the module cover promised.</p>
        <p>The palette is fixed so we can test it: brightness
        <code>t = min(1, v·2.5)</code>, painted as
        <code>color(t, t·t, 0.25 + 0.75·t)</code> — a deep-blue ocean at <code>v = 0</code>
        rising through violet to white-hot at the pattern's crest.</p>`,goal:`<strong>Goal:</strong> complete the <code>paint</code> kernel — map this thread's
        <code>v</code> value through the palette and put it on screen.`,requirements:["Read this thread's value: <code>v[this.thread.y][this.thread.x]</code>","Brightness <code>t = Math.min(1, value * 2.5)</code>","Paint <code>this.color(t, t * t, 0.25 + 0.75 * t, 1)</code>"],hints:[{title:"Hint 1 — same move as the luminance painter",body:`<p>This is the paint kernel from track 1 with a fancier ramp: read one number
            from the grid, compute the channels, call <code>this.color()</code>.
            <code>Math.min</code> works inside kernels.</p>`},{title:"Hint 2 — the whole body",body:`<pre><code>const t = Math.min(1, v[this.thread.y][this.thread.x] * 2.5);
this.color(t, t * t, 0.25 + 0.75 * t, 1);</code></pre>`}],transfer:`Compute passes that end in a draw are the shape of every GPU simulation you've
        seen on the web: WebGPU chains compute pipelines into a render pipeline whose fragment
        shader is your <code>paint</code>; Metal apps do the same with a compute encoder feeding
        a fragment function. The data never has to leave the card.`,starterCode:`// 200 steps of Gray–Scott, then paint the V field. The sim is done —
// the painter is yours.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  return uc + (this.constants.du * lap - uc * vc * vc
    + this.constants.f * (1 - uc)) * this.constants.dt;
}, { output: [64, 64], constants: { size: 64, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  return vc + (this.constants.dv * lap + uc * vc * vc
    - (this.constants.f + this.constants.k) * vc) * this.constants.dt;
}, { output: [64, 64], constants: { size: 64, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const paint = gpu.createKernel(function (v) {
  // TODO: t = Math.min(1, value * 2.5), then
  // this.color(t, t * t, 0.25 + 0.75 * t, 1)
  this.color(1, 0, 1, 1);
}, { output: [64, 64], graphical: true });

let u = seedU;
let v = seedV;
for (let i = 0; i < 200; i++) {
  const nextU = stepU(u, v);
  const nextV = stepV(u, v);
  u = nextU;
  v = nextV;
}

paint(v);
render(paint.canvas);
`,solutionCode:`// 200 steps of Gray–Scott, then paint the V field. The sim is done —
// the painter is yours.
const gpu = new GPU({ mode });

const stepU = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = u[y][xl] + u[y][xr] + u[yd][x] + u[yu][x] - 4 * uc;
  return uc + (this.constants.du * lap - uc * vc * vc
    + this.constants.f * (1 - uc)) * this.constants.dt;
}, { output: [64, 64], constants: { size: 64, du: 0.2, f: 0.035, dt: 1 } });

const stepV = gpu.createKernel(function (u, v) {
  const x = this.thread.x;
  const y = this.thread.y;
  let xl = x - 1; if (xl < 0) xl = this.constants.size - 1;
  let xr = x + 1; if (xr > this.constants.size - 1) xr = 0;
  let yd = y - 1; if (yd < 0) yd = this.constants.size - 1;
  let yu = y + 1; if (yu > this.constants.size - 1) yu = 0;
  const uc = u[y][x];
  const vc = v[y][x];
  const lap = v[y][xl] + v[y][xr] + v[yd][x] + v[yu][x] - 4 * vc;
  return vc + (this.constants.dv * lap + uc * vc * vc
    - (this.constants.f + this.constants.k) * vc) * this.constants.dt;
}, { output: [64, 64], constants: { size: 64, dv: 0.1, f: 0.035, k: 0.06, dt: 1 } });

const paint = gpu.createKernel(function (v) {
  const t = Math.min(1, v[this.thread.y][this.thread.x] * 2.5);
  this.color(t, t * t, 0.25 + 0.75 * t, 1);
}, { output: [64, 64], graphical: true });

let u = seedU;
let v = seedV;
for (let i = 0; i < 200; i++) {
  const nextU = stepU(u, v);
  const nextV = stepV(u, v);
  u = nextU;
  v = nextV;
}

paint(v);
render(paint.canvas);
`,inputs:()=>{const e=Lt(64,8);return{seedU:e.u,seedV:e.v}},publicTests:[{name:"a 64×64 canvas is rendered",run:async e=>{e.assert(e.canvas,"no canvas — is paint graphical: true, and did you call render()?"),e.assert(e.canvas.width===64&&e.canvas.height===64,`expected a 64×64 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();e.assert(t.length===4096*4,"pixel buffer should hold 64×64 RGBA values")}},{name:"the palette is exact: still water is deep blue, <code>v = 0.2</code> is half-lit violet",run:async e=>{const t=e.kernels.find(n=>n.kernel&&n.kernel.graphical);e.assert(t,"no graphical kernel found"),t(Le(64,0));let s=t.getPixels();for(let n=0;n<s.length;n+=331*4)e.assertClose(s[n],0,3,Oe(s[n],0,3,xt(0,0))||`red at byte ${n} for v = 0`),e.assertClose(s[n+1],0,3,Oe(s[n+1],0,3,xt(0,1))||`green at byte ${n} for v = 0`),e.assertClose(s[n+2],.25*255,3,Oe(s[n+2],.25*255,3,xt(0,2))||`blue at byte ${n} for v = 0`);t(Le(64,.2)),s=t.getPixels();for(let n=0;n<s.length;n+=331*4)e.assertClose(s[n],.5*255,3,Oe(s[n],.5*255,3,xt(.2,0))||`red at byte ${n} for v = 0.2`),e.assertClose(s[n+1],.25*255,3,Oe(s[n+1],.25*255,3,xt(.2,1))||`green at byte ${n} for v = 0.2`),e.assertClose(s[n+2],.625*255,3,Oe(s[n+2],.625*255,3,xt(.2,2))||`blue at byte ${n} for v = 0.2`)}},{name:"the picture is alive — bright coral on a dark ocean",run:async e=>{const t=e.kernels.find(m=>m.kernel&&m.kernel.graphical);e.assert(t,"no graphical kernel found");const s=Lt(64,8),[,n]=gr(s.u,s.v,200);t(n);const i=t.getPixels();let a=0,f=0;for(let m=0;m<i.length;m+=4)i[m]>150&&a++,i[m]<20&&f++;e.assert(a>=50,`expected at least 50 bright pattern pixels after 200 steps, found ${a}`),e.assert(f>=1e3,`expected a mostly-dark ocean around the pattern, found only ${f} dark pixels`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(n=>n.kernel&&n.kernel.graphical);e.assert(t,"no graphical kernel found"),t(Le(64,.6));let s=t.getPixels();for(let n=0;n<s.length;n+=449*4)e.assertClose(s[n],255,3,`red at byte ${n} for v = 0.6`),e.assertClose(s[n+1],255,3,`green at byte ${n} for v = 0.6`),e.assertClose(s[n+2],255,3,`blue at byte ${n} for v = 0.6`);t(Le(64,.1)),s=t.getPixels();for(let n=0;n<s.length;n+=449*4)e.assertClose(s[n],.25*255,3,Oe(s[n],.25*255,3,xt(.1,0))||`red at byte ${n} for v = 0.1`),e.assertClose(s[n+1],.0625*255,3,Oe(s[n+1],.0625*255,3,xt(.1,1))||`green at byte ${n} for v = 0.1`),e.assertClose(s[n+2],.4375*255,3,Oe(s[n+2],.4375*255,3,xt(.1,2))||`blue at byte ${n} for v = 0.1`)}}]}]},xu=Object.freeze({__proto__:null,default:yu});function dt(e){return(e-32)/16}function bu(e,t,s){const n=Math.max(s-Math.abs(e-t),0)/s;return Math.min(e,t)-n*n*s*.25}function Ge(e,t,s){const n=(s*64+t)*4;return[e[n],e[n+1],e[n+2],e[n+3]]}function Ut(e,t,s,n,i,a,f){const m=Ge(t,s,32)[n],A=Ge(t,s,31)[n];e.assert(Math.abs(m-i)<=a||Math.abs(A-i)<=a,`${f} — expected ≈${i} ±${a}, got ${m} (row 32) / ${A} (row 31)`)}function ke(e,t){return Ge(e,t,32)[0]}function cn(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Ma(e,t,s,n,i){const a=e-s,f=t-n,m=a*a+f*f;return[[Math.sqrt(m),"the radius never got subtracted — the field is length(p − center) − r"],[Math.sqrt(m)+i,"the radius is added where the field subtracts it — inside the sphere the distance is NEGATIVE"],[m-i,"that is the squared distance — take Math.sqrt of it before subtracting r"]]}function $a(e,t,s){const n=Math.max(s-Math.abs(e-t),0)/s,i=Math.min(e,t);return[[i,"that is the hard minimum — smin has to dip below both fields where they are within k of each other"],[i-n*n*s,"the dip is h * h * k * 0.25 — the 0.25 is missing"],[i-n*s*.25,"h is squared in the dip: h * h * k * 0.25"]]}function wu(e){for(let t=0;t<e.length;t+=4)if(e[t]>180)return null;return"not one pixel hit the surface — the ray never leaves its starting plane; each of the 48 passes has to step forward by the distance itself, t += d"}function vu(e,t){return Math.abs(e-128)<=6&&Math.abs(t-128)<=6?"blue came back mid-grey — the raw differences are tiny, so this normal was never normalized; divide nx, ny and nz by len":null}function ku(e){return e>=.15*255-4&&e<=.2*255?"this pixel is sitting on the 0.15 ambient floor — the diffuse term added nothing; the dot product needs the UNIT normal, so divide nx, ny, nz by len first":null}var Tu={id:"3-5",track:3,title:"Ray-Marched Metaballs",blurb:"Signed distance fields and soft shadows — a real-time 3D scene with no triangles at all.",tasks:[{slug:"sphere-sdf",title:"The Sphere as a Number",intro:`<p>A <strong>signed distance field</strong> describes a shape with one function:
        for any point in space it returns the distance to the nearest surface — positive outside,
        <em>negative</em> inside, exactly zero on the skin. A whole sphere collapses into one line:
        <code>length(p - center) - radius</code>. No vertices, no triangles, just math.</p>
        <p>That's a perfect fit for a kernel: one thread per sample point, each evaluating the same
        tiny function. Here every pixel owns a point on the <code>z = 0</code> slice through the
        scene — pixel <code>(ix, iy)</code> maps to world <code>((ix - 32) / 16, (iy - 32) / 16)</code>,
        so the canvas spans −2…2 and the center pixel sits exactly at the origin.</p>`,goal:`<strong>Goal:</strong> make the kernel return the signed distance from this thread's
        world point to a sphere at <code>(cx, cy)</code> with radius <code>r</code>.`,requirements:["Map the thread to world space: <code>(this.thread.x - 32) / 16</code> (already wired up)","Measure the offset from the sphere center: <code>(wx - cx, wy - cy)</code>","Return its length via <code>Math.sqrt</code>, <strong>minus</strong> <code>r</code>"],hints:[{title:"Hint 1 — what should the numbers look like?",body:`<p>For the unit sphere at the origin: the center pixel is <em>inside</em>, distance
            <code>-1</code>. A pixel one unit from the center sits exactly on the surface —
            distance <code>0</code>. The far corner at (−2, −2) reads <code>√8 − 1 ≈ 1.83</code>.</p>`},{title:"Hint 2 — the whole thing",body:`<pre><code>const dx = wx - cx;
const dy = wy - cy;
return Math.sqrt(dx * dx + dy * dy) - r;</code></pre>`}],transfer:`Distance fields are a production technique, not a toy: Valve renders crisp text
        from SDF textures, and every WebGPU fragment shader that draws rounded rectangles is
        evaluating exactly this per-pixel field — one invocation per pixel, one signed distance out.`,starterCode:`// A sphere in one line of math: length(p - center) - radius.
const gpu = new GPU({ mode });

const sliceSDF = gpu.createKernel(function (cx, cy, r) {
  // This thread's point on the z = 0 slice: center pixel = origin.
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  // TODO: return the signed distance from (wx, wy) to the sphere:
  // the length of (wx - cx, wy - cy), minus r.
  return 0;
}, { output: [64, 64] });

const field = sliceSDF(0, 0, 1);
console.log('center (inside, should be -1):', field[32][32]);
console.log('far corner (outside):', field[0][0]);
`,solutionCode:`// A sphere in one line of math: length(p - center) - radius.
const gpu = new GPU({ mode });

const sliceSDF = gpu.createKernel(function (cx, cy, r) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  const dx = wx - cx;
  const dy = wy - cy;
  return Math.sqrt(dx * dx + dy * dy) - r;
}, { output: [64, 64] });

const field = sliceSDF(0, 0, 1);
console.log('center (inside, should be -1):', field[32][32]);
console.log('far corner (outside):', field[0][0]);
`,publicTests:[{name:"unit sphere: center reads −1, surface reads 0, corner reads ≈1.83",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(0,0,1);e.assert(t&&t.length===64&&t[0].length===64,"expected a 64×64 field");const s=(n,i,a)=>cn(t[n][i],a,.002,Ma(dt(i),dt(n),0,0,1));e.assertClose(t[32][32],-1,.002,s(32,32,-1)||"center of the sphere (inside)"),e.assertClose(t[32][48],0,.002,s(32,48,0)||"one unit right of center (on the surface)"),e.assertClose(t[0][0],Math.sqrt(8)-1,.002,s(0,0,Math.sqrt(8)-1)||"far corner (outside)")}},{name:"the field is radially symmetric around the sphere center",run:async e=>{const t=e.kernel(0,0,1);for(const s of[5,10,15])e.assertClose(t[32][32+s],t[32][32-s],.002,`left/right at offset ${s}`),e.assertClose(t[32][32+s],t[32+s][32],.002,`x/y at offset ${s}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(.5,-.25,.75),s=[[3,7],[20,44],[32,32],[50,12],[10,58],[63,63]];for(const[n,i]of s){const a=dt(i)-.5,f=dt(n)+.25,m=Math.sqrt(a*a+f*f)-.75,A=cn(t[n][i],m,.003,Ma(dt(i),dt(n),.5,-.25,.75));e.assertClose(t[n][i],m,.003,A||`cell [${n}][${i}]`)}}}]},{slug:"smooth-min",title:"Two Spheres Melt Into One",intro:`<p>Combining two SDFs is just <code>Math.min</code> — the nearest surface wins.
        But <code>min</code> leaves a hard crease where the shapes meet. Swap it for a
        <strong>smooth minimum</strong> and the fields <em>blend</em>: wherever the two distances
        are within <code>k</code> of each other, the result dips below both, bulging the surfaces
        toward each other. That bulge <em>is</em> a metaball.</p>
        <p>Since every task from here on needs this helper, register it once with
        <code>gpu.addFunction()</code> — gpu.js transpiles it alongside the kernel, and any kernel
        on that GPU instance can call it by name.</p>`,goal:`<strong>Goal:</strong> implement the polynomial smooth minimum and use it to blend
        two sphere fields into one metaball field.`,requirements:["Register <code>smin(a, b, k)</code> with <code>gpu.addFunction()</code>","Blend amount: <code>h = Math.max(k - Math.abs(a - b), 0) / k</code>","Return <code>Math.min(a, b) - h * h * k * 0.25</code>","Call <code>smin(d1, d2, k)</code> in the kernel instead of <code>Math.min</code>"],hints:[{title:"Hint 1 — what should change?",body:`<p>Far from the seam, <code>smin</code> equals plain <code>min</code>. Exactly
            between the spheres the two distances are equal, so <code>h = 1</code> and the field
            dips by <code>k / 4</code>. With the starter's arguments the midpoint should read
            <code>0.2 − 0.1 = 0.1</code>.</p>`},{title:"Hint 2 — the function, verbatim",body:`<pre><code>gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});</code></pre>`}],transfer:`Smooth blends of implicit surfaces are how molecular-surface renderers in CUDA
        draw proteins and how Metal-based sculpting apps merge clay-like blobs — the union operator
        is soft everywhere, and the GPU evaluates it millions of times per frame without blinking.`,starterCode:`// min() gives a hard crease. smin() gives a blend. Metaballs are just smin.
const gpu = new GPU({ mode });

// TODO: register smin(a, b, k) with gpu.addFunction():
//   const h = Math.max(k - Math.abs(a - b), 0.0) / k;
//   return Math.min(a, b) - h * h * k * 0.25;

const metaField = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  // one sphere at (-sep, 0), one at (+sep, 0)
  const d1 = Math.sqrt((wx + sep) * (wx + sep) + wy * wy) - r;
  const d2 = Math.sqrt((wx - sep) * (wx - sep) + wy * wy) - r;
  // TODO: blend with smin(d1, d2, k) instead of the hard minimum
  return Math.min(d1, d2);
}, { output: [64, 64] });

const field = metaField(0.7, 0.5, 0.4);
console.log('midpoint (should dip to 0.1):', field[32][32]);
`,solutionCode:`// min() gives a hard crease. smin() gives a blend. Metaballs are just smin.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

const metaField = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  const d1 = Math.sqrt((wx + sep) * (wx + sep) + wy * wy) - r;
  const d2 = Math.sqrt((wx - sep) * (wx - sep) + wy * wy) - r;
  return smin(d1, d2, k);
}, { output: [64, 64] });

const field = metaField(0.7, 0.5, 0.4);
console.log('midpoint (should dip to 0.1):', field[32][32]);
`,publicTests:[{name:"midpoint dips below the plain minimum by <code>k / 4</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(.7,.5,.4),s=cn(t[32][32],.1,.003,$a(.2,.2,.4));e.assertClose(t[32][32],.1,.003,s||"field at the midpoint"),e.assert(t[32][32]<.2-.05,"the blend should dip clearly below plain min (0.2)")}},{name:"field stays symmetric and returns to plain min far from the seam",run:async e=>{const t=e.kernel(.7,.5,.4);for(const f of[6,12,20])e.assertClose(t[32][32+f],t[32][32-f],.003,`mirror pair at offset ${f}`);const s=dt(0),n=dt(0),i=Math.sqrt((s+.7)*(s+.7)+n*n)-.5,a=Math.sqrt((s-.7)*(s-.7)+n*n)-.5;e.assertClose(t[0][0],Math.min(i,a),.003,"far corner (outside the blend zone)")}}],privateTests:[{name:"private test #1",run:async e=>{const[t,s,n]=[.9,.45,.3],i=e.kernel(t,s,n);for(let a=1;a<64;a+=7)for(let f=1;f<64;f+=7){const m=dt(f),A=dt(a),N=Math.sqrt((m+t)*(m+t)+A*A)-s,H=Math.sqrt((m-t)*(m-t)+A*A)-s,Y=bu(N,H,n),Ae=cn(i[a][f],Y,.003,$a(N,H,n));e.assertClose(i[a][f],Y,.003,Ae||`cell [${a}][${f}]`)}}}]},{slug:"ray-march",title:"March Until You Hit Something",intro:`<p>Now the third dimension. Every pixel fires a ray straight into the screen
        (orthographic: origin <code>(wx, wy, -2.5)</code>, direction <code>(0, 0, 1)</code>), and
        the SDF turns finding the surface into a beautiful trick called <strong>sphere tracing</strong>:
        the distance at your current point is a <em>guaranteed-safe step size</em> — nothing can be
        closer than that. So step exactly that far, re-evaluate, repeat.</p>
        <p>Near a surface the distance shrinks toward zero, so the march converges right onto the
        skin. When <code>d</code> drops below a small epsilon, that ray has hit. Rays that miss just
        keep flying — after a fixed number of steps you paint them background. GPUs need that fixed
        bound: every thread runs the same loop, so give it <code>48</code> iterations and let hits
        simply stop making progress.</p>`,goal:`<strong>Goal:</strong> write the ray-marching loop — 48 steps of
        <code>t += d</code> — and paint hit pixels pink, misses dark blue.`,requirements:["Loop a fixed 48 times; sample the field at <code>(wx, wy, -2.5 + t)</code>","Flag a hit when <code>d &lt; 0.01</code>","Step forward by the distance itself: <code>t += d</code>","Hits get <code>this.color(0.98, 0.63, 0.89, 1)</code>, misses the background color"],hints:[{title:"Hint 1 — the loop shape",body:`<p>Two state variables before the loop: <code>let t = 0.0;</code> and
            <code>let hit = 0.0;</code>. Inside: evaluate <code>sceneDist</code>, set
            <code>hit = 1.0</code> when close enough, then advance <code>t</code>. After the loop,
            color by <code>hit</code>.</p>`},{title:"Hint 2 — the loop, spelled out",body:`<pre><code>for (let i = 0; i &lt; 48; i++) {
  const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
  if (d &lt; 0.01) hit = 1.0;
  t += d;
}</code></pre>`}],transfer:`The fixed bound is a gpu.js/WebGL constraint — shader loop bounds must be
        static, so we guard instead of <code>break</code>. Real marchers on CUDA, WGSL or
        Shadertoy <em>do</em> break on a hit, and it pays off whenever a whole warp hits or
        misses together. The durable lesson is about <em>divergence</em>: warps execute in
        lockstep, so a warp runs as long as its slowest thread.`,starterCode:`// Sphere tracing: the SDF value IS a safe step size. Step, sample, repeat.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

// The full 3D metaball field: two spheres at (±sep, 0, 0), blended by k.
gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const marchScene = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  // TODO: march! Start at t = 0 and repeat 48 times:
  //   d = sceneDist(wx, wy, -2.5 + t, sep, r, k)
  //   if d < 0.01 → this ray has hit the surface
  //   step forward: t += d
  // This only checks the starting plane — nothing is that close, so
  // every pixel comes out background until you write the loop:
  let hit = 0.0;
  if (sceneDist(wx, wy, -2.5, sep, r, k) < 0.01) hit = 1.0;

  if (hit > 0.5) this.color(0.98, 0.63, 0.89, 1);
  else this.color(0.02, 0.03, 0.06, 1);
}, { output: [64, 64], graphical: true });

marchScene(0.55, 0.5, 0.3);
render(marchScene.canvas);
`,solutionCode:`// Sphere tracing: the SDF value IS a safe step size. Step, sample, repeat.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

// The full 3D metaball field: two spheres at (±sep, 0, 0), blended by k.
gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const marchScene = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (d < 0.01) hit = 1.0;
    t += d;
  }
  if (hit > 0.5) this.color(0.98, 0.63, 0.89, 1);
  else this.color(0.02, 0.03, 0.06, 1);
}, { output: [64, 64], graphical: true });

marchScene(0.55, 0.5, 0.3);
render(marchScene.canvas);
`,publicTests:[{name:"a 64×64 canvas whose corners are background",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===64&&e.canvas.height===64,`expected a 64×64 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();for(const[s,n]of[[0,0],[63,0],[0,63],[63,63]]){const[i]=Ge(t,s,n);e.assert(i<40,`corner (${s}, ${n}) should be background, got red ${i}`)}}},{name:"rays through the blob hit: center and both lobes come back pink",run:async e=>{e.kernel(.55,.5,.3);const t=e.getPixels(),s=wu(t);for(const n of[22,32,42]){const i=ke(t,n);e.assert(i>180,s||`pixel (${n}, 32) should be a hit (red > 180), got ${i}`)}}},{name:"the silhouette is left-right symmetric",run:async e=>{e.kernel(.55,.5,.3);const t=e.getPixels();for(const s of[6,10,14,18]){const n=ke(t,32-s)>128,i=ke(t,32+s)>128;e.assert(n===i,`pixels (${32-s}, 32) and (${32+s}, 32) should both hit or both miss`)}}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(1.2,.45,.15);const t=e.getPixels();e.assert(ke(t,32)<40,`separated spheres: the center ray should miss, got red ${ke(t,32)}`);for(const s of[13,51]){const n=ke(t,s);e.assert(n>180,`pixel (${s}, 32) is inside a lobe and should hit, got red ${n}`)}}}]},{slug:"normals",title:"Normals Without Geometry",intro:`<p>Lighting needs surface normals, and a mesh would hand them to you per-vertex. We
        have no mesh — but we have something better. The normal of an implicit surface is the
        <strong>gradient</strong> of its distance field: the direction in which distance grows
        fastest is exactly "straight off the surface".</p>
        <p>Estimate it with <strong>central differences</strong>: nudge the hit point by a tiny
        <code>e</code> along each axis, sample the field on both sides, subtract. Six extra field
        evaluations, then normalize. The classic way to sanity-check normals is to paint them:
        <code>n * 0.5 + 0.5</code> maps each component into color range — a head-on surface
        (normal <code>(0, 0, -1)</code>, pointing at the camera) renders as
        <code>rgb(0.5, 0.5, 0)</code>, that mustard-olive tone every graphics programmer knows.</p>`,goal:`<strong>Goal:</strong> at each hit point, compute the finite-difference normal of
        <code>sceneDist</code> and paint each component as <code>n * 0.5 + 0.5</code> —
        hint 2 has the exact <code>this.color</code> call.`,requirements:["Remember the hit distance: record <code>tHit</code> at the <em>first</em> hit","Sample ± <code>e = 0.01</code> along x, y and z around the hit point","Normalize the three differences with <code>Math.sqrt</code>","Paint <code>n * 0.5 + 0.5</code>; misses keep the background color"],hints:[{title:"Hint 1 — one axis at a time",body:`<p>The x component before normalizing is</p>
<pre><code>sceneDist(wx + e, wy, pz, …) - sceneDist(wx - e, wy, pz, …)</code></pre>
<p>where <code>pz = -2.5 + tHit</code>. Same pattern for y and z.</p>`},{title:"Hint 2 — normalize and paint",body:`<pre><code>const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
this.color(nx / len * 0.5 + 0.5,
  ny / len * 0.5 + 0.5,
  nz / len * 0.5 + 0.5, 1);</code></pre>`}],transfer:`Gradient-by-central-differences is the same stencil you'd write in a CUDA fluid
        solver or a ROCm heightfield pipeline, and Metal deferred renderers reconstruct normals
        from depth buffers with exactly this two-sided sampling.`,starterCode:`// The normal of an SDF surface is its gradient. Six samples buy it.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const showNormals = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    // TODO: central differences with e = 0.01 around (wx, wy, pz),
    // normalize (nx, ny, nz), then paint n * 0.5 + 0.5.
    this.color(1, 1, 1, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

showNormals(0.55, 0.5, 0.3);
render(showNormals.canvas);
`,solutionCode:`// The normal of an SDF surface is its gradient. Six samples buy it.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const showNormals = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    const e = 0.01;
    const nx = sceneDist(wx + e, wy, pz, sep, r, k) - sceneDist(wx - e, wy, pz, sep, r, k);
    const ny = sceneDist(wx, wy + e, pz, sep, r, k) - sceneDist(wx, wy - e, pz, sep, r, k);
    const nz = sceneDist(wx, wy, pz + e, sep, r, k) - sceneDist(wx, wy, pz - e, sep, r, k);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    this.color(nx / len * 0.5 + 0.5, ny / len * 0.5 + 0.5, nz / len * 0.5 + 0.5, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

showNormals(0.55, 0.5, 0.3);
render(showNormals.canvas);
`,publicTests:[{name:"the head-on center pixel paints <code>rgb(0.5, 0.5, 0)</code> — normal (0, 0, −1)",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.kernel(.55,.5,.3);const t=e.getPixels();Ut(e,t,32,0,128,14,"center red (nx = 0)"),Ut(e,t,32,1,128,14,"center green (ny = 0)");const s=Ge(t,32,32)[2],n=Ge(t,32,31)[2];e.assert(s<40&&n<40,vu(s,n)||`center blue should be near 0 (nz = -1, facing the camera), got ${s}/${n}`)}},{name:"mirrored hit pixels have mirrored normals: red channels sum to ≈255",run:async e=>{e.kernel(.55,.5,.3);const t=e.getPixels();for(const s of[8,10]){const n=Ge(t,32-s,32),i=Ge(t,32+s,32);e.assert(n[1]>60&&i[1]>60,`both probes at ±${s} should be hits`),e.assert(Math.abs(n[0]+i[0]-255)<=24,`red(${32-s}) + red(${32+s}) should be ≈255 (nx antisymmetric), got ${n[0]+i[0]}`)}}},{name:"misses keep the background color",run:async e=>{e.kernel(.55,.5,.3);const t=e.getPixels();for(const[s,n]of[[0,0],[63,0],[0,63],[63,63]]){const[,i]=Ge(t,s,n);e.assert(i<40,`corner (${s}, ${n}) should be background, got green ${i}`)}}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(1.2,.45,.15);const t=e.getPixels();e.assert(Ge(t,32,32)[1]<40,"center should be background now"),Ut(e,t,51,0,124,16,"right lobe red (nx ≈ 0)"),Ut(e,t,51,1,128,16,"right lobe green (ny = 0)");const s=Ge(t,13,32),n=Ge(t,51,32);e.assert(Math.abs(s[0]+n[0]-255)<=26,`mirrored lobe pixels should have mirrored nx, got ${s[0]+n[0]}`)}}]},{slug:"diffuse-lighting",title:"Turn On the Light",intro:`<p>With normals in hand, lighting is one dot product. <strong>Lambert's law</strong>:
        a surface facing a light head-on catches full brightness; tilt it away and brightness falls
        with the cosine of the angle — which is exactly <code>n · l</code>, the dot product of the
        unit normal and the unit direction <em>toward</em> the light. Clamp it at zero so surfaces
        facing away go dark instead of negative.</p>
        <p>Two finishing touches make it look right: an <strong>ambient floor</strong> of
        <code>0.15</code> so shadowed sides stay readable, and an albedo tint — multiply the final
        brightness into the metaball's pink <code>(1.0, 0.62, 0.86)</code>.</p>`,goal:`<strong>Goal:</strong> light each hit point with
        <code>c = 0.15 + 0.85 * Math.max(nx * lx + ny * ly + nz * lz, 0)</code> and paint
        <code>this.color(c, c * 0.62, c * 0.86, 1)</code>.`,requirements:["Take the dot product of the normal with the light direction <code>(lx, ly, lz)</code>","Clamp with <code>Math.max(dot, 0.0)</code> — no negative light","Apply the ambient floor: <code>c = 0.15 + 0.85 * diff</code>","Tint by the albedo: <code>this.color(c, c * 0.62, c * 0.86, 1)</code>"],hints:[{title:"Hint 1 — sanity-check the center",body:`<p>The center pixel's normal is <code>(0, 0, -1)</code> and the light is
            <code>(-0.6, 0, -0.8)</code>, so the dot product is <code>0.8</code> and
            <code>c = 0.15 + 0.85 × 0.8 = 0.83</code> — a bright, not-quite-white pink.</p>`},{title:"Hint 2 — the two lines",body:`<pre><code>const diff = Math.max(nx / len * lx + ny / len * ly
  + nz / len * lz, 0.0);
const c = 0.15 + 0.85 * diff;</code></pre>`}],transfer:`<code>max(dot(n, l), 0.0)</code> is character-for-character the same in GLSL,
        WGSL, HLSL and Metal Shading Language — Lambert diffuse may be the single most portable
        line of shading code in existence.`,starterCode:`// Lighting is a dot product: brightness = how squarely you face the light.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const shadeScene = gpu.createKernel(function (sep, r, k, lx, ly, lz) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    const e = 0.01;
    const nx = sceneDist(wx + e, wy, pz, sep, r, k) - sceneDist(wx - e, wy, pz, sep, r, k);
    const ny = sceneDist(wx, wy + e, pz, sep, r, k) - sceneDist(wx, wy - e, pz, sep, r, k);
    const nz = sceneDist(wx, wy, pz + e, sep, r, k) - sceneDist(wx, wy, pz - e, sep, r, k);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    // TODO: Lambert — dot the unit normal with (lx, ly, lz), clamp at 0,
    // then c = 0.15 + 0.85 * diff. Flat 1.0 means "fully lit everywhere":
    const c = 1.0;
    this.color(c, c * 0.62, c * 0.86, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

// light direction: up-left of the camera, pointing at the scene
shadeScene(0.55, 0.5, 0.3, -0.6, 0, -0.8);
render(shadeScene.canvas);
`,solutionCode:`// Lighting is a dot product: brightness = how squarely you face the light.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const shadeScene = gpu.createKernel(function (sep, r, k, lx, ly, lz) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    const e = 0.01;
    const nx = sceneDist(wx + e, wy, pz, sep, r, k) - sceneDist(wx - e, wy, pz, sep, r, k);
    const ny = sceneDist(wx, wy + e, pz, sep, r, k) - sceneDist(wx, wy - e, pz, sep, r, k);
    const nz = sceneDist(wx, wy, pz + e, sep, r, k) - sceneDist(wx, wy, pz - e, sep, r, k);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const diff = Math.max((nx * lx + ny * ly + nz * lz) / len, 0.0);
    const c = 0.15 + 0.85 * diff;
    this.color(c, c * 0.62, c * 0.86, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

// light direction: up-left of the camera, pointing at the scene
shadeScene(0.55, 0.5, 0.3, -0.6, 0, -0.8);
render(shadeScene.canvas);
`,publicTests:[{name:"center pixel brightness matches Lambert: <code>0.15 + 0.85 × 0.8</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.kernel(.55,.5,.3,-.6,0,-.8);const t=e.getPixels(),s=ku(Math.max(ke(t,32),Ge(t,32,31)[0]));Ut(e,t,32,0,212,14,s||"center red"),Ut(e,t,32,1,131,14,s||"center green (red × 0.62)")}},{name:"the side facing the light is brighter than the side facing away",run:async e=>{e.kernel(.55,.5,.3,-.6,0,-.8);const t=e.getPixels(),s=ke(t,22),n=ke(t,42);e.assert(s>n+20,`light comes from the left: red(22) = ${s} should exceed red(42) = ${n} by > 20`)}},{name:"misses keep the background color",run:async e=>{e.kernel(.55,.5,.3,-.6,0,-.8);const t=e.getPixels();for(const[s,n]of[[0,0],[63,63]]){const[i]=Ge(t,s,n);e.assert(i<40,`corner (${s}, ${n}) should be background, got red ${i}`)}}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(.55,.5,.3,.6,0,-.8);const t=e.getPixels();Ut(e,t,32,0,212,14,"center red (same head-on dot product)");const s=ke(t,22),n=ke(t,42);e.assert(n>s+20,`light from the right: red(42) = ${n} should exceed red(22) = ${s} by > 20`)}}]},{slug:"soft-shadows",title:"Soft Shadows, Full Scene",intro:`<p>The payoff. A point is in shadow when something sits between it and the light —
        and you already own the tool that answers that: <em>march again</em>, from the hit point
        toward the light. Here's Inigo Quilez's beautiful upgrade: instead of a binary blocked/clear,
        track how <em>closely</em> the shadow ray grazes the scene. The running minimum of
        <code>3 · d / t</code> (distance over travel) is ≈1 when the ray stays clear, 0 when it's
        blocked, and slides smoothly between when it grazes — a free penumbra.</p>
        <p>The kernel takes a <code>shadowOn</code> switch so you can A/B it: with the light swung
        low to the left, the left lobe casts a soft-edged shadow across the neck of the blob. One
        more march, and a scene with no triangles anywhere gets real cinematography.</p>`,goal:`<strong>Goal:</strong> add the shadow march — 32 steps from the hit point toward the
        light, penumbra factor <code>sh = Math.min(sh, 3.0 * d / st)</code> — and scale the diffuse
        term by it when <code>shadowOn</code> is 1.`,requirements:["Start the shadow ray at <code>st = 0.06</code> so it clears its own surface","Sample at <code>(wx + lx·st, wy + ly·st, pz + lz·st)</code>, 32 steps","Keep the running minimum <code>sh = Math.min(sh, 3.0 * d / st)</code>, clamped ≥ 0","Advance with <code>st += Math.max(d, 0.02)</code>, then shade <code>c = 0.15 + 0.85 * diff * sh</code>"],hints:[{title:"Hint 1 — why 3 · d / t?",body:`<p>At travel distance <code>st</code>, a field value <code>d</code> means the ray
            passes within <code>d</code> of an occluder. The ratio <code>d / st</code> is the sine
            of the "clearance angle" from the surface point — small angle, deep penumbra. The 3
            just sets how sharp the shadow edge is.</p>`},{title:"Hint 2 — the loop, spelled out",body:`<pre><code>let sh = 1.0;
let st = 0.06;
for (let j = 0; j &lt; 32; j++) {
  const d = sceneDist(wx + lx * st, wy + ly * st,
    pz + lz * st, sep, r, k);
  sh = Math.min(sh, 3.0 * d / st);
  st += Math.max(d, 0.02);
}
sh = Math.max(sh, 0.0);</code></pre>
<p>And remember to only apply it when
            <code>shadowOn &gt; 0.5</code>.</p>`}],transfer:`Secondary rays are the moment ray-marching meets real-time ray tracing: shadow
        rays are exactly what RTX/DXR hardware accelerates, and this penumbra estimate ships in
        countless WGSL and GLSL engines as the cheap alternative when you can't afford one.`,starterCode:`// One more march — from the surface toward the light — buys shadows.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const finalScene = gpu.createKernel(function (sep, r, k, lx, ly, lz, shadowOn) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    const e = 0.01;
    const nx = sceneDist(wx + e, wy, pz, sep, r, k) - sceneDist(wx - e, wy, pz, sep, r, k);
    const ny = sceneDist(wx, wy + e, pz, sep, r, k) - sceneDist(wx, wy - e, pz, sep, r, k);
    const nz = sceneDist(wx, wy, pz + e, sep, r, k) - sceneDist(wx, wy, pz - e, sep, r, k);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const diff = Math.max((nx * lx + ny * ly + nz * lz) / len, 0.0);

    let sh = 1.0;
    if (shadowOn > 0.5) {
      // TODO: march toward the light from st = 0.06, 32 steps:
      //   d = sceneDist(wx + lx * st, wy + ly * st, pz + lz * st, sep, r, k)
      //   sh = Math.min(sh, 3.0 * d / st)
      //   st += Math.max(d, 0.02)
      // then clamp: sh = Math.max(sh, 0.0)
    }

    const c = 0.15 + 0.85 * diff * sh;
    this.color(c, c * 0.62, c * 0.86, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

// light swung low to the left — the left lobe should shade the neck
finalScene(0.55, 0.5, 0.3, -0.86, 0, -0.51, 1);
render(finalScene.canvas);
`,solutionCode:`// One more march — from the surface toward the light — buys shadows.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const finalScene = gpu.createKernel(function (sep, r, k, lx, ly, lz, shadowOn) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    const e = 0.01;
    const nx = sceneDist(wx + e, wy, pz, sep, r, k) - sceneDist(wx - e, wy, pz, sep, r, k);
    const ny = sceneDist(wx, wy + e, pz, sep, r, k) - sceneDist(wx, wy - e, pz, sep, r, k);
    const nz = sceneDist(wx, wy, pz + e, sep, r, k) - sceneDist(wx, wy, pz - e, sep, r, k);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const diff = Math.max((nx * lx + ny * ly + nz * lz) / len, 0.0);

    let sh = 1.0;
    if (shadowOn > 0.5) {
      let st = 0.06;
      for (let j = 0; j < 32; j++) {
        const d = sceneDist(wx + lx * st, wy + ly * st, pz + lz * st, sep, r, k);
        sh = Math.min(sh, 3.0 * d / st);
        st += Math.max(d, 0.02);
      }
      sh = Math.max(sh, 0.0);
    }

    const c = 0.15 + 0.85 * diff * sh;
    this.color(c, c * 0.62, c * 0.86, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

// light swung low to the left — the left lobe should shade the neck
finalScene(0.55, 0.5, 0.3, -0.86, 0, -0.51, 1);
render(finalScene.canvas);
`,publicTests:[{name:"a 64×64 canvas: hits keep an ambient floor, corners stay background",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===64&&e.canvas.height===64,`expected a 64×64 canvas, got ${e.canvas.width}×${e.canvas.height}`),e.kernel(.55,.5,.3,-.86,0,-.51,1);const t=e.getPixels();e.assert(ke(t,32)>=28,`center should be a hit with at least the ambient floor, got red ${ke(t,32)}`);for(const[s,n]of[[0,0],[63,63]]){const[i]=Ge(t,s,n);e.assert(i<=18,`corner (${s}, ${n}) should be pure background, got red ${i}`)}}},{name:"toggling <code>shadowOn</code> darkens the neck of the blob by > 60",run:async e=>{e.kernel(.55,.5,.3,-.86,0,-.51,1);const t=Array.from(e.getPixels());e.kernel(.55,.5,.3,-.86,0,-.51,0);const s=Array.from(e.getPixels());let n=0;for(let i=28;i<=40;i++)n=Math.max(n,ke(s,i)-ke(t,i));e.assert(n>60,`expected the shadow to darken some neck pixel by > 60, biggest drop was ${n}`)}},{name:"shadows only ever darken — and the lit flank is untouched",run:async e=>{e.kernel(.55,.5,.3,-.86,0,-.51,1);const t=Array.from(e.getPixels());e.kernel(.55,.5,.3,-.86,0,-.51,0);const s=Array.from(e.getPixels());for(let a=0;a<64;a++)e.assert(ke(t,a)<=ke(s,a)+10,`pixel (${a}, 32) got BRIGHTER with shadows on — sh must stay ≤ 1`);const n=ke(t,18),i=ke(s,18);e.assert(Math.abs(n-i)<=10,`pixel (18, 32) faces the light with a clear path — it should not change (${i} → ${n})`)}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(.6,.48,.25,.86,0,-.51,1);const t=Array.from(e.getPixels());e.kernel(.6,.48,.25,.86,0,-.51,0);const s=Array.from(e.getPixels());e.assert(ke(t,32)<20,"center ray should miss the separated blobs");let n=0;for(let f=24;f<=36;f++)n=Math.max(n,ke(s,f)-ke(t,f));e.assert(n>60,`mirrored light: expected a shadow drop > 60 on the left flank, got ${n}`);const i=ke(t,46),a=ke(s,46);e.assert(Math.abs(i-a)<=10,`pixel (46, 32) faces the mirrored light — it should not change (${a} → ${i})`)}}]}]},Su=Object.freeze({__proto__:null,default:Tu});const _u=[{number:1,title:"GPGPU 101",tagline:"From zero to your first thousand threads"},{number:2,title:"Advanced Math",tagline:"Heavy math, thousands of threads at once"},{number:3,title:"Computational Graphics",tagline:"Pictures computed, not drawn"}],Cu=Object.assign({"./track1/module-1-1.js":vl,"./track1/module-1-2.js":Dl,"./track1/module-1-3.js":zl,"./track1/module-1-4.js":Rl,"./track1/module-1-5.js":Ll,"./track2/module-2-1.js":Kl,"./track2/module-2-2.js":jl,"./track2/module-2-3.js":Jl,"./track2/module-2-4.js":su,"./track2/module-2-5.js":iu,"./track3/module-3-1.js":lu,"./track3/module-3-2.js":cu,"./track3/module-3-3.js":mu,"./track3/module-3-4.js":xu,"./track3/module-3-5.js":Su});function Aa(e){return String(e).split("-").map(Number)}const Xa=Object.values(Cu).map(e=>e.default).filter(Boolean).sort((e,t)=>{const[s,n]=Aa(e.id),[i,a]=Aa(t.id);return s-i||n-a});_u.map(e=>({...e,modules:Xa.filter(t=>t.track===e.number)}));function Eu(e){return Xa.find(t=>t.id===e)||null}function Iu(e,t){return`${e}-${t}`}function Mu(e,t){const s=Eu(e);if(!s)return null;const n=Number(t);return!Number.isInteger(n)||n<1||n>s.tasks.length?null:{module:s,task:s.tasks[n-1],taskNum:n,taskIndex:n-1,taskId:Iu(e,n),total:s.tasks.length}}var dn={exports:{}};/**
 * gpu.js
 * https://gpu.rocks/
 *
 * GPU Accelerated JavaScript
 *
 * @version 2.20.0
 * @date Thu Jul 30 2026 19:36:19 GMT+0800 (Singapore Standard Time)
 *
 * @license MIT
 * The MIT License
 *
 * Copyright (c) 2026 gpu.js Team
 */var $u=dn.exports,Da;function Au(){return Da||(Da=1,(function(e,t){(function(s,n){e.exports=n()})($u,function(){var s=(j,G)=>()=>(G||(j((G={exports:{}}).exports,G),j=null),G.exports),n=s((j,G)=>{function D(g){const k=new Array(g.length);for(let o=0;o<g.length;o++){const l=g[o];l.toArray?k[o]=l.toArray():k[o]=l}return k}function I(){const g=D(arguments);let k=null;for(let o=0;o<this.output.x;o++){this.thread.x=o,this.thread.y=0,this.thread.z=0;const l=this._fn.apply(this,g);k===null&&(k=typeof l=="number"||typeof l=="boolean"?new Float32Array(this.output.x):new Array(this.output.x)),k[o]=typeof l=="object"?new Float32Array(l):l}return k}function _(){const g=D(arguments),k=new Array(this.output.y);for(let o=0;o<this.output.y;o++){let l=null;for(let x=0;x<this.output.x;x++){this.thread.x=x,this.thread.y=o,this.thread.z=0;const w=this._fn.apply(this,g);l===null&&(l=typeof w=="number"||typeof w=="boolean"?new Float32Array(this.output.x):new Array(this.output.x)),l[x]=typeof w=="object"?new Float32Array(w):w}k[o]=l}return k}function c(){const g=D(arguments);for(let k=0;k<this.output.y;k++)for(let o=0;o<this.output.x;o++)this.thread.x=o,this.thread.y=k,this.thread.z=0,this._fn.apply(this,g)}function d(){const g=D(arguments),k=new Array(this.output.z);for(let o=0;o<this.output.z;o++){const l=new Array(this.output.y);for(let x=0;x<this.output.y;x++){let w=null;for(let v=0;v<this.output.x;v++){this.thread.x=v,this.thread.y=x,this.thread.z=o;const C=this._fn.apply(this,g);w===null&&(w=typeof C=="number"||typeof C=="boolean"?new Float32Array(this.output.x):new Array(this.output.x)),w[v]=typeof C=="object"?new Float32Array(C):C}l[x]=w}k[o]=l}return k}function E(g){g.setOutput=l=>{g.output=P(l),g.graphical&&$(g)},g.toJSON=()=>{throw new Error("Not usable with gpuMock")},g.setConstants=l=>(g.constants=l,g),g.setGraphical=l=>(g.graphical=l,g),g.setCanvas=l=>(g.canvas=l,g),g.setContext=l=>(g.context=l,g),g.destroy=()=>{},g.validateSettings=()=>{},g.graphical&&g.output&&$(g),g.exec=function(){return new Promise((l,x)=>{try{l(g.apply(g,arguments))}catch(w){x(w)}})},g.getPixels=l=>{const{x,y:w}=g.output;return l?p(g._imageData.data,x,w):g._imageData.data.slice(0)},g.color=function(l,x,w,v){typeof v>"u"&&(v=1),l=Math.floor(l*255),x=Math.floor(x*255),w=Math.floor(w*255),v=Math.floor(v*255);const C=g.output.x,b=g.output.y,T=g.thread.x+(b-g.thread.y-1)*C;g._colorData[T*4+0]=l,g._colorData[T*4+1]=x,g._colorData[T*4+2]=w,g._colorData[T*4+3]=v};const k=()=>g,o=["setWarnVarUsage","setArgumentTypes","setTactic","setOptimizeFloatMemory","setDebug","setLoopMaxIterations","setConstantTypes","setFunctions","setNativeFunctions","setInjectedNative","setPipeline","setPrecision","setOutputToTexture","setImmutable","setStrictIntegers","setDynamicOutput","setHardcodeConstants","setDynamicArguments","setUseLegacyEncoder","setWarnVarUsage","addSubKernel"];for(let l=0;l<o.length;l++)g[o[l]]=k;return g}function $(g){const{x:k,y:o}=g.output;if(g.context&&g.context.createImageData){const l=new Uint8ClampedArray(k*o*4);g._imageData=g.context.createImageData(k,o),g._colorData=l}else{const l=new Uint8ClampedArray(k*o*4);g._imageData={data:l},g._colorData=l}}function P(g){let k=null;if(g.length)if(g.length===3){const[o,l,x]=g;k={x:o,y:l,z:x}}else if(g.length===2){const[o,l]=g;k={x:o,y:l}}else{const[o]=g;k={x:o}}else k=g;return k}function y(g,k={}){const o=k.output?P(k.output):null;function l(){return l.output.z?d.apply(l,arguments):l.output.y?l.graphical?c.apply(l,arguments):_.apply(l,arguments):I.apply(l,arguments)}return l._fn=g,l.constants=k.constants||null,l.context=k.context||null,l.canvas=k.canvas||null,l.graphical=k.graphical||!1,l._imageData=null,l._colorData=null,l.output=o,l.thread={x:0,y:0,z:0},E(l)}function p(g,k,o){const l=o/2|0,x=k*4,w=new Uint8ClampedArray(k*4),v=g.slice(0);for(let C=0;C<l;++C){const b=C*x,T=(o-C-1)*x;w.set(v.subarray(b,b+x)),v.copyWithin(b,T,T+x),v.set(w,T)}return v}G.exports={gpuMock:y}}),i=s((j,G)=>{(function(D,I){typeof j=="object"&&typeof G<"u"?I(j):(D=typeof globalThis<"u"?globalThis:D||self,I(D.acorn={}))})(j,function(D){var I=[509,0,227,0,150,4,294,9,1368,2,2,1,6,3,41,2,5,0,166,1,574,3,9,9,7,9,32,4,318,1,80,3,71,10,50,3,123,2,54,14,32,10,3,1,11,3,46,10,8,0,46,9,7,2,37,13,2,9,6,1,45,0,13,2,49,13,9,3,2,11,83,11,7,0,3,0,158,11,6,9,7,3,56,1,2,6,3,1,3,2,10,0,11,1,3,6,4,4,68,8,2,0,3,0,2,3,2,4,2,0,15,1,83,17,10,9,5,0,82,19,13,9,214,6,3,8,28,1,83,16,16,9,82,12,9,9,7,19,58,14,5,9,243,14,166,9,71,5,2,1,3,3,2,0,2,1,13,9,120,6,3,6,4,0,29,9,41,6,2,3,9,0,10,10,47,15,343,9,54,7,2,7,17,9,57,21,2,13,123,5,4,0,2,1,2,6,2,0,9,9,49,4,2,1,2,4,9,9,330,3,10,1,2,0,49,6,4,4,14,10,5350,0,7,14,11465,27,2343,9,87,9,39,4,60,6,26,9,535,9,470,0,2,54,8,3,82,0,12,1,19628,1,4178,9,519,45,3,22,543,4,4,5,9,7,3,6,31,3,149,2,1418,49,513,54,5,49,9,0,15,0,23,4,2,14,1361,6,2,16,3,6,2,1,2,4,101,0,161,6,10,9,357,0,62,13,499,13,245,1,2,9,726,6,110,6,6,9,4759,9,787719,239],_=[0,11,2,25,2,18,2,1,2,14,3,13,35,122,70,52,268,28,4,48,48,31,14,29,6,37,11,29,3,35,5,7,2,4,43,157,19,35,5,35,5,39,9,51,13,10,2,14,2,6,2,1,2,10,2,14,2,6,2,1,4,51,13,310,10,21,11,7,25,5,2,41,2,8,70,5,3,0,2,43,2,1,4,0,3,22,11,22,10,30,66,18,2,1,11,21,11,25,71,55,7,1,65,0,16,3,2,2,2,28,43,28,4,28,36,7,2,27,28,53,11,21,11,18,14,17,111,72,56,50,14,50,14,35,39,27,10,22,251,41,7,1,17,2,60,28,11,0,9,21,43,17,47,20,28,22,13,52,58,1,3,0,14,44,33,24,27,35,30,0,3,0,9,34,4,0,13,47,15,3,22,0,2,0,36,17,2,24,20,1,64,6,2,0,2,3,2,14,2,9,8,46,39,7,3,1,3,21,2,6,2,1,2,4,4,0,19,0,13,4,31,9,2,0,3,0,2,37,2,0,26,0,2,0,45,52,19,3,21,2,31,47,21,1,2,0,185,46,42,3,37,47,21,0,60,42,14,0,72,26,38,6,186,43,117,63,32,7,3,0,3,7,2,1,2,23,16,0,2,0,95,7,3,38,17,0,2,0,29,0,11,39,8,0,22,0,12,45,20,0,19,72,200,32,32,8,2,36,18,0,50,29,113,6,2,1,2,37,22,0,26,5,2,1,2,31,15,0,328,18,16,0,2,12,2,33,125,0,80,921,103,110,18,195,2637,96,16,1071,18,5,26,3994,6,582,6842,29,1763,568,8,30,18,78,18,29,19,47,17,3,32,20,6,18,433,44,212,63,129,74,6,0,67,12,65,1,2,0,29,6135,9,1237,42,9,8936,3,2,6,2,1,2,290,16,0,30,2,3,0,15,3,9,395,2309,106,6,12,4,8,8,9,5991,84,2,70,2,1,3,0,3,1,3,3,2,11,2,0,2,6,2,64,2,3,3,7,2,6,2,27,2,3,2,4,2,0,4,6,2,339,3,24,2,24,2,30,2,24,2,30,2,24,2,30,2,24,2,30,2,24,2,7,1845,30,7,5,262,61,147,44,11,6,17,0,322,29,19,43,485,27,229,29,3,0,496,6,2,3,2,1,2,14,2,196,60,67,8,0,1205,3,2,26,2,1,2,0,3,0,2,9,2,3,2,0,2,0,7,0,5,0,2,0,2,0,2,2,2,1,2,0,3,0,2,0,2,0,2,0,2,0,2,1,2,0,3,3,2,6,2,3,2,3,2,0,2,9,2,16,6,2,2,4,2,16,4421,42719,33,4153,7,221,3,5761,15,7472,16,621,2467,541,1507,4938,6,4191],c="‌‍·̀-ͯ·҃-֑҇-ׇֽֿׁׂׅׄؐ-ًؚ-٩ٰۖ-ۜ۟-۪ۤۧۨ-ۭ۰-۹ܑܰ-݊ަ-ް߀-߉߫-߽߳ࠖ-࠙ࠛ-ࠣࠥ-ࠧࠩ-࡙࠭-࡛ࢗ-࢟࣊-ࣣ࣡-ःऺ-़ा-ॏ॑-ॗॢॣ०-९ঁ-ঃ়া-ৄেৈো-্ৗৢৣ০-৯৾ਁ-ਃ਼ਾ-ੂੇੈੋ-੍ੑ੦-ੱੵઁ-ઃ઼ા-ૅે-ૉો-્ૢૣ૦-૯ૺ-૿ଁ-ଃ଼ା-ୄେୈୋ-୍୕-ୗୢୣ୦-୯ஂா-ூெ-ைொ-்ௗ௦-௯ఀ-ఄ఼ా-ౄె-ైొ-్ౕౖౢౣ౦-౯ಁ-ಃ಼ಾ-ೄೆ-ೈೊ-್ೕೖೢೣ೦-೯ೳഀ-ഃ഻഼ാ-ൄെ-ൈൊ-്ൗൢൣ൦-൯ඁ-ඃ්ා-ුූෘ-ෟ෦-෯ෲෳัิ-ฺ็-๎๐-๙ັິ-ຼ່-໎໐-໙༘༙༠-༩༹༵༷༾༿ཱ-྄྆྇ྍ-ྗྙ-ྼ࿆ါ-ှ၀-၉ၖ-ၙၞ-ၠၢ-ၤၧ-ၭၱ-ၴႂ-ႍႏ-ႝ፝-፟፩-፱ᜒ-᜕ᜲ-᜴ᝒᝓᝲᝳ឴-៓៝០-៩᠋-᠍᠏-᠙ᢩᤠ-ᤫᤰ-᤻᥆-᥏᧐-᧚ᨗ-ᨛᩕ-ᩞ᩠-᩿᩼-᪉᪐-᪙᪰-᪽ᪿ-ᫎᬀ-ᬄ᬴-᭄᭐-᭙᭫-᭳ᮀ-ᮂᮡ-ᮭ᮰-᮹᯦-᯳ᰤ-᰷᱀-᱉᱐-᱙᳐-᳔᳒-᳨᳭᳴᳷-᳹᷀-᷿‌‍‿⁀⁔⃐-⃥⃜⃡-⃰⳯-⵿⳱ⷠ-〪ⷿ-゙゚〯・꘠-꘩꙯ꙴ-꙽ꚞꚟ꛰꛱ꠂ꠆ꠋꠣ-ꠧ꠬ꢀꢁꢴ-ꣅ꣐-꣙꣠-꣱ꣿ-꤉ꤦ-꤭ꥇ-꥓ꦀ-ꦃ꦳-꧀꧐-꧙ꧥ꧰-꧹ꨩ-ꨶꩃꩌꩍ꩐-꩙ꩻ-ꩽꪰꪲ-ꪴꪷꪸꪾ꪿꫁ꫫ-ꫯꫵ꫶ꯣ-ꯪ꯬꯭꯰-꯹ﬞ︀-️︠-︯︳︴﹍-﹏０-９＿･",d="ªµºÀ-ÖØ-öø-ˁˆ-ˑˠ-ˤˬˮͰ-ʹͶͷͺ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-Ֆՙՠ-ֈא-תׯ-ײؠ-يٮٯٱ-ۓەۥۦۮۯۺ-ۼۿܐܒ-ܯݍ-ޥޱߊ-ߪߴߵߺࠀ-ࠕࠚࠤࠨࡀ-ࡘࡠ-ࡪࡰ-ࢇࢉ-ࢎࢠ-ࣉऄ-हऽॐक़-ॡॱ-ঀঅ-ঌএঐও-নপ-রলশ-হঽৎড়ঢ়য়-ৡৰৱৼਅ-ਊਏਐਓ-ਨਪ-ਰਲਲ਼ਵਸ਼ਸਹਖ਼-ੜਫ਼ੲ-ੴઅ-ઍએ-ઑઓ-નપ-રલળવ-હઽૐૠૡૹଅ-ଌଏଐଓ-ନପ-ରଲଳଵ-ହଽଡ଼ଢ଼ୟ-ୡୱஃஅ-ஊஎ-ஐஒ-கஙசஜஞடணதந-பம-ஹௐఅ-ఌఎ-ఐఒ-నప-హఽౘ-ౚౝౠౡಀಅ-ಌಎ-ಐಒ-ನಪ-ಳವ-ಹಽೝೞೠೡೱೲഄ-ഌഎ-ഐഒ-ഺഽൎൔ-ൖൟ-ൡൺ-ൿඅ-ඖක-නඳ-රලව-ෆก-ะาำเ-ๆກຂຄຆ-ຊຌ-ຣລວ-ະາຳຽເ-ໄໆໜ-ໟༀཀ-ཇཉ-ཬྈ-ྌက-ဪဿၐ-ၕၚ-ၝၡၥၦၮ-ၰၵ-ႁႎႠ-ჅჇჍა-ჺჼ-ቈቊ-ቍቐ-ቖቘቚ-ቝበ-ኈኊ-ኍነ-ኰኲ-ኵኸ-ኾዀዂ-ዅወ-ዖዘ-ጐጒ-ጕጘ-ፚᎀ-ᎏᎠ-Ᏽᏸ-ᏽᐁ-ᙬᙯ-ᙿᚁ-ᚚᚠ-ᛪᛮ-ᛸᜀ-ᜑᜟ-ᜱᝀ-ᝑᝠ-ᝬᝮ-ᝰក-ឳៗៜᠠ-ᡸᢀ-ᢨᢪᢰ-ᣵᤀ-ᤞᥐ-ᥭᥰ-ᥴᦀ-ᦫᦰ-ᧉᨀ-ᨖᨠ-ᩔᪧᬅ-ᬳᭅ-ᭌᮃ-ᮠᮮᮯᮺ-ᯥᰀ-ᰣᱍ-ᱏᱚ-ᱽᲀ-ᲊᲐ-ᲺᲽ-Ჿᳩ-ᳬᳮ-ᳳᳵᳶᳺᴀ-ᶿḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼⁱⁿₐ-ₜℂℇℊ-ℓℕ℘-ℝℤΩℨK-ℹℼ-ℿⅅ-ⅉⅎⅠ-ↈⰀ-ⳤⳫ-ⳮⳲⳳⴀ-ⴥⴧⴭⴰ-ⵧⵯⶀ-ⶖⶠ-ⶦⶨ-ⶮⶰ-ⶶⶸ-ⶾⷀ-ⷆⷈ-ⷎⷐ-ⷖⷘ-ⷞ々-〇〡-〩〱-〵〸-〼ぁ-ゖ゛-ゟァ-ヺー-ヿㄅ-ㄯㄱ-ㆎㆠ-ㆿㇰ-ㇿ㐀-䶿一-ꒌꓐ-ꓽꔀ-ꘌꘐ-ꘟꘪꘫꙀ-ꙮꙿ-ꚝꚠ-ꛯꜗ-ꜟꜢ-ꞈꞋ-ꟍꟐꟑꟓꟕ-Ƛꟲ-ꠁꠃ-ꠅꠇ-ꠊꠌ-ꠢꡀ-ꡳꢂ-ꢳꣲ-ꣷꣻꣽꣾꤊ-ꤥꤰ-ꥆꥠ-ꥼꦄ-ꦲꧏꧠ-ꧤꧦ-ꧯꧺ-ꧾꨀ-ꨨꩀ-ꩂꩄ-ꩋꩠ-ꩶꩺꩾ-ꪯꪱꪵꪶꪹ-ꪽꫀꫂꫛ-ꫝꫠ-ꫪꫲ-ꫴꬁ-ꬆꬉ-ꬎꬑ-ꬖꬠ-ꬦꬨ-ꬮꬰ-ꭚꭜ-ꭩꭰ-ꯢ가-힣ힰ-ퟆퟋ-ퟻ豈-舘並-龎ﬀ-ﬆﬓ-ﬗיִײַ-ﬨשׁ-זּטּ-לּמּנּסּףּפּצּ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼＡ-Ｚａ-ｚｦ-ﾾￂ-ￇￊ-ￏￒ-ￗￚ-ￜ",E={3:"abstract boolean byte char class double enum export extends final float goto implements import int interface long native package private protected public short static super synchronized throws transient volatile",5:"class enum extends super const export import",6:"enum",strict:"implements interface let package private protected public static yield",strictBind:"eval arguments"},$="break case catch continue debugger default do else finally for function if return switch throw try var while with null true false instanceof typeof void delete new in this",P={5:$,"5module":$+" export import",6:$+" const class extends export import super"},y=/^in(stanceof)?$/,p=new RegExp("["+d+"]"),g=new RegExp("["+d+c+"]");function k(r,u){for(var S=65536,M=0;M<u.length;M+=2){if(S+=u[M],S>r)return!1;if(S+=u[M+1],S>=r)return!0}return!1}function o(r,u){return r<65?r===36:r<91?!0:r<97?r===95:r<123?!0:r<=65535?r>=170&&p.test(String.fromCharCode(r)):u===!1?!1:k(r,_)}function l(r,u){return r<48?r===36:r<58?!0:r<65?!1:r<91?!0:r<97?r===95:r<123?!0:r<=65535?r>=170&&g.test(String.fromCharCode(r)):u===!1?!1:k(r,_)||k(r,I)}var x=function(u,S){S===void 0&&(S={}),this.label=u,this.keyword=S.keyword,this.beforeExpr=!!S.beforeExpr,this.startsExpr=!!S.startsExpr,this.isLoop=!!S.isLoop,this.isAssign=!!S.isAssign,this.prefix=!!S.prefix,this.postfix=!!S.postfix,this.binop=S.binop||null,this.updateContext=null};function w(r,u){return new x(r,{beforeExpr:!0,binop:u})}var v={beforeExpr:!0},C={startsExpr:!0},b={};function T(r,u){return u===void 0&&(u={}),u.keyword=r,b[r]=new x(r,u)}var h={num:new x("num",C),regexp:new x("regexp",C),string:new x("string",C),name:new x("name",C),privateId:new x("privateId",C),eof:new x("eof"),bracketL:new x("[",{beforeExpr:!0,startsExpr:!0}),bracketR:new x("]"),braceL:new x("{",{beforeExpr:!0,startsExpr:!0}),braceR:new x("}"),parenL:new x("(",{beforeExpr:!0,startsExpr:!0}),parenR:new x(")"),comma:new x(",",v),semi:new x(";",v),colon:new x(":",v),dot:new x("."),question:new x("?",v),questionDot:new x("?."),arrow:new x("=>",v),template:new x("template"),invalidTemplate:new x("invalidTemplate"),ellipsis:new x("...",v),backQuote:new x("`",C),dollarBraceL:new x("${",{beforeExpr:!0,startsExpr:!0}),eq:new x("=",{beforeExpr:!0,isAssign:!0}),assign:new x("_=",{beforeExpr:!0,isAssign:!0}),incDec:new x("++/--",{prefix:!0,postfix:!0,startsExpr:!0}),prefix:new x("!/~",{beforeExpr:!0,prefix:!0,startsExpr:!0}),logicalOR:w("||",1),logicalAND:w("&&",2),bitwiseOR:w("|",3),bitwiseXOR:w("^",4),bitwiseAND:w("&",5),equality:w("==/!=/===/!==",6),relational:w("</>/<=/>=",7),bitShift:w("<</>>/>>>",8),plusMin:new x("+/-",{beforeExpr:!0,binop:9,prefix:!0,startsExpr:!0}),modulo:w("%",10),star:w("*",10),slash:w("/",10),starstar:new x("**",{beforeExpr:!0}),coalesce:w("??",1),_break:T("break"),_case:T("case",v),_catch:T("catch"),_continue:T("continue"),_debugger:T("debugger"),_default:T("default",v),_do:T("do",{isLoop:!0,beforeExpr:!0}),_else:T("else",v),_finally:T("finally"),_for:T("for",{isLoop:!0}),_function:T("function",C),_if:T("if"),_return:T("return",v),_switch:T("switch"),_throw:T("throw",v),_try:T("try"),_var:T("var"),_const:T("const"),_while:T("while",{isLoop:!0}),_with:T("with"),_new:T("new",{beforeExpr:!0,startsExpr:!0}),_this:T("this",C),_super:T("super",C),_class:T("class",C),_extends:T("extends",v),_export:T("export"),_import:T("import",C),_null:T("null",C),_true:T("true",C),_false:T("false",C),_in:T("in",{beforeExpr:!0,binop:7}),_instanceof:T("instanceof",{beforeExpr:!0,binop:7}),_typeof:T("typeof",{beforeExpr:!0,prefix:!0,startsExpr:!0}),_void:T("void",{beforeExpr:!0,prefix:!0,startsExpr:!0}),_delete:T("delete",{beforeExpr:!0,prefix:!0,startsExpr:!0})},F=/\r\n?|\n|\u2028|\u2029/,O=new RegExp(F.source,"g");function z(r){return r===10||r===13||r===8232||r===8233}function L(r,u,S){S===void 0&&(S=r.length);for(var M=u;M<S;M++){var R=r.charCodeAt(M);if(z(R))return M<S-1&&R===13&&r.charCodeAt(M+1)===10?M+2:M+1}return-1}var V=/[\u1680\u2000-\u200a\u202f\u205f\u3000\ufeff]/,U=/(?:\s|\/\/.*|\/\*[^]*?\*\/)*/g,X=Object.prototype,q=X.hasOwnProperty,W=X.toString,ee=Object.hasOwn||function(r,u){return q.call(r,u)},se=Array.isArray||function(r){return W.call(r)==="[object Array]"},Z=Object.create(null);function ie(r){return Z[r]||(Z[r]=new RegExp("^(?:"+r.replace(/ /g,"|")+")$"))}function he(r){return r<=65535?String.fromCharCode(r):(r-=65536,String.fromCharCode((r>>10)+55296,(r&1023)+56320))}var we=/(?:[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?:[^\uD800-\uDBFF]|^)[\uDC00-\uDFFF])/,re=function(u,S){this.line=u,this.column=S};re.prototype.offset=function(u){return new re(this.line,this.column+u)};var ce=function(u,S,M){this.start=S,this.end=M,u.sourceFile!==null&&(this.source=u.sourceFile)};function Ve(r,u){for(var S=1,M=0;;){var R=L(r,M,u);if(R<0)return new re(S,u-M);++S,M=R}}var ue={ecmaVersion:null,sourceType:"script",onInsertedSemicolon:null,onTrailingComma:null,allowReserved:null,allowReturnOutsideFunction:!1,allowImportExportEverywhere:!1,allowAwaitOutsideFunction:null,allowSuperOutsideMethod:null,allowHashBang:!1,checkPrivateFields:!0,locations:!1,onToken:null,onComment:null,ranges:!1,program:null,sourceFile:null,directSourceFile:null,preserveParens:!1},Se=!1;function Re(r){var u={};for(var S in ue)u[S]=r&&ee(r,S)?r[S]:ue[S];if(u.ecmaVersion==="latest"?u.ecmaVersion=1e8:u.ecmaVersion==null?(!Se&&typeof console=="object"&&console.warn&&(Se=!0,console.warn(`Since Acorn 8.0.0, options.ecmaVersion is required.
Defaulting to 2020, but this will stop working in the future.`)),u.ecmaVersion=11):u.ecmaVersion>=2015&&(u.ecmaVersion-=2009),u.allowReserved==null&&(u.allowReserved=u.ecmaVersion<5),(!r||r.allowHashBang==null)&&(u.allowHashBang=u.ecmaVersion>=14),se(u.onToken)){var M=u.onToken;u.onToken=function(R){return M.push(R)}}return se(u.onComment)&&(u.onComment=Me(u,u.onComment)),u}function Me(r,u){return function(S,M,R,K,B,J){var Q={type:S?"Block":"Line",value:M,start:R,end:K};r.locations&&(Q.loc=new ce(this,B,J)),r.ranges&&(Q.range=[R,K]),u.push(Q)}}var ae=1,ye=2,Ce=4,Ne=8,ft=16,oi=32,En=64,li=128,ps=256,In=ae|ye|ps;function Mn(r,u){return ye|(r?Ce:0)|(u?Ne:0)}var Gs=0,$n=1,mt=2,ui=3,ci=4,hi=5,_e=function(u,S,M){this.options=u=Re(u),this.sourceFile=u.sourceFile,this.keywords=ie(P[u.ecmaVersion>=6?6:u.sourceType==="module"?"5module":5]);var R="";u.allowReserved!==!0&&(R=E[u.ecmaVersion>=6?6:u.ecmaVersion===5?5:3],u.sourceType==="module"&&(R+=" await")),this.reservedWords=ie(R);var K=(R?R+" ":"")+E.strict;this.reservedWordsStrict=ie(K),this.reservedWordsStrictBind=ie(K+" "+E.strictBind),this.input=String(S),this.containsEsc=!1,M?(this.pos=M,this.lineStart=this.input.lastIndexOf(`
`,M-1)+1,this.curLine=this.input.slice(0,this.lineStart).split(F).length):(this.pos=this.lineStart=0,this.curLine=1),this.type=h.eof,this.value=null,this.start=this.end=this.pos,this.startLoc=this.endLoc=this.curPosition(),this.lastTokEndLoc=this.lastTokStartLoc=null,this.lastTokStart=this.lastTokEnd=this.pos,this.context=this.initialContext(),this.exprAllowed=!0,this.inModule=u.sourceType==="module",this.strict=this.inModule||this.strictDirective(this.pos),this.potentialArrowAt=-1,this.potentialArrowInForAwait=!1,this.yieldPos=this.awaitPos=this.awaitIdentPos=0,this.labels=[],this.undefinedExports=Object.create(null),this.pos===0&&u.allowHashBang&&this.input.slice(0,2)==="#!"&&this.skipLineComment(2),this.scopeStack=[],this.enterScope(ae),this.regexpState=null,this.privateNameStack=[]},it={inFunction:{configurable:!0},inGenerator:{configurable:!0},inAsync:{configurable:!0},canAwait:{configurable:!0},allowSuper:{configurable:!0},allowDirectSuper:{configurable:!0},treatFunctionsAsVar:{configurable:!0},allowNewDotTarget:{configurable:!0},inClassStaticBlock:{configurable:!0}};_e.prototype.parse=function(){var u=this.options.program||this.startNode();return this.nextToken(),this.parseTopLevel(u)},it.inFunction.get=function(){return(this.currentVarScope().flags&ye)>0},it.inGenerator.get=function(){return(this.currentVarScope().flags&Ne)>0&&!this.currentVarScope().inClassFieldInit},it.inAsync.get=function(){return(this.currentVarScope().flags&Ce)>0&&!this.currentVarScope().inClassFieldInit},it.canAwait.get=function(){for(var r=this.scopeStack.length-1;r>=0;r--){var u=this.scopeStack[r];if(u.inClassFieldInit||u.flags&ps)return!1;if(u.flags&ye)return(u.flags&Ce)>0}return this.inModule&&this.options.ecmaVersion>=13||this.options.allowAwaitOutsideFunction},it.allowSuper.get=function(){var r=this.currentThisScope(),u=r.flags,S=r.inClassFieldInit;return(u&En)>0||S||this.options.allowSuperOutsideMethod},it.allowDirectSuper.get=function(){return(this.currentThisScope().flags&li)>0},it.treatFunctionsAsVar.get=function(){return this.treatFunctionsAsVarInScope(this.currentScope())},it.allowNewDotTarget.get=function(){var r=this.currentThisScope(),u=r.flags,S=r.inClassFieldInit;return(u&(ye|ps))>0||S},it.inClassStaticBlock.get=function(){return(this.currentVarScope().flags&ps)>0},_e.extend=function(){for(var u=[],S=arguments.length;S--;)u[S]=arguments[S];for(var M=this,R=0;R<u.length;R++)M=u[R](M);return M},_e.parse=function(u,S){return new this(S,u).parse()},_e.parseExpressionAt=function(u,S,M){var R=new this(M,u,S);return R.nextToken(),R.parseExpression()},_e.tokenizer=function(u,S){return new this(S,u)},Object.defineProperties(_e.prototype,it);var Fe=_e.prototype,Wo=/^(?:'((?:\\[^]|[^'\\])*?)'|"((?:\\[^]|[^"\\])*?)")/;Fe.strictDirective=function(r){if(this.options.ecmaVersion<5)return!1;for(;;){U.lastIndex=r,r+=U.exec(this.input)[0].length;var u=Wo.exec(this.input.slice(r));if(!u)return!1;if((u[1]||u[2])==="use strict"){U.lastIndex=r+u[0].length;var S=U.exec(this.input),M=S.index+S[0].length,R=this.input.charAt(M);return R===";"||R==="}"||F.test(S[0])&&!(/[(`.[+\-/*%<>=,?^&]/.test(R)||R==="!"&&this.input.charAt(M+1)==="=")}r+=u[0].length,U.lastIndex=r,r+=U.exec(this.input)[0].length,this.input[r]===";"&&r++}},Fe.eat=function(r){return this.type===r?(this.next(),!0):!1},Fe.isContextual=function(r){return this.type===h.name&&this.value===r&&!this.containsEsc},Fe.eatContextual=function(r){return this.isContextual(r)?(this.next(),!0):!1},Fe.expectContextual=function(r){this.eatContextual(r)||this.unexpected()},Fe.canInsertSemicolon=function(){return this.type===h.eof||this.type===h.braceR||F.test(this.input.slice(this.lastTokEnd,this.start))},Fe.insertSemicolon=function(){if(this.canInsertSemicolon())return this.options.onInsertedSemicolon&&this.options.onInsertedSemicolon(this.lastTokEnd,this.lastTokEndLoc),!0},Fe.semicolon=function(){!this.eat(h.semi)&&!this.insertSemicolon()&&this.unexpected()},Fe.afterTrailingComma=function(r,u){if(this.type===r)return this.options.onTrailingComma&&this.options.onTrailingComma(this.lastTokStart,this.lastTokStartLoc),u||this.next(),!0},Fe.expect=function(r){this.eat(r)||this.unexpected()},Fe.unexpected=function(r){this.raise(r??this.start,"Unexpected token")};var Ls=function(){this.shorthandAssign=this.trailingComma=this.parenthesizedAssign=this.parenthesizedBind=this.doubleProto=-1};Fe.checkPatternErrors=function(r,u){if(r){r.trailingComma>-1&&this.raiseRecoverable(r.trailingComma,"Comma is not permitted after the rest element");var S=u?r.parenthesizedAssign:r.parenthesizedBind;S>-1&&this.raiseRecoverable(S,u?"Assigning to rvalue":"Parenthesized pattern")}},Fe.checkExpressionErrors=function(r,u){if(!r)return!1;var S=r.shorthandAssign,M=r.doubleProto;if(!u)return S>=0||M>=0;S>=0&&this.raise(S,"Shorthand property assignments are valid only in destructuring patterns"),M>=0&&this.raiseRecoverable(M,"Redefinition of __proto__ property")},Fe.checkYieldAwaitInDefaultParams=function(){this.yieldPos&&(!this.awaitPos||this.yieldPos<this.awaitPos)&&this.raise(this.yieldPos,"Yield expression cannot be a default value"),this.awaitPos&&this.raise(this.awaitPos,"Await expression cannot be a default value")},Fe.isSimpleAssignTarget=function(r){return r.type==="ParenthesizedExpression"?this.isSimpleAssignTarget(r.expression):r.type==="Identifier"||r.type==="MemberExpression"};var ne=_e.prototype;ne.parseTopLevel=function(r){var u=Object.create(null);for(r.body||(r.body=[]);this.type!==h.eof;){var S=this.parseStatement(null,!0,u);r.body.push(S)}if(this.inModule)for(var M=0,R=Object.keys(this.undefinedExports);M<R.length;M+=1){var K=R[M];this.raiseRecoverable(this.undefinedExports[K].start,"Export '"+K+"' is not defined")}return this.adaptDirectivePrologue(r.body),this.next(),r.sourceType=this.options.sourceType,this.finishNode(r,"Program")};var An={kind:"loop"},Ho={kind:"switch"};ne.isLet=function(r){if(this.options.ecmaVersion<6||!this.isContextual("let"))return!1;U.lastIndex=this.pos;var u=U.exec(this.input),S=this.pos+u[0].length,M=this.input.charCodeAt(S);if(M===91||M===92)return!0;if(r)return!1;if(M===123||M>55295&&M<56320)return!0;if(o(M,!0)){for(var R=S+1;l(M=this.input.charCodeAt(R),!0);)++R;if(M===92||M>55295&&M<56320)return!0;var K=this.input.slice(S,R);if(!y.test(K))return!0}return!1},ne.isAsyncFunction=function(){if(this.options.ecmaVersion<8||!this.isContextual("async"))return!1;U.lastIndex=this.pos;var r=U.exec(this.input),u=this.pos+r[0].length,S;return!F.test(this.input.slice(this.pos,u))&&this.input.slice(u,u+8)==="function"&&(u+8===this.input.length||!(l(S=this.input.charCodeAt(u+8))||S>55295&&S<56320))},ne.parseStatement=function(r,u,S){var M=this.type,R=this.startNode(),K;switch(this.isLet(r)&&(M=h._var,K="let"),M){case h._break:case h._continue:return this.parseBreakContinueStatement(R,M.keyword);case h._debugger:return this.parseDebuggerStatement(R);case h._do:return this.parseDoStatement(R);case h._for:return this.parseForStatement(R);case h._function:return r&&(this.strict||r!=="if"&&r!=="label")&&this.options.ecmaVersion>=6&&this.unexpected(),this.parseFunctionStatement(R,!1,!r);case h._class:return r&&this.unexpected(),this.parseClass(R,!0);case h._if:return this.parseIfStatement(R);case h._return:return this.parseReturnStatement(R);case h._switch:return this.parseSwitchStatement(R);case h._throw:return this.parseThrowStatement(R);case h._try:return this.parseTryStatement(R);case h._const:case h._var:return K=K||this.value,r&&K!=="var"&&this.unexpected(),this.parseVarStatement(R,K);case h._while:return this.parseWhileStatement(R);case h._with:return this.parseWithStatement(R);case h.braceL:return this.parseBlock(!0,R);case h.semi:return this.parseEmptyStatement(R);case h._export:case h._import:if(this.options.ecmaVersion>10&&M===h._import){U.lastIndex=this.pos;var B=U.exec(this.input),J=this.pos+B[0].length,Q=this.input.charCodeAt(J);if(Q===40||Q===46)return this.parseExpressionStatement(R,this.parseExpression())}return this.options.allowImportExportEverywhere||(u||this.raise(this.start,"'import' and 'export' may only appear at the top level"),this.inModule||this.raise(this.start,"'import' and 'export' may appear only with 'sourceType: module'")),M===h._import?this.parseImport(R):this.parseExport(R,S);default:if(this.isAsyncFunction())return r&&this.unexpected(),this.next(),this.parseFunctionStatement(R,!0,!r);var de=this.value,le=this.parseExpression();return M===h.name&&le.type==="Identifier"&&this.eat(h.colon)?this.parseLabeledStatement(R,de,le,r):this.parseExpressionStatement(R,le)}},ne.parseBreakContinueStatement=function(r,u){var S=u==="break";this.next(),this.eat(h.semi)||this.insertSemicolon()?r.label=null:this.type!==h.name?this.unexpected():(r.label=this.parseIdent(),this.semicolon());for(var M=0;M<this.labels.length;++M){var R=this.labels[M];if((r.label==null||R.name===r.label.name)&&(R.kind!=null&&(S||R.kind==="loop")||r.label&&S))break}return M===this.labels.length&&this.raise(r.start,"Unsyntactic "+u),this.finishNode(r,S?"BreakStatement":"ContinueStatement")},ne.parseDebuggerStatement=function(r){return this.next(),this.semicolon(),this.finishNode(r,"DebuggerStatement")},ne.parseDoStatement=function(r){return this.next(),this.labels.push(An),r.body=this.parseStatement("do"),this.labels.pop(),this.expect(h._while),r.test=this.parseParenExpression(),this.options.ecmaVersion>=6?this.eat(h.semi):this.semicolon(),this.finishNode(r,"DoWhileStatement")},ne.parseForStatement=function(r){this.next();var u=this.options.ecmaVersion>=9&&this.canAwait&&this.eatContextual("await")?this.lastTokStart:-1;if(this.labels.push(An),this.enterScope(0),this.expect(h.parenL),this.type===h.semi)return u>-1&&this.unexpected(u),this.parseFor(r,null);var S=this.isLet();if(this.type===h._var||this.type===h._const||S){var M=this.startNode(),R=S?"let":this.value;return this.next(),this.parseVar(M,!0,R),this.finishNode(M,"VariableDeclaration"),(this.type===h._in||this.options.ecmaVersion>=6&&this.isContextual("of"))&&M.declarations.length===1?(this.options.ecmaVersion>=9&&(this.type===h._in?u>-1&&this.unexpected(u):r.await=u>-1),this.parseForIn(r,M)):(u>-1&&this.unexpected(u),this.parseFor(r,M))}var K=this.isContextual("let"),B=!1,J=this.containsEsc,Q=new Ls,de=this.start,le=u>-1?this.parseExprSubscripts(Q,"await"):this.parseExpression(!0,Q);return this.type===h._in||(B=this.options.ecmaVersion>=6&&this.isContextual("of"))?(u>-1?(this.type===h._in&&this.unexpected(u),r.await=!0):B&&this.options.ecmaVersion>=8&&(le.start===de&&!J&&le.type==="Identifier"&&le.name==="async"?this.unexpected():this.options.ecmaVersion>=9&&(r.await=!1)),K&&B&&this.raise(le.start,"The left-hand side of a for-of loop may not start with 'let'."),this.toAssignable(le,!1,Q),this.checkLValPattern(le),this.parseForIn(r,le)):(this.checkExpressionErrors(Q,!0),u>-1&&this.unexpected(u),this.parseFor(r,le))},ne.parseFunctionStatement=function(r,u,S){return this.next(),this.parseFunction(r,fs|(S?0:Dn),!1,u)},ne.parseIfStatement=function(r){return this.next(),r.test=this.parseParenExpression(),r.consequent=this.parseStatement("if"),r.alternate=this.eat(h._else)?this.parseStatement("if"):null,this.finishNode(r,"IfStatement")},ne.parseReturnStatement=function(r){return!this.inFunction&&!this.options.allowReturnOutsideFunction&&this.raise(this.start,"'return' outside of function"),this.next(),this.eat(h.semi)||this.insertSemicolon()?r.argument=null:(r.argument=this.parseExpression(),this.semicolon()),this.finishNode(r,"ReturnStatement")},ne.parseSwitchStatement=function(r){this.next(),r.discriminant=this.parseParenExpression(),r.cases=[],this.expect(h.braceL),this.labels.push(Ho),this.enterScope(0);for(var u,S=!1;this.type!==h.braceR;)if(this.type===h._case||this.type===h._default){var M=this.type===h._case;u&&this.finishNode(u,"SwitchCase"),r.cases.push(u=this.startNode()),u.consequent=[],this.next(),M?u.test=this.parseExpression():(S&&this.raiseRecoverable(this.lastTokStart,"Multiple default clauses"),S=!0,u.test=null),this.expect(h.colon)}else u||this.unexpected(),u.consequent.push(this.parseStatement(null));return this.exitScope(),u&&this.finishNode(u,"SwitchCase"),this.next(),this.labels.pop(),this.finishNode(r,"SwitchStatement")},ne.parseThrowStatement=function(r){return this.next(),F.test(this.input.slice(this.lastTokEnd,this.start))&&this.raise(this.lastTokEnd,"Illegal newline after throw"),r.argument=this.parseExpression(),this.semicolon(),this.finishNode(r,"ThrowStatement")};var Xo=[];ne.parseCatchClauseParam=function(){var r=this.parseBindingAtom(),u=r.type==="Identifier";return this.enterScope(u?oi:0),this.checkLValPattern(r,u?ci:mt),this.expect(h.parenR),r},ne.parseTryStatement=function(r){if(this.next(),r.block=this.parseBlock(),r.handler=null,this.type===h._catch){var u=this.startNode();this.next(),this.eat(h.parenL)?u.param=this.parseCatchClauseParam():(this.options.ecmaVersion<10&&this.unexpected(),u.param=null,this.enterScope(0)),u.body=this.parseBlock(!1),this.exitScope(),r.handler=this.finishNode(u,"CatchClause")}return r.finalizer=this.eat(h._finally)?this.parseBlock():null,!r.handler&&!r.finalizer&&this.raise(r.start,"Missing catch or finally clause"),this.finishNode(r,"TryStatement")},ne.parseVarStatement=function(r,u,S){return this.next(),this.parseVar(r,!1,u,S),this.semicolon(),this.finishNode(r,"VariableDeclaration")},ne.parseWhileStatement=function(r){return this.next(),r.test=this.parseParenExpression(),this.labels.push(An),r.body=this.parseStatement("while"),this.labels.pop(),this.finishNode(r,"WhileStatement")},ne.parseWithStatement=function(r){return this.strict&&this.raise(this.start,"'with' in strict mode"),this.next(),r.object=this.parseParenExpression(),r.body=this.parseStatement("with"),this.finishNode(r,"WithStatement")},ne.parseEmptyStatement=function(r){return this.next(),this.finishNode(r,"EmptyStatement")},ne.parseLabeledStatement=function(r,u,S,M){for(var R=0,K=this.labels;R<K.length;R+=1)K[R].name===u&&this.raise(S.start,"Label '"+u+"' is already declared");for(var B=this.type.isLoop?"loop":this.type===h._switch?"switch":null,J=this.labels.length-1;J>=0;J--){var Q=this.labels[J];if(Q.statementStart===r.start)Q.statementStart=this.start,Q.kind=B;else break}return this.labels.push({name:u,kind:B,statementStart:this.start}),r.body=this.parseStatement(M?M.indexOf("label")===-1?M+"label":M:"label"),this.labels.pop(),r.label=S,this.finishNode(r,"LabeledStatement")},ne.parseExpressionStatement=function(r,u){return r.expression=u,this.semicolon(),this.finishNode(r,"ExpressionStatement")},ne.parseBlock=function(r,u,S){for(r===void 0&&(r=!0),u===void 0&&(u=this.startNode()),u.body=[],this.expect(h.braceL),r&&this.enterScope(0);this.type!==h.braceR;){var M=this.parseStatement(null);u.body.push(M)}return S&&(this.strict=!1),this.next(),r&&this.exitScope(),this.finishNode(u,"BlockStatement")},ne.parseFor=function(r,u){return r.init=u,this.expect(h.semi),r.test=this.type===h.semi?null:this.parseExpression(),this.expect(h.semi),r.update=this.type===h.parenR?null:this.parseExpression(),this.expect(h.parenR),r.body=this.parseStatement("for"),this.exitScope(),this.labels.pop(),this.finishNode(r,"ForStatement")},ne.parseForIn=function(r,u){var S=this.type===h._in;return this.next(),u.type==="VariableDeclaration"&&u.declarations[0].init!=null&&(!S||this.options.ecmaVersion<8||this.strict||u.kind!=="var"||u.declarations[0].id.type!=="Identifier")&&this.raise(u.start,(S?"for-in":"for-of")+" loop variable declaration may not have an initializer"),r.left=u,r.right=S?this.parseExpression():this.parseMaybeAssign(),this.expect(h.parenR),r.body=this.parseStatement("for"),this.exitScope(),this.labels.pop(),this.finishNode(r,S?"ForInStatement":"ForOfStatement")},ne.parseVar=function(r,u,S,M){for(r.declarations=[],r.kind=S;;){var R=this.startNode();if(this.parseVarId(R,S),this.eat(h.eq)?R.init=this.parseMaybeAssign(u):!M&&S==="const"&&!(this.type===h._in||this.options.ecmaVersion>=6&&this.isContextual("of"))?this.unexpected():!M&&R.id.type!=="Identifier"&&!(u&&(this.type===h._in||this.isContextual("of")))?this.raise(this.lastTokEnd,"Complex binding patterns require an initialization value"):R.init=null,r.declarations.push(this.finishNode(R,"VariableDeclarator")),!this.eat(h.comma))break}return r},ne.parseVarId=function(r,u){r.id=this.parseBindingAtom(),this.checkLValPattern(r.id,u==="var"?$n:mt,!1)};var fs=1,Dn=2,di=4;ne.parseFunction=function(r,u,S,M,R){this.initFunction(r),(this.options.ecmaVersion>=9||this.options.ecmaVersion>=6&&!M)&&(this.type===h.star&&u&Dn&&this.unexpected(),r.generator=this.eat(h.star)),this.options.ecmaVersion>=8&&(r.async=!!M),u&fs&&(r.id=u&di&&this.type!==h.name?null:this.parseIdent(),r.id&&!(u&Dn)&&this.checkLValSimple(r.id,this.strict||r.generator||r.async?this.treatFunctionsAsVar?$n:mt:ui));var K=this.yieldPos,B=this.awaitPos,J=this.awaitIdentPos;return this.yieldPos=0,this.awaitPos=0,this.awaitIdentPos=0,this.enterScope(Mn(r.async,r.generator)),u&fs||(r.id=this.type===h.name?this.parseIdent():null),this.parseFunctionParams(r),this.parseFunctionBody(r,S,!1,R),this.yieldPos=K,this.awaitPos=B,this.awaitIdentPos=J,this.finishNode(r,u&fs?"FunctionDeclaration":"FunctionExpression")},ne.parseFunctionParams=function(r){this.expect(h.parenL),r.params=this.parseBindingList(h.parenR,!1,this.options.ecmaVersion>=8),this.checkYieldAwaitInDefaultParams()},ne.parseClass=function(r,u){this.next();var S=this.strict;this.strict=!0,this.parseClassId(r,u),this.parseClassSuper(r);var M=this.enterClassBody(),R=this.startNode(),K=!1;for(R.body=[],this.expect(h.braceL);this.type!==h.braceR;){var B=this.parseClassElement(r.superClass!==null);B&&(R.body.push(B),B.type==="MethodDefinition"&&B.kind==="constructor"?(K&&this.raiseRecoverable(B.start,"Duplicate constructor in the same class"),K=!0):B.key&&B.key.type==="PrivateIdentifier"&&Yo(M,B)&&this.raiseRecoverable(B.key.start,"Identifier '#"+B.key.name+"' has already been declared"))}return this.strict=S,this.next(),r.body=this.finishNode(R,"ClassBody"),this.exitClassBody(),this.finishNode(r,u?"ClassDeclaration":"ClassExpression")},ne.parseClassElement=function(r){if(this.eat(h.semi))return null;var u=this.options.ecmaVersion,S=this.startNode(),M="",R=!1,K=!1,B="method",J=!1;if(this.eatContextual("static")){if(u>=13&&this.eat(h.braceL))return this.parseClassStaticBlock(S),S;this.isClassElementNameStart()||this.type===h.star?J=!0:M="static"}if(S.static=J,!M&&u>=8&&this.eatContextual("async")&&((this.isClassElementNameStart()||this.type===h.star)&&!this.canInsertSemicolon()?K=!0:M="async"),!M&&(u>=9||!K)&&this.eat(h.star)&&(R=!0),!M&&!K&&!R){var Q=this.value;(this.eatContextual("get")||this.eatContextual("set"))&&(this.isClassElementNameStart()?B=Q:M=Q)}if(M?(S.computed=!1,S.key=this.startNodeAt(this.lastTokStart,this.lastTokStartLoc),S.key.name=M,this.finishNode(S.key,"Identifier")):this.parseClassElementName(S),u<13||this.type===h.parenL||B!=="method"||R||K){var de=!S.static&&Us(S,"constructor"),le=de&&r;de&&B!=="method"&&this.raise(S.key.start,"Constructor can't have get/set modifier"),S.kind=de?"constructor":B,this.parseClassMethod(S,R,K,le)}else this.parseClassField(S);return S},ne.isClassElementNameStart=function(){return this.type===h.name||this.type===h.privateId||this.type===h.num||this.type===h.string||this.type===h.bracketL||this.type.keyword},ne.parseClassElementName=function(r){this.type===h.privateId?(this.value==="constructor"&&this.raise(this.start,"Classes can't have an element named '#constructor'"),r.computed=!1,r.key=this.parsePrivateIdent()):this.parsePropertyName(r)},ne.parseClassMethod=function(r,u,S,M){var R=r.key;r.kind==="constructor"?(u&&this.raise(R.start,"Constructor can't be a generator"),S&&this.raise(R.start,"Constructor can't be an async method")):r.static&&Us(r,"prototype")&&this.raise(R.start,"Classes may not have a static property named prototype");var K=r.value=this.parseMethod(u,S,M);return r.kind==="get"&&K.params.length!==0&&this.raiseRecoverable(K.start,"getter should have no params"),r.kind==="set"&&K.params.length!==1&&this.raiseRecoverable(K.start,"setter should have exactly one param"),r.kind==="set"&&K.params[0].type==="RestElement"&&this.raiseRecoverable(K.params[0].start,"Setter cannot use rest params"),this.finishNode(r,"MethodDefinition")},ne.parseClassField=function(r){if(Us(r,"constructor")?this.raise(r.key.start,"Classes can't have a field named 'constructor'"):r.static&&Us(r,"prototype")&&this.raise(r.key.start,"Classes can't have a static field named 'prototype'"),this.eat(h.eq)){var u=this.currentThisScope(),S=u.inClassFieldInit;u.inClassFieldInit=!0,r.value=this.parseMaybeAssign(),u.inClassFieldInit=S}else r.value=null;return this.semicolon(),this.finishNode(r,"PropertyDefinition")},ne.parseClassStaticBlock=function(r){r.body=[];var u=this.labels;for(this.labels=[],this.enterScope(ps|En);this.type!==h.braceR;){var S=this.parseStatement(null);r.body.push(S)}return this.next(),this.exitScope(),this.labels=u,this.finishNode(r,"StaticBlock")},ne.parseClassId=function(r,u){this.type===h.name?(r.id=this.parseIdent(),u&&this.checkLValSimple(r.id,mt,!1)):(u===!0&&this.unexpected(),r.id=null)},ne.parseClassSuper=function(r){r.superClass=this.eat(h._extends)?this.parseExprSubscripts(null,!1):null},ne.enterClassBody=function(){var r={declared:Object.create(null),used:[]};return this.privateNameStack.push(r),r.declared},ne.exitClassBody=function(){var r=this.privateNameStack.pop(),u=r.declared,S=r.used;if(this.options.checkPrivateFields)for(var M=this.privateNameStack.length,R=M===0?null:this.privateNameStack[M-1],K=0;K<S.length;++K){var B=S[K];ee(u,B.name)||(R?R.used.push(B):this.raiseRecoverable(B.start,"Private field '#"+B.name+"' must be declared in an enclosing class"))}};function Yo(r,u){var S=u.key.name,M=r[S],R="true";return u.type==="MethodDefinition"&&(u.kind==="get"||u.kind==="set")&&(R=(u.static?"s":"i")+u.kind),M==="iget"&&R==="iset"||M==="iset"&&R==="iget"||M==="sget"&&R==="sset"||M==="sset"&&R==="sget"?(r[S]="true",!1):M?!0:(r[S]=R,!1)}function Us(r,u){var S=r.computed,M=r.key;return!S&&(M.type==="Identifier"&&M.name===u||M.type==="Literal"&&M.value===u)}ne.parseExportAllDeclaration=function(r,u){return this.options.ecmaVersion>=11&&(this.eatContextual("as")?(r.exported=this.parseModuleExportName(),this.checkExport(u,r.exported,this.lastTokStart)):r.exported=null),this.expectContextual("from"),this.type!==h.string&&this.unexpected(),r.source=this.parseExprAtom(),this.options.ecmaVersion>=16&&(r.attributes=this.parseWithClause()),this.semicolon(),this.finishNode(r,"ExportAllDeclaration")},ne.parseExport=function(r,u){if(this.next(),this.eat(h.star))return this.parseExportAllDeclaration(r,u);if(this.eat(h._default))return this.checkExport(u,"default",this.lastTokStart),r.declaration=this.parseExportDefaultDeclaration(),this.finishNode(r,"ExportDefaultDeclaration");if(this.shouldParseExportStatement())r.declaration=this.parseExportDeclaration(r),r.declaration.type==="VariableDeclaration"?this.checkVariableExport(u,r.declaration.declarations):this.checkExport(u,r.declaration.id,r.declaration.id.start),r.specifiers=[],r.source=null;else{if(r.declaration=null,r.specifiers=this.parseExportSpecifiers(u),this.eatContextual("from"))this.type!==h.string&&this.unexpected(),r.source=this.parseExprAtom(),this.options.ecmaVersion>=16&&(r.attributes=this.parseWithClause());else{for(var S=0,M=r.specifiers;S<M.length;S+=1){var R=M[S];this.checkUnreserved(R.local),this.checkLocalExport(R.local),R.local.type==="Literal"&&this.raise(R.local.start,"A string literal cannot be used as an exported binding without `from`.")}r.source=null}this.semicolon()}return this.finishNode(r,"ExportNamedDeclaration")},ne.parseExportDeclaration=function(r){return this.parseStatement(null)},ne.parseExportDefaultDeclaration=function(){var r;if(this.type===h._function||(r=this.isAsyncFunction())){var u=this.startNode();return this.next(),r&&this.next(),this.parseFunction(u,fs|di,!1,r)}else if(this.type===h._class){var S=this.startNode();return this.parseClass(S,"nullableID")}else{var M=this.parseMaybeAssign();return this.semicolon(),M}},ne.checkExport=function(r,u,S){r&&(typeof u!="string"&&(u=u.type==="Identifier"?u.name:u.value),ee(r,u)&&this.raiseRecoverable(S,"Duplicate export '"+u+"'"),r[u]=!0)},ne.checkPatternExport=function(r,u){var S=u.type;if(S==="Identifier")this.checkExport(r,u,u.start);else if(S==="ObjectPattern")for(var M=0,R=u.properties;M<R.length;M+=1){var K=R[M];this.checkPatternExport(r,K)}else if(S==="ArrayPattern")for(var B=0,J=u.elements;B<J.length;B+=1){var Q=J[B];Q&&this.checkPatternExport(r,Q)}else S==="Property"?this.checkPatternExport(r,u.value):S==="AssignmentPattern"?this.checkPatternExport(r,u.left):S==="RestElement"&&this.checkPatternExport(r,u.argument)},ne.checkVariableExport=function(r,u){if(r)for(var S=0,M=u;S<M.length;S+=1){var R=M[S];this.checkPatternExport(r,R.id)}},ne.shouldParseExportStatement=function(){return this.type.keyword==="var"||this.type.keyword==="const"||this.type.keyword==="class"||this.type.keyword==="function"||this.isLet()||this.isAsyncFunction()},ne.parseExportSpecifier=function(r){var u=this.startNode();return u.local=this.parseModuleExportName(),u.exported=this.eatContextual("as")?this.parseModuleExportName():u.local,this.checkExport(r,u.exported,u.exported.start),this.finishNode(u,"ExportSpecifier")},ne.parseExportSpecifiers=function(r){var u=[],S=!0;for(this.expect(h.braceL);!this.eat(h.braceR);){if(S)S=!1;else if(this.expect(h.comma),this.afterTrailingComma(h.braceR))break;u.push(this.parseExportSpecifier(r))}return u},ne.parseImport=function(r){return this.next(),this.type===h.string?(r.specifiers=Xo,r.source=this.parseExprAtom()):(r.specifiers=this.parseImportSpecifiers(),this.expectContextual("from"),r.source=this.type===h.string?this.parseExprAtom():this.unexpected()),this.options.ecmaVersion>=16&&(r.attributes=this.parseWithClause()),this.semicolon(),this.finishNode(r,"ImportDeclaration")},ne.parseImportSpecifier=function(){var r=this.startNode();return r.imported=this.parseModuleExportName(),this.eatContextual("as")?r.local=this.parseIdent():(this.checkUnreserved(r.imported),r.local=r.imported),this.checkLValSimple(r.local,mt),this.finishNode(r,"ImportSpecifier")},ne.parseImportDefaultSpecifier=function(){var r=this.startNode();return r.local=this.parseIdent(),this.checkLValSimple(r.local,mt),this.finishNode(r,"ImportDefaultSpecifier")},ne.parseImportNamespaceSpecifier=function(){var r=this.startNode();return this.next(),this.expectContextual("as"),r.local=this.parseIdent(),this.checkLValSimple(r.local,mt),this.finishNode(r,"ImportNamespaceSpecifier")},ne.parseImportSpecifiers=function(){var r=[],u=!0;if(this.type===h.name&&(r.push(this.parseImportDefaultSpecifier()),!this.eat(h.comma)))return r;if(this.type===h.star)return r.push(this.parseImportNamespaceSpecifier()),r;for(this.expect(h.braceL);!this.eat(h.braceR);){if(u)u=!1;else if(this.expect(h.comma),this.afterTrailingComma(h.braceR))break;r.push(this.parseImportSpecifier())}return r},ne.parseWithClause=function(){var r=[];if(!this.eat(h._with))return r;this.expect(h.braceL);for(var u={},S=!0;!this.eat(h.braceR);){if(S)S=!1;else if(this.expect(h.comma),this.afterTrailingComma(h.braceR))break;var M=this.parseImportAttribute(),R=M.key.type==="Identifier"?M.key.name:M.key.value;ee(u,R)&&this.raiseRecoverable(M.key.start,"Duplicate attribute key '"+R+"'"),u[R]=!0,r.push(M)}return r},ne.parseImportAttribute=function(){var r=this.startNode();return r.key=this.type===h.string?this.parseExprAtom():this.parseIdent(this.options.allowReserved!=="never"),this.expect(h.colon),this.type!==h.string&&this.unexpected(),r.value=this.parseExprAtom(),this.finishNode(r,"ImportAttribute")},ne.parseModuleExportName=function(){if(this.options.ecmaVersion>=13&&this.type===h.string){var r=this.parseLiteral(this.value);return we.test(r.value)&&this.raise(r.start,"An export name cannot include a lone surrogate."),r}return this.parseIdent(!0)},ne.adaptDirectivePrologue=function(r){for(var u=0;u<r.length&&this.isDirectiveCandidate(r[u]);++u)r[u].directive=r[u].expression.raw.slice(1,-1)},ne.isDirectiveCandidate=function(r){return this.options.ecmaVersion>=5&&r.type==="ExpressionStatement"&&r.expression.type==="Literal"&&typeof r.expression.value=="string"&&(this.input[r.start]==='"'||this.input[r.start]==="'")};var Be=_e.prototype;Be.toAssignable=function(r,u,S){if(this.options.ecmaVersion>=6&&r)switch(r.type){case"Identifier":this.inAsync&&r.name==="await"&&this.raise(r.start,"Cannot use 'await' as identifier inside an async function");break;case"ObjectPattern":case"ArrayPattern":case"AssignmentPattern":case"RestElement":break;case"ObjectExpression":r.type="ObjectPattern",S&&this.checkPatternErrors(S,!0);for(var M=0,R=r.properties;M<R.length;M+=1){var K=R[M];this.toAssignable(K,u),K.type==="RestElement"&&(K.argument.type==="ArrayPattern"||K.argument.type==="ObjectPattern")&&this.raise(K.argument.start,"Unexpected token")}break;case"Property":r.kind!=="init"&&this.raise(r.key.start,"Object pattern can't contain getter or setter"),this.toAssignable(r.value,u);break;case"ArrayExpression":r.type="ArrayPattern",S&&this.checkPatternErrors(S,!0),this.toAssignableList(r.elements,u);break;case"SpreadElement":r.type="RestElement",this.toAssignable(r.argument,u),r.argument.type==="AssignmentPattern"&&this.raise(r.argument.start,"Rest elements cannot have a default value");break;case"AssignmentExpression":r.operator!=="="&&this.raise(r.left.end,"Only '=' operator can be used for specifying default value."),r.type="AssignmentPattern",delete r.operator,this.toAssignable(r.left,u);break;case"ParenthesizedExpression":this.toAssignable(r.expression,u,S);break;case"ChainExpression":this.raiseRecoverable(r.start,"Optional chaining cannot appear in left-hand side");break;case"MemberExpression":if(!u)break;default:this.raise(r.start,"Assigning to rvalue")}else S&&this.checkPatternErrors(S,!0);return r},Be.toAssignableList=function(r,u){for(var S=r.length,M=0;M<S;M++){var R=r[M];R&&this.toAssignable(R,u)}if(S){var K=r[S-1];this.options.ecmaVersion===6&&u&&K&&K.type==="RestElement"&&K.argument.type!=="Identifier"&&this.unexpected(K.argument.start)}return r},Be.parseSpread=function(r){var u=this.startNode();return this.next(),u.argument=this.parseMaybeAssign(!1,r),this.finishNode(u,"SpreadElement")},Be.parseRestBinding=function(){var r=this.startNode();return this.next(),this.options.ecmaVersion===6&&this.type!==h.name&&this.unexpected(),r.argument=this.parseBindingAtom(),this.finishNode(r,"RestElement")},Be.parseBindingAtom=function(){if(this.options.ecmaVersion>=6)switch(this.type){case h.bracketL:var r=this.startNode();return this.next(),r.elements=this.parseBindingList(h.bracketR,!0,!0),this.finishNode(r,"ArrayPattern");case h.braceL:return this.parseObj(!0)}return this.parseIdent()},Be.parseBindingList=function(r,u,S,M){for(var R=[],K=!0;!this.eat(r);)if(K?K=!1:this.expect(h.comma),u&&this.type===h.comma)R.push(null);else{if(S&&this.afterTrailingComma(r))break;if(this.type===h.ellipsis){var B=this.parseRestBinding();this.parseBindingListItem(B),R.push(B),this.type===h.comma&&this.raiseRecoverable(this.start,"Comma is not permitted after the rest element"),this.expect(r);break}else R.push(this.parseAssignableListItem(M))}return R},Be.parseAssignableListItem=function(r){var u=this.parseMaybeDefault(this.start,this.startLoc);return this.parseBindingListItem(u),u},Be.parseBindingListItem=function(r){return r},Be.parseMaybeDefault=function(r,u,S){if(S=S||this.parseBindingAtom(),this.options.ecmaVersion<6||!this.eat(h.eq))return S;var M=this.startNodeAt(r,u);return M.left=S,M.right=this.parseMaybeAssign(),this.finishNode(M,"AssignmentPattern")},Be.checkLValSimple=function(r,u,S){u===void 0&&(u=Gs);var M=u!==Gs;switch(r.type){case"Identifier":this.strict&&this.reservedWordsStrictBind.test(r.name)&&this.raiseRecoverable(r.start,(M?"Binding ":"Assigning to ")+r.name+" in strict mode"),M&&(u===mt&&r.name==="let"&&this.raiseRecoverable(r.start,"let is disallowed as a lexically bound name"),S&&(ee(S,r.name)&&this.raiseRecoverable(r.start,"Argument name clash"),S[r.name]=!0),u!==hi&&this.declareName(r.name,u,r.start));break;case"ChainExpression":this.raiseRecoverable(r.start,"Optional chaining cannot appear in left-hand side");break;case"MemberExpression":M&&this.raiseRecoverable(r.start,"Binding member expression");break;case"ParenthesizedExpression":return M&&this.raiseRecoverable(r.start,"Binding parenthesized expression"),this.checkLValSimple(r.expression,u,S);default:this.raise(r.start,(M?"Binding":"Assigning to")+" rvalue")}},Be.checkLValPattern=function(r,u,S){switch(u===void 0&&(u=Gs),r.type){case"ObjectPattern":for(var M=0,R=r.properties;M<R.length;M+=1){var K=R[M];this.checkLValInnerPattern(K,u,S)}break;case"ArrayPattern":for(var B=0,J=r.elements;B<J.length;B+=1){var Q=J[B];Q&&this.checkLValInnerPattern(Q,u,S)}break;default:this.checkLValSimple(r,u,S)}},Be.checkLValInnerPattern=function(r,u,S){switch(u===void 0&&(u=Gs),r.type){case"Property":this.checkLValInnerPattern(r.value,u,S);break;case"AssignmentPattern":this.checkLValPattern(r.left,u,S);break;case"RestElement":this.checkLValPattern(r.argument,u,S);break;default:this.checkLValPattern(r,u,S)}};var je=function(u,S,M,R,K){this.token=u,this.isExpr=!!S,this.preserveSpace=!!M,this.override=R,this.generator=!!K},be={b_stat:new je("{",!1),b_expr:new je("{",!0),b_tmpl:new je("${",!1),p_stat:new je("(",!1),p_expr:new je("(",!0),q_tmpl:new je("`",!0,!0,function(r){return r.tryReadTemplateToken()}),f_stat:new je("function",!1),f_expr:new je("function",!0),f_expr_gen:new je("function",!0,!1,null,!0),f_gen:new je("function",!1,!1,null,!0)},qt=_e.prototype;qt.initialContext=function(){return[be.b_stat]},qt.curContext=function(){return this.context[this.context.length-1]},qt.braceIsBlock=function(r){var u=this.curContext();return u===be.f_expr||u===be.f_stat?!0:r===h.colon&&(u===be.b_stat||u===be.b_expr)?!u.isExpr:r===h._return||r===h.name&&this.exprAllowed?F.test(this.input.slice(this.lastTokEnd,this.start)):r===h._else||r===h.semi||r===h.eof||r===h.parenR||r===h.arrow?!0:r===h.braceL?u===be.b_stat:r===h._var||r===h._const||r===h.name?!1:!this.exprAllowed},qt.inGeneratorContext=function(){for(var r=this.context.length-1;r>=1;r--){var u=this.context[r];if(u.token==="function")return u.generator}return!1},qt.updateContext=function(r){var u,S=this.type;S.keyword&&r===h.dot?this.exprAllowed=!1:(u=S.updateContext)?u.call(this,r):this.exprAllowed=S.beforeExpr},qt.overrideContext=function(r){this.curContext()!==r&&(this.context[this.context.length-1]=r)},h.parenR.updateContext=h.braceR.updateContext=function(){if(this.context.length===1){this.exprAllowed=!0;return}var r=this.context.pop();r===be.b_stat&&this.curContext().token==="function"&&(r=this.context.pop()),this.exprAllowed=!r.isExpr},h.braceL.updateContext=function(r){this.context.push(this.braceIsBlock(r)?be.b_stat:be.b_expr),this.exprAllowed=!0},h.dollarBraceL.updateContext=function(){this.context.push(be.b_tmpl),this.exprAllowed=!0},h.parenL.updateContext=function(r){var u=r===h._if||r===h._for||r===h._with||r===h._while;this.context.push(u?be.p_stat:be.p_expr),this.exprAllowed=!0},h.incDec.updateContext=function(){},h._function.updateContext=h._class.updateContext=function(r){r.beforeExpr&&r!==h._else&&!(r===h.semi&&this.curContext()!==be.p_stat)&&!(r===h._return&&F.test(this.input.slice(this.lastTokEnd,this.start)))&&!((r===h.colon||r===h.braceL)&&this.curContext()===be.b_stat)?this.context.push(be.f_expr):this.context.push(be.f_stat),this.exprAllowed=!1},h.colon.updateContext=function(){this.curContext().token==="function"&&this.context.pop(),this.exprAllowed=!0},h.backQuote.updateContext=function(){this.curContext()===be.q_tmpl?this.context.pop():this.context.push(be.q_tmpl),this.exprAllowed=!1},h.star.updateContext=function(r){if(r===h._function){var u=this.context.length-1;this.context[u]===be.f_expr?this.context[u]=be.f_expr_gen:this.context[u]=be.f_gen}this.exprAllowed=!0},h.name.updateContext=function(r){var u=!1;this.options.ecmaVersion>=6&&r!==h.dot&&(this.value==="of"&&!this.exprAllowed||this.value==="yield"&&this.inGeneratorContext())&&(u=!0),this.exprAllowed=u};var oe=_e.prototype;oe.checkPropClash=function(r,u,S){if(!(this.options.ecmaVersion>=9&&r.type==="SpreadElement")&&!(this.options.ecmaVersion>=6&&(r.computed||r.method||r.shorthand))){var M=r.key,R;switch(M.type){case"Identifier":R=M.name;break;case"Literal":R=String(M.value);break;default:return}var K=r.kind;if(this.options.ecmaVersion>=6){R==="__proto__"&&K==="init"&&(u.proto&&(S?S.doubleProto<0&&(S.doubleProto=M.start):this.raiseRecoverable(M.start,"Redefinition of __proto__ property")),u.proto=!0);return}R="$"+R;var B=u[R];if(B){var J;K==="init"?J=this.strict&&B.init||B.get||B.set:J=B.init||B[K],J&&this.raiseRecoverable(M.start,"Redefinition of property")}else B=u[R]={init:!1,get:!1,set:!1};B[K]=!0}},oe.parseExpression=function(r,u){var S=this.start,M=this.startLoc,R=this.parseMaybeAssign(r,u);if(this.type===h.comma){var K=this.startNodeAt(S,M);for(K.expressions=[R];this.eat(h.comma);)K.expressions.push(this.parseMaybeAssign(r,u));return this.finishNode(K,"SequenceExpression")}return R},oe.parseMaybeAssign=function(r,u,S){if(this.isContextual("yield")){if(this.inGenerator)return this.parseYield(r);this.exprAllowed=!1}var M=!1,R=-1,K=-1,B=-1;u?(R=u.parenthesizedAssign,K=u.trailingComma,B=u.doubleProto,u.parenthesizedAssign=u.trailingComma=-1):(u=new Ls,M=!0);var J=this.start,Q=this.startLoc;(this.type===h.parenL||this.type===h.name)&&(this.potentialArrowAt=this.start,this.potentialArrowInForAwait=r==="await");var de=this.parseMaybeConditional(r,u);if(S&&(de=S.call(this,de,J,Q)),this.type.isAssign){var le=this.startNodeAt(J,Q);return le.operator=this.value,this.type===h.eq&&(de=this.toAssignable(de,!1,u)),M||(u.parenthesizedAssign=u.trailingComma=u.doubleProto=-1),u.shorthandAssign>=de.start&&(u.shorthandAssign=-1),this.type===h.eq?this.checkLValPattern(de):this.checkLValSimple(de),le.left=de,this.next(),le.right=this.parseMaybeAssign(r),B>-1&&(u.doubleProto=B),this.finishNode(le,"AssignmentExpression")}else M&&this.checkExpressionErrors(u,!0);return R>-1&&(u.parenthesizedAssign=R),K>-1&&(u.trailingComma=K),de},oe.parseMaybeConditional=function(r,u){var S=this.start,M=this.startLoc,R=this.parseExprOps(r,u);if(this.checkExpressionErrors(u))return R;if(this.eat(h.question)){var K=this.startNodeAt(S,M);return K.test=R,K.consequent=this.parseMaybeAssign(),this.expect(h.colon),K.alternate=this.parseMaybeAssign(r),this.finishNode(K,"ConditionalExpression")}return R},oe.parseExprOps=function(r,u){var S=this.start,M=this.startLoc,R=this.parseMaybeUnary(u,!1,!1,r);return this.checkExpressionErrors(u)||R.start===S&&R.type==="ArrowFunctionExpression"?R:this.parseExprOp(R,S,M,-1,r)},oe.parseExprOp=function(r,u,S,M,R){var K=this.type.binop;if(K!=null&&(!R||this.type!==h._in)&&K>M){var B=this.type===h.logicalOR||this.type===h.logicalAND,J=this.type===h.coalesce;J&&(K=h.logicalAND.binop);var Q=this.value;this.next();var de=this.start,le=this.startLoc,$e=this.parseExprOp(this.parseMaybeUnary(null,!1,!1,R),de,le,K,R),zt=this.buildBinary(u,S,r,$e,Q,B||J);return(B&&this.type===h.coalesce||J&&(this.type===h.logicalOR||this.type===h.logicalAND))&&this.raiseRecoverable(this.start,"Logical expressions and coalesce expressions cannot be mixed. Wrap either by parentheses"),this.parseExprOp(zt,u,S,M,R)}return r},oe.buildBinary=function(r,u,S,M,R,K){M.type==="PrivateIdentifier"&&this.raise(M.start,"Private identifier can only be left side of binary expression");var B=this.startNodeAt(r,u);return B.left=S,B.operator=R,B.right=M,this.finishNode(B,K?"LogicalExpression":"BinaryExpression")},oe.parseMaybeUnary=function(r,u,S,M){var R=this.start,K=this.startLoc,B;if(this.isContextual("await")&&this.canAwait)B=this.parseAwait(M),u=!0;else if(this.type.prefix){var J=this.startNode(),Q=this.type===h.incDec;J.operator=this.value,J.prefix=!0,this.next(),J.argument=this.parseMaybeUnary(null,!0,Q,M),this.checkExpressionErrors(r,!0),Q?this.checkLValSimple(J.argument):this.strict&&J.operator==="delete"&&pi(J.argument)?this.raiseRecoverable(J.start,"Deleting local variable in strict mode"):J.operator==="delete"&&Pn(J.argument)?this.raiseRecoverable(J.start,"Private fields can not be deleted"):u=!0,B=this.finishNode(J,Q?"UpdateExpression":"UnaryExpression")}else if(!u&&this.type===h.privateId)(M||this.privateNameStack.length===0)&&this.options.checkPrivateFields&&this.unexpected(),B=this.parsePrivateIdent(),this.type!==h._in&&this.unexpected();else{if(B=this.parseExprSubscripts(r,M),this.checkExpressionErrors(r))return B;for(;this.type.postfix&&!this.canInsertSemicolon();){var de=this.startNodeAt(R,K);de.operator=this.value,de.prefix=!1,de.argument=B,this.checkLValSimple(B),this.next(),B=this.finishNode(de,"UpdateExpression")}}if(!S&&this.eat(h.starstar))if(u)this.unexpected(this.lastTokStart);else return this.buildBinary(R,K,B,this.parseMaybeUnary(null,!1,!1,M),"**",!1);else return B};function pi(r){return r.type==="Identifier"||r.type==="ParenthesizedExpression"&&pi(r.expression)}function Pn(r){return r.type==="MemberExpression"&&r.property.type==="PrivateIdentifier"||r.type==="ChainExpression"&&Pn(r.expression)||r.type==="ParenthesizedExpression"&&Pn(r.expression)}oe.parseExprSubscripts=function(r,u){var S=this.start,M=this.startLoc,R=this.parseExprAtom(r,u);if(R.type==="ArrowFunctionExpression"&&this.input.slice(this.lastTokStart,this.lastTokEnd)!==")")return R;var K=this.parseSubscripts(R,S,M,!1,u);return r&&K.type==="MemberExpression"&&(r.parenthesizedAssign>=K.start&&(r.parenthesizedAssign=-1),r.parenthesizedBind>=K.start&&(r.parenthesizedBind=-1),r.trailingComma>=K.start&&(r.trailingComma=-1)),K},oe.parseSubscripts=function(r,u,S,M,R){for(var K=this.options.ecmaVersion>=8&&r.type==="Identifier"&&r.name==="async"&&this.lastTokEnd===r.end&&!this.canInsertSemicolon()&&r.end-r.start===5&&this.potentialArrowAt===r.start,B=!1;;){var J=this.parseSubscript(r,u,S,M,K,B,R);if(J.optional&&(B=!0),J===r||J.type==="ArrowFunctionExpression"){if(B){var Q=this.startNodeAt(u,S);Q.expression=J,J=this.finishNode(Q,"ChainExpression")}return J}r=J}},oe.shouldParseAsyncArrow=function(){return!this.canInsertSemicolon()&&this.eat(h.arrow)},oe.parseSubscriptAsyncArrow=function(r,u,S,M){return this.parseArrowExpression(this.startNodeAt(r,u),S,!0,M)},oe.parseSubscript=function(r,u,S,M,R,K,B){var J=this.options.ecmaVersion>=11,Q=J&&this.eat(h.questionDot);M&&Q&&this.raise(this.lastTokStart,"Optional chaining cannot appear in the callee of new expressions");var de=this.eat(h.bracketL);if(de||Q&&this.type!==h.parenL&&this.type!==h.backQuote||this.eat(h.dot)){var le=this.startNodeAt(u,S);le.object=r,de?(le.property=this.parseExpression(),this.expect(h.bracketR)):this.type===h.privateId&&r.type!=="Super"?le.property=this.parsePrivateIdent():le.property=this.parseIdent(this.options.allowReserved!=="never"),le.computed=!!de,J&&(le.optional=Q),r=this.finishNode(le,"MemberExpression")}else if(!M&&this.eat(h.parenL)){var $e=new Ls,zt=this.yieldPos,ys=this.awaitPos,Wt=this.awaitIdentPos;this.yieldPos=0,this.awaitPos=0,this.awaitIdentPos=0;var js=this.parseExprList(h.parenR,this.options.ecmaVersion>=8,!1,$e);if(R&&!Q&&this.shouldParseAsyncArrow())return this.checkPatternErrors($e,!1),this.checkYieldAwaitInDefaultParams(),this.awaitIdentPos>0&&this.raise(this.awaitIdentPos,"Cannot use 'await' as identifier inside an async function"),this.yieldPos=zt,this.awaitPos=ys,this.awaitIdentPos=Wt,this.parseSubscriptAsyncArrow(u,S,js,B);this.checkExpressionErrors($e,!0),this.yieldPos=zt||this.yieldPos,this.awaitPos=ys||this.awaitPos,this.awaitIdentPos=Wt||this.awaitIdentPos;var Ht=this.startNodeAt(u,S);Ht.callee=r,Ht.arguments=js,J&&(Ht.optional=Q),r=this.finishNode(Ht,"CallExpression")}else if(this.type===h.backQuote){(Q||K)&&this.raise(this.start,"Optional chaining cannot appear in the tag of tagged template expressions");var Xt=this.startNodeAt(u,S);Xt.tag=r,Xt.quasi=this.parseTemplate({isTagged:!0}),r=this.finishNode(Xt,"TaggedTemplateExpression")}return r},oe.parseExprAtom=function(r,u,S){this.type===h.slash&&this.readRegexp();var M,R=this.potentialArrowAt===this.start;switch(this.type){case h._super:return this.allowSuper||this.raise(this.start,"'super' keyword outside a method"),M=this.startNode(),this.next(),this.type===h.parenL&&!this.allowDirectSuper&&this.raise(M.start,"super() call outside constructor of a subclass"),this.type!==h.dot&&this.type!==h.bracketL&&this.type!==h.parenL&&this.unexpected(),this.finishNode(M,"Super");case h._this:return M=this.startNode(),this.next(),this.finishNode(M,"ThisExpression");case h.name:var K=this.start,B=this.startLoc,J=this.containsEsc,Q=this.parseIdent(!1);if(this.options.ecmaVersion>=8&&!J&&Q.name==="async"&&!this.canInsertSemicolon()&&this.eat(h._function))return this.overrideContext(be.f_expr),this.parseFunction(this.startNodeAt(K,B),0,!1,!0,u);if(R&&!this.canInsertSemicolon()){if(this.eat(h.arrow))return this.parseArrowExpression(this.startNodeAt(K,B),[Q],!1,u);if(this.options.ecmaVersion>=8&&Q.name==="async"&&this.type===h.name&&!J&&(!this.potentialArrowInForAwait||this.value!=="of"||this.containsEsc))return Q=this.parseIdent(!1),(this.canInsertSemicolon()||!this.eat(h.arrow))&&this.unexpected(),this.parseArrowExpression(this.startNodeAt(K,B),[Q],!0,u)}return Q;case h.regexp:var de=this.value;return M=this.parseLiteral(de.value),M.regex={pattern:de.pattern,flags:de.flags},M;case h.num:case h.string:return this.parseLiteral(this.value);case h._null:case h._true:case h._false:return M=this.startNode(),M.value=this.type===h._null?null:this.type===h._true,M.raw=this.type.keyword,this.next(),this.finishNode(M,"Literal");case h.parenL:var le=this.start,$e=this.parseParenAndDistinguishExpression(R,u);return r&&(r.parenthesizedAssign<0&&!this.isSimpleAssignTarget($e)&&(r.parenthesizedAssign=le),r.parenthesizedBind<0&&(r.parenthesizedBind=le)),$e;case h.bracketL:return M=this.startNode(),this.next(),M.elements=this.parseExprList(h.bracketR,!0,!0,r),this.finishNode(M,"ArrayExpression");case h.braceL:return this.overrideContext(be.b_expr),this.parseObj(!1,r);case h._function:return M=this.startNode(),this.next(),this.parseFunction(M,0);case h._class:return this.parseClass(this.startNode(),!1);case h._new:return this.parseNew();case h.backQuote:return this.parseTemplate();case h._import:return this.options.ecmaVersion>=11?this.parseExprImport(S):this.unexpected();default:return this.parseExprAtomDefault()}},oe.parseExprAtomDefault=function(){this.unexpected()},oe.parseExprImport=function(r){var u=this.startNode();if(this.containsEsc&&this.raiseRecoverable(this.start,"Escape sequence in keyword import"),this.next(),this.type===h.parenL&&!r)return this.parseDynamicImport(u);if(this.type===h.dot){var S=this.startNodeAt(u.start,u.loc&&u.loc.start);return S.name="import",u.meta=this.finishNode(S,"Identifier"),this.parseImportMeta(u)}else this.unexpected()},oe.parseDynamicImport=function(r){if(this.next(),r.source=this.parseMaybeAssign(),this.options.ecmaVersion>=16)this.eat(h.parenR)?r.options=null:(this.expect(h.comma),this.afterTrailingComma(h.parenR)?r.options=null:(r.options=this.parseMaybeAssign(),this.eat(h.parenR)||(this.expect(h.comma),this.afterTrailingComma(h.parenR)||this.unexpected())));else if(!this.eat(h.parenR)){var u=this.start;this.eat(h.comma)&&this.eat(h.parenR)?this.raiseRecoverable(u,"Trailing comma is not allowed in import()"):this.unexpected(u)}return this.finishNode(r,"ImportExpression")},oe.parseImportMeta=function(r){this.next();var u=this.containsEsc;return r.property=this.parseIdent(!0),r.property.name!=="meta"&&this.raiseRecoverable(r.property.start,"The only valid meta property for import is 'import.meta'"),u&&this.raiseRecoverable(r.start,"'import.meta' must not contain escaped characters"),this.options.sourceType!=="module"&&!this.options.allowImportExportEverywhere&&this.raiseRecoverable(r.start,"Cannot use 'import.meta' outside a module"),this.finishNode(r,"MetaProperty")},oe.parseLiteral=function(r){var u=this.startNode();return u.value=r,u.raw=this.input.slice(this.start,this.end),u.raw.charCodeAt(u.raw.length-1)===110&&(u.bigint=u.raw.slice(0,-1).replace(/_/g,"")),this.next(),this.finishNode(u,"Literal")},oe.parseParenExpression=function(){this.expect(h.parenL);var r=this.parseExpression();return this.expect(h.parenR),r},oe.shouldParseArrow=function(r){return!this.canInsertSemicolon()},oe.parseParenAndDistinguishExpression=function(r,u){var S=this.start,M=this.startLoc,R,K=this.options.ecmaVersion>=8;if(this.options.ecmaVersion>=6){this.next();var B=this.start,J=this.startLoc,Q=[],de=!0,le=!1,$e=new Ls,zt=this.yieldPos,ys=this.awaitPos,Wt;for(this.yieldPos=0,this.awaitPos=0;this.type!==h.parenR;)if(de?de=!1:this.expect(h.comma),K&&this.afterTrailingComma(h.parenR,!0)){le=!0;break}else if(this.type===h.ellipsis){Wt=this.start,Q.push(this.parseParenItem(this.parseRestBinding())),this.type===h.comma&&this.raiseRecoverable(this.start,"Comma is not permitted after the rest element");break}else Q.push(this.parseMaybeAssign(!1,$e,this.parseParenItem));var js=this.lastTokEnd,Ht=this.lastTokEndLoc;if(this.expect(h.parenR),r&&this.shouldParseArrow(Q)&&this.eat(h.arrow))return this.checkPatternErrors($e,!1),this.checkYieldAwaitInDefaultParams(),this.yieldPos=zt,this.awaitPos=ys,this.parseParenArrowList(S,M,Q,u);(!Q.length||le)&&this.unexpected(this.lastTokStart),Wt&&this.unexpected(Wt),this.checkExpressionErrors($e,!0),this.yieldPos=zt||this.yieldPos,this.awaitPos=ys||this.awaitPos,Q.length>1?(R=this.startNodeAt(B,J),R.expressions=Q,this.finishNodeAt(R,"SequenceExpression",js,Ht)):R=Q[0]}else R=this.parseParenExpression();if(this.options.preserveParens){var Xt=this.startNodeAt(S,M);return Xt.expression=R,this.finishNode(Xt,"ParenthesizedExpression")}else return R},oe.parseParenItem=function(r){return r},oe.parseParenArrowList=function(r,u,S,M){return this.parseArrowExpression(this.startNodeAt(r,u),S,!1,M)};var Jo=[];oe.parseNew=function(){this.containsEsc&&this.raiseRecoverable(this.start,"Escape sequence in keyword new");var r=this.startNode();if(this.next(),this.options.ecmaVersion>=6&&this.type===h.dot){var u=this.startNodeAt(r.start,r.loc&&r.loc.start);u.name="new",r.meta=this.finishNode(u,"Identifier"),this.next();var S=this.containsEsc;return r.property=this.parseIdent(!0),r.property.name!=="target"&&this.raiseRecoverable(r.property.start,"The only valid meta property for new is 'new.target'"),S&&this.raiseRecoverable(r.start,"'new.target' must not contain escaped characters"),this.allowNewDotTarget||this.raiseRecoverable(r.start,"'new.target' can only be used in functions and class static block"),this.finishNode(r,"MetaProperty")}var M=this.start,R=this.startLoc;return r.callee=this.parseSubscripts(this.parseExprAtom(null,!1,!0),M,R,!0,!1),this.eat(h.parenL)?r.arguments=this.parseExprList(h.parenR,this.options.ecmaVersion>=8,!1):r.arguments=Jo,this.finishNode(r,"NewExpression")},oe.parseTemplateElement=function(r){var u=r.isTagged,S=this.startNode();return this.type===h.invalidTemplate?(u||this.raiseRecoverable(this.start,"Bad escape sequence in untagged template literal"),S.value={raw:this.value.replace(/\r\n?/g,`
`),cooked:null}):S.value={raw:this.input.slice(this.start,this.end).replace(/\r\n?/g,`
`),cooked:this.value},this.next(),S.tail=this.type===h.backQuote,this.finishNode(S,"TemplateElement")},oe.parseTemplate=function(r){r===void 0&&(r={});var u=r.isTagged;u===void 0&&(u=!1);var S=this.startNode();this.next(),S.expressions=[];var M=this.parseTemplateElement({isTagged:u});for(S.quasis=[M];!M.tail;)this.type===h.eof&&this.raise(this.pos,"Unterminated template literal"),this.expect(h.dollarBraceL),S.expressions.push(this.parseExpression()),this.expect(h.braceR),S.quasis.push(M=this.parseTemplateElement({isTagged:u}));return this.next(),this.finishNode(S,"TemplateLiteral")},oe.isAsyncProp=function(r){return!r.computed&&r.key.type==="Identifier"&&r.key.name==="async"&&(this.type===h.name||this.type===h.num||this.type===h.string||this.type===h.bracketL||this.type.keyword||this.options.ecmaVersion>=9&&this.type===h.star)&&!F.test(this.input.slice(this.lastTokEnd,this.start))},oe.parseObj=function(r,u){var S=this.startNode(),M=!0,R={};for(S.properties=[],this.next();!this.eat(h.braceR);){if(M)M=!1;else if(this.expect(h.comma),this.options.ecmaVersion>=5&&this.afterTrailingComma(h.braceR))break;var K=this.parseProperty(r,u);r||this.checkPropClash(K,R,u),S.properties.push(K)}return this.finishNode(S,r?"ObjectPattern":"ObjectExpression")},oe.parseProperty=function(r,u){var S=this.startNode(),M,R,K,B;if(this.options.ecmaVersion>=9&&this.eat(h.ellipsis))return r?(S.argument=this.parseIdent(!1),this.type===h.comma&&this.raiseRecoverable(this.start,"Comma is not permitted after the rest element"),this.finishNode(S,"RestElement")):(S.argument=this.parseMaybeAssign(!1,u),this.type===h.comma&&u&&u.trailingComma<0&&(u.trailingComma=this.start),this.finishNode(S,"SpreadElement"));this.options.ecmaVersion>=6&&(S.method=!1,S.shorthand=!1,(r||u)&&(K=this.start,B=this.startLoc),r||(M=this.eat(h.star)));var J=this.containsEsc;return this.parsePropertyName(S),!r&&!J&&this.options.ecmaVersion>=8&&!M&&this.isAsyncProp(S)?(R=!0,M=this.options.ecmaVersion>=9&&this.eat(h.star),this.parsePropertyName(S)):R=!1,this.parsePropertyValue(S,r,M,R,K,B,u,J),this.finishNode(S,"Property")},oe.parseGetterSetter=function(r){r.kind=r.key.name,this.parsePropertyName(r),r.value=this.parseMethod(!1);var u=r.kind==="get"?0:1;if(r.value.params.length!==u){var S=r.value.start;r.kind==="get"?this.raiseRecoverable(S,"getter should have no params"):this.raiseRecoverable(S,"setter should have exactly one param")}else r.kind==="set"&&r.value.params[0].type==="RestElement"&&this.raiseRecoverable(r.value.params[0].start,"Setter cannot use rest params")},oe.parsePropertyValue=function(r,u,S,M,R,K,B,J){(S||M)&&this.type===h.colon&&this.unexpected(),this.eat(h.colon)?(r.value=u?this.parseMaybeDefault(this.start,this.startLoc):this.parseMaybeAssign(!1,B),r.kind="init"):this.options.ecmaVersion>=6&&this.type===h.parenL?(u&&this.unexpected(),r.kind="init",r.method=!0,r.value=this.parseMethod(S,M)):!u&&!J&&this.options.ecmaVersion>=5&&!r.computed&&r.key.type==="Identifier"&&(r.key.name==="get"||r.key.name==="set")&&this.type!==h.comma&&this.type!==h.braceR&&this.type!==h.eq?((S||M)&&this.unexpected(),this.parseGetterSetter(r)):this.options.ecmaVersion>=6&&!r.computed&&r.key.type==="Identifier"?((S||M)&&this.unexpected(),this.checkUnreserved(r.key),r.key.name==="await"&&!this.awaitIdentPos&&(this.awaitIdentPos=R),r.kind="init",u?r.value=this.parseMaybeDefault(R,K,this.copyNode(r.key)):this.type===h.eq&&B?(B.shorthandAssign<0&&(B.shorthandAssign=this.start),r.value=this.parseMaybeDefault(R,K,this.copyNode(r.key))):r.value=this.copyNode(r.key),r.shorthand=!0):this.unexpected()},oe.parsePropertyName=function(r){if(this.options.ecmaVersion>=6){if(this.eat(h.bracketL))return r.computed=!0,r.key=this.parseMaybeAssign(),this.expect(h.bracketR),r.key;r.computed=!1}return r.key=this.type===h.num||this.type===h.string?this.parseExprAtom():this.parseIdent(this.options.allowReserved!=="never")},oe.initFunction=function(r){r.id=null,this.options.ecmaVersion>=6&&(r.generator=r.expression=!1),this.options.ecmaVersion>=8&&(r.async=!1)},oe.parseMethod=function(r,u,S){var M=this.startNode(),R=this.yieldPos,K=this.awaitPos,B=this.awaitIdentPos;return this.initFunction(M),this.options.ecmaVersion>=6&&(M.generator=r),this.options.ecmaVersion>=8&&(M.async=!!u),this.yieldPos=0,this.awaitPos=0,this.awaitIdentPos=0,this.enterScope(Mn(u,M.generator)|En|(S?li:0)),this.expect(h.parenL),M.params=this.parseBindingList(h.parenR,!1,this.options.ecmaVersion>=8),this.checkYieldAwaitInDefaultParams(),this.parseFunctionBody(M,!1,!0,!1),this.yieldPos=R,this.awaitPos=K,this.awaitIdentPos=B,this.finishNode(M,"FunctionExpression")},oe.parseArrowExpression=function(r,u,S,M){var R=this.yieldPos,K=this.awaitPos,B=this.awaitIdentPos;return this.enterScope(Mn(S,!1)|ft),this.initFunction(r),this.options.ecmaVersion>=8&&(r.async=!!S),this.yieldPos=0,this.awaitPos=0,this.awaitIdentPos=0,r.params=this.toAssignableList(u,!0),this.parseFunctionBody(r,!0,!1,M),this.yieldPos=R,this.awaitPos=K,this.awaitIdentPos=B,this.finishNode(r,"ArrowFunctionExpression")},oe.parseFunctionBody=function(r,u,S,M){var R=u&&this.type!==h.braceL,K=this.strict,B=!1;if(R)r.body=this.parseMaybeAssign(M),r.expression=!0,this.checkParams(r,!1);else{var J=this.options.ecmaVersion>=7&&!this.isSimpleParamList(r.params);(!K||J)&&(B=this.strictDirective(this.end),B&&J&&this.raiseRecoverable(r.start,"Illegal 'use strict' directive in function with non-simple parameter list"));var Q=this.labels;this.labels=[],B&&(this.strict=!0),this.checkParams(r,!K&&!B&&!u&&!S&&this.isSimpleParamList(r.params)),this.strict&&r.id&&this.checkLValSimple(r.id,hi),r.body=this.parseBlock(!1,void 0,B&&!K),r.expression=!1,this.adaptDirectivePrologue(r.body.body),this.labels=Q}this.exitScope()},oe.isSimpleParamList=function(r){for(var u=0,S=r;u<S.length;u+=1)if(S[u].type!=="Identifier")return!1;return!0},oe.checkParams=function(r,u){for(var S=Object.create(null),M=0,R=r.params;M<R.length;M+=1){var K=R[M];this.checkLValInnerPattern(K,$n,u?null:S)}},oe.parseExprList=function(r,u,S,M){for(var R=[],K=!0;!this.eat(r);){if(K)K=!1;else if(this.expect(h.comma),u&&this.afterTrailingComma(r))break;var B=void 0;S&&this.type===h.comma?B=null:this.type===h.ellipsis?(B=this.parseSpread(M),M&&this.type===h.comma&&M.trailingComma<0&&(M.trailingComma=this.start)):B=this.parseMaybeAssign(!1,M),R.push(B)}return R},oe.checkUnreserved=function(r){var u=r.start,S=r.end,M=r.name;this.inGenerator&&M==="yield"&&this.raiseRecoverable(u,"Cannot use 'yield' as identifier inside a generator"),this.inAsync&&M==="await"&&this.raiseRecoverable(u,"Cannot use 'await' as identifier inside an async function"),this.currentThisScope().inClassFieldInit&&M==="arguments"&&this.raiseRecoverable(u,"Cannot use 'arguments' in class field initializer"),this.inClassStaticBlock&&(M==="arguments"||M==="await")&&this.raise(u,"Cannot use "+M+" in class static initialization block"),this.keywords.test(M)&&this.raise(u,"Unexpected keyword '"+M+"'"),!(this.options.ecmaVersion<6&&this.input.slice(u,S).indexOf("\\")!==-1)&&(this.strict?this.reservedWordsStrict:this.reservedWords).test(M)&&(!this.inAsync&&M==="await"&&this.raiseRecoverable(u,"Cannot use keyword 'await' outside an async function"),this.raiseRecoverable(u,"The keyword '"+M+"' is reserved"))},oe.parseIdent=function(r){var u=this.parseIdentNode();return this.next(!!r),this.finishNode(u,"Identifier"),r||(this.checkUnreserved(u),u.name==="await"&&!this.awaitIdentPos&&(this.awaitIdentPos=u.start)),u},oe.parseIdentNode=function(){var r=this.startNode();return this.type===h.name?r.name=this.value:this.type.keyword?(r.name=this.type.keyword,(r.name==="class"||r.name==="function")&&(this.lastTokEnd!==this.lastTokStart+1||this.input.charCodeAt(this.lastTokStart)!==46)&&this.context.pop(),this.type=h.name):this.unexpected(),r},oe.parsePrivateIdent=function(){var r=this.startNode();return this.type===h.privateId?r.name=this.value:this.unexpected(),this.next(),this.finishNode(r,"PrivateIdentifier"),this.options.checkPrivateFields&&(this.privateNameStack.length===0?this.raise(r.start,"Private field '#"+r.name+"' must be declared in an enclosing class"):this.privateNameStack[this.privateNameStack.length-1].used.push(r)),r},oe.parseYield=function(r){this.yieldPos||(this.yieldPos=this.start);var u=this.startNode();return this.next(),this.type===h.semi||this.canInsertSemicolon()||this.type!==h.star&&!this.type.startsExpr?(u.delegate=!1,u.argument=null):(u.delegate=this.eat(h.star),u.argument=this.parseMaybeAssign(r)),this.finishNode(u,"YieldExpression")},oe.parseAwait=function(r){this.awaitPos||(this.awaitPos=this.start);var u=this.startNode();return this.next(),u.argument=this.parseMaybeUnary(null,!0,!1,r),this.finishNode(u,"AwaitExpression")};var Vs=_e.prototype;Vs.raise=function(r,u){var S=Ve(this.input,r);u+=" ("+S.line+":"+S.column+")";var M=new SyntaxError(u);throw M.pos=r,M.loc=S,M.raisedAt=this.pos,M},Vs.raiseRecoverable=Vs.raise,Vs.curPosition=function(){if(this.options.locations)return new re(this.curLine,this.pos-this.lineStart)};var kt=_e.prototype,Zo=function(u){this.flags=u,this.var=[],this.lexical=[],this.functions=[],this.inClassFieldInit=!1};kt.enterScope=function(r){this.scopeStack.push(new Zo(r))},kt.exitScope=function(){this.scopeStack.pop()},kt.treatFunctionsAsVarInScope=function(r){return r.flags&ye||!this.inModule&&r.flags&ae},kt.declareName=function(r,u,S){var M=!1;if(u===mt){var R=this.currentScope();M=R.lexical.indexOf(r)>-1||R.functions.indexOf(r)>-1||R.var.indexOf(r)>-1,R.lexical.push(r),this.inModule&&R.flags&ae&&delete this.undefinedExports[r]}else if(u===ci)this.currentScope().lexical.push(r);else if(u===ui){var K=this.currentScope();this.treatFunctionsAsVar?M=K.lexical.indexOf(r)>-1:M=K.lexical.indexOf(r)>-1||K.var.indexOf(r)>-1,K.functions.push(r)}else for(var B=this.scopeStack.length-1;B>=0;--B){var J=this.scopeStack[B];if(J.lexical.indexOf(r)>-1&&!(J.flags&oi&&J.lexical[0]===r)||!this.treatFunctionsAsVarInScope(J)&&J.functions.indexOf(r)>-1){M=!0;break}if(J.var.push(r),this.inModule&&J.flags&ae&&delete this.undefinedExports[r],J.flags&In)break}M&&this.raiseRecoverable(S,"Identifier '"+r+"' has already been declared")},kt.checkLocalExport=function(r){this.scopeStack[0].lexical.indexOf(r.name)===-1&&this.scopeStack[0].var.indexOf(r.name)===-1&&(this.undefinedExports[r.name]=r)},kt.currentScope=function(){return this.scopeStack[this.scopeStack.length-1]},kt.currentVarScope=function(){for(var r=this.scopeStack.length-1;;r--){var u=this.scopeStack[r];if(u.flags&In)return u}},kt.currentThisScope=function(){for(var r=this.scopeStack.length-1;;r--){var u=this.scopeStack[r];if(u.flags&In&&!(u.flags&ft))return u}};var ms=function(u,S,M){this.type="",this.start=S,this.end=0,u.options.locations&&(this.loc=new ce(u,M)),u.options.directSourceFile&&(this.sourceFile=u.options.directSourceFile),u.options.ranges&&(this.range=[S,0])},gs=_e.prototype;gs.startNode=function(){return new ms(this,this.start,this.startLoc)},gs.startNodeAt=function(r,u){return new ms(this,r,u)};function fi(r,u,S,M){return r.type=u,r.end=S,this.options.locations&&(r.loc.end=M),this.options.ranges&&(r.range[1]=S),r}gs.finishNode=function(r,u){return fi.call(this,r,u,this.lastTokEnd,this.lastTokEndLoc)},gs.finishNodeAt=function(r,u,S,M){return fi.call(this,r,u,S,M)},gs.copyNode=function(r){var u=new ms(this,r.start,this.startLoc);for(var S in r)u[S]=r[S];return u};var Qo="Gara Garay Gukh Gurung_Khema Hrkt Katakana_Or_Hiragana Kawi Kirat_Rai Krai Nag_Mundari Nagm Ol_Onal Onao Sunu Sunuwar Todhri Todr Tulu_Tigalari Tutg Unknown Zzzz",mi="ASCII ASCII_Hex_Digit AHex Alphabetic Alpha Any Assigned Bidi_Control Bidi_C Bidi_Mirrored Bidi_M Case_Ignorable CI Cased Changes_When_Casefolded CWCF Changes_When_Casemapped CWCM Changes_When_Lowercased CWL Changes_When_NFKC_Casefolded CWKCF Changes_When_Titlecased CWT Changes_When_Uppercased CWU Dash Default_Ignorable_Code_Point DI Deprecated Dep Diacritic Dia Emoji Emoji_Component Emoji_Modifier Emoji_Modifier_Base Emoji_Presentation Extender Ext Grapheme_Base Gr_Base Grapheme_Extend Gr_Ext Hex_Digit Hex IDS_Binary_Operator IDSB IDS_Trinary_Operator IDST ID_Continue IDC ID_Start IDS Ideographic Ideo Join_Control Join_C Logical_Order_Exception LOE Lowercase Lower Math Noncharacter_Code_Point NChar Pattern_Syntax Pat_Syn Pattern_White_Space Pat_WS Quotation_Mark QMark Radical Regional_Indicator RI Sentence_Terminal STerm Soft_Dotted SD Terminal_Punctuation Term Unified_Ideograph UIdeo Uppercase Upper Variation_Selector VS White_Space space XID_Continue XIDC XID_Start XIDS",gi=mi+" Extended_Pictographic",yi=gi,xi=yi+" EBase EComp EMod EPres ExtPict",bi=xi,el={9:mi,10:gi,11:yi,12:xi,13:bi,14:bi},tl={9:"",10:"",11:"",12:"",13:"",14:"Basic_Emoji Emoji_Keycap_Sequence RGI_Emoji_Modifier_Sequence RGI_Emoji_Flag_Sequence RGI_Emoji_Tag_Sequence RGI_Emoji_ZWJ_Sequence RGI_Emoji"},wi="Cased_Letter LC Close_Punctuation Pe Connector_Punctuation Pc Control Cc cntrl Currency_Symbol Sc Dash_Punctuation Pd Decimal_Number Nd digit Enclosing_Mark Me Final_Punctuation Pf Format Cf Initial_Punctuation Pi Letter L Letter_Number Nl Line_Separator Zl Lowercase_Letter Ll Mark M Combining_Mark Math_Symbol Sm Modifier_Letter Lm Modifier_Symbol Sk Nonspacing_Mark Mn Number N Open_Punctuation Ps Other C Other_Letter Lo Other_Number No Other_Punctuation Po Other_Symbol So Paragraph_Separator Zp Private_Use Co Punctuation P punct Separator Z Space_Separator Zs Spacing_Mark Mc Surrogate Cs Symbol S Titlecase_Letter Lt Unassigned Cn Uppercase_Letter Lu",vi="Adlam Adlm Ahom Anatolian_Hieroglyphs Hluw Arabic Arab Armenian Armn Avestan Avst Balinese Bali Bamum Bamu Bassa_Vah Bass Batak Batk Bengali Beng Bhaiksuki Bhks Bopomofo Bopo Brahmi Brah Braille Brai Buginese Bugi Buhid Buhd Canadian_Aboriginal Cans Carian Cari Caucasian_Albanian Aghb Chakma Cakm Cham Cham Cherokee Cher Common Zyyy Coptic Copt Qaac Cuneiform Xsux Cypriot Cprt Cyrillic Cyrl Deseret Dsrt Devanagari Deva Duployan Dupl Egyptian_Hieroglyphs Egyp Elbasan Elba Ethiopic Ethi Georgian Geor Glagolitic Glag Gothic Goth Grantha Gran Greek Grek Gujarati Gujr Gurmukhi Guru Han Hani Hangul Hang Hanunoo Hano Hatran Hatr Hebrew Hebr Hiragana Hira Imperial_Aramaic Armi Inherited Zinh Qaai Inscriptional_Pahlavi Phli Inscriptional_Parthian Prti Javanese Java Kaithi Kthi Kannada Knda Katakana Kana Kayah_Li Kali Kharoshthi Khar Khmer Khmr Khojki Khoj Khudawadi Sind Lao Laoo Latin Latn Lepcha Lepc Limbu Limb Linear_A Lina Linear_B Linb Lisu Lisu Lycian Lyci Lydian Lydi Mahajani Mahj Malayalam Mlym Mandaic Mand Manichaean Mani Marchen Marc Masaram_Gondi Gonm Meetei_Mayek Mtei Mende_Kikakui Mend Meroitic_Cursive Merc Meroitic_Hieroglyphs Mero Miao Plrd Modi Mongolian Mong Mro Mroo Multani Mult Myanmar Mymr Nabataean Nbat New_Tai_Lue Talu Newa Newa Nko Nkoo Nushu Nshu Ogham Ogam Ol_Chiki Olck Old_Hungarian Hung Old_Italic Ital Old_North_Arabian Narb Old_Permic Perm Old_Persian Xpeo Old_South_Arabian Sarb Old_Turkic Orkh Oriya Orya Osage Osge Osmanya Osma Pahawh_Hmong Hmng Palmyrene Palm Pau_Cin_Hau Pauc Phags_Pa Phag Phoenician Phnx Psalter_Pahlavi Phlp Rejang Rjng Runic Runr Samaritan Samr Saurashtra Saur Sharada Shrd Shavian Shaw Siddham Sidd SignWriting Sgnw Sinhala Sinh Sora_Sompeng Sora Soyombo Soyo Sundanese Sund Syloti_Nagri Sylo Syriac Syrc Tagalog Tglg Tagbanwa Tagb Tai_Le Tale Tai_Tham Lana Tai_Viet Tavt Takri Takr Tamil Taml Tangut Tang Telugu Telu Thaana Thaa Thai Thai Tibetan Tibt Tifinagh Tfng Tirhuta Tirh Ugaritic Ugar Vai Vaii Warang_Citi Wara Yi Yiii Zanabazar_Square Zanb",ki=vi+" Dogra Dogr Gunjala_Gondi Gong Hanifi_Rohingya Rohg Makasar Maka Medefaidrin Medf Old_Sogdian Sogo Sogdian Sogd",Ti=ki+" Elymaic Elym Nandinagari Nand Nyiakeng_Puachue_Hmong Hmnp Wancho Wcho",Si=Ti+" Chorasmian Chrs Diak Dives_Akuru Khitan_Small_Script Kits Yezi Yezidi",_i=Si+" Cypro_Minoan Cpmn Old_Uyghur Ougr Tangsa Tnsa Toto Vithkuqi Vith",sl={9:vi,10:ki,11:Ti,12:Si,13:_i,14:_i+" "+Qo},Ci={};function nl(r){var u=Ci[r]={binary:ie(el[r]+" "+wi),binaryOfStrings:ie(tl[r]),nonBinary:{General_Category:ie(wi),Script:ie(sl[r])}};u.nonBinary.Script_Extensions=u.nonBinary.Script,u.nonBinary.gc=u.nonBinary.General_Category,u.nonBinary.sc=u.nonBinary.Script,u.nonBinary.scx=u.nonBinary.Script_Extensions}for(var zn=0,Ei=[9,10,11,12,13,14];zn<Ei.length;zn+=1){var rl=Ei[zn];nl(rl)}var te=_e.prototype,Ks=function(u,S){this.parent=u,this.base=S||this};Ks.prototype.separatedFrom=function(u){for(var S=this;S;S=S.parent)for(var M=u;M;M=M.parent)if(S.base===M.base&&S!==M)return!0;return!1},Ks.prototype.sibling=function(){return new Ks(this.parent,this.base)};var at=function(u){this.parser=u,this.validFlags="gim"+(u.options.ecmaVersion>=6?"uy":"")+(u.options.ecmaVersion>=9?"s":"")+(u.options.ecmaVersion>=13?"d":"")+(u.options.ecmaVersion>=15?"v":""),this.unicodeProperties=Ci[u.options.ecmaVersion>=14?14:u.options.ecmaVersion],this.source="",this.flags="",this.start=0,this.switchU=!1,this.switchV=!1,this.switchN=!1,this.pos=0,this.lastIntValue=0,this.lastStringValue="",this.lastAssertionIsQuantifiable=!1,this.numCapturingParens=0,this.maxBackReference=0,this.groupNames=Object.create(null),this.backReferenceNames=[],this.branchID=null};at.prototype.reset=function(u,S,M){var R=M.indexOf("v")!==-1,K=M.indexOf("u")!==-1;this.start=u|0,this.source=S+"",this.flags=M,R&&this.parser.options.ecmaVersion>=15?(this.switchU=!0,this.switchV=!0,this.switchN=!0):(this.switchU=K&&this.parser.options.ecmaVersion>=6,this.switchV=!1,this.switchN=K&&this.parser.options.ecmaVersion>=9)},at.prototype.raise=function(u){this.parser.raiseRecoverable(this.start,"Invalid regular expression: /"+this.source+"/: "+u)},at.prototype.at=function(u,S){S===void 0&&(S=!1);var M=this.source,R=M.length;if(u>=R)return-1;var K=M.charCodeAt(u);if(!(S||this.switchU)||K<=55295||K>=57344||u+1>=R)return K;var B=M.charCodeAt(u+1);return B>=56320&&B<=57343?(K<<10)+B-56613888:K},at.prototype.nextIndex=function(u,S){S===void 0&&(S=!1);var M=this.source,R=M.length;if(u>=R)return R;var K=M.charCodeAt(u),B;return!(S||this.switchU)||K<=55295||K>=57344||u+1>=R||(B=M.charCodeAt(u+1))<56320||B>57343?u+1:u+2},at.prototype.current=function(u){return u===void 0&&(u=!1),this.at(this.pos,u)},at.prototype.lookahead=function(u){return u===void 0&&(u=!1),this.at(this.nextIndex(this.pos,u),u)},at.prototype.advance=function(u){u===void 0&&(u=!1),this.pos=this.nextIndex(this.pos,u)},at.prototype.eat=function(u,S){return S===void 0&&(S=!1),this.current(S)===u?(this.advance(S),!0):!1},at.prototype.eatChars=function(u,S){S===void 0&&(S=!1);for(var M=this.pos,R=0,K=u;R<K.length;R+=1){var B=K[R],J=this.at(M,S);if(J===-1||J!==B)return!1;M=this.nextIndex(M,S)}return this.pos=M,!0},te.validateRegExpFlags=function(r){for(var u=r.validFlags,S=r.flags,M=!1,R=!1,K=0;K<S.length;K++){var B=S.charAt(K);u.indexOf(B)===-1&&this.raise(r.start,"Invalid regular expression flag"),S.indexOf(B,K+1)>-1&&this.raise(r.start,"Duplicate regular expression flag"),B==="u"&&(M=!0),B==="v"&&(R=!0)}this.options.ecmaVersion>=15&&M&&R&&this.raise(r.start,"Invalid regular expression flag")};function il(r){for(var u in r)return!0;return!1}te.validateRegExpPattern=function(r){this.regexp_pattern(r),!r.switchN&&this.options.ecmaVersion>=9&&il(r.groupNames)&&(r.switchN=!0,this.regexp_pattern(r))},te.regexp_pattern=function(r){r.pos=0,r.lastIntValue=0,r.lastStringValue="",r.lastAssertionIsQuantifiable=!1,r.numCapturingParens=0,r.maxBackReference=0,r.groupNames=Object.create(null),r.backReferenceNames.length=0,r.branchID=null,this.regexp_disjunction(r),r.pos!==r.source.length&&(r.eat(41)&&r.raise("Unmatched ')'"),(r.eat(93)||r.eat(125))&&r.raise("Lone quantifier brackets")),r.maxBackReference>r.numCapturingParens&&r.raise("Invalid escape");for(var u=0,S=r.backReferenceNames;u<S.length;u+=1){var M=S[u];r.groupNames[M]||r.raise("Invalid named capture referenced")}},te.regexp_disjunction=function(r){var u=this.options.ecmaVersion>=16;for(u&&(r.branchID=new Ks(r.branchID,null)),this.regexp_alternative(r);r.eat(124);)u&&(r.branchID=r.branchID.sibling()),this.regexp_alternative(r);u&&(r.branchID=r.branchID.parent),this.regexp_eatQuantifier(r,!0)&&r.raise("Nothing to repeat"),r.eat(123)&&r.raise("Lone quantifier brackets")},te.regexp_alternative=function(r){for(;r.pos<r.source.length&&this.regexp_eatTerm(r););},te.regexp_eatTerm=function(r){return this.regexp_eatAssertion(r)?(r.lastAssertionIsQuantifiable&&this.regexp_eatQuantifier(r)&&r.switchU&&r.raise("Invalid quantifier"),!0):(r.switchU?this.regexp_eatAtom(r):this.regexp_eatExtendedAtom(r))?(this.regexp_eatQuantifier(r),!0):!1},te.regexp_eatAssertion=function(r){var u=r.pos;if(r.lastAssertionIsQuantifiable=!1,r.eat(94)||r.eat(36))return!0;if(r.eat(92)){if(r.eat(66)||r.eat(98))return!0;r.pos=u}if(r.eat(40)&&r.eat(63)){var S=!1;if(this.options.ecmaVersion>=9&&(S=r.eat(60)),r.eat(61)||r.eat(33))return this.regexp_disjunction(r),r.eat(41)||r.raise("Unterminated group"),r.lastAssertionIsQuantifiable=!S,!0}return r.pos=u,!1},te.regexp_eatQuantifier=function(r,u){return u===void 0&&(u=!1),this.regexp_eatQuantifierPrefix(r,u)?(r.eat(63),!0):!1},te.regexp_eatQuantifierPrefix=function(r,u){return r.eat(42)||r.eat(43)||r.eat(63)||this.regexp_eatBracedQuantifier(r,u)},te.regexp_eatBracedQuantifier=function(r,u){var S=r.pos;if(r.eat(123)){var M=0,R=-1;if(this.regexp_eatDecimalDigits(r)&&(M=r.lastIntValue,r.eat(44)&&this.regexp_eatDecimalDigits(r)&&(R=r.lastIntValue),r.eat(125)))return R!==-1&&R<M&&!u&&r.raise("numbers out of order in {} quantifier"),!0;r.switchU&&!u&&r.raise("Incomplete quantifier"),r.pos=S}return!1},te.regexp_eatAtom=function(r){return this.regexp_eatPatternCharacters(r)||r.eat(46)||this.regexp_eatReverseSolidusAtomEscape(r)||this.regexp_eatCharacterClass(r)||this.regexp_eatUncapturingGroup(r)||this.regexp_eatCapturingGroup(r)},te.regexp_eatReverseSolidusAtomEscape=function(r){var u=r.pos;if(r.eat(92)){if(this.regexp_eatAtomEscape(r))return!0;r.pos=u}return!1},te.regexp_eatUncapturingGroup=function(r){var u=r.pos;if(r.eat(40)){if(r.eat(63)){if(this.options.ecmaVersion>=16){var S=this.regexp_eatModifiers(r),M=r.eat(45);if(S||M){for(var R=0;R<S.length;R++){var K=S.charAt(R);S.indexOf(K,R+1)>-1&&r.raise("Duplicate regular expression modifiers")}if(M){var B=this.regexp_eatModifiers(r);!S&&!B&&r.current()===58&&r.raise("Invalid regular expression modifiers");for(var J=0;J<B.length;J++){var Q=B.charAt(J);(B.indexOf(Q,J+1)>-1||S.indexOf(Q)>-1)&&r.raise("Duplicate regular expression modifiers")}}}}if(r.eat(58)){if(this.regexp_disjunction(r),r.eat(41))return!0;r.raise("Unterminated group")}}r.pos=u}return!1},te.regexp_eatCapturingGroup=function(r){if(r.eat(40)){if(this.options.ecmaVersion>=9?this.regexp_groupSpecifier(r):r.current()===63&&r.raise("Invalid group"),this.regexp_disjunction(r),r.eat(41))return r.numCapturingParens+=1,!0;r.raise("Unterminated group")}return!1},te.regexp_eatModifiers=function(r){for(var u="",S=0;(S=r.current())!==-1&&al(S);)u+=he(S),r.advance();return u};function al(r){return r===105||r===109||r===115}te.regexp_eatExtendedAtom=function(r){return r.eat(46)||this.regexp_eatReverseSolidusAtomEscape(r)||this.regexp_eatCharacterClass(r)||this.regexp_eatUncapturingGroup(r)||this.regexp_eatCapturingGroup(r)||this.regexp_eatInvalidBracedQuantifier(r)||this.regexp_eatExtendedPatternCharacter(r)},te.regexp_eatInvalidBracedQuantifier=function(r){return this.regexp_eatBracedQuantifier(r,!0)&&r.raise("Nothing to repeat"),!1},te.regexp_eatSyntaxCharacter=function(r){var u=r.current();return Ii(u)?(r.lastIntValue=u,r.advance(),!0):!1};function Ii(r){return r===36||r>=40&&r<=43||r===46||r===63||r>=91&&r<=94||r>=123&&r<=125}te.regexp_eatPatternCharacters=function(r){for(var u=r.pos,S=0;(S=r.current())!==-1&&!Ii(S);)r.advance();return r.pos!==u},te.regexp_eatExtendedPatternCharacter=function(r){var u=r.current();return u!==-1&&u!==36&&!(u>=40&&u<=43)&&u!==46&&u!==63&&u!==91&&u!==94&&u!==124?(r.advance(),!0):!1},te.regexp_groupSpecifier=function(r){if(r.eat(63)){this.regexp_eatGroupName(r)||r.raise("Invalid group");var u=this.options.ecmaVersion>=16,S=r.groupNames[r.lastStringValue];if(S)if(u)for(var M=0,R=S;M<R.length;M+=1)R[M].separatedFrom(r.branchID)||r.raise("Duplicate capture group name");else r.raise("Duplicate capture group name");u?(S||(r.groupNames[r.lastStringValue]=[])).push(r.branchID):r.groupNames[r.lastStringValue]=!0}},te.regexp_eatGroupName=function(r){if(r.lastStringValue="",r.eat(60)){if(this.regexp_eatRegExpIdentifierName(r)&&r.eat(62))return!0;r.raise("Invalid capture group name")}return!1},te.regexp_eatRegExpIdentifierName=function(r){if(r.lastStringValue="",this.regexp_eatRegExpIdentifierStart(r)){for(r.lastStringValue+=he(r.lastIntValue);this.regexp_eatRegExpIdentifierPart(r);)r.lastStringValue+=he(r.lastIntValue);return!0}return!1},te.regexp_eatRegExpIdentifierStart=function(r){var u=r.pos,S=this.options.ecmaVersion>=11,M=r.current(S);return r.advance(S),M===92&&this.regexp_eatRegExpUnicodeEscapeSequence(r,S)&&(M=r.lastIntValue),ol(M)?(r.lastIntValue=M,!0):(r.pos=u,!1)};function ol(r){return o(r,!0)||r===36||r===95}te.regexp_eatRegExpIdentifierPart=function(r){var u=r.pos,S=this.options.ecmaVersion>=11,M=r.current(S);return r.advance(S),M===92&&this.regexp_eatRegExpUnicodeEscapeSequence(r,S)&&(M=r.lastIntValue),ll(M)?(r.lastIntValue=M,!0):(r.pos=u,!1)};function ll(r){return l(r,!0)||r===36||r===95||r===8204||r===8205}te.regexp_eatAtomEscape=function(r){return this.regexp_eatBackReference(r)||this.regexp_eatCharacterClassEscape(r)||this.regexp_eatCharacterEscape(r)||r.switchN&&this.regexp_eatKGroupName(r)?!0:(r.switchU&&(r.current()===99&&r.raise("Invalid unicode escape"),r.raise("Invalid escape")),!1)},te.regexp_eatBackReference=function(r){var u=r.pos;if(this.regexp_eatDecimalEscape(r)){var S=r.lastIntValue;if(r.switchU)return S>r.maxBackReference&&(r.maxBackReference=S),!0;if(S<=r.numCapturingParens)return!0;r.pos=u}return!1},te.regexp_eatKGroupName=function(r){if(r.eat(107)){if(this.regexp_eatGroupName(r))return r.backReferenceNames.push(r.lastStringValue),!0;r.raise("Invalid named reference")}return!1},te.regexp_eatCharacterEscape=function(r){return this.regexp_eatControlEscape(r)||this.regexp_eatCControlLetter(r)||this.regexp_eatZero(r)||this.regexp_eatHexEscapeSequence(r)||this.regexp_eatRegExpUnicodeEscapeSequence(r,!1)||!r.switchU&&this.regexp_eatLegacyOctalEscapeSequence(r)||this.regexp_eatIdentityEscape(r)},te.regexp_eatCControlLetter=function(r){var u=r.pos;if(r.eat(99)){if(this.regexp_eatControlLetter(r))return!0;r.pos=u}return!1},te.regexp_eatZero=function(r){return r.current()===48&&!Ns(r.lookahead())?(r.lastIntValue=0,r.advance(),!0):!1},te.regexp_eatControlEscape=function(r){var u=r.current();return u===116?(r.lastIntValue=9,r.advance(),!0):u===110?(r.lastIntValue=10,r.advance(),!0):u===118?(r.lastIntValue=11,r.advance(),!0):u===102?(r.lastIntValue=12,r.advance(),!0):u===114?(r.lastIntValue=13,r.advance(),!0):!1},te.regexp_eatControlLetter=function(r){var u=r.current();return Mi(u)?(r.lastIntValue=u%32,r.advance(),!0):!1};function Mi(r){return r>=65&&r<=90||r>=97&&r<=122}te.regexp_eatRegExpUnicodeEscapeSequence=function(r,u){u===void 0&&(u=!1);var S=r.pos,M=u||r.switchU;if(r.eat(117)){if(this.regexp_eatFixedHexDigits(r,4)){var R=r.lastIntValue;if(M&&R>=55296&&R<=56319){var K=r.pos;if(r.eat(92)&&r.eat(117)&&this.regexp_eatFixedHexDigits(r,4)){var B=r.lastIntValue;if(B>=56320&&B<=57343)return r.lastIntValue=(R-55296)*1024+(B-56320)+65536,!0}r.pos=K,r.lastIntValue=R}return!0}if(M&&r.eat(123)&&this.regexp_eatHexDigits(r)&&r.eat(125)&&ul(r.lastIntValue))return!0;M&&r.raise("Invalid unicode escape"),r.pos=S}return!1};function ul(r){return r>=0&&r<=1114111}te.regexp_eatIdentityEscape=function(r){if(r.switchU)return this.regexp_eatSyntaxCharacter(r)?!0:r.eat(47)?(r.lastIntValue=47,!0):!1;var u=r.current();return u!==99&&(!r.switchN||u!==107)?(r.lastIntValue=u,r.advance(),!0):!1},te.regexp_eatDecimalEscape=function(r){r.lastIntValue=0;var u=r.current();if(u>=49&&u<=57){do r.lastIntValue=10*r.lastIntValue+(u-48),r.advance();while((u=r.current())>=48&&u<=57);return!0}return!1};var $i=0,gt=1,qe=2;te.regexp_eatCharacterClassEscape=function(r){var u=r.current();if(cl(u))return r.lastIntValue=-1,r.advance(),gt;var S=!1;if(r.switchU&&this.options.ecmaVersion>=9&&((S=u===80)||u===112)){r.lastIntValue=-1,r.advance();var M;if(r.eat(123)&&(M=this.regexp_eatUnicodePropertyValueExpression(r))&&r.eat(125))return S&&M===qe&&r.raise("Invalid property name"),M;r.raise("Invalid property name")}return $i};function cl(r){return r===100||r===68||r===115||r===83||r===119||r===87}te.regexp_eatUnicodePropertyValueExpression=function(r){var u=r.pos;if(this.regexp_eatUnicodePropertyName(r)&&r.eat(61)){var S=r.lastStringValue;if(this.regexp_eatUnicodePropertyValue(r)){var M=r.lastStringValue;return this.regexp_validateUnicodePropertyNameAndValue(r,S,M),gt}}if(r.pos=u,this.regexp_eatLoneUnicodePropertyNameOrValue(r)){var R=r.lastStringValue;return this.regexp_validateUnicodePropertyNameOrValue(r,R)}return $i},te.regexp_validateUnicodePropertyNameAndValue=function(r,u,S){ee(r.unicodeProperties.nonBinary,u)||r.raise("Invalid property name"),r.unicodeProperties.nonBinary[u].test(S)||r.raise("Invalid property value")},te.regexp_validateUnicodePropertyNameOrValue=function(r,u){if(r.unicodeProperties.binary.test(u))return gt;if(r.switchV&&r.unicodeProperties.binaryOfStrings.test(u))return qe;r.raise("Invalid property name")},te.regexp_eatUnicodePropertyName=function(r){var u=0;for(r.lastStringValue="";Ai(u=r.current());)r.lastStringValue+=he(u),r.advance();return r.lastStringValue!==""};function Ai(r){return Mi(r)||r===95}te.regexp_eatUnicodePropertyValue=function(r){var u=0;for(r.lastStringValue="";hl(u=r.current());)r.lastStringValue+=he(u),r.advance();return r.lastStringValue!==""};function hl(r){return Ai(r)||Ns(r)}te.regexp_eatLoneUnicodePropertyNameOrValue=function(r){return this.regexp_eatUnicodePropertyValue(r)},te.regexp_eatCharacterClass=function(r){if(r.eat(91)){var u=r.eat(94),S=this.regexp_classContents(r);return r.eat(93)||r.raise("Unterminated character class"),u&&S===qe&&r.raise("Negated character class may contain strings"),!0}return!1},te.regexp_classContents=function(r){return r.current()===93?gt:r.switchV?this.regexp_classSetExpression(r):(this.regexp_nonEmptyClassRanges(r),gt)},te.regexp_nonEmptyClassRanges=function(r){for(;this.regexp_eatClassAtom(r);){var u=r.lastIntValue;if(r.eat(45)&&this.regexp_eatClassAtom(r)){var S=r.lastIntValue;r.switchU&&(u===-1||S===-1)&&r.raise("Invalid character class"),u!==-1&&S!==-1&&u>S&&r.raise("Range out of order in character class")}}},te.regexp_eatClassAtom=function(r){var u=r.pos;if(r.eat(92)){if(this.regexp_eatClassEscape(r))return!0;if(r.switchU){var S=r.current();(S===99||zi(S))&&r.raise("Invalid class escape"),r.raise("Invalid escape")}r.pos=u}var M=r.current();return M!==93?(r.lastIntValue=M,r.advance(),!0):!1},te.regexp_eatClassEscape=function(r){var u=r.pos;if(r.eat(98))return r.lastIntValue=8,!0;if(r.switchU&&r.eat(45))return r.lastIntValue=45,!0;if(!r.switchU&&r.eat(99)){if(this.regexp_eatClassControlLetter(r))return!0;r.pos=u}return this.regexp_eatCharacterClassEscape(r)||this.regexp_eatCharacterEscape(r)},te.regexp_classSetExpression=function(r){var u=gt,S;if(!this.regexp_eatClassSetRange(r))if(S=this.regexp_eatClassSetOperand(r)){S===qe&&(u=qe);for(var M=r.pos;r.eatChars([38,38]);){if(r.current()!==38&&(S=this.regexp_eatClassSetOperand(r))){S!==qe&&(u=gt);continue}r.raise("Invalid character in character class")}if(M!==r.pos)return u;for(;r.eatChars([45,45]);)this.regexp_eatClassSetOperand(r)||r.raise("Invalid character in character class");if(M!==r.pos)return u}else r.raise("Invalid character in character class");for(;;)if(!this.regexp_eatClassSetRange(r)){if(S=this.regexp_eatClassSetOperand(r),!S)return u;S===qe&&(u=qe)}},te.regexp_eatClassSetRange=function(r){var u=r.pos;if(this.regexp_eatClassSetCharacter(r)){var S=r.lastIntValue;if(r.eat(45)&&this.regexp_eatClassSetCharacter(r)){var M=r.lastIntValue;return S!==-1&&M!==-1&&S>M&&r.raise("Range out of order in character class"),!0}r.pos=u}return!1},te.regexp_eatClassSetOperand=function(r){return this.regexp_eatClassSetCharacter(r)?gt:this.regexp_eatClassStringDisjunction(r)||this.regexp_eatNestedClass(r)},te.regexp_eatNestedClass=function(r){var u=r.pos;if(r.eat(91)){var S=r.eat(94),M=this.regexp_classContents(r);if(r.eat(93))return S&&M===qe&&r.raise("Negated character class may contain strings"),M;r.pos=u}if(r.eat(92)){var R=this.regexp_eatCharacterClassEscape(r);if(R)return R;r.pos=u}return null},te.regexp_eatClassStringDisjunction=function(r){var u=r.pos;if(r.eatChars([92,113])){if(r.eat(123)){var S=this.regexp_classStringDisjunctionContents(r);if(r.eat(125))return S}else r.raise("Invalid escape");r.pos=u}return null},te.regexp_classStringDisjunctionContents=function(r){for(var u=this.regexp_classString(r);r.eat(124);)this.regexp_classString(r)===qe&&(u=qe);return u},te.regexp_classString=function(r){for(var u=0;this.regexp_eatClassSetCharacter(r);)u++;return u===1?gt:qe},te.regexp_eatClassSetCharacter=function(r){var u=r.pos;if(r.eat(92))return this.regexp_eatCharacterEscape(r)||this.regexp_eatClassSetReservedPunctuator(r)?!0:r.eat(98)?(r.lastIntValue=8,!0):(r.pos=u,!1);var S=r.current();return S<0||S===r.lookahead()&&dl(S)||pl(S)?!1:(r.advance(),r.lastIntValue=S,!0)};function dl(r){return r===33||r>=35&&r<=38||r>=42&&r<=44||r===46||r>=58&&r<=64||r===94||r===96||r===126}function pl(r){return r===40||r===41||r===45||r===47||r>=91&&r<=93||r>=123&&r<=125}te.regexp_eatClassSetReservedPunctuator=function(r){var u=r.current();return fl(u)?(r.lastIntValue=u,r.advance(),!0):!1};function fl(r){return r===33||r===35||r===37||r===38||r===44||r===45||r>=58&&r<=62||r===64||r===96||r===126}te.regexp_eatClassControlLetter=function(r){var u=r.current();return Ns(u)||u===95?(r.lastIntValue=u%32,r.advance(),!0):!1},te.regexp_eatHexEscapeSequence=function(r){var u=r.pos;if(r.eat(120)){if(this.regexp_eatFixedHexDigits(r,2))return!0;r.switchU&&r.raise("Invalid escape"),r.pos=u}return!1},te.regexp_eatDecimalDigits=function(r){var u=r.pos,S=0;for(r.lastIntValue=0;Ns(S=r.current());)r.lastIntValue=10*r.lastIntValue+(S-48),r.advance();return r.pos!==u};function Ns(r){return r>=48&&r<=57}te.regexp_eatHexDigits=function(r){var u=r.pos,S=0;for(r.lastIntValue=0;Di(S=r.current());)r.lastIntValue=16*r.lastIntValue+Pi(S),r.advance();return r.pos!==u};function Di(r){return r>=48&&r<=57||r>=65&&r<=70||r>=97&&r<=102}function Pi(r){return r>=65&&r<=70?10+(r-65):r>=97&&r<=102?10+(r-97):r-48}te.regexp_eatLegacyOctalEscapeSequence=function(r){if(this.regexp_eatOctalDigit(r)){var u=r.lastIntValue;if(this.regexp_eatOctalDigit(r)){var S=r.lastIntValue;u<=3&&this.regexp_eatOctalDigit(r)?r.lastIntValue=u*64+S*8+r.lastIntValue:r.lastIntValue=u*8+S}else r.lastIntValue=u;return!0}return!1},te.regexp_eatOctalDigit=function(r){var u=r.current();return zi(u)?(r.lastIntValue=u-48,r.advance(),!0):(r.lastIntValue=0,!1)};function zi(r){return r>=48&&r<=55}te.regexp_eatFixedHexDigits=function(r,u){var S=r.pos;r.lastIntValue=0;for(var M=0;M<u;++M){var R=r.current();if(!Di(R))return r.pos=S,!1;r.lastIntValue=16*r.lastIntValue+Pi(R),r.advance()}return!0};var Bs=function(u){this.type=u.type,this.value=u.value,this.start=u.start,this.end=u.end,u.options.locations&&(this.loc=new ce(u,u.startLoc,u.endLoc)),u.options.ranges&&(this.range=[u.start,u.end])},pe=_e.prototype;pe.next=function(r){!r&&this.type.keyword&&this.containsEsc&&this.raiseRecoverable(this.start,"Escape sequence in keyword "+this.type.keyword),this.options.onToken&&this.options.onToken(new Bs(this)),this.lastTokEnd=this.end,this.lastTokStart=this.start,this.lastTokEndLoc=this.endLoc,this.lastTokStartLoc=this.startLoc,this.nextToken()},pe.getToken=function(){return this.next(),new Bs(this)},typeof Symbol<"u"&&(pe[Symbol.iterator]=function(){var r=this;return{next:function(){var u=r.getToken();return{done:u.type===h.eof,value:u}}}}),pe.nextToken=function(){var r=this.curContext();if((!r||!r.preserveSpace)&&this.skipSpace(),this.start=this.pos,this.options.locations&&(this.startLoc=this.curPosition()),this.pos>=this.input.length)return this.finishToken(h.eof);if(r.override)return r.override(this);this.readToken(this.fullCharCodeAtPos())},pe.readToken=function(r){return o(r,this.options.ecmaVersion>=6)||r===92?this.readWord():this.getTokenFromCode(r)},pe.fullCharCodeAtPos=function(){var r=this.input.charCodeAt(this.pos);if(r<=55295||r>=56320)return r;var u=this.input.charCodeAt(this.pos+1);return u<=56319||u>=57344?r:(r<<10)+u-56613888},pe.skipBlockComment=function(){var r=this.options.onComment&&this.curPosition(),u=this.pos,S=this.input.indexOf("*/",this.pos+=2);if(S===-1&&this.raise(this.pos-2,"Unterminated comment"),this.pos=S+2,this.options.locations)for(var M=void 0,R=u;(M=L(this.input,R,this.pos))>-1;)++this.curLine,R=this.lineStart=M;this.options.onComment&&this.options.onComment(!0,this.input.slice(u+2,S),u,this.pos,r,this.curPosition())},pe.skipLineComment=function(r){for(var u=this.pos,S=this.options.onComment&&this.curPosition(),M=this.input.charCodeAt(this.pos+=r);this.pos<this.input.length&&!z(M);)M=this.input.charCodeAt(++this.pos);this.options.onComment&&this.options.onComment(!1,this.input.slice(u+r,this.pos),u,this.pos,S,this.curPosition())},pe.skipSpace=function(){e:for(;this.pos<this.input.length;){var r=this.input.charCodeAt(this.pos);switch(r){case 32:case 160:++this.pos;break;case 13:this.input.charCodeAt(this.pos+1)===10&&++this.pos;case 10:case 8232:case 8233:++this.pos,this.options.locations&&(++this.curLine,this.lineStart=this.pos);break;case 47:switch(this.input.charCodeAt(this.pos+1)){case 42:this.skipBlockComment();break;case 47:this.skipLineComment(2);break;default:break e}break;default:if(r>8&&r<14||r>=5760&&V.test(String.fromCharCode(r)))++this.pos;else break e}}},pe.finishToken=function(r,u){this.end=this.pos,this.options.locations&&(this.endLoc=this.curPosition());var S=this.type;this.type=r,this.value=u,this.updateContext(S)},pe.readToken_dot=function(){var r=this.input.charCodeAt(this.pos+1);if(r>=48&&r<=57)return this.readNumber(!0);var u=this.input.charCodeAt(this.pos+2);return this.options.ecmaVersion>=6&&r===46&&u===46?(this.pos+=3,this.finishToken(h.ellipsis)):(++this.pos,this.finishToken(h.dot))},pe.readToken_slash=function(){var r=this.input.charCodeAt(this.pos+1);return this.exprAllowed?(++this.pos,this.readRegexp()):r===61?this.finishOp(h.assign,2):this.finishOp(h.slash,1)},pe.readToken_mult_modulo_exp=function(r){var u=this.input.charCodeAt(this.pos+1),S=1,M=r===42?h.star:h.modulo;return this.options.ecmaVersion>=7&&r===42&&u===42&&(++S,M=h.starstar,u=this.input.charCodeAt(this.pos+2)),u===61?this.finishOp(h.assign,S+1):this.finishOp(M,S)},pe.readToken_pipe_amp=function(r){var u=this.input.charCodeAt(this.pos+1);return u===r?this.options.ecmaVersion>=12&&this.input.charCodeAt(this.pos+2)===61?this.finishOp(h.assign,3):this.finishOp(r===124?h.logicalOR:h.logicalAND,2):u===61?this.finishOp(h.assign,2):this.finishOp(r===124?h.bitwiseOR:h.bitwiseAND,1)},pe.readToken_caret=function(){return this.input.charCodeAt(this.pos+1)===61?this.finishOp(h.assign,2):this.finishOp(h.bitwiseXOR,1)},pe.readToken_plus_min=function(r){var u=this.input.charCodeAt(this.pos+1);return u===r?u===45&&!this.inModule&&this.input.charCodeAt(this.pos+2)===62&&(this.lastTokEnd===0||F.test(this.input.slice(this.lastTokEnd,this.pos)))?(this.skipLineComment(3),this.skipSpace(),this.nextToken()):this.finishOp(h.incDec,2):u===61?this.finishOp(h.assign,2):this.finishOp(h.plusMin,1)},pe.readToken_lt_gt=function(r){var u=this.input.charCodeAt(this.pos+1),S=1;return u===r?(S=r===62&&this.input.charCodeAt(this.pos+2)===62?3:2,this.input.charCodeAt(this.pos+S)===61?this.finishOp(h.assign,S+1):this.finishOp(h.bitShift,S)):u===33&&r===60&&!this.inModule&&this.input.charCodeAt(this.pos+2)===45&&this.input.charCodeAt(this.pos+3)===45?(this.skipLineComment(4),this.skipSpace(),this.nextToken()):(u===61&&(S=2),this.finishOp(h.relational,S))},pe.readToken_eq_excl=function(r){var u=this.input.charCodeAt(this.pos+1);return u===61?this.finishOp(h.equality,this.input.charCodeAt(this.pos+2)===61?3:2):r===61&&u===62&&this.options.ecmaVersion>=6?(this.pos+=2,this.finishToken(h.arrow)):this.finishOp(r===61?h.eq:h.prefix,1)},pe.readToken_question=function(){var r=this.options.ecmaVersion;if(r>=11){var u=this.input.charCodeAt(this.pos+1);if(u===46){var S=this.input.charCodeAt(this.pos+2);if(S<48||S>57)return this.finishOp(h.questionDot,2)}if(u===63)return r>=12&&this.input.charCodeAt(this.pos+2)===61?this.finishOp(h.assign,3):this.finishOp(h.coalesce,2)}return this.finishOp(h.question,1)},pe.readToken_numberSign=function(){var r=this.options.ecmaVersion,u=35;if(r>=13&&(++this.pos,u=this.fullCharCodeAtPos(),o(u,!0)||u===92))return this.finishToken(h.privateId,this.readWord1());this.raise(this.pos,"Unexpected character '"+he(u)+"'")},pe.getTokenFromCode=function(r){switch(r){case 46:return this.readToken_dot();case 40:return++this.pos,this.finishToken(h.parenL);case 41:return++this.pos,this.finishToken(h.parenR);case 59:return++this.pos,this.finishToken(h.semi);case 44:return++this.pos,this.finishToken(h.comma);case 91:return++this.pos,this.finishToken(h.bracketL);case 93:return++this.pos,this.finishToken(h.bracketR);case 123:return++this.pos,this.finishToken(h.braceL);case 125:return++this.pos,this.finishToken(h.braceR);case 58:return++this.pos,this.finishToken(h.colon);case 96:if(this.options.ecmaVersion<6)break;return++this.pos,this.finishToken(h.backQuote);case 48:var u=this.input.charCodeAt(this.pos+1);if(u===120||u===88)return this.readRadixNumber(16);if(this.options.ecmaVersion>=6){if(u===111||u===79)return this.readRadixNumber(8);if(u===98||u===66)return this.readRadixNumber(2)}case 49:case 50:case 51:case 52:case 53:case 54:case 55:case 56:case 57:return this.readNumber(!1);case 34:case 39:return this.readString(r);case 47:return this.readToken_slash();case 37:case 42:return this.readToken_mult_modulo_exp(r);case 124:case 38:return this.readToken_pipe_amp(r);case 94:return this.readToken_caret();case 43:case 45:return this.readToken_plus_min(r);case 60:case 62:return this.readToken_lt_gt(r);case 61:case 33:return this.readToken_eq_excl(r);case 63:return this.readToken_question();case 126:return this.finishOp(h.prefix,1);case 35:return this.readToken_numberSign()}this.raise(this.pos,"Unexpected character '"+he(r)+"'")},pe.finishOp=function(r,u){var S=this.input.slice(this.pos,this.pos+u);return this.pos+=u,this.finishToken(r,S)},pe.readRegexp=function(){for(var r,u,S=this.pos;;){this.pos>=this.input.length&&this.raise(S,"Unterminated regular expression");var M=this.input.charAt(this.pos);if(F.test(M)&&this.raise(S,"Unterminated regular expression"),r)r=!1;else{if(M==="[")u=!0;else if(M==="]"&&u)u=!1;else if(M==="/"&&!u)break;r=M==="\\"}++this.pos}var R=this.input.slice(S,this.pos);++this.pos;var K=this.pos,B=this.readWord1();this.containsEsc&&this.unexpected(K);var J=this.regexpState||(this.regexpState=new at(this));J.reset(S,R,B),this.validateRegExpFlags(J),this.validateRegExpPattern(J);var Q=null;try{Q=new RegExp(R,B)}catch{}return this.finishToken(h.regexp,{pattern:R,flags:B,value:Q})},pe.readInt=function(r,u,S){for(var M=this.options.ecmaVersion>=12&&u===void 0,R=S&&this.input.charCodeAt(this.pos)===48,K=this.pos,B=0,J=0,Q=0,de=u??1/0;Q<de;++Q,++this.pos){var le=this.input.charCodeAt(this.pos),$e=void 0;if(M&&le===95){R&&this.raiseRecoverable(this.pos,"Numeric separator is not allowed in legacy octal numeric literals"),J===95&&this.raiseRecoverable(this.pos,"Numeric separator must be exactly one underscore"),Q===0&&this.raiseRecoverable(this.pos,"Numeric separator is not allowed at the first of digits"),J=le;continue}if(le>=97?$e=le-97+10:le>=65?$e=le-65+10:le>=48&&le<=57?$e=le-48:$e=1/0,$e>=r)break;J=le,B=B*r+$e}return M&&J===95&&this.raiseRecoverable(this.pos-1,"Numeric separator is not allowed at the last of digits"),this.pos===K||u!=null&&this.pos-K!==u?null:B};function ml(r,u){return u?parseInt(r,8):parseFloat(r.replace(/_/g,""))}function Oi(r){return typeof BigInt!="function"?null:BigInt(r.replace(/_/g,""))}pe.readRadixNumber=function(r){var u=this.pos;this.pos+=2;var S=this.readInt(r);return S==null&&this.raise(this.start+2,"Expected number in radix "+r),this.options.ecmaVersion>=11&&this.input.charCodeAt(this.pos)===110?(S=Oi(this.input.slice(u,this.pos)),++this.pos):o(this.fullCharCodeAtPos())&&this.raise(this.pos,"Identifier directly after number"),this.finishToken(h.num,S)},pe.readNumber=function(r){var u=this.pos;!r&&this.readInt(10,void 0,!0)===null&&this.raise(u,"Invalid number");var S=this.pos-u>=2&&this.input.charCodeAt(u)===48;S&&this.strict&&this.raise(u,"Invalid number");var M=this.input.charCodeAt(this.pos);if(!S&&!r&&this.options.ecmaVersion>=11&&M===110){var R=Oi(this.input.slice(u,this.pos));return++this.pos,o(this.fullCharCodeAtPos())&&this.raise(this.pos,"Identifier directly after number"),this.finishToken(h.num,R)}S&&/[89]/.test(this.input.slice(u,this.pos))&&(S=!1),M===46&&!S&&(++this.pos,this.readInt(10),M=this.input.charCodeAt(this.pos)),(M===69||M===101)&&!S&&(M=this.input.charCodeAt(++this.pos),(M===43||M===45)&&++this.pos,this.readInt(10)===null&&this.raise(u,"Invalid number")),o(this.fullCharCodeAtPos())&&this.raise(this.pos,"Identifier directly after number");var K=ml(this.input.slice(u,this.pos),S);return this.finishToken(h.num,K)},pe.readCodePoint=function(){var r=this.input.charCodeAt(this.pos),u;if(r===123){this.options.ecmaVersion<6&&this.unexpected();var S=++this.pos;u=this.readHexChar(this.input.indexOf("}",this.pos)-this.pos),++this.pos,u>1114111&&this.invalidStringToken(S,"Code point out of bounds")}else u=this.readHexChar(4);return u},pe.readString=function(r){for(var u="",S=++this.pos;;){this.pos>=this.input.length&&this.raise(this.start,"Unterminated string constant");var M=this.input.charCodeAt(this.pos);if(M===r)break;M===92?(u+=this.input.slice(S,this.pos),u+=this.readEscapedChar(!1),S=this.pos):M===8232||M===8233?(this.options.ecmaVersion<10&&this.raise(this.start,"Unterminated string constant"),++this.pos,this.options.locations&&(this.curLine++,this.lineStart=this.pos)):(z(M)&&this.raise(this.start,"Unterminated string constant"),++this.pos)}return u+=this.input.slice(S,this.pos++),this.finishToken(h.string,u)};var Ri={};pe.tryReadTemplateToken=function(){this.inTemplateElement=!0;try{this.readTmplToken()}catch(r){if(r===Ri)this.readInvalidTemplateToken();else throw r}this.inTemplateElement=!1},pe.invalidStringToken=function(r,u){if(this.inTemplateElement&&this.options.ecmaVersion>=9)throw Ri;this.raise(r,u)},pe.readTmplToken=function(){for(var r="",u=this.pos;;){this.pos>=this.input.length&&this.raise(this.start,"Unterminated template");var S=this.input.charCodeAt(this.pos);if(S===96||S===36&&this.input.charCodeAt(this.pos+1)===123)return this.pos===this.start&&(this.type===h.template||this.type===h.invalidTemplate)?S===36?(this.pos+=2,this.finishToken(h.dollarBraceL)):(++this.pos,this.finishToken(h.backQuote)):(r+=this.input.slice(u,this.pos),this.finishToken(h.template,r));if(S===92)r+=this.input.slice(u,this.pos),r+=this.readEscapedChar(!0),u=this.pos;else if(z(S)){switch(r+=this.input.slice(u,this.pos),++this.pos,S){case 13:this.input.charCodeAt(this.pos)===10&&++this.pos;case 10:r+=`
`;break;default:r+=String.fromCharCode(S);break}this.options.locations&&(++this.curLine,this.lineStart=this.pos),u=this.pos}else++this.pos}},pe.readInvalidTemplateToken=function(){for(;this.pos<this.input.length;this.pos++)switch(this.input[this.pos]){case"\\":++this.pos;break;case"$":if(this.input[this.pos+1]!=="{")break;case"`":return this.finishToken(h.invalidTemplate,this.input.slice(this.start,this.pos));case"\r":this.input[this.pos+1]===`
`&&++this.pos;case`
`:case"\u2028":case"\u2029":++this.curLine,this.lineStart=this.pos+1;break}this.raise(this.start,"Unterminated template")},pe.readEscapedChar=function(r){var u=this.input.charCodeAt(++this.pos);switch(++this.pos,u){case 110:return`
`;case 114:return"\r";case 120:return String.fromCharCode(this.readHexChar(2));case 117:return he(this.readCodePoint());case 116:return"	";case 98:return"\b";case 118:return"\v";case 102:return"\f";case 13:this.input.charCodeAt(this.pos)===10&&++this.pos;case 10:return this.options.locations&&(this.lineStart=this.pos,++this.curLine),"";case 56:case 57:if(this.strict&&this.invalidStringToken(this.pos-1,"Invalid escape sequence"),r){var S=this.pos-1;this.invalidStringToken(S,"Invalid escape sequence in template string")}default:if(u>=48&&u<=55){var M=this.input.substr(this.pos-1,3).match(/^[0-7]+/)[0],R=parseInt(M,8);return R>255&&(M=M.slice(0,-1),R=parseInt(M,8)),this.pos+=M.length-1,u=this.input.charCodeAt(this.pos),(M!=="0"||u===56||u===57)&&(this.strict||r)&&this.invalidStringToken(this.pos-1-M.length,r?"Octal literal in template string":"Octal literal in strict mode"),String.fromCharCode(R)}return z(u)?(this.options.locations&&(this.lineStart=this.pos,++this.curLine),""):String.fromCharCode(u)}},pe.readHexChar=function(r){var u=this.pos,S=this.readInt(16,r);return S===null&&this.invalidStringToken(u,"Bad character escape sequence"),S},pe.readWord1=function(){this.containsEsc=!1;for(var r="",u=!0,S=this.pos,M=this.options.ecmaVersion>=6;this.pos<this.input.length;){var R=this.fullCharCodeAtPos();if(l(R,M))this.pos+=R<=65535?1:2;else if(R===92){this.containsEsc=!0,r+=this.input.slice(S,this.pos);var K=this.pos;this.input.charCodeAt(++this.pos)!==117&&this.invalidStringToken(this.pos,"Expecting Unicode escape sequence \\uXXXX"),++this.pos;var B=this.readCodePoint();(u?o:l)(B,M)||this.invalidStringToken(K,"Invalid Unicode escape"),r+=he(B),S=this.pos}else break;u=!1}return r+this.input.slice(S,this.pos)},pe.readWord=function(){var r=this.readWord1(),u=h.name;return this.keywords.test(r)&&(u=b[r]),this.finishToken(u,r)};var Fi="8.14.0";_e.acorn={Parser:_e,version:Fi,defaultOptions:ue,Position:re,SourceLocation:ce,getLineInfo:Ve,Node:ms,TokenType:x,tokTypes:h,keywordTypes:b,TokContext:je,tokContexts:be,isIdentifierChar:l,isIdentifierStart:o,Token:Bs,isNewLine:z,lineBreak:F,lineBreakG:O,nonASCIIwhitespace:V};function gl(r,u){return _e.parse(r,u)}function yl(r,u,S){return _e.parseExpressionAt(r,u,S)}function xl(r,u){return _e.tokenizer(r,u)}D.Node=ms,D.Parser=_e,D.Position=re,D.SourceLocation=ce,D.TokContext=je,D.Token=Bs,D.TokenType=x,D.defaultOptions=ue,D.getLineInfo=Ve,D.isIdentifierChar=l,D.isIdentifierStart=o,D.isNewLine=z,D.keywordTypes=b,D.lineBreak=F,D.lineBreakG=O,D.nonASCIIwhitespace=V,D.parse=gl,D.parseExpressionAt=yl,D.tokContexts=be,D.tokTypes=h,D.tokenizer=xl,D.version=Fi})}),a=s((j,G)=>{var D=class{constructor(_,c){this.value=_,Array.isArray(c)?this.size=c:(this.size=new Int32Array(3),c.z?this.size=new Int32Array([c.x,c.y,c.z]):c.y?this.size=new Int32Array([c.x,c.y]):this.size=new Int32Array([c.x]));const[d,E,$]=this.size;if($){if(this.value.length!==d*E*$)throw new Error(`Input size ${this.value.length} does not match ${d} * ${E} * ${$} = ${E*d*$}`)}else if(E){if(this.value.length!==d*E)throw new Error(`Input size ${this.value.length} does not match ${d} * ${E} = ${E*d}`)}else if(this.value.length!==d)throw new Error(`Input size ${this.value.length} does not match ${d}`)}toArray(){const{utils:_}=m(),[c,d,E]=this.size;return E?_.erectMemoryOptimized3DFloat(this.value.subarray?this.value:new Float32Array(this.value),c,d,E):d?_.erectMemoryOptimized2DFloat(this.value.subarray?this.value:new Float32Array(this.value),c,d):this.value}};function I(_,c){return new D(_,c)}G.exports={Input:D,input:I}}),f=s((j,G)=>{var D=class{constructor(I){const{texture:_,size:c,dimensions:d,output:E,context:$,type:P="NumberTexture",kernel:y,internalFormat:p,textureFormat:g}=I;if(!E)throw new Error('settings property "output" required.');if(!$)throw new Error('settings property "context" required.');if(!_)throw new Error('settings property "texture" required.');if(!y)throw new Error('settings property "kernel" required.');this.texture=_,_._refs?_._refs++:_._refs=1,this.size=c,this.dimensions=d,this.output=E,this.context=$,this.kernel=y,this.type=P,this._deleted=!1,this.internalFormat=p,this.textureFormat=g}toArray(){throw new Error(`Not implemented on ${this.constructor.name}`)}clone(){throw new Error(`Not implemented on ${this.constructor.name}`)}delete(){throw new Error(`Not implemented on ${this.constructor.name}`)}clear(){throw new Error(`Not implemented on ${this.constructor.name}`)}};G.exports={Texture:D}}),m=s((j,G)=>{const D=i(),{Input:I}=a(),{Texture:_}=f(),c=/function ([^(]*)/,d=/((\/\/.*$)|(\/\*[\s\S]*?\*\/))/gm,E=/([^\s,]+)/g,$={systemEndianness(){return g},getSystemEndianness(){const k=new ArrayBuffer(4),o=new Uint32Array(k),l=new Uint8Array(k);if(o[0]=3735928559,l[0]===239)return"LE";if(l[0]===222)return"BE";throw new Error("unknown endianness")},isFunction(k){return typeof k=="function"},isFunctionString(k){return typeof k=="string"?k.slice(0,8).toLowerCase()==="function":!1},getFunctionNameFromString(k){const o=c.exec(k);return!o||o.length===0?null:o[1].trim()},getFunctionBodyFromString(k){return k.substring(k.indexOf("{")+1,k.lastIndexOf("}"))},getArgumentNamesFromString(k){const o=k.replace(d,"");let l=o.slice(o.indexOf("(")+1,o.indexOf(")")).match(E);return l===null&&(l=[]),l},clone(k){if(k===null||typeof k!="object"||k.hasOwnProperty("isActiveClone"))return k;const o=k.constructor();for(let l in k)Object.prototype.hasOwnProperty.call(k,l)&&(k.isActiveClone=null,o[l]=$.clone(k[l]),delete k.isActiveClone);return o},isArray(k){return!isNaN(k.length)},typeFitsValue(k,o){if(typeof k!="string"||o===null||o===void 0||o.type)return!0;switch(k){case"Input":return o instanceof I;case"Boolean":return typeof o=="boolean";case"Number":case"Integer":case"Float":return typeof o=="number"}return k.indexOf("Texture")!==-1?!!o.type:k.indexOf("Array")===0?$.isArray(o):!0},getVariableType(k,o){if($.isArray(k))return k.length>0&&k[0].nodeName==="IMG"?"HTMLImageArray":"Array";switch(k.constructor){case Boolean:return"Boolean";case Number:return o&&Number.isInteger(k)?"Integer":"Float";case _:return k.type;case I:return"Input"}if("nodeName"in k)switch(k.nodeName){case"IMG":return"HTMLImage";case"CANVAS":return"HTMLImage";case"VIDEO":return"HTMLVideo"}else{if(k.hasOwnProperty("type"))return k.type;if(typeof OffscreenCanvas<"u"&&k instanceof OffscreenCanvas)return"OffscreenCanvas";if(typeof ImageBitmap<"u"&&k instanceof ImageBitmap)return"ImageBitmap";if(typeof ImageData<"u"&&k instanceof ImageData)return"ImageData"}return"Unknown"},getKernelTextureSize(k,o){let[l,x,w]=o,v=(l||1)*(x||1)*(w||1);return k.optimizeFloatMemory&&k.precision==="single"&&(l=v=Math.ceil(v/4)),x>1&&l*x===v?new Int32Array([l,x]):$.closestSquareDimensions(v)},closestSquareDimensions(k){const o=Math.sqrt(k);let l=Math.ceil(o),x=Math.floor(o);for(;l*x<k;)l--,x=Math.ceil(k/l);return new Int32Array([x,Math.ceil(k/x)])},getMemoryOptimizedFloatTextureSize(k,o){const l=$.roundTo((k[0]||1)*(k[1]||1)*(k[2]||1)*(k[3]||1),4)/o;return $.closestSquareDimensions(l)},getMemoryOptimizedPackedTextureSize(k,o){const[l,x,w]=k,v=$.roundTo((l||1)*(x||1)*(w||1),4)/(4/o);return $.closestSquareDimensions(v)},roundTo(k,o){return Math.floor((k+o-1)/o)*o},getDimensions(k,o){let l;if($.isArray(k)){const x=[];let w=k;for(;$.isArray(w);)x.push(w.length),w=w[0];l=x.reverse()}else if(k instanceof _)l=k.output;else if(k instanceof I)l=k.size;else throw new Error(`Unknown dimensions of ${k}`);if(o)for(l=Array.from(l);l.length<3;)l.push(1);return new Int32Array(l)},flatten2dArrayTo(k,o){let l=0;for(let x=0;x<k.length;x++)o.set(k[x],l),l+=k[x].length},flatten3dArrayTo(k,o){let l=0;for(let x=0;x<k.length;x++)for(let w=0;w<k[x].length;w++)o.set(k[x][w],l),l+=k[x][w].length},flatten4dArrayTo(k,o){let l=0;for(let x=0;x<k.length;x++)for(let w=0;w<k[x].length;w++)for(let v=0;v<k[x][w].length;v++)o.set(k[x][w][v],l),l+=k[x][w][v].length},flattenTo(k,o){$.isArray(k[0])?$.isArray(k[0][0])?$.isArray(k[0][0][0])?$.flatten4dArrayTo(k,o):$.flatten3dArrayTo(k,o):$.flatten2dArrayTo(k,o):o.set(k)},splitArray(k,o){const l=[];for(let x=0;x<k.length;x+=o)l.push(new k.constructor(k.buffer,x*4+k.byteOffset,o));return l},getAstString(k,o){const l=Array.isArray(k)?k:k.split(/\r?\n/g),x=o.loc.start,w=o.loc.end,v=[];if(x.line===w.line)v.push(l[x.line-1].substring(x.column,w.column));else{v.push(l[x.line-1].slice(x.column));for(let C=x.line;C<w.line;C++)v.push(l[C]);v.push(l[w.line-1].slice(0,w.column))}return v.join(`
`)},allPropertiesOf(k){const o=[];do o.push.apply(o,Object.getOwnPropertyNames(k));while(k=Object.getPrototypeOf(k));return o},linesToString(k){return k.length>0?k.join(`;
`)+`;
`:`
`},warnDeprecated(k,o,l){console.warn(l?`You are using a deprecated ${k} "${o}". It has been replaced with "${l}". Fixing, but please upgrade as it will soon be removed.`:`You are using a deprecated ${k} "${o}". It has been removed. Fixing, but please upgrade as it will soon be removed.`)},flipPixels:(k,o,l)=>{const x=l/2|0,w=o*4,v=new Uint8ClampedArray(o*4),C=k.slice(0);for(let b=0;b<x;++b){const T=b*w,h=(l-b-1)*w;v.set(C.subarray(T,T+w)),C.copyWithin(T,h,h+w),C.set(v,h)}return C},erectPackedFloat:(k,o)=>k.subarray(0,o),erect2DPackedFloat:(k,o,l)=>{const x=new Array(l);for(let w=0;w<l;w++){const v=w*o,C=v+o;x[w]=k.subarray(v,C)}return x},erect3DPackedFloat:(k,o,l,x)=>{const w=new Array(x);for(let v=0;v<x;v++){const C=new Array(l);for(let b=0;b<l;b++){const T=v*l*o+b*o,h=T+o;C[b]=k.subarray(T,h)}w[v]=C}return w},erectMemoryOptimizedFloat:(k,o)=>k.subarray(0,o),erectMemoryOptimized2DFloat:(k,o,l)=>{const x=new Array(l);for(let w=0;w<l;w++){const v=w*o;x[w]=k.subarray(v,v+o)}return x},erectMemoryOptimized3DFloat:(k,o,l,x)=>{const w=new Array(x);for(let v=0;v<x;v++){const C=new Array(l);for(let b=0;b<l;b++){const T=v*l*o+b*o;C[b]=k.subarray(T,T+o)}w[v]=C}return w},erectFloat:(k,o)=>{const l=new Float32Array(o);let x=0;for(let w=0;w<o;w++)l[w]=k[x],x+=4;return l},erect2DFloat:(k,o,l)=>{const x=new Array(l);let w=0;for(let v=0;v<l;v++){const C=new Float32Array(o);for(let b=0;b<o;b++)C[b]=k[w],w+=4;x[v]=C}return x},erect3DFloat:(k,o,l,x)=>{const w=new Array(x);let v=0;for(let C=0;C<x;C++){const b=new Array(l);for(let T=0;T<l;T++){const h=new Float32Array(o);for(let F=0;F<o;F++)h[F]=k[v],v+=4;b[T]=h}w[C]=b}return w},erectArray2:(k,o)=>{const l=new Array(o),x=o*4;let w=0;for(let v=0;v<x;v+=4)l[w++]=k.subarray(v,v+2);return l},erect2DArray2:(k,o,l)=>{const x=new Array(l),w=o*4;for(let v=0;v<l;v++){const C=new Array(o),b=v*w;let T=0;for(let h=0;h<w;h+=4)C[T++]=k.subarray(h+b,h+b+2);x[v]=C}return x},erect3DArray2:(k,o,l,x)=>{const w=o*4,v=new Array(x);for(let C=0;C<x;C++){const b=new Array(l);for(let T=0;T<l;T++){const h=new Array(o),F=C*w*l+T*w;let O=0;for(let z=0;z<w;z+=4)h[O++]=k.subarray(z+F,z+F+2);b[T]=h}v[C]=b}return v},erectArray3:(k,o)=>{const l=new Array(o),x=o*4;let w=0;for(let v=0;v<x;v+=4)l[w++]=k.subarray(v,v+3);return l},erect2DArray3:(k,o,l)=>{const x=o*4,w=new Array(l);for(let v=0;v<l;v++){const C=new Array(o),b=v*x;let T=0;for(let h=0;h<x;h+=4)C[T++]=k.subarray(h+b,h+b+3);w[v]=C}return w},erect3DArray3:(k,o,l,x)=>{const w=o*4,v=new Array(x);for(let C=0;C<x;C++){const b=new Array(l);for(let T=0;T<l;T++){const h=new Array(o),F=C*w*l+T*w;let O=0;for(let z=0;z<w;z+=4)h[O++]=k.subarray(z+F,z+F+3);b[T]=h}v[C]=b}return v},erectArray4:(k,o)=>{const l=new Array(k),x=o*4;let w=0;for(let v=0;v<x;v+=4)l[w++]=k.subarray(v,v+4);return l},erect2DArray4:(k,o,l)=>{const x=o*4,w=new Array(l);for(let v=0;v<l;v++){const C=new Array(o),b=v*x;let T=0;for(let h=0;h<x;h+=4)C[T++]=k.subarray(h+b,h+b+4);w[v]=C}return w},erect3DArray4:(k,o,l,x)=>{const w=o*4,v=new Array(x);for(let C=0;C<x;C++){const b=new Array(l);for(let T=0;T<l;T++){const h=new Array(o),F=C*w*l+T*w;let O=0;for(let z=0;z<w;z+=4)h[O++]=k.subarray(z+F,z+F+4);b[T]=h}v[C]=b}return v},flattenFunctionToString:(k,o)=>{const{findDependency:l,thisLookup:x,doNotDefine:w}=o;let v=o.flattened;v||(v=o.flattened={});const C=D.parse(k,{ecmaVersion:2020}),b=[];let T=0;function h(O){if(Array.isArray(O)){const z=[];for(let L=0;L<O.length;L++)z.push(h(O[L]));return z.join("")}switch(O.type){case"Program":return h(O.body)+(O.body[0].type==="VariableDeclaration"?";":"");case"FunctionDeclaration":return`function ${O.id.name}(${O.params.map(h).join(", ")}) ${h(O.body)}`;case"BlockStatement":{const L=[];T+=2;for(let V=0;V<O.body.length;V++){const U=h(O.body[V]);U&&L.push(" ".repeat(T)+U,`;
`)}return T-=2,`{
${L.join("")}}`}case"VariableDeclaration":const z=$.normalizeDeclarations(O).map(h).filter(L=>L!==null);return z.length<1?"":`${O.kind} ${z.join(",")}`;case"VariableDeclarator":return O.init?O.init.object&&O.init.object.type==="ThisExpression"?x(O.init.property.name,!0)?`${O.id.name} = ${h(O.init)}`:null:`${O.id.name} = ${h(O.init)}`:O.id.name;case"CallExpression":if(O.callee.property.name==="subarray")return`${h(O.callee.object)}.${h(O.callee.property)}(${O.arguments.map(L=>h(L)).join(", ")})`;if(O.callee.object.name==="gl"||O.callee.object.name==="context")return`${h(O.callee.object)}.${h(O.callee.property)}(${O.arguments.map(L=>h(L)).join(", ")})`;if(O.callee.object.type==="ThisExpression")return b.push(l("this",O.callee.property.name)),`${O.callee.property.name}(${O.arguments.map(L=>h(L)).join(", ")})`;if(O.callee.object.name){const L=l(O.callee.object.name,O.callee.property.name);return L===null?`${O.callee.object.name}.${O.callee.property.name}(${O.arguments.map(V=>h(V)).join(", ")})`:(b.push(L),`${O.callee.property.name}(${O.arguments.map(V=>h(V)).join(", ")})`)}else{if(O.callee.object.type==="MemberExpression")return`${h(O.callee.object)}.${O.callee.property.name}(${O.arguments.map(L=>h(L)).join(", ")})`;throw new Error("unknown ast.callee")}case"ReturnStatement":return`return ${h(O.argument)}`;case"BinaryExpression":return`(${h(O.left)}${O.operator}${h(O.right)})`;case"UnaryExpression":return O.prefix?`${O.operator} ${h(O.argument)}`:`${h(O.argument)} ${O.operator}`;case"ExpressionStatement":return`${h(O.expression)}`;case"SequenceExpression":return`(${h(O.expressions)})`;case"ArrowFunctionExpression":return`(${O.params.map(h).join(", ")}) => ${h(O.body)}`;case"Literal":return O.raw;case"Identifier":return O.name;case"MemberExpression":return O.object.type==="ThisExpression"?x(O.property.name):O.computed?`${h(O.object)}[${h(O.property)}]`:h(O.object)+"."+h(O.property);case"ThisExpression":return"this";case"NewExpression":return`new ${h(O.callee)}(${O.arguments.map(L=>h(L)).join(", ")})`;case"ForStatement":return`for (${h(O.init)};${h(O.test)};${h(O.update)}) ${h(O.body)}`;case"AssignmentExpression":return`${h(O.left)}${O.operator}${h(O.right)}`;case"UpdateExpression":return`${h(O.argument)}${O.operator}`;case"IfStatement":{const L=h(O.consequent);if(!O.alternate)return`if (${h(O.test)}) ${L}`;const V=O.consequent.type==="BlockStatement"?"":";";return`if (${h(O.test)}) ${L}${V} else ${h(O.alternate)}`}case"ThrowStatement":return`throw ${h(O.argument)}`;case"ObjectPattern":return O.properties.map(h).join(", ");case"ArrayPattern":return O.elements.map(h).join(", ");case"DebuggerStatement":return"debugger;";case"ConditionalExpression":return`${h(O.test)}?${h(O.consequent)}:${h(O.alternate)}`;case"Property":if(O.kind==="init")return h(O.key)}throw new Error(`unhandled ast.type of ${O.type}`)}const F=h(C);if(b.length>0){const O=[];for(let z=0;z<b.length;z++){const L=b[z];v[L]||(v[L]=!0),L&&O.push($.flattenFunctionToString(L,o)+`
`)}return O.join("")+F}return F},normalizeDeclarations:k=>{if(k.type!=="VariableDeclaration")throw new Error('Ast is not of type "VariableDeclaration"');const o=[];for(let l=0;l<k.declarations.length;l++){const x=k.declarations[l];if(x.id&&x.id.type==="ObjectPattern"&&x.id.properties){const{properties:w}=x.id;for(let v=0;v<w.length;v++){const C=w[v];if(C.value.type==="ObjectPattern"&&C.value.properties)for(let b=0;b<C.value.properties.length;b++){const T=C.value.properties[b];if(T.type==="Property")o.push({type:"VariableDeclarator",id:{type:"Identifier",name:T.key.name},init:{type:"MemberExpression",object:{type:"MemberExpression",object:x.init,property:{type:"Identifier",name:C.key.name},computed:!1},property:{type:"Identifier",name:T.key.name},computed:!1}});else throw new Error("unexpected state")}else if(C.value.type==="Identifier")o.push({type:"VariableDeclarator",id:{type:"Identifier",name:C.value&&C.value.name?C.value.name:C.key.name},init:{type:"MemberExpression",object:x.init,property:{type:"Identifier",name:C.key.name},computed:!1}});else throw new Error("unexpected state")}}else if(x.id&&x.id.type==="ArrayPattern"&&x.id.elements){const{elements:w}=x.id;for(let v=0;v<w.length;v++){const C=w[v];if(C.type==="Identifier")o.push({type:"VariableDeclarator",id:{type:"Identifier",name:C.name},init:{type:"MemberExpression",object:x.init,property:{type:"Literal",value:v,raw:v.toString(),start:C.start,end:C.end},computed:!0}});else throw new Error("unexpected state")}}else o.push(x)}return o},splitHTMLImageToRGB:(k,o)=>{const l=k.createKernel(function(b){return b[this.thread.y][this.thread.x].r*255},{output:[o.width,o.height],precision:"unsigned",argumentTypes:{a:"HTMLImage"}}),x=k.createKernel(function(b){return b[this.thread.y][this.thread.x].g*255},{output:[o.width,o.height],precision:"unsigned",argumentTypes:{a:"HTMLImage"}}),w=k.createKernel(function(b){return b[this.thread.y][this.thread.x].b*255},{output:[o.width,o.height],precision:"unsigned",argumentTypes:{a:"HTMLImage"}}),v=k.createKernel(function(b){return b[this.thread.y][this.thread.x].a*255},{output:[o.width,o.height],precision:"unsigned",argumentTypes:{a:"HTMLImage"}}),C=[l(o),x(o),w(o),v(o)];return C.rKernel=l,C.gKernel=x,C.bKernel=w,C.aKernel=v,C.gpu=k,C},splitRGBAToCanvases:(k,o,l,x)=>{const w=k.createKernel(function(T){const h=T[this.thread.y][this.thread.x];this.color(h.r/255,0,0,255)},{output:[l,x],graphical:!0,argumentTypes:{v:"Array2D(4)"}});w(o);const v=k.createKernel(function(T){const h=T[this.thread.y][this.thread.x];this.color(0,h.g/255,0,255)},{output:[l,x],graphical:!0,argumentTypes:{v:"Array2D(4)"}});v(o);const C=k.createKernel(function(T){const h=T[this.thread.y][this.thread.x];this.color(0,0,h.b/255,255)},{output:[l,x],graphical:!0,argumentTypes:{v:"Array2D(4)"}});C(o);const b=k.createKernel(function(T){const h=T[this.thread.y][this.thread.x];this.color(255,255,255,h.a/255)},{output:[l,x],graphical:!0,argumentTypes:{v:"Array2D(4)"}});return b(o),[w.canvas,v.canvas,C.canvas,b.canvas]},getMinifySafeName:k=>{try{const{init:o}=D.parse(`const value = ${k.toString()}`,{ecmaVersion:2020}).body[0].declarations[0];return o.body.name||o.body.body[0].argument.name}catch{throw new Error("Unrecognized function type.  Please use `() => yourFunctionVariableHere` or function() { return yourFunctionVariableHere; }")}},sanitizeName:function(k){return P.test(k)&&(k=k.replace(P,"S_S")),y.test(k)?k=k.replace(y,"U_U"):p.test(k)&&(k=k.replace(p,"u_u")),k}},P=/\$/,y=/__/,p=/_/,g=$.getSystemEndianness();G.exports={utils:$}}),A=s((j,G)=>{const{utils:D}=m(),{Input:I}=a();var _=class{static get isSupported(){throw new Error(`"isSupported" not implemented on ${this.name}`)}static isContextMatch(d){throw new Error(`"isContextMatch" not implemented on ${this.name}`)}static getFeatures(){throw new Error(`"getFeatures" not implemented on ${this.name}`)}static destroyContext(d){throw new Error(`"destroyContext" called on ${this.name}`)}static nativeFunctionArguments(){throw new Error(`"nativeFunctionArguments" called on ${this.name}`)}static nativeFunctionReturnType(){throw new Error(`"nativeFunctionReturnType" called on ${this.name}`)}static combineKernels(){throw new Error(`"combineKernels" called on ${this.name}`)}constructor(d,E){if(typeof d!="object"){if(typeof d!="string")throw new Error("source not a string");if(!D.isFunctionString(d))throw new Error("source not a function string")}this.useLegacyEncoder=!1,this.fallbackRequested=!1,this.onRequestFallback=null,this.onRequestSwitchKernel=null,this.argumentNames=typeof d=="string"?D.getArgumentNamesFromString(d):null,this.argumentTypes=null,this.argumentSizes=null,this.argumentBitRatios=null,this.kernelArguments=null,this.kernelConstants=null,this.forceUploadKernelConstants=null,this.source=d,this.output=null,this.debug=!1,this.graphical=!1,this.loopMaxIterations=0,this.constants=null,this.constantTypes=null,this.constantBitRatios=null,this.dynamicArguments=!1,this.dynamicOutput=!1,this.canvas=null,this.context=null,this.checkContext=null,this.gpu=null,this.functions=null,this.nativeFunctions=null,this.injectedNative=null,this.subKernels=null,this.validate=!0,this.immutable=!1,this.pipeline=!1,this.asyncMode=!1,this.precision=null,this.tactic=null,this.plugins=null,this.returnType=null,this.leadingReturnStatement=null,this.followingReturnStatement=null,this.optimizeFloatMemory=null,this.strictIntegers=!1,this.fixIntegerDivisionAccuracy=null,this.randomSeed=null,this.built=!1,this.signature=null,this.switchingKernels=null}mergeSettings(d){for(let E in d)if(!(!d.hasOwnProperty(E)||!this.hasOwnProperty(E))){switch(E){case"output":if(!Array.isArray(d.output)){this.setOutput(d.output);continue}break;case"functions":this.functions=[];for(let $=0;$<d.functions.length;$++)this.addFunction(d.functions[$]);continue;case"graphical":d[E]&&!d.hasOwnProperty("precision")&&(this.precision="unsigned"),this[E]=d[E];continue;case"nativeFunctions":if(!d.nativeFunctions)continue;this.nativeFunctions=[];for(let $=0;$<d.nativeFunctions.length;$++){const P=d.nativeFunctions[$],{name:y,source:p}=P;this.addNativeFunction(y,p,P)}continue}this[E]=d[E]}this.canvas||(this.canvas=this.initCanvas()),this.context||(this.context=this.initContext()),this.plugins||(this.plugins=this.initPlugins(d))}build(){throw new Error(`"build" not defined on ${this.constructor.name}`)}run(){throw new Error(`"run" not defined on ${this.constructor.name}`)}initCanvas(){throw new Error(`"initCanvas" not defined on ${this.constructor.name}`)}initContext(){throw new Error(`"initContext" not defined on ${this.constructor.name}`)}initPlugins(d){throw new Error(`"initPlugins" not defined on ${this.constructor.name}`)}addFunction(d,E={}){if(d.name&&d.source&&d.argumentTypes&&"returnType"in d)this.functions.push(d);else if(typeof d=="string"||typeof d=="function")this.functions.push(this.functionToIGPUFunction(d,E));else if("settings"in d&&"source"in d)this.functions.push(this.functionToIGPUFunction(d.source,d.settings));else throw new Error("function not properly defined");return this}addNativeFunction(d,E,$={}){const{argumentTypes:P,argumentNames:y}=$.argumentTypes?c($.argumentTypes):this.constructor.nativeFunctionArguments(E)||{};return this.nativeFunctions.push({name:d,source:E,settings:$,argumentTypes:P,argumentNames:y,returnType:$.returnType||this.constructor.nativeFunctionReturnType(E)}),this}setupArguments(d){if(this.kernelArguments=[],this.argumentTypes)for(let E=0;E<this.argumentTypes.length;E++)this.kernelArguments.push({type:this.argumentTypes[E]});else if(!this.argumentTypes){this.argumentTypes=[];for(let E=0;E<d.length;E++){const $=D.getVariableType(d[E],this.strictIntegers),P=$==="Integer"?"Number":$;this.argumentTypes.push(P),this.kernelArguments.push({type:P})}}this.argumentSizes=new Array(d.length),this.argumentBitRatios=new Int32Array(d.length);for(let E=0;E<d.length;E++){const $=d[E];this.argumentSizes[E]=$.constructor===I?$.size:null,this.argumentBitRatios[E]=this.getBitRatio($)}if(this.argumentNames.length!==d.length)throw new Error("arguments are miss-aligned")}setupConstants(){this.kernelConstants=[];let d=this.constantTypes===null;if(d&&(this.constantTypes={}),this.constantBitRatios={},this.constants)for(let E in this.constants){if(d){const $=D.getVariableType(this.constants[E],this.strictIntegers);this.constantTypes[E]=$,this.kernelConstants.push({name:E,type:$})}else this.kernelConstants.push({name:E,type:this.constantTypes[E]});this.constantBitRatios[E]=this.getBitRatio(this.constants[E])}}setOptimizeFloatMemory(d){return this.optimizeFloatMemory=d,this}toKernelOutput(d){return d.hasOwnProperty("x")?d.hasOwnProperty("y")?d.hasOwnProperty("z")?[d.x,d.y,d.z]:[d.x,d.y]:[d.x]:d}setOutput(d){return this.output=this.toKernelOutput(d),this}setDebug(d){return this.debug=d,this}setGraphical(d){return this.graphical=d,this.precision="unsigned",this}setLoopMaxIterations(d){return this.loopMaxIterations=d,this}setConstants(d){return this.constants=d,this}setConstantTypes(d){return this.constantTypes=d,this}setFunctions(d){for(let E=0;E<d.length;E++)this.addFunction(d[E]);return this}setNativeFunctions(d){for(let E=0;E<d.length;E++){const $=d[E],{name:P,source:y}=$;this.addNativeFunction(P,y,$)}return this}setInjectedNative(d){return this.injectedNative=d,this}setPipeline(d){return this.pipeline=d,this}setAsyncMode(d){return this.asyncMode=d,this}setPrecision(d){return this.precision=d,this}setDimensions(d){return D.warnDeprecated("method","setDimensions","setOutput"),this.output=d,this}setOutputToTexture(d){return D.warnDeprecated("method","setOutputToTexture","setPipeline"),this.pipeline=d,this}setImmutable(d){return this.immutable=d,this}setCanvas(d){return this.canvas=d,this}setStrictIntegers(d){return this.strictIntegers=d,this}setDynamicOutput(d){return this.dynamicOutput=d,this}setRandomSeed(d){return this.randomSeed=d,this._mathRandomGenerator=null,this}setHardcodeConstants(d){return D.warnDeprecated("method","setHardcodeConstants"),this.setDynamicOutput(d),this.setDynamicArguments(d),this}setDynamicArguments(d){return this.dynamicArguments=d,this}setUseLegacyEncoder(d){return this.useLegacyEncoder=d,this}setWarnVarUsage(d){return D.warnDeprecated("method","setWarnVarUsage"),this}getCanvas(){return D.warnDeprecated("method","getCanvas"),this.canvas}getWebGl(){return D.warnDeprecated("method","getWebGl"),this.context}setContext(d){return this.context=d,this}setArgumentTypes(d){if(Array.isArray(d))this.argumentTypes=d;else{this.argumentTypes=[];for(const E in d){if(!d.hasOwnProperty(E))continue;const $=this.argumentNames.indexOf(E);if($===-1)throw new Error(`unable to find argument ${E}`);this.argumentTypes[$]=d[E]}}return this}setTactic(d){return this.tactic=d,this}requestFallback(d){if(!this.onRequestFallback)throw new Error(`"onRequestFallback" not defined on ${this.constructor.name}`);return this.fallbackRequested=!0,this.onRequestFallback(d)}validateSettings(){throw new Error(`"validateSettings" not defined on ${this.constructor.name}`)}addSubKernel(d){if(this.subKernels===null&&(this.subKernels=[]),!d.source)throw new Error('subKernel missing "source" property');if(!d.property&&isNaN(d.property))throw new Error('subKernel missing "property" property');if(!d.name)throw new Error('subKernel missing "name" property');return this.subKernels.push(d),this}destroy(d){throw new Error(`"destroy" called on ${this.constructor.name}`)}getBitRatio(d){if(this.precision==="single")return 4;if(Array.isArray(d[0]))return this.getBitRatio(d[0]);if(d.constructor===I)return this.getBitRatio(d.value);switch(d.constructor){case Uint8ClampedArray:case Uint8Array:case Int8Array:return 1;case Uint16Array:case Int16Array:return 2;case Float32Array:case Int32Array:default:return 4}}getPixels(d){throw new Error(`"getPixels" called on ${this.constructor.name}`)}checkOutput(){if(!this.output||!D.isArray(this.output))throw new Error("kernel.output not an array");if(this.output.length<1)throw new Error("kernel.output is empty, needs at least 1 value");for(let d=0;d<this.output.length;d++)if(isNaN(this.output[d])||this.output[d]<1)throw new Error(`${this.constructor.name}.output[${d}] incorrectly defined as \`${this.output[d]}\`, needs to be numeric, and greater than 0`)}prependString(d){throw new Error(`"prependString" called on ${this.constructor.name}`)}hasPrependString(d){throw new Error(`"hasPrependString" called on ${this.constructor.name}`)}toJSON(){return{settings:{output:this.output,pipeline:this.pipeline,argumentNames:this.argumentNames,argumentsTypes:this.argumentTypes,constants:this.constants,pluginNames:this.plugins?this.plugins.map(d=>d.name):null,returnType:this.returnType}}}buildSignature(d){const E=this.constructor;this.signature=E.getSignature(this,E.getArgumentTypes(this,d))}static getArgumentTypes(d,E){const $=new Array(E.length);for(let P=0;P<E.length;P++){const y=E[P],p=d.argumentTypes[P];if(y.type)$[P]=y.type;else switch(p){case"Number":case"Integer":case"Float":case"ArrayTexture(1)":$[P]=D.getVariableType(y,d.strictIntegers);break;default:$[P]=D.typeFitsValue(p,y)?p:D.getVariableType(y,d.strictIntegers)}}return $}static getSignature(d,E){throw new Error(`"getSignature" not implemented on ${this.name}`)}functionToIGPUFunction(d,E={}){if(typeof d!="string"&&typeof d!="function")throw new Error("source not a string or function");const $=typeof d=="string"?d:d.toString();let P=[];return Array.isArray(E.argumentTypes)?P=E.argumentTypes:typeof E.argumentTypes=="object"?P=D.getArgumentNamesFromString($).map(y=>E.argumentTypes[y])||[]:P=E.argumentTypes||[],{name:D.getFunctionNameFromString($)||null,source:$,argumentTypes:P,returnType:E.returnType||null}}onActivate(d){}switchKernels(d){this.switchingKernels?this.switchingKernels.push(d):this.switchingKernels=[d]}resetSwitchingKernels(){const d=this.switchingKernels;return this.switchingKernels=null,d}checkArgumentTypes(d){if(!this.argumentTypes)return;const E=Math.min(d.length,this.argumentTypes.length);for(let $=0;$<E;$++)D.typeFitsValue(this.argumentTypes[$],d[$])||this.switchKernels({type:"argumentTypeMismatch",index:$,needed:D.getVariableType(d[$],this.strictIntegers)})}};function c(d){const E=Object.keys(d),$=[];for(let P=0;P<E.length;P++){const y=E[P];$.push(d[y])}return{argumentTypes:$,argumentNames:E}}G.exports={Kernel:_}}),N=s((j,G)=>{G.exports={FunctionBuilder:class Cr{static fromKernel(I,_,c){const{kernelArguments:d,kernelConstants:E,argumentNames:$,argumentSizes:P,argumentBitRatios:y,constants:p,constantBitRatios:g,debug:k,loopMaxIterations:o,nativeFunctions:l,output:x,optimizeFloatMemory:w,precision:v,plugins:C,source:b,subKernels:T,functions:h,leadingReturnStatement:F,followingReturnStatement:O,dynamicArguments:z,dynamicOutput:L}=I,V=new Array(d.length),U={};for(let ae=0;ae<d.length;ae++)V[ae]=d[ae].type;for(let ae=0;ae<E.length;ae++){const ye=E[ae];U[ye.name]=ye.type}const X=(ae,ye)=>Me.needsArgumentType(ae,ye),q=(ae,ye,Ce)=>{Me.assignArgumentType(ae,ye,Ce)},W=(ae,ye,Ce)=>Me.lookupReturnType(ae,ye,Ce),ee=ae=>Me.lookupFunctionArgumentTypes(ae),se=(ae,ye)=>Me.lookupFunctionArgumentName(ae,ye),Z=(ae,ye)=>Me.lookupFunctionArgumentBitRatio(ae,ye),ie=(ae,ye,Ce,Ne)=>{Me.assignArgumentType(ae,ye,Ce,Ne)},he=(ae,ye,Ce,Ne)=>{Me.assignArgumentBitRatio(ae,ye,Ce,Ne)},we=(ae,ye,Ce)=>{Me.trackFunctionCall(ae,ye,Ce)},re=(ae,ye)=>{const Ce=[];for(let ft=0;ft<ae.params.length;ft++)Ce.push(ae.params[ft].name);const Ne=new _(ye,Object.assign({},ce,{returnType:null,ast:ae,name:ae.id.name,argumentNames:Ce,lookupReturnType:W,lookupFunctionArgumentTypes:ee,lookupFunctionArgumentName:se,lookupFunctionArgumentBitRatio:Z,needsArgumentType:X,assignArgumentType:q,triggerImplyArgumentType:ie,triggerImplyArgumentBitRatio:he,onFunctionCall:we}));Ne.traceFunctionAST(ae),Me.addFunctionNode(Ne)},ce=Object.assign({isRootKernel:!1,onNestedFunction:re,lookupReturnType:W,lookupFunctionArgumentTypes:ee,lookupFunctionArgumentName:se,lookupFunctionArgumentBitRatio:Z,needsArgumentType:X,assignArgumentType:q,triggerImplyArgumentType:ie,triggerImplyArgumentBitRatio:he,onFunctionCall:we,optimizeFloatMemory:w,precision:v,constants:p,constantTypes:U,constantBitRatios:g,debug:k,loopMaxIterations:o,output:x,plugins:C,dynamicArguments:z,dynamicOutput:L},c||{}),Ve=Object.assign({},ce,{isRootKernel:!0,name:"kernel",argumentNames:$,argumentTypes:V,argumentSizes:P,argumentBitRatios:y,leadingReturnStatement:F,followingReturnStatement:O});if(typeof b=="object"&&b.functionNodes)return new Cr().fromJSON(b.functionNodes,_);const ue=new _(b,Ve);let Se=null;h&&(Se=h.map(ae=>new _(ae.source,{returnType:ae.returnType,argumentTypes:ae.argumentTypes,output:x,plugins:C,constants:p,constantTypes:U,constantBitRatios:g,optimizeFloatMemory:w,precision:v,lookupReturnType:W,lookupFunctionArgumentTypes:ee,lookupFunctionArgumentName:se,lookupFunctionArgumentBitRatio:Z,needsArgumentType:X,assignArgumentType:q,triggerImplyArgumentType:ie,triggerImplyArgumentBitRatio:he,onFunctionCall:we,onNestedFunction:re})));let Re=null;T&&(Re=T.map(ae=>{const{name:ye,source:Ce}=ae;return new _(Ce,Object.assign({},ce,{name:ye,isSubKernel:!0,isRootKernel:!1}))}));const Me=new Cr({kernel:I,rootNode:ue,functionNodes:Se,nativeFunctions:l,subKernelNodes:Re});return Me}constructor(I){if(I=I||{},this.kernel=I.kernel,this.rootNode=I.rootNode,this.functionNodes=I.functionNodes||[],this.subKernelNodes=I.subKernelNodes||[],this.nativeFunctions=I.nativeFunctions||[],this.functionMap={},this.nativeFunctionNames=[],this.lookupChain=[],this.functionNodeDependencies={},this.functionCalls={},this.rootNode&&(this.functionMap.kernel=this.rootNode),this.functionNodes)for(let _=0;_<this.functionNodes.length;_++)this.functionMap[this.functionNodes[_].name]=this.functionNodes[_];if(this.subKernelNodes)for(let _=0;_<this.subKernelNodes.length;_++)this.functionMap[this.subKernelNodes[_].name]=this.subKernelNodes[_];if(this.nativeFunctions)for(let _=0;_<this.nativeFunctions.length;_++){const c=this.nativeFunctions[_];this.nativeFunctionNames.push(c.name)}}addFunctionNode(I){if(!I.name)throw new Error("functionNode.name needs set");this.functionMap[I.name]=I,I.isRootKernel&&(this.rootNode=I)}traceFunctionCalls(I,_){if(I=I||"kernel",_=_||[],this.nativeFunctionNames.indexOf(I)>-1){const d=_.indexOf(I);if(d===-1)_.push(I);else{const E=_.splice(d,1)[0];_.push(E)}return _}const c=this.functionMap[I];if(c){const d=_.indexOf(I);if(d===-1){_.push(I),c.toString();for(let E=0;E<c.calledFunctions.length;++E)this.traceFunctionCalls(c.calledFunctions[E],_)}else{const E=_.splice(d,1)[0];_.push(E)}}return _}getPrototypeString(I){return this.getPrototypes(I).join(`
`)}getPrototypes(I){return this.rootNode&&this.rootNode.toString(),I?this.getPrototypesFromFunctionNames(this.traceFunctionCalls(I,[]).reverse()):this.getPrototypesFromFunctionNames(Object.keys(this.functionMap))}getStringFromFunctionNames(I){const _=[];for(let c=0;c<I.length;++c)this.functionMap[I[c]]&&_.push(this.functionMap[I[c]].toString());return _.join(`
`)}getPrototypesFromFunctionNames(I){const _=[];for(let c=0;c<I.length;++c){const d=I[c],E=this.nativeFunctionNames.indexOf(d);if(E>-1){_.push(this.nativeFunctions[E].source);continue}const $=this.functionMap[d];$&&_.push($.toString())}return _}toJSON(){return this.traceFunctionCalls(this.rootNode.name).reverse().map(I=>{const _=this.nativeFunctions.indexOf(I);if(_>-1)return{name:I,source:this.nativeFunctions[_].source};if(this.functionMap[I])return this.functionMap[I].toJSON();throw new Error(`function ${I} not found`)})}fromJSON(I,_){this.functionMap={};for(let c=0;c<I.length;c++){const d=I[c];this.functionMap[d.settings.name]=new _(d.ast,d.settings)}return this}getString(I){return I?this.getStringFromFunctionNames(this.traceFunctionCalls(I).reverse()):this.getStringFromFunctionNames(Object.keys(this.functionMap))}lookupReturnType(I,_,c){if(_.type!=="CallExpression")throw new Error(`expected ast type of "CallExpression", but is ${_.type}`);if(this._isNativeFunction(I))return this._lookupNativeFunctionReturnType(I);if(this._isFunction(I)){const d=this._getFunction(I);if(d.returnType)return d.returnType;{for(let $=0;$<this.lookupChain.length;$++)if(this.lookupChain[$].ast===_){if(d.argumentTypes.length===0&&_.arguments.length>0){const P=_.arguments;for(let y=0;y<P.length;y++)this.lookupChain.push({name:c.name,ast:P[$],requestingNode:c}),d.argumentTypes[y]=c.getType(P[y]),this.lookupChain.pop();return d.returnType=d.getType(d.getJsAST())}throw new Error("circlical logic detected!")}this.lookupChain.push({name:c.name,ast:_,requestingNode:c});const E=d.getType(d.getJsAST());return this.lookupChain.pop(),d.returnType=E}}return null}_getFunction(I){return this._isFunction(I),this.functionMap[I]}_isFunction(I){return!!this.functionMap[I]}_getNativeFunction(I){for(let _=0;_<this.nativeFunctions.length;_++)if(this.nativeFunctions[_].name===I)return this.nativeFunctions[_];return null}_isNativeFunction(I){return!!this._getNativeFunction(I)}_lookupNativeFunctionReturnType(I){let _=this._getNativeFunction(I);if(_)return _.returnType;throw new Error(`Native function ${I} not found`)}lookupFunctionArgumentTypes(I){return this._isNativeFunction(I)?this._getNativeFunction(I).argumentTypes:this._isFunction(I)?this._getFunction(I).argumentTypes:null}lookupFunctionArgumentName(I,_){return this._getFunction(I).argumentNames[_]}lookupFunctionArgumentBitRatio(I,_){if(!this._isFunction(I))throw new Error("function not found");if(this.rootNode.name===I){const $=this.rootNode.argumentNames.indexOf(_);if($!==-1)return this.rootNode.argumentBitRatios[$]}const c=this._getFunction(I),d=c.argumentNames.indexOf(_);if(d===-1)throw new Error("argument not found");const E=c.argumentBitRatios[d];if(typeof E!="number")throw new Error("argument bit ratio not found");return E}needsArgumentType(I,_){return this._isFunction(I)?!this._getFunction(I).argumentTypes[_]:!1}assignArgumentType(I,_,c,d){if(!this._isFunction(I))return;const E=this._getFunction(I);E.argumentTypes[_]||(E.argumentTypes[_]=c)}assignArgumentBitRatio(I,_,c,d){const E=this._getFunction(I);if(this._isNativeFunction(c))return null;const $=this._getFunction(c),P=E.argumentNames.indexOf(_);if(P===-1)throw new Error(`Argument ${_} not found in arguments from function ${I}`);const y=E.argumentBitRatios[P];if(typeof y!="number")throw new Error(`Bit ratio for argument ${_} not found in function ${I}`);$.argumentBitRatios||($.argumentBitRatios=new Array($.argumentNames.length));const p=$.argumentBitRatios[d];if(typeof p=="number"){if(p!==y)throw new Error(`Incompatible bit ratio found at function ${I} at argument ${_}`);return p}return $.argumentBitRatios[d]=y,y}trackFunctionCall(I,_,c){this.functionNodeDependencies[I]||(this.functionNodeDependencies[I]=new Set,this.functionCalls[I]=[]),this.functionNodeDependencies[I].add(_),this.functionCalls[I].push(c)}getKernelResultType(){return this.rootNode.returnType||this.rootNode.getType(this.rootNode.ast)}getSubKernelResultType(I){const _=this.subKernelNodes[I];let c=!1;for(let d=0;d<this.rootNode.functionCalls.length;d++)this.rootNode.functionCalls[d].ast.callee.name===_.name&&(c=!0);if(!c)throw new Error(`SubKernel ${_.name} never called by kernel`);return _.returnType||_.getType(_.getJsAST())}getReturnTypes(){const I={[this.rootNode.name]:this.rootNode.getType(this.rootNode.ast)},_=this.traceFunctionCalls(this.rootNode.name);for(let c=0;c<_.length;c++){const d=_[c],E=this.functionMap[d];I[d]=E.getType(E.ast)}return I}}}}),H=s((j,G)=>{const{utils:D}=m();function I(d){return d.length>0?d[d.length-1]:null}const _={trackIdentifiers:"trackIdentifiers",memberExpression:"memberExpression",inForLoopInit:"inForLoopInit"};var c=class{constructor(d){this.runningContexts=[],this.functionContexts=[],this.contexts=[],this.functionCalls=[],this.declarations=[],this.identifiers=[],this.functions=[],this.returnStatements=[],this.trackedIdentifiers=null,this.states=[],this.newFunctionContext(),this.scan(d)}isState(d){return this.states[this.states.length-1]===d}hasState(d){return this.states.indexOf(d)>-1}pushState(d){this.states.push(d)}popState(d){if(this.isState(d))this.states.pop();else throw new Error(`Cannot pop the non-active state "${d}"`)}get currentFunctionContext(){return I(this.functionContexts)}get currentContext(){return I(this.runningContexts)}newFunctionContext(){const d={"@contextType":"function"};this.contexts.push(d),this.functionContexts.push(d)}newContext(d){const E=Object.assign({"@contextType":"const/let"},this.currentContext);this.contexts.push(E),this.runningContexts.push(E),d();const{currentFunctionContext:$}=this;for(const P in $)!$.hasOwnProperty(P)||E.hasOwnProperty(P)||(E[P]=$[P]);return this.runningContexts.pop(),E}useFunctionContext(d){const E=I(this.functionContexts);this.runningContexts.push(E),d(),this.runningContexts.pop()}getIdentifiers(d){const E=this.trackedIdentifiers=[];return this.pushState(_.trackIdentifiers),d(),this.trackedIdentifiers=null,this.popState(_.trackIdentifiers),E}getDeclaration(d){const{currentContext:E,currentFunctionContext:$,runningContexts:P}=this,y=E[d]||$[d]||null;if(!y&&E===$&&P.length>0){const p=P[P.length-2];if(p[d])return p[d]}return y}scan(d){if(d){if(Array.isArray(d)){for(let E=0;E<d.length;E++)this.scan(d[E]);return}switch(d.type){case"Program":this.useFunctionContext(()=>{this.scan(d.body)});break;case"BlockStatement":this.newContext(()=>{this.scan(d.body)});break;case"AssignmentExpression":case"LogicalExpression":this.scan(d.left),this.scan(d.right);break;case"BinaryExpression":this.scan(d.left),this.scan(d.right);break;case"UpdateExpression":if(d.operator==="++"){const E=this.getDeclaration(d.argument.name);E&&(E.suggestedType="Integer")}this.scan(d.argument);break;case"UnaryExpression":this.scan(d.argument);break;case"VariableDeclaration":d.kind==="var"?this.useFunctionContext(()=>{d.declarations=D.normalizeDeclarations(d),this.scan(d.declarations)}):(d.declarations=D.normalizeDeclarations(d),this.scan(d.declarations));break;case"VariableDeclarator":{const{currentContext:E}=this,$=this.hasState(_.inForLoopInit),P={ast:d,context:E,name:d.id.name,origin:"declaration",inForLoopInit:$,inForLoopTest:null,assignable:E===this.currentFunctionContext||!$&&!E.hasOwnProperty(d.id.name),suggestedType:null,valueType:null,dependencies:null,isSafe:null};E[d.id.name]||(E[d.id.name]=P),this.declarations.push(P),this.scan(d.id),this.scan(d.init);break}case"FunctionExpression":case"FunctionDeclaration":this.runningContexts.length===0?this.scan(d.body):this.functions.push(d);break;case"IfStatement":this.scan(d.test),this.scan(d.consequent),d.alternate&&this.scan(d.alternate);break;case"ForStatement":{let E;const $=this.newContext(()=>{this.pushState(_.inForLoopInit),this.scan(d.init),this.popState(_.inForLoopInit),E=this.getIdentifiers(()=>{this.scan(d.test)}),this.scan(d.update),this.newContext(()=>{this.scan(d.body)})});if(E)for(const P in $)P!=="@contextType"&&E.indexOf(P)>-1&&($[P].inForLoopTest=!0);break}case"DoWhileStatement":case"WhileStatement":this.newContext(()=>{this.scan(d.body),this.scan(d.test)});break;case"Identifier":this.isState(_.trackIdentifiers)&&this.trackedIdentifiers.push(d.name),this.identifiers.push({context:this.currentContext,declaration:this.getDeclaration(d.name),ast:d});break;case"ReturnStatement":this.returnStatements.push(d),this.scan(d.argument);break;case"MemberExpression":this.pushState(_.memberExpression),this.scan(d.object),this.scan(d.property),this.popState(_.memberExpression);break;case"ExpressionStatement":this.scan(d.expression);break;case"SequenceExpression":this.scan(d.expressions);break;case"CallExpression":this.functionCalls.push({context:this.currentContext,ast:d}),this.scan(d.arguments);break;case"ArrayExpression":this.scan(d.elements);break;case"ConditionalExpression":this.scan(d.test),this.scan(d.alternate),this.scan(d.consequent);break;case"SwitchStatement":this.scan(d.discriminant),this.scan(d.cases);break;case"SwitchCase":this.scan(d.test),this.scan(d.consequent);break;case"ThisExpression":case"Literal":case"DebuggerStatement":case"EmptyStatement":case"BreakStatement":case"ContinueStatement":break;default:throw new Error(`unhandled type "${d.type}"`)}}}};G.exports={FunctionTracer:c}}),Y=s((j,G)=>{const D=i(),{utils:I}=m(),{FunctionTracer:_}=H(),c=["E","PI","SQRT2","SQRT1_2","LN2","LN10","LOG2E","LOG10E"],d=["abs","acos","acosh","asin","asinh","atan","atan2","atanh","cbrt","ceil","clz32","cos","cosh","expm1","exp","floor","fround","imul","log","log2","log10","log1p","max","min","pow","random","round","sign","sin","sinh","sqrt","tan","tanh","trunc"],E=["value","value[]","value[][]","value[][][]","value[][][][]","value.value","value.thread.value","this.thread.value","this.output.value","this.constants.value","this.constants.value[]","this.constants.value[][]","this.constants.value[][][]","this.constants.value[][][][]","fn()[]","fn()[][]","fn()[][][]","[][]"];var $=class{constructor(y,p){if(!y&&!p.ast)throw new Error("source parameter is missing");if(p=p||{},this.source=y,this.ast=null,this.name=typeof y=="string"?p.isRootKernel?"kernel":p.name||I.getFunctionNameFromString(y):null,this.calledFunctions=[],this.constants={},this.constantTypes={},this.constantBitRatios={},this.isRootKernel=!1,this.isSubKernel=!1,this.debug=null,this.functions=null,this.identifiers=null,this.contexts=null,this.functionCalls=null,this.states=[],this.needsArgumentType=null,this.assignArgumentType=null,this.lookupReturnType=null,this.lookupFunctionArgumentTypes=null,this.lookupFunctionArgumentBitRatio=null,this.triggerImplyArgumentType=null,this.triggerImplyArgumentBitRatio=null,this.onNestedFunction=null,this.onFunctionCall=null,this.optimizeFloatMemory=null,this.precision=null,this.loopMaxIterations=null,this.argumentNames=typeof this.source=="string"?I.getArgumentNamesFromString(this.source):null,this.argumentTypes=[],this.argumentSizes=[],this.argumentBitRatios=null,this.returnType=null,this.output=[],this.plugins=null,this.leadingReturnStatement=null,this.followingReturnStatement=null,this.dynamicOutput=null,this.dynamicArguments=null,this.strictTypingChecking=!1,this.fixIntegerDivisionAccuracy=null,p)for(const g in p)p.hasOwnProperty(g)&&this.hasOwnProperty(g)&&(this[g]=p[g]);this.literalTypes={},this.validate(),this._string=null,this._internalVariableNames={}}validate(){if(typeof this.source!="string"&&!this.ast)throw new Error("this.source not a string");if(!this.ast&&!I.isFunctionString(this.source))throw new Error("this.source not a function string");if(!this.name)throw new Error("this.name could not be set");if(this.argumentTypes.length>0&&this.argumentTypes.length!==this.argumentNames.length)throw new Error(`argumentTypes count of ${this.argumentTypes.length} exceeds ${this.argumentNames.length}`);if(this.output.length<1)throw new Error("this.output is not big enough")}isIdentifierConstant(y){return this.constants?this.constants.hasOwnProperty(y):!1}isInput(y){return this.argumentTypes[this.argumentNames.indexOf(y)]==="Input"}pushState(y){this.states.push(y)}popState(y){if(this.state!==y)throw new Error(`Cannot popState ${y} when in ${this.state}`);this.states.pop()}isState(y){return this.state===y}get state(){return this.states[this.states.length-1]}astMemberExpressionUnroll(y){if(y.type==="Identifier")return y.name;if(y.type==="ThisExpression")return"this";if(y.type==="MemberExpression"&&y.object&&y.property)return y.object.hasOwnProperty("name")&&y.object.name!=="Math"?this.astMemberExpressionUnroll(y.property):this.astMemberExpressionUnroll(y.object)+"."+this.astMemberExpressionUnroll(y.property);if(y.hasOwnProperty("expressions")){const p=y.expressions[0];if(p.type==="Literal"&&p.value===0&&y.expressions.length===2)return this.astMemberExpressionUnroll(y.expressions[1])}throw this.astErrorOutput("Unknown astMemberExpressionUnroll",y)}getJsAST(y){if(this.ast)return this.ast;if(typeof this.source=="object")return this.traceFunctionAST(this.source),this.ast=this.source;if(y=y||D,y===null)throw new Error("Missing JS to AST parser");const p=Object.freeze(y.parse(`const parser_${this.name} = ${this.source};`,{locations:!0,ecmaVersion:2020})),g=p.body[0].declarations[0].init;if(this.traceFunctionAST(g),!p)throw new Error("Failed to parse JS code");return this.ast=g}traceFunctionAST(y){const{contexts:p,declarations:g,functions:k,identifiers:o,functionCalls:l}=new _(y);this.contexts=p,this.identifiers=o,this.functionCalls=l,this.functions=k;for(let x=0;x<g.length;x++){const w=g[x],{ast:v,inForLoopInit:C,inForLoopTest:b}=w,{init:T}=v,h=this.getDependencies(T);let F=null;if(C&&b)F="Integer";else if(T){const O=this.getType(T);switch(O){case"Integer":case"Float":case"Number":T.type==="MemberExpression"?F=O:F="Number";break;case"LiteralInteger":F="Number";break;default:F=O}}w.valueType=F,w.dependencies=h,w.isSafe=this.isSafeDependencies(h)}for(let x=0;x<k.length;x++)this.onNestedFunction(k[x],this.source)}getDeclaration(y){for(let p=0;p<this.identifiers.length;p++){const g=this.identifiers[p];if(y===g.ast)return g.declaration}return null}getVariableType(y){if(y.type!=="Identifier")throw new Error(`ast of ${y.type} not "Identifier"`);let p=null;const g=this.argumentNames.indexOf(y.name);if(g===-1){const k=this.getDeclaration(y);if(k)return k.valueType}else{const k=this.argumentTypes[g];k&&(p=k)}if(!p&&this.strictTypingChecking)throw new Error(`Declaration of ${name} not found`);return p}getLookupType(y){if(!P.hasOwnProperty(y))throw new Error(`unknown typeLookupMap ${y}`);return P[y]}getConstantType(y){if(this.constantTypes[y]){const p=this.constantTypes[y];return p==="Float"?"Number":p}throw new Error(`Type for constant "${y}" not declared`)}toString(){return this._string?this._string:this._string=this.astGeneric(this.getJsAST(),[]).join("").trim()}toJSON(){const y={source:this.source,name:this.name,constants:this.constants,constantTypes:this.constantTypes,isRootKernel:this.isRootKernel,isSubKernel:this.isSubKernel,debug:this.debug,output:this.output,loopMaxIterations:this.loopMaxIterations,argumentNames:this.argumentNames,argumentTypes:this.argumentTypes,argumentSizes:this.argumentSizes,returnType:this.returnType,leadingReturnStatement:this.leadingReturnStatement,followingReturnStatement:this.followingReturnStatement};return{ast:this.ast,settings:y}}getType(y){if(Array.isArray(y))return this.getType(y[y.length-1]);switch(y.type){case"BlockStatement":return this.getType(y.body);case"ArrayExpression":switch(this.getType(y.elements[0])){case"Array(2)":case"Array(3)":case"Array(4)":return`Matrix(${y.elements.length})`}return`Array(${y.elements.length})`;case"Literal":const p=this.astKey(y);return this.literalTypes[p]?this.literalTypes[p]:Number.isInteger(y.value)?"LiteralInteger":y.value===!0||y.value===!1?"Boolean":"Number";case"AssignmentExpression":return this.getType(y.left);case"CallExpression":if(this.isAstMathFunction(y))return"Number";if(!y.callee||!y.callee.name){if(y.callee.type==="SequenceExpression"&&y.callee.expressions[y.callee.expressions.length-1].property.name){const x=y.callee.expressions[y.callee.expressions.length-1].property.name;return this.inferArgumentTypesIfNeeded(x,y.arguments),this.lookupReturnType(x,y,this)}if(this.getVariableSignature(y.callee,!0)==="this.color")return null;if(y.callee.type==="MemberExpression"&&y.callee.object&&y.callee.property&&y.callee.property.name&&y.arguments){const x=y.callee.property.name;return this.inferArgumentTypesIfNeeded(x,y.arguments),this.lookupReturnType(x,y,this)}throw this.astErrorOutput("Unknown call expression",y)}if(y.callee&&y.callee.name){const x=y.callee.name;return this.inferArgumentTypesIfNeeded(x,y.arguments),this.lookupReturnType(x,y,this)}throw this.astErrorOutput(`Unhandled getType Type "${y.type}"`,y);case"LogicalExpression":return"Boolean";case"BinaryExpression":switch(y.operator){case"%":return"Number";case"/":return"Number";case">":case"<":return"Boolean";case"&":case"|":case"^":case"<<":case">>":case">>>":return"Integer"}const g=this.getType(y.left);if(this.isState("skip-literal-correction"))return g;if(g==="LiteralInteger"){const x=this.getType(y.right);return x==="LiteralInteger"?y.left.value%1===0?"Integer":"Float":x}if(g==="Integer"){const x=this.getType(y.right);if(x==="Number"||x==="Float")return x}return P[g]||g;case"UpdateExpression":return this.getType(y.argument);case"UnaryExpression":return y.operator==="~"?"Integer":this.getType(y.argument);case"VariableDeclaration":{const x=y.declarations;let w;for(let v=0;v<x.length;v++){const C=x[v];w=this.getType(C)}if(!w)throw this.astErrorOutput("Unable to find type for declaration",y);return w}case"VariableDeclarator":const k=this.getDeclaration(y.id);if(!k)throw this.astErrorOutput("Unable to find declarator",y);if(!k.valueType)throw this.astErrorOutput("Unable to find declarator valueType",y);return k.valueType;case"Identifier":if(y.name==="Infinity")return"Number";if(this.isAstVariable(y)&&this.getVariableSignature(y)==="value")return this.getCheckVariableType(y);const o=this.findIdentifierOrigin(y);return o&&o.init?this.getType(o.init):null;case"ReturnStatement":return this.getType(y.argument);case"MemberExpression":if(this.isAstMathFunction(y)){switch(y.property.name){case"ceil":return"Integer";case"floor":return"Integer";case"round":return"Integer"}return"Number"}if(this.isAstVariable(y)){switch(this.getVariableSignature(y)){case"value[]":return this.getLookupType(this.getCheckVariableType(y.object));case"value[][]":return this.getLookupType(this.getCheckVariableType(y.object.object));case"value[][][]":return this.getLookupType(this.getCheckVariableType(y.object.object.object));case"value[][][][]":return this.getLookupType(this.getCheckVariableType(y.object.object.object.object));case"value.thread.value":case"this.thread.value":return"Integer";case"this.output.value":return this.dynamicOutput?"Integer":"LiteralInteger";case"this.constants.value":return this.getConstantType(y.property.name);case"this.constants.value[]":return this.getLookupType(this.getConstantType(y.object.property.name));case"this.constants.value[][]":return this.getLookupType(this.getConstantType(y.object.object.property.name));case"this.constants.value[][][]":return this.getLookupType(this.getConstantType(y.object.object.object.property.name));case"this.constants.value[][][][]":return this.getLookupType(this.getConstantType(y.object.object.object.object.property.name));case"fn()[]":case"fn()[][]":case"fn()[][][]":return this.getLookupType(this.getType(y.object));case"value.value":if(this.isAstMathVariable(y))return"Number";switch(y.property.name){case"r":case"g":case"b":case"a":return this.getLookupType(this.getCheckVariableType(y.object))}case"[][]":return"Number"}throw this.astErrorOutput("Unhandled getType MemberExpression",y)}throw this.astErrorOutput("Unhandled getType MemberExpression",y);case"ConditionalExpression":return this.getType(y.consequent);case"FunctionDeclaration":case"FunctionExpression":const l=this.findLastReturn(y.body);return l?this.getType(l):null;case"IfStatement":return this.getType(y.consequent);case"SequenceExpression":return this.getType(y.expressions[y.expressions.length-1]);default:throw this.astErrorOutput(`Unhandled getType Type "${y.type}"`,y)}}getCheckVariableType(y){const p=this.getVariableType(y);if(!p)throw this.astErrorOutput(`${y.type} is not defined`,y);return p}inferArgumentTypesIfNeeded(y,p){for(let g=0;g<p.length;g++){if(!this.needsArgumentType(y,g))continue;const k=this.getType(p[g]);if(!k)throw this.astErrorOutput(`Unable to infer argument ${g}`,p[g]);this.assignArgumentType(y,g,k)}}isAstMathVariable(y){return y.type==="MemberExpression"&&y.object&&y.object.type==="Identifier"&&y.object.name==="Math"&&y.property&&y.property.type==="Identifier"&&c.includes(y.property.name)}isAstMathFunction(y){return y.type==="CallExpression"&&y.callee&&y.callee.type==="MemberExpression"&&y.callee.object&&y.callee.object.type==="Identifier"&&y.callee.object.name==="Math"&&y.callee.property&&y.callee.property.type==="Identifier"&&d.includes(y.callee.property.name)}isAstVariable(y){return y.type==="Identifier"||y.type==="MemberExpression"}isSafe(y){return this.isSafeDependencies(this.getDependencies(y))}isSafeDependencies(y){return y&&y.every?y.every(p=>p.isSafe):!0}getDependencies(y,p,g){if(p||(p=[]),!y)return null;if(Array.isArray(y)){for(let k=0;k<y.length;k++)this.getDependencies(y[k],p,g);return p}switch(y.type){case"AssignmentExpression":return this.getDependencies(y.left,p,g),this.getDependencies(y.right,p,g),p;case"ConditionalExpression":return this.getDependencies(y.test,p,g),this.getDependencies(y.alternate,p,g),this.getDependencies(y.consequent,p,g),p;case"Literal":p.push({origin:"literal",value:y.value,isSafe:g===!0?!1:y.value>-1/0&&y.value<1/0&&!isNaN(y.value)});break;case"VariableDeclarator":return this.getDependencies(y.init,p,g);case"Identifier":const k=this.getDeclaration(y);if(k)p.push({name:y.name,origin:"declaration",isSafe:g?!1:this.isSafeDependencies(k.dependencies)});else if(this.argumentNames.indexOf(y.name)>-1)p.push({name:y.name,origin:"argument",isSafe:!1});else if(this.strictTypingChecking)throw new Error(`Cannot find identifier origin "${y.name}"`);break;case"FunctionDeclaration":return this.getDependencies(y.body.body[y.body.body.length-1],p,g);case"ReturnStatement":return this.getDependencies(y.argument,p);case"BinaryExpression":case"LogicalExpression":return g=y.operator==="/"||y.operator==="*",this.getDependencies(y.left,p,g),this.getDependencies(y.right,p,g),p;case"UnaryExpression":case"UpdateExpression":return this.getDependencies(y.argument,p,g);case"VariableDeclaration":return this.getDependencies(y.declarations,p,g);case"ArrayExpression":return p.push({origin:"declaration",isSafe:!0}),p;case"CallExpression":return p.push({origin:"function",isSafe:!0}),p;case"MemberExpression":const o=this.getMemberExpressionDetails(y);switch(o.signature){case"value[]":this.getDependencies(y.object,p,g);break;case"value[][]":this.getDependencies(y.object.object,p,g);break;case"value[][][]":this.getDependencies(y.object.object.object,p,g);break;case"this.output.value":this.dynamicOutput&&p.push({name:o.name,origin:"output",isSafe:!1});break}if(o)return o.property&&this.getDependencies(o.property,p,g),o.xProperty&&this.getDependencies(o.xProperty,p,g),o.yProperty&&this.getDependencies(o.yProperty,p,g),o.zProperty&&this.getDependencies(o.zProperty,p,g),p;case"SequenceExpression":return this.getDependencies(y.expressions,p,g);default:throw this.astErrorOutput(`Unhandled type ${y.type} in getDependencies`,y)}return p}getVariableSignature(y,p){if(!this.isAstVariable(y))throw new Error(`ast of type "${y.type}" is not a variable signature`);if(y.type==="Identifier")return"value";const g=[];for(;y;)y.computed?g.push("[]"):y.type==="ThisExpression"?g.unshift("this"):y.property&&y.property.name?y.property.name==="x"||y.property.name==="y"||y.property.name==="z"?g.unshift(p?"."+y.property.name:".value"):y.property.name==="constants"||y.property.name==="thread"||y.property.name==="output"?g.unshift("."+y.property.name):g.unshift(p?"."+y.property.name:".value"):y.name?g.unshift(p?y.name:"value"):y.callee&&y.callee.name?g.unshift(p?y.callee.name+"()":"fn()"):y.elements?g.unshift("[]"):g.unshift("unknown"),y=y.object;const k=g.join("");return p||E.includes(k)?k:null}build(){return this.toString().length>0}astGeneric(y,p){if(y===null)throw this.astErrorOutput("NULL ast",y);if(Array.isArray(y)){for(let g=0;g<y.length;g++)this.astGeneric(y[g],p);return p}switch(y.type){case"FunctionDeclaration":return this.astFunctionDeclaration(y,p);case"FunctionExpression":return this.astFunctionExpression(y,p);case"ReturnStatement":return this.astReturnStatement(y,p);case"Literal":return this.astLiteral(y,p);case"BinaryExpression":return this.astBinaryExpression(y,p);case"Identifier":return this.astIdentifierExpression(y,p);case"AssignmentExpression":return this.astAssignmentExpression(y,p);case"ExpressionStatement":return this.astExpressionStatement(y,p);case"EmptyStatement":return this.astEmptyStatement(y,p);case"BlockStatement":return this.astBlockStatement(y,p);case"IfStatement":return this.astIfStatement(y,p);case"SwitchStatement":return this.astSwitchStatement(y,p);case"BreakStatement":return this.astBreakStatement(y,p);case"ContinueStatement":return this.astContinueStatement(y,p);case"ForStatement":return this.astForStatement(y,p);case"WhileStatement":return this.astWhileStatement(y,p);case"DoWhileStatement":return this.astDoWhileStatement(y,p);case"VariableDeclaration":return this.astVariableDeclaration(y,p);case"VariableDeclarator":return this.astVariableDeclarator(y,p);case"ThisExpression":return this.astThisExpression(y,p);case"SequenceExpression":return this.astSequenceExpression(y,p);case"UnaryExpression":return this.astUnaryExpression(y,p);case"UpdateExpression":return this.astUpdateExpression(y,p);case"LogicalExpression":return this.astLogicalExpression(y,p);case"MemberExpression":return this.astMemberExpression(y,p);case"CallExpression":return this.astCallExpression(y,p);case"ArrayExpression":return this.astArrayExpression(y,p);case"DebuggerStatement":return this.astDebuggerStatement(y,p);case"ConditionalExpression":return this.astConditionalExpression(y,p)}throw this.astErrorOutput("Unknown ast type : "+y.type,y)}astErrorOutput(y,p){if(typeof this.source!="string")return new Error(y);const g=I.getAstString(this.source,p),k=this.source.slice(p.start).split(/\n/),o=k.length>0?k[k.length-1]:0;return new Error(`${y} on line ${k.length}, position ${o.length}:
 ${g}`)}astDebuggerStatement(y,p){return p}astConditionalExpression(y,p){if(y.type!=="ConditionalExpression")throw this.astErrorOutput("Not a conditional expression",y);return p.push("("),this.astGeneric(y.test,p),p.push("?"),this.astGeneric(y.consequent,p),p.push(":"),this.astGeneric(y.alternate,p),p.push(")"),p}astFunction(y,p){throw new Error(`"astFunction" not defined on ${this.constructor.name}`)}astFunctionDeclaration(y,p){return this.isChildFunction(y)?p:this.astFunction(y,p)}astFunctionExpression(y,p){return this.isChildFunction(y)?p:this.astFunction(y,p)}isChildFunction(y){for(let p=0;p<this.functions.length;p++)if(this.functions[p]===y)return!0;return!1}astReturnStatement(y,p){return p}astLiteral(y,p){return this.literalTypes[this.astKey(y)]="Number",p}astBinaryExpression(y,p){return p}astIdentifierExpression(y,p){return p}astAssignmentExpression(y,p){return p}astExpressionStatement(y,p){return y.expression.type==="AssignmentExpression"&&this.pushState("assignment-as-statement"),this.astGeneric(y.expression,p),p.push(";"),p}astEmptyStatement(y,p){return p}astBlockStatement(y,p){return p}astIfStatement(y,p){return p}astSwitchStatement(y,p){return p}astBreakStatement(y,p){return p.push("break;"),p}astContinueStatement(y,p){return p.push(`continue;
`),p}astForStatement(y,p){return p}astWhileStatement(y,p){return p}astDoWhileStatement(y,p){return p}astVariableDeclarator(y,p){return this.astGeneric(y.id,p),y.init!==null&&(p.push("="),this.astGeneric(y.init,p)),p}astThisExpression(y,p){return p}astSequenceExpression(y,p){const{expressions:g}=y,k=[];for(let o=0;o<g.length;o++){const l=g[o],x=[];this.astGeneric(l,x),k.push(x.join(""))}return k.length>1?p.push("(",k.join(","),")"):p.push(k[0]),p}astUnaryExpression(y,p){return this.checkAndUpconvertBitwiseUnary(y,p)||(y.prefix?(p.push(y.operator),this.astGeneric(y.argument,p)):(this.astGeneric(y.argument,p),p.push(y.operator))),p}checkAndUpconvertBitwiseUnary(y,p){}astUpdateExpression(y,p){return y.prefix?(p.push(y.operator),this.astGeneric(y.argument,p)):(this.astGeneric(y.argument,p),p.push(y.operator)),p}astLogicalExpression(y,p){return p.push("("),this.astGeneric(y.left,p),p.push(y.operator),this.astGeneric(y.right,p),p.push(")"),p}astMemberExpression(y,p){return p}astCallExpression(y,p){return p}astArrayExpression(y,p){return p}getMemberExpressionDetails(y){if(y.type!=="MemberExpression")throw this.astErrorOutput(`Expression ${y.type} not a MemberExpression`,y);let p=null,g=null;const k=this.getVariableSignature(y);switch(k){case"value":return null;case"value.thread.value":case"this.thread.value":case"this.output.value":return{signature:k,type:"Integer",name:y.property.name};case"value[]":if(typeof y.object.name!="string")throw this.astErrorOutput("Unexpected expression",y);return p=y.object.name,{name:p,origin:"user",signature:k,type:this.getVariableType(y.object),xProperty:y.property};case"value[][]":if(typeof y.object.object.name!="string")throw this.astErrorOutput("Unexpected expression",y);return p=y.object.object.name,{name:p,origin:"user",signature:k,type:this.getVariableType(y.object.object),yProperty:y.object.property,xProperty:y.property};case"value[][][]":if(typeof y.object.object.object.name!="string")throw this.astErrorOutput("Unexpected expression",y);return p=y.object.object.object.name,{name:p,origin:"user",signature:k,type:this.getVariableType(y.object.object.object),zProperty:y.object.object.property,yProperty:y.object.property,xProperty:y.property};case"value[][][][]":if(typeof y.object.object.object.object.name!="string")throw this.astErrorOutput("Unexpected expression",y);return p=y.object.object.object.object.name,{name:p,origin:"user",signature:k,type:this.getVariableType(y.object.object.object.object),zProperty:y.object.object.property,yProperty:y.object.property,xProperty:y.property};case"value.value":if(typeof y.property.name!="string")throw this.astErrorOutput("Unexpected expression",y);if(this.isAstMathVariable(y))return p=y.property.name,{name:p,origin:"Math",type:"Number",signature:k};switch(y.property.name){case"r":case"g":case"b":case"a":return p=y.object.name,{name:p,property:y.property.name,origin:"user",signature:k,type:"Number"};default:throw this.astErrorOutput("Unexpected expression",y)}case"this.constants.value":if(typeof y.property.name!="string")throw this.astErrorOutput("Unexpected expression",y);if(p=y.property.name,g=this.getConstantType(p),!g)throw this.astErrorOutput("Constant has no type",y);return{name:p,type:g,origin:"constants",signature:k};case"this.constants.value[]":if(typeof y.object.property.name!="string")throw this.astErrorOutput("Unexpected expression",y);if(p=y.object.property.name,g=this.getConstantType(p),!g)throw this.astErrorOutput("Constant has no type",y);return{name:p,type:g,origin:"constants",signature:k,xProperty:y.property};case"this.constants.value[][]":if(typeof y.object.object.property.name!="string")throw this.astErrorOutput("Unexpected expression",y);if(p=y.object.object.property.name,g=this.getConstantType(p),!g)throw this.astErrorOutput("Constant has no type",y);return{name:p,type:g,origin:"constants",signature:k,yProperty:y.object.property,xProperty:y.property};case"this.constants.value[][][]":if(typeof y.object.object.object.property.name!="string")throw this.astErrorOutput("Unexpected expression",y);if(p=y.object.object.object.property.name,g=this.getConstantType(p),!g)throw this.astErrorOutput("Constant has no type",y);return{name:p,type:g,origin:"constants",signature:k,zProperty:y.object.object.property,yProperty:y.object.property,xProperty:y.property};case"fn()[]":case"fn()[][]":case"[][]":return{signature:k,property:y.property};default:throw this.astErrorOutput("Unexpected expression",y)}}findIdentifierOrigin(y){const p=[this.ast];for(;p.length>0;){const g=p[0];if(g.type==="VariableDeclarator"&&g.id&&g.id.name&&g.id.name===y.name)return g;if(p.shift(),g.argument)p.push(g.argument);else if(g.body)p.push(g.body);else if(g.declarations)p.push(g.declarations);else if(Array.isArray(g))for(let k=0;k<g.length;k++)p.push(g[k])}return null}findLastReturn(y){const p=[y||this.ast];for(;p.length>0;){const g=p.pop();if(g.type==="ReturnStatement")return g;if(g.type!=="FunctionDeclaration")if(g.argument)p.push(g.argument);else if(g.body)p.push(g.body);else if(g.declarations)p.push(g.declarations);else if(Array.isArray(g))for(let k=0;k<g.length;k++)p.push(g[k]);else g.consequent?p.push(g.consequent):g.cases&&p.push(g.cases)}return null}getInternalVariableName(y){return this._internalVariableNames.hasOwnProperty(y)||(this._internalVariableNames[y]=0),this._internalVariableNames[y]++,this._internalVariableNames[y]===1?y:y+this._internalVariableNames[y]}astKey(y,p=","){if(!y.start||!y.end)throw new Error("AST start and end needed");return`${y.start}${p}${y.end}`}};const P={Number:"Number",Float:"Float",Integer:"Integer",Array:"Number","Array(2)":"Number","Array(3)":"Number","Array(4)":"Number","Matrix(2)":"Number","Matrix(3)":"Number","Matrix(4)":"Number",Array2D:"Number",Array3D:"Number",Input:"Number",HTMLCanvas:"Array(4)",OffscreenCanvas:"Array(4)",HTMLImage:"Array(4)",ImageBitmap:"Array(4)",ImageData:"Array(4)",HTMLVideo:"Array(4)",HTMLImageArray:"Array(4)",NumberTexture:"Number",MemoryOptimizedNumberTexture:"Number","Array1D(2)":"Array(2)","Array1D(3)":"Array(3)","Array1D(4)":"Array(4)","Array2D(2)":"Array(2)","Array2D(3)":"Array(3)","Array2D(4)":"Array(4)","Array3D(2)":"Array(2)","Array3D(3)":"Array(3)","Array3D(4)":"Array(4)","ArrayTexture(1)":"Number","ArrayTexture(2)":"Array(2)","ArrayTexture(3)":"Array(3)","ArrayTexture(4)":"Array(4)"};G.exports={FunctionNode:$}}),Ae=s((j,G)=>{const{FunctionNode:D}=Y();var I=class extends D{astFunction(_,c){if(!this.isRootKernel){c.push("function"),c.push(" "),c.push(this.name),c.push("(");for(let d=0;d<this.argumentNames.length;++d){const E=this.argumentNames[d];d>0&&c.push(", "),c.push("user_"),c.push(E)}c.push(`) {
`)}for(let d=0;d<_.body.body.length;++d)this.astGeneric(_.body.body[d],c),c.push(`
`);return this.isRootKernel||c.push(`}
`),c}astReturnStatement(_,c){const d=this.returnType||this.getType(_.argument);return this.returnType||(this.returnType=d),this.isRootKernel?(c.push(this.leadingReturnStatement),this.astGeneric(_.argument,c),c.push(`;
`),c.push(this.followingReturnStatement),c.push(`continue;
`)):this.isSubKernel?(c.push(`subKernelResult_${this.name} = `),this.astGeneric(_.argument,c),c.push(";"),c.push(`return subKernelResult_${this.name};`)):(c.push("return "),this.astGeneric(_.argument,c),c.push(";")),c}astLiteral(_,c){if(isNaN(_.value))throw this.astErrorOutput("Non-numeric literal not supported : "+_.value,_);return c.push(_.value),c}astBinaryExpression(_,c){return c.push("("),this.astGeneric(_.left,c),c.push(_.operator),this.astGeneric(_.right,c),c.push(")"),c}astIdentifierExpression(_,c){if(_.type!=="Identifier")throw this.astErrorOutput("IdentifierExpression - not an Identifier",_);switch(_.name){case"Infinity":c.push("Infinity");break;default:this.constants&&this.constants.hasOwnProperty(_.name)?c.push("constants_"+_.name):c.push("user_"+_.name)}return c}astForStatement(_,c){if(_.type!=="ForStatement")throw this.astErrorOutput("Invalid for statement",_);const d=[],E=[],$=[],P=[];let y=null;if(_.init){this.pushState("in-for-loop-init"),this.astGeneric(_.init,d);for(let p=0;p<d.length;p++)d[p].includes&&d[p].includes(",")&&(y=!1);this.popState("in-for-loop-init")}else y=!1;if(_.test?this.astGeneric(_.test,E):y=!1,_.update?(_.update.type==="AssignmentExpression"&&this.pushState("assignment-as-statement"),this.astGeneric(_.update,$)):y=!1,_.body&&(this.pushState("loop-body"),this.astGeneric(_.body,P),this.popState("loop-body")),y===null&&(y=this.isSafe(_.init)&&this.isSafe(_.test)),y)c.push(`for (${d.join("")};${E.join("")};${$.join("")}){
`),c.push(P.join("")),c.push(`}
`);else{const p=this.getInternalVariableName("safeI");d.length>0&&c.push(d.join(""),`;
`),c.push(`for (let ${p}=0;${p}<LOOP_MAX;${p}++){
`),E.length>0&&c.push(`if (!${E.join("")}) break;
`),c.push(P.join("")),c.push(`
${$.join("")};`),c.push(`}
`)}return c}astWhileStatement(_,c){if(_.type!=="WhileStatement")throw this.astErrorOutput("Invalid while statement",_);return c.push("for (let i = 0; i < LOOP_MAX; i++) {"),c.push("if ("),this.astGeneric(_.test,c),c.push(`) {
`),this.astGeneric(_.body,c),c.push(`} else {
`),c.push(`break;
`),c.push(`}
`),c.push(`}
`),c}astDoWhileStatement(_,c){if(_.type!=="DoWhileStatement")throw this.astErrorOutput("Invalid while statement",_);return c.push("for (let i = 0; i < LOOP_MAX; i++) {"),this.astGeneric(_.body,c),c.push("if (!"),this.astGeneric(_.test,c),c.push(`) {
`),c.push(`break;
`),c.push(`}
`),c.push(`}
`),c}astAssignmentExpression(_,c){const d=this.getDeclaration(_.left);if(d&&!d.assignable)throw this.astErrorOutput(`Variable ${_.left.name} is not assignable here`,_);const E=this.isState("assignment-as-statement");return E?this.popState("assignment-as-statement"):c.push("("),this.astGeneric(_.left,c),c.push(_.operator),this.astGeneric(_.right,c),E||c.push(")"),c}astBlockStatement(_,c){if(this.isState("loop-body")){this.pushState("block-body");for(let d=0;d<_.body.length;d++)this.astGeneric(_.body[d],c);this.popState("block-body")}else{c.push(`{
`);for(let d=0;d<_.body.length;d++)this.astGeneric(_.body[d],c);c.push(`}
`)}return c}astVariableDeclaration(_,c){c.push(`${_.kind} `);const{declarations:d}=_;for(let E=0;E<d.length;E++){E>0&&c.push(",");const $=d[E],P=this.getDeclaration($.id);P.valueType||(P.valueType=this.getType($.init)),this.astGeneric($,c)}return this.isState("in-for-loop-init")||c.push(";"),c}astIfStatement(_,c){return c.push("if ("),this.astGeneric(_.test,c),c.push(")"),_.consequent.type==="BlockStatement"?this.astGeneric(_.consequent,c):(c.push(` {
`),this.astGeneric(_.consequent,c),c.push(`
}
`)),_.alternate&&(c.push("else "),_.alternate.type==="BlockStatement"||_.alternate.type==="IfStatement"?this.astGeneric(_.alternate,c):(c.push(` {
`),this.astGeneric(_.alternate,c),c.push(`
}
`))),c}astSwitchStatement(_,c){const{discriminant:d,cases:E}=_;c.push("switch ("),this.astGeneric(d,c),c.push(`) {
`);for(let $=0;$<E.length;$++){if(E[$].test===null){c.push(`default:
`),this.astGeneric(E[$].consequent,c),E[$].consequent&&E[$].consequent.length>0&&c.push(`break;
`);continue}c.push("case "),this.astGeneric(E[$].test,c),c.push(`:
`),E[$].consequent&&E[$].consequent.length>0&&(this.astGeneric(E[$].consequent,c),c.push(`break;
`))}c.push(`
}`)}astThisExpression(_,c){return c.push("_this"),c}astMemberExpression(_,c){const{signature:d,type:E,property:$,xProperty:P,yProperty:y,zProperty:p,name:g,origin:k}=this.getMemberExpressionDetails(_);switch(d){case"this.thread.value":return c.push(`_this.thread.${g}`),c;case"this.output.value":switch(g){case"x":c.push("outputX");break;case"y":c.push("outputY");break;case"z":c.push("outputZ");break;default:throw this.astErrorOutput("Unexpected expression",_)}return c;case"value":throw this.astErrorOutput("Unexpected expression",_);case"value[]":case"value[][]":case"value[][][]":case"value.value":if(k==="Math")return c.push(Math[g]),c;switch($){case"r":return c.push(`user_${g}[0]`),c;case"g":return c.push(`user_${g}[1]`),c;case"b":return c.push(`user_${g}[2]`),c;case"a":return c.push(`user_${g}[3]`),c}break;case"this.constants.value":case"this.constants.value[]":case"this.constants.value[][]":case"this.constants.value[][][]":break;case"fn()[]":return this.astGeneric(_.object,c),c.push("["),this.astGeneric(_.property,c),c.push("]"),c;case"fn()[][]":return this.astGeneric(_.object.object,c),c.push("["),this.astGeneric(_.object.property,c),c.push("]"),c.push("["),this.astGeneric(_.property,c),c.push("]"),c;default:throw this.astErrorOutput("Unexpected expression",_)}if(!_.computed)switch(E){case"Number":case"Integer":case"Float":case"Boolean":return c.push(`${k}_${g}`),c}const o=`${k}_${g}`;switch(E){default:let l,x;if(k==="constants"){const w=this.constants[g];x=this.constantTypes[g]==="Input",l=x?w.size:null}else x=this.isInput(g),l=x?this.argumentSizes[this.argumentNames.indexOf(g)]:null;c.push(`${o}`),p&&y?x?(c.push("[("),this.astGeneric(p,c),c.push(`*${this.dynamicArguments?"(outputY * outputX)":l[1]*l[0]})+(`),this.astGeneric(y,c),c.push(`*${this.dynamicArguments?"outputX":l[0]})+`),this.astGeneric(P,c),c.push("]")):(c.push("["),this.astGeneric(p,c),c.push("]"),c.push("["),this.astGeneric(y,c),c.push("]"),c.push("["),this.astGeneric(P,c),c.push("]")):y?x?(c.push("[("),this.astGeneric(y,c),c.push(`*${this.dynamicArguments?"outputX":l[0]})+`),this.astGeneric(P,c),c.push("]")):(c.push("["),this.astGeneric(y,c),c.push("]"),c.push("["),this.astGeneric(P,c),c.push("]")):typeof P<"u"&&(c.push("["),this.astGeneric(P,c),c.push("]"))}return c}astCallExpression(_,c){if(_.type!=="CallExpression")throw this.astErrorOutput("Unknown CallExpression",_);let d=this.astMemberExpressionUnroll(_.callee);this.calledFunctions.indexOf(d)<0&&this.calledFunctions.push(d),this.isAstMathFunction(_),this.onFunctionCall&&this.onFunctionCall(this.name,d,_.arguments),c.push(d),c.push("(");const E=this.lookupFunctionArgumentTypes(d)||[];for(let $=0;$<_.arguments.length;++$){const P=_.arguments[$];let y=this.getType(P);E[$]||this.triggerImplyArgumentType(d,$,y,this),$>0&&c.push(", "),this.astGeneric(P,c)}return c.push(")"),c}astArrayExpression(_,c){const d=this.getType(_),E=_.elements.length,$=[];for(let P=0;P<E;++P){const y=[];this.astGeneric(_.elements[P],y),$.push(y.join(""))}switch(d){case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":c.push(`[${$.join(", ")}]`);break;default:c.push(`new Float32Array([${$.join(", ")}])`)}return c}astDebuggerStatement(_,c){return c.push("debugger;"),c}};G.exports={CPUFunctionNode:I}}),Ze=s((j,G)=>{const{utils:D}=m();function I(c,d){const E=[];for(const $ in d){if(!d.hasOwnProperty($))continue;const P=d[$],y=c[$];switch(P){case"Number":case"Integer":case"Float":case"Boolean":E.push(`${$}:${y}`);break;case"Array(2)":case"Array(3)":case"Array(4)":case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":E.push(`${$}:new ${y.constructor.name}(${JSON.stringify(Array.from(y))})`);break}}return`{ ${E.join()} }`}function _(c,d){const E=[],$=[],P=[],y=!/^function/.test(c.color.toString());if(E.push("  const { context, canvas, constants: incomingConstants } = settings;",`  const output = new Int32Array(${JSON.stringify(Array.from(c.output))});`,`  const _constantTypes = ${JSON.stringify(c.constantTypes)};`,`  const _constants = ${I(c.constants,c.constantTypes)};`),$.push("    constants: _constants,","    context,","    output,","    thread: {x: 0, y: 0, z: 0},"),c.graphical){E.push(`  const _imageData = context.createImageData(${c.output[0]}, ${c.output[1]});`),E.push(`  const _colorData = new Uint8ClampedArray(${c.output[0]} * ${c.output[1]} * 4);`);const k=D.flattenFunctionToString((y?"function ":"")+c.color.toString(),{thisLookup:l=>{switch(l){case"_colorData":return"_colorData";case"_imageData":return"_imageData";case"output":return"output";case"thread":return"this.thread"}return JSON.stringify(c[l])},findDependency:(l,x)=>null}),o=D.flattenFunctionToString((y?"function ":"")+c.getPixels.toString(),{thisLookup:l=>{switch(l){case"_colorData":return"_colorData";case"_imageData":return"_imageData";case"output":return"output";case"thread":return"this.thread"}return JSON.stringify(c[l])},findDependency:()=>null});$.push("    _imageData,","    _colorData,",`    color: ${k},`),P.push(`  kernel.getPixels = ${o};`)}const p=[],g=Object.keys(c.constantTypes);for(let k=0;k<g.length;k++)p.push(c.constantTypes[g]);if(c.argumentTypes.indexOf("HTMLImageArray")!==-1||p.indexOf("HTMLImageArray")!==-1){const k=D.flattenFunctionToString((y?"function ":"")+c._imageTo3DArray.toString(),{doNotDefine:["canvas"],findDependency:(o,l)=>o==="this"?(y?"function ":"")+c[l].toString():null,thisLookup:o=>{switch(o){case"canvas":return;case"context":return"context"}}});P.push(k),$.push("    _mediaTo2DArray,"),$.push("    _imageTo3DArray,")}else if(c.argumentTypes.indexOf("HTMLImage")!==-1||p.indexOf("HTMLImage")!==-1){const k=D.flattenFunctionToString((y?"function ":"")+c._mediaTo2DArray.toString(),{findDependency:(o,l)=>null,thisLookup:o=>{switch(o){case"canvas":return"settings.canvas";case"context":return"settings.context"}throw new Error("unhandled thisLookup")}});P.push(k),$.push("    _mediaTo2DArray,")}return`function(settings) {
${E.join(`
`)}
  for (const p in _constantTypes) {
    if (!_constantTypes.hasOwnProperty(p)) continue;
    const type = _constantTypes[p];
    switch (type) {
      case 'Number':
      case 'Integer':
      case 'Float':
      case 'Boolean':
      case 'Array(2)':
      case 'Array(3)':
      case 'Array(4)':
      case 'Matrix(2)':
      case 'Matrix(3)':
      case 'Matrix(4)':
        if (incomingConstants.hasOwnProperty(p)) {
          console.warn('constant ' + p + ' of type ' + type + ' cannot be resigned');
        }
        continue;
    }
    if (!incomingConstants.hasOwnProperty(p)) {
      throw new Error('constant ' + p + ' not found');
    }
    _constants[p] = incomingConstants[p];
  }
  const kernel = (function() {
${c._kernelString}
  })
    .apply({ ${$.join(`
`)} });
  ${P.join(`
`)}
  return kernel;
}`}G.exports={cpuKernelString:_}}),rt=s((j,G)=>{const{Kernel:D}=A(),{FunctionBuilder:I}=N(),{CPUFunctionNode:_}=Ae(),{utils:c}=m(),{cpuKernelString:d}=Ze();var E=class extends D{static getFeatures(){return this.features}static get features(){return Object.freeze({kernelMap:!0,isIntegerDivisionAccurate:!0})}static get isSupported(){return!0}static isContextMatch($){return!1}static get mode(){return"cpu"}static nativeFunctionArguments(){return null}static nativeFunctionReturnType(){throw new Error(`Looking up native function return type not supported on ${this.name}`)}static combineKernels($){return $}static getSignature($,P){return"cpu"+(P.length>0?":"+P.join(","):"")}constructor($,P){super($,P),this.mergeSettings($.settings||P),this._imageData=null,this._colorData=null,this._kernelString=null,this._prependedString=[],this.thread={x:0,y:0,z:0},this.translatedSources=null}initCanvas(){if(typeof document<"u")return document.createElement("canvas");if(typeof OffscreenCanvas<"u")return new OffscreenCanvas(0,0)}initContext(){return this.canvas?this.canvas.getContext("2d",{willReadFrequently:!0}):null}initPlugins($){return[]}validateSettings($){if(!this.output||this.output.length===0){if($.length!==1)throw new Error("Auto output only supported for kernels with only one input");const P=c.getVariableType($[0],this.strictIntegers);if(P==="Array")this.output=c.getDimensions(P);else if(P==="NumberTexture"||P==="ArrayTexture(4)")this.output=$[0].output;else throw new Error("Auto output not supported for input type: "+P)}if(this.graphical&&this.output.length!==2)throw new Error("Output must have 2 dimensions on graphical mode");this.checkOutput()}translateSource(){if(this.leadingReturnStatement=this.output.length>1?"resultX[x] = ":"result[x] = ",this.subKernels){const P=[];for(let y=0;y<this.subKernels.length;y++){const{name:p}=this.subKernels[y];P.push(this.output.length>1?`resultX_${p}[x] = subKernelResult_${p};
`:`result_${p}[x] = subKernelResult_${p};
`)}this.followingReturnStatement=P.join("")}const $=I.fromKernel(this,_);this.translatedSources=$.getPrototypes("kernel"),!this.graphical&&!this.returnType&&(this.returnType=$.getKernelResultType())}build(){if(this.built)return;if(this.randomSeed!==null&&console.warn("randomSeed is not supported in cpu mode; Math.random() will be unseeded"),this.setupConstants(),this.setupArguments(arguments),this.validateSettings(arguments),this.translateSource(),this.graphical){const{canvas:P,output:y}=this;if(!P)throw new Error("no canvas available for using graphical output");const p=y[0],g=y[1]||1;P.width=p,P.height=g,this._imageData=this.context.createImageData(p,g),this._colorData=new Uint8ClampedArray(p*g*4)}const $=this.getKernelString();this.kernelString=$,this.debug&&(console.log("Function output:"),console.log($));try{this.run=new Function([],$).bind(this)()}catch(P){console.error("An error occurred compiling the javascript: ",P)}this.buildSignature(arguments),this.built=!0}color($,P,y,p){typeof p>"u"&&(p=1),$=Math.floor($*255),P=Math.floor(P*255),y=Math.floor(y*255),p=Math.floor(p*255);const g=this.output[0],k=this.output[1],o=this.thread.x+(k-this.thread.y-1)*g;this._colorData[o*4+0]=$,this._colorData[o*4+1]=P,this._colorData[o*4+2]=y,this._colorData[o*4+3]=p}getKernelString(){if(this._kernelString!==null)return this._kernelString;let $=null,{translatedSources:P}=this;return P.length>1?P=P.filter(y=>/^function/.test(y)?y:($=y,!1)):$=P.shift(),this._kernelString=`  const LOOP_MAX = ${this._getLoopMaxString()};
  ${this.injectedNative||""}
  const _this = this;
  ${this._resultKernelHeader()}
  ${this._processConstants()}
  return (${this.argumentNames.map(y=>"user_"+y).join(", ")}) => {
    ${this._prependedString.join("")}
    ${this._earlyThrows()}
    ${this._processArguments()}
    ${this.graphical?this._graphicalKernelBody($):this._resultKernelBody($)}
    ${P.length>0?P.join(`
`):""}
  };`}toString(){return d(this)}_getLoopMaxString(){return this.loopMaxIterations?` ${parseInt(this.loopMaxIterations)};`:" 1000;"}_processConstants(){if(!this.constants)return"";const $=[];for(let P in this.constants)switch(this.constantTypes[P]){case"HTMLCanvas":case"OffscreenCanvas":case"HTMLImage":case"ImageBitmap":case"ImageData":case"HTMLVideo":$.push(`    const constants_${P} = this._mediaTo2DArray(this.constants.${P});
`);break;case"HTMLImageArray":$.push(`    const constants_${P} = this._imageTo3DArray(this.constants.${P});
`);break;case"Input":$.push(`    const constants_${P} = this.constants.${P}.value;
`);break;default:$.push(`    const constants_${P} = this.constants.${P};
`)}return $.join("")}_earlyThrows(){if(this.graphical||this.immutable||!this.pipeline)return"";const $=[];for(let y=0;y<this.argumentTypes.length;y++)this.argumentTypes[y]==="Array"&&$.push(this.argumentNames[y]);if($.length===0)return"";const P=[];for(let y=0;y<$.length;y++){const p=$[y],g=this._mapSubKernels(k=>`user_${p} === result_${k.name}`).join(" || ");P.push(`user_${p} === result${g?` || ${g}`:""}`)}return`if (${P.join(" || ")}) throw new Error('Source and destination arrays are the same.  Use immutable = true');`}_processArguments(){const $=[];for(let P=0;P<this.argumentTypes.length;P++){const y=`user_${this.argumentNames[P]}`;switch(this.argumentTypes[P]){case"HTMLCanvas":case"OffscreenCanvas":case"HTMLImage":case"ImageBitmap":case"ImageData":case"HTMLVideo":$.push(`    ${y} = this._mediaTo2DArray(${y});
`);break;case"HTMLImageArray":$.push(`    ${y} = this._imageTo3DArray(${y});
`);break;case"Input":$.push(`    ${y} = ${y}.value;
`);break;case"ArrayTexture(1)":case"ArrayTexture(2)":case"ArrayTexture(3)":case"ArrayTexture(4)":case"NumberTexture":case"MemoryOptimizedNumberTexture":$.push(`
    if (${y}.toArray) {
      if (!_this.textureCache) {
        _this.textureCache = [];
        _this.arrayCache = [];
      }
      const textureIndex = _this.textureCache.indexOf(${y});
      if (textureIndex !== -1) {
        ${y} = _this.arrayCache[textureIndex];
      } else {
        _this.textureCache.push(${y});
        ${y} = ${y}.toArray();
        _this.arrayCache.push(${y});
      }
    }`);break}}return $.join("")}_mediaTo2DArray($){const P=this.canvas,y=$.width>0?$.width:$.videoWidth,p=$.height>0?$.height:$.videoHeight;P.width<y&&(P.width=y),P.height<p&&(P.height=p);const g=this.context;let k;$.constructor===ImageData?k=$.data:(g.drawImage($,0,0,y,p),k=g.getImageData(0,0,y,p).data);const o=new Array(p);let l=0;for(let x=p-1;x>=0;x--){const w=o[x]=new Array(y);for(let v=0;v<y;v++){const C=new Float32Array(4);C[0]=k[l++]/255,C[1]=k[l++]/255,C[2]=k[l++]/255,C[3]=k[l++]/255,w[v]=C}}return o}getPixels($){const[P,y]=this.output;return $?c.flipPixels(this._imageData.data,P,y):this._imageData.data.slice(0)}_imageTo3DArray($){const P=new Array($.length);for(let y=0;y<$.length;y++)P[y]=this._mediaTo2DArray($[y]);return P}_resultKernelHeader(){if(this.graphical||this.immutable||!this.pipeline)return"";switch(this.output.length){case 1:return this._mutableKernel1DResults();case 2:return this._mutableKernel2DResults();case 3:return this._mutableKernel3DResults()}}_resultKernelBody($){switch(this.output.length){case 1:return(!this.immutable&&this.pipeline?this._resultMutableKernel1DLoop($):this._resultImmutableKernel1DLoop($))+this._kernelOutput();case 2:return(!this.immutable&&this.pipeline?this._resultMutableKernel2DLoop($):this._resultImmutableKernel2DLoop($))+this._kernelOutput();case 3:return(!this.immutable&&this.pipeline?this._resultMutableKernel3DLoop($):this._resultImmutableKernel3DLoop($))+this._kernelOutput();default:throw new Error("unsupported size kernel")}}_graphicalKernelBody($){switch(this.output.length){case 2:return this._graphicalKernel2DLoop($)+this._graphicalOutput();default:throw new Error("unsupported size kernel")}}_graphicalOutput(){return`
    this._imageData.data.set(this._colorData);
    this.context.putImageData(this._imageData, 0, 0);
    return;`}_getKernelResultTypeConstructorString(){switch(this.returnType){case"LiteralInteger":case"Number":case"Integer":case"Float":return"Float32Array";case"Array(2)":case"Array(3)":case"Array(4)":return"Array";default:if(this.graphical)return"Float32Array";throw new Error(`unhandled returnType ${this.returnType}`)}}_resultImmutableKernel1DLoop($){const P=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const result = new ${P}(outputX);
    ${this._mapSubKernels(y=>`const result_${y.name} = new ${P}(outputX);
`).join("    ")}
    ${this._mapSubKernels(y=>`let subKernelResult_${y.name};
`).join("    ")}
    for (let x = 0; x < outputX; x++) {
      this.thread.x = x;
      this.thread.y = 0;
      this.thread.z = 0;
      ${$}
    }`}_mutableKernel1DResults(){const $=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const result = new ${$}(outputX);
    ${this._mapSubKernels(P=>`const result_${P.name} = new ${$}(outputX);
`).join("    ")}
    ${this._mapSubKernels(P=>`let subKernelResult_${P.name};
`).join("    ")}`}_resultMutableKernel1DLoop($){return`  const outputX = _this.output[0];
    for (let x = 0; x < outputX; x++) {
      this.thread.x = x;
      this.thread.y = 0;
      this.thread.z = 0;
      ${$}
    }`}_resultImmutableKernel2DLoop($){const P=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    const result = new Array(outputY);
    ${this._mapSubKernels(y=>`const result_${y.name} = new Array(outputY);
`).join("    ")}
    ${this._mapSubKernels(y=>`let subKernelResult_${y.name};
`).join("    ")}
    for (let y = 0; y < outputY; y++) {
      this.thread.z = 0;
      this.thread.y = y;
      const resultX = result[y] = new ${P}(outputX);
      ${this._mapSubKernels(y=>`const resultX_${y.name} = result_${y.name}[y] = new ${P}(outputX);
`).join("")}
      for (let x = 0; x < outputX; x++) {
        this.thread.x = x;
        ${$}
      }
    }`}_mutableKernel2DResults(){const $=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    const result = new Array(outputY);
    ${this._mapSubKernels(P=>`const result_${P.name} = new Array(outputY);
`).join("    ")}
    ${this._mapSubKernels(P=>`let subKernelResult_${P.name};
`).join("    ")}
    for (let y = 0; y < outputY; y++) {
      const resultX = result[y] = new ${$}(outputX);
      ${this._mapSubKernels(P=>`const resultX_${P.name} = result_${P.name}[y] = new ${$}(outputX);
`).join("")}
    }`}_resultMutableKernel2DLoop($){const P=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    for (let y = 0; y < outputY; y++) {
      this.thread.z = 0;
      this.thread.y = y;
      const resultX = result[y];
      ${this._mapSubKernels(y=>`const resultX_${y.name} = result_${y.name}[y] = new ${P}(outputX);
`).join("")}
      for (let x = 0; x < outputX; x++) {
        this.thread.x = x;
        ${$}
      }
    }`}_graphicalKernel2DLoop($){return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    for (let y = 0; y < outputY; y++) {
      this.thread.z = 0;
      this.thread.y = y;
      for (let x = 0; x < outputX; x++) {
        this.thread.x = x;
        ${$}
      }
    }`}_resultImmutableKernel3DLoop($){const P=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    const outputZ = _this.output[2];
    const result = new Array(outputZ);
    ${this._mapSubKernels(y=>`const result_${y.name} = new Array(outputZ);
`).join("    ")}
    ${this._mapSubKernels(y=>`let subKernelResult_${y.name};
`).join("    ")}
    for (let z = 0; z < outputZ; z++) {
      this.thread.z = z;
      const resultY = result[z] = new Array(outputY);
      ${this._mapSubKernels(y=>`const resultY_${y.name} = result_${y.name}[z] = new Array(outputY);
`).join("      ")}
      for (let y = 0; y < outputY; y++) {
        this.thread.y = y;
        const resultX = resultY[y] = new ${P}(outputX);
        ${this._mapSubKernels(y=>`const resultX_${y.name} = resultY_${y.name}[y] = new ${P}(outputX);
`).join("        ")}
        for (let x = 0; x < outputX; x++) {
          this.thread.x = x;
          ${$}
        }
      }
    }`}_mutableKernel3DResults(){const $=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    const outputZ = _this.output[2];
    const result = new Array(outputZ);
    ${this._mapSubKernels(P=>`const result_${P.name} = new Array(outputZ);
`).join("    ")}
    ${this._mapSubKernels(P=>`let subKernelResult_${P.name};
`).join("    ")}
    for (let z = 0; z < outputZ; z++) {
      const resultY = result[z] = new Array(outputY);
      ${this._mapSubKernels(P=>`const resultY_${P.name} = result_${P.name}[z] = new Array(outputY);
`).join("      ")}
      for (let y = 0; y < outputY; y++) {
        const resultX = resultY[y] = new ${$}(outputX);
        ${this._mapSubKernels(P=>`const resultX_${P.name} = resultY_${P.name}[y] = new ${$}(outputX);
`).join("        ")}
      }
    }`}_resultMutableKernel3DLoop($){return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    const outputZ = _this.output[2];
    for (let z = 0; z < outputZ; z++) {
      this.thread.z = z;
      const resultY = result[z];
      for (let y = 0; y < outputY; y++) {
        this.thread.y = y;
        const resultX = resultY[y];
        for (let x = 0; x < outputX; x++) {
          this.thread.x = x;
          ${$}
        }
      }
    }`}_kernelOutput(){return this.subKernels?`
    return {
      result: result,
      ${this.subKernels.map($=>`${$.property}: result_${$.name}`).join(`,
      `)}
    };`:`
    return result;`}_mapSubKernels($){return this.subKernels===null?[""]:this.subKernels.map($)}destroy($){$&&delete this.canvas}static destroyContext($){}toJSON(){const $=super.toJSON();return $.functionNodes=I.fromKernel(this,_).toJSON(),$}setOutput($){super.setOutput($);const[P,y]=this.output;this.graphical&&(this._imageData=this.context.createImageData(P,y),this._colorData=new Uint8ClampedArray(P*y*4))}prependString($){if(this._kernelString)throw new Error("Kernel already built");this._prependedString.push($)}hasPrependString($){return this._prependedString.indexOf($)>-1}};G.exports={CPUKernel:E}}),cs=s((j,G)=>{G.exports={}}),jt=s((j,G)=>{const{Texture:D}=f();var I=class extends D{get textureType(){throw new Error(`"textureType" not implemented on ${this.name}`)}clone(){return new this.constructor(this)}beforeMutate(){return this.texture._refs>1?(this.newTexture(),!0):!1}cloneTexture(){this.texture._refs--;const{context:c,size:d,texture:E,kernel:$}=this;$.debug&&console.warn("cloning internal texture"),c.bindFramebuffer(c.FRAMEBUFFER,this.framebuffer()),_(c,E),c.framebufferTexture2D(c.FRAMEBUFFER,c.COLOR_ATTACHMENT0,c.TEXTURE_2D,E,0);const P=c.createTexture();_(c,P),c.texImage2D(c.TEXTURE_2D,0,this.internalFormat,d[0],d[1],0,this.textureFormat,this.textureType,null),c.copyTexSubImage2D(c.TEXTURE_2D,0,0,0,0,0,d[0],d[1]),P._refs=1,this.texture=P}newTexture(){this.texture._refs--;const c=this.context,d=this.size;this.kernel.debug&&console.warn("new internal texture");const E=c.createTexture();_(c,E),c.texImage2D(c.TEXTURE_2D,0,this.internalFormat,d[0],d[1],0,this.textureFormat,this.textureType,null),E._refs=1,this.texture=E}clear(){if(this.texture._refs){this.texture._refs--;const E=this.context,$=this.texture=E.createTexture();_(E,$);const P=this.size;$._refs=1,E.texImage2D(E.TEXTURE_2D,0,this.internalFormat,P[0],P[1],0,this.textureFormat,this.textureType,null)}const{context:c,texture:d}=this;c.bindFramebuffer(c.FRAMEBUFFER,this.framebuffer()),c.bindTexture(c.TEXTURE_2D,d),_(c,d),c.framebufferTexture2D(c.FRAMEBUFFER,c.COLOR_ATTACHMENT0,c.TEXTURE_2D,d,0),c.clearColor(0,0,0,0),c.clear(c.COLOR_BUFFER_BIT|c.DEPTH_BUFFER_BIT)}delete(){this._deleted||(this._deleted=!0,!(this.texture._refs&&(this.texture._refs--,this.texture._refs))&&(this.kernel&&this.kernel.deleteTexture?this.kernel.deleteTexture(this.texture):this.context.deleteTexture(this.texture)))}framebuffer(){return this._framebuffer||(this._framebuffer=this.kernel.getRawValueFramebuffer(this.size[0],this.size[1])),this._framebuffer}};function _(c,d){c.activeTexture(c.TEXTURE15),c.bindTexture(c.TEXTURE_2D,d),c.texParameteri(c.TEXTURE_2D,c.TEXTURE_WRAP_S,c.CLAMP_TO_EDGE),c.texParameteri(c.TEXTURE_2D,c.TEXTURE_WRAP_T,c.CLAMP_TO_EDGE),c.texParameteri(c.TEXTURE_2D,c.TEXTURE_MIN_FILTER,c.NEAREST),c.texParameteri(c.TEXTURE_2D,c.TEXTURE_MAG_FILTER,c.NEAREST)}G.exports={GLTexture:I}}),De=s((j,G)=>{const{utils:D}=m(),{GLTexture:I}=jt();var _=class extends I{get textureType(){return this.context.FLOAT}constructor(c){super(c),this.type="ArrayTexture(1)"}renderRawOutput(){const c=this.context,d=this.size;c.bindFramebuffer(c.FRAMEBUFFER,this.framebuffer()),c.framebufferTexture2D(c.FRAMEBUFFER,c.COLOR_ATTACHMENT0,c.TEXTURE_2D,this.texture,0);const E=new Float32Array(d[0]*d[1]*4);return c.readPixels(0,0,d[0],d[1],c.RGBA,c.FLOAT,E),E}renderValues(){return this._deleted?null:this.renderRawOutput()}toArray(){return D.erectFloat(this.renderValues(),this.output[0])}};G.exports={GLTextureFloat:_}}),At=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(2)"}toArray(){return D.erectArray2(this.renderValues(),this.output[0],this.output[1])}};G.exports={GLTextureArray2Float:_}}),Ps=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(2)"}toArray(){return D.erect2DArray2(this.renderValues(),this.output[0],this.output[1])}};G.exports={GLTextureArray2Float2D:_}}),hs=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(2)"}toArray(){return D.erect3DArray2(this.renderValues(),this.output[0],this.output[1],this.output[2])}};G.exports={GLTextureArray2Float3D:_}}),gn=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(3)"}toArray(){return D.erectArray3(this.renderValues(),this.output[0])}};G.exports={GLTextureArray3Float:_}}),ds=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(3)"}toArray(){return D.erect2DArray3(this.renderValues(),this.output[0],this.output[1])}};G.exports={GLTextureArray3Float2D:_}}),me=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(3)"}toArray(){return D.erect3DArray3(this.renderValues(),this.output[0],this.output[1],this.output[2])}};G.exports={GLTextureArray3Float3D:_}}),Ee=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(4)"}toArray(){return D.erectArray4(this.renderValues(),this.output[0])}};G.exports={GLTextureArray4Float:_}}),Ue=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(4)"}toArray(){return D.erect2DArray4(this.renderValues(),this.output[0],this.output[1])}};G.exports={GLTextureArray4Float2D:_}}),Dt=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(4)"}toArray(){return D.erect3DArray4(this.renderValues(),this.output[0],this.output[1],this.output[2])}};G.exports={GLTextureArray4Float3D:_}}),Qa=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(1)"}toArray(){return D.erect2DFloat(this.renderValues(),this.output[0],this.output[1])}};G.exports={GLTextureFloat2D:_}}),eo=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="ArrayTexture(1)"}toArray(){return D.erect3DFloat(this.renderValues(),this.output[0],this.output[1],this.output[2])}};G.exports={GLTextureFloat3D:_}}),to=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="MemoryOptimizedNumberTexture"}toArray(){return D.erectMemoryOptimizedFloat(this.renderValues(),this.output[0])}};G.exports={GLTextureMemoryOptimized:_}}),so=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="MemoryOptimizedNumberTexture"}toArray(){return D.erectMemoryOptimized2DFloat(this.renderValues(),this.output[0],this.output[1])}};G.exports={GLTextureMemoryOptimized2D:_}}),no=s((j,G)=>{const{utils:D}=m(),{GLTextureFloat:I}=De();var _=class extends I{constructor(c){super(c),this.type="MemoryOptimizedNumberTexture"}toArray(){return D.erectMemoryOptimized3DFloat(this.renderValues(),this.output[0],this.output[1],this.output[2])}};G.exports={GLTextureMemoryOptimized3D:_}}),zs=s((j,G)=>{const{utils:D}=m(),{GLTexture:I}=jt();var _=class extends I{get textureType(){return this.context.UNSIGNED_BYTE}constructor(c){super(c),this.type="NumberTexture"}renderRawOutput(){const{context:c}=this;c.bindFramebuffer(c.FRAMEBUFFER,this.framebuffer()),c.framebufferTexture2D(c.FRAMEBUFFER,c.COLOR_ATTACHMENT0,c.TEXTURE_2D,this.texture,0);const d=new Uint8Array(this.size[0]*this.size[1]*4);return c.readPixels(0,0,this.size[0],this.size[1],c.RGBA,c.UNSIGNED_BYTE,d),d}renderValues(){return this._deleted?null:new Float32Array(this.renderRawOutput().buffer)}toArray(){return D.erectPackedFloat(this.renderValues(),this.output[0])}};G.exports={GLTextureUnsigned:_}}),ro=s((j,G)=>{const{utils:D}=m(),{GLTextureUnsigned:I}=zs();var _=class extends I{constructor(c){super(c),this.type="NumberTexture"}toArray(){return D.erect2DPackedFloat(this.renderValues(),this.output[0],this.output[1])}};G.exports={GLTextureUnsigned2D:_}}),io=s((j,G)=>{const{utils:D}=m(),{GLTextureUnsigned:I}=zs();var _=class extends I{constructor(c){super(c),this.type="NumberTexture"}toArray(){return D.erect3DPackedFloat(this.renderValues(),this.output[0],this.output[1],this.output[2])}};G.exports={GLTextureUnsigned3D:_}}),ao=s((j,G)=>{const{GLTextureUnsigned:D}=zs();var I=class extends D{constructor(_){super(_),this.type="ArrayTexture(4)"}toArray(){return this.renderValues()}};G.exports={GLTextureGraphical:I}}),Ar=s((j,G)=>{const{Kernel:D}=A(),{utils:I}=m(),{GLTextureArray2Float:_}=At(),{GLTextureArray2Float2D:c}=Ps(),{GLTextureArray2Float3D:d}=hs(),{GLTextureArray3Float:E}=gn(),{GLTextureArray3Float2D:$}=ds(),{GLTextureArray3Float3D:P}=me(),{GLTextureArray4Float:y}=Ee(),{GLTextureArray4Float2D:p}=Ue(),{GLTextureArray4Float3D:g}=Dt(),{GLTextureFloat:k}=De(),{GLTextureFloat2D:o}=Qa(),{GLTextureFloat3D:l}=eo(),{GLTextureMemoryOptimized:x}=to(),{GLTextureMemoryOptimized2D:w}=so(),{GLTextureMemoryOptimized3D:v}=no(),{GLTextureUnsigned:C}=zs(),{GLTextureUnsigned2D:b}=ro(),{GLTextureUnsigned3D:T}=io(),{GLTextureGraphical:h}=ao();var F=class extends D{static get mode(){return"gpu"}static getIsFloatRead(){const z=new this(`function kernelFunction() {
      return 1;
    }`,{context:this.testContext,canvas:this.testCanvas,validate:!1,output:[1],precision:"single",returnType:"Number",tactic:"speed"});z.build(),z.run();const L=z.renderOutput();return z.destroy(!0),L[0]===1}static getIsIntegerDivisionAccurate(){function z(X,q){return X[this.thread.x]/q[this.thread.x]}const L=new this(z.toString(),{context:this.testContext,canvas:this.testCanvas,validate:!1,output:[2],returnType:"Number",precision:"unsigned",tactic:"speed"}),V=[[6,6030401],[3,3991]];L.build.apply(L,V),L.run.apply(L,V);const U=L.renderOutput();return L.destroy(!0),U[0]===2&&U[1]===1511}static getIsSpeedTacticSupported(){function z(X){return X[this.thread.x]}const L=new this(z.toString(),{context:this.testContext,canvas:this.testCanvas,validate:!1,output:[4],returnType:"Number",precision:"unsigned",tactic:"speed"}),V=[[0,1,2,3]];L.build.apply(L,V),L.run.apply(L,V);const U=L.renderOutput();return L.destroy(!0),Math.round(U[0])===0&&Math.round(U[1])===1&&Math.round(U[2])===2&&Math.round(U[3])===3}static get testCanvas(){throw new Error(`"testCanvas" not defined on ${this.name}`)}static get testContext(){throw new Error(`"testContext" not defined on ${this.name}`)}static getFeatures(){const z=this.testContext,L=this.getIsDrawBuffers();return Object.freeze({isFloatRead:this.getIsFloatRead(),isIntegerDivisionAccurate:this.getIsIntegerDivisionAccurate(),isSpeedTacticSupported:this.getIsSpeedTacticSupported(),isTextureFloat:this.getIsTextureFloat(),isDrawBuffers:L,kernelMap:L,channelCount:this.getChannelCount(),maxTextureSize:this.getMaxTextureSize(),lowIntPrecision:z.getShaderPrecisionFormat(z.FRAGMENT_SHADER,z.LOW_INT),lowFloatPrecision:z.getShaderPrecisionFormat(z.FRAGMENT_SHADER,z.LOW_FLOAT),mediumIntPrecision:z.getShaderPrecisionFormat(z.FRAGMENT_SHADER,z.MEDIUM_INT),mediumFloatPrecision:z.getShaderPrecisionFormat(z.FRAGMENT_SHADER,z.MEDIUM_FLOAT),highIntPrecision:z.getShaderPrecisionFormat(z.FRAGMENT_SHADER,z.HIGH_INT),highFloatPrecision:z.getShaderPrecisionFormat(z.FRAGMENT_SHADER,z.HIGH_FLOAT)})}static setupFeatureChecks(){throw new Error(`"setupFeatureChecks" not defined on ${this.name}`)}static getSignature(z,L){return z.getVariablePrecisionString()+(L.length>0?":"+L.join(","):"")}setFixIntegerDivisionAccuracy(z){return this.fixIntegerDivisionAccuracy=z,this}setPrecision(z){return this.precision=z,this}setFloatTextures(z){return I.warnDeprecated("method","setFloatTextures","setOptimizeFloatMemory"),this.floatTextures=z,this}static nativeFunctionArguments(z){const L=[],V=[],U=[],X=/^[a-zA-Z_]/,q=/[a-zA-Z_0-9]/;let W=0,ee=null,se=null;for(;W<z.length;){const Z=z[W],ie=z[W+1],he=U.length>0?U[U.length-1]:null;if(he==="FUNCTION_ARGUMENTS"&&Z==="/"&&ie==="*"){U.push("MULTI_LINE_COMMENT"),W+=2;continue}else if(he==="MULTI_LINE_COMMENT"&&Z==="*"&&ie==="/"){U.pop(),W+=2;continue}else if(he==="FUNCTION_ARGUMENTS"&&Z==="/"&&ie==="/"){U.push("COMMENT"),W+=2;continue}else if(he==="COMMENT"&&Z===`
`){U.pop(),W++;continue}else if(he===null&&Z==="("){U.push("FUNCTION_ARGUMENTS"),W++;continue}else if(he==="FUNCTION_ARGUMENTS"){if(Z===")"){U.pop();break}if(Z==="f"&&ie==="l"&&z[W+2]==="o"&&z[W+3]==="a"&&z[W+4]==="t"&&z[W+5]===" "){U.push("DECLARE_VARIABLE"),se="float",ee="",W+=6;continue}else if(Z==="i"&&ie==="n"&&z[W+2]==="t"&&z[W+3]===" "){U.push("DECLARE_VARIABLE"),se="int",ee="",W+=4;continue}else if(Z==="v"&&ie==="e"&&z[W+2]==="c"&&z[W+3]==="2"&&z[W+4]===" "){U.push("DECLARE_VARIABLE"),se="vec2",ee="",W+=5;continue}else if(Z==="v"&&ie==="e"&&z[W+2]==="c"&&z[W+3]==="3"&&z[W+4]===" "){U.push("DECLARE_VARIABLE"),se="vec3",ee="",W+=5;continue}else if(Z==="v"&&ie==="e"&&z[W+2]==="c"&&z[W+3]==="4"&&z[W+4]===" "){U.push("DECLARE_VARIABLE"),se="vec4",ee="",W+=5;continue}}else if(he==="DECLARE_VARIABLE"){if(ee===""){if(Z===" "){W++;continue}if(!X.test(Z))throw new Error("variable name is not expected string")}ee+=Z,q.test(ie)||(U.pop(),V.push(ee),L.push(O[se]))}W++}if(U.length>0)throw new Error("GLSL function was not parsable");return{argumentNames:V,argumentTypes:L}}static nativeFunctionReturnType(z){return O[z.match(/int|float|vec[2-4]/)[0]]}static combineKernels(z,L){z.apply(null,arguments);const{texSize:V,context:U,threadDim:X}=L.texSize;let q;if(L.precision==="single"){const W=V[0],ee=Math.ceil(V[1]/4);q=new Float32Array(W*ee*4*4),U.readPixels(0,0,W,ee*4,U.RGBA,U.FLOAT,q)}else{const W=new Uint8Array(V[0]*V[1]*4);U.readPixels(0,0,V[0],V[1],U.RGBA,U.UNSIGNED_BYTE,W),q=new Float32Array(W.buffer)}if(q=q.subarray(0,X[0]*X[1]*X[2]),L.output.length===1)return q;if(L.output.length===2)return I.splitArray(q,L.output[0]);if(L.output.length===3)return I.splitArray(q,L.output[0]*L.output[1]).map(function(W){return I.splitArray(W,L.output[0])})}constructor(z,L){super(z,L),this.transferValues=null,this.formatValues=null,this.TextureConstructor=null,this.renderOutput=null,this.renderRawOutput=null,this.texSize=null,this.translatedSource=null,this.compiledFragmentShader=null,this.compiledVertexShader=null,this.switchingKernels=null,this._textureSwitched=null,this._mappedTextureSwitched=null}checkTextureSize(){const{features:z}=this.constructor;if(this.texSize[0]>z.maxTextureSize||this.texSize[1]>z.maxTextureSize)throw new Error(`Texture size [${this.texSize[0]},${this.texSize[1]}] generated by kernel is larger than supported size [${z.maxTextureSize},${z.maxTextureSize}]`)}translateSource(){throw new Error(`"translateSource" not defined on ${this.constructor.name}`)}pickRenderStrategy(z){if(this.graphical)return this.renderRawOutput=this.readPackedPixelsToUint8Array,this.transferValues=L=>L,this.TextureConstructor=h,null;if(this.precision==="unsigned")if(this.renderRawOutput=this.readPackedPixelsToUint8Array,this.transferValues=this.readPackedPixelsToFloat32Array,this.pipeline)switch(this.renderOutput=this.renderTexture,this.subKernels!==null&&(this.renderKernels=this.renderKernelsToTextures),this.returnType){case"LiteralInteger":case"Float":case"Number":case"Integer":return this.output[2]>0?(this.TextureConstructor=T,null):this.output[1]>0?(this.TextureConstructor=b,null):(this.TextureConstructor=C,null);case"Array(2)":case"Array(3)":case"Array(4)":return this.requestFallback(z)}else switch(this.subKernels!==null&&(this.renderKernels=this.renderKernelsToArrays),this.returnType){case"LiteralInteger":case"Float":case"Number":case"Integer":return this.renderOutput=this.renderValues,this.output[2]>0?(this.TextureConstructor=T,this.formatValues=I.erect3DPackedFloat,null):this.output[1]>0?(this.TextureConstructor=b,this.formatValues=I.erect2DPackedFloat,null):(this.TextureConstructor=C,this.formatValues=I.erectPackedFloat,null);case"Array(2)":case"Array(3)":case"Array(4)":return this.requestFallback(z)}else if(this.precision==="single"){if(this.renderRawOutput=this.readFloatPixelsToFloat32Array,this.transferValues=this.readFloatPixelsToFloat32Array,this.pipeline)switch(this.renderOutput=this.renderTexture,this.subKernels!==null&&(this.renderKernels=this.renderKernelsToTextures),this.returnType){case"LiteralInteger":case"Float":case"Number":case"Integer":return this.optimizeFloatMemory?this.output[2]>0?(this.TextureConstructor=v,null):this.output[1]>0?(this.TextureConstructor=w,null):(this.TextureConstructor=x,null):this.output[2]>0?(this.TextureConstructor=l,null):this.output[1]>0?(this.TextureConstructor=o,null):(this.TextureConstructor=k,null);case"Array(2)":return this.output[2]>0?(this.TextureConstructor=d,null):this.output[1]>0?(this.TextureConstructor=c,null):(this.TextureConstructor=_,null);case"Array(3)":return this.output[2]>0?(this.TextureConstructor=P,null):this.output[1]>0?(this.TextureConstructor=$,null):(this.TextureConstructor=E,null);case"Array(4)":return this.output[2]>0?(this.TextureConstructor=g,null):this.output[1]>0?(this.TextureConstructor=p,null):(this.TextureConstructor=y,null)}if(this.renderOutput=this.renderValues,this.subKernels!==null&&(this.renderKernels=this.renderKernelsToArrays),this.optimizeFloatMemory)switch(this.returnType){case"LiteralInteger":case"Float":case"Number":case"Integer":return this.output[2]>0?(this.TextureConstructor=v,this.formatValues=I.erectMemoryOptimized3DFloat,null):this.output[1]>0?(this.TextureConstructor=w,this.formatValues=I.erectMemoryOptimized2DFloat,null):(this.TextureConstructor=x,this.formatValues=I.erectMemoryOptimizedFloat,null);case"Array(2)":return this.output[2]>0?(this.TextureConstructor=d,this.formatValues=I.erect3DArray2,null):this.output[1]>0?(this.TextureConstructor=c,this.formatValues=I.erect2DArray2,null):(this.TextureConstructor=_,this.formatValues=I.erectArray2,null);case"Array(3)":return this.output[2]>0?(this.TextureConstructor=P,this.formatValues=I.erect3DArray3,null):this.output[1]>0?(this.TextureConstructor=$,this.formatValues=I.erect2DArray3,null):(this.TextureConstructor=E,this.formatValues=I.erectArray3,null);case"Array(4)":return this.output[2]>0?(this.TextureConstructor=g,this.formatValues=I.erect3DArray4,null):this.output[1]>0?(this.TextureConstructor=p,this.formatValues=I.erect2DArray4,null):(this.TextureConstructor=y,this.formatValues=I.erectArray4,null)}else switch(this.returnType){case"LiteralInteger":case"Float":case"Number":case"Integer":return this.output[2]>0?(this.TextureConstructor=l,this.formatValues=I.erect3DFloat,null):this.output[1]>0?(this.TextureConstructor=o,this.formatValues=I.erect2DFloat,null):(this.TextureConstructor=k,this.formatValues=I.erectFloat,null);case"Array(2)":return this.output[2]>0?(this.TextureConstructor=d,this.formatValues=I.erect3DArray2,null):this.output[1]>0?(this.TextureConstructor=c,this.formatValues=I.erect2DArray2,null):(this.TextureConstructor=_,this.formatValues=I.erectArray2,null);case"Array(3)":return this.output[2]>0?(this.TextureConstructor=P,this.formatValues=I.erect3DArray3,null):this.output[1]>0?(this.TextureConstructor=$,this.formatValues=I.erect2DArray3,null):(this.TextureConstructor=E,this.formatValues=I.erectArray3,null);case"Array(4)":return this.output[2]>0?(this.TextureConstructor=g,this.formatValues=I.erect3DArray4,null):this.output[1]>0?(this.TextureConstructor=p,this.formatValues=I.erect2DArray4,null):(this.TextureConstructor=y,this.formatValues=I.erectArray4,null)}}else throw new Error(`unhandled precision of "${this.precision}"`);throw new Error(`unhandled return type "${this.returnType}"`)}getKernelString(){throw new Error("abstract method call")}getMainResultTexture(){switch(this.returnType){case"LiteralInteger":case"Float":case"Integer":case"Number":return this.getMainResultNumberTexture();case"Array(2)":return this.getMainResultArray2Texture();case"Array(3)":return this.getMainResultArray3Texture();case"Array(4)":return this.getMainResultArray4Texture();default:throw new Error(`unhandled returnType type ${this.returnType}`)}}getMainResultKernelNumberTexture(){throw new Error("abstract method call")}getMainResultSubKernelNumberTexture(){throw new Error("abstract method call")}getMainResultKernelArray2Texture(){throw new Error("abstract method call")}getMainResultSubKernelArray2Texture(){throw new Error("abstract method call")}getMainResultKernelArray3Texture(){throw new Error("abstract method call")}getMainResultSubKernelArray3Texture(){throw new Error("abstract method call")}getMainResultKernelArray4Texture(){throw new Error("abstract method call")}getMainResultSubKernelArray4Texture(){throw new Error("abstract method call")}getMainResultGraphical(){throw new Error("abstract method call")}getMainResultMemoryOptimizedFloats(){throw new Error("abstract method call")}getMainResultPackedPixels(){throw new Error("abstract method call")}getMainResultString(){return this.graphical?this.getMainResultGraphical():this.precision==="single"?this.optimizeFloatMemory?this.getMainResultMemoryOptimizedFloats():this.getMainResultTexture():this.getMainResultPackedPixels()}getMainResultNumberTexture(){return I.linesToString(this.getMainResultKernelNumberTexture())+I.linesToString(this.getMainResultSubKernelNumberTexture())}getMainResultArray2Texture(){return I.linesToString(this.getMainResultKernelArray2Texture())+I.linesToString(this.getMainResultSubKernelArray2Texture())}getMainResultArray3Texture(){return I.linesToString(this.getMainResultKernelArray3Texture())+I.linesToString(this.getMainResultSubKernelArray3Texture())}getMainResultArray4Texture(){return I.linesToString(this.getMainResultKernelArray4Texture())+I.linesToString(this.getMainResultSubKernelArray4Texture())}getFloatTacticDeclaration(){return`precision ${this.getVariablePrecisionString(this.texSize,this.tactic)} float;
`}getIntTacticDeclaration(){return`precision ${this.getVariablePrecisionString(this.texSize,this.tactic,!0)} int;
`}getSampler2DTacticDeclaration(){return`precision ${this.getVariablePrecisionString(this.texSize,this.tactic)} sampler2D;
`}getSampler2DArrayTacticDeclaration(){return`precision ${this.getVariablePrecisionString(this.texSize,this.tactic)} sampler2DArray;
`}renderTexture(){return this.immutable?this.texture.clone():this.texture}readPackedPixelsToUint8Array(){if(this.precision!=="unsigned")throw new Error('Requires this.precision to be "unsigned"');const{texSize:z,context:L}=this,V=new Uint8Array(z[0]*z[1]*4);return L.readPixels(0,0,z[0],z[1],L.RGBA,L.UNSIGNED_BYTE,V),V}readPackedPixelsToFloat32Array(){return new Float32Array(this.readPackedPixelsToUint8Array().buffer)}readFloatPixelsToFloat32Array(){if(this.precision!=="single")throw new Error('Requires this.precision to be "single"');const{texSize:z,context:L}=this,V=z[0],U=z[1],X=new Float32Array(V*U*4);return L.readPixels(0,0,V,U,L.RGBA,L.FLOAT,X),X}getPixels(z){const{context:L,output:V}=this,[U,X]=V,q=new Uint8Array(U*X*4);return L.readPixels(0,0,U,X,L.RGBA,L.UNSIGNED_BYTE,q),new Uint8ClampedArray((z?q:I.flipPixels(q,U,X)).buffer)}renderKernelsToArrays(){const z={result:this.renderOutput()};for(let L=0;L<this.subKernels.length;L++)z[this.subKernels[L].property]=this.mappedTextures[L].toArray();return z}renderKernelsToTextures(){const z={result:this.renderOutput()};if(this.immutable)for(let L=0;L<this.subKernels.length;L++)z[this.subKernels[L].property]=this.mappedTextures[L].clone();else for(let L=0;L<this.subKernels.length;L++)z[this.subKernels[L].property]=this.mappedTextures[L];return z}setOutput(z){const L=this.toKernelOutput(z);if(this.program){if(!this.dynamicOutput)throw new Error("Resizing a kernel with dynamicOutput: false is not possible");const V=[L[0],L[1]||1,L[2]||1],U=I.getKernelTextureSize({optimizeFloatMemory:this.optimizeFloatMemory,precision:this.precision},V),X=this.texSize;if(X){const W=this.getVariablePrecisionString(X,this.tactic),ee=this.getVariablePrecisionString(U,this.tactic);if(W!==ee){this.debug&&console.warn("Precision requirement changed, asking GPU instance to recompile"),this.switchKernels({type:"outputPrecisionMismatch",precision:ee,needed:z});return}}this.output=L,this.threadDim=V,this.texSize=U;const{context:q}=this;if(q.bindFramebuffer(q.FRAMEBUFFER,this.framebuffer),this.updateMaxTexSize(),this.framebuffer.width=this.texSize[0],this.framebuffer.height=this.texSize[1],q.viewport(0,0,this.maxTexSize[0],this.maxTexSize[1]),this.canvas.width=this.maxTexSize[0],this.canvas.height=this.maxTexSize[1],this.texture&&this.texture.delete(),this.texture=null,this._setupOutputTexture(),this.mappedTextures&&this.mappedTextures.length>0){for(let W=0;W<this.mappedTextures.length;W++)this.mappedTextures[W].delete();this.mappedTextures=null,this._setupSubOutputTextures()}}else this.output=L;return this}renderValues(){return this.formatValues(this.transferValues(),this.output[0],this.output[1],this.output[2])}getVariablePrecisionString(z=this.texSize,L=this.tactic,V=!1){if(!L){if(!this.constructor.features.isSpeedTacticSupported)return"highp";const U=this.constructor.features[V?"lowIntPrecision":"lowFloatPrecision"],X=this.constructor.features[V?"mediumIntPrecision":"mediumFloatPrecision"],q=this.constructor.features[V?"highIntPrecision":"highFloatPrecision"],W=Math.log2(z[0]*z[1]);if(W<=U.rangeMax)return"lowp";if(W<=X.rangeMax)return"mediump";if(W<=q.rangeMax)return"highp";throw new Error("The required size exceeds that of the ability of your system")}switch(L){case"speed":return"lowp";case"balanced":return"mediump";case"precision":return"highp";default:throw new Error(`Unknown tactic "${L}" use "speed", "balanced", "precision", or empty for auto`)}}updateTextureArgumentRefs(z,L){if(this.immutable){if(this.texture.texture===L.texture){const{prevArg:V}=z;V&&(V.texture._refs===1&&(this.texture.delete(),this.texture=V.clone(),this._textureSwitched=!0),V.delete()),z.prevArg=L.clone()}else if(this.mappedTextures&&this.mappedTextures.length>0){const{mappedTextures:V}=this;for(let U=0;U<V.length;U++){const X=V[U];if(X.texture===L.texture){const{prevArg:q}=z;q&&(q.texture._refs===1&&(X.delete(),V[U]=q.clone(),this._mappedTextureSwitched[U]=!0),q.delete()),z.prevArg=L.clone();return}}}}}onActivate(z){if(this._textureSwitched=!0,this.texture=z.texture,this.mappedTextures){for(let L=0;L<this.mappedTextures.length;L++)this._mappedTextureSwitched[L]=!0;this.mappedTextures=z.mappedTextures}}initCanvas(){}};const O={int:"Integer",float:"Number",vec2:"Array(2)",vec3:"Array(3)",vec4:"Array(4)"};G.exports={GLKernel:F}}),yn=s((j,G)=>{const{utils:D}=m(),{FunctionNode:I}=Y(),_={"<":"ceil",">=":"ceil",">":"floor","<=":"floor"};var c=class extends I{constructor(o,l){super(o,l),l&&l.hasOwnProperty("fixIntegerDivisionAccuracy")&&(this.fixIntegerDivisionAccuracy=l.fixIntegerDivisionAccuracy)}astConditionalExpression(o,l){if(o.type!=="ConditionalExpression")throw this.astErrorOutput("Not a conditional expression",o);const x=this.getType(o.consequent),w=this.getType(o.alternate);return x===null&&w===null?(l.push("if ("),this.astGeneric(o.test,l),l.push(") {"),this.astGeneric(o.consequent,l),l.push(";"),l.push("} else {"),this.astGeneric(o.alternate,l),l.push(";"),l.push("}"),l):(l.push("("),this.astGeneric(o.test,l),l.push("?"),this.astGeneric(o.consequent,l),l.push(":"),this.astGeneric(o.alternate,l),l.push(")"),l)}astFunction(o,l){if(this.isRootKernel)l.push("void");else{this.returnType||this.findLastReturn()&&(this.returnType=this.getType(o.body),this.returnType==="LiteralInteger"&&(this.returnType="Number"));const{returnType:x}=this;if(!x)l.push("void");else{const w=g[x];if(!w)throw new Error(`unknown type ${x}`);l.push(w)}}if(l.push(" "),l.push(this.name),l.push("("),!this.isRootKernel)for(let x=0;x<this.argumentNames.length;++x){const w=this.argumentNames[x];x>0&&l.push(", ");let v=this.argumentTypes[this.argumentNames.indexOf(w)];if(!v)throw this.astErrorOutput(`Unknown argument ${w} type`,o);v==="LiteralInteger"&&(this.argumentTypes[x]=v="Number");const C=g[v];if(!C)throw this.astErrorOutput("Unexpected expression",o);const b=D.sanitizeName(w);C==="sampler2D"||C==="sampler2DArray"?l.push(`${C} user_${b},ivec2 user_${b}Size,ivec3 user_${b}Dim`):l.push(`${C} user_${b}`)}l.push(`) {
`);for(let x=0;x<o.body.body.length;++x)this.astStatementWithHoisting(o.body.body[x],l),l.push(`
`);return l.push(`}
`),l}astReturnStatement(o,l){if(!o.argument)throw this.astErrorOutput("Unexpected return statement",o);this.pushState("skip-literal-correction");const x=this.getType(o.argument);this.popState("skip-literal-correction");const w=[];switch(this.returnType||(x==="LiteralInteger"||x==="Integer"?this.returnType="Number":this.returnType=x),this.returnType){case"LiteralInteger":case"Number":case"Float":switch(x){case"Integer":w.push("float("),this.astGeneric(o.argument,w),w.push(")");break;case"LiteralInteger":this.castLiteralToFloat(o.argument,w),this.getType(o)==="Integer"&&(w.unshift("float("),w.push(")"));break;default:this.astGeneric(o.argument,w)}break;case"Integer":switch(x){case"Float":case"Number":this.castValueToInteger(o.argument,w);break;case"LiteralInteger":this.castLiteralToInteger(o.argument,w);break;default:this.astGeneric(o.argument,w)}break;case"Array(4)":case"Array(3)":case"Array(2)":case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":case"Input":this.astGeneric(o.argument,w);break;default:throw this.astErrorOutput(`unhandled return type ${this.returnType}`,o)}return this.isRootKernel?(l.push(`kernelResult = ${w.join("")};`),l.push("return;")):this.isSubKernel?(l.push(`subKernelResult_${this.name} = ${w.join("")};`),l.push(`return subKernelResult_${this.name};`)):l.push(`return ${w.join("")};`),l}astLiteral(o,l){if(isNaN(o.value))throw this.astErrorOutput("Non-numeric literal not supported : "+o.value,o);const x=this.astKey(o);return Number.isInteger(o.value)?this.isState("casting-to-integer")||this.isState("building-integer")?(this.literalTypes[x]="Integer",l.push(`${o.value}`)):this.isState("casting-to-float")||this.isState("building-float")?(this.literalTypes[x]="Number",l.push(`${o.value}.0`)):(this.literalTypes[x]="Number",l.push(`${o.value}.0`)):this.isState("casting-to-integer")||this.isState("building-integer")?(this.literalTypes[x]="Integer",l.push(Math.round(o.value))):(this.literalTypes[x]="Number",l.push(`${o.value}`)),l}astBinaryExpression(o,l){if(this.checkAndUpconvertOperator(o,l))return l;if(o.operator==="/"){const C=this.fixIntegerDivisionAccuracy;switch(l.push(C?"divWithIntCheck(":"("),this.pushState("building-float"),this.getType(o.left)){case"Integer":this.castValueToFloat(o.left,l);break;case"LiteralInteger":this.castLiteralToFloat(o.left,l);break;default:this.astGeneric(o.left,l)}switch(l.push(C?", ":"/"),this.getType(o.right)){case"Integer":this.castValueToFloat(o.right,l);break;case"LiteralInteger":this.castLiteralToFloat(o.right,l);break;default:this.astGeneric(o.right,l)}return this.popState("building-float"),l.push(")"),l}l.push("(");const x=this.getType(o.left)||"Number",w=this.getType(o.right)||"Number",v=x+" & "+w;switch(v){case"Integer & Integer":this.pushState("building-integer"),this.astGeneric(o.left,l),l.push(k[o.operator]||o.operator),this.astGeneric(o.right,l),this.popState("building-integer");break;case"Number & Float":case"Float & Number":case"Float & Float":case"Number & Number":this.pushState("building-float"),this.astGeneric(o.left,l),l.push(k[o.operator]||o.operator),this.astGeneric(o.right,l),this.popState("building-float");break;case"LiteralInteger & LiteralInteger":this.isState("casting-to-integer")||this.isState("building-integer")?(this.pushState("building-integer"),this.astGeneric(o.left,l),l.push(k[o.operator]||o.operator),this.astGeneric(o.right,l),this.popState("building-integer")):(this.pushState("building-float"),this.castLiteralToFloat(o.left,l),l.push(k[o.operator]||o.operator),this.castLiteralToFloat(o.right,l),this.popState("building-float"));break;case"Integer & Float":case"Integer & Number":{const C=_[o.operator];if(C){this.pushState("building-integer"),this.astGeneric(o.left,l),l.push(k[o.operator]||o.operator),o.right.type==="Literal"&&typeof o.right.value=="number"?l.push(`${Math[C](o.right.value)}`):(l.push(`int(${C}(`),this.pushState("building-float"),this.astGeneric(o.right,l),this.popState("building-float"),l.push("))")),this.popState("building-integer");break}this.pushState("building-float"),this.castValueToFloat(o.left,l),l.push(k[o.operator]||o.operator),this.astGeneric(o.right,l),this.popState("building-float");break}case"Integer & LiteralInteger":this.pushState("building-integer"),this.astGeneric(o.left,l),l.push(k[o.operator]||o.operator),this.castLiteralToInteger(o.right,l),this.popState("building-integer");break;case"Number & Integer":this.pushState("building-float"),this.astGeneric(o.left,l),l.push(k[o.operator]||o.operator),this.castValueToFloat(o.right,l),this.popState("building-float");break;case"Float & LiteralInteger":case"Number & LiteralInteger":this.pushState("building-float"),this.astGeneric(o.left,l),l.push(k[o.operator]||o.operator),this.castLiteralToFloat(o.right,l),this.popState("building-float");break;case"LiteralInteger & Float":case"LiteralInteger & Number":this.isState("casting-to-integer")?(this.pushState("building-integer"),this.castLiteralToInteger(o.left,l),l.push(k[o.operator]||o.operator),this.castValueToInteger(o.right,l),this.popState("building-integer")):(this.pushState("building-float"),this.astGeneric(o.left,l),l.push(k[o.operator]||o.operator),this.pushState("casting-to-float"),this.astGeneric(o.right,l),this.popState("casting-to-float"),this.popState("building-float"));break;case"LiteralInteger & Integer":this.pushState("building-integer"),this.castLiteralToInteger(o.left,l),l.push(k[o.operator]||o.operator),this.astGeneric(o.right,l),this.popState("building-integer");break;case"Boolean & Boolean":this.pushState("building-boolean"),this.astGeneric(o.left,l),l.push(k[o.operator]||o.operator),this.astGeneric(o.right,l),this.popState("building-boolean");break;case"Float & Integer":this.pushState("building-float"),this.astGeneric(o.left,l),l.push(k[o.operator]||o.operator),this.castValueToFloat(o.right,l),this.popState("building-float");break;default:throw this.astErrorOutput(`Unhandled binary expression between ${v}`,o)}return l.push(")"),l}checkAndUpconvertOperator(o,l){const x=this.checkAndUpconvertBitwiseOperators(o,l);if(x)return x;const w={"%":this.fixIntegerDivisionAccuracy?"integerCorrectionModulo":"modulo","**":"pow"}[o.operator];if(!w)return null;switch(l.push(w),l.push("("),this.getType(o.left)){case"Integer":this.castValueToFloat(o.left,l);break;case"LiteralInteger":this.castLiteralToFloat(o.left,l);break;default:this.astGeneric(o.left,l)}switch(l.push(","),this.getType(o.right)){case"Integer":this.castValueToFloat(o.right,l);break;case"LiteralInteger":this.castLiteralToFloat(o.right,l);break;default:this.astGeneric(o.right,l)}return l.push(")"),l}checkAndUpconvertBitwiseOperators(o,l){const x={"&":"bitwiseAnd","|":"bitwiseOr","^":"bitwiseXOR","<<":"bitwiseZeroFillLeftShift",">>":"bitwiseSignedRightShift",">>>":"bitwiseZeroFillRightShift"}[o.operator];if(!x)return null;switch(l.push(x),l.push("("),this.getType(o.left)){case"Number":case"Float":this.castValueToInteger(o.left,l);break;case"LiteralInteger":this.castLiteralToInteger(o.left,l);break;default:this.astGeneric(o.left,l)}switch(l.push(","),this.getType(o.right)){case"Number":case"Float":this.castValueToInteger(o.right,l);break;case"LiteralInteger":this.castLiteralToInteger(o.right,l);break;default:this.astGeneric(o.right,l)}return l.push(")"),l}checkAndUpconvertBitwiseUnary(o,l){const x={"~":"bitwiseNot"}[o.operator];if(!x)return null;switch(l.push(x),l.push("("),this.getType(o.argument)){case"Number":case"Float":this.castValueToInteger(o.argument,l);break;case"LiteralInteger":this.castLiteralToInteger(o.argument,l);break;default:this.astGeneric(o.argument,l)}return l.push(")"),l}castLiteralToInteger(o,l){return this.pushState("casting-to-integer"),this.astGeneric(o,l),this.popState("casting-to-integer"),l}castLiteralToFloat(o,l){return this.pushState("casting-to-float"),this.astGeneric(o,l),this.popState("casting-to-float"),l}castValueToInteger(o,l){return this.pushState("casting-to-integer"),l.push("int("),this.astGeneric(o,l),l.push(")"),this.popState("casting-to-integer"),l}castValueToFloat(o,l){return this.pushState("casting-to-float"),l.push("float("),this.astGeneric(o,l),l.push(")"),this.popState("casting-to-float"),l}astIdentifierExpression(o,l){if(o.type!=="Identifier")throw this.astErrorOutput("IdentifierExpression - not an Identifier",o);const x=this.getType(o),w=D.sanitizeName(o.name);return o.name==="Infinity"?l.push("3.402823466e+38"):x==="Boolean"?this.argumentNames.indexOf(w)>-1?l.push(`bool(user_${w})`):l.push(`user_${w}`):l.push(`user_${w}`),l}astForStatement(o,l){if(o.type!=="ForStatement")throw this.astErrorOutput("Invalid for statement",o);const x=[],w=[],v=[],C=[];let b=null;if(o.init){const{declarations:T}=o.init;T.length>1&&(b=!1),this.astGeneric(o.init,x);for(let h=0;h<T.length;h++)T[h].init&&T[h].init.type!=="Literal"&&(b=!1)}else b=!1;if(o.test?this.astGeneric(o.test,w):b=!1,o.update?(o.update.type==="AssignmentExpression"&&this.pushState("assignment-as-statement"),this.astGeneric(o.update,v)):b=!1,o.body&&(this.pushState("loop-body"),this.astGeneric(o.body,C),this.popState("loop-body")),b===null&&(b=this.isSafe(o.init)&&this.isSafe(o.test)),b){const T=x.join(""),h=T[T.length-1]!==";";l.push(`for (${T}${h?";":""}${w.join("")};${v.join("")}){
`),l.push(C.join("")),l.push(`}
`)}else{const T=this.getInternalVariableName("safeI");x.length>0&&l.push(x.join(""),`
`),l.push(`for (int ${T}=0;${T}<LOOP_MAX;${T}++){
`),w.length>0&&l.push(`if (!${w.join("")}) break;
`),l.push(C.join("")),l.push(`
${v.join("")};`),l.push(`}
`)}return l}astWhileStatement(o,l){if(o.type!=="WhileStatement")throw this.astErrorOutput("Invalid while statement",o);const x=this.getInternalVariableName("safeI");return l.push(`for (int ${x}=0;${x}<LOOP_MAX;${x}++){
`),l.push("if (!"),this.astGeneric(o.test,l),l.push(`) break;
`),this.astGeneric(o.body,l),l.push(`}
`),l}astDoWhileStatement(o,l){if(o.type!=="DoWhileStatement")throw this.astErrorOutput("Invalid while statement",o);const x=this.getInternalVariableName("safeI");return l.push(`for (int ${x}=0;${x}<LOOP_MAX;${x}++){
`),this.astGeneric(o.body,l),l.push("if (!"),this.astGeneric(o.test,l),l.push(`) break;
`),l.push(`}
`),l}astAssignmentExpression(o,l){const x=this.isState("assignment-as-statement");if(x?this.popState("assignment-as-statement"):l.push("("),o.operator==="%=")this.astGeneric(o.left,l),l.push("="),l.push("mod("),this.astGeneric(o.left,l),l.push(","),this.astGeneric(o.right,l),l.push(")");else if(o.operator==="**=")this.astGeneric(o.left,l),l.push("="),l.push("pow("),this.astGeneric(o.left,l),l.push(","),this.astGeneric(o.right,l),l.push(")");else{const w=this.getType(o.left),v=this.getType(o.right);this.astGeneric(o.left,l),l.push(o.operator),w!=="Integer"&&v==="Integer"?(l.push("float("),this.astGeneric(o.right,l),l.push(")")):this.astGeneric(o.right,l)}return x||l.push(")"),l}astBlockStatement(o,l){if(this.isState("loop-body")){this.pushState("block-body");for(let x=0;x<o.body.length;x++)this.astStatementWithHoisting(o.body[x],l);this.popState("block-body")}else{l.push(`{
`);for(let x=0;x<o.body.length;x++)this.astStatementWithHoisting(o.body[x],l);l.push(`}
`)}return l}traceFunctionAST(o){this.normalizeBlock(o.body),super.traceFunctionAST(o)}normalizeBlock(o){if(!o||o.type!=="BlockStatement")return;const l=o.body;for(let x=0;x<l.length;x++){const w=l[x];switch(w.type){case"ExpressionStatement":case"VariableDeclaration":case"ReturnStatement":if(!y(w)&&E(w)||P(w)){const v=this.linearizeStatement(w);v!==null&&(l.splice(x,1,...v),x+=v.length-1)}break;case"IfStatement":if(E(w.test)||P(w.test)){const v={type:"VariableDeclaration",kind:"const",declarations:[{type:"VariableDeclarator",id:{type:"Identifier",name:`hoistSeqIf${this.linearTempId=(this.linearTempId||0)+1}`},init:w.test}]},C=this.linearizeStatement(v);if(C!==null){const b={type:"Identifier",name:v.declarations[0].id.name,start:this.syntheticNodeId,end:this.syntheticNodeId+1};this.syntheticNodeId+=2,w.test=b,l.splice(x,0,...C),x+=C.length}}this.normalizeBranch(w,"consequent"),this.normalizeBranch(w,"alternate");break;case"ForStatement":case"WhileStatement":case"DoWhileStatement":{const v=this.normalizeLoopHeader(w);if(v!==null){l.splice(x,1,v),this.normalizeBlock(v),x--;break}this.normalizeBranch(w,"body");break}case"SwitchStatement":if(E(w.discriminant)){const v={type:"VariableDeclaration",kind:"const",declarations:[{type:"VariableDeclarator",id:{type:"Identifier",name:`hoistSeqIf${this.linearTempId=(this.linearTempId||0)+1}`},init:w.discriminant}]},C=this.linearizeStatement(v);C!==null&&(w.discriminant={type:"Identifier",name:v.declarations[0].id.name,start:this.syntheticNodeId,end:this.syntheticNodeId+1},this.syntheticNodeId+=2,l.splice(x,0,...C),x+=C.length)}for(let v=0;v<w.cases.length;v++){const C={type:"BlockStatement",body:w.cases[v].consequent};this.normalizeBlock(C),w.cases[v].consequent=C.body}break;case"BlockStatement":this.normalizeBlock(w);break}}}normalizeBranch(o,l){const x=o[l];if(x){if(x.type==="BlockStatement"){this.normalizeBlock(x);return}!E(x)&&!P(x)||(o[l]={type:"BlockStatement",body:[x]},this.normalizeBlock(o[l]))}}normalizeLoopHeader(o){const{type:l}=o,x=l==="ForStatement"?o.init:null,w=o.test||null,v=l==="ForStatement"?o.update:null;if(![x,w,v].some(U=>U!==null&&(E(U)||P(U))))return null;const C=U=>JSON.parse(JSON.stringify(U)),b=U=>({type:"IfStatement",test:{type:"UnaryExpression",operator:"!",prefix:!0,argument:U},consequent:{type:"BlockStatement",body:[{type:"BreakStatement",label:null}]},alternate:null}),T=U=>U.type==="VariableDeclaration"?U:{type:"ExpressionStatement",expression:U},h=o.body.type==="BlockStatement"?o.body.body.slice():[o.body],F=(U,X)=>{const q=W=>{if(!W||typeof W!="object")return W;if(Array.isArray(W))return W.map(q);switch(W.type){case"ContinueStatement":return{type:"BlockStatement",body:[...X(),W]};case"ForStatement":case"WhileStatement":case"DoWhileStatement":return W;case"IfStatement":return{...W,consequent:q(W.consequent),alternate:q(W.alternate)};case"BlockStatement":return{...W,body:W.body.map(q)};case"SwitchStatement":return{...W,cases:W.cases.map(ee=>({...ee,consequent:ee.consequent.map(q)}))};default:return W}};return U.map(q)},O=[];l==="DoWhileStatement"?(O.push(...w?F(h,()=>[b(C(w))]):h),w&&O.push(b(w))):(w&&O.push(b(w)),O.push(...v?F(h,()=>[T(C(v))]):h),v&&O.push(T(v)));const z={type:"BlockStatement",body:[...x?[T(x)]:[],{type:"WhileStatement",test:{type:"Literal",value:!0,raw:"true"},body:{type:"BlockStatement",body:O}}]};let L=this.syntheticNodeId||1073741824;const V=U=>{if(!(!U||typeof U!="object")){if(Array.isArray(U)){U.forEach(V);return}typeof U.type=="string"&&U.start===void 0&&(U.start=L,U.end=L+1,L+=2);for(const X in U)X==="loc"||X==="range"||X==="parent"||V(U[X])}};return V(z),this.syntheticNodeId=L,z}linearizeStatement(o){const l=[];let x=!1,w=this.linearTempId||0;const v=z=>({type:"Identifier",name:z}),C=(z,L,V)=>({type:"VariableDeclaration",kind:z,declarations:[{type:"VariableDeclarator",id:v(L),init:V}]}),b=(z,L)=>{const V=`hoistSeq${w++}`;return z.push(C("const",V,L)),v(V)},T=z=>!d(z),h=(z,L)=>{if(x||!z||typeof z!="object")return z;switch(z.type){case"Identifier":case"Literal":case"ThisExpression":return z;case"MemberExpression":{const V=h(z.object,L),U=z.computed?h(z.property,L):z.property;return{...z,object:V,property:U}}case"CallExpression":{const V=z.arguments.map(U=>h(U,L));if(z.callee.type==="Identifier")for(let U=0;U<V.length;U++)$(V[U],z.callee.name)&&(V[U]=b(L,V[U]));return{...z,arguments:V}}case"BinaryExpression":{const V=h(z.left,L),U=T(z.right)?b(L,V):V;return{...z,left:U,right:h(z.right,L)}}case"UnaryExpression":return{...z,argument:h(z.argument,L)};case"ArrayExpression":return{...z,elements:z.elements.map(V=>h(V,L))};case"UpdateExpression":{if(z.argument.type!=="Identifier")return x=!0,z;if(z.prefix)return L.push({type:"ExpressionStatement",expression:z}),b(L,z.argument);const V=b(L,z.argument);return L.push({type:"ExpressionStatement",expression:z}),V}case"AssignmentExpression":{if(z.left.type!=="Identifier")return x=!0,z;const V=h(z.right,L);return L.push({type:"ExpressionStatement",expression:{...z,right:V}}),b(L,z.left)}case"SequenceExpression":for(let V=0;V<z.expressions.length-1;V++){const U=h(z.expressions[V],L);(U.type==="UpdateExpression"||U.type==="AssignmentExpression")&&L.push({type:"ExpressionStatement",expression:U})}return h(z.expressions[z.expressions.length-1],L);case"ConditionalExpression":{if(!T(z.consequent)&&!T(z.alternate))return{...z,test:h(z.test,L)};const V=h(z.test,L),U=`hoistSeq${w++}`;L.push(C("let",U,{type:"Literal",value:0,raw:"0"}));const X=[],q=[],W=h(z.consequent,X),ee=h(z.alternate,q),se=(Z,ie)=>({type:"ExpressionStatement",expression:{type:"AssignmentExpression",operator:"=",left:v(Z),right:ie}});return X.push(se(U,W)),q.push(se(U,ee)),L.push({type:"IfStatement",test:V,consequent:{type:"BlockStatement",body:X},alternate:{type:"BlockStatement",body:q}}),v(U)}case"LogicalExpression":{if(!T(z.right))return{...z,left:h(z.left,L)};const V=h(z.left,L),U=`hoistSeq${w++}`;L.push(C("let",U,V));const X=[],q=h(z.right,X);return X.push({type:"ExpressionStatement",expression:{type:"AssignmentExpression",operator:"=",left:v(U),right:q}}),L.push({type:"IfStatement",test:z.operator==="&&"?v(U):{type:"UnaryExpression",operator:"!",prefix:!0,argument:v(U)},consequent:{type:"BlockStatement",body:X},alternate:null}),v(U)}default:return x=!0,z}};switch(o.type){case"ExpressionStatement":{const z=o.expression;if(z.type==="AssignmentExpression"&&z.left.type==="Identifier"){const L=h(z.right,l);l.push({type:"ExpressionStatement",expression:{...z,right:L}})}else{const L=h(z,l);(L.type==="UpdateExpression"||L.type==="AssignmentExpression")&&l.push({type:"ExpressionStatement",expression:L})}break}case"VariableDeclaration":for(let z=0;z<o.declarations.length;z++){const L=o.declarations[z],V=h(L.init,l);l.push({...o,declarations:[{...L,init:V}]})}break;case"ReturnStatement":{const z=h(o.argument,l);l.push({...o,argument:z});break}default:return null}if(x)return null;this.linearTempId=w;let F=this.syntheticNodeId||1073741824;const O=z=>{if(!(!z||typeof z!="object")){if(Array.isArray(z)){z.forEach(O);return}typeof z.type=="string"&&z.start===void 0&&(z.start=F,z.end=F+1,F+=2);for(const L in z)L==="loc"||L==="range"||L==="parent"||O(z[L])}};return O(l),this.syntheticNodeId=F,l}astStatementWithHoisting(o,l){switch(o.type){case"ExpressionStatement":case"VariableDeclaration":case"ReturnStatement":{if(!y(o))return this.astGeneric(o,l);const x=this.hoistedIndexReads,w=this.hoistedIndexReads=[],v=[];return this.astGeneric(o,v),this.hoistedIndexReads=x,l.push(...w,...v),l}default:return this.astGeneric(o,l)}}astVariableDeclaration(o,l){const x=o.declarations;if(!x||!x[0]||!x[0].init)throw this.astErrorOutput("Unexpected expression",o);const w=[];let v=null;const C=[];let b=[];for(let T=0;T<x.length;T++){const h=x[T],F=h.init,O=this.getDeclaration(h.id),z=this.getType(h.init);let L=z;L==="LiteralInteger"&&(O.suggestedType==="Integer"?L="Integer":L="Number");const V=g[L];if(!V)throw this.astErrorOutput(`Markup type ${L} not handled`,o);const U=[];if(z==="Integer"&&L==="Integer"){if(O.valueType="Number",T===0||v===null)U.push("float ");else if(L!==v)throw new Error("Unhandled declaration");v=L,U.push(`user_${D.sanitizeName(h.id.name)}=`),U.push("float("),this.astGeneric(F,U),U.push(")")}else O.valueType=L,T===0||v===null?U.push(`${V} `):L!==v&&(C.push(b.join(",")),b=[],U.push(`${V} `)),v=L,U.push(`user_${D.sanitizeName(h.id.name)}=`),z==="Number"&&L==="Integer"?F.left&&F.left.type==="Literal"?this.astGeneric(F,U):(U.push("int("),this.astGeneric(F,U),U.push(")")):z==="LiteralInteger"&&L==="Integer"?this.castLiteralToInteger(F,U):this.astGeneric(F,U);b.push(U.join(""))}return b.length>0&&C.push(b.join(",")),w.push(C.join(";")),l.push(w.join("")),l.push(";"),l}astIfStatement(o,l){return l.push("if ("),this.astGeneric(o.test,l),l.push(")"),o.consequent.type==="BlockStatement"?this.astGeneric(o.consequent,l):(l.push(` {
`),this.astGeneric(o.consequent,l),l.push(`
}
`)),o.alternate&&(l.push("else "),o.alternate.type==="BlockStatement"||o.alternate.type==="IfStatement"?this.astGeneric(o.alternate,l):(l.push(` {
`),this.astGeneric(o.alternate,l),l.push(`
}
`))),l}astSwitchCaseConsequent(o,l){const x=[];for(let w=0;w<o.length&&o[w].type!=="BreakStatement";w++)x.push(o[w]);for(let w=0;w<x.length;w++){const v=C=>{if(!C||typeof C!="object")return!1;if(Array.isArray(C))return C.some(v);if(C.type==="BreakStatement")return!0;if(C.type==="ForStatement"||C.type==="WhileStatement"||C.type==="DoWhileStatement"||C.type==="SwitchStatement")return!1;for(const b in C)if(!(b==="loc"||b==="range"||b==="parent")&&v(C[b]))return!0;return!1};if(v(x[w]))throw this.astErrorOutput("break inside a switch case is only supported as the case terminator",x[w])}for(let w=0;w<x.length;w++)this.astStatementWithHoisting(x[w],l),l.push(`
`);return l}astSwitchStatement(o,l){if(o.type!=="SwitchStatement")throw this.astErrorOutput("Invalid switch statement",o);const{discriminant:x,cases:w}=o,v=this.getType(x),C=`switchDiscriminant${this.astKey(o,"_")}`;switch(v){case"Float":case"Number":l.push(`float ${C} = `),this.astGeneric(x,l),l.push(`;
`);break;case"Integer":l.push(`int ${C} = `),this.astGeneric(x,l),l.push(`;
`);break}if(w.length===1&&!w[0].test)return this.astSwitchCaseConsequent(w[0].consequent,l),l;let b=!1,T=[],h=!1,F=!1;for(let O=0;O<w.length;O++){if(w[O].test){if(O===0||!F?(F=!0,l.push(`if (${C} == `)):b?(l.push(`${C} == `),b=!1):l.push(` else if (${C} == `),v==="Integer")switch(this.getType(w[O].test)){case"Number":case"Float":this.castValueToInteger(w[O].test,l);break;case"LiteralInteger":this.castLiteralToInteger(w[O].test,l);break}else if(v==="Float"||v==="Number")switch(this.getType(w[O].test)){case"LiteralInteger":this.castLiteralToFloat(w[O].test,l);break;case"Integer":this.castValueToFloat(w[O].test,l);break}else throw this.astErrorOutput(`Unhandled switch discriminant type "${v}"`,o);if(!w[O].consequent||w[O].consequent.length===0){b=!0,l.push(" || ");continue}l.push(`) {
`)}else if(w.length>O+1){h=!0,this.astSwitchCaseConsequent(w[O].consequent,T);continue}else l.push(` else {
`);this.astSwitchCaseConsequent(w[O].consequent,l),l.push(`
}`)}return h&&(l.push(" else {"),l.push(T.join("")),l.push("}")),l}astThisExpression(o,l){return l.push("this"),l}astMemberExpression(o,l){const{property:x,name:w,signature:v,origin:C,type:b,xProperty:T,yProperty:h,zProperty:F}=this.getMemberExpressionDetails(o);switch(v){case"value.thread.value":case"this.thread.value":if(w!=="x"&&w!=="y"&&w!=="z")throw this.astErrorOutput("Unexpected expression, expected `this.thread.x`, `this.thread.y`, or `this.thread.z`",o);return l.push(`threadId.${w}`),l;case"this.output.value":if(this.dynamicOutput)switch(w){case"x":this.isState("casting-to-float")?l.push("float(uOutputDim.x)"):l.push("uOutputDim.x");break;case"y":this.isState("casting-to-float")?l.push("float(uOutputDim.y)"):l.push("uOutputDim.y");break;case"z":this.isState("casting-to-float")?l.push("float(uOutputDim.z)"):l.push("uOutputDim.z");break;default:throw this.astErrorOutput("Unexpected expression",o)}else switch(w){case"x":this.isState("casting-to-integer")?l.push(this.output[0]):l.push(this.output[0],".0");break;case"y":this.isState("casting-to-integer")?l.push(this.output[1]):l.push(this.output[1],".0");break;case"z":this.isState("casting-to-integer")?l.push(this.output[2]):l.push(this.output[2],".0");break;default:throw this.astErrorOutput("Unexpected expression",o)}return l;case"value":throw this.astErrorOutput("Unexpected expression",o);case"value[]":case"value[][]":case"value[][][]":case"value[][][][]":case"value.value":if(C==="Math")return l.push(Math[w]),l;const z=D.sanitizeName(w);switch(x){case"r":return l.push(`user_${z}.r`),l;case"g":return l.push(`user_${z}.g`),l;case"b":return l.push(`user_${z}.b`),l;case"a":return l.push(`user_${z}.a`),l}break;case"this.constants.value":if(typeof T>"u")switch(b){case"Array(2)":case"Array(3)":case"Array(4)":return l.push(`constants_${D.sanitizeName(w)}`),l}case"this.constants.value[]":case"this.constants.value[][]":case"this.constants.value[][][]":case"this.constants.value[][][][]":break;case"fn()[]":return this.astCallExpression(o.object,l),l.push("["),l.push(this.memberExpressionPropertyMarkup(x)),l.push("]"),l;case"fn()[][]":{const L=o.object.property,V=o.property,U=p[this.getType(o.object.object)],X=q=>this.getType(q)==="LiteralInteger";return U&&!(X(L)&&X(V))?(l.push(`getMatrix${U}(`),this.astCallExpression(o.object.object,l),l.push(", "),l.push(this.memberExpressionPropertyMarkup(L)),l.push(", "),l.push(this.memberExpressionPropertyMarkup(V)),l.push(")"),l):(this.astCallExpression(o.object.object,l),l.push("["),l.push(this.memberExpressionPropertyMarkup(L)),l.push("]"),l.push("["),l.push(this.memberExpressionPropertyMarkup(V)),l.push("]"),l)}case"[][]":return this.astArrayExpression(o.object,l),l.push("["),l.push(this.memberExpressionPropertyMarkup(x)),l.push("]"),l;default:throw this.astErrorOutput("Unexpected expression",o)}if(o.computed===!1)switch(b){case"Number":case"Integer":case"Float":case"Boolean":return l.push(`${C}_${D.sanitizeName(w)}`),l}const O=`${C}_${D.sanitizeName(w)}`;switch(b){case"Array(2)":case"Array(3)":case"Array(4)":this.astGeneric(o.object,l),l.push("["),l.push(this.memberExpressionPropertyMarkup(T)),l.push("]");break;case"HTMLImageArray":l.push(`getImage3D(${O}, ${O}Size, ${O}Dim, `),this.memberExpressionXYZ(T,h,F,l),l.push(")");break;case"ArrayTexture(1)":l.push(`getFloatFromSampler2D(${O}, ${O}Size, ${O}Dim, `),this.memberExpressionXYZ(T,h,F,l),l.push(")");break;case"Array1D(2)":case"Array2D(2)":case"Array3D(2)":l.push(`getMemoryOptimizedVec2(${O}, ${O}Size, ${O}Dim, `),this.memberExpressionXYZ(T,h,F,l),l.push(")");break;case"ArrayTexture(2)":l.push(`getVec2FromSampler2D(${O}, ${O}Size, ${O}Dim, `),this.memberExpressionXYZ(T,h,F,l),l.push(")");break;case"Array1D(3)":case"Array2D(3)":case"Array3D(3)":l.push(`getMemoryOptimizedVec3(${O}, ${O}Size, ${O}Dim, `),this.memberExpressionXYZ(T,h,F,l),l.push(")");break;case"ArrayTexture(3)":l.push(`getVec3FromSampler2D(${O}, ${O}Size, ${O}Dim, `),this.memberExpressionXYZ(T,h,F,l),l.push(")");break;case"Array1D(4)":case"Array2D(4)":case"Array3D(4)":l.push(`getMemoryOptimizedVec4(${O}, ${O}Size, ${O}Dim, `),this.memberExpressionXYZ(T,h,F,l),l.push(")");break;case"ArrayTexture(4)":case"HTMLCanvas":case"OffscreenCanvas":case"HTMLImage":case"ImageBitmap":case"ImageData":case"HTMLVideo":l.push(`getVec4FromSampler2D(${O}, ${O}Size, ${O}Dim, `),this.memberExpressionXYZ(T,h,F,l),l.push(")");break;case"NumberTexture":case"Array":case"Array2D":case"Array3D":case"Array4D":case"Input":case"Number":case"Float":case"Integer":if(this.precision==="single")l.push(`getMemoryOptimized32(${O}, ${O}Size, ${O}Dim, `),this.memberExpressionXYZ(T,h,F,l),l.push(")");else{const z=C==="user"?this.lookupFunctionArgumentBitRatio(this.name,w):this.constantBitRatios[w];switch(z){case 1:l.push(`get8(${O}, ${O}Size, ${O}Dim, `);break;case 2:l.push(`get16(${O}, ${O}Size, ${O}Dim, `);break;case 4:case 0:l.push(`get32(${O}, ${O}Size, ${O}Dim, `);break;default:throw new Error(`unhandled bit ratio of ${z}`)}this.memberExpressionXYZ(T,h,F,l),l.push(")")}break;case"MemoryOptimizedNumberTexture":l.push(`getMemoryOptimized32(${O}, ${O}Size, ${O}Dim, `),this.memberExpressionXYZ(T,h,F,l),l.push(")");break;case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":l.push(`${O}[${this.memberExpressionPropertyMarkup(h)}]`),h&&l.push(`[${this.memberExpressionPropertyMarkup(T)}]`);break;default:throw new Error(`unhandled member expression "${b}"`)}return l}astCallExpression(o,l){if(!o.callee)throw this.astErrorOutput("Unknown CallExpression",o);let x=null;const w=this.isAstMathFunction(o);if(w||o.callee.object&&o.callee.object.type==="ThisExpression"?x=o.callee.property.name:o.callee.type==="SequenceExpression"&&o.callee.expressions[0].type==="Literal"&&!isNaN(o.callee.expressions[0].raw)?x=o.callee.expressions[1].property.name:x=o.callee.name,!x)throw this.astErrorOutput("Unhandled function, couldn't find name",o);switch(x){case"pow":x="_pow";break;case"round":x="_round";break}if(this.calledFunctions.indexOf(x)<0&&this.calledFunctions.push(x),x==="random"&&this.plugins&&this.plugins.length>0)for(let v=0;v<this.plugins.length;v++){const C=this.plugins[v];if(C.functionMatch==="Math.random()"&&C.functionReplace)return l.push(C.functionReplace),l}if(this.onFunctionCall&&this.onFunctionCall(this.name,x,o.arguments),l.push(x),l.push("("),w)for(let v=0;v<o.arguments.length;++v){const C=o.arguments[v],b=this.getType(C);switch(v>0&&l.push(", "),b){case"Integer":this.castValueToFloat(C,l);break;default:this.astGeneric(C,l);break}}else{const v=this.lookupFunctionArgumentTypes(x)||[];for(let C=0;C<o.arguments.length;++C){const b=o.arguments[C];let T=v[C];C>0&&l.push(", ");const h=this.getType(b);switch(T||(this.triggerImplyArgumentType(x,C,h,this),T=h),h){case"Boolean":this.astGeneric(b,l);continue;case"Number":case"Float":if(T==="Integer"){l.push("int("),this.astGeneric(b,l),l.push(")");continue}else if(T==="Number"||T==="Float"){this.astGeneric(b,l);continue}else if(T==="LiteralInteger"){this.castLiteralToFloat(b,l);continue}break;case"Integer":if(T==="Number"||T==="Float"){l.push("float("),this.astGeneric(b,l),l.push(")");continue}else if(T==="Integer"){this.astGeneric(b,l);continue}break;case"LiteralInteger":if(T==="Integer"){this.castLiteralToInteger(b,l);continue}else if(T==="Number"||T==="Float"){this.castLiteralToFloat(b,l);continue}else if(T==="LiteralInteger"){this.astGeneric(b,l);continue}break;case"Array(2)":case"Array(3)":case"Array(4)":if(T===h){if(b.type==="Identifier")l.push(`user_${D.sanitizeName(b.name)}`);else if(b.type==="ArrayExpression"||b.type==="MemberExpression"||b.type==="CallExpression")this.astGeneric(b,l);else throw this.astErrorOutput(`Unhandled argument type ${b.type}`,o);continue}break;case"HTMLCanvas":case"OffscreenCanvas":case"HTMLImage":case"ImageBitmap":case"ImageData":case"HTMLImageArray":case"HTMLVideo":case"ArrayTexture(1)":case"ArrayTexture(2)":case"ArrayTexture(3)":case"ArrayTexture(4)":case"Array":case"Input":if(T===h){if(b.type!=="Identifier")throw this.astErrorOutput(`Unhandled argument type ${b.type}`,o);this.triggerImplyArgumentBitRatio(this.name,b.name,x,C);const F=D.sanitizeName(b.name);l.push(`user_${F},user_${F}Size,user_${F}Dim`);continue}break}throw this.astErrorOutput(`Unhandled argument combination of ${h} and ${T} for argument named "${b.name}"`,o)}}return l.push(")"),l}astArrayExpression(o,l){const x=this.getType(o),w=o.elements.length;switch(x){case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":l.push(`mat${w}(`);break;default:l.push(`vec${w}(`)}for(let v=0;v<w;++v){v>0&&l.push(", ");const C=o.elements[v];this.astGeneric(C,l)}return l.push(")"),l}memberExpressionXYZ(o,l,x,w){return x?w.push(this.memberExpressionPropertyMarkup(x),", "):w.push("0, "),l?w.push(this.memberExpressionPropertyMarkup(l),", "):w.push("0, "),w.push(this.memberExpressionPropertyMarkup(o)),w}memberExpressionPropertyMarkup(o){if(!o)throw new Error("Property not set");const l=this.getType(o),x=[];switch(l){case"Number":case"Float":this.castValueToInteger(o,x);break;case"LiteralInteger":this.castLiteralToInteger(o,x);break;default:this.astGeneric(o,x)}const w=x.join("");if(this.hoistedIndexReads&&/\b\w+\((user_|constants_)\w+, \1\w+Size/.test(w)){const v=`hoisted_${this.hoistedIndexReads.length}_${D.sanitizeName(this.name)}`,C=w.startsWith("int(");return this.hoistedIndexReads.push(`${C?"int":"float"} ${v}=${w};
`),v}return w}};function d(o){if(!o||typeof o!="object")return!0;if(Array.isArray(o))return o.every(d);if(o.type==="UpdateExpression"||o.type==="AssignmentExpression"||o.type==="SequenceExpression")return!1;for(const l in o)if(!(l==="loc"||l==="range"||l==="parent")&&!d(o[l]))return!1;return!0}function E(o){let l=!1;function x(v){if(!v||typeof v!="object"||l)return!1;if(Array.isArray(v))return v.some(x);if(v.type==="MemberExpression"&&v.computed)return!0;for(const C in v)if(!(C==="loc"||C==="range"||C==="parent")&&x(v[C]))return!0;return!1}function w(v){if(!(!v||typeof v!="object"||l)){if(Array.isArray(v)){v.forEach(w);return}if(v.type==="MemberExpression"&&v.computed&&x(v.property)){l=!0;return}for(const C in v)C==="loc"||C==="range"||C==="parent"||w(v[C])}}return w(o),l}function $(o,l){if(!o||typeof o!="object")return!1;if(Array.isArray(o))return o.some(x=>$(x,l));if(o.type==="CallExpression"&&o.callee.type==="Identifier"&&o.callee.name===l)return!0;for(const x in o)if(!(x==="loc"||x==="range"||x==="parent")&&$(o[x],l))return!0;return!1}function P(o){let l=!1;function x(w){if(!(!w||typeof w!="object"||l)){if(Array.isArray(w)){w.forEach(x);return}if(w.type==="CallExpression"&&w.callee.type==="Identifier"&&w.arguments.some(v=>$(v,w.callee.name))){l=!0;return}for(const v in w)v==="loc"||v==="range"||v==="parent"||x(w[v])}}return x(o),l}function y(o){const l=o.type==="ExpressionStatement"&&o.expression.type==="AssignmentExpression"?o.expression:null;function x(w){if(!w||typeof w!="object")return!0;if(Array.isArray(w))return w.every(x);if(typeof w.type=="string"&&(w.type==="UpdateExpression"||w.type==="SequenceExpression"||w.type==="AssignmentExpression"&&w!==l))return!1;for(const v in w)if(!(v==="loc"||v==="range"||v==="parent")&&!x(w[v]))return!1;return!0}return x(o)}const p={"Matrix(2)":2,"Matrix(3)":3,"Matrix(4)":4},g={Array:"sampler2D","Array(2)":"vec2","Array(3)":"vec3","Array(4)":"vec4","Matrix(2)":"mat2","Matrix(3)":"mat3","Matrix(4)":"mat4",Array2D:"sampler2D",Array3D:"sampler2D",Boolean:"bool",Float:"float",Input:"sampler2D",Integer:"int",Number:"float",LiteralInteger:"float",NumberTexture:"sampler2D",MemoryOptimizedNumberTexture:"sampler2D","ArrayTexture(1)":"sampler2D","ArrayTexture(2)":"sampler2D","ArrayTexture(3)":"sampler2D","ArrayTexture(4)":"sampler2D",HTMLVideo:"sampler2D",HTMLCanvas:"sampler2D",OffscreenCanvas:"sampler2D",HTMLImage:"sampler2D",ImageBitmap:"sampler2D",ImageData:"sampler2D",HTMLImageArray:"sampler2DArray"},k={"===":"==","!==":"!="};G.exports={WebGLFunctionNode:c}}),Dr=s((j,G)=>{const D=`// https://www.shadertoy.com/view/4t2SDh
//note: uniformly distributed, normalized rand, [0,1]
highp float randomSeedShift = 1.0;
highp float slide = 1.0;
uniform highp float randomSeed1;
uniform highp float randomSeed2;

highp float nrand(highp vec2 n) {
  highp float result = fract(sin(dot((n.xy + 1.0) * vec2(randomSeed1 * slide, randomSeed2 * randomSeedShift), vec2(12.9898, 78.233))) * 43758.5453);
  randomSeedShift = result;
  if (randomSeedShift > 0.5) {
    slide += 0.00009; 
  } else {
    slide += 0.0009;
  }
  return result;
}`,I="math-random-uniformly-distributed",_="Math.random()",c="nrand(vTexCoord)",d="Number";function E(P){let y=P>>>0;return function(){y=y+1831565813>>>0;let p=y;return p=Math.imul(p^p>>>15,p|1),p^=p+Math.imul(p^p>>>7,p|61),((p^p>>>14)>>>0)/4294967296}}const $=P=>{if(P.randomSeed===null||P.randomSeed===void 0){P.setUniform1f("randomSeed1",Math.random()),P.setUniform1f("randomSeed2",Math.random());return}(!P._mathRandomGenerator||P._mathRandomGeneratorSeed!==P.randomSeed)&&(P._mathRandomGenerator=E(P.randomSeed),P._mathRandomGeneratorSeed=P.randomSeed),P.setUniform1f("randomSeed1",P._mathRandomGenerator()),P.setUniform1f("randomSeed2",P._mathRandomGenerator())};G.exports={name:I,onBeforeRun:$,functionMatch:_,functionReplace:c,functionReturnType:d,source:D}}),oo=s((j,G)=>{G.exports={fragmentShader:`__HEADER__;
__FLOAT_TACTIC_DECLARATION__;
__INT_TACTIC_DECLARATION__;
__SAMPLER_2D_TACTIC_DECLARATION__;

const int LOOP_MAX = __LOOP_MAX__;

__PLUGINS__;
__CONSTANTS__;

varying vec2 vTexCoord;

float acosh(float x) {
  return log(x + sqrt(x * x - 1.0));
}

float sinh(float x) {
  return (pow(${Math.E}, x) - pow(${Math.E}, -x)) / 2.0;
}

float asinh(float x) {
  return log(x + sqrt(x * x + 1.0));
}

float atan2(float v1, float v2) {
  if (v2 == 0.0) {
    if (v1 == 0.0) return 0.0;
    if (v1 > 0.0) return 1.5707963267948966;
    if (v1 < 0.0) return -1.5707963267948966;
  }
  return atan(v1, v2);
}

float atanh(float x) {
  x = (x + 1.0) / (x - 1.0);
  if (x < 0.0) {
    return 0.5 * log(-x);
  }
  return 0.5 * log(x);
}

float cbrt(float x) {
  if (x >= 0.0) {
    return pow(x, 1.0 / 3.0);
  } else {
    return -pow(x, 1.0 / 3.0);
  }
}

float cosh(float x) {
  return (pow(${Math.E}, x) + pow(${Math.E}, -x)) / 2.0; 
}

float expm1(float x) {
  return pow(${Math.E}, x) - 1.0; 
}

float fround(highp float x) {
  return x;
}

float imul(float v1, float v2) {
  return float(int(v1) * int(v2));
}

float log10(float x) {
  return log2(x) * (1.0 / log2(10.0));
}

float log1p(float x) {
  return log(1.0 + x);
}

float _pow(float v1, float v2) {
  if (v2 == 0.0) return 1.0;
  return pow(v1, v2);
}

float tanh(float x) {
  float e = exp(2.0 * x);
  return (e - 1.0) / (e + 1.0);
}

float trunc(float x) {
  if (x >= 0.0) {
    return floor(x); 
  } else {
    return ceil(x);
  }
}

vec4 _round(vec4 x) {
  return floor(x + 0.5);
}

float _round(float x) {
  return floor(x + 0.5);
}

const int BIT_COUNT = 32;
int modi(int x, int y) {
  return x - y * (x / y);
}

int bitwiseOr(int a, int b) {
  int result = 0;
  int n = 1;
  
  for (int i = 0; i < BIT_COUNT; i++) {
    if ((modi(a, 2) == 1) || (modi(b, 2) == 1)) {
      result += n;
    }
    a = a / 2;
    b = b / 2;
    n = n * 2;
    if(!(a > 0 || b > 0)) {
      break;
    }
  }
  return result;
}
int bitwiseXOR(int a, int b) {
  int result = 0;
  int n = 1;
  
  for (int i = 0; i < BIT_COUNT; i++) {
    if ((modi(a, 2) == 1) != (modi(b, 2) == 1)) {
      result += n;
    }
    a = a / 2;
    b = b / 2;
    n = n * 2;
    if(!(a > 0 || b > 0)) {
      break;
    }
  }
  return result;
}
int bitwiseAnd(int a, int b) {
  int result = 0;
  int n = 1;
  for (int i = 0; i < BIT_COUNT; i++) {
    if ((modi(a, 2) == 1) && (modi(b, 2) == 1)) {
      result += n;
    }
    a = a / 2;
    b = b / 2;
    n = n * 2;
    if(!(a > 0 && b > 0)) {
      break;
    }
  }
  return result;
}
int bitwiseNot(int a) {
  // ~a is identically -a - 1 in two's complement, for every value including
  // negatives. The previous bit-by-bit loop only worked for a >= 0, where it
  // leaned on 32-bit overflow wrapping to reach the negative answer; given a
  // negative input it computed ~abs(a), so ~(-1) gave -2 and ~~x never
  // returned x.
  return -a - 1;
}
int bitwiseZeroFillLeftShift(int n, int shift) {
  int maxBytes = BIT_COUNT;
  for (int i = 0; i < BIT_COUNT; i++) {
    if (maxBytes >= n) {
      break;
    }
    maxBytes *= 2;
  }
  for (int i = 0; i < BIT_COUNT; i++) {
    if (i >= shift) {
      break;
    }
    n *= 2;
  }

  int result = 0;
  int byteVal = 1;
  for (int i = 0; i < BIT_COUNT; i++) {
    if (i >= maxBytes) break;
    if (modi(n, 2) > 0) { result += byteVal; }
    n = int(n / 2);
    byteVal *= 2;
  }
  return result;
}

// _pow2 is defined further down, alongside encode32/decode32
float _pow2(float e);
int bitwiseSignedRightShift(int num, int shifts) {
  // pow(2.0, n) is approximate on many GPUs, and landing 1 ulp high makes the
  // division fall just under a whole number, which floor() then rounds away:
  // 2 >> 1 came out 0, 8 >> 1 came out 3. Only exact left operands were
  // affected, odd ones having enough slack to survive. _pow2 is exact.
  return int(floor(float(num) / _pow2(float(shifts))));
}

int bitwiseZeroFillRightShift(int n, int shift) {
  int maxBytes = BIT_COUNT;
  for (int i = 0; i < BIT_COUNT; i++) {
    if (maxBytes >= n) {
      break;
    }
    maxBytes *= 2;
  }
  for (int i = 0; i < BIT_COUNT; i++) {
    if (i >= shift) {
      break;
    }
    n /= 2;
  }
  int result = 0;
  int byteVal = 1;
  for (int i = 0; i < BIT_COUNT; i++) {
    if (i >= maxBytes) break;
    if (modi(n, 2) > 0) { result += byteVal; }
    n = int(n / 2);
    byteVal *= 2;
  }
  return result;
}

vec2 integerMod(vec2 x, float y) {
  vec2 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}

vec3 integerMod(vec3 x, float y) {
  vec3 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}

vec4 integerMod(vec4 x, vec4 y) {
  vec4 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}

float integerMod(float x, float y) {
  float res = floor(mod(x, y));
  return res * (res > floor(y) - 1.0 ? 0.0 : 1.0);
}

int integerMod(int x, int y) {
  return x - (y * int(x / y));
}

// GLSL ES 1.00 accepts only a constant or a loop symbol inside an index
// expression, so m[y][x] does not compile when y and x come from kernel
// arguments -- the error is "Index expression can only contain const or loop
// symbols". Loop counters are legal indices, so walk the matrix with them
// instead. These are 2x2 to 4x4, so it costs at most sixteen comparisons.
float getMatrix2(mat2 m, int y, int x) {
  float result = 0.0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == y && j == x) result = m[i][j];
    }
  }
  return result;
}

float getMatrix3(mat3 m, int y, int x) {
  float result = 0.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == y && j == x) result = m[i][j];
    }
  }
  return result;
}

float getMatrix4(mat4 m, int y, int x) {
  float result = 0.0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == y && j == x) result = m[i][j];
    }
  }
  return result;
}

__DIVIDE_WITH_INTEGER_CHECK__;

// Here be dragons!
// DO NOT OPTIMIZE THIS CODE
// YOU WILL BREAK SOMETHING ON SOMEBODY'S MACHINE
// LEAVE IT AS IT IS, LEST YOU WASTE YOUR OWN TIME
// Exact powers of two built from exact constant multiplies: exp2/log2/pow
// are approximate on some GPUs (notably Apple silicon), and 1-2 ulp there
// corrupts the packed bytes (#659)
float _pow2(float e) {
  float r = 1.0;
  float a = abs(e);
  bool n = e < 0.0;
  if (a >= 64.0) { r *= n ? 5.421010862427522e-20 : 18446744073709551616.0; a -= 64.0; }
  if (a >= 64.0) { r *= n ? 5.421010862427522e-20 : 18446744073709551616.0; a -= 64.0; }
  if (a >= 32.0) { r *= n ? 2.3283064365386963e-10 : 4294967296.0; a -= 32.0; }
  if (a >= 16.0) { r *= n ? 0.0000152587890625 : 65536.0; a -= 16.0; }
  if (a >= 8.0) { r *= n ? 0.00390625 : 256.0; a -= 8.0; }
  if (a >= 4.0) { r *= n ? 0.0625 : 16.0; a -= 4.0; }
  if (a >= 2.0) { r *= n ? 0.25 : 4.0; a -= 2.0; }
  if (a >= 1.0) { r *= n ? 0.5 : 2.0; }
  return r;
}
const vec2 MAGIC_VEC = vec2(1.0, -256.0);
const vec4 SCALE_FACTOR = vec4(1.0, 256.0, 65536.0, 0.0);
const vec4 SCALE_FACTOR_INV = vec4(1.0, 0.00390625, 0.0000152587890625, 0.0); // 1, 1/256, 1/65536
float decode32(vec4 texel) {
  __DECODE32_ENDIANNESS__;
  texel *= 255.0;
  vec2 gte128;
  gte128.x = texel.b >= 128.0 ? 1.0 : 0.0;
  gte128.y = texel.a >= 128.0 ? 1.0 : 0.0;
  float exponent = 2.0 * texel.a - 127.0 + dot(gte128, MAGIC_VEC);
  float res = _pow2(_round(exponent));
  texel.b = texel.b - 128.0 * gte128.x;
  res = dot(texel, SCALE_FACTOR) * _pow2(_round(exponent-23.0)) + res;
  res *= gte128.y * -2.0 + 1.0;
  return res;
}

float decode16(vec4 texel, int index) {
  int channel = integerMod(index, 2);
  if (channel == 0) return texel.r * 255.0 + texel.g * 65280.0;
  if (channel == 1) return texel.b * 255.0 + texel.a * 65280.0;
  return 0.0;
}

float decode8(vec4 texel, int index) {
  int channel = integerMod(index, 4);
  if (channel == 0) return texel.r * 255.0;
  if (channel == 1) return texel.g * 255.0;
  if (channel == 2) return texel.b * 255.0;
  if (channel == 3) return texel.a * 255.0;
  return 0.0;
}

vec4 legacyEncode32(float f) {
  float F = abs(f);
  float sign = f < 0.0 ? 1.0 : 0.0;
  float exponent = floor(log2(F));
  float mantissa = (exp2(-exponent) * F);
  // exponent += floor(log2(mantissa));
  vec4 texel = vec4(F * exp2(23.0-exponent)) * SCALE_FACTOR_INV;
  texel.rg = integerMod(texel.rg, 256.0);
  texel.b = integerMod(texel.b, 128.0);
  texel.a = exponent*0.5 + 63.5;
  texel.ba += vec2(integerMod(exponent+127.0, 2.0), sign) * 128.0;
  texel = floor(texel);
  texel *= 0.003921569; // 1/255
  __ENCODE32_ENDIANNESS__;
  return texel;
}

// https://github.com/gpujs/gpu.js/wiki/Encoder-details
vec4 encode32(float value) {
  if (value == 0.0) return vec4(0, 0, 0, 0);

  float exponent;
  float mantissa;
  vec4  result;
  float sgn;

  sgn = step(0.0, -value);
  value = abs(value);

  exponent = floor(log2(value));
  float p2 = _pow2(exponent);
  // approximate log2 can land one off; correct by direct comparison
  if (p2 > value) { exponent -= 1.0; p2 *= 0.5; }
  else if (p2 * 2.0 <= value) { exponent += 1.0; p2 *= 2.0; }

  mantissa = value / p2 - 1.0;
  exponent = exponent+127.0;
  result   = vec4(0,0,0,0);

  result.a = floor(exponent/2.0);
  exponent = exponent - result.a*2.0;
  result.a = result.a + 128.0*sgn;

  result.b = floor(mantissa * 128.0);
  mantissa = mantissa - result.b / 128.0;
  result.b = result.b + exponent*128.0;

  result.g = floor(mantissa*32768.0);
  mantissa = mantissa - result.g/32768.0;

  result.r = floor(mantissa*8388608.0);
  return result/255.0;
}
// Dragons end here

int index;
ivec3 threadId;

ivec3 indexTo3D(int idx, ivec3 texDim) {
  int z = int(idx / (texDim.x * texDim.y));
  idx -= z * int(texDim.x * texDim.y);
  int y = int(idx / texDim.x);
  int x = int(integerMod(idx, texDim.x));
  return ivec3(x, y, z);
}

float get32(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture2D(tex, st / vec2(texSize));
  return decode32(texel);
}

float get16(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int w = texSize.x * 2;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture2D(tex, st / vec2(texSize.x * 2, texSize.y));
  return decode16(texel, index);
}

float get8(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int w = texSize.x * 4;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture2D(tex, st / vec2(texSize.x * 4, texSize.y));
  return decode8(texel, index);
}

float getMemoryOptimized32(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int channel = integerMod(index, 4);
  index = index / 4;
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture2D(tex, st / vec2(texSize));
  if (channel == 0) return texel.r;
  if (channel == 1) return texel.g;
  if (channel == 2) return texel.b;
  if (channel == 3) return texel.a;
  return 0.0;
}

vec4 getImage2D(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  return texture2D(tex, st / vec2(texSize));
}

float getFloatFromSampler2D(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  vec4 result = getImage2D(tex, texSize, texDim, z, y, x);
  return result[0];
}

vec2 getVec2FromSampler2D(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  vec4 result = getImage2D(tex, texSize, texDim, z, y, x);
  return vec2(result[0], result[1]);
}

vec2 getMemoryOptimizedVec2(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + (texDim.x * (y + (texDim.y * z)));
  int channel = integerMod(index, 2);
  index = index / 2;
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture2D(tex, st / vec2(texSize));
  if (channel == 0) return vec2(texel.r, texel.g);
  if (channel == 1) return vec2(texel.b, texel.a);
  return vec2(0.0, 0.0);
}

vec3 getVec3FromSampler2D(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  vec4 result = getImage2D(tex, texSize, texDim, z, y, x);
  return vec3(result[0], result[1], result[2]);
}

vec3 getMemoryOptimizedVec3(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int fieldIndex = 3 * (x + texDim.x * (y + texDim.y * z));
  int vectorIndex = fieldIndex / 4;
  int vectorOffset = fieldIndex - vectorIndex * 4;
  int readY = vectorIndex / texSize.x;
  int readX = vectorIndex - readY * texSize.x;
  vec4 tex1 = texture2D(tex, (vec2(readX, readY) + 0.5) / vec2(texSize));
  
  if (vectorOffset == 0) {
    return tex1.xyz;
  } else if (vectorOffset == 1) {
    return tex1.yzw;
  } else {
    readX++;
    if (readX >= texSize.x) {
      readX = 0;
      readY++;
    }
    vec4 tex2 = texture2D(tex, vec2(readX, readY) / vec2(texSize));
    if (vectorOffset == 2) {
      return vec3(tex1.z, tex1.w, tex2.x);
    } else {
      return vec3(tex1.w, tex2.x, tex2.y);
    }
  }
}

vec4 getVec4FromSampler2D(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  return getImage2D(tex, texSize, texDim, z, y, x);
}

vec4 getMemoryOptimizedVec4(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int channel = integerMod(index, 2);
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture2D(tex, st / vec2(texSize));
  return vec4(texel.r, texel.g, texel.b, texel.a);
}

vec4 actualColor;
void color(float r, float g, float b, float a) {
  actualColor = vec4(r,g,b,a);
}

void color(float r, float g, float b) {
  color(r,g,b,1.0);
}

void color(sampler2D image) {
  actualColor = texture2D(image, vTexCoord);
}

float modulo(float number, float divisor) {
  if (number < 0.0) {
    number = abs(number);
    if (divisor < 0.0) {
      divisor = abs(divisor);
    }
    return -mod(number, divisor);
  }
  if (divisor < 0.0) {
    divisor = abs(divisor);
  }
  return mod(number, divisor);
}

__INJECTED_NATIVE__;
__MAIN_CONSTANTS__;
__MAIN_ARGUMENTS__;
__KERNEL__;

void main(void) {
  index = int(vTexCoord.s * float(uTexSize.x)) + int(vTexCoord.t * float(uTexSize.y)) * uTexSize.x;
  __MAIN_RESULT__;
}`}}),lo=s((j,G)=>{G.exports={vertexShader:`__FLOAT_TACTIC_DECLARATION__;
__INT_TACTIC_DECLARATION__;
__SAMPLER_2D_TACTIC_DECLARATION__;

attribute vec2 aPos;
attribute vec2 aTexCoord;

varying vec2 vTexCoord;
uniform vec2 ratio;

void main(void) {
  gl_Position = vec4((aPos + vec2(1)) * ratio + vec2(-1), 0, 1);
  vTexCoord = aTexCoord;
}`}}),uo=s((j,G)=>{function D(E,$={}){const{contextName:P="gl",throwGetError:y,useTrackablePrimitives:p,recording:g=[],variables:k={},onReadPixels:o,onUnrecognizedArgumentLookup:l}=$,x=new Proxy(E,{get:T}),w=[],v={};let C="",b;return x;function T(se,Z){switch(Z){case"addComment":return U;case"checkThrowError":return X;case"getReadPixelsVariableName":return b;case"insertVariable":return O;case"reset":return F;case"setIndent":return L;case"toString":return h;case"getContextVariableName":return ee}return typeof E[Z]=="function"?function(){switch(Z){case"getError":return y?g.push(`${C}if (${P}.getError() !== ${P}.NONE) throw new Error('error');`):g.push(`${C}${P}.getError();`),E.getError();case"getExtension":{const ce=`${P}Variables${w.length}`;g.push(`${C}const ${ce} = ${P}.getExtension('${arguments[0]}');`);const Ve=E.getExtension(arguments[0]);if(Ve&&typeof Ve=="object"){const ue=I(Ve,{getEntity:z,useTrackablePrimitives:p,recording:g,contextName:ce,contextVariables:w,variables:k,indent:C,onUnrecognizedArgumentLookup:l});return w.push(ue),ue}else w.push(null);return Ve}case"readPixels":const he=w.indexOf(arguments[6]);let we;if(he===-1){const ce=W(arguments[6]);ce?(we=ce,g.push(`${C}${ce}`)):(we=`${P}Variable${w.length}`,w.push(arguments[6]),g.push(`${C}const ${we} = new ${arguments[6].constructor.name}(${arguments[6].length});`))}else we=`${P}Variable${he}`;b=we;const re=[arguments[0],arguments[1],arguments[2],arguments[3],z(arguments[4]),z(arguments[5]),we];return g.push(`${C}${P}.readPixels(${re.join(", ")});`),o&&o(we,re),E.readPixels.apply(E,arguments);case"drawBuffers":return g.push(`${C}${P}.drawBuffers([${_(arguments[0],{contextName:P,contextVariables:w,getEntity:z,addVariable:V,variables:k,onUnrecognizedArgumentLookup:l})}]);`),E.drawBuffers(arguments[0])}let ie=E[Z].apply(E,arguments);switch(typeof ie){case"undefined":g.push(`${C}${q(Z,arguments)};`);return;case"number":case"boolean":if(p&&w.indexOf(d(ie))===-1){g.push(`${C}const ${P}Variable${w.length} = ${q(Z,arguments)};`),w.push(ie=d(ie));break}default:ie===null?g.push(`${q(Z,arguments)};`):g.push(`${C}const ${P}Variable${w.length} = ${q(Z,arguments)};`),w.push(ie)}return ie}:(v[E[Z]]=Z,E[Z])}function h(){return g.join(`
`)}function F(){for(;g.length>0;)g.pop()}function O(se,Z){k[se]=Z}function z(se){const Z=v[se];return Z?P+"."+Z:se}function L(se){C=" ".repeat(se)}function V(se,Z){const ie=`${P}Variable${w.length}`;return g.push(`${C}const ${ie} = ${Z};`),w.push(se),ie}function U(se){g.push(`${C}// ${se}`)}function X(){g.push(`${C}(() => {
${C}const error = ${P}.getError();
${C}if (error !== ${P}.NONE) {
${C}  const names = Object.getOwnPropertyNames(gl);
${C}  for (let i = 0; i < names.length; i++) {
${C}    const name = names[i];
${C}    if (${P}[name] === error) {
${C}      throw new Error('${P} threw ' + name);
${C}    }
${C}  }
${C}}
${C}})();`)}function q(se,Z){return`${P}.${se}(${_(Z,{contextName:P,contextVariables:w,getEntity:z,addVariable:V,variables:k,onUnrecognizedArgumentLookup:l})})`}function W(se){if(k){for(const Z in k)if(k[Z]===se)return Z}return null}function ee(se){const Z=w.indexOf(se);return Z!==-1?`${P}Variable${Z}`:null}}function I(E,$){const P=new Proxy(E,{get:C}),y={},{contextName:p,contextVariables:g,getEntity:k,useTrackablePrimitives:o,recording:l,variables:x,indent:w,onUnrecognizedArgumentLookup:v}=$;return P;function C(F,O){return typeof F[O]=="function"?function(){switch(O){case"drawBuffersWEBGL":return l.push(`${w}${p}.drawBuffersWEBGL([${_(arguments[0],{contextName:p,contextVariables:g,getEntity:b,addVariable:h,variables:x,onUnrecognizedArgumentLookup:v})}]);`),E.drawBuffersWEBGL(arguments[0])}let z=E[O].apply(E,arguments);switch(typeof z){case"undefined":l.push(`${w}${T(O,arguments)};`);return;case"number":case"boolean":o&&g.indexOf(d(z))===-1?(l.push(`${w}const ${p}Variable${g.length} = ${T(O,arguments)};`),g.push(z=d(z))):(l.push(`${w}const ${p}Variable${g.length} = ${T(O,arguments)};`),g.push(z));break;default:z===null?l.push(`${T(O,arguments)};`):l.push(`${w}const ${p}Variable${g.length} = ${T(O,arguments)};`),g.push(z)}return z}:(y[E[O]]=O,E[O])}function b(F){return y.hasOwnProperty(F)?`${p}.${y[F]}`:k(F)}function T(F,O){return`${p}.${F}(${_(O,{contextName:p,contextVariables:g,getEntity:b,addVariable:h,variables:x,onUnrecognizedArgumentLookup:v})})`}function h(F,O){const z=`${p}Variable${g.length}`;return g.push(F),l.push(`${w}const ${z} = ${O};`),z}}function _(E,$){const{variables:P,onUnrecognizedArgumentLookup:y}=$;return Array.from(E).map(g=>{const k=p(g);return k||c(g,$)}).join(", ");function p(g){if(P){for(const k in P)if(P.hasOwnProperty(k)&&P[k]===g)return k}return y?y(g):null}}function c(E,$){const{contextName:P,contextVariables:y,getEntity:p,addVariable:g,onUnrecognizedArgumentLookup:k}=$;if(typeof E>"u")return"undefined";if(E===null)return"null";const o=y.indexOf(E);if(o>-1)return`${P}Variable${o}`;switch(E.constructor.name){case"String":const l=/\n/.test(E),x=/'/.test(E),w=/"/.test(E);return l?"`"+E+"`":x&&!w?'"'+E+'"':"'"+E+"'";case"Number":return p(E);case"Boolean":return p(E);case"Array":return g(E,`new ${E.constructor.name}([${Array.from(E).join(",")}])`);case"Float32Array":case"Uint8Array":case"Uint16Array":case"Int32Array":return g(E,`new ${E.constructor.name}(${JSON.stringify(Array.from(E))})`);default:if(k){const v=k(E);if(v)return v}throw new Error(`unrecognized argument type ${E.constructor.name}`)}}function d(E){return new E.constructor(E)}typeof G<"u"&&(G.exports={glWiretap:D,glExtensionWiretap:I}),typeof window<"u"&&(D.glExtensionWiretap=I,window.glWiretap=D)}),Pr=s((j,G)=>{const{glWiretap:D}=uo(),{utils:I}=m();function _(y){let p=y.toString().replace(/^function /,"");const g=p.indexOf("=>");if(g!==-1&&!/[{]|\bfunction\b/.test(p.slice(0,g))){const k=p.slice(0,g).trim(),o=p.slice(g+2).trim();p=o.startsWith("{")?`${k} ${o}`:`${k} { return ${o}; }`}return p.replace(/utils[.]/g,"/*utils.*/")}function c(y,p,g,k,o){g.built||g.build.apply(g,p),p=p?Array.from(p).map(ue=>{switch(typeof ue){case"boolean":return new Boolean(ue);case"number":return new Number(ue);default:return ue}}):null;const l=[],x=D(g.context,{useTrackablePrimitives:!0,onReadPixels:ue=>{if(re.subKernels){if(!w)l.push(`    const result = { result: ${d(ue,re)} };`),w=!0;else{const Se=re.subKernels[v++].property;l.push(`    result${isNaN(Se)?"."+Se:`[${Se}]`} = ${d(ue,re)};`)}v===re.subKernels.length&&l.push("    return result;");return}ue?l.push(`    return ${d(ue,re)};`):l.push("    return null;")},onUnrecognizedArgumentLookup:ue=>{const Se=P(ue,re.kernelArguments,[],x);if(Se)return Se;const Re=P(ue,re.kernelConstants,z?Object.keys(z).map(Me=>z[Me]):[],x);return Re||null}});let w=!1,v=0;const{source:C,canvas:b,output:T,pipeline:h,graphical:F,loopMaxIterations:O,constants:z,optimizeFloatMemory:L,precision:V,fixIntegerDivisionAccuracy:U,functions:X,nativeFunctions:q,subKernels:W,immutable:ee,argumentTypes:se,constantTypes:Z,kernelArguments:ie,kernelConstants:he,tactic:we}=g,re=new y(C,{canvas:b,context:x,checkContext:!1,output:T,pipeline:h,graphical:F,loopMaxIterations:O,constants:z,optimizeFloatMemory:L,precision:V,fixIntegerDivisionAccuracy:U,functions:X,nativeFunctions:q,subKernels:W,immutable:ee,argumentTypes:se,constantTypes:Z,tactic:we});let ce=[];if(x.setIndent(2),re.build.apply(re,p),ce.push(x.toString()),x.reset(),re.kernelArguments.forEach((ue,Se)=>{switch(ue.type){case"Integer":case"Boolean":case"Number":case"Float":case"Array":case"Array(2)":case"Array(3)":case"Array(4)":case"HTMLCanvas":case"HTMLImage":case"HTMLVideo":x.insertVariable(`uploadValue_${ue.name}`,ue.uploadValue);break;case"HTMLImageArray":for(let Re=0;Re<p[Se].length;Re++){const Me=p[Se];x.insertVariable(`uploadValue_${ue.name}[${Re}]`,Me[Re])}break;case"Input":x.insertVariable(`uploadValue_${ue.name}`,ue.uploadValue);break;case"MemoryOptimizedNumberTexture":case"NumberTexture":case"Array1D(2)":case"Array1D(3)":case"Array1D(4)":case"Array2D(2)":case"Array2D(3)":case"Array2D(4)":case"Array3D(2)":case"Array3D(3)":case"Array3D(4)":case"ArrayTexture(1)":case"ArrayTexture(2)":case"ArrayTexture(3)":case"ArrayTexture(4)":x.insertVariable(`uploadValue_${ue.name}`,p[Se].texture);break;default:throw new Error(`unhandled kernelArgumentType insertion for glWiretap of type ${ue.type}`)}}),ce.push("/** start of injected functions **/"),ce.push(`function ${_(I.flattenTo)}`),ce.push(`function ${_(I.flatten2dArrayTo)}`),ce.push(`function ${_(I.flatten3dArrayTo)}`),ce.push(`function ${_(I.flatten4dArrayTo)}`),ce.push(`function ${_(I.isArray)}`),re.renderOutput!==re.renderTexture&&re.formatValues&&ce.push(`  const renderOutput = function ${_(re.formatValues)};`),ce.push(`let readFramebuffer = null;
function getReadFramebuffer() {
  if (!readFramebuffer) readFramebuffer = gl.createFramebuffer();
  return readFramebuffer;
}`),ce.push("/** end of injected functions **/"),ce.push(`  const innerKernel = function (${re.kernelArguments.map(ue=>ue.varName).join(", ")}) {`),x.setIndent(4),re.run.apply(re,p),re.renderKernels?re.renderKernels():re.renderOutput&&re.renderOutput(),ce.push("    /** start setup uploads for kernel values **/"),re.kernelArguments.forEach(ue=>{ce.push("    "+ue.getStringValueHandler().split(`
`).join(`
    `))}),ce.push("    /** end setup uploads for kernel values **/"),ce.push(x.toString()),re.renderOutput===re.renderTexture)if(x.reset(),re.renderKernels){const ue=re.renderKernels(),Se=x.getContextVariableName(re.texture.texture);ce.push(`    return {
      result: {
        texture: ${Se},
        type: '${ue.result.type}',
        toArray: ${$(ue.result,Se)}
      },`);const{subKernels:Re,mappedTextures:Me}=re;for(let ae=0;ae<Re.length;ae++){const ye=Me[ae],Ce=Re[ae],Ne=ue[Ce.property],ft=x.getContextVariableName(ye.texture);ce.push(`
      ${Ce.property}: {
        texture: ${ft},
        type: '${Ne.type}',
        toArray: ${$(Ne,ft)}
      },`)}ce.push("    };")}else{const ue=re.renderOutput(),Se=x.getContextVariableName(re.texture.texture);ce.push(`    return {
        texture: ${Se},
        type: '${ue.type}',
        toArray: ${$(ue,Se)}
      };`)}ce.push(`    ${o?`
`+o+"    ":""}`),ce.push(l.join(`
`)),ce.push("  };"),re.graphical&&(ce.push(E(re)),ce.push("  innerKernel.getPixels = getPixels;")),ce.push("  return innerKernel;");let Ve=[];return he.forEach(ue=>{Ve.push(`${ue.getStringValueHandler()}`)}),`function kernel(settings) {
  const { context, constants } = settings;
  ${Ve.join("")}
  ${k||""}
${ce.join(`
`)}
}`}function d(y,p){const g=p.precision==="single"?y:`new Float32Array(${y}.buffer)`;return p.output[2]?`renderOutput(${g}, ${p.output[0]}, ${p.output[1]}, ${p.output[2]})`:p.output[1]?`renderOutput(${g}, ${p.output[0]}, ${p.output[1]})`:`renderOutput(${g}, ${p.output[0]})`}function E(y){const p=y.getPixels.toString(),g=!/^function/.test(p);return I.flattenFunctionToString(`${g?"function ":""}${p}`,{findDependency:(k,o)=>k==="utils"?`const ${o} = ${I[o].toString()};`:null,thisLookup:k=>{if(k==="context")return null;if(y.hasOwnProperty(k))return JSON.stringify(y[k]);throw new Error(`unhandled thisLookup ${k}`)}})}function $(y,p){const g=y.toArray.toString(),k=!/^function/.test(g);return`() => {
  function framebuffer() { return getReadFramebuffer(); };
  ${I.flattenFunctionToString(`${k?"function ":""}${g}`,{findDependency:(o,l)=>{if(o==="utils")return`const ${l} = ${I[l].toString()};`;if(o==="this")return l==="framebuffer"?"":`${k?"function ":""}${y[l].toString()}`;throw new Error("unhandled fromObject")},thisLookup:(o,l)=>{if(o==="texture")return p;if(o==="context")return l?null:"gl";if(y.hasOwnProperty(o))return JSON.stringify(y[o]);throw new Error(`unhandled thisLookup ${o}`)}})}
  return toArray();
  }`}function P(y,p,g,k,o){if(y===null||p===null)return null;switch(typeof y){case"boolean":case"number":return null}if(typeof HTMLImageElement<"u"&&y instanceof HTMLImageElement)for(let l=0;l<p.length;l++){const x=p[l];if(x.type!=="HTMLImageArray"&&x||x.uploadValue!==y)continue;const w=g[l].indexOf(y);if(w===-1)continue;const v=`uploadValue_${x.name}[${w}]`;return k.insertVariable(v,y),v}for(let l=0;l<p.length;l++){const x=p[l];if(y!==x.uploadValue)continue;const w=`uploadValue_${x.name}`;return k.insertVariable(w,x),w}return null}G.exports={glKernelString:c}}),co=s((j,G)=>{var D=class{constructor(I,_){const{name:c,kernel:d,context:E,checkContext:$,onRequestContextHandle:P,onUpdateValueMismatch:y,origin:p,strictIntegers:g,type:k,tactic:o}=_;if(!c)throw new Error("name not set");if(!k)throw new Error("type not set");if(!p)throw new Error("origin not set");if(p!=="user"&&p!=="constants")throw new Error(`origin must be "user" or "constants" value is "${p}"`);if(!P)throw new Error("onRequestContextHandle is not set");this.name=c,this.origin=p,this.tactic=o,this.varName=p==="constants"?`constants.${c}`:c,this.kernel=d,this.strictIntegers=g,this.type=I.type||k,this.size=I.size||null,this.index=null,this.context=E,this.checkContext=$??!0,this.contextHandle=null,this.onRequestContextHandle=P,this.onUpdateValueMismatch=y,this.forceUploadEachRun=null}get id(){return`${this.origin}_${name}`}getSource(){throw new Error(`"getSource" not defined on ${this.constructor.name}`)}updateValue(I){throw new Error(`"updateValue" not defined on ${this.constructor.name}`)}};G.exports={KernelValue:D}}),Pt=s((j,G)=>{const{utils:D}=m(),{KernelValue:I}=co();var _=class extends I{constructor(c,d){super(c,d),this.dimensionsId=null,this.sizeId=null,this.initialValueConstructor=c.constructor,this.onRequestTexture=d.onRequestTexture,this.onRequestIndex=d.onRequestIndex,this.uploadValue=null,this.textureSize=null,this.bitRatio=null,this.prevArg=null}get id(){return`${this.origin}_${D.sanitizeName(this.name)}`}setup(){}getTransferArrayType(c){if(Array.isArray(c[0]))return this.getTransferArrayType(c[0]);switch(c.constructor){case Array:case Int32Array:case Int16Array:case Int8Array:return Float32Array;case Uint8ClampedArray:case Uint8Array:case Uint16Array:case Uint32Array:case Float32Array:case Float64Array:return c.constructor}return console.warn("Unfamiliar constructor type.  Will go ahead and use, but likley this may result in a transfer of zeros"),c.constructor}getStringValueHandler(){throw new Error(`"getStringValueHandler" not implemented on ${this.constructor.name}`)}getVariablePrecisionString(){return this.kernel.getVariablePrecisionString(this.textureSize||void 0,this.tactic||void 0)}destroy(){}};G.exports={WebGLKernelValue:_}}),zr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValue:I}=Pt();var _=class extends I{constructor(c,d){super(c,d),this.uploadValue=c}getSource(c){return this.origin==="constants"?`const bool ${this.id} = ${c};
`:`uniform bool ${this.id};
`}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName};
`}updateValue(c){this.origin!=="constants"&&this.kernel.setUniform1i(this.id,this.uploadValue=c)}};G.exports={WebGLKernelValueBoolean:_}}),Or=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValue:I}=Pt();var _=class extends I{constructor(c,d){super(c,d),this.uploadValue=c}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName};
`}getSource(c){return this.origin==="constants"?Number.isInteger(c)?`const float ${this.id} = ${c}.0;
`:`const float ${this.id} = ${c};
`:`uniform float ${this.id};
`}updateValue(c){this.origin!=="constants"&&this.kernel.setUniform1f(this.id,this.uploadValue=c)}};G.exports={WebGLKernelValueFloat:_}}),Rr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValue:I}=Pt();var _=class extends I{constructor(c,d){super(c,d),this.uploadValue=c}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName};
`}getSource(c){return this.origin==="constants"?`const int ${this.id} = ${parseInt(c)};
`:`uniform int ${this.id};
`}updateValue(c){this.origin!=="constants"&&this.kernel.setUniform1i(this.id,this.uploadValue=c)}};G.exports={WebGLKernelValueInteger:_}}),Qe=s((j,G)=>{const{WebGLKernelValue:D}=Pt(),{Input:I}=a();var _=class extends D{checkSize(c,d){if(!this.kernel.validate)return;const{maxTextureSize:E}=this.kernel.constructor.features;if(c>E||d>E)throw c>d?new Error(`Argument texture width of ${c} larger than maximum size of ${E} for your GPU`):c<d?new Error(`Argument texture height of ${d} larger than maximum size of ${E} for your GPU`):new Error(`Argument texture height and width of ${d} larger than maximum size of ${E} for your GPU`)}setup(){this.requestTexture(),this.setupTexture(),this.defineTexture()}requestTexture(){this.texture=this.onRequestTexture()}defineTexture(){const{context:c}=this;c.activeTexture(this.contextHandle),c.bindTexture(c.TEXTURE_2D,this.texture),c.texParameteri(c.TEXTURE_2D,c.TEXTURE_WRAP_S,c.CLAMP_TO_EDGE),c.texParameteri(c.TEXTURE_2D,c.TEXTURE_WRAP_T,c.CLAMP_TO_EDGE),c.texParameteri(c.TEXTURE_2D,c.TEXTURE_MIN_FILTER,c.NEAREST),c.texParameteri(c.TEXTURE_2D,c.TEXTURE_MAG_FILTER,c.NEAREST)}setupTexture(){this.contextHandle=this.onRequestContextHandle(),this.index=this.onRequestIndex(),this.dimensionsId=this.id+"Dim",this.sizeId=this.id+"Size"}getBitRatio(c){if(Array.isArray(c[0]))return this.getBitRatio(c[0]);if(c.constructor===I)return this.getBitRatio(c.value);switch(c.constructor){case Uint8ClampedArray:case Uint8Array:return 1;case Uint16Array:return 2;case Int8Array:case Int16Array:case Float32Array:case Int32Array:default:return 4}}destroy(){this.prevArg&&this.prevArg.delete(),this.context.deleteTexture(this.texture)}};G.exports={WebGLKernelArray:_}}),Os=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe();function _(d){return{width:d.width>0?d.width:d.videoWidth,height:d.height>0?d.height:d.videoHeight}}var c=class extends I{constructor(d,E){super(d,E);const{width:$,height:P}=_(d);this.checkSize($,P),this.dimensions=[$,P,1],this.textureSize=[$,P],this.uploadValue=d}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName};
`}getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(d){if(d.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(d.constructor);return}const{context:E}=this;E.activeTexture(this.contextHandle),E.bindTexture(E.TEXTURE_2D,this.texture),E.pixelStorei(E.UNPACK_FLIP_Y_WEBGL,!0),E.texImage2D(E.TEXTURE_2D,0,E.RGBA,E.RGBA,E.UNSIGNED_BYTE,this.uploadValue=d),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGLKernelValueHTMLImage:c,mediaSize:_}}),xn=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueHTMLImage:I,mediaSize:_}=Os();var c=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(d){const{width:E,height:$}=_(d);this.checkSize(E,$),this.dimensions=[E,$,1],this.textureSize=[E,$],this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(d)}};G.exports={WebGLKernelValueDynamicHTMLImage:c}}),ho=s((j,G)=>{const{WebGLKernelValueHTMLImage:D}=Os();var I=class extends D{};G.exports={WebGLKernelValueHTMLVideo:I}}),po=s((j,G)=>{const{WebGLKernelValueDynamicHTMLImage:D}=xn();var I=class extends D{};G.exports={WebGLKernelValueDynamicHTMLVideo:I}}),bn=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe();var _=class extends I{constructor(c,d){super(c,d),this.bitRatio=4;let[E,$,P]=c.size;this.dimensions=new Int32Array([E||1,$||1,P||1]),this.textureSize=D.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength)}getStringValueHandler(){return D.linesToString([`const uploadValue_${this.name} = new Float32Array(${this.uploadArrayLength})`,`flattenTo(${this.varName}.value, uploadValue_${this.name})`])}getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(c.constructor);return}const{context:d}=this;D.flattenTo(c.value,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGLKernelValueSingleInput:_}}),fo=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueSingleInput:I}=bn();var _=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(c){let[d,E,$]=c.size;this.dimensions=new Int32Array([d||1,E||1,$||1]),this.textureSize=D.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGLKernelValueDynamicSingleInput:_}}),wn=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe();var _=class extends I{constructor(c,d){super(c,d),this.bitRatio=this.getBitRatio(c);const[E,$,P]=c.size;this.dimensions=new Int32Array([E||1,$||1,P||1]),this.textureSize=D.getMemoryOptimizedPackedTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*(4/this.bitRatio),this.checkSize(this.textureSize[0],this.textureSize[1]),this.TranserArrayType=this.getTransferArrayType(c.value),this.preUploadValue=new this.TranserArrayType(this.uploadArrayLength),this.uploadValue=new Uint8Array(this.preUploadValue.buffer)}getStringValueHandler(){return D.linesToString([`const preUploadValue_${this.name} = new ${this.TranserArrayType.name}(${this.uploadArrayLength})`,`const uploadValue_${this.name} = new Uint8Array(preUploadValue_${this.name}.buffer)`,`flattenTo(${this.varName}.value, preUploadValue_${this.name})`])}getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(value.constructor);return}const{context:d}=this;D.flattenTo(c.value,this.preUploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.UNSIGNED_BYTE,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGLKernelValueUnsignedInput:_}}),Fr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueUnsignedInput:I}=wn();var _=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(c){let[d,E,$]=c.size;this.dimensions=new Int32Array([d||1,E||1,$||1]),this.textureSize=D.getMemoryOptimizedPackedTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*(4/this.bitRatio),this.checkSize(this.textureSize[0],this.textureSize[1]);const P=this.getTransferArrayType(c.value);this.preUploadValue=new P(this.uploadArrayLength),this.uploadValue=new Uint8Array(this.preUploadValue.buffer),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGLKernelValueDynamicUnsignedInput:_}}),Rs=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe(),_="Source and destination textures are the same.  Use immutable = true and manually cleanup kernel output texture memory with texture.delete()";var c=class extends I{constructor(d,E){super(d,E);const[$,P]=d.size;this.checkSize($,P),this.dimensions=d.dimensions,this.textureSize=d.size,this.uploadValue=d.texture,this.forceUploadEachRun=!0}setup(){this.setupTexture()}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName}.texture;
`}getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(d){if(d.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(d.constructor);return}if(this.checkContext&&d.context!==this.context)throw new Error(`Value ${this.name} (${this.type}) must be from same context`);const{kernel:E,context:$}=this;if(E.pipeline)if(E.immutable)E.updateTextureArgumentRefs(this,d);else{if(E.texture&&E.texture.texture===d.texture)throw new Error(_);if(E.mappedTextures){const{mappedTextures:P}=E;for(let y=0;y<P.length;y++)if(P[y].texture===d.texture)throw new Error(_)}}$.activeTexture(this.contextHandle),$.bindTexture($.TEXTURE_2D,this.uploadValue=d.texture),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGLKernelValueMemoryOptimizedNumberTexture:c,sameError:_}}),Gr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueMemoryOptimizedNumberTexture:I}=Rs();var _=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(c){this.dimensions=c.dimensions,this.checkSize(c.size[0],c.size[1]),this.textureSize=c.size,this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGLKernelValueDynamicMemoryOptimizedNumberTexture:_}}),vn=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe(),{sameError:_}=Rs();var c=class extends I{constructor(d,E){super(d,E);const[$,P]=d.size;this.checkSize($,P);const{size:y,dimensions:p}=d;this.bitRatio=this.getBitRatio(d),this.dimensions=p,this.textureSize=y,this.uploadValue=d.texture,this.forceUploadEachRun=!0}setup(){this.setupTexture()}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName}.texture;
`}getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(d){if(d.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(d.constructor);return}if(this.checkContext&&d.context!==this.context)throw new Error(`Value ${this.name} (${this.type}) must be from same context`);const{kernel:E,context:$}=this;if(E.pipeline)if(E.immutable)E.updateTextureArgumentRefs(this,d);else{if(E.texture&&E.texture.texture===d.texture)throw new Error(_);if(E.mappedTextures){const{mappedTextures:P}=E;for(let y=0;y<P.length;y++)if(P[y].texture===d.texture)throw new Error(_)}}$.activeTexture(this.contextHandle),$.bindTexture($.TEXTURE_2D,this.uploadValue=d.texture),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGLKernelValueNumberTexture:c}}),Lr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueNumberTexture:I}=vn();var _=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(c){this.dimensions=c.dimensions,this.checkSize(c.size[0],c.size[1]),this.textureSize=c.size,this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGLKernelValueDynamicNumberTexture:_}}),kn=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe();var _=class extends I{constructor(c,d){super(c,d),this.bitRatio=4,this.dimensions=D.getDimensions(c,!0),this.textureSize=D.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength)}getStringValueHandler(){return D.linesToString([`const uploadValue_${this.name} = new Float32Array(${this.uploadArrayLength})`,`flattenTo(${this.varName}, uploadValue_${this.name})`])}getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(c.constructor);return}const{context:d}=this;D.flattenTo(c,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGLKernelValueSingleArray:_}}),mo=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueSingleArray:I}=kn();var _=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(c){this.dimensions=D.getDimensions(c,!0),this.textureSize=D.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGLKernelValueDynamicSingleArray:_}}),Tn=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe();var _=class extends I{constructor(c,d){super(c,d),this.bitRatio=4,this.setShape(c)}setShape(c){const d=D.getDimensions(c,!0);this.textureSize=D.getMemoryOptimizedFloatTextureSize(d,this.bitRatio),this.dimensions=new Int32Array([d[1],1,1]),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength)}getStringValueHandler(){return D.linesToString([`const uploadValue_${this.name} = new Float32Array(${this.uploadArrayLength})`,`flattenTo(${this.varName}, uploadValue_${this.name})`])}getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(c.constructor);return}const{context:d}=this;D.flatten2dArrayTo(c,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGLKernelValueSingleArray1DI:_}}),go=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueSingleArray1DI:I}=Tn();var _=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(c){this.setShape(c),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGLKernelValueDynamicSingleArray1DI:_}}),Sn=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe();var _=class extends I{constructor(c,d){super(c,d),this.bitRatio=4,this.setShape(c)}setShape(c){const d=D.getDimensions(c,!0);this.textureSize=D.getMemoryOptimizedFloatTextureSize(d,this.bitRatio),this.dimensions=new Int32Array([d[1],d[2],1]),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength)}getStringValueHandler(){return D.linesToString([`const uploadValue_${this.name} = new Float32Array(${this.uploadArrayLength})`,`flattenTo(${this.varName}, uploadValue_${this.name})`])}getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(c.constructor);return}const{context:d}=this;D.flatten3dArrayTo(c,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGLKernelValueSingleArray2DI:_}}),yo=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueSingleArray2DI:I}=Sn();var _=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(c){this.setShape(c),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGLKernelValueDynamicSingleArray2DI:_}}),_n=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe();var _=class extends I{constructor(c,d){super(c,d),this.bitRatio=4,this.setShape(c)}setShape(c){const d=D.getDimensions(c,!0);this.textureSize=D.getMemoryOptimizedFloatTextureSize(d,this.bitRatio),this.dimensions=new Int32Array([d[1],d[2],d[3]]),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength)}getStringValueHandler(){return D.linesToString([`const uploadValue_${this.name} = new Float32Array(${this.uploadArrayLength})`,`flattenTo(${this.varName}, uploadValue_${this.name})`])}getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(c.constructor);return}const{context:d}=this;D.flatten4dArrayTo(c,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGLKernelValueSingleArray3DI:_}}),xo=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueSingleArray3DI:I}=_n();var _=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(c){this.setShape(c),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGLKernelValueDynamicSingleArray3DI:_}}),Ur=s((j,G)=>{const{WebGLKernelValue:D}=Pt();var I=class extends D{constructor(_,c){super(_,c),this.uploadValue=_}getSource(_){return this.origin==="constants"?`const vec2 ${this.id} = vec2(${_[0]},${_[1]});
`:`uniform vec2 ${this.id};
`}getStringValueHandler(){return this.origin==="constants"?"":`const uploadValue_${this.name} = ${this.varName};
`}updateValue(_){this.origin!=="constants"&&this.kernel.setUniform2fv(this.id,this.uploadValue=_)}};G.exports={WebGLKernelValueArray2:I}}),Vr=s((j,G)=>{const{WebGLKernelValue:D}=Pt();var I=class extends D{constructor(_,c){super(_,c),this.uploadValue=_}getSource(_){return this.origin==="constants"?`const vec3 ${this.id} = vec3(${_[0]},${_[1]},${_[2]});
`:`uniform vec3 ${this.id};
`}getStringValueHandler(){return this.origin==="constants"?"":`const uploadValue_${this.name} = ${this.varName};
`}updateValue(_){this.origin!=="constants"&&this.kernel.setUniform3fv(this.id,this.uploadValue=_)}};G.exports={WebGLKernelValueArray3:I}}),Kr=s((j,G)=>{const{WebGLKernelValue:D}=Pt();var I=class extends D{constructor(_,c){super(_,c),this.uploadValue=_}getSource(_){return this.origin==="constants"?`const vec4 ${this.id} = vec4(${_[0]},${_[1]},${_[2]},${_[3]});
`:`uniform vec4 ${this.id};
`}getStringValueHandler(){return this.origin==="constants"?"":`const uploadValue_${this.name} = ${this.varName};
`}updateValue(_){this.origin!=="constants"&&this.kernel.setUniform4fv(this.id,this.uploadValue=_)}};G.exports={WebGLKernelValueArray4:I}}),Cn=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe();var _=class extends I{constructor(c,d){super(c,d),this.bitRatio=this.getBitRatio(c),this.dimensions=D.getDimensions(c,!0),this.textureSize=D.getMemoryOptimizedPackedTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*(4/this.bitRatio),this.checkSize(this.textureSize[0],this.textureSize[1]),this.TranserArrayType=this.getTransferArrayType(c),this.preUploadValue=new this.TranserArrayType(this.uploadArrayLength),this.uploadValue=new Uint8Array(this.preUploadValue.buffer)}getStringValueHandler(){return D.linesToString([`const preUploadValue_${this.name} = new ${this.TranserArrayType.name}(${this.uploadArrayLength})`,`const uploadValue_${this.name} = new Uint8Array(preUploadValue_${this.name}.buffer)`,`flattenTo(${this.varName}, preUploadValue_${this.name})`])}getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(c.constructor);return}const{context:d}=this;D.flattenTo(c,this.preUploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.UNSIGNED_BYTE,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGLKernelValueUnsignedArray:_}}),Nr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueUnsignedArray:I}=Cn();var _=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(c){this.dimensions=D.getDimensions(c,!0),this.textureSize=D.getMemoryOptimizedPackedTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*(4/this.bitRatio),this.checkSize(this.textureSize[0],this.textureSize[1]);const d=this.getTransferArrayType(c);this.preUploadValue=new d(this.uploadArrayLength),this.uploadValue=new Uint8Array(this.preUploadValue.buffer),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGLKernelValueDynamicUnsignedArray:_}}),Br=s((j,G)=>{const{WebGLKernelValueBoolean:D}=zr(),{WebGLKernelValueFloat:I}=Or(),{WebGLKernelValueInteger:_}=Rr(),{WebGLKernelValueHTMLImage:c}=Os(),{WebGLKernelValueDynamicHTMLImage:d}=xn(),{WebGLKernelValueHTMLVideo:E}=ho(),{WebGLKernelValueDynamicHTMLVideo:$}=po(),{WebGLKernelValueSingleInput:P}=bn(),{WebGLKernelValueDynamicSingleInput:y}=fo(),{WebGLKernelValueUnsignedInput:p}=wn(),{WebGLKernelValueDynamicUnsignedInput:g}=Fr(),{WebGLKernelValueMemoryOptimizedNumberTexture:k}=Rs(),{WebGLKernelValueDynamicMemoryOptimizedNumberTexture:o}=Gr(),{WebGLKernelValueNumberTexture:l}=vn(),{WebGLKernelValueDynamicNumberTexture:x}=Lr(),{WebGLKernelValueSingleArray:w}=kn(),{WebGLKernelValueDynamicSingleArray:v}=mo(),{WebGLKernelValueSingleArray1DI:C}=Tn(),{WebGLKernelValueDynamicSingleArray1DI:b}=go(),{WebGLKernelValueSingleArray2DI:T}=Sn(),{WebGLKernelValueDynamicSingleArray2DI:h}=yo(),{WebGLKernelValueSingleArray3DI:F}=_n(),{WebGLKernelValueDynamicSingleArray3DI:O}=xo(),{WebGLKernelValueArray2:z}=Ur(),{WebGLKernelValueArray3:L}=Vr(),{WebGLKernelValueArray4:V}=Kr(),{WebGLKernelValueUnsignedArray:U}=Cn(),{WebGLKernelValueDynamicUnsignedArray:X}=Nr(),q={unsigned:{dynamic:{Boolean:D,Integer:_,Float:I,Array:X,"Array(2)":z,"Array(3)":L,"Array(4)":V,"Array1D(2)":!1,"Array1D(3)":!1,"Array1D(4)":!1,"Array2D(2)":!1,"Array2D(3)":!1,"Array2D(4)":!1,"Array3D(2)":!1,"Array3D(3)":!1,"Array3D(4)":!1,Input:g,NumberTexture:x,"ArrayTexture(1)":x,"ArrayTexture(2)":x,"ArrayTexture(3)":x,"ArrayTexture(4)":x,MemoryOptimizedNumberTexture:o,HTMLCanvas:d,OffscreenCanvas:d,HTMLImage:d,ImageBitmap:d,ImageData:d,HTMLImageArray:!1,HTMLVideo:$},static:{Boolean:D,Float:I,Integer:_,Array:U,"Array(2)":z,"Array(3)":L,"Array(4)":V,"Array1D(2)":!1,"Array1D(3)":!1,"Array1D(4)":!1,"Array2D(2)":!1,"Array2D(3)":!1,"Array2D(4)":!1,"Array3D(2)":!1,"Array3D(3)":!1,"Array3D(4)":!1,Input:p,NumberTexture:l,"ArrayTexture(1)":l,"ArrayTexture(2)":l,"ArrayTexture(3)":l,"ArrayTexture(4)":l,MemoryOptimizedNumberTexture:k,HTMLCanvas:c,OffscreenCanvas:c,HTMLImage:c,ImageBitmap:c,ImageData:c,HTMLImageArray:!1,HTMLVideo:E}},single:{dynamic:{Boolean:D,Integer:_,Float:I,Array:v,"Array(2)":z,"Array(3)":L,"Array(4)":V,"Array1D(2)":b,"Array1D(3)":b,"Array1D(4)":b,"Array2D(2)":h,"Array2D(3)":h,"Array2D(4)":h,"Array3D(2)":O,"Array3D(3)":O,"Array3D(4)":O,Input:y,NumberTexture:x,"ArrayTexture(1)":x,"ArrayTexture(2)":x,"ArrayTexture(3)":x,"ArrayTexture(4)":x,MemoryOptimizedNumberTexture:o,HTMLCanvas:d,OffscreenCanvas:d,HTMLImage:d,ImageBitmap:d,ImageData:d,HTMLImageArray:!1,HTMLVideo:$},static:{Boolean:D,Float:I,Integer:_,Array:w,"Array(2)":z,"Array(3)":L,"Array(4)":V,"Array1D(2)":C,"Array1D(3)":C,"Array1D(4)":C,"Array2D(2)":T,"Array2D(3)":T,"Array2D(4)":T,"Array3D(2)":F,"Array3D(3)":F,"Array3D(4)":F,Input:P,NumberTexture:l,"ArrayTexture(1)":l,"ArrayTexture(2)":l,"ArrayTexture(3)":l,"ArrayTexture(4)":l,MemoryOptimizedNumberTexture:k,HTMLCanvas:c,OffscreenCanvas:c,HTMLImage:c,ImageBitmap:c,ImageData:c,HTMLImageArray:!1,HTMLVideo:E}}};function W(ee,se,Z,ie){if(!ee)throw new Error("type missing");if(!se)throw new Error("dynamic missing");if(!Z)throw new Error("precision missing");ie.type&&(ee=ie.type);const he=q[Z][se];if(ee==="WebGPUBuffer")throw new Error("this kernel runs on WebGL but received a WebGPU pipeline buffer; await handle.toArray() first, or give this kernel the async contract (asyncMode: true / mode: 'async') so the readback happens for you");if(he[ee]===!1)return null;if(he[ee]===void 0)throw new Error(`Could not find a KernelValue for ${ee}`);return he[ee]}G.exports={lookupKernelValueType:W,kernelValueMaps:q}}),Fs=s((j,G)=>{const{GLKernel:D}=Ar(),{FunctionBuilder:I}=N(),{WebGLFunctionNode:_}=yn(),{utils:c}=m(),d=Dr(),{fragmentShader:E}=oo(),{vertexShader:$}=lo(),{glKernelString:P}=Pr(),{lookupKernelValueType:y}=Br();let p=null,g=null,k=null,o=null,l=null;const x=[d],w=[],v={};var C=class extends D{static get isSupported(){return p!==null||(this.setupFeatureChecks(),p=this.isContextMatch(k)),p}static setupFeatureChecks(){typeof document<"u"?g=document.createElement("canvas"):typeof OffscreenCanvas<"u"&&(g=new OffscreenCanvas(0,0)),g&&(k=g.getContext("webgl"),!k&&!(g instanceof OffscreenCanvas)&&(k=g.getContext("experimental-webgl")),!(!k||!k.getExtension)&&(o={OES_texture_float:k.getExtension("OES_texture_float"),OES_texture_float_linear:k.getExtension("OES_texture_float_linear"),OES_element_index_uint:k.getExtension("OES_element_index_uint"),WEBGL_draw_buffers:k.getExtension("WEBGL_draw_buffers")},l=this.getFeatures()))}static isContextMatch(b){return typeof WebGLRenderingContext<"u"?b instanceof WebGLRenderingContext:!1}static getIsTextureFloat(){return!!o.OES_texture_float}static getIsDrawBuffers(){return!!o.WEBGL_draw_buffers}static getChannelCount(){return o.WEBGL_draw_buffers?k.getParameter(o.WEBGL_draw_buffers.MAX_DRAW_BUFFERS_WEBGL):1}static getMaxTextureSize(){return k.getParameter(k.MAX_TEXTURE_SIZE)}static lookupKernelValueType(b,T,h,F){return y(b,T,h,F)}static get testCanvas(){return g}static get testContext(){return k}static get features(){return l}static get fragmentShader(){return E}static get vertexShader(){return $}constructor(b,T){super(b,T),this.program=null,this.pipeline=T.pipeline,this.endianness=c.systemEndianness(),this.extensions={},this.argumentTextureCount=0,this.constantTextureCount=0,this.fragShader=null,this.vertShader=null,this.drawBuffersMap=null,this.maxTexSize=null,this.onRequestSwitchKernel=null,this.texture=null,this.mappedTextures=null,this.mergeSettings(b.settings||T),this.threadDim=null,this.framebuffer=null,this.buffer=null,this.textureCache=[],this.programUniformLocationCache={},this.uniform1fCache={},this.uniform1iCache={},this.uniform2fCache={},this.uniform2fvCache={},this.uniform2ivCache={},this.uniform3fvCache={},this.uniform3ivCache={},this.uniform4fvCache={},this.uniform4ivCache={}}initCanvas(){if(typeof document<"u"){const b=document.createElement("canvas");return b.width=2,b.height=2,b}else if(typeof OffscreenCanvas<"u")return new OffscreenCanvas(0,0)}initContext(){const b={alpha:!1,depth:!1,antialias:!1};return this.canvas.getContext("webgl",b)||this.canvas.getContext("experimental-webgl",b)}initPlugins(b){const T=[],{source:h}=this;if(typeof h=="string")for(let F=0;F<x.length;F++){const O=x[F];h.match(O.functionMatch)&&T.push(O)}else if(typeof h=="object"&&b.pluginNames)for(let F=0;F<x.length;F++){const O=x[F];b.pluginNames.some(z=>z===O.name)&&T.push(O)}return T}initExtensions(){this.extensions={OES_texture_float:this.context.getExtension("OES_texture_float"),OES_texture_float_linear:this.context.getExtension("OES_texture_float_linear"),OES_element_index_uint:this.context.getExtension("OES_element_index_uint"),WEBGL_draw_buffers:this.context.getExtension("WEBGL_draw_buffers"),WEBGL_color_buffer_float:this.context.getExtension("WEBGL_color_buffer_float")}}validateSettings(b){if(!this.validate){this.texSize=c.getKernelTextureSize({optimizeFloatMemory:this.optimizeFloatMemory,precision:this.precision},this.output);return}const{features:T}=this.constructor;if(this.optimizeFloatMemory===!0&&!T.isTextureFloat)throw new Error("Float textures are not supported");if(this.precision==="single"&&!T.isFloatRead)throw new Error("Single precision not supported");if(!this.graphical&&this.precision===null&&(this.precision=T.isTextureFloat&&T.isFloatRead?"single":"unsigned"),this.subKernels&&this.subKernels.length>0&&!this.extensions.WEBGL_draw_buffers)throw new Error("could not instantiate draw buffers extension");if(this.fixIntegerDivisionAccuracy===null?this.fixIntegerDivisionAccuracy=!T.isIntegerDivisionAccurate:this.fixIntegerDivisionAccuracy&&T.isIntegerDivisionAccurate&&(this.fixIntegerDivisionAccuracy=!1),this.checkOutput(),!this.output||this.output.length===0){if(b.length!==1)throw new Error("Auto output only supported for kernels with only one input");const h=c.getVariableType(b[0],this.strictIntegers);switch(h){case"Array":this.output=c.getDimensions(h);break;case"NumberTexture":case"MemoryOptimizedNumberTexture":case"ArrayTexture(1)":case"ArrayTexture(2)":case"ArrayTexture(3)":case"ArrayTexture(4)":this.output=b[0].output;break;default:throw new Error("Auto output not supported for input type: "+h)}}if(this.graphical){if(this.output.length!==2)throw new Error("Output must have 2 dimensions on graphical mode");this.precision==="precision"&&(this.precision="unsigned",console.warn("Cannot use graphical mode and single precision at the same time")),this.texSize=c.clone(this.output);return}else this.precision===null&&T.isTextureFloat&&(this.precision="single");this.texSize=c.getKernelTextureSize({optimizeFloatMemory:this.optimizeFloatMemory,precision:this.precision},this.output),this.checkTextureSize()}updateMaxTexSize(){const{texSize:b,canvas:T}=this;if(this.maxTexSize===null){let h=w.indexOf(T);h===-1&&(h=w.length,w.push(T),v[h]=[b[0],b[1]]),this.maxTexSize=v[h]}this.maxTexSize[0]<b[0]&&(this.maxTexSize[0]=b[0]),this.maxTexSize[1]<b[1]&&(this.maxTexSize[1]=b[1])}setupArguments(b){this.kernelArguments=[],this.argumentTextureCount=0;const T=this.argumentTypes===null;if(T&&(this.argumentTypes=[]),this.argumentSizes=[],this.argumentBitRatios=[],b.length<this.argumentNames.length)throw new Error("not enough arguments for kernel");if(b.length>this.argumentNames.length)throw new Error("too many arguments for kernel");const{context:h}=this;let F=0;const O=()=>this.createTexture(),z=()=>this.constantTextureCount+F++,L=U=>{this.switchKernels({type:"argumentMismatch",needed:U})},V=()=>h.TEXTURE0+this.constantTextureCount+this.argumentTextureCount++;for(let U=0;U<b.length;U++){const X=b[U],q=this.argumentNames[U];let W;T?(W=c.getVariableType(X,this.strictIntegers),this.argumentTypes.push(W)):W=this.argumentTypes[U];const ee=this.constructor.lookupKernelValueType(W,this.dynamicArguments?"dynamic":"static",this.precision,b[U]);if(ee===null)return this.requestFallback(b);const se=new ee(X,{name:q,type:W,tactic:this.tactic,origin:"user",context:h,checkContext:this.checkContext,kernel:this,strictIntegers:this.strictIntegers,onRequestTexture:O,onRequestIndex:z,onUpdateValueMismatch:L,onRequestContextHandle:V});this.kernelArguments.push(se),se.setup(),this.argumentSizes.push(se.textureSize),this.argumentBitRatios[U]=se.bitRatio}}createTexture(){const b=this.context.createTexture();return this.textureCache.push(b),b}deleteTexture(b){const T=this.textureCache.indexOf(b);T!==-1&&this.textureCache.splice(T,1),this.context&&this.context.deleteTexture(b)}setupConstants(b){const{context:T}=this;this.kernelConstants=[],this.forceUploadKernelConstants=[];let h=this.constantTypes===null;h&&(this.constantTypes={}),this.constantBitRatios={};let F=0;for(const O in this.constants){const z=this.constants[O];let L;h?(L=c.getVariableType(z,this.strictIntegers),this.constantTypes[O]=L):L=this.constantTypes[O];const V=this.constructor.lookupKernelValueType(L,"static",this.precision,z);if(V===null)return this.requestFallback(b);const U=new V(z,{name:O,type:L,tactic:this.tactic,origin:"constants",context:this.context,checkContext:this.checkContext,kernel:this,strictIntegers:this.strictIntegers,onRequestTexture:()=>this.createTexture(),onRequestIndex:()=>F++,onRequestContextHandle:()=>T.TEXTURE0+this.constantTextureCount++});this.constantBitRatios[O]=U.bitRatio,this.kernelConstants.push(U),U.setup(),U.forceUploadEachRun&&this.forceUploadKernelConstants.push(U)}}build(){if(this.built||(this.initExtensions(),this.validateSettings(arguments),this.setupConstants(arguments),this.fallbackRequested)||(this.setupArguments(arguments),this.fallbackRequested))return;this.updateMaxTexSize(),this.translateSource();const b=this.pickRenderStrategy(arguments);if(b)return b;const{texSize:T,context:h,canvas:F}=this;h.enable(h.SCISSOR_TEST),this.pipeline&&this.precision==="single"?(h.viewport(0,0,this.maxTexSize[0],this.maxTexSize[1]),F.width=this.maxTexSize[0],F.height=this.maxTexSize[1]):(h.viewport(0,0,this.maxTexSize[0],this.maxTexSize[1]),F.width=this.maxTexSize[0],F.height=this.maxTexSize[1]);const O=this.threadDim=Array.from(this.output);for(;O.length<3;)O.push(1);const z=this.getVertexShader(arguments),L=h.createShader(h.VERTEX_SHADER);h.shaderSource(L,z),h.compileShader(L),this.vertShader=L;const V=this.getFragmentShader(arguments),U=h.createShader(h.FRAGMENT_SHADER);if(h.shaderSource(U,V),h.compileShader(U),this.fragShader=U,this.debug&&(console.log("GLSL Shader Output:"),console.log(V)),!h.getShaderParameter(L,h.COMPILE_STATUS))throw new Error("Error compiling vertex shader: "+h.getShaderInfoLog(L));if(!h.getShaderParameter(U,h.COMPILE_STATUS))throw new Error("Error compiling fragment shader: "+h.getShaderInfoLog(U));const X=this.program=h.createProgram();h.attachShader(X,L),h.attachShader(X,U),h.linkProgram(X),this.framebuffer=h.createFramebuffer(),this.framebuffer.width=T[0],this.framebuffer.height=T[1],this.rawValueFramebuffers={};const q=new Float32Array([-1,-1,1,-1,-1,1,1,1]),W=new Float32Array([0,0,1,0,0,1,1,1]),ee=q.byteLength;let se=this.buffer;se?h.bindBuffer(h.ARRAY_BUFFER,se):(se=this.buffer=h.createBuffer(),h.bindBuffer(h.ARRAY_BUFFER,se),h.bufferData(h.ARRAY_BUFFER,q.byteLength+W.byteLength,h.STATIC_DRAW)),h.bufferSubData(h.ARRAY_BUFFER,0,q),h.bufferSubData(h.ARRAY_BUFFER,ee,W);const Z=h.getAttribLocation(this.program,"aPos");Z!==-1&&(h.enableVertexAttribArray(Z),h.vertexAttribPointer(Z,2,h.FLOAT,!1,0,0));const ie=h.getAttribLocation(this.program,"aTexCoord");ie!==-1&&(h.enableVertexAttribArray(ie),h.vertexAttribPointer(ie,2,h.FLOAT,!1,0,ee)),h.bindFramebuffer(h.FRAMEBUFFER,this.framebuffer);let he=0;h.useProgram(this.program);for(let we in this.constants)this.kernelConstants[he++].updateValue(this.constants[we]);this._setupOutputTexture(),this.subKernels!==null&&this.subKernels.length>0&&(this._mappedTextureSwitched={},this._setupSubOutputTextures()),this.buildSignature(arguments),this.built=!0}translateSource(){const b=I.fromKernel(this,_,{fixIntegerDivisionAccuracy:this.fixIntegerDivisionAccuracy});this.translatedSource=b.getPrototypeString("kernel"),this.setupReturnTypes(b)}setupReturnTypes(b){if(!this.graphical&&!this.returnType&&(this.returnType=b.getKernelResultType()),this.subKernels&&this.subKernels.length>0)for(let T=0;T<this.subKernels.length;T++){const h=this.subKernels[T];h.returnType||(h.returnType=b.getSubKernelResultType(T))}}run(){const{kernelArguments:b,texSize:T,forceUploadKernelConstants:h,context:F}=this;F.useProgram(this.program),F.scissor(0,0,T[0],T[1]),this.dynamicOutput&&(this.setUniform3iv("uOutputDim",new Int32Array(this.threadDim)),this.setUniform2iv("uTexSize",T)),this.setUniform2f("ratio",T[0]/this.maxTexSize[0],T[1]/this.maxTexSize[1]);for(let O=0;O<h.length;O++){const z=h[O];if(z.updateValue(this.constants[z.name]),this.switchingKernels)return}for(let O=0;O<b.length;O++)if(b[O].updateValue(arguments[O]),this.switchingKernels)return;if(this.plugins)for(let O=0;O<this.plugins.length;O++){const z=this.plugins[O];z.onBeforeRun&&z.onBeforeRun(this)}if(this.graphical){if(this.pipeline)return F.bindRenderbuffer(F.RENDERBUFFER,null),F.bindFramebuffer(F.FRAMEBUFFER,this.framebuffer),this.immutable&&this._replaceOutputTexture(),F.drawArrays(F.TRIANGLE_STRIP,0,4),this.immutable?this.texture.clone():this.texture;F.bindRenderbuffer(F.RENDERBUFFER,null),F.bindFramebuffer(F.FRAMEBUFFER,null),F.drawArrays(F.TRIANGLE_STRIP,0,4);return}F.bindFramebuffer(F.FRAMEBUFFER,this.framebuffer),this.immutable&&this._replaceOutputTexture(),this.subKernels!==null&&(this.immutable&&this._replaceSubOutputTextures(),this.drawBuffers()),F.drawArrays(F.TRIANGLE_STRIP,0,4)}drawBuffers(){this.extensions.WEBGL_draw_buffers.drawBuffersWEBGL(this.drawBuffersMap)}getInternalFormat(){return this.context.RGBA}getTextureFormat(){const{context:b}=this;switch(this.getInternalFormat()){case b.RGBA:return b.RGBA;default:throw new Error("Unknown internal format")}}_replaceOutputTexture(){if(this.texture.beforeMutate()||this._textureSwitched){const b=this.context;b.framebufferTexture2D(b.FRAMEBUFFER,b.COLOR_ATTACHMENT0,b.TEXTURE_2D,this.texture.texture,0),this._textureSwitched=!1}}_setupOutputTexture(){const b=this.context,T=this.texSize;if(this.texture){b.framebufferTexture2D(b.FRAMEBUFFER,b.COLOR_ATTACHMENT0,b.TEXTURE_2D,this.texture.texture,0);return}const h=this.createTexture();b.activeTexture(b.TEXTURE0+this.constantTextureCount+this.argumentTextureCount),b.bindTexture(b.TEXTURE_2D,h),b.texParameteri(b.TEXTURE_2D,b.TEXTURE_WRAP_S,b.CLAMP_TO_EDGE),b.texParameteri(b.TEXTURE_2D,b.TEXTURE_WRAP_T,b.CLAMP_TO_EDGE),b.texParameteri(b.TEXTURE_2D,b.TEXTURE_MIN_FILTER,b.NEAREST),b.texParameteri(b.TEXTURE_2D,b.TEXTURE_MAG_FILTER,b.NEAREST);const F=this.getInternalFormat();this.precision==="single"?b.texImage2D(b.TEXTURE_2D,0,F,T[0],T[1],0,b.RGBA,b.FLOAT,null):b.texImage2D(b.TEXTURE_2D,0,F,T[0],T[1],0,F,b.UNSIGNED_BYTE,null),b.framebufferTexture2D(b.FRAMEBUFFER,b.COLOR_ATTACHMENT0,b.TEXTURE_2D,h,0),this.texture=new this.TextureConstructor({texture:h,size:T,dimensions:this.threadDim,output:this.output,context:this.context,internalFormat:this.getInternalFormat(),textureFormat:this.getTextureFormat(),kernel:this})}_replaceSubOutputTextures(){const b=this.context;for(let T=0;T<this.mappedTextures.length;T++){const h=this.mappedTextures[T];(h.beforeMutate()||this._mappedTextureSwitched[T])&&(b.framebufferTexture2D(b.FRAMEBUFFER,b.COLOR_ATTACHMENT0+T+1,b.TEXTURE_2D,h.texture,0),this._mappedTextureSwitched[T]=!1)}}_setupSubOutputTextures(){const b=this.context;if(this.mappedTextures){for(let h=0;h<this.subKernels.length;h++)b.framebufferTexture2D(b.FRAMEBUFFER,b.COLOR_ATTACHMENT0+h+1,b.TEXTURE_2D,this.mappedTextures[h].texture,0);return}const T=this.texSize;this.drawBuffersMap=[b.COLOR_ATTACHMENT0],this.mappedTextures=[];for(let h=0;h<this.subKernels.length;h++){const F=this.createTexture();this.drawBuffersMap.push(b.COLOR_ATTACHMENT0+h+1),b.activeTexture(b.TEXTURE0+this.constantTextureCount+this.argumentTextureCount+h),b.bindTexture(b.TEXTURE_2D,F),b.texParameteri(b.TEXTURE_2D,b.TEXTURE_WRAP_S,b.CLAMP_TO_EDGE),b.texParameteri(b.TEXTURE_2D,b.TEXTURE_WRAP_T,b.CLAMP_TO_EDGE),b.texParameteri(b.TEXTURE_2D,b.TEXTURE_MIN_FILTER,b.NEAREST),b.texParameteri(b.TEXTURE_2D,b.TEXTURE_MAG_FILTER,b.NEAREST),this.precision==="single"?b.texImage2D(b.TEXTURE_2D,0,b.RGBA,T[0],T[1],0,b.RGBA,b.FLOAT,null):b.texImage2D(b.TEXTURE_2D,0,b.RGBA,T[0],T[1],0,b.RGBA,b.UNSIGNED_BYTE,null),b.framebufferTexture2D(b.FRAMEBUFFER,b.COLOR_ATTACHMENT0+h+1,b.TEXTURE_2D,F,0),this.mappedTextures.push(new this.TextureConstructor({texture:F,size:T,dimensions:this.threadDim,output:this.output,context:this.context,internalFormat:this.getInternalFormat(),textureFormat:this.getTextureFormat(),kernel:this}))}}setUniform1f(b,T){if(this.uniform1fCache.hasOwnProperty(b)&&T===this.uniform1fCache[b])return;this.uniform1fCache[b]=T;const h=this.getUniformLocation(b);this.context.uniform1f(h,T)}setUniform1i(b,T){if(this.uniform1iCache.hasOwnProperty(b)&&T===this.uniform1iCache[b])return;this.uniform1iCache[b]=T;const h=this.getUniformLocation(b);this.context.uniform1i(h,T)}setUniform2f(b,T,h){if(this.uniform2fCache.hasOwnProperty(b)){const O=this.uniform2fCache[b];if(T===O[0]&&h===O[1])return}this.uniform2fCache[b]=[T,h];const F=this.getUniformLocation(b);this.context.uniform2f(F,T,h)}setUniform2fv(b,T){if(this.uniform2fvCache.hasOwnProperty(b)){const F=this.uniform2fvCache[b];if(T[0]===F[0]&&T[1]===F[1])return}this.uniform2fvCache[b]=T;const h=this.getUniformLocation(b);this.context.uniform2fv(h,T)}setUniform2iv(b,T){if(this.uniform2ivCache.hasOwnProperty(b)){const F=this.uniform2ivCache[b];if(T[0]===F[0]&&T[1]===F[1])return}this.uniform2ivCache[b]=T;const h=this.getUniformLocation(b);this.context.uniform2iv(h,T)}setUniform3fv(b,T){if(this.uniform3fvCache.hasOwnProperty(b)){const F=this.uniform3fvCache[b];if(T[0]===F[0]&&T[1]===F[1]&&T[2]===F[2])return}this.uniform3fvCache[b]=T;const h=this.getUniformLocation(b);this.context.uniform3fv(h,T)}setUniform3iv(b,T){if(this.uniform3ivCache.hasOwnProperty(b)){const F=this.uniform3ivCache[b];if(T[0]===F[0]&&T[1]===F[1]&&T[2]===F[2])return}this.uniform3ivCache[b]=T;const h=this.getUniformLocation(b);this.context.uniform3iv(h,T)}setUniform4fv(b,T){if(this.uniform4fvCache.hasOwnProperty(b)){const F=this.uniform4fvCache[b];if(T[0]===F[0]&&T[1]===F[1]&&T[2]===F[2]&&T[3]===F[3])return}this.uniform4fvCache[b]=T;const h=this.getUniformLocation(b);this.context.uniform4fv(h,T)}setUniform4iv(b,T){if(this.uniform4ivCache.hasOwnProperty(b)){const F=this.uniform4ivCache[b];if(T[0]===F[0]&&T[1]===F[1]&&T[2]===F[2]&&T[3]===F[3])return}this.uniform4ivCache[b]=T;const h=this.getUniformLocation(b);this.context.uniform4iv(h,T)}getUniformLocation(b){return this.programUniformLocationCache.hasOwnProperty(b)?this.programUniformLocationCache[b]:this.programUniformLocationCache[b]=this.context.getUniformLocation(this.program,b)}_getFragShaderArtifactMap(b){return{HEADER:this._getHeaderString(),LOOP_MAX:this._getLoopMaxString(),PLUGINS:this._getPluginsString(),CONSTANTS:this._getConstantsString(),DECODE32_ENDIANNESS:this._getDecode32EndiannessString(),ENCODE32_ENDIANNESS:this._getEncode32EndiannessString(),DIVIDE_WITH_INTEGER_CHECK:this._getDivideWithIntegerCheckString(),INJECTED_NATIVE:this._getInjectedNative(),MAIN_CONSTANTS:this._getMainConstantsString(),MAIN_ARGUMENTS:this._getMainArgumentsString(b),KERNEL:this.getKernelString(),MAIN_RESULT:this.getMainResultString(),FLOAT_TACTIC_DECLARATION:this.getFloatTacticDeclaration(),INT_TACTIC_DECLARATION:this.getIntTacticDeclaration(),SAMPLER_2D_TACTIC_DECLARATION:this.getSampler2DTacticDeclaration(),SAMPLER_2D_ARRAY_TACTIC_DECLARATION:this.getSampler2DArrayTacticDeclaration()}}_getVertShaderArtifactMap(b){return{FLOAT_TACTIC_DECLARATION:this.getFloatTacticDeclaration(),INT_TACTIC_DECLARATION:this.getIntTacticDeclaration(),SAMPLER_2D_TACTIC_DECLARATION:this.getSampler2DTacticDeclaration(),SAMPLER_2D_ARRAY_TACTIC_DECLARATION:this.getSampler2DArrayTacticDeclaration()}}_getHeaderString(){return this.subKernels!==null?`#extension GL_EXT_draw_buffers : require
`:""}_getLoopMaxString(){return this.loopMaxIterations?` ${parseInt(this.loopMaxIterations)};
`:` 1000;
`}_getPluginsString(){return this.plugins?this.plugins.map(b=>b.source&&this.source.match(b.functionMatch)?b.source:"").join(`
`):`
`}_getConstantsString(){const b=[],{threadDim:T,texSize:h}=this;return this.dynamicOutput?b.push("uniform ivec3 uOutputDim","uniform ivec2 uTexSize"):b.push(`ivec3 uOutputDim = ivec3(${T[0]}, ${T[1]}, ${T[2]})`,`ivec2 uTexSize = ivec2(${h[0]}, ${h[1]})`),c.linesToString(b)}_getTextureCoordinate(){const b=this.subKernels;return b===null||b.length<1?`varying vec2 vTexCoord;
`:`out vec2 vTexCoord;
`}_getDecode32EndiannessString(){return this.endianness==="LE"?"":`  texel.rgba = texel.abgr;
`}_getEncode32EndiannessString(){return this.endianness==="LE"?"":`  texel.rgba = texel.abgr;
`}_getDivideWithIntegerCheckString(){return this.fixIntegerDivisionAccuracy?`float divWithIntCheck(float x, float y) {
  if (floor(x) == x && floor(y) == y) {
    float q = floor(x / y + 0.5);
    if (y * q == x) {
      return q;
    }
  }
  return x / y;
}

float integerCorrectionModulo(float number, float divisor) {
  if (number < 0.0) {
    number = abs(number);
    if (divisor < 0.0) {
      divisor = abs(divisor);
    }
    return -(number - (divisor * floor(divWithIntCheck(number, divisor))));
  }
  if (divisor < 0.0) {
    divisor = abs(divisor);
  }
  return number - (divisor * floor(divWithIntCheck(number, divisor)));
}`:""}_getMainArgumentsString(b){const T=[],{argumentNames:h}=this;for(let F=0;F<h.length;F++)T.push(this.kernelArguments[F].getSource(b[F]));return T.join("")}_getInjectedNative(){return this.injectedNative||""}_getMainConstantsString(){const b=[],{constants:T}=this;if(T){let h=0;for(const F in T)this.constants.hasOwnProperty(F)&&b.push(this.kernelConstants[h++].getSource(this.constants[F]))}return b.join("")}getRawValueFramebuffer(b,T){if(this.rawValueFramebuffers[b]||(this.rawValueFramebuffers[b]={}),!this.rawValueFramebuffers[b][T]){const h=this.context.createFramebuffer();h.width=b,h.height=T,this.rawValueFramebuffers[b][T]=h}return this.rawValueFramebuffers[b][T]}getKernelResultDeclaration(){switch(this.returnType){case"Array(2)":return"vec2 kernelResult";case"Array(3)":return"vec3 kernelResult";case"Array(4)":return"vec4 kernelResult";case"LiteralInteger":case"Float":case"Number":case"Integer":return"float kernelResult";default:if(this.graphical)return"float kernelResult";throw new Error(`unrecognized output type "${this.returnType}"`)}}getKernelString(){const b=[this.getKernelResultDeclaration()],{subKernels:T}=this;if(T!==null)switch(this.returnType){case"Number":case"Float":case"Integer":for(let h=0;h<T.length;h++){const F=T[h];b.push(F.returnType==="Integer"?`int subKernelResult_${F.name} = 0`:`float subKernelResult_${F.name} = 0.0`)}break;case"Array(2)":for(let h=0;h<T.length;h++)b.push(`vec2 subKernelResult_${T[h].name}`);break;case"Array(3)":for(let h=0;h<T.length;h++)b.push(`vec3 subKernelResult_${T[h].name}`);break;case"Array(4)":for(let h=0;h<T.length;h++)b.push(`vec4 subKernelResult_${T[h].name}`);break}return c.linesToString(b)+this.translatedSource}getMainResultGraphical(){return c.linesToString(["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  gl_FragColor = actualColor"])}getMainResultPackedPixels(){switch(this.returnType){case"LiteralInteger":case"Number":case"Integer":case"Float":return this.getMainResultKernelPackedPixels()+this.getMainResultSubKernelPackedPixels();default:throw new Error(`packed output only usable with Numbers, "${this.returnType}" specified`)}}getMainResultKernelPackedPixels(){return c.linesToString(["  threadId = indexTo3D(index, uOutputDim)","  kernel()",`  gl_FragData[0] = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(kernelResult)`])}getMainResultSubKernelPackedPixels(){const b=[];if(!this.subKernels)return"";for(let T=0;T<this.subKernels.length;T++)this.subKernels[T].returnType==="Integer"?b.push(`  gl_FragData[${T+1}] = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(float(subKernelResult_${this.subKernels[T].name}))`):b.push(`  gl_FragData[${T+1}] = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(subKernelResult_${this.subKernels[T].name})`);return c.linesToString(b)}getMainResultMemoryOptimizedFloats(){const b=["  index *= 4"];switch(this.returnType){case"Number":case"Integer":case"Float":const T=["r","g","b","a"];for(let h=0;h<T.length;h++){const F=T[h];this.getMainResultKernelMemoryOptimizedFloats(b,F),this.getMainResultSubKernelMemoryOptimizedFloats(b,F),h+1<T.length&&b.push("  index += 1")}break;default:throw new Error(`optimized output only usable with Numbers, ${this.returnType} specified`)}return c.linesToString(b)}getMainResultKernelMemoryOptimizedFloats(b,T){b.push("  threadId = indexTo3D(index, uOutputDim)","  kernel()",`  gl_FragData[0].${T} = kernelResult`)}getMainResultSubKernelMemoryOptimizedFloats(b,T){if(!this.subKernels)return b;for(let h=0;h<this.subKernels.length;h++)this.subKernels[h].returnType==="Integer"?b.push(`  gl_FragData[${h+1}].${T} = float(subKernelResult_${this.subKernels[h].name})`):b.push(`  gl_FragData[${h+1}].${T} = subKernelResult_${this.subKernels[h].name}`)}getMainResultKernelNumberTexture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  gl_FragData[0][0] = kernelResult"]}getMainResultSubKernelNumberTexture(){const b=[];if(!this.subKernels)return b;for(let T=0;T<this.subKernels.length;++T){const h=this.subKernels[T];h.returnType==="Integer"?b.push(`  gl_FragData[${T+1}][0] = float(subKernelResult_${h.name})`):b.push(`  gl_FragData[${T+1}][0] = subKernelResult_${h.name}`)}return b}getMainResultKernelArray2Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  gl_FragData[0][0] = kernelResult[0]","  gl_FragData[0][1] = kernelResult[1]"]}getMainResultSubKernelArray2Texture(){const b=[];if(!this.subKernels)return b;for(let T=0;T<this.subKernels.length;++T)b.push(`  gl_FragData[${T+1}][0] = subKernelResult_${this.subKernels[T].name}[0]`,`  gl_FragData[${T+1}][1] = subKernelResult_${this.subKernels[T].name}[1]`);return b}getMainResultKernelArray3Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  gl_FragData[0][0] = kernelResult[0]","  gl_FragData[0][1] = kernelResult[1]","  gl_FragData[0][2] = kernelResult[2]"]}getMainResultSubKernelArray3Texture(){const b=[];if(!this.subKernels)return b;for(let T=0;T<this.subKernels.length;++T)b.push(`  gl_FragData[${T+1}][0] = subKernelResult_${this.subKernels[T].name}[0]`,`  gl_FragData[${T+1}][1] = subKernelResult_${this.subKernels[T].name}[1]`,`  gl_FragData[${T+1}][2] = subKernelResult_${this.subKernels[T].name}[2]`);return b}getMainResultKernelArray4Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  gl_FragData[0] = kernelResult"]}getMainResultSubKernelArray4Texture(){const b=[];if(!this.subKernels)return b;switch(this.returnType){case"Number":case"Float":case"Integer":for(let T=0;T<this.subKernels.length;++T)this.subKernels[T].returnType==="Integer"?b.push(`  gl_FragData[${T+1}] = float(subKernelResult_${this.subKernels[T].name})`):b.push(`  gl_FragData[${T+1}] = subKernelResult_${this.subKernels[T].name}`);break;case"Array(2)":for(let T=0;T<this.subKernels.length;++T)b.push(`  gl_FragData[${T+1}][0] = subKernelResult_${this.subKernels[T].name}[0]`,`  gl_FragData[${T+1}][1] = subKernelResult_${this.subKernels[T].name}[1]`);break;case"Array(3)":for(let T=0;T<this.subKernels.length;++T)b.push(`  gl_FragData[${T+1}][0] = subKernelResult_${this.subKernels[T].name}[0]`,`  gl_FragData[${T+1}][1] = subKernelResult_${this.subKernels[T].name}[1]`,`  gl_FragData[${T+1}][2] = subKernelResult_${this.subKernels[T].name}[2]`);break;case"Array(4)":for(let T=0;T<this.subKernels.length;++T)b.push(`  gl_FragData[${T+1}][0] = subKernelResult_${this.subKernels[T].name}[0]`,`  gl_FragData[${T+1}][1] = subKernelResult_${this.subKernels[T].name}[1]`,`  gl_FragData[${T+1}][2] = subKernelResult_${this.subKernels[T].name}[2]`,`  gl_FragData[${T+1}][3] = subKernelResult_${this.subKernels[T].name}[3]`);break}return b}replaceArtifacts(b,T){return b.replace(/[ ]*__([A-Z]+[0-9]*([_]?[A-Z]*[0-9]?)*)__;\n/g,(h,F)=>{if(T.hasOwnProperty(F))return T[F];throw`unhandled artifact ${F}`})}getFragmentShader(b){return this.compiledFragmentShader!==null?this.compiledFragmentShader:this.compiledFragmentShader=this.replaceArtifacts(this.constructor.fragmentShader,this._getFragShaderArtifactMap(b))}getVertexShader(b){return this.compiledVertexShader!==null?this.compiledVertexShader:this.compiledVertexShader=this.replaceArtifacts(this.constructor.vertexShader,this._getVertShaderArtifactMap(b))}toString(){const b=c.linesToString(["const gl = context"]);return P(this.constructor,arguments,this,b)}destroy(b){if(!this.context)return;this.buffer&&this.context.deleteBuffer(this.buffer),this.framebuffer&&this.context.deleteFramebuffer(this.framebuffer);for(const h in this.rawValueFramebuffers){for(const F in this.rawValueFramebuffers[h])this.context.deleteFramebuffer(this.rawValueFramebuffers[h][F]),delete this.rawValueFramebuffers[h][F];delete this.rawValueFramebuffers[h]}if(this.vertShader&&this.context.deleteShader(this.vertShader),this.fragShader&&this.context.deleteShader(this.fragShader),this.program&&this.context.deleteProgram(this.program),this.texture){this.texture.delete();const h=this.textureCache.indexOf(this.texture.texture);h>-1&&this.textureCache.splice(h,1),this.texture=null}if(this.mappedTextures&&this.mappedTextures.length){for(let h=0;h<this.mappedTextures.length;h++){const F=this.mappedTextures[h];F.delete();const O=this.textureCache.indexOf(F.texture);O>-1&&this.textureCache.splice(O,1)}this.mappedTextures=null}if(this.kernelArguments)for(let h=0;h<this.kernelArguments.length;h++)this.kernelArguments[h].destroy();if(this.kernelConstants)for(let h=0;h<this.kernelConstants.length;h++)this.kernelConstants[h].destroy();for(;this.textureCache.length>0;){const h=this.textureCache.pop();this.context.deleteTexture(h)}if(b){const h=w.indexOf(this.canvas);h>=0&&(w[h]=null,v[h]=null)}if(this.destroyExtensions(),delete this.context,delete this.canvas,!this.gpu)return;const T=this.gpu.kernels.indexOf(this);T!==-1&&this.gpu.kernels.splice(T,1)}destroyExtensions(){this.extensions.OES_texture_float=null,this.extensions.OES_texture_float_linear=null,this.extensions.OES_element_index_uint=null,this.extensions.WEBGL_draw_buffers=null}static destroyContext(b){const T=b.getExtension("WEBGL_lose_context");T&&T.loseContext()}toJSON(){const b=super.toJSON();return b.functionNodes=I.fromKernel(this,_).toJSON(),b.settings.threadDim=this.threadDim,b}};G.exports={WebGLKernel:C}}),jr=s((j,G)=>{const D=cs(),{WebGLKernel:I}=Fs(),{glKernelString:_}=Pr();let c=null,d=null,E=null,$=null,P=null;var y=class extends I{static get isSupported(){return c!==null||(this.setupFeatureChecks(),c=E!==null),c}static setupFeatureChecks(){if(d=null,$=null,typeof D=="function")try{if(E=D(2,2,{preserveDrawingBuffer:!0}),!E||!E.getExtension)return;$={STACKGL_resize_drawingbuffer:E.getExtension("STACKGL_resize_drawingbuffer"),STACKGL_destroy_context:E.getExtension("STACKGL_destroy_context"),OES_texture_float:E.getExtension("OES_texture_float"),OES_texture_float_linear:E.getExtension("OES_texture_float_linear"),OES_element_index_uint:E.getExtension("OES_element_index_uint"),WEBGL_draw_buffers:E.getExtension("WEBGL_draw_buffers"),WEBGL_color_buffer_float:E.getExtension("WEBGL_color_buffer_float")},P=this.getFeatures()}catch(p){console.warn(p)}}static isContextMatch(p){try{return p.getParameter(p.RENDERER)==="ANGLE"}catch{return!1}}static getIsTextureFloat(){return!!$.OES_texture_float}static getIsDrawBuffers(){return!!$.WEBGL_draw_buffers}static getChannelCount(){return $.WEBGL_draw_buffers?E.getParameter($.WEBGL_draw_buffers.MAX_DRAW_BUFFERS_WEBGL):1}static getMaxTextureSize(){return E.getParameter(E.MAX_TEXTURE_SIZE)}static get testCanvas(){return d}static get testContext(){return E}static get features(){return P}initCanvas(){return{}}initContext(){return D(2,2,{preserveDrawingBuffer:!0})}initExtensions(){this.extensions={STACKGL_resize_drawingbuffer:this.context.getExtension("STACKGL_resize_drawingbuffer"),STACKGL_destroy_context:this.context.getExtension("STACKGL_destroy_context"),OES_texture_float:this.context.getExtension("OES_texture_float"),OES_texture_float_linear:this.context.getExtension("OES_texture_float_linear"),OES_element_index_uint:this.context.getExtension("OES_element_index_uint"),WEBGL_draw_buffers:this.context.getExtension("WEBGL_draw_buffers")}}build(){super.build.apply(this,arguments),this.fallbackRequested||this.extensions.STACKGL_resize_drawingbuffer.resize(this.maxTexSize[0],this.maxTexSize[1])}destroyExtensions(){this.extensions.STACKGL_resize_drawingbuffer=null,this.extensions.STACKGL_destroy_context=null,this.extensions.OES_texture_float=null,this.extensions.OES_texture_float_linear=null,this.extensions.OES_element_index_uint=null,this.extensions.WEBGL_draw_buffers=null}static destroyContext(p){const g=p.getExtension("STACKGL_destroy_context");g&&g.destroy&&g.destroy()}toString(){return _(this.constructor,arguments,this,`const gl = context || require('gl')(1, 1);
`,`    if (!context) { gl.getExtension('STACKGL_destroy_context').destroy(); }
`)}setOutput(p){return super.setOutput(p),this.graphical&&this.extensions.STACKGL_resize_drawingbuffer&&this.extensions.STACKGL_resize_drawingbuffer.resize(this.maxTexSize[0],this.maxTexSize[1]),this}};G.exports={HeadlessGLKernel:y}}),qr=s((j,G)=>{const{utils:D}=m(),{WebGLFunctionNode:I}=yn();var _=class extends I{astIdentifierExpression(c,d){if(c.type!=="Identifier")throw this.astErrorOutput("IdentifierExpression - not an Identifier",c);const E=this.getType(c),$=D.sanitizeName(c.name);return c.name==="Infinity"?d.push("intBitsToFloat(2139095039)"):E==="Boolean"?this.argumentNames.indexOf($)>-1?d.push(`bool(user_${$})`):d.push(`user_${$}`):d.push(`user_${$}`),d}};G.exports={WebGL2FunctionNode:_}}),bo=s((j,G)=>{G.exports={fragmentShader:`#version 300 es
__HEADER__;
__FLOAT_TACTIC_DECLARATION__;
__INT_TACTIC_DECLARATION__;
__SAMPLER_2D_TACTIC_DECLARATION__;
__SAMPLER_2D_ARRAY_TACTIC_DECLARATION__;

const int LOOP_MAX = __LOOP_MAX__;

__PLUGINS__;
__CONSTANTS__;

in vec2 vTexCoord;

float atan2(float v1, float v2) {
  if (v2 == 0.0) {
    if (v1 == 0.0) return 0.0;
    if (v1 > 0.0) return 1.5707963267948966;
    if (v1 < 0.0) return -1.5707963267948966;
  }
  return atan(v1, v2);
}

float cbrt(float x) {
  if (x >= 0.0) {
    return pow(x, 1.0 / 3.0);
  } else {
    return -pow(x, 1.0 / 3.0);
  }
}

float expm1(float x) {
  return pow(${Math.E}, x) - 1.0; 
}

float fround(highp float x) {
  return x;
}

float imul(float v1, float v2) {
  return float(int(v1) * int(v2));
}

float log10(float x) {
  return log2(x) * (1.0 / log2(10.0));
}

float log1p(float x) {
  return log(1.0 + x);
}

float _pow(float v1, float v2) {
  if (v2 == 0.0) return 1.0;
  return pow(v1, v2);
}

float _round(float x) {
  return floor(x + 0.5);
}


const int BIT_COUNT = 32;
int modi(int x, int y) {
  return x - y * (x / y);
}

int bitwiseOr(int a, int b) {
  int result = 0;
  int n = 1;
  
  for (int i = 0; i < BIT_COUNT; i++) {
    if ((modi(a, 2) == 1) || (modi(b, 2) == 1)) {
      result += n;
    }
    a = a / 2;
    b = b / 2;
    n = n * 2;
    if(!(a > 0 || b > 0)) {
      break;
    }
  }
  return result;
}
int bitwiseXOR(int a, int b) {
  int result = 0;
  int n = 1;
  
  for (int i = 0; i < BIT_COUNT; i++) {
    if ((modi(a, 2) == 1) != (modi(b, 2) == 1)) {
      result += n;
    }
    a = a / 2;
    b = b / 2;
    n = n * 2;
    if(!(a > 0 || b > 0)) {
      break;
    }
  }
  return result;
}
int bitwiseAnd(int a, int b) {
  int result = 0;
  int n = 1;
  for (int i = 0; i < BIT_COUNT; i++) {
    if ((modi(a, 2) == 1) && (modi(b, 2) == 1)) {
      result += n;
    }
    a = a / 2;
    b = b / 2;
    n = n * 2;
    if(!(a > 0 && b > 0)) {
      break;
    }
  }
  return result;
}
int bitwiseNot(int a) {
  // ~a is identically -a - 1 in two's complement, for every value including
  // negatives. The previous bit-by-bit loop only worked for a >= 0, where it
  // leaned on 32-bit overflow wrapping to reach the negative answer; given a
  // negative input it computed ~abs(a), so ~(-1) gave -2 and ~~x never
  // returned x.
  return -a - 1;
}
int bitwiseZeroFillLeftShift(int n, int shift) {
  int maxBytes = BIT_COUNT;
  for (int i = 0; i < BIT_COUNT; i++) {
    if (maxBytes >= n) {
      break;
    }
    maxBytes *= 2;
  }
  for (int i = 0; i < BIT_COUNT; i++) {
    if (i >= shift) {
      break;
    }
    n *= 2;
  }

  int result = 0;
  int byteVal = 1;
  for (int i = 0; i < BIT_COUNT; i++) {
    if (i >= maxBytes) break;
    if (modi(n, 2) > 0) { result += byteVal; }
    n = int(n / 2);
    byteVal *= 2;
  }
  return result;
}

// _pow2 is defined further down, alongside encode32/decode32
float _pow2(float e);
int bitwiseSignedRightShift(int num, int shifts) {
  // pow(2.0, n) is approximate on many GPUs, and landing 1 ulp high makes the
  // division fall just under a whole number, which floor() then rounds away:
  // 2 >> 1 came out 0, 8 >> 1 came out 3. Only exact left operands were
  // affected, odd ones having enough slack to survive. _pow2 is exact.
  return int(floor(float(num) / _pow2(float(shifts))));
}

int bitwiseZeroFillRightShift(int n, int shift) {
  int maxBytes = BIT_COUNT;
  for (int i = 0; i < BIT_COUNT; i++) {
    if (maxBytes >= n) {
      break;
    }
    maxBytes *= 2;
  }
  for (int i = 0; i < BIT_COUNT; i++) {
    if (i >= shift) {
      break;
    }
    n /= 2;
  }
  int result = 0;
  int byteVal = 1;
  for (int i = 0; i < BIT_COUNT; i++) {
    if (i >= maxBytes) break;
    if (modi(n, 2) > 0) { result += byteVal; }
    n = int(n / 2);
    byteVal *= 2;
  }
  return result;
}

vec2 integerMod(vec2 x, float y) {
  vec2 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}

vec3 integerMod(vec3 x, float y) {
  vec3 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}

vec4 integerMod(vec4 x, vec4 y) {
  vec4 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}

float integerMod(float x, float y) {
  float res = floor(mod(x, y));
  return res * (res > floor(y) - 1.0 ? 0.0 : 1.0);
}

int integerMod(int x, int y) {
  return x - (y * int(x/y));
}

// GLSL ES 1.00 accepts only a constant or a loop symbol inside an index
// expression, so m[y][x] does not compile when y and x come from kernel
// arguments -- the error is "Index expression can only contain const or loop
// symbols". Loop counters are legal indices, so walk the matrix with them
// instead. These are 2x2 to 4x4, so it costs at most sixteen comparisons.
float getMatrix2(mat2 m, int y, int x) {
  float result = 0.0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == y && j == x) result = m[i][j];
    }
  }
  return result;
}

float getMatrix3(mat3 m, int y, int x) {
  float result = 0.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i == y && j == x) result = m[i][j];
    }
  }
  return result;
}

float getMatrix4(mat4 m, int y, int x) {
  float result = 0.0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == y && j == x) result = m[i][j];
    }
  }
  return result;
}

__DIVIDE_WITH_INTEGER_CHECK__;

// Here be dragons!
// DO NOT OPTIMIZE THIS CODE
// YOU WILL BREAK SOMETHING ON SOMEBODY'S MACHINE
// LEAVE IT AS IT IS, LEST YOU WASTE YOUR OWN TIME
// Exact powers of two built from exact constant multiplies: exp2/log2/pow
// are approximate on some GPUs (notably Apple silicon), and 1-2 ulp there
// corrupts the packed bytes (#659)
float _pow2(float e) {
  float r = 1.0;
  float a = abs(e);
  bool n = e < 0.0;
  if (a >= 64.0) { r *= n ? 5.421010862427522e-20 : 18446744073709551616.0; a -= 64.0; }
  if (a >= 64.0) { r *= n ? 5.421010862427522e-20 : 18446744073709551616.0; a -= 64.0; }
  if (a >= 32.0) { r *= n ? 2.3283064365386963e-10 : 4294967296.0; a -= 32.0; }
  if (a >= 16.0) { r *= n ? 0.0000152587890625 : 65536.0; a -= 16.0; }
  if (a >= 8.0) { r *= n ? 0.00390625 : 256.0; a -= 8.0; }
  if (a >= 4.0) { r *= n ? 0.0625 : 16.0; a -= 4.0; }
  if (a >= 2.0) { r *= n ? 0.25 : 4.0; a -= 2.0; }
  if (a >= 1.0) { r *= n ? 0.5 : 2.0; }
  return r;
}
const vec2 MAGIC_VEC = vec2(1.0, -256.0);
const vec4 SCALE_FACTOR = vec4(1.0, 256.0, 65536.0, 0.0);
const vec4 SCALE_FACTOR_INV = vec4(1.0, 0.00390625, 0.0000152587890625, 0.0); // 1, 1/256, 1/65536
float decode32(vec4 texel) {
  __DECODE32_ENDIANNESS__;
  texel *= 255.0;
  vec2 gte128;
  gte128.x = texel.b >= 128.0 ? 1.0 : 0.0;
  gte128.y = texel.a >= 128.0 ? 1.0 : 0.0;
  float exponent = 2.0 * texel.a - 127.0 + dot(gte128, MAGIC_VEC);
  float res = _pow2(round(exponent));
  texel.b = texel.b - 128.0 * gte128.x;
  res = dot(texel, SCALE_FACTOR) * _pow2(round(exponent-23.0)) + res;
  res *= gte128.y * -2.0 + 1.0;
  return res;
}

float decode16(vec4 texel, int index) {
  int channel = integerMod(index, 2);
  return texel[channel*2] * 255.0 + texel[channel*2 + 1] * 65280.0;
}

float decode8(vec4 texel, int index) {
  int channel = integerMod(index, 4);
  return texel[channel] * 255.0;
}

vec4 legacyEncode32(float f) {
  float F = abs(f);
  float sign = f < 0.0 ? 1.0 : 0.0;
  float exponent = floor(log2(F));
  float mantissa = (exp2(-exponent) * F);
  // exponent += floor(log2(mantissa));
  vec4 texel = vec4(F * exp2(23.0-exponent)) * SCALE_FACTOR_INV;
  texel.rg = integerMod(texel.rg, 256.0);
  texel.b = integerMod(texel.b, 128.0);
  texel.a = exponent*0.5 + 63.5;
  texel.ba += vec2(integerMod(exponent+127.0, 2.0), sign) * 128.0;
  texel = floor(texel);
  texel *= 0.003921569; // 1/255
  __ENCODE32_ENDIANNESS__;
  return texel;
}

// https://github.com/gpujs/gpu.js/wiki/Encoder-details
vec4 encode32(float value) {
  if (value == 0.0) return vec4(0, 0, 0, 0);

  float exponent;
  float mantissa;
  vec4  result;
  float sgn;

  sgn = step(0.0, -value);
  value = abs(value);

  exponent = floor(log2(value));
  float p2 = _pow2(exponent);
  // approximate log2 can land one off; correct by direct comparison
  if (p2 > value) { exponent -= 1.0; p2 *= 0.5; }
  else if (p2 * 2.0 <= value) { exponent += 1.0; p2 *= 2.0; }

  mantissa = value / p2 - 1.0;
  exponent = exponent+127.0;
  result   = vec4(0,0,0,0);

  result.a = floor(exponent/2.0);
  exponent = exponent - result.a*2.0;
  result.a = result.a + 128.0*sgn;

  result.b = floor(mantissa * 128.0);
  mantissa = mantissa - result.b / 128.0;
  result.b = result.b + exponent*128.0;

  result.g = floor(mantissa*32768.0);
  mantissa = mantissa - result.g/32768.0;

  result.r = floor(mantissa*8388608.0);
  return result/255.0;
}
// Dragons end here

int index;
ivec3 threadId;

ivec3 indexTo3D(int idx, ivec3 texDim) {
  int z = int(idx / (texDim.x * texDim.y));
  idx -= z * int(texDim.x * texDim.y);
  int y = int(idx / texDim.x);
  int x = int(integerMod(idx, texDim.x));
  return ivec3(x, y, z);
}

float get32(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture(tex, st / vec2(texSize));
  return decode32(texel);
}

float get16(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + (texDim.x * (y + (texDim.y * z)));
  int w = texSize.x * 2;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture(tex, st / vec2(texSize.x * 2, texSize.y));
  return decode16(texel, index);
}

float get8(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + (texDim.x * (y + (texDim.y * z)));
  int w = texSize.x * 4;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture(tex, st / vec2(texSize.x * 4, texSize.y));
  return decode8(texel, index);
}

float getMemoryOptimized32(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + (texDim.x * (y + (texDim.y * z)));
  int channel = integerMod(index, 4);
  index = index / 4;
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  index = index / 4;
  vec4 texel = texture(tex, st / vec2(texSize));
  return texel[channel];
}

vec4 getImage2D(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  return texture(tex, st / vec2(texSize));
}

vec4 getImage3D(sampler2DArray tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  return texture(tex, vec3(st / vec2(texSize), z));
}

float getFloatFromSampler2D(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  vec4 result = getImage2D(tex, texSize, texDim, z, y, x);
  return result[0];
}

vec2 getVec2FromSampler2D(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  vec4 result = getImage2D(tex, texSize, texDim, z, y, x);
  return vec2(result[0], result[1]);
}

vec2 getMemoryOptimizedVec2(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int channel = integerMod(index, 2);
  index = index / 2;
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture(tex, st / vec2(texSize));
  if (channel == 0) return vec2(texel.r, texel.g);
  if (channel == 1) return vec2(texel.b, texel.a);
  return vec2(0.0, 0.0);
}

vec3 getVec3FromSampler2D(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  vec4 result = getImage2D(tex, texSize, texDim, z, y, x);
  return vec3(result[0], result[1], result[2]);
}

vec3 getMemoryOptimizedVec3(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int fieldIndex = 3 * (x + texDim.x * (y + texDim.y * z));
  int vectorIndex = fieldIndex / 4;
  int vectorOffset = fieldIndex - vectorIndex * 4;
  int readY = vectorIndex / texSize.x;
  int readX = vectorIndex - readY * texSize.x;
  vec4 tex1 = texture(tex, (vec2(readX, readY) + 0.5) / vec2(texSize));

  if (vectorOffset == 0) {
    return tex1.xyz;
  } else if (vectorOffset == 1) {
    return tex1.yzw;
  } else {
    readX++;
    if (readX >= texSize.x) {
      readX = 0;
      readY++;
    }
    vec4 tex2 = texture(tex, vec2(readX, readY) / vec2(texSize));
    if (vectorOffset == 2) {
      return vec3(tex1.z, tex1.w, tex2.x);
    } else {
      return vec3(tex1.w, tex2.x, tex2.y);
    }
  }
}

vec4 getVec4FromSampler2D(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  return getImage2D(tex, texSize, texDim, z, y, x);
}

vec4 getMemoryOptimizedVec4(sampler2D tex, ivec2 texSize, ivec3 texDim, int z, int y, int x) {
  int index = x + texDim.x * (y + texDim.y * z);
  int channel = integerMod(index, 2);
  int w = texSize.x;
  vec2 st = vec2(float(integerMod(index, w)), float(index / w)) + 0.5;
  vec4 texel = texture(tex, st / vec2(texSize));
  return vec4(texel.r, texel.g, texel.b, texel.a);
}

vec4 actualColor;
void color(float r, float g, float b, float a) {
  actualColor = vec4(r,g,b,a);
}

void color(float r, float g, float b) {
  color(r,g,b,1.0);
}

float modulo(float number, float divisor) {
  if (number < 0.0) {
    number = abs(number);
    if (divisor < 0.0) {
      divisor = abs(divisor);
    }
    return -mod(number, divisor);
  }
  if (divisor < 0.0) {
    divisor = abs(divisor);
  }
  return mod(number, divisor);
}

__INJECTED_NATIVE__;
__MAIN_CONSTANTS__;
__MAIN_ARGUMENTS__;
__KERNEL__;

void main(void) {
  index = int(vTexCoord.s * float(uTexSize.x)) + int(vTexCoord.t * float(uTexSize.y)) * uTexSize.x;
  __MAIN_RESULT__;
}`}}),wo=s((j,G)=>{G.exports={vertexShader:`#version 300 es
__FLOAT_TACTIC_DECLARATION__;
__INT_TACTIC_DECLARATION__;
__SAMPLER_2D_TACTIC_DECLARATION__;

in vec2 aPos;
in vec2 aTexCoord;

out vec2 vTexCoord;
uniform vec2 ratio;

void main(void) {
  gl_Position = vec4((aPos + vec2(1)) * ratio + vec2(-1), 0, 1);
  vTexCoord = aTexCoord;
}`}}),vo=s((j,G)=>{const{WebGLKernelValueBoolean:D}=zr();var I=class extends D{};G.exports={WebGL2KernelValueBoolean:I}}),ko=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueFloat:I}=Or();var _=class extends I{};G.exports={WebGL2KernelValueFloat:_}}),To=s((j,G)=>{const{WebGLKernelValueInteger:D}=Rr();var I=class extends D{getSource(_){const c=this.getVariablePrecisionString();return this.origin==="constants"?`const ${c} int ${this.id} = ${parseInt(_)};
`:`uniform ${c} int ${this.id};
`}updateValue(_){this.origin!=="constants"&&this.kernel.setUniform1i(this.id,this.uploadValue=_)}};G.exports={WebGL2KernelValueInteger:I}}),Wr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueHTMLImage:I}=Os();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`${c} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${c} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}};G.exports={WebGL2KernelValueHTMLImage:_}}),Hr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueDynamicHTMLImage:I}=xn();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`uniform ${c} ivec2 ${this.sizeId}`,`uniform ${c} ivec3 ${this.dimensionsId}`])}};G.exports={WebGL2KernelValueDynamicHTMLImage:_}}),Xr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelArray:I}=Qe();var _=class extends I{constructor(c,d){super(c,d),this.checkSize(c[0].width,c[0].height),this.dimensions=[c[0].width,c[0].height,c.length],this.textureSize=[c[0].width,c[0].height]}defineTexture(){const{context:c}=this;c.activeTexture(this.contextHandle),c.bindTexture(c.TEXTURE_2D_ARRAY,this.texture),c.texParameteri(c.TEXTURE_2D_ARRAY,c.TEXTURE_MAG_FILTER,c.NEAREST),c.texParameteri(c.TEXTURE_2D_ARRAY,c.TEXTURE_MIN_FILTER,c.NEAREST)}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName};
`}getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2DArray ${this.id}`,`${c} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${c} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(c){const{context:d}=this;d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D_ARRAY,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!0),d.texImage3D(d.TEXTURE_2D_ARRAY,0,d.RGBA,c[0].width,c[0].height,c.length,0,d.RGBA,d.UNSIGNED_BYTE,null);for(let E=0;E<c.length;E++)d.texSubImage3D(d.TEXTURE_2D_ARRAY,0,0,0,E,c[E].width,c[E].height,1,d.RGBA,d.UNSIGNED_BYTE,this.uploadValue=c[E]);this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGL2KernelValueHTMLImageArray:_}}),So=s((j,G)=>{const{utils:D}=m(),{WebGL2KernelValueHTMLImageArray:I}=Xr();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2DArray ${this.id}`,`uniform ${c} ivec2 ${this.sizeId}`,`uniform ${c} ivec3 ${this.dimensionsId}`])}updateValue(c){const{width:d,height:E}=c[0];this.checkSize(d,E),this.dimensions=[d,E,c.length],this.textureSize=[d,E],this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGL2KernelValueDynamicHTMLImageArray:_}}),_o=s((j,G)=>{const{utils:D}=m(),{WebGL2KernelValueHTMLImage:I}=Wr();var _=class extends I{};G.exports={WebGL2KernelValueHTMLVideo:_}}),Co=s((j,G)=>{const{utils:D}=m(),{WebGL2KernelValueDynamicHTMLImage:I}=Hr();var _=class extends I{};G.exports={WebGL2KernelValueDynamicHTMLVideo:_}}),Yr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueSingleInput:I}=bn();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`${c} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${c} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(c){const{context:d}=this;D.flattenTo(c.value,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA32F,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGL2KernelValueSingleInput:_}}),Eo=s((j,G)=>{const{utils:D}=m(),{WebGL2KernelValueSingleInput:I}=Yr();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`uniform ${c} ivec2 ${this.sizeId}`,`uniform ${c} ivec3 ${this.dimensionsId}`])}updateValue(c){let[d,E,$]=c.size;this.dimensions=new Int32Array([d||1,E||1,$||1]),this.textureSize=D.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGL2KernelValueDynamicSingleInput:_}}),Io=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueUnsignedInput:I}=wn();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`${c} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${c} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}};G.exports={WebGL2KernelValueUnsignedInput:_}}),Mo=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueDynamicUnsignedInput:I}=Fr();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`uniform ${c} ivec2 ${this.sizeId}`,`uniform ${c} ivec3 ${this.dimensionsId}`])}};G.exports={WebGL2KernelValueDynamicUnsignedInput:_}}),$o=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueMemoryOptimizedNumberTexture:I}=Rs();var _=class extends I{getSource(){const{id:c,sizeId:d,textureSize:E,dimensionsId:$,dimensions:P}=this,y=this.getVariablePrecisionString();return D.linesToString([`uniform sampler2D ${c}`,`${y} ivec2 ${d} = ivec2(${E[0]}, ${E[1]})`,`${y} ivec3 ${$} = ivec3(${P[0]}, ${P[1]}, ${P[2]})`])}};G.exports={WebGL2KernelValueMemoryOptimizedNumberTexture:_}}),Ao=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueDynamicMemoryOptimizedNumberTexture:I}=Gr();var _=class extends I{getSource(){return D.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}};G.exports={WebGL2KernelValueDynamicMemoryOptimizedNumberTexture:_}}),Do=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueNumberTexture:I}=vn();var _=class extends I{getSource(){const{id:c,sizeId:d,textureSize:E,dimensionsId:$,dimensions:P}=this,y=this.getVariablePrecisionString();return D.linesToString([`uniform ${y} sampler2D ${c}`,`${y} ivec2 ${d} = ivec2(${E[0]}, ${E[1]})`,`${y} ivec3 ${$} = ivec3(${P[0]}, ${P[1]}, ${P[2]})`])}};G.exports={WebGL2KernelValueNumberTexture:_}}),Po=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueDynamicNumberTexture:I}=Lr();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`uniform ${c} ivec2 ${this.sizeId}`,`uniform ${c} ivec3 ${this.dimensionsId}`])}};G.exports={WebGL2KernelValueDynamicNumberTexture:_}}),Jr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueSingleArray:I}=kn();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`${c} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${c} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(c.constructor);return}const{context:d}=this;D.flattenTo(c,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA32F,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGL2KernelValueSingleArray:_}}),zo=s((j,G)=>{const{utils:D}=m(),{WebGL2KernelValueSingleArray:I}=Jr();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`uniform ${c} ivec2 ${this.sizeId}`,`uniform ${c} ivec3 ${this.dimensionsId}`])}updateValue(c){this.dimensions=D.getDimensions(c,!0),this.textureSize=D.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGL2KernelValueDynamicSingleArray:_}}),Zr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueSingleArray1DI:I}=Tn();var _=class extends I{updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(c.constructor);return}const{context:d}=this;D.flattenTo(c,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA32F,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGL2KernelValueSingleArray1DI:_}}),Oo=s((j,G)=>{const{utils:D}=m(),{WebGL2KernelValueSingleArray1DI:I}=Zr();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`uniform ${c} ivec2 ${this.sizeId}`,`uniform ${c} ivec3 ${this.dimensionsId}`])}updateValue(c){this.setShape(c),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGL2KernelValueDynamicSingleArray1DI:_}}),Qr=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueSingleArray2DI:I}=Sn();var _=class extends I{updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(c.constructor);return}const{context:d}=this;D.flattenTo(c,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA32F,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGL2KernelValueSingleArray2DI:_}}),Ro=s((j,G)=>{const{utils:D}=m(),{WebGL2KernelValueSingleArray2DI:I}=Qr();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`uniform ${c} ivec2 ${this.sizeId}`,`uniform ${c} ivec3 ${this.dimensionsId}`])}updateValue(c){this.setShape(c),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGL2KernelValueDynamicSingleArray2DI:_}}),ei=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueSingleArray3DI:I}=_n();var _=class extends I{updateValue(c){if(c.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(c.constructor);return}const{context:d}=this;D.flattenTo(c,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA32F,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};G.exports={WebGL2KernelValueSingleArray3DI:_}}),Fo=s((j,G)=>{const{utils:D}=m(),{WebGL2KernelValueSingleArray3DI:I}=ei();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`uniform ${c} ivec2 ${this.sizeId}`,`uniform ${c} ivec3 ${this.dimensionsId}`])}updateValue(c){this.setShape(c),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(c)}};G.exports={WebGL2KernelValueDynamicSingleArray3DI:_}}),Go=s((j,G)=>{const{WebGLKernelValueArray2:D}=Ur();var I=class extends D{};G.exports={WebGL2KernelValueArray2:I}}),Lo=s((j,G)=>{const{WebGLKernelValueArray3:D}=Vr();var I=class extends D{};G.exports={WebGL2KernelValueArray3:I}}),Uo=s((j,G)=>{const{WebGLKernelValueArray4:D}=Kr();var I=class extends D{};G.exports={WebGL2KernelValueArray4:I}}),Vo=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueUnsignedArray:I}=Cn();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`${c} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${c} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}};G.exports={WebGL2KernelValueUnsignedArray:_}}),Ko=s((j,G)=>{const{utils:D}=m(),{WebGLKernelValueDynamicUnsignedArray:I}=Nr();var _=class extends I{getSource(){const c=this.getVariablePrecisionString();return D.linesToString([`uniform ${c} sampler2D ${this.id}`,`uniform ${c} ivec2 ${this.sizeId}`,`uniform ${c} ivec3 ${this.dimensionsId}`])}};G.exports={WebGL2KernelValueDynamicUnsignedArray:_}}),ti=s((j,G)=>{const{WebGL2KernelValueBoolean:D}=vo(),{WebGL2KernelValueFloat:I}=ko(),{WebGL2KernelValueInteger:_}=To(),{WebGL2KernelValueHTMLImage:c}=Wr(),{WebGL2KernelValueDynamicHTMLImage:d}=Hr(),{WebGL2KernelValueHTMLImageArray:E}=Xr(),{WebGL2KernelValueDynamicHTMLImageArray:$}=So(),{WebGL2KernelValueHTMLVideo:P}=_o(),{WebGL2KernelValueDynamicHTMLVideo:y}=Co(),{WebGL2KernelValueSingleInput:p}=Yr(),{WebGL2KernelValueDynamicSingleInput:g}=Eo(),{WebGL2KernelValueUnsignedInput:k}=Io(),{WebGL2KernelValueDynamicUnsignedInput:o}=Mo(),{WebGL2KernelValueMemoryOptimizedNumberTexture:l}=$o(),{WebGL2KernelValueDynamicMemoryOptimizedNumberTexture:x}=Ao(),{WebGL2KernelValueNumberTexture:w}=Do(),{WebGL2KernelValueDynamicNumberTexture:v}=Po(),{WebGL2KernelValueSingleArray:C}=Jr(),{WebGL2KernelValueDynamicSingleArray:b}=zo(),{WebGL2KernelValueSingleArray1DI:T}=Zr(),{WebGL2KernelValueDynamicSingleArray1DI:h}=Oo(),{WebGL2KernelValueSingleArray2DI:F}=Qr(),{WebGL2KernelValueDynamicSingleArray2DI:O}=Ro(),{WebGL2KernelValueSingleArray3DI:z}=ei(),{WebGL2KernelValueDynamicSingleArray3DI:L}=Fo(),{WebGL2KernelValueArray2:V}=Go(),{WebGL2KernelValueArray3:U}=Lo(),{WebGL2KernelValueArray4:X}=Uo(),{WebGL2KernelValueUnsignedArray:q}=Vo(),{WebGL2KernelValueDynamicUnsignedArray:W}=Ko(),ee={unsigned:{dynamic:{Boolean:D,Integer:_,Float:I,Array:W,"Array(2)":V,"Array(3)":U,"Array(4)":X,"Array1D(2)":!1,"Array1D(3)":!1,"Array1D(4)":!1,"Array2D(2)":!1,"Array2D(3)":!1,"Array2D(4)":!1,"Array3D(2)":!1,"Array3D(3)":!1,"Array3D(4)":!1,Input:o,NumberTexture:v,"ArrayTexture(1)":v,"ArrayTexture(2)":v,"ArrayTexture(3)":v,"ArrayTexture(4)":v,MemoryOptimizedNumberTexture:x,HTMLCanvas:d,OffscreenCanvas:d,HTMLImage:d,ImageBitmap:d,ImageData:d,HTMLImageArray:$,HTMLVideo:y},static:{Boolean:D,Float:I,Integer:_,Array:q,"Array(2)":V,"Array(3)":U,"Array(4)":X,"Array1D(2)":!1,"Array1D(3)":!1,"Array1D(4)":!1,"Array2D(2)":!1,"Array2D(3)":!1,"Array2D(4)":!1,"Array3D(2)":!1,"Array3D(3)":!1,"Array3D(4)":!1,Input:k,NumberTexture:w,"ArrayTexture(1)":w,"ArrayTexture(2)":w,"ArrayTexture(3)":w,"ArrayTexture(4)":w,MemoryOptimizedNumberTexture:x,HTMLCanvas:c,OffscreenCanvas:c,HTMLImage:c,ImageBitmap:c,ImageData:c,HTMLImageArray:E,HTMLVideo:P}},single:{dynamic:{Boolean:D,Integer:_,Float:I,Array:b,"Array(2)":V,"Array(3)":U,"Array(4)":X,"Array1D(2)":h,"Array1D(3)":h,"Array1D(4)":h,"Array2D(2)":O,"Array2D(3)":O,"Array2D(4)":O,"Array3D(2)":L,"Array3D(3)":L,"Array3D(4)":L,Input:g,NumberTexture:v,"ArrayTexture(1)":v,"ArrayTexture(2)":v,"ArrayTexture(3)":v,"ArrayTexture(4)":v,MemoryOptimizedNumberTexture:x,HTMLCanvas:d,OffscreenCanvas:d,HTMLImage:d,ImageBitmap:d,ImageData:d,HTMLImageArray:$,HTMLVideo:y},static:{Boolean:D,Float:I,Integer:_,Array:C,"Array(2)":V,"Array(3)":U,"Array(4)":X,"Array1D(2)":T,"Array1D(3)":T,"Array1D(4)":T,"Array2D(2)":F,"Array2D(3)":F,"Array2D(4)":F,"Array3D(2)":z,"Array3D(3)":z,"Array3D(4)":z,Input:p,NumberTexture:w,"ArrayTexture(1)":w,"ArrayTexture(2)":w,"ArrayTexture(3)":w,"ArrayTexture(4)":w,MemoryOptimizedNumberTexture:l,HTMLCanvas:c,OffscreenCanvas:c,HTMLImage:c,ImageBitmap:c,ImageData:c,HTMLImageArray:E,HTMLVideo:P}}};function se(Z,ie,he,we){if(!Z)throw new Error("type missing");if(!ie)throw new Error("dynamic missing");if(!he)throw new Error("precision missing");we.type&&(Z=we.type);const re=ee[he][ie];if(Z==="WebGPUBuffer")throw new Error("this kernel runs on WebGL but received a WebGPU pipeline buffer; await handle.toArray() first, or give this kernel the async contract (asyncMode: true / mode: 'async') so the readback happens for you");if(re[Z]===!1)return null;if(re[Z]===void 0)throw new Error(`Could not find a KernelValue for ${Z}`);return re[Z]}G.exports={kernelValueMaps:ee,lookupKernelValueType:se}}),si=s((j,G)=>{const{WebGLKernel:D}=Fs(),{WebGL2FunctionNode:I}=qr(),{FunctionBuilder:_}=N(),{utils:c}=m(),{fragmentShader:d}=bo(),{vertexShader:E}=wo(),{lookupKernelValueType:$}=ti();let P=null,y=null,p=null,g=null;var k=class extends D{static get isSupported(){return P!==null||(this.setupFeatureChecks(),P=this.isContextMatch(p)),P}static setupFeatureChecks(){typeof document<"u"?y=document.createElement("canvas"):typeof OffscreenCanvas<"u"&&(y=new OffscreenCanvas(0,0)),y&&(p=y.getContext("webgl2"),!(!p||!p.getExtension)&&(p.getExtension("EXT_color_buffer_float"),p.getExtension("OES_texture_float_linear"),g=this.getFeatures()))}static isContextMatch(o){return typeof WebGL2RenderingContext<"u"?o instanceof WebGL2RenderingContext:!1}static getFeatures(){const o=this.testContext;return Object.freeze({isFloatRead:this.getIsFloatRead(),isIntegerDivisionAccurate:this.getIsIntegerDivisionAccurate(),isSpeedTacticSupported:this.getIsSpeedTacticSupported(),kernelMap:!0,isTextureFloat:!0,isDrawBuffers:!0,channelCount:this.getChannelCount(),maxTextureSize:this.getMaxTextureSize(),lowIntPrecision:o.getShaderPrecisionFormat(o.FRAGMENT_SHADER,o.LOW_INT),lowFloatPrecision:o.getShaderPrecisionFormat(o.FRAGMENT_SHADER,o.LOW_FLOAT),mediumIntPrecision:o.getShaderPrecisionFormat(o.FRAGMENT_SHADER,o.MEDIUM_INT),mediumFloatPrecision:o.getShaderPrecisionFormat(o.FRAGMENT_SHADER,o.MEDIUM_FLOAT),highIntPrecision:o.getShaderPrecisionFormat(o.FRAGMENT_SHADER,o.HIGH_INT),highFloatPrecision:o.getShaderPrecisionFormat(o.FRAGMENT_SHADER,o.HIGH_FLOAT)})}static getIsTextureFloat(){return!0}static getChannelCount(){return p.getParameter(p.MAX_DRAW_BUFFERS)}static getMaxTextureSize(){return p.getParameter(p.MAX_TEXTURE_SIZE)}static lookupKernelValueType(o,l,x,w){return $(o,l,x,w)}static get testCanvas(){return y}static get testContext(){return p}static get features(){return g}static get fragmentShader(){return d}static get vertexShader(){return E}initContext(){return this.canvas.getContext("webgl2",{alpha:!1,depth:!1,antialias:!1})}initExtensions(){this.extensions={EXT_color_buffer_float:this.context.getExtension("EXT_color_buffer_float"),OES_texture_float_linear:this.context.getExtension("OES_texture_float_linear")}}validateSettings(o){if(!this.validate){this.texSize=c.getKernelTextureSize({optimizeFloatMemory:this.optimizeFloatMemory,precision:this.precision},this.output);return}const{features:l}=this.constructor;if(this.precision==="single"&&!l.isFloatRead)throw new Error("Float texture outputs are not supported");if(!this.graphical&&this.precision===null&&(this.precision=l.isFloatRead?"single":"unsigned"),this.fixIntegerDivisionAccuracy===null?this.fixIntegerDivisionAccuracy=!l.isIntegerDivisionAccurate:this.fixIntegerDivisionAccuracy&&l.isIntegerDivisionAccurate&&(this.fixIntegerDivisionAccuracy=!1),this.checkOutput(),!this.output||this.output.length===0){if(o.length!==1)throw new Error("Auto output only supported for kernels with only one input");const x=c.getVariableType(o[0],this.strictIntegers);switch(x){case"Array":this.output=c.getDimensions(x);break;case"NumberTexture":case"MemoryOptimizedNumberTexture":case"ArrayTexture(1)":case"ArrayTexture(2)":case"ArrayTexture(3)":case"ArrayTexture(4)":this.output=o[0].output;break;default:throw new Error("Auto output not supported for input type: "+x)}}if(this.graphical){if(this.output.length!==2)throw new Error("Output must have 2 dimensions on graphical mode");this.precision==="single"&&(console.warn("Cannot use graphical mode and single precision at the same time"),this.precision="unsigned"),this.texSize=c.clone(this.output);return}else!this.graphical&&this.precision===null&&l.isTextureFloat&&(this.precision="single");this.texSize=c.getKernelTextureSize({optimizeFloatMemory:this.optimizeFloatMemory,precision:this.precision},this.output),this.checkTextureSize()}translateSource(){const o=_.fromKernel(this,I,{fixIntegerDivisionAccuracy:this.fixIntegerDivisionAccuracy});this.translatedSource=o.getPrototypeString("kernel"),this.setupReturnTypes(o)}drawBuffers(){this.context.drawBuffers(this.drawBuffersMap)}getTextureFormat(){const{context:o}=this;switch(this.getInternalFormat()){case o.R32F:return o.RED;case o.RG32F:return o.RG;case o.RGBA32F:return o.RGBA;case o.RGBA:return o.RGBA;default:throw new Error("Unknown internal format")}}renderValues(){return this._tightRead===void 0&&this._detectTightRead(),super.renderValues()}renderKernelsToArrays(){return this._tightRead===void 0&&this._detectTightRead(),super.renderKernelsToArrays()}readFloatPixelsToFloat32Array(){if(!this._tightRead)return super.readFloatPixelsToFloat32Array();const{texSize:o,context:l}=this,x=o[0],w=o[1],v=new Float32Array(x*w);return l.readPixels(0,0,x,w,l.RED,l.FLOAT,v),v}renderOutputAsync(){return this.renderOutput!==this.renderValues?Promise.resolve(this.renderOutput()):this.renderValuesAsync()}renderValuesAsync(){this._tightRead===void 0&&this._detectTightRead();const o=this.formatValues,[l,x,w]=this.output;return this.transferValuesAsync().then(v=>o(v,l,x,w))}transferValuesAsync(){const{texSize:o,context:l}=this,x=o[0],w=o[1];let v,C,b;this.precision==="single"?(v=this._tightRead?l.RED:l.RGBA,C=l.FLOAT,b=new Float32Array(x*w*(this._tightRead?1:4))):(v=l.RGBA,C=l.UNSIGNED_BYTE,b=new Uint8Array(x*w*4));const T=l.createBuffer();l.bindBuffer(l.PIXEL_PACK_BUFFER,T),l.bufferData(l.PIXEL_PACK_BUFFER,b.byteLength,l.STREAM_READ),l.readPixels(0,0,x,w,v,C,0),l.bindBuffer(l.PIXEL_PACK_BUFFER,null);const h=l.fenceSync(l.SYNC_GPU_COMMANDS_COMPLETE,0);return l.flush(),this._pollFence(h).then(()=>(l.bindBuffer(l.PIXEL_PACK_BUFFER,T),l.getBufferSubData(l.PIXEL_PACK_BUFFER,0,b),l.bindBuffer(l.PIXEL_PACK_BUFFER,null),l.deleteBuffer(T),this.precision==="single"?b:new Float32Array(b.buffer)),F=>{throw l.deleteBuffer(T),F})}_pollFence(o){const l=this.context;return new Promise((x,w)=>{let v,C=null;typeof MessageChannel<"u"?(C=new MessageChannel,C.port1.onmessage=()=>T(),v=()=>C.port2.postMessage(0)):v=()=>setTimeout(T,0);const b=(h,F)=>{l.deleteSync(o),C&&(C.port1.close(),C.port2.close()),h(F)},T=()=>{if(l.isContextLost())return b(w,new Error("WebGL context lost while awaiting kernel result"));const h=l.clientWaitSync(o,0,0);if(h===l.ALREADY_SIGNALED||h===l.CONDITION_SATISFIED)return b(x);if(h===l.WAIT_FAILED)return b(w,new Error("clientWaitSync failed while awaiting kernel result"));v()};T()})}_detectTightRead(){const o=this.context;this._tightRead=!1,o.bindFramebuffer(o.FRAMEBUFFER,this.framebuffer);const l=this.returnType==="Number"||this.returnType==="Float"||this.returnType==="Integer"||this.returnType==="LiteralInteger";if(!(this.precision!=="single"||this.optimizeFloatMemory||this.graphical||!l)&&!(o.getParameter(o.IMPLEMENTATION_COLOR_READ_FORMAT)!==o.RED||o.getParameter(o.IMPLEMENTATION_COLOR_READ_TYPE)!==o.FLOAT)){if(this.formatValues===c.erectFloat)this.formatValues=c.erectMemoryOptimizedFloat;else if(this.formatValues===c.erect2DFloat)this.formatValues=c.erectMemoryOptimized2DFloat;else if(this.formatValues===c.erect3DFloat)this.formatValues=c.erectMemoryOptimized3DFloat;else if(this.formatValues!==c.erectMemoryOptimizedFloat&&this.formatValues!==c.erectMemoryOptimized2DFloat&&this.formatValues!==c.erectMemoryOptimized3DFloat)return;this._tightRead=!0}}getInternalFormat(){const{context:o}=this;if(this.precision==="single")switch(this.returnType){case"Number":case"Float":case"Integer":return this.optimizeFloatMemory?o.RGBA32F:o.R32F;case"Array(2)":return o.RG32F;case"Array(3)":case"Array(4)":return o.RGBA32F;default:throw new Error("Unhandled return type")}return o.RGBA}_setupOutputTexture(){const o=this.context;if(this.texture){o.framebufferTexture2D(o.FRAMEBUFFER,o.COLOR_ATTACHMENT0,o.TEXTURE_2D,this.texture.texture,0),this._tightRead=void 0;return}o.bindFramebuffer(o.FRAMEBUFFER,this.framebuffer);const l=o.createTexture(),x=this.texSize;o.activeTexture(o.TEXTURE0+this.constantTextureCount+this.argumentTextureCount),o.bindTexture(o.TEXTURE_2D,l),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_WRAP_S,o.REPEAT),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_WRAP_T,o.REPEAT),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_MIN_FILTER,o.NEAREST),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_MAG_FILTER,o.NEAREST);const w=this.getInternalFormat();this.precision==="single"?o.texStorage2D(o.TEXTURE_2D,1,w,x[0],x[1]):o.texImage2D(o.TEXTURE_2D,0,w,x[0],x[1],0,w,o.UNSIGNED_BYTE,null),o.framebufferTexture2D(o.FRAMEBUFFER,o.COLOR_ATTACHMENT0,o.TEXTURE_2D,l,0),this.texture=new this.TextureConstructor({texture:l,size:x,dimensions:this.threadDim,output:this.output,context:this.context,internalFormat:this.getInternalFormat(),textureFormat:this.getTextureFormat(),kernel:this}),this._tightRead=void 0}_setupSubOutputTextures(){const o=this.context;if(this.mappedTextures){for(let x=0;x<this.subKernels.length;x++)o.framebufferTexture2D(o.FRAMEBUFFER,o.COLOR_ATTACHMENT0+x+1,o.TEXTURE_2D,this.mappedTextures[x].texture,0);return}const l=this.texSize;this.drawBuffersMap=[o.COLOR_ATTACHMENT0],this.mappedTextures=[];for(let x=0;x<this.subKernels.length;x++){const w=this.createTexture();this.drawBuffersMap.push(o.COLOR_ATTACHMENT0+x+1),o.activeTexture(o.TEXTURE0+this.constantTextureCount+this.argumentTextureCount+x),o.bindTexture(o.TEXTURE_2D,w),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_WRAP_S,o.CLAMP_TO_EDGE),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_WRAP_T,o.CLAMP_TO_EDGE),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_MIN_FILTER,o.NEAREST),o.texParameteri(o.TEXTURE_2D,o.TEXTURE_MAG_FILTER,o.NEAREST);const v=this.getInternalFormat();this.precision==="single"?o.texStorage2D(o.TEXTURE_2D,1,v,l[0],l[1]):o.texImage2D(o.TEXTURE_2D,0,o.RGBA,l[0],l[1],0,o.RGBA,o.UNSIGNED_BYTE,null),o.framebufferTexture2D(o.FRAMEBUFFER,o.COLOR_ATTACHMENT0+x+1,o.TEXTURE_2D,w,0),this.mappedTextures.push(new this.TextureConstructor({texture:w,size:l,dimensions:this.threadDim,output:this.output,context:this.context,internalFormat:this.getInternalFormat(),textureFormat:this.getTextureFormat(),kernel:this}))}}_getHeaderString(){return""}_getTextureCoordinate(){const o=this.subKernels,l=this.getVariablePrecisionString(this.texSize,this.tactic);return o===null||o.length<1?`in ${l} vec2 vTexCoord;
`:`out ${l} vec2 vTexCoord;
`}_getMainArgumentsString(o){const l=[],x=this.argumentNames;for(let w=0;w<x.length;w++)l.push(this.kernelArguments[w].getSource(o[w]));return l.join("")}getKernelString(){const o=[this.getKernelResultDeclaration()],l=this.subKernels;if(l!==null)switch(o.push("layout(location = 0) out vec4 data0"),this.returnType){case"Number":case"Float":case"Integer":for(let x=0;x<l.length;x++){const w=l[x];o.push(w.returnType==="Integer"?`int subKernelResult_${w.name} = 0`:`float subKernelResult_${w.name} = 0.0`,`layout(location = ${x+1}) out vec4 data${x+1}`)}break;case"Array(2)":for(let x=0;x<l.length;x++)o.push(`vec2 subKernelResult_${l[x].name}`,`layout(location = ${x+1}) out vec4 data${x+1}`);break;case"Array(3)":for(let x=0;x<l.length;x++)o.push(`vec3 subKernelResult_${l[x].name}`,`layout(location = ${x+1}) out vec4 data${x+1}`);break;case"Array(4)":for(let x=0;x<l.length;x++)o.push(`vec4 subKernelResult_${l[x].name}`,`layout(location = ${x+1}) out vec4 data${x+1}`);break}else o.push("out vec4 data0");return c.linesToString(o)+this.translatedSource}getMainResultGraphical(){return c.linesToString(["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  data0 = actualColor"])}getMainResultPackedPixels(){switch(this.returnType){case"LiteralInteger":case"Number":case"Integer":case"Float":return this.getMainResultKernelPackedPixels()+this.getMainResultSubKernelPackedPixels();default:throw new Error(`packed output only usable with Numbers, "${this.returnType}" specified`)}}getMainResultKernelPackedPixels(){return c.linesToString(["  threadId = indexTo3D(index, uOutputDim)","  kernel()",`  data0 = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(kernelResult)`])}getMainResultSubKernelPackedPixels(){const o=[];if(!this.subKernels)return"";for(let l=0;l<this.subKernels.length;l++)this.subKernels[l].returnType==="Integer"?o.push(`  data${l+1} = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(float(subKernelResult_${this.subKernels[l].name}))`):o.push(`  data${l+1} = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(subKernelResult_${this.subKernels[l].name})`);return c.linesToString(o)}getMainResultKernelMemoryOptimizedFloats(o,l){o.push("  threadId = indexTo3D(index, uOutputDim)","  kernel()",`  data0.${l} = kernelResult`)}getMainResultSubKernelMemoryOptimizedFloats(o,l){if(!this.subKernels)return o;for(let x=0;x<this.subKernels.length;x++){const w=this.subKernels[x];w.returnType==="Integer"?o.push(`  data${x+1}.${l} = float(subKernelResult_${w.name})`):o.push(`  data${x+1}.${l} = subKernelResult_${w.name}`)}}getMainResultKernelNumberTexture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  data0[0] = kernelResult"]}getMainResultSubKernelNumberTexture(){const o=[];if(!this.subKernels)return o;for(let l=0;l<this.subKernels.length;++l){const x=this.subKernels[l];x.returnType==="Integer"?o.push(`  data${l+1}[0] = float(subKernelResult_${x.name})`):o.push(`  data${l+1}[0] = subKernelResult_${x.name}`)}return o}getMainResultKernelArray2Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  data0[0] = kernelResult[0]","  data0[1] = kernelResult[1]"]}getMainResultSubKernelArray2Texture(){const o=[];if(!this.subKernels)return o;for(let l=0;l<this.subKernels.length;++l){const x=this.subKernels[l];o.push(`  data${l+1}[0] = subKernelResult_${x.name}[0]`,`  data${l+1}[1] = subKernelResult_${x.name}[1]`)}return o}getMainResultKernelArray3Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  data0[0] = kernelResult[0]","  data0[1] = kernelResult[1]","  data0[2] = kernelResult[2]"]}getMainResultSubKernelArray3Texture(){const o=[];if(!this.subKernels)return o;for(let l=0;l<this.subKernels.length;++l){const x=this.subKernels[l];o.push(`  data${l+1}[0] = subKernelResult_${x.name}[0]`,`  data${l+1}[1] = subKernelResult_${x.name}[1]`,`  data${l+1}[2] = subKernelResult_${x.name}[2]`)}return o}getMainResultKernelArray4Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  data0 = kernelResult"]}getMainResultSubKernelArray4Texture(){const o=[];if(!this.subKernels)return o;for(let l=0;l<this.subKernels.length;++l)o.push(`  data${l+1} = subKernelResult_${this.subKernels[l].name}`);return o}destroyExtensions(){this.extensions.EXT_color_buffer_float=null,this.extensions.OES_texture_float_linear=null}toJSON(){const o=super.toJSON();return o.functionNodes=_.fromKernel(this,I).toJSON(),o.settings.threadDim=this.threadDim,o}};G.exports={WebGL2Kernel:k}}),ni=s((j,G)=>{const{utils:D}=m(),{FunctionNode:I}=Y();var _=class extends I{wgslFloat(p){if(p===1/0)return"0x1.fffffep+127";if(p===-1/0)return"-0x1.fffffep+127";if(p>34028234663852886e22)return"0x1.fffffep+127";if(p<-34028234663852886e22)return"-0x1.fffffep+127";const g=`${p}`;return g.indexOf(".")!==-1||g.indexOf("e")!==-1||g.indexOf("E")!==-1?g:`${g}.0`}wgslInt(p){return`${Math.round(p)}`}mangleFunctionName(p){return y.indexOf(p)!==-1?`fn_${p}`:D.sanitizeName(p)}getLookupType(p){return p==="WebGPUBuffer"?"Number":super.getLookupType(p)}astUpdateExpression(p,g){return this.astGeneric(p.argument,g),g.push(p.operator),g}getType(p){if(p&&p.type==="ConditionalExpression"){const g=this.getType(p.consequent);if(g==="Integer"||g==="LiteralInteger"){const k=this.getType(p.alternate);if(k==="Number"||k==="Float")return"Number"}}return super.getType(p)}astConditionalExpression(p,g){if(p.type!=="ConditionalExpression")throw this.astErrorOutput("Not a conditional expression",p);const k=this.getType(p.consequent),o=this.getType(p.alternate);if(k===null&&o===null)return g.push("if ("),this.astGeneric(p.test,g),g.push(") {"),this.astGeneric(p.consequent,g),g.push(";"),g.push("} else {"),this.astGeneric(p.alternate,g),g.push(";"),g.push("}"),g;let l=k==="LiteralInteger"?"Number":k;l==="Integer"&&(o==="Number"||o==="Float")&&(l="Number");const x=w=>{const v=this.getType(w);switch(l){case"Number":case"Float":v==="Integer"?this.castValueToFloat(w,g):v==="LiteralInteger"?this.castLiteralToFloat(w,g):this.astGeneric(w,g);break;case"Integer":v==="Number"||v==="Float"?this.castValueToInteger(w,g):v==="LiteralInteger"?this.castLiteralToInteger(w,g):this.astGeneric(w,g);break;default:this.astGeneric(w,g)}};return g.push("select("),x(p.alternate),g.push(", "),x(p.consequent),g.push(", "),this.astGeneric(p.test,g),g.push(")"),g}astFunction(p,g){if(this.isRootKernel){for(let l=0;l<p.body.body.length;++l)this.astGeneric(p.body.body[l],g),g.push(`
`);return g}this.returnType||this.findLastReturn()&&(this.returnType=this.getType(p.body),this.returnType==="LiteralInteger"&&(this.returnType="Number"));const{returnType:k}=this;let o=null;if(k&&(o=c[k],!o))throw this.astErrorOutput(`unknown return type ${k}`,p);g.push(`fn ${this.mangleFunctionName(this.name)}(`);for(let l=0;l<this.argumentNames.length;++l){const x=this.argumentNames[l];l>0&&g.push(", ");let w=this.argumentTypes[this.argumentNames.indexOf(x)];if(!w)throw this.astErrorOutput(`Unknown argument ${x} type`,p);w==="LiteralInteger"&&(this.argumentTypes[l]=w="Number");const v=c[w];if(!v)throw this.astErrorOutput(`WebGPU backend does not yet support ${w} arguments to helper functions`,p);g.push(`user_${D.sanitizeName(x)} : ${v}`)}g.push(")"),o&&g.push(` -> ${o}`),g.push(` {
`);for(let l=0;l<p.body.body.length;++l)this.astGeneric(p.body.body[l],g),g.push(`
`);return g.push(`}
`),g}astReturnStatement(p,g){if(!p.argument)throw this.astErrorOutput("Unexpected return statement",p);this.pushState("skip-literal-correction");const k=this.getType(p.argument);this.popState("skip-literal-correction");const o=[];switch(this.returnType||(k==="LiteralInteger"||k==="Integer"?this.returnType="Number":this.returnType=k),this.returnType){case"LiteralInteger":case"Number":case"Float":switch(k){case"Integer":o.push("f32("),this.astGeneric(p.argument,o),o.push(")");break;case"LiteralInteger":this.castLiteralToFloat(p.argument,o),this.getType(p.argument)==="Integer"&&(o.unshift("f32("),o.push(")"));break;default:this.astGeneric(p.argument,o)}break;case"Integer":switch(k){case"Float":case"Number":this.castValueToInteger(p.argument,o);break;case"LiteralInteger":this.castLiteralToInteger(p.argument,o);break;default:this.astGeneric(p.argument,o)}break;case"Boolean":case"Array(4)":case"Array(3)":case"Array(2)":this.astGeneric(p.argument,o);break;default:throw this.astErrorOutput(`unhandled return type ${this.returnType}`,p)}if(this.isRootKernel)switch(this.returnType){case"Array(4)":case"Array(3)":case"Array(2)":{const l=parseInt(this.returnType.substring(6),10),x=this.getInternalVariableName("kernelResultVec");g.push(`let ${x} : ${c[this.returnType]} = ${o.join("")};
`);for(let w=0;w<l;w++)g.push(`result[data_index * ${l} + ${w}] = ${x}.${E[w]};
`);g.push("return;");break}case"Integer":g.push(`result[data_index] = f32(${o.join("")});`),g.push("return;");break;default:g.push(`result[data_index] = ${o.join("")};`),g.push("return;")}else{if(this.isSubKernel)throw this.astErrorOutput("WebGPU backend does not yet support createKernelMap",p);g.push(`return ${o.join("")};`)}return g}astLiteral(p,g){if(p.value===!0||p.value===!1)return g.push(p.value?"true":"false"),g;if(isNaN(p.value))throw this.astErrorOutput("Non-numeric literal not supported : "+p.value,p);const k=this.astKey(p);return Number.isInteger(p.value)?this.isState("casting-to-integer")||this.isState("building-integer")?(this.literalTypes[k]="Integer",g.push(this.wgslInt(p.value))):(this.literalTypes[k]="Number",g.push(this.wgslFloat(p.value))):this.isState("casting-to-integer")||this.isState("building-integer")?(this.literalTypes[k]="Integer",g.push(this.wgslInt(p.value))):(this.literalTypes[k]="Number",g.push(this.wgslFloat(p.value))),g}astBinaryExpression(p,g){if(this.checkAndUpconvertOperator(p,g))return g;if(p.operator==="/"||p.operator==="%"){switch(g.push("("),this.pushState("building-float"),this.getType(p.left)){case"Integer":this.castValueToFloat(p.left,g);break;case"LiteralInteger":this.castLiteralToFloat(p.left,g);break;default:this.astGeneric(p.left,g)}switch(g.push(p.operator),this.getType(p.right)){case"Integer":this.castValueToFloat(p.right,g);break;case"LiteralInteger":this.castLiteralToFloat(p.right,g);break;default:this.astGeneric(p.right,g)}return this.popState("building-float"),g.push(")"),g}g.push("(");const k=this.getType(p.left)||"Number",o=this.getType(p.right)||"Number",l=k+" & "+o;switch(l){case"Integer & Integer":this.pushState("building-integer"),this.astGeneric(p.left,g),g.push(d[p.operator]||p.operator),this.astGeneric(p.right,g),this.popState("building-integer");break;case"Number & Float":case"Float & Number":case"Float & Float":case"Number & Number":this.pushState("building-float"),this.astGeneric(p.left,g),g.push(d[p.operator]||p.operator),this.astGeneric(p.right,g),this.popState("building-float");break;case"LiteralInteger & LiteralInteger":this.isState("casting-to-integer")||this.isState("building-integer")?(this.pushState("building-integer"),this.astGeneric(p.left,g),g.push(d[p.operator]||p.operator),this.astGeneric(p.right,g),this.popState("building-integer")):(this.pushState("building-float"),this.castLiteralToFloat(p.left,g),g.push(d[p.operator]||p.operator),this.castLiteralToFloat(p.right,g),this.popState("building-float"));break;case"Integer & Float":case"Integer & Number":this.pushState("building-float"),this.castValueToFloat(p.left,g),g.push(d[p.operator]||p.operator),this.astGeneric(p.right,g),this.popState("building-float");break;case"Integer & LiteralInteger":this.pushState("building-integer"),this.astGeneric(p.left,g),g.push(d[p.operator]||p.operator),this.castLiteralToInteger(p.right,g),this.popState("building-integer");break;case"Number & Integer":this.pushState("building-float"),this.astGeneric(p.left,g),g.push(d[p.operator]||p.operator),this.castValueToFloat(p.right,g),this.popState("building-float");break;case"Float & LiteralInteger":case"Number & LiteralInteger":this.pushState("building-float"),this.astGeneric(p.left,g),g.push(d[p.operator]||p.operator),this.castLiteralToFloat(p.right,g),this.popState("building-float");break;case"LiteralInteger & Float":case"LiteralInteger & Number":this.isState("casting-to-integer")?(this.pushState("building-integer"),this.castLiteralToInteger(p.left,g),g.push(d[p.operator]||p.operator),this.castValueToInteger(p.right,g),this.popState("building-integer")):(this.pushState("building-float"),this.castLiteralToFloat(p.left,g),g.push(d[p.operator]||p.operator),this.pushState("casting-to-float"),this.astGeneric(p.right,g),this.popState("casting-to-float"),this.popState("building-float"));break;case"LiteralInteger & Integer":this.pushState("building-integer"),this.castLiteralToInteger(p.left,g),g.push(d[p.operator]||p.operator),this.astGeneric(p.right,g),this.popState("building-integer");break;case"Boolean & Boolean":this.pushState("building-boolean"),this.astGeneric(p.left,g),g.push(d[p.operator]||p.operator),this.astGeneric(p.right,g),this.popState("building-boolean");break;case"Float & Integer":this.pushState("building-float"),this.astGeneric(p.left,g),g.push(d[p.operator]||p.operator),this.castValueToFloat(p.right,g),this.popState("building-float");break;default:throw this.astErrorOutput(`Unhandled binary expression between ${l}`,p)}return g.push(")"),g}checkAndUpconvertOperator(p,g){if(this.checkAndUpconvertBitwiseOperators(p,g))return g;if(p.operator!=="**")return null;switch(g.push("_pow"),g.push("("),this.getType(p.left)){case"Integer":this.castValueToFloat(p.left,g);break;case"LiteralInteger":this.castLiteralToFloat(p.left,g);break;default:this.astGeneric(p.left,g)}switch(g.push(","),this.getType(p.right)){case"Integer":this.castValueToFloat(p.right,g);break;case"LiteralInteger":this.castLiteralToFloat(p.right,g);break;default:this.astGeneric(p.right,g)}return g.push(")"),g}checkAndUpconvertBitwiseOperators(p,g){if(!{"&":!0,"|":!0,"^":!0,"<<":!0,">>":!0,">>>":!0}[p.operator])return null;const k=o=>{switch(this.getType(o)){case"Number":case"Float":this.castValueToInteger(o,g);break;case"LiteralInteger":this.castLiteralToInteger(o,g);break;default:this.pushState("building-integer"),this.astGeneric(o,g),this.popState("building-integer")}};return g.push("("),p.operator===">>>"?(g.push("bitcast<i32>(bitcast<u32>("),k(p.left),g.push(") >> u32("),k(p.right),g.push("))")):p.operator==="<<"||p.operator===">>"?(k(p.left),g.push(` ${p.operator} u32(`),k(p.right),g.push(")")):(k(p.left),g.push(` ${p.operator} `),k(p.right)),g.push(")"),g}checkAndUpconvertBitwiseUnary(p,g){if(p.operator!=="~")return null;switch(g.push("~("),this.getType(p.argument)){case"Number":case"Float":this.castValueToInteger(p.argument,g);break;case"LiteralInteger":this.castLiteralToInteger(p.argument,g);break;default:this.astGeneric(p.argument,g)}return g.push(")"),g}astUnaryExpression(p,g){return this.checkAndUpconvertBitwiseUnary(p,g)?g:p.operator==="+"?(this.astGeneric(p.argument,g),g):(p.prefix?(g.push(p.operator),this.astGeneric(p.argument,g)):(this.astGeneric(p.argument,g),g.push(p.operator)),g)}castLiteralToInteger(p,g){return this.pushState("casting-to-integer"),this.astGeneric(p,g),this.popState("casting-to-integer"),g}castLiteralToFloat(p,g){return this.pushState("casting-to-float"),this.astGeneric(p,g),this.popState("casting-to-float"),g}castValueToInteger(p,g){return this.pushState("casting-to-integer"),g.push("i32("),this.astGeneric(p,g),g.push(")"),this.popState("casting-to-integer"),g}castValueToFloat(p,g){return this.pushState("casting-to-float"),g.push("f32("),this.astGeneric(p,g),g.push(")"),this.popState("casting-to-float"),g}astIdentifierExpression(p,g){if(p.type!=="Identifier")throw this.astErrorOutput("IdentifierExpression - not an Identifier",p);const k=this.getType(p),o=D.sanitizeName(p.name);return p.name==="Infinity"?(g.push("0x1.fffffep+127"),g):this.isRootKernel&&this.argumentNames.indexOf(p.name)!==-1&&(k==="Number"||k==="Float"||k==="Integer"||k==="Boolean")?(k==="Boolean"?g.push(`bool(params.user_${o})`):g.push(`params.user_${o}`),g):(g.push(`user_${o}`),g)}astForStatement(p,g){if(p.type!=="ForStatement")throw this.astErrorOutput("Invalid for statement",p);const k=[],o=[],l=[],x=[];let w=null;if(p.init){const{declarations:v}=p.init;v.length>1&&(w=!1),this.astGeneric(p.init,k);for(let C=0;C<v.length;C++)v[C].init&&v[C].init.type!=="Literal"&&(w=!1)}else w=!1;if(p.test?this.astGeneric(p.test,o):w=!1,p.update?(p.update.type==="AssignmentExpression"&&this.pushState("assignment-as-statement"),this.astGeneric(p.update,l)):w=!1,p.body&&(this.pushState("loop-body"),this.astGeneric(p.body,x),this.popState("loop-body")),w===null&&(w=this.isSafe(p.init)&&this.isSafe(p.test)),w){const v=k.join(""),C=v[v.length-1]!==";";g.push(`for (${v}${C?";":""}${o.join("")};${l.join("")}){
`),g.push(x.join("")),g.push(`}
`)}else{const v=this.getInternalVariableName("safeI");k.length>0&&g.push(k.join(""),`
`),g.push(`for (var ${v} : i32 = 0;${v}<LOOP_MAX;${v}++){
`),o.length>0&&g.push(`if (!(${o.join("")})) { break; }
`),g.push(x.join("")),g.push(`
${l.join("")};`),g.push(`}
`)}return g}astWhileStatement(p,g){if(p.type!=="WhileStatement")throw this.astErrorOutput("Invalid while statement",p);const k=this.getInternalVariableName("safeI");return g.push(`for (var ${k} : i32 = 0;${k}<LOOP_MAX;${k}++){
`),g.push("if (!("),this.astGeneric(p.test,g),g.push(`)) { break; }
`),this.astGeneric(p.body,g),g.push(`}
`),g}astDoWhileStatement(p,g){if(p.type!=="DoWhileStatement")throw this.astErrorOutput("Invalid while statement",p);const k=this.getInternalVariableName("safeI");return g.push(`for (var ${k} : i32 = 0;${k}<LOOP_MAX;${k}++){
`),this.astGeneric(p.body,g),g.push("if (!("),this.astGeneric(p.test,g),g.push(`)) { break; }
`),g.push(`}
`),g}astAssignmentExpression(p,g){if(this.isState("assignment-as-statement"))this.popState("assignment-as-statement");else throw this.astErrorOutput("WebGPU backend does not yet support assignment used as an expression",p);if(p.operator==="%="){this.astGeneric(p.left,g),g.push("=("),this.astGeneric(p.left,g),g.push("%");const k=this.getType(p.right);k==="Integer"?this.castValueToFloat(p.right,g):k==="LiteralInteger"?this.castLiteralToFloat(p.right,g):this.astGeneric(p.right,g),g.push(")")}else if(p.operator==="**="){this.astGeneric(p.left,g),g.push("="),g.push("_pow("),this.astGeneric(p.left,g),g.push(",");const k=this.getType(p.right);k==="Integer"?this.castValueToFloat(p.right,g):k==="LiteralInteger"?this.castLiteralToFloat(p.right,g):this.astGeneric(p.right,g),g.push(")")}else{const k=this.getType(p.left),o=this.getType(p.right);this.astGeneric(p.left,g),g.push(p.operator),k!=="Integer"&&o==="Integer"?(g.push("f32("),this.astGeneric(p.right,g),g.push(")")):k!=="Integer"&&o==="LiteralInteger"?this.castLiteralToFloat(p.right,g):k==="Integer"&&o==="LiteralInteger"?this.castLiteralToInteger(p.right,g):k==="Integer"&&(o==="Number"||o==="Float")?(g.push("i32("),this.astGeneric(p.right,g),g.push(")")):this.astGeneric(p.right,g)}return g}astBlockStatement(p,g){if(this.isState("loop-body")){this.pushState("block-body");for(let k=0;k<p.body.length;k++)this.astGeneric(p.body[k],g);this.popState("block-body")}else{g.push(`{
`);for(let k=0;k<p.body.length;k++)this.astGeneric(p.body[k],g);g.push(`}
`)}return g}astVariableDeclaration(p,g){const k=p.declarations;if(!k||!k[0]||!k[0].init)throw this.astErrorOutput("Unexpected expression",p);for(let o=0;o<k.length;o++){const l=k[o],x=l.init,w=this.getDeclaration(l.id),v=this.getType(l.init);let C=v;C==="LiteralInteger"&&(w.suggestedType==="Integer"?C="Integer":C="Number");const b=D.sanitizeName(l.id.name);if(v==="Integer"&&C==="Integer")w.valueType="Number",g.push(`var user_${b} : f32 = `),g.push("f32("),this.astGeneric(x,g),g.push(")");else{const T=c[C];if(!T)throw this.astErrorOutput(`Markup type ${C} not handled`,p);w.valueType=C,g.push(`var user_${b} : ${T} = `),v==="Number"&&C==="Integer"?(g.push("i32("),this.astGeneric(x,g),g.push(")")):v==="LiteralInteger"&&C==="Integer"?this.castLiteralToInteger(x,g):v==="LiteralInteger"&&C==="Number"?this.castLiteralToFloat(x,g):v==="Integer"&&C==="Number"?this.castValueToFloat(x,g):this.astGeneric(x,g)}g.push(";")}return g}astIfStatement(p,g){return g.push("if ("),this.astGeneric(p.test,g),g.push(")"),p.consequent.type==="BlockStatement"?(this.pushState("if-body"),this.astGeneric(p.consequent,g),this.popState("if-body")):(g.push(` {
`),this.astGeneric(p.consequent,g),g.push(`
}
`)),p.alternate&&(g.push("else "),p.alternate.type==="IfStatement"?this.astGeneric(p.alternate,g):p.alternate.type==="BlockStatement"?(this.pushState("if-body"),this.astGeneric(p.alternate,g),this.popState("if-body")):(g.push(` {
`),this.astGeneric(p.alternate,g),g.push(`
}
`))),g}astSwitchCaseConsequent(p,g){const k=[];for(let o=0;o<p.length&&p[o].type!=="BreakStatement";o++)k.push(p[o]);for(let o=0;o<k.length;o++){const l=x=>{if(!x||typeof x!="object")return!1;if(Array.isArray(x))return x.some(l);if(x.type==="BreakStatement")return!0;if(x.type==="ForStatement"||x.type==="WhileStatement"||x.type==="DoWhileStatement"||x.type==="SwitchStatement")return!1;for(const w in x)if(!(w==="loc"||w==="range"||w==="parent")&&l(x[w]))return!0;return!1};if(l(k[o]))throw this.astErrorOutput("break inside a switch case is only supported as the case terminator",k[o])}for(let o=0;o<k.length;o++)this.astGeneric(k[o],g),g.push(`
`);return g}astSwitchStatement(p,g){if(p.type!=="SwitchStatement")throw this.astErrorOutput("Invalid switch statement",p);const{discriminant:k,cases:o}=p,l=this.getType(k),x=`switchDiscriminant${this.astKey(p,"_")}`;switch(l){case"Float":case"Number":g.push(`var ${x} : f32 = `),this.astGeneric(k,g),g.push(`;
`);break;case"Integer":g.push(`var ${x} : i32 = `),this.astGeneric(k,g),g.push(`;
`);break;default:throw this.astErrorOutput(`Unhandled switch discriminant type "${l}"`,p)}if(o.length===1&&!o[0].test)return this.astSwitchCaseConsequent(o[0].consequent,g),g;let w=!1,v=[],C=!1,b=!1;for(let T=0;T<o.length;T++){if(o[T].test){if(T===0||!b?(b=!0,g.push(`if (${x} == `)):w?(g.push(`${x} == `),w=!1):g.push(` else if (${x} == `),l==="Integer")switch(this.getType(o[T].test)){case"Number":case"Float":this.castValueToInteger(o[T].test,g);break;case"LiteralInteger":this.castLiteralToInteger(o[T].test,g);break}else switch(this.getType(o[T].test)){case"LiteralInteger":this.castLiteralToFloat(o[T].test,g);break;case"Integer":this.castValueToFloat(o[T].test,g);break;default:this.astGeneric(o[T].test,g)}if(!o[T].consequent||o[T].consequent.length===0){w=!0,g.push(" || ");continue}g.push(`) {
`)}else if(o.length>T+1){C=!0,this.astSwitchCaseConsequent(o[T].consequent,v);continue}else g.push(` else {
`);this.astSwitchCaseConsequent(o[T].consequent,g),g.push(`
}`)}return C&&(g.push(" else {"),g.push(v.join("")),g.push("}")),g.push(`
`),g}astThisExpression(p,g){return g.push("this"),g}astSequenceExpression(p,g){const{expressions:k}=p;if(k.length===1)return this.astGeneric(k[0],g),g;throw this.astErrorOutput("WebGPU backend does not yet support the comma operator",p)}astMemberExpression(p,g){const{property:k,name:o,signature:l,origin:x,type:w,xProperty:v,yProperty:C,zProperty:b}=this.getMemberExpressionDetails(p);switch(l){case"value.thread.value":case"this.thread.value":if(o!=="x"&&o!=="y"&&o!=="z")throw this.astErrorOutput("Unexpected expression, expected `this.thread.x`, `this.thread.y`, or `this.thread.z`",p);return g.push(`i32(threadGid.${o})`),g;case"this.output.value":{const h={x:0,y:1,z:2}[o];if(h===void 0)throw this.astErrorOutput("Unexpected expression",p);if(this.dynamicOutput){const F=`params.output${o.toUpperCase()}`;this.isState("casting-to-float")?g.push(`f32(${F})`):g.push(`i32(${F})`)}else this.isState("casting-to-integer")?g.push(`${this.output[h]}`):g.push(`${this.output[h]}.0`);return g}case"value":throw this.astErrorOutput("Unexpected expression",p);case"value[]":case"value[][]":case"value[][][]":case"value[][][][]":case"value.value":if(x==="Math")return g.push(this.wgslFloat(Math[o])),g;switch(k){case"r":return g.push(`user_${D.sanitizeName(o)}.x`),g;case"g":return g.push(`user_${D.sanitizeName(o)}.y`),g;case"b":return g.push(`user_${D.sanitizeName(o)}.z`),g;case"a":return g.push(`user_${D.sanitizeName(o)}.w`),g}break;case"this.constants.value":{const h=this.constants[o];switch(w){case"Integer":return this.isState("casting-to-float")?g.push(this.wgslFloat(h)):g.push(this.wgslInt(h)),g;case"Number":case"Float":return this.isState("casting-to-integer")?g.push(this.wgslInt(h)):g.push(this.wgslFloat(h)),g;case"Boolean":return g.push(h?"true":"false"),g;case"Array(2)":case"Array(3)":case"Array(4)":{const F=parseInt(w.substring(6),10),O=[];for(let z=0;z<F;z++)O.push(this.wgslFloat(h[z]));return g.push(`${c[w]}(${O.join(", ")})`),g}default:throw this.astErrorOutput(`WebGPU backend does not yet support constant type ${w}`,p)}}case"this.constants.value[]":case"this.constants.value[][]":case"this.constants.value[][][]":case"this.constants.value[][][][]":break;case"fn()[]":return this.astCallExpression(p.object,g),g.push("["),g.push(this.memberExpressionPropertyMarkup(k)),g.push("]"),g;default:throw this.astErrorOutput(`WebGPU backend does not yet support expression signature "${l}"`,p)}const T=`${x}_${D.sanitizeName(o)}`;switch(w){case"Array(2)":case"Array(3)":case"Array(4)":this.astGeneric(p.object,g),g.push("["),g.push(this.memberExpressionPropertyMarkup(v)),g.push("]");break;case"Array":case"Array2D":case"Array3D":case"Input":case"WebGPUBuffer":case"Number":case"Float":case"Integer":g.push(`get_${T}(`),this.memberExpressionXYZ(v,C,b,g),g.push(")");break;case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":throw this.astErrorOutput("WebGPU backend does not yet support Matrix types",p);default:throw this.astErrorOutput(`WebGPU backend does not yet support member expression type "${w}"`,p)}return g}astCallExpression(p,g){if(!p.callee)throw this.astErrorOutput("Unknown CallExpression",p);let k=null;const o=this.isAstMathFunction(p);if(o||p.callee.object&&p.callee.object.type==="ThisExpression"?k=p.callee.property.name:p.callee.type==="SequenceExpression"&&p.callee.expressions[0].type==="Literal"&&!isNaN(p.callee.expressions[0].raw)?k=p.callee.expressions[1].property.name:k=p.callee.name,!k)throw this.astErrorOutput("Unhandled function, couldn't find name",p);let l=k;if(o){if(k==="random")throw this.astErrorOutput("WebGPU backend does not yet support Math.random",p);$[k]&&(k=$[k]),l=k}else l=this.mangleFunctionName(k);this.calledFunctions.indexOf(k)<0&&this.calledFunctions.push(k),this.onFunctionCall&&this.onFunctionCall(this.name,k,p.arguments);const x=o&&P[k]&&this.isState("building-integer");if(x&&g.push("i32("),g.push(l),g.push("("),o)for(let w=0;w<p.arguments.length;++w){const v=p.arguments[w],C=this.getType(v);switch(w>0&&g.push(", "),C){case"Integer":this.castValueToFloat(v,g);break;case"LiteralInteger":this.castLiteralToFloat(v,g);break;default:this.astGeneric(v,g);break}}else{const w=this.lookupFunctionArgumentTypes(k)||[];for(let v=0;v<p.arguments.length;++v){const C=p.arguments[v];let b=w[v];v>0&&g.push(", ");const T=this.getType(C);switch(b||(this.triggerImplyArgumentType(k,v,T,this),b=T),T){case"Boolean":this.astGeneric(C,g);continue;case"Number":case"Float":if(b==="Integer"){g.push("i32("),this.astGeneric(C,g),g.push(")");continue}else if(b==="Number"||b==="Float"){this.astGeneric(C,g);continue}else if(b==="LiteralInteger"){this.castLiteralToFloat(C,g);continue}break;case"Integer":if(b==="Number"||b==="Float"){g.push("f32("),this.astGeneric(C,g),g.push(")");continue}else if(b==="Integer"){this.astGeneric(C,g);continue}break;case"LiteralInteger":if(b==="Integer"){this.castLiteralToInteger(C,g);continue}else if(b==="Number"||b==="Float"){this.castLiteralToFloat(C,g);continue}else if(b==="LiteralInteger"){this.astGeneric(C,g);continue}break;case"Array(2)":case"Array(3)":case"Array(4)":if(b===T){C.type==="Identifier"?g.push(`user_${D.sanitizeName(C.name)}`):this.astGeneric(C,g);continue}break;case"Array":case"Array2D":case"Array3D":case"Input":case"WebGPUBuffer":throw this.astErrorOutput("WebGPU backend does not yet support array arguments to helper functions",p)}throw this.astErrorOutput(`Unhandled argument combination of ${T} and ${b} for argument named "${C.name}"`,p)}}return g.push(")"),x&&g.push(")"),g}astArrayExpression(p,g){switch(this.getType(p)){case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":throw this.astErrorOutput("WebGPU backend does not yet support Matrix types",p)}const k=p.elements.length;g.push(`vec${k}<f32>(`);for(let o=0;o<k;++o){o>0&&g.push(", ");const l=p.elements[o];switch(this.getType(l)){case"Integer":this.castValueToFloat(l,g);break;case"LiteralInteger":this.castLiteralToFloat(l,g);break;default:this.astGeneric(l,g)}}return g.push(")"),g}memberExpressionXYZ(p,g,k,o){return k?o.push(this.memberExpressionPropertyMarkup(k),", "):o.push("0, "),g?o.push(this.memberExpressionPropertyMarkup(g),", "):o.push("0, "),o.push(this.memberExpressionPropertyMarkup(p)),o}memberExpressionPropertyMarkup(p){if(!p)throw new Error("Property not set");const g=this.getType(p),k=[];switch(g){case"Number":case"Float":this.castValueToInteger(p,k);break;case"LiteralInteger":this.castLiteralToInteger(p,k);break;case"Integer":this.pushState("building-integer"),k.push("i32("),this.astGeneric(p,k),k.push(")"),this.popState("building-integer");break;default:this.astGeneric(p,k)}return k.join("")}};const c={Number:"f32",Float:"f32",Integer:"i32",LiteralInteger:"f32",Boolean:"bool","Array(2)":"vec2<f32>","Array(3)":"vec3<f32>","Array(4)":"vec4<f32>"},d={"===":"==","!==":"!="},E=["x","y","z","w"],$={pow:"_pow",round:"_round"},P={ceil:!0,floor:!0,_round:!0},y=["alias","break","case","const","const_assert","continue","continuing","default","diagnostic","discard","else","enable","false","fn","for","if","let","loop","override","requires","return","struct","switch","true","var","while","main","params","result","gid","threadGid","data_index","select","abs","acos","acosh","asin","asinh","atan","atan2","atanh","ceil","clamp","cos","cosh","cross","degrees","distance","dot","exp","exp2","floor","fma","fract","inverseSqrt","length","log","log2","max","min","mix","modf","normalize","pow","radians","round","sign","sin","sinh","smoothstep","sqrt","step","tan","tanh","trunc","cbrt","expm1","fround","imul","log10","log1p","clz32","_pow","_round","LOOP_MAX","bitcast","ptr","array","vec2","vec3","vec4","mat2x2","mat3x3","mat4x4","f32","i32","u32","bool"];G.exports={WGSLFunctionNode:_}}),ri=s((j,G)=>{let D=null;G.exports={WebGPUContext:class Ya{static get isSupported(){return typeof navigator<"u"&&!!navigator.gpu}static acquire(){if(D)return D;const _=(async()=>{if(!Ya.isSupported)throw new Error("WebGPU is not supported on this platform (navigator.gpu is missing)");const c=await navigator.gpu.requestAdapter();if(!c)throw new Error("WebGPU is present (navigator.gpu) but no adapter is available. On headless Chromium there is no adapter; run headed. Use `await GPU.isWebGPUAvailable()` to feature-detect.");const d=await c.requestDevice({requiredLimits:{maxStorageBufferBindingSize:c.limits.maxStorageBufferBindingSize,maxBufferSize:c.limits.maxBufferSize}}),E={adapter:c,device:d,isLost:!1};return d.lost.then($=>{E.isLost=!0,$.reason!=="destroyed"&&console.error(`gpu.js [webgpu]: device lost: ${$.message}`),D===_&&(D=null)}),d.onuncapturederror=$=>{console.error(`gpu.js [webgpu]: ${$.error.message}`)},E})();return _.catch(()=>{D===_&&(D=null)}),D=_}static destroy(){if(!D)return Promise.resolve();const _=D;return D=null,_.then(({device:c})=>{c.destroy()},()=>{})}}}}),ii=s((j,G)=>{G.exports={WebGPUBufferResult:class Ja{constructor(I){this.buffer=I.buffer,this.output=I.output,this.componentCount=I.componentCount||1,this.context=I.context,this.kernel=I.kernel,this.type="WebGPUBuffer",this._deleted=!1,this.buffer._refs?this.buffer._refs++:this.buffer._refs=1}toArray(){return this._deleted?Promise.reject(new Error("WebGPUBufferResult has been deleted")):this.kernel.readBufferResult(this)}delete(){this._deleted||(this._deleted=!0,--this.buffer._refs===0&&this.buffer.destroy())}clone(){return new Ja(this)}}}}),ai=s((j,G)=>{const{Kernel:D}=A(),{FunctionBuilder:I}=N(),{WGSLFunctionNode:_}=ni(),{WebGPUContext:c}=ri(),{WebGPUBufferResult:d}=ii(),{utils:E}=m(),{Input:$}=a(),P=128,y=1,p=Object.freeze({kernelMap:!1,isIntegerDivisionAccurate:!0,isSpeedTacticSupported:!1,isTextureFloat:!0,isDrawBuffers:!1,kernelMapSize:0,channelCount:1,maxTextureSize:1/0,isFloatRead:!0}),g={_pow:`fn _pow(v1 : f32, v2 : f32) -> f32 {
  if (v2 == 0.0) { return 1.0; }
  return pow(v1, v2);
}`,_round:`fn _round(x : f32) -> f32 {
  return floor(x + 0.5);
}`,cbrt:`fn cbrt(x : f32) -> f32 {
  return sign(x) * pow(abs(x), 1.0 / 3.0);
}`,expm1:`fn expm1(x : f32) -> f32 {
  return exp(x) - 1.0;
}`,fround:`fn fround(x : f32) -> f32 {
  return x;
}`,imul:`fn imul(a : f32, b : f32) -> f32 {
  return f32(i32(a) * i32(b));
}`,log10:`fn log10(x : f32) -> f32 {
  return log2(x) * ${1/Math.log2(10)};
}`,log1p:`fn log1p(x : f32) -> f32 {
  return log(1.0 + x);
}`,clz32:`fn clz32(x : f32) -> f32 {
  return f32(countLeadingZeros(u32(x)));
}`};var k=class extends D{static get isSupported(){return c.isSupported}static get isAsync(){return!0}static isContextMatch(o){return!!(o&&typeof o.createShaderModule=="function"&&typeof o.createComputePipeline=="function")}static getFeatures(){return p}static get features(){return p}static get mode(){return"webgpu"}static getSignature(o,l){return"webgpu"+(l.length>0?":"+l.join(","):"")}static destroyContext(o){}static nativeFunctionArguments(){throw new Error("WebGPU backend does not yet support native functions")}static nativeFunctionReturnType(){throw new Error("WebGPU backend does not yet support native functions")}static combineKernels(){throw new Error("WebGPU backend does not yet support combineKernels; chain kernels with `await` and pipeline mode instead")}constructor(o,l){if(super(o,l),l){if(l.graphical)throw new Error("WebGPU backend does not yet support graphical mode; use the webgl backend");if(l.precision==="unsigned")throw new Error("WebGPU backend does not yet support precision: 'unsigned'; it is single precision only");if(l.subKernels)throw new Error("WebGPU backend does not yet support createKernelMap")}this.mergeSettings(o.settings||l),this.precision===null&&(this.precision="single"),this.asyncMode=!0,this.threadDim=null,this.componentCount=1,this.compiledSource=null,this.translatedBody=null,this.translatedFunctions=null,this.paramsLayout=null,this._buildPromise=null,this._device=null,this.computePipeline=null,this.bindGroupLayout=null,this.bindGroup=null,this.bindGroupDirty=!0,this.paramsBuffer=null,this.paramsMirror=null,this.outputBuffer=null,this.argumentBuffers=null,this.constantBuffers=null,this.stagingPool=[]}initCanvas(){return null}initContext(){return null}initPlugins(o){return[]}setGraphical(o){if(o)throw new Error("WebGPU backend does not yet support graphical mode; use the webgl backend");return super.setGraphical(o)}setOutput(o){const l=this.toKernelOutput(o);if(this.built){if(!this.dynamicOutput)throw new Error("Resizing a kernel with dynamicOutput: false is not possible");if(l.length!==this.output.length)throw new Error("WebGPU backend does not yet support changing the output rank of a built kernel; the workgroup shape is fixed at build")}return this.output=l,this}toString(){throw new Error("WebGPU backend does not yet support toString")}validateSettings(o){if(this.graphical)throw new Error("WebGPU backend does not yet support graphical mode; use the webgl backend");if(this.precision==="unsigned")throw new Error("WebGPU backend does not yet support precision: 'unsigned'; it is single precision only");if(this.precision="single",this.subKernels&&this.subKernels.length>0)throw new Error("WebGPU backend does not yet support createKernelMap");if(!this.output||this.output.length===0){if(o.length!==1)throw new Error("Auto output only supported for kernels with only one input");const l=E.getVariableType(o[0],this.strictIntegers);if(l==="Array")this.output=Array.from(E.getDimensions(o[0]));else if(l==="WebGPUBuffer")this.output=Array.from(o[0].output);else throw new Error("Auto output not supported for input type: "+l)}this.checkOutput()}setupArguments(o){super.setupArguments(o);for(let l=0;l<this.argumentTypes.length;l++)switch(this.argumentTypes[l]){case"Array":case"Input":case"WebGPUBuffer":case"Number":case"Float":case"Integer":case"Boolean":continue;default:throw new Error(`WebGPU backend does not yet support argument type ${this.argumentTypes[l]} (argument "${this.argumentNames[l]}")`)}}setupConstants(){super.setupConstants();for(const o in this.constantTypes)switch(this.constantTypes[o]){case"Array":case"Input":case"Number":case"Float":case"Integer":case"Boolean":case"Array(2)":case"Array(3)":case"Array(4)":continue;default:throw new Error(`WebGPU backend does not yet support constant type ${this.constantTypes[o]} (constant "${o}")`)}}build(){if(this.built)return Promise.resolve();if(this._buildPromise)return this._buildPromise;this.setupConstants(),this.setupArguments(arguments),this.validateSettings(arguments);const o=this.threadDim=Array.from(this.output);for(;o.length<3;)o.push(1);return this.translateSource(),this.paramsLayout=this.computeParamsLayout(),this.compiledSource=this.assembleWGSL(),this.debug&&(console.log("WGSL Shader Output:"),console.log(this.compiledSource)),this.buildSignature(arguments),this._buildPromise=this._buildAsync()}translateSource(){const o=I.fromKernel(this,_),l=o.getPrototypes("kernel");switch(this.translatedBody=l[l.length-1],this.translatedFunctions=l.slice(0,-1).join(`
`),this.returnType||(this.returnType=o.getKernelResultType()),this.returnType){case"Number":case"Float":case"Integer":case"LiteralInteger":this.componentCount=1;break;case"Array(2)":this.componentCount=2;break;case"Array(3)":this.componentCount=3;break;case"Array(4)":this.componentCount=4;break;default:throw new Error(`WebGPU backend does not yet support returning ${this.returnType}`)}}computeParamsLayout(){const o=[],l=[];let x=16;for(let v=0;v<this.argumentTypes.length;v++){const C=this.argumentTypes[v],b=E.sanitizeName(this.argumentNames[v]);C==="Array"||C==="Input"||C==="WebGPUBuffer"?(o.push({name:b,index:v,type:C,dimsOffset:x,buffer:null,boundBuffer:null}),x+=16):l.push({name:b,index:v,type:C,offset:null})}for(let v=0;v<l.length;v++)l[v].offset=x,x+=4;const w=[];if(this.constants)for(const v in this.constants){if(!this.constants.hasOwnProperty(v))continue;const C=this.constantTypes[v];(C==="Array"||C==="Input")&&w.push({name:E.sanitizeName(v),constantName:v,buffer:null})}return{arrayArgs:o,scalarArgs:l,bufferConstants:w,byteLength:Math.ceil(x/16)*16}}scalarWGSLType(o){switch(o){case"Integer":return"i32";case"Boolean":return"u32";default:return"f32"}}assembleWGSL(){const{arrayArgs:o,scalarArgs:l,bufferConstants:x}=this.paramsLayout,w=[],v=["  outputX : u32,","  outputY : u32,","  outputZ : u32,","  dispatchWidth : u32,"];for(let h=0;h<o.length;h++)v.push(`  user_${o[h].name}_dims : vec4<u32>,`);for(let h=0;h<l.length;h++)v.push(`  user_${l[h].name} : ${this.scalarWGSLType(l[h].type)},`);w.push("struct Params {",v.join(`
`),"}"),w.push("@group(0) @binding(0) var<uniform> params : Params;");for(let h=0;h<o.length;h++)w.push(`@group(0) @binding(${1+h}) var<storage, read> user_${o[h].name} : array<f32>;`);const C=1+o.length;w.push(`@group(0) @binding(${C}) var<storage, read_write> result : array<f32>;`);for(let h=0;h<x.length;h++)w.push(`@group(0) @binding(${C+1+h}) var<storage, read> constants_${x[h].name} : array<f32>;`);w.push("var<private> threadGid : vec3<u32>;");const b=`${this.translatedFunctions}
${this.translatedBody}`;/\bLOOP_MAX\b/.test(b)&&w.push(`const LOOP_MAX : i32 = ${parseInt(this.loopMaxIterations,10)||1e3};`);for(const h in g)new RegExp(`\\b${h}\\(`).test(b)&&w.push(g[h]);for(let h=0;h<o.length;h++){const F=o[h].name;w.push(`fn get_user_${F}(z : i32, y : i32, x : i32) -> f32 {
  return user_${F}[u32(x + i32(params.user_${F}_dims.x) * (y + i32(params.user_${F}_dims.y) * z))];
}`)}for(let h=0;h<x.length;h++){const F=x[h],O=this.constants[F.constantName],z=this.constantDimensions(O);w.push(`fn get_constants_${F.name}(z : i32, y : i32, x : i32) -> f32 {
  return constants_${F.name}[u32(x + ${z[0]} * (y + ${z[1]} * z))];
}`)}this.translatedFunctions&&w.push(this.translatedFunctions);const T=this.output.length===1?[64,1,1]:[8,8,1];return this.workgroupSize=T,this.output.length===1?w.push(`@compute @workgroup_size(${T[0]}, ${T[1]}, ${T[2]})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let flat_index : u32 = gid.x + gid.y * params.dispatchWidth;
  threadGid = vec3<u32>(flat_index, 0u, 0u);
  if (flat_index >= params.outputX) { return; }
  let data_index : i32 = i32(flat_index);
${this.translatedBody}
}`):w.push(`@compute @workgroup_size(${T[0]}, ${T[1]}, ${T[2]})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  threadGid = gid;
  if (gid.x >= params.outputX || gid.y >= params.outputY || gid.z >= params.outputZ) { return; }
  let data_index : i32 = i32(gid.x + params.outputX * (gid.y + params.outputY * gid.z));
${this.translatedBody}
}`),w.join(`
`)}constantDimensions(o){const l=o instanceof $?Array.from(o.size):Array.from(E.getDimensions(o));for(;l.length<3;)l.push(1);return l}async _buildAsync(){const o=await c.acquire();this.context=o;const l=this._device=o.device,x=l.createShaderModule({code:this.compiledSource}),w=(await x.getCompilationInfo()).messages.filter(O=>O.type==="error");if(w.length>0)throw new Error(`Error compiling WGSL compute shader:
`+w.map(O=>`  ${O.lineNum}:${O.linePos} ${O.message}`).join(`
`)+`
--- generated WGSL ---
${this.compiledSource}`);const{arrayArgs:v,bufferConstants:C,byteLength:b}=this.paramsLayout,T=[{binding:0,visibility:4,buffer:{type:"uniform"}}];for(let O=0;O<v.length;O++)T.push({binding:1+O,visibility:4,buffer:{type:"read-only-storage"}});const h=1+v.length;T.push({binding:h,visibility:4,buffer:{type:"storage"}});for(let O=0;O<C.length;O++)T.push({binding:h+1+O,visibility:4,buffer:{type:"read-only-storage"}});this.bindGroupLayout=l.createBindGroupLayout({entries:T}),l.pushErrorScope("validation"),this.computePipeline=l.createComputePipeline({layout:l.createPipelineLayout({bindGroupLayouts:[this.bindGroupLayout]}),compute:{module:x,entryPoint:"main"}});const F=await l.popErrorScope();if(F)throw new Error(`Error creating WebGPU compute pipeline for kernel: ${F.message}`);this.paramsBuffer=l.createBuffer({size:b,usage:72}),this.paramsMirror=new ArrayBuffer(b),this.paramsU32=new Uint32Array(this.paramsMirror),this.paramsI32=new Int32Array(this.paramsMirror),this.paramsF32=new Float32Array(this.paramsMirror),this.constantBuffers=[];for(let O=0;O<C.length;O++){const z=C[O],L=this.constants[z.constantName],V=this.constantDimensions(L),U=V[0]*V[1]*V[2];this._checkBufferSize(U*4,`constant "${z.constantName}"`);const X=l.createBuffer({size:Math.max(U*4,4),usage:P,mappedAtCreation:!0}),q=new Float32Array(X.getMappedRange());E.flattenTo(L instanceof $?L.value:L,q.subarray(0,U)),X.unmap(),z.buffer=X,this.constantBuffers.push(X)}this._ensureOutputBuffer(),this.bindGroupDirty=!0,this.built=!0}_computeDispatch(o){const[l,x,w]=this.workgroupSize,v=[Math.ceil(o[0]/l),Math.ceil(o[1]/x),Math.ceil(o[2]/w)],C=this._device.limits.maxComputeWorkgroupsPerDimension;let b=0;this.output.length===1&&(v[0]>C&&(v[1]=Math.ceil(v[0]/C),v[0]=Math.ceil(v[0]/v[1])),b=v[0]*l);for(let T=0;T<3;T++)if(v[T]>C)throw new Error(`output dimension ${T} needs ${v[T]} workgroups, over this device's limit of ${C}`);return{groups:v,dispatchWidth:b}}_ensureOutputBuffer(){const[o,l,x]=this.threadDim,w=o*l*x*4*this.componentCount;this.immutable&&this.pipeline&&this.outputBuffer&&(--this.outputBuffer._refs===0&&this.outputBuffer.destroy(),this.outputBuffer=null,this.bindGroupDirty=!0),!(this.outputBuffer&&this.outputBuffer.size>=w)&&(this.outputBuffer&&--this.outputBuffer._refs===0&&this.outputBuffer.destroy(),this._checkBufferSize(w,`output [${this.output.join(", ")}]`),this.outputBuffer=this._device.createBuffer({size:w,usage:132}),this.outputBuffer._refs=1,this.bindGroupDirty=!0)}_checkBufferSize(o,l){const x=this._device.limits,w=Math.min(x.maxStorageBufferBindingSize,x.maxBufferSize);if(o>w)throw new Error(`WebGPU backend: ${l} needs ${o} bytes but this device allows ${w} per storage buffer (maxStorageBufferBindingSize/maxBufferSize); reduce the output or split the work across kernels`)}_snapshotArguments(o){const l=new Array(o.length);for(let x=0;x<o.length;x++){const w=o[x],v=this.argumentTypes[x];if(w instanceof d){l[x]={kind:"buffer",handle:w};continue}switch(v){case"Array":{const C=Array.from(E.getDimensions(w));for(;C.length<3;)C.push(1);const b=new Float32Array(C[0]*C[1]*C[2]);E.flattenTo(w,b),l[x]={kind:"array",dims:C,flat:b};break}case"Input":{const C=Array.from(w.size);for(;C.length<3;)C.push(1);const b=new Float32Array(C[0]*C[1]*C[2]);E.flattenTo(w.value,b),l[x]={kind:"array",dims:C,flat:b};break}case"WebGPUBuffer":l[x]={kind:"buffer",handle:w};break;case"Boolean":l[x]={kind:"scalar",value:w?1:0};break;default:l[x]={kind:"scalar",value:w}}}return l}run(){!this.built&&!this._buildPromise&&this.build.apply(this,arguments);const o=this._snapshotArguments(arguments);return this.built?this._runInternal(o):this._buildPromise.then(()=>this._runInternal(o))}_runInternal(o){if(this.context&&this.context.isLost)throw new Error("WebGPU device was lost; call kernel.destroy() (or gpu.destroy()) and run again to rebuild on a fresh device");const l=this._device,x=l.queue,{arrayArgs:w,scalarArgs:v,bufferConstants:C}=this.paramsLayout,b=this.threadDim=Array.from(this.output);for(;b.length<3;)b.push(1);this._ensureOutputBuffer(),this.paramsU32[0]=b[0],this.paramsU32[1]=b[1],this.paramsU32[2]=b[2],this.paramsU32[3]=this._computeDispatch(b).dispatchWidth;for(let V=0;V<w.length;V++){const U=w[V],X=o[U.index];let q;if(X.kind==="buffer"){const ee=X.handle;if(ee._deleted)throw new Error(`WebGPUBufferResult passed as argument "${this.argumentNames[U.index]}" has been deleted`);if(ee.context!==this.context)throw new Error(`WebGPUBufferResult passed as argument "${this.argumentNames[U.index]}" is from a different WebGPU device`);if(ee.buffer===this.outputBuffer)throw new Error(`WebGPUBufferResult passed as argument "${this.argumentNames[U.index]}" is this kernel's own output buffer; use a second kernel or clone the result`);if(ee.componentCount!==1)throw new Error(`WebGPU backend does not yet support Array(${ee.componentCount}) pipeline results as kernel arguments`);for(q=Array.from(ee.output);q.length<3;)q.push(1);U.boundBuffer!==ee.buffer&&(U.boundBuffer=ee.buffer,this.bindGroupDirty=!0)}else{q=X.dims;const ee=X.flat.byteLength;if(!U.buffer||U.buffer.size<ee){if(U.buffer){if(!this.dynamicArguments)throw new Error(`argument "${this.argumentNames[U.index]}" grew from ${U.buffer.size/4} to ${X.flat.length} values; use dynamicArguments: true for varying input sizes`);U.buffer.destroy()}this._checkBufferSize(ee,`argument "${this.argumentNames[U.index]}"`),U.buffer=l.createBuffer({size:ee,usage:136}),this.bindGroupDirty=!0}x.writeBuffer(U.buffer,0,X.flat),U.boundBuffer!==U.buffer&&(U.boundBuffer=U.buffer,this.bindGroupDirty=!0)}const W=U.dimsOffset/4;this.paramsU32[W]=q[0],this.paramsU32[W+1]=q[1],this.paramsU32[W+2]=q[2],this.paramsU32[W+3]=q[0]*q[1]*q[2]}for(let V=0;V<v.length;V++){const U=v[V],X=U.offset/4;switch(U.type){case"Integer":this.paramsI32[X]=o[U.index].value;break;case"Boolean":this.paramsU32[X]=o[U.index].value;break;default:this.paramsF32[X]=o[U.index].value}}if(x.writeBuffer(this.paramsBuffer,0,this.paramsMirror),this.bindGroupDirty){const V=[{binding:0,resource:{buffer:this.paramsBuffer}}];for(let X=0;X<w.length;X++)V.push({binding:1+X,resource:{buffer:w[X].boundBuffer}});const U=1+w.length;V.push({binding:U,resource:{buffer:this.outputBuffer}});for(let X=0;X<C.length;X++)V.push({binding:U+1+X,resource:{buffer:C[X].buffer}});this.bindGroup=l.createBindGroup({layout:this.bindGroupLayout,entries:V}),this.bindGroupDirty=!1}const{groups:T}=this._computeDispatch(b),h=b[0]*b[1]*b[2]*4*this.componentCount,F=l.createCommandEncoder(),O=F.beginComputePass();if(O.setPipeline(this.computePipeline),O.setBindGroup(0,this.bindGroup),O.dispatchWorkgroups(T[0],T[1],T[2]),O.end(),this.pipeline)return x.submit([F.finish()]),Promise.resolve(new d({buffer:this.outputBuffer,output:Array.from(this.output),componentCount:this.componentCount,context:this.context,kernel:this}));const z=this._acquireStaging(h);F.copyBufferToBuffer(this.outputBuffer,0,z.buffer,0,h),x.submit([F.finish()]);const L=Array.from(this.output);return z.buffer.mapAsync(y,0,h).then(()=>{const V=new Float32Array(z.buffer.getMappedRange(0,h).slice(0));return z.buffer.unmap(),this._releaseStaging(z),this._shapeOutput(V,L,this.componentCount)},V=>{throw this._releaseStaging(z),V})}_acquireStaging(o){for(let x=0;x<this.stagingPool.length;x++){const w=this.stagingPool[x];if(!w.busy&&w.size>=o)return w.busy=!0,w}const l={buffer:this._device.createBuffer({size:o,usage:9}),size:o,busy:!0,pooled:this.stagingPool.length<3};return l.pooled&&this.stagingPool.push(l),l}_releaseStaging(o){o.pooled?o.busy=!1:o.buffer.destroy()}_shapeOutput(o,l,x){const[w,v,C]=[l[0],l[1]||1,l[2]||1];if(x===1)switch(l.length){case 1:return E.erectMemoryOptimizedFloat(o,w);case 2:return E.erectMemoryOptimized2DFloat(o,w,v);default:return E.erectMemoryOptimized3DFloat(o,w,v,C)}const b=x,T=h=>{const F=new Array(w);for(let O=0;O<w;O++)F[O]=o.subarray(h+O*b,h+O*b+b);return F};switch(l.length){case 1:return T(0);case 2:{const h=new Array(v);for(let F=0;F<v;F++)h[F]=T(F*w*b);return h}default:{const h=new Array(C);for(let F=0;F<C;F++){const O=new Array(v);for(let z=0;z<v;z++)O[z]=T((F*v+z)*w*b);h[F]=O}return h}}}readBufferResult(o){const l=this._device||o.context&&o.context.device;if(!l)return Promise.reject(new Error("no WebGPU device available to read this buffer"));if(o.context&&o.context.isLost)return Promise.reject(new Error("WebGPU device was lost; this buffer no longer holds data — rebuild the producing kernel and run again"));const x=Array.from(o.output),w=Array.from(x);for(;w.length<3;)w.push(1);const v=w[0]*w[1]*w[2]*4*o.componentCount,C=this._acquireStaging(v),b=l.createCommandEncoder();return b.copyBufferToBuffer(o.buffer,0,C.buffer,0,v),l.queue.submit([b.finish()]),C.buffer.mapAsync(y,0,v).then(()=>{const T=new Float32Array(C.buffer.getMappedRange(0,v).slice(0));return C.buffer.unmap(),this._releaseStaging(C),this._shapeOutput(T,x,o.componentCount)},T=>{throw this._releaseStaging(C),T})}destroy(o){if(this.paramsBuffer&&(this.paramsBuffer.destroy(),this.paramsBuffer=null),this.paramsLayout)for(let l=0;l<this.paramsLayout.arrayArgs.length;l++){const x=this.paramsLayout.arrayArgs[l];x.buffer&&(x.buffer.destroy(),x.buffer=null),x.boundBuffer=null}if(this.constantBuffers){for(let l=0;l<this.constantBuffers.length;l++)this.constantBuffers[l].destroy();this.constantBuffers=null}for(let l=0;l<this.stagingPool.length;l++)this.stagingPool[l].buffer.destroy();if(this.stagingPool=[],this.outputBuffer&&(--this.outputBuffer._refs===0&&this.outputBuffer.destroy(),this.outputBuffer=null),this.bindGroup=null,this.bindGroupLayout=null,this.computePipeline=null,this.built=!1,this._buildPromise=null,this.gpu&&this.gpu.kernels){const l=this.gpu.kernels.indexOf(this);l!==-1&&this.gpu.kernels.splice(l,1)}}};G.exports={WebGPUKernel:k}}),No=s((j,G)=>{const{utils:D}=m(),{Input:I}=a();function _(d){function $(v){d.build.apply(d,v),d.checkArgumentTypes(v);let C=d.switchingKernels?void 0:d.run.apply(d,v);for(let b=0;d.switchingKernels;b++){if(b>=4){const F=d.resetSwitchingKernels();throw new Error(`this kernel cannot run the arguments it was given (${P(F)}); it did not settle on a kernel for them after 4 attempts. Create a separate kernel for this call's argument types.`)}const T=d.resetSwitchingKernels(),h=d.onRequestSwitchKernel(T,v,d);w.kernel=d=h,h.checkArgumentTypes(v),C=h.switchingKernels?void 0:h.run.apply(h,v)}return C}function P(v){return!v||!v.length?"unknown reason":v.map(C=>C.type==="argumentTypeMismatch"?`argument ${C.index} is now ${C.needed}`:C.type).join(", ")}function y(v){const C=$(v);return d.renderKernels?d.renderKernels():d.renderOutput?d.renderOutput():C}function p(v){if(d.onAsyncModeUpgrade){const C=d.onAsyncModeUpgrade;d.onAsyncModeUpgrade=null;const b=o(v);return C(b,d).then(T=>(T&&w.replaceKernel(T),p(b)))}try{if(d.constructor.isAsync===!0)return d.build.apply(d,v),Promise.resolve(d.run.apply(d,v));for(let b=0;b<v.length;b++)if(g(v[b]))return k(v).then(T=>p(T));const C=$(v);return d.renderKernels?Promise.resolve(d.renderKernels()):d.renderOutput?d.renderOutputAsync?d.renderOutputAsync():Promise.resolve(d.renderOutput()):Promise.resolve(C)}catch(C){return Promise.reject(C)}}function g(v){return!!v&&v.type==="WebGPUBuffer"}function k(v){const C=o(v),b=[];for(let T=0;T<C.length;T++)if(g(C[T])){const h=T;b.push(Promise.resolve(C[h].toArray()).then(F=>{C[h]=F}))}return Promise.all(b).then(()=>C)}function o(v){const C=new Array(v.length);for(let b=0;b<v.length;b++)C[b]=l(v[b]);return C}function l(v){return!v||typeof v!="object"||g(v)||typeof v.delete=="function"?v:ArrayBuffer.isView(v)?v.slice(0):Array.isArray(v)?v.map(l):v instanceof I?new I(l(v.value),v.size):v}function x(){return d.constructor.isAsync===!0||d.asyncMode===!0?p(arguments):y(arguments)}const w=function(){return x.apply(d,arguments)};return w.exec=function(){return new Promise((v,C)=>{try{v(x.apply(this,arguments))}catch(b){C(b)}})},w.replaceKernel=function(v){d=v,c(d,w)},c(d,w),w}function c(d,E){if(E.kernel){E.kernel=d;return}const $=D.allPropertiesOf(d);for(let P=0;P<$.length;P++){const y=$[P];y[0]==="_"&&y[1]==="_"||(typeof d[y]=="function"?y.substring(0,3)==="add"||y.substring(0,3)==="set"?E[y]=function(){return E.kernel[y].apply(E.kernel,arguments),E}:E[y]=function(){return E.kernel[y].apply(E.kernel,arguments)}:(E.__defineGetter__(y,()=>E.kernel[y]),E.__defineSetter__(y,p=>{E.kernel[y]=p})))}E.kernel=d}G.exports={kernelRunShortcut:_}}),Bo=s((j,G)=>{const{gpuMock:D}=n(),{utils:I}=m(),{Kernel:_}=A(),{CPUKernel:c}=rt(),{HeadlessGLKernel:d}=jr(),{WebGL2Kernel:E}=si(),{WebGLKernel:$}=Fs(),{WebGPUKernel:P}=ai(),{kernelRunShortcut:y}=No(),p=[d,E,$],g=["gpu","cpu"],k={headlessgl:d,webgl2:E,webgl:$,webgpu:P};let o=!0;var l=class Za{static disableValidation(){o=!1}static enableValidation(){o=!0}static get isGPUSupported(){return p.some(v=>v.isSupported)}static get isKernelMapSupported(){return p.some(v=>v.isSupported&&v.features.kernelMap)}static get isOffscreenCanvasSupported(){return typeof Worker<"u"&&typeof OffscreenCanvas<"u"||typeof importScripts<"u"}static get isWebGLSupported(){return $.isSupported}static get isWebGL2Supported(){return E.isSupported}static get isHeadlessGLSupported(){return d.isSupported}static get isWebGPUSupported(){return P.isSupported}static isWebGPUAvailable(){return P.isSupported?navigator.gpu.requestAdapter().then(v=>v!==null,()=>!1):Promise.resolve(!1)}static get isCanvasSupported(){return typeof HTMLCanvasElement<"u"}static get isGPUHTMLImageArraySupported(){return E.isSupported}static get isSinglePrecisionSupported(){return p.some(v=>v.isSupported&&v.features.isFloatRead&&v.features.isTextureFloat)}constructor(v){if(v=v||{},this.canvas=v.canvas||null,this.context=v.context||null,this.mode=v.mode,this.Kernel=null,this.kernels=[],this.functions=[],this.nativeFunctions=[],this.injectedNative=null,this.mode!=="dev"){if(this.chooseKernel(),v.functions)for(let C=0;C<v.functions.length;C++)this.addFunction(v.functions[C]);if(v.nativeFunctions)for(const C in v.nativeFunctions){if(!v.nativeFunctions.hasOwnProperty(C))continue;const b=v.nativeFunctions[C],{name:T,source:h}=b;this.addNativeFunction(T,h,b)}}}chooseKernel(){if(this.Kernel)return;let v=null;if(this.context){for(let C=0;C<p.length;C++){const b=p[C];if(b.isContextMatch(this.context)){if(!b.isSupported)throw new Error(`Kernel type ${b.name} not supported`);v=b;break}}if(v===null)throw new Error("unknown Context")}else if(this.mode){if(this.mode in k)(!o||k[this.mode].isSupported)&&(v=k[this.mode]);else if(this.mode==="gpu"){for(let C=0;C<p.length;C++)if(p[C].isSupported){v=p[C];break}}else if(this.mode==="async"){for(let C=0;C<p.length;C++)if(p[C].isSupported){v=p[C];break}v||(v=c)}else this.mode==="cpu"&&(v=c);if(!v)throw new Error(`A requested mode of "${this.mode}" and is not supported`)}else{for(let C=0;C<p.length;C++)if(p[C].isSupported){v=p[C];break}v||(v=c)}this.mode||(this.mode=v.mode),this.Kernel=v}createKernel(v,C){if(typeof v>"u")throw new Error("Missing source parameter");if(typeof v!="object"&&!I.isFunction(v)&&typeof v!="string")throw new Error("source parameter not a function");const b=this.kernels;if(this.mode==="dev"){const U=D(v,x(C));return b.push(U),U}v=typeof v=="function"?v.toString():v;const T={},h=x(C)||{};C&&typeof C.argumentTypes=="object"&&(h.argumentTypes=Object.keys(C.argumentTypes).map(U=>C.argumentTypes[U]));function F(U){console.warn("Falling back to CPU");const X=new c(v,{argumentTypes:V.argumentTypes,constantTypes:V.constantTypes,graphical:V.graphical,loopMaxIterations:V.loopMaxIterations,constants:V.constants,dynamicOutput:V.dynamicOutput,dynamicArgument:V.dynamicArguments,output:V.output,precision:V.precision,pipeline:V.pipeline,immutable:V.immutable,optimizeFloatMemory:V.optimizeFloatMemory,fixIntegerDivisionAccuracy:V.fixIntegerDivisionAccuracy,functions:V.functions,nativeFunctions:V.nativeFunctions,injectedNative:V.injectedNative,subKernels:V.subKernels,strictIntegers:V.strictIntegers,randomSeed:V.randomSeed,debug:V.debug,asyncMode:V.asyncMode});X.build.apply(X,U);const q=X.run.apply(X,U);return V.replaceKernel(X),q}function O(U,X,q){q.debug&&console.warn("Switching kernels");let W=null;if(q.signature&&!T[q.signature]&&(T[q.signature]=q),q.dynamicOutput)for(let we=U.length-1;we>=0;we--){const re=U[we];re.type==="outputPrecisionMismatch"&&(W=re.needed)}const ee=q.constructor,se=ee.getArgumentTypes(q,X),Z=ee.getSignature(q,se),ie=T[Z];if(ie)return ie.onActivate(q),ie;const he=T[Z]=new ee(v,{argumentTypes:se,constantTypes:q.constantTypes,graphical:q.graphical,loopMaxIterations:q.loopMaxIterations,constants:q.constants,dynamicOutput:q.dynamicOutput,dynamicArgument:q.dynamicArguments,context:q.context,canvas:q.canvas,output:W||q.output,precision:q.precision,pipeline:q.pipeline,immutable:q.immutable,optimizeFloatMemory:q.optimizeFloatMemory,fixIntegerDivisionAccuracy:q.fixIntegerDivisionAccuracy,functions:q.functions,nativeFunctions:q.nativeFunctions,injectedNative:q.injectedNative,subKernels:q.subKernels,strictIntegers:q.strictIntegers,randomSeed:q.randomSeed,debug:q.debug,asyncMode:q.asyncMode,gpu:q.gpu,validate:o,returnType:q.returnType,tactic:q.tactic,onRequestFallback:F,onRequestSwitchKernel:O,texture:q.texture,mappedTextures:q.mappedTextures,drawBuffersMap:q.drawBuffersMap});return he.build.apply(he,X),V.replaceKernel(he),b.push(he),he}const z=Object.assign({context:this.context,canvas:this.canvas,functions:this.functions,nativeFunctions:this.nativeFunctions,injectedNative:this.injectedNative,gpu:this,validate:o,onRequestFallback:F,onRequestSwitchKernel:O},h);this.mode==="async"&&(z.asyncMode=!0);const L=new this.Kernel(v,z),V=y(L);if(this.mode==="async"&&P.isSupported&&!(L instanceof P)){const U=this;L.onAsyncModeUpgrade=function(q,W){return Za.isWebGPUAvailable().then(ee=>{if(!ee)return null;let se;try{se=new P(v,{functions:W.functions,nativeFunctions:W.nativeFunctions,injectedNative:W.injectedNative,gpu:U,validate:o,asyncMode:!0,output:W.output,pipeline:W.pipeline,immutable:W.immutable,dynamicOutput:W.dynamicOutput,dynamicArguments:!0,loopMaxIterations:W.loopMaxIterations,constants:W.constants,constantTypes:W.constantTypes,argumentTypes:W.argumentTypes,precision:W.precision,tactic:W.tactic,strictIntegers:W.strictIntegers,fixIntegerDivisionAccuracy:W.fixIntegerDivisionAccuracy,subKernels:W.subKernels,graphical:W.graphical,debug:W.debug}),se.build.apply(se,q)}catch(Z){return W.debug&&console.warn("webgpu upgrade declined: "+Z.message),null}return se._buildPromise.then(()=>(b.push(se),se),Z=>(W.debug&&console.warn("webgpu upgrade declined: "+Z.message),se.destroy(),null))},()=>null)}}return this.canvas||(this.canvas=L.canvas),this.context||(this.context=L.context),b.push(L),V}createKernelMap(){let v,C;const b=typeof arguments[arguments.length-2];if(b==="function"||b==="string"?(v=arguments[arguments.length-2],C=arguments[arguments.length-1]):v=arguments[arguments.length-1],this.mode!=="dev"&&(!this.Kernel.isSupported||!this.Kernel.features.kernelMap)){if(this.Kernel.mode==="webgpu")throw new Error("WebGPU backend does not yet support createKernelMap");if(this.mode&&g.indexOf(this.mode)<0)throw new Error(`kernelMap not supported on ${this.Kernel.name}`)}const T=x(C);if(C&&typeof C.argumentTypes=="object"&&(T.argumentTypes=Object.keys(C.argumentTypes).map(h=>C.argumentTypes[h])),Array.isArray(arguments[0])){T.subKernels=[];const h=arguments[0];for(let F=0;F<h.length;F++){const O=h[F].toString(),z=I.getFunctionNameFromString(O);T.subKernels.push({name:z,source:O,property:F})}}else{T.subKernels=[];const h=arguments[0];for(let F in h){if(!h.hasOwnProperty(F))continue;const O=h[F].toString(),z=I.getFunctionNameFromString(O);T.subKernels.push({name:z||F,source:O,property:F})}}return this.createKernel(v,T)}combineKernels(){const v=arguments[0],C=arguments[arguments.length-1];if(this.mode==="async"||v.kernel.asyncMode)throw new Error("mode 'async' does not yet support combineKernels; chain kernels with `await` and pipeline mode instead");if(v.kernel.constructor.mode==="cpu")return C;if(v.kernel.constructor.mode==="webgpu")throw new Error("WebGPU backend does not yet support combineKernels; chain kernels with `await` and pipeline mode instead");const b=arguments[0].canvas,T=arguments[0].context,h=arguments.length-1;for(let F=0;F<h;F++)arguments[F].setCanvas(b).setContext(T).setPipeline(!0);return function(){const F=C.apply(this,arguments);return F.toArray?F.toArray():F}}setFunctions(v){return this.functions=v,this}setNativeFunctions(v){return this.nativeFunctions=v,this}addFunction(v,C){return this.functions.push({source:v,settings:C}),this}addNativeFunction(v,C,b){if(this.kernels.length>0)throw new Error('Cannot call "addNativeFunction" after "createKernels" has been called.');return this.nativeFunctions.push(Object.assign({name:v,source:C},b)),this}injectNative(v){return this.injectedNative=v,this}destroy(){return new Promise((v,C)=>{this.kernels||v(),setTimeout(()=>{try{const b=this.kernels.slice();for(let h=0;h<b.length;h++)b[h].destroy(!0);let T=b[0];T&&(T.kernel&&(T=T.kernel),T.constructor.destroyContext&&T.constructor.destroyContext(this.context))}catch(b){C(b)}v()},0)})}};function x(w){if(!w)return{};const v=Object.assign({},w);return w.hasOwnProperty("floatOutput")&&(I.warnDeprecated("setting","floatOutput","precision"),v.precision=w.floatOutput?"single":"unsigned"),w.hasOwnProperty("outputToTexture")&&(I.warnDeprecated("setting","outputToTexture","pipeline"),v.pipeline=!!w.outputToTexture),w.hasOwnProperty("outputImmutable")&&(I.warnDeprecated("setting","outputImmutable","immutable"),v.immutable=!!w.outputImmutable),w.hasOwnProperty("floatTextures")&&(I.warnDeprecated("setting","floatTextures","optimizeFloatMemory"),v.optimizeFloatMemory=!!w.floatTextures),v}G.exports={GPU:l,kernelOrder:p,kernelTypes:g}}),jo=s((j,G)=>{const{utils:D}=m();function I(_,c){const d=c.toString();return new Function(`return function ${_} (${D.getArgumentNamesFromString(d).join(", ")}) {
  ${D.getFunctionBodyFromString(d)}
}`)()}G.exports={alias:I}}),qo=s((j,G)=>{const{GPU:D}=Bo(),{alias:I}=jo(),{utils:_}=m(),{Input:c,input:d}=a(),{Texture:E}=f(),{FunctionBuilder:$}=N(),{FunctionNode:P}=Y(),{CPUFunctionNode:y}=Ae(),{CPUKernel:p}=rt(),{HeadlessGLKernel:g}=jr(),{WebGLFunctionNode:k}=yn(),{WebGLKernel:o}=Fs(),{kernelValueMaps:l}=Br(),{WebGL2FunctionNode:x}=qr(),{WebGL2Kernel:w}=si(),{kernelValueMaps:v}=ti(),{WGSLFunctionNode:C}=ni(),{WebGPUKernel:b}=ai(),{WebGPUContext:T}=ri(),{WebGPUBufferResult:h}=ii(),{GLKernel:F}=Ar(),{Kernel:O}=A(),{FunctionTracer:z}=H();G.exports={alias:I,CPUFunctionNode:y,CPUKernel:p,GPU:D,FunctionBuilder:$,FunctionNode:P,HeadlessGLKernel:g,Input:c,input:d,Texture:E,utils:_,WebGL2FunctionNode:x,WebGL2Kernel:w,webGL2KernelValueMaps:v,WebGLFunctionNode:k,WebGLKernel:o,webGLKernelValueMaps:l,WGSLFunctionNode:C,WebGPUKernel:b,WebGPUContext:T,WebGPUBufferResult:h,GLKernel:F,Kernel:O,FunctionTracer:z,plugins:{mathRandom:Dr()}}});return s((j,G)=>{const D=qo(),I=D.GPU;for(const c in D)D.hasOwnProperty(c)&&c!=="GPU"&&(I[c]=D[c]);I.GPU=I,typeof window<"u"&&_(window),typeof self<"u"&&_(self);function _(c){c.GPU&&c.GPU.prototype&&c.GPU.prototype.createKernel||Object.defineProperty(c,"GPU",{configurable:!0,get(){return I},set(){}})}G.exports=I})()})})(dn)),dn.exports}var Ds=Au();const Du=typeof document>"u",Pu=Du?"▸ sandbox: Web Worker — a runaway kernel can be stopped":"▸ sandbox: main thread (no Worker sandbox in use) — a runaway kernel cannot be stopped",zu=/^[A-Za-z_$][A-Za-z0-9_$]*$/;let Er=[];async function vr(){const e=Er;Er=[];for(const t of e)try{await t.destroy()}catch{}}function Mr(){try{return!!Ds.GPU.isGPUSupported}catch{return!1}}function Ou(e){try{const t=e.kernel;return t?typeof Ds.CPUKernel=="function"&&t instanceof Ds.CPUKernel?!0:!!(t.constructor&&t.constructor.name==="CPUKernel"):!1}catch{return!1}}function Ru(e){const t=typeof console<"u"?console:null,s=n=>(...i)=>{t&&t[n]&&t[n](...i),e({type:n==="warn"?"warn":n==="error"?"error":"log",time:Je(),text:i.map(a=>Tr(a)).join(" ")})};return{log:s("log"),info:s("info"),debug:s("debug"),warn:s("warn"),error:s("error")}}function Fu(e){return(Array.isArray(e)||ArrayBuffer.isView(e))&&e.length===4&&typeof e[0]=="number"}function Gu(e){return Array.isArray(e)&&e.length>0&&Array.isArray(e[0])&&e[0].length>0&&Fu(e[0][0])}function Lu(e,t){try{const s=e.kernel;if(!s||s.built||s.argumentTypes||(s.argumentNames||[]).length!==t.length)return;let i=!1;const a=t.map(f=>{if(Gu(f))return i=!0,"Array2D(4)";const m=Ds.utils.getVariableType(f,s.strictIntegers);return m==="Integer"?"Number":m});i&&e.setArgumentTypes(a)}catch{}}function Uu(e){if(typeof ImageData>"u")return e;let t=!1;const s=e.map(n=>n instanceof ImageData&&Array.isArray(n.plain)?(t=!0,n.plain):n);return t?s:e}function Pa(e,t,s){let n=!1;return new Proxy(e,{apply(a,f,m){a.lastArgs=m;const A=s==="cpu"?Uu(m):m;Lu(a,A);const N=Reflect.apply(a,f,A);if(!n){n=!0;try{const H=a.kernel,Y=H&&H.output?Array.from(H.output):null;if(Y&&Y.length){const Ae=Y.reduce((Ze,rt)=>Ze*rt,1);t({type:"system",time:Je(),text:`▸ kernel compiled · output ${Y.join("×")} · ${Ae.toLocaleString("en-US")} threads`})}}catch{}}return N}})}const pn=64,Vu=65536,za=5e3;function Ku(e){if(Array.isArray(e)){const t=e.map(n=>typeof n=="number"?Math.min(n,pn):n),s=n=>n.reduce((i,a)=>i*(typeof a=="number"?a:1),1);return{clamped:t,requestedThreads:s(e),clampedThreads:s(t)}}if(e&&typeof e=="object"){const t=["x","y","z"].filter(i=>typeof e[i]=="number");if(!t.length)return null;const s={...e};t.forEach(i=>{s[i]=Math.min(e[i],pn)});const n=i=>t.reduce((a,f)=>a*i[f],1);return{clamped:s,requestedThreads:n(e),clampedThreads:n(s)}}return null}function Oa(e,t){const s=e[1],n=s&&typeof s=="object"?Ku(s.output):null;if(!n)return t.unclamped=!0,e;t.requestedThreads=Math.max(t.requestedThreads,n.requestedThreads),t.clampedThreads=Math.max(t.clampedThreads,n.clampedThreads);const i=e.slice(2);return[e[0],{...s,output:n.clamped},...i]}async function Ra(e,t,s){try{return await fn(e,{mode:t,task:s,probe:!0})}catch{return null}}async function Nu(e,t,s){const n=await Ra(e,t,s);if(!n)return null;const i=n.probeStats||{};if(!n.ok||i.unclamped||!i.requestedThreads||!i.clampedThreads||i.clampedThreads>=i.requestedThreads||i.requestedThreads<=Vu)return null;const a=i.requestedThreads/i.clampedThreads;if(n.durationMs*a<=za)return null;const f=await Ra(e,t,s),m=f&&f.ok?Math.min(n.durationMs,f.durationMs):n.durationMs,A=m*a;return A<=za?null:{probeMs:m,estimateMs:A,threads:i.requestedThreads}}async function fn(e,{mode:t="auto",task:s,probe:n=!1,onLog:i}={}){await vr();const a=[],f=me=>{if(a.push(me),i)try{i(me)}catch{}},m=[],A={requestedThreads:0,clampedThreads:0,unclamped:!1};let N=null;n||f({type:"system",time:Je(),text:Pu});const H=Mr();let Y;t==="cpu"?(Y="cpu",f({type:"system",time:Je(),text:'▸ mode "cpu" → selected cpu'})):t==="gpu"?H?(Y="gpu",f({type:"system",time:Je(),text:'▸ mode "gpu" → selected gpu (WebGL)'})):(Y="cpu",f({type:"system",time:Je(),text:'▸ mode "gpu" requested but WebGL is unavailable here — falling back to cpu'})):(Y=H?"gpu":"cpu",f({type:"system",time:Je(),text:H?'▸ mode "auto" → selected gpu (WebGL)':'▸ mode "auto" → selected cpu (WebGL unavailable)'}));class Ae extends Ds.GPU{constructor(Ee={}){super({...Ee,mode:Y}),Er.push(this)}createKernel(...Ee){const Ue=super.createKernel(...n?Oa(Ee,A):Ee),Dt=Pa(Ue,f,Y);return m.push(Dt),Dt}createKernelMap(...Ee){const Ue=super.createKernelMap(...n?Oa(Ee,A):Ee),Dt=Pa(Ue,f,Y);return m.push(Dt),Dt}}const Ze=me=>{N=me||N,f({type:"canvas",time:Je(),text:`render: ${me&&me.constructor?me.constructor.name:"canvas"}`,canvas:me||null,snapshot:Sl(me)})},rt=Ru(f),cs={GPU:Ae,console:rt,render:Ze,utils:kr,mode:Y};if(!n){const me=await Nu(e,t,s);if(me){const Ue=`refused to run: this would take about ${Math.round(me.estimateMs/1e3)}s and freeze the page. A ${pn}×${pn} slice took ${me.probeMs.toFixed(0)} ms, and the kernel asks for ${me.threads.toLocaleString("en-US")} threads. That much work per thread usually means a kernel is handling a whole row or array where it should handle one value — check that every array is indexed down to a number before you do arithmetic on it.`;return f({type:"error",time:Je(),text:Ue}),await vr(),{ok:!1,error:{message:Ue},logs:a,kernels:[],canvas:null,resolvedMode:Y,durationMs:0,fellBackToCPU:!1,refusedAsTooSlow:!0}}await vr()}let jt=null;try{if(s&&typeof s.inputs=="function"){const me=s.inputs(kr)||{};for(const[Ee,Ue]of Object.entries(me))zu.test(Ee)&&(cs[Ee]=Ue)}}catch(me){jt=me}const De=performance.now();let At=null;try{if(jt)throw jt;const me=Object.keys(cs);await new Function(...me,`"use strict";
return (async () => {
${e}
})();`)(...me.map(Ue=>cs[Ue]))}catch(me){At={message:mn(me),stack:me&&me.stack?String(me.stack):void 0},f({type:"error",time:Je(),text:At.message})}const Ps=performance.now()-De;let hs=N;if(!hs)for(let me=m.length-1;me>=0;me--){const Ee=m[me];try{if(Ee.kernel&&Ee.kernel.graphical&&Ee.canvas){hs=Ee.canvas;break}}catch{}}const gn=m.some(Ou),ds=Y==="gpu"&&gn;return ds&&!n&&f({type:"warn",time:Je(),text:"▸ gpu.js could not compile this kernel for WebGL and ran it on the CPU backend instead. A graphical kernel always uses unsigned precision, which has no 2D pixel-array type, so an image built as a nested array (image[y][x] = [r, g, b, a]) can only run on the CPU. Pass this task's image through untouched — the images it hands you are ImageData, which both backends read as the same image[this.thread.y][this.thread.x] pixel."}),At||f({type:"ok",time:Je(),text:`✓ run complete in ${Ps.toFixed(1)} ms${ds?" (on the CPU backend)":""}`}),{ok:!At,error:At,logs:a,kernels:m,canvas:hs,resolvedMode:Y,durationMs:Ps,fellBackToCPU:ds,probeStats:n?A:void 0}}function Bu(e,t={}){const s=[],n=e.logs.map(a=>{if(!a.canvas&&!a.snapshot)return a;const{canvas:f,snapshot:m,...A}=a;return m&&m.bitmap?(s.push(m.bitmap),{...A,snapshot:m}):m?{...A,snapshot:m}:A}),i=e.canvas;return{result:{ok:e.ok,error:e.error||null,logs:n,canvasInfo:i&&i.width?{width:i.width,height:i.height}:null,kernelCount:(e.kernels||[]).length,resolvedMode:e.resolvedMode,durationMs:e.durationMs,fellBackToCPU:!!e.fellBackToCPU,refusedAsTooSlow:!!e.refusedAsTooSlow,...t},transfer:s}}function ju(e,t){const s=e.kernels||[];return{...e,task:t,kernel:s.length?s[s.length-1]:null,utils:kr,assert:Na,assertClose:Ba,getPixels(n){for(let a=s.length-1;a>=0;a--){const f=s[a];try{if(f.kernel&&f.kernel.graphical&&typeof f.getPixels=="function")return f.getPixels(n)}catch{}}const i=_l(e.canvas);if(i)return i;throw new Error("no graphical kernel or canvas to read pixels from")}}}async function qu(e,t){const s=[{tests:e.publicTests||[],isPrivate:!1},{tests:e.privateTests||[],isPrivate:!0}],n=[];for(const{tests:a,isPrivate:f}of s)for(const m of a){const A=ju(t,e),N=performance.now();let H=!0,Y;try{await m.run(A)}catch(Ae){H=!1,Y=mn(Ae)}n.push({name:m.name,private:f,passed:H,ms:performance.now()-N,error:Y})}const i=n.filter(a=>a.passed).length;return{results:n,passed:i,total:n.length,allPassed:i===n.length}}function Fa(e){return(e.kernels||[]).filter(t=>Array.isArray(t.lastArgs))}function Ga(e,t){try{if(t&&typeof t.toArray=="function"){t.toArray();return}const s=e.kernel;if(s&&s.graphical){const n=s.context;n&&typeof n.readPixels=="function"&&n.readPixels(0,0,1,1,n.RGBA,n.UNSIGNED_BYTE,new Uint8Array(4))}}catch{}}function La(e){for(const n of e)Ga(n,n(...n.lastArgs));const t=[],s=performance.now();for(;t.length<5&&performance.now()-s<250;){const n=performance.now();for(const i of e)Ga(i,i(...i.lastArgs));t.push(performance.now()-n)}return t.sort((n,i)=>n-i),t[Math.floor(t.length/2)]}async function Wu(e,t){try{const s=await fn(e,{mode:"cpu",task:t});if(!s.ok)return{error:s.error};const n=Fa(s);if(!n.length)return{error:{message:"nothing to benchmark — the code never invoked a kernel"}};const i=La(n);if(!Mr())return{gpuUnavailable:!0,cpuMs:i};const a=await fn(e,{mode:"gpu",task:t});if(!a.ok)return{gpuFailed:!0,cpuMs:i,error:a.error};const f=Fa(a);if(!f.length)return{gpuFailed:!0,cpuMs:i,error:{message:"the code never invoked a kernel in gpu mode"}};const m=La(f);return{cpuMs:i,gpuMs:m,ratio:m>0?i/m:1/0,fasterOn:m<=i?"gpu":"cpu",gpuRanOnCpu:!!a.fellBackToCPU}}catch(s){return{error:{message:mn(s)}}}}let $s=null,Hu=0;const Xu=300;function $r(e){if(!e)return null;const t=Mu(e.moduleId,e.taskNum);return t?t.task:null}function vt(e,t){self.postMessage(e,t&&t.length?t:void 0)}function Yu(e){let t=0;return s=>{t>=Xu||(t++,vt({id:e,kind:"log",log:{type:s.type,time:s.time,text:s.text}}))}}async function Ju(e){const t=$r(e.taskRef),s=await fn(e.code,{mode:e.mode,task:t,onLog:Yu(e.id)}),n=`run-${++Hu}`;$s={token:n,internal:s};const{result:i,transfer:a}=Bu(s,{runToken:n});vt({id:e.id,kind:"result",result:i},a)}async function Zu(e){const t=$r(e.taskRef);if(!t){vt({id:e.id,kind:"result",result:{unknownTask:!0}});return}if(!$s||e.runToken&&e.runToken!==$s.token){vt({id:e.id,kind:"result",result:{staleToken:!0}});return}const s=await qu(t,$s.internal);vt({id:e.id,kind:"result",result:s})}async function Qu(e){const t=$r(e.taskRef),s=await Wu(e.code,t);$s=null,vt({id:e.id,kind:"result",result:s})}self.onmessage=async e=>{const t=e.data||{};try{switch(t.kind){case"hello":vt({id:t.id,kind:"result",result:{gpuSupported:Mr(),sandbox:"worker"}});break;case"run":await Ju(t);break;case"tests":await Zu(t);break;case"benchmark":await Qu(t);break;default:vt({id:t.id,kind:"failed",error:{message:`unknown request "${t.kind}"`}})}}catch(s){vt({id:t.id,kind:"failed",error:{message:mn(s)}})}};
