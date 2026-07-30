const Oa=Math.PI*2;function zi(e=64){const t=new Array(e);for(let s=0;s<e;s++)t[s]=Math.sin(s/e*Oa);return t}function qs(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function La(e,t,s,n,i){const a=i.filter(([c])=>{let h=!1;for(let I=0;I<e;I++){if(!(Math.abs(t(I)-c(I))<=n))return!1;Math.abs(s(I)-c(I))>n&&(h=!0)}return h}).map(c=>c[1]);return a.length&&a.every(c=>c===a[0])?a[0]:null}function dl(e,t,s,n,i){const a=i.filter(([c])=>{let h=!1;for(let I=0;I<e;I++)for(let U=0;U<e;U++){const j=c(I,U);if(!(t[I]&&Math.abs(t[I][U]-j)<=n))return!1;Math.abs(s(I,U)-j)>n&&(h=!0)}return h}).map(c=>c[1]);return a.length&&a.every(c=>c===a[0])?a[0]:null}function Hs(e,t=64){return[[Math.sin(e),"you sampled Math.sin(this.thread.x) directly — the index counts samples, not radians"],[Math.sin(e/t*Math.PI),"that is half a cycle — a full turn is 2 * Math.PI, not Math.PI"],[Math.sin(e/t*360),"Math.sin takes radians, not degrees — a full turn is 2 * Math.PI, not 360"],[Math.sin(e/t),"the 2π factor is missing — x / 64 on its own spans about one radian, not a full cycle"],[Math.sin(e/(t-1)*Oa),"you divided by 63 instead of 64 — the cycle spans all 64 samples, so sample 64 is where it would repeat"]]}function Ri(e){return dl(8,e,(t,s)=>(s+t)%2,.001,[[(t,s)=>s%2,"the whole board is the parity of this.thread.x alone — vertical stripes; a checkerboard flips on both axes, so add this.thread.y"],[(t,s)=>t%2,"the whole board is the parity of this.thread.y alone — horizontal stripes; a checkerboard flips on both axes, so add this.thread.x"]])}function Rn(e,t){return La(64,s=>e[s],s=>s*t,.01,[[s=>s*Math.trunc(t),"every cell is this.thread.x times a truncated scale — the integer thread id on the left makes gpu.js compile an integer multiply; put the float first: scale * this.thread.x"],[s=>s,"the scale never reached the result — every cell is the bare this.thread.x"]])}var pl={id:"1-1",track:1,title:"Hello, Kernel",blurb:"What a kernel is, what a thread is, and why <code>this.thread.x</code> replaces your for-loop.",tasks:[{slug:"first-kernel",title:"Your First Kernel",intro:`<p>A <strong>kernel</strong> is an ordinary-looking JavaScript function with one twist:
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
`,publicTests:[{name:"result holds 32 values",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===32,`expected 32 output values, got ${t&&t.length}`)}},{name:"cell <code>i</code> holds <code>i</code> — each thread reports its index",run:async e=>{const t=e.kernel(),s=La(32,n=>t[n],n=>n,.001,[[n=>n+1,"every cell is one more than its index — this.thread.x already counts from 0, so the first cell holds 0"]]);for(let n=0;n<32;n++)e.assertClose(t[n],n,.001,s||`cell ${n}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel();let s=0;for(let n=0;n<t.length;n++)s+=t[n];e.assertClose(s,992/2,.01,"the indices should total 496"),e.assertClose(t[0],0,.001,"first thread is index 0"),e.assertClose(t[17],17,.001,"thread 17"),e.assertClose(t[31],31,.001,"last thread is index 31")}}]},{slug:"index-formula",title:"From For-Loop to Formula",intro:`<p>Here's the payoff of the thread index. On the CPU you'd sample a sine wave like
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
`,publicTests:[{name:"kernel produces 64 samples",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===64,`expected 64 samples, got ${t&&t.length}`)}},{name:"samples trace <code>sin(x / 64 · 2π)</code> — starts at 0, peaks at thread 16",run:async e=>{const t=e.kernel(),s=zi(64);e.assertClose(t[0],0,.001,"thread 0: sin(0) = 0");const n=qs(t[16],1,.001,Hs(16));e.assertClose(t[16],1,.001,n||"thread 16: quarter cycle, sin = 1");const i=qs(t[48],-1,.001,Hs(48));e.assertClose(t[48],-1,.001,i||"thread 48: three-quarter cycle, sin = -1");for(let a=0;a<64;a+=7){const c=qs(t[a],s[a],.001,Hs(a));e.assertClose(t[a],s[a],.001,c||`sample ${a}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(),s=zi(64);for(let n=0;n<64;n++){const i=qs(t[n],s[n],.001,Hs(n));e.assertClose(t[n],s[n],.001,i||`sample ${n}`)}}}]},{slug:"checkerboard",title:"A Second Dimension: this.thread.y",intro:`<p>Threads don't have to line up in a row. Give <code>output</code> two numbers —
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
`,publicTests:[{name:"result is an 8×8 grid — 8 rows of 8 values",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===8,`expected 8 rows, got ${t&&t.length}`),e.assert(t[0]&&typeof t[0]!="number"&&t[0].length===8,"each row should hold 8 values — is your output still 1D?")}},{name:"cells alternate like a checkerboard: <code>(x + y) % 2</code>",run:async e=>{const t=e.kernel(),s=Ri(t);e.assertClose(t[0][0],0,.001,s||"corner [0][0] is 0"),e.assertClose(t[0][1],1,.001,s||"its neighbour [0][1] is 1"),e.assertClose(t[1][0],1,.001,s||"its neighbour [1][0] is 1"),e.assertClose(t[7][7],0,.001,"far corner [7][7] is 0 (7 + 7 is even)")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(),s=Ri(t);for(let n=0;n<8;n++)for(let i=0;i<8;i++)e.assertClose(t[n][i],(i+n)%2,.001,s||`cell [${n}][${i}]`)}}]},{slug:"first-argument",title:"Pass Something In",intro:`<p>So far every kernel has conjured its output from thread coordinates alone. Real
        kernels also take <strong>arguments</strong> — declare a parameter on the kernel function,
        pass a value when you call it, and every thread sees that same value. Combine it with
        <code>this.thread.x</code> and each thread computes something different from shared
        input.</p>
        <p>Here's the payoff: a compiled kernel is <strong>reusable</strong>. Build it once, call it
        with <code>2.5</code>, call it again with <code>0.5</code> — two parallel launches, zero
        recompiles. That build-once/call-many rhythm is how all real GPU code is structured.
        One gpu.js habit to pick up now: <code>this.thread.x</code> is an <em>integer</em>, and
        gpu.js's transpiler types a <code>*</code>, <code>+</code> or <code>-</code> expression
        from its <strong>left</strong> operand (division always produces a float) — so write
        <code>scale * this.thread.x</code> (float first) to get float math. That's a quirk of
        this framework, not a GPU law: CUDA promotes mixed int/float math to float, and WGSL or
        GLSL refuse to compile the mix outright.</p>`,goal:`<strong>Goal:</strong> make <code>ramp</code> return <code>scale * this.thread.x</code>,
        then call it twice — once with <code>2.5</code>, once with <code>0.5</code>.`,requirements:["Give the kernel function a <code>scale</code> parameter","Multiply the shared argument by this thread's index — shared argument × thread identity","Keep the float on the left so gpu.js compiles a float multiply, not an integer one","Call the kernel twice with different scales (already wired up)"],hints:[{title:"Hint 1 — where arguments come from",body:`<p>Kernel arguments are ordinary function parameters:
            <code>function (scale) { … }</code>, called as <code>ramp(3)</code>. Every one of the
            64 threads receives the same <code>3</code>.</p>`},{title:"Hint 2 — the body",body:`<p><code>return scale * this.thread.x;</code> — the argument is shared, the
            index is per-thread, the product is different in every cell. Written the other way
            round (<code>this.thread.x * scale</code>) gpu.js's GL backend compiles an
            <em>integer</em> multiplication and truncates <code>scale</code> — a transpiler
            gotcha specific to gpu.js (CUDA would promote to float; WGSL would refuse the
            mixed types at compile time).</p>`}],transfer:`A value shared by all threads is a <em>uniform</em>: WebGPU binds it as a uniform
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
`,publicTests:[{name:"called with <code>2.5</code>, cell <code>i</code> holds <code>i * 2.5</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(2.5);e.assert(t&&t.length===64,`expected 64 output values, got ${t&&t.length}`);const s=Rn(t,2.5);for(let n=0;n<64;n++)e.assertClose(t[n],n*2.5,.01,s||`cell ${n} with scale 2.5`)}},{name:"the same kernel re-launches with <code>0.5</code> — no rebuild needed",run:async e=>{const t=e.kernel(.5),s=Rn(t,.5);for(let n=0;n<64;n++)e.assertClose(t[n],n*.5,.01,s||`cell ${n} with scale 0.5`)}}],privateTests:[{name:"private test #1",run:async e=>{const s=e.kernel(-2.25);e.assert(s.length===64,"expected 64 output values");const n=Rn(s,-2.25);for(let i=0;i<64;i++)e.assertClose(s[i],i*-2.25,.01,n||`cell ${i} with scale ${-2.25}`)}}]}]},fl=Object.freeze({__proto__:null,default:pl});const Mt=`<div class="layout-note">
  <b>Array layout in gpu.js</b>
  <p>Image data comes in row-major: <code>image[y][x]</code> is the pixel in row <em>y</em>,
    column <em>x</em>, and each pixel is an <code>[r, g, b, a]</code> array with channels from
    0 to 1. Mind the inversion that catches everyone — sizes are given width-first
    (<code>output: [width, height]</code>), but indexing runs row-first, so this thread's own
    pixel is <code>image[this.thread.y][this.thread.x]</code>. Swap those two and you read the
    transpose of your image. Three-dimensional data follows the same rule:
    <code>output: [w, h, d]</code> is indexed <code>[z][y][x]</code>.</p>
</div>`;function Fa(e){let t=e>>>0;return function(){t=t+1831565813>>>0;let n=t;return n=Math.imul(n^n>>>15,n|1),n^=n+Math.imul(n^n>>>7,n|61),((n^n>>>14)>>>0)/4294967296}}function _s(e){return e<0?0:e>1?1:e}function Kt(e){return Math.round(_s(e)*255)/255}function Bt(e){return[Kt(e[0]),Kt(e[1]),Kt(e[2]),Kt(e[3]===void 0?1:e[3])]}function us(e){const t=e.length,s=e[0].length,n=new Uint8ClampedArray(s*t*4);let i=0;for(let c=t-1;c>=0;c--){const h=e[c];for(let I=0;I<s;I++){const U=h[I];n[i++]=Math.round(_s(U[0])*255),n[i++]=Math.round(_s(U[1])*255),n[i++]=Math.round(_s(U[2])*255),n[i++]=Math.round(_s(U[3]===void 0?1:U[3])*255)}}const a=new ImageData(n,s,t);return Object.defineProperties(a,{plain:{value:e,enumerable:!1},at:{value:(c,h)=>e[h][c],enumerable:!1}}),a}function ml(e){const t=Fa(1735423278^e*2654435761),s=new Array(e);for(let n=0;n<e;n++){const i=new Array(e),a=n/e;for(let c=0;c<e;c++){const h=c/e;i[c]=[Kt(.2+.55*h+.25*t()),Kt(.2+.55*a+.25*t()),Kt(.15+.6*Math.abs(Math.sin(3.1*(h+a)))+.25*t()),1]}s[n]=i}return us(s)}function gl(e){const t=[],s=[e];for(;s.length;){const n=s.pop();if(Array.isArray(n)||ArrayBuffer.isView(n))for(let i=n.length-1;i>=0;i--)s.push(n[i]);else t.push(n)}return t}function Ga(e,t){if(!e)throw new Error(t||"assertion failed")}function Ua(e,t,s=1e-4,n){const i=n?`${n} — `:"";if(typeof e!="number"||Number.isNaN(e))throw new Error(`${i}expected a number close to ${t}, got ${e}`);if(Math.abs(e-t)>s)throw new Error(`${i}expected ${t} ± ${s}, got ${e}`)}const kr={seededRandom:Fa,makeTestImage:ml,flatten:gl,assert:Ga,assertClose:Ua};function Je(){const e=new Date,t=(s,n)=>String(s).padStart(n,"0");return`${t(e.getHours(),2)}:${t(e.getMinutes(),2)}:${t(e.getSeconds(),2)}.${t(e.getMilliseconds(),3)}`}function Tr(e,t=0){if(e===null)return"null";if(e===void 0)return"undefined";const s=typeof e;if(s==="string")return t===0?e:JSON.stringify(e);if(s==="number"||s==="boolean"||s==="bigint")return String(e);if(s==="function")return`ƒ ${e.name||"(anonymous)"}`;if(e instanceof Error)return`${e.name||"Error"}: ${e.message}`;if(ArrayBuffer.isView(e)){const n=Array.from(e.slice(0,8),a=>Tr(a,t+1)),i=e.length>8?", …":"";return`${e.constructor.name}(${e.length}) [${n.join(", ")}${i}]`}if(typeof ImageData<"u"&&e instanceof ImageData)return`ImageData(${e.width}×${e.height})`;if(Array.isArray(e)){if(t>=2)return`Array(${e.length})`;const n=e.slice(0,8).map(a=>Tr(a,t+1)),i=e.length>8?", …":"";return`[${n.join(", ")}${i}]`}if(typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement)return"HTMLCanvasElement";if(typeof OffscreenCanvas<"u"&&e instanceof OffscreenCanvas)return`OffscreenCanvas(${e.width}×${e.height})`;try{const n=JSON.stringify(e);return n&&n.length>200?`${n.slice(0,200)}…`:n||String(e)}catch{try{return String(e)}catch{return"[unprintable object]"}}}function mn(e){try{const t=e&&e.message;return String(t||e)}catch{return"unprintable error"}}function yl(e){try{if(!e)return null;const t=e.width,s=e.height;if(!t||!s)return null;if(typeof document<"u"){const n=document.createElement("canvas");return n.width=t,n.height=s,n.getContext("2d").drawImage(e,0,0),{url:n.toDataURL(),w:t,h:s}}if(typeof OffscreenCanvas<"u"){const n=new OffscreenCanvas(t,s);return n.getContext("2d").drawImage(e,0,0),{bitmap:n.transferToImageBitmap(),w:t,h:s}}return null}catch{return null}}function xl(e){if(!e||!e.width)return null;let t;if(typeof document<"u")t=document.createElement("canvas"),t.width=e.width,t.height=e.height;else if(typeof OffscreenCanvas<"u")t=new OffscreenCanvas(e.width,e.height);else return null;const s=t.getContext("2d");return s.drawImage(e,0,0),s.getImageData(0,0,t.width,t.height).data}const as=[.299,.587,.114];function pt(e){return as[0]*e[0]+as[1]*e[1]+as[2]*e[2]}function On(e,t=4201){const s=e.seededRandom(t),n=new Array(64);for(let i=0;i<64;i++)n[i]=Math.round(s()*1e3)/100;return n}function bl(e,t,s,n){return[t[n][0],t[n][s]].some(c=>Math.abs(e-pt(c))<=2/255)?"that value is the luminance of the transposed pixel — looks like this.thread.x and this.thread.y are swapped. Rows come first: image[this.thread.y][this.thread.x].":null}function Ln(e,t,s,n,i){return Math.abs(e-t)<=s?`that is the value for cell [${i}][${n}] — this.thread.x and this.thread.y are swapped. Rows come first: image[this.thread.y][this.thread.x]`:null}function Oi(e,t){const s=new Array(e).fill(Bt(t));return us(new Array(e).fill(s))}function Ws(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Va(e,t,s,n,i){const a=i.filter(([c])=>{let h=!1;for(let I=0;I<e;I++){if(!(Math.abs(t(I)-c(I))<=n))return!1;Math.abs(s(I)-c(I))>n&&(h=!0)}return h}).map(c=>c[1]);return a.length&&a.every(c=>c===a[0])?a[0]:null}function vl(e,t,s,n,i){const a=i.filter(([c])=>{let h=!1;for(let I=0;I<e;I++)for(let U=0;U<e;U++){const j=c(I,U);if(!(t[I]&&Math.abs(t[I][U]-j)<=n))return!1;Math.abs(s(I,U)-j)>n&&(h=!0)}return h}).map(c=>c[1]);return a.length&&a.every(c=>c===a[0])?a[0]:null}function Li(e,t){return Va(64,s=>e[s],s=>t[s]*2,.001,[[s=>t[s],"every cell is the element itself — the doubling never happened"],[s=>2*s,"every cell is twice the thread index, not twice the element — index the array with it: data[this.thread.x]"]])}function Fi(e){const t="a + 1 is missing — this.thread.x and this.thread.y both start at 0, so cell [y][x] holds (x + 1) * (y + 1)";return vl(16,e,(s,n)=>(n+1)*(s+1),.001,[[(s,n)=>n*s,t],[(s,n)=>(n+1)*s,t],[(s,n)=>n*(s+1),t],[(s,n)=>n+1+(s+1),"the coordinates were added, not multiplied — the cell is (x + 1) * (y + 1)"]])}function wl(e){return Va(128,t=>e[t],t=>t*t,.01,[[t=>t,"you returned the thread index itself, not its square — every cell is exactly this.thread.x"],[t=>2*t,"every cell is twice the index, not the index squared — x * x, not x * 2"]])}function Gi(e){return[[pt(e),"that is the weighted luminance — this map wants the plain average (r + g + b) / 3"],[e[0]+e[1]+e[2],"the three channels were summed but never divided by 3"],[(e[0]+e[1]+e[2]+e[3])/4,"alpha crept into the average — only r, g and b belong in it"]]}function Fn(e){return[[(e[0]+e[1]+e[2])/3,"that is the plain channel average — luminance weights the channels 0.299 R + 0.587 G + 0.114 B"],[as[2]*e[0]+as[1]*e[1]+as[0]*e[2],"the weights are in the wrong order — 0.299 belongs on red and 0.114 on blue"]]}function kl(e,t,s,n){const i=.00784313725490196,a=t.filter(c=>Math.abs(e-c[0])<=i&&Math.abs(s-c[0])>i&&Math.abs(n-c[0])>i).map(c=>c[1]);return a.length&&a.every(c=>c===a[0])?a[0]:null}function Tl(e,t,s){return e>=254/255&&Math.max(t,s)<.9?"that pixel is clamped to full white — this.color() takes 0–1 channels and the image already is 0–1, so scaling by 255 saturates everything":null}var Sl={id:"1-2",track:1,title:"Data In, Data Out",blurb:"Feeding arrays and images into kernels, shaping 1D/2D/3D output, and reading results back.",tasks:[{slug:"pass-an-array",title:"Pass an Array In",intro:`<p>Kernels don't reach out and grab data — data is <strong>handed to them</strong>
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
`,inputs:e=>({data:On(e)}),publicTests:[{name:"kernel returns 64 values — one per thread",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(On(e.utils));e.assert(t&&t.length===64,`expected 64 output values, got ${t&&t.length}`)}},{name:"every element is doubled: <code>data[i] * 2</code>",run:async e=>{const t=new Array(64);for(let i=0;i<64;i++)t[i]=i*1.5-10;const s=e.kernel(t),n=Li(s,t);for(let i=0;i<64;i++)e.assertClose(s[i],t[i]*2,.001,n||`element ${i}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=On(e.utils,777),s=e.kernel(t);e.assert(s.length===64,"expected 64 output values");const n=Li(s,t);for(let i=0;i<64;i++)e.assertClose(s[i],t[i]*2,.001,n||`element ${i}`)}}]},{slug:"output-shapes",title:"Shape the Output: 2D",intro:`<p><code>output</code> is not just a size — it's a <strong>shape</strong>.
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
`,publicTests:[{name:"result is a 16×16 grid — 16 rows of 16 values",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===16,`expected 16 rows, got ${t&&t.length}`),e.assert(t[0]&&typeof t[0]!="number"&&t[0].length===16,"each row should hold 16 values — is your output still 1D?")}},{name:"cell [y][x] equals <code>(x + 1) * (y + 1)</code>",run:async e=>{const t=e.kernel(),s=[[0,0,1],[2,3,12],[7,0,8],[0,7,8],[15,15,256]],n=Fi(t);for(const[i,a,c]of s)e.assertClose(t[i][a],c,.001,n||`cell [${i}][${a}]`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(),s=Fi(t);for(let n=0;n<16;n++)for(let i=0;i<16;i++)e.assertClose(t[n][i],(i+1)*(n+1),.001,s||`cell [${n}][${i}]`)}}]},{slug:"grayscale",title:"Grayscale, the GPU way",intro:`<p>On the CPU you'd loop over 262,144 pixels one by one. On the GPU, every pixel gets
        <strong>its own thread</strong> — the kernel body runs once per pixel, all at the same time.</p>
        ${Mt}`,goal:`<strong>Goal:</strong> write a graphical kernel that converts <code>image</code> to
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
`,inputs:e=>({inputImage:e.makeTestImage(512)}),publicTests:[{name:"returns a graphical canvas of size <code>512×512</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===512&&e.canvas.height===512,`expected a 512×512 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();e.assert(t.length===512*512*4,"pixel buffer should hold 512×512 RGBA values")}},{name:"pixel (0,0) matches <code>0.299r + 0.587g + 0.114b</code> within ±1/255",run:async e=>{const t=e.utils.makeTestImage(512).plain,n=e.getPixels()[0]/255,i=pt(t[0][0]),a=pt(t[511][0]),c=Math.abs(n-i)<=2/255||Math.abs(n-a)<=2/255,h=kl(n,[...Fn(t[0][0]),...Fn(t[511][0])],i,a);e.assert(c,bl(n,t,511,0)||Tl(n,i,a)||h||`corner pixel should be its luminance (got ${n.toFixed(3)}, expected ≈${i.toFixed(3)})`)}},{name:"output is monochrome — <code>r == g == b</code> for sampled pixels",run:async e=>{const t=e.getPixels();for(let s=0;s<t.length;s+=997*4){const n=t[s],i=t[s+1],a=t[s+2];e.assert(Math.abs(n-i)<=1&&Math.abs(i-a)<=1,`pixel at byte ${s} is not gray: rgb(${n}, ${i}, ${a})`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Oi(512,[.2,.4,.6,1]),s=pt(t.at(0,0))*255;e.kernel(t);const n=e.getPixels();for(let i=0;i<n.length;i+=4999*4)e.assertClose(n[i],s,2,`red at byte ${i}`),e.assertClose(n[i+1],s,2,`green at byte ${i}`),e.assertClose(n[i+2],s,2,`blue at byte ${i}`)}},{name:"private test #2",run:async e=>{const t=e.utils.makeTestImage(512);e.kernel(t);const s=e.getPixels();let n=0;for(let a=0;a<s.length;a+=4)n+=s[a];n/=s.length/4*255;let i=0;for(let a=0;a<512;a++)for(let c=0;c<512;c++)i+=pt(t.plain[a][c]);i/=512*512,e.assertClose(n,i,1.5/255,"mean luminance")}}]},{slug:"read-it-back",title:"Read the Results Back",intro:`<p>A kernel's return value doesn't stay on the GPU — invoking the kernel hands you the
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
`,publicTests:[{name:"kernel returns <code>x²</code> for each of 128 threads",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===128,`expected 128 values, got ${t&&t.length}`);const s=wl(t);for(let n=0;n<128;n++)e.assertClose(t[n],n*n,.01,s||`element ${n}`)}},{name:"the total <code>690880</code> is computed and logged",run:async e=>{const t=e.logs.some(s=>s.type==="log"&&s.text&&s.text.includes("690880"));e.assert(t,"log the sum with console.log — expected to see 690880 in the console output")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel();let s=0;for(let i=0;i<t.length;i++)s+=t[i];const n=Ws(s,690880,1,[[8128,"that total is the sum of the indices themselves — the kernel is returning this.thread.x, not its square"],[2*8128,"that total is the sum of twice each index — the kernel is doubling where it should square"]]);e.assertClose(s,690880,1,n||"sum of the kernel output")}}]},{slug:"image-as-data",title:"Images Are Just Arrays",intro:`<p>Task 3 painted pixels. But an image doesn't have to <em>stay</em> an image: in this
        course an image is a nested array — <code>photo[y][x]</code> is an <code>[r, g, b, a]</code>
        pixel with channels 0–1 — and a kernel can read it like any other array argument.</p>
        <p>Drop <code>graphical: true</code>, and the same per-pixel indexing produces
        <strong>numbers</strong> instead of colors: a measurement per pixel, ready for JavaScript.</p>
        ${Mt}`,goal:`<strong>Goal:</strong> compute a 64×64 brightness map of <code>photo</code> — each cell
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
`,inputs:e=>({photo:e.makeTestImage(64)}),publicTests:[{name:"produces a 64×64 brightness map",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(e.utils.makeTestImage(64));e.assert(t&&t.length===64,`expected 64 rows, got ${t&&t.length}`),e.assert(t[0]&&t[0].length===64,"each row should hold 64 values")}},{name:"each cell averages the channels — <code>(r + g + b) / 3</code>",run:async e=>{const t=e.utils.makeTestImage(64),s=e.kernel(t),n=t.plain,i=[[0,0],[10,3],[31,40],[63,63]];for(const[a,c]of i){const h=n[a][c],I=n[c][a],U=(h[0]+h[1]+h[2])/3,j=Ln(s[a][c],(I[0]+I[1]+I[2])/3,.002,a,c)||Ws(s[a][c],U,.002,Gi(h));e.assertClose(s[a][c],U,.002,j||`cell [${a}][${c}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.utils.makeTestImage(64),s=e.kernel(t),n=t.plain;for(let i=0;i<64;i++)for(let a=0;a<64;a++){const c=n[i][a],h=n[a][i],I=(c[0]+c[1]+c[2])/3,U=Ln(s[i][a],(h[0]+h[1]+h[2])/3,.002,i,a)||Ws(s[i][a],I,.002,Gi(c));e.assertClose(s[i][a],I,.002,U||`cell [${i}][${a}]`)}}}]},{slug:"two-kernels",title:"Put It Together: Two Kernels",intro:`<p>Everything from this module in one pipeline. Kernel one reads the
        <code>photo</code> and produces a 64×64 <strong>luminance map</strong> — pure numbers.
        That result comes back to JavaScript, and you pass it straight into kernel two, a
        <strong>graphical</strong> kernel that paints the map as a grayscale picture.</p>
        <p>Array in → numbers out → array in again → pixels out. Data flowing <em>through</em>
        kernels is the whole game (and module 1.4 shows how to keep that flow on the GPU).</p>
        ${Mt}`,goal:`<strong>Goal:</strong> finish both kernels — <code>luminance</code> returns
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
`,inputs:e=>({photo:e.makeTestImage(64)}),publicTests:[{name:"two kernels: a numeric pass and a graphical pass",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=e.kernels.find(n=>n.kernel&&n.kernel.graphical),s=e.kernels.find(n=>n.kernel&&!n.kernel.graphical);e.assert(s,"no numeric (non-graphical) kernel found"),e.assert(t,"no graphical kernel found")}},{name:"luminance pass: cell [y][x] = <code>0.299r + 0.587g + 0.114b</code>",run:async e=>{const t=e.kernels.find(c=>c.kernel&&!c.kernel.graphical);e.assert(t,"no numeric kernel found");const s=e.utils.makeTestImage(64),n=t(s),i=s.plain,a=[[0,0],[5,50],[33,12],[63,63]];for(const[c,h]of a){const I=Ln(n[c][h],pt(i[h][c]),.002,c,h)||Ws(n[c][h],pt(i[c][h]),.002,Fn(i[c][h]));e.assertClose(n[c][h],pt(i[c][h]),.002,I||`cell [${c}][${h}]`)}}},{name:"painted canvas is monochrome and <code>64×64</code>",run:async e=>{e.assert(e.canvas,"no canvas — did you call render(paint.canvas)?"),e.assert(e.canvas.width===64&&e.canvas.height===64,`expected a 64×64 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();for(let s=0;s<t.length;s+=244){const n=t[s],i=t[s+1],a=t[s+2];e.assert(Math.abs(n-i)<=1&&Math.abs(i-a)<=1,`pixel at byte ${s} is not gray: rgb(${n}, ${i}, ${a})`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(h=>h.kernel&&!h.kernel.graphical),s=e.kernels.find(h=>h.kernel&&h.kernel.graphical);e.assert(t&&s,"expected a numeric and a graphical kernel");const n=Oi(64,[.6,.2,.4,1]),i=pt(n.at(0,0))*255,a=t(n);s(a);const c=s.getPixels();for(let h=0;h<c.length;h+=596)e.assertClose(c[h],i,2,`red at byte ${h}`),e.assertClose(c[h+1],i,2,`green at byte ${h}`),e.assertClose(c[h+2],i,2,`blue at byte ${h}`)}}]}]},_l=Object.freeze({__proto__:null,default:Sl});function Gn(e,t=1303){const s=e.seededRandom(t),n=new Array(64);for(let i=0;i<64;i++)n[i]=Math.round((s()*50-10)*100)/100;return n}function He(e,t,s=2718){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++)i[a]=Math.round((1+n()*8)*100)/100;return i}function xs(e,t,s=917){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++){const c=new Array(t);for(let h=0;h<t;h++)c[h]=Math.round(n()*1e3)/1e3;i[a]=c}return i}function ls(e,t){return Math.max(0,Math.min(t-1,e))}function Un(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){let i=0;for(let a=-2;a<=2;a++)i+=e[ls(n+a,t)];s[n]=i/5}return s}function Es(e){const t=e.length;return e.map(s=>{const n=new Array(t);for(let i=0;i<t;i++)n[i]=(s[ls(i-1,t)]+s[i]+s[ls(i+1,t)])/3;return n})}function Is(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){const i=new Array(t);for(let a=0;a<t;a++)i[a]=(e[ls(n-1,t)][a]+e[n][a]+e[ls(n+1,t)][a])/3;s[n]=i}return s}function Xs(e,t){const s=e.kernels[0](t);return e.kernels[1](s)}function ze(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Ui(e){return[[e*9/5,"the + 32 offset is missing — °F = °C × 9/5 + 32"],[e*5/9+32,"that is the °F → °C ratio — this direction multiplies by 9/5, not 5/9"],[(e+32)*9/5,"you added 32 before scaling — the formula scales first, then adds"],[e,"that reading came back unconverted — the formula never ran on it"]]}function Ys(e,t){const s=[[e[t],"that is your own element — a gather reads the mirrored index, data[n − 1 − this.thread.x]"]],n=e[e.length-t];return Number.isFinite(n)&&s.push([n,"that is data[n − this.thread.x] — the last valid index is n − 1, so the mirror of i is n − 1 − i"]),s}function Vn(e,t){const s=e.length;return[[e[(t+1)%s],"that value came from your right — rotating right means pulling from the left, index this.thread.x − 1"],[e[t],"that is your own element — the shift has to be expressed as a read from the neighbor"]]}function Vi(e){return Number.isFinite(e)&&e!==0?null:"thread 0 read index −1 — % keeps its left operand's sign, so add n first: (this.thread.x − 1 + n) % n"}function Kn(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){let i=0;for(let a=0;a<5;a++)i+=e[ls(n+a,t)];s[n]=i/5}return s}function Js(e,t,s,n){return[[e[n],"every tap read your own element — the offset d − 2 never reached the index, so you averaged five copies of yourself"],[t[n],"the window is shifted right — the offset d − 2 is what centers the five taps on this.thread.x"],[5*s[n],"that is the window sum — a mean divides by 5"]]}function Nn(e){const t=Es(e),s=Is(e);return[[Es(t),"both passes blurred along x — the second one has to walk the column: clamp this.thread.y and read grid[j][this.thread.x]"],[Is(s),"both passes blurred along y — the first one has to walk the row: clamp this.thread.x and read grid[this.thread.y][j]"],[s,"the x pass is a passthrough — only the y blur reached this cell"],[t,"the y pass is a passthrough — only the x blur reached this cell"]]}function Bn(e,t,s,n,i,a){if(!Number.isFinite(e))return"that cell read past the edge of the grid — clamp the index with Math.max(0, Math.min(n − 1, …))";const c=s.map(I=>[I[0][i][a],I[1]]),h="a pass returned the sum of its three taps without dividing by 3";return c.push([3*t[i][a],h],[9*t[i][a],h]),ze(e,t[i][a],n,c)}var Cl={id:"1-3",track:1,title:"Thinking in Parallel",blurb:"Map and gather patterns, why kernels write only their own cell, and how to design around it.",tasks:[{slug:"map-pattern",title:"Map: One Thread, One Value",intro:`<p>Nearly every GPU program you'll ever write is built from a handful of patterns,
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
`,inputs:e=>({celsius:Gn(e)}),publicTests:[{name:"converts all 64 readings — one output per thread",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(Gn(e.utils));e.assert(t&&t.length===64,`expected 64 output values, got ${t&&t.length}`)}},{name:"each cell is <code>c × 9/5 + 32</code> — 0 °C → 32 °F, 100 °C → 212 °F",run:async e=>{const t=new Array(64);for(let n=0;n<64;n++)t[n]=n*2-20;const s=e.kernel(t);for(let n=0;n<64;n++){const i=t[n]*9/5+32,a=ze(s[n],i,.01,Ui(t[n]));e.assertClose(s[n],i,.01,a||`reading ${n} (${t[n]} °C)`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Gn(e.utils,999),s=e.kernel(t);e.assert(s.length===64,"expected 64 output values");for(let n=0;n<64;n++){const i=t[n]*9/5+32,a=ze(s[n],i,.01,Ui(t[n]));e.assertClose(s[n],i,.01,a||`reading ${n}`)}}}]},{slug:"gather-pattern",title:"Gather: Read Anywhere",intro:`<p>A map reads its own cell. A <strong>gather</strong> reads <em>any</em> cell —
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
`,inputs:e=>({data:He(e,64,1101)}),publicTests:[{name:"the ends swap places — <code>out[0] = data[63]</code>, <code>out[63] = data[0]</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(64);for(let a=0;a<64;a++)t[a]=a+1;const s=e.kernel(t);e.assert(s&&s.length===64,`expected 64 output values, got ${s&&s.length}`);const n=ze(s[0],64,.001,Ys(t,0));e.assertClose(s[0],64,.001,n||"out[0] should hold the last input value");const i=ze(s[63],1,.001,Ys(t,63));e.assertClose(s[63],1,.001,i||"out[63] should hold the first input value")}},{name:"every cell mirrors: <code>out[i] = data[63 − i]</code>",run:async e=>{const t=new Array(64);for(let n=0;n<64;n++)t[n]=n*1.5+3;const s=e.kernel(t);for(let n=0;n<64;n++){const i=ze(s[n],t[63-n],.001,Ys(t,n));e.assertClose(s[n],t[63-n],.001,i||`element ${n}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=He(e.utils,64,4242),s=e.kernel(t);e.assert(s.length===64,"expected 64 output values");for(let n=0;n<64;n++){const i=ze(s[n],t[63-n],.001,Ys(t,n));e.assertClose(s[n],t[63-n],.001,i||`element ${n}`)}}}]},{slug:"invert-the-scatter",title:"No Scatter Allowed",intro:`<p>Here's the rule that shapes gpu.js kernels (and any fragment shader): a thread
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
`,inputs:e=>({ring:He(e,64,3301)}),publicTests:[{name:"values move one slot right: <code>out[i] = ring[i − 1]</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(64);for(let n=0;n<64;n++)t[n]=n*2+5;const s=e.kernel(t);e.assert(s&&s.length===64,`expected 64 output values, got ${s&&s.length}`);for(let n=1;n<64;n++){const i=ze(s[n],t[n-1],.001,Vn(t,n));e.assertClose(s[n],t[n-1],.001,i||`element ${n} should hold ring[${n-1}]`)}}},{name:"the first cell wraps around: <code>out[0] = ring[63]</code>",run:async e=>{const t=new Array(64);for(let i=0;i<64;i++)t[i]=i+10;const s=e.kernel(t),n=Vi(s[0])||ze(s[0],t[63],.001,Vn(t,0));e.assertClose(s[0],t[63],.001,n||"out[0] should hold the last input value")}}],privateTests:[{name:"private test #1",run:async e=>{const t=He(e.utils,64,8088),s=e.kernel(t);e.assert(s.length===64,"expected 64 output values");for(let n=0;n<64;n++){const i=t[(n-1+64)%64],a=(n===0?Vi(s[0]):null)||ze(s[n],i,.001,Vn(t,n));e.assertClose(s[n],i,.001,a||`element ${n}`)}}}]},{slug:"edges-and-clamps",title:"Life on the Edge",intro:`<p>The moment a gather reads a <em>neighbor</em>, the edges bite. Take the forward
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
`,inputs:e=>({signal:He(e,64,5150)}),publicTests:[{name:"interior cells hold the jump: <code>out[i] = signal[i+1] − signal[i]</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=He(e.utils,64,5150),s=e.kernel(t);e.assert(s&&s.length===64,`expected 64 output values, got ${s&&s.length}`);for(let n=0;n<63;n++){const i=t[n+1]-t[n],a=ze(s[n],i,.001,[[-i,"the sign is flipped — a forward difference is signal[i + 1] − signal[i], next minus current"],[t[n+1],"that is the neighbor's value, not the jump — subtract your own signal[this.thread.x]"]]);e.assertClose(s[n],i,.001,a||`element ${n}`)}}},{name:"the last cell clamps to <code>0</code> — no read past the end",run:async e=>{const t=He(e.utils,64,5150),s=e.kernel(t);e.assert(Number.isFinite(s[63]),`out[63] is ${s[63]} — an out-of-bounds read`),e.assertClose(s[63],0,1e-4,"the clamped edge cell should be exactly 0")}}],privateTests:[{name:"private test #1",run:async e=>{const t=He(e.utils,64,6006),s=e.kernel(t);for(let n=0;n<63;n++){const i=t[n+1]-t[n],a=ze(s[n],i,.001,[[-i,"the sign is flipped — a forward difference is signal[i + 1] − signal[i], next minus current"],[t[n+1],"that is the neighbor's value, not the jump — subtract your own signal[this.thread.x]"]]);e.assertClose(s[n],i,.001,a||`element ${n}`)}e.assertClose(s[63],0,1e-4,"last cell should clamp to 0")}}]},{slug:"moving-average",title:"Smooth a Signal",intro:`<p>Time to combine everything: a <strong>5-tap moving average</strong>. Each output
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
`,inputs:e=>({signal:He(e,128,7203)}),publicTests:[{name:"mid-signal cells average their five neighbors",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=He(e.utils,128,7203),s=e.kernel(t);e.assert(s&&s.length===128,`expected 128 output values, got ${s&&s.length}`);const n=Un(t),i=Kn(t);for(const a of[2,17,64,99,125]){const c=ze(s[a],n[a],.001,Js(t,i,n,a));e.assertClose(s[a],n[a],.001,c||`element ${a}`)}}},{name:"edge cells clamp — <code>out[0]</code> averages indexes 0, 0, 0, 1, 2",run:async e=>{const t=He(e.utils,128,7203),s=e.kernel(t),n=t.length,i=Un(t),a=Kn(t),c=ze(s[0],i[0],.001,Js(t,a,i,0));e.assertClose(s[0],(3*t[0]+t[1]+t[2])/5,.001,c||"left edge");const h=ze(s[n-1],i[n-1],.001,Js(t,a,i,n-1));e.assertClose(s[n-1],(3*t[n-1]+t[n-2]+t[n-3])/5,.001,h||"right edge")}}],privateTests:[{name:"private test #1",run:async e=>{const t=He(e.utils,128,9090),s=e.kernel(t),n=Un(t),i=Kn(t);for(let a=0;a<128;a++){const c=ze(s[a],n[a],.001,Js(t,i,n,a));e.assertClose(s[a],n[a],.001,c||`element ${a}`)}}}]},{slug:"two-pass-blur",title:"The Two-Pass Blur",intro:`<p>The payoff. A 3×3 box blur of a 2D grid needs nine reads per cell — but the box
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
`,inputs:e=>({heightmap:xs(e,48)}),publicTests:[{name:"two passes compose into a 48×48 grid",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=Xs(e,xs(e.utils,48));e.assert(t&&t.length===48,`expected 48 rows, got ${t&&t.length}`),e.assert(t[0]&&t[0].length===48,"each row should hold 48 values")}},{name:"interior cells equal the full 3×3 box average",run:async e=>{const t=xs(e.utils,48),s=Xs(e,t),n=Is(Es(t)),i=Nn(t);for(const[a,c]of[[1,1],[10,30],[24,24],[40,7],[46,46]]){const h=Bn(s[a][c],n,i,.002,a,c);e.assertClose(s[a][c],n[a][c],.002,h||`cell [${a}][${c}]`)}}},{name:"edges and corners clamp — no zero-padding creeping in",run:async e=>{const t=xs(e.utils,48),s=Xs(e,t),n=Is(Es(t)),i=Nn(t);for(const[a,c]of[[0,0],[0,47],[47,0],[47,47],[0,20],[20,0]]){const h=Bn(s[a][c],n,i,.002,a,c);e.assertClose(s[a][c],n[a][c],.002,h||`cell [${a}][${c}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=xs(e.utils,48,555),s=Xs(e,t),n=Is(Es(t)),i=Nn(t);for(let a=0;a<48;a++)for(let c=0;c<48;c++){const h=Bn(s[a][c],n,i,.002,a,c);e.assertClose(s[a][c],n[a][c],.002,h||`cell [${a}][${c}]`)}}}]}]},Al=Object.freeze({__proto__:null,default:Cl});const jn=[.299,.587,.114];function Cs(e){return jn[0]*e[0]+jn[1]*e[1]+jn[2]*e[2]}function qn(e,t,s,n,i){return Math.abs(e-t)<=s?`that is the value for cell [${i}][${n}] — this.thread.x and this.thread.y are swapped. Rows come first: image[this.thread.y][this.thread.x]`:null}function Rt(e){return e&&typeof e.toArray=="function"?e.toArray():e}function Ki(e,t=1701){const s=e.seededRandom(t),n=new Array(256);for(let i=0;i<256;i++)n[i]=Math.round(s()*100)/100;return n}function Hn(e,t=2026){const s=e.seededRandom(t),n=new Array(256);for(let i=0;i<256;i++)n[i]=Math.round(s()*1e3)/100;return n}function Zs(e,t,s){const n=new Array(e).fill(0);return n[t]=s,n}function Ni(e,t){const s=new Array(e).fill(Bt(t));return us(new Array(e).fill(s))}function Bi(e){const t=e.map(n=>{const i=n/10;return i*i}),s=new Array(t.length);for(let n=0;n<t.length;n++){const i=Math.max(n-1,0),a=Math.min(n+1,t.length-1);s[n]=(t[i]+t[n]+t[a])/3}return s}function ji(e,t){let s=e.slice();const n=s.length;for(let i=0;i<t;i++){const a=s.slice();for(let c=1;c<n-1;c++)a[c]=.25*s[c-1]+.5*s[c]+.25*s[c+1];s=a}return s}function qi(e){return e.plain.map(t=>t.map(Cs))}function Hi(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){s[n]=new Array(t);for(let i=0;i<t;i++){let a=0;for(let c=-1;c<=1;c++)for(let h=-1;h<=1;h++){const I=Math.min(Math.max(n+c,0),t-1),U=Math.min(Math.max(i+h,0),t-1);a+=e[I][U]}s[n][i]=a/9}}return s}function Qs(e){return Math.min(Math.max((e-.5)*2+.5,0),1)}function Sr(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Wi(e){return[[(e-.5)*2+.5,"that is the stretch without its clamp — Math.min / Math.max keep the result inside 0–1"],[e,"that is the luminance unchanged — the stretch never reached the return value"],[Math.min(Math.max(e*2,0),1),"the midpoint is missing — subtract 0.5 before doubling and add it back afterwards"]]}function Xi(e,t,s,n,i,a){return Number.isFinite(e)?Sr(e,t[i][a],n,[[s[i][a],"that is the unblurred luminance — the 3×3 average never happened"],[9*t[i][a],"that is the sum of the nine samples — a mean divides by 9"]]):"that cell read past the edge of the map — clamp yy and xx into 0…63 before indexing"}var El={id:"1-4",track:1,title:"Pipelines & Textures",blurb:"Chaining kernels so data stays on the GPU — the single biggest real-world speedup.",tasks:[{slug:"pipeline-on",title:"Flip On the Pipeline",intro:`<p>Until now, every kernel call ended the same way: the GPU finished computing,
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
`,inputs:e=>({signal:Ki(e)}),publicTests:[{name:"kernel is created with <code>pipeline: true</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.kernel.kernel&&e.kernel.kernel.pipeline===!0,"the kernel is not a pipeline kernel — add pipeline: true to its settings")}},{name:"boosted values read back correctly through <code>toArray()</code>",run:async e=>{const t=new Array(256);for(let n=0;n<256;n++)t[n]=n%100/100;const s=Rt(e.kernel(t));e.assert(s&&s.length===256,`expected 256 values, got ${s&&s.length}`);for(let n=0;n<256;n++)e.assertClose(s[n],Math.min(t[n]*1.5,1),.002,`sample ${n}`)}},{name:"the downloaded first sample is logged as 'first sample:'",run:async e=>{const t=e.logs.some(s=>s.type==="log"&&s.text&&s.text.includes("first sample"));e.assert(t,"expected a console.log('first sample:', values[0]) after the download")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ki(e.utils,8842),s=Rt(e.kernel(t));e.assert(s.length===256,"expected 256 values");for(let n=0;n<256;n++)e.assertClose(s[n],Math.min(t[n]*1.5,1),.002,`sample ${n}`)}}]},{slug:"chain-two-kernels",title:"Chain Kernels, Skip the Round Trip",intro:`<p>Here's the payoff of textures: a texture returned by one kernel can be passed
        <strong>straight into the next kernel</strong> as an argument. gpu.js binds the texture
        as the input — no download, no re-upload, no JavaScript in the middle. The data makes
        the whole trip without ever leaving the card.</p>
        <p>In module 1.2 you chained two kernels through JavaScript: the luminance map came
        back as arrays, then went up again for the second pass. Same chain below — except this
        time <code>luminance</code> is a pipeline kernel, and the second pass eats its texture
        directly.</p>
        ${Mt}`,goal:`<strong>Goal:</strong> finish the <code>contrast</code> kernel — stretch each
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
`,inputs:e=>({photo:e.makeTestImage(64)}),publicTests:[{name:"two kernels: a pipeline pass feeding a plain pass",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=e.kernels.find(n=>n.kernel&&n.kernel.pipeline),s=e.kernels.find(n=>n.kernel&&!n.kernel.pipeline);e.assert(t,"no pipeline kernel found — keep pipeline: true on luminance"),e.assert(s,"no plain kernel found — contrast should NOT be a pipeline kernel"),e.resolvedMode==="gpu"&&e.assert(s.lastArgs&&s.lastArgs[0]&&typeof s.lastArgs[0].toArray=="function","contrast should be fed the texture itself — no .toArray() in between")}},{name:"chained result: clamped <code>(l - 0.5) * 2 + 0.5</code> per cell",run:async e=>{const t=e.kernels.find(h=>h.kernel&&h.kernel.pipeline),s=e.kernels.find(h=>h.kernel&&!h.kernel.pipeline);e.assert(t&&s,"expected a pipeline kernel and a plain kernel");const n=e.utils.makeTestImage(64),i=s(t(n)),a=n.plain,c=[[0,0],[7,41],[32,32],[63,63]];for(const[h,I]of c){const U=Cs(a[h][I]),j=qn(i[h][I],Qs(Cs(a[I][h])),.003,h,I)||Sr(i[h][I],Qs(U),.003,Wi(U));e.assertClose(i[h][I],Qs(U),.003,j||`cell [${h}][${I}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(I=>I.kernel&&I.kernel.pipeline),s=e.kernels.find(I=>I.kernel&&!I.kernel.pipeline);e.assert(t&&s,"expected a pipeline kernel and a plain kernel");const n=Ni(64,[.8,.3,.5,1]),i=Cs(n.at(0,0)),a=Qs(i),c=s(t(n)),h=Wi(i);for(let I=0;I<64;I++)for(let U=0;U<64;U++){const j=Sr(c[I][U],a,.003,h);e.assertClose(c[I][U],a,.003,j||`cell [${I}][${U}]`)}}}]},{slug:"tollbooth",title:"toArray() Is a Tollbooth",intro:`<p>Here's the mental model that makes GPU code fast: computation on the card is
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
`,inputs:e=>({signal:Hn(e)}),publicTests:[{name:"stages 1–2 are pipeline kernels; the final stage is not",run:async e=>{e.assert(e.kernels.length>=3,`expected 3 kernels, found ${e.kernels.length}`);const[t,s,n]=e.kernels;e.assert(t.kernel&&t.kernel.pipeline===!0,"normalize should have pipeline: true"),e.assert(s.kernel&&s.kernel.pipeline===!0,"gamma should have pipeline: true"),e.assert(n.kernel&&!n.kernel.pipeline,"smooth should stay a plain kernel — its return IS the readback you want"),e.resolvedMode==="gpu"&&e.assert(s.lastArgs&&s.lastArgs[0]&&typeof s.lastArgs[0].toArray=="function","gamma should receive a texture from normalize, not an array")}},{name:"the numbers survive the refactor — chain output is unchanged",run:async e=>{const[t,s,n]=e.kernels,i=Hn(e.utils),a=n(s(t(i))),c=Bi(i);e.assert(a&&a.length===256,`expected 256 values, got ${a&&a.length}`);for(let h=0;h<256;h++)e.assertClose(a[h],c[h],.003,`sample ${h}`)}}],privateTests:[{name:"private test #1",run:async e=>{const[t,s,n]=e.kernels,i=Hn(e.utils,5150),a=n(s(t(i))),c=Bi(i);for(let h=0;h<256;h++)e.assertClose(a[h],c[h],.003,`sample ${h}`)}}]},{slug:"iterate-immutable",title:"Feedback Loops: immutable Textures",intro:`<p>Simulations don't run once — they <strong>step</strong>: the output of step
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
`,inputs:()=>({field:Zs(128,64,1)}),publicTests:[{name:"the stepping kernel opts into <code>immutable: true</code>",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=e.kernels.find(s=>s.kernel&&s.kernel.immutable);e.assert(t,"no immutable kernel found — the feedback loop needs immutable: true on step"),e.assert(t.kernel.pipeline===!0,"step should keep pipeline: true too")}},{name:"twelve steps match a reference diffusion of the spike",run:async e=>{const t=e.kernels.find(h=>h.kernel&&h.kernel.pipeline&&!h.kernel.immutable),s=e.kernels.find(h=>h.kernel&&h.kernel.immutable);e.assert(t&&s,"expected an upload kernel and an immutable step kernel");const n=Zs(128,64,1);let i=t(n);for(let h=0;h<12;h++)i=s(i);const a=Rt(i),c=ji(n,12);for(let h=0;h<128;h++)e.assertClose(a[h],c[h],.002,`cell ${h}`)}},{name:"heat is conserved — the field still sums to 1.0",run:async e=>{const t=e.kernels.find(c=>c.kernel&&c.kernel.pipeline&&!c.kernel.immutable),s=e.kernels.find(c=>c.kernel&&c.kernel.immutable);e.assert(t&&s,"expected an upload kernel and an immutable step kernel");let n=t(Zs(128,64,1));for(let c=0;c<12;c++)n=s(n);const i=Rt(n);let a=0;for(let c=0;c<128;c++)a+=i[c];e.assertClose(a,1,.01,"total heat in the field")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(I=>I.kernel&&I.kernel.pipeline&&!I.kernel.immutable),s=e.kernels.find(I=>I.kernel&&I.kernel.immutable);e.assert(t&&s,"expected an upload kernel and an immutable step kernel");const n=Zs(128,40,.75);let i=t(n);for(let I=0;I<12;I++)i=s(i);const a=Rt(i),c=ji(n,12);let h=0;for(let I=0;I<128;I++)e.assertClose(a[I],c[I],.002,`cell ${I}`),h+=a[I];e.assertClose(h,.75,.01,"total heat in the field")}}]},{slug:"photo-to-screen",title:"The Payoff: Photo to Screen, Zero Readbacks",intro:`<p>Time to cash in the whole module. In module 1.2's finale, a two-kernel chain
        hauled the luminance map down to JavaScript and back up again — two transfers it didn't
        need. This pipeline does more work with <em>fewer</em> transfers: photo →
        <strong>luminance</strong> → <strong>3×3 blur</strong> → <strong>painted canvas</strong>,
        and after the photo is uploaded, nothing comes back. The graphical kernel eats the blur
        texture and writes pixels; readbacks: zero.</p>
        <p>The missing piece is the blur. Each cell averages its 3×3 neighbourhood — two little
        loops over <code>dy</code>/<code>dx</code>, indices clamped to 0…63 so the edges don't
        read out of bounds. When it works, hit <strong>Benchmark</strong> and watch what
        keeping data on the card does to the gap.</p>
        ${Mt}`,goal:`<strong>Goal:</strong> implement the 3×3 box blur so the full three-pass pipeline —
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
`,inputs:e=>({photo:e.makeTestImage(64)}),publicTests:[{name:"three passes: two texture kernels feeding a graphical finale",run:async e=>{e.assert(e.kernels.length>=3,`expected 3 kernels, found ${e.kernels.length}`);const[t,s,n]=e.kernels;e.assert(t.kernel&&t.kernel.pipeline===!0,"luminance should have pipeline: true"),e.assert(s.kernel&&s.kernel.pipeline===!0,"blur should have pipeline: true"),e.assert(n.kernel&&n.kernel.graphical,"the third kernel should be graphical"),e.assert(e.canvas,"no canvas — did you call render(paint.canvas)?"),e.assert(e.canvas.width===64&&e.canvas.height===64,`expected a 64×64 canvas, got ${e.canvas.width}×${e.canvas.height}`),e.resolvedMode==="gpu"&&e.assert(n.lastArgs&&n.lastArgs[0]&&typeof n.lastArgs[0].toArray=="function","paint should be fed the blur texture directly — zero readbacks")}},{name:"blur pass: each cell is the mean of its 3×3 neighbourhood",run:async e=>{const[t,s]=e.kernels,n=e.utils.makeTestImage(64),i=Rt(s(t(n))),a=qi(n),c=Hi(a),h=[[32,32],[0,20],[63,20],[20,0],[20,63],[0,0],[10,47]];for(const[I,U]of h){const j=qn(i[I][U],c[U][I],.003,I,U)||Xi(i[I][U],c,a,.003,I,U);e.assertClose(i[I][U],c[I][U],.003,j||`cell [${I}][${U}]`)}}},{name:"painted canvas is monochrome",run:async e=>{const t=e.getPixels();e.assert(t.length===4096*4,"pixel buffer should hold 64×64 RGBA values");for(let s=0;s<t.length;s+=244){const n=t[s],i=t[s+1],a=t[s+2];e.assert(Math.abs(n-i)<=1&&Math.abs(i-a)<=1,`pixel at byte ${s} is not gray: rgb(${n}, ${i}, ${a})`)}}}],privateTests:[{name:"private test #1",run:async e=>{const[t,s,n]=e.kernels,i=Ni(64,[.35,.65,.15,1]),a=Cs(i.at(0,0))*255;n(s(t(i)));const c=n.getPixels();for(let h=0;h<c.length;h+=596)e.assertClose(c[h],a,2,`red at byte ${h}`),e.assertClose(c[h+1],a,2,`green at byte ${h}`),e.assertClose(c[h+2],a,2,`blue at byte ${h}`)}},{name:"private test #2",run:async e=>{const[t,s]=e.kernels,n=e.utils.makeTestImage(64),i=Rt(s(t(n))),a=qi(n),c=Hi(a);for(let h=0;h<64;h++)for(let I=0;I<64;I++){const U=qn(i[h][I],c[I][h],.004,h,I)||Xi(i[h][I],c,a,.004,h,I);e.assertClose(i[h][I],c[h][I],.004,U||`cell [${h}][${I}]`)}}}]}]},Il=Object.freeze({__proto__:null,default:El});function Tt(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++)i[a]=Math.round(n()*1e3)/100;return i}function ns(e){let t=0;for(let s=1;s<=1e3;s++)t+=1/(s+e);return t}function We(e,t){return e.logs.some(s=>s.type==="log"&&s.text&&s.text.includes(t))}function ot(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Yi(e){return[[Math.sin(e/100),"the amplitude is missing — the sample is Math.sin(x / 100) * 100"],[Math.sin(e)*100,"you sampled Math.sin(this.thread.x) — the index has to be divided by 100 first"],[Math.sin(e*100)*100,"the index is multiplied by 100 where it should be divided by it"]]}function Yt(e,t,s){return[e[t],`that is the element unchanged — the ${s} never happened`]}function Ji(e,t){return Ml(16,s=>e[s],s=>t[s]*2,.001,[[s=>2*s,"every cell is twice the thread index, not twice the element — index the array with it: data[this.thread.x]"]])}function Ml(e,t,s,n,i){const a=i.filter(([c])=>{let h=!1;for(let I=0;I<e;I++){if(!(Math.abs(t(I)-c(I))<=n))return!1;Math.abs(s(I)-c(I))>n&&(h=!0)}return h}).map(c=>c[1]);return a.length&&a.every(c=>c===a[0])?a[0]:null}function Zi(e){const t=[[1/(1+e),"each pass overwrote the running total — accumulate it with sum += inside the loop"],[ns(0)+1e3*e,"the parentheses are missing — each term is 1 / (k + this.thread.x), not 1 / k + this.thread.x"]];return e>0&&t.push([ns(e)+1/e,"the loop started at k = 0 — that extra 1 / this.thread.x term does not belong to the sum"]),t}var $l={id:"1-5",track:1,title:"Measuring Speed Honestly",blurb:"Warm-up, transfer costs, and precision — when the GPU wins, and when the CPU quietly beats it.",tasks:[{slug:"first-call-lie",title:"The First Call Is a Lie",intro:`<p>The first time you invoke a kernel, gpu.js does far more than run it: it
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
`,publicTests:[{name:"kernel computes <code>sin(x / 100) · 100</code> for all 2048 threads",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===2048,`expected 2048 output values, got ${t&&t.length}`);for(const s of[0,1,100,777,1023,2047]){const n=Math.sin(s/100)*100,i=ot(t[s],n,.05,Yi(s));e.assertClose(t[s],n,.05,i||`element ${s}`)}}},{name:"both timings are logged: <code>first call</code> and <code>warm call</code>",run:async e=>{e.assert(We(e,"first call"),"time the first call and log it — console.log('first call:', ms, 'ms')"),e.assert(We(e,"warm call"),"time 10 warmed-up calls and log the average — console.log('warm call:', avg, 'ms')")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel();e.assert(t.length===2048,"expected 2048 output values");for(let s=0;s<2048;s++){const n=Math.sin(s/100)*100,i=ot(t[s],n,.05,Yi(s));e.assertClose(t[s],n,.05,i||`element ${s}`)}}}]},{slug:"transfer-tax",title:"Pay the Transfer Tax",intro:`<p>A kernel call isn't just compute. Every invocation ships your input array from
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
`,inputs:e=>({small:Tt(e,1024,1101),big:Tt(e,65536,1102)}),publicTests:[{name:"two kernels exist: output sizes <code>1024</code> and <code>65536</code>",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=e.kernels.find(n=>n.kernel&&n.kernel.output&&n.kernel.output[0]===1024),s=e.kernels.find(n=>n.kernel&&n.kernel.output&&n.kernel.output[0]===65536);e.assert(t,"no kernel with output [1024] found"),e.assert(s,"no kernel with output [65536] found")}},{name:"every element comes back as <code>value + 1</code>",run:async e=>{const t=e.kernels.find(h=>h.kernel&&h.kernel.output&&h.kernel.output[0]===1024),s=e.kernels.find(h=>h.kernel&&h.kernel.output&&h.kernel.output[0]===65536);e.assert(t&&s,"expected kernels with outputs [1024] and [65536]");const n=Tt(e.utils,1024,2201),i=t(n);for(let h=0;h<1024;h++){const I=ot(i[h],n[h]+1,.001,[Yt(n,h,"+ 1")]);e.assertClose(i[h],n[h]+1,.001,I||`small element ${h}`)}const a=Tt(e.utils,65536,2202),c=s(a);for(let h=0;h<65536;h+=271){const I=ot(c[h],a[h]+1,.001,[Yt(a,h,"+ 1")]);e.assertClose(c[h],a[h]+1,.001,I||`big element ${h}`)}}},{name:"per-call cost logged for both payloads (<code>ms/call</code>)",run:async e=>{e.assert(We(e,"small:"),"log the small kernel's cost — the console.log is in the starter"),e.assert(We(e,"big:"),"log the big kernel's cost — the console.log is in the starter"),e.assert(We(e,"ms/call"),"timeKernel should return ms per call (did it return 0 forever?)")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(h=>h.kernel&&h.kernel.output&&h.kernel.output[0]===1024),s=e.kernels.find(h=>h.kernel&&h.kernel.output&&h.kernel.output[0]===65536);e.assert(t&&s,"expected kernels with outputs [1024] and [65536]");const n=Tt(e.utils,1024,3301),i=t(n);e.assert(i.length===1024,"small kernel should produce 1024 values");for(let h=0;h<1024;h++){const I=ot(i[h],n[h]+1,.001,[Yt(n,h,"+ 1")]);e.assertClose(i[h],n[h]+1,.001,I||`small element ${h}`)}const a=Tt(e.utils,65536,3302),c=s(a);e.assert(c.length===65536,"big kernel should produce 65536 values");for(let h=0;h<65536;h+=97){const I=ot(c[h],a[h]+1,.001,[Yt(a,h,"+ 1")]);e.assertClose(c[h],a[h]+1,.001,I||`big element ${h}`)}}}]},{slug:"two-answers",title:"Two Machines, Two Answers",intro:`<p>JavaScript numbers are 64-bit floats — about 16 significant digits. GPU shaders
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
`,publicTests:[{name:"all 64 partial sums match the float64 reference within <code>1e-3</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel();e.assert(t&&t.length===64,`expected 64 output values, got ${t&&t.length}`);for(let s=0;s<64;s++){const n=ot(t[s],ns(s),.001,Zi(s));e.assertClose(t[s],ns(s),.001,n||`partial sum for thread ${s}`)}}},{name:"verdict uses a tolerance — <code>close enough: true</code> is logged",run:async e=>{e.assert(We(e,"difference:"),"keep the difference log — it shows the float32/float64 drift"),e.assert(We(e,"close enough: true"),"compare with Math.abs(result[0] - ref) < 1e-3, not === — the verdict should log true")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel();for(let s=0;s<64;s++){const n=ot(t[s],ns(s),.001,Zi(s));e.assertClose(t[s],ns(s),.001,n||`partial sum for thread ${s}`)}for(let s=0;s<63;s++)e.assert(t[s]>t[s+1],`sum for thread ${s} should exceed thread ${s+1}`)}}]},{slug:"when-cpu-wins",title:"When the CPU Wins",intro:`<p>Sixteen numbers, doubled. The GPU <em>can</em> do it — but every kernel call pays
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
`,inputs:e=>({tiny:Tt(e,16,4404)}),publicTests:[{name:"kernel doubles all 16 values",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(16);for(let i=0;i<16;i++)t[i]=i*1.25-3;const s=e.kernel(t);e.assert(s&&s.length===16,`expected 16 output values, got ${s&&s.length}`);const n=Ji(s,t);for(let i=0;i<16;i++){const a=n||ot(s[i],t[i]*2,.001,[Yt(t,i,"doubling")]);e.assertClose(s[i],t[i]*2,.001,a||`element ${i}`)}}},{name:"results agree within tolerance — <code>match: true</code> is logged",run:async e=>{e.assert(We(e,"match: true"),"compare fromKernel and fromLoop with Math.abs(a - b) <= 1e-4 and log the verdict — expected 'match: true'")}},{name:"both contenders timed and a winner declared",run:async e=>{e.assert(We(e,"ms/round"),"time both contenders and log each as ms/round"),e.assert(We(e,"kernel:"),"log the kernel's time — console.log('kernel:  ', kernelMs, 'ms/round')"),e.assert(We(e,"plain js:"),"log the loop's time — console.log('plain js:', loopMs, 'ms/round')"),e.assert(We(e,"winner:"),"declare the faster contender with a 'winner:' log")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Tt(e.utils,16,5505),s=e.kernel(t);e.assert(s.length===16,"expected exactly 16 output values");const n=Ji(s,t);for(let i=0;i<16;i++){const a=n||ot(s[i],t[i]*2,.001,[Yt(t,i,"doubling")]);e.assertClose(s[i],t[i]*2,.001,a||`element ${i}`)}}}]}]},Dl=Object.freeze({__proto__:null,default:$l});function Jt(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++)i[a]=Math.round(n()*100-50)/10;return i}function ge(e,t,s,n){const i=e.seededRandom(n),a=new Array(t);for(let c=0;c<t;c++){const h=new Array(s);for(let I=0;I<s;I++)h[I]=Math.round(i()*100-50)/10;a[c]=h}return a}function Qi(e){const t=new Array(e);for(let s=0;s<e;s++){const n=new Array(e).fill(0);n[s]=1,t[s]=n}return t}function As(e,t){let s=0;for(let n=0;n<e.length;n++)s+=e[n]*t[n];return s}function yt(e,t){const s=e.length,n=t.length,i=t[0].length,a=new Array(s);for(let c=0;c<s;c++){const h=new Array(i);for(let I=0;I<i;I++){let U=0;for(let j=0;j<n;j++)U+=e[c][j]*t[j][I];h[I]=U}a[c]=h}return a}function Pl(e,t,s){const n=e.length,i=t[0].length,a=new Array(n);for(let c=0;c<n;c++){const h=new Array(i);for(let I=0;I<i;I++){let U=0;for(let j=0;j<s;j++)U+=e[c][j]*t[j][I];h[I]=U}a[c]=h}return a}function Xe(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Wn(e,t){const s=e.length-1;return[[e[0]*t[0],"that is only the first product — a dot product accumulates all 16 pairs in the loop"],[As(e,t)-e[s]*t[s],`the loop stopped one pair short — with k < ${s} the pair a[${s}]·b[${s}] never gets added`]]}function ea(e,t,s,n,i,a){return[[e[i][a]*t[i][a],"that is the elementwise product of the two cells — C[y][x] is the whole dot product of row y of A with column x of B"],[s[a][i],`that is cell [${a}][${i}] of the product — this.thread.y picks A's row and this.thread.x picks B's column`],[n[a][i],"both matrices were read with their indices swapped — the walks are a[this.thread.y][k] across the row and b[k][this.thread.x] down the column"]]}function bs(e,t,s,n,i){return n.map(a=>[Pl(e,t,a),`only the first ${a} of the ${s} shared terms were summed — ${i}`])}function vs(e,t,s){return e.map(n=>[n[0][t][s],n[1]])}function ta(e,t,s){return t<e.length?[[e[t][s],`that is the input cell [${t}][${s}] — a transpose reads with the indices swapped: m[this.thread.x][this.thread.y]`]]:[]}var zl={id:"2-1",track:2,title:"Matrix Multiply",blurb:"The canonical GPGPU workload: from naive triple loop to a kernel that scales.",tasks:[{slug:"dot-product",title:"One Cell, One Dot Product",intro:`<p>Matrix multiply is the workload GPUs were born for, and every cell of the result
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
`,inputs:e=>({a:Jt(e,16,1101),b:Jt(e,16,1102)}),publicTests:[{name:"output is a single cell — 1 value, not 16",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(16).fill(1),s=e.kernel(t,t);e.assert(s&&s.length===1,`expected 1 output value, got ${s&&s.length} — a dot product is one number`);const n=Xe(s[0],16,.01,Wn(t,t));e.assertClose(s[0],16,.01,n||"dot of two all-ones vectors should be 16")}},{name:"the sum is right: <code>Σ a[k]·b[k]</code>",run:async e=>{const t=Jt(e.utils,16,1101),s=Jt(e.utils,16,1102),n=e.kernel(t,s),i=Xe(n[0],As(t,s),.01,Wn(t,s));e.assertClose(n[0],As(t,s),.01,i||"dot product of the provided vectors")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Jt(e.utils,16,1177),s=Jt(e.utils,16,1178),n=e.kernel(t,s)[0],i=Xe(n,As(t,s),.01,Wn(t,s));e.assertClose(n,As(t,s),.01,i||"dot of fresh vectors");const a=new Array(16).fill(0);a[11]=1,e.assertClose(e.kernel(t,a)[0],t[11],.01,"dot with a basis vector picks a[11]")}}]},{slug:"full-matmul",title:"The Full Grid: Matrix × Matrix",intro:`<p>On the CPU, <code>C = A × B</code> is the classic triple loop: over rows, over
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
`,inputs:e=>({matA:ge(e,16,16,2101),matB:ge(e,16,16,2102)}),publicTests:[{name:"result is a 16×16 grid",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=ge(e.utils,16,16,2101),s=ge(e.utils,16,16,2102),n=e.kernel(t,s);e.assert(n&&n.length===16,`expected 16 rows, got ${n&&n.length}`),e.assert(n[0]&&n[0].length===16,"each row should hold 16 values")}},{name:"cells match the dot product of row × column",run:async e=>{const t=ge(e.utils,16,16,2101),s=ge(e.utils,16,16,2102),n=e.kernel(t,s),i=yt(t,s),a=yt(s,t),c=[[0,0],[3,12],[8,8],[15,1],[15,15]];for(const[h,I]of c){const U=Xe(n[h][I],i[h][I],.01,ea(t,s,i,a,h,I));e.assertClose(n[h][I],i[h][I],.01,U||`cell [${h}][${I}]`)}}},{name:"multiplying by the identity gives A back",run:async e=>{const t=ge(e.utils,16,16,2101),s=e.kernel(t,Qi(16));for(let n=0;n<16;n++)for(let i=0;i<16;i++)e.assertClose(s[n][i],t[n][i],.01,`cell [${n}][${i}] of A × I`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=ge(e.utils,16,16,2777),s=ge(e.utils,16,16,2778),n=e.kernel(t,s),i=yt(t,s),a=yt(s,t);for(let c=0;c<16;c++)for(let h=0;h<16;h++){const I=Xe(n[c][h],i[c][h],.01,ea(t,s,i,a,c,h));e.assertClose(n[c][h],i[c][h],.01,I||`cell [${c}][${h}]`)}}}]},{slug:"rectangular",title:"Rectangular: Three Different Sizes",intro:`<p>Square matrices hide a trap: every dimension is 16, so any loop bound "works".
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
`,inputs:e=>({rectA:ge(e,8,32,3101),rectB:ge(e,32,12,3102)}),publicTests:[{name:"result is 8 rows × 12 columns",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=ge(e.utils,8,32,3101),s=ge(e.utils,32,12,3102),n=e.kernel(t,s);e.assert(n&&n.length===8,`expected 8 rows, got ${n&&n.length}`),e.assert(n[0]&&n[0].length===12,`expected 12 columns, got ${n[0]&&n[0].length}`)}},{name:"every term counted — all 32 of the shared dimension",run:async e=>{const t=ge(e.utils,8,32,3101),s=ge(e.utils,32,12,3102),n=e.kernel(t,s),i=yt(t,s),a=bs(t,s,32,[12,8],"k has to run over the shared dimension — A's columns and B's rows, not the output's width or height"),c=[[0,0],[2,11],[5,6],[7,0],[7,11]];for(const[h,I]of c){const U=Xe(n[h][I],i[h][I],.02,vs(a,h,I));e.assertClose(n[h][I],i[h][I],.02,U||`cell [${h}][${I}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=ge(e.utils,8,32,3777),s=ge(e.utils,32,12,3778),n=e.kernel(t,s),i=yt(t,s),a=bs(t,s,32,[12,8],"k has to run over the shared dimension — A's columns and B's rows, not the output's width or height");for(let c=0;c<8;c++)for(let h=0;h<12;h++){const I=Xe(n[c][h],i[c][h],.02,vs(a,c,h));e.assertClose(n[c][h],i[c][h],.02,I||`cell [${c}][${h}]`)}}}]},{slug:"transpose",title:"Transpose: Swap the Axes",intro:`<p>Look back at the matmul loop: <code>b[k][x]</code> walks <em>down a column</em> —
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
`,inputs:e=>({matWide:ge(e,24,40,4101)}),publicTests:[{name:"shape flips: 24×40 in, 40×24 out",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=ge(e.utils,24,40,4101),s=e.kernel(t);e.assert(s&&s.length===40,`expected 40 rows, got ${s&&s.length}`),e.assert(s[0]&&s[0].length===24,`expected 24 columns, got ${s[0]&&s[0].length}`)}},{name:"cell [y][x] equals input [x][y]",run:async e=>{const t=ge(e.utils,24,40,4101),s=e.kernel(t),n=[[0,0],[0,23],[39,0],[17,5],[39,23]];for(const[i,a]of n){const c=Xe(s[i][a],t[a][i],.001,ta(t,i,a));e.assertClose(s[i][a],t[a][i],.001,c||`cell [${i}][${a}] should hold input [${a}][${i}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=ge(e.utils,24,40,4777),s=e.kernel(t);for(let n=0;n<40;n++)for(let i=0;i<24;i++){const a=Xe(s[n][i],t[i][n],.001,ta(t,n,i));e.assertClose(s[n][i],t[i][n],.001,a||`cell [${n}][${i}]`)}}}]},{slug:"any-size",title:"One Kernel, Any Size",intro:`<p>Every kernel so far had its size welded on: <code>output: [16, 16]</code>, loop
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
`,inputs:e=>({smallA:ge(e,8,8,5101),smallB:ge(e,8,8,5102),bigA:ge(e,48,48,5103),bigB:ge(e,48,48,5104)}),publicTests:[{name:"one kernel serves both sizes",run:async e=>{e.assert(e.kernels.length===1,`expected exactly 1 kernel to handle every size, found ${e.kernels.length}`)}},{name:"8×8 product is correct",run:async e=>{const t=ge(e.utils,8,8,5101),s=ge(e.utils,8,8,5102);e.kernel.setOutput([8,8]);const n=e.kernel(t,s,8),i=yt(t,s);for(let a=0;a<8;a++)for(let c=0;c<8;c++)e.assertClose(n[a][c],i[a][c],.02,`cell [${a}][${c}]`)}},{name:"48×48 product is correct — same kernel, bigger launch",run:async e=>{const t=ge(e.utils,48,48,5103),s=ge(e.utils,48,48,5104);e.kernel.setOutput([48,48]);const n=e.kernel(t,s,48);e.assert(n.length===48&&n[0].length===48,"expected a 48×48 result");const i=yt(t,s),a=bs(t,s,48,[8],"the loop bound has to be the size argument, not the literal the kernel was born with"),c=[[0,0],[7,33],[24,24],[40,3],[47,47]];for(const[h,I]of c){const U=Xe(n[h][I],i[h][I],.05,vs(a,h,I));e.assertClose(n[h][I],i[h][I],.05,U||`cell [${h}][${I}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=ge(e.utils,32,32,5777),s=ge(e.utils,32,32,5778);e.kernel.setOutput([32,32]);const n=e.kernel(t,s,32),i=yt(t,s),a=bs(t,s,32,[8],"the loop bound has to be the size argument, not the literal the kernel was born with");for(let c=0;c<32;c++)for(let h=0;h<32;h++){const I=Xe(n[c][h],i[c][h],.05,vs(a,c,h));e.assertClose(n[c][h],i[c][h],.05,I||`cell [${c}][${h}]`)}}},{name:"private test #2",run:async e=>{const t=ge(e.utils,16,16,5888),s=Qi(16);e.kernel.setOutput([16,16]);const n=e.kernel(t,s,16),i=bs(t,s,16,[8],"the loop bound has to be the size argument, not the literal the kernel was born with");for(let a=0;a<16;a++)for(let c=0;c<16;c++){const h=Xe(n[a][c],t[a][c],.02,vs(i,a,c));e.assertClose(n[a][c],t[a][c],.02,h||`cell [${a}][${c}] of A × I`)}}}]}]},Rl=Object.freeze({__proto__:null,default:zl});function Pe(e,t,s=2207){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++)i[a]=Math.round(n()*2e3)/1e3;return i}function Ye(e){let t=0;for(let s=0;s<e.length;s++)t+=e[s];return t}function sa(e){let t=e[0];for(let s=1;s<e.length;s++)e[s]<t&&(t=e[s]);return t}function na(e){let t=e[0];for(let s=1;s<e.length;s++)e[s]>t&&(t=e[s]);return t}function Ka(e,t,s,n){let i=0;for(let a=0;a<n;a++)i+=e[a*s+t];return i}function lt(e,t){let s=t instanceof Float32Array?t:Float32Array.from(t),n=s.length;for(;n>1;)n=n/2,e.setOutput([n]),s=e(s);return s[0]}function Ol(e,t,s){let n=0;for(let i=0;i<s;i++)n+=e[t*s+i];return n}function St(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function ra(e,t,s,n){return[[Ol(e,t,n),"that is the sum of a contiguous block — the strided walk is data[i * this.constants.threads + this.thread.x], so neighbouring threads touch neighbouring elements"],[Ka(e,0,s,n),"every thread summed thread 0's slice — this.thread.x has to appear in the index"]]}function en(e,t){return[[e[t]+e[t+1],"you paired with your immediate neighbour — the partner sits one output width away: data[this.thread.x + this.output.x]"],[e[t],"only your own element came back — the partner in the top half never got added"]]}function Xn(e){const t=new Array(4096).fill(2);for(const s of e.kernels){if(!s.kernel||s.kernel.dynamicOutput)continue;let n;try{n=s(t)}catch{continue}if(n&&n.length===64&&Math.abs(n[0]-16384)<=.01)return"one kernel squared its finished partial sum — the square belongs on each value as it is read, inside the loop"}return null}function Yn(e){const t=[];for(const s of e){if(s.type!=="log"||!s.text)continue;const n=s.text.match(/-?\d+(?:\.\d+)?/g);if(n)for(const i of n)t.push(parseFloat(i))}return t}function Zt(e){return e.kernels.find(t=>t.kernel&&t.kernel.dynamicOutput)||null}function Jn(e){const t=new Array(4096).fill(2);let s=null,n=null;for(const i of e.kernels){if(!i.kernel||i.kernel.dynamicOutput)continue;let a;try{a=i(t)}catch{continue}!a||a.length!==64||(Math.abs(a[0]-128)<=.01?s=i:Math.abs(a[0]-256)<=.01&&(n=i))}return{sums:s,squares:n}}var Ll={id:"2-2",track:2,title:"Reductions",blurb:"Sum, min, max and mean over millions of values — the ladder pattern every platform uses.",tasks:[{slug:"one-thread-sum",title:"The One-Thread Trap",intro:`<p>Meet the <strong>reduction</strong>: many values in, one value out — sum, min,
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
`,inputs:e=>({data:Pe(e,4096,707)}),publicTests:[{name:"64 partial sums — all-ones input gives 64 everywhere",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(new Array(4096).fill(1));e.assert(t&&t.length===64,`expected 64 partial sums, got ${t&&t.length}`);for(let s=0;s<64;s++)e.assertClose(t[s],64,.001,`partial ${s} should sum 64 ones`)}},{name:"partials are strided — thread x sums <code>data[x], data[x + 64], …</code>",run:async e=>{const t=new Array(4096);for(let n=0;n<4096;n++)t[n]=n;const s=e.kernel(t);for(const n of[0,1,31,63]){const i=St(s[n],129024+64*n,.5,ra(t,n,64,64));e.assertClose(s[n],129024+64*n,.5,i||`partial ${n} should sum data[${n}], data[${n} + 64], data[${n} + 128], …`)}}},{name:"the grand total is computed and logged",run:async e=>{const t=Ye(Pe(e.utils,4096,707)),s=Yn(e.logs);e.assert(s.some(n=>Math.abs(n-t)<=.5),`log the total of the partials — expected ≈${t.toFixed(2)} in the console output`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=Pe(e.utils,4096,555),s=e.kernel(t);e.assert(s&&s.length===64,"expected 64 partial sums");for(let n=0;n<64;n++){const i=Ka(t,n,64,64),a=St(s[n],i,.02,ra(t,n,64,64));e.assertClose(s[n],i,.02,a||`partial ${n}`)}e.assertClose(Ye(Array.from(s)),Ye(t),.5,"total of the partials")}}]},{slug:"halving-step",title:"One Rung of the Ladder",intro:`<p>Sixty-four partials finished in JavaScript is fine. A million wouldn't be. To
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
`,inputs:e=>({data:Pe(e,1024,5150)}),publicTests:[{name:"one ladder keeps the smaller value, one the larger",run:async e=>{let t=null,s=null;for(const n of e.kernels){if(!n.kernel||!n.kernel.dynamicOutput)continue;n.setOutput([1]);const i=n(Float32Array.from([3,5]))[0],a=n(Float32Array.from([8,2]))[0];Math.abs(i-3)<.001&&Math.abs(a-2)<.001&&(t=n),Math.abs(i-5)<.001&&Math.abs(a-8)<.001&&(s=n)}e.assert(t,"no min ladder found — one kernel should fold with Math.min"),e.assert(s,"no max ladder found — one kernel should fold with Math.max")}},{name:"min and max of a fresh 512-value array",run:async e=>{const t=e.utils.seededRandom(88),s=new Array(512);for(let a=0;a<512;a++)s[a]=Math.round(t()*4e3)/1e3-2;let n=null,i=null;for(const a of e.kernels){if(!a.kernel||!a.kernel.dynamicOutput)continue;a.setOutput([1]);const c=a(Float32Array.from([3,5]))[0];Math.abs(c-3)<.001?n=a:Math.abs(c-5)<.001&&(i=a)}e.assert(n&&i,"expected a Math.min ladder and a Math.max ladder"),e.assertClose(lt(n,s),sa(s),.001,"the minimum"),e.assertClose(lt(i,s),na(s),.001,"the maximum")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Pe(e.utils,1024,31337);let s=null,n=null;for(const i of e.kernels){if(!i.kernel||!i.kernel.dynamicOutput)continue;i.setOutput([1]);const a=i(Float32Array.from([-4,9]))[0];Math.abs(a- -4)<.001?s=i:Math.abs(a-9)<.001&&(n=i)}e.assert(s&&n,"expected a Math.min ladder and a Math.max ladder"),e.assertClose(lt(s,t),sa(t),.001,"the minimum"),e.assertClose(lt(n,t),na(t),.001,"the maximum")}}]},{slug:"fused-mean-rms",title:"Payoff: Mean and RMS, Fused",intro:`<p>The payoff. Two statistics over 4,096 values: the <strong>mean</strong>
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
`,inputs:e=>({data:Pe(e,4096,6001)}),publicTests:[{name:"three kernels: plain partials, fused squared partials, dynamic ladder",run:async e=>{const{sums:t,squares:s}=Jn(e);e.assert(t,"no kernel producing 64 partial sums found (all-2s input should give 128 per thread)");const n=s?null:Xn(e);e.assert(s,n||"no fused kernel producing 64 partial sums of squares found (all-2s input should give 256 per thread)"),e.assert(Zt(e),"no dynamicOutput halving-ladder kernel found")}},{name:"full pipeline: mean and RMS of a fresh array",run:async e=>{const{sums:t,squares:s}=Jn(e),n=Zt(e),i=s?null:Xn(e);e.assert(t&&s&&n,i||"expected partialSums, partialSquares and a dynamic ladder kernel");const a=new Array(4096);for(let j=0;j<4096;j++)a[j]=(j%8+1)/4;let c=0,h=0;for(let j=0;j<4096;j++)c+=a[j],h+=a[j]*a[j];const I=lt(n,t(a)),U=lt(n,s(a));e.assertClose(I/4096,c/4096,.001,"the mean"),e.assertClose(Math.sqrt(U/4096),Math.sqrt(h/4096),.001,"the RMS")}},{name:"mean and RMS of <code>data</code> are logged",run:async e=>{const t=Pe(e.utils,4096,6001);let s=0,n=0;for(let h=0;h<t.length;h++)s+=t[h],n+=t[h]*t[h];const i=s/4096,a=Math.sqrt(n/4096),c=Yn(e.logs);e.assert(c.some(h=>Math.abs(h-i)<=.01),`log the mean — expected ≈${i.toFixed(3)} in the console output`),e.assert(c.some(h=>Math.abs(h-a)<=.01),`log the RMS — expected ≈${a.toFixed(3)} in the console output`)}}],privateTests:[{name:"private test #1",run:async e=>{const{sums:t,squares:s}=Jn(e),n=Zt(e),i=s?null:Xn(e);e.assert(t&&s&&n,i||"expected partialSums, partialSquares and a dynamic ladder kernel");const a=Pe(e.utils,4096,909);let c=0,h=0;for(let j=0;j<a.length;j++)c+=a[j],h+=a[j]*a[j];const I=lt(n,t(a)),U=lt(n,s(a));e.assertClose(I/4096,c/4096,.01,"the mean"),e.assertClose(Math.sqrt(U/4096),Math.sqrt(h/4096),.01,"the RMS")}}]}]},Fl=Object.freeze({__proto__:null,default:Ll});function ia(e,t,s,n,i){return Math.abs(e-t)<=s?`that is the value for cell [${i}][${n}] — this.thread.x and this.thread.y are swapped. Rows come first: image[this.thread.y][this.thread.x]`:null}function ws(e,t,s){return e?`the picture is transposed — the value for row ${t}, col ${s} turned up at row ${s}, col ${t}. this.thread.x and this.thread.y are swapped; rows come first: image[this.thread.y][this.thread.x].`:null}function ut(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function Zn(e,t){const s=e.length,n=h=>Math.max(0,Math.min(s-1,h)),i=e[n(t-1)],a=e[t],c=e[n(t+1)];return[[(i+a+c)/3,"that is the plain 3-tap mean — this filter weights the taps 0.25 / 0.5 / 0.25"],[a,"that is the sample itself — the weighted average of its neighborhood never happened"],[.25*(i+a+c),"the center tap carries 0.5, not 0.25 — the three weights have to sum to 1"]]}function aa(e){return Number.isFinite(e)?null:"that sample read outside the signal — clamp the neighbor indexes into 0…127 before reading"}function Qn(e,t){const s="the window is not centered on this thread — tap i belongs at x + i − this.constants.radius";return e.map(n=>[n[t],s])}function Gl(e,t,s,n){const i=(t*128+s)*4;return e[i]>=253&&e[i+1]>=253&&e[i+2]>=253&&Math.max(n[0],n[1],n[2])<.9?"every channel is clamped to white — that is the sum of the nine samples; divide each one by 9":null}function er(e,t,s){const n=e.length,i=h=>Math.max(0,Math.min(n-1,h)),a=e[t][s],c=e[t][i(s-1)]+e[t][i(s+1)]+e[i(t-1)][s]+e[i(t+1)][s];return[[4*a-c,"the center weight is 4, not 5 — the five weights have to sum to 1 so flat areas pass through unchanged"],[5*a+c,"the four neighbors are being added — a sharpen subtracts them: 5·center − left − right − up − down"],[a,"that is the value unchanged — none of the five weights reached the return value"],[c/4,"that is the average of the four neighbors — the 5·center term is missing"]]}function Qt(e,t=2301){const s=e.seededRandom(t),n=new Array(128);for(let i=0;i<128;i++)n[i]=Math.round((Math.sin(i/6)*3+s()*4)*100)/100;return n}function et(e,t,s){const n=e.length,i=new Array(n);for(let a=0;a<n;a++){let c=0;for(let h=0;h<t.length;h++){let I=a+h-s;I<0&&(I=0),I>n-1&&(I=n-1),c+=t[h]*e[I]}i[a]=c}return i}function Ul(e){const t=e.plain,s=t.length,n=new Array(s);for(let i=0;i<s;i++){const a=new Array(s);for(let c=0;c<s;c++){let h=0,I=0,U=0;for(let j=-1;j<=1;j++)for(let q=-1;q<=1;q++){const $e=Math.min(s-1,Math.max(0,i+j)),Ze=Math.min(s-1,Math.max(0,c+q)),rt=t[$e][Ze];h+=rt[0],I+=rt[1],U+=rt[2]}a[c]=[h/9,I/9,U/9]}n[i]=a}return n}function oa(e){const t=e.plain,s=t.length,n=new Array(s);for(let i=0;i<s;i++){const a=new Array(s);for(let c=0;c<s;c++){const h=t[i][c];a[c]=.299*h[0]+.587*h[1]+.114*h[2]}n[i]=a}return n}function Vl(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++){const c=new Array(t);for(let h=0;h<t;h++)c[h]=Math.round(n()*1e3)/1e3;i[a]=c}return i}function la(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){const i=new Array(t),a=Math.max(0,n-1),c=Math.min(t-1,n+1);for(let h=0;h<t;h++){const I=Math.max(0,h-1),U=Math.min(t-1,h+1);i[h]=5*e[n][h]-e[n][I]-e[n][U]-e[a][h]-e[c][h]}s[n]=i}return s}function ua(e,t){const s=new Array(e).fill(Bt(t));return us(new Array(e).fill(s))}function ca(e,t,s){const n=Bt([t,t,t,1]),i=Bt([s,s,s,1]),a=new Array(e);for(let c=0;c<e;c++){const h=new Array(e);for(let I=0;I<e;I++)h[I]=I<e/2?n:i;a[c]=h}return us(a)}function Kl(e,t,s){const n=Bt([t,t,t,1]),i=Bt([s,s,s,1]),a=new Array(e);for(let c=0;c<e;c++)a[c]=new Array(e).fill(c<e/2?n:i);return us(a)}var Nl={id:"2-3",track:2,title:"Convolution & Filters",blurb:"Sliding-window math on signals and images: blur, sharpen, edge detection.",tasks:[{slug:"smooth-a-signal",title:"Slide a Window: 1D Convolution",intro:`<p>A <strong>convolution</strong> slides a small window of weights along a signal:
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
`,inputs:e=>({signal:Qt(e)}),publicTests:[{name:"returns 128 samples, each a <code>[0.25, 0.5, 0.25]</code> weighted average",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Qt(e.utils),s=e.kernel(t);e.assert(s&&s.length===128,`expected 128 output samples, got ${s&&s.length}`);const n=et(t,[.25,.5,.25],1);for(const i of[1,17,42,63,100,126]){const a=ut(s[i],n[i],.001,Zn(t,i));e.assertClose(s[i],n[i],.001,a||`sample ${i}`)}}},{name:"edges clamp: sample 0 is <code>0.75·s[0] + 0.25·s[1]</code>",run:async e=>{const t=new Array(128);for(let a=0;a<128;a++)t[a]=a*37%23-11;const s=e.kernel(t),n=et(t,[.25,.5,.25],1),i=a=>aa(s[a])||ut(s[a],n[a],.001,Zn(t,a));e.assertClose(s[0],.75*t[0]+.25*t[1],.001,i(0)||"sample 0"),e.assertClose(s[127],.25*t[126]+.75*t[127],.001,i(127)||"sample 127");for(let a=0;a<128;a++)e.assertClose(s[a],n[a],.001,i(a)||`sample ${a}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=Qt(e.utils,909),s=e.kernel(t),n=et(t,[.25,.5,.25],1);e.assert(s.length===128,"expected 128 output samples");for(let i=0;i<128;i++){const a=aa(s[i])||ut(s[i],n[i],.001,Zn(t,i));e.assertClose(s[i],n[i],.001,a||`sample ${i}`)}}}]},{slug:"filter-as-data",title:"Any Filter, One Kernel",intro:`<p>Hardcoded weights mean writing a new kernel for every filter. The fix: pass the
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
`,inputs:e=>({signal:Qt(e)}),publicTests:[{name:"the identity filter <code>[0, 0, 1, 0, 0]</code> returns the signal untouched",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Qt(e.utils),s=[0,0,1,0,0],n=e.kernel(t,s);e.assert(n&&n.length===128,`expected 128 output samples, got ${n&&n.length}`);const i=[et(t,s,0),et(t,s,-2)];for(let a=0;a<128;a++){const c=ut(n[a],t[a],.001,Qn(i,a));e.assertClose(n[a],t[a],.001,c||`sample ${a}`)}}},{name:"a box filter matches the clamped-edge reference everywhere",run:async e=>{const t=new Array(128);for(let c=0;c<128;c++)t[c]=c*29%17-8;const s=[.2,.2,.2,.2,.2],n=e.kernel(t,s),i=et(t,s,2),a=[et(t,s,0),et(t,s,-2)];for(let c=0;c<128;c++){const h=ut(n[c],i[c],.002,Qn(a,c));e.assertClose(n[c],i[c],.002,h||`sample ${c}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Qt(e.utils,777),s=e.utils.seededRandom(31),n=new Array(5);for(let h=0;h<5;h++)n[h]=Math.round((s()*.6-.1)*100)/100;const i=e.kernel(t,n),a=et(t,n,2),c=[et(t,n,0),et(t,n,-2)];for(let h=0;h<128;h++){const I=ut(i[h],a[h],.002,Qn(c,h));e.assertClose(i[h],a[h],.002,I||`sample ${h}`)}}}]},{slug:"box-blur",title:"Box Blur: the Window Goes 2D",intro:`<p>Take the sliding window into two dimensions and you have image filtering. A
        <strong>3×3 box blur</strong> is the simplest case: every output pixel is the plain
        average of the 3×3 patch centered on it — nine reads, per color channel, per pixel.
        131,072 threads each do their nine reads at once.</p>
        <p>Same edge problem, now on four sides: clamp <em>both</em> coordinates into
        <code>0…this.constants.last</code> before indexing. Average red, green and blue
        separately and hand the result to <code>this.color()</code>.</p>
        ${Mt}`,goal:`<strong>Goal:</strong> blur <code>inputImage</code> with a 3×3 box filter — each
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
`,inputs:e=>({inputImage:e.makeTestImage(128)}),publicTests:[{name:"produces a <code>128×128</code> graphical canvas",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===128&&e.canvas.height===128,`expected a 128×128 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();e.assert(t.length===16384*4,"pixel buffer should hold 128×128 RGBA values")}},{name:"blurring a flat color changes nothing",run:async e=>{const t=ua(128,[.3,.5,.7,1]),s=t.at(0,0);e.kernel(t);const n=e.getPixels();for(let i=0;i<n.length;i+=331*4)e.assertClose(n[i],s[0]*255,2,`red at byte ${i}`),e.assertClose(n[i+1],s[1]*255,2,`green at byte ${i}`),e.assertClose(n[i+2],s[2]*255,2,`blue at byte ${i}`)}},{name:"each pixel is the average of its 3×3 neighborhood",run:async e=>{const t=e.utils.makeTestImage(128);e.kernel(t);const s=e.getPixels(),n=Ul(t),i=(a,c,h)=>{const I=(a*128+c)*4;return Math.abs(s[I]-h[0]*255)<=3&&Math.abs(s[I+1]-h[1]*255)<=3&&Math.abs(s[I+2]-h[2]*255)<=3};for(const a of[3,17,40,64,90,121])for(const c of[5,33,64,101,124]){const h=i(a,c,n[c][a])||i(a,c,n[c][127-a]);e.assert(i(a,c,n[a][c])||i(a,c,n[127-a][c]),ws(h,a,c)||Gl(s,a,c,n[a][c])||`pixel at row ${a}, col ${c} is not the 3×3 average of its neighborhood`)}}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(ca(128,.2,.8));const t=e.getPixels(),s=n=>n<=62?.2:n===63?.4:n===64?.6:.8;for(const n of[8,60,119])for(const i of[0,20,63,64,90,127]){const a=(n*128+i)*4,c=s(i)*255,h=(i<64?.2:.8)*255,I=ut(t[a],c,2,[[h,"that is the original pixel — the 3×3 average never happened, so the seam did not soften"]]);e.assertClose(t[a],c,2,I||`red at row ${n}, col ${i}`),e.assertClose(t[a+1],c,2,I||`green at row ${n}, col ${i}`),e.assertClose(t[a+2],c,2,I||`blue at row ${n}, col ${i}`)}}}]},{slug:"sharpen",title:"Sharpen: Negative Weights",intro:`<p>Filters are not all averages. Give the window <strong>negative weights</strong>
        and it starts measuring <em>differences</em>. The classic sharpen filter is a cross:
        <code>5</code> at the center, <code>−1</code> at each direct neighbor. Where the image is
        flat, the terms cancel to exactly the original value; where it changes, the difference
        gets amplified — edges pop.</p>
        <p>Sharpened values can overshoot right out of the 0–1 range, so this task computes on a
        numeric <strong>luminance map</strong> (<code>gray[y][x]</code>, one number per pixel)
        and returns raw numbers you can inspect — no color clamping hiding the math.</p>
        ${Mt}`,goal:`<strong>Goal:</strong> sharpen the 96×96 <code>gray</code> map — each cell becomes
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
`,inputs:e=>({gray:oa(e.makeTestImage(96))}),publicTests:[{name:"flat regions are a fixed point — sharpening a constant map returns it unchanged",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(96).fill(new Array(96).fill(.5)),s=e.kernel(t);e.assert(s&&s.length===96,`expected 96 rows, got ${s&&s.length}`),e.assert(s[0]&&s[0].length===96,"each row should hold 96 values");for(let n=0;n<96;n+=7)for(let i=0;i<96;i+=7){const a=ut(s[n][i],.5,.001,er(t,n,i));e.assertClose(s[n][i],.5,.001,a||`cell [${n}][${i}]`)}}},{name:"cell [y][x] equals <code>5·center − left − right − up − down</code>",run:async e=>{const t=oa(e.utils.makeTestImage(96)),s=e.kernel(t),n=la(t),i=[[0,0],[0,48],[11,60],[48,48],[77,3],[95,95]];for(const[a,c]of i){const h=ia(s[a][c],n[c][a],.002,a,c)||ut(s[a][c],n[a][c],.002,er(t,a,c));e.assertClose(s[a][c],n[a][c],.002,h||`cell [${a}][${c}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Vl(e.utils,96,4242),s=e.kernel(t),n=la(t);for(let i=0;i<96;i++)for(let a=0;a<96;a++){const c=ia(s[i][a],n[a][i],.002,i,a)||ut(s[i][a],n[i][a],.002,er(t,i,a));e.assertClose(s[i][a],n[i][a],.002,c||`cell [${i}][${a}]`)}}}]},{slug:"sobel",title:"Sobel Edge Detection",intro:`<p>The payoff: run <strong>two convolutions at once</strong>. Sobel's
        <code>Gx</code> filter responds to horizontal change, <code>Gy</code> to vertical change,
        and the length of that gradient vector — <code>√(gx² + gy²)</code> — is how
        <em>edge-like</em> the pixel is, whatever the edge's direction.</p>
        <p>This is a two-kernel pipeline like module 1.2's finale: a numeric pass turns the image
        into a luminance map (written for you), then the Sobel pass reads each map cell's eight
        neighbors, applies both weight grids, and paints the magnitude. Border pixels have no
        full neighborhood, so the starter already paints them black — your work lives in the
        <code>else</code> branch.</p>
        ${Mt}`,goal:`<strong>Goal:</strong> finish the Sobel kernel — read the 3×3 neighborhood of
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
`,inputs:e=>({inputImage:e.makeTestImage(128)}),publicTests:[{name:"a numeric luminance pass feeding a graphical Sobel pass",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels, found ${e.kernels.length}`);const t=e.kernels.find(n=>n.kernel&&!n.kernel.graphical),s=e.kernels.find(n=>n.kernel&&n.kernel.graphical);e.assert(t,"no numeric (non-graphical) kernel found"),e.assert(s,"no graphical kernel found"),e.assert(e.canvas,"no canvas — did you call render(sobel.canvas)?"),e.assert(e.canvas.width===128&&e.canvas.height===128,`expected a 128×128 canvas, got ${e.canvas.width}×${e.canvas.height}`)}},{name:"a flat image has no edges — constant in, all black out",run:async e=>{const t=e.kernels.find(i=>i.kernel&&!i.kernel.graphical),s=e.kernels.find(i=>i.kernel&&i.kernel.graphical);e.assert(t&&s,"expected a numeric and a graphical kernel"),s(t(ua(128,[.4,.6,.2,1])));const n=s.getPixels();for(let i=0;i<n.length;i+=1004)e.assert(n[i]<=1&&n[i+1]<=1&&n[i+2]<=1,`pixel at byte ${i} should be black, got rgb(${n[i]}, ${n[i+1]}, ${n[i+2]})`),e.assert(n[i+3]===255,`alpha at byte ${i} should be 255`)}},{name:"a vertical brightness step lights up exactly the step columns",run:async e=>{const t=e.kernels.find(c=>c.kernel&&!c.kernel.graphical),s=e.kernels.find(c=>c.kernel&&c.kernel.graphical);e.assert(t&&s,"expected a numeric and a graphical kernel"),s(t(ca(128,.1,.9)));const n=s.getPixels(),i=(c,h)=>n[(c*128+h)*4],a=(c,h,I)=>I(i(h,c))||I(i(127-h,c));for(const c of[10,64,100]){for(const h of[63,64]){const I=(c*128+h)*4,U=a(c,h,j=>j>=253);e.assert(n[I]>=253,ws(U,c,h)||`the step at col ${h} should saturate white, got ${n[I]} (row ${c})`)}for(const h of[30,96]){const I=(c*128+h)*4,U=a(c,h,j=>j<=1);e.assert(n[I]<=1,ws(U,c,h)||`flat area at col ${h} should be black, got ${n[I]} (row ${c})`)}}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(c=>c.kernel&&!c.kernel.graphical),s=e.kernels.find(c=>c.kernel&&c.kernel.graphical);e.assert(t&&s,"expected a numeric and a graphical kernel"),s(t(Kl(128,.15,.85)));const n=s.getPixels(),i=(c,h)=>n[(c*128+h)*4],a=(c,h,I)=>I(i(h,c))||I(i(127-h,c));for(const c of[10,64,120]){for(const h of[63,64]){const I=(h*128+c)*4,U=a(h,c,j=>j>=253);e.assert(n[I]>=253,ws(U,h,c)||`the step at row ${h} should saturate white, got ${n[I]} (col ${c})`)}for(const h of[20,100]){const I=(h*128+c)*4,U=a(h,c,j=>j<=1);e.assert(n[I]<=1,ws(U,h,c)||`flat area at row ${h} should be black, got ${n[I]} (col ${c})`)}}}}]}]},Bl=Object.freeze({__proto__:null,default:Nl});function tr(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++)i[a]=n();return i}function es(e,t,s){const n=e.seededRandom(s),i=new Array(t),a=new Array(t);for(let c=0;c<t;c++)i[c]=n(),a[c]=n();return{xs:i,ys:a}}function tn(e,t){let s=0;for(let n=0;n<e.length;n++)e[n]*e[n]+t[n]*t[n]<=1&&s++;return s}function sn(e){let t=0;for(let s=0;s<e.length;s++)t+=Math.exp(-e[s]*e[s]);return t}function sr(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a+=2){const c=1-n(),h=n(),I=Math.sqrt(-2*Math.log(c));i[a]=I*Math.cos(2*Math.PI*h),a+1<t&&(i[a+1]=I*Math.sin(2*Math.PI*h))}return i}const ke={s0:100,strike:105,rate:.03,sigma:.2,t:1},rs=(ke.rate-ke.sigma*ke.sigma/2)*ke.t,is=ke.sigma*Math.sqrt(ke.t);function ha(e){const t=1/(1+.2316419*Math.abs(e)),n=.3989422804014327*Math.exp(-e*e/2)*t*(.31938153+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));return e>=0?1-n:n}function da(){const{s0:e,strike:t,rate:s,sigma:n,t:i}=ke,a=(Math.log(e/t)+(s+n*n/2)*i)/(n*Math.sqrt(i)),c=a-n*Math.sqrt(i);return e*ha(a)-t*Math.exp(-s*i)*ha(c)}function nn(e){let t=0;for(let s=0;s<e.length;s++){const n=ke.s0*Math.exp(rs+is*e[s]);t+=Math.max(n-ke.strike,0)}return Math.exp(-.03*ke.t)*(t/e.length)}function pa(e){let t=0;for(let s=0;s<e.length;s++){const n=ke.s0*Math.exp(rs+is*e[s]);t+=n-ke.strike}return Math.exp(-.03*ke.t)*(t/e.length)}function Ot(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function jl(e,t,s,n){const i=a=>n.every((c,h)=>Math.abs(e[h]-a(h))<=1e-6);return i(a=>t[a]+s[a]<=1?1:0)?"those verdicts are x + y ≤ 1 — the inside test compares squared distance: x * x + y * y <= 1":i(a=>1-n[a])?"the verdicts are inverted — return 1 when the dart lands inside, 0 when it misses":null}function hn(e,t,s,n){let i=0;for(let a=0;a<s;a++){const c=e[t+a];i+=n?n(c):c}return i}function ql(e,t,s){const n=[[e[t],"that is a single verdict — this thread has to total all 256 in its own slice"]];return t+s<=e.length&&n.push([hn(e,t,s),"the slice starts at this.thread.x * 256 — with this.thread.x alone every thread walks an overlapping window"]),n}function nr(e,t,s){return[[hn(e,t,s),"that is the sum of the samples themselves — the accumulator wants Math.exp(-x * x)"],[hn(e,t,s,n=>Math.exp(-n)),"that is e^(−x), not e^(−x²) — square x inside the exponent"],[hn(e,t,s,n=>Math.exp(n*n)),"the exponent is missing its minus sign — e^(−x²) falls off as x grows"]]}function Hl(e,t){return[[e-t,"that is st − strike with no floor — a losing path pays exactly 0, never a negative amount"],[Math.max(t-e,0),"that is the put payoff — a call pays max(st − strike, 0)"]]}var Wl={id:"2-4",track:2,title:"Monte Carlo Methods",blurb:"Estimate π, price an option, integrate the un-integrable — with a million random samples.",tasks:[{slug:"darts-at-a-circle",title:"Darts at a Quarter Circle",intro:`<p>Monte Carlo is statistics as a weapon: throw random darts at a square, and the
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
`,inputs:e=>es(e,4096,9001),publicTests:[{name:"clearly-inside darts return 1, clearly-outside darts return 0",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(4096).fill(.5),s=new Array(4096).fill(.5);t[0]=.1,s[0]=.1,t[1]=.9,s[1]=.9,t[2]=0,s[2]=0,t[3]=.99,s[3]=.3,t[4]=.6,s[4]=.6;const n=e.kernel(t,s);e.assert(n&&n.length===4096,`expected 4096 verdicts, got ${n&&n.length}`);const i=[1,0,1,0,1],a=jl(n,t,s,i);for(let c=0;c<i.length;c++)e.assertClose(n[c],i[c],1e-6,a||`dart ${c} at (${t[c]}, ${s[c]})`)}},{name:"hit fraction over the seeded darts approaches <code>π/4</code>",run:async e=>{const{xs:t,ys:s}=es(e.utils,4096,9001),n=e.kernel(t,s);let i=0;for(let a=0;a<n.length;a++)e.assert(n[a]===0||n[a]===1,`verdict ${a} is ${n[a]} — return exactly 1 or 0`),i+=n[a];e.assertClose(i,tn(t,s),2,"hit count over the seeded darts"),e.assertClose(4*i/4096,Math.PI,.06,"π estimate from 4096 darts")}}],privateTests:[{name:"private test #1",run:async e=>{const{xs:t,ys:s}=es(e.utils,4096,4242),n=e.kernel(t,s);let i=0;for(let a=0;a<n.length;a++)i+=n[a];e.assertClose(i,tn(t,s),2,"hit count on unseen darts")}}]},{slug:"reduce-to-pi",title:"Reduce 65,536 Hits to π",intro:`<p>Last task summed the verdicts with a JavaScript loop — fine for 4,096 darts,
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
`,inputs:e=>es(e,65536,1337),publicTests:[{name:"reduction kernel collapses a known array to correct partial sums",run:async e=>{const t=e.kernels.find(i=>i.kernel&&Array.isArray(i.kernel.output)&&i.kernel.output[0]===256);e.assert(t,"no kernel with output [256] found — keep the partialSums kernel");const s=new Array(65536);for(let i=0;i<65536;i++)s[i]=i%3;const n=t(new Float32Array(s));e.assert(n&&n.length===256,`expected 256 partials, got ${n&&n.length}`);for(const i of[0,1,17,128,255]){let a=0;for(let h=0;h<256;h++)a+=(i*256+h)%3;const c=Ot(n[i],a,.5,ql(s,i,256));e.assertClose(n[i],a,.5,c||`partial sum for thread ${i}`)}}},{name:"π comes out within <code>±0.05</code> over the 65,536 seeded darts",run:async e=>{const t=e.kernels.find(I=>I.kernel&&Array.isArray(I.kernel.output)&&I.kernel.output[0]===65536),s=e.kernels.find(I=>I.kernel&&Array.isArray(I.kernel.output)&&I.kernel.output[0]===256);e.assert(t&&s,"expected the inside kernel [65536] and the partialSums kernel [256]");const{xs:n,ys:i}=es(e.utils,65536,1337),a=t(n,i),c=s(new Float32Array(a));let h=0;for(let I=0;I<c.length;I++)h+=c[I];e.assertClose(h,tn(n,i),4,"total hit count after reduction"),e.assertClose(4*h/65536,Math.PI,.05,"π estimate")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(I=>I.kernel&&Array.isArray(I.kernel.output)&&I.kernel.output[0]===65536),s=e.kernels.find(I=>I.kernel&&Array.isArray(I.kernel.output)&&I.kernel.output[0]===256);e.assert(t&&s,"expected the inside kernel [65536] and the partialSums kernel [256]");const{xs:n,ys:i}=es(e.utils,65536,2718),a=t(n,i),c=s(new Float32Array(a));let h=0;for(let I=0;I<c.length;I++)h+=c[I];e.assertClose(h,tn(n,i),4,"hit count on unseen darts")}}]},{slug:"integrate-the-unintegrable",title:"Integrate the Un-integrable",intro:`<p><code>e^(−x²)</code> — the bell curve — famously has <strong>no elementary
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
`,inputs:e=>({samples:tr(e,16384,6077)}),publicTests:[{name:"each thread sums <code>e^(−x²)</code> over its own 64-sample slice",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(16384);for(let n=0;n<16384;n++)t[n]=n/16384;const s=e.kernel(t);e.assert(s&&s.length===256,`expected 256 partial sums, got ${s&&s.length}`);for(const n of[0,3,100,255]){const i=t.slice(n*64,n*64+64),a=Ot(s[n],sn(i),.05,nr(t,n*64,64));e.assertClose(s[n],sn(i),.05,a||`partial sum for thread ${n}`)}}},{name:"estimate lands within <code>±0.01</code> of the true value <code>0.746824</code>",run:async e=>{const t=tr(e.utils,16384,6077),s=e.kernel(t);let n=0;for(let a=0;a<s.length;a++)n+=s[a];const i=Ot(n/16384,.7468241328124271,.01,nr(t,0,16384).map(a=>[a[0]/16384,a[1]]));e.assertClose(n/16384,.7468241328124271,.01,i||"Monte Carlo integral estimate")}}],privateTests:[{name:"private test #1",run:async e=>{const t=tr(e.utils,16384,1912),s=e.kernel(t);let n=0;for(let a=0;a<s.length;a++)n+=s[a];const i=Ot(n/16384,sn(t)/16384,.002,nr(t,0,16384).map(a=>[a[0]/16384,a[1]]));e.assertClose(n/16384,sn(t)/16384,.002,i||"estimate vs float64 reference"),e.assertClose(n/16384,.7468241328124271,.01,"estimate vs the true integral")}}]},{slug:"price-an-option",title:"Price an Option",intro:`<p>The payoff. A <strong>European call option</strong> is the right to buy a stock at
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
`,inputs:e=>({normals:sr(e,16384,8128)}),publicTests:[{name:"payoffs are <code>max(S_T − K, 0)</code> — losing paths pay exactly zero",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=new Array(16384).fill(0);t[0]=-3,t[1]=2,t[2]=.5,t[3]=-.5;const s=e.kernel(t,ke.s0,ke.strike,rs,is);e.assert(s&&s.length===16384,`expected 16384 payoffs, got ${s&&s.length}`);for(const n of[0,1,2,3]){const i=ke.s0*Math.exp(rs+is*t[n]),a=Math.max(i-ke.strike,0),c=Ot(s[n],a,.05,Hl(i,ke.strike));e.assertClose(s[n],a,.05,c||`payoff for shock z = ${t[n]}`),e.assert(s[n]>=0,`payoff for z = ${t[n]} is negative (${s[n]}) — options never go below zero`)}}},{name:"simulated price agrees with Black–Scholes (<code>≈ 7.13</code>) within <code>±0.4</code>",run:async e=>{const t=sr(e.utils,16384,8128),s=e.kernel(t,ke.s0,ke.strike,rs,is);let n=0;for(let c=0;c<s.length;c++)n+=s[c];const i=Math.exp(-.03*ke.t)*(n/s.length),a=Ot(i,nn(t),.05,[[pa(t),"that is the average of st − strike with no floor — every losing path dragged the mean below zero"]]);e.assertClose(i,nn(t),.05,a||"price vs float64 reference simulation"),e.assertClose(i,da(),.4,"price vs the Black–Scholes closed form")}}],privateTests:[{name:"private test #1",run:async e=>{const t=sr(e.utils,16384,6174),s=e.kernel(t,ke.s0,ke.strike,rs,is);let n=0;for(let c=0;c<s.length;c++)n+=s[c];const i=Math.exp(-.03*ke.t)*(n/s.length),a=Ot(i,nn(t),.05,[[pa(t),"that is the average of st − strike with no floor — every losing path dragged the mean below zero"]]);e.assertClose(i,nn(t),.05,a||"price vs float64 reference on unseen shocks"),e.assertClose(i,da(),.5,"price vs Black–Scholes on unseen shocks")}}]}]},Xl=Object.freeze({__proto__:null,default:Wl});function Lt(e,t,s){const n=e.seededRandom(s),i=new Array(t),a=new Array(t),c=new Array(t);for(let h=0;h<t;h++){const I=2*Math.PI*h/t,U=.7+.6*n();i[h]=Math.round(U*Math.cos(I)*1e4)/1e4,a[h]=Math.round(U*Math.sin(I)*1e4)/1e4,c[h]=Math.round((.5+n())*100)/100}return{posX:i,posY:a,mass:c}}function tt(e,t,s){const n=e.seededRandom(s),i=new Array(t),a=new Array(t),c=new Array(t),h=new Array(t),I=new Array(t);for(let U=0;U<t;U++)i[U]=Math.round((n()*2-1)*1e4)/1e4,a[U]=Math.round((n()*2-1)*1e4)/1e4,c[U]=Math.round((n()-.5)*.2*1e4)/1e4,h[U]=Math.round((n()-.5)*.2*1e4)/1e4,I[U]=Math.round((.5+n())*100)/100;return i[1]=i[0]+.001,a[1]=a[0],{posX:i,posY:a,velX:c,velY:h,mass:I}}function Nt(e,t,s,n,i){let a=0,c=0;for(let h=0;h<t.length;h++){if(i===0&&h===e)continue;const I=t[h]-t[e],U=s[h]-s[e],j=I*I+U*U+i*i;if(j===0)continue;const q=n[h]/(j*Math.sqrt(j));a+=I*q,c+=U*q}return[a,c]}function _r(e,t){const s=new Array(e.posX.length),n=new Array(e.posX.length);for(let i=0;i<e.posX.length;i++){const a=Nt(i,e.posX,e.posY,e.mass,t);s[i]=a[0],n[i]=a[1]}return{accX:s,accY:n}}function rr(e,t,s,n){const i={posX:e.posX.slice(),posY:e.posY.slice(),velX:e.velX.slice(),velY:e.velY.slice()};for(let a=0;a<t;a++){const c=_r({posX:i.posX,posY:i.posY,mass:e.mass},n);for(let h=0;h<i.posX.length;h++)i.velX[h]+=c.accX[h]*s,i.velY[h]+=c.accY[h]*s,i.posX[h]+=i.velX[h]*s,i.posY[h]+=i.velY[h]*s}return i}function ir(e){const t=new Array(e.length),s=new Array(e.length);for(let n=0;n<e.length;n++)t[n]=e[n][0],s[n]=e[n][1];return[t,s]}function fa(e,t,s,n,i,a,c){const[h,I]=ir(e(n.posX,n.posY,i,c)),[U,j]=ir(t(n.velX,n.velY,h,I,a)),[q,$e]=ir(s(n.posX,n.posY,U,j,a));return{posX:q,posY:$e,velX:U,velY:j}}function ve(e,t,s,n,i){e.assertClose(t,s,n*(1+Math.abs(s)),i)}function At(e,t){return e*(1+Math.abs(t))}function bt(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function ma(e,t){return[[e/Math.sqrt(t),"that is M / r — dx * dx + dy * dy is already r², so there is no square root to take"],[1/t,"the star's mass never entered the result — the pull is starMass / r²"],[e*t,"that multiplies by r² where the law divides by it"]]}function Yl(e,t,s,n){let i=0,a=0;for(let c=0;c<t.length;c++){if(c===e)continue;const h=t[c]-t[e],I=s[c]-s[e],U=h*h+I*I;i+=n[c]*h/U,a+=n[c]*I/U}return[i,a]}function ga(e,t,s,n){return[[Yl(e,t.posX,t.posY,t.mass)[n],"that is mass[j]·d / r², one factor of r short — the unit direction is d / r, so each term is mass[j]·d / r³"],[-s,"the offset points the wrong way — dx is posX[j] minus your OWN x, so the pull points at the other body"]]}function rn(e,t,s,n){return[[Nt(e,t.posX,t.posY,t.mass,Math.sqrt(s))[n],"that adds soft where it should add soft · soft — Plummer softening replaces r² with r² + ε²"],[Nt(e,t.posX,t.posY,t.mass,0)[n],"that is the unsoftened sum — ε never reached the denominator"]]}function _t(e,t,s,n,i){const a=t+s*n;return bt(e,a,At(i,a),[[t+s,"the time step is missing — the update is value + rate · dt"],[t,"that value came back unchanged — nothing was added to it"]])}var Jl={id:"2-5",track:2,title:"N-Body Gravity",blurb:"Every particle pulls on every other: an O(n²) problem the GPU eats for breakfast.",tasks:[{slug:"one-star-pull",title:"The Pull of One Star",intro:`<p>Newton, in one line: the gravitational pull between two bodies is
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
`,inputs:e=>{const t=Lt(e,64,901);return{posX:t.posX,posY:t.posY}},publicTests:[{name:"one pull strength per body — 64 positive numbers",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Lt(e.utils,64,901),s=e.kernel(t.posX,t.posY,0,0,100);e.assert(s&&s.length===64,`expected 64 values, got ${s&&s.length}`);for(let n=0;n<64;n++)e.assert(Number.isFinite(s[n])&&s[n]>0,`body ${n}: a star of mass 100 should pull with positive strength, got ${s[n]}`)}},{name:"doubling the distance quarters the pull — <code>M / r²</code>",run:async e=>{const t=new Array(64),s=new Array(64);for(let a=0;a<64;a++)t[a]=a+1,s[a]=0;const n=e.kernel(t,s,0,0,100),i=(a,c)=>bt(n[a],c,.01,ma(100,(a+1)*(a+1)));e.assertClose(n[0],100,.01,"body at distance 1"),e.assertClose(n[1],25,.01,i(1,25)||"body at distance 2 (quarter the pull)"),e.assertClose(n[3],6.25,.01,i(3,6.25)||"body at distance 4 (a sixteenth)"),e.assertClose(n[9],1,.01,i(9,1)||"body at distance 10")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Lt(e.utils,64,4242),s=e.kernel(t.posX,t.posY,-1.5,2.5,77);for(let n=0;n<64;n++){const i=-1.5-t.posX[n],a=2.5-t.posY[n],c=i*i+a*a,h=77/c,I=bt(s[n],h,At(.001,h),ma(77,c));ve(e,s[n],h,.001,I||`body ${n}`)}}}]},{slug:"sum-the-sky",title:"Every Body Pulls on Every Body",intro:`<p>Real gravity has no star at the center — <strong>every body pulls on every
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
`,inputs:e=>Lt(e,64,1702),publicTests:[{name:"pulls are real — and Newton's third law holds",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Lt(e.utils,64,1702),s=e.kernel(t.posX,t.posY,t.mass);e.assert(s&&s.length===64,`expected 64 values, got ${s&&s.length}`);let n=!1,i=0;for(let a=0;a<64;a++)Math.abs(s[a])>.001&&(n=!0),i+=t.mass[a]*s[a];e.assert(n,"every net pull came out ~0 — is the loop body still empty?"),e.assertClose(i,0,.05,"Σ mass[i]·ax[i] should cancel to ~0")}},{name:"body-by-body against a reference O(n²) loop",run:async e=>{const t=Lt(e.utils,64,1702),s=e.kernel(t.posX,t.posY,t.mass);for(const n of[0,17,40,63]){const i=Nt(n,t.posX,t.posY,t.mass,0),a=bt(s[n],i[0],At(.002,i[0]),ga(n,t,i[0],0));ve(e,s[n],i[0],.002,a||`net x-acceleration on body ${n}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=Lt(e.utils,64,555),s=e.kernel(t.posX,t.posY,t.mass);for(let n=0;n<64;n++){const i=Nt(n,t.posX,t.posY,t.mass,0),a=bt(s[n],i[0],At(.002,i[0]),ga(n,t,i[0],0));ve(e,s[n],i[0],.002,a||`net x-acceleration on body ${n}`)}}}]},{slug:"softening",title:"Softening the Singularity",intro:`<p>Two of this task's bodies sit <code>0.001</code> apart. Plug that into
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
`,inputs:e=>{const t=tt(e,64,33);return{posX:t.posX,posY:t.posY,mass:t.mass}},publicTests:[{name:"the close pair no longer explodes — every value stays finite and small",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=tt(e.utils,64,33),s=e.kernel(t.posX,t.posY,t.mass,.1);e.assert(s&&s.length===64,`expected 64 [ax, ay] pairs, got ${s&&s.length}`);let n=!1;for(let i=0;i<64;i++){e.assert(s[i]&&s[i].length===2,`body ${i}: expected an [ax, ay] pair`);const a=Math.abs(s[i][0])+Math.abs(s[i][1]);e.assert(Number.isFinite(a)&&a<1e3,`body ${i}: |acceleration| ≈ ${a.toFixed(1)} — the 0.001-apart pair is still unsoftened`),a>.001&&(n=!0)}e.assert(n,"every acceleration came out ~0 — did the loop body survive?")}},{name:"matches the softened reference — <code>mass · d / (r² + ε²)^{3/2}</code>",run:async e=>{const t=tt(e.utils,64,33),s=e.kernel(t.posX,t.posY,t.mass,.1);for(const n of[0,1,7,63]){const i=Nt(n,t.posX,t.posY,t.mass,.1),a=bt(s[n][0],i[0],At(.002,i[0]),rn(n,t,.1,0)),c=bt(s[n][1],i[1],At(.002,i[1]),rn(n,t,.1,1));ve(e,s[n][0],i[0],.002,a||`ax on body ${n}`),ve(e,s[n][1],i[1],.002,c||`ay on body ${n}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=tt(e.utils,64,909),s=e.kernel(t.posX,t.posY,t.mass,.25);for(let n=0;n<64;n++){const i=Nt(n,t.posX,t.posY,t.mass,.25),a=bt(s[n][0],i[0],At(.002,i[0]),rn(n,t,.25,0)),c=bt(s[n][1],i[1],At(.002,i[1]),rn(n,t,.25,1));ve(e,s[n][0],i[0],.002,a||`ax on body ${n}`),ve(e,s[n][1],i[1],.002,c||`ay on body ${n}`)}}}]},{slug:"euler-step",title:"One Tick of the Clock",intro:`<p>Accelerations are just numbers until an integrator turns them into motion. The
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
`,inputs:e=>{const t=tt(e,64,74),s=_r(t,.1);return{posX:t.posX,posY:t.posY,velX:t.velX,velY:t.velY,accX:s.accX,accY:s.accY}},publicTests:[{name:"the position step consumed the NEW velocities (semi-implicit)",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels (stepVel, stepPos), found ${e.kernels.length}`);const t=e.kernels[1];e.assert(Array.isArray(t.lastArgs),"stepPos was never called");const s=tt(e.utils,64,74),n=_r(s,.1),i=t.lastArgs[2],a=t.lastArgs[3];for(let c=0;c<64;c++)ve(e,i[c],s.velX[c]+n.accX[c]*.01,.001,`stepPos got a stale vx for body ${c} — did stepVel add a·dt?`),ve(e,a[c],s.velY[c]+n.accY[c]*.01,.001,`stepPos got a stale vy for body ${c}`)}},{name:"velocity kernel: <code>v' = v + a·dt</code>",run:async e=>{const t=e.kernels[0],s=new Array(64),n=new Array(64),i=new Array(64),a=new Array(64);for(let h=0;h<64;h++)s[h]=h*.1-3,n[h]=2-h*.05,i[h]=Math.sin(h)*4,a[h]=Math.cos(h)*4;const c=t(s,n,i,a,.5);for(let h=0;h<64;h++)ve(e,c[h][0],s[h]+i[h]*.5,.001,_t(c[h][0],s[h],i[h],.5,.001)||`vx of body ${h}`),ve(e,c[h][1],n[h]+a[h]*.5,.001,_t(c[h][1],n[h],a[h],.5,.001)||`vy of body ${h}`)}},{name:"position kernel: <code>x' = x + v·dt</code>",run:async e=>{const t=e.kernels[1],s=new Array(64),n=new Array(64),i=new Array(64),a=new Array(64);for(let h=0;h<64;h++)s[h]=h*.25,n[h]=-h*.125,i[h]=1+h*.02,a[h]=-2+h*.03;const c=t(s,n,i,a,.2);for(let h=0;h<64;h++)ve(e,c[h][0],s[h]+i[h]*.2,.001,_t(c[h][0],s[h],i[h],.2,.001)||`x of body ${h}`),ve(e,c[h][1],n[h]+a[h]*.2,.001,_t(c[h][1],n[h],a[h],.2,.001)||`y of body ${h}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.utils.seededRandom(31),s=[];for(let q=0;q<6;q++){const $e=new Array(64);for(let Ze=0;Ze<64;Ze++)$e[Ze]=t()*4-2;s.push($e)}const[n,i,a,c,h,I]=s,U=e.kernels[0](n,i,a,c,.025),j=e.kernels[1](h,I,n,i,.025);for(let q=0;q<64;q++)ve(e,U[q][0],n[q]+a[q]*.025,.001,_t(U[q][0],n[q],a[q],.025,.001)||`vx of body ${q}`),ve(e,U[q][1],i[q]+c[q]*.025,.001,_t(U[q][1],i[q],c[q],.025,.001)||`vy of body ${q}`),ve(e,j[q][0],h[q]+n[q]*.025,.001,_t(j[q][0],h[q],n[q],.025,.001)||`x of body ${q}`),ve(e,j[q][1],I[q]+i[q]*.025,.001,_t(j[q][1],I[q],i[q],.025,.001)||`y of body ${q}`)}}]},{slug:"full-simulation",title:"Put It Together: 128 Bodies",intro:`<p>Everything from this module, running as one machine. The three kernels below are
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
`,inputs:e=>tt(e,128,55),publicTests:[{name:"all ten ticks ran — the final tick saw step-nine positions",run:async e=>{e.assert(e.kernels.length>=3,`expected 3 kernels (accel, stepVel, stepPos), found ${e.kernels.length}`);const t=e.kernels[0];e.assert(Array.isArray(t.lastArgs),"the accel kernel was never called — is the loop wired up?");const s=tt(e.utils,128,55),n=rr(s,9,.01,.1),i=t.lastArgs[0],a=t.lastArgs[1];for(let c=0;c<128;c+=7)ve(e,i[c],n.posX[c],.005,`tick 10 saw a wrong x for body ${c} — is the new state carried between ticks?`),ve(e,a[c],n.posY[c],.005,`tick 10 saw a wrong y for body ${c}`)}},{name:"momentum is conserved across the whole run",run:async e=>{const t=e.kernels[1];e.assert(Array.isArray(t.lastArgs),"stepVel was never called — is the loop wired up?");const s=t(...t.lastArgs),n=tt(e.utils,128,55);let i=0,a=0,c=0,h=0;for(let I=0;I<128;I++)i+=n.mass[I]*n.velX[I],a+=n.mass[I]*n.velY[I],c+=n.mass[I]*s[I][0],h+=n.mass[I]*s[I][1];e.assertClose(c,i,.05,"total x-momentum drifted — forces should cancel pairwise"),e.assertClose(h,a,.05,"total y-momentum drifted")}},{name:"one tick, rebuilt from scratch, matches the physics",run:async e=>{const t=tt(e.utils,128,55),s=fa(e.kernels[0],e.kernels[1],e.kernels[2],t,t.mass,.01,.1),n=rr(t,1,.01,.1);for(const i of[0,1,42,127])ve(e,s.posX[i],n.posX[i],.002,`x of body ${i} after one tick`),ve(e,s.posY[i],n.posY[i],.002,`y of body ${i} after one tick`),ve(e,s.velX[i],n.velX[i],.002,`vx of body ${i} after one tick`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=tt(e.utils,128,991);let s=t;for(let i=0;i<5;i++)s=fa(e.kernels[0],e.kernels[1],e.kernels[2],s,t.mass,.01,.1);const n=rr(t,5,.01,.1);for(let i=0;i<128;i++)ve(e,s.posX[i],n.posX[i],.005,`x of body ${i} after five ticks`),ve(e,s.posY[i],n.posY[i],.005,`y of body ${i} after five ticks`)}}]}]},Zl=Object.freeze({__proto__:null,default:Jl});function xe(e,t,s){return(s*e+t)*4}function ts(e){return 64+40*Math.sin(e*2*Math.PI/128)}function ks(e,t,s){const n=[];for(let i=0;i<t;i++)e[xe(t,s,i)]>128&&n.push(i);return n}function Ts(e){let t=0;for(let s=0;s<e.length;s++)t+=e[s];return t/e.length}function an(e,t){const s=e-63.5,n=t-63.5,i=Math.sqrt(s*s+n*n),a=.5+.5*Math.cos(i*.35),c=Math.max(0,1-i/96),h=a*c;return[.4*h*255,.75*h*255,h*255]}function st(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function ya(e,t){const s="red is following the row instead of the column — the horizontal ramp is this.thread.x / 128";return[[255*t/128,s],[255*(127-t)/128,s],[Math.min(255,e*255),"color channels run 0–1, so an undivided this.thread.x saturates every column past the first — divide it by 128"]]}function Ql(e,t,s){return Math.abs(e-t)<=2&&Math.abs(e-255*s/128)<=3?"green is constant down the canvas and matches this column's x ramp — this.thread.x and this.thread.y are swapped; green rises with this.thread.y":null}function xa(e,t){const s=e[xe(128,0,t)];let n=1;for(;n<128&&e[xe(128,n,t)]===s;)n++;return n===16||n===128?null:`your cells are ${n} pixels wide, not 16 — the cell index is Math.floor(coordinate / 16)`}function ba(e,t){return[[t(64+40*Math.sin(e)),"that is Math.sin() of the raw pixel count — one period across 128 px needs x * 2 * Math.PI / 128"],[t(64+40*Math.sin(e*2*Math.PI/64)),"two periods fit the canvas — divide x by 128, the full width, for one"],[t(64+40*Math.sin(e*360/128)),"Math.sin takes radians, not degrees — the scale is 2 * Math.PI / 128"],[t(64),"the curve is still the constant 64 — it never became a function of x"]]}function Ft(e,t,s){const n=[.4,.75,1][s],i=e-63.5,a=t-63.5,c=Math.sqrt(i*i+a*a),h=.5+.5*Math.cos(c*.35),I=Math.max(0,1-c/96);return[[h,"the fade never got multiplied in — v = wave * fade"],[I,"that is the bare fade — the cosine ripple is missing from v"],[Math.max(0,Math.cos(c*.35))*I,"the cosine still swings negative — remap it with 0.5 + 0.5 * Math.cos(r * 0.35)"]].map(U=>[Math.min(1,U[0])*n*255,U[1]])}var eu={id:"3-1",track:3,title:"Pixels from Scratch",blurb:"Graphical kernels and <code>this.color()</code>: gradients, patterns and plots, one thread per pixel.",tasks:[{slug:"coordinate-gradient",title:"Paint with Coordinates",intro:`<p>Set <code>graphical: true</code> and a kernel stops returning numbers — instead
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
`,publicTests:[{name:"paints a graphical <code>128×128</code> canvas",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===128&&e.canvas.height===128,`expected a 128×128 canvas, got ${e.canvas.width}×${e.canvas.height}`),e.assert(e.getPixels().length===16384*4,"pixel buffer should hold 128×128 RGBA values")}},{name:"red rises left to right: <code>this.thread.x / 128</code>",run:async e=>{const t=e.getPixels();for(const s of[3,64,124])for(let n=0;n<128;n+=7){const i=t[xe(128,n,s)],a=255*n/128,c=st(i,a,2.5,ya(n,s));e.assertClose(i,a,2.5,c||`red at column ${n} (buffer row ${s})`)}}},{name:"green rises with <code>this.thread.y</code>; blue holds at 0.5",run:async e=>{const t=e.getPixels(),s=t[xe(128,20,0)+1],n=t[xe(128,20,127)+1];e.assert(Math.min(s,n)<=4&&Math.max(s,n)>=248,Ql(s,n,20)||`green should ramp 0 → 252 across the canvas, got edge values ${s} and ${n}`);for(const[i,a]of[[10,10],[90,40],[64,100]])e.assertClose(t[xe(128,i,a)+2],127.5,2.5,`blue at (${i}, row ${a})`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.getPixels(),s=t[xe(128,5,0)+1]<t[xe(128,5,127)+1];for(let n=0;n<128;n+=5){const i=s?n:127-n;for(let a=0;a<128;a+=5){const c=xe(128,a,n),h=st(t[c],255*a/128,2.5,ya(a,n));e.assertClose(t[c],255*a/128,2.5,h||`red at (${a}, y=${i})`),e.assertClose(t[c+1],255*i/128,2.5,`green at (${a}, y=${i})`),e.assertClose(t[c+2],127.5,2.5,`blue at (${a}, y=${i})`),e.assert(t[c+3]===255,`alpha at (${a}, y=${i}) should be 255`)}}}}]},{slug:"checkerboard",title:"Checkerboard Logic",intro:`<p>Smooth ramps become hard-edged patterns with two tools:
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
`,publicTests:[{name:"every pixel is pure black or pure white",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.getPixels();e.assert(t.length===16384*4,"expected a 128×128 canvas");for(let s=0;s<t.length;s+=4){const n=t[s];e.assert(n<=1||n>=254,`pixel at byte ${s} is gray (${n}) — the parity should be exactly 0 or 1`),e.assert(Math.abs(t[s+1]-n)<=1&&Math.abs(t[s+2]-n)<=1,`pixel at byte ${s} is tinted — use the same value for r, g and b`)}}},{name:"cells are 16 pixels wide and alternate along x",run:async e=>{const t=e.getPixels();for(const s of[8,40,100]){const n=t[xe(128,2,s)],i=t[xe(128,13,s)];e.assert(Math.abs(n-i)<=1,xa(t,s)||`columns 2 and 13 share a 16-px cell but differ on buffer row ${s}`);const a=t[xe(128,8,s)],c=t[xe(128,24,s)];e.assert(Math.abs(a-c)>=250,xa(t,s)||`columns 8 and 24 are in adjacent cells but match on buffer row ${s} — still stripes?`)}}},{name:"cells alternate along y too — that's what makes it a checkerboard",run:async e=>{const t=e.getPixels();for(const s of[8,40,100]){const n=t[xe(128,s,2)],i=t[xe(128,s,13)];e.assert(Math.abs(n-i)<=1,`rows 2 and 13 share a 16-px cell but differ in column ${s}`);const a=t[xe(128,s,8)],c=t[xe(128,s,24)];e.assert(Math.abs(a-c)>=250,`rows 8 and 24 are in adjacent cells but match in column ${s} — did you use this.thread.y?`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.getPixels(),s=t[0]>=254?255:0;let n=0;for(let i=0;i<128;i++)for(let a=0;a<128;a++){const h=(Math.floor(a/16)+Math.floor(i/16))%2===0?s:255-s,I=t[xe(128,a,i)];e.assert(Math.abs(I-h)<=1,`pixel (${a}, row ${i}) breaks the checkerboard: got ${I}`),I>=254&&n++}e.assert(n===16384/2,`expected exactly half the pixels white, got ${n}`)}}]},{slug:"plot-a-wave",title:"Plot a Function",intro:`<p>How do you plot <code>y = f(x)</code> when no thread can draw a line? Flip the
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
`,publicTests:[{name:"a thin curve on a dark background",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.getPixels();e.assert(t.length===16384*4,"expected a 128×128 canvas");let s=0;for(let i=0;i<t.length;i+=4)t[i]>128?s++:e.assert(t[i]<40,`background pixel at byte ${i} is not dark (red ${t[i]})`);const n=s/16384;e.assert(n>.01&&n<.15,`expected a thin curve (1–15% of pixels lit), got ${(n*100).toFixed(1)}%`)}},{name:"every column crosses the curve exactly once",run:async e=>{const t=e.getPixels();for(let s=0;s<128;s+=4){const n=ks(t,128,s);e.assert(n.length>=1,`column ${s} has no lit pixels`),e.assert(n.length<=14,`column ${s} has ${n.length} lit pixels — the band should stay thin`),e.assert(n[n.length-1]-n[0]===n.length-1,`column ${s} lights two separate bands — the curve should cross it once`)}}},{name:"the curve follows <code>64 + 40·sin(2πx/128)</code>",run:async e=>{const t=e.getPixels(),s=Ts(ks(t,128,32)),n=Math.abs(s-ts(32))<=4,i=Math.abs(s-(127-ts(32)))<=4;e.assert(n||i,`at x=32 the curve should sit ~40 px from the middle (y≈104), found its center at buffer row ${s.toFixed(1)}`);const a=c=>n?c:127-c;for(const c of[0,8,16,32,48,64,80,96,112,120]){const h=Ts(ks(t,128,c)),I=a(ts(c)),U=st(h,I,3,ba(c,a));e.assertClose(h,I,3,U||`curve center in column ${c}`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.getPixels(),s=Ts(ks(t,128,32)),n=Math.abs(s-ts(32))<=4;e.assert(n||Math.abs(s-(127-ts(32)))<=4,"curve peak is not where sin() puts it");for(let i=0;i<128;i++){const a=ks(t,128,i);e.assert(a.length>=1&&a.length<=14,`column ${i}: ${a.length} lit pixels`);const c=U=>n?U:127-U,h=c(ts(i)),I=st(Ts(a),h,3,ba(i,c));e.assertClose(Ts(a),h,3,I||`curve center in column ${i}`)}}}]},{slug:"radial-ripples",title:"Ripples: Think in Polar",intro:`<p>Gradients, cells and curves all thought in x and y. The last move of this module
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
`,publicTests:[{name:"the picture is radially symmetric about the center",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.getPixels();e.assert(t.length===16384*4,"expected a 128×128 canvas");for(const[s,n]of[[10,30],[45,8],[70,100],[120,60],[33,33]]){const i=xe(128,s,n),a=xe(128,127-s,n),c=xe(128,s,127-n);for(let h=0;h<3;h++)e.assert(Math.abs(t[i+h]-t[a+h])<=2,`pixel (${s}, row ${n}) and its horizontal mirror disagree — is the center at (63.5, 63.5)?`),e.assert(Math.abs(t[i+h]-t[c+h])<=2,`pixel (${s}, row ${n}) and its vertical mirror disagree — is the center at (63.5, 63.5)?`)}}},{name:"crests and troughs land where <code>cos(0.35r)</code> puts them",run:async e=>{const t=e.getPixels();for(const i of[64,73,81,99,120]){const a=xe(128,i,63),c=an(i,63.5+.5),h=st(t[a+2],c[2],3,Ft(i,64,2));e.assertClose(t[a+2],c[2],3,h||`blue in column ${i} of the center row`)}const s=t[xe(128,64,63)+2],n=t[xe(128,73,63)+2];e.assert(s-n>200,`the first trough should be nearly black next to the bright center (got ${s} vs ${n})`)}},{name:"blue tint and edge fade: <code>b &gt; g &gt; r</code>, corners dark",run:async e=>{const t=e.getPixels(),s=xe(128,81,63),[n,i,a]=an(81,64);e.assertClose(t[s],n,3,st(t[s],n,3,Ft(81,64,0))||"red on the first ring"),e.assertClose(t[s+1],i,3,st(t[s+1],i,3,Ft(81,64,1))||"green on the first ring"),e.assertClose(t[s+2],a,3,st(t[s+2],a,3,Ft(81,64,2))||"blue on the first ring"),e.assert(t[s+2]>t[s+1]&&t[s+1]>t[s],"ring pixels should be tinted blue: b > g > r");for(const[c,h]of[[0,0],[127,0],[0,127],[127,127]]){const I=xe(128,c,h),[U,j,q]=an(c<64?0:127,h<64?0:127);e.assertClose(t[I],U,3,`red in corner (${c}, row ${h})`),e.assertClose(t[I+1],j,3,`green in corner (${c}, row ${h})`),e.assertClose(t[I+2],q,3,`blue in corner (${c}, row ${h})`),e.assert(t[s+2]-t[I+2]>150,`corner (${c}, row ${h}) should be far dimmer than the first ring`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.getPixels();for(let s=0;s<128;s+=3)for(let n=0;n<128;n+=3){const i=xe(128,n,s),[a,c,h]=an(n,s);e.assertClose(t[i],a,3,st(t[i],a,3,Ft(n,s,0))||`red at (${n}, row ${s})`),e.assertClose(t[i+1],c,3,st(t[i+1],c,3,Ft(n,s,1))||`green at (${n}, row ${s})`),e.assertClose(t[i+2],h,3,st(t[i+2],h,3,Ft(n,s,2))||`blue at (${n}, row ${s})`)}}}]}]},tu=Object.freeze({__proto__:null,default:eu});const os=100;function Na(e,t,s,n,i=os){let a=e,c=t,h=0;for(let I=0;I<i;I++)if(a*a+c*c<4){const U=a*a-c*c+s;c=2*a*c+n,a=U,h+=1}return{count:h,zr:a,zi:c}}function on(e,t,s,n,i=os){const{count:a,zr:c,zi:h}=Na(e,t,s,n,i);return a>=i?i:a+1-Math.log2(.5*Math.log2(c*c+h*h))}function ar(e){return[Math.round(e*255),Math.round(e*e*255),Math.round((.5+.5*e)*255)]}function ct(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function or(e,t,s,n,i){const a=e+n*s,c=t+i*s;return[[Math.sqrt(a*a+c*c),"that is |c|, not |c|² — return cr * cr + ci * ci without the square root"],[n*s*(n*s)+i*s*(i*s),"the view offsets never got added — cr is xMin + x * step, ci is yMin + y * step"],[e*e+t*t,"every cell reports the corner of the view — the integer this.thread.x on the left of the multiply makes gpu.js truncate step to 0; hoist the thread id into a const first"]]}function ln(){return[[os,"every point reached the 100 cap — counting continued after z escaped, so the guard zr * zr + zi * zi < 4 is either missing or not wrapping the count"],[0,"count came back 0 — no guarded pass ever ran; the guard admits z while zr * zr + zi * zi is BELOW 4"]]}function va(e,t,s){return e>=253&&t>=253&&s>=253?"the interior came out white — count = 100 pixels are falling into the shade branch; shade only when count < 100 and paint the rest black":null}function lr(e,t){const s=Na(e,t),n=s.zr*s.zr+s.zi*s.zi;return[[s.count,"that is the raw integer count — the fractional correction 1 − log2(0.5 · log2|z|²) is missing"],[s.count+1-Math.log(.5*Math.log(n)),"Math.log is the natural logarithm — the normalized iteration count takes log2 twice"],[s.count+1-Math.log2(Math.log2(n)),"the halving is missing — log2|z| is 0.5 * Math.log2(zr * zr + zi * zi)"]]}var su={id:"3-2",track:3,title:"Escape-Time Fractals",blurb:"Mandelbrot and Julia sets with smooth coloring — infinite detail from a ten-line kernel.",tasks:[{slug:"pixel-to-plane",title:"Map Pixels to the Complex Plane",intro:`<p>A fractal isn't drawn — it's <strong>evaluated</strong>. There is a function
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
`,publicTests:[{name:"the view <code>(-2, -2, 4/64)</code> puts the origin at cell [32][32]",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(-2,-2,4/64);e.assert(t&&t.length===64,`expected 64 rows, got ${t&&t.length}`),e.assert(t[0]&&t[0].length===64,"each row should hold 64 values");const s=4/64,n=(i,a,c,h)=>ct(t[i][a],c,h,or(-2,-2,s,a,i));e.assertClose(t[32][32],0,.001,n(32,32,0,.001)||"cell [32][32] should be the origin, |c|² = 0"),e.assertClose(t[32][48],1,.001,n(32,48,1,.001)||"cell [32][48] sits at c = 1 + 0i, so |c|² = 1"),e.assertClose(t[0][0],8,.01,n(0,0,8,.01)||"cell [0][0] sits at c = -2 - 2i, so |c|² = 8")}},{name:"a different camera — <code>(0, 0, 0.5)</code> — moves every cell",run:async e=>{const t=e.kernel(0,0,.5),s=[[0,0],[2,3],[7,7],[63,1]];for(const[n,i]of s){const a=.25*(i*i+n*n),c=ct(t[n][i],a,.01,or(0,0,.5,i,n));e.assertClose(t[n][i],a,.01,c||`cell [${n}][${i}]`)}}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(-1,2,.25);for(let s=0;s<64;s+=3)for(let n=0;n<64;n+=3){const i=-1+n*.25,a=2+s*.25,c=ct(t[s][n],i*i+a*a,.01,or(-1,2,.25,n,s));e.assertClose(t[s][n],i*i+a*a,.01,c||`cell [${s}][${n}]`)}}}]},{slug:"escape-time",title:"The Escape-Time Loop",intro:`<p>The Mandelbrot set asks one question at every point <code>c</code>: start
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
`,publicTests:[{name:"the classic view shows both worlds — black interior AND shaded exterior",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===128&&e.canvas.height===128,`expected a 128×128 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();let s=0,n=0;for(let i=0;i<t.length;i+=4)t[i]+t[i+1]+t[i+2]<=3?s++:t[i+2]>100&&n++;e.assert(s>300,`expected a black interior — found only ${s} black pixels`),e.assert(n>300,`expected a shaded exterior — found only ${n} blue-ish pixels`)}},{name:"a window inside the set is pure black",run:async e=>{e.kernel(-.2,-.05,.001);const t=e.getPixels();for(let s=0;s<t.length;s+=401*4)e.assert(t[s]<=2&&t[s+1]<=2&&t[s+2]<=2,va(t[s],t[s+1],t[s+2])||`interior pixel at byte ${s} should be black, got rgb(${t[s]}, ${t[s+1]}, ${t[s+2]})`)}},{name:"far outside, every pixel wears the count-1 shade",run:async e=>{e.kernel(2.5,2.5,.001);const t=e.getPixels(),[s,n,i]=ar(1/os);for(let a=0;a<t.length;a+=401*4)e.assertClose(t[a],s,2,`red at byte ${a}`),e.assertClose(t[a+1],n,2,`green at byte ${a}`),e.assertClose(t[a+2],i,2,`blue at byte ${a}`)}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(-1.05,-.05,8e-4);let t=e.getPixels();for(let a=0;a<t.length;a+=293*4)e.assert(t[a]<=2&&t[a+1]<=2&&t[a+2]<=2,va(t[a],t[a+1],t[a+2])||`bulb pixel at byte ${a} should be black, got rgb(${t[a]}, ${t[a+1]}, ${t[a+2]})`);e.kernel(-9,0,.001),t=e.getPixels();const[s,n,i]=ar(1/os);for(let a=0;a<t.length;a+=293*4)e.assertClose(t[a],s,2,`red at byte ${a}`),e.assertClose(t[a+1],n,2,`green at byte ${a}`),e.assertClose(t[a+2],i,2,`blue at byte ${a}`)}}]},{slug:"smooth-coloring",title:"Smooth Out the Bands",intro:`<p>Look closely at task 3's exterior and you'll see hard rings: iteration counts are
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
`,publicTests:[{name:"interior cells still return exactly 100",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(-.2,-.05,.002);e.assert(t&&t.length===64&&t[0].length===64,"expected a 64×64 grid");for(let s=0;s<64;s+=7)for(let n=0;n<64;n+=7)e.assertClose(t[s][n],100,.001,`interior cell [${s}][${n}]`)}},{name:"escaped cells carry a fraction that matches the formula",run:async e=>{const t=e.kernel(3,1,.01);let s=!1;const n=[[0,0],[10,20],[33,7],[63,63]];for(const[i,a]of n){const c=on(0,0,3+a*.01,1+i*.01),h=ct(t[i][a],c,.02,lr(3+a*.01,1+i*.01));e.assertClose(t[i][a],c,.02,h||`cell [${i}][${a}]`),Math.abs(t[i][a]-Math.round(t[i][a]))>.05&&(s=!0)}e.assert(s,"every sampled value is a whole number — are you still returning the raw count?")}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(-4,2,.005);for(let n=0;n<64;n+=9)for(let i=0;i<64;i+=9){const a=on(0,0,-4+i*.005,2+n*.005),c=ct(t[n][i],a,.02,lr(-4+i*.005,2+n*.005));e.assertClose(t[n][i],a,.02,c||`far cell [${n}][${i}]`)}const s=e.kernel(1.5,-.032,.001);for(let n=0;n<64;n+=9)for(let i=0;i<64;i+=9){const a=on(0,0,1.5+i*.001,-.032+n*.001),c=ct(s[n][i],a,.02,lr(1.5+i*.001,-.032+n*.001));e.assertClose(s[n][i],a,.02,c||`near cell [${n}][${i}]`)}}}]},{slug:"julia-dial",title:"Julia Sets: Turn the Dial",intro:`<p>Here's the payoff. Take the exact loop you've built and <strong>flip the
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
`,publicTests:[{name:"with <code>c = 0</code> the Julia set is the unit disk — inside black, outside shaded",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===128&&e.canvas.height===128,`expected a 128×128 canvas, got ${e.canvas.width}×${e.canvas.height}`),e.kernel(0,0);const t=e.getPixels();for(let s=4;s<128;s+=8)for(let n=4;n<128;n+=8){const i=-1.6+n*.025,a=-1.6+s*.025,c=i*i+a*a,h=(s*128+n)*4;c<.9?e.assert(t[h]+t[h+1]+t[h+2]<=3,`pixel (${n}, ${s}) is inside the unit disk — expected black, got rgb(${t[h]}, ${t[h+1]}, ${t[h+2]})`):c>4.25&&e.assert(t[h+2]>100,`pixel (${n}, ${s}) is far outside the disk — expected a blue-ish shade, got rgb(${t[h]}, ${t[h+1]}, ${t[h+2]})`)}}},{name:"c is a live argument — turn the dial and the center pixel flips",run:async e=>{e.kernel(0,0);let t=e.getPixels();const s=8256*4;e.assert(t[s]+t[s+1]+t[s+2]<=3,"with c = 0 the center pixel (z₀ = 0) never escapes — it should be black"),e.kernel(-2.5,0),t=e.getPixels(),e.assert(t[s+2]>100,"with c = -2.5 the center pixel escapes in one step — it should be shaded, not black. Is c actually used in the loop?")}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(0,0);const t=e.getPixels();let s=10836*4;e.assert(t[s]+t[s+1]+t[s+2]<=3,`pixel (84, 84) lies inside the unit disk — expected black, got rgb(${t[s]}, ${t[s+1]}, ${t[s+2]})`),s=8196*4;const n=on(-1.5,0,0,0)/os,[i,a,c]=ar(n);e.assertClose(t[s],i,4,"red at pixel (4, 64)"),e.assertClose(t[s+1],a,4,"green at pixel (4, 64)"),e.assertClose(t[s+2],c,4,"blue at pixel (4, 64)")}}]}]},nu=Object.freeze({__proto__:null,default:su});const It=16;function $s(e=It){const t=new Array(e);for(let s=0;s<e;s++)t[s]=new Array(e).fill(0);return t}function Ee(e,t=It){const s=$s(t);for(const[n,i]of e)s[n][i]=1;return s}function Ct(e,t,s=It,n=.35){const i=e.seededRandom(t),a=$s(s);for(let c=0;c<s;c++)for(let h=0;h<s;h++)a[c][h]=i()<n?1:0;return a}function Er(e,t,s){const n=e.length;let i=0;for(let a=-1;a<=1;a++)for(let c=-1;c<=1;c++)a===0&&c===0||(i+=e[(t+a+n)%n][(s+c+n)%n]);return i}function vt(e,t=Et,s=Ke){const n=e.length,i=$s(n);for(let a=0;a<n;a++)for(let c=0;c<n;c++){const h=Er(e,a,c);i[a][c]=e[a][c]===1?s[h]:t[h]}return i}function wa(e,t,s=Et,n=Ke){let i=e;for(let a=0;a<t;a++)i=vt(i,s,n);return i}function ka(e){let t=0;for(let s=0;s<e.length;s++)for(let n=0;n<e[s].length;n++)t+=e[s][n];return t}function nt(e,t,s,n,i){e.assert(t&&t.length===s.length,`${n} — expected ${s.length} rows`);const a=i||n;for(let c=0;c<s.length;c++)for(let h=0;h<s.length;h++)e.assertClose(t[c][h],s[c][h],.001,`${a} — cell [${c}][${h}]`)}function ru(e,t){for(let s=0;s<t.length;s++){if(!e[s])return!1;for(let n=0;n<t.length;n++)if(!(Math.abs(e[s][n]-t[s][n])<=.001))return!1}return!0}function ht(e,t,s=Et,n=Ke){const i=[[vt(t,n,n),"a dead cell with 2 neighbors came alive — birth is on exactly 3; 2 is what lets an already-live cell survive"],[vt(t,s,s),"live cells with 2 neighbors died — survival covers 2 or 3, and only birth is limited to exactly 3"],[vt(t,n,s),"the two rules are swapped — a dead cell follows the birth rule, a live cell the survival rule"],[t,"the world came back unchanged — the rule never reached the return value"]];for(const[a,c]of i)if(ru(e,a))return c;return null}function ur(e,t,s){return[[Er(e,t,s)+e[t][s],"your own cell is still inside the 3×3 sum — subtract grid[this.thread.y][this.thread.x] at the end"]]}function cr(e){return Math.abs(e)<=.001?"the edge did not wrap — add the width before the modulo, (this.thread.y + dy + 16) % 16, because a bare % can go negative":null}function iu(e,t,s){const n=`gen ${t}: ${s} alive`;return e.logs.some(i=>i.type==="log"&&i.text&&i.text.includes(n))?"that generation logged the population from BEFORE its step — count the live cells after current = step(current)":null}function hr(e,t,s){return e===s-t?"the two colors are swapped — live cells take the green, dead cells the dark background":null}function dr(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}const Et=[0,0,0,1,0,0,0,0,0],Ke=[0,0,1,1,0,0,0,0,0],ss=[0,0,0,1,0,0,1,0,0],pr=[0,0,0,1,0,0,1,1,1],fr=[0,0,0,1,1,0,1,1,1],Ss=[[7,6],[7,7],[7,8]],mr=[[3,3],[3,4],[4,3],[4,4]],un=[[1,2],[2,3],[3,1],[3,2],[3,3]],Ta=[[6,8],[6,9],[7,7],[7,8],[8,8]];function au(e,t,s,n=It){return e.map(([i,a])=>[(i+t+n)%n,(a+s+n)%n])}var ou={id:"3-3",track:3,title:"Cellular Automata",blurb:"Conway's Life and friends: feed a kernel's output back in and watch worlds evolve.",tasks:[{slug:"neighbor-census",title:"The Neighbor Census",intro:`<p>A cellular automaton is a world of cells, each one dead (<code>0</code>) or alive
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
`,inputs:e=>({grid:Ct(e,1101)}),publicTests:[{name:"a lone cell has zero neighbors — each of its eight neighbors sees one",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ee([[5,5]]),s=e.kernel(t);e.assertClose(s[5][5],0,.001,dr(s[5][5],0,.001,ur(t,5,5))||"the live cell itself (it is not its own neighbor)");const n=[[4,4],[4,5],[4,6],[5,4],[5,6],[6,4],[6,5],[6,6]];for(const[i,a]of n){const c=dr(s[i][a],1,.001,ur(t,i,a));e.assertClose(s[i][a],1,.001,c||`neighbor cell [${i}][${a}]`)}e.assertClose(s[10][10],0,.001,"a far-away cell")}},{name:"the world wraps: a corner cell is seen across all four edges",run:async e=>{const t=e.kernel(Ee([[0,0]]));e.assertClose(t[15][15],1,.001,cr(t[15][15])||"diagonal wrap — cell [15][15]"),e.assertClose(t[0][15],1,.001,cr(t[0][15])||"horizontal wrap — cell [0][15]"),e.assertClose(t[15][0],1,.001,cr(t[15][0])||"vertical wrap — cell [15][0]"),e.assertClose(t[1][1],1,.001,"ordinary diagonal — cell [1][1]"),e.assertClose(t[0][0],0,.001,"the corner cell itself")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ct(e.utils,2202),s=e.kernel(t);for(let n=0;n<It;n++)for(let i=0;i<It;i++){const a=Er(t,n,i),c=dr(s[n][i],a,.001,ur(t,n,i));e.assertClose(s[n][i],a,.001,c||`cell [${n}][${i}]`)}}}]},{slug:"one-tick",title:"One Tick of Life",intro:`<p>In 1970 John Conway picked the simplest rules he could find that make a world worth
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
`,inputs:()=>({world:Ee(Ss)}),publicTests:[{name:"the blinker: three-in-a-row flips to three-in-a-column",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ee(Ss),s=e.kernel(t);nt(e,s,Ee([[6,7],[7,7],[8,7]]),"blinker after one tick",ht(s,t))}},{name:"the block: a 2×2 square is a still life — nothing moves",run:async e=>{const t=Ee(mr),s=e.kernel(t);nt(e,s,Ee(mr),"block after one tick",ht(s,t))}},{name:"an empty world stays empty — no spontaneous generation",run:async e=>{const t=e.kernel($s());nt(e,t,$s(),"empty world after one tick")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ct(e.utils,4404),s=e.kernel(t);nt(e,s,vt(t),"random world, one tick",ht(s,t))}},{name:"private test #2",run:async e=>{const t=Ct(e.utils,5505,It,.6),s=e.kernel(t);nt(e,s,vt(t),"crowded world, one tick",ht(s,t))}}]},{slug:"generations",title:"Generations: Feed It Back",intro:`<p>One tick is a snapshot; a world is a movie. A kernel has no memory of the previous
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
`,inputs:()=>({world:Ee(Ta)}),publicTests:[{name:"six generations logged, matching the R-pentomino's true population history",run:async e=>{let t=Ee(Ta);for(let s=1;s<=6;s++){const n=ka(t);t=vt(t);const i="gen "+s+": "+ka(t)+" alive",a=e.logs.some(c=>c.type==="log"&&c.text&&c.text.includes(i));e.assert(a,iu(e,s,n)||`expected a log line containing "${i}"`)}}},{name:"the step kernel is still a faithful B3/S23 tick",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ee(Ss),s=e.kernel(t);nt(e,s,Ee([[6,7],[7,7],[8,7]]),"blinker after one tick",ht(s,t))}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ct(e.utils,6606);let s=t;for(let n=0;n<3;n++)s=e.kernel(s);nt(e,s,wa(t,3),"random world after three ticks")}}]},{slug:"glider-on-screen",title:"Watch the Glider Fly",intro:`<p>The <strong>glider</strong> is five cells that <em>travel</em>. No individual cell
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
`,inputs:()=>({world:Ee(un)}),publicTests:[{name:"the glider translates: four ticks move the whole pattern down-right by one",run:async e=>{const t=e.kernels.find(n=>n.kernel&&!n.kernel.graphical);e.assert(t,"no numeric (non-graphical) step kernel found");let s=Ee(un);for(let n=0;n<4;n++)s=t(s);nt(e,s,Ee(au(un,1,1)),"glider after four ticks")}},{name:"canvas is 16×16 and shows exactly the 5 glider cells lit green",run:async e=>{e.assert(e.canvas,"no canvas — did you call render(paint.canvas)?"),e.assert(e.canvas.width===16&&e.canvas.height===16,`expected a 16×16 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.kernels.find(i=>i.kernel&&i.kernel.graphical);e.assert(t,"no graphical paint kernel found"),t(wa(Ee(un),8));const s=t.getPixels();let n=0;for(let i=0;i<s.length;i+=4){const a=s[i+1];e.assert(a>200||a<40,`pixel at byte ${i} is neither live-green nor dead-dark (green = ${a})`),a>200&&n++}e.assert(n===5,hr(n,5,256)||`a glider is always 5 cells — found ${n} lit pixels`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(c=>c.kernel&&!c.kernel.graphical),s=e.kernels.find(c=>c.kernel&&c.kernel.graphical);e.assert(t&&s,"expected a numeric and a graphical kernel");const n=c=>{let h=0;for(let I=0;I<c.length;I+=4)c[I+1]>200&&h++;return h};s(t(Ee(Ss)));const i=n(s.getPixels());e.assert(i===3,hr(i,3,256)||"stepped blinker should light 3 pixels"),s(Ee(mr));const a=n(s.getPixels());e.assert(a===4,hr(a,4,256)||"block should light 4 pixels")}}]},{slug:"any-rule",title:"One Kernel, Every Universe",intro:`<p>B3/S23 is one point in a whole family. Any <em>outer-totalistic</em> rule is fully
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
`,inputs:e=>({world:Ct(e,7707),lifeBorn:Et.slice(),lifeStay:Ke.slice(),highlifeBorn:ss.slice()}),publicTests:[{name:"fed the Life tables, it is still Life: the blinker spins",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ee(Ss),s=e.kernel(t,Et,Ke);nt(e,s,Ee([[6,7],[7,7],[8,7]]),"blinker under B3/S23",ht(s,t,Et,Ke))}},{name:"HighLife's B6: six neighbors ignite a dead cell that Life leaves dark",run:async e=>{const t=Ee([[4,4],[4,5],[4,6],[5,4],[5,6],[6,4]]),s=e.kernel(t,Et,Ke),n=e.kernel(t,ss,Ke);e.assertClose(s[5][5],0,.001,ht(s,t,Et,Ke)||"under Life (B3), 6 neighbors do not give birth"),e.assertClose(n[5][5],1,.001,ht(n,t,ss,Ke)||"under HighLife (B36), 6 neighbors do")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ct(e.utils,8808,It,.5),s=e.kernel(t,pr,fr);nt(e,s,vt(t,pr,fr),"Day & Night, one tick",ht(s,t,pr,fr))}},{name:"private test #2",run:async e=>{const t=Ct(e.utils,9909),s=e.kernel(t,ss,Ke);nt(e,s,vt(t,ss,Ke),"HighLife, one tick",ht(s,t,ss,Ke))}}]}]},lu=Object.freeze({__proto__:null,default:ou});const fe={du:.2,dv:.1,f:.035,k:.06,dt:1};function Ge(e,t){const s=new Array(e);for(let n=0;n<e;n++)s[n]=new Array(e).fill(t);return s}function Sa(e,t,s){const n=e.seededRandom(s),i=new Array(t);for(let a=0;a<t;a++){const c=new Array(t);for(let h=0;h<t;h++)c[h]=Math.round(n()*1e3)/1e3;i[a]=c}return i}function Vt(e){const t=e.length,s=new Array(t);for(let n=0;n<t;n++){const i=new Array(t),a=n===0?t-1:n-1,c=n===t-1?0:n+1;for(let h=0;h<t;h++){const I=h===0?t-1:h-1,U=h===t-1?0:h+1;i[h]=e[n][I]+e[n][U]+e[a][h]+e[c][h]-4*e[n][h]}s[n]=i}return s}function Ba(e,t){const s=e.length,n=Vt(e),i=Vt(t),a=new Array(s),c=new Array(s);for(let h=0;h<s;h++){a[h]=new Array(s),c[h]=new Array(s);for(let I=0;I<s;I++){const U=e[h][I],j=t[h][I],q=U*j*j;a[h][I]=U+(fe.du*n[h][I]-q+fe.f*(1-U))*fe.dt,c[h][I]=j+(fe.dv*i[h][I]+q-(fe.f+fe.k)*j)*fe.dt}}return[a,c]}function gr(e,t,s){for(let n=0;n<s;n++)[e,t]=Ba(e,t);return[e,t]}function Gt(e,t){const s=Ge(e,1),n=Ge(e,0),i=e-t>>1;for(let a=i;a<i+t;a++)for(let c=i;c<i+t;c++)s[a][c]=.5,n[a][c]=.25;return{u:s,v:n}}function uu(e,t,s,n){const i=Ge(e,0);return i[t][s]=n,i}function yr(e){const t=e.length,s=i=>Math.max(0,Math.min(t-1,i)),n=new Array(t);for(let i=0;i<t;i++){const a=new Array(t);for(let c=0;c<t;c++)a[c]=e[i][s(c-1)]+e[i][s(c+1)]+e[s(i-1)][c]+e[s(i+1)][c]-4*e[i][c];n[i]=a}return n}function Re(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function xr(e,t,s,n,i,a,c){return Number.isFinite(e)?Re(e,s[a][c],i,[[s[a][c]+4*t[a][c],"the −4·center term is missing — the Laplacian is left + right + up + down − 4·center"],[-s[a][c],"the sign is flipped — it is the four neighbors minus 4·center, not the other way round"],[n[a][c],"that edge clamped instead of wrapping — column 0's left neighbor is the last column, not itself"]]):"that cell read outside the grid — wrap the index instead: below 0 becomes size − 1, past size − 1 becomes 0"}function br(e,t,s){const n=e*t*t;return[[e+(fe.du*s+n+fe.f*(1-e))*fe.dt,"the reaction term is added to U — V eats U, so u·v·v is subtracted here and added in stepV"],[e+(fe.du*s-n)*fe.dt,"the feed term is missing — U is replenished everywhere by f · (1 − u)"],[e+(fe.du*s+fe.f*(1-e))*fe.dt,"the reaction term u·v·v never got subtracted"],[e,"U came back unchanged — none of the three terms reached the return value"]]}function vr(e,t,s){const n=e*t*t;return[[t+(fe.dv*s-n-(fe.f+fe.k)*t)*fe.dt,"the reaction term is subtracted from V — V is what grows on it, so u·v·v is added here"],[t+(fe.dv*s+n)*fe.dt,"the kill term is missing — V is removed at (f + k) · v"],[t+(fe.dv*s-(fe.f+fe.k)*t)*fe.dt,"the reaction term u·v·v never got added"],[t,"V came back unchanged — none of the three terms reached the return value"]]}function xt(e,t){const s=Math.min(1,e*2.5),n=Math.min(1,e);return[[[n,n*n,.25+.75*n],"the brightness scale is missing — t = Math.min(1, value * 2.5)"],[[s,s,.25+.75*s],"green is t · t, not t — the square is what holds the mid-tones back"],[[.25+.75*s,s*s,s],"the 0.25 floor belongs on blue — the order is this.color(t, t * t, 0.25 + 0.75 * t, 1)"],[[s,s*s,s],"blue is missing its 0.25 floor — still water should be deep blue, not black"]].map(i=>[i[0][t]*255,i[1]])}var cu={id:"3-4",track:3,title:"Reaction–Diffusion",blurb:"Two chemicals, two equations, and suddenly: coral, fingerprints, leopard spots.",tasks:[{slug:"laplacian",title:"The Laplacian: Ask Your Neighbors",intro:`<p>Diffusion is gossip: every cell drifts toward the average of its neighbors.
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
`,inputs:e=>({field:Sa(e,32,3401)}),publicTests:[{name:"a uniform field has zero Laplacian everywhere",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=Ge(32,.7),s=e.kernel(t);e.assert(s&&s.length===32&&s[0].length===32,"expected a 32×32 result");const n=Vt(t),i=yr(t);for(let a=0;a<32;a++)for(let c=0;c<32;c++){const h=xr(s[a][c],t,n,i,1e-4,a,c);e.assertClose(s[a][c],0,1e-4,h||`cell [${a}][${c}] of a flat field`)}}},{name:"a single spike: <code>−4·s</code> at the peak, <code>+s</code> on each neighbor — even across the wrap",run:async e=>{const t=uu(32,0,0,2),s=e.kernel(t),n=Vt(t),i=yr(t),a=(c,h)=>xr(s[c][h],t,n,i,1e-4,c,h);e.assertClose(s[0][0],-8,1e-4,a(0,0)||"the peak itself (−4 × 2)"),e.assertClose(s[0][1],2,1e-4,a(0,1)||"right neighbor"),e.assertClose(s[1][0],2,1e-4,a(1,0)||"neighbor above"),e.assertClose(s[0][31],2,1e-4,a(0,31)||"LEFT neighbor — wraps to column 31"),e.assertClose(s[31][0],2,1e-4,a(31,0)||"neighbor below — wraps to row 31"),e.assertClose(s[5][5],0,1e-4,a(5,5)||"a far-away cell")}}],privateTests:[{name:"private test #1",run:async e=>{const t=Sa(e.utils,32,909),s=e.kernel(t),n=Vt(t),i=yr(t);let a=0;for(let c=0;c<32;c++)for(let h=0;h<32;h++){const I=xr(s[c][h],t,n,i,.001,c,h);e.assertClose(s[c][h],n[c][h],.001,I||`cell [${c}][${h}]`),a+=s[c][h]}e.assertClose(a,0,.01,"on a torus, gains and losses cancel exactly")}}]},{slug:"gray-scott-step",title:"One Step of Gray–Scott",intro:`<p>Now the chemistry. Gray–Scott tracks two chemicals on the same grid:
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
`,inputs:()=>{const e=Gt(32,6);return{u0:e.u,v0:e.v}},publicTests:[{name:"the calm ocean is a fixed point: U=1, V=0 stays exactly put",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels (stepU then stepV), found ${e.kernels.length}`);const t=Ge(32,1),s=Ge(32,0),n=e.kernels[0](t,s),i=e.kernels[1](t,s);for(let a=0;a<32;a+=5)for(let c=0;c<32;c+=5)e.assertClose(n[a][c],1,1e-5,`U at [${a}][${c}] — nothing to react, nothing to feed`),e.assertClose(i[a][c],0,1e-5,`V at [${a}][${c}] — no V, no reaction`)}},{name:"a well-mixed beaker (u=0.6, v=0.3) follows the equations exactly",run:async e=>{const t=Ge(32,.6),s=Ge(32,.3),n=.6*.3*.3,i=.6+(-n+fe.f*(1-.6))*fe.dt,a=.3+(n-(fe.f+fe.k)*.3)*fe.dt,c=e.kernels[0](t,s),h=e.kernels[1](t,s),I=br(.6,.3,0),U=vr(.6,.3,0);for(let j=0;j<32;j+=7)for(let q=0;q<32;q+=7)e.assertClose(c[j][q],i,1e-4,Re(c[j][q],i,1e-4,I)||`U at [${j}][${q}]`),e.assertClose(h[j][q],a,1e-4,Re(h[j][q],a,1e-4,U)||`V at [${j}][${q}]`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=Ge(32,.8),s=Ge(32,.1),n=.8*.1*.1,i=.8+(-n+fe.f*(1-.8))*fe.dt,a=.1+(n-(fe.f+fe.k)*.1)*fe.dt,c=e.kernels[0](t,s),h=e.kernels[1](t,s),I=br(.8,.1,0),U=vr(.8,.1,0);for(let j=0;j<32;j+=3)for(let q=0;q<32;q+=3)e.assertClose(c[j][q],i,1e-4,Re(c[j][q],i,1e-4,I)||`U at [${j}][${q}]`),e.assertClose(h[j][q],a,1e-4,Re(h[j][q],a,1e-4,U)||`V at [${j}][${q}]`)}},{name:"private test #2",run:async e=>{const t=Gt(32,10),[s,n]=Ba(t.u,t.v),i=e.kernels[0](t.u,t.v),a=e.kernels[1](t.u,t.v),c=Vt(t.u),h=Vt(t.v);for(let I=0;I<32;I++)for(let U=0;U<32;U++){const j=t.u[I][U],q=t.v[I][U];e.assertClose(i[I][U],s[I][U],1e-4,Re(i[I][U],s[I][U],1e-4,br(j,q,c[I][U]))||`U at [${I}][${U}]`),e.assertClose(a[I][U],n[I][U],1e-4,Re(a[I][U],n[I][U],1e-4,vr(j,q,h[I][U]))||`V at [${I}][${U}]`)}}}]},{slug:"feed-it-back",title:"Feed It Back: 100 Steps",intro:`<p>One step is chemistry; a hundred steps is <em>morphogenesis</em>. The kernels
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
`,inputs:()=>{const e=Gt(48,8);return{seedU:e.u,seedV:e.v}},publicTests:[{name:"after 100 steps the kernels are seeing step 99, not the seed",run:async e=>{e.assert(e.kernels.length>=2,`expected 2 kernels (stepU then stepV), found ${e.kernels.length}`);const t=Gt(48,8),[s,n]=gr(t.u,t.v,99),i=e.kernels[0].lastArgs;e.assert(i&&i.length>=2,"stepU should have been called with (u, v)");const[a,c]=i;e.assert(Math.abs(a[24][24]-t.u[24][24])>.05,"the last stepU call still saw the seed — did the loop actually feed results back?");const h=[[24,24],[24,20],[20,28],[24,12],[4,4]];for(const[I,U]of h)e.assertClose(a[I][U],s[I][U],.002,`U at [${I}][${U}] after 99 steps`),e.assertClose(c[I][U],n[I][U],.002,`V at [${I}][${U}] after 99 steps`)}},{name:"stepU and stepV read the same snapshot — no future food",run:async e=>{const t=e.kernels[0].lastArgs[0],s=e.kernels[1].lastArgs[0];e.assert(t&&s,"both kernels should have been called with (u, v)");const n=[[24,24],[24,21],[27,24],[21,27],[24,16]];for(const[i,a]of n)e.assertClose(s[i][a],t[i][a],1e-4,`u[${i}][${a}] differs between the stepU and stepV calls — swap only after BOTH kernels ran`)}},{name:"V has escaped the seed square, and both fields stay in [0, 1]",run:async e=>{const[t,s]=e.kernels[0].lastArgs;e.assert(s[24][12]>1e-4,"V should have diffused well outside the 8×8 seed by step 99");for(let n=0;n<48;n+=3)for(let i=0;i<48;i+=3)e.assert(t[n][i]>=-1e-6&&t[n][i]<=1+1e-6,`U at [${n}][${i}] left [0, 1] — unstable loop?`),e.assert(s[n][i]>=-1e-6&&s[n][i]<=1+1e-6,`V at [${n}][${i}] left [0, 1] — unstable loop?`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=Gt(48,12);let s=t.u,n=t.v;for(let c=0;c<40;c++){const h=e.kernels[0](s,n),I=e.kernels[1](s,n);s=h,n=I}const[i,a]=gr(t.u,t.v,40);for(let c=0;c<48;c++)for(let h=0;h<48;h++)e.assertClose(s[c][h],i[c][h],.001,`U at [${c}][${h}] after 40 steps`),e.assertClose(n[c][h],a[c][h],.001,`V at [${c}][${h}] after 40 steps`)}}]},{slug:"paint-the-pattern",title:"Paint the Pattern",intro:`<p>Payoff time. The whole simulation is wired below — 64×64 grid, 200 steps —
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
`,inputs:()=>{const e=Gt(64,8);return{seedU:e.u,seedV:e.v}},publicTests:[{name:"a 64×64 canvas is rendered",run:async e=>{e.assert(e.canvas,"no canvas — is paint graphical: true, and did you call render()?"),e.assert(e.canvas.width===64&&e.canvas.height===64,`expected a 64×64 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();e.assert(t.length===4096*4,"pixel buffer should hold 64×64 RGBA values")}},{name:"the palette is exact: still water is deep blue, <code>v = 0.2</code> is half-lit violet",run:async e=>{const t=e.kernels.find(n=>n.kernel&&n.kernel.graphical);e.assert(t,"no graphical kernel found"),t(Ge(64,0));let s=t.getPixels();for(let n=0;n<s.length;n+=331*4)e.assertClose(s[n],0,3,Re(s[n],0,3,xt(0,0))||`red at byte ${n} for v = 0`),e.assertClose(s[n+1],0,3,Re(s[n+1],0,3,xt(0,1))||`green at byte ${n} for v = 0`),e.assertClose(s[n+2],.25*255,3,Re(s[n+2],.25*255,3,xt(0,2))||`blue at byte ${n} for v = 0`);t(Ge(64,.2)),s=t.getPixels();for(let n=0;n<s.length;n+=331*4)e.assertClose(s[n],.5*255,3,Re(s[n],.5*255,3,xt(.2,0))||`red at byte ${n} for v = 0.2`),e.assertClose(s[n+1],.25*255,3,Re(s[n+1],.25*255,3,xt(.2,1))||`green at byte ${n} for v = 0.2`),e.assertClose(s[n+2],.625*255,3,Re(s[n+2],.625*255,3,xt(.2,2))||`blue at byte ${n} for v = 0.2`)}},{name:"the picture is alive — bright coral on a dark ocean",run:async e=>{const t=e.kernels.find(h=>h.kernel&&h.kernel.graphical);e.assert(t,"no graphical kernel found");const s=Gt(64,8),[,n]=gr(s.u,s.v,200);t(n);const i=t.getPixels();let a=0,c=0;for(let h=0;h<i.length;h+=4)i[h]>150&&a++,i[h]<20&&c++;e.assert(a>=50,`expected at least 50 bright pattern pixels after 200 steps, found ${a}`),e.assert(c>=1e3,`expected a mostly-dark ocean around the pattern, found only ${c} dark pixels`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernels.find(n=>n.kernel&&n.kernel.graphical);e.assert(t,"no graphical kernel found"),t(Ge(64,.6));let s=t.getPixels();for(let n=0;n<s.length;n+=449*4)e.assertClose(s[n],255,3,`red at byte ${n} for v = 0.6`),e.assertClose(s[n+1],255,3,`green at byte ${n} for v = 0.6`),e.assertClose(s[n+2],255,3,`blue at byte ${n} for v = 0.6`);t(Ge(64,.1)),s=t.getPixels();for(let n=0;n<s.length;n+=449*4)e.assertClose(s[n],.25*255,3,Re(s[n],.25*255,3,xt(.1,0))||`red at byte ${n} for v = 0.1`),e.assertClose(s[n+1],.0625*255,3,Re(s[n+1],.0625*255,3,xt(.1,1))||`green at byte ${n} for v = 0.1`),e.assertClose(s[n+2],.4375*255,3,Re(s[n+2],.4375*255,3,xt(.1,2))||`blue at byte ${n} for v = 0.1`)}}]}]},hu=Object.freeze({__proto__:null,default:cu});function dt(e){return(e-32)/16}function du(e,t,s){const n=Math.max(s-Math.abs(e-t),0)/s;return Math.min(e,t)-n*n*s*.25}function Fe(e,t,s){const n=(s*64+t)*4;return[e[n],e[n+1],e[n+2],e[n+3]]}function Ut(e,t,s,n,i,a,c){const h=Fe(t,s,32)[n],I=Fe(t,s,31)[n];e.assert(Math.abs(h-i)<=a||Math.abs(I-i)<=a,`${c} — expected ≈${i} ±${a}, got ${h} (row 32) / ${I} (row 31)`)}function we(e,t){return Fe(e,t,32)[0]}function cn(e,t,s,n){const i=n.filter(a=>Math.abs(e-a[0])<=s&&Math.abs(t-a[0])>s).map(a=>a[1]);return i.length&&i.every(a=>a===i[0])?i[0]:null}function _a(e,t,s,n,i){const a=e-s,c=t-n,h=a*a+c*c;return[[Math.sqrt(h),"the radius never got subtracted — the field is length(p − center) − r"],[Math.sqrt(h)+i,"the radius is added where the field subtracts it — inside the sphere the distance is NEGATIVE"],[h-i,"that is the squared distance — take Math.sqrt of it before subtracting r"]]}function Ca(e,t,s){const n=Math.max(s-Math.abs(e-t),0)/s,i=Math.min(e,t);return[[i,"that is the hard minimum — smin has to dip below both fields where they are within k of each other"],[i-n*n*s,"the dip is h * h * k * 0.25 — the 0.25 is missing"],[i-n*s*.25,"h is squared in the dip: h * h * k * 0.25"]]}function pu(e){for(let t=0;t<e.length;t+=4)if(e[t]>180)return null;return"not one pixel hit the surface — the ray never leaves its starting plane; each of the 48 passes has to step forward by the distance itself, t += d"}function fu(e,t){return Math.abs(e-128)<=6&&Math.abs(t-128)<=6?"blue came back mid-grey — the raw differences are tiny, so this normal was never normalized; divide nx, ny and nz by len":null}function mu(e){return e>=.15*255-4&&e<=.2*255?"this pixel is sitting on the 0.15 ambient floor — the diffuse term added nothing; the dot product needs the UNIT normal, so divide nx, ny, nz by len first":null}var gu={id:"3-5",track:3,title:"Ray-Marched Metaballs",blurb:"Signed distance fields and soft shadows — a real-time 3D scene with no triangles at all.",tasks:[{slug:"sphere-sdf",title:"The Sphere as a Number",intro:`<p>A <strong>signed distance field</strong> describes a shape with one function:
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
`,publicTests:[{name:"unit sphere: center reads −1, surface reads 0, corner reads ≈1.83",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(0,0,1);e.assert(t&&t.length===64&&t[0].length===64,"expected a 64×64 field");const s=(n,i,a)=>cn(t[n][i],a,.002,_a(dt(i),dt(n),0,0,1));e.assertClose(t[32][32],-1,.002,s(32,32,-1)||"center of the sphere (inside)"),e.assertClose(t[32][48],0,.002,s(32,48,0)||"one unit right of center (on the surface)"),e.assertClose(t[0][0],Math.sqrt(8)-1,.002,s(0,0,Math.sqrt(8)-1)||"far corner (outside)")}},{name:"the field is radially symmetric around the sphere center",run:async e=>{const t=e.kernel(0,0,1);for(const s of[5,10,15])e.assertClose(t[32][32+s],t[32][32-s],.002,`left/right at offset ${s}`),e.assertClose(t[32][32+s],t[32+s][32],.002,`x/y at offset ${s}`)}}],privateTests:[{name:"private test #1",run:async e=>{const t=e.kernel(.5,-.25,.75),s=[[3,7],[20,44],[32,32],[50,12],[10,58],[63,63]];for(const[n,i]of s){const a=dt(i)-.5,c=dt(n)+.25,h=Math.sqrt(a*a+c*c)-.75,I=cn(t[n][i],h,.003,_a(dt(i),dt(n),.5,-.25,.75));e.assertClose(t[n][i],h,.003,I||`cell [${n}][${i}]`)}}}]},{slug:"smooth-min",title:"Two Spheres Melt Into One",intro:`<p>Combining two SDFs is just <code>Math.min</code> — the nearest surface wins.
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
`,publicTests:[{name:"midpoint dips below the plain minimum by <code>k / 4</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()");const t=e.kernel(.7,.5,.4),s=cn(t[32][32],.1,.003,Ca(.2,.2,.4));e.assertClose(t[32][32],.1,.003,s||"field at the midpoint"),e.assert(t[32][32]<.2-.05,"the blend should dip clearly below plain min (0.2)")}},{name:"field stays symmetric and returns to plain min far from the seam",run:async e=>{const t=e.kernel(.7,.5,.4);for(const c of[6,12,20])e.assertClose(t[32][32+c],t[32][32-c],.003,`mirror pair at offset ${c}`);const s=dt(0),n=dt(0),i=Math.sqrt((s+.7)*(s+.7)+n*n)-.5,a=Math.sqrt((s-.7)*(s-.7)+n*n)-.5;e.assertClose(t[0][0],Math.min(i,a),.003,"far corner (outside the blend zone)")}}],privateTests:[{name:"private test #1",run:async e=>{const[t,s,n]=[.9,.45,.3],i=e.kernel(t,s,n);for(let a=1;a<64;a+=7)for(let c=1;c<64;c+=7){const h=dt(c),I=dt(a),U=Math.sqrt((h+t)*(h+t)+I*I)-s,j=Math.sqrt((h-t)*(h-t)+I*I)-s,q=du(U,j,n),$e=cn(i[a][c],q,.003,Ca(U,j,n));e.assertClose(i[a][c],q,.003,$e||`cell [${a}][${c}]`)}}}]},{slug:"ray-march",title:"March Until You Hit Something",intro:`<p>Now the third dimension. Every pixel fires a ray straight into the screen
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
`,publicTests:[{name:"a 64×64 canvas whose corners are background",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===64&&e.canvas.height===64,`expected a 64×64 canvas, got ${e.canvas.width}×${e.canvas.height}`);const t=e.getPixels();for(const[s,n]of[[0,0],[63,0],[0,63],[63,63]]){const[i]=Fe(t,s,n);e.assert(i<40,`corner (${s}, ${n}) should be background, got red ${i}`)}}},{name:"rays through the blob hit: center and both lobes come back pink",run:async e=>{e.kernel(.55,.5,.3);const t=e.getPixels(),s=pu(t);for(const n of[22,32,42]){const i=we(t,n);e.assert(i>180,s||`pixel (${n}, 32) should be a hit (red > 180), got ${i}`)}}},{name:"the silhouette is left-right symmetric",run:async e=>{e.kernel(.55,.5,.3);const t=e.getPixels();for(const s of[6,10,14,18]){const n=we(t,32-s)>128,i=we(t,32+s)>128;e.assert(n===i,`pixels (${32-s}, 32) and (${32+s}, 32) should both hit or both miss`)}}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(1.2,.45,.15);const t=e.getPixels();e.assert(we(t,32)<40,`separated spheres: the center ray should miss, got red ${we(t,32)}`);for(const s of[13,51]){const n=we(t,s);e.assert(n>180,`pixel (${s}, 32) is inside a lobe and should hit, got red ${n}`)}}}]},{slug:"normals",title:"Normals Without Geometry",intro:`<p>Lighting needs surface normals, and a mesh would hand them to you per-vertex. We
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
`,publicTests:[{name:"the head-on center pixel paints <code>rgb(0.5, 0.5, 0)</code> — normal (0, 0, −1)",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.kernel(.55,.5,.3);const t=e.getPixels();Ut(e,t,32,0,128,14,"center red (nx = 0)"),Ut(e,t,32,1,128,14,"center green (ny = 0)");const s=Fe(t,32,32)[2],n=Fe(t,32,31)[2];e.assert(s<40&&n<40,fu(s,n)||`center blue should be near 0 (nz = -1, facing the camera), got ${s}/${n}`)}},{name:"mirrored hit pixels have mirrored normals: red channels sum to ≈255",run:async e=>{e.kernel(.55,.5,.3);const t=e.getPixels();for(const s of[8,10]){const n=Fe(t,32-s,32),i=Fe(t,32+s,32);e.assert(n[1]>60&&i[1]>60,`both probes at ±${s} should be hits`),e.assert(Math.abs(n[0]+i[0]-255)<=24,`red(${32-s}) + red(${32+s}) should be ≈255 (nx antisymmetric), got ${n[0]+i[0]}`)}}},{name:"misses keep the background color",run:async e=>{e.kernel(.55,.5,.3);const t=e.getPixels();for(const[s,n]of[[0,0],[63,0],[0,63],[63,63]]){const[,i]=Fe(t,s,n);e.assert(i<40,`corner (${s}, ${n}) should be background, got green ${i}`)}}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(1.2,.45,.15);const t=e.getPixels();e.assert(Fe(t,32,32)[1]<40,"center should be background now"),Ut(e,t,51,0,124,16,"right lobe red (nx ≈ 0)"),Ut(e,t,51,1,128,16,"right lobe green (ny = 0)");const s=Fe(t,13,32),n=Fe(t,51,32);e.assert(Math.abs(s[0]+n[0]-255)<=26,`mirrored lobe pixels should have mirrored nx, got ${s[0]+n[0]}`)}}]},{slug:"diffuse-lighting",title:"Turn On the Light",intro:`<p>With normals in hand, lighting is one dot product. <strong>Lambert's law</strong>:
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
`,publicTests:[{name:"center pixel brightness matches Lambert: <code>0.15 + 0.85 × 0.8</code>",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.kernel(.55,.5,.3,-.6,0,-.8);const t=e.getPixels(),s=mu(Math.max(we(t,32),Fe(t,32,31)[0]));Ut(e,t,32,0,212,14,s||"center red"),Ut(e,t,32,1,131,14,s||"center green (red × 0.62)")}},{name:"the side facing the light is brighter than the side facing away",run:async e=>{e.kernel(.55,.5,.3,-.6,0,-.8);const t=e.getPixels(),s=we(t,22),n=we(t,42);e.assert(s>n+20,`light comes from the left: red(22) = ${s} should exceed red(42) = ${n} by > 20`)}},{name:"misses keep the background color",run:async e=>{e.kernel(.55,.5,.3,-.6,0,-.8);const t=e.getPixels();for(const[s,n]of[[0,0],[63,63]]){const[i]=Fe(t,s,n);e.assert(i<40,`corner (${s}, ${n}) should be background, got red ${i}`)}}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(.55,.5,.3,.6,0,-.8);const t=e.getPixels();Ut(e,t,32,0,212,14,"center red (same head-on dot product)");const s=we(t,22),n=we(t,42);e.assert(n>s+20,`light from the right: red(42) = ${n} should exceed red(22) = ${s} by > 20`)}}]},{slug:"soft-shadows",title:"Soft Shadows, Full Scene",intro:`<p>The payoff. A point is in shadow when something sits between it and the light —
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
`,publicTests:[{name:"a 64×64 canvas: hits keep an ambient floor, corners stay background",run:async e=>{e.assert(e.kernels.length>=1,"no kernel was created — call gpu.createKernel()"),e.assert(e.canvas,"no canvas — is the kernel graphical: true, and did you call render()?"),e.assert(e.canvas.width===64&&e.canvas.height===64,`expected a 64×64 canvas, got ${e.canvas.width}×${e.canvas.height}`),e.kernel(.55,.5,.3,-.86,0,-.51,1);const t=e.getPixels();e.assert(we(t,32)>=28,`center should be a hit with at least the ambient floor, got red ${we(t,32)}`);for(const[s,n]of[[0,0],[63,63]]){const[i]=Fe(t,s,n);e.assert(i<=18,`corner (${s}, ${n}) should be pure background, got red ${i}`)}}},{name:"toggling <code>shadowOn</code> darkens the neck of the blob by > 60",run:async e=>{e.kernel(.55,.5,.3,-.86,0,-.51,1);const t=Array.from(e.getPixels());e.kernel(.55,.5,.3,-.86,0,-.51,0);const s=Array.from(e.getPixels());let n=0;for(let i=28;i<=40;i++)n=Math.max(n,we(s,i)-we(t,i));e.assert(n>60,`expected the shadow to darken some neck pixel by > 60, biggest drop was ${n}`)}},{name:"shadows only ever darken — and the lit flank is untouched",run:async e=>{e.kernel(.55,.5,.3,-.86,0,-.51,1);const t=Array.from(e.getPixels());e.kernel(.55,.5,.3,-.86,0,-.51,0);const s=Array.from(e.getPixels());for(let a=0;a<64;a++)e.assert(we(t,a)<=we(s,a)+10,`pixel (${a}, 32) got BRIGHTER with shadows on — sh must stay ≤ 1`);const n=we(t,18),i=we(s,18);e.assert(Math.abs(n-i)<=10,`pixel (18, 32) faces the light with a clear path — it should not change (${i} → ${n})`)}}],privateTests:[{name:"private test #1",run:async e=>{e.kernel(.6,.48,.25,.86,0,-.51,1);const t=Array.from(e.getPixels());e.kernel(.6,.48,.25,.86,0,-.51,0);const s=Array.from(e.getPixels());e.assert(we(t,32)<20,"center ray should miss the separated blobs");let n=0;for(let c=24;c<=36;c++)n=Math.max(n,we(s,c)-we(t,c));e.assert(n>60,`mirrored light: expected a shadow drop > 60 on the left flank, got ${n}`);const i=we(t,46),a=we(s,46);e.assert(Math.abs(i-a)<=10,`pixel (46, 32) faces the mirrored light — it should not change (${a} → ${i})`)}}]}]},yu=Object.freeze({__proto__:null,default:gu});const xu=[{number:1,title:"GPGPU 101",tagline:"From zero to your first thousand threads"},{number:2,title:"Advanced Math",tagline:"Heavy math, thousands of threads at once"},{number:3,title:"Computational Graphics",tagline:"Pictures computed, not drawn"}],bu=Object.assign({"./track1/module-1-1.js":fl,"./track1/module-1-2.js":_l,"./track1/module-1-3.js":Al,"./track1/module-1-4.js":Il,"./track1/module-1-5.js":Dl,"./track2/module-2-1.js":Rl,"./track2/module-2-2.js":Fl,"./track2/module-2-3.js":Bl,"./track2/module-2-4.js":Xl,"./track2/module-2-5.js":Zl,"./track3/module-3-1.js":tu,"./track3/module-3-2.js":nu,"./track3/module-3-3.js":lu,"./track3/module-3-4.js":hu,"./track3/module-3-5.js":yu});function Aa(e){return String(e).split("-").map(Number)}const ja=Object.values(bu).map(e=>e.default).filter(Boolean).sort((e,t)=>{const[s,n]=Aa(e.id),[i,a]=Aa(t.id);return s-i||n-a});xu.map(e=>({...e,modules:ja.filter(t=>t.track===e.number)}));function vu(e){return ja.find(t=>t.id===e)||null}function wu(e,t){return`${e}-${t}`}function ku(e,t){const s=vu(e);if(!s)return null;const n=Number(t);return!Number.isInteger(n)||n<1||n>s.tasks.length?null:{module:s,task:s.tasks[n-1],taskNum:n,taskIndex:n-1,taskId:wu(e,n),total:s.tasks.length}}var dn={exports:{}};/**
 * gpu.js
 * http://gpu.rocks/
 *
 * GPU Accelerated JavaScript
 *
 * @version 2.19.9
 * @date Tue Jul 28 2026 17:41:46 GMT+0800 (Singapore Standard Time)
 *
 * @license MIT
 * The MIT License
 *
 * Copyright (c) 2026 gpu.js Team
 */var Tu=dn.exports,Ea;function Su(){return Ea||(Ea=1,(function(e,t){(function(s,n){e.exports=n()})(Tu,function(){var s=(B,z)=>()=>(z||(B((z={exports:{}}).exports,z),B=null),z.exports),n=s((B,z)=>{function M(E){const w=new Array(E.length);for(let y=0;y<E.length;y++){const k=E[y];k.toArray?w[y]=k.toArray():w[y]=k}return w}function _(){const E=M(arguments);let w=null;for(let y=0;y<this.output.x;y++){this.thread.x=y,this.thread.y=0,this.thread.z=0;const k=this._fn.apply(this,E);w===null&&(w=typeof k=="number"||typeof k=="boolean"?new Float32Array(this.output.x):new Array(this.output.x)),w[y]=typeof k=="object"?new Float32Array(k):k}return w}function b(){const E=M(arguments),w=new Array(this.output.y);for(let y=0;y<this.output.y;y++){let k=null;for(let v=0;v<this.output.x;v++){this.thread.x=v,this.thread.y=y,this.thread.z=0;const $=this._fn.apply(this,E);k===null&&(k=typeof $=="number"||typeof $=="boolean"?new Float32Array(this.output.x):new Array(this.output.x)),k[v]=typeof $=="object"?new Float32Array($):$}w[y]=k}return w}function l(){const E=M(arguments);for(let w=0;w<this.output.y;w++)for(let y=0;y<this.output.x;y++)this.thread.x=y,this.thread.y=w,this.thread.z=0,this._fn.apply(this,E)}function d(){const E=M(arguments),w=new Array(this.output.z);for(let y=0;y<this.output.z;y++){const k=new Array(this.output.y);for(let v=0;v<this.output.y;v++){let $=null;for(let P=0;P<this.output.x;P++){this.thread.x=P,this.thread.y=v,this.thread.z=y;const O=this._fn.apply(this,E);$===null&&($=typeof O=="number"||typeof O=="boolean"?new Float32Array(this.output.x):new Array(this.output.x)),$[P]=typeof O=="object"?new Float32Array(O):O}k[v]=$}w[y]=k}return w}function C(E){E.setOutput=k=>{E.output=p(k),E.graphical&&m(E)},E.toJSON=()=>{throw new Error("Not usable with gpuMock")},E.setConstants=k=>(E.constants=k,E),E.setGraphical=k=>(E.graphical=k,E),E.setCanvas=k=>(E.canvas=k,E),E.setContext=k=>(E.context=k,E),E.destroy=()=>{},E.validateSettings=()=>{},E.graphical&&E.output&&m(E),E.exec=function(){return new Promise((k,v)=>{try{k(E.apply(E,arguments))}catch($){v($)}})},E.getPixels=k=>{const{x:v,y:$}=E.output;return k?g(E._imageData.data,v,$):E._imageData.data.slice(0)},E.color=function(k,v,$,P){typeof P>"u"&&(P=1),k=Math.floor(k*255),v=Math.floor(v*255),$=Math.floor($*255),P=Math.floor(P*255);const O=E.output.x,T=E.output.y,A=E.thread.x+(T-E.thread.y-1)*O;E._colorData[A*4+0]=k,E._colorData[A*4+1]=v,E._colorData[A*4+2]=$,E._colorData[A*4+3]=P};const w=()=>E,y=["setWarnVarUsage","setArgumentTypes","setTactic","setOptimizeFloatMemory","setDebug","setLoopMaxIterations","setConstantTypes","setFunctions","setNativeFunctions","setInjectedNative","setPipeline","setPrecision","setOutputToTexture","setImmutable","setStrictIntegers","setDynamicOutput","setHardcodeConstants","setDynamicArguments","setUseLegacyEncoder","setWarnVarUsage","addSubKernel"];for(let k=0;k<y.length;k++)E[y[k]]=w;return E}function m(E){const{x:w,y}=E.output;if(E.context&&E.context.createImageData){const k=new Uint8ClampedArray(w*y*4);E._imageData=E.context.createImageData(w,y),E._colorData=k}else{const k=new Uint8ClampedArray(w*y*4);E._imageData={data:k},E._colorData=k}}function p(E){let w=null;if(E.length)if(E.length===3){const[y,k,v]=E;w={x:y,y:k,z:v}}else if(E.length===2){const[y,k]=E;w={x:y,y:k}}else{const[y]=E;w={x:y}}else w=E;return w}function u(E,w={}){const y=w.output?p(w.output):null;function k(){return k.output.z?d.apply(k,arguments):k.output.y?k.graphical?l.apply(k,arguments):b.apply(k,arguments):_.apply(k,arguments)}return k._fn=E,k.constants=w.constants||null,k.context=w.context||null,k.canvas=w.canvas||null,k.graphical=w.graphical||!1,k._imageData=null,k._colorData=null,k.output=y,k.thread={x:0,y:0,z:0},C(k)}function g(E,w,y){const k=y/2|0,v=w*4,$=new Uint8ClampedArray(w*4),P=E.slice(0);for(let O=0;O<k;++O){const T=O*v,A=(y-O-1)*v;$.set(P.subarray(T,T+v)),P.copyWithin(T,A,A+v),P.set($,A)}return P}z.exports={gpuMock:u}}),i=s((B,z)=>{(function(M,_){typeof B=="object"&&typeof z<"u"?_(B):(M=typeof globalThis<"u"?globalThis:M||self,_(M.acorn={}))})(B,function(M){var _=[509,0,227,0,150,4,294,9,1368,2,2,1,6,3,41,2,5,0,166,1,574,3,9,9,7,9,32,4,318,1,80,3,71,10,50,3,123,2,54,14,32,10,3,1,11,3,46,10,8,0,46,9,7,2,37,13,2,9,6,1,45,0,13,2,49,13,9,3,2,11,83,11,7,0,3,0,158,11,6,9,7,3,56,1,2,6,3,1,3,2,10,0,11,1,3,6,4,4,68,8,2,0,3,0,2,3,2,4,2,0,15,1,83,17,10,9,5,0,82,19,13,9,214,6,3,8,28,1,83,16,16,9,82,12,9,9,7,19,58,14,5,9,243,14,166,9,71,5,2,1,3,3,2,0,2,1,13,9,120,6,3,6,4,0,29,9,41,6,2,3,9,0,10,10,47,15,343,9,54,7,2,7,17,9,57,21,2,13,123,5,4,0,2,1,2,6,2,0,9,9,49,4,2,1,2,4,9,9,330,3,10,1,2,0,49,6,4,4,14,10,5350,0,7,14,11465,27,2343,9,87,9,39,4,60,6,26,9,535,9,470,0,2,54,8,3,82,0,12,1,19628,1,4178,9,519,45,3,22,543,4,4,5,9,7,3,6,31,3,149,2,1418,49,513,54,5,49,9,0,15,0,23,4,2,14,1361,6,2,16,3,6,2,1,2,4,101,0,161,6,10,9,357,0,62,13,499,13,245,1,2,9,726,6,110,6,6,9,4759,9,787719,239],b=[0,11,2,25,2,18,2,1,2,14,3,13,35,122,70,52,268,28,4,48,48,31,14,29,6,37,11,29,3,35,5,7,2,4,43,157,19,35,5,35,5,39,9,51,13,10,2,14,2,6,2,1,2,10,2,14,2,6,2,1,4,51,13,310,10,21,11,7,25,5,2,41,2,8,70,5,3,0,2,43,2,1,4,0,3,22,11,22,10,30,66,18,2,1,11,21,11,25,71,55,7,1,65,0,16,3,2,2,2,28,43,28,4,28,36,7,2,27,28,53,11,21,11,18,14,17,111,72,56,50,14,50,14,35,39,27,10,22,251,41,7,1,17,2,60,28,11,0,9,21,43,17,47,20,28,22,13,52,58,1,3,0,14,44,33,24,27,35,30,0,3,0,9,34,4,0,13,47,15,3,22,0,2,0,36,17,2,24,20,1,64,6,2,0,2,3,2,14,2,9,8,46,39,7,3,1,3,21,2,6,2,1,2,4,4,0,19,0,13,4,31,9,2,0,3,0,2,37,2,0,26,0,2,0,45,52,19,3,21,2,31,47,21,1,2,0,185,46,42,3,37,47,21,0,60,42,14,0,72,26,38,6,186,43,117,63,32,7,3,0,3,7,2,1,2,23,16,0,2,0,95,7,3,38,17,0,2,0,29,0,11,39,8,0,22,0,12,45,20,0,19,72,200,32,32,8,2,36,18,0,50,29,113,6,2,1,2,37,22,0,26,5,2,1,2,31,15,0,328,18,16,0,2,12,2,33,125,0,80,921,103,110,18,195,2637,96,16,1071,18,5,26,3994,6,582,6842,29,1763,568,8,30,18,78,18,29,19,47,17,3,32,20,6,18,433,44,212,63,129,74,6,0,67,12,65,1,2,0,29,6135,9,1237,42,9,8936,3,2,6,2,1,2,290,16,0,30,2,3,0,15,3,9,395,2309,106,6,12,4,8,8,9,5991,84,2,70,2,1,3,0,3,1,3,3,2,11,2,0,2,6,2,64,2,3,3,7,2,6,2,27,2,3,2,4,2,0,4,6,2,339,3,24,2,24,2,30,2,24,2,30,2,24,2,30,2,24,2,30,2,24,2,7,1845,30,7,5,262,61,147,44,11,6,17,0,322,29,19,43,485,27,229,29,3,0,496,6,2,3,2,1,2,14,2,196,60,67,8,0,1205,3,2,26,2,1,2,0,3,0,2,9,2,3,2,0,2,0,7,0,5,0,2,0,2,0,2,2,2,1,2,0,3,0,2,0,2,0,2,0,2,0,2,1,2,0,3,3,2,6,2,3,2,3,2,0,2,9,2,16,6,2,2,4,2,16,4421,42719,33,4153,7,221,3,5761,15,7472,16,621,2467,541,1507,4938,6,4191],l="‌‍·̀-ͯ·҃-֑҇-ׇֽֿׁׂׅׄؐ-ًؚ-٩ٰۖ-ۜ۟-۪ۤۧۨ-ۭ۰-۹ܑܰ-݊ަ-ް߀-߉߫-߽߳ࠖ-࠙ࠛ-ࠣࠥ-ࠧࠩ-࡙࠭-࡛ࢗ-࢟࣊-ࣣ࣡-ःऺ-़ा-ॏ॑-ॗॢॣ०-९ঁ-ঃ়া-ৄেৈো-্ৗৢৣ০-৯৾ਁ-ਃ਼ਾ-ੂੇੈੋ-੍ੑ੦-ੱੵઁ-ઃ઼ા-ૅે-ૉો-્ૢૣ૦-૯ૺ-૿ଁ-ଃ଼ା-ୄେୈୋ-୍୕-ୗୢୣ୦-୯ஂா-ூெ-ைொ-்ௗ௦-௯ఀ-ఄ఼ా-ౄె-ైొ-్ౕౖౢౣ౦-౯ಁ-ಃ಼ಾ-ೄೆ-ೈೊ-್ೕೖೢೣ೦-೯ೳഀ-ഃ഻഼ാ-ൄെ-ൈൊ-്ൗൢൣ൦-൯ඁ-ඃ්ා-ුූෘ-ෟ෦-෯ෲෳัิ-ฺ็-๎๐-๙ັິ-ຼ່-໎໐-໙༘༙༠-༩༹༵༷༾༿ཱ-྄྆྇ྍ-ྗྙ-ྼ࿆ါ-ှ၀-၉ၖ-ၙၞ-ၠၢ-ၤၧ-ၭၱ-ၴႂ-ႍႏ-ႝ፝-፟፩-፱ᜒ-᜕ᜲ-᜴ᝒᝓᝲᝳ឴-៓៝០-៩᠋-᠍᠏-᠙ᢩᤠ-ᤫᤰ-᤻᥆-᥏᧐-᧚ᨗ-ᨛᩕ-ᩞ᩠-᩿᩼-᪉᪐-᪙᪰-᪽ᪿ-ᫎᬀ-ᬄ᬴-᭄᭐-᭙᭫-᭳ᮀ-ᮂᮡ-ᮭ᮰-᮹᯦-᯳ᰤ-᰷᱀-᱉᱐-᱙᳐-᳔᳒-᳨᳭᳴᳷-᳹᷀-᷿‌‍‿⁀⁔⃐-⃥⃜⃡-⃰⳯-⵿⳱ⷠ-〪ⷿ-゙゚〯・꘠-꘩꙯ꙴ-꙽ꚞꚟ꛰꛱ꠂ꠆ꠋꠣ-ꠧ꠬ꢀꢁꢴ-ꣅ꣐-꣙꣠-꣱ꣿ-꤉ꤦ-꤭ꥇ-꥓ꦀ-ꦃ꦳-꧀꧐-꧙ꧥ꧰-꧹ꨩ-ꨶꩃꩌꩍ꩐-꩙ꩻ-ꩽꪰꪲ-ꪴꪷꪸꪾ꪿꫁ꫫ-ꫯꫵ꫶ꯣ-ꯪ꯬꯭꯰-꯹ﬞ︀-️︠-︯︳︴﹍-﹏０-９＿･",d="ªµºÀ-ÖØ-öø-ˁˆ-ˑˠ-ˤˬˮͰ-ʹͶͷͺ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-Ֆՙՠ-ֈא-תׯ-ײؠ-يٮٯٱ-ۓەۥۦۮۯۺ-ۼۿܐܒ-ܯݍ-ޥޱߊ-ߪߴߵߺࠀ-ࠕࠚࠤࠨࡀ-ࡘࡠ-ࡪࡰ-ࢇࢉ-ࢎࢠ-ࣉऄ-हऽॐक़-ॡॱ-ঀঅ-ঌএঐও-নপ-রলশ-হঽৎড়ঢ়য়-ৡৰৱৼਅ-ਊਏਐਓ-ਨਪ-ਰਲਲ਼ਵਸ਼ਸਹਖ਼-ੜਫ਼ੲ-ੴઅ-ઍએ-ઑઓ-નપ-રલળવ-હઽૐૠૡૹଅ-ଌଏଐଓ-ନପ-ରଲଳଵ-ହଽଡ଼ଢ଼ୟ-ୡୱஃஅ-ஊஎ-ஐஒ-கஙசஜஞடணதந-பம-ஹௐఅ-ఌఎ-ఐఒ-నప-హఽౘ-ౚౝౠౡಀಅ-ಌಎ-ಐಒ-ನಪ-ಳವ-ಹಽೝೞೠೡೱೲഄ-ഌഎ-ഐഒ-ഺഽൎൔ-ൖൟ-ൡൺ-ൿඅ-ඖක-නඳ-රලව-ෆก-ะาำเ-ๆກຂຄຆ-ຊຌ-ຣລວ-ະາຳຽເ-ໄໆໜ-ໟༀཀ-ཇཉ-ཬྈ-ྌက-ဪဿၐ-ၕၚ-ၝၡၥၦၮ-ၰၵ-ႁႎႠ-ჅჇჍა-ჺჼ-ቈቊ-ቍቐ-ቖቘቚ-ቝበ-ኈኊ-ኍነ-ኰኲ-ኵኸ-ኾዀዂ-ዅወ-ዖዘ-ጐጒ-ጕጘ-ፚᎀ-ᎏᎠ-Ᏽᏸ-ᏽᐁ-ᙬᙯ-ᙿᚁ-ᚚᚠ-ᛪᛮ-ᛸᜀ-ᜑᜟ-ᜱᝀ-ᝑᝠ-ᝬᝮ-ᝰក-ឳៗៜᠠ-ᡸᢀ-ᢨᢪᢰ-ᣵᤀ-ᤞᥐ-ᥭᥰ-ᥴᦀ-ᦫᦰ-ᧉᨀ-ᨖᨠ-ᩔᪧᬅ-ᬳᭅ-ᭌᮃ-ᮠᮮᮯᮺ-ᯥᰀ-ᰣᱍ-ᱏᱚ-ᱽᲀ-ᲊᲐ-ᲺᲽ-Ჿᳩ-ᳬᳮ-ᳳᳵᳶᳺᴀ-ᶿḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼⁱⁿₐ-ₜℂℇℊ-ℓℕ℘-ℝℤΩℨK-ℹℼ-ℿⅅ-ⅉⅎⅠ-ↈⰀ-ⳤⳫ-ⳮⳲⳳⴀ-ⴥⴧⴭⴰ-ⵧⵯⶀ-ⶖⶠ-ⶦⶨ-ⶮⶰ-ⶶⶸ-ⶾⷀ-ⷆⷈ-ⷎⷐ-ⷖⷘ-ⷞ々-〇〡-〩〱-〵〸-〼ぁ-ゖ゛-ゟァ-ヺー-ヿㄅ-ㄯㄱ-ㆎㆠ-ㆿㇰ-ㇿ㐀-䶿一-ꒌꓐ-ꓽꔀ-ꘌꘐ-ꘟꘪꘫꙀ-ꙮꙿ-ꚝꚠ-ꛯꜗ-ꜟꜢ-ꞈꞋ-ꟍꟐꟑꟓꟕ-Ƛꟲ-ꠁꠃ-ꠅꠇ-ꠊꠌ-ꠢꡀ-ꡳꢂ-ꢳꣲ-ꣷꣻꣽꣾꤊ-ꤥꤰ-ꥆꥠ-ꥼꦄ-ꦲꧏꧠ-ꧤꧦ-ꧯꧺ-ꧾꨀ-ꨨꩀ-ꩂꩄ-ꩋꩠ-ꩶꩺꩾ-ꪯꪱꪵꪶꪹ-ꪽꫀꫂꫛ-ꫝꫠ-ꫪꫲ-ꫴꬁ-ꬆꬉ-ꬎꬑ-ꬖꬠ-ꬦꬨ-ꬮꬰ-ꭚꭜ-ꭩꭰ-ꯢ가-힣ힰ-ퟆퟋ-ퟻ豈-舘並-龎ﬀ-ﬆﬓ-ﬗיִײַ-ﬨשׁ-זּטּ-לּמּנּסּףּפּצּ-ﮱﯓ-ﴽﵐ-ﶏﶒ-ﷇﷰ-ﷻﹰ-ﹴﹶ-ﻼＡ-Ｚａ-ｚｦ-ﾾￂ-ￇￊ-ￏￒ-ￗￚ-ￜ",C={3:"abstract boolean byte char class double enum export extends final float goto implements import int interface long native package private protected public short static super synchronized throws transient volatile",5:"class enum extends super const export import",6:"enum",strict:"implements interface let package private protected public static yield",strictBind:"eval arguments"},m="break case catch continue debugger default do else finally for function if return switch throw try var while with null true false instanceof typeof void delete new in this",p={5:m,"5module":m+" export import",6:m+" const class extends export import super"},u=/^in(stanceof)?$/,g=new RegExp("["+d+"]"),E=new RegExp("["+d+l+"]");function w(r,o){for(var x=65536,S=0;S<o.length;S+=2){if(x+=o[S],x>r)return!1;if(x+=o[S+1],x>=r)return!0}return!1}function y(r,o){return r<65?r===36:r<91?!0:r<97?r===95:r<123?!0:r<=65535?r>=170&&g.test(String.fromCharCode(r)):o===!1?!1:w(r,b)}function k(r,o){return r<48?r===36:r<58?!0:r<65?!1:r<91?!0:r<97?r===95:r<123?!0:r<=65535?r>=170&&E.test(String.fromCharCode(r)):o===!1?!1:w(r,b)||w(r,_)}var v=function(o,x){x===void 0&&(x={}),this.label=o,this.keyword=x.keyword,this.beforeExpr=!!x.beforeExpr,this.startsExpr=!!x.startsExpr,this.isLoop=!!x.isLoop,this.isAssign=!!x.isAssign,this.prefix=!!x.prefix,this.postfix=!!x.postfix,this.binop=x.binop||null,this.updateContext=null};function $(r,o){return new v(r,{beforeExpr:!0,binop:o})}var P={beforeExpr:!0},O={startsExpr:!0},T={};function A(r,o){return o===void 0&&(o={}),o.keyword=r,T[r]=new v(r,o)}var f={num:new v("num",O),regexp:new v("regexp",O),string:new v("string",O),name:new v("name",O),privateId:new v("privateId",O),eof:new v("eof"),bracketL:new v("[",{beforeExpr:!0,startsExpr:!0}),bracketR:new v("]"),braceL:new v("{",{beforeExpr:!0,startsExpr:!0}),braceR:new v("}"),parenL:new v("(",{beforeExpr:!0,startsExpr:!0}),parenR:new v(")"),comma:new v(",",P),semi:new v(";",P),colon:new v(":",P),dot:new v("."),question:new v("?",P),questionDot:new v("?."),arrow:new v("=>",P),template:new v("template"),invalidTemplate:new v("invalidTemplate"),ellipsis:new v("...",P),backQuote:new v("`",O),dollarBraceL:new v("${",{beforeExpr:!0,startsExpr:!0}),eq:new v("=",{beforeExpr:!0,isAssign:!0}),assign:new v("_=",{beforeExpr:!0,isAssign:!0}),incDec:new v("++/--",{prefix:!0,postfix:!0,startsExpr:!0}),prefix:new v("!/~",{beforeExpr:!0,prefix:!0,startsExpr:!0}),logicalOR:$("||",1),logicalAND:$("&&",2),bitwiseOR:$("|",3),bitwiseXOR:$("^",4),bitwiseAND:$("&",5),equality:$("==/!=/===/!==",6),relational:$("</>/<=/>=",7),bitShift:$("<</>>/>>>",8),plusMin:new v("+/-",{beforeExpr:!0,binop:9,prefix:!0,startsExpr:!0}),modulo:$("%",10),star:$("*",10),slash:$("/",10),starstar:new v("**",{beforeExpr:!0}),coalesce:$("??",1),_break:A("break"),_case:A("case",P),_catch:A("catch"),_continue:A("continue"),_debugger:A("debugger"),_default:A("default",P),_do:A("do",{isLoop:!0,beforeExpr:!0}),_else:A("else",P),_finally:A("finally"),_for:A("for",{isLoop:!0}),_function:A("function",O),_if:A("if"),_return:A("return",P),_switch:A("switch"),_throw:A("throw",P),_try:A("try"),_var:A("var"),_const:A("const"),_while:A("while",{isLoop:!0}),_with:A("with"),_new:A("new",{beforeExpr:!0,startsExpr:!0}),_this:A("this",O),_super:A("super",O),_class:A("class",O),_extends:A("extends",P),_export:A("export"),_import:A("import",O),_null:A("null",O),_true:A("true",O),_false:A("false",O),_in:A("in",{beforeExpr:!0,binop:7}),_instanceof:A("instanceof",{beforeExpr:!0,binop:7}),_typeof:A("typeof",{beforeExpr:!0,prefix:!0,startsExpr:!0}),_void:A("void",{beforeExpr:!0,prefix:!0,startsExpr:!0}),_delete:A("delete",{beforeExpr:!0,prefix:!0,startsExpr:!0})},F=/\r\n?|\n|\u2028|\u2029/,L=new RegExp(F.source,"g");function R(r){return r===10||r===13||r===8232||r===8233}function V(r,o,x){x===void 0&&(x=r.length);for(var S=o;S<x;S++){var D=r.charCodeAt(S);if(R(D))return S<x-1&&D===13&&r.charCodeAt(S+1)===10?S+2:S+1}return-1}var W=/[\u1680\u2000-\u200a\u202f\u205f\u3000\ufeff]/,N=/(?:\s|\/\/.*|\/\*[^]*?\*\/)*/g,te=Object.prototype,ee=te.hasOwnProperty,X=te.toString,ie=Object.hasOwn||function(r,o){return ee.call(r,o)},se=Array.isArray||function(r){return X.call(r)==="[object Array]"},J=Object.create(null);function ae(r){return J[r]||(J[r]=new RegExp("^(?:"+r.replace(/ /g,"|")+")$"))}function he(r){return r<=65535?String.fromCharCode(r):(r-=65536,String.fromCharCode((r>>10)+55296,(r&1023)+56320))}var Ae=/(?:[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?:[^\uD800-\uDBFF]|^)[\uDC00-\uDFFF])/,ne=function(o,x){this.line=o,this.column=x};ne.prototype.offset=function(o){return new ne(this.line,this.column+o)};var ce=function(o,x,S){this.start=x,this.end=S,o.sourceFile!==null&&(this.source=o.sourceFile)};function Ve(r,o){for(var x=1,S=0;;){var D=V(r,S,o);if(D<0)return new ne(x,o-S);++x,S=D}}var ue={ecmaVersion:null,sourceType:"script",onInsertedSemicolon:null,onTrailingComma:null,allowReserved:null,allowReturnOutsideFunction:!1,allowImportExportEverywhere:!1,allowAwaitOutsideFunction:null,allowSuperOutsideMethod:null,allowHashBang:!1,checkPrivateFields:!0,locations:!1,onToken:null,onComment:null,ranges:!1,program:null,sourceFile:null,directSourceFile:null,preserveParens:!1},Te=!1;function Oe(r){var o={};for(var x in ue)o[x]=r&&ie(r,x)?r[x]:ue[x];if(o.ecmaVersion==="latest"?o.ecmaVersion=1e8:o.ecmaVersion==null?(!Te&&typeof console=="object"&&console.warn&&(Te=!0,console.warn(`Since Acorn 8.0.0, options.ecmaVersion is required.
Defaulting to 2020, but this will stop working in the future.`)),o.ecmaVersion=11):o.ecmaVersion>=2015&&(o.ecmaVersion-=2009),o.allowReserved==null&&(o.allowReserved=o.ecmaVersion<5),(!r||r.allowHashBang==null)&&(o.allowHashBang=o.ecmaVersion>=14),se(o.onToken)){var S=o.onToken;o.onToken=function(D){return S.push(D)}}return se(o.onComment)&&(o.onComment=Ie(o,o.onComment)),o}function Ie(r,o){return function(x,S,D,G,K,H){var Y={type:x?"Block":"Line",value:S,start:D,end:G};r.locations&&(Y.loc=new ce(this,K,H)),r.ranges&&(Y.range=[D,G]),o.push(Y)}}var re=1,ye=2,_e=4,Ne=8,ft=16,ni=32,An=64,ri=128,ps=256,En=re|ye|ps;function In(r,o){return ye|(r?_e:0)|(o?Ne:0)}var Fs=0,Mn=1,mt=2,ii=3,ai=4,oi=5,Se=function(o,x,S){this.options=o=Oe(o),this.sourceFile=o.sourceFile,this.keywords=ae(p[o.ecmaVersion>=6?6:o.sourceType==="module"?"5module":5]);var D="";o.allowReserved!==!0&&(D=C[o.ecmaVersion>=6?6:o.ecmaVersion===5?5:3],o.sourceType==="module"&&(D+=" await")),this.reservedWords=ae(D);var G=(D?D+" ":"")+C.strict;this.reservedWordsStrict=ae(G),this.reservedWordsStrictBind=ae(G+" "+C.strictBind),this.input=String(x),this.containsEsc=!1,S?(this.pos=S,this.lineStart=this.input.lastIndexOf(`
`,S-1)+1,this.curLine=this.input.slice(0,this.lineStart).split(F).length):(this.pos=this.lineStart=0,this.curLine=1),this.type=f.eof,this.value=null,this.start=this.end=this.pos,this.startLoc=this.endLoc=this.curPosition(),this.lastTokEndLoc=this.lastTokStartLoc=null,this.lastTokStart=this.lastTokEnd=this.pos,this.context=this.initialContext(),this.exprAllowed=!0,this.inModule=o.sourceType==="module",this.strict=this.inModule||this.strictDirective(this.pos),this.potentialArrowAt=-1,this.potentialArrowInForAwait=!1,this.yieldPos=this.awaitPos=this.awaitIdentPos=0,this.labels=[],this.undefinedExports=Object.create(null),this.pos===0&&o.allowHashBang&&this.input.slice(0,2)==="#!"&&this.skipLineComment(2),this.scopeStack=[],this.enterScope(re),this.regexpState=null,this.privateNameStack=[]},it={inFunction:{configurable:!0},inGenerator:{configurable:!0},inAsync:{configurable:!0},canAwait:{configurable:!0},allowSuper:{configurable:!0},allowDirectSuper:{configurable:!0},treatFunctionsAsVar:{configurable:!0},allowNewDotTarget:{configurable:!0},inClassStaticBlock:{configurable:!0}};Se.prototype.parse=function(){var o=this.options.program||this.startNode();return this.nextToken(),this.parseTopLevel(o)},it.inFunction.get=function(){return(this.currentVarScope().flags&ye)>0},it.inGenerator.get=function(){return(this.currentVarScope().flags&Ne)>0&&!this.currentVarScope().inClassFieldInit},it.inAsync.get=function(){return(this.currentVarScope().flags&_e)>0&&!this.currentVarScope().inClassFieldInit},it.canAwait.get=function(){for(var r=this.scopeStack.length-1;r>=0;r--){var o=this.scopeStack[r];if(o.inClassFieldInit||o.flags&ps)return!1;if(o.flags&ye)return(o.flags&_e)>0}return this.inModule&&this.options.ecmaVersion>=13||this.options.allowAwaitOutsideFunction},it.allowSuper.get=function(){var r=this.currentThisScope(),o=r.flags,x=r.inClassFieldInit;return(o&An)>0||x||this.options.allowSuperOutsideMethod},it.allowDirectSuper.get=function(){return(this.currentThisScope().flags&ri)>0},it.treatFunctionsAsVar.get=function(){return this.treatFunctionsAsVarInScope(this.currentScope())},it.allowNewDotTarget.get=function(){var r=this.currentThisScope(),o=r.flags,x=r.inClassFieldInit;return(o&(ye|ps))>0||x},it.inClassStaticBlock.get=function(){return(this.currentVarScope().flags&ps)>0},Se.extend=function(){for(var o=[],x=arguments.length;x--;)o[x]=arguments[x];for(var S=this,D=0;D<o.length;D++)S=o[D](S);return S},Se.parse=function(o,x){return new this(x,o).parse()},Se.parseExpressionAt=function(o,x,S){var D=new this(S,o,x);return D.nextToken(),D.parseExpression()},Se.tokenizer=function(o,x){return new this(x,o)},Object.defineProperties(Se.prototype,it);var Le=Se.prototype,Uo=/^(?:'((?:\\[^]|[^'\\])*?)'|"((?:\\[^]|[^"\\])*?)")/;Le.strictDirective=function(r){if(this.options.ecmaVersion<5)return!1;for(;;){N.lastIndex=r,r+=N.exec(this.input)[0].length;var o=Uo.exec(this.input.slice(r));if(!o)return!1;if((o[1]||o[2])==="use strict"){N.lastIndex=r+o[0].length;var x=N.exec(this.input),S=x.index+x[0].length,D=this.input.charAt(S);return D===";"||D==="}"||F.test(x[0])&&!(/[(`.[+\-/*%<>=,?^&]/.test(D)||D==="!"&&this.input.charAt(S+1)==="=")}r+=o[0].length,N.lastIndex=r,r+=N.exec(this.input)[0].length,this.input[r]===";"&&r++}},Le.eat=function(r){return this.type===r?(this.next(),!0):!1},Le.isContextual=function(r){return this.type===f.name&&this.value===r&&!this.containsEsc},Le.eatContextual=function(r){return this.isContextual(r)?(this.next(),!0):!1},Le.expectContextual=function(r){this.eatContextual(r)||this.unexpected()},Le.canInsertSemicolon=function(){return this.type===f.eof||this.type===f.braceR||F.test(this.input.slice(this.lastTokEnd,this.start))},Le.insertSemicolon=function(){if(this.canInsertSemicolon())return this.options.onInsertedSemicolon&&this.options.onInsertedSemicolon(this.lastTokEnd,this.lastTokEndLoc),!0},Le.semicolon=function(){!this.eat(f.semi)&&!this.insertSemicolon()&&this.unexpected()},Le.afterTrailingComma=function(r,o){if(this.type===r)return this.options.onTrailingComma&&this.options.onTrailingComma(this.lastTokStart,this.lastTokStartLoc),o||this.next(),!0},Le.expect=function(r){this.eat(r)||this.unexpected()},Le.unexpected=function(r){this.raise(r??this.start,"Unexpected token")};var Gs=function(){this.shorthandAssign=this.trailingComma=this.parenthesizedAssign=this.parenthesizedBind=this.doubleProto=-1};Le.checkPatternErrors=function(r,o){if(r){r.trailingComma>-1&&this.raiseRecoverable(r.trailingComma,"Comma is not permitted after the rest element");var x=o?r.parenthesizedAssign:r.parenthesizedBind;x>-1&&this.raiseRecoverable(x,o?"Assigning to rvalue":"Parenthesized pattern")}},Le.checkExpressionErrors=function(r,o){if(!r)return!1;var x=r.shorthandAssign,S=r.doubleProto;if(!o)return x>=0||S>=0;x>=0&&this.raise(x,"Shorthand property assignments are valid only in destructuring patterns"),S>=0&&this.raiseRecoverable(S,"Redefinition of __proto__ property")},Le.checkYieldAwaitInDefaultParams=function(){this.yieldPos&&(!this.awaitPos||this.yieldPos<this.awaitPos)&&this.raise(this.yieldPos,"Yield expression cannot be a default value"),this.awaitPos&&this.raise(this.awaitPos,"Await expression cannot be a default value")},Le.isSimpleAssignTarget=function(r){return r.type==="ParenthesizedExpression"?this.isSimpleAssignTarget(r.expression):r.type==="Identifier"||r.type==="MemberExpression"};var Q=Se.prototype;Q.parseTopLevel=function(r){var o=Object.create(null);for(r.body||(r.body=[]);this.type!==f.eof;){var x=this.parseStatement(null,!0,o);r.body.push(x)}if(this.inModule)for(var S=0,D=Object.keys(this.undefinedExports);S<D.length;S+=1){var G=D[S];this.raiseRecoverable(this.undefinedExports[G].start,"Export '"+G+"' is not defined")}return this.adaptDirectivePrologue(r.body),this.next(),r.sourceType=this.options.sourceType,this.finishNode(r,"Program")};var $n={kind:"loop"},Vo={kind:"switch"};Q.isLet=function(r){if(this.options.ecmaVersion<6||!this.isContextual("let"))return!1;N.lastIndex=this.pos;var o=N.exec(this.input),x=this.pos+o[0].length,S=this.input.charCodeAt(x);if(S===91||S===92)return!0;if(r)return!1;if(S===123||S>55295&&S<56320)return!0;if(y(S,!0)){for(var D=x+1;k(S=this.input.charCodeAt(D),!0);)++D;if(S===92||S>55295&&S<56320)return!0;var G=this.input.slice(x,D);if(!u.test(G))return!0}return!1},Q.isAsyncFunction=function(){if(this.options.ecmaVersion<8||!this.isContextual("async"))return!1;N.lastIndex=this.pos;var r=N.exec(this.input),o=this.pos+r[0].length,x;return!F.test(this.input.slice(this.pos,o))&&this.input.slice(o,o+8)==="function"&&(o+8===this.input.length||!(k(x=this.input.charCodeAt(o+8))||x>55295&&x<56320))},Q.parseStatement=function(r,o,x){var S=this.type,D=this.startNode(),G;switch(this.isLet(r)&&(S=f._var,G="let"),S){case f._break:case f._continue:return this.parseBreakContinueStatement(D,S.keyword);case f._debugger:return this.parseDebuggerStatement(D);case f._do:return this.parseDoStatement(D);case f._for:return this.parseForStatement(D);case f._function:return r&&(this.strict||r!=="if"&&r!=="label")&&this.options.ecmaVersion>=6&&this.unexpected(),this.parseFunctionStatement(D,!1,!r);case f._class:return r&&this.unexpected(),this.parseClass(D,!0);case f._if:return this.parseIfStatement(D);case f._return:return this.parseReturnStatement(D);case f._switch:return this.parseSwitchStatement(D);case f._throw:return this.parseThrowStatement(D);case f._try:return this.parseTryStatement(D);case f._const:case f._var:return G=G||this.value,r&&G!=="var"&&this.unexpected(),this.parseVarStatement(D,G);case f._while:return this.parseWhileStatement(D);case f._with:return this.parseWithStatement(D);case f.braceL:return this.parseBlock(!0,D);case f.semi:return this.parseEmptyStatement(D);case f._export:case f._import:if(this.options.ecmaVersion>10&&S===f._import){N.lastIndex=this.pos;var K=N.exec(this.input),H=this.pos+K[0].length,Y=this.input.charCodeAt(H);if(Y===40||Y===46)return this.parseExpressionStatement(D,this.parseExpression())}return this.options.allowImportExportEverywhere||(o||this.raise(this.start,"'import' and 'export' may only appear at the top level"),this.inModule||this.raise(this.start,"'import' and 'export' may appear only with 'sourceType: module'")),S===f._import?this.parseImport(D):this.parseExport(D,x);default:if(this.isAsyncFunction())return r&&this.unexpected(),this.next(),this.parseFunctionStatement(D,!0,!r);var de=this.value,le=this.parseExpression();return S===f.name&&le.type==="Identifier"&&this.eat(f.colon)?this.parseLabeledStatement(D,de,le,r):this.parseExpressionStatement(D,le)}},Q.parseBreakContinueStatement=function(r,o){var x=o==="break";this.next(),this.eat(f.semi)||this.insertSemicolon()?r.label=null:this.type!==f.name?this.unexpected():(r.label=this.parseIdent(),this.semicolon());for(var S=0;S<this.labels.length;++S){var D=this.labels[S];if((r.label==null||D.name===r.label.name)&&(D.kind!=null&&(x||D.kind==="loop")||r.label&&x))break}return S===this.labels.length&&this.raise(r.start,"Unsyntactic "+o),this.finishNode(r,x?"BreakStatement":"ContinueStatement")},Q.parseDebuggerStatement=function(r){return this.next(),this.semicolon(),this.finishNode(r,"DebuggerStatement")},Q.parseDoStatement=function(r){return this.next(),this.labels.push($n),r.body=this.parseStatement("do"),this.labels.pop(),this.expect(f._while),r.test=this.parseParenExpression(),this.options.ecmaVersion>=6?this.eat(f.semi):this.semicolon(),this.finishNode(r,"DoWhileStatement")},Q.parseForStatement=function(r){this.next();var o=this.options.ecmaVersion>=9&&this.canAwait&&this.eatContextual("await")?this.lastTokStart:-1;if(this.labels.push($n),this.enterScope(0),this.expect(f.parenL),this.type===f.semi)return o>-1&&this.unexpected(o),this.parseFor(r,null);var x=this.isLet();if(this.type===f._var||this.type===f._const||x){var S=this.startNode(),D=x?"let":this.value;return this.next(),this.parseVar(S,!0,D),this.finishNode(S,"VariableDeclaration"),(this.type===f._in||this.options.ecmaVersion>=6&&this.isContextual("of"))&&S.declarations.length===1?(this.options.ecmaVersion>=9&&(this.type===f._in?o>-1&&this.unexpected(o):r.await=o>-1),this.parseForIn(r,S)):(o>-1&&this.unexpected(o),this.parseFor(r,S))}var G=this.isContextual("let"),K=!1,H=this.containsEsc,Y=new Gs,de=this.start,le=o>-1?this.parseExprSubscripts(Y,"await"):this.parseExpression(!0,Y);return this.type===f._in||(K=this.options.ecmaVersion>=6&&this.isContextual("of"))?(o>-1?(this.type===f._in&&this.unexpected(o),r.await=!0):K&&this.options.ecmaVersion>=8&&(le.start===de&&!H&&le.type==="Identifier"&&le.name==="async"?this.unexpected():this.options.ecmaVersion>=9&&(r.await=!1)),G&&K&&this.raise(le.start,"The left-hand side of a for-of loop may not start with 'let'."),this.toAssignable(le,!1,Y),this.checkLValPattern(le),this.parseForIn(r,le)):(this.checkExpressionErrors(Y,!0),o>-1&&this.unexpected(o),this.parseFor(r,le))},Q.parseFunctionStatement=function(r,o,x){return this.next(),this.parseFunction(r,fs|(x?0:Dn),!1,o)},Q.parseIfStatement=function(r){return this.next(),r.test=this.parseParenExpression(),r.consequent=this.parseStatement("if"),r.alternate=this.eat(f._else)?this.parseStatement("if"):null,this.finishNode(r,"IfStatement")},Q.parseReturnStatement=function(r){return!this.inFunction&&!this.options.allowReturnOutsideFunction&&this.raise(this.start,"'return' outside of function"),this.next(),this.eat(f.semi)||this.insertSemicolon()?r.argument=null:(r.argument=this.parseExpression(),this.semicolon()),this.finishNode(r,"ReturnStatement")},Q.parseSwitchStatement=function(r){this.next(),r.discriminant=this.parseParenExpression(),r.cases=[],this.expect(f.braceL),this.labels.push(Vo),this.enterScope(0);for(var o,x=!1;this.type!==f.braceR;)if(this.type===f._case||this.type===f._default){var S=this.type===f._case;o&&this.finishNode(o,"SwitchCase"),r.cases.push(o=this.startNode()),o.consequent=[],this.next(),S?o.test=this.parseExpression():(x&&this.raiseRecoverable(this.lastTokStart,"Multiple default clauses"),x=!0,o.test=null),this.expect(f.colon)}else o||this.unexpected(),o.consequent.push(this.parseStatement(null));return this.exitScope(),o&&this.finishNode(o,"SwitchCase"),this.next(),this.labels.pop(),this.finishNode(r,"SwitchStatement")},Q.parseThrowStatement=function(r){return this.next(),F.test(this.input.slice(this.lastTokEnd,this.start))&&this.raise(this.lastTokEnd,"Illegal newline after throw"),r.argument=this.parseExpression(),this.semicolon(),this.finishNode(r,"ThrowStatement")};var Ko=[];Q.parseCatchClauseParam=function(){var r=this.parseBindingAtom(),o=r.type==="Identifier";return this.enterScope(o?ni:0),this.checkLValPattern(r,o?ai:mt),this.expect(f.parenR),r},Q.parseTryStatement=function(r){if(this.next(),r.block=this.parseBlock(),r.handler=null,this.type===f._catch){var o=this.startNode();this.next(),this.eat(f.parenL)?o.param=this.parseCatchClauseParam():(this.options.ecmaVersion<10&&this.unexpected(),o.param=null,this.enterScope(0)),o.body=this.parseBlock(!1),this.exitScope(),r.handler=this.finishNode(o,"CatchClause")}return r.finalizer=this.eat(f._finally)?this.parseBlock():null,!r.handler&&!r.finalizer&&this.raise(r.start,"Missing catch or finally clause"),this.finishNode(r,"TryStatement")},Q.parseVarStatement=function(r,o,x){return this.next(),this.parseVar(r,!1,o,x),this.semicolon(),this.finishNode(r,"VariableDeclaration")},Q.parseWhileStatement=function(r){return this.next(),r.test=this.parseParenExpression(),this.labels.push($n),r.body=this.parseStatement("while"),this.labels.pop(),this.finishNode(r,"WhileStatement")},Q.parseWithStatement=function(r){return this.strict&&this.raise(this.start,"'with' in strict mode"),this.next(),r.object=this.parseParenExpression(),r.body=this.parseStatement("with"),this.finishNode(r,"WithStatement")},Q.parseEmptyStatement=function(r){return this.next(),this.finishNode(r,"EmptyStatement")},Q.parseLabeledStatement=function(r,o,x,S){for(var D=0,G=this.labels;D<G.length;D+=1)G[D].name===o&&this.raise(x.start,"Label '"+o+"' is already declared");for(var K=this.type.isLoop?"loop":this.type===f._switch?"switch":null,H=this.labels.length-1;H>=0;H--){var Y=this.labels[H];if(Y.statementStart===r.start)Y.statementStart=this.start,Y.kind=K;else break}return this.labels.push({name:o,kind:K,statementStart:this.start}),r.body=this.parseStatement(S?S.indexOf("label")===-1?S+"label":S:"label"),this.labels.pop(),r.label=x,this.finishNode(r,"LabeledStatement")},Q.parseExpressionStatement=function(r,o){return r.expression=o,this.semicolon(),this.finishNode(r,"ExpressionStatement")},Q.parseBlock=function(r,o,x){for(r===void 0&&(r=!0),o===void 0&&(o=this.startNode()),o.body=[],this.expect(f.braceL),r&&this.enterScope(0);this.type!==f.braceR;){var S=this.parseStatement(null);o.body.push(S)}return x&&(this.strict=!1),this.next(),r&&this.exitScope(),this.finishNode(o,"BlockStatement")},Q.parseFor=function(r,o){return r.init=o,this.expect(f.semi),r.test=this.type===f.semi?null:this.parseExpression(),this.expect(f.semi),r.update=this.type===f.parenR?null:this.parseExpression(),this.expect(f.parenR),r.body=this.parseStatement("for"),this.exitScope(),this.labels.pop(),this.finishNode(r,"ForStatement")},Q.parseForIn=function(r,o){var x=this.type===f._in;return this.next(),o.type==="VariableDeclaration"&&o.declarations[0].init!=null&&(!x||this.options.ecmaVersion<8||this.strict||o.kind!=="var"||o.declarations[0].id.type!=="Identifier")&&this.raise(o.start,(x?"for-in":"for-of")+" loop variable declaration may not have an initializer"),r.left=o,r.right=x?this.parseExpression():this.parseMaybeAssign(),this.expect(f.parenR),r.body=this.parseStatement("for"),this.exitScope(),this.labels.pop(),this.finishNode(r,x?"ForInStatement":"ForOfStatement")},Q.parseVar=function(r,o,x,S){for(r.declarations=[],r.kind=x;;){var D=this.startNode();if(this.parseVarId(D,x),this.eat(f.eq)?D.init=this.parseMaybeAssign(o):!S&&x==="const"&&!(this.type===f._in||this.options.ecmaVersion>=6&&this.isContextual("of"))?this.unexpected():!S&&D.id.type!=="Identifier"&&!(o&&(this.type===f._in||this.isContextual("of")))?this.raise(this.lastTokEnd,"Complex binding patterns require an initialization value"):D.init=null,r.declarations.push(this.finishNode(D,"VariableDeclarator")),!this.eat(f.comma))break}return r},Q.parseVarId=function(r,o){r.id=this.parseBindingAtom(),this.checkLValPattern(r.id,o==="var"?Mn:mt,!1)};var fs=1,Dn=2,li=4;Q.parseFunction=function(r,o,x,S,D){this.initFunction(r),(this.options.ecmaVersion>=9||this.options.ecmaVersion>=6&&!S)&&(this.type===f.star&&o&Dn&&this.unexpected(),r.generator=this.eat(f.star)),this.options.ecmaVersion>=8&&(r.async=!!S),o&fs&&(r.id=o&li&&this.type!==f.name?null:this.parseIdent(),r.id&&!(o&Dn)&&this.checkLValSimple(r.id,this.strict||r.generator||r.async?this.treatFunctionsAsVar?Mn:mt:ii));var G=this.yieldPos,K=this.awaitPos,H=this.awaitIdentPos;return this.yieldPos=0,this.awaitPos=0,this.awaitIdentPos=0,this.enterScope(In(r.async,r.generator)),o&fs||(r.id=this.type===f.name?this.parseIdent():null),this.parseFunctionParams(r),this.parseFunctionBody(r,x,!1,D),this.yieldPos=G,this.awaitPos=K,this.awaitIdentPos=H,this.finishNode(r,o&fs?"FunctionDeclaration":"FunctionExpression")},Q.parseFunctionParams=function(r){this.expect(f.parenL),r.params=this.parseBindingList(f.parenR,!1,this.options.ecmaVersion>=8),this.checkYieldAwaitInDefaultParams()},Q.parseClass=function(r,o){this.next();var x=this.strict;this.strict=!0,this.parseClassId(r,o),this.parseClassSuper(r);var S=this.enterClassBody(),D=this.startNode(),G=!1;for(D.body=[],this.expect(f.braceL);this.type!==f.braceR;){var K=this.parseClassElement(r.superClass!==null);K&&(D.body.push(K),K.type==="MethodDefinition"&&K.kind==="constructor"?(G&&this.raiseRecoverable(K.start,"Duplicate constructor in the same class"),G=!0):K.key&&K.key.type==="PrivateIdentifier"&&No(S,K)&&this.raiseRecoverable(K.key.start,"Identifier '#"+K.key.name+"' has already been declared"))}return this.strict=x,this.next(),r.body=this.finishNode(D,"ClassBody"),this.exitClassBody(),this.finishNode(r,o?"ClassDeclaration":"ClassExpression")},Q.parseClassElement=function(r){if(this.eat(f.semi))return null;var o=this.options.ecmaVersion,x=this.startNode(),S="",D=!1,G=!1,K="method",H=!1;if(this.eatContextual("static")){if(o>=13&&this.eat(f.braceL))return this.parseClassStaticBlock(x),x;this.isClassElementNameStart()||this.type===f.star?H=!0:S="static"}if(x.static=H,!S&&o>=8&&this.eatContextual("async")&&((this.isClassElementNameStart()||this.type===f.star)&&!this.canInsertSemicolon()?G=!0:S="async"),!S&&(o>=9||!G)&&this.eat(f.star)&&(D=!0),!S&&!G&&!D){var Y=this.value;(this.eatContextual("get")||this.eatContextual("set"))&&(this.isClassElementNameStart()?K=Y:S=Y)}if(S?(x.computed=!1,x.key=this.startNodeAt(this.lastTokStart,this.lastTokStartLoc),x.key.name=S,this.finishNode(x.key,"Identifier")):this.parseClassElementName(x),o<13||this.type===f.parenL||K!=="method"||D||G){var de=!x.static&&Us(x,"constructor"),le=de&&r;de&&K!=="method"&&this.raise(x.key.start,"Constructor can't have get/set modifier"),x.kind=de?"constructor":K,this.parseClassMethod(x,D,G,le)}else this.parseClassField(x);return x},Q.isClassElementNameStart=function(){return this.type===f.name||this.type===f.privateId||this.type===f.num||this.type===f.string||this.type===f.bracketL||this.type.keyword},Q.parseClassElementName=function(r){this.type===f.privateId?(this.value==="constructor"&&this.raise(this.start,"Classes can't have an element named '#constructor'"),r.computed=!1,r.key=this.parsePrivateIdent()):this.parsePropertyName(r)},Q.parseClassMethod=function(r,o,x,S){var D=r.key;r.kind==="constructor"?(o&&this.raise(D.start,"Constructor can't be a generator"),x&&this.raise(D.start,"Constructor can't be an async method")):r.static&&Us(r,"prototype")&&this.raise(D.start,"Classes may not have a static property named prototype");var G=r.value=this.parseMethod(o,x,S);return r.kind==="get"&&G.params.length!==0&&this.raiseRecoverable(G.start,"getter should have no params"),r.kind==="set"&&G.params.length!==1&&this.raiseRecoverable(G.start,"setter should have exactly one param"),r.kind==="set"&&G.params[0].type==="RestElement"&&this.raiseRecoverable(G.params[0].start,"Setter cannot use rest params"),this.finishNode(r,"MethodDefinition")},Q.parseClassField=function(r){if(Us(r,"constructor")?this.raise(r.key.start,"Classes can't have a field named 'constructor'"):r.static&&Us(r,"prototype")&&this.raise(r.key.start,"Classes can't have a static field named 'prototype'"),this.eat(f.eq)){var o=this.currentThisScope(),x=o.inClassFieldInit;o.inClassFieldInit=!0,r.value=this.parseMaybeAssign(),o.inClassFieldInit=x}else r.value=null;return this.semicolon(),this.finishNode(r,"PropertyDefinition")},Q.parseClassStaticBlock=function(r){r.body=[];var o=this.labels;for(this.labels=[],this.enterScope(ps|An);this.type!==f.braceR;){var x=this.parseStatement(null);r.body.push(x)}return this.next(),this.exitScope(),this.labels=o,this.finishNode(r,"StaticBlock")},Q.parseClassId=function(r,o){this.type===f.name?(r.id=this.parseIdent(),o&&this.checkLValSimple(r.id,mt,!1)):(o===!0&&this.unexpected(),r.id=null)},Q.parseClassSuper=function(r){r.superClass=this.eat(f._extends)?this.parseExprSubscripts(null,!1):null},Q.enterClassBody=function(){var r={declared:Object.create(null),used:[]};return this.privateNameStack.push(r),r.declared},Q.exitClassBody=function(){var r=this.privateNameStack.pop(),o=r.declared,x=r.used;if(this.options.checkPrivateFields)for(var S=this.privateNameStack.length,D=S===0?null:this.privateNameStack[S-1],G=0;G<x.length;++G){var K=x[G];ie(o,K.name)||(D?D.used.push(K):this.raiseRecoverable(K.start,"Private field '#"+K.name+"' must be declared in an enclosing class"))}};function No(r,o){var x=o.key.name,S=r[x],D="true";return o.type==="MethodDefinition"&&(o.kind==="get"||o.kind==="set")&&(D=(o.static?"s":"i")+o.kind),S==="iget"&&D==="iset"||S==="iset"&&D==="iget"||S==="sget"&&D==="sset"||S==="sset"&&D==="sget"?(r[x]="true",!1):S?!0:(r[x]=D,!1)}function Us(r,o){var x=r.computed,S=r.key;return!x&&(S.type==="Identifier"&&S.name===o||S.type==="Literal"&&S.value===o)}Q.parseExportAllDeclaration=function(r,o){return this.options.ecmaVersion>=11&&(this.eatContextual("as")?(r.exported=this.parseModuleExportName(),this.checkExport(o,r.exported,this.lastTokStart)):r.exported=null),this.expectContextual("from"),this.type!==f.string&&this.unexpected(),r.source=this.parseExprAtom(),this.options.ecmaVersion>=16&&(r.attributes=this.parseWithClause()),this.semicolon(),this.finishNode(r,"ExportAllDeclaration")},Q.parseExport=function(r,o){if(this.next(),this.eat(f.star))return this.parseExportAllDeclaration(r,o);if(this.eat(f._default))return this.checkExport(o,"default",this.lastTokStart),r.declaration=this.parseExportDefaultDeclaration(),this.finishNode(r,"ExportDefaultDeclaration");if(this.shouldParseExportStatement())r.declaration=this.parseExportDeclaration(r),r.declaration.type==="VariableDeclaration"?this.checkVariableExport(o,r.declaration.declarations):this.checkExport(o,r.declaration.id,r.declaration.id.start),r.specifiers=[],r.source=null;else{if(r.declaration=null,r.specifiers=this.parseExportSpecifiers(o),this.eatContextual("from"))this.type!==f.string&&this.unexpected(),r.source=this.parseExprAtom(),this.options.ecmaVersion>=16&&(r.attributes=this.parseWithClause());else{for(var x=0,S=r.specifiers;x<S.length;x+=1){var D=S[x];this.checkUnreserved(D.local),this.checkLocalExport(D.local),D.local.type==="Literal"&&this.raise(D.local.start,"A string literal cannot be used as an exported binding without `from`.")}r.source=null}this.semicolon()}return this.finishNode(r,"ExportNamedDeclaration")},Q.parseExportDeclaration=function(r){return this.parseStatement(null)},Q.parseExportDefaultDeclaration=function(){var r;if(this.type===f._function||(r=this.isAsyncFunction())){var o=this.startNode();return this.next(),r&&this.next(),this.parseFunction(o,fs|li,!1,r)}else if(this.type===f._class){var x=this.startNode();return this.parseClass(x,"nullableID")}else{var S=this.parseMaybeAssign();return this.semicolon(),S}},Q.checkExport=function(r,o,x){r&&(typeof o!="string"&&(o=o.type==="Identifier"?o.name:o.value),ie(r,o)&&this.raiseRecoverable(x,"Duplicate export '"+o+"'"),r[o]=!0)},Q.checkPatternExport=function(r,o){var x=o.type;if(x==="Identifier")this.checkExport(r,o,o.start);else if(x==="ObjectPattern")for(var S=0,D=o.properties;S<D.length;S+=1){var G=D[S];this.checkPatternExport(r,G)}else if(x==="ArrayPattern")for(var K=0,H=o.elements;K<H.length;K+=1){var Y=H[K];Y&&this.checkPatternExport(r,Y)}else x==="Property"?this.checkPatternExport(r,o.value):x==="AssignmentPattern"?this.checkPatternExport(r,o.left):x==="RestElement"&&this.checkPatternExport(r,o.argument)},Q.checkVariableExport=function(r,o){if(r)for(var x=0,S=o;x<S.length;x+=1){var D=S[x];this.checkPatternExport(r,D.id)}},Q.shouldParseExportStatement=function(){return this.type.keyword==="var"||this.type.keyword==="const"||this.type.keyword==="class"||this.type.keyword==="function"||this.isLet()||this.isAsyncFunction()},Q.parseExportSpecifier=function(r){var o=this.startNode();return o.local=this.parseModuleExportName(),o.exported=this.eatContextual("as")?this.parseModuleExportName():o.local,this.checkExport(r,o.exported,o.exported.start),this.finishNode(o,"ExportSpecifier")},Q.parseExportSpecifiers=function(r){var o=[],x=!0;for(this.expect(f.braceL);!this.eat(f.braceR);){if(x)x=!1;else if(this.expect(f.comma),this.afterTrailingComma(f.braceR))break;o.push(this.parseExportSpecifier(r))}return o},Q.parseImport=function(r){return this.next(),this.type===f.string?(r.specifiers=Ko,r.source=this.parseExprAtom()):(r.specifiers=this.parseImportSpecifiers(),this.expectContextual("from"),r.source=this.type===f.string?this.parseExprAtom():this.unexpected()),this.options.ecmaVersion>=16&&(r.attributes=this.parseWithClause()),this.semicolon(),this.finishNode(r,"ImportDeclaration")},Q.parseImportSpecifier=function(){var r=this.startNode();return r.imported=this.parseModuleExportName(),this.eatContextual("as")?r.local=this.parseIdent():(this.checkUnreserved(r.imported),r.local=r.imported),this.checkLValSimple(r.local,mt),this.finishNode(r,"ImportSpecifier")},Q.parseImportDefaultSpecifier=function(){var r=this.startNode();return r.local=this.parseIdent(),this.checkLValSimple(r.local,mt),this.finishNode(r,"ImportDefaultSpecifier")},Q.parseImportNamespaceSpecifier=function(){var r=this.startNode();return this.next(),this.expectContextual("as"),r.local=this.parseIdent(),this.checkLValSimple(r.local,mt),this.finishNode(r,"ImportNamespaceSpecifier")},Q.parseImportSpecifiers=function(){var r=[],o=!0;if(this.type===f.name&&(r.push(this.parseImportDefaultSpecifier()),!this.eat(f.comma)))return r;if(this.type===f.star)return r.push(this.parseImportNamespaceSpecifier()),r;for(this.expect(f.braceL);!this.eat(f.braceR);){if(o)o=!1;else if(this.expect(f.comma),this.afterTrailingComma(f.braceR))break;r.push(this.parseImportSpecifier())}return r},Q.parseWithClause=function(){var r=[];if(!this.eat(f._with))return r;this.expect(f.braceL);for(var o={},x=!0;!this.eat(f.braceR);){if(x)x=!1;else if(this.expect(f.comma),this.afterTrailingComma(f.braceR))break;var S=this.parseImportAttribute(),D=S.key.type==="Identifier"?S.key.name:S.key.value;ie(o,D)&&this.raiseRecoverable(S.key.start,"Duplicate attribute key '"+D+"'"),o[D]=!0,r.push(S)}return r},Q.parseImportAttribute=function(){var r=this.startNode();return r.key=this.type===f.string?this.parseExprAtom():this.parseIdent(this.options.allowReserved!=="never"),this.expect(f.colon),this.type!==f.string&&this.unexpected(),r.value=this.parseExprAtom(),this.finishNode(r,"ImportAttribute")},Q.parseModuleExportName=function(){if(this.options.ecmaVersion>=13&&this.type===f.string){var r=this.parseLiteral(this.value);return Ae.test(r.value)&&this.raise(r.start,"An export name cannot include a lone surrogate."),r}return this.parseIdent(!0)},Q.adaptDirectivePrologue=function(r){for(var o=0;o<r.length&&this.isDirectiveCandidate(r[o]);++o)r[o].directive=r[o].expression.raw.slice(1,-1)},Q.isDirectiveCandidate=function(r){return this.options.ecmaVersion>=5&&r.type==="ExpressionStatement"&&r.expression.type==="Literal"&&typeof r.expression.value=="string"&&(this.input[r.start]==='"'||this.input[r.start]==="'")};var Be=Se.prototype;Be.toAssignable=function(r,o,x){if(this.options.ecmaVersion>=6&&r)switch(r.type){case"Identifier":this.inAsync&&r.name==="await"&&this.raise(r.start,"Cannot use 'await' as identifier inside an async function");break;case"ObjectPattern":case"ArrayPattern":case"AssignmentPattern":case"RestElement":break;case"ObjectExpression":r.type="ObjectPattern",x&&this.checkPatternErrors(x,!0);for(var S=0,D=r.properties;S<D.length;S+=1){var G=D[S];this.toAssignable(G,o),G.type==="RestElement"&&(G.argument.type==="ArrayPattern"||G.argument.type==="ObjectPattern")&&this.raise(G.argument.start,"Unexpected token")}break;case"Property":r.kind!=="init"&&this.raise(r.key.start,"Object pattern can't contain getter or setter"),this.toAssignable(r.value,o);break;case"ArrayExpression":r.type="ArrayPattern",x&&this.checkPatternErrors(x,!0),this.toAssignableList(r.elements,o);break;case"SpreadElement":r.type="RestElement",this.toAssignable(r.argument,o),r.argument.type==="AssignmentPattern"&&this.raise(r.argument.start,"Rest elements cannot have a default value");break;case"AssignmentExpression":r.operator!=="="&&this.raise(r.left.end,"Only '=' operator can be used for specifying default value."),r.type="AssignmentPattern",delete r.operator,this.toAssignable(r.left,o);break;case"ParenthesizedExpression":this.toAssignable(r.expression,o,x);break;case"ChainExpression":this.raiseRecoverable(r.start,"Optional chaining cannot appear in left-hand side");break;case"MemberExpression":if(!o)break;default:this.raise(r.start,"Assigning to rvalue")}else x&&this.checkPatternErrors(x,!0);return r},Be.toAssignableList=function(r,o){for(var x=r.length,S=0;S<x;S++){var D=r[S];D&&this.toAssignable(D,o)}if(x){var G=r[x-1];this.options.ecmaVersion===6&&o&&G&&G.type==="RestElement"&&G.argument.type!=="Identifier"&&this.unexpected(G.argument.start)}return r},Be.parseSpread=function(r){var o=this.startNode();return this.next(),o.argument=this.parseMaybeAssign(!1,r),this.finishNode(o,"SpreadElement")},Be.parseRestBinding=function(){var r=this.startNode();return this.next(),this.options.ecmaVersion===6&&this.type!==f.name&&this.unexpected(),r.argument=this.parseBindingAtom(),this.finishNode(r,"RestElement")},Be.parseBindingAtom=function(){if(this.options.ecmaVersion>=6)switch(this.type){case f.bracketL:var r=this.startNode();return this.next(),r.elements=this.parseBindingList(f.bracketR,!0,!0),this.finishNode(r,"ArrayPattern");case f.braceL:return this.parseObj(!0)}return this.parseIdent()},Be.parseBindingList=function(r,o,x,S){for(var D=[],G=!0;!this.eat(r);)if(G?G=!1:this.expect(f.comma),o&&this.type===f.comma)D.push(null);else{if(x&&this.afterTrailingComma(r))break;if(this.type===f.ellipsis){var K=this.parseRestBinding();this.parseBindingListItem(K),D.push(K),this.type===f.comma&&this.raiseRecoverable(this.start,"Comma is not permitted after the rest element"),this.expect(r);break}else D.push(this.parseAssignableListItem(S))}return D},Be.parseAssignableListItem=function(r){var o=this.parseMaybeDefault(this.start,this.startLoc);return this.parseBindingListItem(o),o},Be.parseBindingListItem=function(r){return r},Be.parseMaybeDefault=function(r,o,x){if(x=x||this.parseBindingAtom(),this.options.ecmaVersion<6||!this.eat(f.eq))return x;var S=this.startNodeAt(r,o);return S.left=x,S.right=this.parseMaybeAssign(),this.finishNode(S,"AssignmentPattern")},Be.checkLValSimple=function(r,o,x){o===void 0&&(o=Fs);var S=o!==Fs;switch(r.type){case"Identifier":this.strict&&this.reservedWordsStrictBind.test(r.name)&&this.raiseRecoverable(r.start,(S?"Binding ":"Assigning to ")+r.name+" in strict mode"),S&&(o===mt&&r.name==="let"&&this.raiseRecoverable(r.start,"let is disallowed as a lexically bound name"),x&&(ie(x,r.name)&&this.raiseRecoverable(r.start,"Argument name clash"),x[r.name]=!0),o!==oi&&this.declareName(r.name,o,r.start));break;case"ChainExpression":this.raiseRecoverable(r.start,"Optional chaining cannot appear in left-hand side");break;case"MemberExpression":S&&this.raiseRecoverable(r.start,"Binding member expression");break;case"ParenthesizedExpression":return S&&this.raiseRecoverable(r.start,"Binding parenthesized expression"),this.checkLValSimple(r.expression,o,x);default:this.raise(r.start,(S?"Binding":"Assigning to")+" rvalue")}},Be.checkLValPattern=function(r,o,x){switch(o===void 0&&(o=Fs),r.type){case"ObjectPattern":for(var S=0,D=r.properties;S<D.length;S+=1){var G=D[S];this.checkLValInnerPattern(G,o,x)}break;case"ArrayPattern":for(var K=0,H=r.elements;K<H.length;K+=1){var Y=H[K];Y&&this.checkLValInnerPattern(Y,o,x)}break;default:this.checkLValSimple(r,o,x)}},Be.checkLValInnerPattern=function(r,o,x){switch(o===void 0&&(o=Fs),r.type){case"Property":this.checkLValInnerPattern(r.value,o,x);break;case"AssignmentPattern":this.checkLValPattern(r.left,o,x);break;case"RestElement":this.checkLValPattern(r.argument,o,x);break;default:this.checkLValPattern(r,o,x)}};var je=function(o,x,S,D,G){this.token=o,this.isExpr=!!x,this.preserveSpace=!!S,this.override=D,this.generator=!!G},be={b_stat:new je("{",!1),b_expr:new je("{",!0),b_tmpl:new je("${",!1),p_stat:new je("(",!1),p_expr:new je("(",!0),q_tmpl:new je("`",!0,!0,function(r){return r.tryReadTemplateToken()}),f_stat:new je("function",!1),f_expr:new je("function",!0),f_expr_gen:new je("function",!0,!1,null,!0),f_gen:new je("function",!1,!1,null,!0)},qt=Se.prototype;qt.initialContext=function(){return[be.b_stat]},qt.curContext=function(){return this.context[this.context.length-1]},qt.braceIsBlock=function(r){var o=this.curContext();return o===be.f_expr||o===be.f_stat?!0:r===f.colon&&(o===be.b_stat||o===be.b_expr)?!o.isExpr:r===f._return||r===f.name&&this.exprAllowed?F.test(this.input.slice(this.lastTokEnd,this.start)):r===f._else||r===f.semi||r===f.eof||r===f.parenR||r===f.arrow?!0:r===f.braceL?o===be.b_stat:r===f._var||r===f._const||r===f.name?!1:!this.exprAllowed},qt.inGeneratorContext=function(){for(var r=this.context.length-1;r>=1;r--){var o=this.context[r];if(o.token==="function")return o.generator}return!1},qt.updateContext=function(r){var o,x=this.type;x.keyword&&r===f.dot?this.exprAllowed=!1:(o=x.updateContext)?o.call(this,r):this.exprAllowed=x.beforeExpr},qt.overrideContext=function(r){this.curContext()!==r&&(this.context[this.context.length-1]=r)},f.parenR.updateContext=f.braceR.updateContext=function(){if(this.context.length===1){this.exprAllowed=!0;return}var r=this.context.pop();r===be.b_stat&&this.curContext().token==="function"&&(r=this.context.pop()),this.exprAllowed=!r.isExpr},f.braceL.updateContext=function(r){this.context.push(this.braceIsBlock(r)?be.b_stat:be.b_expr),this.exprAllowed=!0},f.dollarBraceL.updateContext=function(){this.context.push(be.b_tmpl),this.exprAllowed=!0},f.parenL.updateContext=function(r){var o=r===f._if||r===f._for||r===f._with||r===f._while;this.context.push(o?be.p_stat:be.p_expr),this.exprAllowed=!0},f.incDec.updateContext=function(){},f._function.updateContext=f._class.updateContext=function(r){r.beforeExpr&&r!==f._else&&!(r===f.semi&&this.curContext()!==be.p_stat)&&!(r===f._return&&F.test(this.input.slice(this.lastTokEnd,this.start)))&&!((r===f.colon||r===f.braceL)&&this.curContext()===be.b_stat)?this.context.push(be.f_expr):this.context.push(be.f_stat),this.exprAllowed=!1},f.colon.updateContext=function(){this.curContext().token==="function"&&this.context.pop(),this.exprAllowed=!0},f.backQuote.updateContext=function(){this.curContext()===be.q_tmpl?this.context.pop():this.context.push(be.q_tmpl),this.exprAllowed=!1},f.star.updateContext=function(r){if(r===f._function){var o=this.context.length-1;this.context[o]===be.f_expr?this.context[o]=be.f_expr_gen:this.context[o]=be.f_gen}this.exprAllowed=!0},f.name.updateContext=function(r){var o=!1;this.options.ecmaVersion>=6&&r!==f.dot&&(this.value==="of"&&!this.exprAllowed||this.value==="yield"&&this.inGeneratorContext())&&(o=!0),this.exprAllowed=o};var oe=Se.prototype;oe.checkPropClash=function(r,o,x){if(!(this.options.ecmaVersion>=9&&r.type==="SpreadElement")&&!(this.options.ecmaVersion>=6&&(r.computed||r.method||r.shorthand))){var S=r.key,D;switch(S.type){case"Identifier":D=S.name;break;case"Literal":D=String(S.value);break;default:return}var G=r.kind;if(this.options.ecmaVersion>=6){D==="__proto__"&&G==="init"&&(o.proto&&(x?x.doubleProto<0&&(x.doubleProto=S.start):this.raiseRecoverable(S.start,"Redefinition of __proto__ property")),o.proto=!0);return}D="$"+D;var K=o[D];if(K){var H;G==="init"?H=this.strict&&K.init||K.get||K.set:H=K.init||K[G],H&&this.raiseRecoverable(S.start,"Redefinition of property")}else K=o[D]={init:!1,get:!1,set:!1};K[G]=!0}},oe.parseExpression=function(r,o){var x=this.start,S=this.startLoc,D=this.parseMaybeAssign(r,o);if(this.type===f.comma){var G=this.startNodeAt(x,S);for(G.expressions=[D];this.eat(f.comma);)G.expressions.push(this.parseMaybeAssign(r,o));return this.finishNode(G,"SequenceExpression")}return D},oe.parseMaybeAssign=function(r,o,x){if(this.isContextual("yield")){if(this.inGenerator)return this.parseYield(r);this.exprAllowed=!1}var S=!1,D=-1,G=-1,K=-1;o?(D=o.parenthesizedAssign,G=o.trailingComma,K=o.doubleProto,o.parenthesizedAssign=o.trailingComma=-1):(o=new Gs,S=!0);var H=this.start,Y=this.startLoc;(this.type===f.parenL||this.type===f.name)&&(this.potentialArrowAt=this.start,this.potentialArrowInForAwait=r==="await");var de=this.parseMaybeConditional(r,o);if(x&&(de=x.call(this,de,H,Y)),this.type.isAssign){var le=this.startNodeAt(H,Y);return le.operator=this.value,this.type===f.eq&&(de=this.toAssignable(de,!1,o)),S||(o.parenthesizedAssign=o.trailingComma=o.doubleProto=-1),o.shorthandAssign>=de.start&&(o.shorthandAssign=-1),this.type===f.eq?this.checkLValPattern(de):this.checkLValSimple(de),le.left=de,this.next(),le.right=this.parseMaybeAssign(r),K>-1&&(o.doubleProto=K),this.finishNode(le,"AssignmentExpression")}else S&&this.checkExpressionErrors(o,!0);return D>-1&&(o.parenthesizedAssign=D),G>-1&&(o.trailingComma=G),de},oe.parseMaybeConditional=function(r,o){var x=this.start,S=this.startLoc,D=this.parseExprOps(r,o);if(this.checkExpressionErrors(o))return D;if(this.eat(f.question)){var G=this.startNodeAt(x,S);return G.test=D,G.consequent=this.parseMaybeAssign(),this.expect(f.colon),G.alternate=this.parseMaybeAssign(r),this.finishNode(G,"ConditionalExpression")}return D},oe.parseExprOps=function(r,o){var x=this.start,S=this.startLoc,D=this.parseMaybeUnary(o,!1,!1,r);return this.checkExpressionErrors(o)||D.start===x&&D.type==="ArrowFunctionExpression"?D:this.parseExprOp(D,x,S,-1,r)},oe.parseExprOp=function(r,o,x,S,D){var G=this.type.binop;if(G!=null&&(!D||this.type!==f._in)&&G>S){var K=this.type===f.logicalOR||this.type===f.logicalAND,H=this.type===f.coalesce;H&&(G=f.logicalAND.binop);var Y=this.value;this.next();var de=this.start,le=this.startLoc,Me=this.parseExprOp(this.parseMaybeUnary(null,!1,!1,D),de,le,G,D),zt=this.buildBinary(o,x,r,Me,Y,K||H);return(K&&this.type===f.coalesce||H&&(this.type===f.logicalOR||this.type===f.logicalAND))&&this.raiseRecoverable(this.start,"Logical expressions and coalesce expressions cannot be mixed. Wrap either by parentheses"),this.parseExprOp(zt,o,x,S,D)}return r},oe.buildBinary=function(r,o,x,S,D,G){S.type==="PrivateIdentifier"&&this.raise(S.start,"Private identifier can only be left side of binary expression");var K=this.startNodeAt(r,o);return K.left=x,K.operator=D,K.right=S,this.finishNode(K,G?"LogicalExpression":"BinaryExpression")},oe.parseMaybeUnary=function(r,o,x,S){var D=this.start,G=this.startLoc,K;if(this.isContextual("await")&&this.canAwait)K=this.parseAwait(S),o=!0;else if(this.type.prefix){var H=this.startNode(),Y=this.type===f.incDec;H.operator=this.value,H.prefix=!0,this.next(),H.argument=this.parseMaybeUnary(null,!0,Y,S),this.checkExpressionErrors(r,!0),Y?this.checkLValSimple(H.argument):this.strict&&H.operator==="delete"&&ui(H.argument)?this.raiseRecoverable(H.start,"Deleting local variable in strict mode"):H.operator==="delete"&&Pn(H.argument)?this.raiseRecoverable(H.start,"Private fields can not be deleted"):o=!0,K=this.finishNode(H,Y?"UpdateExpression":"UnaryExpression")}else if(!o&&this.type===f.privateId)(S||this.privateNameStack.length===0)&&this.options.checkPrivateFields&&this.unexpected(),K=this.parsePrivateIdent(),this.type!==f._in&&this.unexpected();else{if(K=this.parseExprSubscripts(r,S),this.checkExpressionErrors(r))return K;for(;this.type.postfix&&!this.canInsertSemicolon();){var de=this.startNodeAt(D,G);de.operator=this.value,de.prefix=!1,de.argument=K,this.checkLValSimple(K),this.next(),K=this.finishNode(de,"UpdateExpression")}}if(!x&&this.eat(f.starstar))if(o)this.unexpected(this.lastTokStart);else return this.buildBinary(D,G,K,this.parseMaybeUnary(null,!1,!1,S),"**",!1);else return K};function ui(r){return r.type==="Identifier"||r.type==="ParenthesizedExpression"&&ui(r.expression)}function Pn(r){return r.type==="MemberExpression"&&r.property.type==="PrivateIdentifier"||r.type==="ChainExpression"&&Pn(r.expression)||r.type==="ParenthesizedExpression"&&Pn(r.expression)}oe.parseExprSubscripts=function(r,o){var x=this.start,S=this.startLoc,D=this.parseExprAtom(r,o);if(D.type==="ArrowFunctionExpression"&&this.input.slice(this.lastTokStart,this.lastTokEnd)!==")")return D;var G=this.parseSubscripts(D,x,S,!1,o);return r&&G.type==="MemberExpression"&&(r.parenthesizedAssign>=G.start&&(r.parenthesizedAssign=-1),r.parenthesizedBind>=G.start&&(r.parenthesizedBind=-1),r.trailingComma>=G.start&&(r.trailingComma=-1)),G},oe.parseSubscripts=function(r,o,x,S,D){for(var G=this.options.ecmaVersion>=8&&r.type==="Identifier"&&r.name==="async"&&this.lastTokEnd===r.end&&!this.canInsertSemicolon()&&r.end-r.start===5&&this.potentialArrowAt===r.start,K=!1;;){var H=this.parseSubscript(r,o,x,S,G,K,D);if(H.optional&&(K=!0),H===r||H.type==="ArrowFunctionExpression"){if(K){var Y=this.startNodeAt(o,x);Y.expression=H,H=this.finishNode(Y,"ChainExpression")}return H}r=H}},oe.shouldParseAsyncArrow=function(){return!this.canInsertSemicolon()&&this.eat(f.arrow)},oe.parseSubscriptAsyncArrow=function(r,o,x,S){return this.parseArrowExpression(this.startNodeAt(r,o),x,!0,S)},oe.parseSubscript=function(r,o,x,S,D,G,K){var H=this.options.ecmaVersion>=11,Y=H&&this.eat(f.questionDot);S&&Y&&this.raise(this.lastTokStart,"Optional chaining cannot appear in the callee of new expressions");var de=this.eat(f.bracketL);if(de||Y&&this.type!==f.parenL&&this.type!==f.backQuote||this.eat(f.dot)){var le=this.startNodeAt(o,x);le.object=r,de?(le.property=this.parseExpression(),this.expect(f.bracketR)):this.type===f.privateId&&r.type!=="Super"?le.property=this.parsePrivateIdent():le.property=this.parseIdent(this.options.allowReserved!=="never"),le.computed=!!de,H&&(le.optional=Y),r=this.finishNode(le,"MemberExpression")}else if(!S&&this.eat(f.parenL)){var Me=new Gs,zt=this.yieldPos,ys=this.awaitPos,Ht=this.awaitIdentPos;this.yieldPos=0,this.awaitPos=0,this.awaitIdentPos=0;var js=this.parseExprList(f.parenR,this.options.ecmaVersion>=8,!1,Me);if(D&&!Y&&this.shouldParseAsyncArrow())return this.checkPatternErrors(Me,!1),this.checkYieldAwaitInDefaultParams(),this.awaitIdentPos>0&&this.raise(this.awaitIdentPos,"Cannot use 'await' as identifier inside an async function"),this.yieldPos=zt,this.awaitPos=ys,this.awaitIdentPos=Ht,this.parseSubscriptAsyncArrow(o,x,js,K);this.checkExpressionErrors(Me,!0),this.yieldPos=zt||this.yieldPos,this.awaitPos=ys||this.awaitPos,this.awaitIdentPos=Ht||this.awaitIdentPos;var Wt=this.startNodeAt(o,x);Wt.callee=r,Wt.arguments=js,H&&(Wt.optional=Y),r=this.finishNode(Wt,"CallExpression")}else if(this.type===f.backQuote){(Y||G)&&this.raise(this.start,"Optional chaining cannot appear in the tag of tagged template expressions");var Xt=this.startNodeAt(o,x);Xt.tag=r,Xt.quasi=this.parseTemplate({isTagged:!0}),r=this.finishNode(Xt,"TaggedTemplateExpression")}return r},oe.parseExprAtom=function(r,o,x){this.type===f.slash&&this.readRegexp();var S,D=this.potentialArrowAt===this.start;switch(this.type){case f._super:return this.allowSuper||this.raise(this.start,"'super' keyword outside a method"),S=this.startNode(),this.next(),this.type===f.parenL&&!this.allowDirectSuper&&this.raise(S.start,"super() call outside constructor of a subclass"),this.type!==f.dot&&this.type!==f.bracketL&&this.type!==f.parenL&&this.unexpected(),this.finishNode(S,"Super");case f._this:return S=this.startNode(),this.next(),this.finishNode(S,"ThisExpression");case f.name:var G=this.start,K=this.startLoc,H=this.containsEsc,Y=this.parseIdent(!1);if(this.options.ecmaVersion>=8&&!H&&Y.name==="async"&&!this.canInsertSemicolon()&&this.eat(f._function))return this.overrideContext(be.f_expr),this.parseFunction(this.startNodeAt(G,K),0,!1,!0,o);if(D&&!this.canInsertSemicolon()){if(this.eat(f.arrow))return this.parseArrowExpression(this.startNodeAt(G,K),[Y],!1,o);if(this.options.ecmaVersion>=8&&Y.name==="async"&&this.type===f.name&&!H&&(!this.potentialArrowInForAwait||this.value!=="of"||this.containsEsc))return Y=this.parseIdent(!1),(this.canInsertSemicolon()||!this.eat(f.arrow))&&this.unexpected(),this.parseArrowExpression(this.startNodeAt(G,K),[Y],!0,o)}return Y;case f.regexp:var de=this.value;return S=this.parseLiteral(de.value),S.regex={pattern:de.pattern,flags:de.flags},S;case f.num:case f.string:return this.parseLiteral(this.value);case f._null:case f._true:case f._false:return S=this.startNode(),S.value=this.type===f._null?null:this.type===f._true,S.raw=this.type.keyword,this.next(),this.finishNode(S,"Literal");case f.parenL:var le=this.start,Me=this.parseParenAndDistinguishExpression(D,o);return r&&(r.parenthesizedAssign<0&&!this.isSimpleAssignTarget(Me)&&(r.parenthesizedAssign=le),r.parenthesizedBind<0&&(r.parenthesizedBind=le)),Me;case f.bracketL:return S=this.startNode(),this.next(),S.elements=this.parseExprList(f.bracketR,!0,!0,r),this.finishNode(S,"ArrayExpression");case f.braceL:return this.overrideContext(be.b_expr),this.parseObj(!1,r);case f._function:return S=this.startNode(),this.next(),this.parseFunction(S,0);case f._class:return this.parseClass(this.startNode(),!1);case f._new:return this.parseNew();case f.backQuote:return this.parseTemplate();case f._import:return this.options.ecmaVersion>=11?this.parseExprImport(x):this.unexpected();default:return this.parseExprAtomDefault()}},oe.parseExprAtomDefault=function(){this.unexpected()},oe.parseExprImport=function(r){var o=this.startNode();if(this.containsEsc&&this.raiseRecoverable(this.start,"Escape sequence in keyword import"),this.next(),this.type===f.parenL&&!r)return this.parseDynamicImport(o);if(this.type===f.dot){var x=this.startNodeAt(o.start,o.loc&&o.loc.start);return x.name="import",o.meta=this.finishNode(x,"Identifier"),this.parseImportMeta(o)}else this.unexpected()},oe.parseDynamicImport=function(r){if(this.next(),r.source=this.parseMaybeAssign(),this.options.ecmaVersion>=16)this.eat(f.parenR)?r.options=null:(this.expect(f.comma),this.afterTrailingComma(f.parenR)?r.options=null:(r.options=this.parseMaybeAssign(),this.eat(f.parenR)||(this.expect(f.comma),this.afterTrailingComma(f.parenR)||this.unexpected())));else if(!this.eat(f.parenR)){var o=this.start;this.eat(f.comma)&&this.eat(f.parenR)?this.raiseRecoverable(o,"Trailing comma is not allowed in import()"):this.unexpected(o)}return this.finishNode(r,"ImportExpression")},oe.parseImportMeta=function(r){this.next();var o=this.containsEsc;return r.property=this.parseIdent(!0),r.property.name!=="meta"&&this.raiseRecoverable(r.property.start,"The only valid meta property for import is 'import.meta'"),o&&this.raiseRecoverable(r.start,"'import.meta' must not contain escaped characters"),this.options.sourceType!=="module"&&!this.options.allowImportExportEverywhere&&this.raiseRecoverable(r.start,"Cannot use 'import.meta' outside a module"),this.finishNode(r,"MetaProperty")},oe.parseLiteral=function(r){var o=this.startNode();return o.value=r,o.raw=this.input.slice(this.start,this.end),o.raw.charCodeAt(o.raw.length-1)===110&&(o.bigint=o.raw.slice(0,-1).replace(/_/g,"")),this.next(),this.finishNode(o,"Literal")},oe.parseParenExpression=function(){this.expect(f.parenL);var r=this.parseExpression();return this.expect(f.parenR),r},oe.shouldParseArrow=function(r){return!this.canInsertSemicolon()},oe.parseParenAndDistinguishExpression=function(r,o){var x=this.start,S=this.startLoc,D,G=this.options.ecmaVersion>=8;if(this.options.ecmaVersion>=6){this.next();var K=this.start,H=this.startLoc,Y=[],de=!0,le=!1,Me=new Gs,zt=this.yieldPos,ys=this.awaitPos,Ht;for(this.yieldPos=0,this.awaitPos=0;this.type!==f.parenR;)if(de?de=!1:this.expect(f.comma),G&&this.afterTrailingComma(f.parenR,!0)){le=!0;break}else if(this.type===f.ellipsis){Ht=this.start,Y.push(this.parseParenItem(this.parseRestBinding())),this.type===f.comma&&this.raiseRecoverable(this.start,"Comma is not permitted after the rest element");break}else Y.push(this.parseMaybeAssign(!1,Me,this.parseParenItem));var js=this.lastTokEnd,Wt=this.lastTokEndLoc;if(this.expect(f.parenR),r&&this.shouldParseArrow(Y)&&this.eat(f.arrow))return this.checkPatternErrors(Me,!1),this.checkYieldAwaitInDefaultParams(),this.yieldPos=zt,this.awaitPos=ys,this.parseParenArrowList(x,S,Y,o);(!Y.length||le)&&this.unexpected(this.lastTokStart),Ht&&this.unexpected(Ht),this.checkExpressionErrors(Me,!0),this.yieldPos=zt||this.yieldPos,this.awaitPos=ys||this.awaitPos,Y.length>1?(D=this.startNodeAt(K,H),D.expressions=Y,this.finishNodeAt(D,"SequenceExpression",js,Wt)):D=Y[0]}else D=this.parseParenExpression();if(this.options.preserveParens){var Xt=this.startNodeAt(x,S);return Xt.expression=D,this.finishNode(Xt,"ParenthesizedExpression")}else return D},oe.parseParenItem=function(r){return r},oe.parseParenArrowList=function(r,o,x,S){return this.parseArrowExpression(this.startNodeAt(r,o),x,!1,S)};var Bo=[];oe.parseNew=function(){this.containsEsc&&this.raiseRecoverable(this.start,"Escape sequence in keyword new");var r=this.startNode();if(this.next(),this.options.ecmaVersion>=6&&this.type===f.dot){var o=this.startNodeAt(r.start,r.loc&&r.loc.start);o.name="new",r.meta=this.finishNode(o,"Identifier"),this.next();var x=this.containsEsc;return r.property=this.parseIdent(!0),r.property.name!=="target"&&this.raiseRecoverable(r.property.start,"The only valid meta property for new is 'new.target'"),x&&this.raiseRecoverable(r.start,"'new.target' must not contain escaped characters"),this.allowNewDotTarget||this.raiseRecoverable(r.start,"'new.target' can only be used in functions and class static block"),this.finishNode(r,"MetaProperty")}var S=this.start,D=this.startLoc;return r.callee=this.parseSubscripts(this.parseExprAtom(null,!1,!0),S,D,!0,!1),this.eat(f.parenL)?r.arguments=this.parseExprList(f.parenR,this.options.ecmaVersion>=8,!1):r.arguments=Bo,this.finishNode(r,"NewExpression")},oe.parseTemplateElement=function(r){var o=r.isTagged,x=this.startNode();return this.type===f.invalidTemplate?(o||this.raiseRecoverable(this.start,"Bad escape sequence in untagged template literal"),x.value={raw:this.value.replace(/\r\n?/g,`
`),cooked:null}):x.value={raw:this.input.slice(this.start,this.end).replace(/\r\n?/g,`
`),cooked:this.value},this.next(),x.tail=this.type===f.backQuote,this.finishNode(x,"TemplateElement")},oe.parseTemplate=function(r){r===void 0&&(r={});var o=r.isTagged;o===void 0&&(o=!1);var x=this.startNode();this.next(),x.expressions=[];var S=this.parseTemplateElement({isTagged:o});for(x.quasis=[S];!S.tail;)this.type===f.eof&&this.raise(this.pos,"Unterminated template literal"),this.expect(f.dollarBraceL),x.expressions.push(this.parseExpression()),this.expect(f.braceR),x.quasis.push(S=this.parseTemplateElement({isTagged:o}));return this.next(),this.finishNode(x,"TemplateLiteral")},oe.isAsyncProp=function(r){return!r.computed&&r.key.type==="Identifier"&&r.key.name==="async"&&(this.type===f.name||this.type===f.num||this.type===f.string||this.type===f.bracketL||this.type.keyword||this.options.ecmaVersion>=9&&this.type===f.star)&&!F.test(this.input.slice(this.lastTokEnd,this.start))},oe.parseObj=function(r,o){var x=this.startNode(),S=!0,D={};for(x.properties=[],this.next();!this.eat(f.braceR);){if(S)S=!1;else if(this.expect(f.comma),this.options.ecmaVersion>=5&&this.afterTrailingComma(f.braceR))break;var G=this.parseProperty(r,o);r||this.checkPropClash(G,D,o),x.properties.push(G)}return this.finishNode(x,r?"ObjectPattern":"ObjectExpression")},oe.parseProperty=function(r,o){var x=this.startNode(),S,D,G,K;if(this.options.ecmaVersion>=9&&this.eat(f.ellipsis))return r?(x.argument=this.parseIdent(!1),this.type===f.comma&&this.raiseRecoverable(this.start,"Comma is not permitted after the rest element"),this.finishNode(x,"RestElement")):(x.argument=this.parseMaybeAssign(!1,o),this.type===f.comma&&o&&o.trailingComma<0&&(o.trailingComma=this.start),this.finishNode(x,"SpreadElement"));this.options.ecmaVersion>=6&&(x.method=!1,x.shorthand=!1,(r||o)&&(G=this.start,K=this.startLoc),r||(S=this.eat(f.star)));var H=this.containsEsc;return this.parsePropertyName(x),!r&&!H&&this.options.ecmaVersion>=8&&!S&&this.isAsyncProp(x)?(D=!0,S=this.options.ecmaVersion>=9&&this.eat(f.star),this.parsePropertyName(x)):D=!1,this.parsePropertyValue(x,r,S,D,G,K,o,H),this.finishNode(x,"Property")},oe.parseGetterSetter=function(r){r.kind=r.key.name,this.parsePropertyName(r),r.value=this.parseMethod(!1);var o=r.kind==="get"?0:1;if(r.value.params.length!==o){var x=r.value.start;r.kind==="get"?this.raiseRecoverable(x,"getter should have no params"):this.raiseRecoverable(x,"setter should have exactly one param")}else r.kind==="set"&&r.value.params[0].type==="RestElement"&&this.raiseRecoverable(r.value.params[0].start,"Setter cannot use rest params")},oe.parsePropertyValue=function(r,o,x,S,D,G,K,H){(x||S)&&this.type===f.colon&&this.unexpected(),this.eat(f.colon)?(r.value=o?this.parseMaybeDefault(this.start,this.startLoc):this.parseMaybeAssign(!1,K),r.kind="init"):this.options.ecmaVersion>=6&&this.type===f.parenL?(o&&this.unexpected(),r.kind="init",r.method=!0,r.value=this.parseMethod(x,S)):!o&&!H&&this.options.ecmaVersion>=5&&!r.computed&&r.key.type==="Identifier"&&(r.key.name==="get"||r.key.name==="set")&&this.type!==f.comma&&this.type!==f.braceR&&this.type!==f.eq?((x||S)&&this.unexpected(),this.parseGetterSetter(r)):this.options.ecmaVersion>=6&&!r.computed&&r.key.type==="Identifier"?((x||S)&&this.unexpected(),this.checkUnreserved(r.key),r.key.name==="await"&&!this.awaitIdentPos&&(this.awaitIdentPos=D),r.kind="init",o?r.value=this.parseMaybeDefault(D,G,this.copyNode(r.key)):this.type===f.eq&&K?(K.shorthandAssign<0&&(K.shorthandAssign=this.start),r.value=this.parseMaybeDefault(D,G,this.copyNode(r.key))):r.value=this.copyNode(r.key),r.shorthand=!0):this.unexpected()},oe.parsePropertyName=function(r){if(this.options.ecmaVersion>=6){if(this.eat(f.bracketL))return r.computed=!0,r.key=this.parseMaybeAssign(),this.expect(f.bracketR),r.key;r.computed=!1}return r.key=this.type===f.num||this.type===f.string?this.parseExprAtom():this.parseIdent(this.options.allowReserved!=="never")},oe.initFunction=function(r){r.id=null,this.options.ecmaVersion>=6&&(r.generator=r.expression=!1),this.options.ecmaVersion>=8&&(r.async=!1)},oe.parseMethod=function(r,o,x){var S=this.startNode(),D=this.yieldPos,G=this.awaitPos,K=this.awaitIdentPos;return this.initFunction(S),this.options.ecmaVersion>=6&&(S.generator=r),this.options.ecmaVersion>=8&&(S.async=!!o),this.yieldPos=0,this.awaitPos=0,this.awaitIdentPos=0,this.enterScope(In(o,S.generator)|An|(x?ri:0)),this.expect(f.parenL),S.params=this.parseBindingList(f.parenR,!1,this.options.ecmaVersion>=8),this.checkYieldAwaitInDefaultParams(),this.parseFunctionBody(S,!1,!0,!1),this.yieldPos=D,this.awaitPos=G,this.awaitIdentPos=K,this.finishNode(S,"FunctionExpression")},oe.parseArrowExpression=function(r,o,x,S){var D=this.yieldPos,G=this.awaitPos,K=this.awaitIdentPos;return this.enterScope(In(x,!1)|ft),this.initFunction(r),this.options.ecmaVersion>=8&&(r.async=!!x),this.yieldPos=0,this.awaitPos=0,this.awaitIdentPos=0,r.params=this.toAssignableList(o,!0),this.parseFunctionBody(r,!0,!1,S),this.yieldPos=D,this.awaitPos=G,this.awaitIdentPos=K,this.finishNode(r,"ArrowFunctionExpression")},oe.parseFunctionBody=function(r,o,x,S){var D=o&&this.type!==f.braceL,G=this.strict,K=!1;if(D)r.body=this.parseMaybeAssign(S),r.expression=!0,this.checkParams(r,!1);else{var H=this.options.ecmaVersion>=7&&!this.isSimpleParamList(r.params);(!G||H)&&(K=this.strictDirective(this.end),K&&H&&this.raiseRecoverable(r.start,"Illegal 'use strict' directive in function with non-simple parameter list"));var Y=this.labels;this.labels=[],K&&(this.strict=!0),this.checkParams(r,!G&&!K&&!o&&!x&&this.isSimpleParamList(r.params)),this.strict&&r.id&&this.checkLValSimple(r.id,oi),r.body=this.parseBlock(!1,void 0,K&&!G),r.expression=!1,this.adaptDirectivePrologue(r.body.body),this.labels=Y}this.exitScope()},oe.isSimpleParamList=function(r){for(var o=0,x=r;o<x.length;o+=1)if(x[o].type!=="Identifier")return!1;return!0},oe.checkParams=function(r,o){for(var x=Object.create(null),S=0,D=r.params;S<D.length;S+=1){var G=D[S];this.checkLValInnerPattern(G,Mn,o?null:x)}},oe.parseExprList=function(r,o,x,S){for(var D=[],G=!0;!this.eat(r);){if(G)G=!1;else if(this.expect(f.comma),o&&this.afterTrailingComma(r))break;var K=void 0;x&&this.type===f.comma?K=null:this.type===f.ellipsis?(K=this.parseSpread(S),S&&this.type===f.comma&&S.trailingComma<0&&(S.trailingComma=this.start)):K=this.parseMaybeAssign(!1,S),D.push(K)}return D},oe.checkUnreserved=function(r){var o=r.start,x=r.end,S=r.name;this.inGenerator&&S==="yield"&&this.raiseRecoverable(o,"Cannot use 'yield' as identifier inside a generator"),this.inAsync&&S==="await"&&this.raiseRecoverable(o,"Cannot use 'await' as identifier inside an async function"),this.currentThisScope().inClassFieldInit&&S==="arguments"&&this.raiseRecoverable(o,"Cannot use 'arguments' in class field initializer"),this.inClassStaticBlock&&(S==="arguments"||S==="await")&&this.raise(o,"Cannot use "+S+" in class static initialization block"),this.keywords.test(S)&&this.raise(o,"Unexpected keyword '"+S+"'"),!(this.options.ecmaVersion<6&&this.input.slice(o,x).indexOf("\\")!==-1)&&(this.strict?this.reservedWordsStrict:this.reservedWords).test(S)&&(!this.inAsync&&S==="await"&&this.raiseRecoverable(o,"Cannot use keyword 'await' outside an async function"),this.raiseRecoverable(o,"The keyword '"+S+"' is reserved"))},oe.parseIdent=function(r){var o=this.parseIdentNode();return this.next(!!r),this.finishNode(o,"Identifier"),r||(this.checkUnreserved(o),o.name==="await"&&!this.awaitIdentPos&&(this.awaitIdentPos=o.start)),o},oe.parseIdentNode=function(){var r=this.startNode();return this.type===f.name?r.name=this.value:this.type.keyword?(r.name=this.type.keyword,(r.name==="class"||r.name==="function")&&(this.lastTokEnd!==this.lastTokStart+1||this.input.charCodeAt(this.lastTokStart)!==46)&&this.context.pop(),this.type=f.name):this.unexpected(),r},oe.parsePrivateIdent=function(){var r=this.startNode();return this.type===f.privateId?r.name=this.value:this.unexpected(),this.next(),this.finishNode(r,"PrivateIdentifier"),this.options.checkPrivateFields&&(this.privateNameStack.length===0?this.raise(r.start,"Private field '#"+r.name+"' must be declared in an enclosing class"):this.privateNameStack[this.privateNameStack.length-1].used.push(r)),r},oe.parseYield=function(r){this.yieldPos||(this.yieldPos=this.start);var o=this.startNode();return this.next(),this.type===f.semi||this.canInsertSemicolon()||this.type!==f.star&&!this.type.startsExpr?(o.delegate=!1,o.argument=null):(o.delegate=this.eat(f.star),o.argument=this.parseMaybeAssign(r)),this.finishNode(o,"YieldExpression")},oe.parseAwait=function(r){this.awaitPos||(this.awaitPos=this.start);var o=this.startNode();return this.next(),o.argument=this.parseMaybeUnary(null,!0,!1,r),this.finishNode(o,"AwaitExpression")};var Vs=Se.prototype;Vs.raise=function(r,o){var x=Ve(this.input,r);o+=" ("+x.line+":"+x.column+")";var S=new SyntaxError(o);throw S.pos=r,S.loc=x,S.raisedAt=this.pos,S},Vs.raiseRecoverable=Vs.raise,Vs.curPosition=function(){if(this.options.locations)return new ne(this.curLine,this.pos-this.lineStart)};var kt=Se.prototype,jo=function(o){this.flags=o,this.var=[],this.lexical=[],this.functions=[],this.inClassFieldInit=!1};kt.enterScope=function(r){this.scopeStack.push(new jo(r))},kt.exitScope=function(){this.scopeStack.pop()},kt.treatFunctionsAsVarInScope=function(r){return r.flags&ye||!this.inModule&&r.flags&re},kt.declareName=function(r,o,x){var S=!1;if(o===mt){var D=this.currentScope();S=D.lexical.indexOf(r)>-1||D.functions.indexOf(r)>-1||D.var.indexOf(r)>-1,D.lexical.push(r),this.inModule&&D.flags&re&&delete this.undefinedExports[r]}else if(o===ai)this.currentScope().lexical.push(r);else if(o===ii){var G=this.currentScope();this.treatFunctionsAsVar?S=G.lexical.indexOf(r)>-1:S=G.lexical.indexOf(r)>-1||G.var.indexOf(r)>-1,G.functions.push(r)}else for(var K=this.scopeStack.length-1;K>=0;--K){var H=this.scopeStack[K];if(H.lexical.indexOf(r)>-1&&!(H.flags&ni&&H.lexical[0]===r)||!this.treatFunctionsAsVarInScope(H)&&H.functions.indexOf(r)>-1){S=!0;break}if(H.var.push(r),this.inModule&&H.flags&re&&delete this.undefinedExports[r],H.flags&En)break}S&&this.raiseRecoverable(x,"Identifier '"+r+"' has already been declared")},kt.checkLocalExport=function(r){this.scopeStack[0].lexical.indexOf(r.name)===-1&&this.scopeStack[0].var.indexOf(r.name)===-1&&(this.undefinedExports[r.name]=r)},kt.currentScope=function(){return this.scopeStack[this.scopeStack.length-1]},kt.currentVarScope=function(){for(var r=this.scopeStack.length-1;;r--){var o=this.scopeStack[r];if(o.flags&En)return o}},kt.currentThisScope=function(){for(var r=this.scopeStack.length-1;;r--){var o=this.scopeStack[r];if(o.flags&En&&!(o.flags&ft))return o}};var ms=function(o,x,S){this.type="",this.start=x,this.end=0,o.options.locations&&(this.loc=new ce(o,S)),o.options.directSourceFile&&(this.sourceFile=o.options.directSourceFile),o.options.ranges&&(this.range=[x,0])},gs=Se.prototype;gs.startNode=function(){return new ms(this,this.start,this.startLoc)},gs.startNodeAt=function(r,o){return new ms(this,r,o)};function ci(r,o,x,S){return r.type=o,r.end=x,this.options.locations&&(r.loc.end=S),this.options.ranges&&(r.range[1]=x),r}gs.finishNode=function(r,o){return ci.call(this,r,o,this.lastTokEnd,this.lastTokEndLoc)},gs.finishNodeAt=function(r,o,x,S){return ci.call(this,r,o,x,S)},gs.copyNode=function(r){var o=new ms(this,r.start,this.startLoc);for(var x in r)o[x]=r[x];return o};var qo="Gara Garay Gukh Gurung_Khema Hrkt Katakana_Or_Hiragana Kawi Kirat_Rai Krai Nag_Mundari Nagm Ol_Onal Onao Sunu Sunuwar Todhri Todr Tulu_Tigalari Tutg Unknown Zzzz",hi="ASCII ASCII_Hex_Digit AHex Alphabetic Alpha Any Assigned Bidi_Control Bidi_C Bidi_Mirrored Bidi_M Case_Ignorable CI Cased Changes_When_Casefolded CWCF Changes_When_Casemapped CWCM Changes_When_Lowercased CWL Changes_When_NFKC_Casefolded CWKCF Changes_When_Titlecased CWT Changes_When_Uppercased CWU Dash Default_Ignorable_Code_Point DI Deprecated Dep Diacritic Dia Emoji Emoji_Component Emoji_Modifier Emoji_Modifier_Base Emoji_Presentation Extender Ext Grapheme_Base Gr_Base Grapheme_Extend Gr_Ext Hex_Digit Hex IDS_Binary_Operator IDSB IDS_Trinary_Operator IDST ID_Continue IDC ID_Start IDS Ideographic Ideo Join_Control Join_C Logical_Order_Exception LOE Lowercase Lower Math Noncharacter_Code_Point NChar Pattern_Syntax Pat_Syn Pattern_White_Space Pat_WS Quotation_Mark QMark Radical Regional_Indicator RI Sentence_Terminal STerm Soft_Dotted SD Terminal_Punctuation Term Unified_Ideograph UIdeo Uppercase Upper Variation_Selector VS White_Space space XID_Continue XIDC XID_Start XIDS",di=hi+" Extended_Pictographic",pi=di,fi=pi+" EBase EComp EMod EPres ExtPict",mi=fi,Ho={9:hi,10:di,11:pi,12:fi,13:mi,14:mi},Wo={9:"",10:"",11:"",12:"",13:"",14:"Basic_Emoji Emoji_Keycap_Sequence RGI_Emoji_Modifier_Sequence RGI_Emoji_Flag_Sequence RGI_Emoji_Tag_Sequence RGI_Emoji_ZWJ_Sequence RGI_Emoji"},gi="Cased_Letter LC Close_Punctuation Pe Connector_Punctuation Pc Control Cc cntrl Currency_Symbol Sc Dash_Punctuation Pd Decimal_Number Nd digit Enclosing_Mark Me Final_Punctuation Pf Format Cf Initial_Punctuation Pi Letter L Letter_Number Nl Line_Separator Zl Lowercase_Letter Ll Mark M Combining_Mark Math_Symbol Sm Modifier_Letter Lm Modifier_Symbol Sk Nonspacing_Mark Mn Number N Open_Punctuation Ps Other C Other_Letter Lo Other_Number No Other_Punctuation Po Other_Symbol So Paragraph_Separator Zp Private_Use Co Punctuation P punct Separator Z Space_Separator Zs Spacing_Mark Mc Surrogate Cs Symbol S Titlecase_Letter Lt Unassigned Cn Uppercase_Letter Lu",yi="Adlam Adlm Ahom Anatolian_Hieroglyphs Hluw Arabic Arab Armenian Armn Avestan Avst Balinese Bali Bamum Bamu Bassa_Vah Bass Batak Batk Bengali Beng Bhaiksuki Bhks Bopomofo Bopo Brahmi Brah Braille Brai Buginese Bugi Buhid Buhd Canadian_Aboriginal Cans Carian Cari Caucasian_Albanian Aghb Chakma Cakm Cham Cham Cherokee Cher Common Zyyy Coptic Copt Qaac Cuneiform Xsux Cypriot Cprt Cyrillic Cyrl Deseret Dsrt Devanagari Deva Duployan Dupl Egyptian_Hieroglyphs Egyp Elbasan Elba Ethiopic Ethi Georgian Geor Glagolitic Glag Gothic Goth Grantha Gran Greek Grek Gujarati Gujr Gurmukhi Guru Han Hani Hangul Hang Hanunoo Hano Hatran Hatr Hebrew Hebr Hiragana Hira Imperial_Aramaic Armi Inherited Zinh Qaai Inscriptional_Pahlavi Phli Inscriptional_Parthian Prti Javanese Java Kaithi Kthi Kannada Knda Katakana Kana Kayah_Li Kali Kharoshthi Khar Khmer Khmr Khojki Khoj Khudawadi Sind Lao Laoo Latin Latn Lepcha Lepc Limbu Limb Linear_A Lina Linear_B Linb Lisu Lisu Lycian Lyci Lydian Lydi Mahajani Mahj Malayalam Mlym Mandaic Mand Manichaean Mani Marchen Marc Masaram_Gondi Gonm Meetei_Mayek Mtei Mende_Kikakui Mend Meroitic_Cursive Merc Meroitic_Hieroglyphs Mero Miao Plrd Modi Mongolian Mong Mro Mroo Multani Mult Myanmar Mymr Nabataean Nbat New_Tai_Lue Talu Newa Newa Nko Nkoo Nushu Nshu Ogham Ogam Ol_Chiki Olck Old_Hungarian Hung Old_Italic Ital Old_North_Arabian Narb Old_Permic Perm Old_Persian Xpeo Old_South_Arabian Sarb Old_Turkic Orkh Oriya Orya Osage Osge Osmanya Osma Pahawh_Hmong Hmng Palmyrene Palm Pau_Cin_Hau Pauc Phags_Pa Phag Phoenician Phnx Psalter_Pahlavi Phlp Rejang Rjng Runic Runr Samaritan Samr Saurashtra Saur Sharada Shrd Shavian Shaw Siddham Sidd SignWriting Sgnw Sinhala Sinh Sora_Sompeng Sora Soyombo Soyo Sundanese Sund Syloti_Nagri Sylo Syriac Syrc Tagalog Tglg Tagbanwa Tagb Tai_Le Tale Tai_Tham Lana Tai_Viet Tavt Takri Takr Tamil Taml Tangut Tang Telugu Telu Thaana Thaa Thai Thai Tibetan Tibt Tifinagh Tfng Tirhuta Tirh Ugaritic Ugar Vai Vaii Warang_Citi Wara Yi Yiii Zanabazar_Square Zanb",xi=yi+" Dogra Dogr Gunjala_Gondi Gong Hanifi_Rohingya Rohg Makasar Maka Medefaidrin Medf Old_Sogdian Sogo Sogdian Sogd",bi=xi+" Elymaic Elym Nandinagari Nand Nyiakeng_Puachue_Hmong Hmnp Wancho Wcho",vi=bi+" Chorasmian Chrs Diak Dives_Akuru Khitan_Small_Script Kits Yezi Yezidi",wi=vi+" Cypro_Minoan Cpmn Old_Uyghur Ougr Tangsa Tnsa Toto Vithkuqi Vith",Xo={9:yi,10:xi,11:bi,12:vi,13:wi,14:wi+" "+qo},ki={};function Yo(r){var o=ki[r]={binary:ae(Ho[r]+" "+gi),binaryOfStrings:ae(Wo[r]),nonBinary:{General_Category:ae(gi),Script:ae(Xo[r])}};o.nonBinary.Script_Extensions=o.nonBinary.Script,o.nonBinary.gc=o.nonBinary.General_Category,o.nonBinary.sc=o.nonBinary.Script,o.nonBinary.scx=o.nonBinary.Script_Extensions}for(var zn=0,Ti=[9,10,11,12,13,14];zn<Ti.length;zn+=1){var Jo=Ti[zn];Yo(Jo)}var Z=Se.prototype,Ks=function(o,x){this.parent=o,this.base=x||this};Ks.prototype.separatedFrom=function(o){for(var x=this;x;x=x.parent)for(var S=o;S;S=S.parent)if(x.base===S.base&&x!==S)return!0;return!1},Ks.prototype.sibling=function(){return new Ks(this.parent,this.base)};var at=function(o){this.parser=o,this.validFlags="gim"+(o.options.ecmaVersion>=6?"uy":"")+(o.options.ecmaVersion>=9?"s":"")+(o.options.ecmaVersion>=13?"d":"")+(o.options.ecmaVersion>=15?"v":""),this.unicodeProperties=ki[o.options.ecmaVersion>=14?14:o.options.ecmaVersion],this.source="",this.flags="",this.start=0,this.switchU=!1,this.switchV=!1,this.switchN=!1,this.pos=0,this.lastIntValue=0,this.lastStringValue="",this.lastAssertionIsQuantifiable=!1,this.numCapturingParens=0,this.maxBackReference=0,this.groupNames=Object.create(null),this.backReferenceNames=[],this.branchID=null};at.prototype.reset=function(o,x,S){var D=S.indexOf("v")!==-1,G=S.indexOf("u")!==-1;this.start=o|0,this.source=x+"",this.flags=S,D&&this.parser.options.ecmaVersion>=15?(this.switchU=!0,this.switchV=!0,this.switchN=!0):(this.switchU=G&&this.parser.options.ecmaVersion>=6,this.switchV=!1,this.switchN=G&&this.parser.options.ecmaVersion>=9)},at.prototype.raise=function(o){this.parser.raiseRecoverable(this.start,"Invalid regular expression: /"+this.source+"/: "+o)},at.prototype.at=function(o,x){x===void 0&&(x=!1);var S=this.source,D=S.length;if(o>=D)return-1;var G=S.charCodeAt(o);if(!(x||this.switchU)||G<=55295||G>=57344||o+1>=D)return G;var K=S.charCodeAt(o+1);return K>=56320&&K<=57343?(G<<10)+K-56613888:G},at.prototype.nextIndex=function(o,x){x===void 0&&(x=!1);var S=this.source,D=S.length;if(o>=D)return D;var G=S.charCodeAt(o),K;return!(x||this.switchU)||G<=55295||G>=57344||o+1>=D||(K=S.charCodeAt(o+1))<56320||K>57343?o+1:o+2},at.prototype.current=function(o){return o===void 0&&(o=!1),this.at(this.pos,o)},at.prototype.lookahead=function(o){return o===void 0&&(o=!1),this.at(this.nextIndex(this.pos,o),o)},at.prototype.advance=function(o){o===void 0&&(o=!1),this.pos=this.nextIndex(this.pos,o)},at.prototype.eat=function(o,x){return x===void 0&&(x=!1),this.current(x)===o?(this.advance(x),!0):!1},at.prototype.eatChars=function(o,x){x===void 0&&(x=!1);for(var S=this.pos,D=0,G=o;D<G.length;D+=1){var K=G[D],H=this.at(S,x);if(H===-1||H!==K)return!1;S=this.nextIndex(S,x)}return this.pos=S,!0},Z.validateRegExpFlags=function(r){for(var o=r.validFlags,x=r.flags,S=!1,D=!1,G=0;G<x.length;G++){var K=x.charAt(G);o.indexOf(K)===-1&&this.raise(r.start,"Invalid regular expression flag"),x.indexOf(K,G+1)>-1&&this.raise(r.start,"Duplicate regular expression flag"),K==="u"&&(S=!0),K==="v"&&(D=!0)}this.options.ecmaVersion>=15&&S&&D&&this.raise(r.start,"Invalid regular expression flag")};function Zo(r){for(var o in r)return!0;return!1}Z.validateRegExpPattern=function(r){this.regexp_pattern(r),!r.switchN&&this.options.ecmaVersion>=9&&Zo(r.groupNames)&&(r.switchN=!0,this.regexp_pattern(r))},Z.regexp_pattern=function(r){r.pos=0,r.lastIntValue=0,r.lastStringValue="",r.lastAssertionIsQuantifiable=!1,r.numCapturingParens=0,r.maxBackReference=0,r.groupNames=Object.create(null),r.backReferenceNames.length=0,r.branchID=null,this.regexp_disjunction(r),r.pos!==r.source.length&&(r.eat(41)&&r.raise("Unmatched ')'"),(r.eat(93)||r.eat(125))&&r.raise("Lone quantifier brackets")),r.maxBackReference>r.numCapturingParens&&r.raise("Invalid escape");for(var o=0,x=r.backReferenceNames;o<x.length;o+=1){var S=x[o];r.groupNames[S]||r.raise("Invalid named capture referenced")}},Z.regexp_disjunction=function(r){var o=this.options.ecmaVersion>=16;for(o&&(r.branchID=new Ks(r.branchID,null)),this.regexp_alternative(r);r.eat(124);)o&&(r.branchID=r.branchID.sibling()),this.regexp_alternative(r);o&&(r.branchID=r.branchID.parent),this.regexp_eatQuantifier(r,!0)&&r.raise("Nothing to repeat"),r.eat(123)&&r.raise("Lone quantifier brackets")},Z.regexp_alternative=function(r){for(;r.pos<r.source.length&&this.regexp_eatTerm(r););},Z.regexp_eatTerm=function(r){return this.regexp_eatAssertion(r)?(r.lastAssertionIsQuantifiable&&this.regexp_eatQuantifier(r)&&r.switchU&&r.raise("Invalid quantifier"),!0):(r.switchU?this.regexp_eatAtom(r):this.regexp_eatExtendedAtom(r))?(this.regexp_eatQuantifier(r),!0):!1},Z.regexp_eatAssertion=function(r){var o=r.pos;if(r.lastAssertionIsQuantifiable=!1,r.eat(94)||r.eat(36))return!0;if(r.eat(92)){if(r.eat(66)||r.eat(98))return!0;r.pos=o}if(r.eat(40)&&r.eat(63)){var x=!1;if(this.options.ecmaVersion>=9&&(x=r.eat(60)),r.eat(61)||r.eat(33))return this.regexp_disjunction(r),r.eat(41)||r.raise("Unterminated group"),r.lastAssertionIsQuantifiable=!x,!0}return r.pos=o,!1},Z.regexp_eatQuantifier=function(r,o){return o===void 0&&(o=!1),this.regexp_eatQuantifierPrefix(r,o)?(r.eat(63),!0):!1},Z.regexp_eatQuantifierPrefix=function(r,o){return r.eat(42)||r.eat(43)||r.eat(63)||this.regexp_eatBracedQuantifier(r,o)},Z.regexp_eatBracedQuantifier=function(r,o){var x=r.pos;if(r.eat(123)){var S=0,D=-1;if(this.regexp_eatDecimalDigits(r)&&(S=r.lastIntValue,r.eat(44)&&this.regexp_eatDecimalDigits(r)&&(D=r.lastIntValue),r.eat(125)))return D!==-1&&D<S&&!o&&r.raise("numbers out of order in {} quantifier"),!0;r.switchU&&!o&&r.raise("Incomplete quantifier"),r.pos=x}return!1},Z.regexp_eatAtom=function(r){return this.regexp_eatPatternCharacters(r)||r.eat(46)||this.regexp_eatReverseSolidusAtomEscape(r)||this.regexp_eatCharacterClass(r)||this.regexp_eatUncapturingGroup(r)||this.regexp_eatCapturingGroup(r)},Z.regexp_eatReverseSolidusAtomEscape=function(r){var o=r.pos;if(r.eat(92)){if(this.regexp_eatAtomEscape(r))return!0;r.pos=o}return!1},Z.regexp_eatUncapturingGroup=function(r){var o=r.pos;if(r.eat(40)){if(r.eat(63)){if(this.options.ecmaVersion>=16){var x=this.regexp_eatModifiers(r),S=r.eat(45);if(x||S){for(var D=0;D<x.length;D++){var G=x.charAt(D);x.indexOf(G,D+1)>-1&&r.raise("Duplicate regular expression modifiers")}if(S){var K=this.regexp_eatModifiers(r);!x&&!K&&r.current()===58&&r.raise("Invalid regular expression modifiers");for(var H=0;H<K.length;H++){var Y=K.charAt(H);(K.indexOf(Y,H+1)>-1||x.indexOf(Y)>-1)&&r.raise("Duplicate regular expression modifiers")}}}}if(r.eat(58)){if(this.regexp_disjunction(r),r.eat(41))return!0;r.raise("Unterminated group")}}r.pos=o}return!1},Z.regexp_eatCapturingGroup=function(r){if(r.eat(40)){if(this.options.ecmaVersion>=9?this.regexp_groupSpecifier(r):r.current()===63&&r.raise("Invalid group"),this.regexp_disjunction(r),r.eat(41))return r.numCapturingParens+=1,!0;r.raise("Unterminated group")}return!1},Z.regexp_eatModifiers=function(r){for(var o="",x=0;(x=r.current())!==-1&&Qo(x);)o+=he(x),r.advance();return o};function Qo(r){return r===105||r===109||r===115}Z.regexp_eatExtendedAtom=function(r){return r.eat(46)||this.regexp_eatReverseSolidusAtomEscape(r)||this.regexp_eatCharacterClass(r)||this.regexp_eatUncapturingGroup(r)||this.regexp_eatCapturingGroup(r)||this.regexp_eatInvalidBracedQuantifier(r)||this.regexp_eatExtendedPatternCharacter(r)},Z.regexp_eatInvalidBracedQuantifier=function(r){return this.regexp_eatBracedQuantifier(r,!0)&&r.raise("Nothing to repeat"),!1},Z.regexp_eatSyntaxCharacter=function(r){var o=r.current();return Si(o)?(r.lastIntValue=o,r.advance(),!0):!1};function Si(r){return r===36||r>=40&&r<=43||r===46||r===63||r>=91&&r<=94||r>=123&&r<=125}Z.regexp_eatPatternCharacters=function(r){for(var o=r.pos,x=0;(x=r.current())!==-1&&!Si(x);)r.advance();return r.pos!==o},Z.regexp_eatExtendedPatternCharacter=function(r){var o=r.current();return o!==-1&&o!==36&&!(o>=40&&o<=43)&&o!==46&&o!==63&&o!==91&&o!==94&&o!==124?(r.advance(),!0):!1},Z.regexp_groupSpecifier=function(r){if(r.eat(63)){this.regexp_eatGroupName(r)||r.raise("Invalid group");var o=this.options.ecmaVersion>=16,x=r.groupNames[r.lastStringValue];if(x)if(o)for(var S=0,D=x;S<D.length;S+=1)D[S].separatedFrom(r.branchID)||r.raise("Duplicate capture group name");else r.raise("Duplicate capture group name");o?(x||(r.groupNames[r.lastStringValue]=[])).push(r.branchID):r.groupNames[r.lastStringValue]=!0}},Z.regexp_eatGroupName=function(r){if(r.lastStringValue="",r.eat(60)){if(this.regexp_eatRegExpIdentifierName(r)&&r.eat(62))return!0;r.raise("Invalid capture group name")}return!1},Z.regexp_eatRegExpIdentifierName=function(r){if(r.lastStringValue="",this.regexp_eatRegExpIdentifierStart(r)){for(r.lastStringValue+=he(r.lastIntValue);this.regexp_eatRegExpIdentifierPart(r);)r.lastStringValue+=he(r.lastIntValue);return!0}return!1},Z.regexp_eatRegExpIdentifierStart=function(r){var o=r.pos,x=this.options.ecmaVersion>=11,S=r.current(x);return r.advance(x),S===92&&this.regexp_eatRegExpUnicodeEscapeSequence(r,x)&&(S=r.lastIntValue),el(S)?(r.lastIntValue=S,!0):(r.pos=o,!1)};function el(r){return y(r,!0)||r===36||r===95}Z.regexp_eatRegExpIdentifierPart=function(r){var o=r.pos,x=this.options.ecmaVersion>=11,S=r.current(x);return r.advance(x),S===92&&this.regexp_eatRegExpUnicodeEscapeSequence(r,x)&&(S=r.lastIntValue),tl(S)?(r.lastIntValue=S,!0):(r.pos=o,!1)};function tl(r){return k(r,!0)||r===36||r===95||r===8204||r===8205}Z.regexp_eatAtomEscape=function(r){return this.regexp_eatBackReference(r)||this.regexp_eatCharacterClassEscape(r)||this.regexp_eatCharacterEscape(r)||r.switchN&&this.regexp_eatKGroupName(r)?!0:(r.switchU&&(r.current()===99&&r.raise("Invalid unicode escape"),r.raise("Invalid escape")),!1)},Z.regexp_eatBackReference=function(r){var o=r.pos;if(this.regexp_eatDecimalEscape(r)){var x=r.lastIntValue;if(r.switchU)return x>r.maxBackReference&&(r.maxBackReference=x),!0;if(x<=r.numCapturingParens)return!0;r.pos=o}return!1},Z.regexp_eatKGroupName=function(r){if(r.eat(107)){if(this.regexp_eatGroupName(r))return r.backReferenceNames.push(r.lastStringValue),!0;r.raise("Invalid named reference")}return!1},Z.regexp_eatCharacterEscape=function(r){return this.regexp_eatControlEscape(r)||this.regexp_eatCControlLetter(r)||this.regexp_eatZero(r)||this.regexp_eatHexEscapeSequence(r)||this.regexp_eatRegExpUnicodeEscapeSequence(r,!1)||!r.switchU&&this.regexp_eatLegacyOctalEscapeSequence(r)||this.regexp_eatIdentityEscape(r)},Z.regexp_eatCControlLetter=function(r){var o=r.pos;if(r.eat(99)){if(this.regexp_eatControlLetter(r))return!0;r.pos=o}return!1},Z.regexp_eatZero=function(r){return r.current()===48&&!Ns(r.lookahead())?(r.lastIntValue=0,r.advance(),!0):!1},Z.regexp_eatControlEscape=function(r){var o=r.current();return o===116?(r.lastIntValue=9,r.advance(),!0):o===110?(r.lastIntValue=10,r.advance(),!0):o===118?(r.lastIntValue=11,r.advance(),!0):o===102?(r.lastIntValue=12,r.advance(),!0):o===114?(r.lastIntValue=13,r.advance(),!0):!1},Z.regexp_eatControlLetter=function(r){var o=r.current();return _i(o)?(r.lastIntValue=o%32,r.advance(),!0):!1};function _i(r){return r>=65&&r<=90||r>=97&&r<=122}Z.regexp_eatRegExpUnicodeEscapeSequence=function(r,o){o===void 0&&(o=!1);var x=r.pos,S=o||r.switchU;if(r.eat(117)){if(this.regexp_eatFixedHexDigits(r,4)){var D=r.lastIntValue;if(S&&D>=55296&&D<=56319){var G=r.pos;if(r.eat(92)&&r.eat(117)&&this.regexp_eatFixedHexDigits(r,4)){var K=r.lastIntValue;if(K>=56320&&K<=57343)return r.lastIntValue=(D-55296)*1024+(K-56320)+65536,!0}r.pos=G,r.lastIntValue=D}return!0}if(S&&r.eat(123)&&this.regexp_eatHexDigits(r)&&r.eat(125)&&sl(r.lastIntValue))return!0;S&&r.raise("Invalid unicode escape"),r.pos=x}return!1};function sl(r){return r>=0&&r<=1114111}Z.regexp_eatIdentityEscape=function(r){if(r.switchU)return this.regexp_eatSyntaxCharacter(r)?!0:r.eat(47)?(r.lastIntValue=47,!0):!1;var o=r.current();return o!==99&&(!r.switchN||o!==107)?(r.lastIntValue=o,r.advance(),!0):!1},Z.regexp_eatDecimalEscape=function(r){r.lastIntValue=0;var o=r.current();if(o>=49&&o<=57){do r.lastIntValue=10*r.lastIntValue+(o-48),r.advance();while((o=r.current())>=48&&o<=57);return!0}return!1};var Ci=0,gt=1,qe=2;Z.regexp_eatCharacterClassEscape=function(r){var o=r.current();if(nl(o))return r.lastIntValue=-1,r.advance(),gt;var x=!1;if(r.switchU&&this.options.ecmaVersion>=9&&((x=o===80)||o===112)){r.lastIntValue=-1,r.advance();var S;if(r.eat(123)&&(S=this.regexp_eatUnicodePropertyValueExpression(r))&&r.eat(125))return x&&S===qe&&r.raise("Invalid property name"),S;r.raise("Invalid property name")}return Ci};function nl(r){return r===100||r===68||r===115||r===83||r===119||r===87}Z.regexp_eatUnicodePropertyValueExpression=function(r){var o=r.pos;if(this.regexp_eatUnicodePropertyName(r)&&r.eat(61)){var x=r.lastStringValue;if(this.regexp_eatUnicodePropertyValue(r)){var S=r.lastStringValue;return this.regexp_validateUnicodePropertyNameAndValue(r,x,S),gt}}if(r.pos=o,this.regexp_eatLoneUnicodePropertyNameOrValue(r)){var D=r.lastStringValue;return this.regexp_validateUnicodePropertyNameOrValue(r,D)}return Ci},Z.regexp_validateUnicodePropertyNameAndValue=function(r,o,x){ie(r.unicodeProperties.nonBinary,o)||r.raise("Invalid property name"),r.unicodeProperties.nonBinary[o].test(x)||r.raise("Invalid property value")},Z.regexp_validateUnicodePropertyNameOrValue=function(r,o){if(r.unicodeProperties.binary.test(o))return gt;if(r.switchV&&r.unicodeProperties.binaryOfStrings.test(o))return qe;r.raise("Invalid property name")},Z.regexp_eatUnicodePropertyName=function(r){var o=0;for(r.lastStringValue="";Ai(o=r.current());)r.lastStringValue+=he(o),r.advance();return r.lastStringValue!==""};function Ai(r){return _i(r)||r===95}Z.regexp_eatUnicodePropertyValue=function(r){var o=0;for(r.lastStringValue="";rl(o=r.current());)r.lastStringValue+=he(o),r.advance();return r.lastStringValue!==""};function rl(r){return Ai(r)||Ns(r)}Z.regexp_eatLoneUnicodePropertyNameOrValue=function(r){return this.regexp_eatUnicodePropertyValue(r)},Z.regexp_eatCharacterClass=function(r){if(r.eat(91)){var o=r.eat(94),x=this.regexp_classContents(r);return r.eat(93)||r.raise("Unterminated character class"),o&&x===qe&&r.raise("Negated character class may contain strings"),!0}return!1},Z.regexp_classContents=function(r){return r.current()===93?gt:r.switchV?this.regexp_classSetExpression(r):(this.regexp_nonEmptyClassRanges(r),gt)},Z.regexp_nonEmptyClassRanges=function(r){for(;this.regexp_eatClassAtom(r);){var o=r.lastIntValue;if(r.eat(45)&&this.regexp_eatClassAtom(r)){var x=r.lastIntValue;r.switchU&&(o===-1||x===-1)&&r.raise("Invalid character class"),o!==-1&&x!==-1&&o>x&&r.raise("Range out of order in character class")}}},Z.regexp_eatClassAtom=function(r){var o=r.pos;if(r.eat(92)){if(this.regexp_eatClassEscape(r))return!0;if(r.switchU){var x=r.current();(x===99||Mi(x))&&r.raise("Invalid class escape"),r.raise("Invalid escape")}r.pos=o}var S=r.current();return S!==93?(r.lastIntValue=S,r.advance(),!0):!1},Z.regexp_eatClassEscape=function(r){var o=r.pos;if(r.eat(98))return r.lastIntValue=8,!0;if(r.switchU&&r.eat(45))return r.lastIntValue=45,!0;if(!r.switchU&&r.eat(99)){if(this.regexp_eatClassControlLetter(r))return!0;r.pos=o}return this.regexp_eatCharacterClassEscape(r)||this.regexp_eatCharacterEscape(r)},Z.regexp_classSetExpression=function(r){var o=gt,x;if(!this.regexp_eatClassSetRange(r))if(x=this.regexp_eatClassSetOperand(r)){x===qe&&(o=qe);for(var S=r.pos;r.eatChars([38,38]);){if(r.current()!==38&&(x=this.regexp_eatClassSetOperand(r))){x!==qe&&(o=gt);continue}r.raise("Invalid character in character class")}if(S!==r.pos)return o;for(;r.eatChars([45,45]);)this.regexp_eatClassSetOperand(r)||r.raise("Invalid character in character class");if(S!==r.pos)return o}else r.raise("Invalid character in character class");for(;;)if(!this.regexp_eatClassSetRange(r)){if(x=this.regexp_eatClassSetOperand(r),!x)return o;x===qe&&(o=qe)}},Z.regexp_eatClassSetRange=function(r){var o=r.pos;if(this.regexp_eatClassSetCharacter(r)){var x=r.lastIntValue;if(r.eat(45)&&this.regexp_eatClassSetCharacter(r)){var S=r.lastIntValue;return x!==-1&&S!==-1&&x>S&&r.raise("Range out of order in character class"),!0}r.pos=o}return!1},Z.regexp_eatClassSetOperand=function(r){return this.regexp_eatClassSetCharacter(r)?gt:this.regexp_eatClassStringDisjunction(r)||this.regexp_eatNestedClass(r)},Z.regexp_eatNestedClass=function(r){var o=r.pos;if(r.eat(91)){var x=r.eat(94),S=this.regexp_classContents(r);if(r.eat(93))return x&&S===qe&&r.raise("Negated character class may contain strings"),S;r.pos=o}if(r.eat(92)){var D=this.regexp_eatCharacterClassEscape(r);if(D)return D;r.pos=o}return null},Z.regexp_eatClassStringDisjunction=function(r){var o=r.pos;if(r.eatChars([92,113])){if(r.eat(123)){var x=this.regexp_classStringDisjunctionContents(r);if(r.eat(125))return x}else r.raise("Invalid escape");r.pos=o}return null},Z.regexp_classStringDisjunctionContents=function(r){for(var o=this.regexp_classString(r);r.eat(124);)this.regexp_classString(r)===qe&&(o=qe);return o},Z.regexp_classString=function(r){for(var o=0;this.regexp_eatClassSetCharacter(r);)o++;return o===1?gt:qe},Z.regexp_eatClassSetCharacter=function(r){var o=r.pos;if(r.eat(92))return this.regexp_eatCharacterEscape(r)||this.regexp_eatClassSetReservedPunctuator(r)?!0:r.eat(98)?(r.lastIntValue=8,!0):(r.pos=o,!1);var x=r.current();return x<0||x===r.lookahead()&&il(x)||al(x)?!1:(r.advance(),r.lastIntValue=x,!0)};function il(r){return r===33||r>=35&&r<=38||r>=42&&r<=44||r===46||r>=58&&r<=64||r===94||r===96||r===126}function al(r){return r===40||r===41||r===45||r===47||r>=91&&r<=93||r>=123&&r<=125}Z.regexp_eatClassSetReservedPunctuator=function(r){var o=r.current();return ol(o)?(r.lastIntValue=o,r.advance(),!0):!1};function ol(r){return r===33||r===35||r===37||r===38||r===44||r===45||r>=58&&r<=62||r===64||r===96||r===126}Z.regexp_eatClassControlLetter=function(r){var o=r.current();return Ns(o)||o===95?(r.lastIntValue=o%32,r.advance(),!0):!1},Z.regexp_eatHexEscapeSequence=function(r){var o=r.pos;if(r.eat(120)){if(this.regexp_eatFixedHexDigits(r,2))return!0;r.switchU&&r.raise("Invalid escape"),r.pos=o}return!1},Z.regexp_eatDecimalDigits=function(r){var o=r.pos,x=0;for(r.lastIntValue=0;Ns(x=r.current());)r.lastIntValue=10*r.lastIntValue+(x-48),r.advance();return r.pos!==o};function Ns(r){return r>=48&&r<=57}Z.regexp_eatHexDigits=function(r){var o=r.pos,x=0;for(r.lastIntValue=0;Ei(x=r.current());)r.lastIntValue=16*r.lastIntValue+Ii(x),r.advance();return r.pos!==o};function Ei(r){return r>=48&&r<=57||r>=65&&r<=70||r>=97&&r<=102}function Ii(r){return r>=65&&r<=70?10+(r-65):r>=97&&r<=102?10+(r-97):r-48}Z.regexp_eatLegacyOctalEscapeSequence=function(r){if(this.regexp_eatOctalDigit(r)){var o=r.lastIntValue;if(this.regexp_eatOctalDigit(r)){var x=r.lastIntValue;o<=3&&this.regexp_eatOctalDigit(r)?r.lastIntValue=o*64+x*8+r.lastIntValue:r.lastIntValue=o*8+x}else r.lastIntValue=o;return!0}return!1},Z.regexp_eatOctalDigit=function(r){var o=r.current();return Mi(o)?(r.lastIntValue=o-48,r.advance(),!0):(r.lastIntValue=0,!1)};function Mi(r){return r>=48&&r<=55}Z.regexp_eatFixedHexDigits=function(r,o){var x=r.pos;r.lastIntValue=0;for(var S=0;S<o;++S){var D=r.current();if(!Ei(D))return r.pos=x,!1;r.lastIntValue=16*r.lastIntValue+Ii(D),r.advance()}return!0};var Bs=function(o){this.type=o.type,this.value=o.value,this.start=o.start,this.end=o.end,o.options.locations&&(this.loc=new ce(o,o.startLoc,o.endLoc)),o.options.ranges&&(this.range=[o.start,o.end])},pe=Se.prototype;pe.next=function(r){!r&&this.type.keyword&&this.containsEsc&&this.raiseRecoverable(this.start,"Escape sequence in keyword "+this.type.keyword),this.options.onToken&&this.options.onToken(new Bs(this)),this.lastTokEnd=this.end,this.lastTokStart=this.start,this.lastTokEndLoc=this.endLoc,this.lastTokStartLoc=this.startLoc,this.nextToken()},pe.getToken=function(){return this.next(),new Bs(this)},typeof Symbol<"u"&&(pe[Symbol.iterator]=function(){var r=this;return{next:function(){var o=r.getToken();return{done:o.type===f.eof,value:o}}}}),pe.nextToken=function(){var r=this.curContext();if((!r||!r.preserveSpace)&&this.skipSpace(),this.start=this.pos,this.options.locations&&(this.startLoc=this.curPosition()),this.pos>=this.input.length)return this.finishToken(f.eof);if(r.override)return r.override(this);this.readToken(this.fullCharCodeAtPos())},pe.readToken=function(r){return y(r,this.options.ecmaVersion>=6)||r===92?this.readWord():this.getTokenFromCode(r)},pe.fullCharCodeAtPos=function(){var r=this.input.charCodeAt(this.pos);if(r<=55295||r>=56320)return r;var o=this.input.charCodeAt(this.pos+1);return o<=56319||o>=57344?r:(r<<10)+o-56613888},pe.skipBlockComment=function(){var r=this.options.onComment&&this.curPosition(),o=this.pos,x=this.input.indexOf("*/",this.pos+=2);if(x===-1&&this.raise(this.pos-2,"Unterminated comment"),this.pos=x+2,this.options.locations)for(var S=void 0,D=o;(S=V(this.input,D,this.pos))>-1;)++this.curLine,D=this.lineStart=S;this.options.onComment&&this.options.onComment(!0,this.input.slice(o+2,x),o,this.pos,r,this.curPosition())},pe.skipLineComment=function(r){for(var o=this.pos,x=this.options.onComment&&this.curPosition(),S=this.input.charCodeAt(this.pos+=r);this.pos<this.input.length&&!R(S);)S=this.input.charCodeAt(++this.pos);this.options.onComment&&this.options.onComment(!1,this.input.slice(o+r,this.pos),o,this.pos,x,this.curPosition())},pe.skipSpace=function(){e:for(;this.pos<this.input.length;){var r=this.input.charCodeAt(this.pos);switch(r){case 32:case 160:++this.pos;break;case 13:this.input.charCodeAt(this.pos+1)===10&&++this.pos;case 10:case 8232:case 8233:++this.pos,this.options.locations&&(++this.curLine,this.lineStart=this.pos);break;case 47:switch(this.input.charCodeAt(this.pos+1)){case 42:this.skipBlockComment();break;case 47:this.skipLineComment(2);break;default:break e}break;default:if(r>8&&r<14||r>=5760&&W.test(String.fromCharCode(r)))++this.pos;else break e}}},pe.finishToken=function(r,o){this.end=this.pos,this.options.locations&&(this.endLoc=this.curPosition());var x=this.type;this.type=r,this.value=o,this.updateContext(x)},pe.readToken_dot=function(){var r=this.input.charCodeAt(this.pos+1);if(r>=48&&r<=57)return this.readNumber(!0);var o=this.input.charCodeAt(this.pos+2);return this.options.ecmaVersion>=6&&r===46&&o===46?(this.pos+=3,this.finishToken(f.ellipsis)):(++this.pos,this.finishToken(f.dot))},pe.readToken_slash=function(){var r=this.input.charCodeAt(this.pos+1);return this.exprAllowed?(++this.pos,this.readRegexp()):r===61?this.finishOp(f.assign,2):this.finishOp(f.slash,1)},pe.readToken_mult_modulo_exp=function(r){var o=this.input.charCodeAt(this.pos+1),x=1,S=r===42?f.star:f.modulo;return this.options.ecmaVersion>=7&&r===42&&o===42&&(++x,S=f.starstar,o=this.input.charCodeAt(this.pos+2)),o===61?this.finishOp(f.assign,x+1):this.finishOp(S,x)},pe.readToken_pipe_amp=function(r){var o=this.input.charCodeAt(this.pos+1);return o===r?this.options.ecmaVersion>=12&&this.input.charCodeAt(this.pos+2)===61?this.finishOp(f.assign,3):this.finishOp(r===124?f.logicalOR:f.logicalAND,2):o===61?this.finishOp(f.assign,2):this.finishOp(r===124?f.bitwiseOR:f.bitwiseAND,1)},pe.readToken_caret=function(){return this.input.charCodeAt(this.pos+1)===61?this.finishOp(f.assign,2):this.finishOp(f.bitwiseXOR,1)},pe.readToken_plus_min=function(r){var o=this.input.charCodeAt(this.pos+1);return o===r?o===45&&!this.inModule&&this.input.charCodeAt(this.pos+2)===62&&(this.lastTokEnd===0||F.test(this.input.slice(this.lastTokEnd,this.pos)))?(this.skipLineComment(3),this.skipSpace(),this.nextToken()):this.finishOp(f.incDec,2):o===61?this.finishOp(f.assign,2):this.finishOp(f.plusMin,1)},pe.readToken_lt_gt=function(r){var o=this.input.charCodeAt(this.pos+1),x=1;return o===r?(x=r===62&&this.input.charCodeAt(this.pos+2)===62?3:2,this.input.charCodeAt(this.pos+x)===61?this.finishOp(f.assign,x+1):this.finishOp(f.bitShift,x)):o===33&&r===60&&!this.inModule&&this.input.charCodeAt(this.pos+2)===45&&this.input.charCodeAt(this.pos+3)===45?(this.skipLineComment(4),this.skipSpace(),this.nextToken()):(o===61&&(x=2),this.finishOp(f.relational,x))},pe.readToken_eq_excl=function(r){var o=this.input.charCodeAt(this.pos+1);return o===61?this.finishOp(f.equality,this.input.charCodeAt(this.pos+2)===61?3:2):r===61&&o===62&&this.options.ecmaVersion>=6?(this.pos+=2,this.finishToken(f.arrow)):this.finishOp(r===61?f.eq:f.prefix,1)},pe.readToken_question=function(){var r=this.options.ecmaVersion;if(r>=11){var o=this.input.charCodeAt(this.pos+1);if(o===46){var x=this.input.charCodeAt(this.pos+2);if(x<48||x>57)return this.finishOp(f.questionDot,2)}if(o===63)return r>=12&&this.input.charCodeAt(this.pos+2)===61?this.finishOp(f.assign,3):this.finishOp(f.coalesce,2)}return this.finishOp(f.question,1)},pe.readToken_numberSign=function(){var r=this.options.ecmaVersion,o=35;if(r>=13&&(++this.pos,o=this.fullCharCodeAtPos(),y(o,!0)||o===92))return this.finishToken(f.privateId,this.readWord1());this.raise(this.pos,"Unexpected character '"+he(o)+"'")},pe.getTokenFromCode=function(r){switch(r){case 46:return this.readToken_dot();case 40:return++this.pos,this.finishToken(f.parenL);case 41:return++this.pos,this.finishToken(f.parenR);case 59:return++this.pos,this.finishToken(f.semi);case 44:return++this.pos,this.finishToken(f.comma);case 91:return++this.pos,this.finishToken(f.bracketL);case 93:return++this.pos,this.finishToken(f.bracketR);case 123:return++this.pos,this.finishToken(f.braceL);case 125:return++this.pos,this.finishToken(f.braceR);case 58:return++this.pos,this.finishToken(f.colon);case 96:if(this.options.ecmaVersion<6)break;return++this.pos,this.finishToken(f.backQuote);case 48:var o=this.input.charCodeAt(this.pos+1);if(o===120||o===88)return this.readRadixNumber(16);if(this.options.ecmaVersion>=6){if(o===111||o===79)return this.readRadixNumber(8);if(o===98||o===66)return this.readRadixNumber(2)}case 49:case 50:case 51:case 52:case 53:case 54:case 55:case 56:case 57:return this.readNumber(!1);case 34:case 39:return this.readString(r);case 47:return this.readToken_slash();case 37:case 42:return this.readToken_mult_modulo_exp(r);case 124:case 38:return this.readToken_pipe_amp(r);case 94:return this.readToken_caret();case 43:case 45:return this.readToken_plus_min(r);case 60:case 62:return this.readToken_lt_gt(r);case 61:case 33:return this.readToken_eq_excl(r);case 63:return this.readToken_question();case 126:return this.finishOp(f.prefix,1);case 35:return this.readToken_numberSign()}this.raise(this.pos,"Unexpected character '"+he(r)+"'")},pe.finishOp=function(r,o){var x=this.input.slice(this.pos,this.pos+o);return this.pos+=o,this.finishToken(r,x)},pe.readRegexp=function(){for(var r,o,x=this.pos;;){this.pos>=this.input.length&&this.raise(x,"Unterminated regular expression");var S=this.input.charAt(this.pos);if(F.test(S)&&this.raise(x,"Unterminated regular expression"),r)r=!1;else{if(S==="[")o=!0;else if(S==="]"&&o)o=!1;else if(S==="/"&&!o)break;r=S==="\\"}++this.pos}var D=this.input.slice(x,this.pos);++this.pos;var G=this.pos,K=this.readWord1();this.containsEsc&&this.unexpected(G);var H=this.regexpState||(this.regexpState=new at(this));H.reset(x,D,K),this.validateRegExpFlags(H),this.validateRegExpPattern(H);var Y=null;try{Y=new RegExp(D,K)}catch{}return this.finishToken(f.regexp,{pattern:D,flags:K,value:Y})},pe.readInt=function(r,o,x){for(var S=this.options.ecmaVersion>=12&&o===void 0,D=x&&this.input.charCodeAt(this.pos)===48,G=this.pos,K=0,H=0,Y=0,de=o??1/0;Y<de;++Y,++this.pos){var le=this.input.charCodeAt(this.pos),Me=void 0;if(S&&le===95){D&&this.raiseRecoverable(this.pos,"Numeric separator is not allowed in legacy octal numeric literals"),H===95&&this.raiseRecoverable(this.pos,"Numeric separator must be exactly one underscore"),Y===0&&this.raiseRecoverable(this.pos,"Numeric separator is not allowed at the first of digits"),H=le;continue}if(le>=97?Me=le-97+10:le>=65?Me=le-65+10:le>=48&&le<=57?Me=le-48:Me=1/0,Me>=r)break;H=le,K=K*r+Me}return S&&H===95&&this.raiseRecoverable(this.pos-1,"Numeric separator is not allowed at the last of digits"),this.pos===G||o!=null&&this.pos-G!==o?null:K};function ll(r,o){return o?parseInt(r,8):parseFloat(r.replace(/_/g,""))}function $i(r){return typeof BigInt!="function"?null:BigInt(r.replace(/_/g,""))}pe.readRadixNumber=function(r){var o=this.pos;this.pos+=2;var x=this.readInt(r);return x==null&&this.raise(this.start+2,"Expected number in radix "+r),this.options.ecmaVersion>=11&&this.input.charCodeAt(this.pos)===110?(x=$i(this.input.slice(o,this.pos)),++this.pos):y(this.fullCharCodeAtPos())&&this.raise(this.pos,"Identifier directly after number"),this.finishToken(f.num,x)},pe.readNumber=function(r){var o=this.pos;!r&&this.readInt(10,void 0,!0)===null&&this.raise(o,"Invalid number");var x=this.pos-o>=2&&this.input.charCodeAt(o)===48;x&&this.strict&&this.raise(o,"Invalid number");var S=this.input.charCodeAt(this.pos);if(!x&&!r&&this.options.ecmaVersion>=11&&S===110){var D=$i(this.input.slice(o,this.pos));return++this.pos,y(this.fullCharCodeAtPos())&&this.raise(this.pos,"Identifier directly after number"),this.finishToken(f.num,D)}x&&/[89]/.test(this.input.slice(o,this.pos))&&(x=!1),S===46&&!x&&(++this.pos,this.readInt(10),S=this.input.charCodeAt(this.pos)),(S===69||S===101)&&!x&&(S=this.input.charCodeAt(++this.pos),(S===43||S===45)&&++this.pos,this.readInt(10)===null&&this.raise(o,"Invalid number")),y(this.fullCharCodeAtPos())&&this.raise(this.pos,"Identifier directly after number");var G=ll(this.input.slice(o,this.pos),x);return this.finishToken(f.num,G)},pe.readCodePoint=function(){var r=this.input.charCodeAt(this.pos),o;if(r===123){this.options.ecmaVersion<6&&this.unexpected();var x=++this.pos;o=this.readHexChar(this.input.indexOf("}",this.pos)-this.pos),++this.pos,o>1114111&&this.invalidStringToken(x,"Code point out of bounds")}else o=this.readHexChar(4);return o},pe.readString=function(r){for(var o="",x=++this.pos;;){this.pos>=this.input.length&&this.raise(this.start,"Unterminated string constant");var S=this.input.charCodeAt(this.pos);if(S===r)break;S===92?(o+=this.input.slice(x,this.pos),o+=this.readEscapedChar(!1),x=this.pos):S===8232||S===8233?(this.options.ecmaVersion<10&&this.raise(this.start,"Unterminated string constant"),++this.pos,this.options.locations&&(this.curLine++,this.lineStart=this.pos)):(R(S)&&this.raise(this.start,"Unterminated string constant"),++this.pos)}return o+=this.input.slice(x,this.pos++),this.finishToken(f.string,o)};var Di={};pe.tryReadTemplateToken=function(){this.inTemplateElement=!0;try{this.readTmplToken()}catch(r){if(r===Di)this.readInvalidTemplateToken();else throw r}this.inTemplateElement=!1},pe.invalidStringToken=function(r,o){if(this.inTemplateElement&&this.options.ecmaVersion>=9)throw Di;this.raise(r,o)},pe.readTmplToken=function(){for(var r="",o=this.pos;;){this.pos>=this.input.length&&this.raise(this.start,"Unterminated template");var x=this.input.charCodeAt(this.pos);if(x===96||x===36&&this.input.charCodeAt(this.pos+1)===123)return this.pos===this.start&&(this.type===f.template||this.type===f.invalidTemplate)?x===36?(this.pos+=2,this.finishToken(f.dollarBraceL)):(++this.pos,this.finishToken(f.backQuote)):(r+=this.input.slice(o,this.pos),this.finishToken(f.template,r));if(x===92)r+=this.input.slice(o,this.pos),r+=this.readEscapedChar(!0),o=this.pos;else if(R(x)){switch(r+=this.input.slice(o,this.pos),++this.pos,x){case 13:this.input.charCodeAt(this.pos)===10&&++this.pos;case 10:r+=`
`;break;default:r+=String.fromCharCode(x);break}this.options.locations&&(++this.curLine,this.lineStart=this.pos),o=this.pos}else++this.pos}},pe.readInvalidTemplateToken=function(){for(;this.pos<this.input.length;this.pos++)switch(this.input[this.pos]){case"\\":++this.pos;break;case"$":if(this.input[this.pos+1]!=="{")break;case"`":return this.finishToken(f.invalidTemplate,this.input.slice(this.start,this.pos));case"\r":this.input[this.pos+1]===`
`&&++this.pos;case`
`:case"\u2028":case"\u2029":++this.curLine,this.lineStart=this.pos+1;break}this.raise(this.start,"Unterminated template")},pe.readEscapedChar=function(r){var o=this.input.charCodeAt(++this.pos);switch(++this.pos,o){case 110:return`
`;case 114:return"\r";case 120:return String.fromCharCode(this.readHexChar(2));case 117:return he(this.readCodePoint());case 116:return"	";case 98:return"\b";case 118:return"\v";case 102:return"\f";case 13:this.input.charCodeAt(this.pos)===10&&++this.pos;case 10:return this.options.locations&&(this.lineStart=this.pos,++this.curLine),"";case 56:case 57:if(this.strict&&this.invalidStringToken(this.pos-1,"Invalid escape sequence"),r){var x=this.pos-1;this.invalidStringToken(x,"Invalid escape sequence in template string")}default:if(o>=48&&o<=55){var S=this.input.substr(this.pos-1,3).match(/^[0-7]+/)[0],D=parseInt(S,8);return D>255&&(S=S.slice(0,-1),D=parseInt(S,8)),this.pos+=S.length-1,o=this.input.charCodeAt(this.pos),(S!=="0"||o===56||o===57)&&(this.strict||r)&&this.invalidStringToken(this.pos-1-S.length,r?"Octal literal in template string":"Octal literal in strict mode"),String.fromCharCode(D)}return R(o)?(this.options.locations&&(this.lineStart=this.pos,++this.curLine),""):String.fromCharCode(o)}},pe.readHexChar=function(r){var o=this.pos,x=this.readInt(16,r);return x===null&&this.invalidStringToken(o,"Bad character escape sequence"),x},pe.readWord1=function(){this.containsEsc=!1;for(var r="",o=!0,x=this.pos,S=this.options.ecmaVersion>=6;this.pos<this.input.length;){var D=this.fullCharCodeAtPos();if(k(D,S))this.pos+=D<=65535?1:2;else if(D===92){this.containsEsc=!0,r+=this.input.slice(x,this.pos);var G=this.pos;this.input.charCodeAt(++this.pos)!==117&&this.invalidStringToken(this.pos,"Expecting Unicode escape sequence \\uXXXX"),++this.pos;var K=this.readCodePoint();(o?y:k)(K,S)||this.invalidStringToken(G,"Invalid Unicode escape"),r+=he(K),x=this.pos}else break;o=!1}return r+this.input.slice(x,this.pos)},pe.readWord=function(){var r=this.readWord1(),o=f.name;return this.keywords.test(r)&&(o=T[r]),this.finishToken(o,r)};var Pi="8.14.0";Se.acorn={Parser:Se,version:Pi,defaultOptions:ue,Position:ne,SourceLocation:ce,getLineInfo:Ve,Node:ms,TokenType:v,tokTypes:f,keywordTypes:T,TokContext:je,tokContexts:be,isIdentifierChar:k,isIdentifierStart:y,Token:Bs,isNewLine:R,lineBreak:F,lineBreakG:L,nonASCIIwhitespace:W};function ul(r,o){return Se.parse(r,o)}function cl(r,o,x){return Se.parseExpressionAt(r,o,x)}function hl(r,o){return Se.tokenizer(r,o)}M.Node=ms,M.Parser=Se,M.Position=ne,M.SourceLocation=ce,M.TokContext=je,M.Token=Bs,M.TokenType=v,M.defaultOptions=ue,M.getLineInfo=Ve,M.isIdentifierChar=k,M.isIdentifierStart=y,M.isNewLine=R,M.keywordTypes=T,M.lineBreak=F,M.lineBreakG=L,M.nonASCIIwhitespace=W,M.parse=ul,M.parseExpressionAt=cl,M.tokContexts=be,M.tokTypes=f,M.tokenizer=hl,M.version=Pi})}),a=s((B,z)=>{var M=class{constructor(b,l){this.value=b,Array.isArray(l)?this.size=l:(this.size=new Int32Array(3),l.z?this.size=new Int32Array([l.x,l.y,l.z]):l.y?this.size=new Int32Array([l.x,l.y]):this.size=new Int32Array([l.x]));const[d,C,m]=this.size;if(m){if(this.value.length!==d*C*m)throw new Error(`Input size ${this.value.length} does not match ${d} * ${C} * ${m} = ${C*d*m}`)}else if(C){if(this.value.length!==d*C)throw new Error(`Input size ${this.value.length} does not match ${d} * ${C} = ${C*d}`)}else if(this.value.length!==d)throw new Error(`Input size ${this.value.length} does not match ${d}`)}toArray(){const{utils:b}=h(),[l,d,C]=this.size;return C?b.erectMemoryOptimized3DFloat(this.value.subarray?this.value:new Float32Array(this.value),l,d,C):d?b.erectMemoryOptimized2DFloat(this.value.subarray?this.value:new Float32Array(this.value),l,d):this.value}};function _(b,l){return new M(b,l)}z.exports={Input:M,input:_}}),c=s((B,z)=>{var M=class{constructor(_){const{texture:b,size:l,dimensions:d,output:C,context:m,type:p="NumberTexture",kernel:u,internalFormat:g,textureFormat:E}=_;if(!C)throw new Error('settings property "output" required.');if(!m)throw new Error('settings property "context" required.');if(!b)throw new Error('settings property "texture" required.');if(!u)throw new Error('settings property "kernel" required.');this.texture=b,b._refs?b._refs++:b._refs=1,this.size=l,this.dimensions=d,this.output=C,this.context=m,this.kernel=u,this.type=p,this._deleted=!1,this.internalFormat=g,this.textureFormat=E}toArray(){throw new Error(`Not implemented on ${this.constructor.name}`)}clone(){throw new Error(`Not implemented on ${this.constructor.name}`)}delete(){throw new Error(`Not implemented on ${this.constructor.name}`)}clear(){throw new Error(`Not implemented on ${this.constructor.name}`)}};z.exports={Texture:M}}),h=s((B,z)=>{const M=i(),{Input:_}=a(),{Texture:b}=c(),l=/function ([^(]*)/,d=/((\/\/.*$)|(\/\*[\s\S]*?\*\/))/gm,C=/([^\s,]+)/g,m={systemEndianness(){return E},getSystemEndianness(){const w=new ArrayBuffer(4),y=new Uint32Array(w),k=new Uint8Array(w);if(y[0]=3735928559,k[0]===239)return"LE";if(k[0]===222)return"BE";throw new Error("unknown endianness")},isFunction(w){return typeof w=="function"},isFunctionString(w){return typeof w=="string"?w.slice(0,8).toLowerCase()==="function":!1},getFunctionNameFromString(w){const y=l.exec(w);return!y||y.length===0?null:y[1].trim()},getFunctionBodyFromString(w){return w.substring(w.indexOf("{")+1,w.lastIndexOf("}"))},getArgumentNamesFromString(w){const y=w.replace(d,"");let k=y.slice(y.indexOf("(")+1,y.indexOf(")")).match(C);return k===null&&(k=[]),k},clone(w){if(w===null||typeof w!="object"||w.hasOwnProperty("isActiveClone"))return w;const y=w.constructor();for(let k in w)Object.prototype.hasOwnProperty.call(w,k)&&(w.isActiveClone=null,y[k]=m.clone(w[k]),delete w.isActiveClone);return y},isArray(w){return!isNaN(w.length)},getVariableType(w,y){if(m.isArray(w))return w.length>0&&w[0].nodeName==="IMG"?"HTMLImageArray":"Array";switch(w.constructor){case Boolean:return"Boolean";case Number:return y&&Number.isInteger(w)?"Integer":"Float";case b:return w.type;case _:return"Input"}if("nodeName"in w)switch(w.nodeName){case"IMG":return"HTMLImage";case"CANVAS":return"HTMLImage";case"VIDEO":return"HTMLVideo"}else{if(w.hasOwnProperty("type"))return w.type;if(typeof OffscreenCanvas<"u"&&w instanceof OffscreenCanvas)return"OffscreenCanvas";if(typeof ImageBitmap<"u"&&w instanceof ImageBitmap)return"ImageBitmap";if(typeof ImageData<"u"&&w instanceof ImageData)return"ImageData"}return"Unknown"},getKernelTextureSize(w,y){let[k,v,$]=y,P=(k||1)*(v||1)*($||1);return w.optimizeFloatMemory&&w.precision==="single"&&(k=P=Math.ceil(P/4)),v>1&&k*v===P?new Int32Array([k,v]):m.closestSquareDimensions(P)},closestSquareDimensions(w){const y=Math.sqrt(w);let k=Math.ceil(y),v=Math.floor(y);for(;k*v<w;)k--,v=Math.ceil(w/k);return new Int32Array([v,Math.ceil(w/v)])},getMemoryOptimizedFloatTextureSize(w,y){const k=m.roundTo((w[0]||1)*(w[1]||1)*(w[2]||1)*(w[3]||1),4)/y;return m.closestSquareDimensions(k)},getMemoryOptimizedPackedTextureSize(w,y){const[k,v,$]=w,P=m.roundTo((k||1)*(v||1)*($||1),4)/(4/y);return m.closestSquareDimensions(P)},roundTo(w,y){return Math.floor((w+y-1)/y)*y},getDimensions(w,y){let k;if(m.isArray(w)){const v=[];let $=w;for(;m.isArray($);)v.push($.length),$=$[0];k=v.reverse()}else if(w instanceof b)k=w.output;else if(w instanceof _)k=w.size;else throw new Error(`Unknown dimensions of ${w}`);if(y)for(k=Array.from(k);k.length<3;)k.push(1);return new Int32Array(k)},flatten2dArrayTo(w,y){let k=0;for(let v=0;v<w.length;v++)y.set(w[v],k),k+=w[v].length},flatten3dArrayTo(w,y){let k=0;for(let v=0;v<w.length;v++)for(let $=0;$<w[v].length;$++)y.set(w[v][$],k),k+=w[v][$].length},flatten4dArrayTo(w,y){let k=0;for(let v=0;v<w.length;v++)for(let $=0;$<w[v].length;$++)for(let P=0;P<w[v][$].length;P++)y.set(w[v][$][P],k),k+=w[v][$][P].length},flattenTo(w,y){m.isArray(w[0])?m.isArray(w[0][0])?m.isArray(w[0][0][0])?m.flatten4dArrayTo(w,y):m.flatten3dArrayTo(w,y):m.flatten2dArrayTo(w,y):y.set(w)},splitArray(w,y){const k=[];for(let v=0;v<w.length;v+=y)k.push(new w.constructor(w.buffer,v*4+w.byteOffset,y));return k},getAstString(w,y){const k=Array.isArray(w)?w:w.split(/\r?\n/g),v=y.loc.start,$=y.loc.end,P=[];if(v.line===$.line)P.push(k[v.line-1].substring(v.column,$.column));else{P.push(k[v.line-1].slice(v.column));for(let O=v.line;O<$.line;O++)P.push(k[O]);P.push(k[$.line-1].slice(0,$.column))}return P.join(`
`)},allPropertiesOf(w){const y=[];do y.push.apply(y,Object.getOwnPropertyNames(w));while(w=Object.getPrototypeOf(w));return y},linesToString(w){return w.length>0?w.join(`;
`)+`;
`:`
`},warnDeprecated(w,y,k){console.warn(k?`You are using a deprecated ${w} "${y}". It has been replaced with "${k}". Fixing, but please upgrade as it will soon be removed.`:`You are using a deprecated ${w} "${y}". It has been removed. Fixing, but please upgrade as it will soon be removed.`)},flipPixels:(w,y,k)=>{const v=k/2|0,$=y*4,P=new Uint8ClampedArray(y*4),O=w.slice(0);for(let T=0;T<v;++T){const A=T*$,f=(k-T-1)*$;P.set(O.subarray(A,A+$)),O.copyWithin(A,f,f+$),O.set(P,f)}return O},erectPackedFloat:(w,y)=>w.subarray(0,y),erect2DPackedFloat:(w,y,k)=>{const v=new Array(k);for(let $=0;$<k;$++){const P=$*y,O=P+y;v[$]=w.subarray(P,O)}return v},erect3DPackedFloat:(w,y,k,v)=>{const $=new Array(v);for(let P=0;P<v;P++){const O=new Array(k);for(let T=0;T<k;T++){const A=P*k*y+T*y,f=A+y;O[T]=w.subarray(A,f)}$[P]=O}return $},erectMemoryOptimizedFloat:(w,y)=>w.subarray(0,y),erectMemoryOptimized2DFloat:(w,y,k)=>{const v=new Array(k);for(let $=0;$<k;$++){const P=$*y;v[$]=w.subarray(P,P+y)}return v},erectMemoryOptimized3DFloat:(w,y,k,v)=>{const $=new Array(v);for(let P=0;P<v;P++){const O=new Array(k);for(let T=0;T<k;T++){const A=P*k*y+T*y;O[T]=w.subarray(A,A+y)}$[P]=O}return $},erectFloat:(w,y)=>{const k=new Float32Array(y);let v=0;for(let $=0;$<y;$++)k[$]=w[v],v+=4;return k},erect2DFloat:(w,y,k)=>{const v=new Array(k);let $=0;for(let P=0;P<k;P++){const O=new Float32Array(y);for(let T=0;T<y;T++)O[T]=w[$],$+=4;v[P]=O}return v},erect3DFloat:(w,y,k,v)=>{const $=new Array(v);let P=0;for(let O=0;O<v;O++){const T=new Array(k);for(let A=0;A<k;A++){const f=new Float32Array(y);for(let F=0;F<y;F++)f[F]=w[P],P+=4;T[A]=f}$[O]=T}return $},erectArray2:(w,y)=>{const k=new Array(y),v=y*4;let $=0;for(let P=0;P<v;P+=4)k[$++]=w.subarray(P,P+2);return k},erect2DArray2:(w,y,k)=>{const v=new Array(k),$=y*4;for(let P=0;P<k;P++){const O=new Array(y),T=P*$;let A=0;for(let f=0;f<$;f+=4)O[A++]=w.subarray(f+T,f+T+2);v[P]=O}return v},erect3DArray2:(w,y,k,v)=>{const $=y*4,P=new Array(v);for(let O=0;O<v;O++){const T=new Array(k);for(let A=0;A<k;A++){const f=new Array(y),F=O*$*k+A*$;let L=0;for(let R=0;R<$;R+=4)f[L++]=w.subarray(R+F,R+F+2);T[A]=f}P[O]=T}return P},erectArray3:(w,y)=>{const k=new Array(y),v=y*4;let $=0;for(let P=0;P<v;P+=4)k[$++]=w.subarray(P,P+3);return k},erect2DArray3:(w,y,k)=>{const v=y*4,$=new Array(k);for(let P=0;P<k;P++){const O=new Array(y),T=P*v;let A=0;for(let f=0;f<v;f+=4)O[A++]=w.subarray(f+T,f+T+3);$[P]=O}return $},erect3DArray3:(w,y,k,v)=>{const $=y*4,P=new Array(v);for(let O=0;O<v;O++){const T=new Array(k);for(let A=0;A<k;A++){const f=new Array(y),F=O*$*k+A*$;let L=0;for(let R=0;R<$;R+=4)f[L++]=w.subarray(R+F,R+F+3);T[A]=f}P[O]=T}return P},erectArray4:(w,y)=>{const k=new Array(w),v=y*4;let $=0;for(let P=0;P<v;P+=4)k[$++]=w.subarray(P,P+4);return k},erect2DArray4:(w,y,k)=>{const v=y*4,$=new Array(k);for(let P=0;P<k;P++){const O=new Array(y),T=P*v;let A=0;for(let f=0;f<v;f+=4)O[A++]=w.subarray(f+T,f+T+4);$[P]=O}return $},erect3DArray4:(w,y,k,v)=>{const $=y*4,P=new Array(v);for(let O=0;O<v;O++){const T=new Array(k);for(let A=0;A<k;A++){const f=new Array(y),F=O*$*k+A*$;let L=0;for(let R=0;R<$;R+=4)f[L++]=w.subarray(R+F,R+F+4);T[A]=f}P[O]=T}return P},flattenFunctionToString:(w,y)=>{const{findDependency:k,thisLookup:v,doNotDefine:$}=y;let P=y.flattened;P||(P=y.flattened={});const O=M.parse(w,{ecmaVersion:2020}),T=[];let A=0;function f(L){if(Array.isArray(L)){const R=[];for(let V=0;V<L.length;V++)R.push(f(L[V]));return R.join("")}switch(L.type){case"Program":return f(L.body)+(L.body[0].type==="VariableDeclaration"?";":"");case"FunctionDeclaration":return`function ${L.id.name}(${L.params.map(f).join(", ")}) ${f(L.body)}`;case"BlockStatement":{const V=[];A+=2;for(let W=0;W<L.body.length;W++){const N=f(L.body[W]);N&&V.push(" ".repeat(A)+N,`;
`)}return A-=2,`{
${V.join("")}}`}case"VariableDeclaration":const R=m.normalizeDeclarations(L).map(f).filter(V=>V!==null);return R.length<1?"":`${L.kind} ${R.join(",")}`;case"VariableDeclarator":return L.init?L.init.object&&L.init.object.type==="ThisExpression"?v(L.init.property.name,!0)?`${L.id.name} = ${f(L.init)}`:null:`${L.id.name} = ${f(L.init)}`:L.id.name;case"CallExpression":if(L.callee.property.name==="subarray")return`${f(L.callee.object)}.${f(L.callee.property)}(${L.arguments.map(V=>f(V)).join(", ")})`;if(L.callee.object.name==="gl"||L.callee.object.name==="context")return`${f(L.callee.object)}.${f(L.callee.property)}(${L.arguments.map(V=>f(V)).join(", ")})`;if(L.callee.object.type==="ThisExpression")return T.push(k("this",L.callee.property.name)),`${L.callee.property.name}(${L.arguments.map(V=>f(V)).join(", ")})`;if(L.callee.object.name){const V=k(L.callee.object.name,L.callee.property.name);return V===null?`${L.callee.object.name}.${L.callee.property.name}(${L.arguments.map(W=>f(W)).join(", ")})`:(T.push(V),`${L.callee.property.name}(${L.arguments.map(W=>f(W)).join(", ")})`)}else{if(L.callee.object.type==="MemberExpression")return`${f(L.callee.object)}.${L.callee.property.name}(${L.arguments.map(V=>f(V)).join(", ")})`;throw new Error("unknown ast.callee")}case"ReturnStatement":return`return ${f(L.argument)}`;case"BinaryExpression":return`(${f(L.left)}${L.operator}${f(L.right)})`;case"UnaryExpression":return L.prefix?`${L.operator} ${f(L.argument)}`:`${f(L.argument)} ${L.operator}`;case"ExpressionStatement":return`${f(L.expression)}`;case"SequenceExpression":return`(${f(L.expressions)})`;case"ArrowFunctionExpression":return`(${L.params.map(f).join(", ")}) => ${f(L.body)}`;case"Literal":return L.raw;case"Identifier":return L.name;case"MemberExpression":return L.object.type==="ThisExpression"?v(L.property.name):L.computed?`${f(L.object)}[${f(L.property)}]`:f(L.object)+"."+f(L.property);case"ThisExpression":return"this";case"NewExpression":return`new ${f(L.callee)}(${L.arguments.map(V=>f(V)).join(", ")})`;case"ForStatement":return`for (${f(L.init)};${f(L.test)};${f(L.update)}) ${f(L.body)}`;case"AssignmentExpression":return`${f(L.left)}${L.operator}${f(L.right)}`;case"UpdateExpression":return`${f(L.argument)}${L.operator}`;case"IfStatement":{const V=f(L.consequent);if(!L.alternate)return`if (${f(L.test)}) ${V}`;const W=L.consequent.type==="BlockStatement"?"":";";return`if (${f(L.test)}) ${V}${W} else ${f(L.alternate)}`}case"ThrowStatement":return`throw ${f(L.argument)}`;case"ObjectPattern":return L.properties.map(f).join(", ");case"ArrayPattern":return L.elements.map(f).join(", ");case"DebuggerStatement":return"debugger;";case"ConditionalExpression":return`${f(L.test)}?${f(L.consequent)}:${f(L.alternate)}`;case"Property":if(L.kind==="init")return f(L.key)}throw new Error(`unhandled ast.type of ${L.type}`)}const F=f(O);if(T.length>0){const L=[];for(let R=0;R<T.length;R++){const V=T[R];P[V]||(P[V]=!0),V&&L.push(m.flattenFunctionToString(V,y)+`
`)}return L.join("")+F}return F},normalizeDeclarations:w=>{if(w.type!=="VariableDeclaration")throw new Error('Ast is not of type "VariableDeclaration"');const y=[];for(let k=0;k<w.declarations.length;k++){const v=w.declarations[k];if(v.id&&v.id.type==="ObjectPattern"&&v.id.properties){const{properties:$}=v.id;for(let P=0;P<$.length;P++){const O=$[P];if(O.value.type==="ObjectPattern"&&O.value.properties)for(let T=0;T<O.value.properties.length;T++){const A=O.value.properties[T];if(A.type==="Property")y.push({type:"VariableDeclarator",id:{type:"Identifier",name:A.key.name},init:{type:"MemberExpression",object:{type:"MemberExpression",object:v.init,property:{type:"Identifier",name:O.key.name},computed:!1},property:{type:"Identifier",name:A.key.name},computed:!1}});else throw new Error("unexpected state")}else if(O.value.type==="Identifier")y.push({type:"VariableDeclarator",id:{type:"Identifier",name:O.value&&O.value.name?O.value.name:O.key.name},init:{type:"MemberExpression",object:v.init,property:{type:"Identifier",name:O.key.name},computed:!1}});else throw new Error("unexpected state")}}else if(v.id&&v.id.type==="ArrayPattern"&&v.id.elements){const{elements:$}=v.id;for(let P=0;P<$.length;P++){const O=$[P];if(O.type==="Identifier")y.push({type:"VariableDeclarator",id:{type:"Identifier",name:O.name},init:{type:"MemberExpression",object:v.init,property:{type:"Literal",value:P,raw:P.toString(),start:O.start,end:O.end},computed:!0}});else throw new Error("unexpected state")}}else y.push(v)}return y},splitHTMLImageToRGB:(w,y)=>{const k=w.createKernel(function(T){return T[this.thread.y][this.thread.x].r*255},{output:[y.width,y.height],precision:"unsigned",argumentTypes:{a:"HTMLImage"}}),v=w.createKernel(function(T){return T[this.thread.y][this.thread.x].g*255},{output:[y.width,y.height],precision:"unsigned",argumentTypes:{a:"HTMLImage"}}),$=w.createKernel(function(T){return T[this.thread.y][this.thread.x].b*255},{output:[y.width,y.height],precision:"unsigned",argumentTypes:{a:"HTMLImage"}}),P=w.createKernel(function(T){return T[this.thread.y][this.thread.x].a*255},{output:[y.width,y.height],precision:"unsigned",argumentTypes:{a:"HTMLImage"}}),O=[k(y),v(y),$(y),P(y)];return O.rKernel=k,O.gKernel=v,O.bKernel=$,O.aKernel=P,O.gpu=w,O},splitRGBAToCanvases:(w,y,k,v)=>{const $=w.createKernel(function(A){const f=A[this.thread.y][this.thread.x];this.color(f.r/255,0,0,255)},{output:[k,v],graphical:!0,argumentTypes:{v:"Array2D(4)"}});$(y);const P=w.createKernel(function(A){const f=A[this.thread.y][this.thread.x];this.color(0,f.g/255,0,255)},{output:[k,v],graphical:!0,argumentTypes:{v:"Array2D(4)"}});P(y);const O=w.createKernel(function(A){const f=A[this.thread.y][this.thread.x];this.color(0,0,f.b/255,255)},{output:[k,v],graphical:!0,argumentTypes:{v:"Array2D(4)"}});O(y);const T=w.createKernel(function(A){const f=A[this.thread.y][this.thread.x];this.color(255,255,255,f.a/255)},{output:[k,v],graphical:!0,argumentTypes:{v:"Array2D(4)"}});return T(y),[$.canvas,P.canvas,O.canvas,T.canvas]},getMinifySafeName:w=>{try{const{init:y}=M.parse(`const value = ${w.toString()}`,{ecmaVersion:2020}).body[0].declarations[0];return y.body.name||y.body.body[0].argument.name}catch{throw new Error("Unrecognized function type.  Please use `() => yourFunctionVariableHere` or function() { return yourFunctionVariableHere; }")}},sanitizeName:function(w){return p.test(w)&&(w=w.replace(p,"S_S")),u.test(w)?w=w.replace(u,"U_U"):g.test(w)&&(w=w.replace(g,"u_u")),w}},p=/\$/,u=/__/,g=/_/,E=m.getSystemEndianness();z.exports={utils:m}}),I=s((B,z)=>{const{utils:M}=h(),{Input:_}=a();var b=class{static get isSupported(){throw new Error(`"isSupported" not implemented on ${this.name}`)}static isContextMatch(d){throw new Error(`"isContextMatch" not implemented on ${this.name}`)}static getFeatures(){throw new Error(`"getFeatures" not implemented on ${this.name}`)}static destroyContext(d){throw new Error(`"destroyContext" called on ${this.name}`)}static nativeFunctionArguments(){throw new Error(`"nativeFunctionArguments" called on ${this.name}`)}static nativeFunctionReturnType(){throw new Error(`"nativeFunctionReturnType" called on ${this.name}`)}static combineKernels(){throw new Error(`"combineKernels" called on ${this.name}`)}constructor(d,C){if(typeof d!="object"){if(typeof d!="string")throw new Error("source not a string");if(!M.isFunctionString(d))throw new Error("source not a function string")}this.useLegacyEncoder=!1,this.fallbackRequested=!1,this.onRequestFallback=null,this.argumentNames=typeof d=="string"?M.getArgumentNamesFromString(d):null,this.argumentTypes=null,this.argumentSizes=null,this.argumentBitRatios=null,this.kernelArguments=null,this.kernelConstants=null,this.forceUploadKernelConstants=null,this.source=d,this.output=null,this.debug=!1,this.graphical=!1,this.loopMaxIterations=0,this.constants=null,this.constantTypes=null,this.constantBitRatios=null,this.dynamicArguments=!1,this.dynamicOutput=!1,this.canvas=null,this.context=null,this.checkContext=null,this.gpu=null,this.functions=null,this.nativeFunctions=null,this.injectedNative=null,this.subKernels=null,this.validate=!0,this.immutable=!1,this.pipeline=!1,this.precision=null,this.tactic=null,this.plugins=null,this.returnType=null,this.leadingReturnStatement=null,this.followingReturnStatement=null,this.optimizeFloatMemory=null,this.strictIntegers=!1,this.fixIntegerDivisionAccuracy=null,this.randomSeed=null,this.built=!1,this.signature=null}mergeSettings(d){for(let C in d)if(!(!d.hasOwnProperty(C)||!this.hasOwnProperty(C))){switch(C){case"output":if(!Array.isArray(d.output)){this.setOutput(d.output);continue}break;case"functions":this.functions=[];for(let m=0;m<d.functions.length;m++)this.addFunction(d.functions[m]);continue;case"graphical":d[C]&&!d.hasOwnProperty("precision")&&(this.precision="unsigned"),this[C]=d[C];continue;case"nativeFunctions":if(!d.nativeFunctions)continue;this.nativeFunctions=[];for(let m=0;m<d.nativeFunctions.length;m++){const p=d.nativeFunctions[m],{name:u,source:g}=p;this.addNativeFunction(u,g,p)}continue}this[C]=d[C]}this.canvas||(this.canvas=this.initCanvas()),this.context||(this.context=this.initContext()),this.plugins||(this.plugins=this.initPlugins(d))}build(){throw new Error(`"build" not defined on ${this.constructor.name}`)}run(){throw new Error(`"run" not defined on ${this.constructor.name}`)}initCanvas(){throw new Error(`"initCanvas" not defined on ${this.constructor.name}`)}initContext(){throw new Error(`"initContext" not defined on ${this.constructor.name}`)}initPlugins(d){throw new Error(`"initPlugins" not defined on ${this.constructor.name}`)}addFunction(d,C={}){if(d.name&&d.source&&d.argumentTypes&&"returnType"in d)this.functions.push(d);else if(typeof d=="string"||typeof d=="function")this.functions.push(this.functionToIGPUFunction(d,C));else if("settings"in d&&"source"in d)this.functions.push(this.functionToIGPUFunction(d.source,d.settings));else throw new Error("function not properly defined");return this}addNativeFunction(d,C,m={}){const{argumentTypes:p,argumentNames:u}=m.argumentTypes?l(m.argumentTypes):this.constructor.nativeFunctionArguments(C)||{};return this.nativeFunctions.push({name:d,source:C,settings:m,argumentTypes:p,argumentNames:u,returnType:m.returnType||this.constructor.nativeFunctionReturnType(C)}),this}setupArguments(d){if(this.kernelArguments=[],this.argumentTypes)for(let C=0;C<this.argumentTypes.length;C++)this.kernelArguments.push({type:this.argumentTypes[C]});else if(!this.argumentTypes){this.argumentTypes=[];for(let C=0;C<d.length;C++){const m=M.getVariableType(d[C],this.strictIntegers),p=m==="Integer"?"Number":m;this.argumentTypes.push(p),this.kernelArguments.push({type:p})}}this.argumentSizes=new Array(d.length),this.argumentBitRatios=new Int32Array(d.length);for(let C=0;C<d.length;C++){const m=d[C];this.argumentSizes[C]=m.constructor===_?m.size:null,this.argumentBitRatios[C]=this.getBitRatio(m)}if(this.argumentNames.length!==d.length)throw new Error("arguments are miss-aligned")}setupConstants(){this.kernelConstants=[];let d=this.constantTypes===null;if(d&&(this.constantTypes={}),this.constantBitRatios={},this.constants)for(let C in this.constants){if(d){const m=M.getVariableType(this.constants[C],this.strictIntegers);this.constantTypes[C]=m,this.kernelConstants.push({name:C,type:m})}else this.kernelConstants.push({name:C,type:this.constantTypes[C]});this.constantBitRatios[C]=this.getBitRatio(this.constants[C])}}setOptimizeFloatMemory(d){return this.optimizeFloatMemory=d,this}toKernelOutput(d){return d.hasOwnProperty("x")?d.hasOwnProperty("y")?d.hasOwnProperty("z")?[d.x,d.y,d.z]:[d.x,d.y]:[d.x]:d}setOutput(d){return this.output=this.toKernelOutput(d),this}setDebug(d){return this.debug=d,this}setGraphical(d){return this.graphical=d,this.precision="unsigned",this}setLoopMaxIterations(d){return this.loopMaxIterations=d,this}setConstants(d){return this.constants=d,this}setConstantTypes(d){return this.constantTypes=d,this}setFunctions(d){for(let C=0;C<d.length;C++)this.addFunction(d[C]);return this}setNativeFunctions(d){for(let C=0;C<d.length;C++){const m=d[C],{name:p,source:u}=m;this.addNativeFunction(p,u,m)}return this}setInjectedNative(d){return this.injectedNative=d,this}setPipeline(d){return this.pipeline=d,this}setPrecision(d){return this.precision=d,this}setDimensions(d){return M.warnDeprecated("method","setDimensions","setOutput"),this.output=d,this}setOutputToTexture(d){return M.warnDeprecated("method","setOutputToTexture","setPipeline"),this.pipeline=d,this}setImmutable(d){return this.immutable=d,this}setCanvas(d){return this.canvas=d,this}setStrictIntegers(d){return this.strictIntegers=d,this}setDynamicOutput(d){return this.dynamicOutput=d,this}setRandomSeed(d){return this.randomSeed=d,this._mathRandomGenerator=null,this}setHardcodeConstants(d){return M.warnDeprecated("method","setHardcodeConstants"),this.setDynamicOutput(d),this.setDynamicArguments(d),this}setDynamicArguments(d){return this.dynamicArguments=d,this}setUseLegacyEncoder(d){return this.useLegacyEncoder=d,this}setWarnVarUsage(d){return M.warnDeprecated("method","setWarnVarUsage"),this}getCanvas(){return M.warnDeprecated("method","getCanvas"),this.canvas}getWebGl(){return M.warnDeprecated("method","getWebGl"),this.context}setContext(d){return this.context=d,this}setArgumentTypes(d){if(Array.isArray(d))this.argumentTypes=d;else{this.argumentTypes=[];for(const C in d){if(!d.hasOwnProperty(C))continue;const m=this.argumentNames.indexOf(C);if(m===-1)throw new Error(`unable to find argument ${C}`);this.argumentTypes[m]=d[C]}}return this}setTactic(d){return this.tactic=d,this}requestFallback(d){if(!this.onRequestFallback)throw new Error(`"onRequestFallback" not defined on ${this.constructor.name}`);return this.fallbackRequested=!0,this.onRequestFallback(d)}validateSettings(){throw new Error(`"validateSettings" not defined on ${this.constructor.name}`)}addSubKernel(d){if(this.subKernels===null&&(this.subKernels=[]),!d.source)throw new Error('subKernel missing "source" property');if(!d.property&&isNaN(d.property))throw new Error('subKernel missing "property" property');if(!d.name)throw new Error('subKernel missing "name" property');return this.subKernels.push(d),this}destroy(d){throw new Error(`"destroy" called on ${this.constructor.name}`)}getBitRatio(d){if(this.precision==="single")return 4;if(Array.isArray(d[0]))return this.getBitRatio(d[0]);if(d.constructor===_)return this.getBitRatio(d.value);switch(d.constructor){case Uint8ClampedArray:case Uint8Array:case Int8Array:return 1;case Uint16Array:case Int16Array:return 2;case Float32Array:case Int32Array:default:return 4}}getPixels(d){throw new Error(`"getPixels" called on ${this.constructor.name}`)}checkOutput(){if(!this.output||!M.isArray(this.output))throw new Error("kernel.output not an array");if(this.output.length<1)throw new Error("kernel.output is empty, needs at least 1 value");for(let d=0;d<this.output.length;d++)if(isNaN(this.output[d])||this.output[d]<1)throw new Error(`${this.constructor.name}.output[${d}] incorrectly defined as \`${this.output[d]}\`, needs to be numeric, and greater than 0`)}prependString(d){throw new Error(`"prependString" called on ${this.constructor.name}`)}hasPrependString(d){throw new Error(`"hasPrependString" called on ${this.constructor.name}`)}toJSON(){return{settings:{output:this.output,pipeline:this.pipeline,argumentNames:this.argumentNames,argumentsTypes:this.argumentTypes,constants:this.constants,pluginNames:this.plugins?this.plugins.map(d=>d.name):null,returnType:this.returnType}}}buildSignature(d){const C=this.constructor;this.signature=C.getSignature(this,C.getArgumentTypes(this,d))}static getArgumentTypes(d,C){const m=new Array(C.length);for(let p=0;p<C.length;p++){const u=C[p],g=d.argumentTypes[p];if(u.type)m[p]=u.type;else switch(g){case"Number":case"Integer":case"Float":case"ArrayTexture(1)":m[p]=M.getVariableType(u);break;default:m[p]=g}}return m}static getSignature(d,C){throw new Error(`"getSignature" not implemented on ${this.name}`)}functionToIGPUFunction(d,C={}){if(typeof d!="string"&&typeof d!="function")throw new Error("source not a string or function");const m=typeof d=="string"?d:d.toString();let p=[];return Array.isArray(C.argumentTypes)?p=C.argumentTypes:typeof C.argumentTypes=="object"?p=M.getArgumentNamesFromString(m).map(u=>C.argumentTypes[u])||[]:p=C.argumentTypes||[],{name:M.getFunctionNameFromString(m)||null,source:m,argumentTypes:p,returnType:C.returnType||null}}onActivate(d){}};function l(d){const C=Object.keys(d),m=[];for(let p=0;p<C.length;p++){const u=C[p];m.push(d[u])}return{argumentTypes:m,argumentNames:C}}z.exports={Kernel:b}}),U=s((B,z)=>{z.exports={FunctionBuilder:class Cr{static fromKernel(_,b,l){const{kernelArguments:d,kernelConstants:C,argumentNames:m,argumentSizes:p,argumentBitRatios:u,constants:g,constantBitRatios:E,debug:w,loopMaxIterations:y,nativeFunctions:k,output:v,optimizeFloatMemory:$,precision:P,plugins:O,source:T,subKernels:A,functions:f,leadingReturnStatement:F,followingReturnStatement:L,dynamicArguments:R,dynamicOutput:V}=_,W=new Array(d.length),N={};for(let re=0;re<d.length;re++)W[re]=d[re].type;for(let re=0;re<C.length;re++){const ye=C[re];N[ye.name]=ye.type}const te=(re,ye)=>Ie.needsArgumentType(re,ye),ee=(re,ye,_e)=>{Ie.assignArgumentType(re,ye,_e)},X=(re,ye,_e)=>Ie.lookupReturnType(re,ye,_e),ie=re=>Ie.lookupFunctionArgumentTypes(re),se=(re,ye)=>Ie.lookupFunctionArgumentName(re,ye),J=(re,ye)=>Ie.lookupFunctionArgumentBitRatio(re,ye),ae=(re,ye,_e,Ne)=>{Ie.assignArgumentType(re,ye,_e,Ne)},he=(re,ye,_e,Ne)=>{Ie.assignArgumentBitRatio(re,ye,_e,Ne)},Ae=(re,ye,_e)=>{Ie.trackFunctionCall(re,ye,_e)},ne=(re,ye)=>{const _e=[];for(let ft=0;ft<re.params.length;ft++)_e.push(re.params[ft].name);const Ne=new b(ye,Object.assign({},ce,{returnType:null,ast:re,name:re.id.name,argumentNames:_e,lookupReturnType:X,lookupFunctionArgumentTypes:ie,lookupFunctionArgumentName:se,lookupFunctionArgumentBitRatio:J,needsArgumentType:te,assignArgumentType:ee,triggerImplyArgumentType:ae,triggerImplyArgumentBitRatio:he,onFunctionCall:Ae}));Ne.traceFunctionAST(re),Ie.addFunctionNode(Ne)},ce=Object.assign({isRootKernel:!1,onNestedFunction:ne,lookupReturnType:X,lookupFunctionArgumentTypes:ie,lookupFunctionArgumentName:se,lookupFunctionArgumentBitRatio:J,needsArgumentType:te,assignArgumentType:ee,triggerImplyArgumentType:ae,triggerImplyArgumentBitRatio:he,onFunctionCall:Ae,optimizeFloatMemory:$,precision:P,constants:g,constantTypes:N,constantBitRatios:E,debug:w,loopMaxIterations:y,output:v,plugins:O,dynamicArguments:R,dynamicOutput:V},l||{}),Ve=Object.assign({},ce,{isRootKernel:!0,name:"kernel",argumentNames:m,argumentTypes:W,argumentSizes:p,argumentBitRatios:u,leadingReturnStatement:F,followingReturnStatement:L});if(typeof T=="object"&&T.functionNodes)return new Cr().fromJSON(T.functionNodes,b);const ue=new b(T,Ve);let Te=null;f&&(Te=f.map(re=>new b(re.source,{returnType:re.returnType,argumentTypes:re.argumentTypes,output:v,plugins:O,constants:g,constantTypes:N,constantBitRatios:E,optimizeFloatMemory:$,precision:P,lookupReturnType:X,lookupFunctionArgumentTypes:ie,lookupFunctionArgumentName:se,lookupFunctionArgumentBitRatio:J,needsArgumentType:te,assignArgumentType:ee,triggerImplyArgumentType:ae,triggerImplyArgumentBitRatio:he,onFunctionCall:Ae,onNestedFunction:ne})));let Oe=null;A&&(Oe=A.map(re=>{const{name:ye,source:_e}=re;return new b(_e,Object.assign({},ce,{name:ye,isSubKernel:!0,isRootKernel:!1}))}));const Ie=new Cr({kernel:_,rootNode:ue,functionNodes:Te,nativeFunctions:k,subKernelNodes:Oe});return Ie}constructor(_){if(_=_||{},this.kernel=_.kernel,this.rootNode=_.rootNode,this.functionNodes=_.functionNodes||[],this.subKernelNodes=_.subKernelNodes||[],this.nativeFunctions=_.nativeFunctions||[],this.functionMap={},this.nativeFunctionNames=[],this.lookupChain=[],this.functionNodeDependencies={},this.functionCalls={},this.rootNode&&(this.functionMap.kernel=this.rootNode),this.functionNodes)for(let b=0;b<this.functionNodes.length;b++)this.functionMap[this.functionNodes[b].name]=this.functionNodes[b];if(this.subKernelNodes)for(let b=0;b<this.subKernelNodes.length;b++)this.functionMap[this.subKernelNodes[b].name]=this.subKernelNodes[b];if(this.nativeFunctions)for(let b=0;b<this.nativeFunctions.length;b++){const l=this.nativeFunctions[b];this.nativeFunctionNames.push(l.name)}}addFunctionNode(_){if(!_.name)throw new Error("functionNode.name needs set");this.functionMap[_.name]=_,_.isRootKernel&&(this.rootNode=_)}traceFunctionCalls(_,b){if(_=_||"kernel",b=b||[],this.nativeFunctionNames.indexOf(_)>-1){const d=b.indexOf(_);if(d===-1)b.push(_);else{const C=b.splice(d,1)[0];b.push(C)}return b}const l=this.functionMap[_];if(l){const d=b.indexOf(_);if(d===-1){b.push(_),l.toString();for(let C=0;C<l.calledFunctions.length;++C)this.traceFunctionCalls(l.calledFunctions[C],b)}else{const C=b.splice(d,1)[0];b.push(C)}}return b}getPrototypeString(_){return this.getPrototypes(_).join(`
`)}getPrototypes(_){return this.rootNode&&this.rootNode.toString(),_?this.getPrototypesFromFunctionNames(this.traceFunctionCalls(_,[]).reverse()):this.getPrototypesFromFunctionNames(Object.keys(this.functionMap))}getStringFromFunctionNames(_){const b=[];for(let l=0;l<_.length;++l)this.functionMap[_[l]]&&b.push(this.functionMap[_[l]].toString());return b.join(`
`)}getPrototypesFromFunctionNames(_){const b=[];for(let l=0;l<_.length;++l){const d=_[l],C=this.nativeFunctionNames.indexOf(d);if(C>-1){b.push(this.nativeFunctions[C].source);continue}const m=this.functionMap[d];m&&b.push(m.toString())}return b}toJSON(){return this.traceFunctionCalls(this.rootNode.name).reverse().map(_=>{const b=this.nativeFunctions.indexOf(_);if(b>-1)return{name:_,source:this.nativeFunctions[b].source};if(this.functionMap[_])return this.functionMap[_].toJSON();throw new Error(`function ${_} not found`)})}fromJSON(_,b){this.functionMap={};for(let l=0;l<_.length;l++){const d=_[l];this.functionMap[d.settings.name]=new b(d.ast,d.settings)}return this}getString(_){return _?this.getStringFromFunctionNames(this.traceFunctionCalls(_).reverse()):this.getStringFromFunctionNames(Object.keys(this.functionMap))}lookupReturnType(_,b,l){if(b.type!=="CallExpression")throw new Error(`expected ast type of "CallExpression", but is ${b.type}`);if(this._isNativeFunction(_))return this._lookupNativeFunctionReturnType(_);if(this._isFunction(_)){const d=this._getFunction(_);if(d.returnType)return d.returnType;{for(let m=0;m<this.lookupChain.length;m++)if(this.lookupChain[m].ast===b){if(d.argumentTypes.length===0&&b.arguments.length>0){const p=b.arguments;for(let u=0;u<p.length;u++)this.lookupChain.push({name:l.name,ast:p[m],requestingNode:l}),d.argumentTypes[u]=l.getType(p[u]),this.lookupChain.pop();return d.returnType=d.getType(d.getJsAST())}throw new Error("circlical logic detected!")}this.lookupChain.push({name:l.name,ast:b,requestingNode:l});const C=d.getType(d.getJsAST());return this.lookupChain.pop(),d.returnType=C}}return null}_getFunction(_){return this._isFunction(_),this.functionMap[_]}_isFunction(_){return!!this.functionMap[_]}_getNativeFunction(_){for(let b=0;b<this.nativeFunctions.length;b++)if(this.nativeFunctions[b].name===_)return this.nativeFunctions[b];return null}_isNativeFunction(_){return!!this._getNativeFunction(_)}_lookupNativeFunctionReturnType(_){let b=this._getNativeFunction(_);if(b)return b.returnType;throw new Error(`Native function ${_} not found`)}lookupFunctionArgumentTypes(_){return this._isNativeFunction(_)?this._getNativeFunction(_).argumentTypes:this._isFunction(_)?this._getFunction(_).argumentTypes:null}lookupFunctionArgumentName(_,b){return this._getFunction(_).argumentNames[b]}lookupFunctionArgumentBitRatio(_,b){if(!this._isFunction(_))throw new Error("function not found");if(this.rootNode.name===_){const m=this.rootNode.argumentNames.indexOf(b);if(m!==-1)return this.rootNode.argumentBitRatios[m]}const l=this._getFunction(_),d=l.argumentNames.indexOf(b);if(d===-1)throw new Error("argument not found");const C=l.argumentBitRatios[d];if(typeof C!="number")throw new Error("argument bit ratio not found");return C}needsArgumentType(_,b){return this._isFunction(_)?!this._getFunction(_).argumentTypes[b]:!1}assignArgumentType(_,b,l,d){if(!this._isFunction(_))return;const C=this._getFunction(_);C.argumentTypes[b]||(C.argumentTypes[b]=l)}assignArgumentBitRatio(_,b,l,d){const C=this._getFunction(_);if(this._isNativeFunction(l))return null;const m=this._getFunction(l),p=C.argumentNames.indexOf(b);if(p===-1)throw new Error(`Argument ${b} not found in arguments from function ${_}`);const u=C.argumentBitRatios[p];if(typeof u!="number")throw new Error(`Bit ratio for argument ${b} not found in function ${_}`);m.argumentBitRatios||(m.argumentBitRatios=new Array(m.argumentNames.length));const g=m.argumentBitRatios[d];if(typeof g=="number"){if(g!==u)throw new Error(`Incompatible bit ratio found at function ${_} at argument ${b}`);return g}return m.argumentBitRatios[d]=u,u}trackFunctionCall(_,b,l){this.functionNodeDependencies[_]||(this.functionNodeDependencies[_]=new Set,this.functionCalls[_]=[]),this.functionNodeDependencies[_].add(b),this.functionCalls[_].push(l)}getKernelResultType(){return this.rootNode.returnType||this.rootNode.getType(this.rootNode.ast)}getSubKernelResultType(_){const b=this.subKernelNodes[_];let l=!1;for(let d=0;d<this.rootNode.functionCalls.length;d++)this.rootNode.functionCalls[d].ast.callee.name===b.name&&(l=!0);if(!l)throw new Error(`SubKernel ${b.name} never called by kernel`);return b.returnType||b.getType(b.getJsAST())}getReturnTypes(){const _={[this.rootNode.name]:this.rootNode.getType(this.rootNode.ast)},b=this.traceFunctionCalls(this.rootNode.name);for(let l=0;l<b.length;l++){const d=b[l],C=this.functionMap[d];_[d]=C.getType(C.ast)}return _}}}}),j=s((B,z)=>{const{utils:M}=h();function _(d){return d.length>0?d[d.length-1]:null}const b={trackIdentifiers:"trackIdentifiers",memberExpression:"memberExpression",inForLoopInit:"inForLoopInit"};var l=class{constructor(d){this.runningContexts=[],this.functionContexts=[],this.contexts=[],this.functionCalls=[],this.declarations=[],this.identifiers=[],this.functions=[],this.returnStatements=[],this.trackedIdentifiers=null,this.states=[],this.newFunctionContext(),this.scan(d)}isState(d){return this.states[this.states.length-1]===d}hasState(d){return this.states.indexOf(d)>-1}pushState(d){this.states.push(d)}popState(d){if(this.isState(d))this.states.pop();else throw new Error(`Cannot pop the non-active state "${d}"`)}get currentFunctionContext(){return _(this.functionContexts)}get currentContext(){return _(this.runningContexts)}newFunctionContext(){const d={"@contextType":"function"};this.contexts.push(d),this.functionContexts.push(d)}newContext(d){const C=Object.assign({"@contextType":"const/let"},this.currentContext);this.contexts.push(C),this.runningContexts.push(C),d();const{currentFunctionContext:m}=this;for(const p in m)!m.hasOwnProperty(p)||C.hasOwnProperty(p)||(C[p]=m[p]);return this.runningContexts.pop(),C}useFunctionContext(d){const C=_(this.functionContexts);this.runningContexts.push(C),d(),this.runningContexts.pop()}getIdentifiers(d){const C=this.trackedIdentifiers=[];return this.pushState(b.trackIdentifiers),d(),this.trackedIdentifiers=null,this.popState(b.trackIdentifiers),C}getDeclaration(d){const{currentContext:C,currentFunctionContext:m,runningContexts:p}=this,u=C[d]||m[d]||null;if(!u&&C===m&&p.length>0){const g=p[p.length-2];if(g[d])return g[d]}return u}scan(d){if(d){if(Array.isArray(d)){for(let C=0;C<d.length;C++)this.scan(d[C]);return}switch(d.type){case"Program":this.useFunctionContext(()=>{this.scan(d.body)});break;case"BlockStatement":this.newContext(()=>{this.scan(d.body)});break;case"AssignmentExpression":case"LogicalExpression":this.scan(d.left),this.scan(d.right);break;case"BinaryExpression":this.scan(d.left),this.scan(d.right);break;case"UpdateExpression":if(d.operator==="++"){const C=this.getDeclaration(d.argument.name);C&&(C.suggestedType="Integer")}this.scan(d.argument);break;case"UnaryExpression":this.scan(d.argument);break;case"VariableDeclaration":d.kind==="var"?this.useFunctionContext(()=>{d.declarations=M.normalizeDeclarations(d),this.scan(d.declarations)}):(d.declarations=M.normalizeDeclarations(d),this.scan(d.declarations));break;case"VariableDeclarator":{const{currentContext:C}=this,m=this.hasState(b.inForLoopInit),p={ast:d,context:C,name:d.id.name,origin:"declaration",inForLoopInit:m,inForLoopTest:null,assignable:C===this.currentFunctionContext||!m&&!C.hasOwnProperty(d.id.name),suggestedType:null,valueType:null,dependencies:null,isSafe:null};C[d.id.name]||(C[d.id.name]=p),this.declarations.push(p),this.scan(d.id),this.scan(d.init);break}case"FunctionExpression":case"FunctionDeclaration":this.runningContexts.length===0?this.scan(d.body):this.functions.push(d);break;case"IfStatement":this.scan(d.test),this.scan(d.consequent),d.alternate&&this.scan(d.alternate);break;case"ForStatement":{let C;const m=this.newContext(()=>{this.pushState(b.inForLoopInit),this.scan(d.init),this.popState(b.inForLoopInit),C=this.getIdentifiers(()=>{this.scan(d.test)}),this.scan(d.update),this.newContext(()=>{this.scan(d.body)})});if(C)for(const p in m)p!=="@contextType"&&C.indexOf(p)>-1&&(m[p].inForLoopTest=!0);break}case"DoWhileStatement":case"WhileStatement":this.newContext(()=>{this.scan(d.body),this.scan(d.test)});break;case"Identifier":this.isState(b.trackIdentifiers)&&this.trackedIdentifiers.push(d.name),this.identifiers.push({context:this.currentContext,declaration:this.getDeclaration(d.name),ast:d});break;case"ReturnStatement":this.returnStatements.push(d),this.scan(d.argument);break;case"MemberExpression":this.pushState(b.memberExpression),this.scan(d.object),this.scan(d.property),this.popState(b.memberExpression);break;case"ExpressionStatement":this.scan(d.expression);break;case"SequenceExpression":this.scan(d.expressions);break;case"CallExpression":this.functionCalls.push({context:this.currentContext,ast:d}),this.scan(d.arguments);break;case"ArrayExpression":this.scan(d.elements);break;case"ConditionalExpression":this.scan(d.test),this.scan(d.alternate),this.scan(d.consequent);break;case"SwitchStatement":this.scan(d.discriminant),this.scan(d.cases);break;case"SwitchCase":this.scan(d.test),this.scan(d.consequent);break;case"ThisExpression":case"Literal":case"DebuggerStatement":case"EmptyStatement":case"BreakStatement":case"ContinueStatement":break;default:throw new Error(`unhandled type "${d.type}"`)}}}};z.exports={FunctionTracer:l}}),q=s((B,z)=>{const M=i(),{utils:_}=h(),{FunctionTracer:b}=j(),l=["E","PI","SQRT2","SQRT1_2","LN2","LN10","LOG2E","LOG10E"],d=["abs","acos","acosh","asin","asinh","atan","atan2","atanh","cbrt","ceil","clz32","cos","cosh","expm1","exp","floor","fround","imul","log","log2","log10","log1p","max","min","pow","random","round","sign","sin","sinh","sqrt","tan","tanh","trunc"],C=["value","value[]","value[][]","value[][][]","value[][][][]","value.value","value.thread.value","this.thread.value","this.output.value","this.constants.value","this.constants.value[]","this.constants.value[][]","this.constants.value[][][]","this.constants.value[][][][]","fn()[]","fn()[][]","fn()[][][]","[][]"];var m=class{constructor(u,g){if(!u&&!g.ast)throw new Error("source parameter is missing");if(g=g||{},this.source=u,this.ast=null,this.name=typeof u=="string"?g.isRootKernel?"kernel":g.name||_.getFunctionNameFromString(u):null,this.calledFunctions=[],this.constants={},this.constantTypes={},this.constantBitRatios={},this.isRootKernel=!1,this.isSubKernel=!1,this.debug=null,this.functions=null,this.identifiers=null,this.contexts=null,this.functionCalls=null,this.states=[],this.needsArgumentType=null,this.assignArgumentType=null,this.lookupReturnType=null,this.lookupFunctionArgumentTypes=null,this.lookupFunctionArgumentBitRatio=null,this.triggerImplyArgumentType=null,this.triggerImplyArgumentBitRatio=null,this.onNestedFunction=null,this.onFunctionCall=null,this.optimizeFloatMemory=null,this.precision=null,this.loopMaxIterations=null,this.argumentNames=typeof this.source=="string"?_.getArgumentNamesFromString(this.source):null,this.argumentTypes=[],this.argumentSizes=[],this.argumentBitRatios=null,this.returnType=null,this.output=[],this.plugins=null,this.leadingReturnStatement=null,this.followingReturnStatement=null,this.dynamicOutput=null,this.dynamicArguments=null,this.strictTypingChecking=!1,this.fixIntegerDivisionAccuracy=null,g)for(const E in g)g.hasOwnProperty(E)&&this.hasOwnProperty(E)&&(this[E]=g[E]);this.literalTypes={},this.validate(),this._string=null,this._internalVariableNames={}}validate(){if(typeof this.source!="string"&&!this.ast)throw new Error("this.source not a string");if(!this.ast&&!_.isFunctionString(this.source))throw new Error("this.source not a function string");if(!this.name)throw new Error("this.name could not be set");if(this.argumentTypes.length>0&&this.argumentTypes.length!==this.argumentNames.length)throw new Error(`argumentTypes count of ${this.argumentTypes.length} exceeds ${this.argumentNames.length}`);if(this.output.length<1)throw new Error("this.output is not big enough")}isIdentifierConstant(u){return this.constants?this.constants.hasOwnProperty(u):!1}isInput(u){return this.argumentTypes[this.argumentNames.indexOf(u)]==="Input"}pushState(u){this.states.push(u)}popState(u){if(this.state!==u)throw new Error(`Cannot popState ${u} when in ${this.state}`);this.states.pop()}isState(u){return this.state===u}get state(){return this.states[this.states.length-1]}astMemberExpressionUnroll(u){if(u.type==="Identifier")return u.name;if(u.type==="ThisExpression")return"this";if(u.type==="MemberExpression"&&u.object&&u.property)return u.object.hasOwnProperty("name")&&u.object.name!=="Math"?this.astMemberExpressionUnroll(u.property):this.astMemberExpressionUnroll(u.object)+"."+this.astMemberExpressionUnroll(u.property);if(u.hasOwnProperty("expressions")){const g=u.expressions[0];if(g.type==="Literal"&&g.value===0&&u.expressions.length===2)return this.astMemberExpressionUnroll(u.expressions[1])}throw this.astErrorOutput("Unknown astMemberExpressionUnroll",u)}getJsAST(u){if(this.ast)return this.ast;if(typeof this.source=="object")return this.traceFunctionAST(this.source),this.ast=this.source;if(u=u||M,u===null)throw new Error("Missing JS to AST parser");const g=Object.freeze(u.parse(`const parser_${this.name} = ${this.source};`,{locations:!0,ecmaVersion:2020})),E=g.body[0].declarations[0].init;if(this.traceFunctionAST(E),!g)throw new Error("Failed to parse JS code");return this.ast=E}traceFunctionAST(u){const{contexts:g,declarations:E,functions:w,identifiers:y,functionCalls:k}=new b(u);this.contexts=g,this.identifiers=y,this.functionCalls=k,this.functions=w;for(let v=0;v<E.length;v++){const $=E[v],{ast:P,inForLoopInit:O,inForLoopTest:T}=$,{init:A}=P,f=this.getDependencies(A);let F=null;if(O&&T)F="Integer";else if(A){const L=this.getType(A);switch(L){case"Integer":case"Float":case"Number":A.type==="MemberExpression"?F=L:F="Number";break;case"LiteralInteger":F="Number";break;default:F=L}}$.valueType=F,$.dependencies=f,$.isSafe=this.isSafeDependencies(f)}for(let v=0;v<w.length;v++)this.onNestedFunction(w[v],this.source)}getDeclaration(u){for(let g=0;g<this.identifiers.length;g++){const E=this.identifiers[g];if(u===E.ast)return E.declaration}return null}getVariableType(u){if(u.type!=="Identifier")throw new Error(`ast of ${u.type} not "Identifier"`);let g=null;const E=this.argumentNames.indexOf(u.name);if(E===-1){const w=this.getDeclaration(u);if(w)return w.valueType}else{const w=this.argumentTypes[E];w&&(g=w)}if(!g&&this.strictTypingChecking)throw new Error(`Declaration of ${name} not found`);return g}getLookupType(u){if(!p.hasOwnProperty(u))throw new Error(`unknown typeLookupMap ${u}`);return p[u]}getConstantType(u){if(this.constantTypes[u]){const g=this.constantTypes[u];return g==="Float"?"Number":g}throw new Error(`Type for constant "${u}" not declared`)}toString(){return this._string?this._string:this._string=this.astGeneric(this.getJsAST(),[]).join("").trim()}toJSON(){const u={source:this.source,name:this.name,constants:this.constants,constantTypes:this.constantTypes,isRootKernel:this.isRootKernel,isSubKernel:this.isSubKernel,debug:this.debug,output:this.output,loopMaxIterations:this.loopMaxIterations,argumentNames:this.argumentNames,argumentTypes:this.argumentTypes,argumentSizes:this.argumentSizes,returnType:this.returnType,leadingReturnStatement:this.leadingReturnStatement,followingReturnStatement:this.followingReturnStatement};return{ast:this.ast,settings:u}}getType(u){if(Array.isArray(u))return this.getType(u[u.length-1]);switch(u.type){case"BlockStatement":return this.getType(u.body);case"ArrayExpression":switch(this.getType(u.elements[0])){case"Array(2)":case"Array(3)":case"Array(4)":return`Matrix(${u.elements.length})`}return`Array(${u.elements.length})`;case"Literal":const g=this.astKey(u);return this.literalTypes[g]?this.literalTypes[g]:Number.isInteger(u.value)?"LiteralInteger":u.value===!0||u.value===!1?"Boolean":"Number";case"AssignmentExpression":return this.getType(u.left);case"CallExpression":if(this.isAstMathFunction(u))return"Number";if(!u.callee||!u.callee.name){if(u.callee.type==="SequenceExpression"&&u.callee.expressions[u.callee.expressions.length-1].property.name){const v=u.callee.expressions[u.callee.expressions.length-1].property.name;return this.inferArgumentTypesIfNeeded(v,u.arguments),this.lookupReturnType(v,u,this)}if(this.getVariableSignature(u.callee,!0)==="this.color")return null;if(u.callee.type==="MemberExpression"&&u.callee.object&&u.callee.property&&u.callee.property.name&&u.arguments){const v=u.callee.property.name;return this.inferArgumentTypesIfNeeded(v,u.arguments),this.lookupReturnType(v,u,this)}throw this.astErrorOutput("Unknown call expression",u)}if(u.callee&&u.callee.name){const v=u.callee.name;return this.inferArgumentTypesIfNeeded(v,u.arguments),this.lookupReturnType(v,u,this)}throw this.astErrorOutput(`Unhandled getType Type "${u.type}"`,u);case"LogicalExpression":return"Boolean";case"BinaryExpression":switch(u.operator){case"%":return"Number";case"/":return"Number";case">":case"<":return"Boolean";case"&":case"|":case"^":case"<<":case">>":case">>>":return"Integer"}const E=this.getType(u.left);if(this.isState("skip-literal-correction"))return E;if(E==="LiteralInteger"){const v=this.getType(u.right);return v==="LiteralInteger"?u.left.value%1===0?"Integer":"Float":v}return p[E]||E;case"UpdateExpression":return this.getType(u.argument);case"UnaryExpression":return u.operator==="~"?"Integer":this.getType(u.argument);case"VariableDeclaration":{const v=u.declarations;let $;for(let P=0;P<v.length;P++){const O=v[P];$=this.getType(O)}if(!$)throw this.astErrorOutput("Unable to find type for declaration",u);return $}case"VariableDeclarator":const w=this.getDeclaration(u.id);if(!w)throw this.astErrorOutput("Unable to find declarator",u);if(!w.valueType)throw this.astErrorOutput("Unable to find declarator valueType",u);return w.valueType;case"Identifier":if(u.name==="Infinity")return"Number";if(this.isAstVariable(u)&&this.getVariableSignature(u)==="value")return this.getCheckVariableType(u);const y=this.findIdentifierOrigin(u);return y&&y.init?this.getType(y.init):null;case"ReturnStatement":return this.getType(u.argument);case"MemberExpression":if(this.isAstMathFunction(u)){switch(u.property.name){case"ceil":return"Integer";case"floor":return"Integer";case"round":return"Integer"}return"Number"}if(this.isAstVariable(u)){switch(this.getVariableSignature(u)){case"value[]":return this.getLookupType(this.getCheckVariableType(u.object));case"value[][]":return this.getLookupType(this.getCheckVariableType(u.object.object));case"value[][][]":return this.getLookupType(this.getCheckVariableType(u.object.object.object));case"value[][][][]":return this.getLookupType(this.getCheckVariableType(u.object.object.object.object));case"value.thread.value":case"this.thread.value":return"Integer";case"this.output.value":return this.dynamicOutput?"Integer":"LiteralInteger";case"this.constants.value":return this.getConstantType(u.property.name);case"this.constants.value[]":return this.getLookupType(this.getConstantType(u.object.property.name));case"this.constants.value[][]":return this.getLookupType(this.getConstantType(u.object.object.property.name));case"this.constants.value[][][]":return this.getLookupType(this.getConstantType(u.object.object.object.property.name));case"this.constants.value[][][][]":return this.getLookupType(this.getConstantType(u.object.object.object.object.property.name));case"fn()[]":case"fn()[][]":case"fn()[][][]":return this.getLookupType(this.getType(u.object));case"value.value":if(this.isAstMathVariable(u))return"Number";switch(u.property.name){case"r":case"g":case"b":case"a":return this.getLookupType(this.getCheckVariableType(u.object))}case"[][]":return"Number"}throw this.astErrorOutput("Unhandled getType MemberExpression",u)}throw this.astErrorOutput("Unhandled getType MemberExpression",u);case"ConditionalExpression":return this.getType(u.consequent);case"FunctionDeclaration":case"FunctionExpression":const k=this.findLastReturn(u.body);return k?this.getType(k):null;case"IfStatement":return this.getType(u.consequent);case"SequenceExpression":return this.getType(u.expressions[u.expressions.length-1]);default:throw this.astErrorOutput(`Unhandled getType Type "${u.type}"`,u)}}getCheckVariableType(u){const g=this.getVariableType(u);if(!g)throw this.astErrorOutput(`${u.type} is not defined`,u);return g}inferArgumentTypesIfNeeded(u,g){for(let E=0;E<g.length;E++){if(!this.needsArgumentType(u,E))continue;const w=this.getType(g[E]);if(!w)throw this.astErrorOutput(`Unable to infer argument ${E}`,g[E]);this.assignArgumentType(u,E,w)}}isAstMathVariable(u){return u.type==="MemberExpression"&&u.object&&u.object.type==="Identifier"&&u.object.name==="Math"&&u.property&&u.property.type==="Identifier"&&l.includes(u.property.name)}isAstMathFunction(u){return u.type==="CallExpression"&&u.callee&&u.callee.type==="MemberExpression"&&u.callee.object&&u.callee.object.type==="Identifier"&&u.callee.object.name==="Math"&&u.callee.property&&u.callee.property.type==="Identifier"&&d.includes(u.callee.property.name)}isAstVariable(u){return u.type==="Identifier"||u.type==="MemberExpression"}isSafe(u){return this.isSafeDependencies(this.getDependencies(u))}isSafeDependencies(u){return u&&u.every?u.every(g=>g.isSafe):!0}getDependencies(u,g,E){if(g||(g=[]),!u)return null;if(Array.isArray(u)){for(let w=0;w<u.length;w++)this.getDependencies(u[w],g,E);return g}switch(u.type){case"AssignmentExpression":return this.getDependencies(u.left,g,E),this.getDependencies(u.right,g,E),g;case"ConditionalExpression":return this.getDependencies(u.test,g,E),this.getDependencies(u.alternate,g,E),this.getDependencies(u.consequent,g,E),g;case"Literal":g.push({origin:"literal",value:u.value,isSafe:E===!0?!1:u.value>-1/0&&u.value<1/0&&!isNaN(u.value)});break;case"VariableDeclarator":return this.getDependencies(u.init,g,E);case"Identifier":const w=this.getDeclaration(u);if(w)g.push({name:u.name,origin:"declaration",isSafe:E?!1:this.isSafeDependencies(w.dependencies)});else if(this.argumentNames.indexOf(u.name)>-1)g.push({name:u.name,origin:"argument",isSafe:!1});else if(this.strictTypingChecking)throw new Error(`Cannot find identifier origin "${u.name}"`);break;case"FunctionDeclaration":return this.getDependencies(u.body.body[u.body.body.length-1],g,E);case"ReturnStatement":return this.getDependencies(u.argument,g);case"BinaryExpression":case"LogicalExpression":return E=u.operator==="/"||u.operator==="*",this.getDependencies(u.left,g,E),this.getDependencies(u.right,g,E),g;case"UnaryExpression":case"UpdateExpression":return this.getDependencies(u.argument,g,E);case"VariableDeclaration":return this.getDependencies(u.declarations,g,E);case"ArrayExpression":return g.push({origin:"declaration",isSafe:!0}),g;case"CallExpression":return g.push({origin:"function",isSafe:!0}),g;case"MemberExpression":const y=this.getMemberExpressionDetails(u);switch(y.signature){case"value[]":this.getDependencies(u.object,g,E);break;case"value[][]":this.getDependencies(u.object.object,g,E);break;case"value[][][]":this.getDependencies(u.object.object.object,g,E);break;case"this.output.value":this.dynamicOutput&&g.push({name:y.name,origin:"output",isSafe:!1});break}if(y)return y.property&&this.getDependencies(y.property,g,E),y.xProperty&&this.getDependencies(y.xProperty,g,E),y.yProperty&&this.getDependencies(y.yProperty,g,E),y.zProperty&&this.getDependencies(y.zProperty,g,E),g;case"SequenceExpression":return this.getDependencies(u.expressions,g,E);default:throw this.astErrorOutput(`Unhandled type ${u.type} in getDependencies`,u)}return g}getVariableSignature(u,g){if(!this.isAstVariable(u))throw new Error(`ast of type "${u.type}" is not a variable signature`);if(u.type==="Identifier")return"value";const E=[];for(;u;)u.computed?E.push("[]"):u.type==="ThisExpression"?E.unshift("this"):u.property&&u.property.name?u.property.name==="x"||u.property.name==="y"||u.property.name==="z"?E.unshift(g?"."+u.property.name:".value"):u.property.name==="constants"||u.property.name==="thread"||u.property.name==="output"?E.unshift("."+u.property.name):E.unshift(g?"."+u.property.name:".value"):u.name?E.unshift(g?u.name:"value"):u.callee&&u.callee.name?E.unshift(g?u.callee.name+"()":"fn()"):u.elements?E.unshift("[]"):E.unshift("unknown"),u=u.object;const w=E.join("");return g||C.includes(w)?w:null}build(){return this.toString().length>0}astGeneric(u,g){if(u===null)throw this.astErrorOutput("NULL ast",u);if(Array.isArray(u)){for(let E=0;E<u.length;E++)this.astGeneric(u[E],g);return g}switch(u.type){case"FunctionDeclaration":return this.astFunctionDeclaration(u,g);case"FunctionExpression":return this.astFunctionExpression(u,g);case"ReturnStatement":return this.astReturnStatement(u,g);case"Literal":return this.astLiteral(u,g);case"BinaryExpression":return this.astBinaryExpression(u,g);case"Identifier":return this.astIdentifierExpression(u,g);case"AssignmentExpression":return this.astAssignmentExpression(u,g);case"ExpressionStatement":return this.astExpressionStatement(u,g);case"EmptyStatement":return this.astEmptyStatement(u,g);case"BlockStatement":return this.astBlockStatement(u,g);case"IfStatement":return this.astIfStatement(u,g);case"SwitchStatement":return this.astSwitchStatement(u,g);case"BreakStatement":return this.astBreakStatement(u,g);case"ContinueStatement":return this.astContinueStatement(u,g);case"ForStatement":return this.astForStatement(u,g);case"WhileStatement":return this.astWhileStatement(u,g);case"DoWhileStatement":return this.astDoWhileStatement(u,g);case"VariableDeclaration":return this.astVariableDeclaration(u,g);case"VariableDeclarator":return this.astVariableDeclarator(u,g);case"ThisExpression":return this.astThisExpression(u,g);case"SequenceExpression":return this.astSequenceExpression(u,g);case"UnaryExpression":return this.astUnaryExpression(u,g);case"UpdateExpression":return this.astUpdateExpression(u,g);case"LogicalExpression":return this.astLogicalExpression(u,g);case"MemberExpression":return this.astMemberExpression(u,g);case"CallExpression":return this.astCallExpression(u,g);case"ArrayExpression":return this.astArrayExpression(u,g);case"DebuggerStatement":return this.astDebuggerStatement(u,g);case"ConditionalExpression":return this.astConditionalExpression(u,g)}throw this.astErrorOutput("Unknown ast type : "+u.type,u)}astErrorOutput(u,g){if(typeof this.source!="string")return new Error(u);const E=_.getAstString(this.source,g),w=this.source.slice(g.start).split(/\n/),y=w.length>0?w[w.length-1]:0;return new Error(`${u} on line ${w.length}, position ${y.length}:
 ${E}`)}astDebuggerStatement(u,g){return g}astConditionalExpression(u,g){if(u.type!=="ConditionalExpression")throw this.astErrorOutput("Not a conditional expression",u);return g.push("("),this.astGeneric(u.test,g),g.push("?"),this.astGeneric(u.consequent,g),g.push(":"),this.astGeneric(u.alternate,g),g.push(")"),g}astFunction(u,g){throw new Error(`"astFunction" not defined on ${this.constructor.name}`)}astFunctionDeclaration(u,g){return this.isChildFunction(u)?g:this.astFunction(u,g)}astFunctionExpression(u,g){return this.isChildFunction(u)?g:this.astFunction(u,g)}isChildFunction(u){for(let g=0;g<this.functions.length;g++)if(this.functions[g]===u)return!0;return!1}astReturnStatement(u,g){return g}astLiteral(u,g){return this.literalTypes[this.astKey(u)]="Number",g}astBinaryExpression(u,g){return g}astIdentifierExpression(u,g){return g}astAssignmentExpression(u,g){return g}astExpressionStatement(u,g){return this.astGeneric(u.expression,g),g.push(";"),g}astEmptyStatement(u,g){return g}astBlockStatement(u,g){return g}astIfStatement(u,g){return g}astSwitchStatement(u,g){return g}astBreakStatement(u,g){return g.push("break;"),g}astContinueStatement(u,g){return g.push(`continue;
`),g}astForStatement(u,g){return g}astWhileStatement(u,g){return g}astDoWhileStatement(u,g){return g}astVariableDeclarator(u,g){return this.astGeneric(u.id,g),u.init!==null&&(g.push("="),this.astGeneric(u.init,g)),g}astThisExpression(u,g){return g}astSequenceExpression(u,g){const{expressions:E}=u,w=[];for(let y=0;y<E.length;y++){const k=E[y],v=[];this.astGeneric(k,v),w.push(v.join(""))}return w.length>1?g.push("(",w.join(","),")"):g.push(w[0]),g}astUnaryExpression(u,g){return this.checkAndUpconvertBitwiseUnary(u,g)||(u.prefix?(g.push(u.operator),this.astGeneric(u.argument,g)):(this.astGeneric(u.argument,g),g.push(u.operator))),g}checkAndUpconvertBitwiseUnary(u,g){}astUpdateExpression(u,g){return u.prefix?(g.push(u.operator),this.astGeneric(u.argument,g)):(this.astGeneric(u.argument,g),g.push(u.operator)),g}astLogicalExpression(u,g){return g.push("("),this.astGeneric(u.left,g),g.push(u.operator),this.astGeneric(u.right,g),g.push(")"),g}astMemberExpression(u,g){return g}astCallExpression(u,g){return g}astArrayExpression(u,g){return g}getMemberExpressionDetails(u){if(u.type!=="MemberExpression")throw this.astErrorOutput(`Expression ${u.type} not a MemberExpression`,u);let g=null,E=null;const w=this.getVariableSignature(u);switch(w){case"value":return null;case"value.thread.value":case"this.thread.value":case"this.output.value":return{signature:w,type:"Integer",name:u.property.name};case"value[]":if(typeof u.object.name!="string")throw this.astErrorOutput("Unexpected expression",u);return g=u.object.name,{name:g,origin:"user",signature:w,type:this.getVariableType(u.object),xProperty:u.property};case"value[][]":if(typeof u.object.object.name!="string")throw this.astErrorOutput("Unexpected expression",u);return g=u.object.object.name,{name:g,origin:"user",signature:w,type:this.getVariableType(u.object.object),yProperty:u.object.property,xProperty:u.property};case"value[][][]":if(typeof u.object.object.object.name!="string")throw this.astErrorOutput("Unexpected expression",u);return g=u.object.object.object.name,{name:g,origin:"user",signature:w,type:this.getVariableType(u.object.object.object),zProperty:u.object.object.property,yProperty:u.object.property,xProperty:u.property};case"value[][][][]":if(typeof u.object.object.object.object.name!="string")throw this.astErrorOutput("Unexpected expression",u);return g=u.object.object.object.object.name,{name:g,origin:"user",signature:w,type:this.getVariableType(u.object.object.object.object),zProperty:u.object.object.property,yProperty:u.object.property,xProperty:u.property};case"value.value":if(typeof u.property.name!="string")throw this.astErrorOutput("Unexpected expression",u);if(this.isAstMathVariable(u))return g=u.property.name,{name:g,origin:"Math",type:"Number",signature:w};switch(u.property.name){case"r":case"g":case"b":case"a":return g=u.object.name,{name:g,property:u.property.name,origin:"user",signature:w,type:"Number"};default:throw this.astErrorOutput("Unexpected expression",u)}case"this.constants.value":if(typeof u.property.name!="string")throw this.astErrorOutput("Unexpected expression",u);if(g=u.property.name,E=this.getConstantType(g),!E)throw this.astErrorOutput("Constant has no type",u);return{name:g,type:E,origin:"constants",signature:w};case"this.constants.value[]":if(typeof u.object.property.name!="string")throw this.astErrorOutput("Unexpected expression",u);if(g=u.object.property.name,E=this.getConstantType(g),!E)throw this.astErrorOutput("Constant has no type",u);return{name:g,type:E,origin:"constants",signature:w,xProperty:u.property};case"this.constants.value[][]":if(typeof u.object.object.property.name!="string")throw this.astErrorOutput("Unexpected expression",u);if(g=u.object.object.property.name,E=this.getConstantType(g),!E)throw this.astErrorOutput("Constant has no type",u);return{name:g,type:E,origin:"constants",signature:w,yProperty:u.object.property,xProperty:u.property};case"this.constants.value[][][]":if(typeof u.object.object.object.property.name!="string")throw this.astErrorOutput("Unexpected expression",u);if(g=u.object.object.object.property.name,E=this.getConstantType(g),!E)throw this.astErrorOutput("Constant has no type",u);return{name:g,type:E,origin:"constants",signature:w,zProperty:u.object.object.property,yProperty:u.object.property,xProperty:u.property};case"fn()[]":case"fn()[][]":case"[][]":return{signature:w,property:u.property};default:throw this.astErrorOutput("Unexpected expression",u)}}findIdentifierOrigin(u){const g=[this.ast];for(;g.length>0;){const E=g[0];if(E.type==="VariableDeclarator"&&E.id&&E.id.name&&E.id.name===u.name)return E;if(g.shift(),E.argument)g.push(E.argument);else if(E.body)g.push(E.body);else if(E.declarations)g.push(E.declarations);else if(Array.isArray(E))for(let w=0;w<E.length;w++)g.push(E[w])}return null}findLastReturn(u){const g=[u||this.ast];for(;g.length>0;){const E=g.pop();if(E.type==="ReturnStatement")return E;if(E.type!=="FunctionDeclaration")if(E.argument)g.push(E.argument);else if(E.body)g.push(E.body);else if(E.declarations)g.push(E.declarations);else if(Array.isArray(E))for(let w=0;w<E.length;w++)g.push(E[w]);else E.consequent?g.push(E.consequent):E.cases&&g.push(E.cases)}return null}getInternalVariableName(u){return this._internalVariableNames.hasOwnProperty(u)||(this._internalVariableNames[u]=0),this._internalVariableNames[u]++,this._internalVariableNames[u]===1?u:u+this._internalVariableNames[u]}astKey(u,g=","){if(!u.start||!u.end)throw new Error("AST start and end needed");return`${u.start}${g}${u.end}`}};const p={Number:"Number",Float:"Float",Integer:"Integer",Array:"Number","Array(2)":"Number","Array(3)":"Number","Array(4)":"Number","Matrix(2)":"Number","Matrix(3)":"Number","Matrix(4)":"Number",Array2D:"Number",Array3D:"Number",Input:"Number",HTMLCanvas:"Array(4)",OffscreenCanvas:"Array(4)",HTMLImage:"Array(4)",ImageBitmap:"Array(4)",ImageData:"Array(4)",HTMLVideo:"Array(4)",HTMLImageArray:"Array(4)",NumberTexture:"Number",MemoryOptimizedNumberTexture:"Number","Array1D(2)":"Array(2)","Array1D(3)":"Array(3)","Array1D(4)":"Array(4)","Array2D(2)":"Array(2)","Array2D(3)":"Array(3)","Array2D(4)":"Array(4)","Array3D(2)":"Array(2)","Array3D(3)":"Array(3)","Array3D(4)":"Array(4)","ArrayTexture(1)":"Number","ArrayTexture(2)":"Array(2)","ArrayTexture(3)":"Array(3)","ArrayTexture(4)":"Array(4)"};z.exports={FunctionNode:m}}),$e=s((B,z)=>{const{FunctionNode:M}=q();var _=class extends M{astFunction(b,l){if(!this.isRootKernel){l.push("function"),l.push(" "),l.push(this.name),l.push("(");for(let d=0;d<this.argumentNames.length;++d){const C=this.argumentNames[d];d>0&&l.push(", "),l.push("user_"),l.push(C)}l.push(`) {
`)}for(let d=0;d<b.body.body.length;++d)this.astGeneric(b.body.body[d],l),l.push(`
`);return this.isRootKernel||l.push(`}
`),l}astReturnStatement(b,l){const d=this.returnType||this.getType(b.argument);return this.returnType||(this.returnType=d),this.isRootKernel?(l.push(this.leadingReturnStatement),this.astGeneric(b.argument,l),l.push(`;
`),l.push(this.followingReturnStatement),l.push(`continue;
`)):this.isSubKernel?(l.push(`subKernelResult_${this.name} = `),this.astGeneric(b.argument,l),l.push(";"),l.push(`return subKernelResult_${this.name};`)):(l.push("return "),this.astGeneric(b.argument,l),l.push(";")),l}astLiteral(b,l){if(isNaN(b.value))throw this.astErrorOutput("Non-numeric literal not supported : "+b.value,b);return l.push(b.value),l}astBinaryExpression(b,l){return l.push("("),this.astGeneric(b.left,l),l.push(b.operator),this.astGeneric(b.right,l),l.push(")"),l}astIdentifierExpression(b,l){if(b.type!=="Identifier")throw this.astErrorOutput("IdentifierExpression - not an Identifier",b);switch(b.name){case"Infinity":l.push("Infinity");break;default:this.constants&&this.constants.hasOwnProperty(b.name)?l.push("constants_"+b.name):l.push("user_"+b.name)}return l}astForStatement(b,l){if(b.type!=="ForStatement")throw this.astErrorOutput("Invalid for statement",b);const d=[],C=[],m=[],p=[];let u=null;if(b.init){this.pushState("in-for-loop-init"),this.astGeneric(b.init,d);for(let g=0;g<d.length;g++)d[g].includes&&d[g].includes(",")&&(u=!1);this.popState("in-for-loop-init")}else u=!1;if(b.test?this.astGeneric(b.test,C):u=!1,b.update?this.astGeneric(b.update,m):u=!1,b.body&&(this.pushState("loop-body"),this.astGeneric(b.body,p),this.popState("loop-body")),u===null&&(u=this.isSafe(b.init)&&this.isSafe(b.test)),u)l.push(`for (${d.join("")};${C.join("")};${m.join("")}){
`),l.push(p.join("")),l.push(`}
`);else{const g=this.getInternalVariableName("safeI");d.length>0&&l.push(d.join(""),`;
`),l.push(`for (let ${g}=0;${g}<LOOP_MAX;${g}++){
`),C.length>0&&l.push(`if (!${C.join("")}) break;
`),l.push(p.join("")),l.push(`
${m.join("")};`),l.push(`}
`)}return l}astWhileStatement(b,l){if(b.type!=="WhileStatement")throw this.astErrorOutput("Invalid while statement",b);return l.push("for (let i = 0; i < LOOP_MAX; i++) {"),l.push("if ("),this.astGeneric(b.test,l),l.push(`) {
`),this.astGeneric(b.body,l),l.push(`} else {
`),l.push(`break;
`),l.push(`}
`),l.push(`}
`),l}astDoWhileStatement(b,l){if(b.type!=="DoWhileStatement")throw this.astErrorOutput("Invalid while statement",b);return l.push("for (let i = 0; i < LOOP_MAX; i++) {"),this.astGeneric(b.body,l),l.push("if (!"),this.astGeneric(b.test,l),l.push(`) {
`),l.push(`break;
`),l.push(`}
`),l.push(`}
`),l}astAssignmentExpression(b,l){const d=this.getDeclaration(b.left);if(d&&!d.assignable)throw this.astErrorOutput(`Variable ${b.left.name} is not assignable here`,b);return this.astGeneric(b.left,l),l.push(b.operator),this.astGeneric(b.right,l),l}astBlockStatement(b,l){if(this.isState("loop-body")){this.pushState("block-body");for(let d=0;d<b.body.length;d++)this.astGeneric(b.body[d],l);this.popState("block-body")}else{l.push(`{
`);for(let d=0;d<b.body.length;d++)this.astGeneric(b.body[d],l);l.push(`}
`)}return l}astVariableDeclaration(b,l){l.push(`${b.kind} `);const{declarations:d}=b;for(let C=0;C<d.length;C++){C>0&&l.push(",");const m=d[C],p=this.getDeclaration(m.id);p.valueType||(p.valueType=this.getType(m.init)),this.astGeneric(m,l)}return this.isState("in-for-loop-init")||l.push(";"),l}astIfStatement(b,l){return l.push("if ("),this.astGeneric(b.test,l),l.push(")"),b.consequent.type==="BlockStatement"?this.astGeneric(b.consequent,l):(l.push(` {
`),this.astGeneric(b.consequent,l),l.push(`
}
`)),b.alternate&&(l.push("else "),b.alternate.type==="BlockStatement"||b.alternate.type==="IfStatement"?this.astGeneric(b.alternate,l):(l.push(` {
`),this.astGeneric(b.alternate,l),l.push(`
}
`))),l}astSwitchStatement(b,l){const{discriminant:d,cases:C}=b;l.push("switch ("),this.astGeneric(d,l),l.push(`) {
`);for(let m=0;m<C.length;m++){if(C[m].test===null){l.push(`default:
`),this.astGeneric(C[m].consequent,l),C[m].consequent&&C[m].consequent.length>0&&l.push(`break;
`);continue}l.push("case "),this.astGeneric(C[m].test,l),l.push(`:
`),C[m].consequent&&C[m].consequent.length>0&&(this.astGeneric(C[m].consequent,l),l.push(`break;
`))}l.push(`
}`)}astThisExpression(b,l){return l.push("_this"),l}astMemberExpression(b,l){const{signature:d,type:C,property:m,xProperty:p,yProperty:u,zProperty:g,name:E,origin:w}=this.getMemberExpressionDetails(b);switch(d){case"this.thread.value":return l.push(`_this.thread.${E}`),l;case"this.output.value":switch(E){case"x":l.push("outputX");break;case"y":l.push("outputY");break;case"z":l.push("outputZ");break;default:throw this.astErrorOutput("Unexpected expression",b)}return l;case"value":throw this.astErrorOutput("Unexpected expression",b);case"value[]":case"value[][]":case"value[][][]":case"value.value":if(w==="Math")return l.push(Math[E]),l;switch(m){case"r":return l.push(`user_${E}[0]`),l;case"g":return l.push(`user_${E}[1]`),l;case"b":return l.push(`user_${E}[2]`),l;case"a":return l.push(`user_${E}[3]`),l}break;case"this.constants.value":case"this.constants.value[]":case"this.constants.value[][]":case"this.constants.value[][][]":break;case"fn()[]":return this.astGeneric(b.object,l),l.push("["),this.astGeneric(b.property,l),l.push("]"),l;case"fn()[][]":return this.astGeneric(b.object.object,l),l.push("["),this.astGeneric(b.object.property,l),l.push("]"),l.push("["),this.astGeneric(b.property,l),l.push("]"),l;default:throw this.astErrorOutput("Unexpected expression",b)}if(!b.computed)switch(C){case"Number":case"Integer":case"Float":case"Boolean":return l.push(`${w}_${E}`),l}const y=`${w}_${E}`;switch(C){default:let k,v;if(w==="constants"){const $=this.constants[E];v=this.constantTypes[E]==="Input",k=v?$.size:null}else v=this.isInput(E),k=v?this.argumentSizes[this.argumentNames.indexOf(E)]:null;l.push(`${y}`),g&&u?v?(l.push("[("),this.astGeneric(g,l),l.push(`*${this.dynamicArguments?"(outputY * outputX)":k[1]*k[0]})+(`),this.astGeneric(u,l),l.push(`*${this.dynamicArguments?"outputX":k[0]})+`),this.astGeneric(p,l),l.push("]")):(l.push("["),this.astGeneric(g,l),l.push("]"),l.push("["),this.astGeneric(u,l),l.push("]"),l.push("["),this.astGeneric(p,l),l.push("]")):u?v?(l.push("[("),this.astGeneric(u,l),l.push(`*${this.dynamicArguments?"outputX":k[0]})+`),this.astGeneric(p,l),l.push("]")):(l.push("["),this.astGeneric(u,l),l.push("]"),l.push("["),this.astGeneric(p,l),l.push("]")):typeof p<"u"&&(l.push("["),this.astGeneric(p,l),l.push("]"))}return l}astCallExpression(b,l){if(b.type!=="CallExpression")throw this.astErrorOutput("Unknown CallExpression",b);let d=this.astMemberExpressionUnroll(b.callee);this.calledFunctions.indexOf(d)<0&&this.calledFunctions.push(d),this.isAstMathFunction(b),this.onFunctionCall&&this.onFunctionCall(this.name,d,b.arguments),l.push(d),l.push("(");const C=this.lookupFunctionArgumentTypes(d)||[];for(let m=0;m<b.arguments.length;++m){const p=b.arguments[m];let u=this.getType(p);C[m]||this.triggerImplyArgumentType(d,m,u,this),m>0&&l.push(", "),this.astGeneric(p,l)}return l.push(")"),l}astArrayExpression(b,l){const d=this.getType(b),C=b.elements.length,m=[];for(let p=0;p<C;++p){const u=[];this.astGeneric(b.elements[p],u),m.push(u.join(""))}switch(d){case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":l.push(`[${m.join(", ")}]`);break;default:l.push(`new Float32Array([${m.join(", ")}])`)}return l}astDebuggerStatement(b,l){return l.push("debugger;"),l}};z.exports={CPUFunctionNode:_}}),Ze=s((B,z)=>{const{utils:M}=h();function _(l,d){const C=[];for(const m in d){if(!d.hasOwnProperty(m))continue;const p=d[m],u=l[m];switch(p){case"Number":case"Integer":case"Float":case"Boolean":C.push(`${m}:${u}`);break;case"Array(2)":case"Array(3)":case"Array(4)":case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":C.push(`${m}:new ${u.constructor.name}(${JSON.stringify(Array.from(u))})`);break}}return`{ ${C.join()} }`}function b(l,d){const C=[],m=[],p=[],u=!/^function/.test(l.color.toString());if(C.push("  const { context, canvas, constants: incomingConstants } = settings;",`  const output = new Int32Array(${JSON.stringify(Array.from(l.output))});`,`  const _constantTypes = ${JSON.stringify(l.constantTypes)};`,`  const _constants = ${_(l.constants,l.constantTypes)};`),m.push("    constants: _constants,","    context,","    output,","    thread: {x: 0, y: 0, z: 0},"),l.graphical){C.push(`  const _imageData = context.createImageData(${l.output[0]}, ${l.output[1]});`),C.push(`  const _colorData = new Uint8ClampedArray(${l.output[0]} * ${l.output[1]} * 4);`);const w=M.flattenFunctionToString((u?"function ":"")+l.color.toString(),{thisLookup:k=>{switch(k){case"_colorData":return"_colorData";case"_imageData":return"_imageData";case"output":return"output";case"thread":return"this.thread"}return JSON.stringify(l[k])},findDependency:(k,v)=>null}),y=M.flattenFunctionToString((u?"function ":"")+l.getPixels.toString(),{thisLookup:k=>{switch(k){case"_colorData":return"_colorData";case"_imageData":return"_imageData";case"output":return"output";case"thread":return"this.thread"}return JSON.stringify(l[k])},findDependency:()=>null});m.push("    _imageData,","    _colorData,",`    color: ${w},`),p.push(`  kernel.getPixels = ${y};`)}const g=[],E=Object.keys(l.constantTypes);for(let w=0;w<E.length;w++)g.push(l.constantTypes[E]);if(l.argumentTypes.indexOf("HTMLImageArray")!==-1||g.indexOf("HTMLImageArray")!==-1){const w=M.flattenFunctionToString((u?"function ":"")+l._imageTo3DArray.toString(),{doNotDefine:["canvas"],findDependency:(y,k)=>y==="this"?(u?"function ":"")+l[k].toString():null,thisLookup:y=>{switch(y){case"canvas":return;case"context":return"context"}}});p.push(w),m.push("    _mediaTo2DArray,"),m.push("    _imageTo3DArray,")}else if(l.argumentTypes.indexOf("HTMLImage")!==-1||g.indexOf("HTMLImage")!==-1){const w=M.flattenFunctionToString((u?"function ":"")+l._mediaTo2DArray.toString(),{findDependency:(y,k)=>null,thisLookup:y=>{switch(y){case"canvas":return"settings.canvas";case"context":return"settings.context"}throw new Error("unhandled thisLookup")}});p.push(w),m.push("    _mediaTo2DArray,")}return`function(settings) {
${C.join(`
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
${l._kernelString}
  })
    .apply({ ${m.join(`
`)} });
  ${p.join(`
`)}
  return kernel;
}`}z.exports={cpuKernelString:b}}),rt=s((B,z)=>{const{Kernel:M}=I(),{FunctionBuilder:_}=U(),{CPUFunctionNode:b}=$e(),{utils:l}=h(),{cpuKernelString:d}=Ze();var C=class extends M{static getFeatures(){return this.features}static get features(){return Object.freeze({kernelMap:!0,isIntegerDivisionAccurate:!0})}static get isSupported(){return!0}static isContextMatch(m){return!1}static get mode(){return"cpu"}static nativeFunctionArguments(){return null}static nativeFunctionReturnType(){throw new Error(`Looking up native function return type not supported on ${this.name}`)}static combineKernels(m){return m}static getSignature(m,p){return"cpu"+(p.length>0?":"+p.join(","):"")}constructor(m,p){super(m,p),this.mergeSettings(m.settings||p),this._imageData=null,this._colorData=null,this._kernelString=null,this._prependedString=[],this.thread={x:0,y:0,z:0},this.translatedSources=null}initCanvas(){if(typeof document<"u")return document.createElement("canvas");if(typeof OffscreenCanvas<"u")return new OffscreenCanvas(0,0)}initContext(){return this.canvas?this.canvas.getContext("2d",{willReadFrequently:!0}):null}initPlugins(m){return[]}validateSettings(m){if(!this.output||this.output.length===0){if(m.length!==1)throw new Error("Auto output only supported for kernels with only one input");const p=l.getVariableType(m[0],this.strictIntegers);if(p==="Array")this.output=l.getDimensions(p);else if(p==="NumberTexture"||p==="ArrayTexture(4)")this.output=m[0].output;else throw new Error("Auto output not supported for input type: "+p)}if(this.graphical&&this.output.length!==2)throw new Error("Output must have 2 dimensions on graphical mode");this.checkOutput()}translateSource(){if(this.leadingReturnStatement=this.output.length>1?"resultX[x] = ":"result[x] = ",this.subKernels){const p=[];for(let u=0;u<this.subKernels.length;u++){const{name:g}=this.subKernels[u];p.push(this.output.length>1?`resultX_${g}[x] = subKernelResult_${g};
`:`result_${g}[x] = subKernelResult_${g};
`)}this.followingReturnStatement=p.join("")}const m=_.fromKernel(this,b);this.translatedSources=m.getPrototypes("kernel"),!this.graphical&&!this.returnType&&(this.returnType=m.getKernelResultType())}build(){if(this.built)return;if(this.randomSeed!==null&&console.warn("randomSeed is not supported in cpu mode; Math.random() will be unseeded"),this.setupConstants(),this.setupArguments(arguments),this.validateSettings(arguments),this.translateSource(),this.graphical){const{canvas:p,output:u}=this;if(!p)throw new Error("no canvas available for using graphical output");const g=u[0],E=u[1]||1;p.width=g,p.height=E,this._imageData=this.context.createImageData(g,E),this._colorData=new Uint8ClampedArray(g*E*4)}const m=this.getKernelString();this.kernelString=m,this.debug&&(console.log("Function output:"),console.log(m));try{this.run=new Function([],m).bind(this)()}catch(p){console.error("An error occurred compiling the javascript: ",p)}this.buildSignature(arguments),this.built=!0}color(m,p,u,g){typeof g>"u"&&(g=1),m=Math.floor(m*255),p=Math.floor(p*255),u=Math.floor(u*255),g=Math.floor(g*255);const E=this.output[0],w=this.output[1],y=this.thread.x+(w-this.thread.y-1)*E;this._colorData[y*4+0]=m,this._colorData[y*4+1]=p,this._colorData[y*4+2]=u,this._colorData[y*4+3]=g}getKernelString(){if(this._kernelString!==null)return this._kernelString;let m=null,{translatedSources:p}=this;return p.length>1?p=p.filter(u=>/^function/.test(u)?u:(m=u,!1)):m=p.shift(),this._kernelString=`  const LOOP_MAX = ${this._getLoopMaxString()};
  ${this.injectedNative||""}
  const _this = this;
  ${this._resultKernelHeader()}
  ${this._processConstants()}
  return (${this.argumentNames.map(u=>"user_"+u).join(", ")}) => {
    ${this._prependedString.join("")}
    ${this._earlyThrows()}
    ${this._processArguments()}
    ${this.graphical?this._graphicalKernelBody(m):this._resultKernelBody(m)}
    ${p.length>0?p.join(`
`):""}
  };`}toString(){return d(this)}_getLoopMaxString(){return this.loopMaxIterations?` ${parseInt(this.loopMaxIterations)};`:" 1000;"}_processConstants(){if(!this.constants)return"";const m=[];for(let p in this.constants)switch(this.constantTypes[p]){case"HTMLCanvas":case"OffscreenCanvas":case"HTMLImage":case"ImageBitmap":case"ImageData":case"HTMLVideo":m.push(`    const constants_${p} = this._mediaTo2DArray(this.constants.${p});
`);break;case"HTMLImageArray":m.push(`    const constants_${p} = this._imageTo3DArray(this.constants.${p});
`);break;case"Input":m.push(`    const constants_${p} = this.constants.${p}.value;
`);break;default:m.push(`    const constants_${p} = this.constants.${p};
`)}return m.join("")}_earlyThrows(){if(this.graphical||this.immutable||!this.pipeline)return"";const m=[];for(let u=0;u<this.argumentTypes.length;u++)this.argumentTypes[u]==="Array"&&m.push(this.argumentNames[u]);if(m.length===0)return"";const p=[];for(let u=0;u<m.length;u++){const g=m[u],E=this._mapSubKernels(w=>`user_${g} === result_${w.name}`).join(" || ");p.push(`user_${g} === result${E?` || ${E}`:""}`)}return`if (${p.join(" || ")}) throw new Error('Source and destination arrays are the same.  Use immutable = true');`}_processArguments(){const m=[];for(let p=0;p<this.argumentTypes.length;p++){const u=`user_${this.argumentNames[p]}`;switch(this.argumentTypes[p]){case"HTMLCanvas":case"OffscreenCanvas":case"HTMLImage":case"ImageBitmap":case"ImageData":case"HTMLVideo":m.push(`    ${u} = this._mediaTo2DArray(${u});
`);break;case"HTMLImageArray":m.push(`    ${u} = this._imageTo3DArray(${u});
`);break;case"Input":m.push(`    ${u} = ${u}.value;
`);break;case"ArrayTexture(1)":case"ArrayTexture(2)":case"ArrayTexture(3)":case"ArrayTexture(4)":case"NumberTexture":case"MemoryOptimizedNumberTexture":m.push(`
    if (${u}.toArray) {
      if (!_this.textureCache) {
        _this.textureCache = [];
        _this.arrayCache = [];
      }
      const textureIndex = _this.textureCache.indexOf(${u});
      if (textureIndex !== -1) {
        ${u} = _this.arrayCache[textureIndex];
      } else {
        _this.textureCache.push(${u});
        ${u} = ${u}.toArray();
        _this.arrayCache.push(${u});
      }
    }`);break}}return m.join("")}_mediaTo2DArray(m){const p=this.canvas,u=m.width>0?m.width:m.videoWidth,g=m.height>0?m.height:m.videoHeight;p.width<u&&(p.width=u),p.height<g&&(p.height=g);const E=this.context;let w;m.constructor===ImageData?w=m.data:(E.drawImage(m,0,0,u,g),w=E.getImageData(0,0,u,g).data);const y=new Array(g);let k=0;for(let v=g-1;v>=0;v--){const $=y[v]=new Array(u);for(let P=0;P<u;P++){const O=new Float32Array(4);O[0]=w[k++]/255,O[1]=w[k++]/255,O[2]=w[k++]/255,O[3]=w[k++]/255,$[P]=O}}return y}getPixels(m){const[p,u]=this.output;return m?l.flipPixels(this._imageData.data,p,u):this._imageData.data.slice(0)}_imageTo3DArray(m){const p=new Array(m.length);for(let u=0;u<m.length;u++)p[u]=this._mediaTo2DArray(m[u]);return p}_resultKernelHeader(){if(this.graphical||this.immutable||!this.pipeline)return"";switch(this.output.length){case 1:return this._mutableKernel1DResults();case 2:return this._mutableKernel2DResults();case 3:return this._mutableKernel3DResults()}}_resultKernelBody(m){switch(this.output.length){case 1:return(!this.immutable&&this.pipeline?this._resultMutableKernel1DLoop(m):this._resultImmutableKernel1DLoop(m))+this._kernelOutput();case 2:return(!this.immutable&&this.pipeline?this._resultMutableKernel2DLoop(m):this._resultImmutableKernel2DLoop(m))+this._kernelOutput();case 3:return(!this.immutable&&this.pipeline?this._resultMutableKernel3DLoop(m):this._resultImmutableKernel3DLoop(m))+this._kernelOutput();default:throw new Error("unsupported size kernel")}}_graphicalKernelBody(m){switch(this.output.length){case 2:return this._graphicalKernel2DLoop(m)+this._graphicalOutput();default:throw new Error("unsupported size kernel")}}_graphicalOutput(){return`
    this._imageData.data.set(this._colorData);
    this.context.putImageData(this._imageData, 0, 0);
    return;`}_getKernelResultTypeConstructorString(){switch(this.returnType){case"LiteralInteger":case"Number":case"Integer":case"Float":return"Float32Array";case"Array(2)":case"Array(3)":case"Array(4)":return"Array";default:if(this.graphical)return"Float32Array";throw new Error(`unhandled returnType ${this.returnType}`)}}_resultImmutableKernel1DLoop(m){const p=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const result = new ${p}(outputX);
    ${this._mapSubKernels(u=>`const result_${u.name} = new ${p}(outputX);
`).join("    ")}
    ${this._mapSubKernels(u=>`let subKernelResult_${u.name};
`).join("    ")}
    for (let x = 0; x < outputX; x++) {
      this.thread.x = x;
      this.thread.y = 0;
      this.thread.z = 0;
      ${m}
    }`}_mutableKernel1DResults(){const m=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const result = new ${m}(outputX);
    ${this._mapSubKernels(p=>`const result_${p.name} = new ${m}(outputX);
`).join("    ")}
    ${this._mapSubKernels(p=>`let subKernelResult_${p.name};
`).join("    ")}`}_resultMutableKernel1DLoop(m){return`  const outputX = _this.output[0];
    for (let x = 0; x < outputX; x++) {
      this.thread.x = x;
      this.thread.y = 0;
      this.thread.z = 0;
      ${m}
    }`}_resultImmutableKernel2DLoop(m){const p=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    const result = new Array(outputY);
    ${this._mapSubKernels(u=>`const result_${u.name} = new Array(outputY);
`).join("    ")}
    ${this._mapSubKernels(u=>`let subKernelResult_${u.name};
`).join("    ")}
    for (let y = 0; y < outputY; y++) {
      this.thread.z = 0;
      this.thread.y = y;
      const resultX = result[y] = new ${p}(outputX);
      ${this._mapSubKernels(u=>`const resultX_${u.name} = result_${u.name}[y] = new ${p}(outputX);
`).join("")}
      for (let x = 0; x < outputX; x++) {
        this.thread.x = x;
        ${m}
      }
    }`}_mutableKernel2DResults(){const m=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    const result = new Array(outputY);
    ${this._mapSubKernels(p=>`const result_${p.name} = new Array(outputY);
`).join("    ")}
    ${this._mapSubKernels(p=>`let subKernelResult_${p.name};
`).join("    ")}
    for (let y = 0; y < outputY; y++) {
      const resultX = result[y] = new ${m}(outputX);
      ${this._mapSubKernels(p=>`const resultX_${p.name} = result_${p.name}[y] = new ${m}(outputX);
`).join("")}
    }`}_resultMutableKernel2DLoop(m){const p=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    for (let y = 0; y < outputY; y++) {
      this.thread.z = 0;
      this.thread.y = y;
      const resultX = result[y];
      ${this._mapSubKernels(u=>`const resultX_${u.name} = result_${u.name}[y] = new ${p}(outputX);
`).join("")}
      for (let x = 0; x < outputX; x++) {
        this.thread.x = x;
        ${m}
      }
    }`}_graphicalKernel2DLoop(m){return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    for (let y = 0; y < outputY; y++) {
      this.thread.z = 0;
      this.thread.y = y;
      for (let x = 0; x < outputX; x++) {
        this.thread.x = x;
        ${m}
      }
    }`}_resultImmutableKernel3DLoop(m){const p=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    const outputZ = _this.output[2];
    const result = new Array(outputZ);
    ${this._mapSubKernels(u=>`const result_${u.name} = new Array(outputZ);
`).join("    ")}
    ${this._mapSubKernels(u=>`let subKernelResult_${u.name};
`).join("    ")}
    for (let z = 0; z < outputZ; z++) {
      this.thread.z = z;
      const resultY = result[z] = new Array(outputY);
      ${this._mapSubKernels(u=>`const resultY_${u.name} = result_${u.name}[z] = new Array(outputY);
`).join("      ")}
      for (let y = 0; y < outputY; y++) {
        this.thread.y = y;
        const resultX = resultY[y] = new ${p}(outputX);
        ${this._mapSubKernels(u=>`const resultX_${u.name} = resultY_${u.name}[y] = new ${p}(outputX);
`).join("        ")}
        for (let x = 0; x < outputX; x++) {
          this.thread.x = x;
          ${m}
        }
      }
    }`}_mutableKernel3DResults(){const m=this._getKernelResultTypeConstructorString();return`  const outputX = _this.output[0];
    const outputY = _this.output[1];
    const outputZ = _this.output[2];
    const result = new Array(outputZ);
    ${this._mapSubKernels(p=>`const result_${p.name} = new Array(outputZ);
`).join("    ")}
    ${this._mapSubKernels(p=>`let subKernelResult_${p.name};
`).join("    ")}
    for (let z = 0; z < outputZ; z++) {
      const resultY = result[z] = new Array(outputY);
      ${this._mapSubKernels(p=>`const resultY_${p.name} = result_${p.name}[z] = new Array(outputY);
`).join("      ")}
      for (let y = 0; y < outputY; y++) {
        const resultX = resultY[y] = new ${m}(outputX);
        ${this._mapSubKernels(p=>`const resultX_${p.name} = resultY_${p.name}[y] = new ${m}(outputX);
`).join("        ")}
      }
    }`}_resultMutableKernel3DLoop(m){return`  const outputX = _this.output[0];
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
          ${m}
        }
      }
    }`}_kernelOutput(){return this.subKernels?`
    return {
      result: result,
      ${this.subKernels.map(m=>`${m.property}: result_${m.name}`).join(`,
      `)}
    };`:`
    return result;`}_mapSubKernels(m){return this.subKernels===null?[""]:this.subKernels.map(m)}destroy(m){m&&delete this.canvas}static destroyContext(m){}toJSON(){const m=super.toJSON();return m.functionNodes=_.fromKernel(this,b).toJSON(),m}setOutput(m){super.setOutput(m);const[p,u]=this.output;this.graphical&&(this._imageData=this.context.createImageData(p,u),this._colorData=new Uint8ClampedArray(p*u*4))}prependString(m){if(this._kernelString)throw new Error("Kernel already built");this._prependedString.push(m)}hasPrependString(m){return this._prependedString.indexOf(m)>-1}};z.exports={CPUKernel:C}}),cs=s((B,z)=>{z.exports={}}),jt=s((B,z)=>{const{Texture:M}=c();var _=class extends M{get textureType(){throw new Error(`"textureType" not implemented on ${this.name}`)}clone(){return new this.constructor(this)}beforeMutate(){return this.texture._refs>1?(this.newTexture(),!0):!1}cloneTexture(){this.texture._refs--;const{context:l,size:d,texture:C,kernel:m}=this;m.debug&&console.warn("cloning internal texture"),l.bindFramebuffer(l.FRAMEBUFFER,this.framebuffer()),b(l,C),l.framebufferTexture2D(l.FRAMEBUFFER,l.COLOR_ATTACHMENT0,l.TEXTURE_2D,C,0);const p=l.createTexture();b(l,p),l.texImage2D(l.TEXTURE_2D,0,this.internalFormat,d[0],d[1],0,this.textureFormat,this.textureType,null),l.copyTexSubImage2D(l.TEXTURE_2D,0,0,0,0,0,d[0],d[1]),p._refs=1,this.texture=p}newTexture(){this.texture._refs--;const l=this.context,d=this.size;this.kernel.debug&&console.warn("new internal texture");const C=l.createTexture();b(l,C),l.texImage2D(l.TEXTURE_2D,0,this.internalFormat,d[0],d[1],0,this.textureFormat,this.textureType,null),C._refs=1,this.texture=C}clear(){if(this.texture._refs){this.texture._refs--;const C=this.context,m=this.texture=C.createTexture();b(C,m);const p=this.size;m._refs=1,C.texImage2D(C.TEXTURE_2D,0,this.internalFormat,p[0],p[1],0,this.textureFormat,this.textureType,null)}const{context:l,texture:d}=this;l.bindFramebuffer(l.FRAMEBUFFER,this.framebuffer()),l.bindTexture(l.TEXTURE_2D,d),b(l,d),l.framebufferTexture2D(l.FRAMEBUFFER,l.COLOR_ATTACHMENT0,l.TEXTURE_2D,d,0),l.clearColor(0,0,0,0),l.clear(l.COLOR_BUFFER_BIT|l.DEPTH_BUFFER_BIT)}delete(){this._deleted||(this._deleted=!0,!(this.texture._refs&&(this.texture._refs--,this.texture._refs))&&(this.kernel&&this.kernel.deleteTexture?this.kernel.deleteTexture(this.texture):this.context.deleteTexture(this.texture)))}framebuffer(){return this._framebuffer||(this._framebuffer=this.kernel.getRawValueFramebuffer(this.size[0],this.size[1])),this._framebuffer}};function b(l,d){l.activeTexture(l.TEXTURE15),l.bindTexture(l.TEXTURE_2D,d),l.texParameteri(l.TEXTURE_2D,l.TEXTURE_WRAP_S,l.CLAMP_TO_EDGE),l.texParameteri(l.TEXTURE_2D,l.TEXTURE_WRAP_T,l.CLAMP_TO_EDGE),l.texParameteri(l.TEXTURE_2D,l.TEXTURE_MIN_FILTER,l.NEAREST),l.texParameteri(l.TEXTURE_2D,l.TEXTURE_MAG_FILTER,l.NEAREST)}z.exports={GLTexture:_}}),De=s((B,z)=>{const{utils:M}=h(),{GLTexture:_}=jt();var b=class extends _{get textureType(){return this.context.FLOAT}constructor(l){super(l),this.type="ArrayTexture(1)"}renderRawOutput(){const l=this.context,d=this.size;l.bindFramebuffer(l.FRAMEBUFFER,this.framebuffer()),l.framebufferTexture2D(l.FRAMEBUFFER,l.COLOR_ATTACHMENT0,l.TEXTURE_2D,this.texture,0);const C=new Float32Array(d[0]*d[1]*4);return l.readPixels(0,0,d[0],d[1],l.RGBA,l.FLOAT,C),C}renderValues(){return this._deleted?null:this.renderRawOutput()}toArray(){return M.erectFloat(this.renderValues(),this.output[0])}};z.exports={GLTextureFloat:b}}),$t=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(2)"}toArray(){return M.erectArray2(this.renderValues(),this.output[0],this.output[1])}};z.exports={GLTextureArray2Float:b}}),Ps=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(2)"}toArray(){return M.erect2DArray2(this.renderValues(),this.output[0],this.output[1])}};z.exports={GLTextureArray2Float2D:b}}),hs=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(2)"}toArray(){return M.erect3DArray2(this.renderValues(),this.output[0],this.output[1],this.output[2])}};z.exports={GLTextureArray2Float3D:b}}),gn=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(3)"}toArray(){return M.erectArray3(this.renderValues(),this.output[0])}};z.exports={GLTextureArray3Float:b}}),ds=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(3)"}toArray(){return M.erect2DArray3(this.renderValues(),this.output[0],this.output[1])}};z.exports={GLTextureArray3Float2D:b}}),me=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(3)"}toArray(){return M.erect3DArray3(this.renderValues(),this.output[0],this.output[1],this.output[2])}};z.exports={GLTextureArray3Float3D:b}}),Ce=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(4)"}toArray(){return M.erectArray4(this.renderValues(),this.output[0])}};z.exports={GLTextureArray4Float:b}}),Ue=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(4)"}toArray(){return M.erect2DArray4(this.renderValues(),this.output[0],this.output[1])}};z.exports={GLTextureArray4Float2D:b}}),Dt=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(4)"}toArray(){return M.erect3DArray4(this.renderValues(),this.output[0],this.output[1],this.output[2])}};z.exports={GLTextureArray4Float3D:b}}),qa=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(1)"}toArray(){return M.erect2DFloat(this.renderValues(),this.output[0],this.output[1])}};z.exports={GLTextureFloat2D:b}}),Ha=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="ArrayTexture(1)"}toArray(){return M.erect3DFloat(this.renderValues(),this.output[0],this.output[1],this.output[2])}};z.exports={GLTextureFloat3D:b}}),Wa=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="MemoryOptimizedNumberTexture"}toArray(){return M.erectMemoryOptimizedFloat(this.renderValues(),this.output[0])}};z.exports={GLTextureMemoryOptimized:b}}),Xa=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="MemoryOptimizedNumberTexture"}toArray(){return M.erectMemoryOptimized2DFloat(this.renderValues(),this.output[0],this.output[1])}};z.exports={GLTextureMemoryOptimized2D:b}}),Ya=s((B,z)=>{const{utils:M}=h(),{GLTextureFloat:_}=De();var b=class extends _{constructor(l){super(l),this.type="MemoryOptimizedNumberTexture"}toArray(){return M.erectMemoryOptimized3DFloat(this.renderValues(),this.output[0],this.output[1],this.output[2])}};z.exports={GLTextureMemoryOptimized3D:b}}),zs=s((B,z)=>{const{utils:M}=h(),{GLTexture:_}=jt();var b=class extends _{get textureType(){return this.context.UNSIGNED_BYTE}constructor(l){super(l),this.type="NumberTexture"}renderRawOutput(){const{context:l}=this;l.bindFramebuffer(l.FRAMEBUFFER,this.framebuffer()),l.framebufferTexture2D(l.FRAMEBUFFER,l.COLOR_ATTACHMENT0,l.TEXTURE_2D,this.texture,0);const d=new Uint8Array(this.size[0]*this.size[1]*4);return l.readPixels(0,0,this.size[0],this.size[1],l.RGBA,l.UNSIGNED_BYTE,d),d}renderValues(){return this._deleted?null:new Float32Array(this.renderRawOutput().buffer)}toArray(){return M.erectPackedFloat(this.renderValues(),this.output[0])}};z.exports={GLTextureUnsigned:b}}),Ja=s((B,z)=>{const{utils:M}=h(),{GLTextureUnsigned:_}=zs();var b=class extends _{constructor(l){super(l),this.type="NumberTexture"}toArray(){return M.erect2DPackedFloat(this.renderValues(),this.output[0],this.output[1])}};z.exports={GLTextureUnsigned2D:b}}),Za=s((B,z)=>{const{utils:M}=h(),{GLTextureUnsigned:_}=zs();var b=class extends _{constructor(l){super(l),this.type="NumberTexture"}toArray(){return M.erect3DPackedFloat(this.renderValues(),this.output[0],this.output[1],this.output[2])}};z.exports={GLTextureUnsigned3D:b}}),Qa=s((B,z)=>{const{GLTextureUnsigned:M}=zs();var _=class extends M{constructor(b){super(b),this.type="ArrayTexture(4)"}toArray(){return this.renderValues()}};z.exports={GLTextureGraphical:_}}),$r=s((B,z)=>{const{Kernel:M}=I(),{utils:_}=h(),{GLTextureArray2Float:b}=$t(),{GLTextureArray2Float2D:l}=Ps(),{GLTextureArray2Float3D:d}=hs(),{GLTextureArray3Float:C}=gn(),{GLTextureArray3Float2D:m}=ds(),{GLTextureArray3Float3D:p}=me(),{GLTextureArray4Float:u}=Ce(),{GLTextureArray4Float2D:g}=Ue(),{GLTextureArray4Float3D:E}=Dt(),{GLTextureFloat:w}=De(),{GLTextureFloat2D:y}=qa(),{GLTextureFloat3D:k}=Ha(),{GLTextureMemoryOptimized:v}=Wa(),{GLTextureMemoryOptimized2D:$}=Xa(),{GLTextureMemoryOptimized3D:P}=Ya(),{GLTextureUnsigned:O}=zs(),{GLTextureUnsigned2D:T}=Ja(),{GLTextureUnsigned3D:A}=Za(),{GLTextureGraphical:f}=Qa();var F=class extends M{static get mode(){return"gpu"}static getIsFloatRead(){const R=new this(`function kernelFunction() {
      return 1;
    }`,{context:this.testContext,canvas:this.testCanvas,validate:!1,output:[1],precision:"single",returnType:"Number",tactic:"speed"});R.build(),R.run();const V=R.renderOutput();return R.destroy(!0),V[0]===1}static getIsIntegerDivisionAccurate(){function R(te,ee){return te[this.thread.x]/ee[this.thread.x]}const V=new this(R.toString(),{context:this.testContext,canvas:this.testCanvas,validate:!1,output:[2],returnType:"Number",precision:"unsigned",tactic:"speed"}),W=[[6,6030401],[3,3991]];V.build.apply(V,W),V.run.apply(V,W);const N=V.renderOutput();return V.destroy(!0),N[0]===2&&N[1]===1511}static getIsSpeedTacticSupported(){function R(te){return te[this.thread.x]}const V=new this(R.toString(),{context:this.testContext,canvas:this.testCanvas,validate:!1,output:[4],returnType:"Number",precision:"unsigned",tactic:"speed"}),W=[[0,1,2,3]];V.build.apply(V,W),V.run.apply(V,W);const N=V.renderOutput();return V.destroy(!0),Math.round(N[0])===0&&Math.round(N[1])===1&&Math.round(N[2])===2&&Math.round(N[3])===3}static get testCanvas(){throw new Error(`"testCanvas" not defined on ${this.name}`)}static get testContext(){throw new Error(`"testContext" not defined on ${this.name}`)}static getFeatures(){const R=this.testContext,V=this.getIsDrawBuffers();return Object.freeze({isFloatRead:this.getIsFloatRead(),isIntegerDivisionAccurate:this.getIsIntegerDivisionAccurate(),isSpeedTacticSupported:this.getIsSpeedTacticSupported(),isTextureFloat:this.getIsTextureFloat(),isDrawBuffers:V,kernelMap:V,channelCount:this.getChannelCount(),maxTextureSize:this.getMaxTextureSize(),lowIntPrecision:R.getShaderPrecisionFormat(R.FRAGMENT_SHADER,R.LOW_INT),lowFloatPrecision:R.getShaderPrecisionFormat(R.FRAGMENT_SHADER,R.LOW_FLOAT),mediumIntPrecision:R.getShaderPrecisionFormat(R.FRAGMENT_SHADER,R.MEDIUM_INT),mediumFloatPrecision:R.getShaderPrecisionFormat(R.FRAGMENT_SHADER,R.MEDIUM_FLOAT),highIntPrecision:R.getShaderPrecisionFormat(R.FRAGMENT_SHADER,R.HIGH_INT),highFloatPrecision:R.getShaderPrecisionFormat(R.FRAGMENT_SHADER,R.HIGH_FLOAT)})}static setupFeatureChecks(){throw new Error(`"setupFeatureChecks" not defined on ${this.name}`)}static getSignature(R,V){return R.getVariablePrecisionString()+(V.length>0?":"+V.join(","):"")}setFixIntegerDivisionAccuracy(R){return this.fixIntegerDivisionAccuracy=R,this}setPrecision(R){return this.precision=R,this}setFloatTextures(R){return _.warnDeprecated("method","setFloatTextures","setOptimizeFloatMemory"),this.floatTextures=R,this}static nativeFunctionArguments(R){const V=[],W=[],N=[],te=/^[a-zA-Z_]/,ee=/[a-zA-Z_0-9]/;let X=0,ie=null,se=null;for(;X<R.length;){const J=R[X],ae=R[X+1],he=N.length>0?N[N.length-1]:null;if(he==="FUNCTION_ARGUMENTS"&&J==="/"&&ae==="*"){N.push("MULTI_LINE_COMMENT"),X+=2;continue}else if(he==="MULTI_LINE_COMMENT"&&J==="*"&&ae==="/"){N.pop(),X+=2;continue}else if(he==="FUNCTION_ARGUMENTS"&&J==="/"&&ae==="/"){N.push("COMMENT"),X+=2;continue}else if(he==="COMMENT"&&J===`
`){N.pop(),X++;continue}else if(he===null&&J==="("){N.push("FUNCTION_ARGUMENTS"),X++;continue}else if(he==="FUNCTION_ARGUMENTS"){if(J===")"){N.pop();break}if(J==="f"&&ae==="l"&&R[X+2]==="o"&&R[X+3]==="a"&&R[X+4]==="t"&&R[X+5]===" "){N.push("DECLARE_VARIABLE"),se="float",ie="",X+=6;continue}else if(J==="i"&&ae==="n"&&R[X+2]==="t"&&R[X+3]===" "){N.push("DECLARE_VARIABLE"),se="int",ie="",X+=4;continue}else if(J==="v"&&ae==="e"&&R[X+2]==="c"&&R[X+3]==="2"&&R[X+4]===" "){N.push("DECLARE_VARIABLE"),se="vec2",ie="",X+=5;continue}else if(J==="v"&&ae==="e"&&R[X+2]==="c"&&R[X+3]==="3"&&R[X+4]===" "){N.push("DECLARE_VARIABLE"),se="vec3",ie="",X+=5;continue}else if(J==="v"&&ae==="e"&&R[X+2]==="c"&&R[X+3]==="4"&&R[X+4]===" "){N.push("DECLARE_VARIABLE"),se="vec4",ie="",X+=5;continue}}else if(he==="DECLARE_VARIABLE"){if(ie===""){if(J===" "){X++;continue}if(!te.test(J))throw new Error("variable name is not expected string")}ie+=J,ee.test(ae)||(N.pop(),W.push(ie),V.push(L[se]))}X++}if(N.length>0)throw new Error("GLSL function was not parsable");return{argumentNames:W,argumentTypes:V}}static nativeFunctionReturnType(R){return L[R.match(/int|float|vec[2-4]/)[0]]}static combineKernels(R,V){R.apply(null,arguments);const{texSize:W,context:N,threadDim:te}=V.texSize;let ee;if(V.precision==="single"){const X=W[0],ie=Math.ceil(W[1]/4);ee=new Float32Array(X*ie*4*4),N.readPixels(0,0,X,ie*4,N.RGBA,N.FLOAT,ee)}else{const X=new Uint8Array(W[0]*W[1]*4);N.readPixels(0,0,W[0],W[1],N.RGBA,N.UNSIGNED_BYTE,X),ee=new Float32Array(X.buffer)}if(ee=ee.subarray(0,te[0]*te[1]*te[2]),V.output.length===1)return ee;if(V.output.length===2)return _.splitArray(ee,V.output[0]);if(V.output.length===3)return _.splitArray(ee,V.output[0]*V.output[1]).map(function(X){return _.splitArray(X,V.output[0])})}constructor(R,V){super(R,V),this.transferValues=null,this.formatValues=null,this.TextureConstructor=null,this.renderOutput=null,this.renderRawOutput=null,this.texSize=null,this.translatedSource=null,this.compiledFragmentShader=null,this.compiledVertexShader=null,this.switchingKernels=null,this._textureSwitched=null,this._mappedTextureSwitched=null}checkTextureSize(){const{features:R}=this.constructor;if(this.texSize[0]>R.maxTextureSize||this.texSize[1]>R.maxTextureSize)throw new Error(`Texture size [${this.texSize[0]},${this.texSize[1]}] generated by kernel is larger than supported size [${R.maxTextureSize},${R.maxTextureSize}]`)}translateSource(){throw new Error(`"translateSource" not defined on ${this.constructor.name}`)}pickRenderStrategy(R){if(this.graphical)return this.renderRawOutput=this.readPackedPixelsToUint8Array,this.transferValues=V=>V,this.TextureConstructor=f,null;if(this.precision==="unsigned")if(this.renderRawOutput=this.readPackedPixelsToUint8Array,this.transferValues=this.readPackedPixelsToFloat32Array,this.pipeline)switch(this.renderOutput=this.renderTexture,this.subKernels!==null&&(this.renderKernels=this.renderKernelsToTextures),this.returnType){case"LiteralInteger":case"Float":case"Number":case"Integer":return this.output[2]>0?(this.TextureConstructor=A,null):this.output[1]>0?(this.TextureConstructor=T,null):(this.TextureConstructor=O,null);case"Array(2)":case"Array(3)":case"Array(4)":return this.requestFallback(R)}else switch(this.subKernels!==null&&(this.renderKernels=this.renderKernelsToArrays),this.returnType){case"LiteralInteger":case"Float":case"Number":case"Integer":return this.renderOutput=this.renderValues,this.output[2]>0?(this.TextureConstructor=A,this.formatValues=_.erect3DPackedFloat,null):this.output[1]>0?(this.TextureConstructor=T,this.formatValues=_.erect2DPackedFloat,null):(this.TextureConstructor=O,this.formatValues=_.erectPackedFloat,null);case"Array(2)":case"Array(3)":case"Array(4)":return this.requestFallback(R)}else if(this.precision==="single"){if(this.renderRawOutput=this.readFloatPixelsToFloat32Array,this.transferValues=this.readFloatPixelsToFloat32Array,this.pipeline)switch(this.renderOutput=this.renderTexture,this.subKernels!==null&&(this.renderKernels=this.renderKernelsToTextures),this.returnType){case"LiteralInteger":case"Float":case"Number":case"Integer":return this.optimizeFloatMemory?this.output[2]>0?(this.TextureConstructor=P,null):this.output[1]>0?(this.TextureConstructor=$,null):(this.TextureConstructor=v,null):this.output[2]>0?(this.TextureConstructor=k,null):this.output[1]>0?(this.TextureConstructor=y,null):(this.TextureConstructor=w,null);case"Array(2)":return this.output[2]>0?(this.TextureConstructor=d,null):this.output[1]>0?(this.TextureConstructor=l,null):(this.TextureConstructor=b,null);case"Array(3)":return this.output[2]>0?(this.TextureConstructor=p,null):this.output[1]>0?(this.TextureConstructor=m,null):(this.TextureConstructor=C,null);case"Array(4)":return this.output[2]>0?(this.TextureConstructor=E,null):this.output[1]>0?(this.TextureConstructor=g,null):(this.TextureConstructor=u,null)}if(this.renderOutput=this.renderValues,this.subKernels!==null&&(this.renderKernels=this.renderKernelsToArrays),this.optimizeFloatMemory)switch(this.returnType){case"LiteralInteger":case"Float":case"Number":case"Integer":return this.output[2]>0?(this.TextureConstructor=P,this.formatValues=_.erectMemoryOptimized3DFloat,null):this.output[1]>0?(this.TextureConstructor=$,this.formatValues=_.erectMemoryOptimized2DFloat,null):(this.TextureConstructor=v,this.formatValues=_.erectMemoryOptimizedFloat,null);case"Array(2)":return this.output[2]>0?(this.TextureConstructor=d,this.formatValues=_.erect3DArray2,null):this.output[1]>0?(this.TextureConstructor=l,this.formatValues=_.erect2DArray2,null):(this.TextureConstructor=b,this.formatValues=_.erectArray2,null);case"Array(3)":return this.output[2]>0?(this.TextureConstructor=p,this.formatValues=_.erect3DArray3,null):this.output[1]>0?(this.TextureConstructor=m,this.formatValues=_.erect2DArray3,null):(this.TextureConstructor=C,this.formatValues=_.erectArray3,null);case"Array(4)":return this.output[2]>0?(this.TextureConstructor=E,this.formatValues=_.erect3DArray4,null):this.output[1]>0?(this.TextureConstructor=g,this.formatValues=_.erect2DArray4,null):(this.TextureConstructor=u,this.formatValues=_.erectArray4,null)}else switch(this.returnType){case"LiteralInteger":case"Float":case"Number":case"Integer":return this.output[2]>0?(this.TextureConstructor=k,this.formatValues=_.erect3DFloat,null):this.output[1]>0?(this.TextureConstructor=y,this.formatValues=_.erect2DFloat,null):(this.TextureConstructor=w,this.formatValues=_.erectFloat,null);case"Array(2)":return this.output[2]>0?(this.TextureConstructor=d,this.formatValues=_.erect3DArray2,null):this.output[1]>0?(this.TextureConstructor=l,this.formatValues=_.erect2DArray2,null):(this.TextureConstructor=b,this.formatValues=_.erectArray2,null);case"Array(3)":return this.output[2]>0?(this.TextureConstructor=p,this.formatValues=_.erect3DArray3,null):this.output[1]>0?(this.TextureConstructor=m,this.formatValues=_.erect2DArray3,null):(this.TextureConstructor=C,this.formatValues=_.erectArray3,null);case"Array(4)":return this.output[2]>0?(this.TextureConstructor=E,this.formatValues=_.erect3DArray4,null):this.output[1]>0?(this.TextureConstructor=g,this.formatValues=_.erect2DArray4,null):(this.TextureConstructor=u,this.formatValues=_.erectArray4,null)}}else throw new Error(`unhandled precision of "${this.precision}"`);throw new Error(`unhandled return type "${this.returnType}"`)}getKernelString(){throw new Error("abstract method call")}getMainResultTexture(){switch(this.returnType){case"LiteralInteger":case"Float":case"Integer":case"Number":return this.getMainResultNumberTexture();case"Array(2)":return this.getMainResultArray2Texture();case"Array(3)":return this.getMainResultArray3Texture();case"Array(4)":return this.getMainResultArray4Texture();default:throw new Error(`unhandled returnType type ${this.returnType}`)}}getMainResultKernelNumberTexture(){throw new Error("abstract method call")}getMainResultSubKernelNumberTexture(){throw new Error("abstract method call")}getMainResultKernelArray2Texture(){throw new Error("abstract method call")}getMainResultSubKernelArray2Texture(){throw new Error("abstract method call")}getMainResultKernelArray3Texture(){throw new Error("abstract method call")}getMainResultSubKernelArray3Texture(){throw new Error("abstract method call")}getMainResultKernelArray4Texture(){throw new Error("abstract method call")}getMainResultSubKernelArray4Texture(){throw new Error("abstract method call")}getMainResultGraphical(){throw new Error("abstract method call")}getMainResultMemoryOptimizedFloats(){throw new Error("abstract method call")}getMainResultPackedPixels(){throw new Error("abstract method call")}getMainResultString(){return this.graphical?this.getMainResultGraphical():this.precision==="single"?this.optimizeFloatMemory?this.getMainResultMemoryOptimizedFloats():this.getMainResultTexture():this.getMainResultPackedPixels()}getMainResultNumberTexture(){return _.linesToString(this.getMainResultKernelNumberTexture())+_.linesToString(this.getMainResultSubKernelNumberTexture())}getMainResultArray2Texture(){return _.linesToString(this.getMainResultKernelArray2Texture())+_.linesToString(this.getMainResultSubKernelArray2Texture())}getMainResultArray3Texture(){return _.linesToString(this.getMainResultKernelArray3Texture())+_.linesToString(this.getMainResultSubKernelArray3Texture())}getMainResultArray4Texture(){return _.linesToString(this.getMainResultKernelArray4Texture())+_.linesToString(this.getMainResultSubKernelArray4Texture())}getFloatTacticDeclaration(){return`precision ${this.getVariablePrecisionString(this.texSize,this.tactic)} float;
`}getIntTacticDeclaration(){return`precision ${this.getVariablePrecisionString(this.texSize,this.tactic,!0)} int;
`}getSampler2DTacticDeclaration(){return`precision ${this.getVariablePrecisionString(this.texSize,this.tactic)} sampler2D;
`}getSampler2DArrayTacticDeclaration(){return`precision ${this.getVariablePrecisionString(this.texSize,this.tactic)} sampler2DArray;
`}renderTexture(){return this.immutable?this.texture.clone():this.texture}readPackedPixelsToUint8Array(){if(this.precision!=="unsigned")throw new Error('Requires this.precision to be "unsigned"');const{texSize:R,context:V}=this,W=new Uint8Array(R[0]*R[1]*4);return V.readPixels(0,0,R[0],R[1],V.RGBA,V.UNSIGNED_BYTE,W),W}readPackedPixelsToFloat32Array(){return new Float32Array(this.readPackedPixelsToUint8Array().buffer)}readFloatPixelsToFloat32Array(){if(this.precision!=="single")throw new Error('Requires this.precision to be "single"');const{texSize:R,context:V}=this,W=R[0],N=R[1],te=new Float32Array(W*N*4);return V.readPixels(0,0,W,N,V.RGBA,V.FLOAT,te),te}getPixels(R){const{context:V,output:W}=this,[N,te]=W,ee=new Uint8Array(N*te*4);return V.readPixels(0,0,N,te,V.RGBA,V.UNSIGNED_BYTE,ee),new Uint8ClampedArray((R?ee:_.flipPixels(ee,N,te)).buffer)}renderKernelsToArrays(){const R={result:this.renderOutput()};for(let V=0;V<this.subKernels.length;V++)R[this.subKernels[V].property]=this.mappedTextures[V].toArray();return R}renderKernelsToTextures(){const R={result:this.renderOutput()};if(this.immutable)for(let V=0;V<this.subKernels.length;V++)R[this.subKernels[V].property]=this.mappedTextures[V].clone();else for(let V=0;V<this.subKernels.length;V++)R[this.subKernels[V].property]=this.mappedTextures[V];return R}resetSwitchingKernels(){const R=this.switchingKernels;return this.switchingKernels=null,R}setOutput(R){const V=this.toKernelOutput(R);if(this.program){if(!this.dynamicOutput)throw new Error("Resizing a kernel with dynamicOutput: false is not possible");const W=[V[0],V[1]||1,V[2]||1],N=_.getKernelTextureSize({optimizeFloatMemory:this.optimizeFloatMemory,precision:this.precision},W),te=this.texSize;if(te){const X=this.getVariablePrecisionString(te,this.tactic),ie=this.getVariablePrecisionString(N,this.tactic);if(X!==ie){this.debug&&console.warn("Precision requirement changed, asking GPU instance to recompile"),this.switchKernels({type:"outputPrecisionMismatch",precision:ie,needed:R});return}}this.output=V,this.threadDim=W,this.texSize=N;const{context:ee}=this;if(ee.bindFramebuffer(ee.FRAMEBUFFER,this.framebuffer),this.updateMaxTexSize(),this.framebuffer.width=this.texSize[0],this.framebuffer.height=this.texSize[1],ee.viewport(0,0,this.maxTexSize[0],this.maxTexSize[1]),this.canvas.width=this.maxTexSize[0],this.canvas.height=this.maxTexSize[1],this.texture&&this.texture.delete(),this.texture=null,this._setupOutputTexture(),this.mappedTextures&&this.mappedTextures.length>0){for(let X=0;X<this.mappedTextures.length;X++)this.mappedTextures[X].delete();this.mappedTextures=null,this._setupSubOutputTextures()}}else this.output=V;return this}renderValues(){return this.formatValues(this.transferValues(),this.output[0],this.output[1],this.output[2])}switchKernels(R){this.switchingKernels?this.switchingKernels.push(R):this.switchingKernels=[R]}getVariablePrecisionString(R=this.texSize,V=this.tactic,W=!1){if(!V){if(!this.constructor.features.isSpeedTacticSupported)return"highp";const N=this.constructor.features[W?"lowIntPrecision":"lowFloatPrecision"],te=this.constructor.features[W?"mediumIntPrecision":"mediumFloatPrecision"],ee=this.constructor.features[W?"highIntPrecision":"highFloatPrecision"],X=Math.log2(R[0]*R[1]);if(X<=N.rangeMax)return"lowp";if(X<=te.rangeMax)return"mediump";if(X<=ee.rangeMax)return"highp";throw new Error("The required size exceeds that of the ability of your system")}switch(V){case"speed":return"lowp";case"balanced":return"mediump";case"precision":return"highp";default:throw new Error(`Unknown tactic "${V}" use "speed", "balanced", "precision", or empty for auto`)}}updateTextureArgumentRefs(R,V){if(this.immutable){if(this.texture.texture===V.texture){const{prevArg:W}=R;W&&(W.texture._refs===1&&(this.texture.delete(),this.texture=W.clone(),this._textureSwitched=!0),W.delete()),R.prevArg=V.clone()}else if(this.mappedTextures&&this.mappedTextures.length>0){const{mappedTextures:W}=this;for(let N=0;N<W.length;N++){const te=W[N];if(te.texture===V.texture){const{prevArg:ee}=R;ee&&(ee.texture._refs===1&&(te.delete(),W[N]=ee.clone(),this._mappedTextureSwitched[N]=!0),ee.delete()),R.prevArg=V.clone();return}}}}}onActivate(R){if(this._textureSwitched=!0,this.texture=R.texture,this.mappedTextures){for(let V=0;V<this.mappedTextures.length;V++)this._mappedTextureSwitched[V]=!0;this.mappedTextures=R.mappedTextures}}initCanvas(){}};const L={int:"Integer",float:"Number",vec2:"Array(2)",vec3:"Array(3)",vec4:"Array(4)"};z.exports={GLKernel:F}}),yn=s((B,z)=>{const{utils:M}=h(),{FunctionNode:_}=q();var b=class extends _{constructor(m,p){super(m,p),p&&p.hasOwnProperty("fixIntegerDivisionAccuracy")&&(this.fixIntegerDivisionAccuracy=p.fixIntegerDivisionAccuracy)}astConditionalExpression(m,p){if(m.type!=="ConditionalExpression")throw this.astErrorOutput("Not a conditional expression",m);const u=this.getType(m.consequent),g=this.getType(m.alternate);return u===null&&g===null?(p.push("if ("),this.astGeneric(m.test,p),p.push(") {"),this.astGeneric(m.consequent,p),p.push(";"),p.push("} else {"),this.astGeneric(m.alternate,p),p.push(";"),p.push("}"),p):(p.push("("),this.astGeneric(m.test,p),p.push("?"),this.astGeneric(m.consequent,p),p.push(":"),this.astGeneric(m.alternate,p),p.push(")"),p)}astFunction(m,p){if(this.isRootKernel)p.push("void");else{this.returnType||this.findLastReturn()&&(this.returnType=this.getType(m.body),this.returnType==="LiteralInteger"&&(this.returnType="Number"));const{returnType:u}=this;if(!u)p.push("void");else{const g=d[u];if(!g)throw new Error(`unknown type ${u}`);p.push(g)}}if(p.push(" "),p.push(this.name),p.push("("),!this.isRootKernel)for(let u=0;u<this.argumentNames.length;++u){const g=this.argumentNames[u];u>0&&p.push(", ");let E=this.argumentTypes[this.argumentNames.indexOf(g)];if(!E)throw this.astErrorOutput(`Unknown argument ${g} type`,m);E==="LiteralInteger"&&(this.argumentTypes[u]=E="Number");const w=d[E];if(!w)throw this.astErrorOutput("Unexpected expression",m);const y=M.sanitizeName(g);w==="sampler2D"||w==="sampler2DArray"?p.push(`${w} user_${y},ivec2 user_${y}Size,ivec3 user_${y}Dim`):p.push(`${w} user_${y}`)}p.push(`) {
`);for(let u=0;u<m.body.body.length;++u)this.astStatementWithHoisting(m.body.body[u],p),p.push(`
`);return p.push(`}
`),p}astReturnStatement(m,p){if(!m.argument)throw this.astErrorOutput("Unexpected return statement",m);this.pushState("skip-literal-correction");const u=this.getType(m.argument);this.popState("skip-literal-correction");const g=[];switch(this.returnType||(u==="LiteralInteger"||u==="Integer"?this.returnType="Number":this.returnType=u),this.returnType){case"LiteralInteger":case"Number":case"Float":switch(u){case"Integer":g.push("float("),this.astGeneric(m.argument,g),g.push(")");break;case"LiteralInteger":this.castLiteralToFloat(m.argument,g),this.getType(m)==="Integer"&&(g.unshift("float("),g.push(")"));break;default:this.astGeneric(m.argument,g)}break;case"Integer":switch(u){case"Float":case"Number":this.castValueToInteger(m.argument,g);break;case"LiteralInteger":this.castLiteralToInteger(m.argument,g);break;default:this.astGeneric(m.argument,g)}break;case"Array(4)":case"Array(3)":case"Array(2)":case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":case"Input":this.astGeneric(m.argument,g);break;default:throw this.astErrorOutput(`unhandled return type ${this.returnType}`,m)}return this.isRootKernel?(p.push(`kernelResult = ${g.join("")};`),p.push("return;")):this.isSubKernel?(p.push(`subKernelResult_${this.name} = ${g.join("")};`),p.push(`return subKernelResult_${this.name};`)):p.push(`return ${g.join("")};`),p}astLiteral(m,p){if(isNaN(m.value))throw this.astErrorOutput("Non-numeric literal not supported : "+m.value,m);const u=this.astKey(m);return Number.isInteger(m.value)?this.isState("casting-to-integer")||this.isState("building-integer")?(this.literalTypes[u]="Integer",p.push(`${m.value}`)):this.isState("casting-to-float")||this.isState("building-float")?(this.literalTypes[u]="Number",p.push(`${m.value}.0`)):(this.literalTypes[u]="Number",p.push(`${m.value}.0`)):this.isState("casting-to-integer")||this.isState("building-integer")?(this.literalTypes[u]="Integer",p.push(Math.round(m.value))):(this.literalTypes[u]="Number",p.push(`${m.value}`)),p}astBinaryExpression(m,p){if(this.checkAndUpconvertOperator(m,p))return p;if(m.operator==="/"){const w=this.fixIntegerDivisionAccuracy;switch(p.push(w?"divWithIntCheck(":"("),this.pushState("building-float"),this.getType(m.left)){case"Integer":this.castValueToFloat(m.left,p);break;case"LiteralInteger":this.castLiteralToFloat(m.left,p);break;default:this.astGeneric(m.left,p)}switch(p.push(w?", ":"/"),this.getType(m.right)){case"Integer":this.castValueToFloat(m.right,p);break;case"LiteralInteger":this.castLiteralToFloat(m.right,p);break;default:this.astGeneric(m.right,p)}return this.popState("building-float"),p.push(")"),p}p.push("(");const u=this.getType(m.left)||"Number",g=this.getType(m.right)||"Number",E=u+" & "+g;switch(E){case"Integer & Integer":this.pushState("building-integer"),this.astGeneric(m.left,p),p.push(C[m.operator]||m.operator),this.astGeneric(m.right,p),this.popState("building-integer");break;case"Number & Float":case"Float & Number":case"Float & Float":case"Number & Number":this.pushState("building-float"),this.astGeneric(m.left,p),p.push(C[m.operator]||m.operator),this.astGeneric(m.right,p),this.popState("building-float");break;case"LiteralInteger & LiteralInteger":this.isState("casting-to-integer")||this.isState("building-integer")?(this.pushState("building-integer"),this.astGeneric(m.left,p),p.push(C[m.operator]||m.operator),this.astGeneric(m.right,p),this.popState("building-integer")):(this.pushState("building-float"),this.castLiteralToFloat(m.left,p),p.push(C[m.operator]||m.operator),this.castLiteralToFloat(m.right,p),this.popState("building-float"));break;case"Integer & Float":case"Integer & Number":if((m.operator===">"||m.operator==="<"&&m.right.type==="Literal")&&!Number.isInteger(m.right.value)){this.pushState("building-float"),this.castValueToFloat(m.left,p),p.push(C[m.operator]||m.operator),this.astGeneric(m.right,p),this.popState("building-float");break}if(this.pushState("building-integer"),this.astGeneric(m.left,p),p.push(C[m.operator]||m.operator),this.pushState("casting-to-integer"),m.right.type==="Literal"){const w=[];if(this.astGeneric(m.right,w),this.getType(m.right)==="Integer")p.push(w.join(""));else throw this.astErrorOutput("Unhandled binary expression with literal",m)}else p.push("int("),this.astGeneric(m.right,p),p.push(")");this.popState("casting-to-integer"),this.popState("building-integer");break;case"Integer & LiteralInteger":this.pushState("building-integer"),this.astGeneric(m.left,p),p.push(C[m.operator]||m.operator),this.castLiteralToInteger(m.right,p),this.popState("building-integer");break;case"Number & Integer":this.pushState("building-float"),this.astGeneric(m.left,p),p.push(C[m.operator]||m.operator),this.castValueToFloat(m.right,p),this.popState("building-float");break;case"Float & LiteralInteger":case"Number & LiteralInteger":this.pushState("building-float"),this.astGeneric(m.left,p),p.push(C[m.operator]||m.operator),this.castLiteralToFloat(m.right,p),this.popState("building-float");break;case"LiteralInteger & Float":case"LiteralInteger & Number":this.isState("casting-to-integer")?(this.pushState("building-integer"),this.castLiteralToInteger(m.left,p),p.push(C[m.operator]||m.operator),this.castValueToInteger(m.right,p),this.popState("building-integer")):(this.pushState("building-float"),this.astGeneric(m.left,p),p.push(C[m.operator]||m.operator),this.pushState("casting-to-float"),this.astGeneric(m.right,p),this.popState("casting-to-float"),this.popState("building-float"));break;case"LiteralInteger & Integer":this.pushState("building-integer"),this.castLiteralToInteger(m.left,p),p.push(C[m.operator]||m.operator),this.astGeneric(m.right,p),this.popState("building-integer");break;case"Boolean & Boolean":this.pushState("building-boolean"),this.astGeneric(m.left,p),p.push(C[m.operator]||m.operator),this.astGeneric(m.right,p),this.popState("building-boolean");break;case"Float & Integer":this.pushState("building-float"),this.astGeneric(m.left,p),p.push(C[m.operator]||m.operator),this.castValueToFloat(m.right,p),this.popState("building-float");break;default:throw this.astErrorOutput(`Unhandled binary expression between ${E}`,m)}return p.push(")"),p}checkAndUpconvertOperator(m,p){const u=this.checkAndUpconvertBitwiseOperators(m,p);if(u)return u;const g={"%":this.fixIntegerDivisionAccuracy?"integerCorrectionModulo":"modulo","**":"pow"}[m.operator];if(!g)return null;switch(p.push(g),p.push("("),this.getType(m.left)){case"Integer":this.castValueToFloat(m.left,p);break;case"LiteralInteger":this.castLiteralToFloat(m.left,p);break;default:this.astGeneric(m.left,p)}switch(p.push(","),this.getType(m.right)){case"Integer":this.castValueToFloat(m.right,p);break;case"LiteralInteger":this.castLiteralToFloat(m.right,p);break;default:this.astGeneric(m.right,p)}return p.push(")"),p}checkAndUpconvertBitwiseOperators(m,p){const u={"&":"bitwiseAnd","|":"bitwiseOr","^":"bitwiseXOR","<<":"bitwiseZeroFillLeftShift",">>":"bitwiseSignedRightShift",">>>":"bitwiseZeroFillRightShift"}[m.operator];if(!u)return null;switch(p.push(u),p.push("("),this.getType(m.left)){case"Number":case"Float":this.castValueToInteger(m.left,p);break;case"LiteralInteger":this.castLiteralToInteger(m.left,p);break;default:this.astGeneric(m.left,p)}switch(p.push(","),this.getType(m.right)){case"Number":case"Float":this.castValueToInteger(m.right,p);break;case"LiteralInteger":this.castLiteralToInteger(m.right,p);break;default:this.astGeneric(m.right,p)}return p.push(")"),p}checkAndUpconvertBitwiseUnary(m,p){const u={"~":"bitwiseNot"}[m.operator];if(!u)return null;switch(p.push(u),p.push("("),this.getType(m.argument)){case"Number":case"Float":this.castValueToInteger(m.argument,p);break;case"LiteralInteger":this.castLiteralToInteger(m.argument,p);break;default:this.astGeneric(m.argument,p)}return p.push(")"),p}castLiteralToInteger(m,p){return this.pushState("casting-to-integer"),this.astGeneric(m,p),this.popState("casting-to-integer"),p}castLiteralToFloat(m,p){return this.pushState("casting-to-float"),this.astGeneric(m,p),this.popState("casting-to-float"),p}castValueToInteger(m,p){return this.pushState("casting-to-integer"),p.push("int("),this.astGeneric(m,p),p.push(")"),this.popState("casting-to-integer"),p}castValueToFloat(m,p){return this.pushState("casting-to-float"),p.push("float("),this.astGeneric(m,p),p.push(")"),this.popState("casting-to-float"),p}astIdentifierExpression(m,p){if(m.type!=="Identifier")throw this.astErrorOutput("IdentifierExpression - not an Identifier",m);const u=this.getType(m),g=M.sanitizeName(m.name);return m.name==="Infinity"?p.push("3.402823466e+38"):u==="Boolean"?this.argumentNames.indexOf(g)>-1?p.push(`bool(user_${g})`):p.push(`user_${g}`):p.push(`user_${g}`),p}astForStatement(m,p){if(m.type!=="ForStatement")throw this.astErrorOutput("Invalid for statement",m);const u=[],g=[],E=[],w=[];let y=null;if(m.init){const{declarations:k}=m.init;k.length>1&&(y=!1),this.astGeneric(m.init,u);for(let v=0;v<k.length;v++)k[v].init&&k[v].init.type!=="Literal"&&(y=!1)}else y=!1;if(m.test?this.astGeneric(m.test,g):y=!1,m.update?this.astGeneric(m.update,E):y=!1,m.body&&(this.pushState("loop-body"),this.astGeneric(m.body,w),this.popState("loop-body")),y===null&&(y=this.isSafe(m.init)&&this.isSafe(m.test)),y){const k=u.join(""),v=k[k.length-1]!==";";p.push(`for (${k}${v?";":""}${g.join("")};${E.join("")}){
`),p.push(w.join("")),p.push(`}
`)}else{const k=this.getInternalVariableName("safeI");u.length>0&&p.push(u.join(""),`
`),p.push(`for (int ${k}=0;${k}<LOOP_MAX;${k}++){
`),g.length>0&&p.push(`if (!${g.join("")}) break;
`),p.push(w.join("")),p.push(`
${E.join("")};`),p.push(`}
`)}return p}astWhileStatement(m,p){if(m.type!=="WhileStatement")throw this.astErrorOutput("Invalid while statement",m);const u=this.getInternalVariableName("safeI");return p.push(`for (int ${u}=0;${u}<LOOP_MAX;${u}++){
`),p.push("if (!"),this.astGeneric(m.test,p),p.push(`) break;
`),this.astGeneric(m.body,p),p.push(`}
`),p}astDoWhileStatement(m,p){if(m.type!=="DoWhileStatement")throw this.astErrorOutput("Invalid while statement",m);const u=this.getInternalVariableName("safeI");return p.push(`for (int ${u}=0;${u}<LOOP_MAX;${u}++){
`),this.astGeneric(m.body,p),p.push("if (!"),this.astGeneric(m.test,p),p.push(`) break;
`),p.push(`}
`),p}astAssignmentExpression(m,p){if(m.operator==="%=")this.astGeneric(m.left,p),p.push("="),p.push("mod("),this.astGeneric(m.left,p),p.push(","),this.astGeneric(m.right,p),p.push(")");else if(m.operator==="**=")this.astGeneric(m.left,p),p.push("="),p.push("pow("),this.astGeneric(m.left,p),p.push(","),this.astGeneric(m.right,p),p.push(")");else{const u=this.getType(m.left),g=this.getType(m.right);return this.astGeneric(m.left,p),p.push(m.operator),u!=="Integer"&&g==="Integer"?(p.push("float("),this.astGeneric(m.right,p),p.push(")")):this.astGeneric(m.right,p),p}}astBlockStatement(m,p){if(this.isState("loop-body")){this.pushState("block-body");for(let u=0;u<m.body.length;u++)this.astStatementWithHoisting(m.body[u],p);this.popState("block-body")}else{p.push(`{
`);for(let u=0;u<m.body.length;u++)this.astStatementWithHoisting(m.body[u],p);p.push(`}
`)}return p}astStatementWithHoisting(m,p){switch(m.type){case"ExpressionStatement":case"VariableDeclaration":case"ReturnStatement":{const u=this.hoistedIndexReads,g=this.hoistedIndexReads=[],E=[];return this.astGeneric(m,E),this.hoistedIndexReads=u,p.push(...g,...E),p}default:return this.astGeneric(m,p)}}astVariableDeclaration(m,p){const u=m.declarations;if(!u||!u[0]||!u[0].init)throw this.astErrorOutput("Unexpected expression",m);const g=[];let E=null;const w=[];let y=[];for(let k=0;k<u.length;k++){const v=u[k],$=v.init,P=this.getDeclaration(v.id),O=this.getType(v.init);let T=O;T==="LiteralInteger"&&(P.suggestedType==="Integer"?T="Integer":T="Number");const A=d[T];if(!A)throw this.astErrorOutput(`Markup type ${T} not handled`,m);const f=[];if(O==="Integer"&&T==="Integer"){if(P.valueType="Number",k===0||E===null)f.push("float ");else if(T!==E)throw new Error("Unhandled declaration");E=T,f.push(`user_${M.sanitizeName(v.id.name)}=`),f.push("float("),this.astGeneric($,f),f.push(")")}else P.valueType=T,k===0||E===null?f.push(`${A} `):T!==E&&(w.push(y.join(",")),y=[],f.push(`${A} `)),E=T,f.push(`user_${M.sanitizeName(v.id.name)}=`),O==="Number"&&T==="Integer"?$.left&&$.left.type==="Literal"?this.astGeneric($,f):(f.push("int("),this.astGeneric($,f),f.push(")")):O==="LiteralInteger"&&T==="Integer"?this.castLiteralToInteger($,f):this.astGeneric($,f);y.push(f.join(""))}return y.length>0&&w.push(y.join(",")),g.push(w.join(";")),p.push(g.join("")),p.push(";"),p}astIfStatement(m,p){return p.push("if ("),this.astGeneric(m.test,p),p.push(")"),m.consequent.type==="BlockStatement"?this.astGeneric(m.consequent,p):(p.push(` {
`),this.astGeneric(m.consequent,p),p.push(`
}
`)),m.alternate&&(p.push("else "),m.alternate.type==="BlockStatement"||m.alternate.type==="IfStatement"?this.astGeneric(m.alternate,p):(p.push(` {
`),this.astGeneric(m.alternate,p),p.push(`
}
`))),p}astSwitchStatement(m,p){if(m.type!=="SwitchStatement")throw this.astErrorOutput("Invalid switch statement",m);const{discriminant:u,cases:g}=m,E=this.getType(u),w=`switchDiscriminant${this.astKey(m,"_")}`;switch(E){case"Float":case"Number":p.push(`float ${w} = `),this.astGeneric(u,p),p.push(`;
`);break;case"Integer":p.push(`int ${w} = `),this.astGeneric(u,p),p.push(`;
`);break}if(g.length===1&&!g[0].test)return this.astGeneric(g[0].consequent,p),p;let y=!1,k=[],v=!1,$=!1;for(let P=0;P<g.length;P++){if(g[P].test){if(P===0||!$?($=!0,p.push(`if (${w} == `)):y?(p.push(`${w} == `),y=!1):p.push(` else if (${w} == `),E==="Integer")switch(this.getType(g[P].test)){case"Number":case"Float":this.castValueToInteger(g[P].test,p);break;case"LiteralInteger":this.castLiteralToInteger(g[P].test,p);break}else if(E==="Float")switch(this.getType(g[P].test)){case"LiteralInteger":this.castLiteralToFloat(g[P].test,p);break;case"Integer":this.castValueToFloat(g[P].test,p);break}else throw new Error("unhanlded");if(!g[P].consequent||g[P].consequent.length===0){y=!0,p.push(" || ");continue}p.push(`) {
`)}else if(g.length>P+1){v=!0,this.astGeneric(g[P].consequent,k);continue}else p.push(` else {
`);this.astGeneric(g[P].consequent,p),p.push(`
}`)}return v&&(p.push(" else {"),p.push(k.join("")),p.push("}")),p}astThisExpression(m,p){return p.push("this"),p}astMemberExpression(m,p){const{property:u,name:g,signature:E,origin:w,type:y,xProperty:k,yProperty:v,zProperty:$}=this.getMemberExpressionDetails(m);switch(E){case"value.thread.value":case"this.thread.value":if(g!=="x"&&g!=="y"&&g!=="z")throw this.astErrorOutput("Unexpected expression, expected `this.thread.x`, `this.thread.y`, or `this.thread.z`",m);return p.push(`threadId.${g}`),p;case"this.output.value":if(this.dynamicOutput)switch(g){case"x":this.isState("casting-to-float")?p.push("float(uOutputDim.x)"):p.push("uOutputDim.x");break;case"y":this.isState("casting-to-float")?p.push("float(uOutputDim.y)"):p.push("uOutputDim.y");break;case"z":this.isState("casting-to-float")?p.push("float(uOutputDim.z)"):p.push("uOutputDim.z");break;default:throw this.astErrorOutput("Unexpected expression",m)}else switch(g){case"x":this.isState("casting-to-integer")?p.push(this.output[0]):p.push(this.output[0],".0");break;case"y":this.isState("casting-to-integer")?p.push(this.output[1]):p.push(this.output[1],".0");break;case"z":this.isState("casting-to-integer")?p.push(this.output[2]):p.push(this.output[2],".0");break;default:throw this.astErrorOutput("Unexpected expression",m)}return p;case"value":throw this.astErrorOutput("Unexpected expression",m);case"value[]":case"value[][]":case"value[][][]":case"value[][][][]":case"value.value":if(w==="Math")return p.push(Math[g]),p;const O=M.sanitizeName(g);switch(u){case"r":return p.push(`user_${O}.r`),p;case"g":return p.push(`user_${O}.g`),p;case"b":return p.push(`user_${O}.b`),p;case"a":return p.push(`user_${O}.a`),p}break;case"this.constants.value":if(typeof k>"u")switch(y){case"Array(2)":case"Array(3)":case"Array(4)":return p.push(`constants_${M.sanitizeName(g)}`),p}case"this.constants.value[]":case"this.constants.value[][]":case"this.constants.value[][][]":case"this.constants.value[][][][]":break;case"fn()[]":return this.astCallExpression(m.object,p),p.push("["),p.push(this.memberExpressionPropertyMarkup(u)),p.push("]"),p;case"fn()[][]":{const T=m.object.property,A=m.property,f=l[this.getType(m.object.object)],F=L=>this.getType(L)==="LiteralInteger";return f&&!(F(T)&&F(A))?(p.push(`getMatrix${f}(`),this.astCallExpression(m.object.object,p),p.push(", "),p.push(this.memberExpressionPropertyMarkup(T)),p.push(", "),p.push(this.memberExpressionPropertyMarkup(A)),p.push(")"),p):(this.astCallExpression(m.object.object,p),p.push("["),p.push(this.memberExpressionPropertyMarkup(T)),p.push("]"),p.push("["),p.push(this.memberExpressionPropertyMarkup(A)),p.push("]"),p)}case"[][]":return this.astArrayExpression(m.object,p),p.push("["),p.push(this.memberExpressionPropertyMarkup(u)),p.push("]"),p;default:throw this.astErrorOutput("Unexpected expression",m)}if(m.computed===!1)switch(y){case"Number":case"Integer":case"Float":case"Boolean":return p.push(`${w}_${M.sanitizeName(g)}`),p}const P=`${w}_${M.sanitizeName(g)}`;switch(y){case"Array(2)":case"Array(3)":case"Array(4)":this.astGeneric(m.object,p),p.push("["),p.push(this.memberExpressionPropertyMarkup(k)),p.push("]");break;case"HTMLImageArray":p.push(`getImage3D(${P}, ${P}Size, ${P}Dim, `),this.memberExpressionXYZ(k,v,$,p),p.push(")");break;case"ArrayTexture(1)":p.push(`getFloatFromSampler2D(${P}, ${P}Size, ${P}Dim, `),this.memberExpressionXYZ(k,v,$,p),p.push(")");break;case"Array1D(2)":case"Array2D(2)":case"Array3D(2)":p.push(`getMemoryOptimizedVec2(${P}, ${P}Size, ${P}Dim, `),this.memberExpressionXYZ(k,v,$,p),p.push(")");break;case"ArrayTexture(2)":p.push(`getVec2FromSampler2D(${P}, ${P}Size, ${P}Dim, `),this.memberExpressionXYZ(k,v,$,p),p.push(")");break;case"Array1D(3)":case"Array2D(3)":case"Array3D(3)":p.push(`getMemoryOptimizedVec3(${P}, ${P}Size, ${P}Dim, `),this.memberExpressionXYZ(k,v,$,p),p.push(")");break;case"ArrayTexture(3)":p.push(`getVec3FromSampler2D(${P}, ${P}Size, ${P}Dim, `),this.memberExpressionXYZ(k,v,$,p),p.push(")");break;case"Array1D(4)":case"Array2D(4)":case"Array3D(4)":p.push(`getMemoryOptimizedVec4(${P}, ${P}Size, ${P}Dim, `),this.memberExpressionXYZ(k,v,$,p),p.push(")");break;case"ArrayTexture(4)":case"HTMLCanvas":case"OffscreenCanvas":case"HTMLImage":case"ImageBitmap":case"ImageData":case"HTMLVideo":p.push(`getVec4FromSampler2D(${P}, ${P}Size, ${P}Dim, `),this.memberExpressionXYZ(k,v,$,p),p.push(")");break;case"NumberTexture":case"Array":case"Array2D":case"Array3D":case"Array4D":case"Input":case"Number":case"Float":case"Integer":if(this.precision==="single")p.push(`getMemoryOptimized32(${P}, ${P}Size, ${P}Dim, `),this.memberExpressionXYZ(k,v,$,p),p.push(")");else{const O=w==="user"?this.lookupFunctionArgumentBitRatio(this.name,g):this.constantBitRatios[g];switch(O){case 1:p.push(`get8(${P}, ${P}Size, ${P}Dim, `);break;case 2:p.push(`get16(${P}, ${P}Size, ${P}Dim, `);break;case 4:case 0:p.push(`get32(${P}, ${P}Size, ${P}Dim, `);break;default:throw new Error(`unhandled bit ratio of ${O}`)}this.memberExpressionXYZ(k,v,$,p),p.push(")")}break;case"MemoryOptimizedNumberTexture":p.push(`getMemoryOptimized32(${P}, ${P}Size, ${P}Dim, `),this.memberExpressionXYZ(k,v,$,p),p.push(")");break;case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":p.push(`${P}[${this.memberExpressionPropertyMarkup(v)}]`),v&&p.push(`[${this.memberExpressionPropertyMarkup(k)}]`);break;default:throw new Error(`unhandled member expression "${y}"`)}return p}astCallExpression(m,p){if(!m.callee)throw this.astErrorOutput("Unknown CallExpression",m);let u=null;const g=this.isAstMathFunction(m);if(g||m.callee.object&&m.callee.object.type==="ThisExpression"?u=m.callee.property.name:m.callee.type==="SequenceExpression"&&m.callee.expressions[0].type==="Literal"&&!isNaN(m.callee.expressions[0].raw)?u=m.callee.expressions[1].property.name:u=m.callee.name,!u)throw this.astErrorOutput("Unhandled function, couldn't find name",m);switch(u){case"pow":u="_pow";break;case"round":u="_round";break}if(this.calledFunctions.indexOf(u)<0&&this.calledFunctions.push(u),u==="random"&&this.plugins&&this.plugins.length>0)for(let E=0;E<this.plugins.length;E++){const w=this.plugins[E];if(w.functionMatch==="Math.random()"&&w.functionReplace)return p.push(w.functionReplace),p}if(this.onFunctionCall&&this.onFunctionCall(this.name,u,m.arguments),p.push(u),p.push("("),g)for(let E=0;E<m.arguments.length;++E){const w=m.arguments[E],y=this.getType(w);switch(E>0&&p.push(", "),y){case"Integer":this.castValueToFloat(w,p);break;default:this.astGeneric(w,p);break}}else{const E=this.lookupFunctionArgumentTypes(u)||[];for(let w=0;w<m.arguments.length;++w){const y=m.arguments[w];let k=E[w];w>0&&p.push(", ");const v=this.getType(y);switch(k||(this.triggerImplyArgumentType(u,w,v,this),k=v),v){case"Boolean":this.astGeneric(y,p);continue;case"Number":case"Float":if(k==="Integer"){p.push("int("),this.astGeneric(y,p),p.push(")");continue}else if(k==="Number"||k==="Float"){this.astGeneric(y,p);continue}else if(k==="LiteralInteger"){this.castLiteralToFloat(y,p);continue}break;case"Integer":if(k==="Number"||k==="Float"){p.push("float("),this.astGeneric(y,p),p.push(")");continue}else if(k==="Integer"){this.astGeneric(y,p);continue}break;case"LiteralInteger":if(k==="Integer"){this.castLiteralToInteger(y,p);continue}else if(k==="Number"||k==="Float"){this.castLiteralToFloat(y,p);continue}else if(k==="LiteralInteger"){this.astGeneric(y,p);continue}break;case"Array(2)":case"Array(3)":case"Array(4)":if(k===v){if(y.type==="Identifier")p.push(`user_${M.sanitizeName(y.name)}`);else if(y.type==="ArrayExpression"||y.type==="MemberExpression"||y.type==="CallExpression")this.astGeneric(y,p);else throw this.astErrorOutput(`Unhandled argument type ${y.type}`,m);continue}break;case"HTMLCanvas":case"OffscreenCanvas":case"HTMLImage":case"ImageBitmap":case"ImageData":case"HTMLImageArray":case"HTMLVideo":case"ArrayTexture(1)":case"ArrayTexture(2)":case"ArrayTexture(3)":case"ArrayTexture(4)":case"Array":case"Input":if(k===v){if(y.type!=="Identifier")throw this.astErrorOutput(`Unhandled argument type ${y.type}`,m);this.triggerImplyArgumentBitRatio(this.name,y.name,u,w);const $=M.sanitizeName(y.name);p.push(`user_${$},user_${$}Size,user_${$}Dim`);continue}break}throw this.astErrorOutput(`Unhandled argument combination of ${v} and ${k} for argument named "${y.name}"`,m)}}return p.push(")"),p}astArrayExpression(m,p){const u=this.getType(m),g=m.elements.length;switch(u){case"Matrix(2)":case"Matrix(3)":case"Matrix(4)":p.push(`mat${g}(`);break;default:p.push(`vec${g}(`)}for(let E=0;E<g;++E){E>0&&p.push(", ");const w=m.elements[E];this.astGeneric(w,p)}return p.push(")"),p}memberExpressionXYZ(m,p,u,g){return u?g.push(this.memberExpressionPropertyMarkup(u),", "):g.push("0, "),p?g.push(this.memberExpressionPropertyMarkup(p),", "):g.push("0, "),g.push(this.memberExpressionPropertyMarkup(m)),g}memberExpressionPropertyMarkup(m){if(!m)throw new Error("Property not set");const p=this.getType(m),u=[];switch(p){case"Number":case"Float":this.castValueToInteger(m,u);break;case"LiteralInteger":this.castLiteralToInteger(m,u);break;default:this.astGeneric(m,u)}const g=u.join("");if(this.hoistedIndexReads&&/\b\w+\((user_|constants_)\w+, \1\w+Size/.test(g)){const E=`hoisted_${this.hoistedIndexReads.length}_${M.sanitizeName(this.name)}`,w=g.startsWith("int(");return this.hoistedIndexReads.push(`${w?"int":"float"} ${E}=${g};
`),E}return g}};const l={"Matrix(2)":2,"Matrix(3)":3,"Matrix(4)":4},d={Array:"sampler2D","Array(2)":"vec2","Array(3)":"vec3","Array(4)":"vec4","Matrix(2)":"mat2","Matrix(3)":"mat3","Matrix(4)":"mat4",Array2D:"sampler2D",Array3D:"sampler2D",Boolean:"bool",Float:"float",Input:"sampler2D",Integer:"int",Number:"float",LiteralInteger:"float",NumberTexture:"sampler2D",MemoryOptimizedNumberTexture:"sampler2D","ArrayTexture(1)":"sampler2D","ArrayTexture(2)":"sampler2D","ArrayTexture(3)":"sampler2D","ArrayTexture(4)":"sampler2D",HTMLVideo:"sampler2D",HTMLCanvas:"sampler2D",OffscreenCanvas:"sampler2D",HTMLImage:"sampler2D",ImageBitmap:"sampler2D",ImageData:"sampler2D",HTMLImageArray:"sampler2DArray"},C={"===":"==","!==":"!="};z.exports={WebGLFunctionNode:b}}),Dr=s((B,z)=>{const M=`// https://www.shadertoy.com/view/4t2SDh
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
}`,_="math-random-uniformly-distributed",b="Math.random()",l="nrand(vTexCoord)",d="Number";function C(p){let u=p>>>0;return function(){u=u+1831565813>>>0;let g=u;return g=Math.imul(g^g>>>15,g|1),g^=g+Math.imul(g^g>>>7,g|61),((g^g>>>14)>>>0)/4294967296}}const m=p=>{if(p.randomSeed===null||p.randomSeed===void 0){p.setUniform1f("randomSeed1",Math.random()),p.setUniform1f("randomSeed2",Math.random());return}(!p._mathRandomGenerator||p._mathRandomGeneratorSeed!==p.randomSeed)&&(p._mathRandomGenerator=C(p.randomSeed),p._mathRandomGeneratorSeed=p.randomSeed),p.setUniform1f("randomSeed1",p._mathRandomGenerator()),p.setUniform1f("randomSeed2",p._mathRandomGenerator())};z.exports={name:_,onBeforeRun:m,functionMatch:b,functionReplace:l,functionReturnType:d,source:M}}),eo=s((B,z)=>{z.exports={fragmentShader:`__HEADER__;
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
}`}}),to=s((B,z)=>{z.exports={vertexShader:`__FLOAT_TACTIC_DECLARATION__;
__INT_TACTIC_DECLARATION__;
__SAMPLER_2D_TACTIC_DECLARATION__;

attribute vec2 aPos;
attribute vec2 aTexCoord;

varying vec2 vTexCoord;
uniform vec2 ratio;

void main(void) {
  gl_Position = vec4((aPos + vec2(1)) * ratio + vec2(-1), 0, 1);
  vTexCoord = aTexCoord;
}`}}),so=s((B,z)=>{function M(C,m={}){const{contextName:p="gl",throwGetError:u,useTrackablePrimitives:g,recording:E=[],variables:w={},onReadPixels:y,onUnrecognizedArgumentLookup:k}=m,v=new Proxy(C,{get:A}),$=[],P={};let O="",T;return v;function A(se,J){switch(J){case"addComment":return N;case"checkThrowError":return te;case"getReadPixelsVariableName":return T;case"insertVariable":return L;case"reset":return F;case"setIndent":return V;case"toString":return f;case"getContextVariableName":return ie}return typeof C[J]=="function"?function(){switch(J){case"getError":return u?E.push(`${O}if (${p}.getError() !== ${p}.NONE) throw new Error('error');`):E.push(`${O}${p}.getError();`),C.getError();case"getExtension":{const ce=`${p}Variables${$.length}`;E.push(`${O}const ${ce} = ${p}.getExtension('${arguments[0]}');`);const Ve=C.getExtension(arguments[0]);if(Ve&&typeof Ve=="object"){const ue=_(Ve,{getEntity:R,useTrackablePrimitives:g,recording:E,contextName:ce,contextVariables:$,variables:w,indent:O,onUnrecognizedArgumentLookup:k});return $.push(ue),ue}else $.push(null);return Ve}case"readPixels":const he=$.indexOf(arguments[6]);let Ae;if(he===-1){const ce=X(arguments[6]);ce?(Ae=ce,E.push(`${O}${ce}`)):(Ae=`${p}Variable${$.length}`,$.push(arguments[6]),E.push(`${O}const ${Ae} = new ${arguments[6].constructor.name}(${arguments[6].length});`))}else Ae=`${p}Variable${he}`;T=Ae;const ne=[arguments[0],arguments[1],arguments[2],arguments[3],R(arguments[4]),R(arguments[5]),Ae];return E.push(`${O}${p}.readPixels(${ne.join(", ")});`),y&&y(Ae,ne),C.readPixels.apply(C,arguments);case"drawBuffers":return E.push(`${O}${p}.drawBuffers([${b(arguments[0],{contextName:p,contextVariables:$,getEntity:R,addVariable:W,variables:w,onUnrecognizedArgumentLookup:k})}]);`),C.drawBuffers(arguments[0])}let ae=C[J].apply(C,arguments);switch(typeof ae){case"undefined":E.push(`${O}${ee(J,arguments)};`);return;case"number":case"boolean":if(g&&$.indexOf(d(ae))===-1){E.push(`${O}const ${p}Variable${$.length} = ${ee(J,arguments)};`),$.push(ae=d(ae));break}default:ae===null?E.push(`${ee(J,arguments)};`):E.push(`${O}const ${p}Variable${$.length} = ${ee(J,arguments)};`),$.push(ae)}return ae}:(P[C[J]]=J,C[J])}function f(){return E.join(`
`)}function F(){for(;E.length>0;)E.pop()}function L(se,J){w[se]=J}function R(se){const J=P[se];return J?p+"."+J:se}function V(se){O=" ".repeat(se)}function W(se,J){const ae=`${p}Variable${$.length}`;return E.push(`${O}const ${ae} = ${J};`),$.push(se),ae}function N(se){E.push(`${O}// ${se}`)}function te(){E.push(`${O}(() => {
${O}const error = ${p}.getError();
${O}if (error !== ${p}.NONE) {
${O}  const names = Object.getOwnPropertyNames(gl);
${O}  for (let i = 0; i < names.length; i++) {
${O}    const name = names[i];
${O}    if (${p}[name] === error) {
${O}      throw new Error('${p} threw ' + name);
${O}    }
${O}  }
${O}}
${O}})();`)}function ee(se,J){return`${p}.${se}(${b(J,{contextName:p,contextVariables:$,getEntity:R,addVariable:W,variables:w,onUnrecognizedArgumentLookup:k})})`}function X(se){if(w){for(const J in w)if(w[J]===se)return J}return null}function ie(se){const J=$.indexOf(se);return J!==-1?`${p}Variable${J}`:null}}function _(C,m){const p=new Proxy(C,{get:O}),u={},{contextName:g,contextVariables:E,getEntity:w,useTrackablePrimitives:y,recording:k,variables:v,indent:$,onUnrecognizedArgumentLookup:P}=m;return p;function O(F,L){return typeof F[L]=="function"?function(){switch(L){case"drawBuffersWEBGL":return k.push(`${$}${g}.drawBuffersWEBGL([${b(arguments[0],{contextName:g,contextVariables:E,getEntity:T,addVariable:f,variables:v,onUnrecognizedArgumentLookup:P})}]);`),C.drawBuffersWEBGL(arguments[0])}let R=C[L].apply(C,arguments);switch(typeof R){case"undefined":k.push(`${$}${A(L,arguments)};`);return;case"number":case"boolean":y&&E.indexOf(d(R))===-1?(k.push(`${$}const ${g}Variable${E.length} = ${A(L,arguments)};`),E.push(R=d(R))):(k.push(`${$}const ${g}Variable${E.length} = ${A(L,arguments)};`),E.push(R));break;default:R===null?k.push(`${A(L,arguments)};`):k.push(`${$}const ${g}Variable${E.length} = ${A(L,arguments)};`),E.push(R)}return R}:(u[C[L]]=L,C[L])}function T(F){return u.hasOwnProperty(F)?`${g}.${u[F]}`:w(F)}function A(F,L){return`${g}.${F}(${b(L,{contextName:g,contextVariables:E,getEntity:T,addVariable:f,variables:v,onUnrecognizedArgumentLookup:P})})`}function f(F,L){const R=`${g}Variable${E.length}`;return E.push(F),k.push(`${$}const ${R} = ${L};`),R}}function b(C,m){const{variables:p,onUnrecognizedArgumentLookup:u}=m;return Array.from(C).map(E=>{const w=g(E);return w||l(E,m)}).join(", ");function g(E){if(p){for(const w in p)if(p.hasOwnProperty(w)&&p[w]===E)return w}return u?u(E):null}}function l(C,m){const{contextName:p,contextVariables:u,getEntity:g,addVariable:E,onUnrecognizedArgumentLookup:w}=m;if(typeof C>"u")return"undefined";if(C===null)return"null";const y=u.indexOf(C);if(y>-1)return`${p}Variable${y}`;switch(C.constructor.name){case"String":const k=/\n/.test(C),v=/'/.test(C),$=/"/.test(C);return k?"`"+C+"`":v&&!$?'"'+C+'"':"'"+C+"'";case"Number":return g(C);case"Boolean":return g(C);case"Array":return E(C,`new ${C.constructor.name}([${Array.from(C).join(",")}])`);case"Float32Array":case"Uint8Array":case"Uint16Array":case"Int32Array":return E(C,`new ${C.constructor.name}(${JSON.stringify(Array.from(C))})`);default:if(w){const P=w(C);if(P)return P}throw new Error(`unrecognized argument type ${C.constructor.name}`)}}function d(C){return new C.constructor(C)}typeof z<"u"&&(z.exports={glWiretap:M,glExtensionWiretap:_}),typeof window<"u"&&(M.glExtensionWiretap=_,window.glWiretap=M)}),Pr=s((B,z)=>{const{glWiretap:M}=so(),{utils:_}=h();function b(u){let g=u.toString().replace(/^function /,"");const E=g.indexOf("=>");if(E!==-1&&!/[{]|\bfunction\b/.test(g.slice(0,E))){const w=g.slice(0,E).trim(),y=g.slice(E+2).trim();g=y.startsWith("{")?`${w} ${y}`:`${w} { return ${y}; }`}return g.replace(/utils[.]/g,"/*utils.*/")}function l(u,g,E,w,y){E.built||E.build.apply(E,g),g=g?Array.from(g).map(ue=>{switch(typeof ue){case"boolean":return new Boolean(ue);case"number":return new Number(ue);default:return ue}}):null;const k=[],v=M(E.context,{useTrackablePrimitives:!0,onReadPixels:ue=>{if(ne.subKernels){if(!$)k.push(`    const result = { result: ${d(ue,ne)} };`),$=!0;else{const Te=ne.subKernels[P++].property;k.push(`    result${isNaN(Te)?"."+Te:`[${Te}]`} = ${d(ue,ne)};`)}P===ne.subKernels.length&&k.push("    return result;");return}ue?k.push(`    return ${d(ue,ne)};`):k.push("    return null;")},onUnrecognizedArgumentLookup:ue=>{const Te=p(ue,ne.kernelArguments,[],v);if(Te)return Te;const Oe=p(ue,ne.kernelConstants,R?Object.keys(R).map(Ie=>R[Ie]):[],v);return Oe||null}});let $=!1,P=0;const{source:O,canvas:T,output:A,pipeline:f,graphical:F,loopMaxIterations:L,constants:R,optimizeFloatMemory:V,precision:W,fixIntegerDivisionAccuracy:N,functions:te,nativeFunctions:ee,subKernels:X,immutable:ie,argumentTypes:se,constantTypes:J,kernelArguments:ae,kernelConstants:he,tactic:Ae}=E,ne=new u(O,{canvas:T,context:v,checkContext:!1,output:A,pipeline:f,graphical:F,loopMaxIterations:L,constants:R,optimizeFloatMemory:V,precision:W,fixIntegerDivisionAccuracy:N,functions:te,nativeFunctions:ee,subKernels:X,immutable:ie,argumentTypes:se,constantTypes:J,tactic:Ae});let ce=[];if(v.setIndent(2),ne.build.apply(ne,g),ce.push(v.toString()),v.reset(),ne.kernelArguments.forEach((ue,Te)=>{switch(ue.type){case"Integer":case"Boolean":case"Number":case"Float":case"Array":case"Array(2)":case"Array(3)":case"Array(4)":case"HTMLCanvas":case"HTMLImage":case"HTMLVideo":v.insertVariable(`uploadValue_${ue.name}`,ue.uploadValue);break;case"HTMLImageArray":for(let Oe=0;Oe<g[Te].length;Oe++){const Ie=g[Te];v.insertVariable(`uploadValue_${ue.name}[${Oe}]`,Ie[Oe])}break;case"Input":v.insertVariable(`uploadValue_${ue.name}`,ue.uploadValue);break;case"MemoryOptimizedNumberTexture":case"NumberTexture":case"Array1D(2)":case"Array1D(3)":case"Array1D(4)":case"Array2D(2)":case"Array2D(3)":case"Array2D(4)":case"Array3D(2)":case"Array3D(3)":case"Array3D(4)":case"ArrayTexture(1)":case"ArrayTexture(2)":case"ArrayTexture(3)":case"ArrayTexture(4)":v.insertVariable(`uploadValue_${ue.name}`,g[Te].texture);break;default:throw new Error(`unhandled kernelArgumentType insertion for glWiretap of type ${ue.type}`)}}),ce.push("/** start of injected functions **/"),ce.push(`function ${b(_.flattenTo)}`),ce.push(`function ${b(_.flatten2dArrayTo)}`),ce.push(`function ${b(_.flatten3dArrayTo)}`),ce.push(`function ${b(_.flatten4dArrayTo)}`),ce.push(`function ${b(_.isArray)}`),ne.renderOutput!==ne.renderTexture&&ne.formatValues&&ce.push(`  const renderOutput = function ${b(ne.formatValues)};`),ce.push(`let readFramebuffer = null;
function getReadFramebuffer() {
  if (!readFramebuffer) readFramebuffer = gl.createFramebuffer();
  return readFramebuffer;
}`),ce.push("/** end of injected functions **/"),ce.push(`  const innerKernel = function (${ne.kernelArguments.map(ue=>ue.varName).join(", ")}) {`),v.setIndent(4),ne.run.apply(ne,g),ne.renderKernels?ne.renderKernels():ne.renderOutput&&ne.renderOutput(),ce.push("    /** start setup uploads for kernel values **/"),ne.kernelArguments.forEach(ue=>{ce.push("    "+ue.getStringValueHandler().split(`
`).join(`
    `))}),ce.push("    /** end setup uploads for kernel values **/"),ce.push(v.toString()),ne.renderOutput===ne.renderTexture)if(v.reset(),ne.renderKernels){const ue=ne.renderKernels(),Te=v.getContextVariableName(ne.texture.texture);ce.push(`    return {
      result: {
        texture: ${Te},
        type: '${ue.result.type}',
        toArray: ${m(ue.result,Te)}
      },`);const{subKernels:Oe,mappedTextures:Ie}=ne;for(let re=0;re<Oe.length;re++){const ye=Ie[re],_e=Oe[re],Ne=ue[_e.property],ft=v.getContextVariableName(ye.texture);ce.push(`
      ${_e.property}: {
        texture: ${ft},
        type: '${Ne.type}',
        toArray: ${m(Ne,ft)}
      },`)}ce.push("    };")}else{const ue=ne.renderOutput(),Te=v.getContextVariableName(ne.texture.texture);ce.push(`    return {
        texture: ${Te},
        type: '${ue.type}',
        toArray: ${m(ue,Te)}
      };`)}ce.push(`    ${y?`
`+y+"    ":""}`),ce.push(k.join(`
`)),ce.push("  };"),ne.graphical&&(ce.push(C(ne)),ce.push("  innerKernel.getPixels = getPixels;")),ce.push("  return innerKernel;");let Ve=[];return he.forEach(ue=>{Ve.push(`${ue.getStringValueHandler()}`)}),`function kernel(settings) {
  const { context, constants } = settings;
  ${Ve.join("")}
  ${w||""}
${ce.join(`
`)}
}`}function d(u,g){const E=g.precision==="single"?u:`new Float32Array(${u}.buffer)`;return g.output[2]?`renderOutput(${E}, ${g.output[0]}, ${g.output[1]}, ${g.output[2]})`:g.output[1]?`renderOutput(${E}, ${g.output[0]}, ${g.output[1]})`:`renderOutput(${E}, ${g.output[0]})`}function C(u){const g=u.getPixels.toString(),E=!/^function/.test(g);return _.flattenFunctionToString(`${E?"function ":""}${g}`,{findDependency:(w,y)=>w==="utils"?`const ${y} = ${_[y].toString()};`:null,thisLookup:w=>{if(w==="context")return null;if(u.hasOwnProperty(w))return JSON.stringify(u[w]);throw new Error(`unhandled thisLookup ${w}`)}})}function m(u,g){const E=u.toArray.toString(),w=!/^function/.test(E);return`() => {
  function framebuffer() { return getReadFramebuffer(); };
  ${_.flattenFunctionToString(`${w?"function ":""}${E}`,{findDependency:(y,k)=>{if(y==="utils")return`const ${k} = ${_[k].toString()};`;if(y==="this")return k==="framebuffer"?"":`${w?"function ":""}${u[k].toString()}`;throw new Error("unhandled fromObject")},thisLookup:(y,k)=>{if(y==="texture")return g;if(y==="context")return k?null:"gl";if(u.hasOwnProperty(y))return JSON.stringify(u[y]);throw new Error(`unhandled thisLookup ${y}`)}})}
  return toArray();
  }`}function p(u,g,E,w,y){if(u===null||g===null)return null;switch(typeof u){case"boolean":case"number":return null}if(typeof HTMLImageElement<"u"&&u instanceof HTMLImageElement)for(let k=0;k<g.length;k++){const v=g[k];if(v.type!=="HTMLImageArray"&&v||v.uploadValue!==u)continue;const $=E[k].indexOf(u);if($===-1)continue;const P=`uploadValue_${v.name}[${$}]`;return w.insertVariable(P,u),P}for(let k=0;k<g.length;k++){const v=g[k];if(u!==v.uploadValue)continue;const $=`uploadValue_${v.name}`;return w.insertVariable($,v),$}return null}z.exports={glKernelString:l}}),no=s((B,z)=>{var M=class{constructor(_,b){const{name:l,kernel:d,context:C,checkContext:m,onRequestContextHandle:p,onUpdateValueMismatch:u,origin:g,strictIntegers:E,type:w,tactic:y}=b;if(!l)throw new Error("name not set");if(!w)throw new Error("type not set");if(!g)throw new Error("origin not set");if(g!=="user"&&g!=="constants")throw new Error(`origin must be "user" or "constants" value is "${g}"`);if(!p)throw new Error("onRequestContextHandle is not set");this.name=l,this.origin=g,this.tactic=y,this.varName=g==="constants"?`constants.${l}`:l,this.kernel=d,this.strictIntegers=E,this.type=_.type||w,this.size=_.size||null,this.index=null,this.context=C,this.checkContext=m??!0,this.contextHandle=null,this.onRequestContextHandle=p,this.onUpdateValueMismatch=u,this.forceUploadEachRun=null}get id(){return`${this.origin}_${name}`}getSource(){throw new Error(`"getSource" not defined on ${this.constructor.name}`)}updateValue(_){throw new Error(`"updateValue" not defined on ${this.constructor.name}`)}};z.exports={KernelValue:M}}),Pt=s((B,z)=>{const{utils:M}=h(),{KernelValue:_}=no();var b=class extends _{constructor(l,d){super(l,d),this.dimensionsId=null,this.sizeId=null,this.initialValueConstructor=l.constructor,this.onRequestTexture=d.onRequestTexture,this.onRequestIndex=d.onRequestIndex,this.uploadValue=null,this.textureSize=null,this.bitRatio=null,this.prevArg=null}get id(){return`${this.origin}_${M.sanitizeName(this.name)}`}setup(){}getTransferArrayType(l){if(Array.isArray(l[0]))return this.getTransferArrayType(l[0]);switch(l.constructor){case Array:case Int32Array:case Int16Array:case Int8Array:return Float32Array;case Uint8ClampedArray:case Uint8Array:case Uint16Array:case Uint32Array:case Float32Array:case Float64Array:return l.constructor}return console.warn("Unfamiliar constructor type.  Will go ahead and use, but likley this may result in a transfer of zeros"),l.constructor}getStringValueHandler(){throw new Error(`"getStringValueHandler" not implemented on ${this.constructor.name}`)}getVariablePrecisionString(){return this.kernel.getVariablePrecisionString(this.textureSize||void 0,this.tactic||void 0)}destroy(){}};z.exports={WebGLKernelValue:b}}),zr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValue:_}=Pt();var b=class extends _{constructor(l,d){super(l,d),this.uploadValue=l}getSource(l){return this.origin==="constants"?`const bool ${this.id} = ${l};
`:`uniform bool ${this.id};
`}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName};
`}updateValue(l){this.origin!=="constants"&&this.kernel.setUniform1i(this.id,this.uploadValue=l)}};z.exports={WebGLKernelValueBoolean:b}}),Rr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValue:_}=Pt();var b=class extends _{constructor(l,d){super(l,d),this.uploadValue=l}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName};
`}getSource(l){return this.origin==="constants"?Number.isInteger(l)?`const float ${this.id} = ${l}.0;
`:`const float ${this.id} = ${l};
`:`uniform float ${this.id};
`}updateValue(l){this.origin!=="constants"&&this.kernel.setUniform1f(this.id,this.uploadValue=l)}};z.exports={WebGLKernelValueFloat:b}}),Or=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValue:_}=Pt();var b=class extends _{constructor(l,d){super(l,d),this.uploadValue=l}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName};
`}getSource(l){return this.origin==="constants"?`const int ${this.id} = ${parseInt(l)};
`:`uniform int ${this.id};
`}updateValue(l){this.origin!=="constants"&&this.kernel.setUniform1i(this.id,this.uploadValue=l)}};z.exports={WebGLKernelValueInteger:b}}),Qe=s((B,z)=>{const{WebGLKernelValue:M}=Pt(),{Input:_}=a();var b=class extends M{checkSize(l,d){if(!this.kernel.validate)return;const{maxTextureSize:C}=this.kernel.constructor.features;if(l>C||d>C)throw l>d?new Error(`Argument texture width of ${l} larger than maximum size of ${C} for your GPU`):l<d?new Error(`Argument texture height of ${d} larger than maximum size of ${C} for your GPU`):new Error(`Argument texture height and width of ${d} larger than maximum size of ${C} for your GPU`)}setup(){this.requestTexture(),this.setupTexture(),this.defineTexture()}requestTexture(){this.texture=this.onRequestTexture()}defineTexture(){const{context:l}=this;l.activeTexture(this.contextHandle),l.bindTexture(l.TEXTURE_2D,this.texture),l.texParameteri(l.TEXTURE_2D,l.TEXTURE_WRAP_S,l.CLAMP_TO_EDGE),l.texParameteri(l.TEXTURE_2D,l.TEXTURE_WRAP_T,l.CLAMP_TO_EDGE),l.texParameteri(l.TEXTURE_2D,l.TEXTURE_MIN_FILTER,l.NEAREST),l.texParameteri(l.TEXTURE_2D,l.TEXTURE_MAG_FILTER,l.NEAREST)}setupTexture(){this.contextHandle=this.onRequestContextHandle(),this.index=this.onRequestIndex(),this.dimensionsId=this.id+"Dim",this.sizeId=this.id+"Size"}getBitRatio(l){if(Array.isArray(l[0]))return this.getBitRatio(l[0]);if(l.constructor===_)return this.getBitRatio(l.value);switch(l.constructor){case Uint8ClampedArray:case Uint8Array:return 1;case Uint16Array:return 2;case Int8Array:case Int16Array:case Float32Array:case Int32Array:default:return 4}}destroy(){this.prevArg&&this.prevArg.delete(),this.context.deleteTexture(this.texture)}};z.exports={WebGLKernelArray:b}}),Rs=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe();function b(d){return{width:d.width>0?d.width:d.videoWidth,height:d.height>0?d.height:d.videoHeight}}var l=class extends _{constructor(d,C){super(d,C);const{width:m,height:p}=b(d);this.checkSize(m,p),this.dimensions=[m,p,1],this.textureSize=[m,p],this.uploadValue=d}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName};
`}getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(d){if(d.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(d.constructor);return}const{context:C}=this;C.activeTexture(this.contextHandle),C.bindTexture(C.TEXTURE_2D,this.texture),C.pixelStorei(C.UNPACK_FLIP_Y_WEBGL,!0),C.texImage2D(C.TEXTURE_2D,0,C.RGBA,C.RGBA,C.UNSIGNED_BYTE,this.uploadValue=d),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGLKernelValueHTMLImage:l,mediaSize:b}}),xn=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueHTMLImage:_,mediaSize:b}=Rs();var l=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(d){const{width:C,height:m}=b(d);this.checkSize(C,m),this.dimensions=[C,m,1],this.textureSize=[C,m],this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(d)}};z.exports={WebGLKernelValueDynamicHTMLImage:l}}),ro=s((B,z)=>{const{WebGLKernelValueHTMLImage:M}=Rs();var _=class extends M{};z.exports={WebGLKernelValueHTMLVideo:_}}),io=s((B,z)=>{const{WebGLKernelValueDynamicHTMLImage:M}=xn();var _=class extends M{};z.exports={WebGLKernelValueDynamicHTMLVideo:_}}),bn=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe();var b=class extends _{constructor(l,d){super(l,d),this.bitRatio=4;let[C,m,p]=l.size;this.dimensions=new Int32Array([C||1,m||1,p||1]),this.textureSize=M.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength)}getStringValueHandler(){return M.linesToString([`const uploadValue_${this.name} = new Float32Array(${this.uploadArrayLength})`,`flattenTo(${this.varName}.value, uploadValue_${this.name})`])}getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(l.constructor);return}const{context:d}=this;M.flattenTo(l.value,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGLKernelValueSingleInput:b}}),ao=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueSingleInput:_}=bn();var b=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(l){let[d,C,m]=l.size;this.dimensions=new Int32Array([d||1,C||1,m||1]),this.textureSize=M.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGLKernelValueDynamicSingleInput:b}}),vn=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe();var b=class extends _{constructor(l,d){super(l,d),this.bitRatio=this.getBitRatio(l);const[C,m,p]=l.size;this.dimensions=new Int32Array([C||1,m||1,p||1]),this.textureSize=M.getMemoryOptimizedPackedTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*(4/this.bitRatio),this.checkSize(this.textureSize[0],this.textureSize[1]),this.TranserArrayType=this.getTransferArrayType(l.value),this.preUploadValue=new this.TranserArrayType(this.uploadArrayLength),this.uploadValue=new Uint8Array(this.preUploadValue.buffer)}getStringValueHandler(){return M.linesToString([`const preUploadValue_${this.name} = new ${this.TranserArrayType.name}(${this.uploadArrayLength})`,`const uploadValue_${this.name} = new Uint8Array(preUploadValue_${this.name}.buffer)`,`flattenTo(${this.varName}.value, preUploadValue_${this.name})`])}getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(value.constructor);return}const{context:d}=this;M.flattenTo(l.value,this.preUploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.UNSIGNED_BYTE,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGLKernelValueUnsignedInput:b}}),Lr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueUnsignedInput:_}=vn();var b=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(l){let[d,C,m]=l.size;this.dimensions=new Int32Array([d||1,C||1,m||1]),this.textureSize=M.getMemoryOptimizedPackedTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*(4/this.bitRatio),this.checkSize(this.textureSize[0],this.textureSize[1]);const p=this.getTransferArrayType(l.value);this.preUploadValue=new p(this.uploadArrayLength),this.uploadValue=new Uint8Array(this.preUploadValue.buffer),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGLKernelValueDynamicUnsignedInput:b}}),Os=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe(),b="Source and destination textures are the same.  Use immutable = true and manually cleanup kernel output texture memory with texture.delete()";var l=class extends _{constructor(d,C){super(d,C);const[m,p]=d.size;this.checkSize(m,p),this.dimensions=d.dimensions,this.textureSize=d.size,this.uploadValue=d.texture,this.forceUploadEachRun=!0}setup(){this.setupTexture()}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName}.texture;
`}getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(d){if(d.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(d.constructor);return}if(this.checkContext&&d.context!==this.context)throw new Error(`Value ${this.name} (${this.type}) must be from same context`);const{kernel:C,context:m}=this;if(C.pipeline)if(C.immutable)C.updateTextureArgumentRefs(this,d);else{if(C.texture&&C.texture.texture===d.texture)throw new Error(b);if(C.mappedTextures){const{mappedTextures:p}=C;for(let u=0;u<p.length;u++)if(p[u].texture===d.texture)throw new Error(b)}}m.activeTexture(this.contextHandle),m.bindTexture(m.TEXTURE_2D,this.uploadValue=d.texture),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGLKernelValueMemoryOptimizedNumberTexture:l,sameError:b}}),Fr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueMemoryOptimizedNumberTexture:_}=Os();var b=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(l){this.dimensions=l.dimensions,this.checkSize(l.size[0],l.size[1]),this.textureSize=l.size,this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGLKernelValueDynamicMemoryOptimizedNumberTexture:b}}),wn=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe(),{sameError:b}=Os();var l=class extends _{constructor(d,C){super(d,C);const[m,p]=d.size;this.checkSize(m,p);const{size:u,dimensions:g}=d;this.bitRatio=this.getBitRatio(d),this.dimensions=g,this.textureSize=u,this.uploadValue=d.texture,this.forceUploadEachRun=!0}setup(){this.setupTexture()}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName}.texture;
`}getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(d){if(d.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(d.constructor);return}if(this.checkContext&&d.context!==this.context)throw new Error(`Value ${this.name} (${this.type}) must be from same context`);const{kernel:C,context:m}=this;if(C.pipeline)if(C.immutable)C.updateTextureArgumentRefs(this,d);else{if(C.texture&&C.texture.texture===d.texture)throw new Error(b);if(C.mappedTextures){const{mappedTextures:p}=C;for(let u=0;u<p.length;u++)if(p[u].texture===d.texture)throw new Error(b)}}m.activeTexture(this.contextHandle),m.bindTexture(m.TEXTURE_2D,this.uploadValue=d.texture),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGLKernelValueNumberTexture:l}}),Gr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueNumberTexture:_}=wn();var b=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(l){this.dimensions=l.dimensions,this.checkSize(l.size[0],l.size[1]),this.textureSize=l.size,this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGLKernelValueDynamicNumberTexture:b}}),kn=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe();var b=class extends _{constructor(l,d){super(l,d),this.bitRatio=4,this.dimensions=M.getDimensions(l,!0),this.textureSize=M.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength)}getStringValueHandler(){return M.linesToString([`const uploadValue_${this.name} = new Float32Array(${this.uploadArrayLength})`,`flattenTo(${this.varName}, uploadValue_${this.name})`])}getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(l.constructor);return}const{context:d}=this;M.flattenTo(l,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGLKernelValueSingleArray:b}}),oo=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueSingleArray:_}=kn();var b=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(l){this.dimensions=M.getDimensions(l,!0),this.textureSize=M.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGLKernelValueDynamicSingleArray:b}}),Tn=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe();var b=class extends _{constructor(l,d){super(l,d),this.bitRatio=4,this.setShape(l)}setShape(l){const d=M.getDimensions(l,!0);this.textureSize=M.getMemoryOptimizedFloatTextureSize(d,this.bitRatio),this.dimensions=new Int32Array([d[1],1,1]),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength)}getStringValueHandler(){return M.linesToString([`const uploadValue_${this.name} = new Float32Array(${this.uploadArrayLength})`,`flattenTo(${this.varName}, uploadValue_${this.name})`])}getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(l.constructor);return}const{context:d}=this;M.flatten2dArrayTo(l,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGLKernelValueSingleArray1DI:b}}),lo=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueSingleArray1DI:_}=Tn();var b=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(l){this.setShape(l),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGLKernelValueDynamicSingleArray1DI:b}}),Sn=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe();var b=class extends _{constructor(l,d){super(l,d),this.bitRatio=4,this.setShape(l)}setShape(l){const d=M.getDimensions(l,!0);this.textureSize=M.getMemoryOptimizedFloatTextureSize(d,this.bitRatio),this.dimensions=new Int32Array([d[1],d[2],1]),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength)}getStringValueHandler(){return M.linesToString([`const uploadValue_${this.name} = new Float32Array(${this.uploadArrayLength})`,`flattenTo(${this.varName}, uploadValue_${this.name})`])}getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(l.constructor);return}const{context:d}=this;M.flatten3dArrayTo(l,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGLKernelValueSingleArray2DI:b}}),uo=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueSingleArray2DI:_}=Sn();var b=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(l){this.setShape(l),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGLKernelValueDynamicSingleArray2DI:b}}),_n=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe();var b=class extends _{constructor(l,d){super(l,d),this.bitRatio=4,this.setShape(l)}setShape(l){const d=M.getDimensions(l,!0);this.textureSize=M.getMemoryOptimizedFloatTextureSize(d,this.bitRatio),this.dimensions=new Int32Array([d[1],d[2],d[3]]),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength)}getStringValueHandler(){return M.linesToString([`const uploadValue_${this.name} = new Float32Array(${this.uploadArrayLength})`,`flattenTo(${this.varName}, uploadValue_${this.name})`])}getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(l.constructor);return}const{context:d}=this;M.flatten4dArrayTo(l,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGLKernelValueSingleArray3DI:b}}),co=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueSingleArray3DI:_}=_n();var b=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(l){this.setShape(l),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGLKernelValueDynamicSingleArray3DI:b}}),Ur=s((B,z)=>{const{WebGLKernelValue:M}=Pt();var _=class extends M{constructor(b,l){super(b,l),this.uploadValue=b}getSource(b){return this.origin==="constants"?`const vec2 ${this.id} = vec2(${b[0]},${b[1]});
`:`uniform vec2 ${this.id};
`}getStringValueHandler(){return this.origin==="constants"?"":`const uploadValue_${this.name} = ${this.varName};
`}updateValue(b){this.origin!=="constants"&&this.kernel.setUniform2fv(this.id,this.uploadValue=b)}};z.exports={WebGLKernelValueArray2:_}}),Vr=s((B,z)=>{const{WebGLKernelValue:M}=Pt();var _=class extends M{constructor(b,l){super(b,l),this.uploadValue=b}getSource(b){return this.origin==="constants"?`const vec3 ${this.id} = vec3(${b[0]},${b[1]},${b[2]});
`:`uniform vec3 ${this.id};
`}getStringValueHandler(){return this.origin==="constants"?"":`const uploadValue_${this.name} = ${this.varName};
`}updateValue(b){this.origin!=="constants"&&this.kernel.setUniform3fv(this.id,this.uploadValue=b)}};z.exports={WebGLKernelValueArray3:_}}),Kr=s((B,z)=>{const{WebGLKernelValue:M}=Pt();var _=class extends M{constructor(b,l){super(b,l),this.uploadValue=b}getSource(b){return this.origin==="constants"?`const vec4 ${this.id} = vec4(${b[0]},${b[1]},${b[2]},${b[3]});
`:`uniform vec4 ${this.id};
`}getStringValueHandler(){return this.origin==="constants"?"":`const uploadValue_${this.name} = ${this.varName};
`}updateValue(b){this.origin!=="constants"&&this.kernel.setUniform4fv(this.id,this.uploadValue=b)}};z.exports={WebGLKernelValueArray4:_}}),Cn=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe();var b=class extends _{constructor(l,d){super(l,d),this.bitRatio=this.getBitRatio(l),this.dimensions=M.getDimensions(l,!0),this.textureSize=M.getMemoryOptimizedPackedTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*(4/this.bitRatio),this.checkSize(this.textureSize[0],this.textureSize[1]),this.TranserArrayType=this.getTransferArrayType(l),this.preUploadValue=new this.TranserArrayType(this.uploadArrayLength),this.uploadValue=new Uint8Array(this.preUploadValue.buffer)}getStringValueHandler(){return M.linesToString([`const preUploadValue_${this.name} = new ${this.TranserArrayType.name}(${this.uploadArrayLength})`,`const uploadValue_${this.name} = new Uint8Array(preUploadValue_${this.name}.buffer)`,`flattenTo(${this.varName}, preUploadValue_${this.name})`])}getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(l.constructor);return}const{context:d}=this;M.flattenTo(l,this.preUploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.UNSIGNED_BYTE,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGLKernelValueUnsignedArray:b}}),Nr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueUnsignedArray:_}=Cn();var b=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}updateValue(l){this.dimensions=M.getDimensions(l,!0),this.textureSize=M.getMemoryOptimizedPackedTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*(4/this.bitRatio),this.checkSize(this.textureSize[0],this.textureSize[1]);const d=this.getTransferArrayType(l);this.preUploadValue=new d(this.uploadArrayLength),this.uploadValue=new Uint8Array(this.preUploadValue.buffer),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGLKernelValueDynamicUnsignedArray:b}}),Br=s((B,z)=>{const{WebGLKernelValueBoolean:M}=zr(),{WebGLKernelValueFloat:_}=Rr(),{WebGLKernelValueInteger:b}=Or(),{WebGLKernelValueHTMLImage:l}=Rs(),{WebGLKernelValueDynamicHTMLImage:d}=xn(),{WebGLKernelValueHTMLVideo:C}=ro(),{WebGLKernelValueDynamicHTMLVideo:m}=io(),{WebGLKernelValueSingleInput:p}=bn(),{WebGLKernelValueDynamicSingleInput:u}=ao(),{WebGLKernelValueUnsignedInput:g}=vn(),{WebGLKernelValueDynamicUnsignedInput:E}=Lr(),{WebGLKernelValueMemoryOptimizedNumberTexture:w}=Os(),{WebGLKernelValueDynamicMemoryOptimizedNumberTexture:y}=Fr(),{WebGLKernelValueNumberTexture:k}=wn(),{WebGLKernelValueDynamicNumberTexture:v}=Gr(),{WebGLKernelValueSingleArray:$}=kn(),{WebGLKernelValueDynamicSingleArray:P}=oo(),{WebGLKernelValueSingleArray1DI:O}=Tn(),{WebGLKernelValueDynamicSingleArray1DI:T}=lo(),{WebGLKernelValueSingleArray2DI:A}=Sn(),{WebGLKernelValueDynamicSingleArray2DI:f}=uo(),{WebGLKernelValueSingleArray3DI:F}=_n(),{WebGLKernelValueDynamicSingleArray3DI:L}=co(),{WebGLKernelValueArray2:R}=Ur(),{WebGLKernelValueArray3:V}=Vr(),{WebGLKernelValueArray4:W}=Kr(),{WebGLKernelValueUnsignedArray:N}=Cn(),{WebGLKernelValueDynamicUnsignedArray:te}=Nr(),ee={unsigned:{dynamic:{Boolean:M,Integer:b,Float:_,Array:te,"Array(2)":R,"Array(3)":V,"Array(4)":W,"Array1D(2)":!1,"Array1D(3)":!1,"Array1D(4)":!1,"Array2D(2)":!1,"Array2D(3)":!1,"Array2D(4)":!1,"Array3D(2)":!1,"Array3D(3)":!1,"Array3D(4)":!1,Input:E,NumberTexture:v,"ArrayTexture(1)":v,"ArrayTexture(2)":v,"ArrayTexture(3)":v,"ArrayTexture(4)":v,MemoryOptimizedNumberTexture:y,HTMLCanvas:d,OffscreenCanvas:d,HTMLImage:d,ImageBitmap:d,ImageData:d,HTMLImageArray:!1,HTMLVideo:m},static:{Boolean:M,Float:_,Integer:b,Array:N,"Array(2)":R,"Array(3)":V,"Array(4)":W,"Array1D(2)":!1,"Array1D(3)":!1,"Array1D(4)":!1,"Array2D(2)":!1,"Array2D(3)":!1,"Array2D(4)":!1,"Array3D(2)":!1,"Array3D(3)":!1,"Array3D(4)":!1,Input:g,NumberTexture:k,"ArrayTexture(1)":k,"ArrayTexture(2)":k,"ArrayTexture(3)":k,"ArrayTexture(4)":k,MemoryOptimizedNumberTexture:w,HTMLCanvas:l,OffscreenCanvas:l,HTMLImage:l,ImageBitmap:l,ImageData:l,HTMLImageArray:!1,HTMLVideo:C}},single:{dynamic:{Boolean:M,Integer:b,Float:_,Array:P,"Array(2)":R,"Array(3)":V,"Array(4)":W,"Array1D(2)":T,"Array1D(3)":T,"Array1D(4)":T,"Array2D(2)":f,"Array2D(3)":f,"Array2D(4)":f,"Array3D(2)":L,"Array3D(3)":L,"Array3D(4)":L,Input:u,NumberTexture:v,"ArrayTexture(1)":v,"ArrayTexture(2)":v,"ArrayTexture(3)":v,"ArrayTexture(4)":v,MemoryOptimizedNumberTexture:y,HTMLCanvas:d,OffscreenCanvas:d,HTMLImage:d,ImageBitmap:d,ImageData:d,HTMLImageArray:!1,HTMLVideo:m},static:{Boolean:M,Float:_,Integer:b,Array:$,"Array(2)":R,"Array(3)":V,"Array(4)":W,"Array1D(2)":O,"Array1D(3)":O,"Array1D(4)":O,"Array2D(2)":A,"Array2D(3)":A,"Array2D(4)":A,"Array3D(2)":F,"Array3D(3)":F,"Array3D(4)":F,Input:p,NumberTexture:k,"ArrayTexture(1)":k,"ArrayTexture(2)":k,"ArrayTexture(3)":k,"ArrayTexture(4)":k,MemoryOptimizedNumberTexture:w,HTMLCanvas:l,OffscreenCanvas:l,HTMLImage:l,ImageBitmap:l,ImageData:l,HTMLImageArray:!1,HTMLVideo:C}}};function X(ie,se,J,ae){if(!ie)throw new Error("type missing");if(!se)throw new Error("dynamic missing");if(!J)throw new Error("precision missing");ae.type&&(ie=ae.type);const he=ee[J][se];if(he[ie]===!1)return null;if(he[ie]===void 0)throw new Error(`Could not find a KernelValue for ${ie}`);return he[ie]}z.exports={lookupKernelValueType:X,kernelValueMaps:ee}}),Ls=s((B,z)=>{const{GLKernel:M}=$r(),{FunctionBuilder:_}=U(),{WebGLFunctionNode:b}=yn(),{utils:l}=h(),d=Dr(),{fragmentShader:C}=eo(),{vertexShader:m}=to(),{glKernelString:p}=Pr(),{lookupKernelValueType:u}=Br();let g=null,E=null,w=null,y=null,k=null;const v=[d],$=[],P={};var O=class extends M{static get isSupported(){return g!==null||(this.setupFeatureChecks(),g=this.isContextMatch(w)),g}static setupFeatureChecks(){typeof document<"u"?E=document.createElement("canvas"):typeof OffscreenCanvas<"u"&&(E=new OffscreenCanvas(0,0)),E&&(w=E.getContext("webgl"),!w&&!(E instanceof OffscreenCanvas)&&(w=E.getContext("experimental-webgl")),!(!w||!w.getExtension)&&(y={OES_texture_float:w.getExtension("OES_texture_float"),OES_texture_float_linear:w.getExtension("OES_texture_float_linear"),OES_element_index_uint:w.getExtension("OES_element_index_uint"),WEBGL_draw_buffers:w.getExtension("WEBGL_draw_buffers")},k=this.getFeatures()))}static isContextMatch(T){return typeof WebGLRenderingContext<"u"?T instanceof WebGLRenderingContext:!1}static getIsTextureFloat(){return!!y.OES_texture_float}static getIsDrawBuffers(){return!!y.WEBGL_draw_buffers}static getChannelCount(){return y.WEBGL_draw_buffers?w.getParameter(y.WEBGL_draw_buffers.MAX_DRAW_BUFFERS_WEBGL):1}static getMaxTextureSize(){return w.getParameter(w.MAX_TEXTURE_SIZE)}static lookupKernelValueType(T,A,f,F){return u(T,A,f,F)}static get testCanvas(){return E}static get testContext(){return w}static get features(){return k}static get fragmentShader(){return C}static get vertexShader(){return m}constructor(T,A){super(T,A),this.program=null,this.pipeline=A.pipeline,this.endianness=l.systemEndianness(),this.extensions={},this.argumentTextureCount=0,this.constantTextureCount=0,this.fragShader=null,this.vertShader=null,this.drawBuffersMap=null,this.maxTexSize=null,this.onRequestSwitchKernel=null,this.texture=null,this.mappedTextures=null,this.mergeSettings(T.settings||A),this.threadDim=null,this.framebuffer=null,this.buffer=null,this.textureCache=[],this.programUniformLocationCache={},this.uniform1fCache={},this.uniform1iCache={},this.uniform2fCache={},this.uniform2fvCache={},this.uniform2ivCache={},this.uniform3fvCache={},this.uniform3ivCache={},this.uniform4fvCache={},this.uniform4ivCache={}}initCanvas(){if(typeof document<"u"){const T=document.createElement("canvas");return T.width=2,T.height=2,T}else if(typeof OffscreenCanvas<"u")return new OffscreenCanvas(0,0)}initContext(){const T={alpha:!1,depth:!1,antialias:!1};return this.canvas.getContext("webgl",T)||this.canvas.getContext("experimental-webgl",T)}initPlugins(T){const A=[],{source:f}=this;if(typeof f=="string")for(let F=0;F<v.length;F++){const L=v[F];f.match(L.functionMatch)&&A.push(L)}else if(typeof f=="object"&&T.pluginNames)for(let F=0;F<v.length;F++){const L=v[F];T.pluginNames.some(R=>R===L.name)&&A.push(L)}return A}initExtensions(){this.extensions={OES_texture_float:this.context.getExtension("OES_texture_float"),OES_texture_float_linear:this.context.getExtension("OES_texture_float_linear"),OES_element_index_uint:this.context.getExtension("OES_element_index_uint"),WEBGL_draw_buffers:this.context.getExtension("WEBGL_draw_buffers"),WEBGL_color_buffer_float:this.context.getExtension("WEBGL_color_buffer_float")}}validateSettings(T){if(!this.validate){this.texSize=l.getKernelTextureSize({optimizeFloatMemory:this.optimizeFloatMemory,precision:this.precision},this.output);return}const{features:A}=this.constructor;if(this.optimizeFloatMemory===!0&&!A.isTextureFloat)throw new Error("Float textures are not supported");if(this.precision==="single"&&!A.isFloatRead)throw new Error("Single precision not supported");if(!this.graphical&&this.precision===null&&(this.precision=A.isTextureFloat&&A.isFloatRead?"single":"unsigned"),this.subKernels&&this.subKernels.length>0&&!this.extensions.WEBGL_draw_buffers)throw new Error("could not instantiate draw buffers extension");if(this.fixIntegerDivisionAccuracy===null?this.fixIntegerDivisionAccuracy=!A.isIntegerDivisionAccurate:this.fixIntegerDivisionAccuracy&&A.isIntegerDivisionAccurate&&(this.fixIntegerDivisionAccuracy=!1),this.checkOutput(),!this.output||this.output.length===0){if(T.length!==1)throw new Error("Auto output only supported for kernels with only one input");const f=l.getVariableType(T[0],this.strictIntegers);switch(f){case"Array":this.output=l.getDimensions(f);break;case"NumberTexture":case"MemoryOptimizedNumberTexture":case"ArrayTexture(1)":case"ArrayTexture(2)":case"ArrayTexture(3)":case"ArrayTexture(4)":this.output=T[0].output;break;default:throw new Error("Auto output not supported for input type: "+f)}}if(this.graphical){if(this.output.length!==2)throw new Error("Output must have 2 dimensions on graphical mode");this.precision==="precision"&&(this.precision="unsigned",console.warn("Cannot use graphical mode and single precision at the same time")),this.texSize=l.clone(this.output);return}else this.precision===null&&A.isTextureFloat&&(this.precision="single");this.texSize=l.getKernelTextureSize({optimizeFloatMemory:this.optimizeFloatMemory,precision:this.precision},this.output),this.checkTextureSize()}updateMaxTexSize(){const{texSize:T,canvas:A}=this;if(this.maxTexSize===null){let f=$.indexOf(A);f===-1&&(f=$.length,$.push(A),P[f]=[T[0],T[1]]),this.maxTexSize=P[f]}this.maxTexSize[0]<T[0]&&(this.maxTexSize[0]=T[0]),this.maxTexSize[1]<T[1]&&(this.maxTexSize[1]=T[1])}setupArguments(T){this.kernelArguments=[],this.argumentTextureCount=0;const A=this.argumentTypes===null;if(A&&(this.argumentTypes=[]),this.argumentSizes=[],this.argumentBitRatios=[],T.length<this.argumentNames.length)throw new Error("not enough arguments for kernel");if(T.length>this.argumentNames.length)throw new Error("too many arguments for kernel");const{context:f}=this;let F=0;const L=()=>this.createTexture(),R=()=>this.constantTextureCount+F++,V=N=>{this.switchKernels({type:"argumentMismatch",needed:N})},W=()=>f.TEXTURE0+this.constantTextureCount+this.argumentTextureCount++;for(let N=0;N<T.length;N++){const te=T[N],ee=this.argumentNames[N];let X;A?(X=l.getVariableType(te,this.strictIntegers),this.argumentTypes.push(X)):X=this.argumentTypes[N];const ie=this.constructor.lookupKernelValueType(X,this.dynamicArguments?"dynamic":"static",this.precision,T[N]);if(ie===null)return this.requestFallback(T);const se=new ie(te,{name:ee,type:X,tactic:this.tactic,origin:"user",context:f,checkContext:this.checkContext,kernel:this,strictIntegers:this.strictIntegers,onRequestTexture:L,onRequestIndex:R,onUpdateValueMismatch:V,onRequestContextHandle:W});this.kernelArguments.push(se),se.setup(),this.argumentSizes.push(se.textureSize),this.argumentBitRatios[N]=se.bitRatio}}createTexture(){const T=this.context.createTexture();return this.textureCache.push(T),T}deleteTexture(T){const A=this.textureCache.indexOf(T);A!==-1&&this.textureCache.splice(A,1),this.context&&this.context.deleteTexture(T)}setupConstants(T){const{context:A}=this;this.kernelConstants=[],this.forceUploadKernelConstants=[];let f=this.constantTypes===null;f&&(this.constantTypes={}),this.constantBitRatios={};let F=0;for(const L in this.constants){const R=this.constants[L];let V;f?(V=l.getVariableType(R,this.strictIntegers),this.constantTypes[L]=V):V=this.constantTypes[L];const W=this.constructor.lookupKernelValueType(V,"static",this.precision,R);if(W===null)return this.requestFallback(T);const N=new W(R,{name:L,type:V,tactic:this.tactic,origin:"constants",context:this.context,checkContext:this.checkContext,kernel:this,strictIntegers:this.strictIntegers,onRequestTexture:()=>this.createTexture(),onRequestIndex:()=>F++,onRequestContextHandle:()=>A.TEXTURE0+this.constantTextureCount++});this.constantBitRatios[L]=N.bitRatio,this.kernelConstants.push(N),N.setup(),N.forceUploadEachRun&&this.forceUploadKernelConstants.push(N)}}build(){if(this.built||(this.initExtensions(),this.validateSettings(arguments),this.setupConstants(arguments),this.fallbackRequested)||(this.setupArguments(arguments),this.fallbackRequested))return;this.updateMaxTexSize(),this.translateSource();const T=this.pickRenderStrategy(arguments);if(T)return T;const{texSize:A,context:f,canvas:F}=this;f.enable(f.SCISSOR_TEST),this.pipeline&&this.precision==="single"?(f.viewport(0,0,this.maxTexSize[0],this.maxTexSize[1]),F.width=this.maxTexSize[0],F.height=this.maxTexSize[1]):(f.viewport(0,0,this.maxTexSize[0],this.maxTexSize[1]),F.width=this.maxTexSize[0],F.height=this.maxTexSize[1]);const L=this.threadDim=Array.from(this.output);for(;L.length<3;)L.push(1);const R=this.getVertexShader(arguments),V=f.createShader(f.VERTEX_SHADER);f.shaderSource(V,R),f.compileShader(V),this.vertShader=V;const W=this.getFragmentShader(arguments),N=f.createShader(f.FRAGMENT_SHADER);if(f.shaderSource(N,W),f.compileShader(N),this.fragShader=N,this.debug&&(console.log("GLSL Shader Output:"),console.log(W)),!f.getShaderParameter(V,f.COMPILE_STATUS))throw new Error("Error compiling vertex shader: "+f.getShaderInfoLog(V));if(!f.getShaderParameter(N,f.COMPILE_STATUS))throw new Error("Error compiling fragment shader: "+f.getShaderInfoLog(N));const te=this.program=f.createProgram();f.attachShader(te,V),f.attachShader(te,N),f.linkProgram(te),this.framebuffer=f.createFramebuffer(),this.framebuffer.width=A[0],this.framebuffer.height=A[1],this.rawValueFramebuffers={};const ee=new Float32Array([-1,-1,1,-1,-1,1,1,1]),X=new Float32Array([0,0,1,0,0,1,1,1]),ie=ee.byteLength;let se=this.buffer;se?f.bindBuffer(f.ARRAY_BUFFER,se):(se=this.buffer=f.createBuffer(),f.bindBuffer(f.ARRAY_BUFFER,se),f.bufferData(f.ARRAY_BUFFER,ee.byteLength+X.byteLength,f.STATIC_DRAW)),f.bufferSubData(f.ARRAY_BUFFER,0,ee),f.bufferSubData(f.ARRAY_BUFFER,ie,X);const J=f.getAttribLocation(this.program,"aPos");J!==-1&&(f.enableVertexAttribArray(J),f.vertexAttribPointer(J,2,f.FLOAT,!1,0,0));const ae=f.getAttribLocation(this.program,"aTexCoord");ae!==-1&&(f.enableVertexAttribArray(ae),f.vertexAttribPointer(ae,2,f.FLOAT,!1,0,ie)),f.bindFramebuffer(f.FRAMEBUFFER,this.framebuffer);let he=0;f.useProgram(this.program);for(let Ae in this.constants)this.kernelConstants[he++].updateValue(this.constants[Ae]);this._setupOutputTexture(),this.subKernels!==null&&this.subKernels.length>0&&(this._mappedTextureSwitched={},this._setupSubOutputTextures()),this.buildSignature(arguments),this.built=!0}translateSource(){const T=_.fromKernel(this,b,{fixIntegerDivisionAccuracy:this.fixIntegerDivisionAccuracy});this.translatedSource=T.getPrototypeString("kernel"),this.setupReturnTypes(T)}setupReturnTypes(T){if(!this.graphical&&!this.returnType&&(this.returnType=T.getKernelResultType()),this.subKernels&&this.subKernels.length>0)for(let A=0;A<this.subKernels.length;A++){const f=this.subKernels[A];f.returnType||(f.returnType=T.getSubKernelResultType(A))}}run(){const{kernelArguments:T,texSize:A,forceUploadKernelConstants:f,context:F}=this;F.useProgram(this.program),F.scissor(0,0,A[0],A[1]),this.dynamicOutput&&(this.setUniform3iv("uOutputDim",new Int32Array(this.threadDim)),this.setUniform2iv("uTexSize",A)),this.setUniform2f("ratio",A[0]/this.maxTexSize[0],A[1]/this.maxTexSize[1]);for(let L=0;L<f.length;L++){const R=f[L];if(R.updateValue(this.constants[R.name]),this.switchingKernels)return}for(let L=0;L<T.length;L++)if(T[L].updateValue(arguments[L]),this.switchingKernels)return;if(this.plugins)for(let L=0;L<this.plugins.length;L++){const R=this.plugins[L];R.onBeforeRun&&R.onBeforeRun(this)}if(this.graphical){if(this.pipeline)return F.bindRenderbuffer(F.RENDERBUFFER,null),F.bindFramebuffer(F.FRAMEBUFFER,this.framebuffer),this.immutable&&this._replaceOutputTexture(),F.drawArrays(F.TRIANGLE_STRIP,0,4),this.immutable?this.texture.clone():this.texture;F.bindRenderbuffer(F.RENDERBUFFER,null),F.bindFramebuffer(F.FRAMEBUFFER,null),F.drawArrays(F.TRIANGLE_STRIP,0,4);return}F.bindFramebuffer(F.FRAMEBUFFER,this.framebuffer),this.immutable&&this._replaceOutputTexture(),this.subKernels!==null&&(this.immutable&&this._replaceSubOutputTextures(),this.drawBuffers()),F.drawArrays(F.TRIANGLE_STRIP,0,4)}drawBuffers(){this.extensions.WEBGL_draw_buffers.drawBuffersWEBGL(this.drawBuffersMap)}getInternalFormat(){return this.context.RGBA}getTextureFormat(){const{context:T}=this;switch(this.getInternalFormat()){case T.RGBA:return T.RGBA;default:throw new Error("Unknown internal format")}}_replaceOutputTexture(){if(this.texture.beforeMutate()||this._textureSwitched){const T=this.context;T.framebufferTexture2D(T.FRAMEBUFFER,T.COLOR_ATTACHMENT0,T.TEXTURE_2D,this.texture.texture,0),this._textureSwitched=!1}}_setupOutputTexture(){const T=this.context,A=this.texSize;if(this.texture){T.framebufferTexture2D(T.FRAMEBUFFER,T.COLOR_ATTACHMENT0,T.TEXTURE_2D,this.texture.texture,0);return}const f=this.createTexture();T.activeTexture(T.TEXTURE0+this.constantTextureCount+this.argumentTextureCount),T.bindTexture(T.TEXTURE_2D,f),T.texParameteri(T.TEXTURE_2D,T.TEXTURE_WRAP_S,T.CLAMP_TO_EDGE),T.texParameteri(T.TEXTURE_2D,T.TEXTURE_WRAP_T,T.CLAMP_TO_EDGE),T.texParameteri(T.TEXTURE_2D,T.TEXTURE_MIN_FILTER,T.NEAREST),T.texParameteri(T.TEXTURE_2D,T.TEXTURE_MAG_FILTER,T.NEAREST);const F=this.getInternalFormat();this.precision==="single"?T.texImage2D(T.TEXTURE_2D,0,F,A[0],A[1],0,T.RGBA,T.FLOAT,null):T.texImage2D(T.TEXTURE_2D,0,F,A[0],A[1],0,F,T.UNSIGNED_BYTE,null),T.framebufferTexture2D(T.FRAMEBUFFER,T.COLOR_ATTACHMENT0,T.TEXTURE_2D,f,0),this.texture=new this.TextureConstructor({texture:f,size:A,dimensions:this.threadDim,output:this.output,context:this.context,internalFormat:this.getInternalFormat(),textureFormat:this.getTextureFormat(),kernel:this})}_replaceSubOutputTextures(){const T=this.context;for(let A=0;A<this.mappedTextures.length;A++){const f=this.mappedTextures[A];(f.beforeMutate()||this._mappedTextureSwitched[A])&&(T.framebufferTexture2D(T.FRAMEBUFFER,T.COLOR_ATTACHMENT0+A+1,T.TEXTURE_2D,f.texture,0),this._mappedTextureSwitched[A]=!1)}}_setupSubOutputTextures(){const T=this.context;if(this.mappedTextures){for(let f=0;f<this.subKernels.length;f++)T.framebufferTexture2D(T.FRAMEBUFFER,T.COLOR_ATTACHMENT0+f+1,T.TEXTURE_2D,this.mappedTextures[f].texture,0);return}const A=this.texSize;this.drawBuffersMap=[T.COLOR_ATTACHMENT0],this.mappedTextures=[];for(let f=0;f<this.subKernels.length;f++){const F=this.createTexture();this.drawBuffersMap.push(T.COLOR_ATTACHMENT0+f+1),T.activeTexture(T.TEXTURE0+this.constantTextureCount+this.argumentTextureCount+f),T.bindTexture(T.TEXTURE_2D,F),T.texParameteri(T.TEXTURE_2D,T.TEXTURE_WRAP_S,T.CLAMP_TO_EDGE),T.texParameteri(T.TEXTURE_2D,T.TEXTURE_WRAP_T,T.CLAMP_TO_EDGE),T.texParameteri(T.TEXTURE_2D,T.TEXTURE_MIN_FILTER,T.NEAREST),T.texParameteri(T.TEXTURE_2D,T.TEXTURE_MAG_FILTER,T.NEAREST),this.precision==="single"?T.texImage2D(T.TEXTURE_2D,0,T.RGBA,A[0],A[1],0,T.RGBA,T.FLOAT,null):T.texImage2D(T.TEXTURE_2D,0,T.RGBA,A[0],A[1],0,T.RGBA,T.UNSIGNED_BYTE,null),T.framebufferTexture2D(T.FRAMEBUFFER,T.COLOR_ATTACHMENT0+f+1,T.TEXTURE_2D,F,0),this.mappedTextures.push(new this.TextureConstructor({texture:F,size:A,dimensions:this.threadDim,output:this.output,context:this.context,internalFormat:this.getInternalFormat(),textureFormat:this.getTextureFormat(),kernel:this}))}}setUniform1f(T,A){if(this.uniform1fCache.hasOwnProperty(T)&&A===this.uniform1fCache[T])return;this.uniform1fCache[T]=A;const f=this.getUniformLocation(T);this.context.uniform1f(f,A)}setUniform1i(T,A){if(this.uniform1iCache.hasOwnProperty(T)&&A===this.uniform1iCache[T])return;this.uniform1iCache[T]=A;const f=this.getUniformLocation(T);this.context.uniform1i(f,A)}setUniform2f(T,A,f){if(this.uniform2fCache.hasOwnProperty(T)){const L=this.uniform2fCache[T];if(A===L[0]&&f===L[1])return}this.uniform2fCache[T]=[A,f];const F=this.getUniformLocation(T);this.context.uniform2f(F,A,f)}setUniform2fv(T,A){if(this.uniform2fvCache.hasOwnProperty(T)){const F=this.uniform2fvCache[T];if(A[0]===F[0]&&A[1]===F[1])return}this.uniform2fvCache[T]=A;const f=this.getUniformLocation(T);this.context.uniform2fv(f,A)}setUniform2iv(T,A){if(this.uniform2ivCache.hasOwnProperty(T)){const F=this.uniform2ivCache[T];if(A[0]===F[0]&&A[1]===F[1])return}this.uniform2ivCache[T]=A;const f=this.getUniformLocation(T);this.context.uniform2iv(f,A)}setUniform3fv(T,A){if(this.uniform3fvCache.hasOwnProperty(T)){const F=this.uniform3fvCache[T];if(A[0]===F[0]&&A[1]===F[1]&&A[2]===F[2])return}this.uniform3fvCache[T]=A;const f=this.getUniformLocation(T);this.context.uniform3fv(f,A)}setUniform3iv(T,A){if(this.uniform3ivCache.hasOwnProperty(T)){const F=this.uniform3ivCache[T];if(A[0]===F[0]&&A[1]===F[1]&&A[2]===F[2])return}this.uniform3ivCache[T]=A;const f=this.getUniformLocation(T);this.context.uniform3iv(f,A)}setUniform4fv(T,A){if(this.uniform4fvCache.hasOwnProperty(T)){const F=this.uniform4fvCache[T];if(A[0]===F[0]&&A[1]===F[1]&&A[2]===F[2]&&A[3]===F[3])return}this.uniform4fvCache[T]=A;const f=this.getUniformLocation(T);this.context.uniform4fv(f,A)}setUniform4iv(T,A){if(this.uniform4ivCache.hasOwnProperty(T)){const F=this.uniform4ivCache[T];if(A[0]===F[0]&&A[1]===F[1]&&A[2]===F[2]&&A[3]===F[3])return}this.uniform4ivCache[T]=A;const f=this.getUniformLocation(T);this.context.uniform4iv(f,A)}getUniformLocation(T){return this.programUniformLocationCache.hasOwnProperty(T)?this.programUniformLocationCache[T]:this.programUniformLocationCache[T]=this.context.getUniformLocation(this.program,T)}_getFragShaderArtifactMap(T){return{HEADER:this._getHeaderString(),LOOP_MAX:this._getLoopMaxString(),PLUGINS:this._getPluginsString(),CONSTANTS:this._getConstantsString(),DECODE32_ENDIANNESS:this._getDecode32EndiannessString(),ENCODE32_ENDIANNESS:this._getEncode32EndiannessString(),DIVIDE_WITH_INTEGER_CHECK:this._getDivideWithIntegerCheckString(),INJECTED_NATIVE:this._getInjectedNative(),MAIN_CONSTANTS:this._getMainConstantsString(),MAIN_ARGUMENTS:this._getMainArgumentsString(T),KERNEL:this.getKernelString(),MAIN_RESULT:this.getMainResultString(),FLOAT_TACTIC_DECLARATION:this.getFloatTacticDeclaration(),INT_TACTIC_DECLARATION:this.getIntTacticDeclaration(),SAMPLER_2D_TACTIC_DECLARATION:this.getSampler2DTacticDeclaration(),SAMPLER_2D_ARRAY_TACTIC_DECLARATION:this.getSampler2DArrayTacticDeclaration()}}_getVertShaderArtifactMap(T){return{FLOAT_TACTIC_DECLARATION:this.getFloatTacticDeclaration(),INT_TACTIC_DECLARATION:this.getIntTacticDeclaration(),SAMPLER_2D_TACTIC_DECLARATION:this.getSampler2DTacticDeclaration(),SAMPLER_2D_ARRAY_TACTIC_DECLARATION:this.getSampler2DArrayTacticDeclaration()}}_getHeaderString(){return this.subKernels!==null?`#extension GL_EXT_draw_buffers : require
`:""}_getLoopMaxString(){return this.loopMaxIterations?` ${parseInt(this.loopMaxIterations)};
`:` 1000;
`}_getPluginsString(){return this.plugins?this.plugins.map(T=>T.source&&this.source.match(T.functionMatch)?T.source:"").join(`
`):`
`}_getConstantsString(){const T=[],{threadDim:A,texSize:f}=this;return this.dynamicOutput?T.push("uniform ivec3 uOutputDim","uniform ivec2 uTexSize"):T.push(`ivec3 uOutputDim = ivec3(${A[0]}, ${A[1]}, ${A[2]})`,`ivec2 uTexSize = ivec2(${f[0]}, ${f[1]})`),l.linesToString(T)}_getTextureCoordinate(){const T=this.subKernels;return T===null||T.length<1?`varying vec2 vTexCoord;
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
}`:""}_getMainArgumentsString(T){const A=[],{argumentNames:f}=this;for(let F=0;F<f.length;F++)A.push(this.kernelArguments[F].getSource(T[F]));return A.join("")}_getInjectedNative(){return this.injectedNative||""}_getMainConstantsString(){const T=[],{constants:A}=this;if(A){let f=0;for(const F in A)this.constants.hasOwnProperty(F)&&T.push(this.kernelConstants[f++].getSource(this.constants[F]))}return T.join("")}getRawValueFramebuffer(T,A){if(this.rawValueFramebuffers[T]||(this.rawValueFramebuffers[T]={}),!this.rawValueFramebuffers[T][A]){const f=this.context.createFramebuffer();f.width=T,f.height=A,this.rawValueFramebuffers[T][A]=f}return this.rawValueFramebuffers[T][A]}getKernelResultDeclaration(){switch(this.returnType){case"Array(2)":return"vec2 kernelResult";case"Array(3)":return"vec3 kernelResult";case"Array(4)":return"vec4 kernelResult";case"LiteralInteger":case"Float":case"Number":case"Integer":return"float kernelResult";default:if(this.graphical)return"float kernelResult";throw new Error(`unrecognized output type "${this.returnType}"`)}}getKernelString(){const T=[this.getKernelResultDeclaration()],{subKernels:A}=this;if(A!==null)switch(this.returnType){case"Number":case"Float":case"Integer":for(let f=0;f<A.length;f++){const F=A[f];T.push(F.returnType==="Integer"?`int subKernelResult_${F.name} = 0`:`float subKernelResult_${F.name} = 0.0`)}break;case"Array(2)":for(let f=0;f<A.length;f++)T.push(`vec2 subKernelResult_${A[f].name}`);break;case"Array(3)":for(let f=0;f<A.length;f++)T.push(`vec3 subKernelResult_${A[f].name}`);break;case"Array(4)":for(let f=0;f<A.length;f++)T.push(`vec4 subKernelResult_${A[f].name}`);break}return l.linesToString(T)+this.translatedSource}getMainResultGraphical(){return l.linesToString(["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  gl_FragColor = actualColor"])}getMainResultPackedPixels(){switch(this.returnType){case"LiteralInteger":case"Number":case"Integer":case"Float":return this.getMainResultKernelPackedPixels()+this.getMainResultSubKernelPackedPixels();default:throw new Error(`packed output only usable with Numbers, "${this.returnType}" specified`)}}getMainResultKernelPackedPixels(){return l.linesToString(["  threadId = indexTo3D(index, uOutputDim)","  kernel()",`  gl_FragData[0] = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(kernelResult)`])}getMainResultSubKernelPackedPixels(){const T=[];if(!this.subKernels)return"";for(let A=0;A<this.subKernels.length;A++)this.subKernels[A].returnType==="Integer"?T.push(`  gl_FragData[${A+1}] = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(float(subKernelResult_${this.subKernels[A].name}))`):T.push(`  gl_FragData[${A+1}] = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(subKernelResult_${this.subKernels[A].name})`);return l.linesToString(T)}getMainResultMemoryOptimizedFloats(){const T=["  index *= 4"];switch(this.returnType){case"Number":case"Integer":case"Float":const A=["r","g","b","a"];for(let f=0;f<A.length;f++){const F=A[f];this.getMainResultKernelMemoryOptimizedFloats(T,F),this.getMainResultSubKernelMemoryOptimizedFloats(T,F),f+1<A.length&&T.push("  index += 1")}break;default:throw new Error(`optimized output only usable with Numbers, ${this.returnType} specified`)}return l.linesToString(T)}getMainResultKernelMemoryOptimizedFloats(T,A){T.push("  threadId = indexTo3D(index, uOutputDim)","  kernel()",`  gl_FragData[0].${A} = kernelResult`)}getMainResultSubKernelMemoryOptimizedFloats(T,A){if(!this.subKernels)return T;for(let f=0;f<this.subKernels.length;f++)this.subKernels[f].returnType==="Integer"?T.push(`  gl_FragData[${f+1}].${A} = float(subKernelResult_${this.subKernels[f].name})`):T.push(`  gl_FragData[${f+1}].${A} = subKernelResult_${this.subKernels[f].name}`)}getMainResultKernelNumberTexture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  gl_FragData[0][0] = kernelResult"]}getMainResultSubKernelNumberTexture(){const T=[];if(!this.subKernels)return T;for(let A=0;A<this.subKernels.length;++A){const f=this.subKernels[A];f.returnType==="Integer"?T.push(`  gl_FragData[${A+1}][0] = float(subKernelResult_${f.name})`):T.push(`  gl_FragData[${A+1}][0] = subKernelResult_${f.name}`)}return T}getMainResultKernelArray2Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  gl_FragData[0][0] = kernelResult[0]","  gl_FragData[0][1] = kernelResult[1]"]}getMainResultSubKernelArray2Texture(){const T=[];if(!this.subKernels)return T;for(let A=0;A<this.subKernels.length;++A)T.push(`  gl_FragData[${A+1}][0] = subKernelResult_${this.subKernels[A].name}[0]`,`  gl_FragData[${A+1}][1] = subKernelResult_${this.subKernels[A].name}[1]`);return T}getMainResultKernelArray3Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  gl_FragData[0][0] = kernelResult[0]","  gl_FragData[0][1] = kernelResult[1]","  gl_FragData[0][2] = kernelResult[2]"]}getMainResultSubKernelArray3Texture(){const T=[];if(!this.subKernels)return T;for(let A=0;A<this.subKernels.length;++A)T.push(`  gl_FragData[${A+1}][0] = subKernelResult_${this.subKernels[A].name}[0]`,`  gl_FragData[${A+1}][1] = subKernelResult_${this.subKernels[A].name}[1]`,`  gl_FragData[${A+1}][2] = subKernelResult_${this.subKernels[A].name}[2]`);return T}getMainResultKernelArray4Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  gl_FragData[0] = kernelResult"]}getMainResultSubKernelArray4Texture(){const T=[];if(!this.subKernels)return T;switch(this.returnType){case"Number":case"Float":case"Integer":for(let A=0;A<this.subKernels.length;++A)this.subKernels[A].returnType==="Integer"?T.push(`  gl_FragData[${A+1}] = float(subKernelResult_${this.subKernels[A].name})`):T.push(`  gl_FragData[${A+1}] = subKernelResult_${this.subKernels[A].name}`);break;case"Array(2)":for(let A=0;A<this.subKernels.length;++A)T.push(`  gl_FragData[${A+1}][0] = subKernelResult_${this.subKernels[A].name}[0]`,`  gl_FragData[${A+1}][1] = subKernelResult_${this.subKernels[A].name}[1]`);break;case"Array(3)":for(let A=0;A<this.subKernels.length;++A)T.push(`  gl_FragData[${A+1}][0] = subKernelResult_${this.subKernels[A].name}[0]`,`  gl_FragData[${A+1}][1] = subKernelResult_${this.subKernels[A].name}[1]`,`  gl_FragData[${A+1}][2] = subKernelResult_${this.subKernels[A].name}[2]`);break;case"Array(4)":for(let A=0;A<this.subKernels.length;++A)T.push(`  gl_FragData[${A+1}][0] = subKernelResult_${this.subKernels[A].name}[0]`,`  gl_FragData[${A+1}][1] = subKernelResult_${this.subKernels[A].name}[1]`,`  gl_FragData[${A+1}][2] = subKernelResult_${this.subKernels[A].name}[2]`,`  gl_FragData[${A+1}][3] = subKernelResult_${this.subKernels[A].name}[3]`);break}return T}replaceArtifacts(T,A){return T.replace(/[ ]*__([A-Z]+[0-9]*([_]?[A-Z]*[0-9]?)*)__;\n/g,(f,F)=>{if(A.hasOwnProperty(F))return A[F];throw`unhandled artifact ${F}`})}getFragmentShader(T){return this.compiledFragmentShader!==null?this.compiledFragmentShader:this.compiledFragmentShader=this.replaceArtifacts(this.constructor.fragmentShader,this._getFragShaderArtifactMap(T))}getVertexShader(T){return this.compiledVertexShader!==null?this.compiledVertexShader:this.compiledVertexShader=this.replaceArtifacts(this.constructor.vertexShader,this._getVertShaderArtifactMap(T))}toString(){const T=l.linesToString(["const gl = context"]);return p(this.constructor,arguments,this,T)}destroy(T){if(!this.context)return;this.buffer&&this.context.deleteBuffer(this.buffer),this.framebuffer&&this.context.deleteFramebuffer(this.framebuffer);for(const f in this.rawValueFramebuffers){for(const F in this.rawValueFramebuffers[f])this.context.deleteFramebuffer(this.rawValueFramebuffers[f][F]),delete this.rawValueFramebuffers[f][F];delete this.rawValueFramebuffers[f]}if(this.vertShader&&this.context.deleteShader(this.vertShader),this.fragShader&&this.context.deleteShader(this.fragShader),this.program&&this.context.deleteProgram(this.program),this.texture){this.texture.delete();const f=this.textureCache.indexOf(this.texture.texture);f>-1&&this.textureCache.splice(f,1),this.texture=null}if(this.mappedTextures&&this.mappedTextures.length){for(let f=0;f<this.mappedTextures.length;f++){const F=this.mappedTextures[f];F.delete();const L=this.textureCache.indexOf(F.texture);L>-1&&this.textureCache.splice(L,1)}this.mappedTextures=null}if(this.kernelArguments)for(let f=0;f<this.kernelArguments.length;f++)this.kernelArguments[f].destroy();if(this.kernelConstants)for(let f=0;f<this.kernelConstants.length;f++)this.kernelConstants[f].destroy();for(;this.textureCache.length>0;){const f=this.textureCache.pop();this.context.deleteTexture(f)}if(T){const f=$.indexOf(this.canvas);f>=0&&($[f]=null,P[f]=null)}if(this.destroyExtensions(),delete this.context,delete this.canvas,!this.gpu)return;const A=this.gpu.kernels.indexOf(this);A!==-1&&this.gpu.kernels.splice(A,1)}destroyExtensions(){this.extensions.OES_texture_float=null,this.extensions.OES_texture_float_linear=null,this.extensions.OES_element_index_uint=null,this.extensions.WEBGL_draw_buffers=null}static destroyContext(T){const A=T.getExtension("WEBGL_lose_context");A&&A.loseContext()}toJSON(){const T=super.toJSON();return T.functionNodes=_.fromKernel(this,b).toJSON(),T.settings.threadDim=this.threadDim,T}};z.exports={WebGLKernel:O}}),jr=s((B,z)=>{const M=cs(),{WebGLKernel:_}=Ls(),{glKernelString:b}=Pr();let l=null,d=null,C=null,m=null,p=null;var u=class extends _{static get isSupported(){return l!==null||(this.setupFeatureChecks(),l=C!==null),l}static setupFeatureChecks(){if(d=null,m=null,typeof M=="function")try{if(C=M(2,2,{preserveDrawingBuffer:!0}),!C||!C.getExtension)return;m={STACKGL_resize_drawingbuffer:C.getExtension("STACKGL_resize_drawingbuffer"),STACKGL_destroy_context:C.getExtension("STACKGL_destroy_context"),OES_texture_float:C.getExtension("OES_texture_float"),OES_texture_float_linear:C.getExtension("OES_texture_float_linear"),OES_element_index_uint:C.getExtension("OES_element_index_uint"),WEBGL_draw_buffers:C.getExtension("WEBGL_draw_buffers"),WEBGL_color_buffer_float:C.getExtension("WEBGL_color_buffer_float")},p=this.getFeatures()}catch(g){console.warn(g)}}static isContextMatch(g){try{return g.getParameter(g.RENDERER)==="ANGLE"}catch{return!1}}static getIsTextureFloat(){return!!m.OES_texture_float}static getIsDrawBuffers(){return!!m.WEBGL_draw_buffers}static getChannelCount(){return m.WEBGL_draw_buffers?C.getParameter(m.WEBGL_draw_buffers.MAX_DRAW_BUFFERS_WEBGL):1}static getMaxTextureSize(){return C.getParameter(C.MAX_TEXTURE_SIZE)}static get testCanvas(){return d}static get testContext(){return C}static get features(){return p}initCanvas(){return{}}initContext(){return M(2,2,{preserveDrawingBuffer:!0})}initExtensions(){this.extensions={STACKGL_resize_drawingbuffer:this.context.getExtension("STACKGL_resize_drawingbuffer"),STACKGL_destroy_context:this.context.getExtension("STACKGL_destroy_context"),OES_texture_float:this.context.getExtension("OES_texture_float"),OES_texture_float_linear:this.context.getExtension("OES_texture_float_linear"),OES_element_index_uint:this.context.getExtension("OES_element_index_uint"),WEBGL_draw_buffers:this.context.getExtension("WEBGL_draw_buffers")}}build(){super.build.apply(this,arguments),this.fallbackRequested||this.extensions.STACKGL_resize_drawingbuffer.resize(this.maxTexSize[0],this.maxTexSize[1])}destroyExtensions(){this.extensions.STACKGL_resize_drawingbuffer=null,this.extensions.STACKGL_destroy_context=null,this.extensions.OES_texture_float=null,this.extensions.OES_texture_float_linear=null,this.extensions.OES_element_index_uint=null,this.extensions.WEBGL_draw_buffers=null}static destroyContext(g){const E=g.getExtension("STACKGL_destroy_context");E&&E.destroy&&E.destroy()}toString(){return b(this.constructor,arguments,this,`const gl = context || require('gl')(1, 1);
`,`    if (!context) { gl.getExtension('STACKGL_destroy_context').destroy(); }
`)}setOutput(g){return super.setOutput(g),this.graphical&&this.extensions.STACKGL_resize_drawingbuffer&&this.extensions.STACKGL_resize_drawingbuffer.resize(this.maxTexSize[0],this.maxTexSize[1]),this}};z.exports={HeadlessGLKernel:u}}),qr=s((B,z)=>{const{utils:M}=h(),{WebGLFunctionNode:_}=yn();var b=class extends _{astIdentifierExpression(l,d){if(l.type!=="Identifier")throw this.astErrorOutput("IdentifierExpression - not an Identifier",l);const C=this.getType(l),m=M.sanitizeName(l.name);return l.name==="Infinity"?d.push("intBitsToFloat(2139095039)"):C==="Boolean"?this.argumentNames.indexOf(m)>-1?d.push(`bool(user_${m})`):d.push(`user_${m}`):d.push(`user_${m}`),d}};z.exports={WebGL2FunctionNode:b}}),ho=s((B,z)=>{z.exports={fragmentShader:`#version 300 es
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
}`}}),po=s((B,z)=>{z.exports={vertexShader:`#version 300 es
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
}`}}),fo=s((B,z)=>{const{WebGLKernelValueBoolean:M}=zr();var _=class extends M{};z.exports={WebGL2KernelValueBoolean:_}}),mo=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueFloat:_}=Rr();var b=class extends _{};z.exports={WebGL2KernelValueFloat:b}}),go=s((B,z)=>{const{WebGLKernelValueInteger:M}=Or();var _=class extends M{getSource(b){const l=this.getVariablePrecisionString();return this.origin==="constants"?`const ${l} int ${this.id} = ${parseInt(b)};
`:`uniform ${l} int ${this.id};
`}updateValue(b){this.origin!=="constants"&&this.kernel.setUniform1i(this.id,this.uploadValue=b)}};z.exports={WebGL2KernelValueInteger:_}}),Hr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueHTMLImage:_}=Rs();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`${l} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${l} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}};z.exports={WebGL2KernelValueHTMLImage:b}}),Wr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueDynamicHTMLImage:_}=xn();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`uniform ${l} ivec2 ${this.sizeId}`,`uniform ${l} ivec3 ${this.dimensionsId}`])}};z.exports={WebGL2KernelValueDynamicHTMLImage:b}}),Xr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelArray:_}=Qe();var b=class extends _{constructor(l,d){super(l,d),this.checkSize(l[0].width,l[0].height),this.dimensions=[l[0].width,l[0].height,l.length],this.textureSize=[l[0].width,l[0].height]}defineTexture(){const{context:l}=this;l.activeTexture(this.contextHandle),l.bindTexture(l.TEXTURE_2D_ARRAY,this.texture),l.texParameteri(l.TEXTURE_2D_ARRAY,l.TEXTURE_MAG_FILTER,l.NEAREST),l.texParameteri(l.TEXTURE_2D_ARRAY,l.TEXTURE_MIN_FILTER,l.NEAREST)}getStringValueHandler(){return`const uploadValue_${this.name} = ${this.varName};
`}getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2DArray ${this.id}`,`${l} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${l} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(l){const{context:d}=this;d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D_ARRAY,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!0),d.texImage3D(d.TEXTURE_2D_ARRAY,0,d.RGBA,l[0].width,l[0].height,l.length,0,d.RGBA,d.UNSIGNED_BYTE,null);for(let C=0;C<l.length;C++)d.texSubImage3D(d.TEXTURE_2D_ARRAY,0,0,0,C,l[C].width,l[C].height,1,d.RGBA,d.UNSIGNED_BYTE,this.uploadValue=l[C]);this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGL2KernelValueHTMLImageArray:b}}),yo=s((B,z)=>{const{utils:M}=h(),{WebGL2KernelValueHTMLImageArray:_}=Xr();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2DArray ${this.id}`,`uniform ${l} ivec2 ${this.sizeId}`,`uniform ${l} ivec3 ${this.dimensionsId}`])}updateValue(l){const{width:d,height:C}=l[0];this.checkSize(d,C),this.dimensions=[d,C,l.length],this.textureSize=[d,C],this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGL2KernelValueDynamicHTMLImageArray:b}}),xo=s((B,z)=>{const{utils:M}=h(),{WebGL2KernelValueHTMLImage:_}=Hr();var b=class extends _{};z.exports={WebGL2KernelValueHTMLVideo:b}}),bo=s((B,z)=>{const{utils:M}=h(),{WebGL2KernelValueDynamicHTMLImage:_}=Wr();var b=class extends _{};z.exports={WebGL2KernelValueDynamicHTMLVideo:b}}),Yr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueSingleInput:_}=bn();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`${l} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${l} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(l){const{context:d}=this;M.flattenTo(l.value,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA32F,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGL2KernelValueSingleInput:b}}),vo=s((B,z)=>{const{utils:M}=h(),{WebGL2KernelValueSingleInput:_}=Yr();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`uniform ${l} ivec2 ${this.sizeId}`,`uniform ${l} ivec3 ${this.dimensionsId}`])}updateValue(l){let[d,C,m]=l.size;this.dimensions=new Int32Array([d||1,C||1,m||1]),this.textureSize=M.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGL2KernelValueDynamicSingleInput:b}}),wo=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueUnsignedInput:_}=vn();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`${l} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${l} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}};z.exports={WebGL2KernelValueUnsignedInput:b}}),ko=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueDynamicUnsignedInput:_}=Lr();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`uniform ${l} ivec2 ${this.sizeId}`,`uniform ${l} ivec3 ${this.dimensionsId}`])}};z.exports={WebGL2KernelValueDynamicUnsignedInput:b}}),To=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueMemoryOptimizedNumberTexture:_}=Os();var b=class extends _{getSource(){const{id:l,sizeId:d,textureSize:C,dimensionsId:m,dimensions:p}=this,u=this.getVariablePrecisionString();return M.linesToString([`uniform sampler2D ${l}`,`${u} ivec2 ${d} = ivec2(${C[0]}, ${C[1]})`,`${u} ivec3 ${m} = ivec3(${p[0]}, ${p[1]}, ${p[2]})`])}};z.exports={WebGL2KernelValueMemoryOptimizedNumberTexture:b}}),So=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueDynamicMemoryOptimizedNumberTexture:_}=Fr();var b=class extends _{getSource(){return M.linesToString([`uniform sampler2D ${this.id}`,`uniform ivec2 ${this.sizeId}`,`uniform ivec3 ${this.dimensionsId}`])}};z.exports={WebGL2KernelValueDynamicMemoryOptimizedNumberTexture:b}}),_o=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueNumberTexture:_}=wn();var b=class extends _{getSource(){const{id:l,sizeId:d,textureSize:C,dimensionsId:m,dimensions:p}=this,u=this.getVariablePrecisionString();return M.linesToString([`uniform ${u} sampler2D ${l}`,`${u} ivec2 ${d} = ivec2(${C[0]}, ${C[1]})`,`${u} ivec3 ${m} = ivec3(${p[0]}, ${p[1]}, ${p[2]})`])}};z.exports={WebGL2KernelValueNumberTexture:b}}),Co=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueDynamicNumberTexture:_}=Gr();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`uniform ${l} ivec2 ${this.sizeId}`,`uniform ${l} ivec3 ${this.dimensionsId}`])}};z.exports={WebGL2KernelValueDynamicNumberTexture:b}}),Jr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueSingleArray:_}=kn();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`${l} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${l} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(l.constructor);return}const{context:d}=this;M.flattenTo(l,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA32F,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGL2KernelValueSingleArray:b}}),Ao=s((B,z)=>{const{utils:M}=h(),{WebGL2KernelValueSingleArray:_}=Jr();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`uniform ${l} ivec2 ${this.sizeId}`,`uniform ${l} ivec3 ${this.dimensionsId}`])}updateValue(l){this.dimensions=M.getDimensions(l,!0),this.textureSize=M.getMemoryOptimizedFloatTextureSize(this.dimensions,this.bitRatio),this.uploadArrayLength=this.textureSize[0]*this.textureSize[1]*this.bitRatio,this.checkSize(this.textureSize[0],this.textureSize[1]),this.uploadValue=new Float32Array(this.uploadArrayLength),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGL2KernelValueDynamicSingleArray:b}}),Zr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueSingleArray1DI:_}=Tn();var b=class extends _{updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(l.constructor);return}const{context:d}=this;M.flattenTo(l,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA32F,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGL2KernelValueSingleArray1DI:b}}),Eo=s((B,z)=>{const{utils:M}=h(),{WebGL2KernelValueSingleArray1DI:_}=Zr();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`uniform ${l} ivec2 ${this.sizeId}`,`uniform ${l} ivec3 ${this.dimensionsId}`])}updateValue(l){this.setShape(l),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGL2KernelValueDynamicSingleArray1DI:b}}),Qr=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueSingleArray2DI:_}=Sn();var b=class extends _{updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(l.constructor);return}const{context:d}=this;M.flattenTo(l,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA32F,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGL2KernelValueSingleArray2DI:b}}),Io=s((B,z)=>{const{utils:M}=h(),{WebGL2KernelValueSingleArray2DI:_}=Qr();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`uniform ${l} ivec2 ${this.sizeId}`,`uniform ${l} ivec3 ${this.dimensionsId}`])}updateValue(l){this.setShape(l),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGL2KernelValueDynamicSingleArray2DI:b}}),ei=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueSingleArray3DI:_}=_n();var b=class extends _{updateValue(l){if(l.constructor!==this.initialValueConstructor){this.onUpdateValueMismatch(l.constructor);return}const{context:d}=this;M.flattenTo(l,this.uploadValue),d.activeTexture(this.contextHandle),d.bindTexture(d.TEXTURE_2D,this.texture),d.pixelStorei(d.UNPACK_FLIP_Y_WEBGL,!1),d.texImage2D(d.TEXTURE_2D,0,d.RGBA32F,this.textureSize[0],this.textureSize[1],0,d.RGBA,d.FLOAT,this.uploadValue),this.kernel.setUniform1i(this.id,this.index)}};z.exports={WebGL2KernelValueSingleArray3DI:b}}),Mo=s((B,z)=>{const{utils:M}=h(),{WebGL2KernelValueSingleArray3DI:_}=ei();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`uniform ${l} ivec2 ${this.sizeId}`,`uniform ${l} ivec3 ${this.dimensionsId}`])}updateValue(l){this.setShape(l),this.kernel.setUniform3iv(this.dimensionsId,this.dimensions),this.kernel.setUniform2iv(this.sizeId,this.textureSize),super.updateValue(l)}};z.exports={WebGL2KernelValueDynamicSingleArray3DI:b}}),$o=s((B,z)=>{const{WebGLKernelValueArray2:M}=Ur();var _=class extends M{};z.exports={WebGL2KernelValueArray2:_}}),Do=s((B,z)=>{const{WebGLKernelValueArray3:M}=Vr();var _=class extends M{};z.exports={WebGL2KernelValueArray3:_}}),Po=s((B,z)=>{const{WebGLKernelValueArray4:M}=Kr();var _=class extends M{};z.exports={WebGL2KernelValueArray4:_}}),zo=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueUnsignedArray:_}=Cn();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`${l} ivec2 ${this.sizeId} = ivec2(${this.textureSize[0]}, ${this.textureSize[1]})`,`${l} ivec3 ${this.dimensionsId} = ivec3(${this.dimensions[0]}, ${this.dimensions[1]}, ${this.dimensions[2]})`])}};z.exports={WebGL2KernelValueUnsignedArray:b}}),Ro=s((B,z)=>{const{utils:M}=h(),{WebGLKernelValueDynamicUnsignedArray:_}=Nr();var b=class extends _{getSource(){const l=this.getVariablePrecisionString();return M.linesToString([`uniform ${l} sampler2D ${this.id}`,`uniform ${l} ivec2 ${this.sizeId}`,`uniform ${l} ivec3 ${this.dimensionsId}`])}};z.exports={WebGL2KernelValueDynamicUnsignedArray:b}}),ti=s((B,z)=>{const{WebGL2KernelValueBoolean:M}=fo(),{WebGL2KernelValueFloat:_}=mo(),{WebGL2KernelValueInteger:b}=go(),{WebGL2KernelValueHTMLImage:l}=Hr(),{WebGL2KernelValueDynamicHTMLImage:d}=Wr(),{WebGL2KernelValueHTMLImageArray:C}=Xr(),{WebGL2KernelValueDynamicHTMLImageArray:m}=yo(),{WebGL2KernelValueHTMLVideo:p}=xo(),{WebGL2KernelValueDynamicHTMLVideo:u}=bo(),{WebGL2KernelValueSingleInput:g}=Yr(),{WebGL2KernelValueDynamicSingleInput:E}=vo(),{WebGL2KernelValueUnsignedInput:w}=wo(),{WebGL2KernelValueDynamicUnsignedInput:y}=ko(),{WebGL2KernelValueMemoryOptimizedNumberTexture:k}=To(),{WebGL2KernelValueDynamicMemoryOptimizedNumberTexture:v}=So(),{WebGL2KernelValueNumberTexture:$}=_o(),{WebGL2KernelValueDynamicNumberTexture:P}=Co(),{WebGL2KernelValueSingleArray:O}=Jr(),{WebGL2KernelValueDynamicSingleArray:T}=Ao(),{WebGL2KernelValueSingleArray1DI:A}=Zr(),{WebGL2KernelValueDynamicSingleArray1DI:f}=Eo(),{WebGL2KernelValueSingleArray2DI:F}=Qr(),{WebGL2KernelValueDynamicSingleArray2DI:L}=Io(),{WebGL2KernelValueSingleArray3DI:R}=ei(),{WebGL2KernelValueDynamicSingleArray3DI:V}=Mo(),{WebGL2KernelValueArray2:W}=$o(),{WebGL2KernelValueArray3:N}=Do(),{WebGL2KernelValueArray4:te}=Po(),{WebGL2KernelValueUnsignedArray:ee}=zo(),{WebGL2KernelValueDynamicUnsignedArray:X}=Ro(),ie={unsigned:{dynamic:{Boolean:M,Integer:b,Float:_,Array:X,"Array(2)":W,"Array(3)":N,"Array(4)":te,"Array1D(2)":!1,"Array1D(3)":!1,"Array1D(4)":!1,"Array2D(2)":!1,"Array2D(3)":!1,"Array2D(4)":!1,"Array3D(2)":!1,"Array3D(3)":!1,"Array3D(4)":!1,Input:y,NumberTexture:P,"ArrayTexture(1)":P,"ArrayTexture(2)":P,"ArrayTexture(3)":P,"ArrayTexture(4)":P,MemoryOptimizedNumberTexture:v,HTMLCanvas:d,OffscreenCanvas:d,HTMLImage:d,ImageBitmap:d,ImageData:d,HTMLImageArray:m,HTMLVideo:u},static:{Boolean:M,Float:_,Integer:b,Array:ee,"Array(2)":W,"Array(3)":N,"Array(4)":te,"Array1D(2)":!1,"Array1D(3)":!1,"Array1D(4)":!1,"Array2D(2)":!1,"Array2D(3)":!1,"Array2D(4)":!1,"Array3D(2)":!1,"Array3D(3)":!1,"Array3D(4)":!1,Input:w,NumberTexture:$,"ArrayTexture(1)":$,"ArrayTexture(2)":$,"ArrayTexture(3)":$,"ArrayTexture(4)":$,MemoryOptimizedNumberTexture:v,HTMLCanvas:l,OffscreenCanvas:l,HTMLImage:l,ImageBitmap:l,ImageData:l,HTMLImageArray:C,HTMLVideo:p}},single:{dynamic:{Boolean:M,Integer:b,Float:_,Array:T,"Array(2)":W,"Array(3)":N,"Array(4)":te,"Array1D(2)":f,"Array1D(3)":f,"Array1D(4)":f,"Array2D(2)":L,"Array2D(3)":L,"Array2D(4)":L,"Array3D(2)":V,"Array3D(3)":V,"Array3D(4)":V,Input:E,NumberTexture:P,"ArrayTexture(1)":P,"ArrayTexture(2)":P,"ArrayTexture(3)":P,"ArrayTexture(4)":P,MemoryOptimizedNumberTexture:v,HTMLCanvas:d,OffscreenCanvas:d,HTMLImage:d,ImageBitmap:d,ImageData:d,HTMLImageArray:m,HTMLVideo:u},static:{Boolean:M,Float:_,Integer:b,Array:O,"Array(2)":W,"Array(3)":N,"Array(4)":te,"Array1D(2)":A,"Array1D(3)":A,"Array1D(4)":A,"Array2D(2)":F,"Array2D(3)":F,"Array2D(4)":F,"Array3D(2)":R,"Array3D(3)":R,"Array3D(4)":R,Input:g,NumberTexture:$,"ArrayTexture(1)":$,"ArrayTexture(2)":$,"ArrayTexture(3)":$,"ArrayTexture(4)":$,MemoryOptimizedNumberTexture:k,HTMLCanvas:l,OffscreenCanvas:l,HTMLImage:l,ImageBitmap:l,ImageData:l,HTMLImageArray:C,HTMLVideo:p}}};function se(J,ae,he,Ae){if(!J)throw new Error("type missing");if(!ae)throw new Error("dynamic missing");if(!he)throw new Error("precision missing");Ae.type&&(J=Ae.type);const ne=ie[he][ae];if(ne[J]===!1)return null;if(ne[J]===void 0)throw new Error(`Could not find a KernelValue for ${J}`);return ne[J]}z.exports={kernelValueMaps:ie,lookupKernelValueType:se}}),si=s((B,z)=>{const{WebGLKernel:M}=Ls(),{WebGL2FunctionNode:_}=qr(),{FunctionBuilder:b}=U(),{utils:l}=h(),{fragmentShader:d}=ho(),{vertexShader:C}=po(),{lookupKernelValueType:m}=ti();let p=null,u=null,g=null,E=null;var w=class extends M{static get isSupported(){return p!==null||(this.setupFeatureChecks(),p=this.isContextMatch(g)),p}static setupFeatureChecks(){typeof document<"u"?u=document.createElement("canvas"):typeof OffscreenCanvas<"u"&&(u=new OffscreenCanvas(0,0)),u&&(g=u.getContext("webgl2"),!(!g||!g.getExtension)&&(g.getExtension("EXT_color_buffer_float"),g.getExtension("OES_texture_float_linear"),E=this.getFeatures()))}static isContextMatch(y){return typeof WebGL2RenderingContext<"u"?y instanceof WebGL2RenderingContext:!1}static getFeatures(){const y=this.testContext;return Object.freeze({isFloatRead:this.getIsFloatRead(),isIntegerDivisionAccurate:this.getIsIntegerDivisionAccurate(),isSpeedTacticSupported:this.getIsSpeedTacticSupported(),kernelMap:!0,isTextureFloat:!0,isDrawBuffers:!0,channelCount:this.getChannelCount(),maxTextureSize:this.getMaxTextureSize(),lowIntPrecision:y.getShaderPrecisionFormat(y.FRAGMENT_SHADER,y.LOW_INT),lowFloatPrecision:y.getShaderPrecisionFormat(y.FRAGMENT_SHADER,y.LOW_FLOAT),mediumIntPrecision:y.getShaderPrecisionFormat(y.FRAGMENT_SHADER,y.MEDIUM_INT),mediumFloatPrecision:y.getShaderPrecisionFormat(y.FRAGMENT_SHADER,y.MEDIUM_FLOAT),highIntPrecision:y.getShaderPrecisionFormat(y.FRAGMENT_SHADER,y.HIGH_INT),highFloatPrecision:y.getShaderPrecisionFormat(y.FRAGMENT_SHADER,y.HIGH_FLOAT)})}static getIsTextureFloat(){return!0}static getChannelCount(){return g.getParameter(g.MAX_DRAW_BUFFERS)}static getMaxTextureSize(){return g.getParameter(g.MAX_TEXTURE_SIZE)}static lookupKernelValueType(y,k,v,$){return m(y,k,v,$)}static get testCanvas(){return u}static get testContext(){return g}static get features(){return E}static get fragmentShader(){return d}static get vertexShader(){return C}initContext(){return this.canvas.getContext("webgl2",{alpha:!1,depth:!1,antialias:!1})}initExtensions(){this.extensions={EXT_color_buffer_float:this.context.getExtension("EXT_color_buffer_float"),OES_texture_float_linear:this.context.getExtension("OES_texture_float_linear")}}validateSettings(y){if(!this.validate){this.texSize=l.getKernelTextureSize({optimizeFloatMemory:this.optimizeFloatMemory,precision:this.precision},this.output);return}const{features:k}=this.constructor;if(this.precision==="single"&&!k.isFloatRead)throw new Error("Float texture outputs are not supported");if(!this.graphical&&this.precision===null&&(this.precision=k.isFloatRead?"single":"unsigned"),this.fixIntegerDivisionAccuracy===null?this.fixIntegerDivisionAccuracy=!k.isIntegerDivisionAccurate:this.fixIntegerDivisionAccuracy&&k.isIntegerDivisionAccurate&&(this.fixIntegerDivisionAccuracy=!1),this.checkOutput(),!this.output||this.output.length===0){if(y.length!==1)throw new Error("Auto output only supported for kernels with only one input");const v=l.getVariableType(y[0],this.strictIntegers);switch(v){case"Array":this.output=l.getDimensions(v);break;case"NumberTexture":case"MemoryOptimizedNumberTexture":case"ArrayTexture(1)":case"ArrayTexture(2)":case"ArrayTexture(3)":case"ArrayTexture(4)":this.output=y[0].output;break;default:throw new Error("Auto output not supported for input type: "+v)}}if(this.graphical){if(this.output.length!==2)throw new Error("Output must have 2 dimensions on graphical mode");this.precision==="single"&&(console.warn("Cannot use graphical mode and single precision at the same time"),this.precision="unsigned"),this.texSize=l.clone(this.output);return}else!this.graphical&&this.precision===null&&k.isTextureFloat&&(this.precision="single");this.texSize=l.getKernelTextureSize({optimizeFloatMemory:this.optimizeFloatMemory,precision:this.precision},this.output),this.checkTextureSize()}translateSource(){const y=b.fromKernel(this,_,{fixIntegerDivisionAccuracy:this.fixIntegerDivisionAccuracy});this.translatedSource=y.getPrototypeString("kernel"),this.setupReturnTypes(y)}drawBuffers(){this.context.drawBuffers(this.drawBuffersMap)}getTextureFormat(){const{context:y}=this;switch(this.getInternalFormat()){case y.R32F:return y.RED;case y.RG32F:return y.RG;case y.RGBA32F:return y.RGBA;case y.RGBA:return y.RGBA;default:throw new Error("Unknown internal format")}}getInternalFormat(){const{context:y}=this;if(this.precision==="single"){if(this.pipeline)switch(this.returnType){case"Number":case"Float":case"Integer":return this.optimizeFloatMemory?y.RGBA32F:y.R32F;case"Array(2)":return y.RG32F;case"Array(3)":case"Array(4)":return y.RGBA32F;default:throw new Error("Unhandled return type")}return y.RGBA32F}return y.RGBA}_setupOutputTexture(){const y=this.context;if(this.texture){y.framebufferTexture2D(y.FRAMEBUFFER,y.COLOR_ATTACHMENT0,y.TEXTURE_2D,this.texture.texture,0);return}y.bindFramebuffer(y.FRAMEBUFFER,this.framebuffer);const k=y.createTexture(),v=this.texSize;y.activeTexture(y.TEXTURE0+this.constantTextureCount+this.argumentTextureCount),y.bindTexture(y.TEXTURE_2D,k),y.texParameteri(y.TEXTURE_2D,y.TEXTURE_WRAP_S,y.REPEAT),y.texParameteri(y.TEXTURE_2D,y.TEXTURE_WRAP_T,y.REPEAT),y.texParameteri(y.TEXTURE_2D,y.TEXTURE_MIN_FILTER,y.NEAREST),y.texParameteri(y.TEXTURE_2D,y.TEXTURE_MAG_FILTER,y.NEAREST);const $=this.getInternalFormat();this.precision==="single"?y.texStorage2D(y.TEXTURE_2D,1,$,v[0],v[1]):y.texImage2D(y.TEXTURE_2D,0,$,v[0],v[1],0,$,y.UNSIGNED_BYTE,null),y.framebufferTexture2D(y.FRAMEBUFFER,y.COLOR_ATTACHMENT0,y.TEXTURE_2D,k,0),this.texture=new this.TextureConstructor({texture:k,size:v,dimensions:this.threadDim,output:this.output,context:this.context,internalFormat:this.getInternalFormat(),textureFormat:this.getTextureFormat(),kernel:this})}_setupSubOutputTextures(){const y=this.context;if(this.mappedTextures){for(let v=0;v<this.subKernels.length;v++)y.framebufferTexture2D(y.FRAMEBUFFER,y.COLOR_ATTACHMENT0+v+1,y.TEXTURE_2D,this.mappedTextures[v].texture,0);return}const k=this.texSize;this.drawBuffersMap=[y.COLOR_ATTACHMENT0],this.mappedTextures=[];for(let v=0;v<this.subKernels.length;v++){const $=this.createTexture();this.drawBuffersMap.push(y.COLOR_ATTACHMENT0+v+1),y.activeTexture(y.TEXTURE0+this.constantTextureCount+this.argumentTextureCount+v),y.bindTexture(y.TEXTURE_2D,$),y.texParameteri(y.TEXTURE_2D,y.TEXTURE_WRAP_S,y.CLAMP_TO_EDGE),y.texParameteri(y.TEXTURE_2D,y.TEXTURE_WRAP_T,y.CLAMP_TO_EDGE),y.texParameteri(y.TEXTURE_2D,y.TEXTURE_MIN_FILTER,y.NEAREST),y.texParameteri(y.TEXTURE_2D,y.TEXTURE_MAG_FILTER,y.NEAREST);const P=this.getInternalFormat();this.precision==="single"?y.texStorage2D(y.TEXTURE_2D,1,P,k[0],k[1]):y.texImage2D(y.TEXTURE_2D,0,y.RGBA,k[0],k[1],0,y.RGBA,y.UNSIGNED_BYTE,null),y.framebufferTexture2D(y.FRAMEBUFFER,y.COLOR_ATTACHMENT0+v+1,y.TEXTURE_2D,$,0),this.mappedTextures.push(new this.TextureConstructor({texture:$,size:k,dimensions:this.threadDim,output:this.output,context:this.context,internalFormat:this.getInternalFormat(),textureFormat:this.getTextureFormat(),kernel:this}))}}_getHeaderString(){return""}_getTextureCoordinate(){const y=this.subKernels,k=this.getVariablePrecisionString(this.texSize,this.tactic);return y===null||y.length<1?`in ${k} vec2 vTexCoord;
`:`out ${k} vec2 vTexCoord;
`}_getMainArgumentsString(y){const k=[],v=this.argumentNames;for(let $=0;$<v.length;$++)k.push(this.kernelArguments[$].getSource(y[$]));return k.join("")}getKernelString(){const y=[this.getKernelResultDeclaration()],k=this.subKernels;if(k!==null)switch(y.push("layout(location = 0) out vec4 data0"),this.returnType){case"Number":case"Float":case"Integer":for(let v=0;v<k.length;v++){const $=k[v];y.push($.returnType==="Integer"?`int subKernelResult_${$.name} = 0`:`float subKernelResult_${$.name} = 0.0`,`layout(location = ${v+1}) out vec4 data${v+1}`)}break;case"Array(2)":for(let v=0;v<k.length;v++)y.push(`vec2 subKernelResult_${k[v].name}`,`layout(location = ${v+1}) out vec4 data${v+1}`);break;case"Array(3)":for(let v=0;v<k.length;v++)y.push(`vec3 subKernelResult_${k[v].name}`,`layout(location = ${v+1}) out vec4 data${v+1}`);break;case"Array(4)":for(let v=0;v<k.length;v++)y.push(`vec4 subKernelResult_${k[v].name}`,`layout(location = ${v+1}) out vec4 data${v+1}`);break}else y.push("out vec4 data0");return l.linesToString(y)+this.translatedSource}getMainResultGraphical(){return l.linesToString(["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  data0 = actualColor"])}getMainResultPackedPixels(){switch(this.returnType){case"LiteralInteger":case"Number":case"Integer":case"Float":return this.getMainResultKernelPackedPixels()+this.getMainResultSubKernelPackedPixels();default:throw new Error(`packed output only usable with Numbers, "${this.returnType}" specified`)}}getMainResultKernelPackedPixels(){return l.linesToString(["  threadId = indexTo3D(index, uOutputDim)","  kernel()",`  data0 = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(kernelResult)`])}getMainResultSubKernelPackedPixels(){const y=[];if(!this.subKernels)return"";for(let k=0;k<this.subKernels.length;k++)this.subKernels[k].returnType==="Integer"?y.push(`  data${k+1} = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(float(subKernelResult_${this.subKernels[k].name}))`):y.push(`  data${k+1} = ${this.useLegacyEncoder?"legacyEncode32":"encode32"}(subKernelResult_${this.subKernels[k].name})`);return l.linesToString(y)}getMainResultKernelMemoryOptimizedFloats(y,k){y.push("  threadId = indexTo3D(index, uOutputDim)","  kernel()",`  data0.${k} = kernelResult`)}getMainResultSubKernelMemoryOptimizedFloats(y,k){if(!this.subKernels)return y;for(let v=0;v<this.subKernels.length;v++){const $=this.subKernels[v];$.returnType==="Integer"?y.push(`  data${v+1}.${k} = float(subKernelResult_${$.name})`):y.push(`  data${v+1}.${k} = subKernelResult_${$.name}`)}}getMainResultKernelNumberTexture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  data0[0] = kernelResult"]}getMainResultSubKernelNumberTexture(){const y=[];if(!this.subKernels)return y;for(let k=0;k<this.subKernels.length;++k){const v=this.subKernels[k];v.returnType==="Integer"?y.push(`  data${k+1}[0] = float(subKernelResult_${v.name})`):y.push(`  data${k+1}[0] = subKernelResult_${v.name}`)}return y}getMainResultKernelArray2Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  data0[0] = kernelResult[0]","  data0[1] = kernelResult[1]"]}getMainResultSubKernelArray2Texture(){const y=[];if(!this.subKernels)return y;for(let k=0;k<this.subKernels.length;++k){const v=this.subKernels[k];y.push(`  data${k+1}[0] = subKernelResult_${v.name}[0]`,`  data${k+1}[1] = subKernelResult_${v.name}[1]`)}return y}getMainResultKernelArray3Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  data0[0] = kernelResult[0]","  data0[1] = kernelResult[1]","  data0[2] = kernelResult[2]"]}getMainResultSubKernelArray3Texture(){const y=[];if(!this.subKernels)return y;for(let k=0;k<this.subKernels.length;++k){const v=this.subKernels[k];y.push(`  data${k+1}[0] = subKernelResult_${v.name}[0]`,`  data${k+1}[1] = subKernelResult_${v.name}[1]`,`  data${k+1}[2] = subKernelResult_${v.name}[2]`)}return y}getMainResultKernelArray4Texture(){return["  threadId = indexTo3D(index, uOutputDim)","  kernel()","  data0 = kernelResult"]}getMainResultSubKernelArray4Texture(){const y=[];if(!this.subKernels)return y;for(let k=0;k<this.subKernels.length;++k)y.push(`  data${k+1} = subKernelResult_${this.subKernels[k].name}`);return y}destroyExtensions(){this.extensions.EXT_color_buffer_float=null,this.extensions.OES_texture_float_linear=null}toJSON(){const y=super.toJSON();return y.functionNodes=b.fromKernel(this,_).toJSON(),y.settings.threadDim=this.threadDim,y}};z.exports={WebGL2Kernel:w}}),Oo=s((B,z)=>{const{utils:M}=h();function _(l){let d=function(){return l.build.apply(l,arguments),d=function(){let m=l.run.apply(l,arguments);if(l.switchingKernels){const p=l.resetSwitchingKernels(),u=l.onRequestSwitchKernel(p,arguments,l);C.kernel=l=u,m=u.run.apply(u,arguments)}return l.renderKernels?l.renderKernels():l.renderOutput?l.renderOutput():m},d.apply(l,arguments)};const C=function(){return d.apply(l,arguments)};return C.exec=function(){return new Promise((m,p)=>{try{m(d.apply(this,arguments))}catch(u){p(u)}})},C.replaceKernel=function(m){l=m,b(l,C)},b(l,C),C}function b(l,d){if(d.kernel){d.kernel=l;return}const C=M.allPropertiesOf(l);for(let m=0;m<C.length;m++){const p=C[m];p[0]==="_"&&p[1]==="_"||(typeof l[p]=="function"?p.substring(0,3)==="add"||p.substring(0,3)==="set"?d[p]=function(){return d.kernel[p].apply(d.kernel,arguments),d}:d[p]=function(){return d.kernel[p].apply(d.kernel,arguments)}:(d.__defineGetter__(p,()=>d.kernel[p]),d.__defineSetter__(p,u=>{d.kernel[p]=u})))}d.kernel=l}z.exports={kernelRunShortcut:_}}),Lo=s((B,z)=>{const{gpuMock:M}=n(),{utils:_}=h(),{Kernel:b}=I(),{CPUKernel:l}=rt(),{HeadlessGLKernel:d}=jr(),{WebGL2Kernel:C}=si(),{WebGLKernel:m}=Ls(),{kernelRunShortcut:p}=Oo(),u=[d,C,m],g=["gpu","cpu"],E={headlessgl:d,webgl2:C,webgl:m};let w=!0;var y=class{static disableValidation(){w=!1}static enableValidation(){w=!0}static get isGPUSupported(){return u.some(v=>v.isSupported)}static get isKernelMapSupported(){return u.some(v=>v.isSupported&&v.features.kernelMap)}static get isOffscreenCanvasSupported(){return typeof Worker<"u"&&typeof OffscreenCanvas<"u"||typeof importScripts<"u"}static get isWebGLSupported(){return m.isSupported}static get isWebGL2Supported(){return C.isSupported}static get isHeadlessGLSupported(){return d.isSupported}static get isCanvasSupported(){return typeof HTMLCanvasElement<"u"}static get isGPUHTMLImageArraySupported(){return C.isSupported}static get isSinglePrecisionSupported(){return u.some(v=>v.isSupported&&v.features.isFloatRead&&v.features.isTextureFloat)}constructor(v){if(v=v||{},this.canvas=v.canvas||null,this.context=v.context||null,this.mode=v.mode,this.Kernel=null,this.kernels=[],this.functions=[],this.nativeFunctions=[],this.injectedNative=null,this.mode!=="dev"){if(this.chooseKernel(),v.functions)for(let $=0;$<v.functions.length;$++)this.addFunction(v.functions[$]);if(v.nativeFunctions)for(const $ in v.nativeFunctions){if(!v.nativeFunctions.hasOwnProperty($))continue;const P=v.nativeFunctions[$],{name:O,source:T}=P;this.addNativeFunction(O,T,P)}}}chooseKernel(){if(this.Kernel)return;let v=null;if(this.context){for(let $=0;$<u.length;$++){const P=u[$];if(P.isContextMatch(this.context)){if(!P.isSupported)throw new Error(`Kernel type ${P.name} not supported`);v=P;break}}if(v===null)throw new Error("unknown Context")}else if(this.mode){if(this.mode in E)(!w||E[this.mode].isSupported)&&(v=E[this.mode]);else if(this.mode==="gpu"){for(let $=0;$<u.length;$++)if(u[$].isSupported){v=u[$];break}}else this.mode==="cpu"&&(v=l);if(!v)throw new Error(`A requested mode of "${this.mode}" and is not supported`)}else{for(let $=0;$<u.length;$++)if(u[$].isSupported){v=u[$];break}v||(v=l)}this.mode||(this.mode=v.mode),this.Kernel=v}createKernel(v,$){if(typeof v>"u")throw new Error("Missing source parameter");if(typeof v!="object"&&!_.isFunction(v)&&typeof v!="string")throw new Error("source parameter not a function");const P=this.kernels;if(this.mode==="dev"){const V=M(v,k($));return P.push(V),V}v=typeof v=="function"?v.toString():v;const O={},T=k($)||{};$&&typeof $.argumentTypes=="object"&&(T.argumentTypes=Object.keys($.argumentTypes).map(V=>$.argumentTypes[V]));function A(V){console.warn("Falling back to CPU");const W=new l(v,{argumentTypes:R.argumentTypes,constantTypes:R.constantTypes,graphical:R.graphical,loopMaxIterations:R.loopMaxIterations,constants:R.constants,dynamicOutput:R.dynamicOutput,dynamicArgument:R.dynamicArguments,output:R.output,precision:R.precision,pipeline:R.pipeline,immutable:R.immutable,optimizeFloatMemory:R.optimizeFloatMemory,fixIntegerDivisionAccuracy:R.fixIntegerDivisionAccuracy,functions:R.functions,nativeFunctions:R.nativeFunctions,injectedNative:R.injectedNative,subKernels:R.subKernels,strictIntegers:R.strictIntegers,randomSeed:R.randomSeed,debug:R.debug});W.build.apply(W,V);const N=W.run.apply(W,V);return R.replaceKernel(W),N}function f(V,W,N){N.debug&&console.warn("Switching kernels");let te=null;if(N.signature&&!O[N.signature]&&(O[N.signature]=N),N.dynamicOutput)for(let ae=V.length-1;ae>=0;ae--){const he=V[ae];he.type==="outputPrecisionMismatch"&&(te=he.needed)}const ee=N.constructor,X=ee.getArgumentTypes(N,W),ie=ee.getSignature(N,X),se=O[ie];if(se)return se.onActivate(N),se;const J=O[ie]=new ee(v,{argumentTypes:X,constantTypes:N.constantTypes,graphical:N.graphical,loopMaxIterations:N.loopMaxIterations,constants:N.constants,dynamicOutput:N.dynamicOutput,dynamicArgument:N.dynamicArguments,context:N.context,canvas:N.canvas,output:te||N.output,precision:N.precision,pipeline:N.pipeline,immutable:N.immutable,optimizeFloatMemory:N.optimizeFloatMemory,fixIntegerDivisionAccuracy:N.fixIntegerDivisionAccuracy,functions:N.functions,nativeFunctions:N.nativeFunctions,injectedNative:N.injectedNative,subKernels:N.subKernels,strictIntegers:N.strictIntegers,randomSeed:N.randomSeed,debug:N.debug,gpu:N.gpu,validate:w,returnType:N.returnType,tactic:N.tactic,onRequestFallback:A,onRequestSwitchKernel:f,texture:N.texture,mappedTextures:N.mappedTextures,drawBuffersMap:N.drawBuffersMap});return J.build.apply(J,W),R.replaceKernel(J),P.push(J),J}const F=Object.assign({context:this.context,canvas:this.canvas,functions:this.functions,nativeFunctions:this.nativeFunctions,injectedNative:this.injectedNative,gpu:this,validate:w,onRequestFallback:A,onRequestSwitchKernel:f},T),L=new this.Kernel(v,F),R=p(L);return this.canvas||(this.canvas=L.canvas),this.context||(this.context=L.context),P.push(L),R}createKernelMap(){let v,$;const P=typeof arguments[arguments.length-2];if(P==="function"||P==="string"?(v=arguments[arguments.length-2],$=arguments[arguments.length-1]):v=arguments[arguments.length-1],this.mode!=="dev"&&(!this.Kernel.isSupported||!this.Kernel.features.kernelMap)&&this.mode&&g.indexOf(this.mode)<0)throw new Error(`kernelMap not supported on ${this.Kernel.name}`);const O=k($);if($&&typeof $.argumentTypes=="object"&&(O.argumentTypes=Object.keys($.argumentTypes).map(T=>$.argumentTypes[T])),Array.isArray(arguments[0])){O.subKernels=[];const T=arguments[0];for(let A=0;A<T.length;A++){const f=T[A].toString(),F=_.getFunctionNameFromString(f);O.subKernels.push({name:F,source:f,property:A})}}else{O.subKernels=[];const T=arguments[0];for(let A in T){if(!T.hasOwnProperty(A))continue;const f=T[A].toString(),F=_.getFunctionNameFromString(f);O.subKernels.push({name:F||A,source:f,property:A})}}return this.createKernel(v,O)}combineKernels(){const v=arguments[0],$=arguments[arguments.length-1];if(v.kernel.constructor.mode==="cpu")return $;const P=arguments[0].canvas,O=arguments[0].context,T=arguments.length-1;for(let A=0;A<T;A++)arguments[A].setCanvas(P).setContext(O).setPipeline(!0);return function(){const A=$.apply(this,arguments);return A.toArray?A.toArray():A}}setFunctions(v){return this.functions=v,this}setNativeFunctions(v){return this.nativeFunctions=v,this}addFunction(v,$){return this.functions.push({source:v,settings:$}),this}addNativeFunction(v,$,P){if(this.kernels.length>0)throw new Error('Cannot call "addNativeFunction" after "createKernels" has been called.');return this.nativeFunctions.push(Object.assign({name:v,source:$},P)),this}injectNative(v){return this.injectedNative=v,this}destroy(){return new Promise((v,$)=>{this.kernels||v(),setTimeout(()=>{try{const P=this.kernels.slice();for(let T=0;T<P.length;T++)P[T].destroy(!0);let O=P[0];O&&(O.kernel&&(O=O.kernel),O.constructor.destroyContext&&O.constructor.destroyContext(this.context))}catch(P){$(P)}v()},0)})}};function k(v){if(!v)return{};const $=Object.assign({},v);return v.hasOwnProperty("floatOutput")&&(_.warnDeprecated("setting","floatOutput","precision"),$.precision=v.floatOutput?"single":"unsigned"),v.hasOwnProperty("outputToTexture")&&(_.warnDeprecated("setting","outputToTexture","pipeline"),$.pipeline=!!v.outputToTexture),v.hasOwnProperty("outputImmutable")&&(_.warnDeprecated("setting","outputImmutable","immutable"),$.immutable=!!v.outputImmutable),v.hasOwnProperty("floatTextures")&&(_.warnDeprecated("setting","floatTextures","optimizeFloatMemory"),$.optimizeFloatMemory=!!v.floatTextures),$}z.exports={GPU:y,kernelOrder:u,kernelTypes:g}}),Fo=s((B,z)=>{const{utils:M}=h();function _(b,l){const d=l.toString();return new Function(`return function ${b} (${M.getArgumentNamesFromString(d).join(", ")}) {
  ${M.getFunctionBodyFromString(d)}
}`)()}z.exports={alias:_}}),Go=s((B,z)=>{const{GPU:M}=Lo(),{alias:_}=Fo(),{utils:b}=h(),{Input:l,input:d}=a(),{Texture:C}=c(),{FunctionBuilder:m}=U(),{FunctionNode:p}=q(),{CPUFunctionNode:u}=$e(),{CPUKernel:g}=rt(),{HeadlessGLKernel:E}=jr(),{WebGLFunctionNode:w}=yn(),{WebGLKernel:y}=Ls(),{kernelValueMaps:k}=Br(),{WebGL2FunctionNode:v}=qr(),{WebGL2Kernel:$}=si(),{kernelValueMaps:P}=ti(),{GLKernel:O}=$r(),{Kernel:T}=I(),{FunctionTracer:A}=j();z.exports={alias:_,CPUFunctionNode:u,CPUKernel:g,GPU:M,FunctionBuilder:m,FunctionNode:p,HeadlessGLKernel:E,Input:l,input:d,Texture:C,utils:b,WebGL2FunctionNode:v,WebGL2Kernel:$,webGL2KernelValueMaps:P,WebGLFunctionNode:w,WebGLKernel:y,webGLKernelValueMaps:k,GLKernel:O,Kernel:T,FunctionTracer:A,plugins:{mathRandom:Dr()}}});return s((B,z)=>{const M=Go(),_=M.GPU;for(const l in M)M.hasOwnProperty(l)&&l!=="GPU"&&(_[l]=M[l]);_.GPU=_,typeof window<"u"&&b(window),typeof self<"u"&&b(self);function b(l){l.GPU&&l.GPU.prototype&&l.GPU.prototype.createKernel||Object.defineProperty(l,"GPU",{configurable:!0,get(){return _},set(){}})}z.exports=_})()})})(dn)),dn.exports}var Ds=Su();const _u=typeof document>"u",Cu=_u?"▸ sandbox: Web Worker — a runaway kernel can be stopped":"▸ sandbox: main thread (no Worker sandbox in use) — a runaway kernel cannot be stopped",Au=/^[A-Za-z_$][A-Za-z0-9_$]*$/;let Ar=[];async function wr(){const e=Ar;Ar=[];for(const t of e)try{await t.destroy()}catch{}}function Ir(){try{return!!Ds.GPU.isGPUSupported}catch{return!1}}function Eu(e){try{const t=e.kernel;return t?typeof Ds.CPUKernel=="function"&&t instanceof Ds.CPUKernel?!0:!!(t.constructor&&t.constructor.name==="CPUKernel"):!1}catch{return!1}}function Iu(e){const t=typeof console<"u"?console:null,s=n=>(...i)=>{t&&t[n]&&t[n](...i),e({type:n==="warn"?"warn":n==="error"?"error":"log",time:Je(),text:i.map(a=>Tr(a)).join(" ")})};return{log:s("log"),info:s("info"),debug:s("debug"),warn:s("warn"),error:s("error")}}function Mu(e){return(Array.isArray(e)||ArrayBuffer.isView(e))&&e.length===4&&typeof e[0]=="number"}function $u(e){return Array.isArray(e)&&e.length>0&&Array.isArray(e[0])&&e[0].length>0&&Mu(e[0][0])}function Du(e,t){try{const s=e.kernel;if(!s||s.built||s.argumentTypes||(s.argumentNames||[]).length!==t.length)return;let i=!1;const a=t.map(c=>{if($u(c))return i=!0,"Array2D(4)";const h=Ds.utils.getVariableType(c,s.strictIntegers);return h==="Integer"?"Number":h});i&&e.setArgumentTypes(a)}catch{}}function Pu(e){if(typeof ImageData>"u")return e;let t=!1;const s=e.map(n=>n instanceof ImageData&&Array.isArray(n.plain)?(t=!0,n.plain):n);return t?s:e}function Ia(e,t,s){let n=!1;return new Proxy(e,{apply(a,c,h){a.lastArgs=h;const I=s==="cpu"?Pu(h):h;Du(a,I);const U=Reflect.apply(a,c,I);if(!n){n=!0;try{const j=a.kernel,q=j&&j.output?Array.from(j.output):null;if(q&&q.length){const $e=q.reduce((Ze,rt)=>Ze*rt,1);t({type:"system",time:Je(),text:`▸ kernel compiled · output ${q.join("×")} · ${$e.toLocaleString("en-US")} threads`})}}catch{}}return U}})}const pn=64,zu=65536,Ma=5e3;function Ru(e){if(Array.isArray(e)){const t=e.map(n=>typeof n=="number"?Math.min(n,pn):n),s=n=>n.reduce((i,a)=>i*(typeof a=="number"?a:1),1);return{clamped:t,requestedThreads:s(e),clampedThreads:s(t)}}if(e&&typeof e=="object"){const t=["x","y","z"].filter(i=>typeof e[i]=="number");if(!t.length)return null;const s={...e};t.forEach(i=>{s[i]=Math.min(e[i],pn)});const n=i=>t.reduce((a,c)=>a*i[c],1);return{clamped:s,requestedThreads:n(e),clampedThreads:n(s)}}return null}function $a(e,t){const s=e[1],n=s&&typeof s=="object"?Ru(s.output):null;if(!n)return t.unclamped=!0,e;t.requestedThreads=Math.max(t.requestedThreads,n.requestedThreads),t.clampedThreads=Math.max(t.clampedThreads,n.clampedThreads);const i=e.slice(2);return[e[0],{...s,output:n.clamped},...i]}async function Da(e,t,s){try{return await fn(e,{mode:t,task:s,probe:!0})}catch{return null}}async function Ou(e,t,s){const n=await Da(e,t,s);if(!n)return null;const i=n.probeStats||{};if(!n.ok||i.unclamped||!i.requestedThreads||!i.clampedThreads||i.clampedThreads>=i.requestedThreads||i.requestedThreads<=zu)return null;const a=i.requestedThreads/i.clampedThreads;if(n.durationMs*a<=Ma)return null;const c=await Da(e,t,s),h=c&&c.ok?Math.min(n.durationMs,c.durationMs):n.durationMs,I=h*a;return I<=Ma?null:{probeMs:h,estimateMs:I,threads:i.requestedThreads}}async function fn(e,{mode:t="auto",task:s,probe:n=!1,onLog:i}={}){await wr();const a=[],c=me=>{if(a.push(me),i)try{i(me)}catch{}},h=[],I={requestedThreads:0,clampedThreads:0,unclamped:!1};let U=null;n||c({type:"system",time:Je(),text:Cu});const j=Ir();let q;t==="cpu"?(q="cpu",c({type:"system",time:Je(),text:'▸ mode "cpu" → selected cpu'})):t==="gpu"?j?(q="gpu",c({type:"system",time:Je(),text:'▸ mode "gpu" → selected gpu (WebGL)'})):(q="cpu",c({type:"system",time:Je(),text:'▸ mode "gpu" requested but WebGL is unavailable here — falling back to cpu'})):(q=j?"gpu":"cpu",c({type:"system",time:Je(),text:j?'▸ mode "auto" → selected gpu (WebGL)':'▸ mode "auto" → selected cpu (WebGL unavailable)'}));class $e extends Ds.GPU{constructor(Ce={}){super({...Ce,mode:q}),Ar.push(this)}createKernel(...Ce){const Ue=super.createKernel(...n?$a(Ce,I):Ce),Dt=Ia(Ue,c,q);return h.push(Dt),Dt}createKernelMap(...Ce){const Ue=super.createKernelMap(...n?$a(Ce,I):Ce),Dt=Ia(Ue,c,q);return h.push(Dt),Dt}}const Ze=me=>{U=me||U,c({type:"canvas",time:Je(),text:`render: ${me&&me.constructor?me.constructor.name:"canvas"}`,canvas:me||null,snapshot:yl(me)})},rt=Iu(c),cs={GPU:$e,console:rt,render:Ze,utils:kr,mode:q};if(!n){const me=await Ou(e,t,s);if(me){const Ue=`refused to run: this would take about ${Math.round(me.estimateMs/1e3)}s and freeze the page. A ${pn}×${pn} slice took ${me.probeMs.toFixed(0)} ms, and the kernel asks for ${me.threads.toLocaleString("en-US")} threads. That much work per thread usually means a kernel is handling a whole row or array where it should handle one value — check that every array is indexed down to a number before you do arithmetic on it.`;return c({type:"error",time:Je(),text:Ue}),await wr(),{ok:!1,error:{message:Ue},logs:a,kernels:[],canvas:null,resolvedMode:q,durationMs:0,fellBackToCPU:!1,refusedAsTooSlow:!0}}await wr()}let jt=null;try{if(s&&typeof s.inputs=="function"){const me=s.inputs(kr)||{};for(const[Ce,Ue]of Object.entries(me))Au.test(Ce)&&(cs[Ce]=Ue)}}catch(me){jt=me}const De=performance.now();let $t=null;try{if(jt)throw jt;const me=Object.keys(cs);await new Function(...me,`"use strict";
return (async () => {
${e}
})();`)(...me.map(Ue=>cs[Ue]))}catch(me){$t={message:mn(me),stack:me&&me.stack?String(me.stack):void 0},c({type:"error",time:Je(),text:$t.message})}const Ps=performance.now()-De;let hs=U;if(!hs)for(let me=h.length-1;me>=0;me--){const Ce=h[me];try{if(Ce.kernel&&Ce.kernel.graphical&&Ce.canvas){hs=Ce.canvas;break}}catch{}}const gn=h.some(Eu),ds=q==="gpu"&&gn;return ds&&!n&&c({type:"warn",time:Je(),text:"▸ gpu.js could not compile this kernel for WebGL and ran it on the CPU backend instead. A graphical kernel always uses unsigned precision, which has no 2D pixel-array type, so an image built as a nested array (image[y][x] = [r, g, b, a]) can only run on the CPU. Pass this task's image through untouched — the images it hands you are ImageData, which both backends read as the same image[this.thread.y][this.thread.x] pixel."}),$t||c({type:"ok",time:Je(),text:`✓ run complete in ${Ps.toFixed(1)} ms${ds?" (on the CPU backend)":""}`}),{ok:!$t,error:$t,logs:a,kernels:h,canvas:hs,resolvedMode:q,durationMs:Ps,fellBackToCPU:ds,probeStats:n?I:void 0}}function Lu(e,t={}){const s=[],n=e.logs.map(a=>{if(!a.canvas&&!a.snapshot)return a;const{canvas:c,snapshot:h,...I}=a;return h&&h.bitmap?(s.push(h.bitmap),{...I,snapshot:h}):h?{...I,snapshot:h}:I}),i=e.canvas;return{result:{ok:e.ok,error:e.error||null,logs:n,canvasInfo:i&&i.width?{width:i.width,height:i.height}:null,kernelCount:(e.kernels||[]).length,resolvedMode:e.resolvedMode,durationMs:e.durationMs,fellBackToCPU:!!e.fellBackToCPU,refusedAsTooSlow:!!e.refusedAsTooSlow,...t},transfer:s}}function Fu(e,t){const s=e.kernels||[];return{...e,task:t,kernel:s.length?s[s.length-1]:null,utils:kr,assert:Ga,assertClose:Ua,getPixels(n){for(let a=s.length-1;a>=0;a--){const c=s[a];try{if(c.kernel&&c.kernel.graphical&&typeof c.getPixels=="function")return c.getPixels(n)}catch{}}const i=xl(e.canvas);if(i)return i;throw new Error("no graphical kernel or canvas to read pixels from")}}}async function Gu(e,t){const s=[{tests:e.publicTests||[],isPrivate:!1},{tests:e.privateTests||[],isPrivate:!0}],n=[];for(const{tests:a,isPrivate:c}of s)for(const h of a){const I=Fu(t,e),U=performance.now();let j=!0,q;try{await h.run(I)}catch($e){j=!1,q=mn($e)}n.push({name:h.name,private:c,passed:j,ms:performance.now()-U,error:q})}const i=n.filter(a=>a.passed).length;return{results:n,passed:i,total:n.length,allPassed:i===n.length}}function Pa(e){return(e.kernels||[]).filter(t=>Array.isArray(t.lastArgs))}function za(e,t){try{if(t&&typeof t.toArray=="function"){t.toArray();return}const s=e.kernel;if(s&&s.graphical){const n=s.context;n&&typeof n.readPixels=="function"&&n.readPixels(0,0,1,1,n.RGBA,n.UNSIGNED_BYTE,new Uint8Array(4))}}catch{}}function Ra(e){for(const n of e)za(n,n(...n.lastArgs));const t=[],s=performance.now();for(;t.length<5&&performance.now()-s<250;){const n=performance.now();for(const i of e)za(i,i(...i.lastArgs));t.push(performance.now()-n)}return t.sort((n,i)=>n-i),t[Math.floor(t.length/2)]}async function Uu(e,t){try{const s=await fn(e,{mode:"cpu",task:t});if(!s.ok)return{error:s.error};const n=Pa(s);if(!n.length)return{error:{message:"nothing to benchmark — the code never invoked a kernel"}};const i=Ra(n);if(!Ir())return{gpuUnavailable:!0,cpuMs:i};const a=await fn(e,{mode:"gpu",task:t});if(!a.ok)return{gpuFailed:!0,cpuMs:i,error:a.error};const c=Pa(a);if(!c.length)return{gpuFailed:!0,cpuMs:i,error:{message:"the code never invoked a kernel in gpu mode"}};const h=Ra(c);return{cpuMs:i,gpuMs:h,ratio:h>0?i/h:1/0,fasterOn:h<=i?"gpu":"cpu",gpuRanOnCpu:!!a.fellBackToCPU}}catch(s){return{error:{message:mn(s)}}}}let Ms=null,Vu=0;const Ku=300;function Mr(e){if(!e)return null;const t=ku(e.moduleId,e.taskNum);return t?t.task:null}function wt(e,t){self.postMessage(e,t&&t.length?t:void 0)}function Nu(e){let t=0;return s=>{t>=Ku||(t++,wt({id:e,kind:"log",log:{type:s.type,time:s.time,text:s.text}}))}}async function Bu(e){const t=Mr(e.taskRef),s=await fn(e.code,{mode:e.mode,task:t,onLog:Nu(e.id)}),n=`run-${++Vu}`;Ms={token:n,internal:s};const{result:i,transfer:a}=Lu(s,{runToken:n});wt({id:e.id,kind:"result",result:i},a)}async function ju(e){const t=Mr(e.taskRef);if(!t){wt({id:e.id,kind:"result",result:{unknownTask:!0}});return}if(!Ms||e.runToken&&e.runToken!==Ms.token){wt({id:e.id,kind:"result",result:{staleToken:!0}});return}const s=await Gu(t,Ms.internal);wt({id:e.id,kind:"result",result:s})}async function qu(e){const t=Mr(e.taskRef),s=await Uu(e.code,t);Ms=null,wt({id:e.id,kind:"result",result:s})}self.onmessage=async e=>{const t=e.data||{};try{switch(t.kind){case"hello":wt({id:t.id,kind:"result",result:{gpuSupported:Ir(),sandbox:"worker"}});break;case"run":await Bu(t);break;case"tests":await ju(t);break;case"benchmark":await qu(t);break;default:wt({id:t.id,kind:"failed",error:{message:`unknown request "${t.kind}"`}})}}catch(s){wt({id:t.id,kind:"failed",error:{message:mn(s)}})}};
