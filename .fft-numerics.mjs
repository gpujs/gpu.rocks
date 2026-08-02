const w = (await import('/Users/fuzzie/Documents/gpu.rocks/src/Bench/workloads/fft.js')).default;
const size = w.size, inputs = w.make(size);
const ref = w.js(size, inputs);
const refCheck = w.reduce(ref, size);

// fp32-throughout: every intermediate operation rounded with Math.fround
const f = Math.fround;
function jsF32({ n, bits }, { re, im, twRe, twIm }) {
  const ar = new Float32Array(n), ai = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0, v = i;
    for (let b = 0; b < bits; b++) { const h = Math.floor(v / 2); r = r * 2 + (v - h * 2); v = h; }
    ar[i] = re[r]; ai[i] = im[r];
  }
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1, step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let j = 0, m = 0; j < halfLen; j++, m += step) {
        const wr = twRe[m], wi = twIm[m], p = i + j, q = p + halfLen;
        const br = ar[q], bi = ai[q];
        const tr = f(f(wr * br) - f(wi * bi));
        const ti = f(f(wr * bi) + f(wi * br));
        ar[q] = f(ar[p] - tr); ai[q] = f(ai[p] - ti);
        ar[p] = f(ar[p] + tr); ai[p] = f(ai[p] + ti);
      }
    }
  }
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = f(Math.sqrt(f(f(ar[i]*ar[i]) + f(ai[i]*ai[i]))));
  return out;
}
const c32 = w.reduce(jsF32(size, inputs), size);
console.log('oracle checksum        ', refCheck);
console.log('fp32-throughout        ', c32);
console.log('relative difference    ', (Math.abs(c32 - refCheck) / Math.abs(refCheck)).toExponential(2));
console.log('checksum tolerance     ', (1e-4).toExponential(1));

// memory of one ping-pong texture: output [n,2], one float per RGBA32F texel
const texels = size.n * 2;
console.log('ping-pong texture      ', texels, 'texels x 16 B =', (texels*16/1048576).toFixed(2), 'MB');
