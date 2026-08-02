const N = 1024;
function lcg(seed) { let s = seed >>> 0; return () => (s = (s * 1664525 + 1013904223) >>> 0); }
const next = lcg(0xc2b2ae35);
const spins = new Float32Array(N*N); for (let i=0;i<spins.length;i++) spins[i] = next() >>> 31 ? 1 : -1;
const rnd = new Float32Array(N*N);   for (let i=0;i<rnd.length;i++)   rnd[i] = next() >>> 16;

// 1. exactness of the LCG in fp64
let s = 0xc2b2ae35>>>0, ok=true;
for (let i=0;i<100000;i++){ const p = s*1664525 + 1013904223; if (p > Number.MAX_SAFE_INTEGER) { ok=false; break; } s = p>>>0; }
console.log('LCG product stays under 2^53 (exact in double):', ok, ' max product =', (4294967295*1664525+1013904223).toExponential(3), ' 2^53 =', Number.MAX_SAFE_INTEGER.toExponential(3));

// 2. are the two sequences identical / correlated?
let same=0; for (let i=0;i<spins.length;i++) if ((rnd[i] >= 32768 ? 1 : -1) === spins[i]) same++;
console.log('sign(rnd) agrees with spin on', (100*same/spins.length).toFixed(3), '% of sites (50% = uncorrelated)');

// 3. row-stripe test on the initial lattice (the failure mode the comment warns about)
const rowSum = []; for (let y=0;y<N;y++){ let a=0; for(let x=0;x<N;x++) a+=spins[y*N+x]; rowSum.push(a); }
let identicalRows=0; for (let y=1;y<N;y++){ let eq=true; for(let x=0;x<N;x++) if(spins[y*N+x]!==spins[x]) {eq=false;break;} if(eq) identicalRows++; }
console.log('rows identical to row 0:', identicalRows, '| row-sum mean', (rowSum.reduce((a,b)=>a+b,0)/N).toFixed(2), 'sd', Math.sqrt(rowSum.reduce((a,b)=>a+b*b,0)/N).toFixed(2), '(sd ~32 expected for 1024 fair coins)');

// 4. period of the 16-bit draw word: does rnd repeat within the 2^20 values drawn?
const seenAt = new Map(); let firstRepeatLag = null;
for (let i=0;i<rnd.length && firstRepeatLag===null;i++){ /* check exact sequence repeat, not value repeat */ }
let lagOK = true;
for (const lag of [1<<17, 1<<18, 1<<19]) {
  let eq = 0; const m = rnd.length - lag;
  for (let i=0;i<m;i++) if (rnd[i]===rnd[i+lag]) eq++;
  console.log(`  fraction of draws equal to the draw ${lag} later: ${(eq/m).toExponential(2)} (chance = ${(1/65536).toExponential(2)})`);
}

// 5. uniformity + how often each threshold fires
const LEVELS=65536, TEMP=2.2;
const A4 = Math.round(Math.exp(-4/TEMP)*LEVELS), A8 = Math.round(Math.exp(-8/TEMP)*LEVELS);
let lt4=0, lt8=0, mx=0, mn=Infinity, sum=0;
for (let i=0;i<rnd.length;i++){ const v=rnd[i]; sum+=v; if(v<A4)lt4++; if(v<A8)lt8++; if(v>mx)mx=v; if(v<mn)mn=v; }
console.log(`ACCEPT_4=${A4} (exp=${(Math.exp(-4/TEMP)).toFixed(6)}) ACCEPT_8=${A8} (exp=${(Math.exp(-8/TEMP)).toFixed(6)})`);
console.log(`draws: min ${mn} max ${mx} mean ${(sum/rnd.length).toFixed(1)} (expect 32767.5); P(r<A4)=${(lt4/rnd.length).toFixed(5)} vs ${(A4/LEVELS).toFixed(5)}; P(r<A8)=${(lt8/rnd.length).toFixed(5)} vs ${(A8/LEVELS).toFixed(5)}`);
console.log('all draws exactly representable in fp32:', rnd.every(v => Number.isInteger(v) && v < (1<<24)));
