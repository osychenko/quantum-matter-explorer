/** Analytical regression tests; run with `npm run test:theory`. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const out=mkdtempSync(join(tmpdir(),'qme-theory40-'));
try {
  execFileSync(process.platform==='win32'?'tsc.cmd':'tsc',[
    '--strict','--target','ES2021','--module','commonjs','--skipLibCheck',
    '--outDir',out,'src/theoryDemos.ts'
  ],{cwd:root,stdio:'pipe'});
} catch(error) {
  rmSync(out,{recursive:true,force:true});
  throw new Error(`Theory calculation compilation failed: ${error.stderr?.toString() || error.message}`);
}
process.on('exit',()=>rmSync(out,{recursive:true,force:true}));
const m=createRequire(import.meta.url)(join(out,'theoryDemos.js'));
const near=(a,b,eps=1e-11)=>assert(Math.abs(a-b)<eps,`${a} differs from ${b}`);

test('imaginary-time projection filters excitations if the initial ground overlap is nonzero',()=>{
  near(m.projectedExcitedWeight(0,1,.7),.7);
  const a=m.projectedExcitedWeight(1,1,.7), b=m.projectedExcitedWeight(5,1,.7);
  assert(.7>a && a>b && b>=0);
  assert(m.projectedExcitedWeight(3,2,.7)<m.projectedExcitedWeight(3,.5,.7));
  near(m.projectedExcitedWeight(20,1,0),0);
  near(m.projectedExcitedWeight(20,1,1),1);
});

test('Gaussian oscillator energy obeys the exact variational principle',()=>{
  for(const omega of [.6,1,1.7,2]) {
    const ground=m.oscillatorExactEnergy(omega);
    near(m.oscillatorTrialEnergy(omega,omega),ground);
    near(m.oscillatorLocalEnergyStd(omega,omega),0);
    for(const alpha of [.4,.8,1.5,2.6]) {
      assert(m.oscillatorTrialEnergy(alpha,omega)>=ground-1e-12);
      assert(m.oscillatorLocalEnergyStd(alpha,omega)>=0);
    }
  }
});

test('two-boson exchange kernel has the correct limiting behavior',()=>{
  near(m.twoBosonExchangeTermFraction(0,2),.5);
  near(m.twoBosonExchangeTermFraction(1,2,.5),Math.exp(-.5)/(1+Math.exp(-.5)));
  assert(m.twoBosonExchangeTermFraction(3,2)<m.twoBosonExchangeTermFraction(1,2));
  assert(m.twoBosonExchangeTermFraction(1,4)>m.twoBosonExchangeTermFraction(1,.5));
});

test('invalid controls are rejected',()=>{
  assert.throws(()=>m.projectedExcitedWeight(-1,1,.7),RangeError);
  assert.throws(()=>m.projectedExcitedWeight(1,1,1.1),RangeError);
  assert.throws(()=>m.oscillatorTrialEnergy(0,1),RangeError);
  assert.throws(()=>m.twoBosonExchangeTermFraction(1,0),RangeError);
});
