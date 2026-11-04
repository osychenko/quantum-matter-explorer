/** Run `npm run test:missions` (Node >=18, TypeScript in devDependencies). */
import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const webRoot=fileURLToPath(new URL('..',import.meta.url));
const tmp=mkdtempSync(join(tmpdir(),'qme-mission-test-'));
const tsc=process.platform==='win32'?'tsc.cmd':'tsc';
try {
  execFileSync(tsc,["--strict","--target","ES2021","--module","commonjs","--skipLibCheck",
    "--outDir",tmp,"src/missionEngine.ts"],{cwd:webRoot,stdio:'pipe'});
} catch(e) {
  rmSync(tmp,{recursive:true,force:true});
  throw new Error(`Mission engine TypeScript compilation failed: ${e.stderr?.toString() || e.message}`);
}
const require=createRequire(import.meta.url);
const engine=require(join(tmp,'missionEngine.js'));
process.on('exit',()=>rmSync(tmp,{recursive:true,force:true}));
const {MISSIONS,EMPTY_PROGRESS,assessEvidence,recordCompletion,totalXp,isUnlocked,earnedBadges,validatedProgress}=engine;

function synthetic(req,value,options={}) {
  const param=req.mode==='recorded_result'?{beta:2.4,M:8,cell_length:2.8}:req.parameters;
  return {
    schema_version:'1.0.0',experiment_id:req.presetId,execution_mode:req.mode,
    preset_status:req.mode==='recorded_result'?'candidate_recording':'supported',
    parameters:param,cautions:['Educational limits apply'],fingerprint_sha256:'f'.repeat(64),
    observables:[{id:'green_addition_midpoint_k0',value,uncertainty:req.mode==='recorded_result'?0.043524585:null,
      uncertainty_kind:req.mode==='recorded_result'?'independent_replica_sem':'not_applicable_analytic',
      spectral_reference_k2:req.mode==='recorded_result'?1.691025596:undefined}],...options,
  };
}
const m=id=>MISSIONS.find(m=>m.id===id);

test('mission catalog has four unique sequential investigations and curated request IDs',()=>{
  assert.equal(MISSIONS.length,4);
  assert.deepEqual(MISSIONS.map(m=>m.chapter),[1,2,3,4]);
  assert.equal(new Set(MISSIONS.map(m=>m.id)).size,4);
  assert(MISSIONS.every(m=>m.evidence.every(r=>r.presetId==='ideal-boson-addition-v1'||r.presetId==='softcore-m8-baseline-recorded-v1')));
  assert(MISSIONS.every(m=>m.choices.some(c=>c.id==='higher') || m.kind==='reference'));
});

test('prediction is derived from returned observations, not pre-awarded answer',()=>{
  const mission=m('thermal-detective');
  const findings=assessEvidence(mission,[synthetic(mission.evidence[0],1.2),synthetic(mission.evidence[1],1.9)]);
  assert.equal(findings.comparison,'higher');
  assert.equal(findings.difference,0.7);
  const reverse=assessEvidence(mission,[synthetic(mission.evidence[0],1.7),synthetic(mission.evidence[1],1.2)]);
  assert.equal(reverse.comparison,'lower');
});

test('recorded observations are not treated as live simulations or exact results',()=>{
  const mission=m('repulsion-case');
  const result=assessEvidence(mission,[synthetic(mission.evidence[0],1.98588),synthetic(mission.evidence[1],1.68505)]);
  assert.equal(result.comparison,'lower');
  assert.equal(result.summaries[1].source,'recorded_result');
  assert.throws(()=>assessEvidence(mission,[synthetic(mission.evidence[0],1.9),synthetic(mission.evidence[1],1.6,{execution_mode:'analytic_control'})]),/contract/);
  assert.throws(()=>assessEvidence(mission,[synthetic(mission.evidence[0],1.9),synthetic(mission.evidence[1],1.6,{preset_status:'supported'})]),/status/);
});

test('uncertainty mission evaluates discrepancy in reported independent-replica SEM',()=>{
  const mission=m('uncertainty-investigator');
  const valid=assessEvidence(mission,[synthetic(mission.evidence[0],1.68505)]);
  assert.equal(valid.comparison,'within_2_sem');
  assert(valid.zScore<1);
  const mismatch=assessEvidence(mission,[synthetic(mission.evidence[0],2.2)]);
  assert.equal(mismatch.comparison,'outside_2_sem');
  assert(mismatch.zScore>2);
});

test('mission rejects incorrect parameters, fingerprint and invalid numeric observations',()=>{
  const mission=m('expanding-universe');
  const a=synthetic(mission.evidence[0],1.8), b=synthetic(mission.evidence[1],1.5);
  assert.throws(()=>assessEvidence(mission,[{...a,parameters:{beta:1,cell_length:1.8}},b]),/parameters/);
  assert.throws(()=>assessEvidence(mission,[{...a,fingerprint_sha256:'nope'},b]),/contract/);
  assert.throws(()=>assessEvidence(mission,[{...a,observables:[{id:'green_addition_midpoint_k0',value:Infinity,uncertainty:null}]},b]),/Invalid numerical/);
});

test('same XP for a wrong hypothesis and correct one; duplicate completion earns none',()=>{
  const mission=m('thermal-detective');
  const finding=assessEvidence(mission,[synthetic(mission.evidence[0],1.2),synthetic(mission.evidence[1],1.6)]);
  const wrong=recordCompletion(EMPTY_PROGRESS,mission,'lower','motion',finding,'My hypothesis was contradicted');
  const right=recordCompletion(EMPTY_PROGRESS,mission,'higher','thermal',finding,'I expected this trend');
  assert.equal(totalXp(wrong),40);
  assert.equal(totalXp(right),40);
  assert.equal(wrong.completions[0].predictionMatched,false);
  assert.equal(wrong.completions[0].interpretationMatched,false);
  const twice=recordCompletion(wrong,mission,'higher','thermal',finding,'');
  assert.equal(totalXp(twice),40);
  assert.equal(twice.completions.length,1);
  assert.equal(isUnlocked(wrong,MISSIONS[1]),true);
  assert.equal(isUnlocked(wrong,MISSIONS[2]),false);
});

test('bounded storage recovery, stable achievements and graduated unlocks',()=>{
  const m1=MISSIONS[0],f1=assessEvidence(m1,[synthetic(m1.evidence[0],1.2),synthetic(m1.evidence[1],1.8)]);
  const p1=recordCompletion(EMPTY_PROGRESS,m1,'lower','thermal',f1,'');
  const p2=validatedProgress({...p1,completions:[...p1.completions,p1.completions[0]]});
  assert.equal(p2.completions.length,1);
  assert.equal(validatedProgress({version:999,completions:[]}).completions.length,0);
  assert.equal(earnedBadges(p1)[0].title,'Hypothesis Maker');
  const m2=MISSIONS[1],f2=assessEvidence(m2,[synthetic(m2.evidence[0],1.9),synthetic(m2.evidence[1],1.2)]);
  const two=recordCompletion(p1,m2,'higher','levels',f2,'');
  assert.equal(totalXp(two),80);
  assert.equal(isUnlocked(two,MISSIONS[2]),true);
});
