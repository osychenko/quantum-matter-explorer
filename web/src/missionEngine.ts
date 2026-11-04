/** Patch 35 — educational missions, independent of the graphical interface.
 *
 * This module never changes the QMC engine and never grades a prediction as
 * "good" or "bad" based on whether it matches the observed outcome. Progress
 * rewards prediction BEFORE evidence, observing, and interpreting with care.
 */
export const MISSION_PROGRESS_KEY = 'qme-missions-v1';
export const MISSION_PROGRESS_VERSION = 1;
export const XP_PER_MISSION = 40;

export type Prediction = 'higher' | 'lower' | 'similar' | 'within_2_sem' | 'outside_2_sem';
export type MissionId = 'thermal-detective' | 'expanding-universe' | 'repulsion-case' | 'uncertainty-investigator';
export type EvidenceRequest = {
  label: string;
  presetId: string;
  parameters: Readonly<Record<string, number>>;
  mode: 'analytic_control' | 'recorded_result';
};
export type Mission = {
  id: MissionId;
  chapter: number;
  title: string;
  subtitle: string;
  story: string;
  hypothesis: string;
  choices: ReadonlyArray<{id: Prediction; label: string}>;
  evidence: ReadonlyArray<EvidenceRequest>;
  interpretQuestion: string;
  interpretations: ReadonlyArray<{id: string; label: string}>;
  correctInterpretation: string;
  lesson: string;
  caution: string;
  kind: 'pair' | 'reference';
};

const IDEAL = 'ideal-boson-addition-v1';
const RECORDED = 'softcore-m8-baseline-recorded-v1';

export const MISSIONS: ReadonlyArray<Mission> = [
  {
    id: 'thermal-detective', chapter: 1, title: 'The Thermal Detective',
    subtitle: 'What changes when the gas cools?',
    story: 'An invisible particle is inside a periodic box. You can change inverse temperature, β. Make a prediction before you uncover the evidence.',
    hypothesis: 'At the same box size, what happens to the zero-momentum particle-addition Green function when β grows?',
    choices: [{id:'higher',label:'It increases'}, {id:'lower',label:'It decreases'}, {id:'similar',label:'It stays about the same'}],
    evidence: [
      {label:'Warmer control · β = 0.7',presetId:IDEAL,parameters:{beta:0.7,cell_length:2.8},mode:'analytic_control'},
      {label:'Colder control · β = 4.5',presetId:IDEAL,parameters:{beta:4.5,cell_length:2.8},mode:'analytic_control'},
    ],
    interpretQuestion:'Which explanation is consistent with this ideal-gas result?',
    interpretations: [
      {id:'thermal',label:'At larger β (lower temperature), the one-particle thermal host has greater zero-mode occupation.'},
      {id:'motion',label:'We directly watched a particle move more slowly in real time.'},
      {id:'phase',label:'The finite one-particle gas has been proven to undergo a bulk phase transition.'},
    ],
    correctInterpretation:'thermal',
    lesson:'For this canonical ideal-gas model, G₀⁺ = 1 + 1/Z₁. As β rises, excited momentum states contribute less to Z₁, so the zero-mode addition correlator increases. This is an imaginary-time quantity, not a movie of particle motion.',
    caution:'Analytical control: finite momentum-sum truncation is reported by the engine. This is not an interacting many-body transition.',
    kind:'pair',
  },
  {
    id:'expanding-universe', chapter:2, title:'The Expanding Box',
    subtitle:'More room, different momentum states',
    story:'Your particle has the same inverse temperature, but you can give it a larger periodic home. Does more space always mean more of every quantum observable?',
    hypothesis:'At the same β, how does G₀⁺ change when the box grows from L = 1.8 to L = 4.8?',
    choices: [{id:'higher',label:'It increases'}, {id:'lower',label:'It decreases'}, {id:'similar',label:'It stays about the same'}],
    evidence: [
      {label:'Small cell · L = 1.8',presetId:IDEAL,parameters:{beta:2.4,cell_length:1.8},mode:'analytic_control'},
      {label:'Large cell · L = 4.8',presetId:IDEAL,parameters:{beta:2.4,cell_length:4.8},mode:'analytic_control'},
    ],
    interpretQuestion:'What physical reason best explains the change?',
    interpretations: [
      {id:'levels',label:'The larger box has more closely spaced momentum energies, increasing the thermal partition sum.'},
      {id:'speed',label:'The box walls give the particle a larger real-time speed.'},
      {id:'density',label:'We have measured a new finite-density phase of interacting particles.'},
    ],
    correctInterpretation:'levels',
    lesson:'Allowed momenta are separated by 2π/L. At fixed β, a larger L makes nonzero-momentum energies smaller. More excited states contribute to Z₁, lowering 1 + 1/Z₁ for this canonical one-host calculation.',
    caution:'The pictured periodic cell is a spatial illustration; the number shown is computed independently by the analytical control.',
    kind:'pair',
  },
  {
    id:'repulsion-case', chapter:3, title:'The Repulsion Case File',
    subtitle:'An experiment and a historical record',
    story:'A researcher has left behind a Monte Carlo result for two interacting bosons. Compare it with the ideal-gas control — then decide what the evidence can actually establish.',
    hypothesis:'Which reported Green-function value will be larger for this pair of scenarios?',
    choices: [{id:'higher',label:'The recorded interacting value'}, {id:'lower',label:'The analytical ideal-gas value'}, {id:'similar',label:'They will be nearly identical'}],
    evidence: [
      {label:'Analytical ideal control · β=2.4, L=2.8',presetId:IDEAL,parameters:{beta:2.4,cell_length:2.8},mode:'analytic_control'},
      {label:'Interacting soft-core · recorded M=8',presetId:RECORDED,parameters:{},mode:'recorded_result'},
    ],
    interpretQuestion:'Which conclusion is justified by these two values?',
    interpretations: [
      {id:'limits',label:'They differ, but this comparison alone cannot isolate the effect of interactions or certify a continuum result.'},
      {id:'proof',label:'The difference definitively proves a many-body phase transition.'},
      {id:'exact',label:'Both values are exact live solutions of the same Hamiltonian.'},
    ],
    correctInterpretation:'limits',
    lesson:'The ideal control and the recorded interacting benchmark have different Hamiltonians and numerical status. The M=8 value is a saved result, with replica uncertainty and primitive-time-step limitations. Their difference is informative but not a controlled causal estimate of interaction alone.',
    caution:'The M=8 interacting observation is a historical candidate recording, NOT a new Monte Carlo run. Do not infer a phase transition from this comparison.',
    kind:'pair',
  },
  {
    id:'uncertainty-investigator', chapter:4, title:'The Uncertainty Investigator',
    subtitle:'When two numbers look different',
    story:'A numerical measurement rarely matches its reference digit for digit. Your task is to decide whether the discrepancy is large relative to its reported uncertainty.',
    hypothesis:'Is the recorded M=8 interacting result within two reported replica SEMs of its K=2 spectral reference?',
    choices: [{id:'within_2_sem',label:'Yes, within two SEMs'}, {id:'outside_2_sem',label:'No, farther than two SEMs'}],
    evidence: [
      {label:'M=8 baseline · historical QMC benchmark',presetId:RECORDED,parameters:{},mode:'recorded_result'},
    ],
    interpretQuestion:'What can such agreement tell us?',
    interpretations: [
      {id:'screen',label:'It passes a useful consistency screen, but does not prove timestep or basis convergence.'},
      {id:'proven',label:'It proves the complete interacting continuum theory is exact.'},
      {id:'zero',label:'It means the reported uncertainty is zero.'},
    ],
    correctInterpretation:'screen',
    lesson:'A difference smaller than two independent-replica SEMs is consistent with the reported statistical error. But the spectral reference has finite basis, the worm has finite imaginary-time step, and mixing must be checked separately. Agreement is evidence, not a proof.',
    caution:'Statistical screening only. The K=2 result is a truncated spectral reference, not an exact continuum result.',
    kind:'reference',
  },
] as const;

export type EngineObservation = {
  schema_version: string;
  experiment_id: string;
  execution_mode: string;
  preset_status: string;
  parameters: Record<string, unknown>;
  observables: Array<{id:string;value:number;uncertainty:number|null; uncertainty_kind:string; spectral_reference_k2?:number}>;
  fingerprint_sha256: string;
  cautions: string[];
};
export type ObservationSummary = { label:string; value:number; uncertainty:number|null; source:'analytic_control'|'recorded_result'; reference?:number; fingerprint:string };
export type Finding = { comparison:Prediction; difference:number; zScore:number|null; summaries:ObservationSummary[] };

/** Reject a result whose model, parameters, or provenance do not match the mission.
 * An educational scoring gate is not a cryptographic proof: it trusts the local API.
 */
export function assessEvidence(mission:Mission, results:EngineObservation[]):Finding {
  if (results.length !== mission.evidence.length) throw new Error('Incomplete mission evidence');
  const summaries = results.map((result, i):ObservationSummary => {
    const requested = mission.evidence[i];
    if (!result || result.schema_version !== '1.0.0' ||
        result.experiment_id !== requested.presetId || result.execution_mode !== requested.mode ||
        !Array.isArray(result.cautions) ||
        !/^[0-9a-f]{64}$/.test(result.fingerprint_sha256)) {
      throw new Error('Experiment result does not match the mission contract');
    }
    if (requested.mode === 'recorded_result' && result.preset_status !== 'candidate_recording') {
      throw new Error('Recorded result has an unexpected scientific status');
    }
    for (const [key,value] of Object.entries(requested.parameters)) {
      if (typeof result.parameters?.[key] !== 'number' ||
          Math.abs((result.parameters[key] as number)-value)>1e-9) {
        throw new Error('Returned parameters differ from the requested experiment');
      }
    }
    const obs = result.observables?.find(o=>o.id==='green_addition_midpoint_k0');
    if (!obs || !Number.isFinite(obs.value) || obs.value<=0 ||
        (obs.uncertainty !== null && (!Number.isFinite(obs.uncertainty) || obs.uncertainty<0))) {
      throw new Error('Invalid numerical observation');
    }
    if (requested.mode==='recorded_result' &&
        (obs.uncertainty === null || obs.uncertainty_kind !== 'independent_replica_sem' ||
         !Number.isFinite(obs.spectral_reference_k2))) {
      throw new Error('Missing independent-replica uncertainty or spectral reference');
    }
    return {label:requested.label,value:obs.value,uncertainty:obs.uncertainty,
      source:requested.mode,reference:obs.spectral_reference_k2,fingerprint:result.fingerprint_sha256};
  });
  if (mission.kind === 'reference') {
    const obs = summaries[0];
    if (obs.reference === undefined || obs.uncertainty === null || obs.uncertainty === 0) {
      throw new Error('Cannot compare against reference without reported uncertainty');
    }
    const delta = obs.value - obs.reference;
    const z = Math.abs(delta) / obs.uncertainty;
    return {comparison: z<=2?'within_2_sem':'outside_2_sem',difference:delta,zScore:z,summaries};
  }
  const delta = summaries[1].value - summaries[0].value;
  const epsilon = 1e-6 * Math.max(1,summaries[0].value,summaries[1].value);
  return {comparison:Math.abs(delta)<=epsilon?'similar':delta>0?'higher':'lower',
    difference:delta,zScore:null,summaries};
}

export type MissionCompletion = {
  missionId:MissionId;
  completedAt:string;
  prediction:Prediction;
  interpretation:string;
  predictionMatched:boolean;
  interpretationMatched:boolean;
  fingerprintHashes:string[];
  researchNote:string;
};
export type MissionProgress = {version:1; completions:MissionCompletion[]};
export const EMPTY_PROGRESS:MissionProgress = {version:1,completions:[]};
export function validatedProgress(data:unknown):MissionProgress {
  if (!data || typeof data!=='object') return EMPTY_PROGRESS;
  const d = data as Record<string,unknown>;
  if (d.version!==MISSION_PROGRESS_VERSION || !Array.isArray(d.completions)) return EMPTY_PROGRESS;
  const seen = new Set<string>();
  const completions:MissionCompletion[] = [];
  for (const raw of d.completions.slice(0,MISSIONS.length)) {
    if (!raw || typeof raw!=='object') continue;
    const r = raw as Record<string,unknown>;
    if (!MISSIONS.some(m=>m.id===r.missionId) || seen.has(String(r.missionId)) ||
        typeof r.completedAt!=='string' || !Number.isFinite(Date.parse(r.completedAt)) ||
        !['higher','lower','similar','within_2_sem','outside_2_sem'].includes(String(r.prediction)) ||
        typeof r.interpretation!=='string' || typeof r.predictionMatched!=='boolean' ||
        typeof r.interpretationMatched!=='boolean' || !Array.isArray(r.fingerprintHashes) ||
        !r.fingerprintHashes.every(x=>typeof x==='string' && /^[0-9a-f]{64}$/.test(x)) ||
        typeof r.researchNote!=='string' || r.researchNote.length>400) continue;
    const id = r.missionId as MissionId;
    const m = MISSIONS.find(m=>m.id===id)!;
    if (!m.choices.some(c=>c.id===r.prediction) || !m.interpretations.some(c=>c.id===r.interpretation) ||
        r.fingerprintHashes.length!==m.evidence.length) continue;
    seen.add(id);
    completions.push({missionId:id,completedAt:r.completedAt as string,prediction:r.prediction as Prediction,
      interpretation:r.interpretation as string,predictionMatched:r.predictionMatched as boolean,
      interpretationMatched:r.interpretationMatched as boolean,
      fingerprintHashes:r.fingerprintHashes as string[],researchNote:r.researchNote as string});
  }
  return {version:1,completions};
}
export function recordCompletion(progress:MissionProgress, mission:Mission, prediction:Prediction,
    interpretation:string, finding:Finding, researchNote:string, completedAt=new Date().toISOString()):MissionProgress {
  if (progress.completions.some(c=>c.missionId===mission.id)) return progress; // No duplicate XP.
  if (!mission.choices.some(c=>c.id===prediction) || !mission.interpretations.some(c=>c.id===interpretation) ||
      finding.summaries.length!==mission.evidence.length) throw new Error('Missing prediction, interpretation or observations');
  if (researchNote.length>400) throw new Error('Research note exceeds 400 characters');
  return {version:1,completions:[...progress.completions,{
    missionId:mission.id,completedAt,prediction,interpretation,
    predictionMatched:prediction===finding.comparison,
    interpretationMatched:interpretation===mission.correctInterpretation,
    fingerprintHashes:finding.summaries.map(o=>o.fingerprint),researchNote,
  }]};
}
export function totalXp(progress:MissionProgress):number {return progress.completions.length*XP_PER_MISSION;}
export function isUnlocked(progress:MissionProgress, mission:Mission):boolean {
  // Guided progression. Completed chapters remain replayable.
  return mission.chapter===1 || progress.completions.some(c=>
    MISSIONS.find(m=>m.id===c.missionId)?.chapter===mission.chapter-1);
}
export function earnedBadges(progress:MissionProgress):Array<{id:string;title:string;detail:string}> {
  const count=progress.completions.length;
  const has = (id:MissionId)=>progress.completions.some(c=>c.missionId===id);
  return [
    ...(count>=1?[{id:'hypothesis',title:'Hypothesis Maker',detail:'Predicted before observing'}]:[]),
    ...(count>=2?[{id:'field-notes',title:'Evidence Collector',detail:'Completed two investigations'}]:[]),
    ...(has('repulsion-case')?[{id:'model-guardian',title:'Model Guardian',detail:'Distinguished recorded data from live calculation'}]:[]),
    ...(has('uncertainty-investigator')?[{id:'skeptic',title:'Careful Skeptic',detail:'Interpreted numerical uncertainty'}]:[]),
    ...(count===MISSIONS.length?[{id:'explorer',title:'Quantum Explorer',detail:'Finished the first research expedition'}]:[]),
  ];
}
