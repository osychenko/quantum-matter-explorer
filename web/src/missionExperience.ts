/** Patch 36: presentation-only guidance derived from verified mission evidence.
 * Do not infer physical claims or trust user-supplied text as simulation output.
 */
import type {Finding, Mission, Prediction} from './missionEngine';

export type MissionStage = 'predict' | 'collect' | 'interpret' | 'review';
export function missionStage(prediction:Prediction|null, finding:Finding|null, feedback:boolean):MissionStage {
  if (feedback) return 'review';
  if (finding) return 'interpret';
  if (prediction) return 'collect';
  return 'predict';
}

export function observedOutcome(mission:Mission, finding:Finding):string {
  if (mission.kind === 'reference') {
    if (finding.zScore === null || !Number.isFinite(finding.zScore)) throw new Error('Missing reference comparison');
    return `The measured difference is ${finding.zScore.toFixed(2)} reported standard errors from the K=2 reference. ${finding.zScore <= 2 ? 'It passes this two-SEM screening rule.' : 'It fails this two-SEM screening rule.'} Neither outcome proves the physical model correct or incorrect.`;
  }
  const names=finding.summaries;
  if (names.length !== 2) throw new Error('Pair mission requires two observations');
  const phrase=finding.comparison === 'higher' ? 'higher than' : finding.comparison === 'lower' ? 'lower than' : 'approximately equal to';
  return `“${names[1].label}” is ${phrase} “${names[0].label}” for this specific Green-function observable. The numerical difference is ${finding.difference.toFixed(5)}.`;
}

export function learningSteps():ReadonlyArray<{id:MissionStage;title:string;detail:string}> {
  return [
    {id:'predict',title:'Predict',detail:'Choose a hypothesis before seeing the result'},
    {id:'collect',title:'Collect',detail:'Request curated numerical evidence'},
    {id:'interpret',title:'Interpret',detail:'Explain what the evidence supports'},
    {id:'review',title:'Reflect',detail:'Read the explanation and record the discovery'},
  ];
}

export const RESEARCH_GLOSSARY = [
  {term:'Inverse temperature β',definition:'A parameter inversely proportional to temperature when the relevant energy units are fixed. Larger β means colder.'},
  {term:'Green function',definition:'The quantity used in these missions describes particle addition at an imaginary-time separation. It is not a real-time particle trajectory.'},
  {term:'Periodic cell',definition:'A simulation box repeated in every direction. Opposite sides represent the same spatial location.'},
  {term:'Recorded benchmark',definition:'A saved result of a previous Monte Carlo experiment. Viewing it is not running a new simulation.'},
  {term:'Replica SEM',definition:'The standard error estimated from independently seeded chain results. It represents sampling uncertainty, not all systematic biases.'},
  {term:'Spectral reference',definition:'An independently computed finite-basis comparison. Matching it is a useful check, not a proof of continuum convergence.'},
] as const;
