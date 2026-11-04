/** Patch 37: a problem-first map of the educational scope and transparent results.
 * Static scientific prose comes from /api/v1/science-map; all numerical values
 * shown in the inspector come from the actual experiment response.
 */
import {useState} from 'react';

export type ScienceProblem = {
  id: string; title: string; area: string; status: 'interactive'|'recorded'|'research_only';
  preset_id: string|null; question: string; method: string; observable: string;
  visual: string; level: string;
};
export type ScienceMethod = {
  id: string; title: string; availability: string; principle: string;
  procedure: string[]; limitation: string;
};
export type ParameterSpec = {path: string; label: string; category: 'physical'|'variational'|'numerical'|'statistical'|'observable'; explanation: string};
export type ScienceProfile = {
  problem_id: string; method_id: string; question: string; model: string;
  observable: string; interpretation: string; source_kind: string;
  parameters: ParameterSpec[];
  comparison: {title: string; target: string; held_fixed: string; changed: string; cautions: string};
  theory_ids: string[];
};
export type TheoryTopic = {id: string; title: string; concept: string; physics: string; equation: string; interpretation: string};
export type ScienceMapData = {
  schema_version: string; problems: ScienceProblem[]; methods: ScienceMethod[];
  profiles: Record<string, ScienceProfile>; theory: TheoryTopic[];
};
type Evidence = {
  experiment_id: string; parameters: Record<string, unknown>;
  provenance: Record<string, unknown>;
  execution_mode: string;
  observables: Array<{value: number; uncertainty: number|null; spectral_reference_k2?: number}>;
};

const SECTIONS = ['Question', 'Model', 'Method', 'Measurement', 'Comparison'];
const CATEGORIES = [
  {id: 'physical', label: 'Physical Hamiltonian', note: 'What system are we studying?'},
  {id: 'variational', label: 'Variational choice', note: 'What trial wavefunction are we using? This does not necessarily alter the Hamiltonian.'},
  {id: 'numerical', label: 'Numerical method', note: 'How is the calculation approximated?'},
  {id: 'statistical', label: 'Statistical evidence', note: 'How precisely is it measured?'},
  {id: 'observable', label: 'Measurement definition', note: 'What quantity is reported?'},
] as const;

function readPath(result: Evidence, path: string): unknown {
  let value: unknown = result;
  for (const key of path.split('.')) {
    if (typeof value !== 'object' || value === null) return undefined;
    if (Array.isArray(value)) {
      const index = Number(key);
      if (!Number.isInteger(index) || index < 0) return undefined;
      value = value[index];
    } else {
      value = (value as Record<string, unknown>)[key];
    }
  }
  return value;
}
function formatValue(value: unknown): string {
  if (typeof value === 'number') return Number.isFinite(value) ? Number(value.toPrecision(7)).toString() : 'Unavailable';
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(formatValue).join(', ');
  return value === undefined || value === null ? 'Not reported' : String(value);
}
function methodSummary(data: ScienceMapData, id: string): ScienceMethod|undefined {
  return data.methods.find(method => method.id === id);
}

export function ScienceBrief({profile, onAtlas}: {profile?: ScienceProfile; onAtlas: () => void}) {
  if (!profile) return <section className="panel science-placeholder" role="status">Loading the science behind this experiment…</section>;
  const items = [profile.question, profile.model, profile.source_kind, profile.observable, profile.comparison.target];
  return <section className="panel science-brief" aria-label="The five questions behind this experiment">
    <div className="science-heading"><div><span className="eyebrow">FIRST / ASK THE PHYSICS QUESTION</span><h2>What are we actually investigating?</h2></div>
      <button className="outline" type="button" onClick={onAtlas}>Explore the Science Map ↗</button></div>
    <div className="science-steps">
      {SECTIONS.map((label, index)=><div className="science-step" key={label}>
        <span className="science-step-index">0{index+1} · {label}</span><p>{items[index]}</p>
      </div>)}
    </div>
  </section>;
}

export function ScienceInspector({data, profile, result, matchedIdeal, sweepVariable}: {
  data: ScienceMapData|null; profile?: ScienceProfile; result: Evidence|null; matchedIdeal: Evidence|null;
  sweepVariable: 'beta'|'cell_length';
}) {
  if (!profile || !data) return null;
  const method = methodSummary(data, profile.method_id);
  const reference = result?.observables[0]?.spectral_reference_k2;
  const measured = result?.observables[0]?.value;
  const sem = result?.observables[0]?.uncertainty;
  const z = reference !== undefined && measured !== undefined && sem != null && sem > 0
    ? Math.abs(measured-reference)/sem : null;
  return <section className="science-inspector" aria-label="Physics and calculation details">
    <div className="science-heading"><div><span className="eyebrow">05 / REPRODUCE & COMPARE</span><h2>Inside the calculation</h2>
      <p className="hint">Explanations are curated; parameter values below are read from this experiment’s JSON response.</p></div>
      <span className="mode-tag">{result?.execution_mode === 'recorded_result' ? 'RECORDED EVIDENCE' : result?.execution_mode === 'bounded_live_vmc' ? 'LIVE VARIATIONAL MC' : 'ANALYTICAL CALCULATION'}</span></div>
    <div className="science-inspector-grid">
      <div className="panel science-detail">
        <span className="eyebrow">METHOD</span><h3>{method?.title ?? 'Method details'}</h3>
        <p>{method?.principle}</p>
        <ol className="science-method-steps">{method?.procedure.map((step,i)=><li key={i}>{step}</li>)}</ol>
        {method && <p className="science-caveat"><strong>What this does not establish:</strong> {method.limitation}</p>}
      </div>
      <div className="panel science-detail">
        <span className="eyebrow">COMPARISON PROTOCOL</span><h3>{profile.comparison.title}</h3>
        <dl className="science-comparison-list"><dt>Compared</dt><dd>{profile.comparison.target}</dd>
          <dt>Held fixed</dt><dd>{profile.comparison.held_fixed}</dd>
          <dt>Changed</dt><dd>{profile.comparison.changed}</dd></dl>
        {result?.execution_mode === 'analytic_control' && <p className="science-caveat"><strong>Current graph:</strong> we vary {sweepVariable === 'beta' ? 'β while holding L fixed' : 'L while holding β fixed'}, using the same ideal-gas formula.</p>}
        {z !== null && <div className="science-comparison-result" role="note">
          <div><span>Recorded worm estimate</span><strong>{measured?.toFixed(5)} ± {sem?.toFixed(5)}</strong></div>
          <div><span>Independent K=2 spectral calculation</span><strong>{reference?.toFixed(5)}</strong></div>
          <div><span>Difference / replica SEM</span><strong>{z.toFixed(2)} σ</strong></div>
          <small>This is a consistency check, not a continuum convergence certificate.</small>
        </div>}
        {result?.execution_mode === 'recorded_result' && matchedIdeal && <div className="science-comparison-result control-comparison" role="note">
          <h4>Different Hamiltonians: ideal vs interacting</h4>
          <div><span>Ideal, calculated at the matching β and L</span><strong>{matchedIdeal.observables[0].value.toFixed(5)}</strong></div>
          <div><span>Interacting, recorded M=8 worm</span><strong>{result.observables[0].value.toFixed(5)}</strong></div>
          <small>Matched: β, L, D, background N and k=0. Changed: repulsive potential; also a different solver and finite timestep are used. This comparison alone cannot isolate an interaction effect.</small>
        </div>}
        <p className="science-caveat">{profile.comparison.cautions}</p>
      </div>
    </div>
    <section className="panel science-parameters">
      <div className="science-heading"><div><span className="eyebrow">PARAMETERS</span><h3>What went into this result?</h3></div>
        <span className="mode-tag">{result ? 'ACTUAL RUN VALUES' : 'WAITING FOR RESULT'}</span></div>
      <div className="science-param-grid">{CATEGORIES.map(category => {
        const parameters = profile.parameters.filter(param => param.category === category.id);
        return <div className="science-param-group" key={category.id}>
          <h4>{category.label}</h4><p>{category.note}</p>
          <dl>{parameters.length ? parameters.map(param => <div key={param.path} className="science-param">
            <dt>{param.label} <span title={param.explanation} aria-label={param.explanation}>ⓘ</span></dt>
            <dd>{result ? formatValue(readPath(result, param.path)) : '…'}</dd>
            <small>{param.explanation}</small>
          </div>) : <div className="science-empty">No adjustable statistical parameter in this calculation.</div>}</dl>
        </div>;
      })}</div>
    </section>
  </section>;
}

export default function ScienceMapPage({data, onOpenExperiment, onOpenTheory}: {
  data: ScienceMapData|null; onOpenExperiment: (presetId: string)=>void;
  onOpenTheory: (topicId: string)=>void;
}) {
  const [level, setLevel] = useState<'concept'|'physics'|'equation'>('physics');
  if (!data) return <div className="panel" role="status">Loading the Science Map…</div>;
  return <div className="science-page">
    <header className="hero"><span className="eyebrow">THE SCIENCE MAP</span><h1>Begin with a question.</h1>
      <p>Quantum Matter Explorer connects physical questions to models, computational methods, observables, and trustworthy comparisons. Only curated experiments can be run in the app.</p></header>
    <section aria-labelledby="science-problems">
      <div className="science-heading"><div><span className="eyebrow">INVESTIGATIONS</span><h2 id="science-problems">Choose a problem to investigate</h2></div><span className="mode-tag">{data.problems.filter(problem => problem.preset_id).length} CURATED · {data.problems.filter(problem => !problem.preset_id).length} RESEARCH PATHS</span></div>
      <div className="science-problem-grid">{data.problems.map((problem,i)=><article key={problem.id} className="panel science-problem-card">
        <div className="science-card-top"><span className="science-problem-n">{String(i+1).padStart(2,'0')}</span>
          <span className={`science-availability ${problem.status}`}>{problem.status === 'interactive' ? (problem.method.includes('VMC') ? 'Live variational MC' : 'Live analytical') : problem.status === 'recorded' ? 'Recorded benchmark' : 'Research only'}</span></div>
        <div className={`science-problem-visual ${problem.visual}`} aria-hidden="true"><span>{problem.visual === 'momentum-levels' ? 'n = 0    1    2' : problem.visual === 'worldlines' ? 'τ →  0 ── β/2 ── β' : problem.visual === 'energy-landscape' ? 'E[Ψ] ≥ E₀' : problem.visual === 'particle-cloud' ? 'g(r)  ·  S(k)' : 'P ↔ exchange'}</span></div>
        <span className="eyebrow">{problem.area} · {problem.level}</span><h3>{problem.title}</h3><p>{problem.question}</p>
        <dl className="science-problem-facts"><dt>Method</dt><dd>{problem.method}</dd><dt>Measure</dt><dd>{problem.observable}</dd></dl>
        {problem.preset_id ? <button className="primary" type="button" onClick={()=>onOpenExperiment(problem.preset_id!)}>{problem.status === 'recorded' ? 'Inspect recorded experiment' : problem.method.includes('VMC') ? 'Open variational laboratory' : 'Run analytical experiment'} →</button>
          : <div className="science-coming" role="note">Research-only module: no public educational preset yet.</div>}
      </article>)}</div>
    </section>
    <section className="science-method-atlas" aria-labelledby="method-atlas-title"><div className="science-heading"><div><span className="eyebrow">HOW DO WE KNOW?</span><h2 id="method-atlas-title">Methods inside the engine</h2></div></div>
      <div className="science-method-list">{data.methods.map(method=><details key={method.id} className="panel science-method-card">
        <summary><strong>{method.title}</strong><span className="science-availability">{method.availability.replaceAll('_',' ')}</span></summary>
        <p>{method.principle}</p><ol>{method.procedure.map((step,i)=><li key={i}>{step}</li>)}</ol><p className="science-caveat"><strong>Limitation:</strong> {method.limitation}</p>
      </details>)}</div>
    </section>
    <section className="science-theory-atlas" aria-labelledby="theory-atlas-title"><div className="science-heading"><div><span className="eyebrow">PHYSICS NOTES</span><h2 id="theory-atlas-title">The theory, at your level</h2></div>
      <button type="button" className="outline" onClick={()=>onOpenTheory('imaginary-time-projection')}>Open interactive Theory Atlas →</button>
      <div className="science-depth" role="group" aria-label="Explanation depth">
        {(['concept','physics','equation'] as const).map(v=><button type="button" key={v} className={level===v?'active':''} aria-pressed={level===v} onClick={()=>setLevel(v)}>{v==='equation'?'Mathematics':v==='physics'?'Physics':'Concept'}</button>)}
      </div></div>
      <div className="science-theory-grid">{data.theory.map(topic=><article className="panel science-theory-card" key={topic.id}>
        <h3>{topic.title}</h3>
        <button type="button" className="outline" onClick={()=>onOpenTheory(topic.id)}>Read and explore →</button>
        {level === 'equation' ? <><pre className="science-equation">{topic.equation}</pre><p>{topic.interpretation}</p></> : <p>{topic[level]}</p>}
      </article>)}</div>
    </section>
  </div>;
}
