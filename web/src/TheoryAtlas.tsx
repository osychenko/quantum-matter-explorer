/** Patch 40: interactive Theory Atlas.
 * Text and topic↔experiment links come from /api/v1/science-map.
 * Three clearly labelled analytical demonstrations NEVER run the QMC solver.
 */
import {useState, type ReactNode} from 'react';
import type {ScienceMapData} from './ScienceMap';
import {oscillatorExactEnergy, oscillatorTrialEnergy, oscillatorLocalEnergyStd,
  projectedExcitedWeight, twoBosonExchangeTermFraction} from './theoryDemos';

type Depth = 'concept' | 'physics' | 'equation';
const DEMOS: Record<string,string> = {
  'imaginary-time-projection': 'Two-level projection',
  'variational-principle': 'Gaussian variational energy',
  'quantum-statistics': 'Two-boson exchange terms',
};

function Range({id,label,value,min,max,step,onChange}: {
  id:string;label:string;value:number;min:number;max:number;step:number;
  onChange:(value:number)=>void;
}) {
  return <div className="theory-range">
    <div><label htmlFor={id}>{label}</label><output htmlFor={id}>{value.toFixed(2)}</output></div>
    <input id={id} type="range" min={min} max={max} step={step} value={value}
      onChange={event=>onChange(Number(event.target.value))}/>
  </div>;
}

function Axes({children,label}: {children:ReactNode;label:string}) {
  return <svg className="theory-plot" viewBox="0 0 620 250" role="img" aria-label={label}>
    <line x1="48" y1="206" x2="584" y2="206" stroke="#91a9ba"/>
    <line x1="48" y1="22" x2="48" y2="206" stroke="#91a9ba"/>
    {children}
  </svg>;
}

function ProjectionDemo() {
  const [gap,setGap]=useState(1);
  const [initial,setInitial]=useState(.7);
  const [tau,setTau]=useState(1.5);
  const x=(v:number)=>48+536*v/5;
  const y=(v:number)=>206-166*v;
  const current=projectedExcitedWeight(tau,gap,initial);
  const curve=Array.from({length:101},(_,i)=>{
    const t=i/20;
    return `${i?'L':'M'} ${x(t).toFixed(2)} ${y(projectedExcitedWeight(t,gap,initial)).toFixed(2)}`;
  }).join(' ');
  return <section className="panel theory-demo" aria-labelledby="theory-projection-title">
    <span className="eyebrow">EXACT TWO-LEVEL TOY MODEL · NOT MONTE CARLO</span>
    <h3 id="theory-projection-title">Imaginary-time projection filters excited states</h3>
    <p>Starting from a superposition of a ground state and an excited state, imaginary-time evolution suppresses the excited amplitude faster. This works only if the initial state has some overlap with the ground state.</p>
    <div className="theory-controls">
      <Range id="theory-gap" label="Excited-state energy gap Δ" min={.3} max={2.5} step={.1} value={gap} onChange={setGap}/>
      <Range id="theory-initial" label="Initial excited weight" min={.1} max={.9} step={.05} value={initial} onChange={setInitial}/>
      <Range id="theory-tau" label="Imaginary time τ" min={0} max={5} step={.1} value={tau} onChange={setTau}/>
    </div>
    <figure className="theory-figure">
      <Axes label="Analytical excited-state weight versus imaginary time">
        {[0,.25,.5,.75,1].map(v=><g key={v}>
          <line x1="48" x2="584" y1={y(v)} y2={y(v)} stroke="#38546b" strokeDasharray="3 6"/>
          <text x="40" y={y(v)+4} textAnchor="end" fill="#c1d0db" fontSize="12">{v.toFixed(2)}</text>
        </g>)}
        <path d={curve} stroke="#74e9dc" strokeWidth="3" fill="none"/>
        <line x1={x(tau)} x2={x(tau)} y1="24" y2="206" stroke="#f1c287" strokeDasharray="5 5"/>
        <circle cx={x(tau)} cy={y(current)} r="6" fill="#f1c287"/>
        <text x="48" y="231" fill="#c1d0db" fontSize="12">τ = 0</text>
        <text x="584" y="231" textAnchor="end" fill="#c1d0db" fontSize="12">τ = 5</text>
      </Axes>
      <figcaption>Normalized excited weight: p(τ) = p(0) exp(−2Δτ) / [1 − p(0) + p(0) exp(−2Δτ)].</figcaption>
    </figure>
    <div className="theory-results">
      <div><span>Excited weight</span><strong>{(100*current).toFixed(2)}%</strong></div>
      <div><span>Ground weight</span><strong>{(100*(1-current)).toFixed(2)}%</strong></div>
    </div>
    <p className="science-caveat"><strong>Scope:</strong> This is a ground-state projection demonstration, not a finite-temperature worm simulation. The worm benchmark samples imaginary-time thermal correlations over a period β, not this normalized two-level projector. Imaginary time is not real time.</p>
  </section>;
}

function VariationalDemo() {
  const [alpha,setAlpha]=useState(1.25);
  const [omega,setOmega]=useState(1);
  const ground=oscillatorExactEnergy(omega);
  const trial=oscillatorTrialEnergy(alpha,omega);
  const sd=oscillatorLocalEnergyStd(alpha,omega);
  const top=1.08*Math.max(ground,oscillatorTrialEnergy(.4,omega),oscillatorTrialEnergy(2.6,omega));
  const x=(a:number)=>48+536*(a-.4)/2.2;
  const y=(energy:number)=>206-176*energy/top;
  const curve=Array.from({length:101},(_,i)=>{
    const a=.4+2.2*i/100;
    return `${i?'L':'M'} ${x(a).toFixed(2)} ${y(oscillatorTrialEnergy(a,omega)).toFixed(2)}`;
  }).join(' ');
  return <section className="panel theory-demo" aria-labelledby="theory-vmc-title">
    <span className="eyebrow">EXACT INTEGRALS · NOT SAMPLED VMC</span>
    <h3 id="theory-vmc-title">Why the trial state cannot beat the true ground state</h3>
    <p>Choose a Gaussian trial function for a 3D harmonic oscillator. Changing its width α changes the approximation. Changing trap frequency ω instead changes the physical Hamiltonian.</p>
    <div className="theory-controls">
      <Range id="theory-alpha" label="Gaussian width α · variational" min={.4} max={2.6} step={.05} value={alpha} onChange={setAlpha}/>
      <Range id="theory-omega" label="Frequency ω · physical" min={.6} max={2} step={.1} value={omega} onChange={setOmega}/>
    </div>
    <figure className="theory-figure">
      <Axes label="Exact Gaussian trial-state energy versus alpha and the exact ground energy">
        <line x1="48" x2="584" y1={y(ground)} y2={y(ground)} stroke="#eab879" strokeWidth="2" strokeDasharray="5 5"/>
        <path d={curve} stroke="#74e9dc" strokeWidth="3" fill="none"/>
        <circle cx={x(alpha)} cy={y(trial)} r="6" fill="#f0f7fc" stroke="#39c8b5" strokeWidth="3"/>
        <text x="56" y={Math.max(18,y(ground)-9)} fill="#eab879" fontSize="12">Exact E₀</text>
        <text x="48" y="231" fill="#c1d0db" fontSize="12">α = 0.4</text>
        <text x="584" y="231" textAnchor="end" fill="#c1d0db" fontSize="12">α = 2.6</text>
      </Axes>
      <figcaption>H = −∇²/2 + ω²r²/2; ψα ∝ exp(−αr²/2). E[ψα] = 3(α + ω²/α)/4, E₀ = 3ω/2.</figcaption>
    </figure>
    <div className="theory-results">
      <div><span>Exact ground E₀</span><strong>{ground.toFixed(4)}</strong></div>
      <div><span>Exact trial E[ψα]</span><strong>{trial.toFixed(4)}</strong></div>
      <div><span>Variational excess</span><strong>{Math.max(0,trial-ground).toFixed(4)}</strong></div>
      <div><span>Local-energy SD · not SEM</span><strong>{sd.toFixed(4)}</strong></div>
    </div>
    <p className="science-caveat"><strong>Scope:</strong> The green line is an exact expectation, not a stochastic VMC trace. When α = ω, the trial state is exact and the local-energy variance is zero. A finite VMC estimate can fluctuate below E₀; only the exact expectation obeys the upper bound.</p>
  </section>;
}

function ExchangeDemo() {
  const [distance,setDistance]=useState(1.3);
  const [beta,setBeta]=useState(2);
  const [exchange,setExchange]=useState(true);
  const fraction=twoBosonExchangeTermFraction(distance,beta);
  return <section className="panel theory-demo" aria-labelledby="theory-exchange-title">
    <span className="eyebrow">TWO NONINTERACTING 1D BOSONS · SCHEMATIC PATHS</span>
    <h3 id="theory-exchange-title">Why identical bosons permit exchanged endpoints</h3>
    <p>The symmetrized two-particle density matrix adds an identity-permutation term and an exchanged-endpoints term. In this free-particle example, their relative weight depends on endpoint separation and imaginary-time extent.</p>
    <div className="theory-controls">
      <Range id="theory-distance" label="Endpoint separation d" min={0} max={3} step={.1} value={distance} onChange={setDistance}/>
      <Range id="theory-beta" label="Imaginary-time β · D = 0.5 fixed" min={.4} max={4} step={.1} value={beta} onChange={setBeta}/>
    </div>
    <div className="theory-mode" role="group" aria-label="Endpoint permutation">
      <button type="button" className={!exchange?'active':''} aria-pressed={!exchange} onClick={()=>setExchange(false)}>Identity pairing</button>
      <button type="button" className={exchange?'active':''} aria-pressed={exchange} onClick={()=>setExchange(true)}>Exchanged pairing</button>
    </div>
    <figure className="theory-figure">
      <svg className="theory-plot" viewBox="0 0 620 250" role="img" aria-label={exchange?'Schematic swapped endpoint connections':'Schematic identity endpoint connections'}>
        <line x1="110" y1="45" x2="510" y2="45" stroke="#668499" strokeDasharray="5 5"/>
        <line x1="110" y1="206" x2="510" y2="206" stroke="#668499" strokeDasharray="5 5"/>
        <text x="98" y="49" fill="#c0d0dd" textAnchor="end" fontSize="13">τ = β</text>
        <text x="98" y="210" fill="#c0d0dd" textAnchor="end" fontSize="13">τ = 0</text>
        <path d={exchange?'M 225 206 C 225 125 395 125 395 45':'M 225 206 C 205 140 245 110 225 45'}
          stroke="#74e9dc" strokeWidth="4" fill="none"/>
        <path d={exchange?'M 395 206 C 395 125 225 125 225 45':'M 395 206 C 415 140 375 110 395 45'}
          stroke="#f1c287" strokeWidth="4" fill="none"/>
        {[225,395].map(cx=><g key={cx}><circle cx={cx} cy="45" r="6" fill="#eff5f8"/>
          <circle cx={cx} cy="206" r="6" fill="#eff5f8"/></g>)}
        <text x="225" y="233" fill="#c0d0dd" textAnchor="middle" fontSize="12">x₁</text>
        <text x="395" y="233" fill="#c0d0dd" textAnchor="middle" fontSize="12">x₂</text>
      </svg>
      <figcaption>These smooth curves illustrate endpoint pairing, not a sampled path, a real-time orbit or measured bosonic exchange.</figcaption>
    </figure>
    <div className="theory-results">
      <div><span>Exchange / identity term</span><strong>{(fraction/(1-fraction)).toFixed(4)}</strong></div>
      <div><span>Exchange share of two terms</span><strong>{(100*fraction).toFixed(2)}%</strong></div>
    </div>
    <p className="science-caveat"><strong>Not a swap probability:</strong> These are relative contributions to the diagonal two-free-boson density matrix at fixed endpoints. They do not count exchange events in an interacting many-body worm calculation. The exchange/identity ratio is exp[−d²/(2Dβ)].</p>
  </section>;
}

export default function TheoryAtlas({data,selectedTopicId,onSelectTopic,onOpenExperiment}: {
  data:ScienceMapData|null; selectedTopicId:string;
  onSelectTopic:(id:string)=>void; onOpenExperiment:(presetId:string)=>void;
}) {
  const [depth,setDepth]=useState<Depth>('concept');
  if(!data) return <section className="panel" role="status">Loading the Theory Atlas…</section>;
  const topic=data.theory.find(t=>t.id===selectedTopicId)??data.theory[0];
  if(!topic) return <section className="panel" role="status">No theory notes are available.</section>;
  const related=Object.entries(data.profiles).filter(([,profile])=>profile.theory_ids.includes(topic.id));
  return <div className="theory-atlas-page">
    <header className="hero"><span className="eyebrow">PATCH 40 / QUANTUM THEORY ATLAS</span>
      <h1>From intuition to equations.</h1>
      <p>Read the concept, inspect the mathematics, then manipulate exact toy models. The analytical demonstrations below do not represent runs of the research QMC engine.</p>
    </header>
    <div className="theory-atlas-layout">
      <nav className="panel theory-topic-nav" aria-label="Theory topics"><span className="eyebrow">CHOOSE A TOPIC</span>
        {data.theory.map(item=><button type="button" key={item.id}
          className={topic.id===item.id?'active':''} aria-current={topic.id===item.id?'true':undefined}
          onClick={()=>onSelectTopic(item.id)}><span>{item.title}</span>
          {DEMOS[item.id]&&<small>INTERACTIVE</small>}</button>)}
      </nav>
      <div className="theory-atlas-content">
        <section className="panel theory-reading" aria-labelledby="theory-current-title">
          <span className="eyebrow">PHYSICS / {topic.id.replaceAll('-',' ').toUpperCase()}</span>
          <h2 id="theory-current-title">{topic.title}</h2>
          <div className="science-depth" role="group" aria-label="Explanation depth">
            {(['concept','physics','equation'] as const).map(v=><button key={v} type="button"
              className={depth===v?'active':''} aria-pressed={depth===v} onClick={()=>setDepth(v)}>
              {v==='concept'?'Concept':v==='physics'?'Physics':'Mathematics'}</button>)}
          </div>
          {depth==='equation'?<><pre className="science-equation">{topic.equation}</pre>
            <p>{topic.interpretation}</p></>:<><p>{topic[depth]}</p><p className="theory-note">{topic.interpretation}</p></>}
        </section>
        {topic.id==='imaginary-time-projection'?<ProjectionDemo/>:
          topic.id==='variational-principle'?<VariationalDemo/>:
          topic.id==='quantum-statistics'?<ExchangeDemo/>:
          <section className="panel theory-reading"><span className="eyebrow">BRIDGE TO WORKED MODELS</span>
            <h3>Explore an interactive example</h3><p>This note has no separate executable simulation. Select one of the three analytical teaching demonstrations:</p>
            <div className="theory-demo-links">{Object.entries(DEMOS).map(([id,title])=><button type="button"
              key={id} className="outline" onClick={()=>onSelectTopic(id)}>{title} →</button>)}</div>
          </section>}
        <section className="panel theory-related"><span className="eyebrow">THEORY → EXPERIMENT</span>
          <h3>Related curated laboratories</h3>
          {related.length?<div className="theory-demo-links">{related.map(([id,profile])=><button type="button"
            className="outline" key={id} onClick={()=>onOpenExperiment(id)}>{profile.question} →</button>)}</div>:
            <p>No curated numerical experiment is directly associated with this specific concept yet.</p>}
          <p className="science-caveat">Exact analytical quantities, numerical variational estimates, and recorded worm measurements must not be confused. Inspect each experiment's model, method, provenance and limitations.</p>
        </section>
      </div>
    </div>
  </div>;
}
