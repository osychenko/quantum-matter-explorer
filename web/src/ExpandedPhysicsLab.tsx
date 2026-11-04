/** Patch 38: two bounded live VMC experiments. Numerical outputs are API data.
 * The plotted oscillator reference curve follows a known exact trial integral;
 * the structure curves are sampled using qmc.observables, not fabricated.
 */
import {useEffect, useRef, useState} from 'react';

export const OSC = 'harmonic-oscillator-vmc-v1';
export const STRUCT = 'softcore-structure-vmc-v1';
export type Curve = {
  id: string; label: string; x: number[]; values: number[]; stderr: number[];
  ideal_reference: number; x_units: string; indices?: number[][];
};
export type LiveResult = {
  experiment_id: string; execution_mode: string; preset_status: string;
  observables: Array<{id: string; value: number; uncertainty: number|null;
    exact_ground_energy?: number; analytic_trial_expectation?: number}>;
  parameters: Record<string, unknown>; provenance: Record<string, unknown>;
  distributions?: Curve[]; cautions: string[]; fingerprint_sha256: string;
};

const format = (n: number) => n.toFixed(4);

function OscillatorPlot({alpha, result}: {alpha: number; result: LiveResult}) {
  const obs = result.observables[0];
  const exact = obs.exact_ground_energy ?? 0;
  const referenceCurve = result.distributions?.find(d=>d.id==='gaussian_variational_curve');
  if (!referenceCurve) return <p className="error">No reference curve supplied by the engine.</p>;
  const curve = referenceCurve.x.map((a,i)=>({a,e:referenceCurve.values[i]}));
  const max = Math.max(...curve.map(c => c.e), exact, obs.value + (obs.uncertainty ?? 0)) + .07;
  const min = Math.min(exact, ...curve.map(c => c.e), obs.value-(obs.uncertainty ?? 0)) - .07;
  const X = (a: number) => 50 + (a-.5)/1.3*490;
  const Y = (e: number) => 230 - (e-min)/(max-min)*190;
  const poly = curve.map((c,i)=>`${i?'L':'M'} ${X(c.a).toFixed(2)} ${Y(c.e).toFixed(2)}`).join(' ');
  return <figure className="physics-plot"><svg viewBox="0 0 590 282" role="img" aria-label="VMC estimate with SEM, exact Gaussian trial expectation as a curve, exact ground energy as a dashed line">
    <line x1="50" x2="540" y1={Y(exact)} y2={Y(exact)} stroke="#d4b177" strokeDasharray="6 5" strokeWidth="2"/>
    <path d={poly} fill="none" stroke="#6eefe1" strokeWidth="3"/>
    <line x1={X(alpha)} x2={X(alpha)} y1={Y(obs.value - (obs.uncertainty ?? 0))} y2={Y(obs.value+(obs.uncertainty ?? 0))} stroke="#ff9b8f" strokeWidth="3"/>
    <circle cx={X(alpha)} cy={Y(obs.value)} r="6" fill="#ff9b8f"/>
    <text x="50" y="261" fill="#9db2c4" fontSize="13">α=0.5</text><text x="540" y="261" textAnchor="end" fill="#9db2c4" fontSize="13">α=1.8</text>
    <text x="50" y="19" fill="#bcd0df" fontSize="12">Energy</text>
  </svg><figcaption><span className="key-gaussian">Gaussian trial expectation</span> · <span className="key-exact">Exact ground energy</span> · <span className="key-vmc">Live VMC ± SEM</span></figcaption></figure>;
}

function SampleCurve({curve}: {curve: Curve; key?: string}) {
  const xs = curve.x, ys = curve.values, es = curve.stderr;
  const ymax = Math.max(1.2, ...ys.map((v,i)=>v + es[i])) * 1.12;
  const maxX = Math.max(...xs), minX = Math.min(...xs);
  const X = (i: number) => curve.indices ? 53 + i/Math.max(1,xs.length-1)*480
    : 53 + (xs[i]-minX)/Math.max(.001,maxX-minX)*480;
  const Y = (v: number) => 229 - v/ymax*199;
  const line = xs.map((_,i)=>`${i?'L':'M'} ${X(i).toFixed(2)} ${Y(ys[i]).toFixed(2)}`).join(' ');
  return <figure className="physics-plot"><h3>{curve.label}</h3><svg viewBox="0 0 590 280" role="img" aria-label={`${curve.label}, pointwise error bars, uniform ideal reference 1`}>
    <line x1="53" x2="533" y1={Y(1)} y2={Y(1)} stroke="#d4b177" strokeDasharray="5 4" strokeWidth="2"/>
    {!curve.indices && <path d={line} fill="none" stroke="#6eefe1" strokeWidth="3"/>}
    {xs.map((v,i)=><g key={i}>
      <line x1={X(i)} x2={X(i)} y1={Y(ys[i]-es[i])} y2={Y(ys[i]+es[i])} stroke="#ff9b8f" strokeWidth="2"/>
      <circle cx={X(i)} cy={Y(ys[i])} r="3.4" fill="#6eefe1"/>
      {(curve.indices || i===0 || i===xs.length-1) && <text x={X(i)} y="252" textAnchor="middle" fill="#9db2c4" fontSize={curve.indices?11:12}>{curve.indices ? curve.indices[i].join(',') : v.toFixed(2)}</text>}
    </g>)}
    <text x="53" y="19" fill="#bcd0df" fontSize="12">{curve.id==='pair_correlation'?'g(r)':'S(k)'}</text>
  </svg><figcaption>{curve.indices ? 'Reciprocal index (nₓ,nᵧ,n_z), each direction separate' : 'Pair separation r'} · turquoise: VMC mean · coral: pointwise SEM · dashed: uniform reference 1</figcaption></figure>;
}

export default function ExpandedPhysicsLab({presetId, onResult, onSave}: {
  presetId: string; onResult: (result: LiveResult|null)=>void; onSave: (result: LiveResult)=>void;
}) {
  const [alpha,setAlpha] = useState(1.0), [omega,setOmega]=useState(1.0), [jastrow,setJastrow]=useState(.6);
  const [result,setResult] = useState<LiveResult|null>(null);
  const [running,setRunning] = useState(false), [error,setError]=useState('');
  const revision = useRef(0);
  const osc = presetId === OSC;
  useEffect(() => {revision.current += 1;setResult(null);onResult(null);setError('');return () => {revision.current+=1;};},[presetId]); // reset on model switch
  function change(set: (n:number)=>void, n:number) {revision.current+=1;set(n);setResult(null);onResult(null);}
  async function run() {
    if (running) return;
    const current = ++revision.current;
    setRunning(true);setError('');setResult(null);onResult(null);
    try {
      const parameters = osc ? {trial_alpha: alpha, omega} : {jastrow_amplitude: jastrow};
      const response = await fetch(`/api/v1/experiments/${presetId}/run`, {method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({parameters})});
      const data = await response.json();
      if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:`HTTP ${response.status}`);
      if (current === revision.current) {setResult(data as LiveResult);onResult(data as LiveResult);}
    } catch(e) {if(current===revision.current)setError(e instanceof Error?e.message:String(e));}
    finally {setRunning(false);}
  }
  return <section className="expanded-lab" aria-label="Live variational physics laboratory">
    <div className="content-grid">
      <section className="panel controls">
        <div className="panel-heading"><span className="eyebrow">01 / DESIGN THE EXPERIMENT</span><h2>{osc?'Harmonic oscillator':'Interacting boson structure'}</h2></div>
        <span className="mode-tag analytic">LIVE BOUNDED VMC · RESEARCH-LEVEL ILLUSTRATION</span>
        {osc ? <>
          <label htmlFor="alpha">Gaussian trial width α</label><output>{alpha.toFixed(2)}</output>
          <input id="alpha" type="range" min="0.5" max="1.8" step="0.05" value={alpha} onChange={(e: {target:{value:string}})=>change(setAlpha,Number(e.target.value))}/>
          <label htmlFor="omega">Trap frequency ω</label><output>{omega.toFixed(2)}</output>
          <input id="omega" type="range" min="0.7" max="1.5" step="0.05" value={omega} onChange={(e: {target:{value:string}})=>change(setOmega,Number(e.target.value))}/>
          <p className="hint">At α=ω, the Gaussian is the exact ground-state wavefunction. Changing α modifies our trial state; changing ω modifies the physical trap.</p>
        </> : <>
          <label htmlFor="jastrow">Jastrow trial amplitude a</label><output>{jastrow.toFixed(2)}</output>
          <input id="jastrow" type="range" min="0" max="1.2" step="0.1" value={jastrow} onChange={(e: {target:{value:string}})=>change(setJastrow,Number(e.target.value))}/>
          <p className="hint">The Hamiltonian stays fixed: N=6, L=5, V₀=3, cutoff=1.2. The control changes only our symmetric trial wavefunction.</p>
        </>}
        <button type="button" className="primary" disabled={running} onClick={()=>void run()}>{running?'Calculating…':'Run Monte Carlo experiment'}</button>
        <p className="hint">Three independent seeded runs. Short fixed CPU workload, intended for scientific exploration, not precision phase determination.</p>
        {error && <div className="error" role="alert">{error}</div>}
      </section>
      <section className="panel result-panel" aria-live="polite">
        <div className="panel-heading"><span className="eyebrow">02 / READ THE EVIDENCE</span><h2>{osc?'Ground-state energy':'Pair and Fourier correlations'}</h2></div>
        {!result ? <p className="hint">{running?'The existing VMC engine is sampling configurations.':'Choose the parameters and run the experiment to see measured data.'}</p> : <>
          <div className="result-stat"><span>{osc?'VMC ENERGY':'VARIATIONAL ENERGY PER PARTICLE'}</span><div className="big-number">{format(result.observables[0].value)}</div><div className="result-detail">± {format(result.observables[0].uncertainty ?? 0)} independent-replica SEM</div></div>
          {osc && <div className="science-comparison-result"><div><span>Gaussian trial expectation (exact integral)</span><strong>{format(result.observables[0].analytic_trial_expectation ?? 0)}</strong></div><div><span>True harmonic ground energy</span><strong>{format(result.observables[0].exact_ground_energy ?? 0)}</strong></div><small>Trial bias is the difference between these analytical reference values, not the VMC sampling error.</small></div>}
          <div className="result-actions"><button className="primary" onClick={()=>onSave(result)}>＋ Save to notebook</button><button className="outline" onClick={()=>{
            const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
            const url=URL.createObjectURL(blob), a=document.createElement('a');a.href=url;a.download=`${presetId}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
          }}>↓ Export JSON</button></div>
        </>}
      </section>
    </div>
    {result && <section className="panel expanded-plots"><div className="panel-heading"><span className="eyebrow">03 / COMPARE</span><h2>{osc?'A variational energy landscape':'Spatial and reciprocal-space structure'}</h2></div>
      {osc?<OscillatorPlot alpha={alpha} result={result}/>:<div className="expanded-curve-grid">{result.distributions?.map(curve=><SampleCurve key={curve.id} curve={curve}/>)}</div>}
      <p className="science-caveat">{osc?'The solid curve integrates the Gaussian trial analytically. The dot is a genuine stochastic estimate; Monte Carlo noise can move it slightly below the theoretical ground energy.': 'These observables describe the selected variational distribution, not a proven finite-temperature phase. Individual points and orientations may be noisy.'}</p>
      <details><summary>Full scientific limitations and reproducibility</summary><ul>{result.cautions.map((c,i)=><li key={i}>{c}</li>)}</ul><p>Result fingerprint: <code>{result.fingerprint_sha256}</code></p></details>
    </section>}
  </section>;
}
