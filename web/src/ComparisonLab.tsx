/** Patch 39: contrasts curated by the scientific backend, not by arbitrary client pairing. */
import {useEffect, useRef, useState} from 'react';

type Parameter = {id:string; label:string; min:number; max:number; step:number; default:number};
type ComparisonPreset = {id:string;title:string;question:string;description:string;comparison_kind:string;parameters:Parameter[];live:boolean};
type Observable = {id:string;value:number;units:string;uncertainty:number|null;uncertainty_kind:string;exact_ground_energy?:number;analytic_trial_expectation?:number};
type Curve = {id:string;label:string;x:number[];values:number[];stderr:number[];indices?:number[][];x_units:string;ideal_reference?:number};
type Experiment = {experiment_id:string;execution_mode:string;preset_status:string;parameters:Record<string,unknown>;provenance:Record<string,unknown>;observables:Observable[];distributions?:Curve[];cautions:string[];fingerprint_sha256:string};
type Arm = {label:string;evidence_kind:string;experiment_id:string;experiment:Experiment;primary_observable:Observable};
type Comparison = {comparison_schema_version:string;comparison_id:string;title:string;question:string;comparison_kind:string;requested_parameters:Record<string,number>;
  arms:[Arm,Arm];controls:{held_fixed:string[];changed:string[];confounders:string[]};
  analysis:{observable_id:string;units:string;difference_b_minus_a:number;combined_sem:null;difference_z:null;inference:string};
  interpretation:{relationship:string;supported_conclusions:string[];not_supported:string[]};fingerprint_sha256:string};

type CatalogResponse = {comparisons:ComparisonPreset[]};
const FORMAT = (n:number) => Number.isFinite(n) ? n.toFixed(5) : '—';
function fileDownload(name:string, contents:string) {
  const uri = URL.createObjectURL(new Blob([contents], {type:'application/json'}));
  const a = document.createElement('a'); a.href=uri; a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(uri),1000);
}
function serialized(v:unknown):string {
  if (v===null || v===undefined) return '—';
  if (Array.isArray(v)) return v.join(', ');
  if(typeof v==='number') return Number.isFinite(v)?String(v):'—';
  return typeof v==='string'?v:JSON.stringify(v);
}
function EvidenceArm({arm}: {arm:Arm}) {
  const o=arm.primary_observable, sample=arm.experiment;
  const reference=arm.evidence_kind==='finite_basis_reference_from_recorded_experiment';
  return <article className="comparison-arm">
    <div className="comparison-arm-head"><h3>{arm.label}</h3><span className="mode-tag">{reference?'Finite-basis reference':sample.execution_mode.replaceAll('_',' ')}</span></div>
    <div className="comparison-number">{FORMAT(o.value)}</div>
    <div className="result-detail">{o.units} {o.uncertainty!==null && <> · ± {FORMAT(o.uncertainty)} independent-replica SEM</>}</div>
    {reference && <p className="hint">This reference is stored inside the historical benchmark. It is not a second Monte Carlo run, and a basis-truncation error bar is not available.</p>}
    {typeof o.analytic_trial_expectation==='number' && <div className="comparison-references">
      <div>Exact expectation of this trial state <strong>{FORMAT(o.analytic_trial_expectation)}</strong></div>
      <div>Exact ground-state energy <strong>{FORMAT(o.exact_ground_energy ?? NaN)}</strong></div>
    </div>}
    <details><summary>Parameters and provenance</summary>
      <p className="hint">Underlying preset: {sample.experiment_id}</p>
      <dl className="comparison-parameters">{Object.entries(sample.parameters).map(([k,v])=><div key={k}><dt>{k.replaceAll('_',' ')}</dt><dd>{serialized(v)}</dd></div>)}</dl>
      <h4>Evidence provenance</h4><dl className="comparison-parameters">{Object.entries(sample.provenance).map(([k,v])=><div key={k}><dt>{k.replaceAll('_',' ')}</dt><dd>{serialized(v)}</dd></div>)}</dl>
      <p className="hint">Original experiment fingerprint: <code>{sample.fingerprint_sha256}</code></p>
    </details>
  </article>;
}

function ComparisonBars({data}: {data:Comparison}) {
  const arms=data.arms, ymax=Math.max(1e-9,...arms.map(a=>a.primary_observable.value+2*(a.primary_observable.uncertainty??0)))*1.18;
  const exact = arms[0].primary_observable.exact_ground_energy;
  const sharedExact = typeof exact==='number' && exact===arms[1].primary_observable.exact_ground_energy ? exact : null;
  const baseline=188, top=18, plotH=baseline-top;
  const yy=(v:number)=>baseline-Math.max(0,v)/ymax*plotH;
  return <figure className="comparison-plot" aria-label="Side-by-side values with individual standard errors where reported">
    <svg viewBox="0 0 560 240" role="img" aria-label="Comparison of the same scalar observable in two experiments. Error whiskers are individual standard errors, not uncertainty on the difference.">
      {[0,.25,.5,.75,1].map(t=><g key={t}><line x1="55" x2="530" y1={yy(t*ymax)} y2={yy(t*ymax)} stroke="#2d4153" strokeDasharray="4 4"/><text x="47" y={yy(t*ymax)+4} fill="#acc2d4" fontSize="11" textAnchor="end">{(t*ymax).toFixed(2)}</text></g>)}
      {sharedExact!==null && <g><line x1="55" x2="530" y1={yy(sharedExact)} y2={yy(sharedExact)} stroke="#faf0d2" strokeDasharray="7 4" strokeWidth="1.8"/><text x="525" y={yy(sharedExact)-6} textAnchor="end" fill="#faf0d2" fontSize="11">Exact E₀</text></g>}
      {arms.map((a,i)=>{const o=a.primary_observable,cx=i?404:180,lo=Math.max(0,o.value-(o.uncertainty??0)),hi=o.value+(o.uncertainty??0);
        return <g key={i}><rect x={cx-48} y={yy(o.value)} width="96" height={baseline-yy(o.value)} fill={i?'#e2b67b':'#68d7d0'} opacity="0.7"/>
          {o.uncertainty!==null && <g stroke={i?'#ffe2b9':'#a7fff5'} strokeWidth="2"><line x1={cx} x2={cx} y1={yy(lo)} y2={yy(hi)}/><line x1={cx-11} x2={cx+11} y1={yy(lo)} y2={yy(lo)}/><line x1={cx-11} x2={cx+11} y1={yy(hi)} y2={yy(hi)}/></g>}
          <text x={cx} y="214" textAnchor="middle" fontSize="13" fill="#dce9f4">{i?'Study B':'Study A'}</text>
        </g>;})}
    </svg>
    <figcaption>Bars show the selected observable. Whiskers, where present, show ±1 replica SEM for each arm separately. {sharedExact!==null?'Dashed line: exact oscillator ground energy. ':''}They do not establish a confidence interval for the difference.</figcaption>
  </figure>;
}

function PairedCurve({left,right}: {left:Curve;right:Curve}) {
  if(left.x.length!==right.x.length || !left.x.every((x,i)=>Math.abs(x-right.x[i])<1e-7)) return <p className="hint">The two curves use different grids and cannot be overlaid without an explicit alignment procedure.</p>;
  const n=left.x.length, all=[...left.values,...right.values], err=[...left.stderr,...right.stderr];
  const low=Math.min(0,...all.map((v,i)=>v-err[i])), high=Math.max(1,...all.map((v,i)=>v+err[i]));
  const minY=low-.07*(high-low), maxY=high+.07*(high-low);
  const X=(i:number)=>52+(n>1?i/(n-1):.5)*494, Y=(v:number)=>194-(v-minY)/(maxY-minY)*174;
  const directional=Boolean(left.indices && right.indices);
  return <figure className="comparison-plot"><h3>{left.label}</h3>
    <svg role="img" aria-label={`${left.label}: side-by-side trial-state means and pointwise errors` } viewBox="0 0 600 255">
      {[0,.25,.5,.75,1].map(t=>{const v=minY+(maxY-minY)*t;return <g key={t}><line x1="50" x2="548" y1={Y(v)} y2={Y(v)} stroke="#2d4153" strokeDasharray="3 5"/><text x="43" y={Y(v)+4} textAnchor="end" fill="#aec5d3" fontSize="11">{v.toFixed(1)}</text></g>;})}
      {left.ideal_reference===1&&right.ideal_reference===1 && <line x1="50" x2="548" y1={Y(1)} y2={Y(1)} stroke="#faf0d2" strokeDasharray="6 4" strokeWidth="1.5"/>}
      {[left,right].map((c,j)=>{const color=j?'#ffc78a':'#76f2e2',off=j?5:-5;
        const valid=c.values.every(Number.isFinite)&&c.stderr.every(Number.isFinite);
        if(!valid)return null;
        const points=c.values.map((v,i)=>`${X(i)+off},${Y(v)}`).join(' ');
        return <g key={j} stroke={color} fill={color}>{!directional&&<polyline points={points} fill="none" strokeWidth="2"/>}{c.values.map((v,i)=><g key={i}><line x1={X(i)+off} x2={X(i)+off} y1={Y(v-c.stderr[i])} y2={Y(v+c.stderr[i])} strokeWidth="1.5"/><circle cx={X(i)+off} cy={Y(v)} r="3"/></g>)}</g>;})}
      {left.x.map((v,i)=>(directional||i===0||i===n-1||i===Math.floor(n/2))&&<text key={i} x={X(i)} y="231" fill="#bacbd7" fontSize={directional?11:12} textAnchor="middle">{directional?(left.indices?.[i]??[]).join(','):v.toFixed(2)}</text>)}
    </svg>
    <figcaption><span className="comparison-color-a">● A</span> and <span className="comparison-color-b">● B</span> · pointwise replica SEM. {left.ideal_reference===1&&right.ideal_reference===1?'Dashed: independent uniform-position reference 1 (not the interacting ground state). ':''}{directional?'Individual reciprocal directions, not a continuous radial S(k) curve.':'Radial bins; connecting points does not establish a phase.'}</figcaption>
  </figure>;
}

export default function ComparisonLab() {
  const [catalog,setCatalog]=useState<ComparisonPreset[]>([]),[chosen,setChosen]=useState('oscillator-trial-choice-v1');
  const [values,setValues]=useState<Record<string,number>>({}),[data,setData]=useState<Comparison|null>(null);
  const [busy,setBusy]=useState(false),[error,setError]=useState('');
  const revision=useRef(0),running=useRef(false);
  const selected=catalog.find(c=>c.id===chosen);
  useEffect(()=>{let alive=true;fetch('/api/v1/comparisons').then(async r=>{
    if(!r.ok)throw new Error(`Comparison catalog unavailable (HTTP ${r.status})`);
    return (await r.json()) as CatalogResponse;
  }).then(r=>{if(alive)setCatalog(r.comparisons);}).catch(e=>{if(alive)setError(String(e));});
    return()=>{alive=false;};},[]);
  function choose(id:string) {revision.current+=1;setChosen(id);setData(null);setError('');setValues({});}
  function setParameter(key:string,value:number) {revision.current+=1;setValues(v=>({...v,[key]:value}));setData(null);setError('');}
  async function execute() {
    if(running.current)return;
    const stamp=++revision.current;
    running.current=true;setBusy(true);setError('');setData(null);
    try {
      const response=await fetch(`/api/v1/comparisons/${chosen}/run`, {method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({parameters:values})});
      const payload=await response.json();
      if(!response.ok)throw new Error(typeof payload.detail==='string'?payload.detail:`HTTP ${response.status}`);
      if(stamp===revision.current)setData(payload as Comparison);
    } catch(e) {if(stamp===revision.current)setError(e instanceof Error?e.message:String(e));}
    finally {running.current=false;setBusy(false);}
  }
  const a=data?.arms[0].experiment.distributions, b=data?.arms[1].experiment.distributions;
  return <div className="comparison-laboratory">
    <header className="hero"><span className="eyebrow">SCIENTIFIC COMPARISONS / PATCH 39</span><h1>Change one thing. Understand what changed.</h1>
      <p>Pick a controlled comparison or a clearly labeled cross-model check. The application records which physical assumptions stay fixed, where methods differ, and which conclusions the evidence supports.</p></header>
    <div className="content-grid">
      <section className="panel controls"><div className="panel-heading"><span className="eyebrow">01 / DESIGN</span><h2>Your comparison</h2></div>
        <label htmlFor="comparison-preset">Research question</label>
        <select id="comparison-preset" value={chosen} onChange={(e:{target:{value:string}})=>choose(e.target.value)}>
          {catalog.length?catalog.map(c=><option key={c.id} value={c.id}>{c.title}</option>):<option value={chosen}>Loading curated studies…</option>}
        </select>
        {selected && <><p className="hint">{selected.question}</p><p className="hint">{selected.description}</p>
          <span className="mode-tag">{selected.comparison_kind.replaceAll('_',' ')}</span>
          {selected.parameters.map(param=><div className="comparison-slider" key={param.id}>
            <div className="control-row"><label htmlFor={`comparison-${param.id}`}>{param.label}</label><output>{(values[param.id]??param.default).toFixed(2)}</output></div>
            <input type="range" id={`comparison-${param.id}`} min={param.min} max={param.max} step={param.step} value={values[param.id]??param.default} onChange={(e:{target:{value:string}})=>setParameter(param.id,Number(e.target.value))}/>
          </div>)}
          <button className="primary" type="button" disabled={busy} onClick={()=>void execute()}>{busy?'Calculating two studies…':selected.live?'Run both VMC studies':'Compare existing evidence'}</button>
          <p className="hint">{selected.live?'Runs two bounded VMC experiments in sequence using the existing QMC engine. This can take a little time.':'Uses existing analytical or historically recorded values; does not launch a new Monte Carlo run.'}</p>
        </>}
        {error&&<p className="error" role="alert">{error}</p>}
      </section>
      <section className="panel result-panel" aria-live="polite"><div className="panel-heading"><span className="eyebrow">02 / MEASUREMENT</span><h2>Compare measured values</h2></div>
        {data ? <><ComparisonBars data={data}/><div className="comparison-delta"><small>Difference B − A · {data.analysis.units}</small><strong>{data.analysis.difference_b_minus_a>=0?'+':''}{FORMAT(data.analysis.difference_b_minus_a)}</strong></div>
          <p className="science-caveat">{data.analysis.inference}</p>
          <button type="button" className="outline" onClick={()=>fileDownload(`${data.comparison_id}.json`,JSON.stringify(data,null,2))}>↓ Export full comparison JSON</button></>
        : <div className="graph-empty">Choose what to compare, then run the study. Only supported observable-matched comparisons are available.</div>}
      </section>
    </div>
    {data && <>
      <section className="panel comparison-controls"><div className="panel-heading"><span className="eyebrow">03 / SCIENTIFIC DESIGN</span><h2>What stays fixed, and what changes?</h2></div>
        <div className="comparison-logic-grid">
          <article><h3>Held fixed</h3><ul>{data.controls.held_fixed.map((v,i)=><li key={i}>{v}</li>)}</ul></article>
          <article><h3>Changed</h3><ul>{data.controls.changed.map((v,i)=><li key={i}>{v}</li>)}</ul></article>
          <article><h3>Confounding factors and uncertainty</h3><ul>{data.controls.confounders.map((v,i)=><li key={i}>{v}</li>)}</ul></article>
        </div>
      </section>
      <section className="panel comparison-evidence"><div className="panel-heading"><span className="eyebrow">04 / EVIDENCE</span><h2>Two transparent experiment records</h2></div>
        <div className="comparison-arm-grid">{data.arms.map((arm,i)=><EvidenceArm key={i} arm={arm}/>)}</div>
      </section>
      {a && b && a.some(c=>c.id==='pair_correlation') && <section className="panel comparison-distributions">
        <div className="panel-heading"><span className="eyebrow">05 / SPATIAL EVIDENCE</span><h2>Overlay structural measurements</h2></div>
        <div className="comparison-curve-grid">{a.filter(c=>c.id==='pair_correlation'||c.id==='structure_factor').map(c=>{
          const right=b.find(v=>v.id===c.id);return right?<PairedCurve key={c.id} left={c} right={right}/>:null;
        })}</div>
        <p className="science-caveat">g(r) and S(k) are measured in two selected variational trial distributions. Shared seed identifiers do not supply their cross-run covariance; pointwise error bars are not simultaneous uncertainty bands.</p>
      </section>}
      <section className="panel comparison-interpretation"><div className="panel-heading"><span className="eyebrow">06 / INTERPRETATION</span><h2>What does the comparison teach?</h2></div>
        <p>{data.interpretation.relationship}</p>
        <div className="comparison-logic-grid"><article><h3>Supported conclusions</h3><ul>{data.interpretation.supported_conclusions.map((v,i)=><li key={i}>{v}</li>)}</ul></article>
        <article><h3>What we cannot conclude</h3><ul>{data.interpretation.not_supported.map((v,i)=><li key={i}>{v}</li>)}</ul></article></div>
        <p className="hint">Comparison fingerprint: <code>{data.fingerprint_sha256}</code></p>
      </section>
    </>}
  </div>;
}
