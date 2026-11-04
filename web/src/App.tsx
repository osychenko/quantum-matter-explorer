import { useEffect, useMemo, useState } from 'react';

type Preset = {id: string; title: string; status: string; mode: string; description: string; adjustable: string[]};
type Observable = {id: string; value: number; units: string; uncertainty: number | null; uncertainty_kind: string; spectral_reference_k2?: number; numerical_truncation_indicator?: number};
type Result = {schema_version: string; engine_release: string; experiment_id: string; preset_status: string; execution_mode: string; parameters: Record<string, unknown>; observables: Observable[]; provenance: Record<string, unknown>; cautions: string[]; fingerprint_sha256: string};
type Point = {x: number; green: number};
type Sweep = {variable: string; points: Point[]; kind: string; units: string; caution: string};
type Saved = {savedAt: string; result: Result};

const IDEAL = 'ideal-boson-addition-v1';
const STORAGE_KEY = 'qme-patch33-notebook-v1';

async function jsonRequest<T>(url: string, body?: object): Promise<T> {
  const response = await fetch(url, body === undefined ? undefined : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const message = typeof payload?.detail === 'string' ? payload.detail : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
function readNotebook(): Saved[] {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]') as unknown;
    return Array.isArray(value) ? value.slice(0, 20) as Saved[] : [];
  } catch { return []; }
}
function saveFile(name: string, contents: string) {
  const blob = new Blob([contents], {type: 'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob); link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function Graph({sweep, current}: {sweep: Sweep | null; current: number}) {
  if (!sweep?.points?.length) return <div className="graph-empty">Run the analytic control to see how the result changes.</div>;
  const points = sweep.points;
  const ymin = Math.min(...points.map(p => p.green)) - .04;
  const ymax = Math.max(...points.map(p => p.green)) + .04;
  const xmin = points[0].x, xmax = points[points.length - 1].x;
  const xp = (x: number) => 48 + (x - xmin) / (xmax - xmin) * 512;
  const yp = (y: number) => 226 - (y - ymin) / (ymax - ymin) * 190;
  const path = points.map((p, i) => `${i ? 'L' : 'M'} ${xp(p.x).toFixed(2)} ${yp(p.green).toFixed(2)}`).join(' ');
  const nearest = points.reduce((a, b) => Math.abs(b.x - current) < Math.abs(a.x - current) ? b : a);
  return <svg className="graph" viewBox="0 0 600 284" role="img" aria-label={`Analytical Green function versus ${sweep.variable}`}>
    {[0, 1, 2, 3, 4].map(k => {
      const val = ymin + (ymax - ymin) * k / 4;
      return <g key={k}><line x1="48" x2="560" y1={yp(val)} y2={yp(val)} stroke="#304052" strokeDasharray="4 6"/>
        <text x="40" y={yp(val) + 4} textAnchor="end" fill="#91a4b6" fontSize="11">{val.toFixed(2)}</text></g>;
    })}
    <path d={path} fill="none" stroke="#6eefe1" strokeWidth="3" strokeLinecap="round" />
    <line x1={xp(nearest.x)} x2={xp(nearest.x)} y1="25" y2="227" stroke="#e8b97b" strokeDasharray="3 4" />
    <circle cx={xp(nearest.x)} cy={yp(nearest.green)} r="6" fill="#e8b97b" stroke="#0a1624" strokeWidth="2" />
    <text x="48" y="259" fontSize="12" fill="#b5c7d5">{xmin.toFixed(1)}</text>
    <text x="560" y="259" fontSize="12" textAnchor="end" fill="#b5c7d5">{xmax.toFixed(1)}</text>
    <text x="303" y="278" textAnchor="middle" fontSize="12" fill="#b5c7d5">{sweep.variable === 'beta' ? 'Inverse temperature β' : 'Cell length L'}</text>
  </svg>;
}

export default function App() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selected, setSelected] = useState(IDEAL);
  const [beta, setBeta] = useState(2.4);
  const [length, setLength] = useState(2.8);
  const [variable, setVariable] = useState<'beta'|'cell_length'>('beta');
  const [result, setResult] = useState<Result | null>(null);
  const [sweep, setSweep] = useState<Sweep | null>(null);
  const [saved, setSaved] = useState<Saved[]>(readNotebook);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [tab, setTab] = useState<'experiment'|'notebook'>('experiment');
  const isIdeal = selected === IDEAL;
  const currentPreset = presets.find(p => p.id === selected);

  useEffect(() => {
    jsonRequest<{presets: Preset[]}>('/api/v1/experiments')
      .then(r => setPresets(r.presets)).catch(e => setError(String(e)));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setBusy(true); setError(''); setResult(null); setSweep(null);
    const parameters = isIdeal ? {beta, cell_length: length} : {};
    const work = async () => {
      const run = await jsonRequest<Result>(`/api/v1/experiments/${selected}/run`, {parameters});
      const series = isIdeal ? await jsonRequest<Sweep>(`/api/v1/experiments/ideal-boson-addition-v1/sweep`,
        {beta, cell_length: length, variable, count: 25}) : null;
      if (!cancelled) {setResult(run); setSweep(series); setBusy(false);}
    };
    // A short delay coalesces changes while the user drags a range slider.
    const timeout = setTimeout(() => {void work().catch(e => {if (!cancelled) {setError(String(e)); setBusy(false);}});}, 180);
    return () => {cancelled = true; clearTimeout(timeout);};
  }, [selected, beta, length, variable, isIdeal]);

  const observable = result?.observables[0];
  const formatted = useMemo(() => observable ? observable.value.toFixed(5) : '—', [observable]);
  const persist = () => {
    if (!result) return;
    const next = [{savedAt: new Date().toISOString(), result}, ...saved].slice(0, 20);
    try {localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); setSaved(next);}
    catch {setError('Browser storage is unavailable. Export the JSON result instead.');}
  };

  return <div className="layout">
    <aside className="sidebar">
      <div className="brand"><div className="brand-icon">◈</div><div><strong>QUANTUM<br/>MATTER</strong><span>EXPLORER</span></div></div>
      <div className="sidebar-label">WORKSPACE</div>
      <button className={`nav-link ${tab === 'experiment' ? 'active' : ''}`} onClick={() => setTab('experiment')}>⌁ <span>Quantum laboratory</span></button>
      <button className={`nav-link ${tab === 'notebook' ? 'active' : ''}`} onClick={() => setTab('notebook')}>▤ <span>Research notebook <small>{saved.length}</small></span></button>
      <div className="sidebar-bottom"><div className="dot"/> Local educational engine<br/><small>Patch 33 · Preview</small></div>
    </aside>
    <main className="main">
      <div className="topbar"><span>LABORATORY / {tab === 'experiment' ? 'EXPERIMENT 01' : 'NOTEBOOK'}</span><span className="top-status">● Engine v0.3 preview</span></div>
      {tab === 'experiment' ? <>
        <header className="hero"><span className="eyebrow">INTERACTIVE QUANTUM PHYSICS</span><h1>Explore the invisible.</h1>
          <p>Choose a scientifically scoped experiment, change its parameters and discover what the quantum model predicts.</p></header>
        <div className="content-grid">
          <section className="panel controls">
            <div className="panel-heading"><span className="eyebrow">01 / SETUP</span><h2>Your experiment</h2></div>
            <label htmlFor="preset">Physics scenario</label>
            <select id="preset" value={selected} onChange={e => setSelected(e.target.value)}>
              {presets.length ? presets.map(p => <option key={p.id} value={p.id}>{p.title}</option>) : <option value={IDEAL}>Free bosons</option>}
            </select>
            <p className="hint">{currentPreset?.description || 'Canonical ideal Bose-gas control.'}</p>
            <span className={`mode-tag ${isIdeal ? 'analytic' : 'recorded'}`}>{isIdeal ? '✦ Calculated analytical control' : '▤ Historical recorded benchmark'}</span>
            {isIdeal ? <div className="sliders">
              <div className="control-row"><label htmlFor="beta">Inverse temperature <em>β</em></label><output>{beta.toFixed(2)}</output></div>
              <input id="beta" aria-label="Inverse temperature" type="range" min="0.5" max="5" step="0.1" value={beta} onChange={e=>setBeta(Number(e.target.value))}/>
              <div className="control-row"><label htmlFor="length">Cell length <em>L</em></label><output>{length.toFixed(2)}</output></div>
              <input id="length" aria-label="Cell length" type="range" min="1.5" max="5" step="0.1" value={length} onChange={e=>setLength(Number(e.target.value))}/>
              <p className="hint">The momentum mode is fixed at k = 0. Particle number is fixed at N = 1 → 2.</p>
            </div> : <div className="note">Parameters are intentionally fixed. This is a previously measured result, not a live Monte Carlo simulation.</div>}
            <div className="parameter-foot">SCIENTIFIC STATUS <strong>{currentPreset?.status?.replaceAll('_', ' ') || 'supported'}</strong></div>
          </section>
          <section className="panel result-panel">
            <div className="panel-heading"><span className="eyebrow">02 / OBSERVE</span><h2>Simulation results</h2></div>
            {error && <div className="error" role="alert">{error}</div>}
            <div className="result-stat"><span>PARTICLE-ADDITION GREEN FUNCTION</span><div className="big-number">{busy && !result ? '…' : formatted}</div>
              <div className="result-detail">G₀(β/2) · dimensionless {observable?.uncertainty != null && <>· ± {observable.uncertainty.toFixed(5)} replica SEM</>}</div></div>
            <div className="chart-heading"><h3>{isIdeal ? 'Parameter response' : 'Benchmark comparison'}</h3>
              {isIdeal && <div className="segmented"><button className={variable==='beta'?'on':''} onClick={()=>setVariable('beta')}>β</button><button className={variable==='cell_length'?'on':''} onClick={()=>setVariable('cell_length')}>L</button></div>}</div>
            {isIdeal ? <Graph sweep={sweep} current={variable==='beta'?beta:length}/> : <div className="reference"><div><span>Observed, historical</span><strong>{observable?.value.toFixed(5) || '—'}</strong></div><div><span>Spectral reference K=2</span><strong>{observable?.spectral_reference_k2?.toFixed(5) || '—'}</strong></div><p>This comparison is not a proof of continuum convergence.</p></div>}
            <div className="result-actions"><button className="primary" disabled={!result} onClick={persist}>＋ Save to notebook</button><button className="outline" disabled={!result} onClick={()=>result&&saveFile(`${result.experiment_id}.json`,JSON.stringify(result,null,2))}>↓ Export JSON</button></div>
          </section>
        </div>
        <section className="panel interpretation"><span className="eyebrow">03 / UNDERSTAND</span><h2>What does this mean?</h2>
          <p>{isIdeal ? 'This observable measures a particle-addition correlation in imaginary time. The analytic ideal-gas formula is evaluated numerically for the thermal partition function. It does not show particles moving in real time.' : 'This is a fixed recorded measurement of interacting bosons at M=8. Its uncertainty comes from independent Monte Carlo replicas. The result passed screening but has not been certified as a continuum-limit value.'}</p>
          {result && <details><summary>Scientific provenance and limitations</summary><ul>{result.cautions.map((c,i)=><li key={i}>{c}</li>)}</ul><p>Fingerprint: <code>{result.fingerprint_sha256}</code></p></details>}
        </section>
      </> : <><header className="hero"><span className="eyebrow">YOUR INVESTIGATIONS</span><h1>Research notebook.</h1><p>Saved results are stored in this browser only. Export JSON files to retain them outside this device.</p></header>
        <section className="panel notebook"><div className="panel-heading"><h2>Saved experiments ({saved.length})</h2></div>
          {saved.length===0 ? <p>No experiments saved yet. Visit the laboratory to create your first entry.</p> : saved.map((item,i)=><div className="saved-item" key={`${item.savedAt}-${i}`}><div><strong>{item.result.experiment_id}</strong><small>{new Date(item.savedAt).toLocaleString()} · {item.result.execution_mode.replaceAll('_',' ')}</small></div><div><strong>{item.result.observables[0].value.toFixed(5)}</strong><button onClick={()=>saveFile(`experiment-${i+1}.json`,JSON.stringify(item.result,null,2))}>Export</button></div></div>)}
          {saved.length>0&&<button className="outline" onClick={()=>{if(confirm('Clear all browser-only notebook entries?')){localStorage.removeItem(STORAGE_KEY);setSaved([]);}}}>Clear notebook</button>}
        </section></>}
      <footer>Quantum Matter Explorer · Local-first educational preview · Numerical research methods remain separate</footer>
    </main>
  </div>;
}
