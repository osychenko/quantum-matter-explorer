/** Patch 35: a small science-first expedition, not an automated physics grader. */
import {useEffect, useRef, useState} from 'react';
import {
  MISSIONS, XP_PER_MISSION, assessEvidence, earnedBadges, isUnlocked,
  recordCompletion, totalXp, type EngineObservation, type Finding,
  type Mission, type MissionId, type MissionProgress, type Prediction,
} from './missionEngine';

type Props = {progress:MissionProgress; onProgress:(next:MissionProgress)=>void; storageWarning:string};

function exportJson(filename:string, data:unknown) {
  const blob = new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
  const url = URL.createObjectURL(blob);
  const link=document.createElement('a');
  link.href=url;link.download=filename;link.click();
  window.setTimeout(()=>URL.revokeObjectURL(url),1000);
}

async function requestEvidence(mission:Mission, signal:AbortSignal):Promise<EngineObservation[]> {
  // The server's allowlist remains the authoritative validation layer.
  const results:EngineObservation[]=[];
  for (const item of mission.evidence) {
    const response=await fetch(`/api/v1/experiments/${item.presetId}/run`,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({parameters:item.parameters}),signal,
    });
    if (!response.ok) {
      const payload=await response.json().catch(()=>null);
      throw new Error(typeof payload?.detail==='string'?payload.detail:`Experiment unavailable (${response.status})`);
    }
    results.push(await response.json() as EngineObservation);
  }
  return results;
}

function EvidencePanel({finding,mission}:{finding:Finding;mission:Mission}) {
  const scale=Math.max(...finding.summaries.map(s=>s.value),1)*1.04;
  return <div className="mission-evidence" aria-live="polite">
    <div className="mission-section-label">02 / EVIDENCE COLLECTED</div>
    {finding.summaries.map((s,i)=><div className="mission-measurement" key={s.fingerprint+':'+i}>
      <div className="mission-measurement-header"><strong>{s.label}</strong>
        <span className={s.source==='recorded_result'?'mission-origin recorded':'mission-origin'}>
          {s.source==='recorded_result'?'Historical QMC recording':'Analytical calculation'}
        </span></div>
      <div className="mission-reading">{s.value.toFixed(5)}{s.uncertainty!==null&&<small> ± {s.uncertainty.toFixed(5)} replica SEM</small>}</div>
      <div className="mission-bar"><div style={{width:`${100*s.value/scale}%`}}/></div>
    </div>)}
    {mission.kind==='reference' ?
      <p className="mission-discovery">Difference from K=2 spectral reference: <strong>{finding.difference.toFixed(5)}</strong> · <strong>{finding.zScore?.toFixed(2)} reported SEM</strong></p> :
      <p className="mission-discovery">Second minus first: <strong>{finding.difference>0?'+':''}{finding.difference.toFixed(5)}</strong> in the zero-momentum addition correlator.</p>}
    <p className="mission-caution">{mission.caution}</p>
  </div>;
}

export default function MissionBoard({progress,onProgress,storageWarning}:Props) {
  const initial=MISSIONS.find(m=>isUnlocked(progress,m) && !progress.completions.some(c=>c.missionId===m.id))?.id || MISSIONS[0].id;
  const [active,setActive]=useState<MissionId>(initial);
  const [prediction,setPrediction]=useState<Prediction|null>(null);
  const [finding,setFinding]=useState<Finding|null>(null);
  const [interpretation,setInterpretation]=useState('');
  const [researchNote,setResearchNote]=useState('');
  const [feedback,setFeedback]=useState(false);
  const [newlyCompleted,setNewlyCompleted]=useState(false);
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState('');
  const controller=useRef<AbortController|null>(null);
  const mission=MISSIONS.find(m=>m.id===active)!;
  const completion=progress.completions.find(c=>c.missionId===active);
  const badges=earnedBadges(progress);
  const xp=totalXp(progress);

  useEffect(()=>()=>controller.current?.abort(),[active]);
  const selectMission=(id:MissionId)=>{
    const next=MISSIONS.find(m=>m.id===id);
    if (!next||!isUnlocked(progress,next)) return;
    controller.current?.abort();
    setActive(id);setPrediction(null);setFinding(null);setInterpretation('');
    setResearchNote('');setFeedback(false);setNewlyCompleted(false);setLoading(false);setError('');
    window.requestAnimationFrame(()=>document.getElementById('mission-chapter')?.scrollIntoView({block:'start',behavior:'smooth'}));
  };
  const reveal=async()=>{
    if (!prediction || loading || finding) return;
    controller.current?.abort();
    const current=new AbortController();controller.current=current;
    setLoading(true);setError('');
    try {
      const results=await requestEvidence(mission,current.signal);
      if (!current.signal.aborted) setFinding(assessEvidence(mission,results));
    } catch(e) {
      if (!current.signal.aborted) setError(e instanceof Error?e.message:'Unable to collect evidence');
    } finally {if (!current.signal.aborted) setLoading(false);}
  };
  const complete=()=>{
    if (!prediction||!finding||!interpretation||loading) return;
    if (!completion) {
      try {onProgress(recordCompletion(progress,mission,prediction,interpretation,finding,researchNote.trim()));setNewlyCompleted(true);}
      catch(e){setError(e instanceof Error?e.message:'Could not record mission');return;}
    }
    setFeedback(true);
  };
  const answeredCorrectly=interpretation===mission.correctInterpretation;
  const matched=prediction===finding?.comparison;

  return <>
    <header className="hero mission-hero">
      <span className="eyebrow">FIELD JOURNAL / THE FIRST EXPEDITION</span>
      <h1>Quantum Field School.</h1>
      <p>Become a researcher. Make a hypothesis, collect real numerical evidence, and learn what the data can—and cannot—tell you. Your curiosity earns progress, not your ability to guess the answer.</p>
    </header>
    <div className="mission-dashboard panel">
      <div className="mission-dashboard-lead"><span className="eyebrow">RESEARCH RANK</span><h2>{xp===160?'Quantum Explorer':xp>=80?'Field Researcher':xp>0?'Junior Investigator':'New Researcher'}</h2>
        <p>{progress.completions.length} of {MISSIONS.length} investigations completed · {xp} research XP</p></div>
      <div className="mission-dashboard-meter"><div className="mission-meter-line"><span>Expedition progress</span><strong>{Math.round(100*progress.completions.length/MISSIONS.length)}%</strong></div>
        <div className="mission-meter"><div style={{width:`${100*progress.completions.length/MISSIONS.length}%`}}/></div>
        <span className="mission-mini-note">{XP_PER_MISSION} XP per completed investigation · No extra points for a “correct” guess</span>
      </div>
    </div>
    <div className="mission-layout">
      <section className="panel mission-map">
        <span className="eyebrow">YOUR EXPEDITION MAP</span><h2>Four discoveries</h2>
        <div className="mission-steps">{MISSIONS.map(m=>{
          const unlocked=isUnlocked(progress,m);
          const finished=progress.completions.some(c=>c.missionId===m.id);
          return <button type="button" key={m.id} className={`mission-step ${active===m.id?'selected':''} ${finished?'finished':''}`}
            onClick={()=>selectMission(m.id)} disabled={!unlocked} aria-current={active===m.id?'step':undefined}>
            <span className="mission-step-symbol">{finished?'✓':unlocked?String(m.chapter).padStart(2,'0'):'⌁'}</span>
            <span><strong>{m.title}</strong><small>{m.subtitle}</small></span>
            <span className="mission-step-status">{finished?'Done':unlocked?'Open':'Locked'}</span>
          </button>;
        })}</div>
        <div className="mission-badges"><h3>Research badges <span>{badges.length}/5</span></h3>
          {badges.length? badges.map(b=><div className="mission-badge" key={b.id}><span>✦</span><div><strong>{b.title}</strong><small>{b.detail}</small></div></div>) :
            <p>Make your first hypothesis to earn a badge.</p>}
        </div>
        <button type="button" className="outline mission-export" onClick={()=>exportJson('qme-research-expedition.json',{
          application:'Quantum Matter Explorer',progress_version:progress.version,
          completions:progress.completions, xp, badges, scientific_note:'Educational progress only; not scientific certification',
        })}>↓ Export expedition journal</button>
        <p className="mission-mini-note">Progress lives only in this browser. Export it to keep a copy; this preview does not support importing a journal.</p>
        {storageWarning&&<p className="mission-caution" role="status">{storageWarning}</p>}
      </section>
      <section id="mission-chapter" className="panel mission-active" key={mission.id}>
        <div className="mission-active-top"><span className="eyebrow">CHAPTER {String(mission.chapter).padStart(2,'0')} / {MISSIONS.length}</span><span className="mission-xp">+{XP_PER_MISSION} XP</span></div>
        <h2>{mission.title}</h2><p className="mission-story">{mission.story}</p>
        {completion&&<div className="mission-complete-note">✓ Expedition completed on {new Date(completion.completedAt).toLocaleDateString()}. You may replay this investigation; XP is awarded once.</div>}
        <div className="mission-question"><span className="mission-section-label">01 / PREDICT FIRST</span><h3>{mission.hypothesis}</h3>
          <fieldset disabled={!!finding||loading} className="mission-radio-set"><legend className="sr-only">Choose your prediction</legend>
            {mission.choices.map(choice=><label key={choice.id} className={`mission-option ${prediction===choice.id?'chosen':''}`}>
              <input type="radio" name={`prediction-${mission.id}`} value={choice.id} checked={prediction===choice.id}
                onChange={()=>setPrediction(choice.id)}/><span>{choice.label}</span></label>)}
          </fieldset>
          <button type="button" className="primary mission-action" disabled={!prediction||loading||!!finding}
            onClick={()=>void reveal()}>{loading?'Collecting evidence…':mission.evidence.some(e=>e.mode==='recorded_result')?'Reveal curated evidence':'Run analytical controls'}</button>
          {!finding&&<p className="mission-mini-note">Choose before revealing the results. Both correct and incorrect hypotheses are valuable.</p>}
        </div>
        {error&&<p role="alert" className="error">{error}</p>}
        {finding&&<>
          <EvidencePanel finding={finding} mission={mission}/>
          <div className="mission-question"><span className="mission-section-label">03 / INTERPRET</span><h3>{mission.interpretQuestion}</h3>
            <fieldset className="mission-radio-set" disabled={feedback}><legend className="sr-only">Choose an interpretation</legend>
              {mission.interpretations.map(choice=><label key={choice.id} className={`mission-option ${interpretation===choice.id?'chosen':''}`}>
                <input type="radio" name={`interpret-${mission.id}`} value={choice.id} checked={interpretation===choice.id}
                  onChange={()=>setInterpretation(choice.id)}/><span>{choice.label}</span></label>)}
            </fieldset>
            <label htmlFor={`mission-note-${mission.id}`} className="mission-note-label">Research notebook note <span>(optional, maximum 400 characters)</span></label>
            <textarea id={`mission-note-${mission.id}`} rows={3} maxLength={400} disabled={feedback}
              placeholder="What surprised you? What would you test next?" value={researchNote} onChange={e=>setResearchNote(e.target.value)}/>
            <button type="button" className="primary mission-action" disabled={!interpretation||feedback} onClick={complete}>
              Record discovery {completion?'(replay)':''}
            </button>
          </div>
        </>}
        {feedback&&finding&&<div className="mission-reveal" role="status">
          <span className="eyebrow">DISCOVERY RECORDED</span><h3>{matched?'Your hypothesis matched the observation.':'Your hypothesis was challenged—that is how science progresses.'}</h3>
          <p>{answeredCorrectly?'Your interpretation matches the evidence.':'There is a stronger explanation to consider:'}</p>
          <p>{mission.lesson}</p><p className="mission-caution">{mission.caution}</p>
          <div className="mission-reveal-actions"><strong>{newlyCompleted?`+${XP_PER_MISSION} research XP`:'XP was earned on your first completion'}</strong>
            {MISSIONS[mission.chapter]&&<button className="outline" type="button" onClick={()=>selectMission(MISSIONS[mission.chapter].id)}>Next mission →</button>}</div>
        </div>}
      </section>
    </div>
    <section className="panel mission-principles"><span className="eyebrow">THE RESEARCHER'S CODE</span><h2>Progress means better questions, not perfect guesses.</h2>
      <p>Each mission uses a curated analytical result or an explicitly identified historical numerical benchmark. No task asserts a phase transition, mistakes imaginary-time paths for real motion, or presents a recorded QMC result as a live simulation. Achievements are educational milestones, not certifications of physical accuracy.</p>
    </section>
  </>;
}
