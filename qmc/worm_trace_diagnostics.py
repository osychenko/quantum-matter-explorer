"""Patch 30: offline diagnostics for correlated canonical Green-function ratios.

Save physical cold-chain measurements only. Split Rhat uses a *common pooled*
ratio in the linearized influence variable, so between-chain shifts survive.
The autocorrelation estimator uses a conservative initial-positive-pair window;
none of these diagnostics certifies stationarity or explores missing rare events.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .worm_audit import ratio_block_jackknife, split_rhat


def write_trace(path, audit, *, method, seed, links, beta, fugacity):
    """Persist the midpoint zero-mode ratio's actual numerator/denominator.

    Atomic replace avoids exposing partial compressed files when worker processes
    finish at different times. No physical states, positions or RNG objects are
    serialized. Always call after the measurement chain, never during it.
    """
    path = Path(path)
    if method not in ('baseline', 'exchange') or type(seed) is not int or seed < 0:
        raise ValueError('invalid trace method or seed')
    if type(links) is not int or links < 4 or links % 2 or not np.isfinite(beta) or beta <= 0:
        raise ValueError('invalid grid or beta')
    if not np.isfinite(fugacity) or fugacity <= 0:
        raise ValueError('invalid fugacity')
    it = links//2-1
    if not np.any(np.all(audit.indices == (0, 0, 0), axis=1)):
        raise ValueError('trace requires zero momentum mode')
    im = int(np.flatnonzero(np.all(audit.indices == (0, 0, 0), axis=1))[0])
    numerator = np.asarray(audit.numerator_trace[:, it, im], dtype=np.float64)
    denominator = np.asarray(audit.denominator_trace, dtype=np.float64)
    if numerator.ndim != 1 or numerator.size < 32 or numerator.shape != denominator.shape:
        raise ValueError('invalid trace shape')
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)) or np.any(denominator < 0) or denominator.sum() <= 0:
        raise ValueError('invalid trace values')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('wb') as handle:
            np.savez_compressed(handle, numerator=numerator, denominator=denominator,
                method=np.array(method), seed=np.int64(seed), M=np.int64(links),
                beta=np.float64(beta), scale=np.float64((links-1)/fugacity),
                green=np.float64(audit.green[it, im]),
                n_measurements=np.int64(numerator.size), schema=np.int64(1))
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return path


def load_trace(path):
    with np.load(path, allow_pickle=False) as data:
        required={'numerator','denominator','method','seed','M','beta','scale','green','n_measurements','schema'}
        if not required.issubset(data.files):
            raise ValueError('missing trace fields')
        n=np.asarray(data['numerator'], dtype=float)
        d=np.asarray(data['denominator'], dtype=float)
        method=str(data['method'].item())
        seed=int(data['seed'].item()); M=int(data['M'].item())
        beta=float(data['beta'].item()); scale=float(data['scale'].item())
        green=float(data['green'].item()); count=int(data['n_measurements'].item())
        schema=int(data['schema'].item())
    if (schema != 1 or method not in ('baseline','exchange') or seed < 0 or M < 4 or M % 2
        or n.ndim != 1 or n.size < 32 or n.shape != d.shape or count != n.size
        or not np.all(np.isfinite(n)) or not np.all(np.isfinite(d)) or np.any(d < 0)
        or d.sum() <= 0 or not np.isfinite(beta) or beta <= 0
        or not np.isfinite(scale) or scale <= 0 or not np.isfinite(green)):
        raise ValueError('invalid or unsupported trace data')
    recomputed=scale*n.sum()/d.sum()
    if not np.isclose(green,recomputed,atol=1e-9,rtol=1e-9):
        raise ValueError('trace does not reconstruct saved Green estimate')
    return dict(method=method,seed=seed,M=M,beta=beta,scale=scale,
                numerator=n,denominator=d,green=green)


def autocorrelation_curve(values, *, max_lag=4096):
    """FFT autocorrelation; window via initial positive adjacent lag pairs.

    The tail may be unresolved. The output includes a boundary/window warning;
    users should check stability with longer chains and independent replicas.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim != 1 or len(x) < 32 or not np.all(np.isfinite(x)) or max_lag < 2:
        raise ValueError('need >=32 finite observations and max_lag>=2')
    n=len(x)
    lag_limit=min(int(max_lag),n//4)
    centered=x-x.mean()
    variance=float(np.dot(centered,centered)/n)
    if variance < 1e-28:
        return dict(lags=np.arange(lag_limit+1),rho=np.r_[1.,np.zeros(lag_limit)],
                    tau_int=.5,ess=float(n),window=0,warning='constant_trace')
    size=1<<(2*n-1).bit_length()
    ft=np.fft.rfft(centered,n=size)
    corr=np.fft.irfft(ft*np.conjugate(ft),n=size)[:lag_limit+1]
    rho=corr/(n-np.arange(lag_limit+1))/variance
    rho[0]=1.
    # Geyer-style initial positive *pair* sequence to avoid chasing noisy tails.
    tau=.5
    window=0
    for lag in range(1,lag_limit+1,2):
        pair=float(rho[lag]+(rho[lag+1] if lag+1<=lag_limit else 0.))
        if pair<=0:
            break
        tau += pair
        window=min(lag+1,lag_limit)
        if window >= 5*tau:
            break
    warning='window_at_limit' if window>=lag_limit-1 else 'none'
    ess=float(min(n,n/(2*max(tau,.5))))
    return dict(lags=np.arange(lag_limit+1),rho=rho,tau_int=float(tau),
                ess=ess,window=window,warning=warning)


def blocking_curve(trace, *, blocks=(4,8,16,32,64,128)):
    n=trace['numerator']; d=trace['denominator']; scale=trace['scale']
    result=[]
    for b in blocks:
        if len(n) >= 8*b:
            _,error=ratio_block_jackknife(n[:,None,None],d,scale=scale,n_blocks=b)
            result.append(dict(blocks=b,block_length=len(n)/b,sem=float(error[0,0])))
    return result


def running_ratio(trace):
    numerator=np.cumsum(trace['numerator'])
    denominator=np.cumsum(trace['denominator'])
    return np.divide(trace['scale']*numerator,denominator,
                    out=np.full_like(numerator,np.nan),where=denominator>0)


def group_diagnostics(traces):
    """Diagnostics for same-model independent replicas with distinct seeds.

    Separate full-chain means are estimated without pooling measurement times;
    equal-length split-Rhat uses common pooled theta and is rank-free.
    """
    traces=list(traces)
    if len(traces)<2:
        raise ValueError('at least two independent chains required')
    if len({t['seed'] for t in traces}) != len(traces):
        raise ValueError('duplicate random seeds')
    first=traces[0]
    if any(t['method']!=first['method'] or t['M']!=first['M'] or
           not np.isclose(t['beta'],first['beta'],atol=1e-12,rtol=0) or
           not np.isclose(t['scale'],first['scale'],atol=1e-12,rtol=0)
           for t in traces):
        raise ValueError('mixed incompatible traces')
    ratio=np.array([t['green'] for t in traces],float)
    mean=float(np.mean(ratio))
    sem=float(np.std(ratio,ddof=1)/np.sqrt(len(ratio)))
    theta=first['scale']*sum(t['numerator'].sum() for t in traces)/sum(t['denominator'].sum() for t in traces)
    infl=[first['scale']*t['numerator']-theta*t['denominator'] for t in traces]
    rh=split_rhat(infl)
    rh_num=split_rhat([t['numerator'] for t in traces])
    rh_den=split_rhat([t['denominator'] for t in traces])
    acs=[autocorrelation_curve(x) for x in infl]
    return dict(method=first['method'],M=first['M'],replicas=len(traces),
        mean_green=mean,replica_sem=sem,pooled_green=float(theta),
        split_rhat_influence=float(rh),split_rhat_numerator=float(rh_num),
        split_rhat_denominator=float(rh_den),median_tau_int=float(np.median([a['tau_int'] for a in acs])),
        min_acf_ess=float(min(a['ess'] for a in acs)),
        window_warnings=sum(a['warning']!='none' for a in acs))
