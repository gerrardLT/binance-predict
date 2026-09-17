#!/usr/bin/env python3
"""V2：BTC 小趋势启动条件研究。

启动 K 收盘时决策；目标是未来 H=3/5/8 根多数同向、同向占比、净位移与 episode 经济性。
baseline 仅使用 discovery/validation；final 读取冻结 registry 后只打开一次 holdout。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260918
DAY_MS = 86_400_000
BAR_MS = {"5m": 300_000, "15m": 900_000}
HORIZONS = (3, 5, 8)

ATOM_FORMULAS = {
    "strong_body": "body_ratio >= 0.60",
    "close_extreme": "signed close location >= 0.80",
    "range_expansion": "range_atr >= 1.25",
    "volume_expansion": "relative_volume >= 1.25",
    "breakout20": "directional close breaks prior 20-bar extreme",
    "breakout50": "directional close breaks prior 50-bar extreme",
    "compression_release": "prior range5/range20 <= 0.75 and current range_atr >= 1.25",
    "efficient_move": "directional path efficiency over 3 bars >= 0.60",
    "hhhl_structure": "up: HH+HL >=4/6; down: LH+LL >=4/6",
    "momentum_accel": "directional 3-bar per-bar return > directional 8-bar per-bar return",
    "align_1h": "direction aligns with last completed 1h move",
    "align_4h": "direction aligns with last completed 4h move",
    "early_htf_slot": "bar lies in first half of its 1h block",
    "sentiment_confirm": "completed window token close >=0.65 for bar direction",
    "sentiment_orderly": "completed window quote path efficiency >=0.60",
    "sentiment_late_lock": "completed window final sign lock occurs after 40% progress",
    "sentiment_btc_align": "completed quote direction and K-line direction agree",
    "streak1_control": "first bar of a color run",
    "streak2_confirm": "second bar of a color run",
    "streak3_confirm": "third bar of a color run",
}


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x): out[k:] = x[:-k]
    return out


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        cs = np.r_[0.0, np.cumsum(np.nan_to_num(x, nan=0.0))]
        out[w-1:] = (cs[w:] - cs[:-w]) / w
    return out


def rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    out=np.full(len(x),np.nan)
    if len(x)>=w: out[w-1:]=np.lib.stride_tricks.sliding_window_view(x,w).max(axis=1)
    return out


def rolling_min(x: np.ndarray, w: int) -> np.ndarray:
    out=np.full(len(x),np.nan)
    if len(x)>=w: out[w-1:]=np.lib.stride_tricks.sliding_window_view(x,w).min(axis=1)
    return out


def run_length(direction: np.ndarray) -> np.ndarray:
    out=np.ones(len(direction),np.int16)
    for i in range(1,len(direction)):
        out[i]=out[i-1]+1 if direction[i]!=0 and direction[i]==direction[i-1] else 1
    return out


def efficiency(close: np.ndarray, w: int) -> np.ndarray:
    path=rolling_mean(np.abs(close-shift(close)),w)*w
    base=shift(close,w)
    return np.abs(close-base)/path


def last_lock(values: np.ndarray) -> float:
    if not len(values): return math.nan
    signs=np.sign(values-.5); final=signs[-1]
    if final==0:return 1.0
    for i in range(len(signs)):
        if np.all(signs[i:]==final):return i/max(len(signs)-1,1)
    return 1.0


def quote_features(path: Path, ms: int) -> dict[int, dict]:
    raw=json.loads(path.read_text(encoding="utf8"));wins={}
    for r in raw:
        try:
            ts=int(r["timestamp"]);up=float(r["up_price"]);down=float(r["down_price"])
        except (KeyError,TypeError,ValueError):continue
        if 0<up<1 and 0<down<1 and .95<=up+down<=1.05:wins.setdefault(ts//ms*ms,[]).append((ts,up,down))
    out={}
    for start,pts in wins.items():
        pts.sort();up=np.array([p[1] for p in pts],float);down=np.array([p[2] for p in pts],float)
        pathlen=float(np.abs(np.diff(up)).sum()) if len(up)>1 else 0
        out[start]={"sent_up_close":float(up[-1]),"sent_down_close":float(down[-1]),"sent_eff":abs(float(up[-1]-up[0]))/pathlen if pathlen else 0.0,"sent_lock":last_lock(up),"sent_samples":len(up)}
    return out


def build_frame(root: Path, tf: str) -> tuple[pd.DataFrame,dict]:
    path=root/f"output/klines_{tf}_720d.csv";df=pd.read_csv(path);ms=BAR_MS[tf]
    t=pd.to_datetime(df.timestamp,utc=True).to_numpy(dtype="datetime64[ms]").astype(np.int64)
    o,h,l,c,v=(df[x].to_numpy(float) for x in ("open","high","low","close","volume"));n=len(t)
    if len(np.unique(t))!=n or not np.all(np.diff(t)==ms):raise ValueError(f"{tf} continuity")
    direction=np.sign(c-o).astype(np.int8);rng=h-l;safe=np.where(rng>0,rng,np.nan);body=np.abs(c-o)/safe;close_loc=(c-l)/safe
    signed_close_loc=np.where(direction>0,close_loc,1-close_loc)
    prev_c=shift(c);tr=np.maximum(rng,np.maximum(np.abs(h-prev_c),np.abs(l-prev_c)));atr20=shift(rolling_mean(np.nan_to_num(tr),20));range_atr=rng/atr20
    rel_volume=v/shift(rolling_mean(v,20));range_pct=rng/o;compression=shift(rolling_mean(range_pct,5))/shift(rolling_mean(range_pct,20))
    ph20,pl20=rolling_max(shift(h),20),rolling_min(shift(l),20);ph50,pl50=rolling_max(shift(h),50),rolling_min(shift(l),50)
    breakout20=np.where(direction>0,c>ph20,c<pl20);breakout50=np.where(direction>0,c>ph50,c<pl50)
    eff3=efficiency(c,3);streak=run_length(direction)
    hh=(h>shift(h)).astype(int);hl=(l>shift(l)).astype(int);lh=(h<shift(h)).astype(int);ll=(l<shift(l)).astype(int)
    up_struct=rolling_mean(hh+hl,3)*3;down_struct=rolling_mean(lh+ll,3)*3;structure=np.where(direction>0,up_struct,down_struct)
    ret3=(c/shift(c,3)-1)/3;ret8=(c/shift(c,8)-1)/8;accel=direction*(ret3-ret8)
    ret1h=direction*(c/shift(c,max(1,3_600_000//ms))-1);ret4h=direction*(c/shift(c,max(1,14_400_000//ms))-1)
    slot=(t//ms)%(3_600_000//ms if 3_600_000>ms else 1)
    early_slot=slot<max(1,(3_600_000//ms)//2)
    qpath=root/f"output/online_{tf}_samples_full_merged.json";qf=quote_features(qpath,ms)
    sent_up=np.array([qf.get(int(x),{}).get("sent_up_close",math.nan) for x in t]);sent_down=np.array([qf.get(int(x),{}).get("sent_down_close",math.nan) for x in t]);sent_eff=np.array([qf.get(int(x),{}).get("sent_eff",math.nan) for x in t]);sent_lock=np.array([qf.get(int(x),{}).get("sent_lock",math.nan) for x in t]);sent_q=np.where(direction>0,sent_up,sent_down)
    rows={"timeframe":tf,"bar_open":t,"bar_utc":[datetime.fromtimestamp(x/1000,timezone.utc).isoformat() for x in t],"direction":direction,"body_ratio":body,"signed_close_loc":signed_close_loc,"range_atr":range_atr,"rel_volume":rel_volume,"compression_5_20":compression,"breakout20":breakout20,"breakout50":breakout50,"efficiency3":eff3,"structure_score":structure,"momentum_accel":accel,"align1h":ret1h>0,"align4h":ret4h>0,"early_htf_slot":early_slot,"streak":streak,"sent_q":sent_q,"sent_eff":sent_eff,"sent_lock":sent_lock,"sent_align":sent_q>=.5,"utc_day":t//DAY_MS,"utc_month":[datetime.fromtimestamp(x/1000,timezone.utc).strftime("%Y-%m") for x in t]}
    for H in HORIZONS:
        valid=np.zeros(n,bool);valid[:-H]=True;future=np.zeros((n,H),np.int8);future[:-H]=np.lib.stride_tricks.sliding_window_view(direction[1:],H)
        same=(future==direction[:,None]);count=same.sum(1);rows[f"h{H}_valid"]=valid&(direction!=0)&np.all(future!=0,axis=1);rows[f"h{H}_same_count"]=count;rows[f"h{H}_same_frac"]=count/H;rows[f"h{H}_majority"]=count>=math.floor(H/2)+1;rows[f"h{H}_strong_majority"]=count>={3:3,5:4,8:6}[H]
        nextc=np.full(n,np.nan);nextc[:-H]=c[H:];rows[f"h{H}_signed_net_ret"]=direction*(nextc-c)/c
        longest=np.zeros(n,int)
        for i in range(n-H):
            run=best=0
            for x in same[i]:run=run+1 if x else 0;best=max(best,run)
            longest[i]=best
        rows[f"h{H}_longest_run"]=longest
    out=pd.DataFrame(rows)
    unique=np.array(sorted(t));s1=int(unique[int(n*.5)]);s2=int(unique[int(n*.75)]);out["segment"]=np.where(t<s1,"discovery",np.where(t<s2,"validation","holdout"))
    audit={"timeframe":tf,"rows":n,"start":out.bar_utc.iloc[0],"end":out.bar_utc.iloc[-1],"split1":datetime.fromtimestamp(s1/1000,timezone.utc).isoformat(),"split2":datetime.fromtimestamp(s2/1000,timezone.utc).isoformat(),"kline_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"quote_sha256":hashlib.sha256(qpath.read_bytes()).hexdigest(),"sentiment_coverage":float(np.isfinite(sent_q).mean()),"feature_clock":"startup bar close and earlier only"}
    return out,audit


def atom_mask(df:pd.DataFrame,atom:str)->pd.Series:
    if atom=="strong_body":return df.body_ratio>=.60
    if atom=="close_extreme":return df.signed_close_loc>=.80
    if atom=="range_expansion":return df.range_atr>=1.25
    if atom=="volume_expansion":return df.rel_volume>=1.25
    if atom=="breakout20":return df.breakout20.astype(bool)
    if atom=="breakout50":return df.breakout50.astype(bool)
    if atom=="compression_release":return (df.compression_5_20<=.75)&(df.range_atr>=1.25)
    if atom=="efficient_move":return df.efficiency3>=.60
    if atom=="hhhl_structure":return df.structure_score>=4
    if atom=="momentum_accel":return df.momentum_accel>0
    if atom=="align_1h":return df.align1h.astype(bool)
    if atom=="align_4h":return df.align4h.astype(bool)
    if atom=="early_htf_slot":return df.early_htf_slot.astype(bool)
    if atom=="sentiment_confirm":return df.sent_q>=.65
    if atom=="sentiment_orderly":return df.sent_eff>=.60
    if atom=="sentiment_late_lock":return df.sent_lock>=.40
    if atom=="sentiment_btc_align":return df.sent_align.astype(bool)
    if atom=="streak1_control":return df.streak==1
    if atom=="streak2_confirm":return df.streak==2
    if atom=="streak3_confirm":return df.streak==3
    raise KeyError(atom)


def candidate_mask(df,c):
    mask=(df.timeframe==c["timeframe"])&(df.direction!=0)
    for atom in c.get("atoms",[]):mask&=atom_mask(df,atom)
    if c.get("direction") in ("UP","DOWN"):mask&=df.direction==(1 if c["direction"]=="UP" else -1)
    return mask


def bootstrap(values,clusters,seed,draws=2000):
    if not len(values):return math.nan,math.nan
    keys=np.unique(clusters)
    if len(keys)<3:return math.nan,math.nan
    groups=[values[clusters==x] for x in keys];rng=np.random.default_rng(seed);out=np.empty(draws)
    for i in range(draws):
        pick=rng.integers(0,len(groups),len(groups));out[i]=np.concatenate([groups[j] for j in pick]).mean()
    return tuple(float(x) for x in np.percentile(out,[2.5,97.5]))


def normal_p(k,n,p0):
    if not n or not 0<p0<1:return math.nan
    z=(k-n*p0-.5)/math.sqrt(n*p0*(1-p0));return .5*math.erfc(z/math.sqrt(2))


def stats(group:pd.DataFrame,H:int,seed=SEED,with_bootstrap=False):
    cols=[f"h{H}_valid",f"h{H}_majority",f"h{H}_strong_majority",f"h{H}_same_count",f"h{H}_same_frac",f"h{H}_signed_net_ret",f"h{H}_longest_run","utc_day","utc_month"]
    group=group.loc[group[f"h{H}_valid"].astype(bool),cols];n=len(group)
    if not n:return {"n":0,"episodes":0,"majority_rate":math.nan,"baseline_majority":math.nan,"lift_pp":math.nan,"same_frac":math.nan,"signed_net_ret":math.nan,"longest_run":math.nan,"episode_return":math.nan,"episode_ci_low":math.nan,"episode_ci_high":math.nan,"p_edge":math.nan}
    y=group[f"h{H}_majority"].astype(float).to_numpy();strong=group[f"h{H}_strong_majority"].astype(float).to_numpy();baseline=float(group[f"h{H}_majority"].mean()) # overwritten by caller
    # One equal-stake bet per future bar. Correct direction pays +1/-1 proxy before market pricing.
    episode=(2*group[f"h{H}_same_count"].to_numpy(float)-H)
    ids=np.arange(n);lo,hi=bootstrap(episode,ids,seed) if with_bootstrap else (math.nan,math.nan)
    return {"n":n,"episodes":n,"majority_rate":float(y.mean()),"baseline_majority":baseline,"lift_pp":0.0,"strong_majority_rate":float(strong.mean()),"baseline_strong_majority":math.nan,"strong_lift_pp":math.nan,"same_frac":float(group[f"h{H}_same_frac"].mean()),"signed_net_ret":float(group[f"h{H}_signed_net_ret"].mean()),"longest_run":float(group[f"h{H}_longest_run"].mean()),"episode_return":float(episode.mean()),"episode_ci_low":lo,"episode_ci_high":hi,"p_edge":math.nan,"weeks":int(group.utc_day.floordiv(7).nunique()),"months":int(group.utc_month.nunique())}


def definitions():
    return [{"hypothesis_id":f"T|{tf}|{direction}|{atom}","timeframe":tf,"direction":direction,"atoms":[atom],"mechanism":formula} for tf in BAR_MS for direction in ("BOTH","UP","DOWN") for atom,formula in ATOM_FORMULAS.items()]


def evaluate(df,defs,segments=("discovery","validation")):
    rows=[]
    target_cols=[x for x in df.columns if x.startswith("h")]+["utc_day","utc_month"]
    for d in defs:
        cm=candidate_mask(df,d)
        for segment in segments:
            pool=df.loc[(df.segment==segment)&(df.timeframe==d["timeframe"])&(df.direction!=0),target_cols]
            group=df.loc[cm&(df.segment==segment),target_cols]
            for H in HORIZONS:
                r=stats(group,H,SEED+len(rows),with_bootstrap=(segments==("holdout",)));valid=pool[pool[f"h{H}_valid"].astype(bool)];p0=float(valid[f"h{H}_majority"].mean()) if len(valid) else math.nan;sp0=float(valid[f"h{H}_strong_majority"].mean()) if len(valid) else math.nan;r["baseline_majority"]=p0;r["lift_pp"]=(r["majority_rate"]-p0)*100 if r["n"] else math.nan;r["baseline_strong_majority"]=sp0;r["strong_lift_pp"]=(r["strong_majority_rate"]-sp0)*100 if r["n"] else math.nan;r["p_edge"]=normal_p(int(round(r["majority_rate"]*r["n"])) if r["n"] else 0,r["n"],p0)
                rows.append({"hypothesis_id":d["hypothesis_id"],"timeframe":d["timeframe"],"direction":d["direction"],"atoms":"+".join(d["atoms"]),"mechanism":d["mechanism"],"segment":segment,"horizon":H}|r)
    # BH by segment+horizon
    for segment in segments:
        for H in HORIZONS:
            idx=[i for i,x in enumerate(rows) if x["segment"]==segment and x["horizon"]==H];vals=sorted(((i,rows[i]["p_edge"]) for i in idx if math.isfinite(rows[i]["p_edge"])),key=lambda x:x[1]);run=1.0;m=len(vals)
            for rank in range(m,0,-1):i,p=vals[rank-1];run=min(run,p*m/rank);rows[i]["fdr_q"]=run
    return rows


def combos(single_rows):
    x=pd.DataFrame(single_rows);d=x[(x.segment=="discovery")&(x.direction=="BOTH")&(x.horizon==5)&(x.n>=300)&(x.lift_pp>=1)&(x.fdr_q<=.10)];v=x[(x.segment=="validation")&(x.horizon==5)][["hypothesis_id","n","lift_pp"]].rename(columns={"n":"vn","lift_pp":"vlift"});s=d.merge(v,on="hypothesis_id");s=s[(s.vn>=150)&(s.vlift>0)];out=[]
    for tf,g in s.groupby("timeframe"):
        atoms=list(g.sort_values("lift_pp",ascending=False).atoms.head(8))
        for i in range(len(atoms)):
            for j in range(i+1,len(atoms)):
                out.append({"hypothesis_id":f"T2|{tf}|BOTH|{atoms[i]}+{atoms[j]}","timeframe":tf,"direction":"BOTH","atoms":[atoms[i],atoms[j]],"mechanism":f"{ATOM_FORMULAS[atoms[i]]} AND {ATOM_FORMULAS[atoms[j]]}"})
    return out[:100]


def self_check():
    d=np.array([1,1,-1,-1,-1,1]);assert run_length(d).tolist()==[1,2,1,2,3,1]
    assert abs(float(efficiency(np.array([1.,2.,3.,4.]),2)[3])-1)<1e-12
    print("self-check: PASS")


def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x)),encoding="utf8")


def baseline(root,out):
    out.mkdir(parents=True,exist_ok=True);cache=out/"trend_event_band_preholdout.csv.gz";audit_path=out/"data_audit.json"
    if cache.exists():
        pre=pd.read_csv(cache);audits=json.loads(audit_path.read_text(encoding="utf8")) if audit_path.exists() else []
    else:
        frames=[];audits=[]
        for tf in BAR_MS:
            df,a=build_frame(root,tf);frames.append(df);audits.append(a)
        data=pd.concat(frames,ignore_index=True);pre=data[data.segment!="holdout"].copy();pre.to_csv(cache,index=False,compression="gzip");write_json(audit_path,audits)
    defs=definitions();single=evaluate(pre,defs);combo_defs=combos(single);combo=evaluate(pre,combo_defs) if combo_defs else []
    pd.DataFrame(single).to_csv(out/"trend_single_factor_preholdout.csv",index=False);pd.DataFrame(combo).to_csv(out/"trend_combo_preholdout.csv",index=False)
    write_json(out/"data_audit.json",audits);write_json(out/"feature_dictionary.json",[{"feature":x,"formula":y,"available":"startup_bar_close"} for x,y in ATOM_FORMULAS.items()]);write_json(out/"hypothesis_registry.json",{"generated_utc":datetime.now(timezone.utc).isoformat(),"holdout_opened_utc":None,"data_hashes":{a["timeframe"]:{"kline":a["kline_sha256"],"quote":a["quote_sha256"]} for a in audits},"single_definitions":defs,"combo_definitions":combo_defs,"frozen_candidates":[]});manifest={"stage":"baseline","preholdout_rows":len(pre),"single_tests":len(defs),"combo_tests":len(combo_defs)};write_json(out/"manifest.json",manifest);print(json.dumps(manifest,ensure_ascii=False,indent=2))


def final(root,out,registry_path):
    reg=json.loads(registry_path.read_text(encoding="utf8"));
    if reg.get("holdout_opened_utc"):raise RuntimeError("holdout already opened")
    frozen=reg.get("frozen_candidates") or []
    if not frozen:raise ValueError("freeze candidates")
    reg["holdout_opened_utc"]=datetime.now(timezone.utc).isoformat();write_json(registry_path,reg)
    data=pd.concat([build_frame(root,tf)[0] for tf in BAR_MS],ignore_index=True);hold=data[data.segment=="holdout"];results=evaluate(hold,frozen,segments=("holdout",));pd.DataFrame(results).to_csv(out/"trend_holdout_results.csv",index=False)
    events=[]
    for d in frozen:
        g=hold[candidate_mask(hold,d)].copy();g["hypothesis_id"]=d["hypothesis_id"];events.append(g)
    ev=pd.concat(events,ignore_index=True) if events else pd.DataFrame();ev.to_csv(out/"trend_holdout_events.csv.gz",index=False,compression="gzip")
    regimes=[]
    if len(ev):
        ev["range_regime"]=np.where(ev.range_atr>=1.25,"HIGH",np.where(ev.range_atr<=.75,"LOW","MID"));ev["session"]=np.where((ev.bar_open//3_600_000)%24<8,"ASIA",np.where((ev.bar_open//3_600_000)%24<16,"EUROPE","US"))
        for key,g in ev.groupby(["hypothesis_id","direction","range_regime","session","utc_month"]):
            for H in HORIZONS:regimes.append(dict(zip(("hypothesis_id","direction","range_regime","session","month"),key))|{"horizon":H}|stats(g,H,SEED+len(regimes),with_bootstrap=True))
    pd.DataFrame(regimes).to_csv(out/"trend_holdout_regimes.csv",index=False);manifest={"stage":"final","frozen":len(frozen),"holdout_events":len(ev),"result_rows":len(results)};write_json(out/"manifest.json",manifest);print(json.dumps(manifest,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("."));ap.add_argument("--out",type=Path,default=Path("output/trend_start_v2_20260918"));ap.add_argument("--stage",choices=("self-check","baseline","final"),required=True);ap.add_argument("--registry",type=Path);a=ap.parse_args()
    if a.stage=="self-check":self_check()
    elif a.stage=="baseline":baseline(a.root,a.out)
    else:
        if not a.registry:ap.error("--registry required")
        final(a.root,a.out,a.registry)


if __name__=="__main__":main()
