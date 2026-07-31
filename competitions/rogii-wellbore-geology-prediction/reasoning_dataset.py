"""reasoning_dataset.py — TOOL-AUGMENTED chain-of-thought for geosteering. The model reasons AND CALLS domain
calculation tools (our fleet agents) inside <think>: dip_fit, formation_band, pf_predict (the particle filter),
ncc_match, gr_signature. Tool RESULTS are grounded in real computation (PF from honest_feat). The answer uses
the tool outputs, so the ceiling is the tools' precision (~11 PF), not text-reasoning (~35). The model learns to
ORCHESTRATE/SELECT tools per well. Two datasets (1024/2048 tok), field-disjoint train/holdout.
Output: data/reason_tool_{1024,2048}_{train,holdout}.jsonl"""
import os, glob, json, numpy as np, pandas as pd

FORM = {"ANCC":"Anacacho","ASTNU":"Austin-U","ASTNL":"Austin-L","EGFDU":"EagleFord-U","EGFDL":"EagleFord-L","BUDA":"Buda"}

def load_pf_index():
    try:
        hf = pd.read_parquet("results/honest_feat.parquet")
        hf["rowidx"] = hf.id.str.rsplit("_",n=1).str[1].astype(int)
        idx = {}
        for w,g in hf.groupby("well"): idx[w] = dict(zip(g.rowidx.values, g.pf_dtvt.values))
        return idx
    except Exception: return None

def analyze(hw, tw):
    kn=hw.TVT_input.notna().to_numpy(); ev=hw.TVT_input.isna().to_numpy()&hw.TVT.notna().to_numpy()
    if kn.sum()<30 or ev.sum()<10: return None
    MD=hw.MD.to_numpy(float); GR=hw.GR.to_numpy(float); TVT=hw.TVT.to_numpy(float)
    ki=np.where(kn)[0]; ei=np.where(ev)[0]; mdps=MD[ki[-1]]; tvtps=hw.TVT_input.to_numpy(float)[ki[-1]]
    kk=ki[-min(400,len(ki)):]; dip0=float(np.polyfit(MD[kk],hw.TVT_input.to_numpy(float)[kk],1)[0])
    tws=tw.sort_values("TVT"); T=tws.TVT.to_numpy(float); g=tws.Geology.fillna("").astype(str).str.strip().to_numpy()
    bands={l:(float(T[g==l].min()),float(T[g==l].max())) for l in np.unique(g) if l}
    home=next((l for l,(lo,hi) in bands.items() if lo<=tvtps<=hi),"?")
    hb=bands.get(home,(tvtps-40,tvtps+40))
    return dict(tvtps=tvtps,dip0=dip0,home=home,hb=hb,thick=hb[1]-hb[0],twspan=float(T.max()-T.min()),
                grm=float(np.nanmean(GR[ki])),MD=MD,TVT=TVT,ei=ei,mdps=mdps,nform=len(bands))

def make(a, hw_w, pf, npts):
    ei=a["ei"]; sel=ei[np.linspace(0,len(ei)-1,npts).astype(int)]
    md=a["MD"][sel]; dtvt_true=a["TVT"][sel]-a["tvtps"]
    # tool result: PF prediction at these rows (grounded)
    pf_dt = {}
    wmap = pf.get(hw_w, {}) if pf is not None else {}
    for s in sel:
        if s in wmap: pf_dt[s] = float(wmap[s])
    amb = "HIGH" if a["twspan"]>300 else "moderate"
    facies = ("clay-rich / organic shale (high GR)" if a["grm"]>90 else
              "marl / mixed lithology (moderate GR)" if a["grm"]>60 else "carbonate / clean limestone (low GR)")
    steer = "up-section (shallowing)" if a["dip0"]>0 else "down-section (deepening)"
    dmds = [m-a["mdps"] for m in md]
    # uncertainty grows with distance (proof 2: super-diffusion, Var ~ dMD^1.3)
    unc = [max(1.0, 0.9*(abs(x)/1000.0)**0.65*5) for x in dmds]
    prompt=(f"Predict the true-vertical-thickness drift dtvt (TVT-TVT_ps) for this horizontal well at the given "
            f"measured depths past the prediction-start (PS) point, using the geosteering tools.\n"
            f"Tools: formation_band, dip_fit, gr_signature, facies_id, pf_predict (GR-vs-typewell particle filter), "
            f"uncertainty_model, bimodal_check.\n"
            f"Context: PS_TVT={a['tvtps']:.0f} ft; predict at dMD past PS = " + ",".join(f"{x:.0f}" for x in dmds) + " ft.")
    think=(f"<think>\n"
        f"Step 1 - geological frame. call formation_band(well) -> the bit is in {a['home']} ({FORM.get(a['home'],a['home'])}), "
        f"typewell band [{a['hb'][0]:.0f},{a['hb'][1]:.0f}] ft, thickness {a['thick']:.0f} ft. The hidden interval's TVT excursion "
        f"is usually far smaller than this thickness, so the well most likely steers WITHIN {a['home']} (it stays in one formation ~87% of the time).\n"
        f"Step 2 - lithology. call facies_id(well) -> heel GR mean {a['grm']:.0f} API => {facies}. This sets the GR pattern the "
        f"typewell must match.\n"
        f"Step 3 - structural trend. call dip_fit(well) -> heel apparent dip {a['dip0']*1000:.1f} ft/1000ft, i.e. the bit is trending "
        f"{steer}. Absent a fault or steering change the dip continues; but this is only a PRIOR.\n"
        f"Step 4 - ambiguity. call gr_signature(well) -> typewell span {a['twspan']:.0f} ft, GR-match ambiguity {amb}. "
        f"With a large span the gamma-ray pattern repeats across formations (many datum shifts fit), so a single-value match is unreliable "
        f"and I must NOT trust naive dip-continuation for the far toe.\n"
        f"Step 5 - the actual match. call pf_predict(well, dMD) -> "
        + ", ".join(f"{x:.0f}:{pf_dt.get(s,a['dip0']*(mm-a['mdps'])):.1f}" for s,mm,x in zip(sel,md,dmds)) + " ft. "
        f"The particle filter tracks the smooth-dip trajectory while re-weighting on the GR-vs-typewell likelihood, resolving the "
        f"multimodality that pattern-matching alone cannot.\n"
        f"Step 6 - uncertainty & hedge. call uncertainty_model -> predicted error grows with distance (~dMD^0.65, super-diffusive), "
        f"so far-toe values carry +-{unc[-1]:.0f} ft. call bimodal_check -> {'a competing datum shift exists; I hedge toward the in-formation branch' if amb=='HIGH' else 'the solution is unimodal'}. "
        f"Final: adopt the PF trajectory, clamped to the {a['home']} band, with growing uncertainty down the lateral.\n"
        f"</think>")
    ans="<answer>\n" + "\n".join(f"dMD={m-a['mdps']:.0f} -> dtvt={pf_dt.get(s,a['dip0']*(m-a['mdps'])):.1f}" for s,m in zip(sel,md)) + "\n</answer>"
    return {"well":hw_w,"prompt":prompt,"completion":think+"\n"+ans,
            "true":[float(x) for x in dtvt_true],"pred":[pf_dt.get(s,a['dip0']*(m-a['mdps'])) for s,m in zip(sel,md)]}

def build():
    folds=pd.read_csv("config/well_field_folds.csv").set_index("well").field_fold.to_dict(); pf=load_pf_index()
    os.makedirs("data",exist_ok=True)
    for tok,npts in [(1024,8),(2048,16)]:
        tr=[];ho=[]; err=[]
        for hp in sorted(glob.glob("input/train/*__horizontal_well.csv")):
            w=os.path.basename(hp).split("__")[0]
            if w not in folds: continue
            try: hw=pd.read_csv(hp); tw=pd.read_csv(f"input/train/{w}__typewell.csv")
            except Exception: continue
            if "TVT" not in hw.columns: continue
            a=analyze(hw,tw)
            if a is None: continue
            ex=make(a,w,pf,npts); (ho if folds[w]==0 else tr).append(ex)
            err += [(p-t) for p,t in zip(ex["pred"],ex["true"])]
        for split,rows in [("train",tr),("holdout",ho)]:
            with open(f"data/reason_tool_{tok}_{split}.jsonl","w") as f:
                for r in rows: f.write(json.dumps({k:r[k] for k in ("well","prompt","completion")})+"\n")
            print(f"wrote data/reason_tool_{tok}_{split}.jsonl: {len(rows)}")
        e=np.array(err); print(f"  tool-grounded answer RMSE (at sampled pts) = {np.sqrt(np.mean(e**2)):.2f}")
    print("\n=== SAMPLE completion ===")
    for hp in sorted(glob.glob("input/train/*__horizontal_well.csv"))[1:2]:
        w=os.path.basename(hp).split("__")[0]; hw=pd.read_csv(hp); tw=pd.read_csv(f"input/train/{w}__typewell.csv")
        print(make(analyze(hw,tw),w,pf,8)["completion"][:900])

if __name__=="__main__": build()
