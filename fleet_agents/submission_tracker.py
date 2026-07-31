"""submission_tracker — the day-over-day SUBMISSION LOG. Records every Kaggle submission with its timestamp,
the model/config behind it, the local CV, and (filled in later) the returned LB score + rerun runtime — so we
can (a) calibrate CV↔LB, (b) pace the daily slot budget, and (c) pick which past submissions to ENSEMBLE as
days accumulate. Appends to results/submissions.csv (git-tracked) + mirrors to the per-comp board.

  log(model, cv, notes, kernel=..., slot_of_day=..)      -> row id (call at submit time)
  set_score(row_id, lb, private=None, runtime_min=None)  -> fill the LB once it lands
  best(n)                                                 -> top-n by LB (or CV if LB missing) for ensembling
"""
from __future__ import annotations
import csv, datetime as _dt, os
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "results" / "submissions.csv"
COLS = ["id","ts_ist","model","cv","lb_public","lb_private","runtime_min","kernel","slot_of_day","notes"]

def _ist():
    return (_dt.datetime.utcnow()+_dt.timedelta(hours=5,minutes=30)).strftime("%Y-%m-%d %H:%M IST")

def _read():
    if not LOG.exists(): return []
    return list(csv.DictReader(LOG.open()))

def _write(rows):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in COLS})

def log(model, cv=None, notes="", kernel="", slot_of_day=None):
    rows=_read(); rid=f"sub{len(rows)+1:03d}"
    rows.append(dict(id=rid, ts_ist=_ist(), model=model, cv=cv if cv is not None else "",
                     lb_public="", lb_private="", runtime_min="", kernel=kernel,
                     slot_of_day=slot_of_day if slot_of_day is not None else len(rows)+1, notes=notes))
    _write(rows); print(f"[submission_tracker] logged {rid}: {model} cv={cv} slot={slot_of_day}")
    return rid

def set_score(rid, lb=None, private=None, runtime_min=None):
    rows=_read()
    for r in rows:
        if r["id"]==rid:
            if lb is not None: r["lb_public"]=lb
            if private is not None: r["lb_private"]=private
            if runtime_min is not None: r["runtime_min"]=runtime_min
    _write(rows); print(f"[submission_tracker] {rid} <- lb={lb} private={private} runtime={runtime_min}")

def best(n=5):
    def key(r):
        for k in ("lb_public","cv"):
            try: return float(r[k])
            except (TypeError,ValueError): continue
        return 1e9
    return sorted(_read(), key=key)[:n]


def _pairs():
    """(cv, lb_public) anchors from the log where BOTH are present — the CV↔LB evidence."""
    out=[]
    for r in _read():
        try: cv=float(r["cv"]); lb=float(r["lb_public"])
        except (TypeError,ValueError,KeyError): continue
        out.append((cv,lb))
    return out

def predict_lb(cv, extra_pairs=None):
    """Guess the Kaggle LB for a local `cv`, metric-agnostic, from this competition's logged (cv,lb) pairs.
    0 pairs -> None (no calibration yet); 1 -> constant offset (lb=cv+off); >=2 -> Theil-Sen robust line.
    Returns dict(pred, model, n, confidence). Reusable across competitions (per-comp submissions.csv)."""
    import statistics as _st
    pts=_pairs()+list(extra_pairs or [])
    if not pts: return {"pred":None,"model":"none","n":0,"confidence":"none"}
    if len(pts)==1:
        cv0,lb0=pts[0]; off=lb0-cv0
        return {"pred":round(cv+off,4),"model":f"offset {off:+.3f}","n":1,"confidence":"low"}
    xs=[a for a,_ in pts]; ys=[b for _,b in pts]
    slopes=[(ys[j]-ys[i])/(xs[j]-xs[i]) for i in range(len(pts)) for j in range(i+1,len(pts)) if abs(xs[j]-xs[i])>1e-9]
    slope=_st.median(slopes) if slopes else 1.0
    inter=_st.median([y-slope*x for x,y in pts])
    resid=_st.median([abs(y-(slope*x+inter)) for x,y in pts])
    conf="high" if (len(pts)>=4 and resid<0.3) else ("med" if len(pts)>=2 else "low")
    return {"pred":round(slope*cv+inter,4),"model":f"lb≈{slope:.3f}·cv{inter:+.3f}","n":len(pts),
            "confidence":conf,"resid":round(resid,3)}


def shakeup_band(field_disjoint_cv, interspersed_cv=None, extra_pairs=None):
    """The PRIVATE-LB RISK BAND from our own data (no private scores needed). Two bounds:
      best_case  = public-calibrated LB for the interspersed CV (private behaves like the easy public test)
      worst_case = the FIELD-DISJOINT CV itself (private holds out whole fields = Deotte shakeup)
    Returns dict(best, worst, mid, spread, verdict). A SMALL spread = shakeup-robust; LARGE = fragile.
    Reusable: pass this competition's field-disjoint CV (+ optional interspersed CV)."""
    icv = interspersed_cv if interspersed_cv is not None else field_disjoint_cv
    best = predict_lb(icv, extra_pairs=extra_pairs).get("pred")
    worst = float(field_disjoint_cv)
    if best is None: best = worst
    best = float(best); mid = round((best+worst)/2,3); spread = round(worst-best,3)
    verdict = ("shakeup-ROBUST" if spread < 1.0 else
               "moderate shakeup risk" if spread < 3.0 else "FRAGILE (large shakeup risk)")
    return {"best_case_lb": round(best,3), "worst_case_lb": worst, "mid": mid,
            "spread": spread, "verdict": verdict}

if __name__=="__main__":
    import sys
    print(best(10) if (len(sys.argv)>1 and sys.argv[1]=="best") else _read())
