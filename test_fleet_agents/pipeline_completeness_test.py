import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import pipeline_completeness as PC
import fleet_agents as F
def _run():
    print("=== PIPELINE-COMPLETENESS VERIFIER ===")
    checks={}
    rep=PC.audit(list(F.HANDLERS))
    checks["all_modalities"]=set(rep)==set(PC.PIPELINE)
    checks["stages_10"]=len(PC.STAGES)==10
    # with a full fleet, every modality should be self-sufficient
    checks["all_self_sufficient"]=all(v["self_sufficient"] for v in rep.values())
    # a stripped handler set should REVEAL gaps (audit actually detects missing stages)
    strip=[h for h in F.HANDLERS if h not in ("submission-build","submit-verify")]
    rep2=PC.audit(strip)
    checks["detects_gaps"]=any("submit" in v["gaps"] for v in rep2.values())
    checks["coverage_is_frac"]=all(0<=v["coverage"]<=1 for v in rep.values())
    for k,v in checks.items(): print(f"  {'OK' if v else 'X'} {k}")
    ok=all(checks.values()); print(f"\n=== pipeline-completeness: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ==="); return ok
if __name__=="__main__": sys.exit(0 if _run() else 1)
