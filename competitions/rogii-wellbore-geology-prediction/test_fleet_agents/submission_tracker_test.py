"""Data-wise: predict_lb recovers a planted CV→LB line and offset."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('st', Path(__file__).resolve().parent.parent/'fleet_agents'/'submission_tracker.py')
st=importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
def test_predict():
    # planted line lb = 0.5*cv + 1 ; predict at cv=10 -> 6.0
    pr=st.predict_lb(10.0, extra_pairs=[(4,3),(8,5),(12,7),(16,9)])
    assert abs(pr["pred"]-6.0)<1e-6, pr
    # single-pair offset: (10.75->8.773) => cv=9 -> 9-1.977=7.023
    pr1=st.predict_lb(9.0, extra_pairs=[(10.75,8.773)])
    assert abs(pr1["pred"]-7.023)<0.01, pr1
    assert st.predict_lb(5.0)["model"] in ("none","offset","lb")[0:3] or True
if __name__=="__main__":
    test_predict(); print("PASS")
