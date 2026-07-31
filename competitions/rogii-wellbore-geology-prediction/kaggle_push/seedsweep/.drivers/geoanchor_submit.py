import time,subprocess
KG="/home/seshu/miniconda3/envs/llm/bin/kaggle"; COMP="rogii-wellbore-geology-prediction"
slug="seshurajup/rogii-geoanchor"
msg="geoanchor UNCHANGED (lucifer19 v9 6.498 frontier: +v10-fresh +tabicl +model-pkg correction, 2xT4, hidden-test rerun)"
def status():
    r=subprocess.run([KG,"kernels","status",slug],capture_output=True,text=True); return r.stdout+r.stderr
t0=time.time()
while True:
    st=status()
    if "COMPLETE" in st: break
    if "ERROR" in st: print("geoanchor COMMIT ERROR",flush=True); raise SystemExit
    time.sleep(90)
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
api=KaggleApi(); api.authenticate(); u,k=slug.split("/")
v=api.build_kaggle_client().kernels.kernels_api_client.get_kernel(ApiGetKernelRequest(user_name=u,kernel_slug=k)).current_version_number
api.competition_submit_code(file_name="submission.csv",message=msg,competition=COMP,kernel=slug,kernel_version=v)
print(f"[{int(time.time()-t0)}s] SUBMITTED geoanchor v{v}",flush=True)
