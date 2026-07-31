import sys,time,subprocess
KG="/home/seshu/miniconda3/envs/llm/bin/kaggle"
slug,msg=sys.argv[1],sys.argv[2]
def status():
    r=subprocess.run([KG,"kernels","status",slug],capture_output=True,text=True); return r.stdout+r.stderr
t0=time.time()
while True:
    st=status()
    if "COMPLETE" in st: break
    if "ERROR" in st: print(f"{slug} COMMIT ERROR after {int(time.time()-t0)}s",flush=True); raise SystemExit
    time.sleep(90)
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
api=KaggleApi(); api.authenticate(); u,k=slug.split("/")
req=ApiGetKernelRequest(); req.user_name=u; req.kernel_slug=k
v=api.build_kaggle_client().kernels.kernels_api_client.get_kernel(req).metadata.current_version_number
api.competition_submit_code(file_name="submission.csv",message=msg,
  competition="rogii-wellbore-geology-prediction",kernel=slug,kernel_version=v)
print(f"[{int(time.time()-t0)}s] SUBMITTED {slug} v{v}",flush=True)
