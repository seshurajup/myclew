import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
slug,msg=sys.argv[1],sys.argv[2]
api=KaggleApi(); api.authenticate(); u,k=slug.split("/")
req=ApiGetKernelRequest(); req.user_name=u; req.kernel_slug=k
v=api.build_kaggle_client().kernels.kernels_api_client.get_kernel(req).metadata.current_version_number
api.competition_submit_code(file_name="submission.csv",message=msg,
  competition="rogii-wellbore-geology-prediction",kernel=slug,kernel_version=v)
print(f"SUBMITTED {slug} v{v}: {msg}")
