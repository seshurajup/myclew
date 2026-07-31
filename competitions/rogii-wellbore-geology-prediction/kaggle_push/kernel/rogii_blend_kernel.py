# rogii SOFT-ANCHOR (dense formation surface + multi-seed blend_AB, 2xT4). NBASE seeds averaged.
import sys, os, shutil, glob
os.environ["NBASE"] = os.environ.get("NBASE", "6")
def fb():
    for p in glob.glob("/kaggle/input/**/geology_structural.py", recursive=True): return os.path.dirname(p)
    raise FileNotFoundError("bundle not found")
B=fb(); print("BUNDLE",B,"NBASE",os.environ["NBASE"])
work="/kaggle/working/pkg"; os.makedirs(work+"/fleet_agents",exist_ok=True)
for f in ("geology_trackB.py","geology_honest.py","geology_structural.py"): shutil.copy(f"{B}/{f}", work+"/fleet_agents/"+f)
open(work+"/fleet_agents/__init__.py","w").close()
shutil.copy(f"{B}/rogii_soft_submit.py", work+"/rogii_soft_submit.py")
if os.path.isdir(B+"/models"): shutil.copytree(B+"/models", work+"/models", dirs_exist_ok=True)
sys.path.insert(0,work)
import rogii_soft_submit as R
R.HERE=work; R.NBASE=int(os.environ["NBASE"])
sub=R.main(out="/kaggle/working/submission.csv"); print("DONE",sub.shape)
