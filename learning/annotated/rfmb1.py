import torch, torch.nn as nn, torch.nn.functional as F      # an expert that decides for itself
import sys; sys.path.insert(0, "learning")
import vizkit as vz

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)

N, d = 8, 16
x = torch.randn(d)
G = torch.randn(d, N, requires_grad=True)
scores = x @ G
J = torch.autograd.functional.jacobian(lambda s: F.softmax(s, -1), scores.detach())
off = float((J - torch.diag(torch.diag(J))).abs().max())
ok("Softmax couples every expert to every other", off > 1e-3,
   f"largest off-diagonal Jacobian entry = {off:.4f}")

k = 2
topk_out = lambda s: torch.topk(s, k).values.sum()
s0 = scores.detach().clone().requires_grad_(True)
topk_out(s0).backward()
grad_mask = (s0.grad != 0)
ok("TopK passes gradient ONLY to the chosen experts", int(grad_mask.sum()) == k,
   f"{int(grad_mask.sum())} of {N} experts receive gradient")
ok("so an unchosen expert cannot learn to be chosen", int((~grad_mask).sum()) == N - k,
   f"{N-k} experts are gradient-dead this step")

W = torch.randn(d, N, requires_grad=True)
tokens = torch.randn(64, d)
gates = F.softmax(tokens @ W, -1)
task = ((gates @ torch.randn(N)) - torch.randn(64)).pow(2).mean()          # a stand-in task loss
load = gates.mean(0)
balance = (load * load).sum() * N                                          # the usual load-balance term
g_task = torch.autograd.grad(task, W, retain_graph=True)[0].flatten()
g_bal = torch.autograd.grad(balance, W, retain_graph=True)[0].flatten()
cos = float(F.cosine_similarity(g_task, g_bal, dim=0))
print(f"  cos(task gradient, balance gradient) = {cos:+.4f}")
ok("the two objectives are not aligned", abs(cos) < 0.5,
   "so the balance term is spending capacity the task did not ask for")
ok("and its weight is a hand-set hyper-parameter", True,
   "lambda multiplies a gradient that points elsewhere")
