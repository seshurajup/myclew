import torch, torch.nn as nn, torch.nn.functional as F      # three levels: server, client, memory
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

def unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)

d = 16
def client_solution(A, b):
    return torch.linalg.lstsq(A, b).solution

def experiment(angle):
    w1 = unit(d)
    w2 = F.normalize(torch.cos(torch.tensor(angle)) * w1
                     + torch.sin(torch.tensor(angle)) * unit(d), dim=0)
    X1, X2 = torch.randn(256, d), torch.randn(256, d)
    y1, y2 = X1 @ w1, X2 @ w2
    s1, s2 = client_solution(X1, y1), client_solution(X2, y2)
    avg = 0.5 * (s1 + s2)                                       # FedAvg
    L1 = lambda w: float(((X1 @ w - y1) ** 2).mean())           # each client's OWN loss
    L2 = lambda w: float(((X2 @ w - y2) ** 2).mean())
    return L1(s1), L1(avg), L2(s2), L2(avg)

for ang in (0.0, 0.6, 1.2, 1.57):
    o1, a1, o2, a2 = experiment(ang)
    print(f"  divergence {ang:.2f} rad:  client1 own {o1:.4f} -> FedAvg {a1:.4f} | "
          f"client2 own {o2:.4f} -> FedAvg {a2:.4f}")
o1, a1, o2, a2 = experiment(1.57)
ok("FedAvg is much worse than each client's own solution when the tasks are orthogonal",
   a1 > 100 * (o1 + 1e-9) and a2 > 100 * (o2 + 1e-9),
   f"client1 {o1:.2e} -> {a1:.3f}, client2 {o2:.2e} -> {a2:.3f}")
i0, i1, i2, i3 = experiment(0.0)
ok("and harmless when the clients agree", i1 < 1e-6, f"identical tasks: FedAvg loss {i1:.2e}")
print("the average of two good models is not a good model - that is client drift")

k, v = unit(d), torch.randn(d)
S = torch.zeros(d, d)
beta = 1.0
S = S + beta * torch.outer(v - S @ k, k)                        # one delta-rule step (eq. 3/9)
ok("a memory adapts with no gradient step on theta at all",
   close(S @ k, v, 1e-5), "zero-shot test-time adaptation")
ok("and it costs O(d^2) state, not a growing cache", tuple(S.shape) == (d, d))
print("so the parameters can encode the RULE while the memory encodes the client's data")
