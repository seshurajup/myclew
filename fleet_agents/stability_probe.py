"""stability_probe — the reusable STABILITY / anti-leak harness. Before any surprising CV win is trusted, this
agent proves it is honest and stable, not a leak or a lucky seed. Model-agnostic: the caller passes ONE
`run_fn(shuffle_target: bool, ablate: list[str]|None, seed: int) -> rmse` closure (train+eval, field-grouped),
and the probe runs:

  • null_test      — shuffle the target (break the input↔target link). An HONEST pipeline must then score no
                     better than the trivial null baseline. If the shuffled model still scores well → LEAK.
                     (Georgy Mamarin's "fork the ruler" wall-test: a real feature clears the null, a leak doesn't.)
  • ablation       — zero each feature group in turn; reports each group's RMSE contribution so a single
                     feature isn't secretly carrying the whole (possibly leaky) signal.
  • seed_stability — re-run with several seeds; reports mean ± std so a "win" isn't one lucky init.

Returns (ok, checks[list of (name, ok, detail)]) like leak_audit. Reusable across competitions — the harness
knows nothing about the model. A math-governance skill: extends the fleet's math_master validation toolkit.
"""
from __future__ import annotations
import numpy as np


class StabilityProbe:
    name = "stability-probe"
    kind = "finding"

    def null_test(self, run_fn, null_baseline: float, tol: float = 0.12, seed: int = 0):
        """Honest if shuffled-target RMSE is NOT meaningfully better than the trivial null baseline."""
        r = float(run_fn(shuffle_target=True, ablate=None, seed=seed))
        ok = r >= null_baseline * (1.0 - tol)
        return ("null-target test (shuffled ≥ null baseline)", ok,
                f"shuffled RMSE={r:.2f} vs null {null_baseline:.2f} — "
                + ("no learning from noise (HONEST)" if ok else "STILL LEARNS SHUFFLED → LEAK"))

    def ablation(self, run_fn, groups: list[str], full_rmse: float, seed: int = 0):
        checks = []
        for g in groups:
            r = float(run_fn(shuffle_target=False, ablate=[g], seed=seed))
            checks.append((f"ablate[{g}] contribution", True,
                           f"RMSE {full_rmse:.2f} → {r:.2f} (Δ={r-full_rmse:+.2f})"))
        return checks

    def seed_stability(self, run_fn, seeds=(0, 1, 2)):
        rs = [float(run_fn(shuffle_target=False, ablate=None, seed=s)) for s in seeds]
        mu, sd = float(np.mean(rs)), float(np.std(rs))
        return ("seed stability (std small)", sd < 0.5,
                f"{mu:.2f} ± {sd:.2f} over {len(seeds)} seeds ({[round(x,2) for x in rs]})")

    def probe(self, run_fn, null_baseline: float, full_rmse: float,
              groups: list[str] | None = None, seeds=(0, 1, 2)) -> tuple[bool, list]:
        checks = [self.null_test(run_fn, null_baseline)]
        checks.append(self.seed_stability(run_fn, seeds))
        if groups:
            checks += self.ablation(run_fn, groups, full_rmse)
        ok = all(c[1] for c in checks)
        return ok, checks

    @staticmethod
    def print_report(ok, checks, title=""):
        print(f"\n🧪 STABILITY PROBE{(' — ' + title) if title else ''}")
        for name, passed, detail in checks:
            print(f"  {'✅' if passed else '❌'} {name} · {detail}")
        print(f"  {'✅ STABLE & HONEST' if ok else '❌ UNSTABLE / LEAK — do not trust this CV'}")
        return ok

    # ---- BaseAgent-style run: caller puts a run_fn + params in the spec ----
    def run(self, q, worker=None):
        spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
        run_fn = spec.get("run_fn"); nb = spec.get("null_baseline", 15.9)
        full = spec.get("full_rmse"); groups = spec.get("groups")
        ok, checks = self.probe(run_fn, nb, full, groups, spec.get("seeds", (0, 1, 2)))
        return ("done", {"ok": ok, "checks": checks}, "all",
                f"stability probe: {'PASS' if ok else 'FAIL'}")

    def verify(self):
        """DATA-WISE self-test: a KNOWN-LEAK model must FAIL null_test, a HONEST model must PASS."""
        rng = np.random.default_rng(0)
        def honest_fn(shuffle_target, ablate, seed):
            # signal model: predicts from a real feature; on shuffled target it degrades to null
            return 15.9 if shuffle_target else 6.0
        def leaky_fn(shuffle_target, ablate, seed):
            return 6.0  # scores great even on shuffled target → leak
        n_honest = self.null_test(honest_fn, 15.9)[1]
        n_leak = self.null_test(leaky_fn, 15.9)[1]
        return bool(n_honest and not n_leak)


if __name__ == "__main__":
    sp = StabilityProbe()
    print("self-test verify():", sp.verify())
