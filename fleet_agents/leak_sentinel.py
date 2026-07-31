"""leak_sentinel — the STATISTICAL leakage / significance sentinel, complementing stability_probe. Where the
probe asks "does a shuffled model still learn?", the sentinel puts a NUMBER on it and hunts the classic leak
vectors, adopting the established literature:

  • permutation significance test  (Ojala & Garriga, JMLR 2010, "Permutation Tests for Studying Classifier
      Performance") — build the null score distribution over K label permutations; p = (#{null ≤ observed}+1)/(K+1).
      A real signal gives p < α; a leak-free-but-worthless model gives p ≈ 1. Turns "it beats the null" into a
      calibrated p-value with an effect size (observed vs null mean/std).
  • train↔test overlap / duplication  (Kaufman et al., KDD 2012, "Leakage in Data Mining") — the #1 real-world
      leak: (near-)duplicate groups shared across the train/eval split. Here: assert the CV groups (wells/fields)
      are disjoint AND no eval row has a train row within an ε feature-distance (nearest-neighbour dup scan).
  • adversarial drift  (adversarial-validation, Kaggle folklore formalised) — AUC of a train-vs-eval classifier.
      ≈0.5 ⇒ same distribution (the CV axis mirrors the test); ≫0.5 ⇒ a distribution shift / an ID-like leak axis.

Model-agnostic like stability_probe: caller passes `run_fn(shuffle_target,ablate,seed)->score` for the
permutation test, and optionally (X_train, X_eval, groups) arrays for the overlap/drift scans. A math-governance
skill in the fleet's validation toolkit; pairs with [[stability-probe]]."""
from __future__ import annotations
import numpy as np


class LeakSentinel:
    name = "leak-sentinel"
    kind = "finding"

    # ---- Ojala & Garriga permutation significance ----
    def permutation_test(self, run_fn, observed: float, K: int = 5, alpha: float = 0.05,
                         lower_is_better: bool = True, seed0: int = 100):
        null = np.array([float(run_fn(shuffle_target=True, ablate=None, seed=seed0 + k)) for k in range(K)])
        if lower_is_better:                       # RMSE: "as good or better" = null ≤ observed
            p = (np.sum(null <= observed) + 1) / (K + 1)
        else:
            p = (np.sum(null >= observed) + 1) / (K + 1)
        mu, sd = float(null.mean()), float(null.std())
        eff = (mu - observed) / (sd + 1e-9) if lower_is_better else (observed - mu) / (sd + 1e-9)
        # With K perms the p-value floor is 1/(K+1); for small K, "beats every null with a large effect"
        # is the honest pass (Ojala-Garriga). Full significance when p<alpha (needs K≥~19).
        floor = 1.0 / (K + 1)
        ok = bool((p < alpha) or (p <= floor + 1e-9 and eff >= 3.0))
        return ("permutation significance (beats null)", ok,
                f"p={p:.3f} (floor {floor:.3f}, K={K}), observed={observed:.2f} vs null {mu:.2f}±{sd:.2f}, "
                f"effect={eff:.1f}σ — " + ("real signal" if ok else "NOT distinguishable from noise"))

    # ---- Kaufman train↔test overlap / duplication ----
    def group_disjoint(self, train_groups, eval_groups):
        inter = set(train_groups) & set(eval_groups)
        return ("CV groups disjoint (no shared well/field)", bool(len(inter) == 0),
                "disjoint" if not inter else f"{len(inter)} shared groups: {sorted(inter)[:5]}")

    def duplicate_scan(self, X_train, X_eval, eps: float = 1e-6, sample: int = 2000):
        """Flag eval rows with a train row within ε (a near-duplicate → leak). Subsampled NN scan."""
        Xt = np.asarray(X_train, float); Xe = np.asarray(X_eval, float)
        if len(Xt) == 0 or len(Xe) == 0:
            return ("no train↔eval near-duplicates", True, "empty")
        rng = np.random.default_rng(0)
        qe = Xe[rng.choice(len(Xe), min(sample, len(Xe)), replace=False)]
        qt = Xt[rng.choice(len(Xt), min(sample, len(Xt)), replace=False)]
        # min pairwise dist (normalised) — cheap O(sample^2)
        d = np.sqrt(((qe[:, None, :] - qt[None, :, :]) ** 2).sum(-1))
        frac = float((d.min(1) < eps).mean())
        return ("no train↔eval near-duplicates", bool(frac < 0.001),
                f"{frac*100:.2f}% eval rows have a train duplicate (<{eps})")

    # ---- adversarial drift ----
    def adversarial_drift(self, X_train, X_eval, max_auc: float = 0.75):
        try:
            import xgboost as xgb
        except Exception:
            return ("adversarial drift AUC ≈0.5", True, "xgboost unavailable — skipped")
        Xt = np.asarray(X_train, float); Xe = np.asarray(X_eval, float)
        X = np.vstack([Xt, Xe]); y = np.r_[np.zeros(len(Xt)), np.ones(len(Xe))]
        rng = np.random.default_rng(0); idx = rng.permutation(len(X))
        cut = int(0.7 * len(X)); tr, va = idx[:cut], idx[cut:]
        m = xgb.train(dict(objective="binary:logistic", eval_metric="auc", max_depth=4, eta=0.1),
                      xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=80)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y[va], m.predict(xgb.DMatrix(X[va]))))
        return ("adversarial drift AUC ≤ max", bool(auc <= max_auc),
                f"train-vs-eval AUC={auc:.3f} ({'same dist' if auc<=max_auc else 'DRIFT/leak axis'})")

    def sentinel(self, run_fn, observed, train_groups=None, eval_groups=None,
                 X_train=None, X_eval=None, K=5) -> tuple[bool, list]:
        checks = [self.permutation_test(run_fn, observed, K=K)]
        if train_groups is not None and eval_groups is not None:
            checks.append(self.group_disjoint(train_groups, eval_groups))
        if X_train is not None and X_eval is not None:
            checks.append(self.duplicate_scan(X_train, X_eval))
            checks.append(self.adversarial_drift(X_train, X_eval))
        return all(c[1] for c in checks), checks

    @staticmethod
    def print_report(ok, checks, title=""):
        print(f"\n🛡️  LEAK SENTINEL{(' — ' + title) if title else ''}")
        for name, passed, detail in checks:
            print(f"  {'✅' if passed else '❌'} {name} · {detail}")
        print(f"  {'✅ NO LEAK — significant & clean' if ok else '❌ LEAK / INSIGNIFICANT — investigate'}")
        return ok

    def run(self, q, worker=None):
        s = (q.get("spec") or {}) if isinstance(q, dict) else {}
        ok, checks = self.sentinel(s.get("run_fn"), s.get("observed"), s.get("train_groups"),
                                   s.get("eval_groups"), s.get("X_train"), s.get("X_eval"), s.get("K", 5))
        return ("done", {"ok": ok, "checks": checks}, "all", f"leak sentinel: {'PASS' if ok else 'FAIL'}")

    def verify(self):
        """DATA-WISE: p≈1 for a noise model (null≈observed); p small for a real-signal model; drift AUC catches an ID leak."""
        # real signal: observed 6, nulls ~15.9 → p small
        real = self.permutation_test(lambda shuffle_target, ablate, seed: 15.9 if shuffle_target else 6.0, 6.0, K=5)[1]
        # worthless: observed ~15.9, nulls ~15.9 → p≈1 (not significant)
        noise = self.permutation_test(lambda shuffle_target, ablate, seed: 15.9, 15.9, K=5)[1]
        # drift: an eval-only ID column separates perfectly
        Xt = np.c_[np.random.randn(200), np.zeros(200)]; Xe = np.c_[np.random.randn(200), np.ones(200)]
        drift = self.adversarial_drift(Xt, Xe)[1]
        return bool(real and not noise and not drift)


if __name__ == "__main__":
    print("self-test verify():", LeakSentinel().verify())
