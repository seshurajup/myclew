"""arch_catalog_test — data-wise verifier for the MODERN-TECHNIQUE catalog added to arch_builder.
Asserts propose(target) COMPOSES the MEASURED session constraints, not generic hype:
  • T4/offline target → FP4/NVFP4 is EXCLUDED (Blackwell-only) and int8-w8a8 is recommended.
  • sub-2-bit budget → qat-bitnet-ternary is REQUIRED (ternary PTQ collapses; QAT recovers).
  • sparse-label regime → trust-region-self-train (weak-to-strong under a firm anchor).
  • heterogeneous regime → moe-conditional-compute.
  • the LOEO/mini-first GATE + hardware-tune config are ALWAYS emitted.
Pure — no torch/data needed."""
import os
import sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import arch_builder as ab


def _names(recs):
    return {r["name"] for r in recs}


def test_t4_excludes_fp4_prefers_int8():
    p = ab.propose({"hardware": "t4", "bit_budget": 8})
    rec, exc = _names(p["recommended"]), {e["name"] for e in p["excluded"]}
    assert "fp4-nvfp4" in exc, f"FP4 must be EXCLUDED on T4, got excluded={exc}"
    assert "int8-w8a8" in rec, f"int8 must be recommended on T4, got rec={rec}"
    assert "fp4-nvfp4" not in rec, "FP4 must never be recommended on T4"
    assert any("int8" in s and "FP4" in s for s in p["substitutions"]), "T4 substitution note missing"
    return True


def test_subtwobit_requires_qat():
    p = ab.propose({"hardware": "t4", "bit_budget": 1.58})
    rec = _names(p["recommended"])
    assert "qat-bitnet-ternary" in rec, f"sub-2-bit must REQUIRE ternary QAT, got {rec}"
    assert any("QAT" in s or "qat" in s for s in p["substitutions"]), "sub-2-bit QAT substitution note missing"
    # and the constraint text must carry the MEASURED collapse fact
    tern = next(r for r in p["recommended"] if r["name"] == "qat-bitnet-ternary")
    assert "collaps" in tern["constraint"].lower() and "9.28" in tern["measured"], "ternary measured constraint missing"
    return True


def test_sparse_label_gets_trust_region():
    p = ab.propose({"hardware": "t4", "data_regime": "sparse_label"})
    rec = _names(p["recommended"])
    assert "trust-region-self-train" in rec, f"sparse-label must get trust-region self-train, got {rec}"
    tr = next(r for r in p["recommended"] if r["name"] == "trust-region-self-train")
    assert "retract" in tr["measured"].lower(), "trust-region must cite the retracted red-herring measurement"
    return True


def test_heterogeneous_gets_moe():
    p = ab.propose({"hardware": "5090", "data_regime": "heterogeneous"})
    assert "moe-conditional-compute" in _names(p["recommended"]), "heterogeneous regime must propose MoE"
    # on 5090 FP4 is NOT excluded (Blackwell)
    assert "fp4-nvfp4" not in {e["name"] for e in p["excluded"]}, "FP4 must not be excluded on Blackwell/5090"
    return True


def test_gate_and_hwtune_always():
    p = ab.propose({})           # defaults = biohub competition target (T4, sparse-label)
    rec = _names(p["recommended"])
    assert "hardware-tune-config" in rec and "gm-training-tricks" in rec, "always-on training configs missing"
    assert "mini-first-loeo-gate" in rec, "the LOEO gate must always be recommended"
    assert p["gate"]["name"] == "mini-first-loeo-gate" and "LOEO" in p["gate"]["rule"].upper() or "disjoint" in p["gate"]["rule"], \
        "propose() must emit the keep-if-improves LOEO gate"
    # default target must be T4 + sparse-label (the competition), so FP4 excluded there too
    assert "fp4-nvfp4" in {e["name"] for e in p["excluded"]}, "default (T4) target must exclude FP4"
    return True


def test_catalog_shape():
    cat = ab.catalog()
    assert len(cat) >= 12, f"catalog too small: {len(cat)}"
    cats = {e["category"] for e in cat}
    assert cats == {"architecture", "quantization", "training", "gate"}, f"unexpected categories {cats}"
    # every entry must carry the load-bearing fields
    for e in cat:
        for k in ("name", "what", "when", "constraint", "plugs_in", "fleet_agent", "source", "measured"):
            assert e.get(k), f"entry {e.get('name')} missing field {k}"
    # filtered view works
    assert all(e["category"] == "quantization" for e in ab.catalog("quantization"))
    return True


def test_handler_returns_proposal():
    s, res, to, msg = ab.catalog_query({"question": "t", "spec": {"target_profile": {"hardware": "t4", "bit_budget": 4}}}, "test")
    assert s == "done" and res["proposal"] is not None, "handler must return a proposal"
    assert "int8" in msg and ("FP4" in msg or "fp4" in msg), "handler message must name the int8/FP4 decision"
    return True


def _run():
    print("=== ARCH-CATALOG (modern-technique) DATA-WISE VERIFIER ===")
    tests = [test_t4_excludes_fp4_prefers_int8, test_subtwobit_requires_qat, test_sparse_label_gets_trust_region,
             test_heterogeneous_gets_moe, test_gate_and_hwtune_always, test_catalog_shape, test_handler_returns_proposal]
    ok = True
    for t in tests:
        try:
            r = t(); print(f"  {'✅' if r else '❌'} {t.__name__}")
            ok = ok and bool(r)
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}"); ok = False
    print(f"\n=== arch-catalog: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
