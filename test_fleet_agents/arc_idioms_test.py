"""arc_idioms_test — data-wise verifier for the arc-idioms catalogue TOOL. Parses a SYNTHETIC patterns.md
(self-contained) to check the band/idiom/exemplar parser + query ranking, and confirms the BUNDLED real
patterns.md parses to a non-trivial catalogue. Pure text parsing — no onnx needed.
"""
import os
import sys
import tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

SYNTH = """# High-Scoring ONNX Solution Patterns

## Band `22-23`

### Recurring ARC Families
- **Pure palette remap.** Same geometry, only channels move.

### ONNX Idioms
- **`Gather(axis=1)` palette table.** A length-10 output-channel vector is the canonical recolor op.

### Cost-Saving Rules
- Emit directly to graph `output`; a helper mask usually costs too much in this band.

### Useful Exemplars
| Task | Score | Cost | Pattern |
| `task016` | 22.697415 | 10 | One `Gather(axis=1)` color-negative palette transform. |

## Exact `25`

### ONNX Idioms
- **Terminal `Transpose`.** `Transpose(input -> output, perm=[0,1,3,2])` solves diagonal reflection at cost 0.

### Useful Exemplars
| Task | Score | Cost | Pattern |
| `task179` | 25.000000 | 0 | Fixed `3x3` main-diagonal reflection by terminal `Transpose`. |
"""


def _run():
    print("=== ARC-IDIOMS DATA-WISE VERIFIER ===")
    from fleet_agents import arc_idioms as I
    checks = {}

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "patterns.md")
        open(p, "w").write(SYNTH)
        cat = I.parse_patterns(p)
    kinds = {r["kind"] for r in cat}
    checks["parsed_bands"] = {r["band"] for r in cat} == {"22-23", "exact-25"}
    checks["has_onnx_idiom"] = "onnx_idiom" in kinds
    checks["has_exemplar"] = "exemplar" in kinds
    checks["has_family"] = "arc_family" in kinds
    ex = [r for r in cat if r["kind"] == "exemplar"]
    checks["exemplar_cost_parsed"] = any(r["task"] == "task016" and r["cost"] == 10 for r in ex)
    checks["exemplar_ops_parsed"] = any("Gather" in (r.get("ops") or []) for r in ex)

    # query ranking: a recolor signature surfaces the Gather palette idiom
    top = I.query(cat, {"tags": ["palette recolor same shape"], "ops": ["Gather"]}, top_k=3)
    checks["query_returns"] = len(top) >= 1
    checks["query_relevant"] = any("Gather" in t.get("title", "") or "palette" in t.get("title", "").lower()
                                   or "Gather" in (t.get("ops") or []) for t in top)

    # bundled REAL patterns.md parses to a rich catalogue
    real = I.parse_patterns()
    summ = I.summarize(real)
    checks["bundled_rich"] = summ["total"] >= 100 and len(summ["by_band"]) >= 5
    checks["bundled_has_exact25"] = any(b.startswith("exact") for b in summ["by_band"])

    # agent contract (empty spec → summary)
    status, res, to, msg = I.run({"question": "smoke", "spec": {}}, "test")
    checks["agent_contract"] = status == "done" and isinstance(res, dict) and res.get("n_idioms", 0) > 0

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    print(f"  -> synthetic parsed {len(cat)} records; bundled patterns.md → {summ['total']} idioms "
          f"across bands {sorted(summ['by_band'])}")
    ok = all(checks.values())
    print(f"=== arc-idioms: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
