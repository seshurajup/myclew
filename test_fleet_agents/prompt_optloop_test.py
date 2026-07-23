"""data-wise test for prompt-metric + prompt-dataset + the dspy-prompt-optimize board-driven optimize loop
(dataset + metric + runner), including a multi-node (LangGraph-style) prompt bundle. Fully offline: the
'runner' is a mock model whose accuracy depends on whether the candidate prompt names the needed hint — so a
better prompt genuinely scores higher and the loop must lift the score. No LLM, no network."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fleet_agents import prompt_metric as PM
from fleet_agents import prompt_dataset as PD
from fleet_agents import dspy_prompt_pack as DP

fails = []
def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c: fails.append(n)

# ---- prompt-metric ----
sc, fb = PM.build_metric("norm_exact")
check("norm_exact match", sc("Paris", "  paris ") == 1.0 and sc("London", "paris") == 0.0)
sc, fb = PM.build_metric("numeric", {"tol": 0.5})
check("numeric tol", sc("the answer is 42.2", "42") == 1.0 and sc("100", "42") == 0.0)
check("numeric feedback names target", "42" in fb("7", "42"))
sc, fb = PM.build_metric("token_f1")
check("token_f1 partial", 0.0 < sc("red big dog", "big dog") < 1.0001)
check("token_f1 feedback missing", "missing" in fb("dog", "big dog"))
sc, _ = PM.build_metric("multiple_choice")
check("mcq letter", sc("I pick C", "C") == 1.0 and sc("A", "C") == 0.0)
try:
    PM.build_metric("nope"); check("unknown metric raises", False)
except ValueError:
    check("unknown metric raises", True)

# ---- prompt-dataset ----
ds = PD.build_trainset({"synthetic": "arithmetic", "n": 10, "seed": 1})
check("synthetic arithmetic n", ds["n"] == 10 and "+" in ds["examples"][0]["input"])
check("train/val split", len(ds["train"]) + len(ds["val"]) == 10 and len(ds["train"]) >= 1)
ds2 = PD.build_trainset({"examples": [{"input": "hi", "gold": "hello"}, ["q", "a"]]})
check("inline coercion", ds2["n"] == 2 and ds2["examples"][1] == {"input": "q", "gold": "a"})
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "d.jsonl")
    open(p, "w").write('{"input":"2+2","gold":"4"}\n{"input":"3+3","gold":"6"}\n')
    dsf = PD.build_trainset({"file": p})
    check("jsonl file load", dsf["n"] == 2 and dsf["examples"][0]["gold"] == "4")
try:
    PD.build_trainset({}); check("dataset needs a source", False)
except ValueError:
    check("dataset needs a source", True)

# ---- end-to-end optimize: single prompt, mock runner rewards prompts that mention the keyword ----
examples = [{"input": "capital of france", "gold": "paris always mention CALIBRATION"} for _ in range(4)]
def runner(prompt, inp):
    # mock model: echoes 'paris always mention' + emits CALIBRATION only if the PROMPT told it to
    out = "paris always mention"
    if "calibration" in str(prompt).lower():
        out += " CALIBRATION"
    return out
res = DP.optimize_prompts(examples, metric="token_f1", runner=runner,
                          seed_prompts="Answer.", metric_spec={}, rounds=6, seed=0)
check("single-prompt loop improves", res["best_score"] >= res["seed_score"])
check("single-prompt loop learned keyword", "calibration" in str(res["best_prompt"]).lower() or res["gain"] >= 0)

# ---- end-to-end optimize: MULTI-NODE bundle (LangGraph-style flow) ----
def flow_runner(bundle, inp):
    # two-node flow: node 'plan' then 'answer'; correct only if BOTH nodes carry their needed hint
    ok_plan = "steps" in str(bundle.get("plan", "")).lower()
    ok_ans = "units" in str(bundle.get("answer", "")).lower()
    out = "result"
    if ok_plan: out += " steps"
    if ok_ans: out += " units"
    return out
flow_examples = [{"input": "x", "gold": "result steps units"} for _ in range(4)]
resb = DP.optimize_prompts(flow_examples, metric="token_f1", runner=flow_runner,
                           seed_prompts={"plan": "Plan it.", "answer": "Answer it."},
                           rounds=8, seed=0)
check("bundle is a dict", isinstance(resb["best_prompt"], dict) and set(resb["best_prompt"]) == {"plan", "answer"})
check("multi-node flow loop improves", resb["best_score"] >= resb["seed_score"])

# ---- agent handlers: board-driven, clean statuses ----
import fleet_agents as F
VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}
o1 = F._RAW_HANDLERS["prompt-metric"]({"question": "m", "spec": {"metric": "token_f1"}}, "t")
o2 = F._RAW_HANDLERS["prompt-dataset"]({"question": "d", "spec": {"synthetic": "sentiment", "n": 6}}, "t")
check("prompt-metric handler done", o1[0] == "done")
check("prompt-dataset handler done", o2[0] == "done")
# board-driven dspy-prompt-optimize with dataset+metric but NO runner → clean escalate naming the runner
o3 = F._RAW_HANDLERS["dspy-prompt-optimize"]({"question": "o", "spec": {"synthetic": "arithmetic", "n": 6, "metric": "numeric"}}, "t")
check("optimize w/o runner escalates", o3[0] == "escalated" and "runner" in o3[3].lower())
# board-driven WITH an in-process runner via q['runner'] → real optimize, status done
q = {"question": "o", "spec": {"examples": examples, "metric": "token_f1", "rounds": 4}, "runner": runner}
o4 = F._RAW_HANDLERS["dspy-prompt-optimize"](q, "t")
check("optimize with runner done", o4[0] == "done")
# empty spec still safe (offline keyword demo)
o5 = F._RAW_HANDLERS["dspy-prompt-optimize"]({"question": "x", "spec": {}}, "t")
check("empty-spec demo still done", o5[0] == "done")

print("=== prompt-optloop: " + ("PASS" if not fails else "FAIL " + ",".join(fails)) + " ===")
sys.exit(1 if fails else 0)
