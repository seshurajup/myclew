import math, random, torch, torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

# a "program" is a dict of knobs (stand-in for a Markdown file's directives)
skill = {"retries": 1, "temperature": 0.9, "verify": 0}
meta = {"edit_size": 1, "explore": 0.5, "verify": 0}          # the EDITOR, in the same representation

def apply_edit(program, editor, rng):
    """one edit of `program`, parameterised by `editor` — and `editor` is itself a program."""
    out = dict(program)
    k = rng.choice(sorted(out))
    step = editor["edit_size"] * (1 if rng.random() > editor["explore"] else -1)
    out[k] = out[k] + step
    return out

rng = random.Random(0)
s1 = apply_edit(skill, meta, rng)
ok("the pipeline edits a TASK skill", s1 != skill, f"{skill} -> {s1}")
m1 = apply_edit(meta, meta, rng)                              # the same operator, applied to ITSELF
ok("and the SAME pipeline edits the META-skill", m1 != meta, f"{meta} -> {m1}")
ok("because both are the same type of object", set(type(v) for v in skill.values()) ==
   set(type(v) for v in meta.values()),
   "representation closure: no second mechanism was needed for recursion")
s2 = apply_edit(skill, m1, rng)
ok("an improved meta-skill then produces DIFFERENT task-skill edits", s2 != s1,
   "which is the entire causal chain the slow loop is betting on")
