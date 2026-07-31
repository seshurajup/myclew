"""Convert a legacy LESSON-dict .py lesson into a `.learning` file (same content)."""
import sys, importlib.util
from pathlib import Path


def to_learning(py_path: Path) -> str:
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    L = mod.LESSON
    out = []
    for k in ("id", "order", "title", "subtitle", "source"):
        if L.get(k) not in (None, ""):
            out.append(f"@ {k}: {L[k]}")
    out.append("")
    for sec in L["sections"]:
        out.append("--- note")
        out.append((sec.get("note") or "").strip("\n"))
        out.append("")
        if sec.get("code"):
            out.append("--- code")
            out.append(sec["code"].strip("\n"))
            out.append("")
        if sec.get("output"):
            out.append("--- output")
            out.append(sec["output"].strip("\n"))
            out.append("")
        if sec.get("image"):
            out.append("--- image")
            out.append(sec["image"].strip("\n"))
            out.append("")
        if sec.get("shape"):
            out.append("--- shape")
            out.append(sec["shape"].strip("\n"))
            out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        py = Path(p)
        text = to_learning(py)
        dst = py.with_suffix(".learning")
        dst.write_text(text)
        py.unlink()
        # sanity: parses?
        n_note = text.count("\n--- note")
        print(f"  {py.name} -> {dst.name}  ({n_note} sections, id in header: {'@ id:' in text})")
