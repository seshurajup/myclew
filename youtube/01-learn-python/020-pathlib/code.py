from pathlib import Path

p = Path("data") / "report.txt"

print(p.name)
print(p.suffix)
print(p.stem)
print(p.parent)
print(p.with_suffix(".md"))
