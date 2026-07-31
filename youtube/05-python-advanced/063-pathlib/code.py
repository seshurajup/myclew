from pathlib import Path

p = Path("folder") / "data" / "file.txt"
print(p)                 # OS-correct path
print(p.name, p.suffix, p.stem)
print(p.parent)

# check existence and properties
here = Path(".")
print(here.exists(), here.is_dir())

# list files matching a pattern
for py in Path(".").glob("*.py"):
    print(py.name)
