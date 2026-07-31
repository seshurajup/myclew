# run: mypy script.py  — catches type errors WITHOUT running the code

def greet(name: str) -> str:
    return "Hello, " + name

def total(items: list[int]) -> int:
    return sum(items)

# mypy flags these mistakes before runtime:
#   greet(42)          -> error: expected str, got int
#   total(["a", "b"])  -> error: expected list[int]

# correct usage type-checks cleanly
print(greet("Alice"))
print(total([1, 2, 3]))

# Optional makes "might be None" explicit
from typing import Optional
def find(x: int) -> Optional[str]:
    return "found" if x > 0 else None
