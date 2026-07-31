def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

print(greet("Alice"))
print(add(3, 4))

# hints for containers and optional values
from typing import Optional

def total(prices: list[float]) -> float:
    return sum(prices)

def find(name: str) -> Optional[str]:
    users = {"alice": "admin"}
    return users.get(name)

print(total([1.5, 2.5, 3.0]))
print(find("alice"), find("bob"))
