def greet(name: str, times: int = 1) -> str:
    return (f"Hi {name}! " * times).strip()

scores: dict[str, int] = {"amy": 90, "ben": 82}
top: list[str] = [n for n, s in scores.items() if s >= 85]

print(greet("Sam"))
print(greet("Sam", 3))
print(top)
