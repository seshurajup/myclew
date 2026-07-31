from dataclasses import dataclass, field

@dataclass
class Product:
    name: str
    price: float
    tags: list = field(default_factory=list)

p = Product("Book", 12.99)
print(p)                 # auto __repr__
print(p == Product("Book", 12.99))  # auto __eq__

p.tags.append("sale")
print(p.tags)

# frozen dataclasses are immutable
@dataclass(frozen=True)
class Point:
    x: int
    y: int

print(Point(1, 2))
