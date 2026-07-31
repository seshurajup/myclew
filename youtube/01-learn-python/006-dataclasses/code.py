from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int

    def dist(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5


p = Point(3, 4)
print(p)
print(p.dist())
