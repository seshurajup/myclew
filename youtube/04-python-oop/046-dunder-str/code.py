class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

p = Point(3, 4)
print(p)          # uses __str__
print(str(p))
print(repr(p))    # uses __repr__

points = [Point(1, 2), Point(5, 6)]
print(points)     # list uses __repr__
