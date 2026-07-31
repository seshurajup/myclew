class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

a = Vector(1, 2)
b = Vector(3, 4)
print(a + b)          # calls __add__
print(a == Vector(1, 2))  # calls __eq__
