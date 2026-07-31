class Point:
    __slots__ = ("x", "y")   # fixed set of attributes

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)
print(p.x, p.y)

# slots block new attributes, catching typos
try:
    p.z = 3
except AttributeError as e:
    print("blocked:", e)

# and they save memory on many objects
print(hasattr(p, "__dict__"))
