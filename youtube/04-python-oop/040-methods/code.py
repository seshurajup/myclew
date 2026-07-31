class Circle:
    pi = 3.14159

    def __init__(self, radius):
        self.radius = radius

    # instance method: uses self
    def area(self):
        return Circle.pi * self.radius ** 2

    # class method: uses cls, great for factories
    @classmethod
    def unit(cls):
        return cls(1)

    # static method: no self, no cls
    @staticmethod
    def describe():
        return "A round shape"

c = Circle(2)
print(round(c.area(), 2))
print(Circle.unit().radius)
print(Circle.describe())
