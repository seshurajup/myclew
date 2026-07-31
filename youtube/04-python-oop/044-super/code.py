class Vehicle:
    def __init__(self, wheels):
        self.wheels = wheels
    def describe(self):
        return f"{self.wheels} wheels"

class Car(Vehicle):
    def __init__(self, brand):
        super().__init__(4)      # run the parent's __init__
        self.brand = brand
    def describe(self):
        base = super().describe()   # extend, not replace
        return f"{self.brand}: {base}"

c = Car("Toyota")
print(c.wheels)
print(c.describe())
