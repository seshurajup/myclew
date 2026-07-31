class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says woof!"

d = Dog("Rex", 3)
print(d.name)
print(d.age)
print(d.bark())

# each object is independent
d2 = Dog("Bella", 5)
print(d2.bark())
