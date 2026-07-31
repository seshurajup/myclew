class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        return "some sound"

class Dog(Animal):
    def speak(self):
        return "woof"

class Cat(Animal):
    def speak(self):
        return "meow"

d = Dog("Rex")
c = Cat("Bella")
print(d.name, d.speak())
print(c.name, c.speak())

# a Dog IS an Animal
print(isinstance(d, Animal))
