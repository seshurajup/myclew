class Swimmer:
    def move(self):
        return "swimming"

class Flyer:
    def fly(self):
        return "flying"

class Duck(Swimmer, Flyer):
    def sound(self):
        return "quack"

d = Duck()
print(d.move())
print(d.fly())
print(d.sound())

# the Method Resolution Order shows the lookup path
print([c.__name__ for c in Duck.__mro__])
