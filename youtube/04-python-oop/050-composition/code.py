class Engine:
    def start(self):
        return "engine running"

class Car:
    def __init__(self):
        self.engine = Engine()   # Car HAS-A Engine
    def drive(self):
        return self.engine.start() + ", car moving"

c = Car()
print(c.drive())

# swap the part without touching Car's users
class ElectricEngine:
    def start(self):
        return "silent motor humming"

c.engine = ElectricEngine()
print(c.drive())
