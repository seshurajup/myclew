class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, n):
        return n * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(10))    # calls __call__
print(triple(10))

# a callable object can keep state between calls
class Counter:
    def __init__(self):
        self.count = 0
    def __call__(self):
        self.count += 1
        return self.count

c = Counter()
print(c(), c(), c())
