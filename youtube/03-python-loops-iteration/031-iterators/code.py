nums = [10, 20, 30]

it = iter(nums)
print(next(it))
print(next(it))
print(next(it))

# a for loop is really iter + next under the hood
it2 = iter("hi")
while True:
    try:
        ch = next(it2)
        print(ch)
    except StopIteration:
        break

# any object with __iter__ and __next__ is iterable
class Countdown:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n + 1

print(list(Countdown(3)))
