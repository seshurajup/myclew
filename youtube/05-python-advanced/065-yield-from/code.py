def letters():
    yield "a"
    yield "b"

def numbers():
    yield 1
    yield 2

def combined():
    yield from letters()   # delegate to another generator
    yield from numbers()

print(list(combined()))

# flatten a nested list with recursion + yield from
def flatten(items):
    for x in items:
        if isinstance(x, list):
            yield from flatten(x)
        else:
            yield x

print(list(flatten([1, [2, [3, 4]], 5])))
