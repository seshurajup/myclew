numbers = [1, 2, 3, 4, 5, 6]

doubled = [x * 2 for x in numbers]
print(doubled)

evens = [x for x in numbers if x % 2 == 0]
print(evens)

squared_evens = [x ** 2 for x in numbers if x % 2 == 0]
print(squared_evens)

# transform strings just as easily
words = ["hello", "world"]
caps = [w.upper() for w in words]
print(caps)

# a nested comprehension builds a grid
grid = [[r * c for c in range(1, 4)] for r in range(1, 4)]
print(grid)
