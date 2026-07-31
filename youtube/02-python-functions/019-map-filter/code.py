numbers = [1, 2, 3, 4, 5, 6]

doubled = list(map(lambda x: x * 2, numbers))
print(doubled)

evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)

# map can take a named function too, not just a lambda
def celsius_to_f(c):
    return c * 9 / 5 + 32

temps = [0, 20, 37]
print(list(map(celsius_to_f, temps)))

# chain filter then map: keep evens, then square them
result = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))
print(result)
