first, *middle, last = [1, 2, 3, 4, 5]
print(first, last)
print(middle)

a, b = 1, 2
a, b = b, a
print(a, b)

defaults = {"size": "M", "color": "red"}
order = {**defaults, "color": "blue"}
print(order)
