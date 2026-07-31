# star captures the middle
first, *middle, last = [1, 2, 3, 4, 5]
print(first, middle, last)

# swap without a temp variable
a, b = 1, 2
a, b = b, a
print(a, b)

# merge dicts and lists with unpacking
d1 = {"x": 1}
d2 = {"y": 2}
merged = {**d1, **d2, "z": 3}
print(merged)

combined = [*[1, 2], *[3, 4]]
print(combined)
