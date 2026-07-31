names = ["Alice", "Bob", "Carol"]
ages = [30, 25, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age}")

# zip stops at the shortest sequence
a = [1, 2, 3, 4]
b = ["x", "y"]
print(list(zip(a, b)))

# build a dict from two lists
prices = dict(zip(names, ages))
print(prices)

# unzip with the star operator
pairs = [(1, "a"), (2, "b"), (3, "c")]
nums, letters = zip(*pairs)
print(nums, letters)
