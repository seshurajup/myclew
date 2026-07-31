nums = [1, 2, 3, 4, 5]

squares = {n: n ** 2 for n in nums}
print(squares)

# filter while building the dict
even_squares = {n: n ** 2 for n in nums if n % 2 == 0}
print(even_squares)

# swap keys and values of an existing dict
prices = {"apple": 3, "banana": 2}
inverted = {v: k for k, v in prices.items()}
print(inverted)

# build a dict from two lists with zip
names = ["Alice", "Bob"]
ages = [30, 25]
people = {n: a for n, a in zip(names, ages)}
print(people)
