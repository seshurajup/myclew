import math, random

# math: constants and functions
print(math.pi, math.e)
print(math.sqrt(144), math.factorial(5))
print(math.floor(3.7), math.ceil(3.2))
print(math.gcd(48, 36))

# random: reproducible with a seed
random.seed(42)
print(random.randint(1, 6))          # dice roll
print(random.choice(["a", "b", "c"]))
print(random.sample(range(1, 50), 3))  # lottery

nums = [1, 2, 3, 4, 5]
random.shuffle(nums)
print(nums)
