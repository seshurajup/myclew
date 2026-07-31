# assign and use in one expression
data = [1, 2, 3, 4, 5]
if (n := len(data)) > 3:
    print(f"list has {n} items")

# avoid computing something twice
import math
if (root := math.isqrt(30)) ** 2 <= 30:
    print(f"floor sqrt is {root}")

# read input until a sentinel (simulated)
values = iter([5, 8, 0, 3])
total = 0
while (v := next(values)) != 0:
    total += v
print("sum before zero:", total)
