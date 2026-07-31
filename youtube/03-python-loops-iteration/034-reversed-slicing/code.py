nums = [1, 2, 3, 4, 5]

# reversed() gives a lazy reverse iterator
for n in reversed(nums):
    print(n, end=" ")
print()

# slicing with a -1 step also reverses
print(nums[::-1])

# reverse a string the same way
print("python"[::-1])

# reversed works with enumerate for countdown indexing
for i, n in enumerate(reversed(nums)):
    print(f"{i}: {n}")
