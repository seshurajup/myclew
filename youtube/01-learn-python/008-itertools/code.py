from itertools import chain, combinations

nums = [1, 2, 3]
letters = ["a", "b"]

print(list(chain(nums, letters)))
print(list(combinations(nums, 2)))
