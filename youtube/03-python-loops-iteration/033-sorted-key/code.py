nums = [3, 1, 4, 1, 5, 9, 2]
print(sorted(nums))
print(sorted(nums, reverse=True))

words = ["banana", "apple", "cherry"]
# sort by length instead of alphabetically
print(sorted(words, key=len))

people = [("Alice", 30), ("Bob", 25), ("Carol", 35)]
# sort by the second element, the age
print(sorted(people, key=lambda p: p[1]))

# sorted returns a NEW list; .sort() changes in place
nums.sort()
print(nums)
