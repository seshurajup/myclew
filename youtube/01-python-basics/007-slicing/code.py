nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(nums[0:3])
print(nums[2:6])
print(nums[:5])
print(nums[5:])
print(nums[::2])

# negative step reverses the list — no need for a special reverse function
print(nums[::-1])

# slicing works on strings too
word = "python programming"
print(word[:6])
print(word[7:])

# slice assignment replaces a whole chunk at once
nums[0:2] = [100, 200]
print(nums)
