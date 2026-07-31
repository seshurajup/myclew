fruits = ["apple", "banana", "cherry"]

print(fruits[0])
print(fruits[1])
print(fruits[2])
print(fruits[-1])
print(fruits[-2])

# indexing works the same way on strings
word = "python"
print(word[0])
print(word[-1])

# out-of-range indexing raises an IndexError
try:
    print(fruits[10])
except IndexError as e:
    print(f"Error: {e}")

# use index() to find WHERE an item lives
print(fruits.index("cherry"))
