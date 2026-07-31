numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "cherry"]
mixed = [10, "hello", 3.14, True]

print(numbers)
print(fruits)
print(mixed)
print(len(fruits))

# lists are mutable — you can change them after creation
fruits.append("date")
print(fruits)

fruits.remove("banana")
print(fruits)

# check membership and combine lists
print("apple" in fruits)
more_numbers = numbers + [6, 7]
print(more_numbers)

# sort() rearranges a list in place
more_numbers.sort(reverse=True)
print(more_numbers)
