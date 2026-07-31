nums = [1, 2, 3, 4, 5, 6]

squares = list(map(lambda x: x * x, nums))
evens = list(filter(lambda x: x % 2 == 0, nums))

print(squares)
print(evens)

total = sum(map(lambda x: x * 10, evens))
print(total)
