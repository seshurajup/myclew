# accumulator: build up a result
total = 0
for n in [10, 20, 30]:
    total += n
print(total)

# search with a flag
found = False
for name in ["Alice", "Bob"]:
    if name == "Bob":
        found = True
        break
print(found)

# for-else: else runs if no break happened
for n in [1, 3, 5]:
    if n % 2 == 0:
        print("found even")
        break
else:
    print("all odd")

# collecting into a new list
squares = []
for n in range(1, 5):
    squares.append(n * n)
print(squares)
