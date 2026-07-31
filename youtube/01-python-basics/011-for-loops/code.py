for i in range(5):
    print(i)

fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

for i in range(1, 4):
    print(f"Count: {i}")

# enumerate gives you BOTH the index and the value
for index, fruit in enumerate(fruits):
    print(index, fruit)

# range with a step: start, stop, step
for i in range(0, 10, 2):
    print(i, end=" ")
print()

# nested loops build a small grid
for row in range(3):
    for col in range(3):
        print(f"({row},{col})", end=" ")
    print()
