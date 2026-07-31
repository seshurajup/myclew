fruits = ["apple", "banana", "cherry"]

for i, fruit in enumerate(fruits):
    print(i, fruit)

# start counting from 1 instead of 0
for rank, fruit in enumerate(fruits, start=1):
    print(f"{rank}. {fruit}")

# enumerate is cleaner than a manual counter
scores = [90, 85, 70]
for i, s in enumerate(scores):
    if s < 80:
        print(f"item {i} needs review")
