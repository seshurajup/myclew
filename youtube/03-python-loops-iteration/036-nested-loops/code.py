# a multiplication table
for row in range(1, 4):
    for col in range(1, 4):
        print(f"{row*col:2}", end=" ")
    print()

# every pairing of two lists
colors = ["red", "blue"]
sizes = ["S", "L"]
for c in colors:
    for s in sizes:
        print(c, s)

# break only exits the INNER loop
for i in range(3):
    for j in range(3):
        if j == 1:
            break
        print(i, j)
