names = ["amy", "ben", "cat"]
ages = [31, 25, 40]

for i, name in enumerate(names, start=1):
    print(i, name)

for name, age in zip(names, ages):
    print(f"{name} is {age}")

paired = dict(zip(names, ages))
print(paired)
