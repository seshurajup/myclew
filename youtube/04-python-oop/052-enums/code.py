from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

print(Color.RED)
print(Color.RED.name, Color.RED.value)

# compare by identity, safe and readable
favorite = Color.GREEN
print(favorite == Color.GREEN)

# iterate over all members
for c in Color:
    print(c.name)
