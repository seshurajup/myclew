try:
    x = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

try:
    nums = [1, 2, 3]
    print(nums[10])
except IndexError:
    print("Index out of range")

# catch multiple exception types in one except
try:
    value = int("not a number")
except (ValueError, TypeError) as e:
    print(f"Conversion failed: {e}")

# else runs only if NO exception occurred; finally ALWAYS runs
try:
    result = 20 / 4
except ZeroDivisionError:
    print("error!")
else:
    print(f"Success: {result}")
finally:
    print("Cleanup runs no matter what")

# raise lets you trigger your own custom error
def check_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age

try:
    check_age(-5)
except ValueError as e:
    print(f"Caught: {e}")
