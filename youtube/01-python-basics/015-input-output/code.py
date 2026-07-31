name = input("What is your name? ")
age_str = input("How old are you? ")
age = int(age_str)

print(f"Hello, {name}!")
print(f"In 10 years, you will be {age + 10}")

# print() has extra options: sep and end
print("a", "b", "c", sep="-")
print("no newline here", end=" ")
print("continues on the same line")

# formatted output with f-strings: control decimal places and padding
price = 19.999
print(f"Price: ${price:.2f}")
print(f"{'right':>10}")
print(f"{'left':<10}|")

# reading a number safely with error handling
try:
    quantity = int(input("How many? "))
except ValueError:
    print("That's not a valid number, defaulting to 1")
    quantity = 1
print(f"You ordered {quantity} item(s)")
