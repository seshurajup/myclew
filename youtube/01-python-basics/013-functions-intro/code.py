def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

print(greet("Alice"))
print(add(5, 3))

# default parameter values make an argument optional
def greet_formal(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet_formal("Bob"))
print(greet_formal("Bob", "Hi"))

# functions can return multiple values as a tuple
def min_max(numbers):
    return min(numbers), max(numbers)

low, high = min_max([4, 1, 9, 2])
print(low, high)

# a function with no return statement returns None
def log_message(msg):
    print(f"LOG: {msg}")

result = log_message("started")
print(result)
