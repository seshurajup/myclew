def greet(first, last):
    return f"Hello, {first} {last}!"

def describe_pet(name, species="dog"):
    return f"{name} is a {species}"

print(greet("Alice", "Smith"))
print(describe_pet("Fluffy"))
print(describe_pet("Tweety", "parrot"))

# keyword arguments: name them explicitly, any order
print(describe_pet(species="cat", name="Whiskers"))

# positional vs keyword — positionals must come first
def order(item, qty, size="medium"):
    return f"{qty} {size} {item}"

print(order("coffee", 2))
print(order("tea", 1, size="large"))
