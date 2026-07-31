name, score = "Alice", 91.5

# alignment and width
print(f"{name:<10}|{score:>8}")

# number formatting: decimals, percent, commas
print(f"{score:.1f}")
print(f"{0.256:.1%}")
print(f"{1234567:,}")

# the = specifier shows name and value (great for debugging)
x = 42
print(f"{x=}")

# expressions and method calls inline
print(f"{name.upper()} scored {score * 2}")
