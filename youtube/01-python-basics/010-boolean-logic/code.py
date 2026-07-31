x = 5
y = 10

print(x < y)
print(x == y)
print(x != y)
print(x > 0 and y > 0)
print(x > 20 or y > 5)
print(not (x > y))

# Python truthiness: empty/zero values are "falsy"
print(bool(0))
print(bool(""))
print(bool([]))
print(bool("hello"))
print(bool([1, 2]))

# short-circuit evaluation: the second side isn't checked if not needed
def noisy():
    print("called!")
    return True

result = False and noisy()
print(result)
