def make_multiplier(factor):
    def multiply(n):
        return n * factor
    return multiply

times_3 = make_multiplier(3)
times_5 = make_multiplier(5)

print(times_3(10))
print(times_5(10))

# a closure can keep private, evolving state
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

counter = make_counter()
print(counter(), counter(), counter())
