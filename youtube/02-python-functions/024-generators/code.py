def countdown(n):
    while n > 0:
        yield n
        n -= 1

for num in countdown(3):
    print(num)

def fibonacci(limit):
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b

print(list(fibonacci(20)))

# generators are lazy — values are produced only on demand
gen = (x ** 2 for x in range(1000000))
print(next(gen))
print(next(gen))
