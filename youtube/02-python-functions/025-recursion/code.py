def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print([fib(i) for i in range(8)])

# recursion walks nested structures naturally
def deep_sum(items):
    total = 0
    for x in items:
        if isinstance(x, list):
            total += deep_sum(x)
        else:
            total += x
    return total

print(deep_sum([1, [2, 3], [4, [5, 6]]]))
