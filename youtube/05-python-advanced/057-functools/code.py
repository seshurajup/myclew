from functools import lru_cache, partial, reduce

@lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)

print(fib(50))          # instant, thanks to caching

# partial pre-fills some arguments
def power(base, exp):
    return base ** exp

square = partial(power, exp=2)
print(square(5))

# reduce folds a sequence into one value
print(reduce(lambda a, b: a * b, [1, 2, 3, 4]))
