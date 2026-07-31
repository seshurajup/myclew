def shout(func):
    def wrapper(name):
        return func(name).upper()
    return wrapper

@shout
def greet(name):
    return f"hello, {name}"

print(greet("alice"))

# a timing decorator that wraps any function
import time
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"took {time.time() - start:.4f}s")
        return result
    return wrapper

@timed
def slow_add(a, b):
    time.sleep(0.1)
    return a + b

print(slow_add(2, 3))
