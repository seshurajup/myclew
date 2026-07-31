def shout(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper() + "!"
    return wrapper


@shout
def greet(name):
    return f"hello {name}"


print(greet("world"))
