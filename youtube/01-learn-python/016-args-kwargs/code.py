def total(*args):
    return sum(args)

def profile(**kwargs):
    return ", ".join(f"{k}={v}" for k, v in kwargs.items())

print(total(1, 2, 3, 4))
print(profile(name="Sam", age=30))

nums = [10, 20, 30]
print(total(*nums))
