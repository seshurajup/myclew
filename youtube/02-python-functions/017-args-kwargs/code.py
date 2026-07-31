def sum_all(*numbers):
    return sum(numbers)

def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print(sum_all(1, 2, 3, 4, 5))
print(sum_all(10, 20))

print_info(name="Alice", age=25, city="NYC")

# combine normal params with *args and **kwargs
def report(title, *scores, **meta):
    print(title, "->", sum(scores), meta)

report("Exam", 80, 90, 70, term="Fall", year=2026)

# unpack a list into *args and a dict into **kwargs
nums = [1, 2, 3]
print(sum_all(*nums))
