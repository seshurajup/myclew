square = lambda x: x ** 2
add = lambda x, y: x + y

print(square(5))
print(add(3, 7))

# lambdas shine as throwaway functions passed to other functions
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)

# sort a list of tuples by the second element
pairs = [(1, "b"), (2, "a"), (3, "c")]
pairs.sort(key=lambda p: p[1])
print(pairs)

# a lambda with a conditional expression
grade = lambda s: "pass" if s >= 60 else "fail"
print(grade(75), grade(40))
