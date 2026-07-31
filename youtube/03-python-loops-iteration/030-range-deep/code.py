# range(stop)
print(list(range(5)))

# range(start, stop)
print(list(range(2, 8)))

# range(start, stop, step)
print(list(range(0, 20, 5)))

# count backwards with a negative step
print(list(range(10, 0, -2)))

# range is lazy — it doesn't build the list in memory
r = range(1_000_000)
print(r[500])
print(len(r))
