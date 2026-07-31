from itertools import count, cycle, chain, combinations

# count: an infinite counter
counter = count(start=1, step=2)
print(next(counter), next(counter), next(counter))

# chain: glue sequences together
print(list(chain([1, 2], [3, 4], [5])))

# combinations: every pair, order-independent
print(list(combinations("ABC", 2)))

# islice to safely take from an infinite iterator
from itertools import islice
print(list(islice(cycle("XY"), 5)))
