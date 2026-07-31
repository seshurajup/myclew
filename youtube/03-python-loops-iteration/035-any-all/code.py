nums = [2, 4, 6, 8]

# all: True only if EVERY item is truthy
print(all(n % 2 == 0 for n in nums))

# any: True if AT LEAST ONE item is truthy
print(any(n > 5 for n in nums))

# they short-circuit, stopping as soon as the answer is known
print(all([True, False, True]))
print(any([False, False, True]))

# handy for validating input
passwords = ["abc123", "xyz789"]
print(all(len(p) >= 6 for p in passwords))

# empty sequence: all is True, any is False
print(all([]), any([]))
