from collections import Counter

words = "red blue red green blue red".split()
counts = Counter(words)

print(counts)
print(counts["red"])
print(counts.most_common(1))

counts.update(["blue", "blue"])
print(counts["blue"])
