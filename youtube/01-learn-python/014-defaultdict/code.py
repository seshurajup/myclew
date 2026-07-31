from collections import defaultdict

pairs = [("fruit", "apple"), ("veg", "kale"), ("fruit", "pear")]
groups = defaultdict(list)

for kind, item in pairs:
    groups[kind].append(item)

print(groups["fruit"])
print(groups["veg"])
print(dict(groups))
