from collections import Counter, defaultdict, namedtuple

# Counter tallies occurrences instantly
c = Counter("mississippi")
print(c.most_common(2))

# defaultdict supplies a default for missing keys
groups = defaultdict(list)
for word in ["ant", "bee", "auk"]:
    groups[word[0]].append(word)
print(dict(groups))

# namedtuple: a lightweight, readable record
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p.x, p.y, p)
