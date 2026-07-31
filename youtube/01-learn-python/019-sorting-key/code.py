words = ["banana", "kiwi", "apple", "fig"]

by_length = sorted(words, key=len)
print(by_length)

people = [("amy", 30), ("ben", 25), ("cat", 40)]
by_age = sorted(people, key=lambda p: p[1], reverse=True)
print(by_age)

print(max(people, key=lambda p: p[1]))
