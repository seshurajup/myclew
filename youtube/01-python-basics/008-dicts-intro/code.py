person = {"name": "Alice", "age": 25, "city": "NYC"}

print(person)
print(person["name"])
print(person["age"])
print(person.get("city"))
print(len(person))

# add or update a key just like a variable
person["email"] = "alice@example.com"
person["age"] = 26
print(person)

# .keys(), .values(), .items() give you views to loop over
print(list(person.keys()))
print(list(person.values()))

# get() with a default avoids a KeyError on missing keys
print(person.get("phone", "not provided"))

# remove a key with pop()
person.pop("city")
print(person)
