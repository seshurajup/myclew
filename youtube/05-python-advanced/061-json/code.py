import json

# Python object -> JSON string
data = {"name": "Alice", "age": 30, "tags": ["a", "b"]}
text = json.dumps(data, indent=2)
print(text)

# JSON string -> Python object
parsed = json.loads('{"x": 1, "y": [2, 3]}')
print(parsed["y"])

# round-trip preserves structure
back = json.loads(json.dumps(data))
print(back == data)
