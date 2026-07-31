text = "Python"
message = "Hello, " + text
repeated = "Ha" * 3

print(text)
print(message)
print(repeated)
print(len(text))

# common string methods
print(text.upper())
print(text.lower())
print(message.replace("Hello", "Hey"))

# strings are immutable — methods return a NEW string
greeting = "  welcome  "
print(greeting.strip())

# split turns a string into a list of words
words = "learn python fast".split()
print(words)
