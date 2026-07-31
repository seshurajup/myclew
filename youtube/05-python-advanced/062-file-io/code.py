# writing with the with-statement auto-closes the file
with open("notes.txt", "w") as f:
    f.write("line one\n")
    f.write("line two\n")

# reading the whole file
with open("notes.txt") as f:
    print(f.read())

# reading line by line
with open("notes.txt") as f:
    for line in f:
        print(line.strip())

# append mode adds without erasing
with open("notes.txt", "a") as f:
    f.write("line three\n")
