import os, sys

# os: interact with the operating system
print(os.getcwd())                     # current directory
print(os.environ.get("HOME", "?"))     # environment variables
print(os.path.join("a", "b", "c.txt"))

# list a directory
for name in os.listdir("."):
    print(name)

# sys: the Python runtime itself
print(sys.version_info[:2])
print(sys.platform)
print(len(sys.argv), "command-line args")
