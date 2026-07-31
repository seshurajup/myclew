from contextlib import contextmanager

@contextmanager
def tag(name):
    print(f"<{name}>")
    yield
    print(f"</{name}>")

with tag("p"):
    print("hello")

# suppress specific errors cleanly
from contextlib import suppress

with suppress(FileNotFoundError):
    open("nope.txt")
print("kept going")
