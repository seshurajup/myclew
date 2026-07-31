x = "global"

def show():
    x = "local"
    print("inside:", x)

show()
print("outside:", x)

# global lets a function rebind a module-level variable
counter = 0
def bump():
    global counter
    counter += 1

bump()
bump()
print("counter:", counter)

# Python resolves names LEGB: Local, Enclosing, Global, Built-in
def outer():
    msg = "enclosing"
    def inner():
        print(msg)
    inner()

outer()
