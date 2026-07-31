count = 0
while count < 5:
    print(count)
    count += 1

user_input = ""
while user_input != "quit":
    user_input = "quit"
    print(f"You entered: {user_input}")

# break exits a loop immediately, even an infinite one
n = 0
while True:
    if n >= 3:
        break
    print(f"n is {n}")
    n += 1

# continue skips to the next iteration, without exiting
i = 0
while i < 6:
    i += 1
    if i % 2 == 0:
        continue
    print(f"odd: {i}")

# while/else: the else runs only if the loop finished WITHOUT a break
i = 0
while i < 3:
    i += 1
else:
    print("loop completed normally")
