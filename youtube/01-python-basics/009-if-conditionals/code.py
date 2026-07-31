age = 20

if age >= 18:
    print("Adult")
else:
    print("Minor")

score = 85
if score >= 90:
    print("A")
elif score >= 80:
    print("B")
else:
    print("C")

# combine conditions with and / or
temperature = 75
is_sunny = True
if temperature > 70 and is_sunny:
    print("Great beach day!")

# nested if: a condition inside another condition
balance = 500
withdrawal = 300
if withdrawal <= balance:
    if withdrawal > 0:
        print(f"Withdrew {withdrawal}, remaining {balance - withdrawal}")
else:
    print("Insufficient funds")

# the ternary (one-line if) expression
status = "adult" if age >= 18 else "minor"
print(status)
