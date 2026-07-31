data = [1, 2, 3, 4, 5, 6]
result = []
i = 0

while (n := data[i] if i < len(data) else None) is not None:
    if n % 2 == 0:
        result.append(n)
    i += 1

print(result)
