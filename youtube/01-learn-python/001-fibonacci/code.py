def fibonacci(n: int) -> list[int]:
    """First n Fibonacci numbers."""
    nums = [0, 1]
    while len(nums) < n:
        nums.append(nums[-1] + nums[-2])
    return nums[:n]

print(fibonacci(10))
