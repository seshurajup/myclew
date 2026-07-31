def average(nums):
    assert len(nums) > 0, "cannot average an empty list"
    return sum(nums) / len(nums)

print(average([2, 4, 6]))

# assert catches broken assumptions early
def apply_discount(price, pct):
    assert 0 <= pct <= 100, f"bad percent: {pct}"
    return price * (1 - pct / 100)

print(apply_discount(100, 20))

try:
    apply_discount(100, 150)
except AssertionError as e:
    print("caught:", e)
