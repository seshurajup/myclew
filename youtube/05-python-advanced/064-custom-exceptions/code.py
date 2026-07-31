class InsufficientFunds(Exception):
    def __init__(self, needed, have):
        self.needed = needed
        self.have = have
        super().__init__(f"need {needed}, have {have}")

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFunds(amount, balance)
    return balance - amount

try:
    withdraw(100, 150)
except InsufficientFunds as e:
    print("denied:", e)
    print("short by", e.needed - e.have)
