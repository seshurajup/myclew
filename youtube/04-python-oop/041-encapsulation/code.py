class Account:
    def __init__(self, balance):
        self._balance = balance      # convention: internal
        self.__pin = "1234"          # name-mangled, truly private

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount

    def get_balance(self):
        return self._balance

acc = Account(100)
acc.deposit(50)
print(acc.get_balance())

# single underscore is a gentle "please don't touch"
print(acc._balance)

# double underscore is name-mangled to prevent accidents
try:
    print(acc.__pin)
except AttributeError as e:
    print("blocked:", e)
