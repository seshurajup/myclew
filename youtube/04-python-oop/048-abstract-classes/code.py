from abc import ABC, abstractmethod

class Payment(ABC):
    @abstractmethod
    def pay(self, amount):
        ...

class Card(Payment):
    def pay(self, amount):
        return f"Paid {amount} by card"

# Payment() alone cannot be instantiated
try:
    Payment()
except TypeError as e:
    print("abstract:", e)

print(Card().pay(50))
