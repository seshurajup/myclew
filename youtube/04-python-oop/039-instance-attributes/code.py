class Player:
    # class attribute: shared by ALL players
    team = "Red"

    def __init__(self, name):
        # instance attribute: unique per player
        self.name = name
        self.score = 0

a = Player("Alice")
b = Player("Bob")

print(a.name, a.team)
print(b.name, b.team)

# changing the class attribute affects everyone
Player.team = "Blue"
print(a.team, b.team)

# but an instance can override it for itself
a.team = "Green"
print(a.team, b.team)
