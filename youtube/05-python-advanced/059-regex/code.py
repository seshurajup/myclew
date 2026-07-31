import re

text = "Contact: alice@mail.com and bob@work.org"

# find all email addresses
emails = re.findall(r"[\w.]+@[\w.]+", text)
print(emails)

# search returns the first match with groups
m = re.search(r"(\d{3})-(\d{4})", "call 555-1234")
print(m.group(0), m.group(1), m.group(2))

# substitute matches
clean = re.sub(r"\s+", " ", "too    many     spaces")
print(clean)
