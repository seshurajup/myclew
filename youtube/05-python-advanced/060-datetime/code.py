from datetime import datetime, timedelta, date

now = datetime.now()
print(now.year, now.month, now.day)

# format a date into a readable string
print(now.strftime("%Y-%m-%d %H:%M"))

# date arithmetic with timedelta
today = date(2026, 7, 24)
next_week = today + timedelta(days=7)
print(next_week)

# difference between two dates
delta = date(2026, 12, 31) - today
print(delta.days, "days left")
