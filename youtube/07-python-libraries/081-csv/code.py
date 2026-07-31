import csv
from io import StringIO

# writing rows to CSV
buf = StringIO()
writer = csv.writer(buf)
writer.writerow(["name", "score"])
writer.writerow(["Alice", 90])
writer.writerow(["Bob", 85])
print(buf.getvalue())

# reading back with DictReader: rows become dicts
buf.seek(0)
reader = csv.DictReader(buf)
for row in reader:
    print(row["name"], "->", row["score"])
