from unittest.mock import Mock, patch

# a Mock stands in for a real object
db = Mock()
db.get_user.return_value = {"name": "Alice"}

print(db.get_user(1))
print(db.get_user.called)
print(db.get_user.call_count)

# patch replaces something during a test
def get_time():
    import time
    return time.time()

with patch("time.time", return_value=42.0):
    print(get_time())
