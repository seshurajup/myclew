import unittest

def add(a, b):
    return a + b

class TestAdd(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(add(2, 3), 5)

    def test_negative(self):
        self.assertEqual(add(-1, -1), -2)

    def test_type(self):
        self.assertIsInstance(add(1, 2), int)

# run the tests
unittest.main(argv=[""], exit=False, verbosity=2)
