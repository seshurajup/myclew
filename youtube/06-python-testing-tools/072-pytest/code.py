# save as test_math.py, then run: pytest

def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5          # plain assert, no boilerplate

def test_add_negative():
    assert add(-1, 1) == 0

import pytest

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / 0

# parametrize runs one test with many inputs
@pytest.mark.parametrize("a,b,expected", [(1,1,2), (2,3,5), (0,0,0)])
def test_many(a, b, expected):
    assert add(a, b) == expected
