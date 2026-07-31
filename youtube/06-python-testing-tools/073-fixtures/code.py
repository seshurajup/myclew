import pytest

# a fixture provides ready-made test data or resources
@pytest.fixture
def sample_users():
    return [{"name": "Alice"}, {"name": "Bob"}]

def test_count(sample_users):
    assert len(sample_users) == 2

def test_first(sample_users):
    assert sample_users[0]["name"] == "Alice"

# fixtures can set up AND tear down
@pytest.fixture
def temp_file(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("hello")
    yield f              # test runs here
    # cleanup happens automatically after
