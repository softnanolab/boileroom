import pytest

from boileroom.utils import validate_sequence, format_time

from conftest import TEST_SEQUENCES

def test_validate_sequence():
    """Test sequence validation."""
    # Valid sequences
    assert validate_sequence(TEST_SEQUENCES["short"]) is True
    assert validate_sequence(TEST_SEQUENCES["medium"]) is True

    # Invalid sequences
    with pytest.raises(ValueError):
        validate_sequence(TEST_SEQUENCES["invalid"])
    with pytest.raises(ValueError):
        validate_sequence("NOT A SEQUENCE")


def test_format_time():
    """Test time formatting."""
    assert format_time(30) == "30s", f"Expected '30s', got {format_time(30)}"
    assert format_time(90) == "1m 30s", f"Expected '1m 30s', got {format_time(90)}"
    assert format_time(3600) == "1h", f"Expected '1h', got {format_time(3600)}"
    assert format_time(3661) == "1h 1m 1s", f"Expected '1h 1m 1s', got {format_time(3661)}"
