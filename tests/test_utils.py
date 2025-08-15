import pytest

from boileroom.utils import validate_sequence, format_time


def test_validate_sequence(test_sequences: dict[str, str]):
    """Test sequence validation."""
    # Valid sequences
    assert validate_sequence(test_sequences["short"]) is True
    assert validate_sequence(test_sequences["medium"]) is True

    # Invalid sequences
    with pytest.raises(ValueError):
        validate_sequence(test_sequences["invalid"])
    with pytest.raises(ValueError):
        validate_sequence("NOT A SEQUENCE")


def test_format_time():
    """Test time formatting."""
    assert format_time(30) == "30s", f"Expected '30s', got {format_time(30)}"
    assert format_time(90) == "1m 30s", f"Expected '1m 30s', got {format_time(90)}"
    assert format_time(3600) == "1h", f"Expected '1h', got {format_time(3600)}"
    assert format_time(3661) == "1h 1m 1s", f"Expected '1h 1m 1s', got {format_time(3661)}"
