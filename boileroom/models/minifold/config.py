"""Dependency-free MiniFold settings shared by lightweight clients and the core."""


def validate_compile(value: object) -> None:
    """Refuse withdrawn compilation before backend startup or model loading."""
    if type(value) is not bool:
        raise ValueError("MiniFold compile must be a boolean")
    if value:
        raise ValueError(
            "MiniFold compile=True is unavailable: compiled inference failed strict upstream parity; use compile=False"
        )
