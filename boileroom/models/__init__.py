from .esm.esmfold import ESMFold, get_esmfold
from .esm.esm2 import ESM2, get_esm2

# Protenix imports are lazy to avoid triggering @app.cls decorator at import time
# when protenix is not needed. Import directly from boileroom.models.protenix.protenix instead.
_LAZY_PROTENIX = ["ProtenixFold", "ProtenixOutput", "get_protenix"]


def __getattr__(name):
    if name in _LAZY_PROTENIX:
        from .protenix.protenix import ProtenixFold, ProtenixOutput, get_protenix

        _mapping = {
            "ProtenixFold": ProtenixFold,
            "ProtenixOutput": ProtenixOutput,
            "get_protenix": get_protenix,
        }
        globals().update(_mapping)
        return _mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ESMFold",
    "ESM2",
    "ProtenixFold",
    "ProtenixOutput",
    "get_esmfold",
    "get_esm2",
    "get_protenix",
]
