import modal

app = modal.App("boileroom")


# Lazy import to avoid circular import
def _import_models():
    from .models import ESMFold, ESM2, get_esmfold, get_esm2, ProtenixFold, ProtenixOutput, get_protenix

    return ESMFold, ESM2, get_esmfold, get_esm2, ProtenixFold, ProtenixOutput, get_protenix


_LAZY_NAMES = [
    "ESMFold", "ESM2", "get_esmfold", "get_esm2",
    "ProtenixFold", "ProtenixOutput", "get_protenix",
]


# Make these available at module level
def __getattr__(name):
    if name in _LAZY_NAMES:
        models = _import_models()
        names = _LAZY_NAMES
        mapping = dict(zip(names, models))
        globals().update(mapping)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + _LAZY_NAMES)


__all__ = _LAZY_NAMES
