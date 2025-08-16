import modal

app = modal.App("boileroom")


# Lazy import to avoid circular import
def _import_models():
    from .models import ESMFold, ESM2, Chai1, get_esmfold, get_esm2, get_chai1

    return ESMFold, ESM2, Chai1, get_esmfold, get_esm2, get_chai1


# Make these available at module level
def __getattr__(name):
    if name in ["ESMFold", "ESM2", "Chai1", "get_esmfold", "get_esm2", "get_chai1"]:
        ESMFold, ESM2, Chai1, get_esmfold, get_esm2, get_chai1 = _import_models()
        globals().update({
            "ESMFold": ESMFold, 
            "ESM2": ESM2, 
            "Chai1": Chai1,
            "get_esmfold": get_esmfold, 
            "get_esm2": get_esm2,
            "get_chai1": get_chai1
        })
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["ESMFold", "ESM2", "Chai1", "get_esmfold", "get_esm2", "get_chai1"])


__all__ = [
    "ESMFold",
    "ESM2",
    "Chai1",
    "get_esmfold",
    "get_esm2",
    "get_chai1",
]
