"""Generic unified model server for Apptainer and image-backed runtimes.

This server dynamically loads any Core class and exposes it via HTTP endpoints.
It runs inside a model-specific runtime image with model dependencies installed.
"""

import argparse
import importlib
import json
import logging
import os
import site
import sys
import sysconfig
from pathlib import Path
from typing import Any

# Ensure Python wheel library paths are in LD_LIBRARY_PATH for CUDA extension modules.
_site_packages = sysconfig.get_path("purelib")
_runtime_lib_paths = [f"{_site_packages}/cuequivariance_ops/lib", f"{_site_packages}/torch/lib"]
_nvidia_package_dir = Path(_site_packages) / "nvidia"
if _nvidia_package_dir.exists():
    _runtime_lib_paths.extend(str(path) for path in _nvidia_package_dir.glob("*/lib"))
_current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
_ld_path_parts = [p for p in _current_ld_path.split(":") if p] if _current_ld_path else []
for lib_path in _runtime_lib_paths:
    if lib_path not in _ld_path_parts:
        _ld_path_parts.insert(0, lib_path)  # Insert at beginning for priority
if _ld_path_parts:
    os.environ["LD_LIBRARY_PATH"] = ":".join(_ld_path_parts)

# Ensure installed packages take precedence over source tree
# This prevents local files (like boileroom/backend/modal.py) from shadowing installed packages
if hasattr(site, "getsitepackages"):
    site_packages = site.getsitepackages()
    # Move site-packages to the front of sys.path
    for site_pkg in reversed(site_packages):
        if site_pkg in sys.path:
            sys.path.remove(site_pkg)
            sys.path.insert(0, site_pkg)

# Add project root to sys.path (after site-packages) so we can import boileroom
_server_file = Path(__file__).resolve()
_boileroom_source_root = _server_file.parent.parent.parent
if str(_boileroom_source_root) not in sys.path:
    sys.path.insert(len(site.getsitepackages()) if hasattr(site, "getsitepackages") else 0, str(_boileroom_source_root))

# Install import hook to prevent local modal.py from shadowing installed modal package
# With lazy imports, this should rarely be needed, but serves as a safety net
_original_import = __import__


def _import_with_modal_fix(name, globals=None, locals=None, fromlist=(), level=0):
    """Import hook that prevents local modal.py from shadowing installed modal package."""
    if name == "modal" and level == 0:
        try:
            import importlib.util

            spec = importlib.util.find_spec("modal")
            if spec and spec.origin:
                origin_path = Path(spec.origin)
                # If it's our local modal.py, raise clear error
                if "boileroom" in str(origin_path) and "backend" in str(origin_path) and origin_path.name == "modal.py":
                    raise ImportError(
                        "The 'modal' package is not installed in this runtime environment. "
                        "The image-backed server does not require modal. "
                        "This error occurred because the local boileroom/backend/modal.py file "
                        "is shadowing the modal package. The image-backed server should not import modal."
                    )
        except Exception:
            pass
    return _original_import(name, globals, locals, fromlist, level)


# Replace __import__ before any imports
import builtins  # noqa: E402

builtins.__import__ = _import_with_modal_fix

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from boileroom.backend.transport import (  # noqa: E402
    TRANSPORT_HMAC_KEY_ENV,
    serialize_transport_payload,
)
from boileroom.optimization import REFUSAL_PROCESS_EXIT_CODE, initialize_core, is_refusal  # noqa: E402

# Set up logging to stderr so it gets captured in the log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global model instance
_model_instance: Any = None


def _extract_device_number(device: str) -> str | None:
    """Extract device number from device string (e.g., 'cuda:0' -> '0').

    Parameters
    ----------
    device : str
        Device string in format 'cuda:N' or 'cpu'.

    Returns
    -------
    Optional[str]
        Device number as string, or None if device is 'cpu' or invalid.
    """
    if device.startswith("cuda:"):
        return device.split(":")[1]
    return None


def _load_model() -> None:
    """Dynamically import and initialize the Core class specified by environment variables.

    Reads MODEL_CLASS, MODEL_CONFIG, and DEVICE from environment variables,
    imports the Core class using importlib, instantiates it, and loads it with
    :func:`boileroom.optimization.initialize_core`. Also sets CUDA_VISIBLE_DEVICES
    if device is a CUDA device.

    Raises
    ------
    Exception
        The load failure: an ``OptimizationUnavailableError`` when the core refuses the requested optimization mode
        (including a kit that exits with the refusal code), any other exception otherwise, with a ``SystemExit``
        turned into a ``RuntimeError``.
    """
    global _model_instance

    model_class_path = os.environ.get("MODEL_CLASS")
    if not model_class_path:
        raise ValueError("MODEL_CLASS environment variable must be set")

    model_config_str = os.environ.get("MODEL_CONFIG", "{}")
    try:
        model_config = json.loads(model_config_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid MODEL_CONFIG JSON: {e}") from e

    device = os.environ.get("DEVICE", "cuda:0")
    device_number = _extract_device_number(device)
    if device_number is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_number

    # Dynamically import the Core class
    module_path, class_name = model_class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        core_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to import {model_class_path}: {e}") from e

    core = core_class(config=model_config)
    failure = initialize_core(core)
    if failure is not None:
        raise failure
    _model_instance = core


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint.

    Returns
    -------
    dict[str, str]
        Status message indicating the server is ready.
    """
    if _model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


class EmbedRequest(BaseModel):
    """Request model for embed endpoint."""

    sequences: str | list[str]
    options: dict[str, Any] | None = None


class FoldRequest(BaseModel):
    """Request model for fold endpoint."""

    sequences: str | list[str] | dict[str, Any] | list[dict[str, Any]]
    options: dict[str, Any] | None = None


class InverseFoldRequest(BaseModel):
    """Request model for inverse_fold endpoint."""

    sequence: str
    backbone_coordinates: list[Any]  # (n_residues, 3, 3) nested lists; null entries mean NaN
    positions: list[int]


def _serialize_output(output: Any) -> dict[str, str]:
    """Serialize output object for signed JSON transport.

    Parameters
    ----------
    output : Any
        Output object to serialize (e.g., ESM2Output).

    Returns
    -------
    dict[str, str]
        Dictionary with base64-encoded pickled data and an HMAC signature.
    """
    transport_secret = os.environ.get(TRANSPORT_HMAC_KEY_ENV)
    if not transport_secret:
        raise RuntimeError(f"{TRANSPORT_HMAC_KEY_ENV} environment variable must be set")
    return serialize_transport_payload(output, transport_secret)


def _error_response(action: str, error: Exception) -> JSONResponse:
    """Return the 500 response for a failed call, naming the exception type so the client can re-raise a refusal.

    Parameters
    ----------
    action : str
        What failed, such as ``"Folding failed"``.
    error : Exception
        The exception the core raised.

    Returns
    -------
    JSONResponse
        Status 500 with ``detail`` (the message) and ``error_type`` (the exception's class name).
    """
    return JSONResponse(
        status_code=500,
        content={"detail": f"{action}: {error}", "error_type": type(error).__name__},
    )


@app.post("/embed")
async def embed(request: EmbedRequest) -> JSONResponse:
    """Embed sequences using the loaded model.

    Parameters
    ----------
    request : EmbedRequest
        Request containing sequences and optional options.

    Returns
    -------
    JSONResponse
        Signed embedding output payload.
    """
    if _model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        output = _model_instance.embed(request.sequences, options=request.options)
        serialized = _serialize_output(output)
        return JSONResponse(content=serialized)
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}", exc_info=True)
        return _error_response("Embedding failed", e)


@app.post("/inverse_fold")
async def inverse_fold(request: InverseFoldRequest) -> JSONResponse:
    """Predict masked-residue amino-acid logits using the loaded model.

    Parameters
    ----------
    request : InverseFoldRequest
        Request containing the sequence, backbone coordinates and positions to mask.

    Returns
    -------
    JSONResponse
        Signed inverse-folding output payload, or the 500 body of :func:`_error_response` for an unexpected failure.

    Raises
    ------
    HTTPException
        422 for invalid inputs (``ValueError``), 501 if the model has no ``inverse_fold`` and 503 if no model is
        loaded.
    """
    if _model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not hasattr(_model_instance, "inverse_fold"):
        raise HTTPException(status_code=501, detail="Loaded model does not support inverse folding")

    try:
        # ``dtype=float32`` turns JSON null entries back into NaN.
        coordinates = np.asarray(request.backbone_coordinates, dtype=np.float32)
        output = _model_instance.inverse_fold(request.sequence, coordinates, request.positions)
        serialized = _serialize_output(output)
        return JSONResponse(content=serialized)
    except ValueError as e:
        # Invalid caller input (duplicate positions, wrong coordinate shape, ...), not a server fault.
        logger.warning(f"Inverse folding rejected invalid input: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid inverse folding input: {str(e)}") from e
    except Exception as e:
        logger.error(f"Inverse folding failed: {str(e)}", exc_info=True)
        return _error_response("Inverse folding failed", e)


@app.post("/fold")
async def fold(request: FoldRequest) -> JSONResponse:
    """Fold sequences using the loaded model.

    Parameters
    ----------
    request : FoldRequest
        Request containing sequences and optional options.

    Returns
    -------
    JSONResponse
        Signed folding output payload.
    """
    if _model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        output = _model_instance.fold(request.sequences, options=request.options)
        serialized = _serialize_output(output)
        return JSONResponse(content=serialized)
    except Exception as e:
        logger.error(f"Folding failed: {str(e)}", exc_info=True)
        return _error_response("Folding failed", e)


def main() -> None:
    """Main entry point for the server: load the model, then serve it.

    The model loads before uvicorn starts, so a failed load ends the process with a code of its own: the kits' refusal
    code (:data:`boileroom.optimization.REFUSAL_PROCESS_EXIT_CODE`) when the core refuses the optimization mode, 1 for any
    other failure. The Apptainer backend turns the refusal code into ``OptimizationUnavailableError``. uvicorn exits
    with the same code 3 when it cannot start, so that exit is reported as 1 instead.
    """
    parser = argparse.ArgumentParser(description="Generic model server for image-backed runtimes")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    try:
        _load_model()
    except Exception as error:
        if is_refusal(error):
            logger.error(f"Model load refused: {error}", exc_info=True)
            sys.exit(REFUSAL_PROCESS_EXIT_CODE)
        logger.error(f"Model load failed: {type(error).__name__}: {error}", exc_info=True)
        sys.exit(1)

    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except SystemExit as exit_:
        # uvicorn's STARTUP_FAILURE is 3 as well; it must not read as a refusal.
        if exit_.code == REFUSAL_PROCESS_EXIT_CODE:
            sys.exit(1)
        raise


if __name__ == "__main__":
    main()
