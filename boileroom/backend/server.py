"""Generic unified model server for conda backend.

This server dynamically loads any Core class and exposes it via HTTP endpoints.
It runs in a separate conda environment with model-specific dependencies.
"""

import argparse
import base64
import importlib
import json
import os
import pickle
import site
import sys
from pathlib import Path
from typing import Any, Optional, Union

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
                        "The 'modal' package is not installed in this conda environment. "
                        "The conda backend does not require modal. "
                        "This error occurred because the local boileroom/backend/modal.py file "
                        "is shadowing the modal package. The conda backend should not import modal."
                    )
        except Exception:
            pass
    return _original_import(name, globals, locals, fromlist, level)


# Replace __import__ before any imports
import builtins  # noqa: E402

builtins.__import__ = _import_with_modal_fix

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI()

# Global model instance
_model_instance: Any = None


def _extract_device_number(device: str) -> Optional[str]:
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
    imports the Core class using importlib, instantiates it, and calls _initialize().
    Also sets CUDA_VISIBLE_DEVICES if device is a CUDA device.
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

    # Instantiate and initialize
    _model_instance = core_class(config=model_config)
    _model_instance._initialize()


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on server startup."""
    _load_model()


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

    sequences: Union[str, list[str]]
    options: Optional[dict[str, Any]] = None


class FoldRequest(BaseModel):
    """Request model for fold endpoint."""

    sequences: Union[str, list[str]]
    options: Optional[dict[str, Any]] = None


def _serialize_output(output: Any) -> dict[str, str]:
    """Serialize output object using pickle and base64 encode for JSON transport.

    Parameters
    ----------
    output : Any
        Output object to serialize (e.g., ESM2Output).

    Returns
    -------
    dict[str, str]
        Dictionary with base64-encoded pickled data.
    """
    pickled_data = pickle.dumps(output)
    base64_encoded = base64.b64encode(pickled_data).decode("utf-8")
    return {"pickled": base64_encoded}


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
        Pickled and base64-encoded embedding output.
    """
    if _model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        output = _model_instance.embed(request.sequences, options=request.options)
        serialized = _serialize_output(output)
        return JSONResponse(content=serialized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}") from e


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
        Pickled and base64-encoded folding output.
    """
    if _model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        output = _model_instance.fold(request.sequences, options=request.options)
        serialized = _serialize_output(output)
        return JSONResponse(content=serialized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Folding failed: {str(e)}") from e


def main() -> None:
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="Generic model server for conda backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
