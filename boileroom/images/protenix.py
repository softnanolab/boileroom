"""Modal image definition for Protenix (AlphaFold 3 reproduction) model."""

from modal import Image

# Protenix requires a specific set of dependencies including
# the protenix package itself installed from the GitHub repository.
# We use a CUDA-enabled base image since protenix relies heavily on GPU.
protenix_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git", "build-essential")
    .pip_install(
        "torch>=2.5.1,<2.8.0",
        "biotite>=1.0.1",
        "scipy>=1.9.0",
        "ml_collections>=1.0.0",
        "tqdm",
        "pandas",
        "PyYAML",
        "rdkit",
        "biopython",
        "modelcif",
        "gemmi",
        "pdbeccdutils",
        "fair-esm",
        "scikit-learn",
        "pydantic>=2.0.0",
        "optree",
        "numpy",
        "click",
    )
    .pip_install("protenix")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
