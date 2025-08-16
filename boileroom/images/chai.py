from modal import Image

from .volumes import model_weights
from ..utils import MODEL_DIR

# Define a dedicated image for Chai-1 with required dependencies
chai_image = (
	Image.debian_slim(python_version="3.12")
	.apt_install("wget", "git")
	.uv_pip_install(
		"chai_lab==0.5.0",
		"hf_transfer>=0.1.8",
		# biotite to parse PDB/CIF into AtomArray when needed
		"biotite>=1.0.1",
	)
	# Use a CUDA-enabled torch version suitable for Modal GPUs; installed inside image so it won't affect local env
	.uv_pip_install(
		"torch==2.7.1",
		index_url="https://download.pytorch.org/whl/cu128",
	)
	.env(
		{
			"HF_HUB_ENABLE_HF_TRANSFER": "1",
			# Let chai_lab cache and read weights from our shared volume
			"CHAI_DOWNLOADS_DIR": f"{MODEL_DIR}/chai1",
		}
	)
)