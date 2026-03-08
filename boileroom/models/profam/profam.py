"""ProFam protein sequence generator — three-tier boileroom implementation.

ProFamCore(GenerationAlgorithm)  — Model loading + sampling logic
ModalProFam(@app.cls)            — Modal deployment with @modal.enter()
ProFam(ModelWrapper)             — User-facing wrapper, backend dispatch
"""

import json
import logging
import time
from typing import Optional, Sequence, Union

import modal

from ...backend import ModalBackend
from ...backend.modal import app
from ...base import GenerationAlgorithm, GenerationOutput, ModelWrapper, PredictionMetadata
from ...utils import MINUTES
from .image import profam_image

logger = logging.getLogger(__name__)

# Default checkpoint path inside Modal container (baked into image)
_MODAL_CHECKPOINT_DIR = "/models/profam"


############################################################
# CORE ALGORITHM
############################################################


class ProFamCore(GenerationAlgorithm):
    """Core ProFam generation logic, independent of execution backend."""

    DEFAULT_CONFIG = {
        "temperature": 1.0,
        "top_p": 0.95,
        "sampler": "single",  # "single" or "ensemble"
        "max_context_tokens": 4096,
        "num_prompts_in_ensemble": 4,
        "checkpoint_path": None,  # auto-resolved
        "num_samples": 8,
        "match_representative_length": True,
    }
    STATIC_CONFIG_KEYS = {"sampler", "checkpoint_path"}

    def __init__(self, config: dict = {}) -> None:
        super().__init__(config)
        self.model = None
        self.name = "ProFam"
        self.version = "ProFam-1"

    def _initialize(self) -> None:
        """Load model — called by Modal @enter or directly."""
        self._load()

    def _load(self) -> None:
        import torch

        from src.models.llama import LlamaLitModule  # type: ignore
        from src.utils.utils import seed_all  # type: ignore

        ckpt_dir = self.config.get("checkpoint_path") or _MODAL_CHECKPOINT_DIR
        ckpt_path = f"{ckpt_dir}/checkpoints/last.ckpt"

        device = self._resolve_device()

        # GPU-aware dtype and attention: T4/L4 (compute < 8.0) use fp16+sdpa,
        # A100+ use bf16+flash_attention_2
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                dtype = torch.bfloat16
                attn_impl = "flash_attention_2"
                try:
                    import flash_attn  # noqa: F401
                except ImportError:
                    attn_impl = "sdpa"
                    logger.warning("flash_attn not installed, falling back to sdpa")
            else:
                dtype = torch.float16
                attn_impl = "sdpa"
        else:
            dtype = torch.float32
            attn_impl = "sdpa"

        logger.info(
            f"Loading ProFam from {ckpt_path} "
            f"(device={device}, dtype={dtype}, attn={attn_impl})"
        )

        # Override attention implementation in checkpoint config
        ckpt_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_obj = ckpt_blob.get("hyper_parameters", {}).get("config")
        if cfg_obj is None:
            raise RuntimeError("Could not find 'config' in checkpoint hyper_parameters")
        setattr(cfg_obj, "attn_implementation", attn_impl)
        setattr(cfg_obj, "_attn_implementation", attn_impl)

        self.model = LlamaLitModule.load_from_checkpoint(
            ckpt_path, config=cfg_obj, strict=False, weights_only=False,
        )
        self.model.eval()
        self.model.to(device, dtype=dtype)
        self._device = device
        self._dtype = dtype
        seed_all(42)
        self.ready = True
        logger.info("ProFam model loaded successfully.")

    def generate(
        self,
        sequences: Union[str, Sequence[str]],
        options: Optional[dict] = None,
    ) -> GenerationOutput:
        import torch

        from src.data.objects import ProteinDocument  # type: ignore
        from src.data.processors.preprocessing import (  # type: ignore
            AlignedProteinPreprocessingConfig,
            ProteinDocumentPreprocessor,
        )
        from src.models.inference import ProFamSampler, PromptBuilder  # type: ignore

        if self.model is None:
            raise RuntimeError("ProFam model not loaded. Call _load() first.")

        seqs = self._validate_sequences(sequences)
        merged = self._merge_options(options)

        num_samples = merged.get("num_samples", self.DEFAULT_CONFIG["num_samples"])
        temperature = merged.get("temperature", self.DEFAULT_CONFIG["temperature"])
        top_p = merged.get("top_p", self.DEFAULT_CONFIG["top_p"])
        max_context_tokens = merged.get("max_context_tokens", self.DEFAULT_CONFIG["max_context_tokens"])
        match_rep_length = merged.get("match_representative_length", True)

        t0 = time.time()

        # Build ProteinDocument from input sequences
        accessions = [f"seq_{i}" for i in range(len(seqs))]
        doc = ProteinDocument(
            sequences=list(seqs),
            accessions=accessions,
            representative_accession=accessions[0],
        )

        longest = max(len(s) for s in seqs)
        max_gen_len = int(longest * 1.2)

        # Preprocessing config for raw (non-MSA) sequences
        preproc_cfg = AlignedProteinPreprocessingConfig(
            document_token="[RAW]",
            defer_sampling=False,
            padding="do_not_pad",
            shuffle_proteins_in_document=True,
            keep_insertions=True,
            to_upper=True,
            keep_gaps=False,
            use_msa_pos=False,
            max_tokens_per_example=max_context_tokens - max_gen_len,
        )
        preprocessor = ProteinDocumentPreprocessor(cfg=preproc_cfg)
        builder = PromptBuilder(preprocessor=preprocessor, prompt_is_aligned=True)

        sampling_kwargs = {}
        if top_p is not None:
            sampling_kwargs["top_p"] = top_p
        if temperature is not None:
            sampling_kwargs["temperature"] = temperature

        sampler = ProFamSampler(
            name="boileroom_sampler",
            model=self.model,
            prompt_builder=builder,
            document_token="[RAW]",
            sampling_kwargs=sampling_kwargs or None,
            match_representative_length=match_rep_length,
            add_final_sep=True,
        )
        sampler.to(self._device)

        with torch.no_grad():
            gen_seqs, scores, _ = sampler.sample_seqs(
                protein_document=doc,
                num_samples=num_samples,
                max_tokens=max_context_tokens,
                max_generated_length=max_gen_len,
                continuous_sampling=False,
                minimum_sequence_length_proportion=0.5,
                minimum_sequence_identity=None,
                maximum_retries=5,
                repeat_guard=True,
            )

        inference_time = time.time() - t0

        metadata = PredictionMetadata(
            model_name=self.name,
            model_version=self.version,
            sequence_lengths=[len(s) for s in gen_seqs],
            inference_time=inference_time,
        )

        return GenerationOutput(
            metadata=metadata,
            sequences=list(gen_seqs),
            log_likelihoods=[float(s) for s in scores],
        )


############################################################
# MODAL BACKEND
############################################################


@app.cls(
    image=profam_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
)
class ModalProFam:
    """Modal-specific wrapper around ProFamCore."""

    config: bytes = modal.parameter(default=b"{}")

    @modal.enter()
    def _initialize(self) -> None:
        cfg = json.loads(self.config.decode("utf-8"))
        self._core = ProFamCore(cfg)
        self._core._initialize()

    @modal.method()
    def generate(
        self,
        sequences: Union[str, Sequence[str]],
        options: Optional[dict] = None,
    ) -> GenerationOutput:
        return self._core.generate(sequences, options=options)


############################################################
# HIGH-LEVEL INTERFACE
############################################################


class ProFam(ModelWrapper):
    """User-facing interface for ProFam protein sequence generation.

    Supports Modal backend for GPU-accelerated generation.
    """

    def __init__(
        self,
        backend: str = "modal",
        device: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = {}
        self.config = config
        self.device = device
        backend_type, _ = ModelWrapper.parse_backend(backend)
        if backend_type == "modal":
            backend_instance = ModalBackend(ModalProFam, config, device=device)
        else:
            raise ValueError(f"Backend {backend_type} not supported for ProFam")
        self._backend = backend_instance
        self._backend.start()

    def generate(
        self,
        sequences: Union[str, Sequence[str]],
        options: Optional[dict] = None,
    ) -> GenerationOutput:
        return self._call_backend_method("generate", sequences, options=options)
