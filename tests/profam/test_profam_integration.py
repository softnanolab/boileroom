"""Opt-in real-checkpoint generation; excluded from default pull-request CI."""

import math

import pytest

from boileroom import ProFam
from boileroom.models.profam.types import ProFamOutput

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow, pytest.mark.xdist_group("profam")]

PROMPT = [
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGA",
]


def test_profam_single_and_ensemble_reuse_runtime(backend_option, device_option, output_ctx) -> None:
    with (
        output_ctx(),
        ProFam(
            backend=backend_option, device=device_option, config={"num_samples": 1, "max_context_tokens": 512}
        ) as model,
    ):
        results = [
            model.generate(
                PROMPT,
                options={
                    "sampler": sampler,
                    "seed": 42,
                    "prompt_accessions": ["ubq_a", "ubq_b"],
                    "num_prompts_in_ensemble": 2,
                },
            )
            for sampler in ("single", "ensemble", "single")
        ]
        for result in results:
            assert isinstance(result, ProFamOutput)
            assert len(result.sequences) == len(result.log_likelihoods) == 1
            assert set(result.sequences[0]) <= set("ACDEFGHIKLMNPQRSTVWY")
            assert all(math.isfinite(score) for score in result.log_likelihoods)
            assert result.conditioning_prompts
            assert result.metadata.runtime is not None
            assert all(set(prompt.accessions) <= {"ubq_a", "ubq_b"} for prompt in result.conditioning_prompts)
            assert (
                result.metadata.runtime["profam.checkpoint_sha256"]
                == "8ec54d80323f188d3454fb4457f63f12158fe55552b919554806b12730c30e08"
            )
            assert "gpu.mem.used_mib" in result.metadata.runtime
        # Same seed on a resident deterministic inference stack should repeat.
        assert results[0].sequences == results[2].sequences
        assert results[0].log_likelihoods == pytest.approx(results[2].log_likelihoods)
