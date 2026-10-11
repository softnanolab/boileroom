"""Lightweight ProFam outputs, independent of the upstream model package."""

from dataclasses import dataclass, field

from ...base import GenerationOutput


@dataclass
class ConditioningPrompt:
    """Sequences and accessions actually selected for one sampler prompt."""

    sequences: list[str]
    accessions: list[str]


@dataclass
class ProFamOutput(GenerationOutput):
    """ProFam generation with the actual (possibly subsampled) family prompts."""

    conditioning_prompts: list[ConditioningPrompt] = field(default_factory=list)
