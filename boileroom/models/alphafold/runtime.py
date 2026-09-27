"""Resident ColabFold adapter, also executable in its Python 3.10 environment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AlphaFold2MultimerRuntime:
    """Keep ColabFold runners, parameters and JAX compilations between requests."""

    def __init__(self, config: dict[str, Any], work_dir: str) -> None:
        """Load the fixed model configuration once in the isolated interpreter."""
        from colabfold.alphafold.models import load_models_and_params
        from colabfold.download import download_alphafold_params
        from colabfold.utils import setup_logging

        setup_logging(Path(work_dir) / "log.txt")
        self.config = config.copy()
        self._msa_pad_depth = 0
        self._max_seq = 508 if config["model_type"] == "alphafold2_multimer_v3" else 252
        data_dir = Path(config["data_dir"])
        download_alphafold_params(config["model_type"], data_dir)
        logger.info("Loading resident AlphaFold2-Multimer runners and parameters")
        self.runners = load_models_and_params(
            num_models=config["num_models"],
            use_templates=config["use_templates"],
            num_recycles=config["num_recycle"],
            model_type=config["model_type"],
            data_dir=data_dir,
            rank_by=config["rank_by"],
            max_seq=self._max_seq,
            max_extra_seq=2048 if config["model_type"] == "alphafold2_multimer_v3" else 1152,
        )

    def predict(self, input_path: str, output_dir: str, options: dict[str, Any]) -> None:
        """Prepare fresh input features and predict with the resident runners."""
        from colabfold.batch import generate_input_feature, msa_to_str, predict_structure
        from colabfold.input import get_queries

        queries, is_complex = get_queries(Path(input_path))
        if len(queries) != 1:
            raise ValueError("Expected exactly one ColabFold query")
        _, sequence, a3m_lines, _ = queries[0]
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        unpaired, paired, unique, cardinalities, templates = self._prepare_msa(sequence, a3m_lines, output, options)
        (output / "query.a3m").write_text(msa_to_str(unpaired, paired, unique, cardinalities), encoding="utf-8")
        features, _ = generate_input_feature(
            unique, cardinalities, unpaired, paired, templates, is_complex, self.config["model_type"], self._max_seq
        )
        lengths = [len(seq) for seq, count in zip(unique, cardinalities, strict=True) for _ in range(count)]
        # ColabFold uses the deepest MSA seen in a batch to avoid recompiling
        # when a later request has fewer rows. Keep that state across calls too.
        self._msa_pad_depth = max(self._msa_pad_depth, len(features["msa"]))
        predict_structure(
            prefix="query",
            result_dir=output,
            feature_dict=features,
            is_complex=is_complex,
            use_templates=self.config["use_templates"],
            sequences_lengths=lengths,
            pad_len=sum(lengths),
            msa_pad_depth=self._msa_pad_depth,
            model_type=self.config["model_type"],
            model_runner_and_params=self.runners,
            rank_by=self.config["rank_by"],
            random_seed=options["random_seed"],
            num_seeds=options["num_seeds"],
            num_relax=self.config["num_models"] * options["num_seeds"] if options["use_amber"] else 0,
            use_gpu_relax=options["use_gpu_relax"],
        )
        (output / "query.done.txt").touch()

    def _prepare_msa(self, sequence: str | list[str], a3m_lines: Any, output: Path, options: dict[str, Any]) -> tuple:
        """Use ColabFold's server or supplied-MSA paths without swallowing errors."""
        from colabfold.batch import get_msa_and_templates, unserialize_msa

        def search(query: str | list[str], alignments: Any, msa_mode: str) -> tuple:
            return get_msa_and_templates(
                jobname="query",
                query_sequences=query,
                a3m_lines=alignments,
                result_dir=output,
                msa_mode=msa_mode,
                use_templates=self.config["use_templates"],
                custom_template_path=None,
                pair_mode=options["pair_mode"],
                host_url=options["msa_server_url"],
                user_agent="boileroom",
            )

        if a3m_lines is None:
            return search(sequence, None, options["msa_mode"])
        if isinstance(a3m_lines, Path):
            a3m_lines = [a3m_lines.read_text(encoding="utf-8")]
        unpaired, paired, unique, cardinalities, templates = unserialize_msa(a3m_lines, sequence)
        if self.config["use_templates"]:
            *_, templates = search(unique, unpaired, "single_sequence")
        return unpaired, paired, unique, cardinalities, templates
