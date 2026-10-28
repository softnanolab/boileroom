"""The Chai-1 side of MSA handling: chai-lab alignment files, cache keys and the MSA directory.

chai-lab takes alignments from a directory of ``<hash of the sequence>.aligned.pqt`` files (``msa_directory``). A protein
without a file there is folded from its sequence alone, and the MSA server cannot be used at the same time. So the
alignments of all chains, whether provided by the caller or fetched from the server (one chain at a time) or taken from
the cache, are put into one such directory, and chai-lab itself never calls the server. Which source each chain's MSA
comes from is decided by :func:`boileroom.msa_sources.resolve_chain_msas`.

The parts that need chai-lab are imported lazily, so the rest can be tested without it.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ...msa_cache import MSACache
from ...msa_sources import EMPTY, MSAField

#: Cache entries are chai-lab alignment files of one sequence, fetched on its own from a given server.
CACHE_KEY_VERSION = "chai1-unpaired-v1"

#: Writes the ``.aligned.pqt`` of one chain into a directory: ``(a3m file, sequence, directory) -> written file``.
PqtWriter = Callable[[Path, str, Path], Path]
#: The file name chai-lab looks for, for a sequence.
Basename = Callable[[str], str]


def cache_key(sequence: str, server_url: str) -> str:
    """Cache key of the alignment of ``sequence``: it depends on the sequence and on the server asked."""
    return MSACache.hash_key(f"{CACHE_KEY_VERSION}|{server_url}|{sequence}")


def expected_basename(sequence: str) -> str:
    """The name of the alignment file of ``sequence`` in chai-lab's MSA directory (needs chai-lab)."""
    from chai_lab.data.parsing.msas.aligned_pqt import expected_basename as chai_expected_basename

    return chai_expected_basename(sequence)


def write_aligned_pqt(a3m_path: Path, sequence: str, directory: Path) -> Path:
    """Convert an A3M file to chai-lab's aligned parquet format (needs chai-lab) and write it into ``directory``."""
    from chai_lab.data.parsing.msas.aligned_pqt import a3m_to_aligned_dataframe
    from chai_lab.data.parsing.msas.data_source import MSADataSource

    # pairing keys come from species names in the headers, which a caller alignment need not have
    frame = a3m_to_aligned_dataframe(a3m_path, source_database=MSADataSource.UNIREF90, insert_pairing_key=False)
    path = directory / expected_basename(sequence)
    try:
        frame.to_parquet(path)
    except ImportError as error:  # pandas has no parquet engine
        raise RuntimeError(
            "Alignments for Chai-1 need a working parquet engine in the Chai-1 image: pyarrow<21 "
            "(newer pyarrow does not import next to numpy<2); rebuild the image from "
            "boileroom/models/chai/requirements.txt"
        ) from error
    return path


def stage_provided_alignments(
    alignments: Mapping[str, str],
    chains: Sequence[str],
    directory: Path,
    write: PqtWriter = write_aligned_pqt,
) -> dict[str, Path]:
    """Convert the provided alignments of the sequences that occur in ``chains`` to chai-lab alignment files.

    Returns the file of each such sequence. An alignment of a sequence that no chain has (the chain was changed) is not
    used and not converted.
    """
    staged: dict[str, Path] = {}
    for sequence in dict.fromkeys(chains):
        if sequence not in alignments:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        a3m_path = directory / f"provided_{len(staged)}.a3m"
        a3m_path.write_text(alignments[sequence], encoding="utf-8")
        staged[sequence] = write(a3m_path, sequence, directory)
    return staged


def assemble_msa_directory(
    chains: Sequence[str],
    fields: Mapping[int, MSAField],
    directory: Path,
    basename: Basename = expected_basename,
) -> Path | None:
    """Put the alignment file of every chain that has one into ``directory``, named as chai-lab expects.

    Chains whose field is ``empty`` get no file: chai-lab folds them from their sequence alone. Returns the directory, or
    ``None`` if no chain has a file.
    """
    placed = False
    for index, sequence in enumerate(chains):
        field = fields.get(index, EMPTY)
        if not isinstance(field, Path):
            continue
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / basename(sequence)
        if not target.exists():  # one file per sequence: identical chains share it
            shutil.copyfile(field, target)
        placed = True
    return directory if placed else None
