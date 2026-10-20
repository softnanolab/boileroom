"""Live production-image regressions for ESM-C and ESM3 embeddings."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.xdist_group("esm3")]

SEQUENCES = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", "ACDEFGHIKL")


@pytest.mark.parametrize(("model_name", "dimension"), [("ESMC", 960), ("ESM3", 1536)])
def test_live_embeddings_preserve_residues(model_name, dimension, backend_option, device_option, output_ctx):
    """The public wrapper loads a real checkpoint and returns finite residue-aligned embeddings."""
    import boileroom

    with output_ctx(), getattr(boileroom, model_name)(backend=backend_option, device=device_option) as model:
        output = model.embed(SEQUENCES, options={"include_fields": ["lm_logits"]})

    assert output.embeddings.shape == (2, len(SEQUENCES[0]), dimension)
    assert np.isfinite(output.embeddings).all() and np.any(output.embeddings)
    assert output.lm_logits is not None and output.lm_logits.shape[:2] == output.embeddings.shape[:2]
    assert np.isfinite(output.lm_logits).all()
    for row, sequence in enumerate(SEQUENCES):
        length = len(sequence)
        np.testing.assert_array_equal(output.residue_index[row, :length], np.arange(length))
        np.testing.assert_array_equal(output.chain_index[row, :length], np.zeros(length))
        assert not output.embeddings[row, length:].any()
        assert (output.residue_index[row, length:] == -1).all()
    assert output.metadata.model_version and output.metadata.model_version != "fake-version"
    print(f"{model_name}: embeddings={output.embeddings.shape}, sdk_version={output.metadata.model_version}")


def _backbone_from_pdb(path: str) -> tuple[str, np.ndarray]:
    """Read a single-chain PDB into its sequence and ``(n_residues, 3, 3)`` N/CA/C coordinates."""
    from biotite.structure.info import one_letter_code
    from biotite.structure.io.pdb import PDBFile

    atoms = PDBFile.read(path).get_structure(model=1)
    residue_ids = sorted(set(atoms.res_id.tolist()))
    sequence = ""
    coordinates = np.full((len(residue_ids), 3, 3), np.nan, dtype=np.float32)
    for row, res_id in enumerate(residue_ids):
        residue = atoms[atoms.res_id == res_id]
        sequence += one_letter_code(residue.res_name[0]) or "X"
        for column, atom_name in enumerate(("N", "CA", "C")):
            match = residue[residue.atom_name == atom_name]
            if len(match) == 1:
                coordinates[row, column] = match.coord[0]
    return sequence, coordinates


def test_live_inverse_generate_redesigns_only_the_masked_residues(backend_option, device_option, output_ctx):
    """A real checkpoint jointly decodes the masked residues and leaves every other residue alone."""
    import boileroom

    sequence, coordinates = _backbone_from_pdb("tests/data/esmfold_server_short.pdb")
    positions = [3, 4, 5, 10, 11]
    assert not np.isnan(coordinates).any(), "test PDB is missing backbone atoms"

    with output_ctx(), boileroom.ESM3(backend=backend_option, device=device_option) as model:
        output = model.inverse_generate(sequence, coordinates, positions, temperature=0.1, seed=0)
        repeat = model.inverse_generate(sequence, coordinates, positions, temperature=0.1, seed=0)
        no_cysteine = model.inverse_generate(
            sequence, coordinates, positions, temperature=1.0, seed=1, excluded_amino_acids="C"
        )

    assert len(output.sequence) == len(sequence)
    assert output.positions.tolist() == positions
    assert len(output.residues) == len(positions)
    assert set(output.residues) <= set("ACDEFGHIKLMNPQRSTVWY")
    # Only the masked positions may differ from the input.
    changed = {index for index, (old, new) in enumerate(zip(sequence, output.sequence, strict=True)) if old != new}
    assert changed <= set(positions)
    # The decoded residues are exactly what the returned sequence carries at those positions.
    assert "".join(output.sequence[index] for index in positions) == output.residues
    # The same seed gives the same sample.
    assert repeat.sequence == output.sequence
    # Excluded amino acids are never sampled, even at a temperature flat enough to reach them.
    assert "C" not in no_cysteine.residues

    print(f"ESM3 inverse_generate: {sequence} -> {output.sequence} (decoded {output.residues} at {positions})")
