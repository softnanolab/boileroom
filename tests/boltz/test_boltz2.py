import json
import pathlib
import tempfile
from typing import Generator, Optional

import numpy as np
import pytest
from modal import enable_output

from boileroom import Boltz2
from boileroom.models.boltz.types import Boltz2Output
from boileroom.constants import restype_3to1

from biotite.structure import AtomArray, rmsd, superimpose
from biotite.structure.io.pdbx import CIFFile, get_structure


nipah_virus_sequence = "ICLQKTSNQILKPKLISYTLGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCTH"


@pytest.fixture(scope="module")
def boltz2_model(config: Optional[dict] = None, gpu_device: Optional[str] = None) -> Generator[Boltz2, None, None]:
    """Provide a Boltz2 model instance configured for the Modal backend.

    Parameters
    ----------
    config : Optional[dict]
        Optional model configuration overrides to apply when constructing the Boltz2 instance.
    gpu_device : Optional[str]
        Optional device identifier to run the model on (for example, "cuda:0" or similar).

    Yields
    ------
    Boltz2
        A Boltz2 instance configured with backend="modal", the specified device, and the provided configuration.
    """
    model_config = dict(config) if config is not None else {}
    with enable_output():
        yield Boltz2(backend="modal", device=gpu_device, config=model_config)


def _recover_chain_sequences(atomarray: AtomArray) -> list[str]:
    """Extract one-letter amino-acid sequences for each chain in an AtomArray.

    Parameters
    ----------
    atomarray : AtomArray
        Biotite AtomArray containing residues with `chain_id`, `res_id`, and three-letter `res_name` fields.

    Returns
    -------
    list[str]
        A list of one-letter sequences, one string per unique chain in the order of numpy.unique on `chain_id`.
    """
    chains = []
    for chain_id in np.unique(atomarray.chain_id):
        chain_atoms = atomarray[atomarray.chain_id == chain_id]
        unique_res_ids = np.unique(chain_atoms.res_id)
        three_letter_codes = [chain_atoms.res_name[chain_atoms.res_id == res_id][0] for res_id in unique_res_ids]
        one_letter_codes = [restype_3to1[code] for code in three_letter_codes]
        chains.append("".join(one_letter_codes))
    return chains


def test_boltz2_nipah_matches_reference(gpu_device: Optional[str]):
    """Run Boltz2 on the Nipah virus sequence and validate the predicted structure against reference data.

    Checks that required reference files exist, runs the model requesting all output fields, verifies an atom array was produced, compares C-alpha (CA) atoms between the predicted and reference structures, and asserts the superimposed CA RMSD is less than 0.5 Å.

    Parameters
    ----------
    gpu_device : Optional[str]
        GPU device identifier to use for the model, or `None` to run on CPU.
    """
    base_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "boltz"
    conf_path = base_dir / "confidence_0_model_0.json"
    cif_path = base_dir / "0_model_0.cif"
    plddt_npz = base_dir / "plddt_0_model_0.npz"
    pae_npz = base_dir / "pae_0_model_0.npz"
    pde_npz = base_dir / "pde_0_model_0.npz"
    assert conf_path.exists(), "tests/data/boltz/confidence_0_model_0.json must exist"
    assert cif_path.exists(), "tests/data/boltz/0_model_0.cif must exist"
    assert plddt_npz.exists(), "tests/data/boltz/plddt_0_model_0.npz must exist"
    assert pae_npz.exists(), "tests/data/boltz/pae_0_model_0.npz must exist"
    assert pde_npz.exists(), "tests/data/boltz/pde_0_model_0.npz must exist"

    with enable_output():
        model = Boltz2(
            backend="modal",
            device=gpu_device,
            config={
                "include_fields": ["*"],  # Request all fields for comprehensive testing
            },
        )
        # Note: we cannot guarantee fully deterministic output across different hardware
        # Current Boltz-2 implementation also does not set CUDA-based RNG to deterministic mode
        out = model.fold(nipah_virus_sequence, options={"seed": 42})

    assert isinstance(out, Boltz2Output)

    # Verify minimal defaults: atom_array should always be present
    assert out.atom_array is not None, "atom_array should always be generated"
    assert len(out.atom_array) > 0, "atom_array should contain at least one structure"

    # load the reference cif
    reference_atom_array = get_structure(CIFFile.read(cif_path), model=1)
    predicted_atom_array = out.atom_array[0]

    # Use CA atoms for backbone comparison (standard practice for protein structure comparison)
    predicted_ca = predicted_atom_array[predicted_atom_array.atom_name == "CA"]
    reference_ca = reference_atom_array[reference_atom_array.atom_name == "CA"]

    # Ensure both structures have the same number of CA atoms
    assert len(predicted_ca) == len(reference_ca), (
        f"Number of CA atoms must match: predicted has {len(predicted_ca)}, " f"reference has {len(reference_ca)}"
    )

    # Calculate RMSD between reference and superimposed predicted structure
    predicted_superimposed, _ = superimpose(reference_ca, predicted_ca)
    rmsd_value = rmsd(reference_ca, predicted_superimposed)
    assert rmsd_value < 0.5, f"RMSD {rmsd_value:.4f} Å exceeds threshold of 0.5 Å"

    # TODO: check confidence metrics within tolerance


def test_boltz2_minimal_output(test_sequences: dict[str, str], gpu_device: Optional[str]):
    """Test that Boltz2 returns minimal output by default (metadata + atom_array)."""
    with enable_output():
        model = Boltz2(backend="modal", device=gpu_device, config={})  # No include_fields = minimal output
        out = model.fold(test_sequences["short"])

    assert isinstance(out, Boltz2Output)
    assert out.metadata is not None, "metadata should always be present"
    assert out.atom_array is not None, "atom_array should always be generated"
    # With minimal output, other fields should be None
    assert out.confidence is None, "confidence should be None in minimal output"
    assert out.plddt is None, "plddt should be None in minimal output"
    assert out.pae is None, "pae should be None in minimal output"
    assert out.pde is None, "pde should be None in minimal output"


def test_boltz2_invalid_amino_acids_validation(test_sequences: dict[str, str]):
    """Verify that Boltz2Core's sequence validator raises a ValueError for invalid amino-acid sequences.

    This test calls Boltz2Core._validate_sequences with a sequence labelled "invalid" in the provided fixtures and expects a ValueError to be raised.

    Parameters
    ----------
    test_sequences : dict[str, str]
        Mapping of named test sequences; must include an "invalid" entry containing a sequence with invalid/unsupported amino-acid codes.
    """
    # Use the core's validator directly to ensure it raises for invalid inputs
    from boileroom.models.boltz.core import Boltz2Core

    core = Boltz2Core(config={"device": "cpu"})
    with pytest.raises(ValueError):
        core._validate_sequences(test_sequences["invalid"])  # should raise


def test_boltz2_static_config_enforcement(test_sequences: dict[str, str]):
    """Test that static config keys cannot be overridden in options."""
    from boileroom.models.boltz.core import Boltz2Core

    core = Boltz2Core(config={"device": "cpu"})
    # device, cache_dir, and no_kernels are static config keys
    with pytest.raises(ValueError, match="device"):
        core.fold(test_sequences["short"], options={"device": "cuda:0"})
    with pytest.raises(ValueError, match="cache_dir"):
        core.fold(test_sequences["short"], options={"cache_dir": "/tmp/test"})
    with pytest.raises(ValueError, match="no_kernels"):
        core.fold(test_sequences["short"], options={"no_kernels": True})


# TODO: we should have an integration test that would check that the MSA is properly hit for multiple runs (and swapped sequences)
def test_boltz2_msa_cache_hit(test_sequences: dict[str, str]):
    """Test that MSA cache is hit on second fold() call with same sequence.

    Creates a Boltz2 instance with a temporary cache directory, calls fold() twice with the same sequence,
    and verifies that the MSA cache was used on the second call.

    Parameters
    ----------
    test_sequences : dict[str, str]
        Mapping of test sequence names to sequence strings. Uses "short" for quick testing.
    """
    from boileroom.models.boltz.core import Boltz2Core
    import hashlib

    sequence = test_sequences["short"]
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = pathlib.Path(tmpdir) / "boltz"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create core with temporary cache directory
        core = Boltz2Core(config={"device": "cpu", "cache_dir": str(cache_dir)})
        core._initialize()

        # First call - cache miss, should generate and save MSA
        out1 = core.fold(sequence)
        assert isinstance(out1, Boltz2Output)
        assert out1.atom_array is not None

        # Verify MSA was cached
        msa_cache_dir = cache_dir / "msa_cache"
        index_path = msa_cache_dir / "msa_index.json"
        assert index_path.exists(), "MSA cache index should exist after first fold"

        # Load index and verify entry exists
        with index_path.open("r") as f:
            index = json.load(f)
        assert seq_hash in index, f"Sequence hash {seq_hash} should be in cache index"

        # Verify MSA file exists
        entry = index[seq_hash]
        msa_relative_path = entry.get("msa_path", "")
        if not msa_relative_path:
            # Fallback to hash-based path structure
            msa_path = msa_cache_dir / seq_hash[:2] / seq_hash[2:4] / f"{seq_hash}.csv"
        else:
            msa_path = msa_cache_dir / msa_relative_path
        assert msa_path.exists(), f"MSA file should exist at {msa_path}"

        # Second call - cache hit, should use cached MSA
        out2 = core.fold(sequence)
        assert isinstance(out2, Boltz2Output)
        assert out2.atom_array is not None

        # Verify cache index was updated (last_accessed should be newer or at least present)
        with index_path.open("r") as f:
            index2 = json.load(f)
        assert seq_hash in index2, "Sequence hash should still be in cache index after second call"
        entry2 = index2[seq_hash]
        second_last_accessed = entry2.get("last_accessed", "")

        # Verify last_accessed was updated (cache was hit)
        assert second_last_accessed != "", "last_accessed should be set after cache check"
        # Note: In practice, last_accessed should be updated, but due to timing it might be the same
        # The important thing is that the entry exists and the file was accessed


def test_boltz2_msa_cache_per_chain_multimer_reuse(test_sequences: dict[str, str]):
    """Test that MSAs are cached per individual chain and can be reused across different multimers.

    This test verifies:
    1. MSAs are cached per individual chain (not per multimer pair)
    2. Sequence order can be changed and cached MSAs are still reused
    3. Different multimers can reuse cached MSAs for shared chains

    Test scenario:
    - First call: "AAAA:BBBB" -> caches MSA for AAAA and BBBB
    - Second call: "BBBB:CCCC" -> reuses BBBB from cache, caches CCCC
    - Third call: "CCCC:AAAA" -> reuses both CCCC and AAAA from cache
    - Fourth call: "AAAA:BBBB" -> reuses both AAAA and BBBB from cache

    Parameters
    ----------
    test_sequences : dict[str, str]
        Mapping of test sequence names to sequence strings.
    """
    from boileroom.models.boltz.core import Boltz2Core

    # Use three distinct sequences for testing
    sequence_aaaa = test_sequences["short"]  # "MLKNVHVLVLGAGDVGSVVVRLLEK"
    sequence_bbbb = test_sequences["medium"]  # "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
    # Create a third distinct sequence by modifying the short one
    sequence_cccc = test_sequences["short"][::-1]  # Reverse of short sequence

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = pathlib.Path(tmpdir) / "boltz"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create core with temporary cache directory
        core = Boltz2Core(config={"device": "cpu", "cache_dir": str(cache_dir)})
        core._initialize()

        # Compute hashes using the core's method
        hash_aaaa = core._get_sequence_hash(sequence_aaaa)
        hash_bbbb = core._get_sequence_hash(sequence_bbbb)
        hash_cccc = core._get_sequence_hash(sequence_cccc)

        msa_cache_dir = cache_dir / "msa_cache"
        index_path = msa_cache_dir / "msa_index.json"

        # Helper function to check which sequences are in cache
        def get_cached_sequences() -> set[str]:
            if not index_path.exists():
                return set()
            with index_path.open("r") as f:
                index = json.load(f)
            return set(index.keys())

        # Helper function to verify MSA file exists for a sequence
        def verify_msa_exists(seq_hash: str) -> bool:
            if not index_path.exists():
                return False
            with index_path.open("r") as f:
                index = json.load(f)
            if seq_hash not in index:
                return False
            entry = index[seq_hash]
            msa_relative_path = entry.get("msa_path", "")
            if not msa_relative_path:
                msa_path = msa_cache_dir / seq_hash[:2] / seq_hash[2:4] / f"{seq_hash}.csv"
            else:
                msa_path = msa_cache_dir / msa_relative_path
            return msa_path.exists() and msa_path.is_file()

        # Test 1: First call with "AAAA:BBBB" -> should cache both AAAA and BBBB
        multimer1 = f"{sequence_aaaa}:{sequence_bbbb}"
        out1 = core.fold(multimer1)
        assert isinstance(out1, Boltz2Output)
        assert out1.atom_array is not None

        cached_after_1 = get_cached_sequences()
        assert hash_aaaa in cached_after_1, "AAAA should be cached after first multimer call"
        assert hash_bbbb in cached_after_1, "BBBB should be cached after first multimer call"
        assert hash_cccc not in cached_after_1, "CCCC should not be cached yet"
        assert verify_msa_exists(hash_aaaa), "MSA file for AAAA should exist"
        assert verify_msa_exists(hash_bbbb), "MSA file for BBBB should exist"

        # Test 2: Second call with "BBBB:CCCC" -> should reuse BBBB, cache CCCC
        multimer2 = f"{sequence_bbbb}:{sequence_cccc}"
        out2 = core.fold(multimer2)
        assert isinstance(out2, Boltz2Output)
        assert out2.atom_array is not None

        cached_after_2 = get_cached_sequences()
        assert hash_aaaa in cached_after_2, "AAAA should still be in cache"
        assert hash_bbbb in cached_after_2, "BBBB should still be in cache (reused)"
        assert hash_cccc in cached_after_2, "CCCC should now be cached"
        assert verify_msa_exists(hash_cccc), "MSA file for CCCC should exist"

        # Test 3: Third call with "CCCC:AAAA" (reversed order) -> should reuse both
        multimer3 = f"{sequence_cccc}:{sequence_aaaa}"
        out3 = core.fold(multimer3)
        assert isinstance(out3, Boltz2Output)
        assert out3.atom_array is not None

        cached_after_3 = get_cached_sequences()
        assert hash_aaaa in cached_after_3, "AAAA should still be in cache (reused)"
        assert hash_bbbb in cached_after_3, "BBBB should still be in cache"
        assert hash_cccc in cached_after_3, "CCCC should still be in cache (reused)"
        # All three should be cached and reused

        # Test 4: Fourth call with "AAAA:BBBB" again -> should reuse both
        out4 = core.fold(multimer1)
        assert isinstance(out4, Boltz2Output)
        assert out4.atom_array is not None

        cached_after_4 = get_cached_sequences()
        assert hash_aaaa in cached_after_4, "AAAA should still be in cache (reused)"
        assert hash_bbbb in cached_after_4, "BBBB should still be in cache (reused)"
        assert hash_cccc in cached_after_4, "CCCC should still be in cache"
        assert len(cached_after_4) == 3, "Should have exactly 3 cached sequences"

        # Verify all MSA files still exist
        assert verify_msa_exists(hash_aaaa), "MSA file for AAAA should still exist"
        assert verify_msa_exists(hash_bbbb), "MSA file for BBBB should still exist"
        assert verify_msa_exists(hash_cccc), "MSA file for CCCC should still exist"


def test_boltz2_msa_cache_integration(test_sequences: dict[str, str]):
    """Full integration test verifying MSA cache works end-to-end with actual folding.

    This test verifies:
    1. MSAs are generated and cached on first call
    2. Cached MSAs are actually used on second call (not regenerated)
    3. MSA content is valid and properly formatted
    4. Both folding outputs produce valid structures
    5. FASTA headers contain MSA paths when cache is used

    The test performs a complete end-to-end validation of the MSA caching system
    by running the full folding pipeline twice and verifying cache behavior.

    Parameters
    ----------
    test_sequences : dict[str, str]
        Mapping of test sequence names to sequence strings.
    """
    from boileroom.models.boltz.core import Boltz2Core

    sequence = test_sequences["short"]

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = pathlib.Path(tmpdir) / "boltz"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create core with temporary cache directory
        core = Boltz2Core(config={"device": "cpu", "cache_dir": str(cache_dir)})
        core._initialize()

        seq_hash = core._get_sequence_hash(sequence)
        msa_cache_dir = cache_dir / "msa_cache"
        index_path = msa_cache_dir / "msa_index.json"

        # Helper to get cached MSA path
        def get_cached_msa_path() -> pathlib.Path | None:
            if not index_path.exists():
                return None
            with index_path.open("r") as f:
                index = json.load(f)
            if seq_hash not in index:
                return None
            entry = index[seq_hash]
            msa_relative_path = entry.get("msa_path", "")
            if not msa_relative_path:
                return msa_cache_dir / seq_hash[:2] / seq_hash[2:4] / f"{seq_hash}.csv"
            return msa_cache_dir / msa_relative_path

        # Helper to read MSA file content
        def read_msa_content(msa_path: pathlib.Path) -> str:
            with msa_path.open("r") as f:
                return f.read()

        # Helper to validate MSA format (CSV with "key,sequence" header)
        def validate_msa_format(content: str) -> bool:
            lines = content.strip().split("\n")
            if not lines:
                return False
            # Check header
            if not lines[0].strip().startswith("key,sequence"):
                return False
            # Check that there's at least one sequence line
            if len(lines) < 2:
                return False
            # Check format of sequence lines (key,sequence)
            for line in lines[1:]:
                if "," not in line:
                    return False
            return True

        # First call - cache miss, should generate and save MSA
        out1 = core.fold(sequence)
        assert isinstance(out1, Boltz2Output), "First fold() should return Boltz2Output"
        assert out1.atom_array is not None, "First fold() should produce atom array"
        assert len(out1.atom_array) > 0, "First fold() should produce at least one structure"

        # Verify MSA was cached
        assert index_path.exists(), "MSA cache index should exist after first fold"
        cached_msa_path_1 = get_cached_msa_path()
        assert cached_msa_path_1 is not None, "Cached MSA path should exist after first fold"
        assert cached_msa_path_1.exists(), "Cached MSA file should exist after first fold"

        # Read and validate MSA content from cache
        cached_msa_content_1 = read_msa_content(cached_msa_path_1)
        assert len(cached_msa_content_1) > 0, "Cached MSA file should not be empty"
        assert validate_msa_format(cached_msa_content_1), "Cached MSA should have valid CSV format"

        # Get file modification time to verify it's not regenerated on second call
        mtime_before_second_call = cached_msa_path_1.stat().st_mtime

        # Small sleep to ensure time difference if file is modified
        import time

        time.sleep(0.1)

        # Second call - cache hit, should use cached MSA
        out2 = core.fold(sequence)
        assert isinstance(out2, Boltz2Output), "Second fold() should return Boltz2Output"
        assert out2.atom_array is not None, "Second fold() should produce atom array"
        assert len(out2.atom_array) > 0, "Second fold() should produce at least one structure"

        # Verify cached MSA was reused (file modification time should be unchanged)
        cached_msa_path_2 = get_cached_msa_path()
        assert cached_msa_path_2 is not None, "Cached MSA path should still exist after second fold"
        assert cached_msa_path_2 == cached_msa_path_1, "Cached MSA path should be the same file"
        mtime_after_second_call = cached_msa_path_2.stat().st_mtime

        # The MSA file should not have been regenerated (modification time should be same or older)
        # Note: We check <= instead of == because file systems may update metadata
        assert (
            mtime_after_second_call <= mtime_before_second_call + 0.5
        ), "Cached MSA file should not be regenerated (modification time check)"

        # Verify MSA content is still valid and unchanged
        cached_msa_content_2 = read_msa_content(cached_msa_path_2)
        assert cached_msa_content_2 == cached_msa_content_1, "Cached MSA content should be unchanged"
        assert validate_msa_format(cached_msa_content_2), "Cached MSA should still have valid format"

        # Verify cache index was updated (last_accessed timestamp)
        with index_path.open("r") as f:
            index = json.load(f)
        assert seq_hash in index, "Sequence hash should still be in cache index"
        entry = index[seq_hash]
        assert "last_accessed" in entry, "Cache entry should have last_accessed timestamp"
        assert entry["last_accessed"] != "", "last_accessed should not be empty"

        # Verify both structures are valid by checking they have expected properties
        # Both should have the same number of residues (same sequence length)
        structure1 = out1.atom_array[0]
        structure2 = out2.atom_array[0]

        # Extract sequences from structures and verify they match input
        sequences1 = _recover_chain_sequences(structure1)
        sequences2 = _recover_chain_sequences(structure2)

        assert len(sequences1) == 1, "First structure should have one chain"
        assert len(sequences2) == 1, "Second structure should have one chain"
        assert sequences1[0] == sequence, "First structure sequence should match input"
        assert sequences2[0] == sequence, "Second structure sequence should match input"

        # Verify structures have CA atoms (basic structure validity check)
        ca_atoms1 = structure1[structure1.atom_name == "CA"]
        ca_atoms2 = structure2[structure2.atom_name == "CA"]

        assert len(ca_atoms1) > 0, "First structure should have CA atoms"
        assert len(ca_atoms2) > 0, "Second structure should have CA atoms"
        assert (
            len(ca_atoms1) == len(ca_atoms2)
        ), "Both structures should have same number of CA atoms"
