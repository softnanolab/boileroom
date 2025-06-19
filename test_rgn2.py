#!/usr/bin/env python
"""
run_rgn2_calpha.py
Minimal end-to-end driver for an RGN2 C-alpha prediction.

Assumptions
-----------
• This script lives in  …/boileroom/run_rgn2_calpha.py
• The RGN2 repository is       …/rgn2/
• You already installed dependencies with the mamba script and
  have activated the `rgn2` environment before running this file.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path

###############################################################################
# Paths (relative to this file)
###############################################################################
ROOT         = Path(__file__).resolve().parent          # boileroom/
RGN2_ROOT    = (ROOT / ".." / "rgn2").resolve()         # ../rgn2
AMINOBERT_CP = RGN2_ROOT / "resources/aminobert_checkpoint" \
                        / "AminoBERT_runs_v2_uniparc_dataset_v2_5-1024_fresh_start_model.ckpt-1100000"
RUN_DIR      = RGN2_ROOT / "runs/15106000"              # pre-trained RGN2 run
CONFIG_FILE  = RUN_DIR / "configuration"

###############################################################################
# Helper functions
###############################################################################
def _validate_sequence(seq: str, max_len: int = 1023) -> str:
    seq = re.sub(r"\s+", "", seq).upper()
    aatypes = set("ACDEFGHIKLMNPQRSTVWY")
    if not set(seq).issubset(aatypes):
        raise ValueError(
            f"Invalid letters: {set(seq) - aatypes}. Only 20 natural AA are supported."
        )
    if len(seq) > max_len:
        raise ValueError(f"Sequence too long ({len(seq)} AA > {max_len})")
    return seq


def main():
    parser = argparse.ArgumentParser(description="Fold one protein with RGN2")
    parser.add_argument("--sequence", default="MALWMRLLPLLALLALWGPDPAAA", help="Amino-acid string")
    parser.add_argument("--job", default="job", help="Job name (used in file names)")
    args = parser.parse_args()

    sequence = _validate_sequence(args.sequence)
    jobname  = re.sub(r"\W+", "", args.job)
    seq_hash = hashlib.blake2b(sequence.encode(), digest_size=3).hexdigest()
    seq_id   = f"{jobname}_{seq_hash}"

    # --- Working directories --------------------------------------------------
    data_dir   = ROOT / "aminobert_output"   # will be recreated
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    # clean old embeddings (AminoBERT reads whole dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir()

    ############################################################################
    # 1.  AminoBERT embeddings
    ############################################################################
    sys.path.append(str(RGN2_ROOT))                          # root for packages
    sys.path.append(str(RGN2_ROOT / "aminobert"))            # AminoBERT package

    # Move temporarily to the RGN2_ROOT directory
    # HACK around poor coding of the AminoBERT package
    old_dir = os.getcwd()
    os.chdir(RGN2_ROOT)

    from aminobert.prediction import aminobert_predict_sequence
    from data_processing.aminobert_postprocessing import aminobert_postprocess

    print(">>> Generating AminoBERT embeddings …")
    aminobert_predict_sequence(
        seq=sequence,
        header=seq_id,
        checkpoint=str(AMINOBERT_CP),
        data_dir=str(data_dir),
        prepend_m=True,
    )
    aminobert_postprocess(
        data_dir=str(data_dir),
        dataset_name="1",
        prepend_m=True,
    )
    os.chdir(old_dir)

    ############################################################################
    # 2.  Run RGN2
    ############################################################################
    print(">>> Running RGN2 …")
    protling = RGN2_ROOT / "rgn" / "protling.py"
    cmd = [
        sys.executable,
        str(protling),
        str(CONFIG_FILE),
        "-p",
        "-e",
        "weighted_testing",
        "-a",
        "-g",
        "0",
    ]
    # Run from boileroom so paths like "aminobert_output" resolve
    subprocess.run(cmd, cwd=ROOT, check=True)

    ############################################################################
    # 3.  Collect result
    ############################################################################
    ter_src = RUN_DIR / "1" / "outputsTesting" / f"{seq_id}.tertiary"
    if not ter_src.exists():
        raise FileNotFoundError(f"Expected output not found: {ter_src}")

    ter_dst = output_dir / ter_src.name
    shutil.copy2(ter_src, ter_dst)
    print(f"\n✅  Finished!  C-alpha trace → {ter_dst}\n")


if __name__ == "__main__":
    main()
