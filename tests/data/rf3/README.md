# RoseTTAFold 3 reference data

## `5vht_*`: upstream regression baseline (CPU tests)

`5vht_model.cif` and `5vht_summary_confidences.json` are an RF3 prediction (two protein chains, PDB 5VHT) that the
RosettaCommons foundry repository commits as its inference-regression baseline
(`models/rf3/tests/` at commit `4010e3e2e7350edada3e25a45c908c6bf407df4d`, files `5vht_from_file_model.cif` and
`5vht_from_file_summary_confidences.json`). They are real model output, used here to check that boileroom parses RF3's
file layout and ranking scores correctly without needing a GPU.

Source: <https://github.com/RosettaCommons/foundry>, BSD 3-Clause License,
Copyright (c) 2025, Institute for Protein Design, University of Washington. The license text is distributed with
that repository (`LICENSE.md`).

## `1ubq_*`, `2zta_*`: stock `rf3 fold` output (GPU integration tests)

For each entry, `<entry>_model.cif`, `<entry>_summary_confidences.json` and `<entry>_confidences.json` are the files that
the stock `rf3 fold` command of the same foundry commit wrote for the entry's sequence (ubiquitin, and the GCN4
leucine-zipper homodimer). `manifest.json` records the sequences, the settings, the checkpoint and the GPU. The integration test
`tests/rf3/test_rf3_integration.py::test_rf3_matches_stock_rf3_reference` folds the same sequences through boileroom and
compares coordinates, confidences and PAE with these files. Regenerate them with
`uv run python scripts/testing/rf3_stock_reference.py`, which runs the stock command on Modal.

The weights are not included. These outputs are model predictions made with the RF3 code; see the weights-license note
in `docs/models.md`.
