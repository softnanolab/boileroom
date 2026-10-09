# RoseTTAFold 3 reference data

`5vht_model.cif` and `5vht_summary_confidences.json` are an RF3 prediction (two protein chains, PDB 5VHT) that the
RosettaCommons foundry repository commits as its inference-regression baseline
(`models/rf3/tests/` at commit `4010e3e2e7350edada3e25a45c908c6bf407df4d`, files `5vht_from_file_model.cif` and
`5vht_from_file_summary_confidences.json`). They are real model output, used here to check that boileroom parses RF3's
file layout and ranking scores correctly without needing a GPU.

Source: <https://github.com/RosettaCommons/foundry>, BSD 3-Clause License,
Copyright (c) 2025, Institute for Protein Design, University of Washington. The license text is distributed with
that repository (`LICENSE.md`).
