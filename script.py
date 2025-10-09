import os
from pathlib import Path
from boileroom import Chai1

import modal

os.environ["MODEL_DIR"] = "/home/mmm1486/Scratch/hugging_face"

os.environ["CHAI_DOWNLOADS_DIR"] = str(Path(os.getenv("MODEL_DIR")) / "chai1")
os.environ["MODAL_INTERACTIVE"] = "True"


if __name__ == "__main__":
    with modal.enable_output():
        model = Chai1(backend="modal", device="A100-40GB", config={"num_diffn_samples": 1, "use_esm_embeddings": True})

        # Predict structure for a protein sequence
        sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK:MLKNVHVLVLGAGDVGSVVVRLLEK"


        restraint_definitions = [
            {
                "restraint_id": "contact_ab_interface",
                "chain_a": "A",
                "residue_token_a": "K3",
                "chain_b": "B",
                "residue_token_b": "D14",
                "connection_type": "contact",
                "confidence": 0.9,
                "minimum_distance_angstrom": 2.0,
                "maximum_distance_angstrom": 10.0,
                "comment": "Encourage Lys3 (chain A) to pair with Asp14 (chain B) as an interface salt bridge",
            },
            {
                "restraint_id": "contact_terminal_clamp",
                "chain_a": "A",
                "residue_token_a": "R21",
                "chain_b": "B",
                "residue_token_b": "E24",
                "connection_type": "contact",
                "confidence": 0.85,
                "minimum_distance_angstrom": 2.0,
                "maximum_distance_angstrom": 8.0,
                "comment": "Hold Arg21 on chain A against Glu24 on chain B to stabilize the dimer tail",
            },
            {
                "restraint_id": "pocket_anchor_b_on_a",
                "chain_a": "B",
                "residue_token_a": None,
                "chain_b": "A",
                "residue_token_b": "L22",
                "connection_type": "pocket",
                "confidence": 0.75,
                "minimum_distance_angstrom": 3.0,
                "maximum_distance_angstrom": 12.0,
                "comment": "Bias chain B to form a hydrophobic pocket around Leu22 on chain A",
            },
        ]

        result = model.fold([sequence], config={"constraint_path": restraint_definitions})

        print(result)
