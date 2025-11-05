import os
from pathlib import Path
from boileroom import Chai1
from boileroom import Boltz2
import modal

os.environ["MODEL_DIR"] = "/home/mmm1486/Scratch/hugging_face"

os.environ["CHAI_DOWNLOADS_DIR"] = str(Path(os.getenv("MODEL_DIR")) / "chai1")
os.environ["MODAL_INTERACTIVE"] = "True"


if __name__ == "__main__":
    with modal.enable_output():
        # model = Chai1(backend="modal", device="A100-40GB", config={"num_diffn_samples": 1, "use_esm_embeddings": True})

        # # Predict structure for a protein sequence
        # sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK:MLKNVHVLVLGAGDVGSVVVRLLEK"


        # restraint_definitions = [
        #     {
        #         "restraint_id": "contact_ab_interface",
        #         "chain_a": "A",
        #         "residue_token_a": "K3",
        #         "chain_b": "B",
        #         "residue_token_b": "D14",
        #         "connection_type": "contact",
        #         "confidence": 0.9,
        #         "minimum_distance_angstrom": 2.0,
        #         "maximum_distance_angstrom": 10.0,
        #         "comment": "Encourage Lys3 (chain A) to pair with Asp14 (chain B) as an interface salt bridge",
        #     },
        #     {
        #         "restraint_id": "contact_terminal_clamp",
        #         "chain_a": "A",
        #         "residue_token_a": "R21",
        #         "chain_b": "B",
        #         "residue_token_b": "E24",
        #         "connection_type": "contact",
        #         "confidence": 0.85,
        #         "minimum_distance_angstrom": 2.0,
        #         "maximum_distance_angstrom": 8.0,
        #         "comment": "Hold Arg21 on chain A against Glu24 on chain B to stabilize the dimer tail",
        #     },
        #     {
        #         "restraint_id": "pocket_anchor_b_on_a",
        #         "chain_a": "B",
        #         "residue_token_a": None,
        #         "chain_b": "A",
        #         "residue_token_b": "L22",
        #         "connection_type": "pocket",
        #         "confidence": 0.75,
        #         "minimum_distance_angstrom": 3.0,
        #         "maximum_distance_angstrom": 12.0,
        #         "comment": "Bias chain B to form a hydrophobic pocket around Leu22 on chain A",
        #     },
        # ]

        # result = model.fold([sequence], config={"constraint_path": restraint_definitions})

        # print(result)

        model = Boltz2(backend="modal", device="A100-80GB")
        print("Model initialized successfully")
        # breakpoint()  # Debug: Check model initialization (uncomment for interactive debugging)

        sequences = ["ICLQKTSNQILKPKLISYTLGQSGTCITDPLLAMDEGYFAYSHLERIGSCSRGVSKQRIIGVGEVLDRGDEVPSLFMTNVWTPPNPNTVYHCSAVYNNEFYYVLCAVSTVGDPILNSTYWSGSLMMTRLAVKPKSNGGGYNQHQLALRSIEKGRYDKVMPYGPSGIKQGDTLYFPAVGFLVRTEFKYNDSNCPITKCQYSKPENCRLSMGIRPNSHYILRSGLLKYNLSDGENPKVVFIEISDQRLSIGSPSKIYDSLGQPVFYQASFSWDTMIKFGDVLTVNPLVVNWRNNTVISRPGQSQCPRFNTCPEICWEGVYNDAFLIDRINWISAGVFLDSNQTAENPVFTVFKDNEILYRAQLASEDTNAQKTITNCFLLKNKIWCISLVEIYDTGDNVIRPKLFAVKIPEQCTH:IMILYWYWNASNHKFTNFQAGQVDPYILLDLDMCEPQKVAQYDYIYWTVHPMIFKDYPWMEPQIQIFKKELWTPENFQDCANPEQHKFIIWFQSNESPNMGGNEFQPGKDYMIISTSNGELDWGFLDMGKVCDESAWIDMSSPNHSQE"]
        result = model.fold(sequences)

        print("\n=== Debugging Output ===")
        print(f"Result type: {type(result)}")
        print(f"Positions shape: {result.positions.shape if hasattr(result.positions, 'shape') else 'N/A'}")
        print(f"Positions dtype: {result.positions.dtype if hasattr(result.positions, 'dtype') else 'N/A'}")
        if result.plddt is not None:
            print(f"plddt list length: {len(result.plddt)}")
            if len(result.plddt) > 0:
                print(f"plddt[0] shape: {result.plddt[0].shape}")
        if result.pae is not None:
            print(f"pae list length: {len(result.pae)}")
            if len(result.pae) > 0:
                print(f"pae[0] shape: {result.pae[0].shape}")
        if result.pde is not None:
            print(f"pde list length: {len(result.pde)}")
            if len(result.pde) > 0:
                print(f"pde[0] shape: {result.pde[0].shape}")
        if result.confidence is not None:
            print(f"Confidence list length: {len(result.confidence)}")
            if len(result.confidence) > 0:
                print(f"Confidence keys: {list(result.confidence[0].keys())}")
        print(f"Metadata: {result.metadata}")
        print(f"Prediction time: {result.metadata.prediction_time}")
        print(f"Sequence lengths: {result.metadata.sequence_lengths}")
        # breakpoint()  # Debug: Inspect final result (uncomment for interactive debugging)