from boileroom import ESMFold, ESM2

import modal

if __name__ == "__main__":
    with modal.enable_output():

        # Initialize the model
        model = ESMFold(backend="modal", config={"output_pdb": True})

        # # Predict structure for a protein sequence
        sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"

        result = model.fold([sequence])

        # Access prediction results
        coordinates = result.positions
        confidence = result.plddt


        new_model = ESM2(backend="modal", config={"output_hidden_states": True})
        new_result = new_model.embed([sequence])

        print(new_result)

print(coordinates)
print(confidence)