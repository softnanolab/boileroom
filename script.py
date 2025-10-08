from boileroom import Chai1

import modal

if __name__ == "__main__":
    with modal.enable_output():
        # Initialize the model
        model = Chai1(backend="modal")

        # # Predict structure for a protein sequence
        sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"

        result = model.fold([sequence])

        print(result)
