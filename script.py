import os
from pathlib import Path
from boileroom import Chai1

# load dotenv
from dotenv import load_dotenv

load_dotenv()

os.environ["CHAI_DOWNLOADS_DIR"] = str(Path(os.getenv("MODEL_DIR")) / "chai1")


if __name__ == "__main__":
    model = Chai1(backend="local")

    # Predict structure for a protein sequence
    sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"

    result = model.fold([sequence])

    print(result)
