import os
from modal import enable_output

from boileroom import app, Chai1, ESMFold

os.environ["MODEL_DIR"] = ".model_cache"

if __name__ == "__main__":

    with enable_output(), app.run():
        model = Chai1()
        result = model.fold.local("MALWMRLLPLLALLALWGPDPAAA")
        print(result)
    
    import pdb; pdb.set_trace()