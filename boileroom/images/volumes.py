from modal import Volume

model_weights = Volume.from_name("model-weights", create_if_missing=True)
