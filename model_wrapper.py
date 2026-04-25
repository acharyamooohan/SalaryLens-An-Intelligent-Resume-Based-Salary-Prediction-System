class ModelWrapper:
    def __init__(self, model, pt):
        self.model = model
        self.pt    = pt

    def predict(self, X):
        import numpy as np
        return self.pt.inverse_transform(
            self.model.predict(X).reshape(-1, 1)).ravel()