import numpy as np


class BCELossCPU:
    """LogLoss for binary/multilabel classification"""

    def __init__(self, clip_value=1e-7):
        self.clip_value = clip_value

    def __call__(self, y_pred):
        pred = 1 / (1 + np.exp(-y_pred))
        pred = np.clip(pred, self.clip_value, 1 - self.clip_value)

        return pred
