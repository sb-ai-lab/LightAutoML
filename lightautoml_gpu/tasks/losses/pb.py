import numpy as np


class MSELossCPU:
    """MSLE Loss function for regression/multiregression"""

    def __call__(self, y_pred):
        return y_pred


class MSLELossCPU:
    """MSLE Loss function for regression/multiregression"""

    def __call__(self, y_pred):
        return np.expm1(y_pred)


class BCELossCPU:
    """LogLoss for binary/multilabel classification"""

    def __init__(self, clip_value=1e-7):
        self.clip_value = clip_value

    def __call__(self, y_pred):
        pred = 1 / (1 + np.exp(-y_pred))
        pred = np.clip(pred, self.clip_value, 1 - self.clip_value)

        return pred


class CrossEntropyLossCPU:
    """CrossEntropy for multiclass classification"""

    def __init__(self, clip_value=1e-6):
        self.clip_value = clip_value

    def __call__(self, y_pred):
        exp_p = np.exp(y_pred - y_pred.max(axis=1, keepdims=True))
        return np.clip(exp_p / exp_p.sum(axis=1, keepdims=True), self.clip_value, 1 - self.clip_value)


postprocess_fns = {

    'binary': BCELossCPU(),
    'reg': MSELossCPU(),
    'multi:reg': MSELossCPU(),
    'multiclass': CrossEntropyLossCPU(),
    'multilabel': BCELossCPU(),
}
