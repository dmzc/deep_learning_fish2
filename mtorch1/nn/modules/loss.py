from mtorch1.nn.modules import Module
from mtorch1.interfaces import ITensor
import mtorch1.operator as F
import numpy as np


class MeanSquareLoss(Module):
    def forward(self, y_actual: np.ndarray, y_expect: np.ndarray) -> ITensor:
        return F.mean_square_loss(y_actual=y_actual, y_expect=y_expect)
