from .module import Module
import mtorch1.operator as F
from mtorch1.interfaces import ITensor

# 激活层


class Sigmoid(Module):
    def forward(self, x) -> ITensor:
        return F.sigmoid(x)
