import numpy as np
import math
from dezero_simple.core import Function, Variable


class Add(Function):
    """
    加法
    """

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 + x2

    def backward(self, dout: Variable) -> list[Variable]:
        return dout, dout


def add(x1: any, x2: any) -> Variable:
    """
    x1、x2 - Variable、np.ndarray、数字 反正都会包装成Variable
    """
    return Add()(x1, x2)


class Neg(Function):
    """
    加法逆元
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: Variable) -> Variable:
        return -gy


def neg(x: any) -> Variable:
    return Neg()(x)


class Mul(Function):
    """
    乘法
    """

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 * x2

    def backward(self, dout: Variable) -> list[Variable]:
        x0, x1 = self.inputs[0], self.inputs[1]
        return x1 * dout, x0 * dout


def mul(x1: any, x2: any):
    return Mul()(x1, x2)


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 - x1
        return y

    def backward(self, gy: Variable) -> list[Variable]:
        return gy, -gy


def sub(x0: any, x1: any):

    # 减法、除法还是在拆解为加法、乘法逆元
    #
    # return x0-Neg(x1)
    return Sub()(x0, x1)


def rsub(x0: any, x1: any):
    return Sub()(x1, x0)


class Div(Function):
    """
    除法
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray):
        return x0 / x1

    def backward(self, gy: Variable) -> list[Variable]:
        x0, x1 = self.inputs[0], self.inputs[1]
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


def div(x0: any, x1: any):
    return Div()(x0, x1)


def rdiv(x0: any, x1: any):
    return Div()(x1, x0)


class Pow(Function):

    def __init__(self, c: int):
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x**self.c
        return y

    def backward(self, gy: Variable) -> Variable:
        x = self.inputs[0]
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x: any, c: int):
    return Pow(c)(x)


class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, dout: Variable) -> Variable:
        return dout * cos(self.inputs[0])


def sin(x: any) -> Variable:
    return Sin()(x)


def maclaurin_sin(x: Variable, threshold=0.0001) -> Variable:
    """
    麦克劳林展开求sin
    """
    y = 0
    for i in range(100000):
        const: int = 2 * i + 1
        c: float = (-1) ** i / math.factorial(const)
        t: Variable = c * (x**const)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def backward(self, dout: Variable) -> Variable:
        return dout * -sin(self.inputs[0])


def cos(x) -> Variable:
    return Cos()(x)
