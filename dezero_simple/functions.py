import numpy as np
import math
from dezero_simple.core import Function, Variable, to_tensor


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, dout):
        return dout * 2 * self.inputs[0]


def square(x):
    return Square()(x)


class Add(Function):
    def forward(self, *xs):
        x0, x1 = xs
        y = x0 + x1
        return y

    def backward(self, dout):
        return dout, dout


def add(*xs: Variable) -> list[Variable]:
    xss = [to_tensor(x) for x in xs]
    return Add()(*xss)


class Mul(Function):
    def forward(self, *xs):
        x0, x1 = xs
        y = x0 * x1
        return y

    def backward(self, dout):
        x0, x1 = self.inputs[0], self.inputs[1]
        return x1 * dout, x0 * dout


def mul(*xs):
    xss = [to_tensor(x) for x in xs]
    return Mul()(*xss)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = to_tensor(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = to_tensor(x1)
    return sub(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0], self.inputs[1]
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


def div(x0, x1):
    x1 = to_tensor(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = to_tensor(x1)
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, dout):
        return dout * cos(self.inputs[0])


def sin(x) -> Variable:
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
    def forward(self, x):
        return np.cos(x)

    def backward(self, dout):
        return dout * -sin(self.inputs[0])


def cos(x) -> Variable:
    return Cos()(x)
