from __future__ import annotations
import numpy as np


class Variable:
    label: str
    data: np.ndarray
    grad: np.ndarray
    creator: Function

    def __init__(self, data: np.ndarray, creator: Function = None, label: str = None):
        if not isinstance(data, np.ndarray):
            raise TypeError("{}is not supportted".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = creator
        if label is not None:
            self.label = label
        else:
            self.label = f"{data}"

    def backward(self) -> None:
        if self.creator is None:
            return
        if self.grad is None:
            self.grad = np.ones_like(self)
        creators = [self.creator]
        while creators:
            creator = creators.pop()
            x = creator.input
            y = creator.output
            x.grad = creator.backward(y.grad)
            if x.creator is not None:
                creators.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    input: Variable
    output: Variable
    label: str

    def __init__(self):
        self.label = self.__class__.__name__
        self.input = None
        self.output = None

    def __call__(self, x: Variable) -> Variable:
        self.input = x
        y_data = as_array(self.forward(x.data))
        y = Variable(y_data, self, f"{self.__class__.__name__}-output-{y_data}")
        self.output = y
        return y

    def forward(self, x: any) -> any:
        raise NotImplementedError

    def backward(self, dout: any) -> any:
        raise NotImplementedError


class Square(Function):
    """
    平方函数
    """

    def forward(self, x: any) -> any:
        return x**2

    def backward(self, dout: any) -> any:
        return dout * 2 * self.input.data


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, dout: any) -> any:
        return dout * np.exp(self.input.data)


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def composite_function(x: Variable) -> Variable:
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_diff(composite_function, x)
print(f"数值微分：{dy}")

y = square(exp(square(x)))
y.backward()
print(f"反向传播：{x.grad}")
