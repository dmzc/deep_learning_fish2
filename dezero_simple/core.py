from __future__ import annotations
import numpy as np
from graphviz import Digraph
import psutil
import os
import math
import weakref
from config import ENABLE_BACKPROGATION


class Variable:
    __name: str
    is_input: bool
    data: np.ndarray
    grad: np.ndarray
    creator: Function
    generation: int
    __array_priority__ = 200

    def __init__(
        self, data: any, creator: Function = None, name: str = None, is_input=False
    ):
        if not isinstance(data, np.ndarray):
            if isinstance(data, (int, float, np.number)):
                data = np.array(data)
            else:
                raise TypeError("{}is not supportted".format(type(data)))
        self.data = data
        self.grad = None
        self.__name = name
        self.is_input = is_input
        if ENABLE_BACKPROGATION:
            self.creator = creator
            if creator is None:
                self.generation = 0
            else:
                self.generation = creator.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad=False) -> None:
        if self.creator is None:
            return

        if self.grad is None:
            self.grad = np.ones_like(self)
        creators: list[Function] = []
        seen_set: set = set()

        def add_creator(creator):
            if creator not in seen_set:
                seen_set.add(creator)
                creators.append(creator)
                creators.sort(key=lambda x: x.generation)

        add_creator(self.creator)

        while creators:
            creator = creators.pop()
            gys = [output().grad for output in creator.outputs]
            gxs = creator.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(creator.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_creator(x.creator)
            if not retain_grad:
                for y in creator.outputs:
                    y().grad = None

    @property
    def id(self) -> str:
        return f"_{id(self)}_"

    @property
    def name(self) -> str:
        ret_name = ""
        if self.__name is not None:
            ret_name = f"{self.__name}"
            if ENABLE_BACKPROGATION:
                ret_name = f"{ret_name}({self.generation})\n数据：{self.data}"
                if self.grad is not None:
                    ret_name = f"{ret_name}\n梯度：{self.grad}"
            else:
                ret_name = f"{ret_name}\n数据：{self.data}"
        else:
            if ENABLE_BACKPROGATION:
                ret_name = f"数据：{self.data}\n层级：{self.generation}"
                if self.grad is not None:
                    ret_name = f"{ret_name}\n梯度：{self.grad}"
            else:
                ret_name = f"数据：{self.data}\n"
        return ret_name

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    inputs: list[Variable]
    outputs: list[weakref.ref[Variable]]
    label: str
    generation: int

    def __init__(self):
        self.label = self.__class__.__name__
        self.inputs = None
        self.outputs = None
        self.generation = None

    def __call__(self, *xs) -> list[Variable] | Variable:
        inputs = [as_variable(x) for x in xs]
        xs_data = [x.data for x in inputs]
        ys = self.forward(*xs_data)
        if not isinstance(ys, tuple):
            ys = (ys,)
        creator = None
        if ENABLE_BACKPROGATION:
            self.generation = max([x.generation for x in inputs])
            creator = self
        outputs = [Variable(as_array(y), creator) for y in ys]
        if ENABLE_BACKPROGATION:
            self.outputs = [weakref.ref(output) for output in outputs]
            self.inputs = inputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: any) -> any:
        raise NotImplementedError

    def backward(self, dout: any) -> any:
        raise NotImplementedError

    @property
    def id(self) -> str:
        return f"_{id(self)}_"

    @property
    def name(self) -> int:
        ret_name = f"{self.__class__.__name__}"
        if ENABLE_BACKPROGATION:
            ret_name = f"{ret_name}({self.generation})"

        return ret_name


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, dout):
        return dout * 2 * self.inputs[0].data


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
    xss = [as_array(x) for x in xs]
    return Add()(*xss)


class Mul(Function):
    def forward(self, *xs):
        x0, x1 = xs
        y = x0 * x1
        return y

    def backward(self, dout):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1 * dout, x0 * dout


def mul(*xs):
    xss = [as_array(x) for x in xs]
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
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return sub(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, dout):
        return dout * np.cos(self.inputs[0].data)


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


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
