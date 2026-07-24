import numpy as np
import math
from dezero_simple.core import Function, Variable


# ==========================================================================
# 基础代数函数
# ==========================================================================
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
    """
    乘幂
    """

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


# ==========================================================================
# 基本代数函数
# ==========================================================================


# ==========================================================================
# 基本超越函数
# ==========================================================================
class Sin(Function):
    """
    正弦
    """

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
    """
    余弦
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def backward(self, dout: Variable) -> Variable:
        return dout * -sin(self.inputs[0])


def cos(x) -> Variable:
    return Cos()(x)


class Tanh(Function):
    """
    双曲正切
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x: any) -> Variable:
    return Tanh()(x)


class Exp(Function):
    """
    指数函数
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.exp(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    """
    对数函数
    """

    def forward(self, x: np.ndarray):
        y = np.log(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x = self.inputs[0]
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# ==========================================================================
# 基本超越函数
# ==========================================================================


# ==========================================================================
# 张量操作函数
# ==========================================================================


class Reshape(Function):
    """
    张量形状变更
    """

    __n_shape: tuple[int]  # 新形状
    __o_shape: tuple[int]  # 旧形状

    def __init__(self, n_shape: tuple[int]):
        super().__init__()
        self.__n_shape = n_shape
        self.__o_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.__o_shape = x.shape
        return np.reshape(x, self.__n_shape)

    def backward(self, dout: Variable) -> Variable:
        return reshape(dout, self.__o_shape)


def reshape(x: np.ndarray | Variable | list[int], shape: tuple[int]) -> Variable:
    """
    list[int]代表原生多维数组
    """
    return Reshape(shape)(x)


class Transpose(Function):
    """
    np.transpose:数据本身在内存中不会变,只是改变shape、stride。

    如果不传递参数，那么就是全部倒序下：

    x.shape # (2, 3, 4)

    x = x.transpose()

    x.shape # (4, 3, 2)
    """

    __n_axis: tuple[int]

    def __init__(self, n_axis=None):
        super().__init__()
        self.__n_axis = n_axis

    def forward(self, x: np.ndarray):
        return x.transpose(self.__n_axis)

    def backward(self, dout: Variable) -> Variable:
        n_axis = self.__n_axis
        if n_axis is None:
            return transpose(dout)
        axis_len = len(n_axis)
        # TODO:这里没搞懂，要连通np.transpose的算法一起搞清楚
        inv_axis = tuple(np.argsort([ax % axis_len for ax in n_axis]))
        return transpose(dout, inv_axis)


def transpose(x: np.ndarray | Variable | list[int], axis=None):
    """
    list[int]代表原生多维数组
    """
    return Transpose(axis)(x)


# ==========================================================================
# 张量操作函数
# ==========================================================================
