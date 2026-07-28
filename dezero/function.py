from __future__ import annotations
import math
import numpy as np
import weakref
from dezero.config import ENABLE_BACKPROGATION
from dezero.interfaces import IVariable, IFunction, IVariableArgs
from dezero.variable import create_variable


class Function(IFunction):

    def __init__(self):
        self.label = self.__class__.__name__
        self.inputs = None
        self.outputs = None
        self.generation = None

    def __call__(self, *xs: tuple[any]) -> list[IVariable] | IVariable:
        inputs = [create_variable(x) for x in xs]
        xs_data = [x.data for x in inputs]
        ys = self.forward(*xs_data)
        if not isinstance(ys, tuple):
            ys = (ys,)
        creator = None
        if ENABLE_BACKPROGATION:
            self.generation = max([x.generation for x in inputs])
            creator = self
        outputs = [create_variable(IVariableArgs(data=y, creator=creator)) for y in ys]
        if ENABLE_BACKPROGATION:
            self.outputs = [weakref.ref(output) for output in outputs]
            self.inputs = inputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: any) -> any:
        raise NotImplementedError

    def backward(self, dout: IVariable) -> any:
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


# ==========================================================================
# 基础代数算子
# ==========================================================================
class Add(Function):
    """
    加法
    """

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 + x2

    def backward(self, dout: IVariable) -> list[IVariable]:
        return dout, dout


def add(x1: any, x2: any) -> IVariable:
    """
    x1、x2 - IVariable、np.ndarray、数字 反正都会包装成IVariable
    """
    return Add()(x1, x2)


class Neg(Function):
    """
    加法逆元
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: IVariable) -> IVariable:
        return -gy


def neg(x: any) -> IVariable:
    return Neg()(x)


class Mul(Function):
    """
    乘法
    """

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 * x2

    def backward(self, dout: IVariable) -> list[IVariable]:
        x0, x1 = self.inputs[0], self.inputs[1]
        return x1 * dout, x0 * dout


def mul(x1: any, x2: any):
    return Mul()(x1, x2)


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 - x1
        return y

    def backward(self, gy: IVariable) -> list[IVariable]:
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

    def backward(self, gy: IVariable) -> list[IVariable]:
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

    def backward(self, gy: IVariable) -> IVariable:
        x = self.inputs[0]
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x: any, c: int):
    return Pow(c)(x)


# ==========================================================================
# 基本代数算子
# ==========================================================================


# ==========================================================================
# 基本超越算子
# ==========================================================================
class Sin(Function):
    """
    正弦
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, dout: IVariable) -> IVariable:
        return dout * cos(self.inputs[0])


def sin(x: any) -> IVariable:
    return Sin()(x)


def maclaurin_sin(x: IVariable, threshold=0.0001) -> IVariable:
    """
    麦克劳林展开求sin
    """
    y = 0
    for i in range(100000):
        const: int = 2 * i + 1
        c: float = (-1) ** i / math.factorial(const)
        t: IVariable = c * (x**const)
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

    def backward(self, dout: IVariable) -> IVariable:
        return dout * -sin(self.inputs[0])


def cos(x) -> IVariable:
    return Cos()(x)


class Tanh(Function):
    """
    双曲正切
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return y

    def backward(self, gy: IVariable) -> IVariable:
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x: any) -> IVariable:
    return Tanh()(x)


class Exp(Function):
    """
    指数函数
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.exp(x)
        return y

    def backward(self, gy: IVariable) -> IVariable:
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

    def backward(self, gy: IVariable) -> IVariable:
        x = self.inputs[0]
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


# ==========================================================================
# 基本超越算子
# ==========================================================================


# ==========================================================================
# 张量形状算子
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

    def backward(self, dout: IVariable) -> IVariable:
        return reshape(dout, self.__o_shape)


def reshape(x: np.ndarray | IVariable | list[int], shape: tuple[int]) -> IVariable:
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

    __n_axes: tuple[int]

    def __init__(self, n_axes=None):
        super().__init__()
        self.__n_axes = n_axes

    def forward(self, x: np.ndarray):
        return x.transpose(self.__n_axes)

    def backward(self, dout: IVariable) -> IVariable:
        n_axes = self.__n_axes
        if n_axes is None:
            return transpose(dout)
        axis_len = len(n_axes)
        # TODO:这里没搞懂，要连通np.transpose的算法一起搞清楚
        inv_axis = tuple(np.argsort([ax % axis_len for ax in n_axes]))
        return transpose(dout, inv_axis)


def transpose(x: np.ndarray | IVariable | list[int], axes=None):
    """
    list[int]代表原生多维数组
    """
    return Transpose(axes)(x)


# ==========================================================================
# 张量形状算子
# ==========================================================================

# ==========================================================================
# 常用张量算子
# ==========================================================================


class Sum(Function):
    __keepdims: bool
    __axes: tuple[int] | int
    __from_shape: tuple[int]

    def __init__(self, keep_dims: bool = False, axes: int | tuple[int] | None = None):
        super().__init__()
        self.__keepdims = keep_dims
        self.__axes = axes

    def forward(self, x: np.ndarray) -> int | float:
        self.__from_shape = x.shape
        return np.sum(axis=self.__axes, keepdims=self.__keepdims)

    def backward(self, dout: IVariable) -> IVariable:
        keepdims = self.__keepdims
        axes = self.__axes
        from_shape = self.__from_shape
        if keepdims:  # 正向传播时求和维度的没有被删除，直接reshape即可
            return broadcast_to(dout, from_shape)
        if axes is None:  # 所有轴求和，dout此时是一个标量，直接广播即可
            return broadcast_to(dout, from_shape)
        to_shape: list[int] = list[dout.shape]
        # 补全被删去的求和维度
        # 比如：原来（2，3，4，5，6)，按轴（1，3）求和得到（2，4，6）
        # 还原时需要先还原小索引，这样才不至于破坏后面的所有
        for axis in sorted(axes):
            to_shape.insert(axis, 1)
        return broadcast_to(reshape(dout, tuple(to_shape)), from_shape)


def sum(x: any, axes: tuple[int] | int = None, keepdims=False) -> IVariable:

    return Sum(axes=axes, keep_dims=keepdims)(x)


class BroadcastTo(Function):
    """
    广播扩展算子
    from_shape - 扩展前张量形状
    to_shape   - 目标扩展形状

    两条约束（遵循标准张量右对齐广播规则）：
    1. len(from_shape) <= len(to_shape)；
       若from_shape维度更少，则在左侧（前置）自动补长度为1的维度；
    2. 两个形状执行右对齐；对齐后的每一维，尺寸必须相等，或from_shape对应维度尺寸为1。
    """

    __from_shape: tuple[int]
    __to_shape: tuple[int]

    def __init__(self, shape: tuple[int]):
        super().__init__()
        self.__to_shape = shape
        self.__from_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        from_shape = self.__from_shape = x.shape
        if from_shape == self.__to_shape:
            return x
        return np.broadcast_to(x, self.__to_shape)

    def backward(self, dout: IVariable) -> IVariable:
        return sum_to(dout, self.__from_shape)


def broadcast_to(x: np.ndarray, shape: tuple[int]) -> IVariable:
    return BroadcastTo(shape=shape)(x)


class SumTo(Function):
    """
    规约对齐求和算子

    shape - 求和前的维度

    to_shape - 求和后的维度

    有一下两点约束：

    1. len(shape) >= len(to_shape)，对于前置多的维度，会被压缩掉
    2. 形状按右对齐，每个维度要想等或to_shape维度为1

    """

    __from_shape: tuple[int]  # 原始形状

    __to_shape: tuple[int]  # 要规约到的形状

    def __init__(self, shape: tuple[int]):
        super().__init__()
        self.__to_shape = shape
        self.__from_shape = None

    def _raise_invalid_ndim_error(shape: tuple[int], to_shape: tuple[int]) -> None:
        raise ValueError(
            f"不符合SumTo的形状规则：\n原始形状：{shape}\n目标形状：{to_shape}"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:

        f_shape = self.__from_shape = x.shape
        t_shape = self.__to_shape
        if f_shape == t_shape:
            return x
        f_ndim = len(f_shape)
        t_ndim = len(t_shape)
        diff = f_ndim - t_ndim
        if diff < 0:
            self._raise_invalid_ndim_error()

        sum_axes: list[int] = list(range(diff))  # 那些轴要求和，前面多的维度肯定要求和

        for f_index in range(diff, f_ndim):
            t_dim = t_shape[f_index - diff]
            if f_shape[f_index] == t_dim:
                continue
            if t_dim == 1:
                sum_axes.append(f_index)
                continue
            # 目标维既不是1，也不相等，无法处理
            self._raise_invalid_ndim_error()
        result: np.ndarray = np.sum(x, axis=tuple(sum_axes), keepdims=True)
        if diff > 0:  # 前面多的维度被压缩了，要裁剪掉
            result = result.squeeze()
        return result

    def backward(self, dout: IVariable) -> IVariable:
        return broadcast_to(dout, self.__from_shape)


def sum_to(x: np.ndarray, shape: tuple[int]) -> IVariable:
    return SumTo(shape=shape)(x)


# ==========================================================================
# 常用张量算子
# ==========================================================================
