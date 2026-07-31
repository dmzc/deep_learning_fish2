from __future__ import annotations
import numpy as np
from graphviz import Digraph
import psutil
import os
import weakref
from steps.draft.config import ENABLE_BACKPROGATION


class Variable:
    name: str
    data: np.ndarray
    grad: np.ndarray
    grad_str: str
    creator: Function
    generation: int
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, creator: Function = None, name: str = None):
        if not isinstance(data, np.ndarray):
            raise TypeError("{}is not supportted".format(type(data)))
        self.data = data
        self.grad = None
        self.grad_str = None
        if name is not None:
            self.name = name
        else:
            self.name = f"{data}"
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
            self.grad_str = f"{self.grad}"
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
                    x.grad_str = f"{gx}"
                else:
                    x.grad = x.grad + gx
                    x.grad_str = f"{x.grad_str}+{gx}"
                if x.creator is not None:
                    add_creator(x.creator)
            if not retain_grad:
                for y in creator.outputs:
                    y().grad = None

    def get_label(self, display=True) -> str:
        if display:
            return f"数据：{self.data}\n梯度：{self.grad_str}"
        return f"{id(self)}"

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
        outputs = [
            Variable(as_array(y), creator, f"{self.__class__.__name__}-output-{y}")
            for y in ys
        ]
        if ENABLE_BACKPROGATION:
            self.outputs = [weakref.ref(output) for output in outputs]
            self.inputs = inputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: any) -> any:
        raise NotImplementedError

    def backward(self, dout: any) -> any:
        raise NotImplementedError

    def get_label(self, display=True) -> str:
        if display:
            return f"{self.__class__.__name__}({self.generation})"
        else:

            return f"{id(self)}"


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
        gx1 = gy * (-x0 / x1 ** 2)
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
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


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


def render(x: Variable):

    # 有向图，从左到右流向LR
    dot = Digraph("signal_flow", format="png")
    dot.attr(rankdir="LR", fontname="SimHei")  # LR=从左到右

    variables: set = set()
    functions: set = set()
    handled_funtions: set = set()

    def add_node(node: Function, is_variable=True):
        name = node.get_label(False)
        label = node.get_label(True)
        if is_variable:
            if name not in variables:
                variables.add(name)
                dot.node(
                    name,
                    label,
                    shape="circle",
                    width="1.2",
                    fixedsize="1",
                    style="filled",
                    fontname="SimHei",
                    fillcolor="green",
                    fontsize="12",
                )
        else:
            if name not in functions:
                functions.add(name)
                dot.node(
                    name,
                    label,
                    shape="box",
                    style="filled,rounded",
                    fillcolor="#87CEEB",
                    fontname="SimHei",
                    fontsize="12",
                )

    def add_edge(left: str, right: str, is_grad=False, label: str = None):
        if label is None:
            label = ""
        if is_grad:
            dot.edge(left, right, style="dashed", label=label)
        else:
            dot.edge(left, right, label=label)

    def process(func: Function):
        if func is None:
            return
        if func not in handled_funtions:
            handled_funtions.add(func)
        else:
            return

        inputs = func.inputs
        outputs = func.outputs

        func_name = func.get_label(False)
        add_node(func, False)

        for output in outputs:
            add_node(output)
            output_name = output.get_label(False)
            add_edge(func_name, output_name)
            # add_edge(output_name, func_name, is_grad=True, label=f"{output.grad}")

        for input in inputs:
            add_node(input)
            add_edge(input.get_label(False), func_name)
            process(input.creator)

    process(x.creator)
    dot.render("signal_diagram", view=True)
    input("图形已打开，按回车键关闭程序...")


ys = add(Variable(np.array(2)), Variable(np.array(3)))
print(ys.data)

x = Variable(np.array(2.0))
y = Variable(np.array(3.0))
z: Variable = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)

print("----" * 20)

x = Variable(np.array(3.0))
y: Variable = add(x, x)
y.backward()
print("相同自变量梯度累加", x.grad)


x = Variable(np.array(3.0))  # or x.cleargrad()
y = add(add(x, square(x)), x)
y.backward()
print(x.grad)
# render(y)

# for i in range(100):
# 存在循环引用时（Function.outputs），内存会一直在800MB
# p = psutil.Process(os.getpid())
# rss_mb = p.memory_info().rss / 1024 / 1024
# print(f"{i}当前内存占用：{rss_mb:.2f} MB")
# x = Variable(np.random.randn(1000000))
# y = square(square(square(x)))
# objgraph.show_growth()
# func_objs = objgraph.by_type("Function")
# if func_objs:
#     # 生成图片，直观看到互相引用的循环
#     objgraph.show_backrefs(func_objs[0], filename="func_cycle_ref.png")

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

# y = add(mul(a, b), c)
y = a * np.array(3) + c
y.backward()

print(a.grad)

y = 3.0 * b
y.backward()
print(b.grad)
