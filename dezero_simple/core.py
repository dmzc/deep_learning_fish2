from __future__ import annotations
import numpy as np
import weakref
from config import ENABLE_BACKPROGATION
from dataclasses import dataclass


@dataclass
class VariableArgs:
    data: any  # 数值、np.ndarray，不能是Variable实例
    creator: Function = None
    name: str = None
    is_input: bool = False


class Variable:
    __name: str
    is_input: bool
    data: np.ndarray
    grad: Variable
    creator: Function
    generation: int
    __array_priority__ = 200

    def __init__(self, args: VariableArgs):
        self.data = args.data
        self.grad = None
        self.__name = args.name
        self.is_input = args.is_input
        if ENABLE_BACKPROGATION:
            creator = self.creator = args.creator
            if args.creator is None:
                self.generation = 0
            else:
                self.generation = creator.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad=False) -> None:
        if self.creator is None:
            return

        if self.grad is None:
            self.grad = create_variable(
                VariableArgs(data=np.ones_like(self), is_input=self.is_input)
            )
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
                x.grad.is_input = x.is_input
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

    def __call__(self, *xs: tuple[any]) -> list[Variable] | Variable:
        inputs = [create_variable(x) for x in xs]
        xs_data = [x.data for x in inputs]
        ys = self.forward(*xs_data)
        if not isinstance(ys, tuple):
            ys = (ys,)
        creator = None
        if ENABLE_BACKPROGATION:
            self.generation = max([x.generation for x in inputs])
            creator = self
        outputs = [create_variable(VariableArgs(data=y, creator=creator)) for y in ys]
        if ENABLE_BACKPROGATION:
            self.outputs = [weakref.ref(output) for output in outputs]
            self.inputs = inputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: any) -> any:
        raise NotImplementedError

    def backward(self, dout: Variable) -> any:
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


def to_tensor(x: any) -> np.ndarray:
    if isinstance(x, Variable):
        return x
    if np.isscalar(x):
        return np.array(x)
    return x


def create_variable(args: VariableArgs) -> Variable:
    data = args.data
    if isinstance(data, Variable):
        return data
    if np.isscalar(data):
        args.data = np.array(data)
    return Variable(args)
