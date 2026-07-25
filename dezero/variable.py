from __future__ import annotations
import numpy as np
from dezero.config import ENABLE_BACKPROGATION
from dezero.interfaces import IVariable, IFunction, IVariableArgs


class Variable(IVariable):

    __name: str

    def __init__(self, args: IVariableArgs):
        super().__init__()
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
                IVariableArgs(data=np.ones_like(self), is_input=self.is_input)
            )
        creators: list[IFunction] = []
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


def create_variable(args: IVariableArgs | any) -> IVariable:
    if not isinstance(args, IVariableArgs):
        args = IVariableArgs(data=args)
    data = args.data
    if isinstance(data, IVariable):
        return data
    if np.isscalar(data) or isinstance(data, list) or isinstance(data, tuple):
        args.data = np.array(data)
    return Variable(args)
