from __future__ import annotations
import numpy as np
import weakref
from dataclasses import dataclass


class IVariable:

    is_input: bool
    data: np.ndarray
    grad: IVariable
    creator: IFunction
    generation: int
    __array_priority__ = 200

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def shape(self): ...

    @property
    def ndim(self): ...

    @property
    def size(self): ...

    @property
    def dtype(self): ...

    def clear_grad(self) -> None: ...

    def backward(self, retain_grad=False) -> None: ...


class IFunction:
    inputs: list[IVariable]
    outputs: list[weakref.ref[IVariable]]
    label: str
    generation: int

    def forward(self, *xs: np.ndarray) -> any: ...

    def backward(self, dout: IVariable) -> IVariable: ...

    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...


@dataclass
class IVariableArgs:
    data: any  # 数值、np.ndarray，不能是Variable实例
    creator: IFunction = None
    name: str = None
    is_input: bool = False
