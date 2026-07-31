from mtorch.interfaces import IModule, ITensor
from collections.abc import Iterable


class Module(IModule):

    __subs: set[str]

    def __init__(self):
        self.__subs = set()

    def __setattr__(self, name, value):
        if isinstance(value, ITensor):
            tensor: ITensor = value
            if tensor.require_grad:
                self.__subs.add(name)
        if isinstance(value, IModule):
            self.__subs.add(name)
        super().__setattr__(name, value)

    def params(self) -> Iterable[ITensor]:
        subs = self.__subs
        if subs is None:
            return
        for sub in subs:
            obj = getattr(sub)
            if isinstance(obj, ITensor):
                return obj
            else:
                m: IModule = obj
                return m.params()
