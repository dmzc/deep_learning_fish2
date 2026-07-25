from dezero.interfaces import IVariable, IVariableArgs
from dezero.variable import create_variable, Variable
import dezero.function as F
from dezero.render import render

__version__ = "0.0.13"

from dezero.setup import _setup

_setup()
del _setup
