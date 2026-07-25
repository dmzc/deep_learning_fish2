from dezero import IVariable, IVariableArgs, render, F, create_variable
from pathlib import Path
import numpy as np

current_file = Path(__file__)


x = create_variable(IVariableArgs(data=np.pi / 4, name="X"))
y: IVariable = F.maclaurin_sin(x)
y.backward(retain_grad=True)
print("麦克劳林-sin", y.data)
print("麦克劳林-梯度", x.grad)
x1 = create_variable(IVariableArgs(data=np.pi / 4, name="X1"))
y1: IVariable = F.sin(x1)
y1.backward(retain_grad=True)
print("np.sin-切比雪夫", y1.data)
print("np.sin-梯度)", x1.grad)
render([y, y1], current_file.parent / (current_file.stem + ".dot"))
