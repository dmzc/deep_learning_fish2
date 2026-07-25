from dezero import IVariable, IVariableArgs, F, render, create_variable
import numpy as np


x: IVariable = create_variable(IVariableArgs(data=np.pi / 4, name="X", is_input=True))
y: IVariable = F.sin(x)
y.backward()
grad = x.grad
x.clear_grad()
y.clear_grad()
grad.backward()
# render([grad])
grad = x.grad
x.clear_grad()
# y.clear_grad()
grad.backward()
render(grad)
