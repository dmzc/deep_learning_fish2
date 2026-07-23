from dezero_simple import Variable, VariableArgs, F, render, create_variable
import numpy as np


x: Variable = create_variable(VariableArgs(data=np.pi / 4, name="X", is_input=True))
y: Variable = F.sin(x)
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
