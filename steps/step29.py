from dezero_simple import Variable
from dezero_simple.core import sin
from dezero_simple.render import render
import numpy as np

x: Variable = Variable(np.pi / 4, name="X", is_input=True)
y: Variable = sin(x)
y.backward()
grad = x.grad
x.clear_grad()
y.clear_grad()
grad.backward()
# render([grad])
grad = x.grad
x.clear_grad()
y.clear_grad()
grad.backward()
render(grad)
