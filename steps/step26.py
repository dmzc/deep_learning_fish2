from dezero_simple import Variable, render
from dezero_simple.core import maclaurin_sin, sin
from pathlib import Path
import numpy as np

current_file = Path(__file__)

x = Variable(np.pi / 4, name="X")
y: Variable = maclaurin_sin(x)
y.backward(retain_grad=True)
print("麦克劳林-sin", y.data)
print("麦克劳林-梯度", x.grad)
x1 = Variable(np.pi / 4, name="X1")
y1: Variable = sin(x1)
y1.backward(retain_grad=True)
print("np.sin-切比雪夫", y1.data)
print("np.sin-梯度)", x1.grad)
render([y, y1], current_file.parent / (current_file.stem + ".dot"))
