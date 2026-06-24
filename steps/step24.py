import numpy as np
from dezero_simple import Variable, render
from pathlib import Path

current_file = Path(__file__)
current_dir = current_file.parent

a = Variable(np.array(2.0), is_input=True, name="a")
b = Variable(np.array(3.0), is_input=True, name="b")
y: Variable = a
while True:
    y = y * b
    if y.data > 23:
        break
y.backward(retain_grad=True)
render(y, current_dir / f"{current_file.stem}.dot")
