from dezero import IVariable, IVariableArgs, render, create_variable
from pathlib import Path

current_file = Path(__file__)
current_dir = current_file.parent

a = create_variable(IVariableArgs(data=2.0, is_input=True, name="a"))
b = create_variable(IVariableArgs(data=3.0, is_input=True, name="b"))
y: IVariable = a
while True:
    y = y * b
    if y.data > 23:
        break
y.backward(retain_grad=True)
render(y, current_dir / f"{current_file.stem}.dot")
