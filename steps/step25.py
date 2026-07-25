from dezero import IVariable, IVariableArgs, render, create_variable
from pathlib import Path

current_file = Path(__file__)
current_dir = current_file.parent


x = create_variable(IVariableArgs(data=2.0, is_input=True))
y = create_variable(IVariableArgs(data=3.0, is_input=True))
z: IVariable = (
    1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
) * (
    30
    + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
)

z.backward()
render(z, current_dir / f"{current_file.stem}.dot")
