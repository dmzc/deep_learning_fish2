from dezero import IVariable, IVariableArgs, create_variable


x = create_variable(IVariableArgs(data=1.0))
y: IVariable = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
