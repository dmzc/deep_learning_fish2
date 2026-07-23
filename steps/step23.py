from dezero_simple import Variable, VariableArgs, create_variable


x = create_variable(VariableArgs(data=1.0))
y: Variable = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
