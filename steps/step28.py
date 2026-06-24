from dezero_simple import Variable


def rosenbrock(x0: Variable, x1: Variable) -> Variable:
    return 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2


x0 = Variable(0.0, name="X0")
x1 = Variable(2.0, name="X1")

lr = 0.001
iters = 10000

for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)
    x0.clear_grad()
    x1.clear_grad()
    y.backward()
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
