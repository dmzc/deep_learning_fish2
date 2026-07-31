from mtorch import ITensor, Tensor


def rosenbrock(x0: ITensor, x1: ITensor) -> ITensor:
    return 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2


x0 = Tensor(data=0.0, name="X0")
x1 = Tensor(data=2.0, name="X1")

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
