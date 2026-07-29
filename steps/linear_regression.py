# 使用dezero实现线性回归
import numpy as np
import matplotlib.pyplot as plt
from dezero import F, create_variable, IVariable, IVariableArgs


np.random.seed(0)
x: np.ndarray = np.random.rand(100, 1)
y: np.ndarray = 5 + 2 * x + np.random.rand(100, 1)

W = create_variable(IVariableArgs(data=np.zeros((1, 1))))
b = create_variable(IVariableArgs(data=np.zeros((1, 1))))

lr = 0.1
iters = 100

for cnt in range(iters):
    y_pred: IVariable = F.dot(x, W) + b
    diff: IVariable = y - y_pred
    # loss: IVariable = F.sum((y - y_pred) ** 2) / len(diff)
    loss: IVariable = F.mean_square_loss(y, y_pred)
    print(f"第{cnt+1}损失：{loss}")

    W.clear_grad()
    b.clear_grad()
    loss.backward(retain_grad=True)

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(f"W:{W.data}    b:{b.data}")


plt.scatter(x, y, s=10)
plt.xlabel("x")
plt.ylabel("y")

y_pred: IVariable = F.dot(x, w=W) + b
plt.plot(x, y_pred.data, color="r")

plt.show()
