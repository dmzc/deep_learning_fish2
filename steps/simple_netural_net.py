from mtorch1 import F, Tensor, ITensor
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x: np.ndarray = np.random.rand(100, 1)
y: np.ndarray = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
w1 = Tensor(data=0.01 * np.random.randn(I, H))
b1 = Tensor(data=np.zeros(H))
w2 = Tensor(data=0.01 * np.random.randn(H, O))
b2 = Tensor(data=np.zeros(O))


def predict(x: np.ndarray) -> np.ndarray:
    y = F.linear(x, w1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, w2, b2)
    return y


lr = 0.2
iters = 10000

for index in range(iters):
    y_actual: ITensor = predict(x)
    loss: ITensor = F.mean_square_loss(y_actual=y_actual, y_expect=y)
    if index % 1000 == 0:
        print(f"第{index}轮损失{loss}")

    w1.clear_grad()
    w2.clear_grad()
    b1.clear_grad()
    b2.clear_grad()

    loss.backward()
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    b1.data -= lr * b1.grad.data
    b2.data -= lr * b2.grad.data

plt.scatter(x, y, s=10)
plt.xlabel("x")
plt.ylabel("y")
# 这里为了画出正弦图像，要用连续的自变量，而x随机生成的，所以不满足要求
t = np.arange(0, 1, 0.01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color="r")

plt.show()
