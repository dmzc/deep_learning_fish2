from mtorch1 import ITensor, F, render, Tensor
import numpy as np

# 测试高阶导

x: ITensor = Tensor(data=np.pi / 4, name="X", is_input=True)
y: ITensor = F.sin(x)
y.backward()
grad = x.grad
x.clear_grad()
y.clear_grad()
grad.backward()
# render([grad])
grad = x.grad
x.clear_grad()
# y.clear_grad()
grad.backward()
render(grad)
