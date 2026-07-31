from mtorch1 import ITensor, Tensor


x = Tensor(data=1.0)
y: ITensor = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
