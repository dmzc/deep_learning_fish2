import numpy as np
from dezero_simple import Variable


x = Variable(np.array(1.0))
y: Variable = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
