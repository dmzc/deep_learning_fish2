from steps.step1_10 import square, exp, Variable, numerical_diff
import numpy as np


def test_square():
    x = Variable(np.array(2.0))
    y = square(x)
    y.backward()
    assert y.data == np.array(4.0), "测试square函数能正向传播"
    assert x.grad == np.array(4.0), "测试square函数能反向传播"
    num_grad = numerical_diff(square, x)
    assert np.allclose(x.grad, num_grad), "square函数数值微分和自动微分结果基本相同"


def test_exp():
    x = Variable(np.array(0))
    y = exp(x)
    y.backward()
    assert y.data == np.array(1), "测试square函数能正向传播"
    assert x.grad == np.array(1), "测试square函数能反向传播"
    assert np.allclose(
        x.grad, numerical_diff(exp, x)
    ), "exp函数数值微分和自动微分结果基本相同"
