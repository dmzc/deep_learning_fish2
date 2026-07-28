from dezero import F, create_variable, IVariableArgs, IVariable
import numpy as np


def test_add():
    v1 = create_variable(IVariableArgs(data=2.0))
    v2 = create_variable(IVariableArgs(data=8.0))
    v3: IVariable = v1 + v2
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == 2.0 and v2.grad.data.tolist() == 2.0, "加法反向传播"

    v1 = create_variable(IVariableArgs(data=2.0))
    v2 = create_variable(IVariableArgs(data=[[1, 2, 3], [4, 5, 6]]))
    v3: IVariable = v1 + v2
    v3.grad = create_variable(IVariableArgs(data=[[1, 1, 1], [3, 3, 3]]))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == 12 and v2.grad.data.tolist() == [
        [1, 1, 1],
        [3, 3, 3],
    ], "前向传播发生补全后也能正常反向传播"


def test_neg():
    v1 = create_variable(IVariableArgs(data=2.0))
    v3: IVariable = -v1
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == -2.0, "加法逆元反向传播"


def test_sub():
    v1 = create_variable(IVariableArgs(data=2.0))
    v2 = create_variable(IVariableArgs(data=8.0))

    v3: IVariable = v1 - v2
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert (
        v1.grad.data.tolist() == 2.0 and v2.grad.data.tolist() == -2.0
    ), "减法反向传播"

    v1 = create_variable(IVariableArgs(data=2.0))
    v2 = create_variable(IVariableArgs(data=[[1, 2, 3], [4, 5, 6]]))
    v3: IVariable = v1 - v2
    v3.grad = create_variable(IVariableArgs(data=[[1, 1, 1], [3, 3, 3]]))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == 12 and v2.grad.data.tolist() == [
        [-1, -1, -1],
        [-3, -3, -3],
    ], "前向传播发生补全后也能正常反向传播"


def test_mul():
    v1 = create_variable(IVariableArgs(data=2.0))
    v2 = create_variable(IVariableArgs(data=8.0))

    v3: IVariable = v1 * v2
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == 16 and v2.grad.data.tolist() == 4, "乘法反向传播"

    v1 = create_variable(IVariableArgs(data=2.0))
    v2 = create_variable(IVariableArgs(data=[[1, 2, 3], [4, 5, 6]]))
    v3: IVariable = v1 * v2
    v3.grad = create_variable(IVariableArgs(data=[[1, 1, 1], [3, 3, 3]]))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == 51 and v2.grad.data.tolist() == [
        [2, 2, 2],
        [6, 6, 6],
    ], "前向传播发生补全后也能正常反向传播"


def test_div():
    v1 = create_variable(IVariableArgs(data=8.0))
    v2 = create_variable(IVariableArgs(data=2.0))

    v3: IVariable = v1 / v2
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == 1 and v2.grad.data.tolist() == -4.0, "乘法反向传播"

    v1 = create_variable(IVariableArgs(data=[[2, 4, 6], [8, 10, 12]]))
    v2 = create_variable(IVariableArgs(data=2.0))
    v3: IVariable = v1 / v2
    v3.grad = create_variable(IVariableArgs(data=[[2, 2, 2], [4, 4, 4]]))
    v3.backward(retain_grad=True)
    assert (
        v1.grad.data.tolist()
        == [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
        and v2.grad.data.tolist() == -36.0
    ), "前向传播发生补全后也能正常反向传播"


def test_pow():

    v1 = create_variable(IVariableArgs(data=2.0))
    v3: IVariable = v1**3
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == 24, "幂操作反向传播"


def test_sin():
    v1 = create_variable(IVariableArgs(data=0))
    v3: IVariable = F.sin(v1)
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == 2, "sin函数反向传播"


def test_cos():
    v1 = create_variable(IVariableArgs(data=np.pi / 2))
    v3: IVariable = F.cos(v1)
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == -2, "cos函数反向传播"


def test_tanh():
    v1 = create_variable(IVariableArgs(data=2))
    v3: IVariable = F.tanh(v1)
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert np.allclose(v1.grad.data.tolist(), 0.1413016), "双曲正切函数反向传播"


def test_exp():
    v1 = create_variable(IVariableArgs(data=0))
    v3: IVariable = F.exp(v1)
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == 2.0, "指数函数反向传播"


def test_log():
    v1 = create_variable(IVariableArgs(data=2))
    v3: IVariable = F.log(v1)
    v3.grad = create_variable(IVariableArgs(data=2.0))
    v3.backward(retain_grad=True)
    assert v1.grad.data.tolist() == 1.0, "对数函数反向传播"


def test_reshape():
    result = F.reshape([1, 2, 3, 4, 5, 6], (2, 3)).data.tolist()
    assert result == [[1, 2, 3], [4, 5, 6]], "单行数组正常reshape"
    result = F.reshape([[1, 2], [3, 4], [5, 6]], (2, 3)).data.tolist()
    assert result == [[1, 2, 3], [4, 5, 6]], "多行数组正常reshape"
    result = F.reshape(np.array([1, 2, 3, 4, 5, 6]), (2, 3)).data.tolist()
    assert result == [[1, 2, 3], [4, 5, 6]], "numpy数组正常reshape"
    result = F.reshape(
        create_variable(IVariableArgs(data=[1, 2, 3, 4, 5, 6])), (2, 3)
    ).data.tolist()
    assert result == [[1, 2, 3], [4, 5, 6]], "variable正常reshape"

    f_variable = create_variable(IVariableArgs(data=[1, 2, 3, 4, 5, 6]))
    t_variable = F.reshape(f_variable, (2, 3))
    # fmt:off
    t_variable.grad=create_variable(
        IVariableArgs(data=[
            [1,2,3],
            [4,5,6]
    ]))
    t_variable.backward(retain_grad=True)
    assert f_variable.grad.data.tolist()==[1,2,3,4,5,6],"反向传播正常"


def test_transpose():
    result = F.transpose([[1, 2, 3], [4, 5, 6]]).data.tolist()
    assert result == [[1, 4], [2, 5], [3, 6]], "矩阵转置"

    # fmt: off
    result = F.transpose(
        [
            [
                [1],
                [3], 
                [5]
            ],
            [
                [7],
                [9],
                [11]
            ]
        ],
        (2,0,1)
    ).data.tolist()
    
    # fmt: off
    assert result == [
        [
            [1, 3, 5],
            [7, 9, 11]
        ]
    ], "交换轴"
    
    f_variable = create_variable(IVariableArgs(data=([[1, 2, 3], [4, 5, 6]])))
    t_variable = F.transpose(f_variable)
    # fmt:off
    t_variable.grad=create_variable(
        IVariableArgs(data=[[1, 2], [3, 4], [5, 6]]))
    t_variable.backward(retain_grad=True)
    assert f_variable.grad.data.tolist()==[[1,3,5],[2,4,6]],"反向传播正常"


def test_sum_to():
    # fmt:off
    assert F.sum_to([
            [1, 2], 
            [3, 4]
        ], (2, 2)).data.tolist() == [
            [1, 2],
            [3, 4]
        ], "形状相同无操作"
    # fmt:off
    assert F.sum_to([
            [1, 2], 
            [3, 4]
        ], (1, 2)).data.tolist() == [
            [4,6]
        ], "按第一维度求和"
    # fmt:off
    assert F.sum_to([
            [1, 2], 
            [3, 4]
        ], (2, 1)).data.tolist() == [
            [3],
            [7]
        ], "按第二维度求和"

    # fmt:off
    assert F.sum_to([
            [
                [1,2,3], 
                [3,4,5]
            ], 
            [
                [6,7,8],
                [9,10,11]
            ]
        ], (2,3)).data.tolist() == [
            [7,9,11],
            [12,14,16]
        ], "压缩维度"
    f_variable = create_variable(IVariableArgs(data=([[1, 2, 3], [4, 5, 6]])))
    t_variable = F.sum_to(f_variable,(2,1))
    # fmt:off
    t_variable.grad=create_variable(
        IVariableArgs(data=[[1],[2]]))
    t_variable.backward(retain_grad=True)
    assert f_variable.grad.data.tolist()==[[1,1,1],[2,2,2]],"反向传播正常"


def test_broadcast_to():
    assert F.broadcast_to([[1], [2]], (2, 3)).data.tolist() == [
        [1, 1, 1],
        [2, 2, 2],
    ], "按第二维度扩充"
    assert F.broadcast_to([[1, 2, 4]], (2, 3)).data.tolist() == [
        [1, 2, 4],
        [1, 2, 4],
    ], "按第一维度扩充"
    assert F.broadcast_to([[1, 2, 3], [4, 5, 6]], (2, 2, 3)).data.tolist() == [
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
    ], "维度长度不足时会在前补充"
    f_variable = create_variable(IVariableArgs(data=([[1, 2, 3], [4, 5, 6]])))
    t_variable = F.broadcast_to(f_variable, (2, 2, 3))
    # fmt:off
    t_variable.grad=create_variable(
        IVariableArgs(data=[
            [
                [1,2,3],
                [4,5,6]
            ],
            [
                [1,1,1],
                [2,2,2]
            ]
        ]))
    t_variable.backward(retain_grad=True)
    assert f_variable.grad.data.tolist()==[[2,3,4],[6,7,8]],"反向传播正常"


def test_sum():
    v1 = create_variable(IVariableArgs(data=[[1, 2], [3, 4]]))
    v2 = F.sum(v1)
    v2.grad = create_variable(IVariableArgs(data=3.0))
    v2.backward(retain_grad=True)
    assert v2.data.tolist() == 10, "无参数求和"
    assert v1.grad.data.tolist() == [[3.0, 3.0], [3.0, 3.0]], "无参数求和反向传播"

    v1.clear_grad()
    v2 = F.sum(v1, axes=(1,))
    v2.grad = create_variable(IVariableArgs(data=[4.0, 5.0]))
    v2.backward(retain_grad=True)
    assert v2.data.tolist() == [3, 7], "按维度2求和"
    assert v1.grad.data.tolist() == [[4.0, 4.0], [5.0, 5.0]], "按维度2求和反向传播"

    v1.clear_grad()
    v2 = F.sum(v1, axes=(1,), keepdims=True)
    v2.grad = create_variable(IVariableArgs(data=[[4.0], [5.0]]))
    v2.backward(retain_grad=True)
    assert v2.data.tolist() == [[3], [7]], "keepdims为true"
    assert v1.grad.data.tolist() == [[4.0, 4.0], [5.0, 5.0]], "keepdims为true反向传播"


def test_dot():
    # fmt:off
    x = create_variable(IVariableArgs(data=[
        [1, 2, 3], 
        [4, 5, 6]
    ]))
    # fmt:off
    w = create_variable(IVariableArgs(data=[
        [1, 1],
        [2, 2],
        [3, 3]
    ]))
    out: IVariable = F.dot(x, w)
    assert out.data.tolist() == [[14,14],[32,32]],"矩阵运算结果"
    out.grad=create_variable(IVariableArgs(data=[[2,2],[3,3]]))
    out.backward()
    assert x.grad.data.tolist()==np.dot([[2,2],[3,3]],w.data.T).tolist(),"点乘左边能反向传播"
    assert w.grad.data.tolist()==np.dot(x.data.T,[[2,2],[3,3]]).tolist(),"点乘右边能反向传播"
