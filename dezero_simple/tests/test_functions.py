from dezero_simple import F, create_variable, VariableArgs
import numpy as np


def test_reshape():
    result = F.reshape([1, 2, 3, 4, 5, 6], (2, 3)).data.tolist()
    assert result == [[1, 2, 3], [4, 5, 6]], "单行数组正常reshape"
    result = F.reshape([[1, 2], [3, 4], [5, 6]], (2, 3)).data.tolist()
    assert result == [[1, 2, 3], [4, 5, 6]], "多行数组正常reshape"
    result = F.reshape(np.array([1, 2, 3, 4, 5, 6]), (2, 3)).data.tolist()
    assert result == [[1, 2, 3], [4, 5, 6]], "numpy数组正常reshape"
    result = F.reshape(
        create_variable(VariableArgs(data=[1, 2, 3, 4, 5, 6])), (2, 3)
    ).data.tolist()
    assert result == [[1, 2, 3], [4, 5, 6]], "variable正常reshape"


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
