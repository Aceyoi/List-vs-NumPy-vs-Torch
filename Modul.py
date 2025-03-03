import time
import numpy as np
import torch

def testcheck():
    testlist = [[1, 2, 3], [4, 5, 6], [7,8,9]]
    testmat = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
    testgpu = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7.,8.,9.]], dtype=torch.long) # matmul не работает с целыми значениями smiledog
    testcpu = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7.,8.,9.]], device='cuda')

    testlist1 = [[30, 36, 42], [66, 81, 96], [102,126,150]]
    testmat1 = np.array([[30, 36, 42], [66, 81, 96], [102,126,150]])
    testgpu1 = torch.tensor([[30., 36., 42.], [66., 81., 96.], [102., 126., 150.]], dtype=torch.long)
    testcpu1 = torch.tensor([[30., 36., 42.], [66., 81., 96.], [102., 126., 150.]], device='cuda')

    testlist2 = multiply_python(testlist, testlist)
    testmat2 = np.dot(testmat, testmat)
    testgpu2 = torch.matmul(testgpu, testgpu)
    testcpu2 = torch.matmul(testcpu, testcpu)


    assert(testlist2 == testlist1)
    assert(np.array_equal(testmat1, testmat2))
    assert(torch.equal(testgpu1, testgpu2))
    assert(torch.equal(testcpu1, testcpu2))

    pass


# Функция для перемножения матриц в виде списков Python
def multiply_python(mat1, mat2):
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result

# Измерение времени для Python List
def timechecklist(matrixlist1, matrixlist2):
    start_time = time.time()
    multiply_python(matrixlist1, matrixlist2)
    end_time = time.time()
    return end_time - start_time

# Измерение времени для NumPy
def timecheckmax(matrixmax1, matrixmax2):
    start_time = time.time()
    np.dot(matrixmax1, matrixmax2)
    end_time = time.time()
    return end_time - start_time

# Измерение времени для PyTorch (CPU)
def timechecktrchcpu(matrixtrchcpu1, matrixtrchcpu2):
    start_time = time.time()
    torch.matmul(matrixtrchcpu1, matrixtrchcpu2)
    end_time = time.time()
    return end_time - start_time

# Измерение времени для PyTorch (GPU)
def timechecktrchgpu(matrixtrchgpu1, matrixtrchgpu2):
    start_time = time.time()
    torch.matmul(matrixtrchgpu1, matrixtrchgpu2)
    end_time = time.time()
    return end_time - start_time