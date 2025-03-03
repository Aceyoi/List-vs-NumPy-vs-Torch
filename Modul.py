import time
import numpy as np
import torch

def testcheck():
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
    torch.cuda.synchronize()  # Синхронизация для точного измерения времени на GPU
    end_time = time.time()
    return end_time - start_time