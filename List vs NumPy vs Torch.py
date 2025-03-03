from Modul import * # подключаем модуль
import numpy as np
import torch
import random

testcheck()# Тест на переумножение матриц

n = int(input("Введите размер матрицы "))
# Будут созданы единичная, единичная обратная, случайная матрицы

matrixlistrand = [[random.randint(0, 100) for _ in range(n)] for _ in range(n)] # Создание случайной матрицы в списках python
matrixlist = [[(i * n) + j + 1 for j in range(n)] for i in range(n)] # Создание матрицы от 1 до n в списках python
matrixlistone = [[1 for _ in range(n)] for _ in range(n)] # Создание единичной матрицы в списках python

matrixmaxrand = np.random.randint(0, 100, (n, n)) # Создание случайной матрицы в numpy
matrixmax = np.arange(1, n * n + 1).reshape(n, n) # Создание матрицы от 1 до n в numpy
matrixmaxone = np.ones((n, n)) # Создание единичной матрицы в numpy

matrixtrchcpurand = torch.randint(0, 100, (n, n)) # Создание случайной матрицы в torch CPU
matrixtrchcpu = torch.arange(1, n * n + 1).reshape(n, n) # Создание матрицы от 1 до n в torch CPU
matrixtrchcpuone = torch.ones(n, n) # Создание единичной матрицы в torch CPU

matrixtrchgpurand = torch.randint(0, 100, (n, n), device='cuda') # Создание случайной матрицы в torch GPU
matrixtrchgpu = torch.arange(1, n * n + 1, device='cuda').reshape(n, n) # Создание матрицы от 1 до n в torch GPU
matrixtrchgpuone = torch.ones(n, n, device='cuda') # Создание единичной матрицы в torch GPU


# Вывод результатов
print(f"Случайный список python: {timechecklist(matrixlistrand,matrixlistrand):.6f} секунд")
print(f"Список python: {timechecklist(matrixlist,matrixlist):.6f} секунд")
print(f"Еденичный список python: {timechecklist(matrixlistone,matrixlistone):.6f} секунд")

print(f"Случайная матрица в numpy: {timechecklist(matrixmaxrand,matrixmaxrand):.6f} секунд")
print(f"Матрица в numpy: {timechecklist(matrixmax,matrixmax):.6f} секунд")
print(f"Еденичная матрица в numpy: {timechecklist(matrixmaxone,matrixmaxone):.6f} секунд")

print(f"Случайная матрица в torch CPU: {timechecklist(matrixtrchcpurand,matrixtrchcpurand):.6f} секунд")
print(f"Матрица в torch CPU: {timechecklist(matrixtrchcpu,matrixtrchcpu):.6f} секунд")
print(f"Еденичная матрица в torch CPU: {timechecklist(matrixtrchcpuone,matrixtrchcpuone):.6f} секунд")

print(f"Случайная матрица в torch GPU: {timechecklist(matrixtrchgpurand,matrixtrchgpurand):.6f} секунд")
print(f"Матрица в torch GPU: {timechecklist(matrixtrchgpu,matrixtrchgpu):.6f} секунд")
print(f"Еденичная матрица в torch GPU: {timechecklist(matrixtrchgpuone,matrixtrchgpuone):.6f} секунд")
