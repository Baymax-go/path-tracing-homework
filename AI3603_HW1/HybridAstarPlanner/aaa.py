import math
import numpy as np

# print(math.hypot(-1, 1))


# s = np.arange(np.deg2rad(40)/ 20.0,
#             np.deg2rad(40), np.deg2rad(40) / 20.0)
#
# print(s)
#
# steer = list(s) + [0.0] + list(-s)
# print(steer)
#
# direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
# print(direc)
#
# steer = steer + steer
# print(steer)


matrix_q = [0.5, 0.0, 1.0, 0.0]     # Q矩阵
matrix_r = [1.0]                    # R矩阵

matrix_r_ = np.diag(matrix_r)  # array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵    #array是一个二维矩阵时，结果输出矩阵的对角线元素
matrix_q_ = np.diag(matrix_q)  # 计算出 Q矩阵， R矩阵

print(matrix_r_)
print(matrix_q_)