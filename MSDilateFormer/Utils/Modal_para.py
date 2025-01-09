import numpy as np
from scipy.linalg import svd, eig
import matplotlib.pyplot as plt
"""
    模态参数识别的方法是快速贝叶斯傅里叶变换方法，是易稳师姐帮忙实现的

"""
# 加载数据
# real_freq = np.loadtxt("D:\Fanyang\GNTT预处理\out_true_freq.txt")
real_shape = np.loadtxt("D:\Fanyang\GNTT预处理\out_true_shape.txt")

# S10_freq = np.loadtxt("D:\Fanyang\GNTT预处理\out_S10_freq.txt")
# S10_shape = np.loadtxt("D:\Fanyang\GNTT预处理\out_S10_shape.txt")

# S30_freq = np.loadtxt("D:\Fanyang\GNTT预处理\out_S30_freq.txt")
S30_shape = np.loadtxt("D:\Fanyang\GNTT预处理\out_S30_shape.txt")

# # 检查数据形状
# print("Shape of real_freq :", real_freq.shape)  
# print("Shape of real_freq :", S10_freq.shape) 
# print("Shape of real_freq :", S30_freq.shape)

print("Shape of real_shape:", real_shape.shape)  # 应该是 (20, 10)
# print("Shape of real_shape:", S10_shape.shape)  # 应该是 (20, 10)
print("Shape of real_shape:", S30_shape.shape)  # 应该是 (20, 10)


# # 计算固有频率相对误差
# relative_errors = ((S10_freq - real_freq) / real_freq)*100
# print("S10 Relative Errors (%):", relative_errors)

# relative_errors = ((S30_freq - real_freq) / real_freq)*100
# print("S30 Relative Errors (%):", relative_errors)

# # 计算振型的MAC评价
# def calculate_MAC(mode_shapes_real, mode_shapes_reconstructed):
#     num_modes = mode_shapes_real.shape[1]
#     MAC = np.zeros((num_modes, num_modes))

#     for i in range(num_modes):
#         for j in range(num_modes):
#             MAC[i, j] = (np.abs(np.dot(mode_shapes_real[:, i].conj().T, mode_shapes_reconstructed[:, j])) ** 2 /
#                          (np.dot(mode_shapes_real[:, i].conj().T, mode_shapes_real[:, i]) *
#                           np.dot(mode_shapes_reconstructed[:, j].conj().T, mode_shapes_reconstructed[:, j])))
#     return MAC
def vec(matrix):
    """
    将矩阵展开成向量
    """
    return matrix.flatten('F')

def calculate_MAC(A, B=None):
    """
    计算模态保证准则 (MAC)
    
    参数:
    A : numpy array
        模态振型矩阵，形状为 (num_sensors, num_modes)
    B : numpy array, optional
        第二个模态振型矩阵，形状为 (num_sensors, num_modes)
        
    返回:
    M : numpy array
        MAC 值矩阵
    """
    if B is None:
        # 只提供 A 时，计算 A 的自 MAC 值
        M = (A.T @ A) / (np.linalg.norm(A, axis=0).reshape(-1, 1) * np.linalg.norm(A, axis=0).reshape(1, -1))
        M = vec(M)
    else:
        if A.shape[0] != B.shape[0]:
            raise ValueError("The dimensions of input matrices must be consistent!")
        
        # 提供 A 和 B 时，计算 A 和 B 的 MAC 值
        M = (A.T @ B) / (np.linalg.norm(A, axis=0).reshape(-1, 1) * np.linalg.norm(B, axis=0).reshape(1, -1))
    
    return M

# S10_MAC = calculate_MAC(real_shape, S10_shape)
S30_MAC = calculate_MAC(real_shape, S30_shape)
# print("MAC Matrix:\n", MAC)

# # 只输出对应同一个模态的MAC
# S10_MAC = np.diag(S10_MAC)
S30_MAC = np.diag(S30_MAC)
# print("S10 MAC Values (same mode):", S10_MAC)
print("S30 MAC Values (same mode):", S30_MAC)


# 绘制每一阶振型图
def plot_mode_shapes(mode_shapes_real, mode_shapes_reconstructed, num_modes=5):
    for i in range(num_modes):
        plt.figure()
        plt.plot(real_shape[:, i], label='Real Mode Shape {}'.format(i+1))
        plt.plot(S30_shape[:, i], label='Reconstructed Mode Shape {}'.format(i+1))
        plt.xlabel('Node')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title('Mode Shape Comparison for Mode {}'.format(i+1))
        plt.savefig(f'mode_shape_comparison_mode_{i+1}.png')
        plt.show()

# # 绘制前5阶振型图
plot_mode_shapes(real_shape, S30_shape, num_modes=min(10, real_shape.shape[1]))