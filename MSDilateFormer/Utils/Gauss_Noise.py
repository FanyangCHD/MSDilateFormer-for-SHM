import numpy as np

def add_gaussian_noise(data, noise_level):
    """
    给数据添加高斯噪声
    :param data: 输入数据
    :param noise_level: 噪声水平（以百分比表示）
    :return: 添加噪声后的数据
    """
    mean = 0
    std = noise_level * np.mean(np.abs(data))
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def mask_rows(data, rows):
    """
    将指定行补0
    :param data: 输入数据
    :param rows: 要补0的行索引列表
    :return: 行补0后的数据
    """
    masked_data = data.copy()
    masked_data[rows, :] = 0
    return masked_data

def save_noisy_data(file_path, data, noise_levels, rows_to_mask):
    """
    保存添加高斯噪声和行补0后的数据
    :param file_path: 原始数据的文件路径
    :param data: 输入数据
    :param noise_levels: 噪声水平列表
    :param rows_to_mask: 要补0的行索引列表
    """
    for noise_level in noise_levels:
        noisy_data = add_gaussian_noise(data, noise_level)
        masked_data = mask_rows(noisy_data, rows_to_mask)
        save_path = file_path.replace('.npy', f'_noise_{int(noise_level*100)}.npy')
        np.save(save_path, masked_data)
        print(f"保存文件: {save_path}")

# 读取原始npy文件
file_path = 'D:\Fanyang\SHM_Data\GNTT\\S30\\feature\missing_1666.npy'
data = np.load(file_path)

# 定义噪声水平和要补0的行
noise_levels = [0.55, 0.6]
rows_to_mask = [2, 5, 7, 12, 14, 15]

file_path1 = 'D:\Fanyang\SHM_Data\GNTT\\S30'
# 保存添加高斯噪声和行补0后的数据
save_noisy_data(file_path, data, noise_levels, rows_to_mask)
