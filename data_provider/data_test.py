import numpy as np
import pandas as pd
import os

root_path = '../dataset/LargeST/SD/'


data_path = '2017,2018,2019,2020,2021'
years = data_path.split(',')  # 将年份字符串分割成列表
concatenated_data = None
for year in years:
    file_path = os.path.join(root_path, year, 'his.npz')  # 构建文件路径
    data = np.load(file_path)
    array = data['data']

    if concatenated_data is None:
        concatenated_data = array
    else:
        # 确保数据在时间维度 T 上拼接
        concatenated_data = np.concatenate((concatenated_data, array), axis=0)

T, N, F = concatenated_data.shape
aggregated_data = concatenated_data.reshape(-1, 4, N, F).sum(axis=1) # 数据按小时聚合
print(aggregated_data.shape)

"""
# 查看邻接矩阵
sd_rn_adj = np.load(root_path + 'sd/sd_rn_adj.npy')
node_num = sd_rn_adj.shape[0]

print(sd_rn_adj[0,0])
sd_rn_adj[np.arange(node_num), np.arange(node_num)] = 0 # 邻接矩阵对角线上的元素设置为0
print(sd_rn_adj[0,0])

# 图中边的数量，计算非零元素的数量
print('edge number', np.count_nonzero(sd_rn_adj))
# 每个节点的平均度数
print('node degree', np.mean(np.count_nonzero(sd_rn_adj, axis=-1)))
# 图的稀疏度。计算非零元素的数量占总可能元素数量的比例并转换为百分比
print('sparsity', np.count_nonzero(sd_rn_adj) / (node_num**2) * 100)
"""

"""
sd_2017 = np.load(root_path + 'sd/2017/his.npz')
data = sd_2017['data']
mean = sd_2017['mean']
std = sd_2017['std']
# data = (data * std) + mean
print(data.shape)

# 每4个时间步进行一个聚合（假设聚合操作是取平均值）
aggregated_data = data.reshape(-1, 4, 716, 3).sum(axis=1)

# 取第一个特征
output_data = aggregated_data[:, :, 0]

# 生成时间戳
start_time = pd.Timestamp('2017-01-01 00:00')
time_index = pd.date_range(start=start_time, periods=output_data.shape[0], freq='H')

# 创建DataFrame
df = pd.DataFrame(output_data, index=time_index)

# 重命名列
df.columns = list(range(715)) + ['OT']

# 保存为CSV文件
df.to_csv(root_path + 'sd/2017/output.csv', index_label='date')

print("CSV文件已成功保存！")
"""













