import os
import pandas as pd

# 设置包含 CSV 文件的文件夹路径
folder_path = '7-covers_new'

# 遍历文件夹中的所有文件和子文件夹
for entry in os.listdir(folder_path):
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, entry)
    
    # 检查文件是否为 CSV 文件
    if file_path.endswith('.csv'):
        try:
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            
            # 在这里对数据框进行操作
            # 获取行数
            num_rows = df.shape[0]  # 获取行数

            # 打印行数
            print(f"{entry} : {num_rows} rows")

        except Exception as e:
            print(f"Failed to read {file_path}: {e}")