import os
import pandas as pd

def merge_csv_to_train_data(input_dir, output_file_data , output_file_label):
    """
    将指定目录下的所有 csv 文件整合成一个 train-data.csv 文件。
    :param input_dir: 包含 csv 文件的输入目录
    :param output_file_data: 输出的 train-data.csv 文件路径
    :param output_file_label：输出的 train-label.csv 文件路径
    """
    # 获取所有 csv 文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    csv_files.sort()  # 按文件名排序

    # 初始化一个空的 DataFrame 用于存储最终结果
    max_columns = 0
    data = []

    # 遍历每个 csv 文件
    for csv_file in csv_files:
        file_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(file_path)
        
        # 提取码流大小列并转置
        bitrate_column = df.iloc[:, 2].values  # 假设码流大小是第三列
        transposed_row = [csv_file] + list(bitrate_column)
        
        # 更新最大列数
        max_columns = max(max_columns, len(transposed_row))
        
        # 将转置后的行添加到数据列表中
        data.append(transposed_row)

    # 创建表头
    header = ["class_label"] + [f"size{i}" for i in range(max_columns - 1)]

    # 填充数据为0
    filled_data = [row + [0] * (max_columns - len(row)) for row in data]

    # 创建最终的 DataFrame
    final_df = pd.DataFrame(filled_data, columns=header)

    df_label = final_df.iloc[:, [0]]
    df_data = final_df.iloc[:, 1:]

    # 逐行读取
    for index, row in df_label.iterrows():
        df_label.iloc[index,0] = str(row.iloc[0]).split('_')[0]
    # 保存到输出文件
    # output_dir = os.path.dirname(output_file_data)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    df_label.to_csv(output_file_label, index=False, header=True)
    df_data.to_csv(output_file_data, index=False, header=True)
    print("Done")


# 获取当前脚本所在的目录
cur_directory = os.path.dirname(os.path.realpath(__file__))
# 定义输入目录和输出目录
input_dir = os.path.join(cur_directory, "7-covers_new/")
output_file_data = os.path.join(cur_directory, "9-train_data_new/train-data.csv")
output_file_label = os.path.join(cur_directory, "9-train_data_new/train-label.csv")
# 调用函数
merge_csv_to_train_data(input_dir, output_file_data, output_file_label)