import os
import shutil

def copy_avi_files(source_dir, target_dir):
    """
    遍历 source_dir 下的所有一级子文件夹中的所有二级子文件夹，
    复制所有 .avi 文件到 target_dir 中的同名一级子文件夹下。
    
    :param source_dir: 源文件夹路径，遍历该文件夹下的所有一级子文件夹
    :param target_dir: 目标文件夹路径，复制 .avi 文件到这个文件夹
    """
    print(f"Starting to search for .avi files in {source_dir}")
    
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory {target_dir}")
    
    # 遍历源文件夹的一级子文件夹
    for first_level_dir in os.listdir(source_dir):
        first_level_path = os.path.join(source_dir, first_level_dir)
        
        # 确保是文件夹
        if os.path.isdir(first_level_path):
            # print(f"Searching in first-level directory {first_level_dir}")
            
            # 遍历一级子文件夹中的所有二级子文件夹
            for root, dirs, files in os.walk(first_level_path):
                # print(f"Searching in directory {root}")
                for file in files:
                    # print(f"Found file {file}")
                    # 检查文件扩展名是否为 .avi
                    if file.endswith(".avi"):
                        # 构建完整的文件路径
                        file_path = os.path.join(root, file)
                        # 构建目标文件路径
                        target_file_path = os.path.join(target_dir, first_level_dir, file)
                        
                        # 创建目标文件夹结构
                        target_folder_path = os.path.dirname(target_file_path)
                        if not os.path.exists(target_folder_path):
                            os.makedirs(target_folder_path)
                            # print(f"Created target folder {target_folder_path}")
                        
                        # 复制文件
                        try:
                            shutil.copy(file_path, target_file_path)
                            print(f"Copied '{file_path}' to '{target_file_path}'")
                        except Exception as e:
                            print(f"Failed to copy '{file_path}' to '{target_file_path}': {e}")


# 获取当前脚本所在的目录
cur_directory = os.path.dirname(os.path.realpath(__file__))
# 定义输入目录和输出目录
source_directory = os.path.join(cur_directory, "ucf_sports_actions/ucf action/")
target_directory = os.path.join(cur_directory, "8-data_set/")

copy_avi_files(source_directory, target_directory)