import os
import subprocess
import shutil

def convert_avi_to_mp4(source_dir, target_dir, file_name):
    """
    使用 ffmpeg 将单个 AVI 文件转换为 MP4 格式。
    
    :param source_dir: 包含 AVI 文件的源目录
    :param target_dir: MP4 文件将要保存的目标目录
    :param file_name: AVI 文件的名称
    """
    # 构造 AVI 文件和 MP4 文件的完整路径
    avi_file_path = os.path.join(source_dir, file_name)
    mp4_file_path = os.path.join(target_dir, file_name.replace('.avi', '.mp4'))
    
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 构造 ffmpeg 命令
    command = [
        'ffmpeg',
        '-i', avi_file_path,  # Input file
        '-c:v', 'libx264',    # Video codec
        '-c:a', 'aac',        # Audio codec
        '-strict', 'experimental',
        mp4_file_path          # Output file
    ]
    
    # 执行 ffmpeg 命令
    try:
        subprocess.run(command, check=True)
        print(f"Converted '{avi_file_path}' to '{mp4_file_path}'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert '{avi_file_path}': {e}")

def main():
    # 获取当前脚本所在的目录
    cur_directory = os.path.dirname(os.path.realpath(__file__))
    # 定义输入目录和输出目录
    input_dir = os.path.join(cur_directory, "8-data_set/")
    output_dir = os.path.join(cur_directory, "6-input_videos_new/")
    
    # 遍历输入目录中的所有目录和文件
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".avi"):
                # 构造源目录和目标目录
                source_subdir = root
                target_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))
                
                # 转换 AVI 文件为 MP4
                convert_avi_to_mp4(source_subdir, target_subdir, file)

    print("Conversion complete!")

if __name__ == "__main__":
    main()