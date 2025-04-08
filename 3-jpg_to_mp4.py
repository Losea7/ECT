import cv2
import os

def jpg_to_mp4(input_base_dir, output_dir, fps=10):
    """
    将指定目录下的所有 jpg 文件拼接成 mp4 视频文件。
    :param input_base_dir: 输入目录的根路径
    :param output_dir: 输出视频文件的保存目录
    :param fps: 输出视频的帧率
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历一级子文件夹
    for action_folder in os.listdir(input_base_dir):
        action_path = os.path.join(input_base_dir, action_folder)
        if not os.path.isdir(action_path):
            continue

        # 遍历二级子文件夹
        for video_folder in os.listdir(action_path):
            video_path = os.path.join(action_path, video_folder)
            if not os.path.isdir(video_path):
                continue

            # 获取所有 jpg 文件
            jpg_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
            jpg_files.sort()  # 按文件名排序，确保顺序正确

            if not jpg_files:
                print(f"No jpg files found in {video_path}")
                continue

            # 读取第一张图片以获取视频的宽度和高度
            first_image_path = os.path.join(video_path, jpg_files[0])
            first_image = cv2.imread(first_image_path)
            height, width, _ = first_image.shape

            # 创建 VideoWriter 对象
            output_file_name = f"{action_folder}_{video_folder}.mp4"
            output_file_path = os.path.join(output_dir, output_file_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

            # 将所有 jpg 文件写入视频
            for jpg_file in jpg_files:
                image_path = os.path.join(video_path, jpg_file)
                image = cv2.imread(image_path)
                video_writer.write(image)

            # 释放 VideoWriter
            video_writer.release()
            print(f"Video saved to {output_file_path}")
    print('Finish!')


# 获取当前脚本所在的目录
cur_directory = os.path.dirname(os.path.realpath(__file__))
# 定义输入目录和输出目录
input_base_dir = os.path.join(cur_directory, "ucf_sports_actions/ucf action/")
output_dir = os.path.join(cur_directory, "6-input_videos_new/")
# 调用函数
jpg_to_mp4(input_base_dir, output_dir)