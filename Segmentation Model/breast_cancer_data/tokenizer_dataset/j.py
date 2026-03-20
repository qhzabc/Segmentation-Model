import os
import json


def generate_simple_file_json(folder_path, output_json="files.json"):

    file_dict = {}

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 只处理文件，跳过子文件夹
        if os.path.isfile(file_path):
            # 使用空字符串作为初始描述
            file_dict[filename] = "This image is of a malignant tumor. The tumor area can be round, oval, lobular, or irregular. The margins can be smooth, irregular, or spiculate. The spiculate margins are often characteristic of malignant breast lesions and radial scars. Irregularly shaped masses with irregular margins and enhanced internal septa are also signs of malignancy."

    # 写入JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(file_dict, f, indent=4, ensure_ascii=False)

    print(f"成功生成JSON文件: {output_json}")
    print(f"共添加了 {len(file_dict)} 个文件")


if __name__ == "__main__":
    # 设置命令行参数

    output = 'files.json'
    generate_simple_file_json(r'E:\project_useful\breast_cancer_data\tokenizer_dataset\all_images', output)