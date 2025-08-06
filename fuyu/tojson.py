import pandas as pd
import json
import os

# 读取 CSV 文件
csv_path = "/home/zczlyf7/Overton-PPO/benchmark/dataset/test_10p.csv"
df = pd.read_csv(csv_path)

# 检查是否有 prompt 列
if 'prompt' not in df.columns:
    raise ValueError("CSV 文件中找不到 'prompt' 列")

# 构建 JSON 数据
json_data = [
    {"id": idx, "input": row["prompt"]}
    for idx, row in df.iterrows()
]

# 输出目录
output_dir = "/home/zczlyf7/Overton-PPO/modular_pluralism/input"
os.makedirs(output_dir, exist_ok=True)

# 保存为 JSON 文件
json_path = os.path.join(output_dir, "test_5p.json")
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"已保存 JSON 文件到: {json_path}")
