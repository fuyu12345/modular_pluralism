
import os
import json
import pandas as pd

# 路径设置
msg_dir = "/home/zczlyf7/Overton-PPO/modular_pluralism/community_lm_msgs/10p"
input_json_path = "/home/zczlyf7/Overton-PPO/modular_pluralism/input/overton_test_10p.json"
output_csv_path = "/home/zczlyf7/Overton-PPO/modular_pluralism/merge_output/merged_output_10p.csv"

# 读取原始输入数据
with open(input_json_path, "r") as f:
    input_data = json.load(f)
id_to_prompt = {str(item["id"]): item["input"] for item in input_data}
all_ids = sorted(id_to_prompt.keys(), key=int)

# 加载所有模型输出
all_model_outputs = []
for fname in sorted(os.listdir(msg_dir)):
    if fname.endswith(".json"):
        with open(os.path.join(msg_dir, fname), "r") as f:
            all_model_outputs.append(json.load(f))

# 构造合并数据
merged_rows = []
for id_str in all_ids:
    answers = []
    for model_output in all_model_outputs:
        msg = model_output.get(id_str, "").strip()
        if msg:
            answers.append(msg)
    merged_rows.append({
        "id": id_str,
        "prompt": id_to_prompt[id_str],
        "answers": "\n\n".join(answers)
    })

# 保存为 CSV
df = pd.DataFrame(merged_rows)
df.to_csv(output_csv_path, index=False)
print(f"✅ CSV saved to: {output_csv_path}")
