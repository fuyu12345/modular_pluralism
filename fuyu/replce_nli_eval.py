#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
from pathlib import Path

# ===== 路径 =====
JSON_IN  = Path("/home/zczlyf7/Overton-PPO/modular_pluralism/output/overton_test_valuekaleidoscope_from_ourmodel_10p.json")
CSV_PATH = Path("/home/zczlyf7/Overton-PPO/modular_pluralism/merge_output/merged_output_10p_with_analysis.csv")
JSON_OUT = JSON_IN.parent / "overton_test_valuekaleidoscope_from_ourmodel_replaced_10p.json"

# ===== 1. 读取文件 =====
with JSON_IN.open("r") as f:
    json_data = json.load(f)

df = pd.read_csv(CSV_PATH)

# 确保 id 唯一，方便映射
csv_map = dict(zip(df["id"], df["answers"]))

# ===== 2. 替换 output =====
for item in json_data:
    i = item.get("id")
    if i in csv_map:
        item["output"] = str(csv_map[i])  # 转成 str 确保 JSON 可写

# ===== 3. 保存 =====
with JSON_OUT.open("w") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"Done!  新文件已写入: {JSON_OUT}")
