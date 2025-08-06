import re
import pandas as pd

# === CONFIG ===
log_path = "/home/zczlyf7/Overton-PPO/modular_pluralism/fuyu/new-reward-model-logs/evalB_chat_hf_grpo-qwen2.5-3b-im-200max_token-200-test_5p.csv_template_maxwords200_0722_144950.log"  # <-- Replace with your actual log path
output_path = "parsed_with_periods_4.csv"

# === READ FILE ===
with open(log_path, "r", encoding="utf-8") as f:
    text = f.read()

# === SPLIT BY ROWS ===
rows = re.split(r"=== Row \d+ ===", text)

# === PROCESS EACH ROW ===
data = []
for row in rows[1:]:  # skip first part before Row 0
    # Extract prompt
    prompt_match = re.search(
          r"Provide a (?:comprehensive|multi[\u200b‑-]?perspective) analysis.*?situation:\s*(.*?)(?:\s*–?\s*max|\(max)",
        row, re.DOTALL | re.IGNORECASE
    )
    #  prompt_match = re.search(
    #     r"Provide a multi[\u200b‑-]perspective analysis to this situation:(.*?)Make sure",
    #     row, re.DOTALL | re.IGNORECASE
    # )
    if not prompt_match:
        continue
    prompt = prompt_match.group(1).strip().replace("\n", " ")

    # Extract Cand(x): block
    cand_block_match = re.search(r"Cand\(\d+\):(.*?)(?:Matched|✔|✗|===|-$|\Z)", row, re.DOTALL)
    if not cand_block_match:
        continue
    cand_block = cand_block_match.group(1)

    # Extract [C#] lines
    cand_lines = re.findall(r"\[C\d+\](.*)", cand_block)
    processed_lines = []
    for line in cand_lines:
        clean = line.strip()
        if not clean.endswith('.'):
            clean += '.'
        processed_lines.append(clean)

    # Join cleaned answers
    answers = "\n".join(processed_lines)

    data.append({
        "prompt": prompt,
        "answers": answers
    })

# === SAVE RESULT ===
df = pd.DataFrame(data)
df.to_csv(output_path, index=False)
print(f"✅ 提取并修复完成，共 {len(df)} 条记录，已保存至：{output_path}")
