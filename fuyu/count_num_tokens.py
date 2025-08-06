import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# Load CSV
csv_path = "/home/zczlyf7/Overton-PPO/modular_pluralism/merge_output/merged_output_10p_with_analysis.csv"
df = pd.read_csv(csv_path)

# Load tokenizer from local path
tokenizer_path = "/scratch/zczlyf7/HF_models/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# Initialize counters
total_words = 0
total_tokens = 0
num_rows = 0

# Loop through answers
for answer in tqdm(df["answers"].dropna(), desc="Processing answers"):
    answer_str = str(answer).strip()
    words = answer_str.split()
    tokens = tokenizer.encode(answer_str, add_special_tokens=False)
    
    total_words += len(words)
    total_tokens += len(tokens)
    num_rows += 1

# Compute averages
avg_words = total_words / num_rows if num_rows > 0 else 0
avg_tokens = total_tokens / num_rows if num_rows > 0 else 0

print(f"Processed {num_rows} answers.")
print(f"Average number of words: {avg_words:.2f}")
print(f"Average number of tokens: {avg_tokens:.2f}")
