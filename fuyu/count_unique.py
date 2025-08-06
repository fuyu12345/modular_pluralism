import pandas as pd
import torch
import re
from sentence_transformers import SentenceTransformer, util
from itertools import combinations
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import stopwords
import nltk

# ──────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────
nltk.download('stopwords')  # 仅用于关键词过滤
CSV_PATH = "/home/zczlyf7/Overton-PPO/modular_pluralism/fuyu/parsed_with_periods_4.csv"
df = pd.read_csv(CSV_PATH)

SIM_THRESHOLD = 0.75
EMBED_MODEL = "/scratch/zczlyf7/st_models/MultipleNegativesRankingLoss/hpo_scale_40/checkpoint-1186"
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(EMBED_MODEL, device=device)

# ──────────────────────────────────────────────────────────────
# Load Qwen3
# ──────────────────────────────────────────────────────────────
LLM_PATH = "/scratch/zczlyf7/HF_models/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
llm_model.eval()

# ──────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────
def extract_sentences(text: str):
    # 基于句号分割
    sents = [s.strip() for s in str(text).split('.') if s.strip()]
    return sents

def extract_keywords(text: str):
    stop_words = set(stopwords.words('english'))
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return set(word for word in tokens if word.isalnum() and word not in stop_words)

def replace_keywords_with_placeholder(text: str, keywords: set, placeholder: str = "[KW]") -> str:
    tokens = re.findall(r"\b\w+\b", text)
    return " ".join([placeholder if word.lower() in keywords else word for word in tokens])

def query_llm_batch_with_voting(pairs, repeat=3):
    prompts = []
    for s1, s2 in pairs:
        for _ in range(repeat):
            prompt = f"""Determine if the following sentence pair expresses the same meaning or perspective. Respond only with 'Yes' or 'No'.\n\nSentence A: {s1}\nSentence B: {s2}\nAnswer:"""
            prompts.append(prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = llm_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=10,
            do_sample=False
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results = []
    for i in range(0, len(decoded), repeat):
        votes = []
        for j in range(repeat):
            response = decoded[i + j]
            answer = response.split("Answer:")[-1].strip().split("\n")[0]
            votes.append("Yes" if "yes" in answer.lower() else "No")
        yes_count = votes.count("Yes")
        final_decision = "Yes" if yes_count >= 2 else "No"
        results.append(final_decision)

    return results

# ──────────────────────────────────────────────────────────────
# MAIN LOOP WITH BATCHING AND PRINT
# ──────────────────────────────────────────────────────────────
batch_size = 8
redundant_ratios = []
idx = 0

while idx < len(df):
    batch_df = df.iloc[idx:idx + batch_size]
    batch_pairs = []
    batch_metadata = []

    batch_sentences_list = []
    batch_keywords_list = []

    for local_i, (row_idx, row) in enumerate(batch_df.iterrows()):
        global_idx = idx + local_i
        prompt = row["prompt"]
        answer = row["answers"]
        keywords = extract_keywords(prompt)
        sentences = extract_sentences(answer)

        print(f"\n{'='*60}")
        print(f"[{global_idx+1}] Prompt: {prompt.strip()}")
        print(f"Answer sentence split (based on '.'): Total {len(sentences)} sentences:")
        for i, s in enumerate(sentences):
            print(f"  [{i+1}] {s}")

        masked_sents = [replace_keywords_with_placeholder(sent, keywords) for sent in sentences]

        batch_sentences_list.append(sentences)
        batch_keywords_list.append(keywords)

        if len(sentences) <= 1:
            redundant_ratios.append(1.0)
            continue

        embs = embed_model.encode(masked_sents, convert_to_tensor=True, normalize_embeddings=True)

        local_pairs = []
        local_indices = []

        for (i, j) in combinations(range(len(sentences)), 2):
            sim = float(util.cos_sim(embs[i], embs[j]))
            if sim >= SIM_THRESHOLD:
                local_pairs.append((sentences[i], sentences[j]))
                local_indices.append((i, j))
                if len(local_pairs) == 10:
                    break

        if not local_pairs:
            redundant_ratios.append(1.0)
            continue

        batch_pairs.extend(local_pairs)
        batch_metadata.extend([(len(redundant_ratios), i, j) for (i, j) in local_indices])
        redundant_ratios.append(None)

    if batch_pairs:
        llm_results = query_llm_batch_with_voting(batch_pairs, repeat=3)

        sample_to_redundant_idx = dict()
        for (sample_idx, i, j), result in zip(batch_metadata, llm_results):
            if result == "Yes":
                sample_to_redundant_idx.setdefault(sample_idx, set()).add(j)

        for sample_idx in set(x[0] for x in batch_metadata):
            sentences = batch_sentences_list[sample_idx - idx]
            redundant_idx = sample_to_redundant_idx.get(sample_idx, set())
            num_original = len(sentences)
            num_filtered = len([s for i, s in enumerate(sentences) if i not in redundant_idx])
            ratio = 1.0 if len(redundant_idx) == 0 else round(num_filtered / num_original, 3)
            redundant_ratios[sample_idx] = ratio

            print(f"→ Kept {num_filtered}/{num_original} unique sentences. Retained Ratio: {ratio:.3f}")

    idx += batch_size

# ──────────────────────────────────────────────────────────────
# RETURN AVERAGE UNIQUENESS SCORE
# ──────────────────────────────────────────────────────────────
average_uniqueness = sum(redundant_ratios) / len(redundant_ratios)
print(f"\n✅ 平均唯一性评分 (perspective_ratio_retained): {average_uniqueness:.3f}")
