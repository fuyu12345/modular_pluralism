#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
refine_with_llm_chat.py
~~~~~~~~~~~~~~~~~~~~~~~
Use Qwen3‑8B in chat mode to refine candidate passages into
multi‑perspective analyses and save the outputs.
"""

import os, sys, pandas as pd, torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────── configuration ───────────────────────────────────
INPUT_CSV   = "/home/zczlyf7/Overton-PPO/modular_pluralism/merge_output/merged_output_10p.csv"
MODEL_PATH  = "/scratch/zczlyf7/HF_models/Qwen3-8B"
OUTPUT_CSV  = os.path.join(
    os.path.dirname(INPUT_CSV), "merged_output_10p_with_analysis.csv")

BATCH_SIZE      = 40
MAX_NEW_TOKENS  = 500
TEMPERATURE     = 0.1
TOP_P           = 0.9
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

LOGDIR          = "llm_refine_logs"
SYSTEM_MSG      = (
    "You are a helpful assistant. Provide a concise multi‑perspective analysis. "
    "Each sentence must reflect a distinct perspective or value."
)
USER_TEMPLATE   = (
    "Please comment and give a multi‑perspective analysis on the given situation "
    "with the help of the following passages. Make sure to reflect diverse values "
    "and perspectives, and ensure each sentence conveys a different perspective.\n\n"
    "Situation: {prompt}\n\n"
    "Helper passages:\n{answers}\n\n"
    "Now give your analysis:"
)
# ──────────────────────────────────────────────────────────────────────────────

# ───────────── logging setup (everything goes to log file) ────────────────────
os.makedirs(LOGDIR, exist_ok=True)
log_path = os.path.join(LOGDIR, f"llm_refine_{datetime.now():%m%d_%H%M%S}.log")
sys.stdout = sys.stderr = open(log_path, "w")

print("=== Parameters ===")
for k, v in {
    "INPUT_CSV": INPUT_CSV,
    "MODEL_PATH": MODEL_PATH,
    "OUTPUT_CSV": OUTPUT_CSV,
    "BATCH_SIZE": BATCH_SIZE,
    "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
    "TEMPERATURE": TEMPERATURE,
    "TOP_P": TOP_P,
    "DEVICE": DEVICE
}.items():
    print(f"{k:>20s} : {v}")
print()
# ───────────── load model ─────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
    device_map="auto"
).eval()

# ───────────── load data ──────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
assert {"prompt", "answers"}.issubset(df.columns), \
    "CSV must contain columns: prompt, answers"

print(f"Loaded {len(df)} rows.\n", flush=True)
analyses = []  # store generated outputs

# ───────────── batch generation ───────────────────────────────────────────────
for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start + BATCH_SIZE]

    # Build chat‑formatted prompts
    chat_texts = []
    for _, row in sub.iterrows():
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_TEMPLATE.format(
                prompt=row["prompt"], answers=row["answers"])}
        ]
        chat_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        chat_texts.append(chat_text)

    enc = tok(chat_texts, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tok.eos_token_id,
        )

    decoded = tok.batch_decode(gen_ids, skip_special_tokens=True)

    # Trim off the template portion so only assistant content remains
    for template, full in zip(chat_texts, decoded):
        analyses.append(full[len(template):].lstrip())

    print(f"Processed rows {start}–{start + len(sub) - 1}", flush=True)

# ───────────── save results ───────────────────────────────────────────────────
result_df = df.copy()
result_df["llm_analysis"] = analyses
result_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(result_df)} analyses to {OUTPUT_CSV}")
