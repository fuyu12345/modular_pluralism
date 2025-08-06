import pandas as pd, json, re, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==== 配置 ====
CSV_PATH   = "/home/zczlyf7/Overton-PPO/benchmark2/dataset/test_10p.csv"
OUT_JSON   = "/home/zczlyf7/Overton-PPO/modular_pluralism/output/overton_test_valuekaleidoscope_from_ourmodel_10p.json"
MODEL_PATH = "/scratch/zczlyf7/HF_models/hf_grpo-qwen2.5-3b-op-ex"
BATCH_SIZE = 20
MAX_WORDS  = 380
MAX_NEW    = 420
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 加载模型 ====
print("Loading model ...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
).eval()

# ==== 加载数据 ====
df = pd.read_csv(CSV_PATH)

def make_prompt(text):
    return (
        f"Provide a structured multi-perspectives analysis: {text}\n"
        f"(withn {MAX_WORDS} words)\n"
        "Make sure each sentence you write can represent one perspective from your analysis.\n"
        "Output Format Example:\n"
        "In the perspective of <Perspective name>, <your explanation to this aspect>\n"
    )

prompts = [make_prompt(x) for x in df["prompt"]]

# ==== 批量生成（不再用 chat_template） ====
print("Generating ...")
outputs = []
for idx in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch_prompts = prompts[idx:idx+BATCH_SIZE]

    # <<< 直接编码纯文本
    enc = tok(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)
    prompt_len = enc.input_ids.shape[1]           # 每句已 pad 为同长

    with torch.no_grad():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )

    for row in gen_ids:
        reply = tok.decode(row[prompt_len:], skip_special_tokens=True).strip()
        outputs.append(reply)

# ==== 解析原 CSV 的 answer 字段 ====
def parse_answer(ans: str):
    pattern = r"In the perspective of (.*?), (.*?)(?=\nIn the perspective of|\Z)"
    pairs   = re.findall(pattern, ans, flags=re.DOTALL)
    vrd  = [p[0].strip() for p in pairs]
    expl = [p[1].strip().replace("\n", " ") for p in pairs]
    return vrd, expl

json_list = []
for i, row in df.iterrows():
    vrd, expl = parse_answer(row["answer"])
    json_list.append({
        "id"         : int(i),
        "situation"  : row["prompt"],
        "input"      : prompts[i],
        "output"     : outputs[i],
        "vrd"        : vrd,
        "explanation": expl
    })

with open(OUT_JSON, "w") as f:
    json.dump(json_list, f, indent=2, ensure_ascii=False)

print(f"Saved {len(json_list)} items to {OUT_JSON}")
