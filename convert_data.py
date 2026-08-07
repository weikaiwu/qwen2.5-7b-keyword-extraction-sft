import json
import random

random.seed(42)

INPUT = "data/keywords_data_train.jsonl"
TRAIN_OUT = "data/keywords_sft_train.jsonl"
VAL_OUT = "data/keywords_sft_val.jsonl"

TRAIN_SIZE = 10000
VAL_SIZE = 500

samples = []
with open(INPUT, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        for conv in d["conversation"]:
            samples.append({
                "messages": [
                    {"role": "user", "content": conv["human"].strip()},
                    {"role": "assistant", "content": conv["assistant"].strip()},
                ]
            })

print(f"总样本数: {len(samples)}")

random.shuffle(samples)
val_samples = samples[:VAL_SIZE]
train_samples = samples[VAL_SIZE:VAL_SIZE + TRAIN_SIZE]

with open(TRAIN_OUT, "w", encoding="utf-8") as f:
    for s in train_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

with open(VAL_OUT, "w", encoding="utf-8") as f:
    for s in val_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"训练集: {len(train_samples)} 条 -> {TRAIN_OUT}")
print(f"验证集: {len(val_samples)} 条 -> {VAL_OUT}")
