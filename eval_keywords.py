import json
import argparse
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def extract_keywords(text):
    """
    从模型输出或参考答案中提取关键词。
    支持格式：
      1. 关键词1
      2. 关键词2
    或 - 关键词1
      - 关键词2
    或 关键词1;关键词2;关键词3
    或 关键词1,关键词2,关键词3
    """
    # 去掉常见的前缀说明
    text = re.sub(r'^(根据.*?(关键词|关键信息)[：:]?\s*)+', '', text, flags=re.S)
    text = re.sub(r'^(以下是|下面是|这些|关键).*?[：:]?\s*', '', text, flags=re.S)
    # 去掉常见的后缀总结
    text = re.split(r'(\n\s*)?这些关键词|(\n\s*)?以上|(\n\s*)?总结', text)[0]
    
    keywords = []
    
    # 先尝试按行分割（处理编号列表）
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 去掉编号 1. / 1、 / - / •
        line = re.sub(r'^[\d一二三四五六七八九十零]+\s*[.．、,，:：)\-]\s*', '', line)
        line = re.sub(r'^[-•·]\s*', '', line)
        line = line.strip()
        # 去掉末尾标点
        line = line.rstrip('。，,；;!！?？')
        if line and len(line) > 0:
            # 如果这一行还包含分号/逗号/顿号，继续拆分
            if ';' in line or '；' in line or '，' in line or ',' in line or '、' in line:
                parts = re.split(r'[;；,，、]', line)
                for part in parts:
                    part = part.strip().rstrip('。，,；;!！?？')
                    if part:
                        keywords.append(part)
            else:
                keywords.append(line)
    
    # 如果没提取到（自由文本），尝试按中文顿号/逗号/分号分
    if not keywords:
        parts = re.split(r'[;；,，、]', text)
        keywords = [p.strip().rstrip('。，,；;!！?？') for p in parts if p.strip()]
    
    return list(set(keywords))  # 去重


def compute_prf1(pred_kw, ref_kw):
    pred_set = set(pred_kw)
    ref_set = set(ref_kw)
    
    if not pred_set and not ref_set:
        return 1.0, 1.0, 1.0
    
    if not pred_set:
        return 0.0, 0.0, 0.0
    
    if not ref_set:
        return 0.0, 0.0, 0.0
    
    tp = len(pred_set & ref_set)
    p = tp / len(pred_set)
    r = tp / len(ref_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--data", default="data/keywords_sft_val.jsonl", help="验证集")
    parser.add_argument("--max_samples", type=int, default=100, help="评估样本数")
    parser.add_argument("--output", required=True, help="输出结果 JSON")
    args = parser.parse_args()

    samples = []
    with open(args.data, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            d = json.loads(line)
            messages = d["messages"]
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            samples.append({"prompt": user_msg, "ref_raw": assistant_msg})

    print(f"加载模型: {args.model}")
    print(f"评估样本数: {len(samples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    formatted_prompts = []
    for s in samples:
        messages = [{"role": "user", "content": s["prompt"]}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(prompt)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=2048,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=512)

    outputs = llm.generate(formatted_prompts, sampling_params)

    results = []
    for s, out in zip(samples, outputs):
        pred_raw = out.outputs[0].text.strip()
        ref_kw = extract_keywords(s["ref_raw"])
        pred_kw = extract_keywords(pred_raw)
        p, r, f1 = compute_prf1(pred_kw, ref_kw)

        results.append({
            "prompt": s["prompt"][:200],
            "ref_raw": s["ref_raw"],
            "pred_raw": pred_raw,
            "ref_kw": ref_kw,
            "pred_kw": pred_kw,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        })

    avg_p = round(sum(r["precision"] for r in results) / len(results), 4)
    avg_r = round(sum(r["recall"] for r in results) / len(results), 4)
    avg_f1 = round(sum(r["f1"] for r in results) / len(results), 4)

    metrics = {
        "model": args.model,
        "n": len(results),
        "precision": avg_p,
        "recall": avg_r,
        "f1": avg_f1,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "samples": results}, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
