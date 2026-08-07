"""
关键词提取服务 (FastAPI 封装层)
=================================
把微调后的 Qwen2.5-7B-Instruct-SFT 模型（通过 vLLM 提供的 OpenAI 兼容 API）
封装成一个稳定、可工程化的 HTTP 服务。

设计要点：
- 与推理引擎解耦：本服务只负责「请求编排 + 输出解析 + 接口规范」，
  真正的模型推理由 vLLM 在 8000 端口提供，两者通过 OpenAI 兼容协议通信。
- 输出归一化：模型原生返回可能带编号、前缀、中英文分号混用，
  parse_keywords() 统一清洗成干净的关键词列表，下游无需再处理脏数据。
- 可观测：记录每次请求的耗时，方便联调与压测时定位瓶颈。

部署（云端 AutoDL 实例）：
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8080
前端访问 http://<实例IP>:8080/ ，API 路径为 POST /api/extract
"""

import os
import re
import time

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="关键词提取服务 (SFT-Qwen2.5-7B)")

# ---- 配置（通过环境变量注入，避免硬编码路径泄露部署细节）----
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy")  # vLLM 默认不校验，填占位即可
VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    "/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct-SFT",  # 未加 --served-model-name 时的注册名
)

client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

# 静态资源目录：基于脚本自身位置，避免依赖启动时的工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 系统提示词：明确约束输出格式，降低后处理成本
SYSTEM_PROMPT = (
    "你是关键词提取专家。请从用户提供的文本中提取最相关的关键词，"
    "多个关键词用英文分号 \";\" 分隔，直接输出关键词，不要解释、不要编号、不要前缀。"
)


class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    keywords: list[str]  # 清洗后的关键词列表
    raw: str              # 模型原始输出（便于排查格式问题）
    elapsed_ms: int       # 本次推理耗时（毫秒）


def parse_keywords(raw: str) -> list[str]:
    """
    把模型原始输出归一化成关键词列表。
    处理以下脏数据：
    - 中英文分号/逗号/顿号/换行混用
    - 行首编号（1. 2、3) 等）
    - 「关键词：」「提取结果：」等前缀
    """
    raw = raw.strip()
    # 去掉逐行编号
    raw = re.sub(r"^\s*\d+[\.、)]\s*", "", raw, flags=re.MULTILINE)
    # 去掉常见前缀
    raw = re.sub(r"^(关键词|关键字|提取结果|关键术语)[:：]\s*", "", raw)
    # 统一分隔符切分
    parts = re.split(r"[;；,，、\n]+", raw)
    keywords = [p.strip() for p in parts if p.strip()]
    return keywords


@app.post("/api/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"文本：{req.text}"},
            ],
            temperature=0.1,      # 关键词提取要求稳定，低温减少随机性
            max_tokens=128,
        )
    except Exception as e:
        # 不暴露内部错误细节，只返回可读信息
        raise HTTPException(status_code=502, detail=f"模型服务调用失败: {e}")

    raw = resp.choices[0].message.content or ""
    keywords = parse_keywords(raw)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return ExtractResponse(keywords=keywords, raw=raw, elapsed_ms=elapsed_ms)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# 挂载静态资源（前端样式/脚本）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
