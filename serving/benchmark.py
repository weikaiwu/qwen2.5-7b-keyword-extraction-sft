"""
关键词提取服务 · 性能压测脚本
==============================
通过并发请求 /api/extract 接口，测量服务的吞吐（QPS）与延迟分布。
用于验证「微调模型 → vLLM → FastAPI」整条链路在工程化部署下的真实表现，
简历上可写「单卡 RTX 4090 支撑 X QPS，平均首结果延迟 XX ms」。

用法：
    python benchmark.py --url http://localhost:8080 --concurrency 16 --requests 200
参数：
    --url          服务地址（默认 http://localhost:8080）
    --concurrency  并发数（默认 16）
    --requests     总请求数（默认 200）
"""

import argparse
import asyncio
import random
import time

import httpx


# 多段不同主题文本轮换，避免 vLLM prefix caching 命中导致延迟偏乐观。
# 真实场景文本各不相同，用混合负载测出的延迟/QPS 更接近生产水位。
SAMPLE_TEXTS = [
    "深度学习在自然语言处理领域取得了突破性进展，特别是在机器翻译、文本生成和问答系统等任务中表现优异。",
    "知识管理是知识经济时代的核心能力，企业需要构建知识市场来提升知识共享与创新的效率。",
    "气候变化导致极端天气事件频发，全球各国正推动碳中和目标，加速向可再生能源结构转型。",
    "量子计算利用量子叠加与纠缠特性，有望在密码破解、药物分子模拟等领域带来算力革命。",
    "新能源汽车渗透率持续提升，动力电池的能量密度、快充技术与回收利用成为产业竞争焦点。",
    "元宇宙通过虚拟现实与数字孪生技术构建沉浸式空间，正在重塑社交、办公与娱乐的交互方式。",
    "半导体先进制程逼近物理极限，Chiplet 异构集成与先进封装成为延续摩尔定律的关键路径。",
    "多模态大模型融合文本、图像、音频的理解与生成能力，推动人机交互向更自然的方向演进。",
]


async def worker(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore, stats: list):
    async with sem:
        text = random.choice(SAMPLE_TEXTS)  # 随机取一段，模拟真实混合负载
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/api/extract", json={"text": text}, timeout=30)
            elapsed = (time.perf_counter() - t0) * 1000
            ok = r.status_code == 200
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            ok = False
        stats.append((elapsed, ok))


def pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


async def main(url: str, concurrency: int, requests: int):
    sem = asyncio.Semaphore(concurrency)
    stats: list = []
    async with httpx.AsyncClient() as client:
        t_start = time.perf_counter()
        tasks = [worker(client, url, sem, stats) for _ in range(requests)]
        await asyncio.gather(*tasks)
        total = time.perf_counter() - t_start

    lats = [s[0] for s in stats if s[1]]
    ok_cnt = sum(1 for s in stats if s[1])

    print("=" * 48)
    print(f"  服务地址      : {url}")
    print(f"  并发数        : {concurrency}")
    print(f"  总请求数      : {requests}")
    print(f"  成功数        : {ok_cnt} ({ok_cnt / requests * 100:.1f}%)")
    print(f"  总耗时        : {total:.2f}s")
    print(f"  吞吐 QPS      : {requests / total:.1f}")
    if lats:
        print(f"  平均延迟      : {sum(lats) / len(lats):.1f} ms")
        print(f"  P50 延迟      : {pct(lats, 0.50):.1f} ms")
        print(f"  P95 延迟      : {pct(lats, 0.95):.1f} ms")
        print(f"  最大延迟      : {max(lats):.1f} ms")
    print("=" * 48)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="关键词提取服务压测")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.concurrency, args.requests))
