"""
LLM Router (R3) — Zero-shot classification via Claude API

設計原則：
- 每個 query 送一次 Claude Haiku API call
- System prompt 描述 7 個 agent + oos 的職責
- 模型回傳 agent 名稱（structured output）
- 這是「品質天花板」，也是「成本天花板」

研究意義：
- 定義路由準確率的上限
- Hybrid Router 的目標：用最少 LLM call 逼近 R3 的準確率
- LLM call rate × R3 cost = Hybrid 的 LLM 成本

成本控制：
- 用 claude-haiku-4-5-20251001（最便宜的 Claude 模型）
- max_tokens=20（只需要 agent 名稱）
- temperature=0（可重現）
"""

import os
import time
import json
from typing import Optional

# ─────────────────────────────────────────────
# API 設定
# ─────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-haiku-4-5-20251001"

AGENT_DESCRIPTIONS = {
    "finance_agent": "Banking, credit cards, accounts, bills, taxes, insurance, rewards, credit score, transfers, payments",
    "travel_agent": "Flights, hotels, reservations, visas, luggage, travel alerts, vaccines, plug types",
    "auto_agent": "Vehicle maintenance, oil change, tires, gas, car rental, uber, directions, traffic, distance",
    "kitchen_agent": "Recipes, nutrition, calories, ingredients, meal suggestions, restaurants, shopping lists, food storage",
    "productivity_agent": "Calendar, reminders, alarms, timers, todo lists, PTO, meetings, weather, translate, spelling, calculator, time, orders",
    "device_agent": "Smart home, music playback, playlists, volume, device settings, sync, language/accent changes, user name",
    "meta_agent": "Greetings, goodbye, jokes, fun facts, bot identity questions, yes/no/maybe, coin flip, dice roll, hobbies, cancel, repeat",
    "oos": "Out-of-scope: queries that don't fit any of the above categories",
}

SYSTEM_PROMPT = """You are a query router for a multi-agent system. Given a user query, classify it into exactly ONE of these agent categories:

""" + "\n".join(f"- {agent}: {desc}" for agent, desc in AGENT_DESCRIPTIONS.items()) + """

Rules:
1. Reply with ONLY the agent name (e.g., "finance_agent"), nothing else.
2. If the query is ambiguous, nonsensical, or doesn't clearly fit any agent, reply "oos".
3. Do not explain your reasoning."""


# ─────────────────────────────────────────────
# 統計
# ─────────────────────────────────────────────

_stats = {
    "total_calls": 0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_latency_ms": 0,
}

VALID_AGENTS = set(AGENT_DESCRIPTIONS.keys())


def get_stats() -> dict:
    """回傳累計的 API 呼叫統計"""
    return _stats.copy()


def reset_stats():
    """重置統計"""
    global _stats
    _stats = {
        "total_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_latency_ms": 0,
    }


# ─────────────────────────────────────────────
# Router 實作
# ─────────────────────────────────────────────

def route(query: str) -> dict:
    """
    用 Claude Haiku zero-shot classification 路由 query。

    Returns:
        {
            "agent": str,
            "confidence": float,    # 1.0（LLM 不回傳 confidence，固定設 1.0）
            "scores": {},           # 空（LLM 只回傳一個答案）
            "method": "llm",
            "input_tokens": int,
            "output_tokens": int,
            "latency_ms": int,
        }
    """
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    t0 = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=20,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    latency_ms = round((time.time() - t0) * 1000)

    # 解析回傳
    raw_text = response.content[0].text.strip().lower()

    # 容錯：模型可能回傳帶引號或多餘文字
    agent = "oos"
    for valid in VALID_AGENTS:
        if valid in raw_text:
            agent = valid
            break

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # 更新統計
    _stats["total_calls"] += 1
    _stats["total_input_tokens"] += input_tokens
    _stats["total_output_tokens"] += output_tokens
    _stats["total_latency_ms"] += latency_ms

    return {
        "agent": agent,
        "confidence": 1.0,
        "scores": {},
        "method": "llm",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
    }


# ─────────────────────────────────────────────
# 直接執行時做簡單測試
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        exit(1)

    test_queries = [
        "what's my credit score",
        "book a flight to tokyo",
        "when should i get my oil changed",
        "how many calories in an avocado",
        "set a reminder for 3pm",
        "play some jazz music",
        "tell me a joke",
    ]

    print(f"Using model: {MODEL}")
    for q in test_queries:
        r = route(q)
        print(f"  [{r['agent']:20s}] ({r['latency_ms']}ms, {r['input_tokens']}+{r['output_tokens']} tok)  {q}")

    stats = get_stats()
    print(f"\nTotal: {stats['total_calls']} calls, {stats['total_input_tokens']} in + {stats['total_output_tokens']} out tokens")
