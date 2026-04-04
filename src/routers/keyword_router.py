"""
Keyword Router (R1) — Rule-based keyword/regex matching baseline

設計原則：
- 每個 agent 有一組關鍵字 pattern
- 查詢文字 lowercase 後做 substring match
- 有多個 agent match 時，取 match 最多關鍵字的（簡單 scoring）
- 無 match → fallback 到 "oos"
- 這是最便宜（零 LLM call）也最脆的 baseline

研究意義：
- 定義 keyword routing 的「天花板」和「系統性失敗模式」
- Hybrid Router (R4) 會基於這些失敗模式設計 fallback 策略
"""

import re
from typing import Optional


# ─────────────────────────────────────────────
# 關鍵字規則表
# ─────────────────────────────────────────────
# 每個 agent 的 patterns: list of (pattern_str, weight)
# weight 允許某些高信號 pattern 有更大影響

AGENT_RULES: dict[str, list[tuple[str, float]]] = {

    "finance_agent": [
        # 帳戶 & 信用卡
        (r"\baccount\b", 1.0),
        (r"\bbank", 1.0),
        (r"\bcredit", 1.5),
        (r"\bdebit", 1.0),
        (r"\bcard\b", 1.0),
        (r"\bpin\b", 1.0),
        (r"\bfreeze", 1.0),
        (r"\bblock", 0.8),
        (r"\bfraud", 1.5),
        (r"\bstolen", 1.0),
        (r"\blost\s*(my\s*)?card", 1.5),
        (r"\breplace\w*\s*card", 1.2),
        (r"\bdamage", 0.8),
        (r"\bnew card", 1.5),
        # 帳單 & 付款
        (r"\bbill\b", 1.2),
        (r"\bpay\b", 0.8),
        (r"\bpayment", 1.0),
        (r"\bbalance\b", 1.2),
        (r"\bowe\b", 0.8),
        (r"\bdue\b", 0.8),
        (r"\bminimum\s*pay", 1.2),
        (r"\bapr\b", 1.5),
        (r"\binterest\s*rate", 1.5),
        (r"\btransaction", 1.2),
        (r"\bspend", 1.0),
        (r"\btransfer\b", 1.2),
        (r"\bdirect\s*deposit", 1.5),
        (r"\bcheck\s*(s|ing)?\b", 0.6),
        (r"\border\s*check", 1.2),
        # 信用分數
        (r"\bcredit\s*score", 2.0),
        (r"\bcredit\s*limit", 2.0),
        (r"\bimprove\s*credit", 1.5),
        # 稅務 & 收入
        (r"\btax", 1.5),
        (r"\bw[\s-]?2\b", 2.0),
        (r"\bincome\b", 1.2),
        (r"\bpayday", 1.5),
        (r"\b401\s*k\b", 2.0),
        (r"\brollover", 1.5),
        # 保險
        (r"\binsurance\b", 1.5),
        # 國際費用
        (r"\binternational\s*fee", 1.5),
        # 獎勵
        (r"\breward", 1.2),
        (r"\bredeem", 1.2),
        # 匯率
        (r"\bexchange\s*rate", 1.5),
        # routing number
        (r"\brouting\s*(number)?", 1.5),
        # expiration
        (r"\bexpir", 1.0),
    ],

    "travel_agent": [
        (r"\bflight\b", 1.5),
        (r"\bfly\b", 1.0),
        (r"\bbook\s*(a\s*)?(flight|hotel|room)", 2.0),
        (r"\bhotel\b", 1.5),
        (r"\breservation", 1.5),
        (r"\bcancel\s*(my\s*)?reservation", 2.0),
        (r"\bconfirm\s*(my\s*)?reservation", 2.0),
        (r"\bluggage\b", 1.5),
        (r"\bcarry[\s-]?on\b", 1.5),
        (r"\bvisa\b", 1.5),
        (r"\bpassport", 1.2),
        (r"\btravel\b", 1.5),
        (r"\bvaccinat", 1.2),
        (r"\bplug\s*type", 1.5),
        (r"\badapter", 1.0),
        (r"\bairport", 1.0),
        (r"\btrip\b", 1.0),
    ],

    "auto_agent": [
        (r"\boil\s*change", 2.0),
        (r"\btire", 1.5),
        (r"\bgas\b", 1.2),
        (r"\bfuel\b", 1.0),
        (r"\bgasoline", 1.2),
        (r"\bmpg\b", 2.0),
        (r"\bmile(s)?\s*per\s*gallon", 2.0),
        (r"\bmaintenance\b", 1.5),
        (r"\bjump\s*start", 2.0),
        (r"\bcar\b", 1.0),
        (r"\bvehicle", 1.0),
        (r"\brental\s*car", 1.5),
        (r"\bcar\s*rental", 1.5),
        (r"\buber\b", 1.5),
        (r"\blyft\b", 1.5),
        (r"\bdirect", 0.6),
        (r"\bdirection", 1.2),
        (r"\bnavigate", 1.0),
        (r"\btraffic\b", 1.5),
        (r"\bdistance\b", 1.0),
        (r"\bhow\s*far\b", 1.0),
        (r"\bschedule\s*(a\s*)?maintenance", 2.0),
    ],

    "kitchen_agent": [
        (r"\brecipe\b", 2.0),
        (r"\bcook\b", 1.5),
        (r"\bcooking", 1.5),
        (r"\bcalorie", 1.5),
        (r"\bnutrition", 1.5),
        (r"\bingredient", 1.5),
        (r"\bsubstitut", 1.2),
        (r"\bmeal\b", 1.2),
        (r"\bfood\b", 1.2),
        (r"\beat\b", 0.8),
        (r"\brestaurant\b", 1.2),
        (r"\bshopping\s*list", 2.0),
        (r"\bgrocery", 1.2),
        (r"\bwhat\s*should\s*i\s*(eat|have|cook|make)", 1.5),
        (r"\bhow\s*long\s*(to|does|should)\s*cook", 2.0),
        (r"\bfresh\b", 0.6),
        (r"\bexpire", 0.5),
        (r"\blast\b.*\b(food|fridge|refrigerator)", 1.5),
    ],

    "productivity_agent": [
        (r"\bcalendar\b", 1.5),
        (r"\bschedule\b", 1.0),
        (r"\bmeeting\b", 1.5),
        (r"\bremind", 1.5),
        (r"\balarm\b", 1.5),
        (r"\btimer\b", 1.5),
        (r"\btodo\b", 1.5),
        (r"\bto[\s-]?do\b", 1.5),
        (r"\bpto\b", 2.0),
        (r"\bvacation\s*day", 1.5),
        (r"\btime\s*off\b", 1.5),
        (r"\bday\s*off\b", 1.2),
        (r"\bholiday", 1.2),
        (r"\btranslate\b", 1.5),
        (r"\bspelling\b", 1.5),
        (r"\bspell\b", 1.0),
        (r"\bdefin(e|ition)", 1.5),
        (r"\bcalculat", 1.5),
        (r"\bconvert\b", 1.0),
        (r"\bconversion", 1.2),
        (r"\bweather\b", 1.5),
        (r"\btemperature\b", 0.8),
        (r"\bforecast", 1.0),
        (r"\btime\b", 0.5),
        (r"\bwhat\s*time\b", 1.5),
        (r"\btimezone\b", 1.5),
        (r"\bdate\b", 0.5),
        (r"\bwhat\s*date\b", 1.2),
        (r"\btoday\b", 0.5),
        (r"\border\s*status", 1.5),
        (r"\btrack\s*(my\s*)?order", 1.5),
        (r"\bwhere\s*(is\s*)?(my\s*)?order", 1.5),
        (r"\bcall\b", 1.0),
        (r"\btext\b", 0.8),
        (r"\bmessage\b", 0.8),
        (r"\bbusy\b", 1.0),
        (r"\bapplication\s*status", 1.5),
        (r"\bshare\s*(my\s*)?location", 1.5),
        (r"\bcurrent\s*location", 1.5),
        (r"\bfind\s*(my\s*)?phone", 1.5),
        (r"\bwhere\s*(is\s*)?(my\s*)?phone", 1.5),
    ],

    "device_agent": [
        (r"\bsmart\s*home", 2.0),
        (r"\bthermostat", 1.5),
        (r"\blight(s)?\b", 0.8),
        (r"\bplay\s*(a\s*)?(song|music)", 2.0),
        (r"\bnext\s*song", 2.0),
        (r"\bplaylist", 1.5),
        (r"\bmusic\b", 1.2),
        (r"\bvolume\b", 1.5),
        (r"\bmute\b", 1.0),
        (r"\blouder\b", 1.0),
        (r"\bquieter\b", 1.0),
        (r"\bspeed\b", 0.8),
        (r"\bchange\s*(the\s*)?(speed|volume|accent|language|name)", 1.5),
        (r"\breset\s*setting", 1.5),
        (r"\bdefault\s*setting", 1.2),
        (r"\bsync\b", 1.2),
        (r"\bbluetooth", 1.2),
        (r"\bpair\b", 0.8),
        (r"\bdevice\b", 0.8),
        (r"\bwhat\s*song\b", 2.0),
        (r"\bwhat('s| is)\s*playing", 1.5),
        (r"\buser\s*name\b", 1.2),
        (r"\bwhisper\b", 1.5),
    ],

    "meta_agent": [
        (r"\bhello\b", 1.5),
        (r"\bhi\b", 1.0),
        (r"\bhey\b", 1.0),
        (r"\bgoodbye\b", 1.5),
        (r"\bbye\b", 1.2),
        (r"\bthank", 1.5),
        (r"\byes\b", 0.5),
        (r"\bno\b", 0.3),
        (r"\bmaybe\b", 0.8),
        (r"\bjoke\b", 2.0),
        (r"\bfunny\b", 0.8),
        (r"\bfun\s*fact", 2.0),
        (r"\bflip\s*(a\s*)?coin", 2.0),
        (r"\broll\s*(a\s*)?(dice|die)", 2.0),
        (r"\bwhat\s*(is\s*)?your\s*name", 2.0),
        (r"\bwho\s*(are|r)\s*you\b", 1.5),
        (r"\bwho\s*made\s*you", 2.0),
        (r"\bwho\s*created\s*you", 2.0),
        (r"\bare\s*you\s*(a\s*)?(bot|robot|ai|human|real)", 2.0),
        (r"\bwhat\s*can\s*(you|i)\s*(do|ask)", 1.5),
        (r"\bhow\s*old\s*are\s*you", 2.0),
        (r"\bwhere\s*(are|r)\s*you\s*from", 2.0),
        (r"\bhobbies\b", 1.5),
        (r"\bpets?\b", 0.8),
        (r"\bdo\s*you\s*have\s*pet", 2.0),
        (r"\bmeaning\s*of\s*life", 2.0),
        (r"\brepeat\s*that\b", 1.5),
        (r"\bsay\s*(that|it)\s*again", 1.5),
        (r"\bcancel\b", 0.8),
        (r"\bnever\s*mind", 1.0),
    ],
}


# ─────────────────────────────────────────────
# Router 實作
# ─────────────────────────────────────────────

def _score_agent(query: str, agent: str) -> float:
    """計算 query 對某 agent 的 keyword 匹配分數"""
    total = 0.0
    for pattern, weight in AGENT_RULES[agent]:
        if re.search(pattern, query, re.IGNORECASE):
            total += weight
    return total


def route(query: str) -> dict:
    """
    用 keyword matching 路由 query 到某個 agent。

    Returns:
        {
            "agent": str,           # 路由結果
            "confidence": float,    # 最高分（用於 hybrid router 的 threshold 判斷）
            "scores": dict,         # 各 agent 的分數（debug 用）
            "method": "keyword"
        }
    """
    q = query.lower().strip()

    scores = {}
    for agent in AGENT_RULES:
        s = _score_agent(q, agent)
        if s > 0:
            scores[agent] = round(s, 2)

    if not scores:
        return {
            "agent": "oos",
            "confidence": 0.0,
            "scores": {},
            "method": "keyword",
        }

    best_agent = max(scores, key=scores.get)
    best_score = scores[best_agent]

    # 如果最高分很低（< 0.5），判定為 oos
    if best_score < 0.5:
        return {
            "agent": "oos",
            "confidence": best_score,
            "scores": scores,
            "method": "keyword",
        }

    return {
        "agent": best_agent,
        "confidence": best_score,
        "scores": scores,
        "method": "keyword",
    }


# ─────────────────────────────────────────────
# 直接執行時做簡單測試
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "what's my credit score",
        "book a flight to tokyo",
        "when should i get my oil changed",
        "how many calories in an avocado",
        "set a reminder for 3pm",
        "play some jazz music",
        "tell me a joke",
        "i need to transfer money to my savings",
        "what's the weather like tomorrow",
        "do i need a visa for japan",
        "asdfghjkl random gibberish xyz",
    ]

    for q in test_queries:
        r = route(q)
        print(f"  [{r['agent']:20s}] (conf={r['confidence']:.1f})  {q}")
