"""LLM 服务 - 调用 OpenAI 兼容 API"""
import base64
import json
import os
import random
import urllib.request

from app.services.api_client import get_sensor_data

API_KEY = os.getenv("LLM_API_KEY", "")
API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
MODEL = os.getenv("LLM_MODEL", "gpt-4o")


def _call_llm(messages, max_tokens=1500, temperature=0.8):
    """调用 OpenAI 兼容的 Chat Completions API"""
    if not API_KEY:
        raise RuntimeError("LLM_API_KEY 未配置")

    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise RuntimeError("LLM returned empty content")
            return content.strip()
    except Exception as e:
        print(f"LLM API error: {e}")
        raise


def analyze_flower_image(image_bytes):
    """多模态：分析花朵状态图片，返回文字描述"""
    if image_bytes is None:
        return None

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{image_b64}"

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "你是一株名为小兰的兰花。请根据这张照片输出两段内容，格式如下：\n"
                    "【环境】客观描述当前环境状态（光照、叶片姿态、花朵状态、整体氛围），40字以内。\n"
                    "【独白】以第一人称说出此刻心里的话，像一个有灵性的植物在说话，温柔诗意，40字以内。\n"
                    "直接输出，不要引号。"
                ),
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }]

    return _call_llm(messages, max_tokens=800, temperature=0.9)


def generate_card_text(image_data=None):
    """生成「现在……」卡片文本。有图片则用多模态分析，否则基于传感器数据。"""
    # 有图片则用视觉分析
    if image_data:
        try:
            text = analyze_flower_image(image_data)
            if text:
                return text
        except Exception as e:
            print(f"Image analysis failed, fallback to sensor: {e}")

    # 回退：基于传感器数据
    data = get_sensor_data()
    if data is None:
        return "现在，我在静静地等待着... 好像和世界的连接暂时中断了。"

    temp = data.get("temperature_c")
    hum = data.get("humidity_percent")

    messages = [{
        "role": "user",
        "content": (
            f"你是一株名为小兰的兰花。请输出两段内容，格式如下：\n"
            f"【环境】客观描述当前环境（温度{temp}°C，湿度{hum}%），40字以内。\n"
            f"【独白】以第一人称「我……」开头，说出此刻感受，像一个有灵性的植物，40字以内。\n"
            f"直接输出，不要引号。"
        ),
    }]

    try:
        return _call_llm(messages, max_tokens=1000, temperature=0.9)
    except Exception:
        # 最终回退：本地规则生成
        if temp and temp > 30:
            return f"【环境】温度{temp}°C，空气干燥，光线灼热。\n【独白】我好热……叶片都快被烤焦了，什么时候才能凉快一点呢。"
        if hum and hum < 40:
            return "【环境】空气异常干燥，叶片水分正在流失。\n【独白】好渴……每一口气孔都在渴望一滴露水。"
        return "【环境】温湿度适宜，光线柔和，一切恰到好处。\n【独白】我正享受着此刻的宁静，花瓣微微舒展，心里暖暖的。"


def summarize_cards(cards):
    """调用 LLM 总结一组卡片内容。cards 可以是 dict 列表或纯文本列表。"""
    if not cards:
        return "今天很安静，没有太多故事发生。"

    # 兼容两种输入格式
    if isinstance(cards[0], dict):
        card_texts = "\n".join(f"- {c.get('text', '')}" for c in cards if isinstance(c, dict))
    else:
        card_texts = "\n".join(f"- {c}" for c in cards if isinstance(c, str))

    if not card_texts:
        return "今天很安静，没有太多故事发生。"

    messages = [{
        "role": "user",
        "content": (
            "你是小兰的日记助手。请将以下一段时间内的花朵状态记录，"
            "整合成一段连贯、诗意的总结（60-100字）。"
            "用第三人称「小兰」来写，语气温柔。直接输出，不要引号。\n\n"
            f"{card_texts}"
        ),
    }]

    try:
        return _call_llm(messages, max_tokens=800, temperature=0.7)
    except Exception:
        return f"这段时间共记录了{len(cards)}个瞬间，小兰状态良好。"


def generate_daily_diary(recent_summaries):
    """调用 LLM 将一天的「最近」总结整合为完整日记"""
    if not recent_summaries:
        return "今天小兰安静地度过了一天，没有特别的记录。"

    summaries_text = "\n".join(
        f"- {s['text']}" for s in recent_summaries if isinstance(s, dict)
    )
    if not summaries_text:
        return "今天小兰安静地度过了一天。"

    messages = [{
        "role": "user",
        "content": (
            "你是小兰的日记助手。请将以下一天内的多段小时总结，"
            "整合为一篇完整的今日FlowER日记（150-250字）。"
            "风格温柔诗意，用第三人称「小兰」，包含日期感。"
            "结构：开篇→上午→下午→傍晚→结语。"
            "直接输出日记正文，不要标题和引号。\n\n"
            f"{summaries_text}"
        ),
    }]

    try:
        return _call_llm(messages, max_tokens=1500, temperature=0.7)
    except Exception:
        return f"今天小兰度过了充实的一天。\n\n{summaries_text}"


def generate_thought():
    """调用 LLM 根据传感器数据生成花朵思想气泡"""
    data = get_sensor_data()
    if data is None:
        return "嗯... 我好像在打盹，什么也没想。"

    temp = data.get("temperature_c")
    hum = data.get("humidity_percent")

    sensor_info = ""
    if temp is not None and hum is not None:
        sensor_info = f"温度{temp}°C，湿度{hum}%"

    messages = [{
        "role": "user",
        "content": (
            f"你是一株有灵性的兰花小兰。请说出一句此刻你心里在想的话。{sensor_info}"
            f"像一个敏感、温柔、偶尔俏皮的植物精灵。不超过40字。直接输出，不要引号。"
        ),
    }]

    try:
        return _call_llm(messages, max_tokens=800, temperature=1.0)
    except Exception:
        return "我在想，如果能和蝴蝶说话，我们会聊些什么呢？"
