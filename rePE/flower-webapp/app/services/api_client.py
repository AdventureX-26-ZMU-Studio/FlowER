"""外部 API 客户端 - 从 OPI 传感器 API 获取数据"""
import json
import random
import urllib.request
import os

# OPI sensor API base URL
API_BASE = os.getenv("FLOWER_API_BASE", "http://10.80.5.215:5001")


def _fetch(url):
    """Simple HTTP GET returning parsed JSON, or None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"API fetch error ({url}): {e}")
        return None


def get_video_stream_url():
    """获取 DCW 相机直连 MJPEG 视频流 URL（绕过 VisionHub）"""
    return "/FlowER视界/api/stream"


def get_flower_image():
    """从本地 DCW 相机获取最新帧（JPEG bytes），供日记多模态分析。"""
    try:
        # 直接从 DCW 相机拉一帧
        from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline
        import cv2, numpy as np

        pipeline, config = Pipeline(), Config()
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(640, 360, OBFormat.MJPG, 30)
        config.enable_stream(color_profile)
        pipeline.start(config)
        frames = pipeline.wait_for_frames(1000)
        if frames:
            color = frames.get_color_frame()
            if color:
                data = np.frombuffer(color.get_data(), dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                _, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                pipeline.stop()
                return jpg.tobytes()
        pipeline.stop()
    except Exception as e:
        print(f"DCW snapshot error: {e}")
    return None


def get_sensor_data():
    """从 OPI 获取传感器数据"""
    data = _fetch(f"{API_BASE}/api/sensors")
    if data is None:
        return None
    return data


def get_mood():
    """根据温度和湿度推断花朵心情"""
    data = get_sensor_data()
    if data is None:
        return {"mood": "离线", "emoji": "💤", "intensity": 0.1}

    temp = data.get("temperature_c")
    hum = data.get("humidity_percent")

    # 基于温度 + 湿度的心情推断
    if temp is not None and hum is not None:
        # 理想状态：20-28°C + 50-80% 湿度
        if 20 <= temp <= 28 and 50 <= hum <= 80:
            return {"mood": "舒适", "emoji": "😊", "intensity": 0.9}
        # 高温高湿
        if temp > 30 and hum > 70:
            return {"mood": "闷热", "emoji": "🥵", "intensity": 0.8}
        # 高温低湿
        if temp > 30 and hum < 50:
            return {"mood": "酷热", "emoji": "🌵", "intensity": 0.85}
        # 低温
        if temp < 18:
            return {"mood": "好冷", "emoji": "🥶", "intensity": 0.7}
        # 干燥
        if hum < 40:
            return {"mood": "干燥", "emoji": "😐", "intensity": 0.6}
        # 温暖偏热
        if 28 < temp <= 30:
            return {"mood": "温暖", "emoji": "🌤️", "intensity": 0.65}

    if temp is not None:
        if temp > 30:
            return {"mood": "好热", "emoji": "🌞", "intensity": 0.75}
        if 20 <= temp <= 28:
            return {"mood": "舒适", "emoji": "😊", "intensity": 0.8}

    if hum is not None:
        if hum < 40:
            return {"mood": "干燥", "emoji": "😐", "intensity": 0.6}
        if hum > 80:
            return {"mood": "湿润", "emoji": "💧", "intensity": 0.7}

    moods = [
        {"mood": "开心", "emoji": "😊", "intensity": 0.8},
        {"mood": "平静", "emoji": "😌", "intensity": 0.5},
        {"mood": "满足", "emoji": "🥰", "intensity": 0.85},
    ]
    return random.choice(moods)


def get_thoughts():
    """根据传感器数据生成花朵思想（已废弃，由 llm_service 替代）"""
    data = get_sensor_data()
    thoughts = [
        "今天的阳光真温暖，我想再多晒一会儿。",
        "有只蝴蝶停在我身边，我们静静地享受着微风。",
        "泥土今天很湿润，感觉根部很舒服。",
        "隔壁的小草又长高了一点，真为它高兴。",
        "今晚的月光一定很美，我要早点休息等明天的朝露。",
        "有人在给我浇水，凉凉的好舒服呀。",
    ]
    if data and data.get("soil_moisture") is False:
        return "好渴呀... 泥土干干的，希望主人快来给我浇水。"
    if data and data.get("temperature_c", 0) > 30:
        return f"今天{data['temperature_c']}°C呢，暖暖的阳光让我充满能量！"
    return random.choice(thoughts)


def get_physiological_data():
    """从 OPI 获取生理数据"""
    data = get_sensor_data()
    if data is None:
        return {
            "temperature": None,
            "humidity": None,
            "soil_moisture": None,
        }
    soil_wet = data.get("soil_moisture")
    return {
        "temperature": data.get("temperature_c"),
        "humidity": data.get("humidity_percent"),
        "soil_moisture": "是" if soil_wet else "否",
    }
