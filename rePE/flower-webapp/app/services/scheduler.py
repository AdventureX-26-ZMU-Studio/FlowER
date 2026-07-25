"""定时任务调度器 - FlowER日记自动化"""
import logging
import os
from datetime import datetime

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:
    BackgroundScheduler = None

from app.services.api_client import get_flower_image
from app.services.llm_service import generate_card_text, summarize_cards, generate_daily_diary
from app.services.data_store import load_json, save_json

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler() if BackgroundScheduler else None


# ==================== 15分钟：生成「现在……」卡片 ====================
def job_generate_now_card():
    """每15分钟从API获取花朵图片，调用LLM生成卡片"""
    try:
        # 获取花朵状态图片
        image_data = get_flower_image()

        # 调用LLM生成卡片文本
        text = generate_card_text(image_data)

        card = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "time": datetime.now().strftime("%H:%M"),
            "text": text,
            "created_at": datetime.now().isoformat(),
        }

        cards = load_json("cards.json", default={"now": [], "recent": []})
        if "now" not in cards:
            cards["now"] = []
        if "recent" not in cards:
            cards["recent"] = []
        cards["now"].append(card)
        save_json("cards.json", cards)

        logger.info(f"✅ 现在卡片已生成: {card['time']} - {text[:30]}...")
    except Exception as e:
        logger.error(f"❌ 现在卡片生成失败: {e}")


# ==================== 1小时：总结「现在」→「最近」 ====================
def job_summarize_to_recent():
    """每小时将「现在」卡片总结为「最近」条目"""
    try:
        cards = load_json("cards.json", default={"now": [], "recent": []})
        if "now" not in cards:
            cards["now"] = []
        if "recent" not in cards:
            cards["recent"] = []

        if not cards["now"]:
            logger.info("⏭ 没有需要总结的卡片，跳过小时总结")
            return

        # 调用LLM总结
        summary_text = summarize_cards(cards["now"])

        summary = {
            "id": datetime.now().strftime("%Y%m%d%H"),
            "time": datetime.now().strftime("%Y-%m-%d %H:00"),
            "text": summary_text,
            "card_count": len(cards["now"]),
            "created_at": datetime.now().isoformat(),
        }

        cards["recent"].append(summary)
        cards["now"] = []
        save_json("cards.json", cards)

        logger.info(f"✅ 小时总结完成: {summary['time']} ({summary['card_count']}条卡片)")
    except Exception as e:
        logger.error(f"❌ 小时总结失败: {e}")


# ==================== 每日：整合「最近」→ 完整日记 ====================
def job_generate_daily_diary():
    """每日将「最近」整合为完整日记，永久保存"""
    try:
        cards = load_json("cards.json", default={"now": [], "recent": []})
        if "recent" not in cards:
            cards["recent"] = []

        recent = cards["recent"]
        if not recent:
            logger.info("⏭ 没有需要整合的日记，跳过每日日记")
            return

        # 调用LLM生成完整日记
        diary_text = generate_daily_diary(recent)

        diary_entry = {
            "id": datetime.now().strftime("%Y%m%d"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "content": diary_text,
            "summary_count": len(recent),
            "created_at": datetime.now().isoformat(),
        }

        entries = load_json("diary_entries.json", default=[])
        entries.append(diary_entry)
        save_json("diary_entries.json", entries)

        # 清空「最近」
        cards["recent"] = []
        save_json("cards.json", cards)

        logger.info(f"✅ 每日日记已生成: {diary_entry['date']}")
    except Exception as e:
        logger.error(f"❌ 每日日记生成失败: {e}")


def init_scheduler(app):
    """初始化定时任务"""
    if scheduler is None:
        return
    # 避免 Flask debug reloader 子进程重复启动
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    interval_snapshot = app.config.get("SNAPSHOT_INTERVAL", 15 * 60)
    interval_summary = app.config.get("SUMMARY_INTERVAL", 60 * 60)
    interval_daily = app.config.get("DAILY_INTERVAL", 24 * 60 * 60)

    scheduler.add_job(
        job_generate_now_card,
        "interval",
        seconds=interval_snapshot,
        id="snapshot",
        replace_existing=True,
    )
    scheduler.add_job(
        job_summarize_to_recent,
        "interval",
        seconds=interval_summary,
        id="summary",
        replace_existing=True,
    )
    scheduler.add_job(
        job_generate_daily_diary,
        "interval",
        seconds=interval_daily,
        id="daily",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("📅 日记定时任务已启动")
    logger.info(f"   🟢 每{interval_snapshot // 60}分钟生成「现在」卡片")
    logger.info(f"   🔵 每{interval_summary // 60}分钟总结「最近」")
    logger.info(f"   🟣 每{interval_daily // 3600}小时生成每日日记")
