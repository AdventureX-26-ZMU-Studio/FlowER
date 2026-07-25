from flask import render_template, jsonify, request
from app.blueprints.diary import bp
from app.services.data_store import load_json, save_json, append_entry
from app.services.llm_service import generate_card_text, summarize_cards
from datetime import datetime


@bp.route('')
def main():
    """FlowER日记主页 - 展示昨日日记 + 实时卡片"""
    diary_entries = load_json('diary_entries.json', default=[])
    cards = load_json('cards.json', default={'now': [], 'recent': []})
    
    # 获取昨日日记（最新一条）
    yesterday_entry = diary_entries[-1] if diary_entries else None
    
    return render_template('diary/main.html',
                           entry=yesterday_entry,
                           now_cards=cards.get('now', []),
                           recent_summaries=cards.get('recent', []))


@bp.route('/历史')
def history():
    """历史日记列表"""
    diary_entries = load_json('diary_entries.json', default=[])
    return render_template('diary/history.html', entries=diary_entries)


@bp.route('/api/cards', methods=['GET'])
def get_cards():
    """获取当前卡片数据"""
    cards = load_json('cards.json', default={'now': [], 'recent': []})
    return jsonify(cards)


@bp.route('/api/generate-card', methods=['POST'])
def generate_card():
    """生成新的"现在……"卡片（调用 LLM 占位符）"""
    text = generate_card_text()
    card = {
        'id': datetime.now().strftime('%Y%m%d%H%M%S'),
        'time': datetime.now().strftime('%H:%M'),
        'text': text,
        'created_at': datetime.now().isoformat()
    }
    
    cards = load_json('cards.json', default={'now': [], 'recent': []})
    cards['now'].append(card)
    save_json('cards.json', cards)
    
    return jsonify({'success': True, 'card': card})


@bp.route('/api/summarize', methods=['POST'])
def summarize():
    """触发小时总结 - 将"现在"卡片压缩为"最近"条目"""
    cards = load_json('cards.json', default={'now': [], 'recent': []})
    
    if not cards['now']:
        return jsonify({'success': False, 'message': '没有需要总结的卡片'})
    
    # 调用 LLM 总结
    card_texts = [c['text'] for c in cards['now']]
    summary_text = summarize_cards(card_texts)
    
    summary = {
        'id': datetime.now().strftime('%Y%m%d%H'),
        'time': datetime.now().strftime('%Y-%m-%d %H:00'),
        'text': summary_text,
        'card_count': len(cards['now']),
        'created_at': datetime.now().isoformat()
    }
    
    # 清理"现在"卡片，添加到"最近"
    cards['recent'].append(summary)
    cards['now'] = []
    save_json('cards.json', cards)
    
    return jsonify({'success': True, 'summary': summary})
