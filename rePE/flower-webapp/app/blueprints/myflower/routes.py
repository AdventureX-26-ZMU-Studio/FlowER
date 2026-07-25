from flask import render_template, jsonify, request
from app.blueprints.myflower import bp
from app.services.api_client import get_mood, get_thoughts, get_physiological_data
from app.services.llm_service import generate_thought
from app.services.data_store import load_json, save_json


@bp.route('')
def card():
    """我的FlowER - 花朵卡片页面"""
    flower_data = load_json('flower.json', default={
        'name': '小兰',
        'species': '兰花',
        'birthday': '2026-06-01'
    })
    return render_template('myflower/card.html', flower=flower_data)


@bp.route('/api/mood', methods=['GET'])
def api_mood():
    """获取当前心情"""
    mood = get_mood()
    return jsonify(mood)


@bp.route('/api/thought', methods=['GET'])
def api_thought():
    """获取 AI 生成的思想气泡"""
    thought = generate_thought()
    return jsonify({'thought': thought})


@bp.route('/api/physiological', methods=['GET'])
def api_physiological():
    """获取生理数据"""
    data = get_physiological_data()
    return jsonify(data)


@bp.route('/api/rename', methods=['POST'])
def api_rename():
    """重命名花朵"""
    data = request.get_json()
    new_name = data.get('name', '').strip()

    if not new_name:
        return jsonify({'success': False, 'message': '名称不能为空'}), 400

    if len(new_name) > 20:
        return jsonify({'success': False, 'message': '名称不能超过20个字符'}), 400

    flower_data = load_json('flower.json', default={'name': '小兰'})
    flower_data['name'] = new_name
    save_json('flower.json', flower_data)

    return jsonify({'success': True, 'name': new_name})
