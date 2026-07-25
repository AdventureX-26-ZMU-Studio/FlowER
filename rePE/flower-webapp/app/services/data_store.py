"""JSON 文件数据读写模块"""
import json
from pathlib import Path


def get_data_path(filename):
    """返回 data 目录下的文件路径"""
    base_dir = Path(__file__).resolve().parent.parent.parent / 'data'
    base_dir.mkdir(exist_ok=True)
    return base_dir / filename


def load_json(filename, default=None):
    """读取 JSON 文件，不存在则返回默认值"""
    if default is None:
        default = []
    filepath = get_data_path(filename)
    if not filepath.exists():
        return default
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(filename, data):
    """写入 JSON 文件"""
    filepath = get_data_path(filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_entry(filename, entry):
    """向 JSON 数组文件追加条目"""
    data = load_json(filename, default=[])
    data.append(entry)
    save_json(filename, data)
