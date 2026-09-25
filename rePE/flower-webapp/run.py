"""FlowER WebApp 启动脚本 - 使用 dimos venv (含 pyorbbecsdk)"""
from app import create_app
from app.config import config_dict
import os

config_name = os.getenv("FLASK_ENV", "development")
app = create_app(config_name)

if __name__ == "__main__":
    config = config_dict.get(config_name, config_dict["development"])
    debug = getattr(config, "DEBUG", False)
    print("🌸 FlowER WebApp 启动中...")
    print(f"   配置: {config_name}")
    print(f"   摄像头: 奥比中光 DaBai DCW (直连模式)")
    print(f"   视频流: /FlowER视界/api/mjpeg")
    app.run(host="0.0.0.0", port=5000, debug=debug)
