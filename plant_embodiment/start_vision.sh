#!/bin/bash
# start-plant-vision — 一键启动 DCW Vision Server
# 安装到 /usr/local/bin/start-plant-vision 后可直接运行

DIMOS_DIR="/home/asus/dimos-advx-zmu"
VENV="$DIMOS_DIR/.venv-mediapipe/bin/python"
MODULE="dimos.hardware.sensors.camera.orbbec_dcw.vision_server"
LD_PATH="/home/asus/orbbec_sdk_libs"

export LD_LIBRARY_PATH="$LD_PATH"

if [ ! -f "$VENV" ]; then
    echo "❌ 找不到 Python 环境: $VENV"
    echo "   检查 dimos 是否安装在 $DIMOS_DIR"
    exit 1
fi

cd "$DIMOS_DIR" || { echo "❌ 无法进入 $DIMOS_DIR"; exit 1; }

echo "🚀 启动 Plant Embodiment Vision Server (port 8890)..."
echo "   SDK: $LD_PATH"
echo "   进程: $VENV -m $MODULE"
echo ""

exec "$VENV" -m "$MODULE" --host 0.0.0.0 --port 8890
