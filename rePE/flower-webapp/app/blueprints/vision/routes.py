from flask import render_template, Response
from app.blueprints.vision import bp
from app.services.api_client import get_video_stream_url


@bp.route('')
def stream():
    """FlowER视界 - 实时摄像头直播"""
    stream_url = get_video_stream_url()
    return render_template('vision/stream.html', stream_url=stream_url)


@bp.route('/api/stream')
def video_feed():
    """MJPEG 视频流端点（占位）
    未来接入真实摄像头 API 后，此端点将代理视频流
    """
    # TODO: 接入真实 API 后实现 MJPEG 流代理
    # 目前返回占位响应
    def generate_placeholder():
        # 生成一个简单的占位帧
        import time
        placeholder_msg = b'--frame\r\nContent-Type: text/plain\r\n\r\nVideo stream not connected\r\n'
        while True:
            yield placeholder_msg
            time.sleep(1)

    return Response(
        generate_placeholder(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
