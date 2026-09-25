"""FlowER视界 - 直接奥比中光 RGB 摄像头视频流"""
from flask import render_template, Response
from app.blueprints.vision import bp


@bp.route('')
def stream():
    """FlowER视界 - 实时摄像头直播页面"""
    return render_template('vision/stream.html')


@bp.route('/api/snapshot')
def snapshot():
    """单帧 JPEG 快照 - 用于 JS 定时刷新"""
    from app.camera import get_camera
    cam = get_camera()
    jpg = cam.capture_jpeg()
    if jpg is not None:
        return Response(jpg, mimetype='image/jpeg')
    else:
        # 返回 204 No Content 让前端保持当前画面
        return Response(status=204)


@bp.route('/api/mjpeg')
def video_feed():
    """MJPEG 视频流（保留给 Firefox 等原生支持的浏览器）"""
    from app.camera import get_camera
    
    def generate():
        cam = get_camera()
        while True:
            jpg = cam.capture_jpeg()
            if jpg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
            else:
                import time
                time.sleep(0.03)
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
