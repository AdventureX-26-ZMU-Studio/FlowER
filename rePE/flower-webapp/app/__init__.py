from flask import Flask
from dotenv import load_dotenv
import os


def create_app(config_name="development"):
    load_dotenv()
    app = Flask(__name__)

    from app.config import config_dict
    app.config.from_object(config_dict.get(config_name, config_dict["development"]))

    from app.blueprints.home import bp as home_bp
    from app.blueprints.vision import bp as vision_bp
    from app.blueprints.diary import bp as diary_bp
    from app.blueprints.myflower import bp as myflower_bp
    from app.blueprints.control import bp as control_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(vision_bp, url_prefix="/FlowER视界")
    app.register_blueprint(diary_bp, url_prefix="/FlowER日记")
    app.register_blueprint(myflower_bp, url_prefix="/我的FlowER")
    app.register_blueprint(control_bp, url_prefix="/k2bin1s")

    # Initialize WebSocket for DCW camera stream
    from app.blueprints.vision.routes import sock
    sock.init_app(app)

    # Force no-cache on ALL HTML responses
    @app.after_request
    def add_no_cache(response):
        if response.content_type and 'text/html' in response.content_type:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    from pathlib import Path
    (Path(app.root_path).parent / "data").mkdir(exist_ok=True)
    (Path(app.root_path).parent / ".temp").mkdir(exist_ok=True)

    try:
        from app.services.scheduler import init_scheduler
        init_scheduler(app)
    except ImportError:
        pass

    return app
