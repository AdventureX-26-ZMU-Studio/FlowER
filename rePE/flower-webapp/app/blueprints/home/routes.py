from flask import render_template
from app.blueprints.home import bp


@bp.route('/')
def index():
    """首页 - 导航枢纽"""
    return render_template('home/index.html')
