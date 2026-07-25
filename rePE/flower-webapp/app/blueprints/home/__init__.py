from flask import Blueprint

bp = Blueprint('home', __name__)

from app.blueprints.home import routes  # noqa: E402, F401
