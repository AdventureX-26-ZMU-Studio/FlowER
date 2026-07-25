from flask import Blueprint

bp = Blueprint('diary', __name__)

from app.blueprints.diary import routes  # noqa: E402, F401
