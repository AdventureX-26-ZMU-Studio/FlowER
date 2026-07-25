from flask import Blueprint

bp = Blueprint('vision', __name__)

from app.blueprints.vision import routes  # noqa: E402, F401
