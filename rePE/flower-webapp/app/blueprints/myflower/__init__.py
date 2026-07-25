from flask import Blueprint

bp = Blueprint('myflower', __name__)

from app.blueprints.myflower import routes  # noqa: E402, F401
