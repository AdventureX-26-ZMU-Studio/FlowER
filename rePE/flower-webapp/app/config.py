import os


class Config:
    """基础配置"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://10.80.5.215:5001')
    LLM_API_KEY = os.getenv('LLM_API_KEY', '')
    LLM_API_BASE = os.getenv('LLM_API_BASE', 'https://api.openai.com/v1')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o')

    # 定时任务间隔（秒）
    SNAPSHOT_INTERVAL = 60 * 60      # 1小时
    SUMMARY_INTERVAL = 60 * 60       # 1小时
    DAILY_INTERVAL = 24 * 60 * 60    # 24小时

    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_dict = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
}
