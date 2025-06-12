import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# 注意：我们不再从这里导入 models，以避免潜在的循环导入问题
# from models import NotificationConfig

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    为应用设置日志记录配置。
    """
    logger_instance = logging.getLogger("wrapper_api")
    
    # 如果已经有处理器，说明已配置过，直接返回，防止重复添加
    if logger_instance.hasHandlers():
        return logger_instance

    log_level_const = getattr(logging, log_level.upper(), logging.INFO)
    logger_instance.setLevel(log_level_const)
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger_instance.addHandler(console_handler)
    
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(detailed_formatter)
        logger_instance.addHandler(file_handler)
    
    logger_instance.propagate = False
    
    logger_instance.info(f"Logging setup completed. Level: {logging.getLevelName(log_level_const)}")
    if log_file:
        logger_instance.info(f"Log file: {log_file}")
    
    return logger_instance

def get_env_log_level() -> str:
    """
    从环境变量获取日志级别或返回默认值。
    """
    return os.getenv("LOG_LEVEL", "INFO").upper()

def notify_on_critical_error(message: str, config=None): # config 类型提示暂时移除
    """
    发送严重错误通知。
    目前只记录到 CRITICAL 级别，可以扩展为邮件/webhook。
    """
    # 直接获取已配置的 logger
    logger = logging.getLogger("wrapper_api")
    alert_message = f"CRITICAL_ALERT: {message}"
    logger.critical(alert_message)

    if config and hasattr(config, 'log_only') and config.log_only:
        pass # 已经记录
    # 未来：在此处添加邮件/webhook逻辑