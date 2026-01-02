import sys
import os
from loguru import logger

def setup_console_logging(
    app_name: str = "my-app",
    log_level: str | None = None,
):
   
    logger.remove()

    log_level = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()

    env = os.getenv("ENV", "production").lower()
    is_production = env in {"production", "prod", "staging"}

    if is_production:
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True,          
            enqueue=True,            
            backtrace=True,          
            diagnose=False,          
            colorize=False,
        )
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            level=log_level,
            format=console_format,
            colorize=True,
            enqueue=True,
            backtrace=True,
            diagnose=True,  
        )

    common_fields = {
        "app": app_name,
        "env": env,
        "pod_name": os.getenv("HOSTNAME", "local"), 
        "namespace": "unknown",     
        "node_name": "unknown",
    }

    return logger.bind(**common_fields)


