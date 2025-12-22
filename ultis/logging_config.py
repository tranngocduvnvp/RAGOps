import sys
import os
from loguru import logger

def setup_console_logging(
    app_name: str = "my-app",
    log_level: str | None = None,
):
    """
    Cấu hình Loguru chỉ sử dụng console sink (stdout).
    Tối ưu cho:
      - Development: log có màu, format đẹp dễ đọc.
      - Production/Kubernetes: log JSON structured, không màu, dễ parse bởi ELK/Loki.
    
    Args:
        app_name: Tên ứng dụng (sẽ được bind vào mọi log).
        log_level: Mức log (DEBUG, INFO, WARNING, ERROR, CRITICAL). 
                   Nếu None thì lấy từ env LOG_LEVEL, mặc định INFO.
    
    Returns:
        logger đã được cấu hình và bind các field chung.
    """
    # Xóa tất cả handler mặc định của Loguru
    logger.remove()

    # Xác định mức log
    log_level = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()

    # Phát hiện môi trường để quyết định format
    env = os.getenv("ENV", "production").lower()
    is_production = env in {"production", "prod", "staging"}

    if is_production:
        # Production / Kubernetes: JSON structured logs (mỗi line một JSON)
        logger.add(
            sys.stdout,
            level=log_level,
            serialize=True,          # Quan trọng: output JSON
            enqueue=True,            # Thread & multiprocess safe
            backtrace=True,          # Giữ full traceback cho exception
            diagnose=False,          # Tắt để tránh lộ dữ liệu nhạy cảm
            colorize=False,
        )
    else:
        # Development: format đẹp, có màu, dễ đọc cho con người
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
            diagnose=True,  # Ở dev có thể bật để debug dễ hơn
        )

    # Các field chung luôn có trong mọi log (rất hữu ích khi query trong Kibana/Grafana)
    common_fields = {
        "app": app_name,
        "env": env,
        "pod_name": os.getenv("HOSTNAME", "local"),  # Luôn có trong K8s
        "namespace": "unknown",     # Nếu không quan trọng có thể để vậy
        "node_name": "unknown",
    }

    # Bind các field này vào logger toàn cục
    return logger.bind(**common_fields)


