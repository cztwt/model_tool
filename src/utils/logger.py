import logging
import os

logger_name = os.path.splitext(os.path.basename(__file__))[0]  # 获取当前文件的名称（不包括扩展名）
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(logger_name)
