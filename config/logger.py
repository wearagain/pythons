"""
로깅 설정
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style, init

from config.settings import settings

init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """ 컬러 설정 """
    COLORS = {
        'DEBUG': Fore.WHITE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        return super().format(record)
    

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # 중복 방지 
    if logger.handlers:
        return logger
    
    log_config = settings.LOG_CONFIG
    log_level = getattr(logging, log_config['level'])
    logger.setLevel(log_level)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(
        log_config['format'],
        datefmt=log_config['date_format']
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    if not settings.DEBUG:
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = settings.LOGS_DIR / f"app_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            log_config['format'],
            datefmt = log_config['date_format']
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    return logger
