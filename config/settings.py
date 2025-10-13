"""
프로젝트 설정 관리    
"""

import os 
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Settings:
    BASE_DIR = Path(__file__).parent.parent
    
    MODULES_DIR = BASE_DIR / "modules"
    API_DIR = BASE_DIR / "api"
    CONFIG_DIR = BASE_DIR / "config"
    OUTPUT_DIR = BASE_DIR / "output"
    DETECTED_DIR = OUTPUT_DIR / "detected"
    LOGS_DIR = OUTPUT_DIR / "logs"
    
    DETECTOR_CONFIG: Dict[str, Any] = {
        'min_area': int(os.getenv('MIN_TICKET_AREA', 15000)),
        'max_area': int(os.getenv('MAX_TICKET_AREA', 1500000)),
        'aspect_ratio_min': float(os.getenv('ASPECT_RATIO_MIN', 0.4)),
        'aspect_ratio_max': float(os.getenv('ASPECT_RATIO_MAX', 0.9)),
        'approx_epsilon': 0.02,        
        'margin': 10,                  
    }
    
    OCR_CONFIG: Dict[str, Any] = {
        'languages': os.getenv('OCR_LANGUAGES', 'ko,en').split(','),
        'gpu': os.getenv('OCR_GPU', 'False').lower() == 'true',
        'min_confidence': 0.3,
        'paragraph': False,
        'detail': 1,
    }
    
    LOG_CONFIG: Dict[str, Any] = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
    }
    
    IMAGE_CONFIG: Dict[str, Any] = {
        'max_width': 1920,
        'max_height': 1080,
        'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp'],
        'jpeg_quality': 95,
    }
    
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        directories = [
            cls.DETECTED_DIR,
            cls.LOGS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
            # .gitkeep 파일 생성
            gitkeep = directory / '.gitkeep'
            if not gitkeep.exists():
                gitkeep.touch()
    
# 싱글톤 인스턴스
settings = Settings()

# 디렉토리 자동 생성
settings.create_directories()