"""
이미지 설정 
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union

from config.settings import settings
from config.logger import get_logger

logger = get_logger(__name__)

def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    image_path = Path(image_path)
    
    if not image_path.exists():
        logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return None
    
    image = cv2.imread(str(image_path))
    
    if image is None:
        logger.error(f"이미지를 읽을 수 없습니다: {image_path}")
        return None
    
    return image

def save_image(image:np.ndarray, output_path: Union[str, Path], quality: Optional[int] = None) -> bool:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if quality is None:
        quality = settings.IMAGE_CONFIG['jpeg_quality']
        
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success = cv2.imwrite(str(output_path), image, encode_param)
    return success

def resize_image(image:np.ndarray, max_width: Optional[int]=None, max_height: Optional[int]=None) -> np.ndarray:
    if max_width is None:
        max_width = settings.IMAGE_CONFIG['max_width']
    if max_height is None:
        max_height = settings.IMAGE_CONFIG['max_height']
    
    height, width = image.shape[:2]
    
    if width <= max_width and height <= max_height:
        return image
    
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def is_valid_image(image_path: Union[str, Path]) -> bool:
    image_path = Path(image_path)
    
    if not image_path.exists():
        return False
    if image_path.suffix.lower() not in settings.IMAGE_CONFIG['supported_formats']:
        return False
    
    image = cv2.imread(str(image_path))
    return image is not None

def convert_to_grayscale(image:np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image