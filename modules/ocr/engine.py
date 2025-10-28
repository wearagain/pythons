"""
OCR 모듈 기초
"""

import cv2
import numpy as np
import easyocr
from typing import Dict, List, Tuple, Any

from config import settings, get_logger

logger = get_logger(__name__)

class OCREngine:
    def __init__(self):
        self.config = settings.OCR_CONFIG
        
        logger.info("EasyOCR 모델 로딩 중... (최초 실행시 다운로드됩니다)")
        
        self.reader = easyocr.Reader(
            lang_list=self.config['languages'],
            gpu=self.config['gpu'],
            verbose=False
        )
        
        logger.info("✓ OCR 모델 로딩 완료")
        
    def extract_text(self, ticket_image: np.ndarray) -> Dict[str, Any]:
        logger.debug("OCR 텍스트 추출 시작")
        
        processed = self._preprocess_for_ocr(ticket_image)
        
        results = self.reader.readtext(processed)
        
        logger.debug(f"OCR 완료: {len(results)}개 텍스트 검출")
        
        all_text = []
        total_confidence = 0.0
        valid_count = 0
        
        for (bbox, text, conf) in results:
            if conf >= self.config['min_confidence']:
                cleaned_text = text.strip()
                if cleaned_text:  
                    all_text.append(cleaned_text)
                    total_confidence += conf
                    valid_count += 1
        
        
        avg_confidence = total_confidence / valid_count if valid_count > 0 else 0.0
        
        
        full_text = '\n'.join(all_text)
        
        result = {
            'text': full_text,
            'confidence': float(avg_confidence),
            'raw_results': results,
            'line_count': len(all_text)
        }
        
        logger.info(
            f"OCR 추출 완료 "
            f"(신뢰도: {avg_confidence:.2%}, 줄 수: {len(all_text)})"
        )
        
        return result
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def get_text_lines(self, ocr_result: Dict[str, Any]) -> List[str]:
        return ocr_result['text'].split('\n')