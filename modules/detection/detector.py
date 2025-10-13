"""
티켓 검출기 모듈 

OpenCV 기반 티켓 검출 및 추출     
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

from config import settings, get_logger, load_image
from modules.detection.preprocessor import Preprocessor
from modules.detection.transformer import PrespectiveTransformer

logger = get_logger(__name__)

class TicketDetector:
    def __init__(self):
        self.config = settings.DETECTOR_CONFIG
        self.preprocessor = Preprocessor()
        self.transformer = PrespectiveTransformer(
            margin = self.config['margin']
        )
        
        # 설정값 로드 
        self.min_area = self.config['min_area']
        self.max_area = self.config['max_area']
        self.aspect_ratio_range = (
            self.config['aspect_ratio_min'],
            self.config['aspect_ratio_max']
        )
        self.approx_epsilon = self.config['approx_epsilon']
        
        logger.info(
            f"티켓 검출기 초기화 완료 "
            f"(면적: {self.min_area}~{self.max_area}, "
            f"종횡비: {self.aspect_ratio_range})"
        )
        
    
    def detect(self, image: Union[np.ndarray, str, Path]) -> List[Tuple[np.ndarray, float]]:
        # 이미지 로드 
        if isinstance(image, (str, Path)):
            image = load_image(image)
            if image is None:
                logger.error("이미지 로드 실패")
                return []
        
        original = image.copy()
        logger.info(f"티켓 검출 시작 (이미지 : {image.shape[:2]})")
        
        # 전처리 
        preprocessed = self.preprocessor.preprocess(image)
        
        # 윤곽선 검출 
        contours = self._find_contours(preprocessed)
        logger.debug(f"윤곽선 검출 완료 (개수: {len(contours)})")
        
        # 카드 형태 필터링
        card_contours = self._filter_card_contours(contours)
        logger.info(f"카드 후보: {len(card_contours)}")
        
        # 티켓 추출 
        tickets = []
        for i, contour in enumerate(card_contours):
            try:
                ticket_img = self.transformer.transform(original, contour)
                
                confidence = self._calculate_confidence(contour)
                
                tickets.append((ticket_img, confidence))
                logger.debug(f"티켓 {i} 추출 완료 (신뢰도: {confidence:.2%})")
            
            except Exception as e:
                logger.warning(f"티켓 {i} 추출 실패: {e}")
                continue
        
        tickets.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"티켓 검출 완료: {len(tickets)}개")
        
        return tickets
    
    
    def _find_contours(self, binary: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return sorted(contours, key=cv2.contourArea, reverse=True)
    

    def _filter_card_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        card_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area < area < self.max_area):
                continue
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_epsilon * peri, True)
            
            if len(approx) != 4:
                continue
            
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h if h != 0 else 0
            
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            card_contours.append(approx)
            
        return card_contours
    
    def _calculate_confidence(self, contour: np.ndarray) -> float:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0
        
        area_score = min(area / self.max_area, 1.0)
        
        ideal_ratio = 0.65
        ratio_diff = abs(aspect_ratio - ideal_ratio)
        ratio_score = max(0, 1.0 - ratio_diff)
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        confidence = (
            area_score * 0.4 +
            ratio_score * 0.3 +
            solidity * 0.3
        )
        
        return float(confidence)