"""
기울어진 티켓 카드를 정면으로 펴기
TL(좌상), TR(우상), BR(우하), BL(좌하)
"""

import cv2
import numpy as np

from config import get_logger

logger = get_logger(__name__)

class PrespectiveTransformer:
    
    def __init__(self, margin: int = 10):
        self.margin = margin
        logger.debug(f"원근 변환기 초기화 (여백: {margin})")
        
    def transform(self, image:np.ndarray, contour:np.ndarray) -> np.ndarray:
        pts = contour.reshape(4, 2).astype(np.float32)
        rect = self._order_points(pts)
        
        width, height = self._calculate_dimensions(rect)
        
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        
        warped = cv2.warpPerspective(image, M, (width, height))
        
        logger.debug(f"변환 완료 : {image.shape[:2]} -> {warped.shape[:2]}")
        return warped
    
    def _order_points(self, pts:np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 좌상단
        rect[2] = pts[np.argmax(s)]  # 우하단
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 우상단
        rect[3] = pts[np.argmax(diff)]  # 좌하단
        
        return rect
    
    def _calculate_dimensions(self, rect: np.ndarray) -> tuple:
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        return maxWidth, maxHeight