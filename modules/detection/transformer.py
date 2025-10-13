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
        
    def transform(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        pts = contour.reshape(4, 2).astype(np.float32)
        
        logger.debug(f"입력 좌표: {pts}")
        
        rect = self._order_points(pts)
        logger.debug(f"정렬된 좌표: {rect}")
        
        width, height = self._calculate_dimensions(rect)
        logger.debug(f"계산된 크기: {width}x{height}")
        
        # 크기 검증
        if width <= 0 or height <= 0:
            logger.error(f"잘못된 크기: {width}x{height}")
            raise ValueError(f"Invalid dimensions: {width}x{height}")
        
        if width > 10000 or height > 10000:
            logger.warning(f"비정상적으로 큰 크기: {width}x{height}")
            # 최대 크기 제한
            max_dim = 5000
            if width > max_dim:
                height = int(height * (max_dim / width))
                width = max_dim
            if height > max_dim:
                width = int(width * (max_dim / height))
                height = max_dim
            logger.info(f"크기 제한 적용: {width}x{height}")
        
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        
        warped = cv2.warpPerspective(image, M, (width, height))
        
        # 결과 검증
        if warped is None or warped.size == 0:
            logger.error("변환 결과가 비어있음")
            raise ValueError("Warped image is empty")
        
        # 검은색 이미지 체크
        mean_val = np.mean(warped)
        if mean_val < 5:
            logger.warning(f"변환 결과가 거의 검은색 (평균값: {mean_val:.2f})")
        
        logger.debug(f"변환 완료: {image.shape[:2]} -> {warped.shape[:2]}, 평균 밝기: {mean_val:.2f}")
        return warped
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """4개 점을 좌상단, 우상단, 우하단, 좌하단 순으로 정렬"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 좌상단 (TL)
        rect[2] = pts[np.argmax(s)]  # 우하단 (BR)
        
        remaining_idx = [i for i in range(4) if i not in [np.argmin(s), np.argmax(s)]]
        remaining_pts = pts[remaining_idx]
        
        if remaining_pts[0][1] < remaining_pts[1][1]:
            rect[1] = remaining_pts[0]  # 우상단 (TR)
            rect[3] = remaining_pts[1]  # 좌하단 (BL)
        else:
            rect[1] = remaining_pts[1]  # 우상단 (TR)
            rect[3] = remaining_pts[0]  # 좌하단 (BL)
        
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