"""
기울어진 티켓 카드를 정면으로 펴기 - 최종 완전판
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
        """원근 변환 수행"""
        # 4개 점 추출
        pts = contour.reshape(4, 2).astype(np.float32)
        
        # 점 정렬
        rect = self._order_points_robust(pts)
        
        # 목표 크기 계산
        width, height = self._calculate_output_size(rect)
        
        # 최소 크기 보장
        min_width, min_height = 500, 700
        if width < min_width or height < min_height:
            scale = max(min_width / width, min_height / height)
            width = int(width * scale)
            height = int(height * scale)
            logger.debug(f"크기 조정: {scale:.2f}배 -> {width}x{height}")
        
        # 목표 좌표
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # 변환 행렬
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # 원근 변환
        warped = cv2.warpPerspective(
            image, M, (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.debug(f"변환 완료: {image.shape[:2]} -> {warped.shape[:2]}")
        return warped
    
    def _order_points_robust(self, pts: np.ndarray) -> np.ndarray:
        """
        강력한 점 정렬 알고리즘
        여러 방법을 시도하여 가장 안정적인 결과 반환
        """
        # 방법 1: 무게중심 기반
        try:
            result = self._order_by_centroid(pts)
            if self._validate_rect(result):
                logger.debug("무게중심 기반 정렬 성공")
                return result
        except Exception as e:
            logger.debug(f"무게중심 방법 실패: {e}")
        
        # 방법 2: 합/차 기반 (기존 방법)
        try:
            result = self._order_by_sum_diff(pts)
            if self._validate_rect(result):
                logger.debug("합/차 기반 정렬 성공")
                return result
        except Exception as e:
            logger.debug(f"합/차 방법 실패: {e}")
        
        # 방법 3: 각도 기반
        try:
            result = self._order_by_angle(pts)
            if self._validate_rect(result):
                logger.debug("각도 기반 정렬 성공")
                return result
        except Exception as e:
            logger.debug(f"각도 방법 실패: {e}")
        
        # 모두 실패하면 기본 방법
        logger.warning("모든 정렬 방법 실패, 기본 정렬 사용")
        return self._order_by_sum_diff(pts)
    
    def _order_by_centroid(self, pts: np.ndarray) -> np.ndarray:
        """무게중심 기준 정렬"""
        center = pts.mean(axis=0)
        
        tl, tr, br, bl = None, None, None, None
        
        for pt in pts:
            if pt[0] < center[0] and pt[1] < center[1]:
                tl = pt
            elif pt[0] >= center[0] and pt[1] < center[1]:
                tr = pt
            elif pt[0] >= center[0] and pt[1] >= center[1]:
                br = pt
            elif pt[0] < center[0] and pt[1] >= center[1]:
                bl = pt
        
        if tl is None or tr is None or br is None or bl is None:
            raise ValueError("사분면 분류 실패")
        
        return np.array([tl, tr, br, bl], dtype=np.float32)
    
    def _order_by_sum_diff(self, pts: np.ndarray) -> np.ndarray:
        """합/차 기반 정렬"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 좌상단
        rect[2] = pts[np.argmax(s)]  # 우하단
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 우상단
        rect[3] = pts[np.argmax(diff)]  # 좌하단
        
        return rect
    
    def _order_by_angle(self, pts: np.ndarray) -> np.ndarray:
        """각도 기준 정렬"""
        center = pts.mean(axis=0)
        
        # 중심에서 각 점까지의 각도 계산
        angles = []
        for pt in pts:
            angle = np.arctan2(pt[1] - center[1], pt[0] - center[0])
            angles.append((angle, pt))
        
        # 각도순 정렬
        angles.sort()
        
        # TL, TR, BR, BL 순서로 재배치
        rect = np.zeros((4, 2), dtype=np.float32)
        for angle, pt in angles:
            deg = np.degrees(angle)
            if -135 <= deg < -45:
                rect[0] = pt  # TL
            elif -45 <= deg < 45:
                rect[1] = pt  # TR
            elif 45 <= deg < 135:
                rect[2] = pt  # BR
            else:
                rect[3] = pt  # BL
        
        return rect
    
    def _validate_rect(self, rect: np.ndarray) -> bool:
        """사각형이 올바른지 검증"""
        if rect is None or len(rect) != 4:
            return False
        
        # 모든 점이 서로 다른지 확인
        for i in range(4):
            for j in range(i + 1, 4):
                if np.allclose(rect[i], rect[j], atol=1):
                    return False
        
        # 너무 작은 사각형인지 확인
        width = max(
            np.linalg.norm(rect[1] - rect[0]),
            np.linalg.norm(rect[2] - rect[3])
        )
        height = max(
            np.linalg.norm(rect[3] - rect[0]),
            np.linalg.norm(rect[2] - rect[1])
        )
        
        if width < 10 or height < 10:
            return False
        
        return True
    
    def _calculate_output_size(self, rect: np.ndarray) -> tuple:
        """출력 크기 계산"""
        (tl, tr, br, bl) = rect
        
        # 너비 계산
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        width = max(int(widthA), int(widthB))
        
        # 높이 계산
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        height = max(int(heightA), int(heightB))
        
        return width, height