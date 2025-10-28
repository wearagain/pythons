"""
중앙 영역 크롭 모듈

흰색 가이드 틀 내부의 티켓 영역을 크롭합니다.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional

from config import settings, get_logger, load_image

logger = get_logger(__name__)

class CenterCropper:
    def __init__(self):
        self.config = settings.CROPPER_CONFIG
        
        # 흰색 테두리 검출 임계값
        self.white_threshold = self.config.get('white_threshold', 200)
        
        # 최소/최대 윤곽선 면적 (이미지 대비 비율)
        self.min_area_ratio = self.config.get('min_area_ratio', 0.1)  # 10%로 낮춤
        self.max_area_ratio = self.config.get('max_area_ratio', 0.95)
        
        # 윤곽선 근사화 엡실론 (값이 클수록 단순화)
        self.approx_epsilon = self.config.get('approx_epsilon', 0.05)  # 0.02 → 0.05
        
        logger.info(
            f"중앙 크롭퍼 초기화 완료 "
            f"(흰색 임계값: {self.white_threshold}, 근사화: {self.approx_epsilon})"
        )
    
    def crop(self, image: Union[np.ndarray, str, Path]) -> Optional[np.ndarray]:
        """
        흰색 가이드 틀 내부 영역을 크롭
        
        Args:
            image: 입력 이미지 (numpy array 또는 파일 경로)
            
        Returns:
            크롭된 이미지 (numpy array) 또는 None
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image = load_image(image)
            if image is None:
                logger.error("이미지 로드 실패")
                return None
        
        original = image.copy()
        height, width = original.shape[:2]
        
        logger.info(f"이미지 크롭 시작 (원본 크기: {width}x{height})")
        
        # 1단계: 전처리
        preprocessed = self._preprocess(original)
        
        # 2단계: 흰색 테두리 검출
        white_frame = self._detect_white_frame(preprocessed, width, height)
        
        if white_frame is None:
            logger.warning("흰색 가이드 틀을 찾을 수 없습니다")
            # 가이드 틀이 없으면 중앙 영역을 크롭
            return self._crop_center_fallback(original)
        
        # 3단계: 틀 내부 영역 추출
        cropped = self._extract_inner_area(original, white_frame)
        
        crop_h, crop_w = cropped.shape[:2]
        logger.info(f"크롭 완료 (크롭 후 크기: {crop_w}x{crop_h})")
        
        return cropped
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 노이즈 제거 (블러)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def _detect_white_frame(self, gray: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
        """
        흰색 가이드 틀(테두리) 검출
        
        Returns:
            윤곽선 좌표 (4개 점) 또는 None
        """
        # 흰색 영역 추출
        _, white_mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        
        # 모폴로지 연산 (구멍 메우기, 노이즈 제거)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("윤곽선을 찾을 수 없습니다")
            return None
        
        # 면적 기준으로 필터링
        min_area = width * height * self.min_area_ratio
        max_area = width * height * self.max_area_ratio
        
        valid_contours = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # 윤곽선을 사각형으로 근사화 (엡실론 값이 클수록 단순화)
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, self.approx_epsilon * peri, True)
                
                logger.debug(f"윤곽선 {i}: 면적={area:.0f}, 원본 꼭짓점={len(contour)}, 근사 꼭짓점={len(approx)}")
                
                # 4개 또는 그 근처의 꼭짓점을 가진 윤곽선 선택
                if 4 <= len(approx) <= 6:  # 4~6개 꼭짓점 허용
                    # 4개로 강제 근사화
                    if len(approx) != 4:
                        # 더 강하게 근사화
                        approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
                        logger.debug(f"  → 강제 근사화: {len(approx)}개 꼭짓점")
                    
                    if len(approx) == 4:
                        valid_contours.append((area, approx))
                        logger.debug(f"  ✓ 유효한 사각형!")
        
        if not valid_contours:
            logger.warning("사각형 윤곽선을 찾을 수 없습니다")
            return None
        
        # 가장 큰 사각형 선택 (가이드 틀)
        valid_contours.sort(reverse=True)  # 면적 기준 내림차순
        _, best_frame = valid_contours[0]
        
        logger.debug(f"가이드 틀 검출 완료 (꼭짓점 4개)")
        
        return best_frame
    
    def _extract_inner_area(self, image: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        가이드 틀 내부 영역만 추출
        
        Args:
            image: 원본 이미지
            frame: 4개 꼭짓점 좌표
            
        Returns:
            크롭된 이미지
        """
        # 꼭짓점 정렬 (좌상, 우상, 우하, 좌하 순서)
        pts = frame.reshape(4, 2).astype(np.float32)
        rect = self._order_points(pts)
        
        # 출력 이미지 크기 계산
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # 목적지 좌표
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)
        
        # 원근 변환 (perspective transform)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        logger.debug(f"원근 변환 완료: {image.shape[:2]} -> {warped.shape[:2]}")
        
        return warped
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        4개의 점을 좌상, 우상, 우하, 좌하 순서로 정렬
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 합이 가장 작은 점: 좌상단
        # 합이 가장 큰 점: 우하단
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # 차이가 가장 작은 점: 우상단
        # 차이가 가장 큰 점: 좌하단
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _crop_center_fallback(self, image: np.ndarray) -> np.ndarray:
        """
        가이드 틀을 찾지 못했을 때 중앙 영역 크롭 (폴백)
        """
        height, width = image.shape[:2]
        
        # 중앙 80% 크롭
        crop_width = int(width * 0.8)
        crop_height = int(height * 0.85)
        
        x = (width - crop_width) // 2
        y = (height - crop_height) // 2
        
        cropped = image[y:y+crop_height, x:x+crop_width]
        
        logger.info("폴백: 중앙 영역 크롭 실행")
        
        return cropped