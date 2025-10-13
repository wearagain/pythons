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
        self.base_min_area = self.config['min_area']
        self.base_max_area = self.config['max_area']
        self.aspect_ratio_range = (
            self.config['aspect_ratio_min'],
            self.config['aspect_ratio_max']
        )
        self.approx_epsilon = self.config['approx_epsilon']
        
        logger.info(
            f"티켓 검출기 초기화 완료 "
            f"(기본 면적: {self.base_min_area}~{self.base_max_area}, "
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
        self.current_image = original
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w
        
        self.min_area = int(img_area * 0.015)  # 1.5%
        self.max_area = int(img_area * 0.95)   # 95%
        
        logger.info(f"티켓 검출 시작 (이미지: {img_w}x{img_h}, 면적: {img_area:,})")
        logger.info(f"동적 면적 범위: {self.min_area:,} ~ {self.max_area:,}")
        
        all_contours = []
        
        logger.debug("엣지 기반 검출")
        try:
            preprocessed_edge = self.preprocessor.preprocess(image)
            contours_edge = self._find_contours(preprocessed_edge)
            logger.debug(f"엣지 방식: {len(contours_edge)}개 윤곽선")
            all_contours.extend(contours_edge)
        except Exception as e:
            logger.warning(f"엣지 전처리 실패: {e}")
            
            
        logger.debug("적응형 임계값")
        try:
            preprocessed_adaptive = self.preprocessor.preprocess_adaptive(image)
            contours_adaptive = self._find_contours(preprocessed_adaptive)
            logger.debug(f"적응형 방식: {len(contours_adaptive)}개 윤곽선")
            all_contours.extend(contours_adaptive)
        except Exception as e:
            logger.warning(f"적응형 전처리 실패: {e}")
        
        logger.debug("Otsu 이진화")
        try:
            preprocessed_otsu = self.preprocessor.preprocess_otsu(image)
            contours_otsu = self._find_contours(preprocessed_otsu)
            logger.debug(f"Otsu 방식: {len(contours_otsu)}개 윤곽선")
            all_contours.extend(contours_otsu)
        except Exception as e:
            logger.warning(f"Otsu 전처리 실패: {e}")
        
        logger.info(f"총 {len(all_contours)}개 윤곽선 발견")
        
        card_contours = self._filter_and_deduplicate(all_contours, image.shape)
        
        if len(card_contours) == 0:
            logger.warning("카드 형태를 찾을 수 없습니다")
            self._log_detection_tips()
            return []
        
        logger.info(f"카드 후보: {len(card_contours)}")
        
        debug_img = original.copy()
        for i, contour in enumerate(card_contours):
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 5)
            # 꼭지점 표시
            for point in contour:
                cv2.circle(debug_img, tuple(point[0]), 10, (0, 0, 255), -1)
        
        debug_dir = settings.OUTPUT_DIR / "debug"
        debug_dir.mkdir(exist_ok=True)
        debug_path = debug_dir / "detected_contours.jpg"
        # 화면에 맞게 리사이즈
        h, w = debug_img.shape[:2]
        if w > 1920 or h > 1080:
            scale = min(1920/w, 1080/h)
            new_w, new_h = int(w*scale), int(h*scale)
            debug_img_resized = cv2.resize(debug_img, (new_w, new_h))
            cv2.imwrite(str(debug_path), debug_img_resized)
        else:
            cv2.imwrite(str(debug_path), debug_img)
        logger.info(f"✓ 디버그 이미지 저장: {debug_path}")
        
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
        img_h, img_w = binary.shape[:2]
        filtered_contours = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 이미지 테두리와 거의 같은 크기면 제외
            if (x <= 5 and y <= 5 and 
                w >= img_w - 10 and h >= img_h - 10):
                logger.debug(f"이미지 테두리 윤곽선 제외: {x},{y},{w},{h}")
                continue
            
            filtered_contours.append(contour)
        
        logger.debug(f"윤곽선 필터링: {len(contours)} → {len(filtered_contours)}개")
        return sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:20]
    

    def _filter_card_contours(self, contours: List[np.ndarray], img_shape: tuple) -> List[np.ndarray]:
        """카드 형태 필터링"""
        card_contours = []
        img_area = img_shape[0] * img_shape[1]
        
        logger.info(f"필터링 시작: {len(contours)}개 윤곽선, 이미지 면적: {img_area:,}")
        logger.info(f"설정 - 면적: {self.min_area:,}~{self.max_area:,}, 종횡비: {self.aspect_ratio_range}")
        
        # 상위 10개 윤곽선 상세 분석
        for idx, contour in enumerate(contours[:10]):
            area = cv2.contourArea(contour)
            peri = cv2.arcLength(contour, True)
            
            # 다각형 근사 시도
            approx = None
            best_epsilon = None
            for epsilon_factor in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
                temp_approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
                if len(temp_approx) == 4:
                    approx = temp_approx
                    best_epsilon = epsilon_factor
                    break
            
            # 종횡비 계산
            x, y, w, h = cv2.boundingRect(contour if approx is None else approx)
            aspect_ratio = w / h if h != 0 else 0
            
            # 볼록도 계산
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # 상세 로그
            logger.info(f"--- 윤곽선 #{idx} ---")
            logger.info(f"  면적: {area:.0f} (범위: {'✓' if self.min_area < area < self.max_area else '✗'})")
            logger.info(f"  꼭지점: {len(approx) if approx is not None else 'N/A'}개 (epsilon={best_epsilon})")
            logger.info(f"  종횡비: {aspect_ratio:.2f} (w={w}, h={h}) (범위: {'✓' if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1] else '✗'})")
            logger.info(f"  볼록도: {solidity:.2f}")
        
        # 실제 필터링
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            # 다각형 근사
            peri = cv2.arcLength(contour, True)
            approx = None
            for epsilon_factor in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
                temp_approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
                if len(temp_approx) == 4:
                    approx = temp_approx
                    break
            
            if approx is None:
                continue
            
            # 종횡비 확인
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h if h != 0 else 0
            
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            # 볼록성 체크
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.75:  # 0.75 기준 유지
                continue
            
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, -1)
            mean_brightness = cv2.mean(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY), mask=mask)[0]
            
            # 평균 밝기가 150 이상이면 흰색으로 간주
            if mean_brightness < 150:
                logger.debug(f"  윤곽선 #{idx}: 밝기 {mean_brightness:.0f} - 너무 어두움 (배경)")
                continue
            
            logger.info(f"✓ 후보 발견: 윤곽선 #{idx}, 면적={area:.0f}, 종횡비={aspect_ratio:.2f}")
            card_contours.append(approx)
            
        return card_contours
    
    def _filter_and_deduplicate(self, contours: List[np.ndarray], img_shape: tuple) -> List[np.ndarray]:
        """윤곽선 필터링 및 중복 제거 (새로 추가)"""
        # 먼저 기본 필터 적용
        filtered = self._filter_card_contours(contours, img_shape)
        
        if len(filtered) == 0:
            return []
        
        # 중복 제거: IoU 기반
        unique_contours = []
        
        for contour in filtered:
            is_duplicate = False
            
            for existing in unique_contours:
                iou = self._calculate_iou(contour, existing)
                if iou > 0.5:  # 50% 이상 겹치면 중복
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_contours.append(contour)
        
        logger.info(f"중복 제거: {len(filtered)} → {len(unique_contours)}개")
        return unique_contours
    
    def _calculate_iou(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """두 윤곽선의 IoU 계산 (새로 추가)"""
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        x2, y2, w2, h2 = cv2.boundingRect(contour2)
        
        # 교집합
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 합집합
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_confidence(self, contour: np.ndarray, img_shape: tuple = None) -> float:
        """신뢰도 계산 (개선된 버전)"""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0
        
        # 면적 점수
        if img_shape is not None:
            img_area = img_shape[0] * img_shape[1]
            area_ratio = area / img_area
            # 이미지의 10~70%가 이상적
            if 0.1 <= area_ratio <= 0.7:
                area_score = 1.0
            elif area_ratio < 0.1:
                area_score = area_ratio / 0.1
            else:
                area_score = max(0, 1.0 - (area_ratio - 0.7) / 0.3)
        else:
            # img_shape 없으면 기본 계산
            area_score = min(area / self.max_area, 1.0)
        
        # 종횡비 점수 (0.65가 이상적)
        ideal_ratio = 0.65
        ratio_diff = abs(aspect_ratio - ideal_ratio)
        ratio_score = max(0, 1.0 - ratio_diff)
        
        # 볼록도 점수
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        confidence = (
            area_score * 0.3 +
            ratio_score * 0.3 +
            solidity * 0.4
        )
        
        return float(confidence)
    
    
    def _log_detection_tips(self):
        """검출 실패 시 도움말"""
        logger.info("확인 사항")
        logger.info("  • 이미지에 카드가 명확하게 보이는지")
        logger.info("  • 카드와 배경의 대비가 충분한지")
        logger.info("  • 카드 크기가 적절한지")
        logger.info("  • 조명이 고르게 분포되어 있는지")
        logger.info("  • config/.env 파일에서 파라미터 조정")