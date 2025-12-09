"""
티켓 검출기 모듈

OpenCV 기반 티켓 검출 및 추출     
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional

from config import settings, get_logger, load_image
from modules.detection.preprocessor import Preprocessor
from modules.detection.transformer import PrespectiveTransformer

logger = get_logger(__name__)

class TicketDetector:
    def __init__(self):
        self.config = settings.DETECTOR_CONFIG
        self.preprocessor = Preprocessor()
        self.transformer = PrespectiveTransformer(margin=self.config['margin'])
        
        self.base_min_area = self.config['min_area']
        self.base_max_area = self.config['max_area']
        self.aspect_ratio_range = (0.3, 0.8)
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
        
        # 동적 면적 범위
        self.min_area = int(img_area * 0.015)
        self.max_area = int(img_area * 0.95)
        
        logger.info(f"티켓 검출 시작 (이미지: {img_w}x{img_h}, 면적: {img_area:,})")
        logger.info(f"동적 면적 범위: {self.min_area:,} ~ {self.max_area:,}")
        
        # 1. 흰색 객체 검출 (최우선)
        contours_white = self._detect_white_tags(image)
        logger.info(f"흰색 태그 검출: {len(contours_white)}개")
        
        # 2. 다른 방법들
        contours_otsu = self._detect_with_otsu(image)
        logger.info(f"Otsu 검출: {len(contours_otsu)}개")
        
        # 모든 윤곽선 합치기
        all_contours = contours_white + contours_otsu
        logger.info(f"총 {len(all_contours)}개 윤곽선 발견")
        
        # 필터링 및 중복 제거
        card_contours = self._filter_and_deduplicate(all_contours, image.shape)
        
        if not card_contours:
            logger.warning("카드를 찾을 수 없습니다")
            self._log_detection_tips()
            return []
        
        logger.info(f"최종 카드 후보: {len(card_contours)}개")
        
        # 디버그 이미지 저장
        self._save_debug_image(original, card_contours)
        
        # 티켓 추출
        tickets = []
        for i, contour in enumerate(card_contours):
            try:
                ticket_img = self.transformer.transform(original, contour)
                confidence = self._calculate_confidence(contour, img_area)
                tickets.append((ticket_img, confidence))
                logger.info(f"티켓 {i+1} 추출 완료 (신뢰도: {confidence:.2%})")
            except Exception as e:
                logger.warning(f"티켓 {i+1} 추출 실패: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        tickets.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"티켓 검출 완료: {len(tickets)}개")
        
        return tickets
    
    def _detect_white_tags(self, image: np.ndarray) -> List[np.ndarray]:
        """흰색 태그 검출"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 밝은 흰색 영역
        lower = np.array([0, 0, 200])    # 밝기 200 이상
        upper = np.array([180, 30, 255]) # 채도 30 이하
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # 형태학적 처리
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > self.min_area]
    
    def _detect_with_otsu(self, image: np.ndarray) -> List[np.ndarray]:
        """Otsu 이진화 검출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 이미지 테두리 제외
        img_h, img_w = image.shape[:2]
        filtered = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if not (x <= 5 and y <= 5 and w >= img_w - 10 and h >= img_h - 10):
                filtered.append(c)
        
        return filtered
    
    def _filter_and_deduplicate(self, contours: List[np.ndarray], img_shape: tuple) -> List[np.ndarray]:
        """
        필터링 및 중복 제거
        4개 점 우선, 3개 점일 때만 조심스럽게 추정
        """
        img_h, img_w = img_shape[:2]
        img_area = img_h * img_w
        candidates = []
        
        logger.info(f"\n=== 윤곽선 필터링 시작 ===")
        logger.info(f"설정 - 면적: {self.min_area:,}~{self.max_area:,}, 종횡비: {self.aspect_ratio_range}")
        
        # 이미지 테두리 마진 설정
        margin_x = int(img_w * 0.05)  # 좌우 5%
        margin_y = int(img_h * 0.05)  # 상하 5%
        
        logger.info(f"이미지 테두리 마진: x={margin_x}, y={margin_y}")
        
        for idx, contour in enumerate(contours[:15]):  # 상위 15개만 체크
            area = cv2.contourArea(contour)
            
            logger.info(f"윤곽선 #{idx} ")
            logger.info(f"  면적: {area:.0f} (범위: {'o' if self.min_area < area < self.max_area else 'x'})")
            
            # 1. 면적 체크
            if area < self.min_area or area > self.max_area:
                logger.info(f"면적 제외")
                continue
            
            peri = cv2.arcLength(contour, True)
            approx = None
            found_4_points = False
            
            # 4개 점 찾기 시도 
            for eps in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06]:
                temp = cv2.approxPolyDP(contour, eps * peri, True)
                if len(temp) == 4:
                    approx = temp
                    found_4_points = True
                    logger.info(f"  꼭지점 4개 찾음 (eps={eps})")
                    break
            
            # 4개 못 찾았으면 3~5개 점으로 시도
            if not found_4_points:
                logger.info(f"  ⚠️ 4개 점 못 찾음, 3~5개 점으로 추정 시도...")
                
                found_alternative = False
                # 더 넓은 epsilon 범위로 3~5개 점 찾기
                for eps in [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
                    temp = cv2.approxPolyDP(contour, eps * peri, True)
                    vertex_count = len(temp)
                    
                    logger.info(f"eps={eps}: {vertex_count}개 점")
                    
                    # 3~5개 점이면 시도
                    if 3 <= vertex_count <= 5:
                        logger.info(f"{vertex_count}개 점 발견! (eps={eps})")
                        found_alternative = True
                        
                        # 면적 체크
                        temp_area = cv2.contourArea(temp)
                        logger.info(f"면적: {temp_area:.0f} (최소: {self.min_area * 0.5:.0f})")
                        
                        if temp_area < self.min_area * 0.5:
                            logger.info(f"면적 너무 작음")
                            continue
                        
                        # 점 좌표 출력
                        for i, pt in enumerate(temp):
                            logger.info(f"점{i}: ({pt[0][0]:.0f}, {pt[0][1]:.0f})")
                        
                        # 4개면 그대로 사용, 3~5개면 3개 선택해서 추정
                        if vertex_count == 4:
                            approx = temp
                            logger.info(f"4개 점 그대로 사용!")
                            break
                        elif vertex_count == 3:
                            estimated = self._estimate_fourth_point(temp, img_shape)
                        else:  # 5개
                            # 5개 중 가장 먼 3개 선택
                            estimated = self._estimate_from_multiple_points(temp, img_shape)
                        
                        if estimated is not None:
                            approx = estimated
                            logger.info(f"4번째 점 추정 성공!")
                            break
                        else:
                            logger.info(f"추정 실패, 다음 eps 시도")
                
                if not found_alternative:
                    logger.info(f"3~5개 점도 찾지 못함")
            
            if approx is None or len(approx) != 4:
                logger.info(f"유효한 4개 꼭짓점 없음")
                continue
            
            if not found_4_points and found_alternative:
                # 추정으로 만든 경우는 테두리 체크 스킵
                logger.info(f"  ⚙️ 추정으로 생성 → 테두리 체크 완화")
                x, y, w, h = cv2.boundingRect(approx)
                logger.info(f"  위치: x={x}, y={y}, w={w}, h={h}")
            else:
                # 2. 이미지 테두리 체크 (원본 4개 점만)
                x, y, w, h = cv2.boundingRect(approx)
                
                too_close_to_edge = (
                    x < margin_x or
                    y < margin_y or
                    (x + w) > (img_w - margin_x) or
                    (y + h) > (img_h - margin_y)
                )
                
                logger.info(f"  위치: x={x}, y={y}, w={w}, h={h}")
                logger.info(f"  테두리 근접: {'✗ (너무 가까움)' if too_close_to_edge else '✓'}")
                
                if too_close_to_edge:
                    logger.info(f"테두리 근접, 3개 안쪽 점으로 구출 시도...")
                    
                    # 4개 점 중 안쪽에 있는 3개 찾기
                    pts_4 = approx.reshape(4, 2)
                    inner_pts = []
                    
                    for pt in pts_4:
                        px, py = pt
                        # 테두리에서 충분히 떨어진 점만 선택
                        if (margin_x <= px < img_w - margin_x and 
                            margin_y <= py < img_h - margin_y):
                            inner_pts.append(pt)
                    
                    logger.info(f"  안쪽 점: {len(inner_pts)}개 발견")
                    
                    if len(inner_pts) == 3:
                        # 3개 안쪽 점으로 4번째 점 추정
                        inner_3pts = np.array(inner_pts).reshape(3, 1, 2)
                        estimated = self._estimate_fourth_point(inner_3pts, img_shape)
                        
                        if estimated is not None:
                            approx = estimated
                            logger.info(f"테두리 근접 상황에서 구출 성공!")
                            # 다시 테두리 체크
                            x, y, w, h = cv2.boundingRect(approx)
                            too_close_to_edge = False 
                        else:
                            logger.info(f"구출 실패")
                            continue
                    else:
                        logger.info(f"안쪽 점 {len(inner_pts)}개로는 구출 불가 (3개 필요)")
                        continue
                
                # 테두리 근접 최종 체크
                if too_close_to_edge:
                    logger.info(f"이미지 테두리에 너무 가까움")
                    continue
            
            # 3. 종횡비 체크 (양방향)
            ar1 = w / h if h != 0 else 0
            ar2 = h / w if w != 0 else 0
            
            logger.info(f"  종횡비: {ar1:.2f} (w={w}, h={h})")
            
            in_range = (
                (self.aspect_ratio_range[0] <= ar1 <= self.aspect_ratio_range[1]) or
                (self.aspect_ratio_range[0] <= ar2 <= self.aspect_ratio_range[1])
            )
            
            logger.info(f"  종횡비 범위: {'✓' if in_range else '✗'}")
            
            if not in_range:
                logger.info(f"종횡비 제외 (태그는 0.3~0.8)")
                continue
            
            # 4. 볼록도 체크
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            logger.info(f"  볼록도: {solidity:.2f}")
            
            if solidity < 0.6:
                logger.info(f"볼록도 부족 (0.6 이상 필요)")
                continue
            
            # 5. 밝기 체크
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, 255, -1)
            brightness = cv2.mean(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY), mask=mask)[0]
            
            logger.info(f"  밝기: {brightness:.0f}")
            
            # 추정으로 만든 경우 밝기 기준 완화
            if not found_4_points and found_alternative:
                min_brightness = 140  # 140 이상이면 OK
                logger.info(f"  밝기 기준 (추정): {min_brightness} 이상")
            else:
                min_brightness = 180  # 원본은 180 이상
                logger.info(f"  밝기 기준 (원본): {min_brightness} 이상")
            
            if brightness < min_brightness:
                logger.info(f"밝기 부족")
                continue
            
            logger.info(f"후보 선정!")
            candidates.append(approx)
        
        # 중복 제거
        unique = []
        for c in candidates:
            is_dup = False
            for u in unique:
                if self._calculate_iou(c, u) > 0.5:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
        
        logger.info(f"\n중복 제거: {len(candidates)} → {len(unique)}개")
        return unique
    
    def _estimate_from_multiple_points(
        self,
        points: np.ndarray,
        img_shape: tuple
    ) -> Optional[np.ndarray]:
        """
        5개 이상의 점에서 가장 적절한 3개를 선택해서 4번째 점 추정
        """
        pts = points.reshape(-1, 2).astype(np.float32)
        logger.info(f"    [5개 점에서 3개 선택]")
        
        # 전략: 가장 코너에 있는 3개 점 선택
        # 1. 좌상단에 가장 가까운 점
        # 2. 우하단에 가장 가까운 점
        # 3. 나머지 중 가장 먼 점
        
        img_h, img_w = img_shape[:2]
        
        # 좌상단 거리
        tl_dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        tl_idx = np.argmin(tl_dists)
        
        # 우하단 거리
        br_dists = np.sqrt((pts[:, 0] - img_w)**2 + (pts[:, 1] - img_h)**2)
        br_idx = np.argmin(br_dists)
        
        # 나머지 점들 중에서 선택
        remaining_indices = [i for i in range(len(pts)) if i not in [tl_idx, br_idx]]
        
        if len(remaining_indices) == 0:
            return None
        
        # 나머지 중 TL, BR과 가장 먼 점
        max_dist = -1
        third_idx = remaining_indices[0]
        for idx in remaining_indices:
            dist = min(
                np.linalg.norm(pts[idx] - pts[tl_idx]),
                np.linalg.norm(pts[idx] - pts[br_idx])
            )
            if dist > max_dist:
                max_dist = dist
                third_idx = idx
        
        # 3개 점 선택
        selected_indices = [tl_idx, br_idx, third_idx]
        three_pts = pts[selected_indices].reshape(3, 1, 2)
        
        logger.info(f"선택된 3개 점:")
        for i, idx in enumerate(selected_indices):
            logger.info(f"점{i} (원본#{idx}): ({pts[idx][0]:.0f}, {pts[idx][1]:.0f})")
        
        # 4번째 점 추정
        return self._estimate_fourth_point(three_pts, img_shape)
    
    def _estimate_fourth_point(
        self, 
        three_points: np.ndarray, 
        img_shape: tuple
    ) -> Optional[np.ndarray]:
        """
        3개 점으로 4번째 점 추정
        
        원리: 평행사변형 대각선 중점이 같음
        TL + BR = TR + BL
        """
        pts = three_points.reshape(3, 2).astype(np.float32)
        img_h, img_w = img_shape[:2]
        
        logger.info(f"[4번째 점 추정 시작]")
        logger.info(f"입력 3점:")
        for i, p in enumerate(pts):
            logger.info(f"      점{i}: ({p[0]:.0f}, {p[1]:.0f})")
        
        # 3개 점을 좌표합 기준으로 정렬
        sums = pts.sum(axis=1)
        sorted_indices = np.argsort(sums)
        p0, p1, p2 = pts[sorted_indices]
        
        logger.info(f"    정렬된 점:")
        logger.info(f"      p0: ({p0[0]:.0f}, {p0[1]:.0f})")
        logger.info(f"      p1: ({p1[0]:.0f}, {p1[1]:.0f})")
        logger.info(f"      p2: ({p2[0]:.0f}, {p2[1]:.0f})")
        
        # 3가지 경우로 4번째 점 계산
        candidates = []
        
        # 경우1: p3 = p0 + p2 - p1
        p3_case1 = p0 + p2 - p1
        # 경우2: p3 = p0 + p1 - p2
        p3_case2 = p0 + p1 - p2
        # 경우3: p3 = p1 + p2 - p0
        p3_case3 = p1 + p2 - p0
        
        logger.info(f"    추정 후보:")
        
        for i, (p3, case_name) in enumerate([
            (p3_case1, "case1"), 
            (p3_case2, "case2"), 
            (p3_case3, "case3")
        ]):
            logger.info(f"      {case_name}: ({p3[0]:.0f}, {p3[1]:.0f})")
            
            # 이미지 범위 체크 (여유 넉넉하게)
            margin = 50  # 50px까지 허용
            in_bounds = (-margin <= p3[0] < img_w + margin and -margin <= p3[1] < img_h + margin)
            
            logger.info(f"        범위 체크: {'o' if in_bounds else 'x'}")
            
            if not in_bounds:
                continue
            
            # 범위 내로 클리핑
            p3_clipped = np.array([
                max(0, min(img_w - 1, p3[0])),
                max(0, min(img_h - 1, p3[1]))
            ])
            
            logger.info(f"        클리핑: ({p3_clipped[0]:.0f}, {p3_clipped[1]:.0f})")
            
            # 4개 점 생성
            four_pts = np.vstack([p0, p1, p2, p3_clipped.reshape(1, 2)])
            
            # 기하학적 타당성 검증 
            is_valid = self._is_valid_quadrilateral(four_pts, img_shape)
            logger.info(f"        사각형 타당성: {'o' if is_valid else 'x'}")
            
            if not is_valid:
                continue
            
            # 종횡비 체크 
            x, y, w, h = cv2.boundingRect(four_pts.astype(np.int32))
            if w == 0 or h == 0:
                logger.info(f"종횡비 체크: x (w={w}, h={h})")
                continue
                
            ar = w / h
            ar_inv = h / w
            
            # 종횡비 범위 확대 (0.25 ~ 1.2)
            wider_range = (0.25, 1.2)
            
            ar_ok = ((wider_range[0] <= ar <= wider_range[1]) or
                     (wider_range[0] <= ar_inv <= wider_range[1]))
            
            logger.info(f"종횡비: {ar:.2f} (역: {ar_inv:.2f}) {'o' if ar_ok else 'x'}")
            
            if ar_ok:
                score = self._calculate_quadrilateral_score(four_pts, img_shape)
                candidates.append((four_pts, score, case_name, p3_clipped))
                logger.info(f"점수: {score:.3f} ")
        
        logger.info(f"총 {len(candidates)}개 후보")
        
        if not candidates:
            logger.info(f"유효한 추정 불가")
            return None
        
        # 점수가 가장 높은 것 선택
        best_pts, best_score, best_case, best_p3 = max(candidates, key=lambda x: x[1])
        logger.info(f"최종 선택: {best_case}, 추정점=({best_p3[0]:.0f},{best_p3[1]:.0f}), 점수={best_score:.3f}")
        
        # 4개 점을 정렬해서 반환
        return best_pts.reshape(4, 1, 2).astype(np.int32)
    
    def _is_valid_quadrilateral(self, pts: np.ndarray, img_shape: tuple) -> bool:
        """4개 점이 유효한 사각형인지 검증"""
        # 모든 점이 서로 충분히 떨어져 있는지
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < 5:  
                    return False
        
        # 볼록 사각형인지
        hull = cv2.convexHull(pts.astype(np.int32))
        if len(hull) != 4:
            return False
        
        return True
    
    def _calculate_quadrilateral_score(self, pts: np.ndarray, img_shape: tuple) -> float:
        """사각형 품질 점수"""
        # 1. 면적 비율 점수
        area = cv2.contourArea(pts.astype(np.int32))
        img_area = img_shape[0] * img_shape[1]
        area_ratio = area / img_area
        
        if 0.05 <= area_ratio <= 0.6:
            area_score = 1.0
        elif area_ratio < 0.05:
            area_score = area_ratio / 0.05
        else:
            area_score = max(0, 1.0 - (area_ratio - 0.6) / 0.4)
        
        # 2. 직사각형 근접도
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        # 3. 볼록도
        hull = cv2.convexHull(pts.astype(np.int32))
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 종합 점수
        score = area_score * 0.4 + rectangularity * 0.3 + solidity * 0.3
        return score
    
    def _calculate_iou(self, c1: np.ndarray, c2: np.ndarray) -> float:
        """IoU 계산"""
        x1, y1, w1, h1 = cv2.boundingRect(c1)
        x2, y2, w2, h2 = cv2.boundingRect(c2)
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_confidence(self, contour: np.ndarray, img_area: float) -> float:
        """신뢰도 계산"""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        ar = w / h if h != 0 else 0
        
        # 면적 점수
        area_ratio = area / img_area
        if 0.05 <= area_ratio <= 0.6:
            area_score = 1.0
        elif area_ratio < 0.05:
            area_score = area_ratio / 0.05
        else:
            area_score = max(0, 1.0 - (area_ratio - 0.6) / 0.4)
        
        # 종횡비 점수
        ideal_ratios = [0.4, 0.5, 0.6, 0.7]
        ratio_score = max(0, 1.0 - min(abs(ar - r) for r in ideal_ratios))
        
        # 볼록도 점수
        hull = cv2.convexHull(contour)
        solidity = area / cv2.contourArea(hull)
        
        return float(area_score * 0.4 + ratio_score * 0.3 + solidity * 0.3)
    
    def _save_debug_image(self, image: np.ndarray, contours: List[np.ndarray]):
        """디버그 이미지 저장"""
        debug_img = image.copy()
        
        for i, contour in enumerate(contours):
            # 윤곽선 그리기
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 3)
            
            # 꼭짓점 표시
            for point in contour:
                cv2.circle(debug_img, tuple(point[0]), 8, (0, 0, 255), -1)
            
            # 번호 표시
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(debug_img, f"#{i+1}", (cx, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        debug_dir = settings.OUTPUT_DIR / "debug"
        debug_dir.mkdir(exist_ok=True)
        debug_path = debug_dir / "detected_contours.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        logger.info(f"✓ 디버그 이미지 저장: {debug_path}")
    
    def _log_detection_tips(self):
        """검출 실패 시 도움말"""
        logger.info("확인 사항:")
        logger.info("  • 이미지에 카드가 명확하게 보이는지")
        logger.info("  • 카드와 배경의 대비가 충분한지")
        logger.info("  • 카드 크기가 적절한지")
        logger.info("  • 조명이 고르게 분포되어 있는지")