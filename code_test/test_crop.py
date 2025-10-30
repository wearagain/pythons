# test_crop.py (디버깅용)
"""
크롭 테스트 및 디버깅
"""

import sys
import cv2
import numpy as np
from pathlib import Path

from config import get_logger, is_valid_image, save_image, settings

logger = get_logger('test_crop')

def test_crop(image_path):
    """크롭 테스트"""
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        logger.error("이미지 로드 실패")
        return
    
    original = image.copy()
    height, width = original.shape[:2]
    
    logger.info(f"원본 이미지 크기: {width}x{height}")
    
    # 1단계: 그레이스케일 변환
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    save_image(gray, settings.OUTPUT_DIR / "debug_1_gray.jpg")
    logger.info("✓ 1. 그레이스케일 저장")
    
    # 2단계: 이진화 (흰색 검출)
    WHITE_THRESHOLD = 200
    _, white_mask = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    save_image(white_mask, settings.OUTPUT_DIR / "debug_2_white_mask.jpg")
    logger.info(f"✓ 2. 흰색 마스크 저장 (임계값: {WHITE_THRESHOLD})")
    
    # 흰색 픽셀 비율 확인
    white_ratio = np.sum(white_mask == 255) / (width * height)
    logger.info(f"   흰색 픽셀 비율: {white_ratio:.2%}")
    
    # 3단계: 윤곽선 검출
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"✓ 3. 윤곽선 개수: {len(contours)}개")
    
    # 윤곽선 시각화
    contour_vis = original.copy()
    cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 3)
    save_image(contour_vis, settings.OUTPUT_DIR / "debug_3_contours.jpg")
    logger.info("✓ 3. 윤곽선 시각화 저장")
    
    # 4단계: 사각형 윤곽선 필터링
    min_area = width * height * 0.2
    max_area = width * height * 0.95
    
    valid_rectangles = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        logger.info(f"   윤곽선 {i}: 면적={area:.0f}, 꼭짓점={len(approx)}개")
        
        if min_area < area < max_area and len(approx) == 4:
            valid_rectangles.append((area, approx))
            logger.info(f"      ✓ 유효한 사각형!")
    
    if not valid_rectangles:
        logger.error("❌ 유효한 사각형을 찾을 수 없습니다!")
        logger.info("\n문제 해결 방법:")
        logger.info("1. debug_2_white_mask.jpg를 확인하세요")
        logger.info("2. 흰색 영역이 제대로 검출되었나요?")
        logger.info("3. 임계값을 조정해보세요 (현재: 200)")
        return
    
    # 가장 큰 사각형 선택
    valid_rectangles.sort(reverse=True)
    best_area, best_rect = valid_rectangles[0]
    
    logger.info(f"✓ 4. 최적 사각형 선택: 면적={best_area:.0f}")
    
    # 선택된 사각형 시각화
    rect_vis = original.copy()
    cv2.drawContours(rect_vis, [best_rect], -1, (0, 0, 255), 5)
    save_image(rect_vis, settings.OUTPUT_DIR / "debug_4_selected_rect.jpg")
    logger.info("✓ 4. 선택된 사각형 저장 (빨간색)")
    
    # 5단계: 꼭짓점 정렬
    pts = best_rect.reshape(4, 2).astype(np.float32)
    
    # 정렬 전 좌표 출력
    logger.info("   정렬 전 좌표:")
    for i, pt in enumerate(pts):
        logger.info(f"      점{i}: ({pt[0]:.0f}, {pt[1]:.0f})")
    
    # 좌상, 우상, 우하, 좌하 순서로 정렬
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상단
    rect[2] = pts[np.argmax(s)]  # 우하단
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상단
    rect[3] = pts[np.argmax(diff)]  # 좌하단
    
    logger.info("   정렬 후 좌표:")
    labels = ['좌상단', '우상단', '우하단', '좌하단']
    for i, (pt, label) in enumerate(zip(rect, labels)):
        logger.info(f"      {label}: ({pt[0]:.0f}, {pt[1]:.0f})")
    
    # 6단계: 원근 변환
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    logger.info(f"✓ 5. 출력 크기 계산: {maxWidth}x{maxHeight}")
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))
    
    save_image(warped, settings.OUTPUT_DIR / "debug_6_final_cropped.jpg")
    logger.info(f"✓ 6. 최종 크롭 이미지 저장!")
    
    # 원본과 크롭 비교
    logger.info(f"\n📊 결과 비교:")
    logger.info(f"   원본: {width}x{height} = {width*height:,} 픽셀")
    logger.info(f"   크롭: {maxWidth}x{maxHeight} = {maxWidth*maxHeight:,} 픽셀")
    logger.info(f"   크기 변화: {(maxWidth*maxHeight)/(width*height)*100:.1f}%")
    
    logger.info(f"\n✅ 모든 디버그 이미지가 {settings.OUTPUT_DIR}/ 에 저장되었습니다!")
    
    return warped


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("이미지 경로를 입력하세요:")
        image_path = input("> ").strip().strip('"\'')
    
    if not is_valid_image(image_path):
        print("유효하지 않은 이미지입니다")
        sys.exit(1)
    
    test_crop(image_path)