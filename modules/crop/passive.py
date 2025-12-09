import cv2
import numpy as np
from typing import List, Union

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_logger
logger = get_logger(__name__)

def passive_crop(image: np.ndarray, corners: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """주어진 좌표를 바로 사용"""
    
    try:
        corner = np.array(corners, dtype=np.float32)
        
        if corner.shape != (4,2):
            raise ValueError(
                f"corners는 정확히 4개의 [x, y] 좌표여야 합니다. "
            )
        
        rect = _order_points(corner)
        
        (tl, tr, br, bl) = rect
        
        # 너비 계산 (상단 vs 하단 중 큰 값)
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # 높이 계산 (좌측 vs 우측 중 큰 값)
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        logger.info(f"수동 크롭 완료: {image.shape[:2]} → {warped.shape[:2]}")
        
        return warped
    
    except ValueError as e:
        # 에러
        logger.error(f"입력 검증 실패: {e}")
        raise
    except Exception as e:
        raise RuntimeError(f"{str(e)}")
    
    
def _order_points(corner: np.ndarray) -> np.ndarray:
    """ 좌상, 우상, 우하, 좌하 순서로 정렬 """
    
    rect = np.zeros((4,2), dtype=np.float32)
    
    s = corner.sum(axis=1)
    rect[0] = corner[np.argmin(s)]  # 좌상단
    rect[2] = corner[np.argmax(s)]  # 우하단
    
    diff = np.diff(corner, axis=1)
    rect[1] = corner[np.argmin(diff)]  # 우상단
    rect[3] = corner[np.argmax(diff)]  # 좌하단
    
    return rect


if __name__ == "__main__":
    """
    테스트 실행:
        python modules/crop/passive.py
    """
    print("=" * 60)
    print("수동 크롭 모듈 테스트 - test2.jpg")
    print("=" * 60)
    
    # ===== test2.jpg 로드 =====
    image_path = Path("test") / "test.jpg"  # test2.jpg 경로
    
    if not image_path.exists():
        print(f"\n❌ 이미지를 찾을 수 없습니다: {image_path}")
        print("\n다른 경로 시도 중...")
        
        # 다른 가능한 경로들
        possible_paths = [
            Path("test/test.jpg"),
            Path("test/test2.jpg"),
            Path("tests/test.jpg"),
        ]
        
        for p in possible_paths:
            if p.exists():
                image_path = p
                print(f"✓ 발견: {image_path}")
                break
        else:
            print("\n❌ test.jpg를 찾을 수 없습니다.")
            print("아래 경로에 이미지를 저장하거나, 코드에서 경로를 수정하세요:")
            print("  - test/test.jpg")
            sys.exit(1)
    
    # 이미지 로드
    test_image = cv2.imread(str(image_path))
    
    if test_image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        sys.exit(1)
    
    height, width = test_image.shape[:2]
    print(f"\n✓ 이미지 로드 완료: {width}x{height}")
    print(f"  경로: {image_path.absolute()}")
    
    # ===== 흰색 카드 영역 좌표 (예시) =====
    # 이미지를 보고 흰색 카드의 4개 모서리 좌표를 입력하세요!
    # 
    # 💡 TIP: 
    # - 그림판으로 열어서 마우스로 좌표 확인 가능
    # - 또는 아래 좌표를 대략적으로 조정하세요
    
    # 기본값: 이미지 중앙 영역 (60%)
    margin_x = int(width * 0.2)
    margin_y = int(height * 0.2)
    
    corners = [
        [margin_x, margin_y],                    # 좌상
        [width - margin_x, margin_y],            # 우상
        [width - margin_x, height - margin_y],   # 우하
        [margin_x, height - margin_y]            # 좌하
    ]
    
    print(f"\n크롭 좌표 (기본값 - 중앙 60%):")
    print(f"  {corners}")
    print("\n💡 좌표를 직접 입력하려면 코드에서 corners 변수를 수정하세요!")
    print("   예시:")
    print("   corners = [")
    print("       [200, 150],   # 좌상")
    print("       [600, 160],   # 우상")
    print("       [590, 950],   # 우하")
    print("       [210, 940]    # 좌하")
    print("   ]")
    
    # ===== 수동 크롭 실행 =====
    print("\n" + "=" * 60)
    print("크롭 실행 중...")
    print("=" * 60)
    
    try:
        # 수동 크롭 실행
        cropped = passive_crop(test_image, corners)
        
        print(f"\n✓ 크롭 완료!")
        print(f"  - 원본 크기: {width}x{height}")
        print(f"  - 크롭 크기: {cropped.shape[1]}x{cropped.shape[0]}")
        
        # 결과 저장
        from config import settings
        output_path = settings.OUTPUT_DIR / "test_passive_crop_real.jpg"
        cv2.imwrite(str(output_path), cropped)
        print(f"\n💾 저장 완료!")
        print(f"  - 경로: {output_path.absolute()}")
        
        # 결과 표시 (선택)
        print("\n이미지를 표시하시겠습니까? (y/n): ", end="")
        show = input().strip().lower()
        
        if show in ['y', 'yes']:
            # 윈도우 크기 조정 (너무 크면)
            display_width = 800
            if width > display_width:
                scale = display_width / width
                display_image = cv2.resize(test_image, None, fx=scale, fy=scale)
            else:
                display_image = test_image
            
            cv2.imshow("원본", display_image)
            cv2.imshow("크롭 결과", cropped)
            
            print("\n이미지 창이 표시되었습니다.")
            print("아무 키나 누르면 종료됩니다...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("✅ 테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()