# create_guided_image.py
"""
앱 가이드 틀 시뮬레이션

실제 앱에서 찍힐 것처럼 가이드 틀을 합성합니다.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

def create_app_simulation(image_path, output_path="guided_ticket.jpg"):
    """
    앱 촬영 환경 시뮬레이션
    
    Args:
        image_path: 티켓 이미지 경로
        output_path: 저장 경로
    """
    # 이미지 로드
    ticket = cv2.imread(image_path)
    if ticket is None:
        print(f"❌ 이미지를 불러올 수 없습니다: {image_path}")
        return None
    
    t_height, t_width = ticket.shape[:2]
    print(f"📸 원본 티켓 크기: {t_width}x{t_height}")
    
    # 앱 화면 크기 (일반적인 스마트폰 비율)
    app_width = 1080
    app_height = 1920
    
    # 가이드 틀 크기 (화면의 75%)
    guide_width = int(app_width * 0.75)
    guide_height = int(app_height * 0.6)
    
    # 가이드 틀 위치 (중앙)
    guide_x = (app_width - guide_width) // 2
    guide_y = (app_height - guide_height) // 2
    
    print(f"📱 앱 화면 크기: {app_width}x{app_height}")
    print(f"🔲 가이드 틀 크기: {guide_width}x{guide_height}")
    
    # 검은 배경 생성
    canvas = np.zeros((app_height, app_width, 3), dtype=np.uint8)
    
    # 티켓을 가이드 틀 크기에 맞게 리사이즈
    ticket_resized = cv2.resize(ticket, (guide_width, guide_height))
    
    # 티켓을 캔버스 중앙에 배치
    canvas[guide_y:guide_y+guide_height, guide_x:guide_x+guide_width] = ticket_resized
    
    # 가이드 틀 그리기 (흰색 테두리)
    # 외부 테두리
    cv2.rectangle(
        canvas, 
        (guide_x-5, guide_y-5), 
        (guide_x+guide_width+5, guide_y+guide_height+5),
        (255, 255, 255), 
        10
    )
    
    # 내부 테두리 (약간 안쪽)
    cv2.rectangle(
        canvas, 
        (guide_x+5, guide_y+5), 
        (guide_x+guide_width-5, guide_y+guide_height-5),
        (255, 255, 255), 
        3
    )
    
    # 모서리 마커 추가 (더 명확한 가이드)
    corner_length = 50
    corner_thickness = 8
    corners = [
        # 좌상단
        [(guide_x, guide_y), (guide_x + corner_length, guide_y)],
        [(guide_x, guide_y), (guide_x, guide_y + corner_length)],
        # 우상단
        [(guide_x + guide_width, guide_y), (guide_x + guide_width - corner_length, guide_y)],
        [(guide_x + guide_width, guide_y), (guide_x + guide_width, guide_y + corner_length)],
        # 우하단
        [(guide_x + guide_width, guide_y + guide_height), (guide_x + guide_width - corner_length, guide_y + guide_height)],
        [(guide_x + guide_width, guide_y + guide_height), (guide_x + guide_width, guide_y + guide_height - corner_length)],
        # 좌하단
        [(guide_x, guide_y + guide_height), (guide_x + corner_length, guide_y + guide_height)],
        [(guide_x, guide_y + guide_height), (guide_x, guide_y + guide_height - corner_length)],
    ]
    
    for start, end in corners:
        cv2.line(canvas, start, end, (255, 255, 255), corner_thickness)
    
    # 상단에 안내 텍스트 추가
    text = "Ticket Guide Frame"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1.2, 2)[0]
    text_x = (app_width - text_size[0]) // 2
    text_y = guide_y - 50
    
    cv2.putText(canvas, text, (text_x, text_y), font, 1.2, (255, 255, 255), 2)
    
    # 저장
    cv2.imwrite(output_path, canvas)
    print(f"✅ 저장 완료: {output_path}")
    
    return canvas


def create_multiple_variations(image_path):
    """
    다양한 상황 시뮬레이션
    """
    ticket = cv2.imread(image_path)
    if ticket is None:
        print("이미지를 불러올 수 없습니다")
        return
    
    variations = [
        # (이름, 틀 크기 비율, 회전 각도)
        ("perfect", 0.75, 0),      # 완벽한 정렬
        ("small", 0.6, 0),          # 작게 촬영
        ("large", 0.85, 0),         # 크게 촬영
        ("rotated_5", 0.75, 5),     # 5도 회전
        ("rotated_10", 0.75, -10),  # -10도 회전
    ]
    
    for name, ratio, angle in variations:
        print(f"\n=== {name} 버전 생성 중 ===")
        
        app_width = 1080
        app_height = 1920
        
        guide_width = int(app_width * ratio)
        guide_height = int(app_height * 0.6 * ratio)
        
        guide_x = (app_width - guide_width) // 2
        guide_y = (app_height - guide_height) // 2
        
        canvas = np.zeros((app_height, app_width, 3), dtype=np.uint8)
        
        # 티켓 리사이즈
        ticket_resized = cv2.resize(ticket, (guide_width, guide_height))
        
        # 회전 적용
        if angle != 0:
            center = (guide_width // 2, guide_height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            ticket_resized = cv2.warpAffine(ticket_resized, matrix, (guide_width, guide_height))
        
        canvas[guide_y:guide_y+guide_height, guide_x:guide_x+guide_width] = ticket_resized
        
        # 가이드 틀
        cv2.rectangle(canvas, (guide_x-5, guide_y-5), 
                     (guide_x+guide_width+5, guide_y+guide_height+5),
                     (255, 255, 255), 10)
        
        output_path = f"guided_{name}.jpg"
        cv2.imwrite(output_path, canvas)
        print(f"✅ 저장: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 앱 가이드 틀 시뮬레이터")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\n티켓 이미지 경로를 입력하세요:")
        image_path = input("> ").strip().strip('"\'')
    
    if not Path(image_path).exists():
        print(f"❌ 파일이 존재하지 않습니다: {image_path}")
        sys.exit(1)
    
    print("\n어떤 버전을 생성할까요?")
    print("1. 기본 버전 (1개)")
    print("2. 다양한 시나리오 버전 (5개)")
    
    choice = input("\n선택 (1 또는 2): ").strip()
    
    if choice == "2":
        create_multiple_variations(image_path)
    else:
        result = create_app_simulation(image_path)
        
        if result is not None:
            print("\n미리보기를 표시하시겠습니까? (y/n)")
            if input("> ").lower() == 'y':
                cv2.imshow("App Simulation", result)
                print("\n아무 키나 누르면 종료됩니다...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    print("\n✨ 완료! 생성된 이미지로 크롭 테스트를 진행하세요:")
    print("   python main.py guided_ticket.jpg")