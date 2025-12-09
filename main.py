"""
로컬 테스트 실행 스크립트
"""

import sys
import cv2
from pathlib import Path

from config import settings, get_logger, is_valid_image, save_image
from modules import TicketDetector, OCREngine, CenterCropper

logger = get_logger('run_local')

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\n이미지 경로를 입력하세요:")
        image_path = input("> ").strip().strip('"\'')
    
    if not is_valid_image(image_path):
        logger.error(f"유효하지 않은 이미지입니다: {image_path}")
        logger.info("지원 포맷: .jpg, .jpeg, .png, .bmp")
        return
    
    logger.info(f"이미지: {Path(image_path).name}")
    
    image = cv2.imread(image_path)
    logger.info(f"이미지 로드 완료 (크기: {image.shape[:2]})")
    
    logger.info("[1단계] 티켓 검출 중...")

    cropper = CenterCropper()
    cropped_image = cropper.crop(image)

    if cropped_image is None:
        logger.error("크롬 실패")
        return
    
    logger.info(f"크롭 완료\n")
    
    logger.info("[2단계] 결과 저장 중...")

    output_filename = f"{Path(image_path).stem}_ticket.jpg"
    output_path = settings.CROPPED_DIR / output_filename

    if save_image(cropped_image, output_path):
        logger.info(f"저장 완료")
        logger.info(f"  - 이미지: {output_path.name}")
        logger.info(f"  - 경로: {output_path}")
    else:
        logger.error("저장 실패")
        return

    logger.info("\n[3단계] 결과 표시")
    
    show = input("결과 이미지를 표시하시겠습니까? (y/n): ").lower()
    
    if show == 'y':
        cv2.imshow("원본", image)
        cv2.imshow("크롭된 티켓", cropped_image)
        
        logger.info("\n이미지 창이 표시되었습니다.")
        logger.info("아무 키나 누르면 종료됩니다...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    logger.info("처리 완료!")
    logger.info(f"\n요약:")
    logger.info(f"  • 저장 위치: {settings.CROPPED_DIR}/")
    logger.info(f"  • 파일명: {output_filename}")
    
    """
    detector = TicketDetector()
    tickets = detector.detect(image)
    
    if not tickets:
        logger.warning("\n 티켓을 찾을 수 없습니다.")
        logger.info("\n확인 사항:")
        logger.info("  • 이미지에 흰색 카드가 명확하게 보이는지")
        logger.info("  • 카드 크기가 적절한지 (너무 작거나 크지 않은지)")
        logger.info("  • 조명이 적절한지")
        logger.info("  • 배경과 카드의 대비가 충분한지")
        return
    
    logger.info(f"✓ {len(tickets)}개 티켓 검출 완료\n")
    
    logger.info("[2단계] OCR 텍스트 인식 중...")

    
    ocr_engine = OCREngine()
    results = []
    
    for i, (ticket_img, confidence) in enumerate(tickets, 1):
        logger.info(f"티켓 {i} (검출 신뢰도: {confidence:.2%})")
        
        
        ocr_result = ocr_engine.extract_text(ticket_img)
        
        logger.info(f"OCR 신뢰도: {ocr_result['confidence']:.2%}")
        logger.info(f"추출된 줄 수: {ocr_result['line_count']}줄")

        if ocr_result['text']:
            logger.info("\n📝 추출된 텍스트:")
            for line in ocr_result['text'].split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        else:
            logger.warning("  (텍스트를 추출할 수 없습니다)")
        
        results.append({
            'ticket_image': ticket_img,
            'detection_confidence': confidence,
            'ocr_result': ocr_result
        })

    logger.info("[3단계] 결과 저장 중...")
    
    for i, result in enumerate(results, 1):
        output_filename = f"{Path(image_path).stem}_ticket_{i}.jpg"
        output_path = settings.DETECTED_DIR / output_filename
        save_image(result['ticket_image'], output_path)
        
        txt_filename = f"{Path(image_path).stem}_ticket_{i}.txt"
        txt_path = settings.DETECTED_DIR / txt_filename
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 티켓 {i} OCR 결과 ===\n\n")
            f.write(f"검출 신뢰도: {result['detection_confidence']:.2%}\n")
            f.write(f"OCR 신뢰도: {result['ocr_result']['confidence']:.2%}\n")
            f.write(f"추출된 줄 수: {result['ocr_result']['line_count']}줄\n\n")
            
            f.write(f"전체 텍스트:\n")
            f.write(result['ocr_result']['text'])
        
        logger.info(f"✓ 티켓 {i} 저장 완료")
        logger.info(f"  - 이미지: {output_path.name}")
        logger.info(f"  - 텍스트: {txt_path.name}")
    
    logger.info(f"\n 저장 위치: {settings.DETECTED_DIR}/")
    
    logger.info("[4단계] 결과 표시")
    
    show = input("결과 이미지를 표시하시겠습니까? (y/n): ").lower()
    
    if show == 'y':
        for i, result in enumerate(results, 1):
            window_name = f"티켓 {i} (신뢰도: {result['detection_confidence']:.2%})"
            cv2.imshow(window_name, result['ticket_image'])
        
        logger.info("\n 이미지 창이 표시되었습니다.")
        logger.info("아무 키나 누르면 종료됩니다...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    logger.info(" 처리 완료!")
    logger.info(f"\n 요약:")
    logger.info(f"  • 검출된 티켓: {len(tickets)}개")
    logger.info(f"  • 저장 위치: {settings.DETECTED_DIR}/")
    """





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()