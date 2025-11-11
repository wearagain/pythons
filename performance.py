"""
태그 인식 성능 평가 스크립트 (이미지 저장 버전)

크롭된 결과를 실제로 저장해서 육안으로 확인 가능
"""

import cv2
import time
import json
from pathlib import Path
from typing import List, Dict
import sys

from config import settings, get_logger, is_valid_image, save_image
from modules import CenterCropper

logger = get_logger('performance_evaluation')


class VisualEvaluator:
    """크롭 결과를 저장하는 평가 클래스"""
    
    def __init__(self, save_cropped: bool = True):
        self.cropper = CenterCropper()
        self.results = []
        self.save_cropped = save_cropped
        
        # 결과 저장 디렉토리 생성
        self.eval_output_dir = settings.OUTPUT_DIR / "evaluation_cropped"
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"성능 평가기 초기화 완료")
        if save_cropped:
            logger.info(f"크롭 이미지 저장 위치: {self.eval_output_dir}")
    
    def evaluate_single(self, image_path: str) -> Dict:
        """
        단일 이미지 평가 + 크롭 결과 저장
        """
        filename = Path(image_path).name
        logger.info(f"평가 중: {filename}")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"이미지 로드 실패: {filename}")
            return {
                'filename': filename,
                'success': False,
                'processing_time': 0,
                'error': 'Failed to load image'
            }
        
        original_h, original_w = image.shape[:2]
        original_size = (original_w, original_h)
        
        # 로그 캡처 (폴백 및 WARNING 감지용)
        import logging
        import io
        
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        
        cropper_logger = logging.getLogger('modules.crop.cropper')
        cropper_logger.addHandler(handler)
        
        # 처리 시간 측정 + 크롭 실행
        start_time = time.time()
        cropped = self.cropper.crop(image)
        processing_time = time.time() - start_time
        
        cropper_logger.removeHandler(handler)
        log_output = log_capture.getvalue()
        
        # 폴백 사용 여부 확인 (WARNING 로그로 감지)
        has_warning = len(log_output.strip()) > 0
        used_fallback = (
            "폴백" in log_output or 
            "fallback" in log_output.lower() or
            "사각형 윤곽선을 찾을 수 없습니다" in log_output or
            "흰색 가이드 틀을 찾을 수 없습니다" in log_output or
            has_warning
        )
        
        if cropped is not None:
            crop_h, crop_w = cropped.shape[:2]
            cropped_size = (crop_w, crop_h)
            
            original_pixels = original_w * original_h
            cropped_pixels = crop_w * crop_h
            size_reduction = (1 - cropped_pixels / original_pixels) * 100
            
            # 📸 크롭된 이미지 저장
            if self.save_cropped:
                output_filename = f"{Path(image_path).stem}_cropped.jpg"
                output_path = self.eval_output_dir / output_filename
                save_image(cropped, output_path)
                logger.info(f"  💾 크롭 이미지 저장: {output_filename}")
            
            # 검출 품질 판단
            if used_fallback:
                detection_quality = 'fallback'
                success_type = 'partial'
                logger.warning(f"⚠️  주의 (폴백 사용) - {processing_time:.3f}초, 크기 감소 {size_reduction:.1f}%")
            else:
                detection_quality = 'excellent'
                success_type = 'full'
                logger.info(f"✓ 완전 성공 - {processing_time:.3f}초, 크기 감소 {size_reduction:.1f}%")
            
            result = {
                'filename': filename,
                'success': True,
                'success_type': success_type,
                'detection_quality': detection_quality,
                'processing_time': round(processing_time, 3),
                'original_size': original_size,
                'cropped_size': cropped_size,
                'size_reduction': round(size_reduction, 2),
                'saved_path': str(output_path) if self.save_cropped else None
            }
        else:
            result = {
                'filename': filename,
                'success': False,
                'detection_quality': 'failed',
                'processing_time': round(processing_time, 3),
                'original_size': original_size,
                'error': 'Cropping completely failed'
            }
            logger.error(f"✗ 완전 실패 - {processing_time:.3f}초")
        
        return result
    
    def evaluate_batch(self, image_paths: List[str]) -> Dict:
        """여러 이미지 일괄 평가"""
        logger.info(f"\n{'='*60}")
        logger.info(f"성능 평가 시작: {len(image_paths)}개 이미지")
        logger.info(f"{'='*60}\n")
        
        self.results = []
        
        for image_path in image_paths:
            result = self.evaluate_single(image_path)
            self.results.append(result)
            print()
        
        # 통계 계산
        total = len(self.results)
        full_success = sum(1 for r in self.results if r.get('success_type') == 'full')
        partial_success = sum(1 for r in self.results if r.get('success_type') == 'partial')
        failed = sum(1 for r in self.results if not r['success'])
        
        full_success_rate = (full_success / total * 100) if total > 0 else 0
        total_success_rate = ((full_success + partial_success) / total * 100) if total > 0 else 0
        
        # 처리 시간 통계
        successful_results = [r for r in self.results if r['success']]
        
        if successful_results:
            processing_times = [r['processing_time'] for r in successful_results]
            avg_time = sum(processing_times) / len(processing_times)
            min_time = min(processing_times)
            max_time = max(processing_times)
            
            size_reductions = [r['size_reduction'] for r in successful_results if 'size_reduction' in r]
            avg_reduction = sum(size_reductions) / len(size_reductions) if size_reductions else None
        else:
            avg_time = min_time = max_time = avg_reduction = 0
        
        metrics = {
            'total_images': total,
            'full_success': full_success,
            'partial_success': partial_success,
            'failed': failed,
            'full_success_rate': round(full_success_rate, 2),
            'total_success_rate': round(total_success_rate, 2),
            'avg_processing_time': round(avg_time, 3) if successful_results else None,
            'min_processing_time': round(min_time, 3) if successful_results else None,
            'max_processing_time': round(max_time, 3) if successful_results else None,
            'avg_size_reduction': round(avg_reduction, 2) if avg_reduction else None,
            'cropped_images_dir': str(self.eval_output_dir) if self.save_cropped else None,
            'detailed_results': self.results
        }
        
        return metrics
    
    def print_summary(self, metrics: Dict):
        """결과 요약 출력"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 성능 평가 결과")
        logger.info(f"{'='*60}\n")
        
        logger.info(f"🔢 전체 통계:")
        logger.info(f"  • 총 이미지 수: {metrics['total_images']}개")
        logger.info(f"  • ✅ 완전 성공: {metrics['full_success']}개 (가이드 프레임 정확 검출)")
        logger.info(f"  • ⚠️  주의사항: {metrics['partial_success']}개 (폴백 사용 - 중앙 크롭)")
        logger.info(f"  • ❌ 실패: {metrics['failed']}개")
        logger.info(f"  • 완전 성공률: {metrics['full_success_rate']:.2f}%")
        logger.info(f"  • 전체 처리율: {metrics['total_success_rate']:.2f}%")
        
        if metrics['avg_processing_time'] is not None:
            logger.info(f"⏱️  처리 시간:")
            logger.info(f"  • 평균: {metrics['avg_processing_time']:.3f}초")
            logger.info(f"  • 최소: {metrics['min_processing_time']:.3f}초")
            logger.info(f"  • 최대: {metrics['max_processing_time']:.3f}초")
        
        if metrics['avg_size_reduction'] is not None:
            logger.info(f"📐 크기 감소:")
            logger.info(f"  • 평균 감소율: {metrics['avg_size_reduction']:.2f}%")
        
        if metrics['cropped_images_dir']:
            logger.info(f"\n💾 크롭 이미지 저장:")
            logger.info(f"  • 위치: {metrics['cropped_images_dir']}")
            logger.info(f"  • 파일명 패턴: [원본파일명]_cropped.jpg")
        
        # 주의사항 케이스
        partial_cases = [r for r in metrics['detailed_results'] if r.get('success_type') == 'partial']
        if partial_cases:
            logger.info(f"\n⚠️  주의사항 케이스 (폴백으로 처리됨):")
            for case in partial_cases:
                logger.info(f"  • {case['filename']}")
                logger.info(f"    - 원인: 흰색 가이드 프레임 검출 실패")
                logger.info(f"    - 처리: 중앙 영역 크롭으로 대체")
                if case.get('saved_path'):
                    logger.info(f"    - 결과: {Path(case['saved_path']).name}")
        
        # 완전 실패 케이스
        failed_cases = [r for r in metrics['detailed_results'] if not r['success']]
        if failed_cases:
            logger.info(f"\n❌ 완전 실패 케이스:")
            for case in failed_cases:
                logger.info(f"  • {case['filename']}: {case.get('error', 'Unknown error')}")
        
        logger.info(f"\n{'='*60}\n")
    
    def print_ppt_summary(self, metrics: Dict):
        """PPT 발표용 요약"""
        print("\n" + "="*70)
        print("📊 PPT 중간발표용 핵심 지표")
        print("="*70 + "\n")
        
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│  1️⃣  검출 성공률                                              │")
        print(f"│     ✅ 완전 성공: {metrics['full_success_rate']:.1f}% ({metrics['full_success']}/{metrics['total_images']}개)                    │")
        print(f"│     ⚠️  주의사항: {(metrics['partial_success']/metrics['total_images']*100):.1f}% ({metrics['partial_success']}개 - 폴백 처리)        │")
        print(f"│     📊 전체 처리: {metrics['total_success_rate']:.1f}%                              │")
        print("└─────────────────────────────────────────────────────────────┘\n")
        
        if metrics['avg_processing_time']:
            print("┌─────────────────────────────────────────────────────────────┐")
            print("│  2️⃣  처리 속도                                                │")
            print(f"│     ⏱️  평균 {metrics['avg_processing_time']:.3f}초/이미지                         │")
            print(f"│     📈 범위: {metrics['min_processing_time']:.3f}~{metrics['max_processing_time']:.3f}초                                  │")
            print("└─────────────────────────────────────────────────────────────┘\n")
        
        if metrics['avg_size_reduction']:
            print("┌─────────────────────────────────────────────────────────────┐")
            print("│  3️⃣  이미지 최적화                                            │")
            print(f"│     📉 평균 {metrics['avg_size_reduction']:.1f}% 크기 감소                          │")
            print("└─────────────────────────────────────────────────────────────┘\n")
        
        if metrics['cropped_images_dir']:
            print("┌─────────────────────────────────────────────────────────────┐")
            print("│  💡 시각적 확인                                               │")
            print(f"│     크롭된 이미지 저장 위치:                                 │")
            print(f"│     {metrics['cropped_images_dir']:55s}│")
            print("└─────────────────────────────────────────────────────────────┘")
        
        print("\n" + "="*70 + "\n")
    
    def save_results(self, metrics: Dict, output_path: str = None):
        """결과 저장"""
        if output_path is None:
            output_path = settings.OUTPUT_DIR / "evaluation_results_visual.json"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 결과 저장 완료: {output_path}")


def main():
    """메인 실행"""
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        print("\n평가할 이미지 디렉토리 또는 이미지 경로들을 입력하세요:")
        print("(여러 개는 공백으로 구분, 디렉토리면 모든 이미지 평가)")
        user_input = input("> ").strip()
        
        if not user_input:
            logger.error("경로가 입력되지 않았습니다.")
            return
        
        inputs = user_input.split()
        image_paths = []
        
        for inp in inputs:
            inp = inp.strip('"\'')
            path = Path(inp)
            
            if path.is_dir():
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                    image_paths.extend([str(p) for p in path.glob(f'*{ext}')])
            elif path.is_file() and is_valid_image(str(path)):
                image_paths.append(str(path))
    
    valid_images = [p for p in image_paths if is_valid_image(p)]
    
    if not valid_images:
        logger.error("유효한 이미지가 없습니다.")
        return
    
    logger.info(f"\n발견된 이미지: {len(valid_images)}개")
    
    # 크롭 이미지 저장 여부 확인
    print("\n크롭된 이미지를 저장하시겠습니까? (y/n) [기본값: y]: ", end='')
    save_choice = input().strip().lower()
    save_cropped = save_choice != 'n'
    
    # 평가 실행
    evaluator = VisualEvaluator(save_cropped=save_cropped)
    metrics = evaluator.evaluate_batch(valid_images)
    
    # 결과 출력
    evaluator.print_summary(metrics)
    evaluator.print_ppt_summary(metrics)
    evaluator.save_results(metrics)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()