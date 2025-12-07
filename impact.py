"""
가치입다 - 환경 임팩트 JSON 생성 (최종 개선 버전)

개선:
- 39회 호출 → 1회 호출
- JsonOutputParser 사용
- 필수 정보만

실행: python generate_impacts_final.py
출력: data/generated/impact_categories.json
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from Model.impact_rag import ImpactRAGImproved
from config import get_logger

logger = get_logger(__name__)


def generate_all_impacts():
    logger.info("가치입다 - 환경 임팩트 데이터 생성")
    
    # RAG 초기화
    logger.info("[1단계] RAG 시스템 초기화...")
    rag = ImpactRAGImproved()
    
    if not rag.rag_enabled:
        logger.error("✗ RAG 비활성화 - data/papers/에 논문 PDF 필요")
        return None
    
    logger.info("✓ RAG 준비 완료")
    logger.info("\n[2단계] LLM 1회 호출로 전체 카테고리 생성...")
    
    try:
        categories_data = rag.generate_all_categories()
        logger.info(f"✓ 생성 완료: {len(categories_data)}개")
        
    except Exception as e:
        logger.error(f"✗ 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # JSON 저장
    logger.info("\n[3단계] JSON 저장...")
    
    output_dir = project_root / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "impact_categories.json"
    
    final_data = {
        "generated_at": datetime.now().isoformat(),
        "total_categories": len(categories_data),
        "categories": categories_data
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ 저장: {output_file}")
    
    return final_data


if __name__ == "__main__":
    try:
        result = generate_all_impacts()
        
        if result:
            print(f"\n🎉 JSON 생성 완료!")
            print(f"📊 총 {result['total_categories']}개")
            print(f"📁 data/generated/impact_categories.json")
            print("\n📄 미리보기 (첫 3개):")
            print(json.dumps(result['categories'][:3], indent=2, ensure_ascii=False))
        else:
            print("\n❌ 생성 실패")
        
    except KeyboardInterrupt:
        logger.info("\n중단")
    except Exception as e:
        logger.error(f"\n오류: {e}")