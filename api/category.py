import json
from pathlib import Path
from typing import Dict, Optional
import sys


# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import get_logger

logger = get_logger(__name__)


class CategoryManager:
    """카테고리 데이터 관리 클래스"""
    
    def __init__(self, json_path: Optional[str] = None):
        """
        Args:
            json_path: JSON 파일 경로 (기본값: data/generated/impact_categories.json)
        """
        if json_path is None:
            self.json_path = project_root / "data" / "generated" / "impact_categories.json"
        else:
            self.json_path = Path(json_path)
        
        self.categories = {}
        self._load_categories()
    
    def _load_categories(self):
        """JSON 파일에서 카테고리 데이터 로드"""
        try:
            if not self.json_path.exists():
                logger.error(f"카테고리 파일을 찾을 수 없습니다: {self.json_path}")
                return
            
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 카테고리 코드를 키로 하는 딕셔너리 생성
            for category in data.get('categories', []):
                code = category.get('code')
                if code:
                    self.categories[code] = category
            
            logger.info(f"카테고리 데이터 로드 완료: {len(self.categories)}개")
            
        except Exception as e:
            logger.error(f"카테고리 데이터 로드 실패: {e}")
            raise
    
    def get_category(self, code: str) -> Optional[Dict]:
        """
        카테고리 코드로 데이터 조회
        """
        category = self.categories.get(code)
        
        if category:
            logger.info(f"카테고리 조회 성공: {code} - {category['sub_category']}")
        else:
            logger.warning(f"카테고리를 찾을 수 없습니다: {code}")
        
        return category
    
    def get_all_categories(self) -> Dict[str, Dict]:
        """모든 카테고리 데이터 반환"""
        return self.categories
    
    def get_categories_by_main(self, main_category: str) -> Dict[str, Dict]:
        """
        대분류별 카테고리 조회
        """
        filtered = {
            code: cat for code, cat in self.categories.items()
            if cat.get('main_category') == main_category
        }
        
        logger.info(f"대분류 '{main_category}' 조회: {len(filtered)}개")
        return filtered


# 싱글톤 인스턴스 생성
category_manager = CategoryManager()
