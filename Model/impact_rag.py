"""
환경 임팩트 RAG 시스템
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdfplumber
import re
import json
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Model.gemini import GeminiClient
from config import get_logger

logger = get_logger(__name__)


class ImpactRAGImproved:
    """환경 임팩트 RAG 시스템"""
    
    def __init__(self):
        logger.info("RAG 시스템 초기화...")
        
        try:
            self.client = GeminiClient(use_rag=True)
            self.rag_enabled = self.client.use_rag
        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            self.client = GeminiClient(use_rag=False)
            self.rag_enabled = False
        
        if self.rag_enabled:
            self._load_papers()
        
        logger.info("RAG 초기화 완료")
    
    def _load_papers(self):
        """논문 PDF 로드"""
        if not hasattr(self.client, 'vector_db') or self.client.vector_db is None:
            return
        
        paper_dir = Path("data/papers")
        if not paper_dir.exists():
            logger.warning(f"디렉토리 없음: {paper_dir}")
            return
        
        pdf_files = list(paper_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("PDF 파일 없음")
            return
        
        for pdf_path in pdf_files:
            logger.info(f"논문 로드: {pdf_path.name}")
            
            try:
                documents = self._extract_pdf(str(pdf_path))
                if documents:
                    self.client.vector_db.add_documents(documents)
                    logger.info(f"  {len(documents)}개 청크 추가")
            except Exception as e:
                logger.error(f"실패: {e}")
    
    def _extract_pdf(self, pdf_path: str) -> List[Document]:
        """PDF 추출 (표 + 캡션)"""
        documents = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                
                # 표 → Markdown
                tables = page.extract_tables()
                if tables:
                    for idx, table in enumerate(tables, 1):
                        md = self._table_to_markdown(table)
                        if md:
                            text += f"\n\n### Table {idx}\n\n{md}\n\n"
                
                # 캡션
                captions = self._extract_captions(text)
                if captions:
                    text += "\n\n### Figures\n\n" + "\n".join(captions)
                
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num}
                    ))
        
        # 청크 분할
        if documents:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            documents = splitter.split_documents(documents)
        
        return documents
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """표 → Markdown"""
        if not table or not table[0]:
            return ""
        
        try:
            headers = table[0]
            lines = [
                "| " + " | ".join(str(h or "").strip() for h in headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |"
            ]
            
            for row in table[1:]:
                if any(cell for cell in row):
                    lines.append("| " + " | ".join(str(c or "").strip() for c in row) + " |")
            
            return "\n".join(lines)
        except:
            return ""
    
    def _extract_captions(self, text: str) -> List[str]:
        """캡션 추출"""
        captions = []
        patterns = [
            r'Figure\s+\d+[:.]\s*([^\n]+)',
            r'Table\s+\d+[:.]\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                caption = match.group(0).strip()
                if caption not in captions:
                    captions.append(caption)
        
        return captions
    
    def generate_all_categories(self) -> List[Dict]:
        """
        1회 LLM 호출로 전체 39개 카테고리 생성
        
        Returns:
            카테고리 리스트
        """
        # 프롬프트 생성
        prompt = self._create_batch_prompt()
        
        logger.info("  - 프롬프트 생성 완료")
        logger.info("  - LLM 호출 중... (약 1-2분 소요)")
        
        try:
            response = self.client.chat(
                message=prompt,
                system_prompt="당신은 환경 임팩트 전문가입니다. 논문 데이터 기반으로 정확한 수치를 제공하세요."
            )
            
            logger.info("  - LLM 응답 받음")
            logger.info(f"  - 응답 길이: {len(response)}자")
            logger.info("  - JSON 파싱 중...")
            
            categories = self._parse_json_response(response)
            
            logger.info(f"  파싱 완료: {len(categories)}개")
            
            return categories
        
        except Exception as e:
            logger.error(f"  LLM 호출 또는 파싱 실패: {e}")
            logger.error(f"  응답 내용: {response[:500] if 'response' in locals() else '없음'}")
            raise
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """
        JSON 응답 파싱
        """
        # 방법 1: Markdown 제거
        try:
            response_clean = response.strip()
            
            # ```json ... ``` 제거
            if response_clean.startswith("```"):
                # 첫 번째 ``` 이후부터
                response_clean = response_clean.split("```", 1)[1]
                # json 키워드 제거
                if response_clean.startswith("json"):
                    response_clean = response_clean[4:]
                # 마지막 ``` 제거
                if "```" in response_clean:
                    response_clean = response_clean.split("```")[0]
            
            # 공백 제거
            response_clean = response_clean.strip()
            
            # JSON 파싱
            data = json.loads(response_clean)
            
            # categories 배열 추출
            if isinstance(data, dict) and "categories" in data:
                return data["categories"]
            elif isinstance(data, list):
                return data
            else:
                raise ValueError("예상치 못한 JSON 구조")
        
        except json.JSONDecodeError as e:
            logger.warning(f"방법 1 실패: {e}")
            
            # 방법 2: 정규식으로 JSON 추출
            try:
                # { "categories": [ ... ] } 패턴 찾기
                match = re.search(r'\{\s*"categories"\s*:\s*\[.*\]\s*\}', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    data = json.loads(json_str)
                    return data["categories"]
                else:
                    raise ValueError("JSON 패턴을 찾을 수 없음")
            
            except Exception as e2:
                logger.warning(f"방법 2 실패: {e2}")
                
                # 방법 3: 불완전한 JSON 처리 (마지막 객체 제거 후 재시도)
                try:
                    # 마지막 , 이후 내용 제거
                    response_clean = response_clean.strip()
                    if response_clean.endswith(","):
                        response_clean = response_clean[:-1]
                    
                    # 불완전한 마지막 객체 제거
                    # [ {...}, {...}, { 여기서 끊김
                    last_complete = response_clean.rfind("},")
                    if last_complete > 0:
                        response_clean = response_clean[:last_complete+1] + "]}"
                    
                    data = json.loads(response_clean)
                    
                    if isinstance(data, dict) and "categories" in data:
                        logger.warning(f"방법 3 성공 (일부 데이터 손실 가능)")
                        return data["categories"]
                
                except Exception as e3:
                    logger.error(f"방법 3 실패: {e3}")
                    logger.error(f"전체 응답:\n{response}")
                    raise ValueError("모든 파싱 방법 실패")
    
    def _create_batch_prompt(self) -> str:
        """전체 카테고리 생성 프롬프트"""
    
        # 39개 카테고리 정의
        categories = [
            ("T", "상의", "티셔츠"), ("S", "상의", "셔츠"), ("B", "상의", "블라우스"),
            ("K", "상의", "니트"), ("H", "상의", "후드티"), ("SW", "상의", "맨투맨"),
            ("J", "하의", "청바지"), ("CP", "하의", "면바지"), ("SL", "하의", "슬랙스"),
            ("SH", "하의", "반바지"), ("SK", "하의", "치마"), ("LG", "하의", "레깅스"),
            ("JK", "아우터", "자켓"), ("CT", "아우터", "코트"), ("PD", "아우터", "패딩"),
            ("CD", "아우터", "가디건"), ("JP", "아우터", "점퍼"), ("VT", "아우터", "조끼"),
            ("DM", "원피스", "미니 원피스"), ("DMI", "원피스", "미디 원피스"),
            ("DMX", "원피스", "맥시 원피스"), ("DS", "원피스", "셔츠 원피스"),
            ("SN", "신발", "운동화"), ("HL", "신발", "힐"), ("FL", "신발", "플랫"),
            ("BT", "신발", "부츠"), ("SD", "신발", "샌들"), ("SLP", "신발", "슬리퍼"),
            ("BP", "가방", "백팩"), ("CB", "가방", "크로스백"), ("TB", "가방", "토트백"),
            ("CL", "가방", "클러치"), ("SB", "가방", "숄더백"),
            ("HT", "액세서리", "모자"), ("SC", "액세서리", "스카프"), ("BL", "액세서리", "벨트"),
            ("JW", "액세서리", "주얼리"), ("WC", "액세서리", "시계"), ("SG", "액세서리", "선글라스"),
        ]
        
        # 카테고리 목록 문자열
        categories_str = "\n".join([
            f"  - 코드: {code}, 대분류: {main}, 소분류: {sub}"
            for code, main, sub in categories
        ])
        
        return f"""
    논문의 LCA 데이터를 기반으로, 다음 39개 의류 카테고리의 환경 임팩트를 한 번에 계산하세요.

    **카테고리 목록 (39개):**
    {categories_str}

    **논문 참고 데이터 (Table 3 기준, 1kg당 스왑 절감량):**
    - Cotton (면): CO2 62.4 kg, Water 14.2 m³, Energy 1303 MJ
    - Polyester (폴리에스터): CO2 50.4 kg, Water 1.1 m³, Energy 1018.4 MJ
    - Viscose (비스코스): CO2 42.7 kg, Water 1.2 m³, Energy 984 MJ
    - Wool (양모): CO2 113.5 kg, Water 1.4 m³, Energy 1553.9 MJ
    - Acrylic (아크릴): CO2 41.2 kg, Water 1.1 m³, Energy 941.4 MJ
    - Leather (가죽): CO2 465.4 kg, Water 2.1 m³, Energy 1273.6 MJ

    **계산 방법:**
    1. 각 의류의 평균 무게 추정:
    - 상의(티셔츠/셔츠/블라우스): 약 0.15-0.2 kg
    - 상의(니트/후드티/맨투맨): 약 0.3-0.5 kg
    - 하의(청바지/면바지/슬랙스): 약 0.4-0.6 kg
    - 하의(반바지/치마/레깅스): 약 0.2-0.3 kg
    - 아우터: 약 0.5-1.0 kg
    - 원피스: 약 0.3-0.5 kg
    - 신발: 약 0.5-0.8 kg
    - 가방: 약 0.3-0.6 kg
    - 액세서리: 약 0.05-0.2 kg

    2. 주요 원단 추정:
    - 티셔츠/블라우스/반바지 → 면 또는 폴리에스터
    - 니트/맨투맨/후드티 → 면 또는 아크릴
    - 청바지 → 면
    - 코트/자켓 → 폴리에스터 또는 양모
    - 신발 → 가죽 또는 합성섬유
    - 가방 → 가죽 또는 합성섬유

    3. 계산식: **평균 무게(kg) × 1kg당 절감량 = 1개당 절감량**

    **출력 형식 (JSON만, 39개 전부):**
    {{
    "categories": [
        {{
        "code": "T",
        "main_category": "상의",
        "sub_category": "티셔츠",
        "co2_kg": 10.08,
        "water_m3": 2.84,
        "energy_mj": 203.68
        }},
        {{
        "code": "S",
        "main_category": "상의",
        "sub_category": "셔츠",
        "co2_kg": 12.48,
        "water_m3": 2.84,
        "energy_mj": 260.6
        }},
        ... (나머지 37개 카테고리)
    ]
    }}

    **중요 규칙:**
    - JSON 형식으로만 출력 (다른 설명 금지)
    - 39개 카테고리 전부 작성
    - 소수점 둘째 자리까지 표기
    - 마지막 항목에는 쉼표(,) 없음
    - 논문 데이터 기반 합리적 추정

    **예시 계산:**
    - 티셔츠 (면 0.2kg): CO2 = 0.2 × 62.4 = 12.48 kg
    - 드레스 (폴리에스터 0.478kg): CO2 = 0.478 × 50.4 = 24.09 kg
    - 코트 (양모 1.0kg): CO2 = 1.0 × 113.5 = 113.5 kg
    """.strip()