# Model/impact_rag.py (수정 버전)

"""
환경 임팩트 전용 RAG 시스템 (표 변환 지원)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdfplumber
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Model.gemini import GeminiClient
from config import settings, get_logger

logger = get_logger(__name__)


class ImpactRAG:
    """환경 임팩트 계산 전용 RAG 시스템 (표 변환 지원)"""
    
    def __init__(self):
        """
        RAG 시스템 초기화
        - Gemini 클라이언트 (RAG 모드)
        - 논문 자동 로드 (표 포함)
        """
        logger.info("환경 임팩트 RAG 시스템 초기화 중...")
        
        # Gemini 클라이언트 (RAG 사용)
        try:
            self.client = GeminiClient(use_rag=True)
            self.rag_enabled = self.client.use_rag
        except Exception as e:
            logger.error(f"Gemini 클라이언트 초기화 실패: {e}")
            self.client = GeminiClient(use_rag=False)
            self.rag_enabled = False
        
        # RAG가 활성화된 경우에만 논문 로드
        if self.rag_enabled:
            self._load_papers_with_tables()
        else:
            logger.warning("⚠️ RAG 비활성화 - 논문 없이 LLM만 사용합니다")
        
        logger.info("✓ 환경 임팩트 RAG 시스템 초기화 완료")
    
    def _load_papers_with_tables(self):
        """
        논문 PDF를 표와 함께 로드
        """
        # vector_db가 없으면 종료
        if not hasattr(self.client, 'vector_db') or self.client.vector_db is None:
            logger.warning("벡터DB가 없습니다 - 논문 로드 건너뜁니다")
            return
        
        paper_dir = Path("data/papers")
        
        if not paper_dir.exists():
            logger.warning(f"논문 디렉토리가 없습니다: {paper_dir}")
            return
        
        pdf_files = list(paper_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("논문 PDF 파일이 없습니다")
            return
        
        for pdf_path in pdf_files:
            logger.info(f"논문 로드 중 (표 변환 지원): {pdf_path.name}")
            
            try:
                logger.info("  - PDF에서 표 포함 텍스트 추출 시작")
                # PDF에서 표를 Markdown으로 변환하여 추출
                documents = self._extract_pdf_with_tables(str(pdf_path))
                logger.info(f"  - PDF 추출 완료, Document {len(documents)}개")

                # 🔍 디버깅: 너무 많으면 일단 일부만 넣어서 속도 확인
                max_docs = 50  # 우선 50개만 테스트용으로
                docs_to_add = documents[:max_docs]
                logger.info(
                    f"  - 벡터DB에 추가할 Document 수: {len(docs_to_add)} "
                    f"(전체 {len(documents)} 중 상위 {max_docs}개만 사용)"
                )

                if docs_to_add:
                    import time
                    start = time.time()
                    logger.info("  - 벡터DB 추가 시작")
                    self.client.vector_db.add_documents(docs_to_add)
                    elapsed = time.time() - start
                    logger.info(f"  ✓ 벡터DB 추가 완료 (소요시간: {elapsed:.2f}초)")
                else:
                    logger.warning("  - 생성된 Document가 없습니다 (빈 PDF로 처리)")
                
            except Exception as e:
                logger.error(f"  ✗ 논문 로드 실패: {e}")
                import traceback
                traceback.print_exc()
        
        # 최종 통계
        try:
            doc_count = len(self.client.vector_db.get()['ids'])
            logger.info(f"✓ 논문 로드 완료 (총 {doc_count}개 청크, 표 포함)")
        except:
            logger.warning("벡터DB 통계 확인 실패")
    
    def _extract_pdf_with_tables(self, pdf_path: str) -> List[Document]:
        """
        PDF에서 표를 Markdown으로 변환하여 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            LangChain Document 리스트
        """
        documents = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # 1. 텍스트 추출
                text = page.extract_text() or ""
                
                # 2. 표 추출
                tables = page.extract_tables()
                
                # 3. 표를 Markdown으로 변환하여 텍스트에 추가
                if tables:
                    for table_idx, table in enumerate(tables, 1):
                        markdown_table = self._table_to_markdown(table)
                        if markdown_table:
                            text += f"\n\n### Table {table_idx} (Page {page_num})\n\n{markdown_table}\n\n"
                
                # 4. Document 생성
                if text.strip():  # 빈 페이지 제외
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "page": page_num,
                            "has_tables": len(tables) > 0,
                            "table_count": len(tables)
                        }
                    )
                    documents.append(doc)
        
        # 5. 텍스트 분할 (너무 긴 경우)
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            documents = text_splitter.split_documents(documents)
        
        return documents
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """
        표를 Markdown 형식으로 변환
        """
        if not table or not table[0]:
            return ""
        
        markdown_lines = []
        
        try:
            # 헤더 행
            headers = table[0]
            header_line = "| " + " | ".join(str(h or "").strip() for h in headers) + " |"
            separator = "| " + " | ".join(["---"] * len(headers)) + " |"
            
            markdown_lines.append(header_line)
            markdown_lines.append(separator)
            
            # 데이터 행
            for row in table[1:]:
                # 완전히 빈 행은 제외
                if any(cell for cell in row):
                    row_line = "| " + " | ".join(str(cell or "").strip() for cell in row) + " |"
                    markdown_lines.append(row_line)
            
            return "\n".join(markdown_lines)
        
        except Exception as e:
            logger.warning(f"표 변환 실패: {e}")
            return ""
    
    def generate_impact_data(
        self,
        main_category: str,
        sub_category: str,
        code: str,
        fabric: str
    ) -> Dict:
        """
        특정 카테고리×소재의 환경 임팩트 데이터 생성
        """
        # 프롬프트 생성
        prompt = self._create_impact_prompt(main_category, sub_category, code, fabric)
        
        # LLM 호출
        try:
            response = self.client.chat(
                message=prompt,
                system_prompt="당신은 환경 임팩트 전문가입니다. 논문 데이터를 기반으로 정확한 수치를 제공하세요."
            )
            
            # JSON 파싱
            import json
            
            # Markdown 코드 블록 제거
            response_clean = response.strip()
            if response_clean.startswith("```"):
                response_clean = response_clean.split("```")[1]
                if response_clean.startswith("json"):
                    response_clean = response_clean[4:]
            
            data = json.loads(response_clean.strip())
            
            logger.info(f"✓ 데이터 생성 완료: {sub_category} ({fabric})")
            return data
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            logger.error(f"응답 내용: {response[:500]}")
            raise
        except Exception as e:
            logger.error(f"데이터 생성 실패: {e}")
            raise
    
    def _create_impact_prompt(
        self,
        main_category: str,
        sub_category: str,
        code: str,
        fabric: str
    ) -> str:
        """환경 임팩트 생성 프롬프트"""
        
        prompt = f"""
논문에서 제공된 LCA 데이터를 기반으로, 다음 의류의 환경 임팩트를 계산하세요.

**의류 정보:**
- 대분류: {main_category}
- 소분류: {sub_category}
- 코드: {code}
- 소재: {fabric}

**중요한 지침:**

1. **논문의 Table 1 데이터를 우선 참고하세요**
   - Table 1은 "1kg당" 환경 임팩트를 제공합니다
   - 각 소재별 CO2, Water, Energy, Land 데이터가 있습니다

2. **계산 방법:**
   - 해당 의류의 평균 무게를 추정하세요 (kg)
   - 평균 무게 × 1kg당 절감량 = 의류 1개당 절감량

3. **신뢰도 판정:**
   - high: 논문에 정확한 데이터가 있음
   - medium: 유사 카테고리로 추정
   - low: 일반적인 업계 추정치

4. **JSON만 출력하세요** (설명 제외)

**출력 형식:**
{{
  "main_category": "{main_category}",
  "sub_category": "{sub_category}",
  "code": "{code}",
  "fabric": "{fabric}",
  "average_weight_kg": 0.2,
  "impact_per_item": {{
    "co2_kg": 12.48,
    "water_m3": 2.84,
    "energy_mj": 260.6,
    "land_m2a": 199.08
  }},
  "source": "MDPI Sustainability 2025, Table 1",
  "confidence": "high",
  "generated_at": "2024-12-02T10:00:00"
}}

**JSON만 출력하세요. 다른 설명은 하지 마세요.**
"""
        return prompt.strip()


# 테스트 코드
if __name__ == "__main__":
    # 간단한 테스트
    try:
        rag = ImpactRAG()
        
        print("\n=== 시스템 상태 ===")
        print(f"RAG 활성화: {rag.rag_enabled}")
        
        if rag.rag_enabled and rag.client.vector_db:
            doc_count = len(rag.client.vector_db.get()['ids'])
            print(f"벡터DB 문서 수: {doc_count}개")
        
        print("\n=== 테스트: 티셔츠(Cotton) 임팩트 생성 ===")
        result = rag.generate_impact_data(
            main_category="상의",
            sub_category="티셔츠",
            code="T",
            fabric="cotton"
        )
        
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()