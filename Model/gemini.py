"""
Gemini API 클라이언트 (RAG 지원 + Qwen3-0.6B Embedding)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from typing import Optional, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

from config import settings, get_logger

logger = get_logger(__name__)


class GeminiClient:
    """Gemini API 클라이언트 (LangChain 통합)"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        use_rag: bool = False,
        embedding_model: str = "intfloat/multilingual-e5-base"
    ):
        """
        Args:
            api_key: Google API 키
            use_rag: RAG 사용 여부
            embedding_model: Hugging Face 임베딩 모델
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 GEMINI_API_KEY를 추가하거나 "
                "api_key 파라미터로 전달해주세요."
            )
        
        # 모델 설정
        self.model_name = settings.LLM_CONFIG.get('model', 'gemini-2.0-flash')
        
        # Generation 설정
        self.generation_config = {
            'temperature': settings.LLM_CONFIG.get('temperature', 0.7),
            'top_p': settings.LLM_CONFIG.get('top_p', 0.95),
            'top_k': settings.LLM_CONFIG.get('top_k', 40),
            'max_output_tokens': settings.LLM_CONFIG.get('max_output_tokens', 2048),
        }
        
        # LangChain LLM 초기화 (통합!)
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.generation_config['temperature'],
            max_output_tokens=self.generation_config['max_output_tokens']
        )
        
        # RAG 설정
        self.use_rag = use_rag
        self.embedding_model_name = embedding_model
        self.embeddings = None
        self.vector_db = None
        self.qa_chain = None
        
        if self.use_rag:
            self._init_rag()
        
        logger.info(
            f"Gemini 클라이언트 초기화 완료 "
            f"(모델: {self.model_name}, RAG: {self.use_rag}, "
            f"Embedding: {self.embedding_model_name if self.use_rag else 'N/A'})"
        )
    
    def _init_rag(self):
        """RAG 시스템 초기화"""
        try:
            persist_directory = settings.OUTPUT_DIR / "vector_db"
            persist_directory.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Qwen3-0.6B Embedding 모델 로드 중: {self.embedding_model_name}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={
                    'device': 'cuda' if settings.LLM_CONFIG.get('use_gpu', False) else 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': True
                }
            )
            
            logger.info("✓ Embedding 모델 로드 완료")
            
            try:
                self.vector_db = Chroma(
                    persist_directory=str(persist_directory),
                    embedding_function=self.embeddings
                )
                doc_count = len(self.vector_db.get()['ids'])
                
                if doc_count > 0:
                    logger.info(f"✓ 벡터DB 로드 완료 (문서 수: {doc_count})")
                else:
                    logger.info("✓ 빈 벡터DB 생성 완료")
                    
            except Exception as e:
                logger.info(f"새 벡터DB 생성 중... ({e})")
                self.vector_db = Chroma(
                    persist_directory=str(persist_directory),
                    embedding_function=self.embeddings
                )
                logger.info("✓ 새 벡터DB 생성 완료")
            
            # RAG 체인 생성
            doc_count = len(self.vector_db.get()['ids'])
            if doc_count > 0:
                from langchain.chains import create_retrieval_chain
                from langchain.chains.combine_documents import create_stuff_documents_chain
                from langchain_core.prompts import ChatPromptTemplate
                
                # 프롬프트 템플릿
                system_prompt = (
                    "당신은 문서 기반 질문답변 AI입니다. "
                    "아래 제공된 문서를 참고하여 정확하게 답변하세요.\n\n"
                    "{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])
                
                # RAG 체인 생성
                retriever = self.vector_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                
                question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
                self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                logger.info("✓ RAG 체인 생성 완료 (문서 검색 활성화)")
            else:
                logger.info("벡터DB가 비어있음 - 일반 LLM 모드로 작동")
            
        except Exception as e:
            logger.error(f"RAG 초기화 실패: {e}")
            logger.info("일반 API 모드로 전환")
            self.use_rag = False
    
    def chat(
        self, 
        message: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        채팅 메시지 전송 (RAG 자동 활성화)
        
        Args:
            message: 사용자 메시지
            chat_history: 이전 대화 히스토리 (사용 안 함)
            system_prompt: 시스템 프롬프트
            
        Returns:
            AI 응답 텍스트
        """
        # RAG 사용
        if self.use_rag and self.qa_chain:
            try:
                result = self.qa_chain.invoke({"input": message})
                response = result["answer"]
                
                # 출처 문서 로깅
                if "context" in result:
                    logger.info(f"RAG 응답 (Qwen3-0.6B Embedding)")
                
                logger.info(f"응답 생성 완료 (길이: {len(response)}자)")
                return response
            except Exception as e:
                logger.error(f"RAG 호출 실패: {e}")
                # 폴백
        
        # 일반 LLM (LangChain 사용!)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=message))
            
            response = self.llm.invoke(messages)
            
            logger.info(f"일반 LLM 응답 생성 완료 (길이: {len(response.content)}자)")
            return response.content
            
        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            raise
    
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """단일 프롬프트 생성"""
        return self.chat(prompt, system_prompt=system_prompt)
    
    def stream_chat(
        self,
        message: str,
        system_prompt: Optional[str] = None
    ):
        """스트리밍 채팅 (LangChain)"""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=message))
            
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content
            
            logger.info("스트리밍 응답 완료")
            
        except Exception as e:
            logger.error(f"스트리밍 실패: {e}")
            raise
    
    def add_documents_from_pdf(
        self, 
        pdf_path: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ):
        """PDF 문서를 벡터DB에 추가"""
        if not self.use_rag:
            logger.warning("RAG가 비활성화되어 있습니다.")
            return
        
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            logger.info(f"PDF 로드 중: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            logger.info(f"Qwen3-0.6B Embedding으로 벡터화 중 ({len(splits)}개 청크)...")
            self.vector_db.add_documents(splits)
            
            # RAG 체인 재초기화
            if not self.qa_chain:
                self._init_rag()
            
            doc_count = len(self.vector_db.get()['ids'])
            logger.info(f"✓ 문서 추가 완료 (총 {doc_count}개 청크)")
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            raise
    
    def add_documents_from_text(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict]] = None
    ):
        """텍스트를 벡터DB에 직접 추가"""
        if not self.use_rag:
            logger.warning("RAG가 비활성화되어 있습니다.")
            return
        
        try:
            from langchain_core.documents import Document
            
            documents = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(texts, metadatas or [{}] * len(texts))
            ]
            
            logger.info(f"Qwen3-0.6B Embedding으로 벡터화 중 ({len(documents)}개 문서)...")
            self.vector_db.add_documents(documents)
            
            if not self.qa_chain:
                self._init_rag()
            
            doc_count = len(self.vector_db.get()['ids'])
            logger.info(f"✓ 문서 추가 완료 (총 {doc_count}개)")
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            raise
    
    def clear_vector_db(self):
        """벡터DB 초기화"""
        if not self.use_rag or not self.vector_db:
            return
        
        try:
            self.vector_db.delete_collection()
            logger.info("벡터DB 초기화 완료")
            self._init_rag()
        except Exception as e:
            logger.error(f"벡터DB 초기화 실패: {e}")
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict]:
        """유사 문서 검색"""
        if not self.use_rag or not self.vector_db:
            logger.warning("RAG가 비활성화되어 있습니다.")
            return []
        
        try:
            docs_with_scores = self.vector_db.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })
            
            logger.info(f"유사 문서 검색 완료: {len(results)}개")
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def get_vector_db_stats(self) -> Dict[str, Any]:
        """벡터DB 통계 정보"""
        if not self.use_rag or not self.vector_db:
            return {
                "enabled": False,
                "embedding_model": None
            }
        
        return {
            "enabled": True,
            "embedding_model": self.embedding_model_name,
            "document_count": len(self.vector_db.get()['ids']),
            "rag_active": self.qa_chain is not None
        }