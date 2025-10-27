"""
Gemini API 클라이언트 (RAG 지원 + Qwen3-0.6B Embedding)
LangChain 0.3 버전

Google Gemini 2.0 Flash + Qwen3-Embedding-0.6B
"""

import os
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

from config import settings, get_logger

logger = get_logger(__name__)


class GeminiClient:
    """Gemini API 클라이언트 (RAG + Qwen3-0.6B)"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        use_rag: bool = False,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
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
        
        # Gemini API 설정
        genai.configure(api_key=self.api_key)
        
        # 모델 설정
        self.model_name = settings.LLM_CONFIG.get('model', 'gemini-2.0-flash-exp')
        self.model = genai.GenerativeModel(self.model_name)
        
        # Generation 설정
        self.generation_config = {
            'temperature': settings.LLM_CONFIG.get('temperature', 0.7),
            'top_p': settings.LLM_CONFIG.get('top_p', 0.95),
            'top_k': settings.LLM_CONFIG.get('top_k', 40),
            'max_output_tokens': settings.LLM_CONFIG.get('max_output_tokens', 2048),
        }
        
        # Safety 설정
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
        ]
        
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
        """RAG 시스템 초기화 (Qwen3-0.6B Embedding 사용)"""
        try:
            # 벡터DB 경로
            persist_directory = settings.OUTPUT_DIR / "vector_db"
            persist_directory.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Qwen3-0.6B Embedding 모델 로드 중: {self.embedding_model_name}")
            
            # HuggingFace Embeddings (Qwen3-0.6B) - LangChain 0.3
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
            
            # 벡터DB 로드 또는 생성 - LangChain 0.3
            try:
                # 기존 벡터DB 로드 시도
                self.vector_db = Chroma(
                    persist_directory=str(persist_directory),
                    embedding_function=self.embeddings
                )
                doc_count = self.vector_db._collection.count()
                
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
            
            # LangChain LLM - 0.3 버전
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.generation_config['temperature'],
                max_output_tokens=self.generation_config['max_output_tokens']
            )
            
            # RAG 체인 생성
            if self.vector_db._collection.count() > 0:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vector_db.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    ),
                    return_source_documents=True
                )
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
            chat_history: 이전 대화 히스토리
            system_prompt: 시스템 프롬프트
            
        Returns:
            AI 응답 텍스트
        """
        if self.use_rag and self.qa_chain and self.vector_db._collection.count() > 0:
            return self._chat_with_rag(message, system_prompt)
        else:
            return self._chat_without_rag(message, chat_history, system_prompt)
    
    def _chat_with_rag(self, message: str, system_prompt: Optional[str] = None) -> str:
        """RAG를 사용한 채팅 (Qwen3-0.6B Embedding으로 검색)"""
        try:
            if system_prompt:
                full_message = f"{system_prompt}\n\n질문: {message}"
            else:
                full_message = message
            
            # RAG 질의 - LangChain 0.3
            result = self.qa_chain.invoke({"query": full_message})
            response = result["result"]
            
            # 출처 문서 로깅
            if "source_documents" in result:
                sources = []
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "")
                    sources.append(f"{source} (p.{page})" if page else source)
                logger.info(f"참조 문서: {sources}")
                logger.info(f"유사도 검색: Qwen3-0.6B Embedding 사용")
            
            logger.info(f"RAG 응답 생성 완료 (길이: {len(response)}자)")
            return response
            
        except Exception as e:
            logger.error(f"RAG 호출 실패, 일반 모드로 전환: {e}")
            return self._chat_without_rag(message, None, system_prompt)
    
    def _chat_without_rag(
        self, 
        message: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """RAG 없이 일반 채팅"""
        try:
            if system_prompt:
                history = [{"role": "user", "parts": system_prompt}]
                if chat_history:
                    history.extend(chat_history)
            else:
                history = chat_history or []
            
            chat = self.model.start_chat(history=history)
            
            response = chat.send_message(
                message,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            logger.info(f"일반 API 응답 생성 완료 (길이: {len(response.text)}자)")
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API 호출 실패: {e}")
            raise
    
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """단일 프롬프트 생성"""
        if self.use_rag and self.qa_chain and self.vector_db._collection.count() > 0:
            return self._chat_with_rag(prompt, system_prompt)
        
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            logger.info(f"생성 완료 (길이: {len(response.text)}자)")
            return response.text
            
        except Exception as e:
            logger.error(f"생성 실패: {e}")
            raise
    
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ):
        """스트리밍 채팅"""
        try:
            if system_prompt:
                history = [{"role": "user", "parts": system_prompt}]
                if chat_history:
                    history.extend(chat_history)
            else:
                history = chat_history or []
            
            chat = self.model.start_chat(history=history)
            
            response = chat.send_message(
                message,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            
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
        """
        PDF 문서를 벡터DB에 추가 (Qwen3-0.6B Embedding 사용)
        
        Args:
            pdf_path: PDF 파일 경로
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
        """
        if not self.use_rag:
            logger.warning("RAG가 비활성화되어 있습니다.")
            return
        
        try:
            # LangChain 0.3 - community 패키지에서 import
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            logger.info(f"PDF 로드 중: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            # 벡터DB에 추가
            logger.info(f"Qwen3-0.6B Embedding으로 벡터화 중 ({len(splits)}개 청크)...")
            self.vector_db.add_documents(splits)
            
            # RAG 체인 재초기화
            if not self.qa_chain:
                self._init_rag()
            
            logger.info(f"✓ 문서 추가 완료 (총 {self.vector_db._collection.count()}개 청크)")
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            raise
    
    def add_documents_from_text(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict]] = None
    ):
        """
        텍스트를 벡터DB에 직접 추가
        
        Args:
            texts: 텍스트 리스트
            metadatas: 메타데이터 리스트
        """
        if not self.use_rag:
            logger.warning("RAG가 비활성화되어 있습니다.")
            return
        
        try:
            # LangChain 0.3
            from langchain_core.documents import Document
            
            documents = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(texts, metadatas or [{}] * len(texts))
            ]
            
            logger.info(f"Qwen3-0.6B Embedding으로 벡터화 중 ({len(documents)}개 문서)...")
            self.vector_db.add_documents(documents)
            
            if not self.qa_chain:
                self._init_rag()
            
            logger.info(f"✓ 문서 추가 완료 (총 {self.vector_db._collection.count()}개)")
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            raise
    
    def clear_vector_db(self):
        """벡터DB 초기화"""
        if not self.use_rag or not self.vector_db:
            return
        
        try:
            # LangChain 0.3 - delete 메서드 사용
            self.vector_db.delete_collection()
            logger.info("벡터DB 초기화 완료")
            self._init_rag()
        except Exception as e:
            logger.error(f"벡터DB 초기화 실패: {e}")
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        유사 문서 검색 (Qwen3-0.6B Embedding 사용)
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            유사 문서 리스트
        """
        if not self.use_rag or not self.vector_db:
            logger.warning("RAG가 비활성화되어 있습니다.")
            return []
        
        try:
            # LangChain 0.3 - 메서드 이름 변경
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
            "document_count": self.vector_db._collection.count(),
            "rag_active": self.qa_chain is not None
        }