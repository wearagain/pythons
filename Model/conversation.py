"""
대화 관리 모듈

채팅 히스토리와 세션을 관리합니다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import json

from config import settings, get_logger

logger = get_logger(__name__)


class ConversationManager:
    """대화 히스토리 관리자"""
    
    def __init__(self, user_id: Optional[str] = None, save_history: bool = True):
        """
        Args:
            user_id: 사용자 ID (세션 식별용)
            save_history: 대화 히스토리 저장 여부
        """
        self.user_id = user_id or "anonymous"
        self.save_history = save_history
        self.history: List[Dict[str, str]] = []
        self.session_start = datetime.now()
        
        # 히스토리 저장 경로
        if self.save_history:
            self.history_dir = settings.OUTPUT_DIR / "chat_history"
            self.history_dir.mkdir(parents=True, exist_ok=True)
            self.history_file = self.history_dir / f"{self.user_id}_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        
        logger.info(f"대화 세션 시작 (사용자: {self.user_id})")
    
    def add_message(self, role: str, content: str):
        """
        메시지 추가
        
        Args:
            role: 'user' 또는 'model'
            content: 메시지 내용
        """
        message = {
            "role": role,
            "parts": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.history.append(message)
        logger.debug(f"메시지 추가 - {role}: {content[:50]}...")
        
        # 자동 저장
        if self.save_history:
            self._save_history()
    
    def add_user_message(self, content: str):
        """사용자 메시지 추가"""
        self.add_message("user", content)
    
    def add_model_message(self, content: str):
        """모델 응답 추가"""
        self.add_message("model", content)
    
    def get_history(
        self, 
        max_messages: Optional[int] = None,
        include_timestamps: bool = False
    ) -> List[Dict[str, str]]:
        """
        대화 히스토리 가져오기
        
        Args:
            max_messages: 최대 메시지 수 (최근 N개)
            include_timestamps: 타임스탬프 포함 여부
            
        Returns:
            대화 히스토리 리스트
        """
        history = self.history.copy()
        
        # 최근 N개만
        if max_messages and len(history) > max_messages:
            history = history[-max_messages:]
        
        # 타임스탬프 제거
        if not include_timestamps:
            history = [
                {"role": msg["role"], "parts": msg["parts"]}
                for msg in history
            ]
        
        return history
    
    def get_last_exchange(self) -> Optional[Dict[str, str]]:
        """
        마지막 대화 교환 가져오기 (user + model 한 쌍)
        
        Returns:
            {"user": "...", "model": "..."} 또는 None
        """
        if len(self.history) < 2:
            return None
        
        # 마지막 두 메시지 확인
        last_two = self.history[-2:]
        
        if last_two[0]["role"] == "user" and last_two[1]["role"] == "model":
            return {
                "user": last_two[0]["parts"],
                "model": last_two[1]["parts"]
            }
        
        return None
    
    def clear_history(self):
        """히스토리 초기화"""
        logger.info(f"대화 히스토리 초기화 (메시지 수: {len(self.history)})")
        self.history.clear()
    
    def get_summary(self) -> Dict[str, any]:
        """대화 세션 요약 정보"""
        total_messages = len(self.history)
        user_messages = sum(1 for msg in self.history if msg["role"] == "user")
        model_messages = sum(1 for msg in self.history if msg["role"] == "model")
        
        duration = datetime.now() - self.session_start
        
        return {
            "user_id": self.user_id,
            "session_start": self.session_start.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "total_messages": total_messages,
            "user_messages": user_messages,
            "model_messages": model_messages,
        }
    
    def _save_history(self):
        """히스토리를 파일로 저장"""
        if not self.save_history:
            return
        
        try:
            data = {
                "user_id": self.user_id,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "messages": self.history,
                "summary": self.get_summary()
            }
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"히스토리 저장 완료: {self.history_file}")
            
        except Exception as e:
            logger.error(f"히스토리 저장 실패: {e}")
    
    @classmethod
    def load_history(cls, filepath: str) -> Optional['ConversationManager']:
        """
        저장된 히스토리 불러오기
        
        Args:
            filepath: 히스토리 파일 경로
            
        Returns:
            ConversationManager 인스턴스 또는 None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            manager = cls(user_id=data["user_id"], save_history=False)
            manager.history = data["messages"]
            manager.session_start = datetime.fromisoformat(data["session_start"])
            
            logger.info(f"히스토리 로드 완료: {filepath}")
            return manager
            
        except Exception as e:
            logger.error(f"히스토리 로드 실패: {e}")
            return None
    
    def export_text(self, output_path: Optional[str] = None) -> str:
        """
        대화를 텍스트 형식으로 내보내기
        
        Args:
            output_path: 저장 경로 (선택)
            
        Returns:
            텍스트 형식의 대화 내용
        """
        lines = [
            "=" * 80,
            f"가치입다 챗봇 대화 기록",
            f"사용자: {self.user_id}",
            f"세션 시작: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"총 메시지: {len(self.history)}개",
            "=" * 80,
            ""
        ]
        
        for i, msg in enumerate(self.history, 1):
            role_name = "사용자" if msg["role"] == "user" else "AI"
            timestamp = msg.get("timestamp", "")
            
            lines.append(f"[{i}] {role_name} ({timestamp})")
            lines.append(msg["parts"])
            lines.append("")
        
        text = "\n".join(lines)
        
        # 파일 저장
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                logger.info(f"대화 내용 저장 완료: {output_path}")
            except Exception as e:
                logger.error(f"대화 내용 저장 실패: {e}")
        
        return text