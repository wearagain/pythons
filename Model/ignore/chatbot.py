
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Model.gemini import GeminiClient
from .prompt import PromptManager
from config import get_logger

logger = get_logger(__name__)


class ChatbotClient:
    """챗봇 클라이언트"""
    
    def __init__(self):
        logger.info("챗봇 초기화...")
        
        self.client = GeminiClient(use_rag=False)
        self.system_prompt = PromptManager.get_system_prompt()
        
        logger.info("✓ 챗봇 준비 완료")
    
    def chat(self, message: str) -> str:
        """일반 대화"""
        try:
            response = self.client.chat(
                message=message,
                system_prompt=self.system_prompt
            )
            return response
        except Exception as e:
            logger.error(f"응답 실패: {e}")
            return "죄송합니다. 오류가 발생했습니다."


# 테스트
if __name__ == "__main__":
    chatbot = ChatbotClient()
    print(chatbot.chat("21% 파티가 뭔가요?"))