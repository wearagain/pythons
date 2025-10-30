"""
가치입다 챗봇 (간단 버전)
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  
sys.path.insert(0, str(project_root))

from Model import GeminiClient, PromptManager, ConversationManager
from config import get_logger

logger = get_logger(__name__)


def main():
    print("\n💡 명령어:")
    print("  - 'quit' 또는 'exit': 종료")
    print("  - 'clear': 대화 히스토리 초기화")
    print("  - 'history': 대화 요약 보기")
    print("=" * 60)
    print()
    
    # 1. GeminiClient 초기화 (RAG 비활성화 - 빠른 응답)
    try:
        client = GeminiClient(use_rag=False)
        logger.info("✓ Gemini 클라이언트 초기화 완료")
    except Exception as e:
        logger.error(f"Gemini 클라이언트 초기화 실패: {e}")
        print("\n API 키를 확인해주세요 (.env 파일의 GEMINI_API_KEY)")
        return
    
    # 2. ConversationManager 초기화
    conversation = ConversationManager(
        user_id="test_user",
        save_history=True
    )
    
    # 3. 시스템 프롬프트 설정
    system_prompt = PromptManager.get_system_prompt()
    
    print("✓ 챗봇이 준비되었습니다. 무엇을 도와드릴까요?\n")
    
    # 4. 대화 루프
    while True:
        try:
            # 사용자 입력
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # 종료 명령어
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("\n이용해주셔서 감사합니다!")
                
                # 대화 요약 출력
                summary = conversation.get_summary()
                print(f"\n대화 요약:")
                print(f"  - 총 메시지: {summary['total_messages']}개")
                print(f"  - 대화 시간: {summary['duration_seconds']:.0f}초")
                break
            
            # 히스토리 초기화
            elif user_input.lower() == 'clear':
                conversation.clear_history()
                print("✓ 대화 히스토리가 초기화되었습니다.\n")
                continue
            
            # 대화 요약 보기
            elif user_input.lower() == 'history':
                summary = conversation.get_summary()
                print(f"\n대화 요약:")
                print(f"  - 사용자 메시지: {summary['user_messages']}개")
                print(f"  - AI 응답: {summary['model_messages']}개")
                print(f"  - 대화 시간: {summary['duration_seconds']:.0f}초\n")
                continue
            
            # 사용자 메시지 저장
            conversation.add_user_message(user_input)
            
            # AI 응답 생성
            print("AI: ", end="", flush=True)
            
            response = client.chat(
                message=user_input,
                system_prompt=system_prompt
            )
            
            print(response)
            print()
            
            # AI 응답 저장
            conversation.add_model_message(response)
            
        except KeyboardInterrupt:
            print("\n\n 중단되었습니다.")
            break
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            print(f"\n 오류가 발생했습니다: {e}\n")
    
    # 대화 내용 저장 확인
    if conversation.save_history:
        print(f"대화 내용이 저장되었습니다: {conversation.history_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()