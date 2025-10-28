"""
프롬프트 관리 모듈

가치입다 챗봇의 프롬프트 템플릿을 관리합니다.
"""

from typing import Dict, Optional
from config import get_logger

logger = get_logger(__name__)


class PromptManager:
    """프롬프트 템플릿 관리자"""
    
    # 시스템 프롬프트 (챗봇 페르소나)
    SYSTEM_PROMPT = """당신은 '가치입다'의 친절한 AI 어시스턴트입니다.

**가치입다 소개:**
- 슬로건: "같이 입어, 가치 있게"
- 의류 교환을 통해 지속가능한 의생활 문화를 만드는 플랫폼
- 가격이 아닌 '관계'와 '가치'에 집중하는 의류 순환 허브
- '21% 파티'를 통한 오프라인 의류 교환 행사 지원

**핵심 기능:**
1. 환경 임팩트 측정: 의류 교환으로 절약되는 CO2, 물 사용량 등을 데이터로 시각화
2. 21% 파티: 온·오프라인 연동 의류 교환 행사
3. 크레딧 보상 시스템: 활동에 따른 보상과 혜택
4. 수선 문화 네트워크: 검증된 수선 전문가 연결

**대화 가이드라인:**
- 친근하고 따뜻한 말투 사용
- 지속가능한 의생활에 대한 열정 표현
- 구체적인 데이터와 사례로 설명
- 사용자의 작은 실천도 격려하고 응원
- 패스트 패션 문제와 순환 경제의 중요성 강조

항상 사용자 친화적이고, 환경을 생각하는 따뜻한 톤으로 대화해주세요."""

    # 주요 프롬프트 템플릿들
    PROMPTS = {
        # 환경 임팩트 설명
        "impact_explanation": """
사용자가 교환한 의류의 환경 임팩트를 설명해주세요.

의류 정보:
- 종류: {clothing_type}
- 수량: {quantity}개

다음 정보를 포함하여 설명:
1. 절약된 CO2 배출량
2. 절약된 물 사용량
3. 나무 심기 효과로 비유
4. 격려 메시지
""",
        
        # 21% 파티 안내
        "party_guide": """
'21% 파티'에 대해 자세히 안내해주세요.

다음 내용 포함:
1. 21% 파티란 무엇인가?
2. 참여 방법
3. 준비물
4. 당일 진행 과정
5. 파티 후 얻을 수 있는 것들
""",
        
        # 의류 등록 도움
        "clothing_registration": """
사용자가 의류를 등록하는 것을 도와주세요.

필요한 정보:
- 의류 종류 (상의/하의/원피스/아우터 등)
- 브랜드 (선택)
- 사이즈
- 상태
- 특이사항 또는 이야기

친절하게 안내하고, 의류에 담긴 이야기를 공유하도록 격려해주세요.
""",
        
        # 지속가능 패션 교육
        "sustainable_fashion": """
지속가능한 패션에 대해 설명해주세요.

주제: {topic}

다음 관점으로 설명:
1. 패스트 패션의 문제점
2. 순환 경제의 중요성
3. 개인이 실천할 수 있는 방법
4. 가치입다가 제공하는 솔루션
""",
        
        # 수선 문화 안내
        "repair_culture": """
수선 문화에 대해 안내해주세요.

사용자 상황: {situation}

다음 정보 제공:
1. 수선의 장점 (환경/경제/감성)
2. 일반적인 수선 종류와 비용
3. 가치입다의 수선 전문가 네트워크
4. 수선 후기나 사례
""",
        
        # 크레딧 시스템 설명
        "credit_system": """
가치입다의 크레딧 보상 시스템을 설명해주세요.

다음 내용 포함:
1. 크레딧을 받는 방법 (활동별)
2. 크레딧 사용처
3. 레벨/배지 시스템
4. 실제 혜택 예시
""",
        
        # 일반 대화
        "general_chat": """
사용자와 자연스럽게 대화해주세요.

사용자 메시지: {message}

가치입다의 가치관(지속가능성, 관계 중심, 커뮤니티)을 자연스럽게 녹여내며 대화하되,
너무 홍보적이지 않게 친근하게 답변해주세요.
""",
    }
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """시스템 프롬프트 반환"""
        return cls.SYSTEM_PROMPT
    
    @classmethod
    def get_prompt(cls, prompt_type: str, **kwargs) -> Optional[str]:
        """
        프롬프트 템플릿 가져오기
        
        Args:
            prompt_type: 프롬프트 타입
            **kwargs: 템플릿에 삽입할 변수들
            
        Returns:
            포맷팅된 프롬프트 또는 None
        """
        template = cls.PROMPTS.get(prompt_type)
        
        if not template:
            logger.warning(f"존재하지 않는 프롬프트 타입: {prompt_type}")
            return None
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"프롬프트 포맷팅 실패 - 필요한 변수: {e}")
            return None
    
    @classmethod
    def create_impact_prompt(
        cls,
        clothing_type: str,
        quantity: int
    ) -> str:
        """환경 임팩트 설명 프롬프트 생성"""
        return cls.get_prompt(
            "impact_explanation",
            clothing_type=clothing_type,
            quantity=quantity
        )
    
    @classmethod
    def create_sustainable_fashion_prompt(cls, topic: str) -> str:
        """지속가능 패션 교육 프롬프트 생성"""
        return cls.get_prompt(
            "sustainable_fashion",
            topic=topic
        )
    
    @classmethod
    def create_repair_prompt(cls, situation: str) -> str:
        """수선 문화 안내 프롬프트 생성"""
        return cls.get_prompt(
            "repair_culture",
            situation=situation
        )
    
    @classmethod
    def create_general_chat_prompt(cls, message: str) -> str:
        """일반 대화 프롬프트 생성"""
        return cls.get_prompt(
            "general_chat",
            message=message
        )
    
    @classmethod
    def list_prompts(cls) -> Dict[str, str]:
        """사용 가능한 모든 프롬프트 목록 반환"""
        return {
            key: template.strip()[:100] + "..."
            for key, template in cls.PROMPTS.items()
        }