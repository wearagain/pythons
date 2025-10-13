"""
Detection 모듈

티켓 검출 관련 기능을 제공합니다.
"""

from .detector import TicketDetector
from .preprocessor import Preprocessor
from .transformer import PrespectiveTransformer

__all__ = [
    'TicketDetector',
    'Preprocessor',
    'PrespectiveTransformer',
]