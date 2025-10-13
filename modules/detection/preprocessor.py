"""
티켓 검출을 위한 이미지 전처리 파이프라인
"""

import cv2
import numpy as np

from config import get_logger

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self):
        logger.debug("이미지 전처리기 초기화")
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """엣지 기반 전처리 (메인 전략)"""
        logger.debug("엣지 기반 전처리 시작")
        
        gray = self._to_grayscale(image)
        
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny 엣지 검출 (임계값 조정)
        edges = cv2.Canny(blurred, 30, 100)
        
        # 모폴로지 연산으로 엣지 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=3)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        logger.debug("엣지 기반 전처리 완료")
        return closed
        
    def preprocess_adaptive(self, image: np.ndarray) -> np.ndarray:
        logger.debug("이미지 전처리 시작")
        
        gray = self._to_grayscale(image)
        denoised = self._denoise(gray)
        enhanced = self._enhance_contrast(denoised)
        binary = self._binarize(enhanced)
        morphed = self._morpholohy(binary)
        
        logger.debug("이미지 전처리 완료")
        return morphed
    
    def preprocess_otsu(self, image: np.ndarray) -> np.ndarray:
        """Otsu 이진화 (추가 전략)"""
        logger.debug("Otsu 전처리 시작")
        
        gray = self._to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu 이진화
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 반전 (어두운 배경의 경우)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # 모폴로지
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        logger.debug("Otsu 전처리 완료")
        return morphed
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h=10,
            templateWindowSize = 7,
            searchWindowSize = 21
        )
        
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit = 3.,
            tileGridSize = (8, 8)
        )
        return clahe.apply(image)

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
    def _morpholohy(self, image: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        
        # 구멍 메우기
        closed = cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            kernel,
            iterations = 2
        )
        
        # 노이즈 제거
        opened = cv2.morphologyEx(
            closed,
            cv2.MORPH_OPEN,
            kernel,
            iterations =1
        )
        
        return opened
        
        