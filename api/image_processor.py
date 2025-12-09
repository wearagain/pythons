"""
이미지 크롭 API

백엔드에서 이미지를 받아 CenterCropper로 크롭 후 S3에 업로드
"""

import cv2
import numpy as np
import requests
import boto3
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import sys
import os
from io import BytesIO

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import get_logger
from modules import CenterCropper, passive_crop
logger = get_logger(__name__)


class ImageProcessor:
    """이미지 처리 및 S3 업로드"""
    
    def __init__(self):
        """초기화"""
        # CenterCropper (자동 크롭)
        self.cropper = CenterCropper()
        
        # AWS S3 설정
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region = os.getenv('AWS_REGION', 'ap-northeast-2')
        self.bucket_name = os.getenv('AWS_S3_BUCKET')
        
        # S3 클라이언트 초기화
        if self.aws_access_key and self.aws_secret_key and self.bucket_name:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.region
                )
                logger.info(f"S3 클라이언트 초기화 완료 (bucket: {self.bucket_name})")
            except Exception as e:
                logger.error(f"S3 클라이언트 초기화 실패: {e}")
                self.s3_client = None
        else:
            logger.warning("AWS 자격증명이 설정되지 않았습니다. 로컬 저장")
            self.s3_client = None
        
        logger.info("ImageProcessor 초기화 완료")
    
    def download_image(self, image_url: str) -> Optional[np.ndarray]:
        """
        URL에서 이미지 다운로드
        
        Args:
            image_url: 이미지 URL
            
        Returns:
            이미지 numpy array 또는 None
        """
        try:
            logger.info(f"이미지 다운로드 중: {image_url}")
            
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # bytes를 numpy array로 변환
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("이미지 디코딩 실패")
                return None
            
            logger.info(f"이미지 다운로드 완료 (크기: {image.shape[:2]})")
            return image
            
        except Exception as e:
            logger.error(f"이미지 다운로드 실패: {e}")
            return None
    
    def upload_to_s3(
        self, 
        image: np.ndarray, 
        filename: Optional[str] = None,
        folder: str = "cropped"
    ) -> Optional[str]:
        """
        이미지를 S3에 업로드
        """
        if self.s3_client is None:
            logger.error("S3 클라이언트가 초기화되지 않았습니다.")
            return None
        
        try:
            # 파일명 생성
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f"{timestamp}.jpg"
            
            # S3 키 생성
            s3_key = f"{folder}/{filename}"
            
            # 이미지를 JPEG로 인코딩
            success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if not success:
                logger.error("이미지 인코딩 실패")
                return None
            
            # BytesIO로 변환
            image_bytes = BytesIO(buffer.tobytes())
            
            # S3에 업로드
            logger.info(f"S3 업로드 중: {s3_key}")
            
            self.s3_client.upload_fileobj(
                image_bytes,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/jpeg'
                }
            )
            
            # S3 URL 생성
            s3_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"S3 업로드 완료: {s3_url}")
            
            return s3_url
            
        except Exception as e:
            logger.error(f"S3 업로드 실패: {e}")
            return None
    
    def process_and_upload(
        self,
        image_url: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        전체 프로세스: 다운로드 → 자동 크롭 → S3 업로드
        """
        
        logger.info("이미지 처리 프로세스 시작")
        
        # 1. 이미지 다운로드
        image = self.download_image(image_url)
        if image is None:
            return None
        
        # 2. 자동 크롭 (CenterCropper 사용!)
        logger.info("CenterCropper로 자동 크롭 중...")
        cropped = self.cropper.crop(image)
        
        if cropped is None:
            logger.error("크롭 실패")
            return None
        
        # 3. S3 업로드
        s3_url = self.upload_to_s3(cropped, filename)
        
        logger.info("이미지 처리 프로세스 완료")
        
        return s3_url
    
    def process_and_upload_passive(self, image_url:str, corners: List[List[float]], filename: Optional[str] = None) -> Optional[str]:
        """ 좌표 수동 크롭 + S3 업로드 """
        
        logger.info(f"수동 크롭 처리 시작")
        
        try:
            # 1. 이미지 다운로드 
            logger.info("이미지 다운로드...")
            image = self.download_image(image_url)
            
            if image is None:
                logger.error("이미지 다운 실패")
                return None
            
            logger.info(f"다운로드 완료: {image.shape}")
            
            # 2. 수동 크롭 
            logger.info("수동 크롭")
            cropped = passive_crop(image, corners)
            
            logger.info(f"크롭 완료 : {cropped.shape}")
            
            # 3. S3 업로드
            logger.info("S3 업로드")
            s3_url = self.upload_to_s3(cropped, filename)
            
            if s3_url:
                logger.info(f"완료 : {s3_url}")
            else:
                logger.warning("S3 업로드 실패 - 로컬 저장")
            
            return s3_url
        
        except Exception as e:
            logger.error(f"수동 크롭 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return None


# 싱글톤 인스턴스
image_processor = ImageProcessor()

