"""
AI 서버
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
import sys
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from api.category import category_manager
from api.image_processor import image_processor
from config import get_logger

logger = get_logger(__name__)

# FastAPI 앱
app = FastAPI(
    title="가치입다 AI API"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Request/Response 모델 ====================

# 환경 임팩트
class ImpactRequest(BaseModel):
    """환경 임팩트 요청)"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category_code": "T"
            }
        }
    )
    
    category_code: str = Field(..., description="카테고리 코드 (T, J, JK 등)")


class ImpactResponse(BaseModel):
    """환경 임팩트 응답"""
    co2_kg: float = Field(..., description="CO2 절감량 (kg)")
    water_m3: float = Field(..., description="물 절약량 (m³)")
    energy_mj: float = Field(..., description="에너지 절감량 (MJ)")


# 이미지 크롭 - 자동 검출
class CropAutoRequest(BaseModel):
    """이미지 자동 크롭 요청)"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_url": "https://example.com/photo.jpg",
                "filename": "my_clothes.jpg"
            }
        }
    )
    
    image_url: str = Field(..., description="원본 이미지 URL")
    filename: Optional[str] = Field(None, description="저장할 파일명 (선택)")


# 이미지 크롭 - 수동 지정
class CropPassiveRequest(BaseModel):
    """이미지 수동 크롭 요청"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_url": "https://example.com/photo.jpg",
                "corners": [[100, 50], [400, 60], [390, 350], [110, 340]],
                "filename": "my_clothes.jpg"
            }
        }
    )
    
    image_url: str = Field(..., description="원본 이미지 URL")
    corners: List[List[float]] = Field(
        ..., 
        description="흰색 테두리 4개 꼭지점 좌표 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )
    filename: Optional[str] = Field(None, description="저장할 파일명 (선택)")


# 응답
class CropResponse(BaseModel):
    """이미지 크롭 응답"""
    success: bool = Field(..., description="크롭 성공 여부")
    s3_url: Optional[str] = Field(None, description="S3 저장된 이미지 URL")
    message: Optional[str] = Field(None, description="처리 결과 메시지")
    corners: Optional[List[List[float]]] = Field(None, description="검출된 꼭지점 (자동 검출 시)")


# ==================== API 엔드포인트 ====================

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "service": "가치입다 AI API",
        "status": "running",
        "endpoints": {
            "환경 임팩트": {
                "POST /api/impact": "카테고리 코드 → CO2/물/에너지 반환 [메인]",
                "GET /api/impact/{code}": "카테고리 코드로 조회 [테스트용]",
                "GET /api/categories": "전체 카테고리 목록"
            },
            "이미지 크롭": {
                "POST /api/crop/auto": "이미지 URL만 → 자동 흰색 틀 검출 → S3 저장",
                "POST /api/crop/passive": "이미지 URL + 꼭지점 좌표 → 크롭 → S3 저장 [메인]",
                "GET /api/crop/passive": "GET 방식 수동 크롭 [테스트용]"
            }
        }
    }


# ==================== 환경 임팩트 API ====================

@app.post("/api/impact", response_model=ImpactResponse)
async def post_impact(request: ImpactRequest):
    """
    환경 임팩트 조회 (POST)
    
    백엔드가 카테고리 코드를 보내면 CO2, 물, 에너지 데이터를 반환합니다.

    """
    logger.info(f"[POST /api/impact] 요청: category_code={request.category_code}")
    
    category_data = category_manager.get_category(request.category_code)
    
    if category_data is None:
        logger.warning(f"카테고리를 찾을 수 없음: {request.category_code}")
        raise HTTPException(
            status_code=404,
            detail=f"카테고리 '{request.category_code}'를 찾을 수 없습니다."
        )
    
    logger.info(f"환경 임팩트 반환: {category_data}")
    return ImpactResponse(**category_data)


@app.get("/api/impact/{category_code}", response_model=ImpactResponse)
async def get_impact(category_code: str):
    """
    환경 임팩트 조회 (GET)
    """
    logger.info(f"[GET /api/impact/{category_code}] 테스트 요청")
    
    category_data = category_manager.get_category(category_code)
    
    if category_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"카테고리 '{category_code}'를 찾을 수 없습니다."
        )
    
    return ImpactResponse(**category_data)


@app.get("/api/categories")
async def get_all_categories():
    """
    전체 카테고리 목록 조회
    """
    logger.info("[GET /api/categories] 전체 카테고리 조회")
    
    categories = category_manager.get_all_categories()
    return {
        "total": len(categories),
        "categories": list(categories.values())
    }


# ==================== 이미지 크롭 API ====================

@app.post("/api/crop/auto", response_model=CropResponse)
async def crop_auto(request: CropAutoRequest):
    """
    이미지 자동 크롭 (POST)
    
    이미지 URL만 받아서 자동으로 흰색 테두리를 찾아 크롭합니다.
    CenterCropper가 자동으로 흰색 가이드 틀을 검출합니다.
    """
    logger.info(f"[POST /api/crop/auto] 자동 크롭 요청: {request.image_url}")
    
    try:
        s3_url = image_processor.process_and_upload(
            image_url=request.image_url,
            filename=request.filename
        )
        
        if s3_url is None:
            return CropResponse(
                success=False,
                s3_url=None,
                message="이미지 처리 실패 (흰색 틀 검출 불가)"
            )
        
        logger.info(f"자동 크롭 완료: {s3_url}")
        
        return CropResponse(
            success=True,
            s3_url=s3_url,
            message="자동 크롭 완료"
        )
        
    except Exception as e:
        logger.error(f"자동 크롭 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/crop/passive", response_model=CropResponse)
async def crop_passive_post(request: CropPassiveRequest):
    """
    이미지 수동 크롭 (POST)
    
    백엔드가 이미지 URL과 4개 꼭지점 좌표를 보내면 크롭 후 S3에 저장합니다.
    """
    logger.info(f"[POST /api/crop/passive] 수동 크롭 요청")
    logger.info(f"  - URL: {request.image_url}")
    logger.info(f"  - Corners: {request.corners}")
    
    try:
        s3_url = image_processor.process_and_upload_passive(
            image_url=request.image_url,
            corners=request.corners,
            filename=request.filename
        )
        
        if s3_url is None:
            return CropResponse(
                success=False,
                s3_url=None,
                message="수동 크롭 실패 (다운로드/크롭/업로드 오류)"
            )
        
        logger.info(f"✓ 수동 크롭 완료: {s3_url}")
        
        return CropResponse(
            success=True,
            s3_url=s3_url,
            message="수동 크롭 완료"
        )
        
    except ValueError as e:
        # 좌표 형식 오류
        logger.error(f"좌표 형식 오류: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"좌표 형식이 잘못되었습니다: {str(e)}"
        )
        
    except Exception as e:
        # 기타 오류
        logger.error(f"수동 크롭 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류: {str(e)}"
        )


@app.get("/api/crop/passive", response_model=CropResponse)
async def crop_passive_get(
    image_url: str = Query(..., description="원본 이미지 URL"),
    corners: str = Query(..., description="4개 꼭지점 JSON string")
):
    """이미지 수동 크롭 (GET) - 테스트용"""
    logger.info(f"[GET /api/crop/passive] 테스트 요청")
    
    try:
        corners_list = json.loads(corners)
        request = CropPassiveRequest(
            image_url=image_url,
            corners=corners_list
        )
        return await crop_passive_post(request)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="corners 파라미터가 올바른 JSON 형식이 아닙니다."
        )


# ==================== 헬스체크 ====================

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "service": "가치입다 AI API",
        "components": {
            "category_manager": "OK" if category_manager else "ERROR",
            "image_processor": "OK" if image_processor else "ERROR",
        }
    }


# ==================== 서버 실행 ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("가치입다 AI 서버 시작...")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )