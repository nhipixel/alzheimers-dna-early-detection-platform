from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Enhanced health check endpoint with detailed system status"""
    try:
        from ...services import get_model_service
        from ...config import get_settings
        
        settings = get_settings()
        model_service = get_model_service()
        loaded_models = model_service.get_loaded_models()
        
        return {
            "status": "healthy",
            "version": settings.app_version,
            "models_loaded": loaded_models,
            "api_prefix": settings.api_v1_prefix,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/models/info")
async def get_models_info() -> Dict[str, Any]:
    """Get detailed information about loaded models"""
    try:
        from ...services import get_model_service
        
        model_service = get_model_service()
        metadata = model_service.get_model_metadata()
        
        return {
            "success": True,
            "models": metadata,
            "total_models": len([m for m in metadata.values() if m.get('loaded', False)])
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_type}/status")
async def get_model_status(model_type: str) -> Dict[str, Any]:
    """Get status of a specific model"""
    try:
        from ...services import get_model_service
        from ...models.schemas import ModelType
        
        if model_type not in ["xgboost", "pytorch"]:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        model_service = get_model_service()
        model_enum = ModelType.XGBOOST if model_type == "xgboost" else ModelType.PYTORCH
        is_loaded = model_service.is_model_loaded(model_enum)
        
        return {
            "model_type": model_type,
            "loaded": is_loaded,
            "status": "ready" if is_loaded else "not_loaded"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/info")
async def get_system_info() -> Dict[str, Any]:
    """Get system information and capabilities"""
    import platform
    import sys
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "api_capabilities": [
            "multi_model_prediction",
            "shap_analysis",
            "feature_importance",
            "batch_processing",
            "csv_upload"
        ]
    }