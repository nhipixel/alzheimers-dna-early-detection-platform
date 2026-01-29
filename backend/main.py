"""
Main FastAPI application.

This is the entry point for the Epigenetic Memory Loss Prediction API.
It provides endpoints for making predictions using XGBoost and PyTorch models.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routes import prediction_router, health_router, analysis_router, websocket_router
from app.services import get_model_service
from app.utils import setup_logging, validate_model_files
from app.models.schemas import ErrorResponse
from app.middleware import setup_middleware, error_handler_middleware

# Initialize settings
settings = get_settings()

# Set up logging
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    
    This function handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting up Epigenetic Memory Loss Prediction API...")
    
    # Validate model files exist
    model_validation = validate_model_files(
        settings.xgboost_model_path,
        settings.pytorch_model_path
    )
    logger.info(f"Model file validation: {model_validation}")
    
    # Initialize model service (loads models)
    model_service = get_model_service()
    loaded_models = model_service.get_loaded_models()
    logger.info(f"Loaded models: {loaded_models}")
    
    if not any(loaded_models.values()):
        logger.warning("No models were successfully loaded!")
    
    logger.info("Application startup complete.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## Epigenetic Memory Loss Prediction API
    
    This API provides machine learning predictions for epigenetic memory loss analysis using:
    - **XGBoost Model**: Gradient boosting classifier
    - **PyTorch Model**: Neural network classifier
    
    ### Features
    - Single endpoint for predictions using either or both models
    - Automatic model loading and management
    - Comprehensive error handling and validation
    - Health checks and monitoring
    - Confidence scores and probability distributions
    
    ### Usage
    1. Send prediction data to `/api/v1/predict`
    2. Choose model type: 'xgboost', 'pytorch', or 'both'
    3. Receive predictions with confidence scores
    
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
setup_middleware(app)
error_handler_middleware(app)


# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"type": str(type(exc).__name__)}
        ).dict()
    )


# Include routers
app.include_router(prediction_router, prefix=f"{settings.api_v1_prefix}", tags=["predictions"])
app.include_router(health_router, prefix=f"{settings.api_v1_prefix}", tags=["health"])
app.include_router(analysis_router, prefix=f"{settings.api_v1_prefix}/analyses", tags=["analyses"])
app.include_router(websocket_router, prefix=f"{settings.api_v1_prefix}", tags=["websocket"])


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Epigenetic Memory Loss Prediction API",
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_check": f"{settings.api_v1_prefix}/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )