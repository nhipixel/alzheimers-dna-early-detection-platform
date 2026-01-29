"""
Initialize routes module.
"""

from .predict import router as prediction_router
from .health import router as health_router
from .analysis import router as analysis_router
from .websocket import router as websocket_router

__all__ = ["prediction_router", "health_router", "analysis_router", "websocket_router"]