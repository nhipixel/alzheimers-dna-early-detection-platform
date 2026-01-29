import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        logger.info(f"Request started: {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"Status: {response.status_code} Duration: {process_time:.3f}s"
        )
        
        return response

def setup_middleware(app):
    app.add_middleware(RequestLoggingMiddleware)
    return app
