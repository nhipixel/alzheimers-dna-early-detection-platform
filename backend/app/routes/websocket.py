import asyncio
import logging
from typing import Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime

from app.services import get_model_service

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.add(connection)
        
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        model_service = get_model_service()
        
        while True:
            loaded_models = model_service.get_loaded_models()
            
            status_update = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "status_update",
                "data": {
                    "backend_status": "online",
                    "models": {
                        "xgboost": {
                            "loaded": loaded_models.get("xgboost", False),
                            "status": "ready" if loaded_models.get("xgboost") else "unavailable"
                        },
                        "pytorch": {
                            "loaded": loaded_models.get("pytorch", False),
                            "status": "ready" if loaded_models.get("pytorch") else "unavailable"
                        }
                    }
                }
            }
            
            await manager.send_message(status_update, websocket)
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from status WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)


@router.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            response = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "prediction_update",
                "data": {
                    "message": f"Received data: {data}",
                    "status": "processing"
                }
            }
            
            await manager.send_message(response, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from predictions WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)
