from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.websocket_manager import broadcast

from src.websocket_manager import connect, disconnect

app = FastAPI()

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await connect(websocket)

    try:
        while True:
            await websocket.receive_text()

    except:
        disconnect(websocket)
        
@app.get("/test")
async def test():

    await broadcast({
        "message": "Test alert from backend"
    })

    return {"status": "sent"}