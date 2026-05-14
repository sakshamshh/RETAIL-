active_connections = []

async def connect(websocket):
    await websocket.accept()
    active_connections.append(websocket)

def disconnect(websocket):
    active_connections.remove(websocket)

async def broadcast(message):
    for connection in active_connections:
        await connection.send_json(message)