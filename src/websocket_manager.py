import asyncio

active_connections = []

async def connect(websocket):
    await websocket.accept()
    active_connections.append(websocket)

def disconnect(websocket):
    if websocket in active_connections:
        active_connections.remove(websocket)

async def broadcast(message):
    dead = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            dead.append(connection)
    for d in dead:
        if d in active_connections:
            active_connections.remove(d)
