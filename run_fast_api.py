from server.server import BaseServer
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()


@app.post("/update_wight")
async def update(request: Request):
    server = BaseServer("config.yaml")
    server.update()


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
