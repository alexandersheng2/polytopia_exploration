from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from agents import registry
from pydantic import BaseModel
from server.trajectory import run_trajectory

app = FastAPI()

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/agents")
def list_agents():
    return [
        {"id": spec.id, "name": spec.name, "description": spec.description}
        for spec in registry.list_public()
    ]

class RunRequest(BaseModel):
    agent_id: str
    seed: int | None = None

@app.post("/api/run")
def run(request: RunRequest):
    return run_trajectory(request.agent_id, request.seed)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")