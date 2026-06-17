from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent.factory import build_memory_agent
from app.api.serialize import agent_result_to_dict, memory_to_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBUI_DIR = PROJECT_ROOT / "webui"


class ChatRequest(BaseModel):
    message: str


def create_app(bundle: Any = None) -> FastAPI:
    app = FastAPI(title="Personal Memory Agent")

    # Build the agent once at startup (uses configured LLM/embedding + persistent memory).
    # A prebuilt bundle can be injected (e.g. for tests) to avoid needing live creds.
    if bundle is None:
        bundle = build_memory_agent(
            use_rerank=os.getenv("USE_RERANK", "").lower() in {"1", "true", "yes"},
        )

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> Dict[str, Any]:
        result = bundle.agent.chat(req.message)
        bundle.save()  # persist memory vector index after each turn
        return agent_result_to_dict(result)

    @app.get("/api/memories")
    def memories() -> Dict[str, Any]:
        facts = bundle.store.list_active()
        return {"count": len(facts), "memories": [memory_to_dict(f) for f in facts]}

    # Serve the web UI at "/" (api routes above take precedence). html=True serves
    # index.html for "/", and assets (styles.css, app.js) resolve relatively.
    if WEBUI_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEBUI_DIR), html=True), name="webui")

    return app


# uvicorn entry point: `uvicorn app.api.server:app`
app = create_app() if os.getenv("SKIP_APP_BUILD") != "1" else None
