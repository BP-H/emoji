from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from emoji_engine import RemixAgent  # Adjust import if your main code file name differs

app = FastAPI(title="MetaKarma Hub API")

# Single global RemixAgent instance — holds all your users, coins, etc.
engine = RemixAgent()

# Pydantic models for request validation matching your event payloads
class AddUserRequest(BaseModel):
    user: str = Field(..., min_length=3, max_length=30)
    is_genesis: Optional[bool] = False
    species: Optional[str] = "human"
    karma: Optional[str] = "0"
    consent: Optional[bool] = True

class MintRequest(BaseModel):
    user: str
    coin_id: str
    value: str
    root_coin_id: str
    references: Optional[List[Dict[str, Any]]] = []
    improvement: Optional[str] = ""
    fractional_pct: Optional[str] = "0"
    ancestors: Optional[List[str]] = []
    timestamp: Optional[str] = None
    is_remix: Optional[bool] = False

class ReactRequest(BaseModel):
    reactor: str
    coin: str
    emoji: str
    message: Optional[str] = ""
    timestamp: Optional[str] = None
    reaction_type: Optional[str] = "react"

@app.post("/add_user")
def add_user(req: AddUserRequest):
    event = {
        "event": "ADD_USER",
        "user": req.user,
        "is_genesis": req.is_genesis,
        "species": req.species,
        "karma": req.karma,
        "consent": req.consent,
        "join_time": None,
        "last_active": None,
        "root_coin_id": f"{req.user}_root_coin",
        "coins_owned": [],
        "initial_root_value": str(engine.Config.ROOT_COIN_INITIAL_VALUE),
        "root_coin_value": str(engine.Config.ROOT_COIN_INITIAL_VALUE),
    }
    try:
        engine._process_event(event)
        engine.save_snapshot()
        return {"status": "success", "message": f"User '{req.user}' added."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mint")
def mint(req: MintRequest):
    event = {
        "event": "MINT",
        "user": req.user,
        "coin_id": req.coin_id,
        "value": req.value,
        "root_coin_id": req.root_coin_id,
        "references": req.references or [],
        "improvement": req.improvement or "",
        "fractional_pct": req.fractional_pct,
        "ancestors": req.ancestors or [],
        "timestamp": req.timestamp,
        "is_remix": req.is_remix,
    }
    try:
        engine._process_event(event)
        engine.save_snapshot()
        return {"status": "success", "message": f"Coin '{req.coin_id}' minted."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/react")
def react(req: ReactRequest):
    event = {
        "event": "REACT",
        "reactor": req.reactor,
        "coin": req.coin,
        "emoji": req.emoji,
        "message": req.message,
        "timestamp": req.timestamp,
        "reaction_type": req.reaction_type,
    }
    try:
        engine._process_event(event)
        engine.save_snapshot()
        return {"status": "success", "message": "Reaction processed."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status")
def status():
    # Simple status endpoint returning some global stats
    return {
        "users_count": len(engine.users),
        "coins_count": len(engine.coins),
        "treasury": str(engine.treasury),
    }

if __name__ == "__main__":
    # Run FastAPI server with Uvicorn on all interfaces, port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
