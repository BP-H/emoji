# main.py

# --- IMPORTS ---
import os
import uvicorn
import atexit
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from typing import Dict, Any, List

# Background task scheduler
from apscheduler.schedulers.background import BackgroundScheduler

# Import your entire engine file as a module
import emoji_engine

# --- CONFIGURATION ---
# Load environment variables from a .env file (optional but recommended)
load_dotenv()

# --- CORE APPLICATION SETUP ---
print("Initializing The Emoji Engine...")

# 1. Create a single, shared instance of your engine.
#    This agent will live in memory for as long as the server is running.
#    The `load_state()` method is automatically called on initialization.
agent = emoji_engine.RemixAgent()

# 2. Create the FastAPI application instance.
app = FastAPI(
    title="The Emoji Engine API",
    description="API for the MetaKarma Hub Ultimate Mega-Agent",
    version="5.28.11"
)

# --- BACKGROUND TASKS (PERSISTENCE) ---
def persist_state():
    """A simple function that tells the agent to save its state."""
    print(f"[{emoji_engine.ts()}] BACKGROUND: Saving snapshot...")
    try:
        agent.save_snapshot()
        print(f"[{emoji_engine.ts()}] BACKGROUND: Snapshot saved successfully.")
    except Exception as e:
        print(f"[{emoji_engine.ts()}] ERROR: Could not save snapshot. Reason: {e}")

# Create a scheduler that will run in a separate thread.
scheduler = BackgroundScheduler()

# Schedule the `persist_state` function to run every 5 minutes.
# You can change "minutes" to "seconds" for more frequent saves during testing.
scheduler.add_job(func=persist_state, trigger="interval", minutes=5, id="save_snapshot_job")
scheduler.start()

# Ensure the scheduler shuts down cleanly and saves state one last time when the app exits.
atexit.register(lambda: persist_state())
atexit.register(lambda: scheduler.shutdown())

print("Emoji Engine Initialized. API is ready.")

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"message": "Welcome to the Emoji Engine API!", "version": agent.get_version()}

# === User Endpoints ===

@app.post("/users/create", status_code=201)
def create_user(payload: Dict[str, Any] = Body(...)):
    """
    Create a new user.
    Payload: { "name": "new_user_name", "species": "human" }
    """
    try:
        name = payload.get("name")
        species = payload.get("species", "human")
        # Call the agent's internal method
        new_user = agent.add_user(name=name, species=species)
        return {"message": "User created successfully", "user": new_user}
    except emoji_engine.UserExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) # Conflict
    except emoji_engine.InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e)) # Bad Request
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/users/{username}")
def get_user(username: str):
    """Get public information and owned coins for a specific user."""
    try:
        user_info = agent.get_user_info(username)
        return user_info
    except emoji_engine.InvalidInputError as e:
        raise HTTPException(status_code=404, detail=str(e))

# === Coin & Minting Endpoints ===

@app.post("/coins/mint", status_code=201)
def mint_coin(payload: Dict[str, Any] = Body(...)):
    """
    Mint a new fractional coin.
    Payload: {
        "user": "minter_username",
        "improvement": "A detailed description of the new content or remix.",
        "fractional_pct": "0.05",
        "references": [{"coin_id": "some_other_coin_id"}]
    }
    """
    try:
        user = payload.get("user")
        improvement = payload.get("improvement")
        fractional_pct = payload.get("fractional_pct")
        references = payload.get("references", []) # References are optional
        
        minted_coin = agent.mint_fractional_coin(
            user_name=user,
            improvement=improvement,
            fractional_pct_str=fractional_pct,
            references=references
        )
        return {"message": "Coin minted successfully", "coin": minted_coin}
    except (emoji_engine.InvalidInputError, emoji_engine.ImprovementRequiredError, emoji_engine.KarmaError, emoji_engine.InsufficientFundsError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/coins/{coin_id}")
def get_coin(coin_id: str):
    """Get detailed information about a specific coin."""
    try:
        coin_info = agent.get_coin_info(coin_id)
        return coin_info
    except emoji_engine.InvalidInputError as e:
        raise HTTPException(status_code=404, detail=str(e))

# === Reaction Endpoint ===

@app.post("/coins/{coin_id}/react")
def react_to_coin(coin_id: str, payload: Dict[str, Any] = Body(...)):
    """
    React to an existing coin.
    Payload: {
        "reactor": "reactor_username",
        "emoji": "🔥",
        "message": "This is amazing content!"
    }
    """
    try:
        reactor = payload.get("reactor")
        emoji = payload.get("emoji")
        message = payload.get("message", "")

        reaction_result = agent.react(
            reactor_name=reactor,
            coin_id=coin_id,
            emoji=emoji,
            message=message
        )
        return {"message": "Reaction successful", "details": reaction_result}
    except (emoji_engine.InvalidInputError, emoji_engine.EmojiRequiredError, emoji_engine.RateLimitError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# === System Endpoint ===

@app.get("/system/status")
def system_status():
    """Get the overall status of the system, including treasury."""
    return agent.get_system_status()
