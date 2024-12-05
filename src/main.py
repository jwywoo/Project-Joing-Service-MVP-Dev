import os

from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pydantic import ValidationError

# Routers
from proposal.router import router as proposal_router
from channel.router import router as channel_router
from rec_system.router import router as rec_router

# config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# router
app = FastAPI()
app.include_router(channel_router)
app.include_router(proposal_router)
app.include_router(rec_router)


@app.get("/")
def root():
    return {"message": "Welcome to Project Joing AI Api Server!"}


@app.get("/ready")
def health_check():
    return JSONResponse(
        status_code=200, content=None
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "Please try again later."},
    )