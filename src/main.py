from dotenv import load_dotenv
import os

from fastapi import FastAPI

# Routers
from proposal import router as proposal_router

# config
load_dotenv()

app = FastAPI()

app.include_router(proposal_router.router)

@app.get("/")
def root():
    return {"message": "Welcome to Project Joing AI Api Server!"}