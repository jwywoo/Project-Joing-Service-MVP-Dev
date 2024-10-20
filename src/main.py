import os

from dotenv import load_dotenv

from fastapi import FastAPI

# Routers
from proposal.router import router as proposal_router
from profile.router import router as profile_router


# config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# router
app = FastAPI()
app.include_router(profile_router)
app.include_router(proposal_router)

@app.get("/")
def root():
    return {"message": "Welcome to Project Joing AI Api Server!"}