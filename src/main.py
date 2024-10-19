from dotenv import load_dotenv
import os

from fastapi import FastAPI

# config
load_dotenv()


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to Project Joing AI Api Server!"}