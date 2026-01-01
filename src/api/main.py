from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
from src.api.routes import report_routes
from datetime import datetime

app= FastAPI(title="Report Generator Agent")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates=Jinja2Templates(directory="src/api/templates")
app.templates= templates # Templates accessible inside router

def basename_filter(path: str):
    return os.path.basename(path)

templates.env.filters["basename"]= basename_filter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
async def health_check():
    """
    Health check for container orchestration
    """
    return {
        "status": "healthy",
        "service": "research-report-generation",
        "timestamp": datetime.now().isoformat()
    }

# Registration Routes
app.include_router(report_routes.router)