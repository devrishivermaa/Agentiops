#!/usr/bin/env python3
"""
AgentOps API Server Entry Point

This starts the complete AgentOps system:
1. FastAPI server (API layer)
2. Ray engine (distributed processing) - if available
3. EventBus (real-time event streaming)

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ REST API    │  │ WebSocket   │  │ File Upload         │  │
│  │ /api/*      │  │ /api/ws     │  │ /api/upload         │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         └────────────────┼─────────────────────┘             │
│                          │                                   │
│                    ┌─────▼─────┐                             │
│                    │ EventBus  │ (Real-time events)          │
│                    └─────┬─────┘                             │
└──────────────────────────┼───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                   Engine Layer (Ray)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Orchestrator│──│ SubMasters  │──│ Workers             │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Mapper      │  │ MasterAgent │  │ Report Generator    │   │
│  └─────────────┘  └─────────────┘  └─────────────────────┘   │
└───────────────────────────────────────────────────────────────┘

Run with: python run_api.py
"""

import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("AgentOps")


def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗ ██████╗ ██████╗███████╗
║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝
║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║   ██║██████╔╝███████╗
║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██╔═══╝ ╚════██║
║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ╚██████╔╝██║     ███████║
║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚══════╝
║                                                               ║
║                 Document Processing Engine                    ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check and report on available dependencies."""
    deps = {
        "ray": False,
        "fastapi": False,
        "uvicorn": False,
        "pydantic": False,
    }
    
    try:
        import ray
        deps["ray"] = f"v{ray.__version__}"
    except ImportError:
        deps["ray"] = "NOT INSTALLED (distributed processing disabled)"
    
    try:
        import fastapi
        deps["fastapi"] = f"v{fastapi.__version__}"
    except ImportError:
        deps["fastapi"] = "NOT INSTALLED"
    
    try:
        import uvicorn
        deps["uvicorn"] = f"v{uvicorn.__version__}"
    except ImportError:
        deps["uvicorn"] = "NOT INSTALLED"
    
    try:
        import pydantic
        deps["pydantic"] = f"v{pydantic.__version__}"
    except ImportError:
        deps["pydantic"] = "NOT INSTALLED"
    
    print("\n📦 Dependencies:")
    for name, status in deps.items():
        icon = "✅" if not isinstance(status, str) or "NOT" not in status else "⚠️"
        print(f"   {icon} {name}: {status}")
    print()


def main():
    """Start the FastAPI server with the AgentOps engine."""
    print_banner()
    check_dependencies()
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    print("🚀 Starting AgentOps Server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    print()
    print("📡 Endpoints:")
    print(f"   API Documentation: http://{host}:{port}/docs")
    print(f"   Health Check:      http://{host}:{port}/health")
    print(f"   Upload PDF:        POST http://{host}:{port}/api/upload")
    print(f"   Start Pipeline:    POST http://{host}:{port}/api/pipeline/start")
    print(f"   WebSocket Events:  ws://{host}:{port}/api/ws")
    print()
    print("=" * 60)
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
