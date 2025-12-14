#!/usr/bin/env python
"""
Démarrage de l'API FastAPI
==========================

Usage:
    python scripts/start_api.py
    python scripts/start_api.py --port 8080 --reload
"""

import argparse
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Démarrage API EIMLIA')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Adresse d\'écoute')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port d\'écoute')
    parser.add_argument('--reload', action='store_true',
                       help='Mode développement avec rechargement automatique')
    parser.add_argument('--workers', type=int, default=1,
                       help='Nombre de workers')
    parser.add_argument('--log-level', type=str, default='info',
                       choices=['debug', 'info', 'warning', 'error'],
                       help='Niveau de log')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  API EIMLIA-3M-TEU")
    print("=" * 70)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Workers: {args.workers}")
    print(f"  Reload: {args.reload}")
    print("=" * 70)
    print(f"\n  📖 Documentation: http://{args.host}:{args.port}/docs")
    print(f"  🔍 ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"  ❤️ Health: http://{args.host}:{args.port}/health\n")
    
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level
    )


if __name__ == '__main__':
    main()
