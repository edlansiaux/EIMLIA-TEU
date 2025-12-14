"""
Configuration du logging
========================
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure le logging pour EIMLIA.
    
    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
        log_file: Fichier de log optionnel
        format_string: Format personnalisé
        
    Returns:
        Logger configuré
    """
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    
    # Configuration de base
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    logger = logging.getLogger('eimlia')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # Handler fichier optionnel
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'eimlia') -> logging.Logger:
    """
    Récupère un logger configuré.
    
    Args:
        name: Nom du logger
        
    Returns:
        Logger
    """
    return logging.getLogger(name)


class ProgressLogger:
    """Logger avec barre de progression."""
    
    def __init__(self, total: int, desc: str = '', logger: logging.Logger = None):
        self.total = total
        self.desc = desc
        self.current = 0
        self.logger = logger or get_logger()
    
    def update(self, n: int = 1) -> None:
        """Met à jour la progression."""
        self.current += n
        pct = self.current / self.total * 100
        
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            self.logger.info(f"{self.desc}: {pct:.0f}% ({self.current}/{self.total})")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
